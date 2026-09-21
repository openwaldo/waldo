#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$(uname -s)" != "Linux" ]; then
  echo "testing: real PyTorch model lifecycle skipped (requires Linux)"
  exit 0
fi

torch_python=""
for candidate in "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
  [ -n "$candidate" ] && [ -x "$candidate" ] || continue
  if "$candidate" -c 'import torch; value=torch.tensor([1.0],device="cuda" if torch.cuda.is_available() else "cpu"); assert value.sum().item() == 1.0' >/dev/null 2>&1; then
    torch_python=$candidate
    break
  fi
done
if [ -z "$torch_python" ]; then
  echo "testing: real PyTorch model lifecycle skipped (no usable PyTorch Python runtime)"
  exit 0
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
revision=$(sed -n 's/.*PyTorchRevision = "\(.*\)".*/\1/p' "$repo_root/internal/training/pytorch.go")
[ -n "$revision" ] || { echo "could not read PyTorchRevision from internal/training/pytorch.go" >&2; exit 1; }
temporary_base=${TMPDIR:-/tmp}
work=$(mktemp -d "$temporary_base/waldo-pytorch-e2e.XXXXXX")

cleanup() {
  if [ "${WALDO_E2E_KEEP:-0}" = "1" ]; then
    echo "preserved PyTorch E2E workspace: $work"
    return
  fi
  case "$work" in
    "$temporary_base"/waldo-pytorch-e2e.*) rm -rf -- "$work" ;;
    *) echo "refusing to remove unexpected workspace: $work" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

binary="$work/waldo"
index_root="$work/waldo-index"
lookaside="$work/lookaside"
staging="$work/staging"
models="$work/models"
source_root="$work/source"
input="$source_root/raw/training.jsonl"
compose="$work/model.yaml"
provider="$work/provider.json"
huggingface_export="$work/huggingface-export"
gguf_export="$work/gguf-export"
export WALDO_CONFIG="$work/config.json"

echo "testing: real PyTorch model lifecycle with $torch_python"
(cd "$repo_root" && GOCACHE="$work/go-cache" go build -o "$binary" ./cmd/waldo)
mkdir -p "$source_root/raw"
cat > "$input" <<'EOF'
{"text":"OpenWALDO trains real weights through PyTorch. This record validates the Linux backend."}
{"text":"Internal checkpoints preserve exact FP32 master weights for deterministic continuation."}
{"text":"Held-out evaluation measures records that optimizer updates never consume."}
{"text":"Portable artifacts are evaluated in their declared parameter representation before publication."}
EOF
file_bytes=$(wc -c < "$input" | tr -d ' ')
if command -v sha256sum >/dev/null 2>&1; then
  file_sha=$(sha256sum "$input" | awk '{print $1}')
  tree_sha=$(printf '%s\t%s\t%s\n' "$file_sha" "$file_bytes" training.jsonl | sha256sum | awk '{print $1}')
else
  file_sha=$(shasum -a 256 "$input" | awk '{print $1}')
  tree_sha=$(printf '%s\t%s\t%s\n' "$file_sha" "$file_bytes" training.jsonl | shasum -a 256 | awk '{print $1}')
fi
cat > "$source_root/manifest.json" <<EOF
{
  "kind":"waldo-source-directory","schema":1,"retrieved_at":"2026-09-13T00:00:00Z",
  "corpus":{"id":"pytorch-e2e","title":"PyTorch-E2E-Corpus","description":"Disposable real PyTorch training input."},
  "sources":[{"id":"pytorch-e2e","path":"","license":"CC0-1.0","source":{"name":"pytorch","version":"fixture-1","url":"https://example.invalid/pytorch-e2e","category":"public-dataset","license_evidence":{"declaration":"CC0-1.0"}},"input":{"format":"jsonl","type":"record-map","fields":{"text":["text"]}},"artifacts":[]}],
  "fetcher":{"name":"pytorch-e2e"},
  "raw":{"path":"raw","file_count":1,"byte_count":$file_bytes,"tree_sha256":"$tree_sha"}
}
EOF

"$binary" index init "$index_root" >/dev/null
"$binary" config set lookaside "file://$lookaside" >/dev/null
"$binary" config set lookaside.cache "$work/cache" >/dev/null
"$binary" config set lookaside.scratch "$work/scratch" >/dev/null
"$binary" config set ingest.staging "$staging" >/dev/null
"$binary" config set model.root "$models" >/dev/null
"$binary" config set model.backend pytorch >/dev/null
"$binary" config set index "$index_root" >/dev/null
cat > "$provider" <<EOF
{
  "kind": "waldo-disclosure-provider",
  "schema": 1,
  "provider": {"name": "OpenWALDO PyTorch E2E", "address": "Local test", "contact": "test@example.invalid"},
  "code_of_practice_status": "not-assessed",
  "copyright_policy_url": "https://example.invalid/copyright"
}
EOF
"$binary" config set disclosure.provider "$provider" >/dev/null

destination="$index_root/core/e2e/pytorch"
"$binary" index ingest "$source_root" "$destination" >/dev/null

contribution=""
for candidate in "$staging"/*/contribution; do
  [ -d "$candidate" ] || continue
  contribution=$candidate
done
[ -n "$contribution" ] || { echo "PyTorch contribution overlay not found" >&2; exit 1; }
cp -R "$contribution"/. "$index_root"/

cat > "$compose" <<EOF
kind: waldo-model-compose
schema: 1
architecture:
  family: decoder-transformer
  context_tokens: 16
  vocabulary_size: 259
  hidden_size: 32
  intermediate_size: 64
  layers: 1
  attention_heads: 4
  key_value_heads: 2
  tie_embeddings: true
  parameter_dtype: bfloat16
  tokenizer:
    name: byte
    revision: builtin-byte-schema-1
stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    corpora:
      - core/e2e/pytorch
    parameters:
      steps: 2
      batch_size: 1
      sequence_length: 16
      learning_rate: 0.001
      seed: 7
      checkpoint_every: 1
      evaluate_every: 1
EOF

output=$("$binary" model train pytorch-smoke "$compose")
printf '%s\n' "$output"
printf '%s\n' "$output" | grep -q 'backend       pytorch@'"$revision"''
summary=$("$binary" --json model summary pytorch-smoke)
printf '%s\n' "$summary" | grep -Eq '"simulated"[[:space:]]*:[[:space:]]*false'
printf '%s\n' "$summary" | grep -Eq '"name"[[:space:]]*:[[:space:]]*"pytorch"'
printf '%s\n' "$summary" | grep -Eq '"selected_checkpoint"[[:space:]]*:[[:space:]]*\{'
printf '%s\n' "$summary" | grep -Eq '"publishable_checkpoint_heldout_loss"[[:space:]]*:'
printf '%s\n' "$summary" | grep -Eq '"artifact_heldout_loss"[[:space:]]*:'

train_output=$("$binary" model train pytorch-smoke core/e2e/pytorch --epochs 2)
printf '%s\n' "$train_output" | grep -q 'backend       pytorch@'"$revision"''
run_count=$(find "$models/pytorch-smoke/runs" -type f -name RUN.json -print | wc -l | tr -d ' ')
[ "$run_count" -eq 2 ] || { echo "found $run_count PyTorch runs, want 2" >&2; exit 1; }
grep -ERq '"initialization"[[:space:]]*:' "$models/pytorch-smoke/runs" || { echo "continued PyTorch run did not pin initialization" >&2; exit 1; }
checkpoint_count=$(find "$models/pytorch-smoke/runs" -type d -name 'step-*' -print | wc -l | tr -d ' ')
[ "$checkpoint_count" -ge 3 ] || { echo "found $checkpoint_count PyTorch checkpoints, want at least 3" >&2; exit 1; }
find "$models/pytorch-smoke/runs" -type d -name 'step-*' -exec test -f '{}/model.safetensors' \; -exec test -f '{}/runtime.pt' \; -exec test -f '{}/state.json' \;
current_weights=$(find "$models/pytorch-smoke/runs" -type f -name model.safetensors ! -path '*/checkpoints/*' -print | sort | tail -1)
[ -s "$current_weights" ] || { echo "real PyTorch weights were not produced" >&2; exit 1; }
current_checkpoint=$(find "$models/pytorch-smoke/runs" -type f -path '*/checkpoints/step-*/model.safetensors' -print | sort | tail -1)
[ -s "$current_checkpoint" ] || { echo "real PyTorch checkpoint weights were not produced" >&2; exit 1; }

"$binary" model export pytorch-smoke "$huggingface_export" --format huggingface --allow-incomplete >/dev/null
"$binary" model export pytorch-smoke "$gguf_export" --format gguf --allow-incomplete >/dev/null
"$torch_python" - "$current_weights" "$huggingface_export" "$gguf_export" <<'PY'
import hashlib
import json
import os
import struct
import sys

source, huggingface_root, gguf_root = sys.argv[1:]

def tensor_payload(path):
    with open(path, "rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        payload = hashlib.sha256(stream.read()).hexdigest()
    return header, payload

source_header, source_payload = tensor_payload(source)
assert source_header["__metadata__"]["backend"] == "pytorch"
assert source_header["__metadata__"]["storage_dtype"] == "bfloat16"
assert "embedding.weight" in source_header
assert source_header["embedding.weight"]["dtype"] == "BF16"
target_header, target_payload = tensor_payload(os.path.join(huggingface_root, "model.safetensors"))
assert source_payload == target_payload
assert target_header["__metadata__"]["format"] == "pt"
assert "model.embed_tokens.weight" in target_header
with open(os.path.join(gguf_root, "model.gguf"), "rb") as stream:
    assert stream.read(4) == b"GGUF"
    assert struct.unpack("<I", stream.read(4))[0] == 3
for root in (huggingface_root, gguf_root):
    with open(os.path.join(root, "BOM.json"), encoding="utf-8") as stream:
        bom = json.load(stream)
    for item in bom["artifacts"]:
        with open(os.path.join(root, item["path"]), "rb") as stream:
            data = stream.read()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
PY

"$torch_python" - "$current_checkpoint" <<'PY'
import json
import struct
import sys

with open(sys.argv[1], "rb") as stream:
    length = struct.unpack("<Q", stream.read(8))[0]
    header = json.loads(stream.read(length))
assert header["__metadata__"]["storage_dtype"] == "float32"
assert header["embedding.weight"]["dtype"] == "F32"
PY

echo "E2E PyTorch model passed: trained, continued, and exported Hugging Face and GGUF packages"
