#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "testing: real MLX model lifecycle skipped (requires Apple Silicon)"
  exit 0
fi

mlx_python=""
for candidate in "$(command -v python3 2>/dev/null || true)" /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  [ -n "$candidate" ] && [ -x "$candidate" ] || continue
  if "$candidate" -c 'import mlx.core as mx; mx.eval(mx.array([1]))' >/dev/null 2>&1; then
    mlx_python=$candidate
    break
  fi
done
if [ -z "$mlx_python" ]; then
  echo "testing: real MLX model lifecycle skipped (no Metal-capable MLX Python runtime)"
  exit 0
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
revision=$(sed -n 's/.*MLXRevision = "\(.*\)".*/\1/p' "$repo_root/internal/training/mlx.go")
[ -n "$revision" ] || { echo "could not read MLXRevision from internal/training/mlx.go" >&2; exit 1; }
temporary_base=${TMPDIR:-/tmp}
work=$(mktemp -d "$temporary_base/waldo-mlx-e2e.XXXXXX")

cleanup() {
  if [ "${WALDO_E2E_KEEP:-0}" = "1" ]; then
    echo "preserved MLX E2E workspace: $work"
    return
  fi
  case "$work" in
    "$temporary_base"/waldo-mlx-e2e.*) rm -rf -- "$work" ;;
    *) echo "refusing to remove unexpected workspace: $work" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

binary="$work/waldo"
index_root="$work/waldo-index"
lookaside="$work/lookaside"
staging="$work/staging"
models="$work/models"
input="$work/training.txt"
compose="$work/model.yaml"
provider="$work/provider.json"
huggingface_export="$work/huggingface-export"
mlx_export="$work/mlx-export"
gguf_export="$work/gguf-export"
ollama_export="$work/ollama-export"
quantized_export="$work/quantized-export"
export WALDO_CONFIG="$work/config.json"

echo "testing: real MLX model lifecycle with $mlx_python"
(cd "$repo_root" && GOCACHE="$work/go-cache" go build -o "$binary" ./cmd/waldo)
printf 'OpenWALDO trains real weights through MLX.\nThis tiny record exists only to validate the complete backend.\n' > "$input"

"$binary" index init "$index_root" >/dev/null
"$binary" config set lookaside "file://$lookaside" >/dev/null
"$binary" config set lookaside.cache "$work/cache" >/dev/null
"$binary" config set lookaside.scratch "$work/scratch" >/dev/null
"$binary" config set ingest.staging "$staging" >/dev/null
"$binary" config set model.root "$models" >/dev/null
"$binary" config set model.backend auto >/dev/null
"$binary" config set index "$index_root" >/dev/null
cat > "$provider" <<EOF
{
  "kind": "waldo-disclosure-provider",
  "schema": 1,
  "provider": {"name": "OpenWALDO MLX E2E", "address": "Local test", "contact": "test@example.invalid"},
  "code_of_practice_status": "not-assessed",
  "copyright_policy_url": "https://example.invalid/copyright"
}
EOF
"$binary" config set disclosure.provider "$provider" >/dev/null

destination="$index_root/core/e2e/mlx"
"$binary" index ingest "$input" "$destination" \
  --title MLX-E2E-Corpus \
  --description Disposable-real-MLX-training-input \
  --license CC0-1.0 \
  --source https://example.invalid/mlx-e2e \
  --source-category public-dataset >/dev/null

contribution=""
for candidate in "$staging"/*/contribution; do
  [ -d "$candidate" ] || continue
  contribution=$candidate
done
[ -n "$contribution" ] || { echo "MLX contribution overlay not found" >&2; exit 1; }
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
      - core/e2e/mlx
    parameters:
      steps: 2
      batch_size: 1
      sequence_length: 16
      learning_rate: 0.001
      seed: 7
      checkpoint_every: 1
      evaluate_every: 1
EOF

output=$("$binary" model train mlx-smoke "$compose")
printf '%s\n' "$output"
printf '%s\n' "$output" | grep -q 'backend       mlx@'"$revision"''
summary=$("$binary" --json model summary mlx-smoke)
printf '%s\n' "$summary" | grep -Eq '"simulated"[[:space:]]*:[[:space:]]*false'
printf '%s\n' "$summary" | grep -Eq '"name"[[:space:]]*:[[:space:]]*"mlx"'
weights=$(find "$models/mlx-smoke/runs" -type f -name model.safetensors ! -path '*/checkpoints/*' -print)
[ -n "$weights" ] && [ -s "$weights" ] || { echo "real MLX weights were not produced" >&2; exit 1; }
checkpoint_count=$(find "$models/mlx-smoke/runs" -type d -name 'step-*' -print | wc -l | tr -d ' ')
[ "$checkpoint_count" -eq 2 ] || { echo "found $checkpoint_count MLX checkpoints, want 2" >&2; exit 1; }
find "$models/mlx-smoke/runs" -type d -name 'step-*' -exec test -f '{}/model.safetensors' \; -exec test -f '{}/optimizer.safetensors' \; -exec test -f '{}/state.json' \;

train_output=$("$binary" model train mlx-smoke core/e2e/mlx --epochs 2)
printf '%s\n' "$train_output"
printf '%s\n' "$train_output" | grep -q 'backend       mlx@'"$revision"''
summary=$("$binary" --json model summary mlx-smoke)
printf '%s\n' "$summary" | grep -Eq '"runs"[[:space:]]*:[[:space:]]*\['
printf '%s\n' "$summary" | grep -Eq '"initialization"[[:space:]]*:'
run_count=$(find "$models/mlx-smoke/runs" -type f -name RUN.json -print | wc -l | tr -d ' ')
[ "$run_count" -eq 2 ] || { echo "found $run_count MLX runs, want 2" >&2; exit 1; }
grep -ERq '"epochs"[[:space:]]*:[[:space:]]*2' "$models/mlx-smoke/runs" || { echo "training run BOM did not persist two epochs" >&2; exit 1; }
weights_count=$(find "$models/mlx-smoke/runs" -type f -name model.safetensors ! -path '*/checkpoints/*' -print | wc -l | tr -d ' ')
[ "$weights_count" -eq 2 ] || { echo "found $weights_count terminal MLX weights, want 2" >&2; exit 1; }
current_weights=$(find "$models/mlx-smoke/runs" -type f -name model.safetensors ! -path '*/checkpoints/*' -print | sort | tail -1)

chat=$("$binary" --json model chat mlx-smoke "OpenWALDO" --max-tokens 2 --temperature 0 --seed 7)
printf '%s\n' "$chat" | grep -Eq '"run_id"[[:space:]]*:[[:space:]]*"[^"]+"'
printf '%s\n' "$chat" | grep -Eq '"tokens"[[:space:]]*:[[:space:]]*[0-2]'
printf '%s\n' "$chat" | grep -Eq '"finish_reason"[[:space:]]*:[[:space:]]*"(eos|max_tokens)"'

"$binary" model export mlx-smoke "$huggingface_export" --format huggingface --allow-incomplete >/dev/null
"$binary" model export mlx-smoke "$mlx_export" --format mlx --allow-incomplete >/dev/null
"$binary" model export mlx-smoke "$gguf_export" --format gguf --allow-incomplete >/dev/null
"$binary" model export mlx-smoke "$ollama_export" --format ollama --allow-incomplete >/dev/null
"$mlx_python" - "$current_weights" "$huggingface_export" "$mlx_export" "$gguf_export" "$ollama_export" <<'PY'
import hashlib
import json
import os
import struct
import sys

source, huggingface_root, mlx_root, gguf_root, ollama_root = sys.argv[1:]

def tensor_payload(path):
    with open(path, "rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        payload = hashlib.sha256(stream.read()).hexdigest()
    return header, payload

_, source_payload = tensor_payload(source)
for root, release_format, container_format in (
    (huggingface_root, "huggingface", "pt"),
    (mlx_root, "mlx", "mlx"),
):
    target_header, target_payload = tensor_payload(os.path.join(root, "model.safetensors"))
    assert source_payload == target_payload
    assert target_header["__metadata__"]["format"] == container_format
    assert "model.embed_tokens.weight" in target_header
    assert "embedding.weight" not in target_header
    for name in ("architecture.py", "tokenization_openwaldo.py"):
        with open(os.path.join(root, name), encoding="utf-8") as stream:
            compile(stream.read(), name, "exec")
    with open(os.path.join(root, "BOM.json"), encoding="utf-8") as stream:
        bom = json.load(stream)
    assert bom["format"] == release_format
    for item in bom["artifacts"]:
        with open(os.path.join(root, item["path"]), "rb") as stream:
            data = stream.read()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]

for root, release_format in ((gguf_root, "gguf"), (ollama_root, "ollama")):
    with open(os.path.join(root, "model.gguf"), "rb") as stream:
        assert stream.read(4) == b"GGUF"
        assert struct.unpack("<I", stream.read(4))[0] == 3
    with open(os.path.join(root, "BOM.json"), encoding="utf-8") as stream:
        bom = json.load(stream)
    assert bom["format"] == release_format
    for item in bom["artifacts"]:
        with open(os.path.join(root, item["path"]), "rb") as stream:
            data = stream.read()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]

with open(os.path.join(ollama_root, "Modelfile"), encoding="utf-8") as stream:
    modelfile = stream.read()
assert "FROM ./model.gguf\n" in modelfile
assert "PARAMETER num_ctx 16\n" in modelfile
PY

if command -v llama-quantize >/dev/null 2>&1 && command -v llama-imatrix >/dev/null 2>&1; then
  "$binary" model export mlx-smoke "$quantized_export" \
    --format gguf --quant 4 --calibration core/e2e/mlx --allow-incomplete >/dev/null
  "$mlx_python" - "$quantized_export" <<'PY'
import json
import os
import sys

root = sys.argv[1]
with open(os.path.join(root, "BOM.json"), encoding="utf-8") as stream:
    bom = json.load(stream)
quant = bom["quantization"]
assert quant["requested"] == "4"
assert quant["resolved"] == "Q4_K_M"
assert quant["quantizer"]["name"] == "llama-quantize"
assert quant["calibrator"]["name"] == "llama-imatrix"
# llama-quantize reports no version, so the digest is the only identity the
# release carries for it. It must never be empty.
assert len(quant["quantizer"]["sha256"]) == 64
assert len(quant["calibrator"]["sha256"]) == 64
calibration = quant["calibration"]
assert calibration["sampled_tokens"] > 0
assert calibration["records"] > 0
assert calibration["shards"] == 1
assert calibration["evidence"]["subject"] == "quantization-calibration"
assert len(calibration["evidence"]["shards"]) == 1
assert os.path.getsize(os.path.join(root, "model.gguf")) > 0
assert not os.path.exists(os.path.join(root, ".waldo-high-precision.gguf"))
assert not os.path.exists(os.path.join(root, ".waldo-imatrix.gguf"))
PY
else
  echo "testing: calibrated GGUF export skipped (llama-quantize and llama-imatrix not both installed)"
fi

echo "E2E MLX model passed: trained, resumed, generated, and exported Hugging Face, MLX, GGUF, and Ollama packages"
