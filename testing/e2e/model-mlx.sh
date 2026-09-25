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
revision=$(awk '{ for (field = 1; field <= NF; field++) if ($field == "MLXRevision" && $(field + 1) == "=") { value = $(field + 2); gsub(/^"|"$/, "", value); print value; exit } }' "$repo_root/internal/training/mlx.go")
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
source_root="$work/source"
input="$source_root/raw/training.jsonl"
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
mkdir -p "$source_root/raw"
cat > "$input" <<'EOF'
{"text":"OpenWALDO trains real weights through MLX. This record validates the complete backend."}
{"text":"Gradient accumulation preserves the logical optimizer batch while using smaller forward passes."}
{"text":"Held-out evaluation measures deterministic records that optimizer updates never consume."}
{"text":"A completed stage publishes the best evaluated checkpoint and verifies its serialized artifact."}
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
  "corpus":{"id":"mlx-e2e","title":"MLX-E2E-Corpus","description":"Disposable real MLX training input."},
  "sources":[{"id":"mlx-e2e","path":"","license":"CC0-1.0","source":{"name":"mlx","version":"fixture-1","url":"https://example.invalid/mlx-e2e","category":"public-dataset","license_evidence":{"declaration":"CC0-1.0"}},"input":{"format":"jsonl","type":"record-map","fields":{"text":["text"]}},"artifacts":[]}],
  "fetcher":{"name":"mlx-e2e"},
  "raw":{"path":"raw","file_count":1,"byte_count":$file_bytes,"tree_sha256":"$tree_sha"}
}
EOF

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
"$binary" index ingest "$source_root" "$destination" >/dev/null

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
      batch_size: 2
      gradient_accumulation_steps: 2
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
printf '%s\n' "$summary" | grep -Eq '"selected_checkpoint"[[:space:]]*:[[:space:]]*\{'
printf '%s\n' "$summary" | grep -Eq '"artifact_heldout_loss"[[:space:]]*:'
telemetry=$(find "$models/mlx-smoke/runs" -type f -name TELEMETRY.csv -print | sort | head -1)
[ -n "$telemetry" ] || { echo "MLX run did not persist telemetry" >&2; exit 1; }
awk -F, '
  NR == 1 {
    for (column = 1; column <= NF; column++) columns[$column] = column
    next
  }
  $(columns["event"]) == "progress" {
    found = 1
    required[1] = "duration_seconds"
    required[2] = "data_wait_seconds"
    required[3] = "peak_memory_bytes"
    required[4] = "training_flops"
    required[5] = "achieved_tflops"
    required[6] = "gradient_norm"
    for (position = 1; position <= 6; position++) {
      name = required[position]
      if (!(name in columns) || $(columns[name]) == "") exit 1
    }
  }
  END { if (!found) exit 1 }
' "$telemetry" || { echo "MLX progress telemetry is incomplete" >&2; exit 1; }
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

interrupted_log="$work/interrupted-training.log"
WALDO_GPU_THROTTLE=0.01 "$binary" model train mlx-resume "$compose" >"$interrupted_log" 2>&1 &
interrupted_pid=$!
checkpoint_state=""
poll=0
while [ "$poll" -lt 200 ]; do
  checkpoint_state=$(find "$models/mlx-resume/runs" -path '*/checkpoints/step-00000001/state.json' -print 2>/dev/null | head -1)
  [ -n "$checkpoint_state" ] && break
  if ! kill -0 "$interrupted_pid" 2>/dev/null; then
    break
  fi
  sleep 0.05
  poll=$((poll + 1))
done
[ -n "$checkpoint_state" ] || { cat "$interrupted_log"; echo "MLX interruption test did not reach checkpoint 1" >&2; exit 1; }
# Give the parent enough time to commit the worker's checkpoint event, while
# throttling guarantees that the next optimizer step cannot finish first.
sleep 0.1
kill -INT "$interrupted_pid"
set +e
wait "$interrupted_pid"
interrupted_code=$?
set -e
[ "$interrupted_code" -ne 0 ] || { echo "interrupted MLX training unexpectedly completed" >&2; exit 1; }

resume_output=$("$binary" model train mlx-resume "$compose")
printf '%s\n' "$resume_output"
grep -ERq '"resume_step"[[:space:]]*:[[:space:]]*1' "$models/mlx-resume/runs" || {
  echo "completed MLX run does not record checkpoint resume from step 1" >&2
  exit 1
}
control_output=$("$binary" model train mlx-control "$compose")
printf '%s\n' "$control_output"
resumed_weights=$(find "$models/mlx-resume/runs" -type f -name model.safetensors ! -path '*/checkpoints/*' -print | head -1)
control_weights=$(find "$models/mlx-control/runs" -type f -name model.safetensors ! -path '*/checkpoints/*' -print | head -1)
"$mlx_python" - "$resumed_weights" "$control_weights" <<'PY'
import sys

import mlx.core as mx

resumed = mx.load(sys.argv[1])
control = mx.load(sys.argv[2])
assert resumed.keys() == control.keys()
for name in resumed:
    assert mx.array_equal(resumed[name], control[name]).item(), name
PY

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

echo "E2E MLX model passed: trained, deterministically checkpoint-resumed, generated, and exported Hugging Face, MLX, GGUF, and Ollama packages"
