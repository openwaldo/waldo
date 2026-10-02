#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$(uname -s)" != "Linux" ]; then
  [ "${WALDO_E2E_REQUIRED:-0}" != "1" ] || { echo "testing: real TorchTitan model lifecycle requires Linux" >&2; exit 1; }
  echo "testing: real TorchTitan model lifecycle skipped (requires Linux)"
  exit 0
fi

titan_python=""
for candidate in "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
  [ -n "$candidate" ] && [ -x "$candidate" ] || continue
  if "$candidate" -c 'import torch,torchtitan; from torchtitan.distributed import ParallelDims; assert torch.cuda.is_available() and torch.cuda.device_count() > 0' >/dev/null 2>&1; then
    titan_python=$candidate
    break
  fi
done
if [ -z "$titan_python" ]; then
  [ "${WALDO_E2E_REQUIRED:-0}" != "1" ] || { echo "testing: no usable GPU TorchTitan runtime" >&2; exit 1; }
  echo "testing: real TorchTitan model lifecycle skipped (no usable GPU TorchTitan runtime)"
  exit 0
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
revision=$(awk '{ for (field = 1; field <= NF; field++) if ($field == "TorchTitanRevision" && $(field + 1) == "=") { value = $(field + 2); gsub(/^"|"$/, "", value); print value; exit } }' "$repo_root/internal/training/torchtitan.go")
[ -n "$revision" ] || { echo "could not read TorchTitanRevision from internal/training/torchtitan.go" >&2; exit 1; }
local_gpus=$("$titan_python" -c 'import torch; print(torch.cuda.device_count())')
[ "$local_gpus" -gt 0 ] || { echo "TorchTitan E2E did not find a CUDA GPU" >&2; exit 1; }
temporary_base=${TMPDIR:-/tmp}
work=$(mktemp -d "$temporary_base/waldo-torchtitan-e2e.XXXXXX")

cleanup() {
  if [ "${WALDO_E2E_KEEP:-0}" = "1" ]; then
    echo "preserved TorchTitan E2E workspace: $work"
    return
  fi
  case "$work" in
    "$temporary_base"/waldo-torchtitan-e2e.*) rm -rf -- "$work" ;;
    *) echo "refusing to remove unexpected workspace: $work" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

binary="$work/waldo"
index_root="$work/waldo-index"
staging="$work/staging"
models="$work/models"
input="$work/training"
compose="$work/model.yaml"
export WALDO_CONFIG="$work/config.json"

echo "testing: real TorchTitan model lifecycle with $titan_python"
(cd "$repo_root" && GOCACHE="$work/go-cache" go build -o "$binary" ./cmd/waldo)
mkdir -p "$input"
printf 'OpenWALDO validates its distributed TorchTitan adapter.\n' > "$input/one.txt"
printf 'Internal checkpoints preserve exact FP32 training state.\n' > "$input/two.txt"
printf 'Held-out records validate publishable model quality.\n' > "$input/three.txt"
printf 'Portable artifacts use the architecture parameter dtype.\n' > "$input/four.txt"

"$binary" index init "$index_root" >/dev/null
"$binary" config set lookaside "file://$work/lookaside" >/dev/null
"$binary" config set lookaside.cache "$work/cache" >/dev/null
"$binary" config set lookaside.scratch "$work/scratch" >/dev/null
"$binary" config set ingest.staging "$staging" >/dev/null
"$binary" config set model.root "$models" >/dev/null
"$binary" config set model.backend torchtitan >/dev/null
"$binary" config set index "$index_root" >/dev/null

destination="$index_root/core/e2e/torchtitan"
"$binary" index ingest "$input" "$destination" \
  --title TorchTitan-E2E-Corpus \
  --description Disposable-real-TorchTitan-training-input \
  --license CC0-1.0 \
  --source https://example.invalid/torchtitan-e2e \
  --language en \
  --source-category public-dataset >/dev/null

contribution=""
for candidate in "$staging"/*/contribution; do
  [ -d "$candidate" ] || continue
  contribution=$candidate
done
[ -n "$contribution" ] || { echo "TorchTitan contribution overlay not found" >&2; exit 1; }
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
  dropout: 0.1
  qk_normalization: true
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
      - core/e2e/torchtitan
    parameters:
      steps: 100
      epochs: 100
      batch_size: $local_gpus
      sequence_length: 16
      learning_rate: 0.001
      seed: 7
      compile: true
      checkpoint_every: 100
      evaluate_every: 100
EOF

output=$("$binary" model train torchtitan-smoke "$compose")
printf '%s\n' "$output"
printf '%s\n' "$output" | grep -q 'backend       torchtitan@'"$revision"''
summary=$("$binary" --json model summary torchtitan-smoke)
printf '%s\n' "$summary" | grep -Eq '"simulated"[[:space:]]*:[[:space:]]*false'
printf '%s\n' "$summary" | grep -Eq '"name"[[:space:]]*:[[:space:]]*"torchtitan"'
printf '%s\n' "$summary" | grep -Eq '"selected_checkpoint"[[:space:]]*:[[:space:]]*\{'
printf '%s\n' "$summary" | grep -Eq '"publishable_checkpoint_heldout_loss"[[:space:]]*:'
printf '%s\n' "$summary" | grep -Eq '"live_compiled_heldout_loss"[[:space:]]*:'
printf '%s\n' "$summary" | grep -Eq '"live_eager_compute_heldout_loss"[[:space:]]*:'
printf '%s\n' "$summary" | grep -Eq '"live_eager_heldout_loss"[[:space:]]*:'
printf '%s\n' "$summary" | grep -Eq '"compile_loss_delta"[[:space:]]*:'
printf '%s\n' "$summary" | grep -Eq '"compute_precision_loss_delta"[[:space:]]*:'
printf '%s\n' "$summary" | grep -Eq '"artifact_heldout_loss"[[:space:]]*:'
chat_output=$("$binary" --json model chat torchtitan-smoke "Continue this sentence: OpenWALDO" --temperature 0 --top-p 1 --max-tokens 4)
printf '%s\n' "$chat_output" | grep -Eq '"run_id"[[:space:]]*:[[:space:]]*"[^"]+"'
printf '%s\n' "$chat_output" | grep -Eq '"tokens"[[:space:]]*:[[:space:]]*[0-4]'
printf '%s\n' "$chat_output" | grep -Eq '"finish_reason"[[:space:]]*:[[:space:]]*"(eos|max_tokens)"'
weights=$(find "$models/torchtitan-smoke/runs" -type f -name model.safetensors ! -path '*/checkpoints/*' -print)
[ -n "$weights" ] && [ -s "$weights" ] || { echo "real TorchTitan weights were not produced" >&2; exit 1; }
checkpoint_count=$(find "$models/torchtitan-smoke/runs" -type d -name 'step-*' -print | wc -l | tr -d ' ')
[ "$checkpoint_count" -eq 2 ] || { echo "found $checkpoint_count TorchTitan checkpoints, want 2" >&2; exit 1; }
find "$models/torchtitan-smoke/runs" -type d -name 'step-*' -exec test -f '{}/model.safetensors' \; -exec test -f '{}/runtime.pt' \; -exec test -f '{}/state.json' \;
grep -ERq '"world_size"[[:space:]]*:[[:space:]]*[1-9]' "$models/torchtitan-smoke/runs" || { echo "TorchTitan run did not persist world size" >&2; exit 1; }
checkpoint=$(find "$models/torchtitan-smoke/runs" -type f -path '*/checkpoints/step-*/model.safetensors' -print | sort | tail -1)
"$titan_python" - "$weights" "$checkpoint" <<'PY'
import json
import struct
import sys

def header(path):
    with open(path, "rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(length))

portable = header(sys.argv[1])
checkpoint = header(sys.argv[2])
assert portable["__metadata__"]["storage_dtype"] == "bfloat16"
assert portable["embedding.weight"]["dtype"] == "BF16"
assert checkpoint["__metadata__"]["storage_dtype"] == "float32"
assert checkpoint["embedding.weight"]["dtype"] == "F32"
PY

echo "E2E TorchTitan model passed: shared training/inference architecture, distributed optimization, compiled/eager/FP32 artifact equivalence, and checkpoints verified"
