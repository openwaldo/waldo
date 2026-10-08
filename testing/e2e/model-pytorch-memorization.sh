#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$(uname -s)" != "Linux" ]; then
  [ "${WALDO_E2E_REQUIRED:-0}" != "1" ] || { echo "testing: PyTorch memorization control requires Linux" >&2; exit 1; }
  echo "testing: PyTorch memorization control skipped (requires Linux)"
  exit 0
fi

torch_python=""
for candidate in "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
  [ -n "$candidate" ] && [ -x "$candidate" ] || continue
  if "$candidate" -c 'import torch; assert torch.cuda.is_available(); assert torch.tensor([1.0], device="cuda").sum().item() == 1.0' >/dev/null 2>&1; then
    torch_python=$candidate
    break
  fi
done
if [ -z "$torch_python" ]; then
  [ "${WALDO_E2E_REQUIRED:-0}" != "1" ] || { echo "testing: PyTorch memorization control requires a CUDA PyTorch runtime" >&2; exit 1; }
  echo "testing: PyTorch memorization control skipped (no CUDA PyTorch runtime)"
  exit 0
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
temporary_base=${TMPDIR:-/tmp}
work=$(mktemp -d "$temporary_base/waldo-pytorch-memorization.XXXXXX")

cleanup() {
  if [ "${WALDO_E2E_KEEP:-0}" = "1" ]; then
    echo "preserved PyTorch memorization workspace: $work"
    return
  fi
  case "$work" in
    "$temporary_base"/waldo-pytorch-memorization.*) rm -rf -- "$work" ;;
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
compose="$work/memorization.yaml"
summary_json="$work/summary.json"
chat_json="$work/chat.json"
export WALDO_CONFIG="$work/config.json"

echo "testing: compose-driven PyTorch memorization control with $torch_python"
(cd "$repo_root" && GOCACHE="$work/go-cache" go build -o "$binary" ./cmd/waldo)
mkdir -p "$source_root/raw"
cat > "$input" <<'EOF'
{"text":"A|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"B|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"C|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"D|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"E|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"F|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"G|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"H|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"I|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"J|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"K|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"L|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"M|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"N|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"O|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
{"text":"P|OpenWALDO memorization control: red green blue. The fixed answer is sapphire. END"}
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
  "kind":"waldo-source-directory","schema":1,"retrieved_at":"2026-10-04T00:00:00Z",
  "corpus":{"id":"pytorch-memorization","title":"PyTorch memorization control","description":"Disposable deterministic learning control."},
  "sources":[{"id":"pytorch-memorization","path":"","license":"CC0-1.0","source":{"name":"pytorch-memorization","version":"fixture-1","url":"https://example.invalid/pytorch-memorization","category":"public-dataset","license_evidence":{"declaration":"CC0-1.0"}},"input":{"format":"jsonl","type":"record-map","fields":{"text":["text"]}},"artifacts":[]}],
  "fetcher":{"name":"pytorch-memorization"},
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

destination="$index_root/core/e2e/memorization"
"$binary" index ingest "$source_root" "$destination" >/dev/null
contribution=""
for candidate in "$staging"/*/contribution; do
  [ -d "$candidate" ] || continue
  contribution=$candidate
done
[ -n "$contribution" ] || { echo "memorization contribution overlay not found" >&2; exit 1; }
cp -R "$contribution"/. "$index_root"/

cat > "$compose" <<'EOF'
kind: waldo-model-compose
schema: 1
architecture:
  family: decoder-transformer
  context_tokens: 96
  vocabulary_size: 259
  hidden_size: 64
  intermediate_size: 192
  layers: 2
  attention_heads: 4
  key_value_heads: 2
  dropout: 0.0
  qk_normalization: false
  initialization: depth-scaled
  tie_embeddings: true
  parameter_dtype: float32
  tokenizer:
    name: byte
    revision: builtin-byte-schema-1
stages:
  - name: memorization-control
    type: pre-training
    objective: causal-language-modeling
    corpora:
      - core/e2e/memorization
    parameters:
      steps: 800
      epochs: 1000
      batch_size: 8
      gradient_accumulation_steps: 1
      compute_precision: float32
      activation_checkpointing: false
      compile: false
      sequence_length: 96
      learning_rate: 0.003
      optimizer: adamw
      schedule: warmup-stable-warmdown
      seed: 7
      weight_decay: 0.0
      warmup_steps: 20
      warmdown_steps: 80
      checkpoint_every: 100
      evaluate_every: 100
      evaluation_max_records: 1
      evaluation_max_bytes: 1048576
EOF

"$binary" model forecast "$compose"
"$binary" model train pytorch-memorization-control "$compose"
"$binary" --json model summary pytorch-memorization-control > "$summary_json"

"$torch_python" - "$summary_json" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
observation = summary["runs"][-1]["observation"]
losses = [
    item["metrics"]["heldout_loss"]
    for item in observation["evaluations"]
    if item["step"] > 0 and "heldout_loss" in item["metrics"]
]
assert len(losses) >= 2, f"need at least two held-out measurements, got {losses}"
initial = losses[0]
best = min(losses)
selected = observation["selected_checkpoint"]
assert all(math.isfinite(value) for value in losses), losses
assert best <= 0.25, f"best held-out loss {best:.6f} exceeds 0.25"
assert best <= initial * 0.10, f"held-out loss fell only from {initial:.6f} to {best:.6f}"
assert selected["metric"] == "heldout_loss"
assert abs(selected["value"] - best) <= 1e-5, (selected, best)
print(f"memorization loss gate passed: initial={initial:.6f}, best={best:.6f}, selected_step={selected['step']}")
PY

"$binary" --json model chat pytorch-memorization-control \
  --raw --temperature 0 --top-p 1 --max-tokens 32 \
  "OpenWALDO memorization control: red green blue. The fixed answer is" > "$chat_json"

"$torch_python" - "$chat_json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    document = json.load(stream)
result = document["result"]
text = result["text"].lstrip()
assert text.startswith("sapphire. END"), repr(result["text"])
assert result["finish_reason"] == "eos", result
print(f"memorization generation gate passed: {result['text']!r} ({result['finish_reason']})")
PY

echo "PyTorch memorization control passed: the compose learned the held-out continuation and emitted EOS"
