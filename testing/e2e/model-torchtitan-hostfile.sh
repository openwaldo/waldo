#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

set -eu

[ "$#" -eq 2 ] || { echo "usage: $0 HOSTFILE SMALL_CONVERSATION_INDEX_PATH" >&2; exit 2; }
hostfile=$1
corpus=$2

[ "$(uname -s)" = "Linux" ] || { echo "hostfile acceptance requires Linux" >&2; exit 1; }
[ -r "$hostfile" ] || { echo "hostfile is not readable: $hostfile" >&2; exit 1; }
case "$corpus" in
  ""|-*|/*|*..*|*[!A-Za-z0-9_./-]*)
    echo "corpus must be a safe relative WALDO index path: $corpus" >&2
    exit 2
    ;;
esac

nodes=$(awk '!/^[[:space:]]*(#|$)/ { count++ } END { print count+0 }' "$hostfile")
[ "$nodes" -ge 2 ] || { echo "hostfile must list at least two hosts" >&2; exit 1; }

titan_python=""
for candidate in "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
  [ -n "$candidate" ] && [ -x "$candidate" ] || continue
  if "$candidate" -c 'import torch,torchtitan; from torchtitan.distributed import ParallelDims; assert torch.cuda.is_available() and torch.cuda.device_count() > 0' >/dev/null 2>&1; then
    titan_python=$candidate
    break
  fi
done
[ -n "$titan_python" ] || { echo "hostfile acceptance requires a usable GPU TorchTitan runtime" >&2; exit 1; }
local_gpus=$("$titan_python" -c 'import torch; print(torch.cuda.device_count())')
world_size=$((nodes * local_gpus))

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
temporary_base=${TMPDIR:-/tmp}
work=$(mktemp -d "$temporary_base/waldo-torchtitan-hostfile-e2e.XXXXXX")
binary="$work/waldo"
compose="$work/compose.yaml"
model_name="hostfile-acceptance-$(date -u +%Y%m%d%H%M%S)-$$"
success=0

cleanup() {
  if [ "$success" -eq 1 ] && [ "${WALDO_E2E_KEEP:-0}" != "1" ]; then
    "$binary" model rm "$model_name" >/dev/null 2>&1 || true
    case "$work" in
      "$temporary_base"/waldo-torchtitan-hostfile-e2e.*) rm -rf -- "$work" ;;
      *) echo "refusing to remove unexpected workspace: $work" >&2 ;;
    esac
  else
    echo "preserved hostfile acceptance model $model_name and workspace $work" >&2
  fi
}
trap cleanup EXIT HUP INT TERM

echo "testing: real $nodes-node/$world_size-GPU TorchTitan hostfile lifecycle using $corpus"
(cd "$repo_root" && GOCACHE="$work/go-cache" go build -o "$binary" ./cmd/waldo)

cat > "$compose" <<EOF
kind: waldo-model-compose
schema: 1
architecture:
  family: decoder-transformer
  context_tokens: 64
  vocabulary_size: 259
  hidden_size: 64
  intermediate_size: 192
  layers: 2
  attention_heads: 4
  key_value_heads: 2
  dropout: 0.1
  qk_normalization: true
  tie_embeddings: true
  parameter_dtype: float32
  tokenizer:
    name: byte
    revision: builtin-byte-schema-1
stages:
  - name: conversation-smoke
    type: fine-tuning
    objective: assistant-response-modeling
    conversation:
      template: user-assistant-v1
      supervised_roles: [assistant]
    corpora:
      - $corpus
    parameters:
      steps: 10
      batch_size: $world_size
      sequence_length: 64
      learning_rate: 0.001
      seed: 7
      compile: false
      checkpoint_every: 5
      evaluate_every: 5
  - name: refine
    type: fine-tuning
    objective: assistant-response-modeling
    conversation:
      template: user-assistant-v1
      supervised_roles: [assistant]
    corpora:
      - $corpus
    parameters:
      steps: 10
      batch_size: $world_size
      sequence_length: 64
      learning_rate: 0.0005
      seed: 8
      compile: false
      checkpoint_every: 5
      evaluate_every: 5
EOF

"$binary" model forecast "$compose" >/dev/null
"$binary" model train "$model_name" "$compose" --hostfile "$hostfile"

summary=$("$binary" --json model summary "$model_name")
printf '%s\n' "$summary" | grep -Eq '"simulated"[[:space:]]*:[[:space:]]*false'
printf '%s\n' "$summary" | grep -Eq '"name"[[:space:]]*:[[:space:]]*"torchtitan"'
printf '%s\n' "$summary" | grep -Eq '"nodes"[[:space:]]*:[[:space:]]*'"$nodes"
printf '%s\n' "$summary" | grep -Eq '"world_size"[[:space:]]*:[[:space:]]*'"$world_size"
printf '%s\n' "$summary" | grep -Eq '"stage"[[:space:]]*:[[:space:]]*"conversation-smoke"'
printf '%s\n' "$summary" | grep -Eq '"stage"[[:space:]]*:[[:space:]]*"refine"'
printf '%s\n' "$summary" | grep -Eq '"publishable_checkpoint_heldout_loss"[[:space:]]*:'
printf '%s\n' "$summary" | grep -Eq '"live_eager_compute_heldout_loss"[[:space:]]*:'
printf '%s\n' "$summary" | grep -Eq '"live_eager_heldout_loss"[[:space:]]*:'
printf '%s\n' "$summary" | grep -Eq '"artifact_heldout_loss"[[:space:]]*:'

chat=$("$binary" --json model chat "$model_name" "OpenWALDO" --max-tokens 4 --temperature 0 --top-p 1)
printf '%s\n' "$chat" | grep -Eq '"run_id"[[:space:]]*:[[:space:]]*"[^"]+"'
printf '%s\n' "$chat" | grep -Eq '"tokens"[[:space:]]*:[[:space:]]*[0-4]'
printf '%s\n' "$chat" | grep -Eq '"finish_reason"[[:space:]]*:[[:space:]]*"(eos|max_tokens)"'

success=1
echo "E2E TorchTitan hostfile passed: SSH staging, $nodes-node rendezvous, $world_size-rank optimization, two-stage handoff, persisted artifact, and rank-0 chat verified"
