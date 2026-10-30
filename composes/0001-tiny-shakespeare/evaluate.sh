#!/usr/bin/env bash
# Evaluate ladder rung 0001 with fixed Shakespeare prompts.
set -euo pipefail

if (($# < 2 || $# > 6)); then
  echo "usage: $0 MODEL OUTPUT.jsonl [TEMPERATURE] [SEED] [RUN_ID] [CHECKPOINT_STEP]" >&2
  exit 2
fi

model="$1"
output="$2"
temperature="${3:-0.8}"
seed="${4:-42}"
run_id="${5:-}"
checkpoint_step="${6:-}"
if [[ -z "$run_id" ]]; then
  [[ -z "$checkpoint_step" ]] || { echo "CHECKPOINT_STEP requires RUN_ID" >&2; exit 2; }
  run_id="$(go run ./cmd/waldo/ --json model summary "$model" | jq -r '.bom.current_run_id')"
fi
selector=(--run-id "$run_id")
[[ -z "$checkpoint_step" ]] || selector+=(--checkpoint-step "$checkpoint_step")

prompts=(
  $'ROMEO:\n'
  $'JULIET:\n'
  $'KING RICHARD II:\n'
  $'First Citizen:\n'
  $'DUKE VINCENTIO:\n'
  $'HAMLET:\n'
  $'To be, or not to be'
  $'What light through yonder window breaks?'
)

: >"$output"
for prompt in "${prompts[@]}"; do
  go run ./cmd/waldo/ --json model chat "$model" \
    "${selector[@]}" --raw --temperature "$temperature" --seed "$seed" \
    --max-tokens 256 "$prompt" >>"$output"
done

jq -s '{
  responses: length,
  eos: map(select(.result.finish_reason == "eos")) | length,
  max_tokens: map(select(.result.finish_reason == "max_tokens")) | length,
  results: map({prompt, tokens: .result.tokens, finish_reason: .result.finish_reason, text: .result.text})
}' "$output"
