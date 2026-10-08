#!/usr/bin/env bash
# Evaluate ladder rung 0001 with fixed Shakespeare prompts.
set -euo pipefail

if (($# < 2 || $# > 4)); then
  echo "usage: $0 MODEL OUTPUT.jsonl [TEMPERATURE] [SEED]" >&2
  exit 2
fi

model="$1"
output="$2"
temperature="${3:-0.8}"
seed="${4:-42}"
run_id="$(go run ./cmd/waldo/ --json model summary "$model" | jq -r '.bom.current_run_id')"

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
    --run-id "$run_id" --raw --temperature "$temperature" --seed "$seed" \
    --max-tokens 256 "$prompt" >>"$output"
done

jq -s '{
  responses: length,
  eos: map(select(.result.finish_reason == "eos")) | length,
  max_tokens: map(select(.result.finish_reason == "max_tokens")) | length,
  results: map({prompt, tokens: .result.tokens, finish_reason: .result.finish_reason, text: .result.text})
}' "$output"
