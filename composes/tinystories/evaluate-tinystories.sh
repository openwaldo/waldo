#!/usr/bin/env bash
set -euo pipefail

if (($# < 2 || $# > 4)); then
  echo "usage: $0 MODEL OUTPUT.jsonl [TEMPERATURE] [SEED]" >&2
  exit 2
fi

model="$1"
output="$2"
temperature="${3:-0}"
seed="${4:-42}"
run_id="$(go run ./cmd/waldo/ --json model summary "$model" | jq -r '.bom.current_run_id')"

prompts=(
  "Once upon a time, a little girl named Lucy found a tiny blue box."
  "One sunny morning, Tom saw a red ball under a tree."
  "Lily was afraid of the dark, but one night she heard a soft sound."
  "Ben wanted to help his mother bake a cake."
  "A small rabbit lived beside a quiet pond."
  "Mia planted a seed and checked it every day."
  "The little dog could not find its favorite toy."
  "Sam and Anna built a boat from a cardboard box."
  "A yellow bird wanted to learn how to sing."
  "Jack opened the garden gate and discovered something surprising."
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
