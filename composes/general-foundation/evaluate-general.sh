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
  "The Linux kernel is"
  "Linux is an operating system whose kernel was created by"
  "The capital city of France is"
  "At sea level, water freezes at"
  "Earth orbits"
  "A central processing unit (CPU) is"
  "The programming language Python was created by"
  "Two plus two equals"
  "Plants use sunlight to"
  "An operating system manages"
  "Once upon a time"
  "The experiment failed because"
  "To install software on Linux,"
  "A backup is useful because"
  "The scientist compared the results and concluded"
)

: >"$output"
for prompt in "${prompts[@]}"; do
  go run ./cmd/waldo/ --json model chat "$model" \
    --run-id "$run_id" --raw --temperature "$temperature" --seed "$seed" \
    --max-tokens 128 "$prompt" >>"$output"
done

jq -s '{
  responses: length,
  eos: map(select(.result.finish_reason == "eos")) | length,
  max_tokens: map(select(.result.finish_reason == "max_tokens")) | length,
  results: map({prompt, tokens: .result.tokens, finish_reason: .result.finish_reason, text: .result.text})
}' "$output"
