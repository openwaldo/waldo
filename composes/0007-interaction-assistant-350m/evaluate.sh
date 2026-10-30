#!/usr/bin/env bash
set -euo pipefail

if (($# < 3 || $# > 5)); then
  echo "usage: $0 MODEL SUITE.json OUTPUT.jsonl [RUN_ID] [CHECKPOINT_STEP]" >&2
  exit 2
fi

model="$1"
suite="$2"
output="$3"
run_id="${4:-}"
checkpoint_step="${5:-}"
if [[ -z "$run_id" ]]; then
  [[ -z "$checkpoint_step" ]] || { echo "CHECKPOINT_STEP requires RUN_ID" >&2; exit 2; }
  run_id="$(go run ./cmd/waldo/ --json model summary "$model" | jq -r '.bom.current_run_id')"
fi
selector=(--run-id "$run_id")
[[ -z "$checkpoint_step" ]] || selector+=(--checkpoint-step "$checkpoint_step")

: >"$output"
while IFS= read -r prompt; do
  go run ./cmd/waldo/ --json model chat "$model" \
    "${selector[@]}" --temperature 0 --seed 42 --max-tokens 128 \
    "$prompt" >>"$output"
done < <(jq -r '.[].prompt' "$suite")
