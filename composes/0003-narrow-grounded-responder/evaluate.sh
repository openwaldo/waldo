#!/usr/bin/env bash
# Evaluate candidate rung 0003 on immutable, fully grounded prompts.
set -euo pipefail

if (($# < 2 || $# > 4)); then
  echo "usage: $0 MODEL OUTPUT.jsonl [RUN_ID] [CHECKPOINT_STEP]" >&2
  exit 2
fi

model="$1"
output="$2"
run_id="${3:-}"
checkpoint_step="${4:-}"
if [[ -z "$run_id" ]]; then
  [[ -z "$checkpoint_step" ]] || { echo "CHECKPOINT_STEP requires RUN_ID" >&2; exit 2; }
  run_id="$(go run ./cmd/waldo/ --json model summary "$model" | jq -r '.bom.current_run_id')"
fi
selector=(--run-id "$run_id")
[[ -z "$checkpoint_step" ]] || selector+=(--checkpoint-step "$checkpoint_step")

prompts=(
  'Read this tiny story. Nora put the teal marble in the cupboard. Where is the marble? Answer with only the place.'
  'Read this tiny story. Owen put the gray ribbon in the bucket. Where is the ribbon? Answer with only the place.'
  'Read this tiny story. Pia put the tan coin in the attic. Where is the coin? Answer with only the place.'
  'Read this tiny story. Quinn put the violet brush in the pantry. Where is the brush? Answer with only the place.'
  'Read this tiny story. Rosa put the cream bell in the garage. Where is the bell? Answer with only the place.'
  'Read this tiny story. Nora gave Owen the teal marble because Owen needed it. Who received the marble? Answer with only the name.'
  'Read this tiny story. Pia gave Quinn the gray ribbon because Quinn needed it. Who received the ribbon? Answer with only the name.'
  'Read this tiny story. Rosa gave Sven the tan coin because Sven needed it. Who received the coin? Answer with only the name.'
  'Read this tiny story. Uma gave Wade the violet brush because Wade needed it. Who received the brush? Answer with only the name.'
  'Read this tiny story. Xena gave Yuri the cream bell because Yuri needed it. Who received the bell? Answer with only the name.'
  'Read this tiny story. Nora carried a teal marble to the cupboard. Was the marble teal? Answer yes or no.'
  'Read this tiny story. Owen carried a gray ribbon to the bucket. Was the ribbon pink? Answer yes or no.'
  'Read this tiny story. Pia carried a tan coin to the attic. Was the coin tan? Answer yes or no.'
  'Read this tiny story. Quinn carried a violet brush to the pantry. Was the brush green? Answer yes or no.'
  'Read this tiny story. Rosa carried a cream bell to the garage. Was the bell cream? Answer yes or no.'
  'Rewrite this sentence by replacing Nora with Owen: "Nora carried the teal marble to the cupboard."'
  'Rewrite this sentence by replacing Pia with Quinn: "Pia carried the gray ribbon to the bucket."'
  'Rewrite this sentence by replacing Rosa with Sven: "Rosa carried the tan coin to the attic."'
  'Rewrite this sentence by replacing Uma with Wade: "Uma carried the violet brush to the pantry."'
  'Rewrite this sentence by replacing Xena with Yuri: "Xena carried the cream bell to the garage."'
)

: >"$output"
for prompt in "${prompts[@]}"; do
  go run ./cmd/waldo/ --json model chat "$model" \
    "${selector[@]}" --temperature 0 --max-tokens 128 "$prompt" >>"$output"
done

python3 "$(dirname "$0")/score.py" "$output"
