#!/usr/bin/env bash
# Run the frozen rung-0004 generation probes.
set -euo pipefail

if (($# < 2 || $# > 7)); then
  echo "usage: $0 MODEL OUTPUT.jsonl [TEMPERATURE] [SEED] [MAX_TOKENS] [RUN_ID] [CHECKPOINT_STEP]" >&2
  exit 2
fi

model="$1"
output="$2"
temperature="${3:-0.8}"
seed="${4:-42}"
max_tokens="${5:-512}"
run_id="${6:-}"
checkpoint_step="${7:-}"
if [[ -z "$run_id" ]]; then
  [[ -z "$checkpoint_step" ]] || { echo "CHECKPOINT_STEP requires RUN_ID" >&2; exit 2; }
  run_id="$(go run ./cmd/waldo/ --json model summary "$model" | jq -r '.bom.current_run_id')"
fi
selector=(--run-id "$run_id")
[[ -z "$checkpoint_step" ]] || selector+=(--checkpoint-step "$checkpoint_step")

prompts=()
while IFS= read -r prompt; do
  prompts+=("$prompt")
done < <(jq -r '.[].prompt' "$(dirname "$0")/suite.json")
: >"$output"
for prompt in "${prompts[@]}"; do
  go run ./cmd/waldo/ --json model chat "$model" \
    "${selector[@]}" --raw --temperature "$temperature" --seed "$seed" \
    --max-tokens "$max_tokens" "$prompt" >>"$output"
done

jq -s --argjson max_tokens_requested "$max_tokens" '{
  responses: length,
  max_tokens_requested: $max_tokens_requested,
  eos: map(select(.result.finish_reason == "eos")) | length,
  max_tokens: map(select(.result.finish_reason == "max_tokens")) | length,
  results: map({prompt, tokens: .result.tokens, finish_reason: .result.finish_reason, text: .result.text})
}' "$output"
