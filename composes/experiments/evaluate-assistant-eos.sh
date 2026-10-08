#!/usr/bin/env bash
set -euo pipefail

if (($# != 2)); then
  echo "usage: $0 baseline|contract|broad OUTPUT.jsonl" >&2
  exit 2
fi

mode="$1"
output="$2"
case "$mode" in
  baseline)
    model="foundation-small-language-full-bpe-01"
    run_id="11c4dd9bde7821c3"
    ;;
  child | contract)
    model="foundation-small-assistant-eos-01"
    run_id="$(go run ./cmd/waldo/ --json model summary "$model" | jq -r '.bom.current_run_id')"
    ;;
  broad)
    model="foundation-small-assistant-broad-01"
    run_id="$(go run ./cmd/waldo/ --json model summary "$model" | jq -r '.bom.current_run_id')"
    ;;
  *)
    echo "mode must be baseline, contract, or broad" >&2
    exit 2
    ;;
esac

questions=(
  "What is Linux?"
  "Who created the Linux kernel?"
  "What is the capital of France?"
  "At what temperature does water freeze at sea level?"
  "What does a CPU do?"
  "Who created the Python programming language?"
  "What is two plus two?"
  "Why are backups useful?"
  "How do plants use sunlight?"
  "Give one safe way to install software on Linux."
)

: >"$output"
for question in "${questions[@]}"; do
  if [[ "$mode" == "baseline" ]]; then
    prompt=$'User: '"$question"$'\n\nAssistant:'
    go run ./cmd/waldo/ --json model chat "$model" \
      --run-id "$run_id" --raw --temperature 0 --max-tokens 128 \
      "$prompt" >>"$output"
  else
    go run ./cmd/waldo/ --json model chat "$model" \
      --run-id "$run_id" --temperature 0 --max-tokens 128 \
      "$question" >>"$output"
  fi
done

jq -s '{responses: length, eos: map(select(.result.finish_reason == "eos")) | length, max_tokens: map(select(.result.finish_reason == "max_tokens")) | length, results: map({prompt, tokens: .result.tokens, finish_reason: .result.finish_reason, text: .result.text})}' "$output"
