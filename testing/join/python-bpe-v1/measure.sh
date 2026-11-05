#!/bin/sh
# Waits for the python-bpe-v1 launcher to finish, then scores the final merge on the 30-request demo
# and the full held-out gate set, and totals the GPU time, into results/.
set -eu
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
bot=$HOME/.waldo/join/mdagosta-bot-2
waldo=$here/../../../waldo
mkdir -p "$here/results"
while pgrep -f "run-identities.sh|waldo model join" >/dev/null; do sleep 300; done
model=$(ls "$bot/models" | grep -E '^waldo-python-bpe-v1-r[0-9]{4}-merge-[0-9a-f]{12}/?$' | tr -d / | sort | tail -1)
echo "$model" > "$here/results/model.txt"
score() {
  WALDO=$waldo WALDO_CONFIG=$bot/waldo-config.json WALDO_GPU_THROTTLE=0.10 uv run -q --with mlx \
    python3 "$here/evaluate_python.py" --show --records "$here/$1" "$model" "$here/eval.bom.json" "$here/results/$2.json" \
    > "$here/results/$2.txt" 2>&1
}
score demo.jsonl demo
score eval.jsonl gate-full
python3 -I "$HOME/src/openwaldo/waldo-join-next/testing/join/timings.py" python-bpe-v1 code-m0 > "$here/results/timings.txt"
echo "done: $(tail -1 "$here/results/demo.txt")"
