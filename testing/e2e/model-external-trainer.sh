#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0

# External trainer lifecycle through the real waldo CLI: pinned source fetch,
# environment verification, tokenizer app, external trainer, interruption after
# the first checkpoint, same-command resume, byte-identical weights, BOMs, and
# Hugging Face export. Opt-in (network, pip, hundreds of MB of wheels):
#   WALDO_E2E_EXTERNAL=1 ./testing/e2e/model-external-trainer.sh
# The CPU run is the fallback for hosts without a GPU; the CUDA run is reported
# as skipped, never passed, when no NVIDIA GPU is visible. Set
# WALDO_E2E_EXTERNAL_LARGE=1 for the budget-capped ~280M GPU run.

set -eu

if [ "${WALDO_E2E_EXTERNAL:-0}" != "1" ]; then
  echo "testing: external trainer lifecycle skipped (set WALDO_E2E_EXTERNAL=1)"
  exit 0
fi
if [ "$(uname -s)" != "Linux" ]; then
  echo "testing: external trainer lifecycle skipped (requires Linux)"
  exit 0
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
smoke="$repo_root/testing/external-qwen3/cli-smoke.py"
python=${WALDO_APPLICATION_PYTHON:-/usr/bin/python3}
temporary_base=${TMPDIR:-/tmp}
work=$(mktemp -d "$temporary_base/waldo-external-e2e.XXXXXX")
cache=${WALDO_APPLICATION_CACHE:-$work/application-cache}

cleanup() {
  if [ "${WALDO_E2E_KEEP:-0}" = "1" ]; then
    echo "preserved external trainer E2E workspace: $work"
    return
  fi
  case "$work" in
    "$temporary_base"/waldo-external-e2e.*) rm -rf -- "$work" ;;
    *) echo "refusing to remove unexpected workspace: $work" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

binary=${WALDO_BIN:-$work/waldo}
if [ -z "${WALDO_BIN:-}" ]; then
  (cd "$repo_root" && go build -o "$binary" ./cmd/waldo)
fi

echo "testing: external trainer, CPU (env/cpu)"
"$python" "$smoke" --binary "$binary" --work "$work/cpu" --application-cache "$cache" --python "$python"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
  echo "testing: external trainer, GPU (env/cuda)"
  "$python" "$smoke" --binary "$binary" --work "$work/cuda" --application-cache "$cache" --python "$python" --trainer-environment cuda
  if [ "${WALDO_E2E_EXTERNAL_LARGE:-0}" = "1" ]; then
    echo "testing: external trainer, GPU ~280M budget-capped (compose.yaml)"
    "$python" "$smoke" --binary "$binary" --work "$work/cuda-large" --application-cache "$cache" --python "$python" \
      --compose examples/trainers/qwen3-hf/compose.yaml --stop none --records 400 \
      --data steps=10 --data sequence_length=512 --data global_batch_sequences=8 \
      --data checkpoint_every=10 --data evaluate_every=10
  fi
else
  echo "testing: external trainer on GPU SKIPPED (no NVIDIA GPU visible); this is not a pass"
fi

echo "testing: schema-1 (native) compose, unchanged path"
"$script_dir/model-fake.sh"
