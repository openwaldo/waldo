#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
set -eu
if [ "${WALDO_E2E_EXTERNAL_DISTRIBUTED:-0}" != 1 ]; then
  echo 'testing: external distributed lifecycle SKIPPED (set WALDO_E2E_EXTERNAL_DISTRIBUTED=1)'
  exit 0
fi
root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
python=${WALDO_APPLICATION_PYTHON:-python3}
work=${WALDO_E2E_WORK:?set WALDO_E2E_WORK to a new test directory}
binary=${WALDO_BIN:?set WALDO_BIN to the built WALDO executable}
cache=${WALDO_APPLICATION_CACHE:?set WALDO_APPLICATION_CACHE}
if [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -lt 2 ]; then
  echo 'external distributed qualification requires two NVIDIA GPUs' >&2
  exit 1
fi
case $(uname -m) in
  x86_64) environment=cuda ;;
  aarch64) environment=cuda ;;
  *) echo 'unsupported CUDA test architecture' >&2; exit 1 ;;
esac
"$python" "$root/testing/external-qwen3/cli-smoke.py" --binary "$binary" --work "$work" \
  --application-cache "$cache" --python "$python" --trainer-environment "$environment" \
  --tokenizer-environment "$environment" --processes-per-node 2 \
  --source-repository "$root" --source-commit "$(git -C "$root" rev-parse HEAD)"
