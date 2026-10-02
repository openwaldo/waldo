#!/usr/bin/env bash
# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Existing runtime only: this script does not install packages, download data,
# acquire GPUs, publish releases, or modify managed models.
: "${WALDO_TRANSFORMERS_PYTHON:?Set the Python executable for the pinned Transformers runtime}"
: "${WALDO_TRANSFORMERS_WHEEL:?Set the local pinned Transformers wheel}"
precision="${1:-fp32}"
case "$precision" in
  fp32) unset WALDO_HF_TEST_PRECISION ;;
  fp16|bf16) export WALDO_HF_TEST_PRECISION="$precision" ;;
  *) echo "usage: bash testing/hf-gpu.sh [fp32|fp16|bf16]" >&2; exit 2 ;;
esac
export WALDO_TRANSFORMERS_DEVICE=cuda
cd "$(dirname "$0")/.."
go test ./internal/model -run '^TestTransformers(FamilyLifecycle|RealLifecycle|Qwen3Smoke)$' -count=1 -v
