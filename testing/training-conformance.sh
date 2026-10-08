#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/.." && pwd)
required=${WALDO_CONFORMANCE_REQUIRED:-1}

torch_python=""
for candidate in "${WALDO_PYTHON:-}" "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
  [ -n "$candidate" ] && [ -x "$candidate" ] || continue
  if "$candidate" -c 'import torch; assert hasattr(torch.nn.functional, "scaled_dot_product_attention")' >/dev/null 2>&1; then
    torch_python=$candidate
    break
  fi
done

if [ -z "$torch_python" ]; then
  if [ "$required" = "1" ]; then
    echo "training conformance requires a Python runtime with PyTorch and scaled_dot_product_attention" >&2
    exit 1
  fi
  echo "testing: PyTorch numerical conformance skipped (no usable PyTorch runtime)"
  exit 0
fi

echo "testing: independent PyTorch logits, loss, gradient, AdamW, and reload conformance"
"$torch_python" "$script_dir/python/pytorch_conformance.py" \
  --model-source "$repo_root/internal/pytorchruntime/model.py" \
  --device cpu

if "$torch_python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  echo "testing: independent CUDA FP32 conformance"
  "$torch_python" "$script_dir/python/pytorch_conformance.py" \
    --model-source "$repo_root/internal/pytorchruntime/model.py" \
    --device cuda
else
  echo "testing: CUDA FP32 conformance skipped (CUDA unavailable)"
fi

echo "Training numerical conformance passed."
