#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

echo "testing: required PyTorch training/inference acceptance gate"
WALDO_E2E_REQUIRED=1 "$script_dir/e2e/model-pytorch.sh"

echo "testing: required TorchTitan training/inference acceptance gate"
WALDO_E2E_REQUIRED=1 "$script_dir/e2e/model-torchtitan.sh"

echo "Training acceptance passed: compiled, eager-compute, eager-FP32, persisted-FP32, and chat paths agree."
