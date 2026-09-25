#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
hostfile=""
corpus=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --hostfile)
      [ "$#" -ge 2 ] || { echo "--hostfile requires a path" >&2; exit 2; }
      hostfile=$2
      shift 2
      ;;
    --corpus)
      [ "$#" -ge 2 ] || { echo "--corpus requires a small structured-conversation index path" >&2; exit 2; }
      corpus=$2
      shift 2
      ;;
    -h|--help)
      echo "usage: $0 [--hostfile PATH --corpus CONVERSATION_INDEX_PATH]"
      echo ""
      echo "Without a hostfile, validates local PyTorch and TorchTitan execution."
      echo "With a hostfile, also validates the real SSH multi-host training path."
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if { [ -n "$hostfile" ] && [ -z "$corpus" ]; } || { [ -z "$hostfile" ] && [ -n "$corpus" ]; }; then
  echo "--hostfile and --corpus must be provided together" >&2
  exit 2
fi

echo "testing: required PyTorch training/inference acceptance gate"
WALDO_E2E_REQUIRED=1 "$script_dir/e2e/model-pytorch.sh"

echo "testing: required TorchTitan training/inference acceptance gate"
WALDO_E2E_REQUIRED=1 "$script_dir/e2e/model-torchtitan.sh"

if [ -n "$hostfile" ]; then
  echo "testing: required TorchTitan hostfile acceptance gate"
  "$script_dir/e2e/model-torchtitan-hostfile.sh" "$hostfile" "$corpus"
  echo "Training acceptance passed: local and hostfile training, compiled, eager-compute, eager-FP32, persisted-FP32, and chat paths agree."
else
  echo "Local training acceptance passed: compiled, eager-compute, eager-FP32, persisted-FP32, and chat paths agree."
  echo "Multi-host acceptance was not tested; rerun with --hostfile PATH --corpus SMALL_INDEX_PATH before a multi-host production run."
fi
