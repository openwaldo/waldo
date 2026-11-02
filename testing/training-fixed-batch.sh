#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
set -eu
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
exec "${WALDO_PYTHON:-python3}" "$script_dir/python/pytorch_fixed_batch.py" "$@"
