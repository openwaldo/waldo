#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

"$script_dir/unit.sh"
"$script_dir/vet.sh"
"$script_dir/docs.sh"
"$script_dir/python.sh"
"$script_dir/e2e/ingest-direct.sh"
"$script_dir/e2e/structured-conversation.sh"
"$script_dir/e2e/model-fake.sh"
"$script_dir/e2e/model-mlx.sh"
"$script_dir/e2e/model-pytorch.sh"
"$script_dir/e2e/model-torchtitan.sh"
"$script_dir/e2e/model-torchtitan-multinode.sh"
