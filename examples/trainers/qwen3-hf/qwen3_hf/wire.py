# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
import json
import sys

LIMIT = 16 * 1024 * 1024


def read(stream=None):
    line = (stream or sys.stdin).readline(LIMIT)
    if not line or not line.endswith("\n") or len(line.encode()) >= LIMIT:
        raise ValueError("missing, incomplete or oversized protocol frame")
    value = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError("frame must be an object")
    return value


def emit(value):
    line = json.dumps(value, separators=(",", ":"), allow_nan=False)
    if len(line.encode()) + 1 >= LIMIT:
        raise ValueError("output frame exceeds protocol limit")
    print(line, flush=True)


def frame(kind, **fields):
    emit(dict(kind=kind, protocol=1, **fields))
