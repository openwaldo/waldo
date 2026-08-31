# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

# internal/training/workers/mlx.py cannot be imported directly: it depends on
# mlx (Apple Silicon + Metal only) and its module-level run() blocks on
# stdin. read_gpu_throttle() has neither dependency, so this pulls just that
# function (plus the emit() and PROTOCOL_SCHEMA it uses) out of the real
# source via ast and executes it in isolation, keeping this test runnable
# anywhere with only the standard library.

import ast
import io
import json
import math
import os
import unittest
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLX_WORKER_PATH = os.path.join(REPO_ROOT, "internal", "training", "workers", "mlx.py")

WANTED_NAMES = {"PROTOCOL_SCHEMA", "emit", "read_gpu_throttle"}


def load_gpu_throttle_module():
    with open(MLX_WORKER_PATH, encoding="utf-8") as source_file:
        tree = ast.parse(source_file.read(), filename=MLX_WORKER_PATH)

    extracted = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in WANTED_NAMES)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in WANTED_NAMES for target in node.targets
        ))
    ]
    found = {node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id for node in extracted}
    if found != WANTED_NAMES:
        raise AssertionError(f"expected to extract {WANTED_NAMES} from {MLX_WORKER_PATH}, found {found}")

    module_ast = ast.Module(body=extracted, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    namespace = {"math": math, "os": os, "json": json}
    exec(compile(module_ast, MLX_WORKER_PATH, "exec"), namespace)
    return namespace


class ReadGPUThrottleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_gpu_throttle_module()

    def setUp(self):
        self.addCleanup(os.environ.pop, "WALDO_GPU_THROTTLE", None)

    def set_env(self, value):
        if value is None:
            os.environ.pop("WALDO_GPU_THROTTLE", None)
        else:
            os.environ["WALDO_GPU_THROTTLE"] = value

    def read_gpu_throttle(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            value = self.namespace["read_gpu_throttle"]()
        return value, buffer.getvalue()

    def test_unset(self):
        self.set_env(None)
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1)
        self.assertEqual(emitted, "")

    def test_empty(self):
        # An env var can't be null, only unset (test_unset) or empty. "" is
        # falsy, so `or 1` catches it before the try/except even runs --
        # a different code path than unset, same as no code path exists for
        # None here at all.
        self.set_env("")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1)
        self.assertEqual(emitted, "")

    def test_text(self):
        self.set_env("non-numeric text")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1)
        self.assertIn("WALDO_GPU_THROTTLE must be a decimal", emitted)

    def test_valid(self):
        self.set_env("0.25")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 0.25)
        self.assertEqual(emitted, "")

    def test_min(self):
        self.set_env("0.01")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 0.01)
        self.assertEqual(emitted, "")

    def test_max(self):
        self.set_env("1.0")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1.0)
        self.assertEqual(emitted, "")

    def test_zero(self):
        self.set_env("0")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1)
        self.assertIn("WALDO_GPU_THROTTLE must be a decimal", emitted)

    def test_negative(self):
        self.set_env("-0.5")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1)
        self.assertIn("WALDO_GPU_THROTTLE must be a decimal", emitted)

    def test_nan(self):
        # float("nan") parses without raising, so this exercises the range
        # check, not the except ValueError branch -- any comparison with NaN
        # is False, so it falls through the same as an out-of-range number.
        self.set_env("nan")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1)
        self.assertIn("WALDO_GPU_THROTTLE must be a decimal", emitted)

    def test_inf(self):
        # Also parses without raising; rejected by the upper bound.
        self.set_env("inf")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1)
        self.assertIn("WALDO_GPU_THROTTLE must be a decimal", emitted)

    def test_malformed(self):
        self.set_env("not-a-number")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1)
        self.assertIn("WALDO_GPU_THROTTLE must be a decimal", emitted)

    def test_too_high(self):
        self.set_env("1.5")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1)
        self.assertIn("WALDO_GPU_THROTTLE must be a decimal", emitted)

    def test_tiny(self):
        # Below the 0.01 floor, 1 / GPU_THROTTLE - 1 would otherwise blow up
        # into an effectively infinite per-step sleep.
        self.set_env("1e-300")
        value, emitted = self.read_gpu_throttle()
        self.assertEqual(value, 1)
        self.assertIn("WALDO_GPU_THROTTLE must be a decimal", emitted)


if __name__ == "__main__":
    unittest.main()
