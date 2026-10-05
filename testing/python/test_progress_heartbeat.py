# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

# The training workers cannot be imported directly: pytorch.py needs torch and mlx.py needs
# Apple Silicon, and both block on stdin at module level. The heartbeat helpers have neither
# dependency, so this extracts them from the real source of each worker via ast and runs them
# in isolation, which keeps the test runnable anywhere with only the standard library.

import ast
import io
import json
import math
import os
import unittest
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKERS = {
    "pytorch": os.path.join(REPO_ROOT, "internal", "training", "workers", "pytorch.py"),
    "mlx": os.path.join(REPO_ROOT, "internal", "training", "workers", "mlx.py"),
}

WANTED_NAMES = {
    "PROTOCOL_SCHEMA",
    "emit",
    "DEFAULT_PROGRESS_HEARTBEAT_SECONDS",
    "read_progress_heartbeat",
    "progress_due",
}

ENV_NAME = "WALDO_PROGRESS_HEARTBEAT_SECONDS"


def load_heartbeat_module(path):
    with open(path, encoding="utf-8") as source_file:
        tree = ast.parse(source_file.read(), filename=path)

    extracted = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in WANTED_NAMES)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in WANTED_NAMES for target in node.targets
        ))
    ]
    found = {node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id for node in extracted}
    if found != WANTED_NAMES:
        raise AssertionError(f"expected to extract {WANTED_NAMES} from {path}, found {found}")

    module_ast = ast.Module(body=extracted, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    namespace = {"math": math, "os": os, "json": json, "IS_PRIMARY": True}
    exec(compile(module_ast, path, "exec"), namespace)
    return namespace


class ProgressHeartbeatTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(os.environ.pop, ENV_NAME, None)

    def set_env(self, value):
        if value is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = value

    def read(self, namespace):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            value = namespace["read_progress_heartbeat"]()
        return value, buffer.getvalue()

    def assert_logged_fallback(self, namespace, value):
        self.set_env(value)
        result, emitted = self.read(namespace)
        self.assertEqual(result, namespace["DEFAULT_PROGRESS_HEARTBEAT_SECONDS"], value)
        frame = json.loads(emitted)
        self.assertEqual(frame["event"]["kind"], "log")
        self.assertIn("WALDO_PROGRESS_HEARTBEAT_SECONDS must be a non-negative number", frame["event"]["message"])

    def test_workers(self):
        for name, path in WORKERS.items():
            with self.subTest(worker=name):
                self.check_worker(load_heartbeat_module(path))

    def check_worker(self, namespace):
        default = namespace["DEFAULT_PROGRESS_HEARTBEAT_SECONDS"]
        self.assertEqual(default, 300)

        for unset in (None, "", "   "):
            self.set_env(unset)
            result, emitted = self.read(namespace)
            self.assertEqual(result, default, repr(unset))
            self.assertEqual(emitted, "", repr(unset))

        for text, expected in (("60", 60.0), ("0.5", 0.5), ("0", 0.0), ("1e3", 1000.0)):
            self.set_env(text)
            result, emitted = self.read(namespace)
            self.assertEqual(result, expected, text)
            self.assertEqual(emitted, "", text)

        for invalid in ("abc", "-1", "-0.1", "nan", "inf", "-inf", "1,5"):
            self.assert_logged_fallback(namespace, invalid)

        due = namespace["progress_due"]

        # The existing 1% rule is unchanged: first step, last step, and every 1% boundary.
        self.assertTrue(due(1, 1000, 0.0, 0.0, 300))
        self.assertTrue(due(1000, 1000, 0.0, 0.0, 300))
        self.assertTrue(due(10, 1000, 0.0, 0.0, 300))
        self.assertTrue(due(500, 1000, 0.0, 0.0, 300))
        self.assertFalse(due(11, 1000, 0.0, 1.0, 300))
        self.assertFalse(due(999, 1000, 0.0, 1.0, 300))
        # A run shorter than 100 steps reports every step.
        self.assertTrue(due(7, 50, 0.0, 0.0, 300))

        # The heartbeat fires only once the interval has passed since the last progress line.
        self.assertFalse(due(11, 1000, 0.0, 299.9, 300))
        self.assertTrue(due(11, 1000, 0.0, 300.0, 300))
        self.assertTrue(due(11, 1000, 0.0, 301.0, 300))
        self.assertFalse(due(11, 1000, 100.0, 399.0, 300))
        self.assertTrue(due(11, 1000, 100.0, 400.0, 300))

        # Zero turns the heartbeat off, however long the gap.
        self.assertFalse(due(11, 1000, 0.0, 10 ** 9, 0))
        self.assertTrue(due(10, 1000, 0.0, 0.0, 0))

    def test_training_loops_use_the_helper(self):
        # The helpers above are only a feature if the training loop calls them: the loop must
        # gate its progress emit on progress_due() and record the report time when it emits.
        for name, path in WORKERS.items():
            with self.subTest(worker=name):
                with open(path, encoding="utf-8") as source_file:
                    tree = ast.parse(source_file.read(), filename=path)
                trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")

                def calls(node, function_name):
                    return any(
                        isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == function_name
                        for item in ast.walk(node)
                    )

                guards = [
                    node for node in ast.walk(trainer)
                    if isinstance(node, ast.If) and calls(node.test, "progress_due")
                ]
                self.assertEqual(len(guards), 1, "expected exactly one progress_due() guard in Trainer")
                body = ast.Module(body=guards[0].body, type_ignores=[])
                self.assertTrue(calls(body, "emit"), "the guarded block must emit the progress event")
                records_report_time = any(
                    isinstance(item, ast.Assign)
                    and any(isinstance(target, ast.Attribute) and target.attr == "last_report" for target in item.targets)
                    for item in ast.walk(body)
                )
                self.assertTrue(records_report_time, "the guarded block must update self.last_report")


if __name__ == "__main__":
    unittest.main()
