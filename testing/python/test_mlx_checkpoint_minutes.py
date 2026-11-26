# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

# internal/training/workers/mlx.py needs mlx and blocks on stdin, so this pulls
# the checkpoint schedule and live config reader out of the source via ast and
# runs them with only the standard library.

import ast
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLX_WORKER_PATH = os.path.join(REPO_ROOT, "internal", "training", "workers", "mlx.py")
WANTED_NAMES = {"config_path", "live_config", "live_setting", "checkpoint_due"}


class CheckpointMinutesTest(unittest.TestCase):
    """`waldo config set training.checkpoint-minutes` also saves on the clock while a worker runs."""

    def setUp(self):
        with open(MLX_WORKER_PATH, encoding="utf-8") as source_file:
            tree = ast.parse(source_file.read(), filename=MLX_WORKER_PATH)
        body = [node for node in tree.body
                if (isinstance(node, ast.FunctionDef) and node.name in WANTED_NAMES)
                or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in WANTED_NAMES)]
        module_ast = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module_ast)
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        self.config = os.path.join(directory, "config.json")
        self.addCleanup(os.environ.pop, "WALDO_CONFIG", None)
        os.environ["WALDO_CONFIG"] = self.config
        self.clock = {"now": 1000.0}
        clock = self.clock

        class Clock:
            @staticmethod
            def monotonic():
                return clock["now"]

        self.namespace = {"math": math, "os": os, "json": json, "sys": sys, "time": Clock}
        exec(compile(module_ast, MLX_WORKER_PATH, "exec"), self.namespace)

    def write(self, minutes):
        with open(self.config, "w", encoding="utf-8") as file:
            json.dump({"schema": 1, "training": {"checkpoint_minutes": minutes}}, file)

    def test_a_checkpoint_comes_by_steps_or_by_minutes(self):
        due = self.namespace["checkpoint_due"]
        self.assertTrue(due(150, 150, 0, 60, None))            # every checkpoint_every steps
        self.assertFalse(due(151, 150, 0, 60, None))           # no minutes set: steps only
        self.assertFalse(due(151, 150, 0, 29 * 60, 30))        # 29 minutes since the last save
        self.assertTrue(due(151, 150, 0, 30 * 60, 30))         # 30 minutes: save now
        self.assertFalse(due(151, 0, 0, 99 * 60, 30))          # a plan without checkpoints stays without
        self.assertFalse(due(151, 150, 0, 99 * 60, "30"))      # not a number: steps only

    def test_minutes_are_reread_while_the_worker_runs(self):
        live_setting = self.namespace["live_setting"]
        self.assertIsNone(live_setting("training", "checkpoint_minutes"))  # no config file yet
        self.write(30)
        self.clock["now"] += 10
        self.assertIsNone(live_setting("training", "checkpoint_minutes"))  # read at most every 30s
        self.clock["now"] += 20
        self.assertEqual(live_setting("training", "checkpoint_minutes"), 30)
        self.write(5)
        self.clock["now"] += 30
        self.assertEqual(live_setting("training", "checkpoint_minutes"), 5)


if __name__ == "__main__":
    unittest.main()
