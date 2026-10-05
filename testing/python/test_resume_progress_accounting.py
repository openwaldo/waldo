# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

# The training workers cannot be imported directly: pytorch.py needs torch and mlx.py needs
# Apple Silicon, and both block on stdin at module level. This pulls progress_estimate, RateWindow
# and the Trainer.train_batch replay branch out of the real source of each worker via ast and runs
# them in isolation, so it stays runnable anywhere with only the standard library. train_batch is
# only driven through its replay branch, which returns before any framework code is reached.

import ast
import os
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKERS = {
    "pytorch": os.path.join(REPO_ROOT, "internal", "training", "workers", "pytorch.py"),
    "mlx": os.path.join(REPO_ROOT, "internal", "training", "workers", "mlx.py"),
}


def parse(path):
    with open(path, encoding="utf-8") as source_file:
        return ast.parse(source_file.read(), filename=path)


def one(nodes, description, path):
    nodes = list(nodes)
    if len(nodes) != 1:
        raise AssertionError(f"expected exactly one {description} in {path}, found {len(nodes)}")
    return nodes[0]


def module_node(tree, kind, name, path):
    return one((n for n in tree.body if isinstance(n, kind) and n.name == name), name, path)


def trainer_method(tree, name, path):
    trainer = module_node(tree, ast.ClassDef, "Trainer", path)
    return one((n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == name), f"Trainer.{name}", path)


def execute(nodes, path, namespace):
    module = ast.Module(body=list(nodes), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, path, "exec"), namespace)
    return namespace


def rate_namespace(tree, path):
    nodes = [
        module_node(tree, ast.FunctionDef, "progress_estimate", path),
        module_node(tree, ast.ClassDef, "RateWindow", path),
    ]
    return execute(nodes, path, {})


class ResumeProgressAccountingTest(unittest.TestCase):
    # Each worker runs in its own subTest so one worker's failure is reported against it and
    # never stops the other worker from being checked.
    def workers(self):
        return [(name, path, parse(path)) for name, path in WORKERS.items()]

    def test_both_workers_carry_the_same_helpers(self):
        dumps = {
            name: (
                ast.dump(module_node(tree, ast.FunctionDef, "progress_estimate", path)),
                ast.dump(module_node(tree, ast.ClassDef, "RateWindow", path)),
            )
            for name, path, tree in self.workers()
        }
        self.assertEqual(len(set(dumps.values())), 1, "the two copies of the rate helpers have drifted apart")

    def test_fresh_run_keeps_the_whole_run_formula(self):
        for name, path, tree in self.workers():
            with self.subTest(worker=name):
                window = rate_namespace(tree, path)["RateWindow"](100.0)
                target = 2400
                for step, tokens, seconds in ((1, 100, 0.5), (800, 80000, 640.0), (2400, 240000, 1900.0)):
                    throughput, eta = window.estimate(step, target, tokens, 100.0 + seconds)
                    self.assertAlmostEqual(throughput, tokens / seconds)
                    self.assertEqual(eta, int((target - step) * seconds / step))

    def test_resumed_run_reports_the_measured_rate(self):
        for name, path, tree in self.workers():
            with self.subTest(worker=name):
                window = rate_namespace(tree, path)["RateWindow"](0.0)
                window.restart()
                # Replay ended after a resume from step 1,600 (160,000 tokens). The first trained step
                # ends at t=100 s and opens the window, so there is no rate yet.
                self.assertEqual(window.estimate(1601, 2400, 160100, 100.0), (0.0, 0))
                # 32 steps and 3,200 tokens later, 25 s on: a true 128 tokens/s with 767 steps to go.
                # The old whole-process formula would have said 163,300 / 125 = 1,306 tokens/s.
                throughput, eta = window.estimate(1633, 2400, 163300, 125.0)
                self.assertAlmostEqual(throughput, 128.0)
                self.assertEqual(eta, 599)

    def test_first_step_start_up_cost_is_not_averaged_in(self):
        for name, path, tree in self.workers():
            with self.subTest(worker=name):
                window = rate_namespace(tree, path)["RateWindow"](0.0)
                window.restart()
                # The first trained step takes 11 s (one-time start-up), then 32 steps take 3.2 s.
                window.estimate(1601, 2400, 160100, 11.0)
                throughput, eta = window.estimate(1633, 2400, 163300, 14.2)
                self.assertAlmostEqual(throughput, 1000.0)
                self.assertEqual(eta, 76)

    def test_empty_or_degenerate_windows_never_raise(self):
        for name, path, tree in self.workers():
            with self.subTest(worker=name):
                estimate = rate_namespace(tree, path)["progress_estimate"]
                self.assertEqual(estimate(1600, 2400, 0, 0, 0.0), (0.0, 0))
                _, eta = estimate(2500, 2400, 10, 1000, 5.0)
                self.assertEqual(eta, 0, "an ETA must never be negative")

    def test_replay_end_restarts_the_window(self):
        for name, path, tree in self.workers():
            with self.subTest(worker=name):
                train_batch = execute([trainer_method(tree, "train_batch", path)], path, {})["train_batch"]
                restarts = []
                trainer = types.SimpleNamespace(
                    batch=[object()],
                    step_number=1600,
                    target_steps=2400,
                    replay_steps=2,
                    rate_window=types.SimpleNamespace(restart=lambda: restarts.append(1)),
                )
                train_batch(trainer)
                self.assertEqual((trainer.replay_steps, restarts, trainer.batch), (1, [], []))
                trainer.batch = [object()]
                train_batch(trainer)
                self.assertEqual((trainer.replay_steps, restarts, trainer.batch), (0, [1], []))

    def test_fresh_trainer_window_starts_at_launch(self):
        for name, path, tree in self.workers():
            with self.subTest(worker=name):
                init = trainer_method(tree, "__init__", path)
                values = [
                    ast.unparse(node.value)
                    for node in ast.walk(init)
                    if isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and ast.unparse(node.targets[0]) == "self.rate_window"
                ]
                self.assertEqual(values, ["RateWindow(self.started)"])

    def test_report_block_feeds_the_window(self):
        for name, path, tree in self.workers():
            with self.subTest(worker=name):
                train_batch = trainer_method(tree, "train_batch", path)
                calls = [
                    node
                    for node in ast.walk(train_batch)
                    if isinstance(node, ast.Call) and ast.unparse(node.func) == "self.rate_window.estimate"
                ]
                call = one(calls, "rate_window.estimate call in train_batch", path)
                self.assertEqual(
                    [ast.unparse(arg) for arg in call.args],
                    ["self.step_number", "self.target_steps", "self.consumed_tokens", "time.perf_counter()"],
                )


if __name__ == "__main__":
    unittest.main()
