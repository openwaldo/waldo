# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Batch-identity checks in the reference trainer's resolve, without frameworks.

The real resolve imports PyTorch and Transformers; those imports are replaced
with inert stand-ins so the arithmetic that enforces micro-batch x accumulation
x world size = global batch runs in the dependency-free Python suite.
"""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2] / "examples/trainers/qwen3-hf"


def load_train_cli():
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
    torch.utils = types.SimpleNamespace(data=types.SimpleNamespace(IterableDataset=object))
    torch.no_grad = lambda: (lambda function: function)
    transformers = types.ModuleType("transformers")
    for name in ("Qwen3Config", "Qwen3ForCausalLM", "TrainingArguments", "enable_full_determinism", "set_seed"):
        setattr(transformers, name, mock.MagicMock(name=name))
    transformers.Trainer = type("Trainer", (), {})
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    callback = types.ModuleType("transformers.trainer_callback")
    callback.PrinterCallback = object
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.Tokenizer = mock.MagicMock()
    stubs = {"torch": torch, "numpy": types.ModuleType("numpy"), "transformers": transformers,
             "transformers.trainer_callback": callback, "tokenizers": tokenizers}
    with mock.patch.dict(sys.modules, stubs), mock.patch.object(sys, "path", [str(ROOT)] + sys.path):
        for name in [m for m in sys.modules if m == "qwen3_hf" or m.startswith("qwen3_hf.")]:
            del sys.modules[name]
        module = importlib.import_module("qwen3_hf.train_cli")
        for name in [m for m in sys.modules if m == "qwen3_hf" or m.startswith("qwen3_hf.")]:
            del sys.modules[name]
        return module


def request(global_batch, world=1, **parameters):
    devices = [dict(name="CPU", kind="cpu", memory_bytes=1)]
    return dict(architecture={}, parameters=parameters,
                world=dict(world_size=world, devices=devices),
                data=dict(global_batch_sequences=global_batch))


class ResolveBatchIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_train_cli()

    def batch(self, *args, **kwargs):
        return self.cli.resolve(request(*args, **kwargs))["batch"]

    def test_defaults_factor_the_global_batch(self):
        for global_batch, world in ((8, 1), (8, 2), (6, 1), (7, 1), (12, 4), (1, 1)):
            b = self.batch(global_batch, world)
            self.assertEqual(b["world_size"], world)
            self.assertEqual(b["micro_batch_sequences"] * b["accumulation_steps"] * world, global_batch)

    def test_supplied_values_are_kept_when_consistent(self):
        b = self.batch(8, 2, per_device_train_batch_size=2, gradient_accumulation_steps=2)
        self.assertEqual((b["micro_batch_sequences"], b["accumulation_steps"]), (2, 2))
        b = self.batch(8, 1, gradient_accumulation_steps=4)
        self.assertEqual((b["micro_batch_sequences"], b["accumulation_steps"]), (2, 4))
        resolved = self.cli.resolve(request(8, 1, per_device_train_batch_size=4))
        self.assertEqual(resolved["parameters"]["gradient_accumulation_steps"], 2)
        self.assertEqual(resolved["parameters"]["per_device_train_batch_size"], 4)

    def test_rejects_batches_that_do_not_multiply_to_the_global_batch(self):
        cases = (
            dict(global_batch=7, world=2),  # not divisible by world size
            dict(global_batch=8, world=1, per_device_train_batch_size=3),
            dict(global_batch=8, world=1, per_device_train_batch_size=2, gradient_accumulation_steps=2),
            dict(global_batch=8, world=2, gradient_accumulation_steps=8),  # exceeds local batch
            dict(global_batch=8, world=1, gradient_accumulation_steps=3),
        )
        for case in cases:
            with self.subTest(case), self.assertRaises(ValueError):
                self.batch(case.pop("global_batch"), case.pop("world"), **case)


if __name__ == "__main__":
    unittest.main()
