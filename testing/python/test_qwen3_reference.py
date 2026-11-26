# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free schema/wire tests; real-framework tests use the Go harness."""
import importlib.util
import io
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "examples/trainers/qwen3-hf/qwen3_hf"


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReferenceContracts(unittest.TestCase):
    def test_roles(self):
        schema = module("schemas")
        for key in ("per_device_train_batch_size", "gradient_accumulation_steps", "gradient_checkpointing",
                    "torch_compile", "attn_implementation", "logging_steps", "report_to"):
            self.assertEqual(schema.PARAMETERS["properties"][key]["x-waldo-role"], "execution")
        for key in ("learning_rate", "bf16", "fp16", "tf32", "init_seed", "max_grad_norm"):
            self.assertNotIn("x-waldo-role", schema.PARAMETERS["properties"][key])

    def test_reject_unsupported_architecture_and_parameters(self):
        schema = module("schemas")
        for values in ({"num_attention_heads": 3, "num_key_value_heads": 2}, {"head_dim": 3},
                       {"num_hidden_layers": 0}, {"unknown": True}):
            with self.assertRaises(ValueError):
                schema.architecture(values)
        for values in ({"torch_compile": True}, {"report_to": ["wandb"]}, {"optim": "sgd"},
                       {"per_device_train_batch_size": 0}, {"bf16": "false"}):
            with self.assertRaises(ValueError):
                schema.validate(values, schema.PARAMETERS)

    def test_wire_rejects_invalid_frames(self):
        wire = module("wire")
        for value in ('', '{}', '[]\n', '{"loss":NaN}\n'):
            with self.assertRaises(ValueError):
                wire.read(io.StringIO(value))
        self.assertEqual(wire.read(io.StringIO('{"protocol":1}\n')), {"protocol": 1})


if __name__ == "__main__":
    unittest.main()
