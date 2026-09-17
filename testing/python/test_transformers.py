# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("waldo_transformers_worker", Path(__file__).resolve().parents[2] / "internal/training/workers/transformers.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
worker.MODEL_CLASSES = json.loads((Path(__file__).resolve().parents[2] / "internal/training/transformers_models.json").read_text())


class DeviceValidationTests(unittest.TestCase):
    def setUp(self):
        tensor = types.SimpleNamespace(sum=lambda: types.SimpleNamespace(item=lambda: 1))
        self.torch = types.SimpleNamespace(
            tensor=lambda *args, **kwargs: tensor,
            cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                      get_device_properties=lambda _: types.SimpleNamespace(name="GPU", total_memory=1024),
                                      is_bf16_supported=lambda: True))
        self.facts = dict(device="cuda", accelerator="GPU", memory_bytes=1024)

    def test_gpu_and_cpu(self):
        self.assertEqual(worker.validate_device(self.torch, self.facts, dict(bf16=True)), "cuda")
        self.assertEqual(worker.validate_device(self.torch, dict(device="cpu")), "cpu")

    def test_precision_and_identity_fail_closed(self):
        for facts, arguments, dtype in [
                (dict(device="cpu"), dict(fp16=True), "float32"),
                (self.facts, dict(fp16=True, bf16=True), "float32"),
                (self.facts, dict(fp16=True), "float16"),
                (dict(self.facts, accelerator="different"), {}, "float32")]:
            with self.subTest(arguments=arguments, facts=facts), self.assertRaises(ValueError):
                worker.validate_device(self.torch, facts, arguments, dtype)
        self.torch.cuda.is_bf16_supported = lambda: False
        with self.assertRaises(ValueError):
            worker.validate_device(self.torch, self.facts, dict(bf16=True))
        self.torch.cuda.device_count = lambda: 2
        with self.assertRaises(ValueError):
            worker.validate_device(self.torch, self.facts)

    def test_mixtral_parameter_evidence(self):
        parameter = lambda n: types.SimpleNamespace(numel=lambda: n)
        values = [("model.embed.weight", parameter(100)), ("layers.0.experts.gate_up_proj", parameter(400))]
        model = types.SimpleNamespace(parameters=lambda: [p for _, p in values], named_parameters=lambda: values)
        config = types.SimpleNamespace(model_type="mixtral", num_local_experts=4, num_experts_per_tok=2)
        result = worker.parameter_evidence(model, config)
        self.assertEqual(result["parameters"], 500)
        self.assertEqual(result["active_parameters_estimate"], 300)


class TransformersStreamTests(unittest.TestCase):
    def test_eos_shift_mask_and_cross_corpus_packing(self):
        records = [dict(tokens=[3, 4], loss_mask=[False, True, True], corpus="a"),
                   dict(tokens=[5, 6], loss_mask=[False, True, True], corpus="b")]
        result = list(worker.windows(records, 3, 0, 259, 2))
        self.assertEqual(result, [dict(input_ids=[3, 4, 2], attention_mask=[1,1,1], labels=[4, 2, -100], _corpora={"a": 2}),
                                  dict(input_ids=[5, 6, 0], attention_mask=[1,1,0], labels=[6, 2, -100], _corpora={"b": 2})])

    def test_evaluation_does_not_join_records(self):
        records = [dict(tokens=[3, 4], corpus="a"), dict(tokens=[5, 6], corpus="b")]
        result = list(worker.windows(records, 3, 0, 259, 2, continuous=False))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["labels"], [4, 2, -100])
        self.assertEqual(result[1]["labels"], [6, 2, -100])

    def test_bad_masks_and_tokens_fail_closed(self):
        for record in (dict(tokens=[259]), dict(tokens=[3], loss_mask=[True]), dict(tokens=[True]), dict(text="not tokenized")):
            with self.subTest(record=record), self.assertRaises(ValueError):
                list(worker.windows([record], 3, 0, 259, 2))

    def test_no_supervised_targets_yields_no_sample(self):
        self.assertEqual(list(worker.windows([dict(tokens=[3], loss_mask=[False, False])], 3, 0, 259, 2)), [])


class TokenizerVerificationTests(unittest.TestCase):
    def test_verified_snapshot_and_failures(self):
        with tempfile.TemporaryDirectory() as source:
            root = Path(source)
            (root / "tokenizer.json").write_text("{}")
            (root / "tokenizer_config.json").write_text("{}")
            (root / "untrusted.py").write_text("raise RuntimeError()")
            pin = dict(files={name: worker.digest(root / name) for name in ("tokenizer.json", "tokenizer_config.json")})
            spec = dict(huggingface=pin, vocabulary_size=4, pad_id=0, bos_id=-1, eos_id=2)
            tokenizer = types.SimpleNamespace(is_fast=True, pad_token_id=0, bos_token_id=None, eos_token_id=2, get_vocab=lambda: {"x": 3})

            def load(path, **kwargs):
                self.assertEqual(kwargs, dict(local_files_only=True, trust_remote_code=False, use_fast=True))
                self.assertEqual(set(item.name for item in Path(path).iterdir()), set(pin["files"]))
                return tokenizer

            package = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=load))
            with worker.pinned_tokenizer(package, spec, root) as (actual, snapshot):
                self.assertIs(actual, tokenizer)
                self.assertNotEqual(snapshot, root)
            for change in (dict(eos_id=1), dict(vocabulary_size=3)):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    with worker.pinned_tokenizer(package, dict(spec, **change), root):
                        pass
            for config in ({"auto_map": {"AutoTokenizer": "evil.Code"}}, {"tokenizer_file": "/elsewhere/tokenizer.json"}):
                (root / "tokenizer_config.json").write_text(json.dumps(config))
                pin["files"]["tokenizer_config.json"] = worker.digest(root / "tokenizer_config.json")
                with self.assertRaises(ValueError):
                    with worker.pinned_tokenizer(package, spec, root):
                        pass
            (root / "tokenizer.json").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                with worker.pinned_tokenizer(package, spec, root):
                    pass


class PackageVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "transformers"
        self.package.mkdir()
        self.source = self.package / "__init__.py"
        self.source.write_bytes(b"# verified source\n")
        self.wheel = self.root / "transformers-5.16.1-py3-none-any.whl"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.write(self.source, "transformers/__init__.py")
        self.pin = dict(distribution="transformers", version="5.16.1", artifact=self.wheel.name, sha256=worker.digest(self.wheel))
        distribution = types.SimpleNamespace(version="5.16.1", locate_file=lambda path: self.root / path)
        self.enterContext(patch.object(worker.importlib.metadata, "distribution", return_value=distribution))
        self.module = types.SimpleNamespace(__file__=str(self.source))
        self.enterContext(patch.object(sys, "meta_path", list(sys.meta_path)))
        self.enterContext(patch.dict(sys.modules, transformers=self.module))

    def test_verified_package(self):
        self.assertIs(worker.verify_package(self.wheel, self.pin), self.module)

    def test_bad_wheel_hash(self):
        self.pin["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            worker.verify_package(self.wheel, self.pin)

    def test_installed_source_tampering(self):
        self.source.write_bytes(b"# different source\n")
        with self.assertRaisesRegex(ValueError, "differs from wheel"):
            worker.verify_package(self.wheel, self.pin)

    def test_extra_installed_source(self):
        (self.package / "extra.py").write_bytes(b"# extra\n")
        with self.assertRaisesRegex(ValueError, "unattested file"):
            worker.verify_package(self.wheel, self.pin)

    def test_import_shadowing(self):
        self.module.__file__ = "/elsewhere/transformers/__init__.py"
        with self.assertRaisesRegex(ValueError, "shadowed"):
            worker.verify_package(self.wheel, self.pin)


if __name__ == "__main__":
    unittest.main()
