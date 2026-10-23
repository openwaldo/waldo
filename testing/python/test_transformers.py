# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import math
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

    def test_unavailable_device_and_allocation_failure(self):
        self.torch.cuda.is_available = lambda: False
        with self.assertRaisesRegex(ValueError, "visible"):
            worker.validate_device(self.torch, self.facts)
        self.torch.cuda.is_available = lambda: True
        with patch.object(self.torch, "tensor", side_effect=RuntimeError("CUDA allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                worker.validate_device(self.torch, self.facts)
        with patch.dict(worker.os.environ, WORLD_SIZE="2"):
            with self.assertRaisesRegex(ValueError, "one CPU process or one GPU"):
                worker.validate_device(self.torch, self.facts)


class TokenizerConfigurationTests(unittest.TestCase):
    def test_special_ids_for_byte_and_hf(self):
        for spec in [dict(name="byte", pad_id=0, bos_id=1, eos_id=2),
                     dict(name="huggingface", huggingface={}, pad_id=0, bos_id=-1, eos_id=2)]:
            values = {name + "_token_id": None if spec[name + "_id"] == -1 else spec[name + "_id"] for name in ("pad", "bos", "eos")}
            worker.validate_tokenizer_configuration(types.SimpleNamespace(**values), spec)
            for name in values:
                for invalid in [99, None]:
                    if invalid == values[name]:
                        continue
                    with self.subTest(tokenizer=spec["name"], field=name, value=invalid), self.assertRaisesRegex(ValueError, "differs from tokenizer"):
                        worker.validate_tokenizer_configuration(types.SimpleNamespace(**dict(values, **{name: invalid})), spec)

    def test_unknown_and_incompatible_configuration(self):
        class Config:
            def __init__(self, **values):
                self.model_type = values.get("model_type", "llama")
            def to_dict(self):
                return dict(model_type=self.model_type)
        package = types.SimpleNamespace(LlamaConfig=Config)
        spec = dict(model_class="LlamaForCausalLM", config_class="LlamaConfig", config={"unknown": 1})
        with self.assertRaisesRegex(ValueError, "unknown architecture.config"):
            worker.configuration(package, spec)
        spec["config"] = dict(model_type="qwen3")
        with self.assertRaisesRegex(ValueError, "model_type"):
            worker.configuration(package, spec)


class FinalModelValidationTests(unittest.TestCase):
    def test_nonfinite_parameters_and_buffers_fail(self):
        class Tensor:
            def __init__(self, values, floating=True):
                self.values, self.floating = values, floating
            def is_floating_point(self): return self.floating
            def is_complex(self): return False
            def detach(self): return self
            def reshape(self, *_): return self
            def split(self, size):
                self_size = size
                return [self.values[i:i+self_size] for i in range(0, len(self.values), self_size)]
        def isfinite(values):
            return types.SimpleNamespace(all=lambda: types.SimpleNamespace(item=lambda: all(math.isfinite(x) for x in values)))
        with patch.dict(sys.modules, torch=types.SimpleNamespace(isfinite=isfinite)):
            for group in ("parameter", "buffer"):
                for value in (0.5, float("nan"), float("inf"), -float("inf")):
                    with self.subTest(group=group, value=value):
                        values = [("bad", Tensor([1.0, value]))]
                        model = types.SimpleNamespace(named_parameters=lambda: values if group == "parameter" else [],
                                                      named_buffers=lambda: values if group == "buffer" else [])
                        if math.isfinite(value):
                            worker.validate_model_finite(model)
                        else:
                            with self.assertRaisesRegex(ValueError, "non-finite model tensor: bad"):
                                worker.validate_model_finite(model)


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


class EvaluationMetricTests(unittest.TestCase):
    """ADR 0081 exact intrinsic metrics over WALDO's pinned held-out views."""

    def pinned(self, targets, text_bytes):
        return dict(records=1, token_targets=targets, text_bytes=text_bytes)

    def test_masks_eos_and_long_records_score_each_target_once(self):
        # A masked prompt token, a record longer than one window (as a
        # contiguous tail produces), and the appended EOS target.
        records = [dict(tokens=[3, 4, 5], loss_mask=[False, False, True, True], corpus="a"),
                   dict(tokens=list(range(3, 13)), corpus="b")]
        windows = list(worker.windows(records, 4, 0, 259, 2, continuous=False))
        # Record 1 scores 5 and EOS; record 2 scores its last 9 tokens and EOS.
        self.assertEqual(worker.evaluation_targets(windows, self.pinned(12, 13)), 12)
        with self.assertRaisesRegex(ValueError, "pins 13"):
            worker.evaluation_targets(windows, self.pinned(13, 13))

    def test_exact_bits_per_byte_uses_summed_nll_and_pinned_bytes(self):
        nll = 41.5
        metrics = worker.heldout_metrics(nll, 20, self.pinned(20, 30))
        self.assertEqual(metrics["heldout_nll_sum"], nll)
        self.assertEqual(metrics["heldout_target_tokens"], 20.0)
        self.assertEqual(metrics["heldout_utf8_bytes"], 30.0)
        self.assertEqual(metrics["heldout_loss"], nll / 20)
        self.assertAlmostEqual(metrics["heldout_perplexity"], math.exp(nll / 20))
        self.assertEqual(metrics["heldout_bits_per_byte"], nll / (math.log(2.0) * 30))

    def test_unicode_byte_and_tokenizer_denominators(self):
        # Seven characters, ten UTF-8 bytes. The byte tokenizer scores one
        # target per byte; a subword tokenizer scores fewer, larger targets.
        text = "na\u00efve \u2603"
        text_bytes = len(text.encode("utf-8"))
        self.assertEqual((len(text), text_bytes), (7, 10))
        byte_record = dict(tokens=[value + 3 for value in text.encode("utf-8")])
        subword_record = dict(tokens=[10, 11, 12])
        byte_targets = worker.evaluation_targets(list(worker.windows([byte_record], 8, 0, 259, 2, continuous=False)), self.pinned(10, text_bytes))
        subword_targets = worker.evaluation_targets(list(worker.windows([subword_record], 8, 0, 259, 2, continuous=False)), self.pinned(3, text_bytes))
        self.assertEqual((byte_targets, subword_targets), (10, 3))
        # Equal total NLL is equal BPB, although per-token loss differs.
        byte_metrics = worker.heldout_metrics(22.0, byte_targets, self.pinned(10, text_bytes))
        subword_metrics = worker.heldout_metrics(22.0, subword_targets, self.pinned(3, text_bytes))
        self.assertEqual(byte_metrics["heldout_bits_per_byte"], subword_metrics["heldout_bits_per_byte"])
        self.assertNotEqual(byte_metrics["heldout_loss"], subword_metrics["heldout_loss"])

    def test_inconsistent_or_nonfinite_totals_fail_closed(self):
        for nll, targets, pinned in [(1.0, 3, self.pinned(4, 8)), (1.0, 0, self.pinned(0, 8)),
                                     (math.nan, 4, self.pinned(4, 8)), (math.inf, 4, self.pinned(4, 8)), (-1.0, 4, self.pinned(4, 8))]:
            with self.subTest(nll=nll, targets=targets), self.assertRaises(worker.ArtifactIntegrityError):
                worker.heldout_metrics(nll, targets, pinned)

    def test_missing_byte_evidence_omits_bits_per_byte(self):
        self.assertNotIn("heldout_bits_per_byte", worker.heldout_metrics(2.0, 2, self.pinned(2, 0)))


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

    def test_wrong_wheel_filename(self):
        self.wheel = self.wheel.rename(self.root / "other.whl")
        with self.assertRaisesRegex(ValueError, "filename"):
            worker.verify_package(self.wheel, self.pin)

    def test_installed_version_mismatch(self):
        worker.importlib.metadata.distribution("transformers").version = "5.16.0"
        with self.assertRaisesRegex(ValueError, "installed Transformers version"):
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


class TrainerPlanTests(unittest.TestCase):
    def arguments(self, **overrides):
        values = dict(device=types.SimpleNamespace(type="cuda"), world_size=1, n_gpu=1, per_device_train_batch_size=1,
                      gradient_accumulation_steps=2, learning_rate=0.001, fp16=False, bf16=True, gradient_checkpointing=True)
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def parameters(self, **overrides):
        values = dict(steps=4, batch_size=2, gradient_accumulation_steps=2, learning_rate=0.001, compute_precision="bfloat16", activation_checkpointing=True)
        values.update(overrides)
        return values

    def test_resolved_plan_matches_trainer(self):
        worker.check_trainer_plan(self.arguments(), self.parameters(), "cuda")
        worker.check_trainer_plan(self.arguments(per_device_train_batch_size=2, gradient_accumulation_steps=1, bf16=False, gradient_checkpointing=False),
                                  dict(steps=4, batch_size=2, learning_rate=0.001), "cuda")

    def test_divergent_plan_fails_closed(self):
        for arguments, parameters in [
                (self.arguments(per_device_train_batch_size=2, gradient_accumulation_steps=1), self.parameters()),
                (self.arguments(), self.parameters(gradient_accumulation_steps=1, batch_size=2)),
                (self.arguments(learning_rate=0.002), self.parameters()),
                (self.arguments(bf16=False), self.parameters()),
                (self.arguments(bf16=False, fp16=True), self.parameters()),
                (self.arguments(), self.parameters(compute_precision="auto")),
                (self.arguments(gradient_checkpointing=False), self.parameters()),
                (self.arguments(world_size=2), self.parameters()),
                (self.arguments(device=types.SimpleNamespace(type="cpu")), self.parameters())]:
            with self.subTest(arguments=arguments, parameters=parameters), self.assertRaises(ValueError):
                worker.check_trainer_plan(arguments, parameters, "cuda")

    def test_compute_dtype_follows_parameters_or_autocast(self):
        self.assertEqual(worker.compute_dtype(dict(compute_precision="auto"), "float32"), "float32")
        self.assertEqual(worker.compute_dtype(dict(), "bfloat16"), "bfloat16")
        self.assertEqual(worker.compute_dtype(dict(compute_precision="float16"), "float32"), "float16")
        self.assertEqual(worker.compute_dtype(dict(compute_precision="bfloat16"), "float32"), "bfloat16")
        with self.assertRaises(ValueError):
            worker.compute_dtype(dict(compute_precision="float64"), "float32")

    def test_resolution_reports_effective_allowlist(self):
        arguments = types.SimpleNamespace(to_dict=lambda: dict(learning_rate=0.001, optim="adamw_torch_fused", output_dir="/run"),
                                          get_warmup_steps=lambda steps: math.ceil(steps * 0.5))
        resolution = worker.trainer_resolution(arguments, dict(steps=3), ["learning_rate", "optim"])
        self.assertEqual(resolution, dict(arguments=dict(learning_rate=0.001, optim="adamw_torch_fused"), warmup_steps=2))

    def test_strict_value_comparison(self):
        self.assertTrue(worker.same_value(dict(a=[1, 2.0], b=None), dict(a=[1.0, 2], b=None)))
        for left, right in [(True, 1), (0, False), ("1", 1), (dict(a=1), dict(a=1, b=2)), ([1], [1, 1]), (None, 0)]:
            with self.subTest(left=left, right=right):
                self.assertFalse(worker.same_value(left, right))

    def test_bf16_requires_compatible_parameters(self):
        torch = types.SimpleNamespace(tensor=lambda *args, **kwargs: types.SimpleNamespace(sum=lambda: types.SimpleNamespace(item=lambda: 1)),
                                      cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda: True,
                                                                 get_device_properties=lambda _: types.SimpleNamespace(name="GPU", total_memory=1024)))
        facts = dict(device="cuda", accelerator="GPU", memory_bytes=1024)
        self.assertEqual(worker.validate_device(torch, facts, dict(bf16=True), "bfloat16"), "cuda")
        with self.assertRaises(ValueError):
            worker.validate_device(torch, facts, dict(bf16=True), "float16")


class PublicationSafetyTests(unittest.TestCase):
    """ADR 0044/0079 decisions that do not need a real model."""

    def candidate(self, step, loss, tokens=None):
        tokens = step * 64 if tokens is None else tokens
        return dict(step=step, tokens=tokens, metrics=dict(heldout_loss=loss)), dict(step=step, tokens=step * 64, artifacts=[{}])

    def history(self, *losses):
        pairs = [self.candidate(step, loss) for step, loss in enumerate(losses, start=1)]
        return [evaluation for evaluation, _ in pairs], [checkpoint for _, checkpoint in pairs]

    def test_worse_final_candidate_does_not_replace_earlier_best(self):
        evaluations, checkpoints = self.history(2.0, 1.25, 1.75)
        selected, checkpoint = worker.select_candidate(evaluations, checkpoints)
        self.assertEqual((selected["step"], checkpoint["step"]), (2, 2))

    def test_ties_select_the_earliest_candidate(self):
        evaluations, checkpoints = self.history(1.5, 1.0, 1.0)
        self.assertEqual(worker.select_candidate(list(reversed(evaluations)), checkpoints)[0]["step"], 2)

    def test_only_persisted_consistent_candidates_are_eligible(self):
        evaluations, checkpoints = self.history(1.0, 2.0)
        self.assertEqual(worker.select_candidate(evaluations, checkpoints[1:])[0]["step"], 2)
        with self.assertRaises(worker.ArtifactIntegrityError):
            worker.select_candidate(evaluations, [])
        evaluations[1]["tokens"] = 1
        with self.assertRaisesRegex(worker.ArtifactIntegrityError, "inconsistent"):
            worker.select_candidate(evaluations, checkpoints)
        evaluations, checkpoints = self.history(1.0, math.nan)
        with self.assertRaisesRegex(worker.ArtifactIntegrityError, "inconsistent"):
            worker.select_candidate(evaluations, checkpoints)

    def test_agreement_tolerance_fails_closed(self):
        self.assertEqual(worker.loss_tolerance(1.0), 0.02)
        self.assertEqual(worker.loss_tolerance(-5.0), 0.05)
        worker.check_agreement(5.04, 5.0, "candidate", "fix")
        for value, reference in [(5.06, 5.0), (1.03, 1.0), (math.nan, 1.0), (math.inf, 1.0), (1.0, math.nan)]:
            with self.subTest(value=value, reference=reference), self.assertRaises(worker.ArtifactIntegrityError):
                worker.check_agreement(value, reference, "candidate", "fix")

    def test_nonfinite_gradient_norm_fails_before_update(self):
        self.assertEqual(worker.checked_gradient_norm(types.SimpleNamespace(item=lambda: 0.5), 1), 0.5)
        self.assertEqual(worker.checked_gradient_norm(2, 1), 2.0)
        for value in [math.nan, math.inf, -math.inf]:
            with self.subTest(value=value), self.assertRaisesRegex(worker.NumericalIntegrityError, "optimizer step 3"):
                worker.checked_gradient_norm(value, 3)

    def test_safety_and_periodic_boundaries(self):
        self.assertEqual([step for step in range(1, 8) if worker.evaluation_due(step, 7, 0)], [1, 7])
        self.assertEqual([step for step in range(1, 8) if worker.evaluation_due(step, 7, 3)], [1, 3, 6, 7])

    def test_live_parameters_are_separated_from_autocast(self):
        model = types.SimpleNamespace(forward="autocast", _original_forward="eager")
        with worker.without_autocast(model):
            self.assertEqual(model.forward, "eager")
        self.assertEqual(model.forward, "autocast")
        with self.assertRaises(worker.ArtifactIntegrityError), worker.without_autocast(types.SimpleNamespace(forward="autocast")):
            pass


if __name__ == "__main__":
    unittest.main()
