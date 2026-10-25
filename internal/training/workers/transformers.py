# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0

"""Experimental CPU Trainer adapter; receives WALDO's canonical token stream."""

import hashlib
import contextlib
import importlib.metadata
import importlib.abc
import importlib.machinery
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback
import zipfile

REVISION = "builtin-transformers-worker-schema-1-r8"
PROTOCOL_OUTPUT = sys.stdout


class ArtifactIntegrityError(ValueError):
    """Persisted or precision-converted weights disagree with the live model."""


class NumericalIntegrityError(ValueError):
    """A training loss or optimizer-step gradient norm is not finite."""


@contextlib.contextmanager
def pinned_tokenizer(package, spec, source):
    """Load only verified data files, with no Hub access or custom Python code."""
    pin = spec["huggingface"]
    allowed = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"}
    if not {"tokenizer.json", "tokenizer_config.json"}.issubset(pin["files"]):
        raise ValueError("missing tokenizer file pins")
    with tempfile.TemporaryDirectory(prefix="waldo-tokenizer-") as temporary:
        root = Path(temporary)
        for name, expected in pin["files"].items():
            if name not in allowed:
                raise ValueError("unsupported tokenizer asset")
            path = Path(source) / name
            with path.open("rb") as stream:
                data = stream.read(128 * 1024 * 1024 + 1)
            if len(data) > 128 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"tokenizer asset hash/size mismatch: {name}")
            (root / name).write_bytes(data)
        configuration = json.loads((root / "tokenizer_config.json").read_text())
        if configuration.get("auto_map"):
            raise ValueError("tokenizer custom code is not supported")
        if any(key.endswith("_file") or key.endswith("_files") for key in configuration):
            raise ValueError("tokenizer configuration may not redirect asset paths")
        tokenizer = package.AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False, use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError("only tokenizer.json-backed fast tokenizers are supported")
        vocabulary = tokenizer.get_vocab()
        if not vocabulary or min(vocabulary.values()) < 0 or max(vocabulary.values()) >= spec["vocabulary_size"]:
            raise ValueError("tokenizer vocabulary exceeds model embeddings")
        for name in ("pad", "eos", "bos"):
            actual = getattr(tokenizer, name + "_token_id")
            expected = spec[name + "_id"]
            if actual != (None if expected == -1 else expected):
                raise ValueError(f"tokenizer {name} ID differs from compose")
        yield tokenizer, root


def tokenizer_service():
    payload = json.loads(sys.argv[4])
    package = verify_package(sys.argv[2], payload["package"])
    with pinned_tokenizer(package, payload["tokenizer"], sys.argv[3]) as (tokenizer, _):
        print(json.dumps({"ready": True}), file=PROTOCOL_OUTPUT, flush=True)
        for line in sys.stdin:
            text = json.loads(line)["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            print(json.dumps({"tokens": tokens}), file=PROTOCOL_OUTPUT, flush=True)


def emit(kind, **payload):
    print(json.dumps(dict(kind=kind, schema=1, **payload), allow_nan=False), file=PROTOCOL_OUTPUT, flush=True)


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_package(wheel, pin):
    """Compare installed package bytes to the pinned wheel before importing it.

    This is integrity evidence, not a sandbox against a hostile Python runtime.
    Dependencies are inventoried separately, not claimed to be wheel-pinned.
    """
    if Path(wheel).name != pin["artifact"] or digest(wheel) != pin["sha256"]:
        raise ValueError("Transformers wheel filename or SHA-256 differs from training.package")
    distribution = importlib.metadata.distribution("transformers")
    if distribution.version != pin["version"]:
        raise ValueError("installed Transformers version differs from training.package")
    root = Path(distribution.locate_file("")).resolve()
    expected = set()
    source_hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith("transformers/"):
                continue
            path = (root / entry.filename).resolve()
            if not path.is_relative_to(root / "transformers"):
                raise ValueError("invalid Transformers wheel member path")
            if not path.is_file() or digest(path) != hashlib.sha256(archive.read(entry)).hexdigest():
                raise ValueError(f"installed Transformers file differs from wheel: {entry.filename}")
            expected.add(entry.filename)
            if path.suffix == ".py":
                source_hashes[str(path)] = hashlib.sha256(archive.read(entry)).hexdigest()
    if "transformers/__init__.py" not in expected:
        raise ValueError("wheel does not contain the Transformers package")
    for path in (root / "transformers").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            if path.relative_to(root).as_posix() not in expected:
                raise ValueError(f"unattested file in installed Transformers package: {path.name}")
    # Compile verified source, ignoring installed bytecode caches. Check again
    # when each lazy module is loaded, so changed source cannot race the probe.
    class SourceLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            data = self.get_data(self.path)
            if hashlib.sha256(data).hexdigest() != source_hashes.get(str(Path(self.path).resolve())):
                raise ImportError(f"Transformers source changed after verification: {fullname}")
            return self.source_to_code(data, self.path)

    class SourceFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "transformers" and not fullname.startswith("transformers."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.origin is None or str(Path(spec.origin).resolve()) not in source_hashes:
                raise ImportError(f"Transformers module is outside the pinned wheel: {fullname}")
            spec.loader = SourceLoader(fullname, spec.origin)
            return spec

    sys.meta_path.insert(0, SourceFinder())
    import transformers
    if Path(transformers.__file__).resolve() != root / "transformers/__init__.py":
        raise ValueError("Transformers import was shadowed outside the verified package")
    return transformers


def configuration(transformers, spec):
    if MODEL_CLASSES.get(spec["model_class"]) != spec["config_class"]:
        raise ValueError("unsupported Transformers model/config class pair")
    config_class = getattr(transformers, spec["config_class"])
    defaults = config_class().to_dict()
    unknown = set(spec["config"]) - set(defaults)
    if unknown:
        raise ValueError(f"unknown architecture.config fields: {sorted(unknown)}")
    config = config_class(**spec["config"])
    if config.model_type != defaults["model_type"]:
        raise ValueError("model_type does not match config_class")
    if config.model_type == "mixtral":
        if not 1 <= config.num_experts_per_tok <= config.num_local_experts:
            raise ValueError("Mixtral requires 1 <= num_experts_per_tok <= num_local_experts")
        if not config.output_router_logits:
            raise ValueError("Mixtral requires output_router_logits=true for router auxiliary loss")
        if not math.isfinite(config.router_aux_loss_coef) or config.router_aux_loss_coef < 0:
            raise ValueError("invalid Mixtral router_aux_loss_coef")
    # Random initialization only. Never from_pretrained, remote code, or Hub I/O.
    return config


def validate_tokenizer_configuration(config, tokenizer):
    for name in ("pad", "bos", "eos"):
        expected = tokenizer[name + "_id"]
        if getattr(config, name + "_token_id", None) != (None if expected == -1 else expected):
            raise ValueError(f"model config {name} ID differs from tokenizer framing")


def runtime_facts(device="cpu"):
    dependencies = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "safetensors", "tokenizers")}
    return dict(python=platform.python_version(), dependencies=dependencies, device=device, worker=REVISION)


def validate_device(torch, facts, arguments=None, dtype="float32"):
    device = facts["device"]
    if device not in ("cpu", "cuda") or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Transformers requires one CPU process or one GPU")
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("expected exactly one visible CUDA/ROCm GPU")
        properties = torch.cuda.get_device_properties(0)
        if properties.name != facts["accelerator"] or properties.total_memory != facts["memory_bytes"]:
            raise ValueError("GPU identity changed after runtime resolution")
    arguments = arguments or {}
    if arguments.get("fp16") and arguments.get("bf16"):
        raise ValueError("fp16 and bf16 are mutually exclusive")
    if device == "cpu" and (arguments.get("fp16") or arguments.get("bf16")):
        raise ValueError("mixed precision requires a GPU in this adapter")
    if device == "cuda" and (dtype == "bfloat16" or arguments.get("bf16")) and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support bfloat16")
    if arguments.get("fp16") and dtype != "float32":
        raise ValueError("fp16 Trainer scaling requires float32 model parameters")
    if arguments.get("bf16") and dtype not in ("float32", "bfloat16"):
        raise ValueError("bf16 Trainer autocast requires float32 or bfloat16 model parameters")
    torch.tensor([1.0], device=device).sum().item()
    return device


def training_arguments(transformers, parameters, device, output_dir):
    """Construct the only TrainingArguments WALDO executes: compose passthroughs
    plus WALDO-owned paths, budget, seed, data loading, and publication."""
    arguments = dict(parameters["trainer"]["arguments"])
    arguments.update(output_dir=str(output_dir), max_steps=parameters["steps"], seed=parameters["seed"], data_seed=parameters["seed"],
                     use_cpu=device == "cpu", report_to="none", push_to_hub=False, save_strategy="no", eval_strategy="no", disable_tqdm=True,
                     dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=False, prediction_loss_only=True)
    return transformers.TrainingArguments(**arguments)


def compute_dtype(parameters, dtype):
    """The floating-point type of forward computation implied by the resolved plan."""
    precision = parameters.get("compute_precision", "auto")
    if precision not in ("auto", "float16", "bfloat16"):
        raise ValueError("unsupported Transformers compute_precision")
    return dtype if precision == "auto" else precision


def check_trainer_plan(args, parameters, device):
    """Fail unless Trainer will execute WALDO's resolved batch, precision, and recomputation."""
    if args.device.type != device or args.world_size != 1 or (device == "cuda" and args.n_gpu != 1):
        raise ValueError("Trainer device/topology differs from WALDO runtime resolution")
    accumulation = parameters.get("gradient_accumulation_steps", 1)
    if args.gradient_accumulation_steps != accumulation or args.per_device_train_batch_size * accumulation != parameters["batch_size"]:
        raise ValueError("Trainer batch or gradient accumulation differs from WALDO's resolved plan")
    if args.learning_rate != parameters["learning_rate"]:
        raise ValueError("Trainer learning rate differs from WALDO's resolved plan")
    precision = "float16" if args.fp16 else "bfloat16" if args.bf16 else "auto"
    if precision != parameters.get("compute_precision", "auto"):
        raise ValueError("Trainer mixed precision differs from WALDO's resolved compute_precision")
    if bool(args.gradient_checkpointing) != bool(parameters.get("activation_checkpointing", False)):
        raise ValueError("Trainer gradient checkpointing differs from WALDO's resolved activation_checkpointing")


def trainer_resolution(args, parameters, names):
    values = args.to_dict()
    resolution = dict(arguments={name: values[name] for name in names}, warmup_steps=args.get_warmup_steps(parameters["steps"]))
    # Round-trip through strict JSON so the recorded and executed forms match.
    return json.loads(json.dumps(resolution, allow_nan=False))


def same_value(left, right):
    """JSON equality that does not let bool and number values stand in for each other."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_value(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def resolve_trainer():
    """Report the pinned package's effective Trainer settings before a run exists."""
    request = json.loads(sys.argv[3])
    device_facts = json.loads(sys.argv[4])
    spec = request["architecture"]["transformers"]
    parameters = request["parameters"]
    transformers = verify_package(sys.argv[2], spec["package"])
    import torch
    dtype = spec["config"].get("dtype", "float32")
    device = validate_device(torch, device_facts, parameters["trainer"]["arguments"], dtype)
    with tempfile.TemporaryDirectory(prefix="waldo-trainer-") as temporary:
        args = training_arguments(transformers, parameters, device, Path(temporary) / "trainer")
        check_trainer_plan(args, parameters, device)
        return trainer_resolution(args, parameters, request["arguments"])


def parameter_evidence(model, config):
    total = sum(p.numel() for p in model.parameters())
    result = dict(parameters=total)
    if config.model_type == "mixtral":
        experts = sum(p.numel() for name, p in model.named_parameters() if ".experts." in name)
        if experts <= 0:
            raise ValueError("cannot identify Mixtral expert parameters")
        result.update(expert_parameters=experts, experts_per_token=config.num_experts_per_tok,
                      experts_per_layer=config.num_local_experts,
                      active_parameters_estimate=total-experts+experts*config.num_experts_per_tok//config.num_local_experts,
                      active_parameters_definition="all non-expert parameters plus selected expert fraction; not activation memory or FLOPs")
    return result


def validate_model_finite(model):
    """Check the final optimizer update too, before publishing any artifacts."""
    import torch
    for values in (model.named_parameters(), model.named_buffers()):
        for name, value in values:
            if value.is_floating_point() or value.is_complex():
                # Bound temporary isfinite masks even for large embedding tables.
                for chunk in value.detach().reshape(-1).split(1024 * 1024):
                    if not torch.isfinite(chunk).all().item():
                        raise ValueError(f"non-finite model tensor: {name}")


def loss_tolerance(reference):
    """ADR 0044 agreement tolerance: the greater of 0.02 loss or one percent."""
    return max(0.02, abs(reference) * 0.01)


def check_agreement(value, reference, description, remedy):
    if not math.isfinite(value) or not math.isfinite(reference):
        raise ArtifactIntegrityError(f"{description} held-out loss is not finite")
    tolerance = loss_tolerance(reference)
    if abs(value - reference) > tolerance:
        raise ArtifactIntegrityError(f"{description} held-out loss {value:.6f} does not match {reference:.6f} within tolerance {tolerance:.6f}; {remedy}")


def checked_gradient_norm(value, step):
    """Fail before the optimizer applies an update from a non-finite gradient."""
    norm = float(value.item() if hasattr(value, "item") else value)
    if not math.isfinite(norm):
        raise NumericalIntegrityError(f"non-finite gradient norm at optimizer step {step}; no optimizer update was applied")
    return norm


def evaluation_due(step, steps, evaluate_every):
    """Step 1 and the terminal step are safety boundaries; evaluate_every adds periodic ones."""
    return step == 1 or step == steps or (evaluate_every > 0 and step % evaluate_every == 0)


def evaluation_targets(windows, evaluation_set):
    """Scored held-out targets; they must equal the run BOM's pinned count."""
    targets = sum(sum(label != -100 for label in window["labels"]) for window in windows)
    if targets != evaluation_set["token_targets"]:
        raise ValueError(f"evaluation stream has {targets} scored targets; run BOM pins {evaluation_set['token_targets']}")
    return targets


def heldout_metrics(nll, targets, evaluation_set):
    """ADR 0081 exact intrinsic metrics from the summed NLL over the pinned view.

    BPB divides by the pinned UTF-8 bytes of the held-out text, never by an
    average bytes-per-token estimate. It is intrinsic distribution fit only.
    """
    if targets <= 0 or targets != evaluation_set["token_targets"]:
        raise ArtifactIntegrityError(f"held-out NLL covers {targets} targets; run BOM pins {evaluation_set['token_targets']}")
    if not math.isfinite(nll) or nll < 0:
        raise ArtifactIntegrityError("held-out NLL is not finite and non-negative")
    loss = nll / targets
    text_bytes = evaluation_set["text_bytes"]
    metrics = dict(heldout_loss=loss, heldout_perplexity=math.exp(min(loss, 80.0)), heldout_nll_sum=nll,
                   heldout_target_tokens=float(targets), heldout_utf8_bytes=float(text_bytes))
    if text_bytes > 0:
        metrics["heldout_bits_per_byte"] = nll / (math.log(2.0) * text_bytes)
    return metrics


def select_candidate(evaluations, checkpoints):
    """Lowest persisted publishable held-out loss; ties keep the earliest step.

    The full budget has already run: selection chooses what to publish and is
    not early stopping.
    """
    persisted = {checkpoint["step"]: checkpoint for checkpoint in checkpoints}
    candidates = sorted((item for item in evaluations if item["step"] in persisted), key=lambda item: item["step"])
    if not candidates:
        raise ArtifactIntegrityError("no evaluated candidate checkpoint was persisted")
    for item in candidates:
        if persisted[item["step"]]["tokens"] != item["tokens"] or not math.isfinite(item["metrics"]["heldout_loss"]):
            raise ArtifactIntegrityError(f"candidate evidence at step {item['step']} is inconsistent")
    selected = min(candidates, key=lambda item: item["metrics"]["heldout_loss"])
    return selected, persisted[selected["step"]]


@contextlib.contextmanager
def without_autocast(model):
    """Run the live model without Accelerate's mixed-precision forward wrapper."""
    original = getattr(model, "_original_forward", None)
    if original is None:
        raise ArtifactIntegrityError("cannot separate mixed-precision execution from the live parameters")
    patched = model.forward
    model.forward = original
    try:
        yield model
    finally:
        model.forward = patched


def write_json(path, value):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def frames(stream):
    for line in stream:
        frame = json.loads(line)
        if frame.get("schema") != 1:
            raise ValueError("unsupported worker input schema")
        yield frame


def windows(records, length, pad, vocabulary, eos, continuous=True):
    tokens, masks, corpora = [], [], []

    def sample():
        piece = tokens[:length + 1]
        target_mask = masks[1:len(piece)]
        targets = [token if keep else -100 for token, keep in zip(piece[1:], target_mask)]
        counts = {}
        for corpus, keep in zip(corpora[1:len(piece)], target_mask):
            if keep:
                counts[corpus] = counts.get(corpus, 0) + 1
        return {"input_ids": piece[:-1] + [pad] * (length - len(piece) + 1),
                "attention_mask": [1] * (len(piece)-1) + [0] * (length-len(piece)+1),
                "labels": targets + [-100] * (length - len(targets)), "_corpora": counts}

    for record in records:
        encoded = record.get("tokens")
        if encoded is None or any(type(token) is not int or not 0 <= token < vocabulary for token in encoded):
            raise ValueError("Transformers requires valid WALDO-pretokenized records")
        encoded = encoded + [eos]
        mask = record.get("loss_mask", [True] * len(encoded))
        if len(mask) != len(encoded) or any(type(value) is not bool for value in mask):
            raise ValueError("loss mask differs from token framing")
        tokens.extend(encoded)
        masks.extend(mask)
        corpora.extend([record.get("corpus", "")] * len(encoded))
        while len(tokens) >= length + 1:
            item = sample()
            if item["_corpora"]:
                yield item
            del tokens[:length], masks[:length], corpora[:length]
        if not continuous:
            if len(tokens) > 1 and any(masks[1:]):
                yield sample()
            tokens, masks, corpora = [], [], []
    if len(tokens) > 1 and any(masks[1:]):
        yield sample()


def run():
    directory, prefix, wheel = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device_facts = json.loads(sys.argv[4])
    stream = frames(sys.stdin)
    first = next(stream)
    if first["kind"] != "begin":
        raise ValueError("missing begin frame")
    begin = first["begin"]
    if begin.get("resume"):
        raise ValueError("Transformers checkpoint resume is not supported yet")
    spec = begin["architecture"]["transformers"]
    transformers = verify_package(wheel, spec["package"])
    import torch
    from torch.utils.data import IterableDataset
    from safetensors.torch import load_model, save_model

    parameters = begin["parameters"]
    trainer_spec = parameters["trainer"]
    device = validate_device(torch, device_facts, trainer_spec["arguments"], spec["config"].get("dtype", "float32"))
    config = configuration(transformers, spec)
    validate_tokenizer_configuration(config, begin["tokenizer"])
    if begin["tokenizer"].get("huggingface"):
        # Verify the assets before training, and again when saving the artifact.
        with pinned_tokenizer(transformers, begin["tokenizer"], os.environ["WALDO_HF_TOKENIZER_DIR"]):
            pass
    transformers.set_seed(parameters["seed"])
    model = getattr(transformers, spec["model_class"])(config)
    if begin.get("initialization"):
        load_model(model, begin["initialization"]["path"], strict=True)

    # Explicit dtype belongs to model identity; mixed precision is a separate
    # Trainer option. Native projections are not used to construct this model.
    dtype = spec["config"].get("dtype", "float32")
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("unsupported model dtype")
    model.to(dtype=getattr(torch, dtype))
    evaluation_records = []
    pending = None
    for frame in stream:
        if frame["kind"] == "evaluation_record":
            evaluation_records.append(frame["record"])
        else:
            pending = frame
            break

    def records():
        nonlocal pending
        if pending is None:
            raise ValueError("worker input ended without end frame")
        frame, pending = pending, None
        while True:
            if frame["kind"] == "end":
                return
            if frame["kind"] != "record":
                raise ValueError("invalid frame in training stream")
            yield frame["record"]
            frame = next(stream, None)
            if frame is None:
                raise ValueError("worker input ended without end frame")

    tokenizer = begin["tokenizer"]
    length = parameters["sequence_length"]
    def pack(source, continuous=True):
        return windows(source, length, tokenizer["pad_id"], tokenizer["vocabulary_size"], tokenizer["eos_id"], continuous)

    class Dataset(IterableDataset):
        def __iter__(self):
            return iter(pack(records()))

    def collate(items):
        return {"input_ids": torch.tensor([x["input_ids"] for x in items]),
                "attention_mask": torch.tensor([x["attention_mask"] for x in items]),
                "labels": torch.tensor([x["labels"] for x in items]),
                "_corpora": [x["_corpora"] for x in items]}

    consumption = {}
    expected_dtype = getattr(torch, compute_dtype(parameters, dtype))
    execution = dict(micro_batches=0, sequences=0, compute_dtypes=set(), head_dtype=None, gradient_norms=0)

    # Accelerate converts autocast outputs back to float32, so observe the
    # output head's own result to see the precision that actually executed.
    def observe_head(module, inputs, output):
        execution["head_dtype"] = output.dtype
    model.get_output_embeddings().register_forward_hook(observe_head)
    class WaldoTrainer(transformers.Trainer):
        def _get_num_items_in_batch(self, batch_samples, device):
            # Our labels are already shifted, unlike standard causal-LM labels.
            return sum(batch["labels"].ne(-100).sum() for batch in batch_samples).to(device) if batch_samples else None

        def _get_grad_norm(self, model, grad_norm=None):
            # Trainer computes the (unscaled, pre-clip) norm here immediately
            # before optimizer.step(), even when max_grad_norm disables clipping.
            norm = checked_gradient_norm(super()._get_grad_norm(model, grad_norm=grad_norm), self.state.global_step + 1)
            execution["gradient_norms"] += 1
            execution["gradient_norm"] = norm
            return norm

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            counts = inputs.pop("_corpora")
            outputs = model(**inputs)
            # Labels are already shifted by WALDO's continuous-eos-v1 packer.
            total = torch.nn.functional.cross_entropy(outputs.logits.float().reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100, reduction="sum")
            targets = labels.ne(-100).sum()
            loss = total / (num_items_in_batch if num_items_in_batch is not None else targets)
            if config.model_type == "mixtral":
                auxiliary = outputs.aux_loss
                if auxiliary is None or not torch.isfinite(auxiliary).all():
                    raise NumericalIntegrityError("Mixtral did not return a finite router auxiliary loss")
                # Scale the per-microbatch mean consistently with WALDO's
                # target-weighted gradient accumulation. Evaluation metrics
                # below remain pure held-out cross entropy, not router loss.
                weight = targets / (num_items_in_batch if num_items_in_batch is not None else targets)
                loss = loss + config.router_aux_loss_coef * auxiliary * weight
            # Trainer's logging filter can hide NaN/Inf in training_loss.
            # Reject the actual objective before backward, not its logged mean.
            if not torch.isfinite(loss).all().item():
                raise NumericalIntegrityError("non-finite model loss")
            if not model.training:
                raise ValueError("WALDO evaluates held-out data outside Trainer.compute_loss")
            # Observe what executed, not only what was requested.
            observed, execution["head_dtype"] = execution["head_dtype"], None
            if observed != expected_dtype:
                raise ValueError(f"Trainer computed {observed}, not the resolved {expected_dtype}")
            execution["micro_batches"] += 1
            execution["sequences"] += labels.shape[0]
            execution["compute_dtypes"].add(str(observed).removeprefix("torch."))
            for item in counts:
                for corpus, count in item.items():
                    consumption[corpus] = consumption.get(corpus, 0) + count
            return (loss, outputs) if return_outputs else loss

    args = training_arguments(transformers, parameters, device, directory / "trainer")
    check_trainer_plan(args, parameters, device)
    pinned = parameters.get("trainer_resolution")
    if pinned is None:
        raise ValueError("run BOM is missing the pinned Trainer resolution")
    if not same_value(trainer_resolution(args, parameters, sorted(pinned["arguments"])), pinned):
        raise ValueError("Trainer arguments differ from the pinned resolution in the run BOM")

    evaluation = list(pack(evaluation_records, continuous=False))
    if len(evaluation_records) != begin["evaluation_set"]["records"]:
        raise ValueError(f"evaluation stream has {len(evaluation_records)} records; run BOM pins {begin['evaluation_set']['records']}")
    if evaluation_records and not evaluation:
        raise ValueError("held-out records produced no scored evaluation targets")
    targets = evaluation_targets(evaluation, begin["evaluation_set"])
    mixed_precision = bool(args.fp16 or args.bf16)
    publication_dtype = dtype if device == "cuda" else "float32"
    evaluations, checkpoints = [], []

    def heldout_nll(target):
        """Summed held-out NLL in nats over WALDO's pinned evaluation windows."""
        was_training = target.training
        target.eval()
        total, count = 0.0, 0
        try:
            with torch.no_grad():
                for offset in range(0, len(evaluation), args.per_device_eval_batch_size):
                    batch = collate(evaluation[offset:offset + args.per_device_eval_batch_size])
                    labels = batch["labels"].to(device)
                    outputs = target(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), use_cache=False)
                    # Per-target FP32 cross entropy, summed in FP64 so the
                    # total is not rounded at FP32 resolution.
                    losses = torch.nn.functional.cross_entropy(outputs.logits.float().reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100, reduction="none")
                    total += losses.sum(dtype=torch.float64).item()
                    count += labels.ne(-100).sum().item()
        finally:
            target.train(was_training)
        if count != targets:
            raise ArtifactIntegrityError(f"held-out evaluation scored {count} targets; run BOM pins {targets}")
        return total

    def heldout_loss(target):
        return heldout_nll(target) / targets

    def construct(target_dtype):
        """Build a fresh model exactly as chat does, without perturbing training RNG."""
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
            fresh = getattr(transformers, spec["model_class"])(getattr(transformers, spec["config_class"])(**json.loads(json.dumps(config.to_dict()))))
        return fresh.to(device=device, dtype=getattr(torch, target_dtype))

    def reload_heldout_nll(path):
        # Chat reloads config.json and model.safetensors into a new model in
        # the parameter dtype on CUDA, FP32 on CPU, eager and without autocast.
        reloaded = construct(publication_dtype)
        load_model(reloaded, str(path), strict=True)
        nll = heldout_nll(reloaded)
        del reloaded
        return nll

    def save_weights(source, path):
        temporary = path.with_name(path.name + ".tmp")
        save_model(source, str(temporary), metadata={"format": "pt", "waldo_backend": "huggingface-transformers", "waldo_architecture_sha256": begin["architecture_sha256"]})
        os.replace(temporary, path)

    def record_evaluation(step):
        compute_loss = heldout_loss(model)
        if mixed_precision:
            with without_autocast(model):
                live_loss = heldout_loss(model)
        else:
            live_loss = compute_loss
        if not math.isfinite(compute_loss) or not math.isfinite(live_loss):
            raise ArtifactIntegrityError(f"live held-out loss at step {step} is not finite")
        check_agreement(compute_loss, live_loss, f"step {step} {expected_dtype} compute-precision",
                        "correct compute-precision behavior before spending more compute")
        tokens = sum(consumption.values())
        name = f"checkpoints/step-{step:08d}"
        candidate = directory / name
        candidate.mkdir(parents=True, exist_ok=False)
        save_weights(model, candidate / "model.safetensors")
        metrics = heldout_metrics(reload_heldout_nll(candidate / "model.safetensors"), targets, begin["evaluation_set"])
        publishable = metrics["heldout_loss"]
        check_agreement(publishable, live_loss, f"step {step} persisted {publication_dtype} candidate",
                        "correct checkpoint serialization or dtype conversion before spending more compute")
        checkpoints.append(dict(step=step, tokens=tokens, artifacts=[dict(path=f"{prefix}/{name}/model.safetensors", sha256=digest(candidate / "model.safetensors"),
                                                                          bytes=(candidate / "model.safetensors").stat().st_size)]))
        item = dict(step=step, tokens=tokens, metrics={
            **metrics,
            "live_eager_compute_heldout_loss": compute_loss,
            "live_eager_heldout_loss": live_loss,
            "publishable_checkpoint_heldout_loss": publishable,
            "compute_precision_loss_delta": compute_loss - live_loss,
            "artifact_loss_delta": publishable - live_loss,
        })
        evaluations.append(item)
        # The standard evaluation event carries the exact intrinsic metrics to
        # telemetry and durable progress. Candidate checkpoints are still not
        # announced as resume points, so an interrupted run restarts at step 0.
        emit("event", event=dict(kind="evaluation", step=step, tokens=tokens, evaluation=item,
                                 message=f"step {step} publishable held-out loss {publishable:.4f} (live {live_loss:.4f}, compute {compute_loss:.4f})"))

    evaluate_every = parameters.get("evaluate_every", 0)

    class Progress(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            emit("event", event=dict(kind="progress", step=state.global_step, tokens=sum(consumption.values()), gradient_norm=execution["gradient_norm"],
                                     message=f"Transformers step {state.global_step}/{parameters['steps']}"))
            if evaluation and evaluation_due(state.global_step, parameters["steps"], evaluate_every):
                record_evaluation(state.global_step)

    directory.mkdir(parents=True, exist_ok=True)
    trainer = WaldoTrainer(model=model, args=args, train_dataset=Dataset(), data_collator=collate, callbacks=[Progress()])
    trainer.model_accepts_loss_kwargs = True
    result = trainer.train()
    if trainer.state.global_step != parameters["steps"]:
        raise ValueError("training stream exhausted before planned optimizer steps")
    if not math.isfinite(result.training_loss):
        raise NumericalIntegrityError("Transformers reported a non-finite training loss")
    if execution["gradient_norms"] != parameters["steps"]:
        raise NumericalIntegrityError("Trainer did not expose a gradient norm for every optimizer step")
    accumulation = parameters.get("gradient_accumulation_steps", 1)
    if execution["micro_batches"] != parameters["steps"] * accumulation or execution["sequences"] != parameters["steps"] * parameters["batch_size"]:
        raise ValueError("Trainer micro-batches or sequences differ from the resolved global batch and accumulation")
    if bool(getattr(model, "is_gradient_checkpointing", False)) != bool(parameters.get("activation_checkpointing", False)):
        raise ValueError("model gradient checkpointing differs from the resolved plan")
    validate_model_finite(model)
    selection = None
    if evaluation:
        if not evaluations or evaluations[-1]["step"] != trainer.state.global_step:
            raise ArtifactIntegrityError("the terminal step was not evaluated")
        selected, checkpoint = select_candidate(evaluations, checkpoints)
        # Reload the selected candidate, write the terminal artifact from it,
        # then reload that file and require it to reproduce the selection.
        published = construct(dtype)
        load_model(published, str(directory / checkpoint["artifacts"][0]["path"].removeprefix(prefix + "/")), strict=True)
        validate_model_finite(published)
        save_weights(published, directory / "model.safetensors")
        del published
        candidate_loss = selected["metrics"]["heldout_loss"]
        artifact_loss = heldout_metrics(reload_heldout_nll(directory / "model.safetensors"), targets, begin["evaluation_set"])["heldout_loss"]
        check_agreement(artifact_loss, candidate_loss, "saved artifact", f"it does not reproduce selected step {selected['step']}")
        selected["metrics"].update(artifact_heldout_loss=artifact_loss, artifact_heldout_perplexity=math.exp(min(artifact_loss, 80.0)),
                                   serialization_loss_delta=artifact_loss - candidate_loss)
        selection = dict(step=selected["step"], tokens=selected["tokens"], metric="heldout_loss", value=candidate_loss)
        emit("event", event=dict(kind="log", step=selected["step"], tokens=selected["tokens"],
                                 message=f"selected checkpoint step {selected['step']} and verified the model artifact at held-out loss {artifact_loss:.4f}"))
    else:
        save_weights(model, directory / "model.safetensors")
    write_json(directory / "config.json", config.to_dict())
    write_json(directory / "training_args.json", args.to_dict())
    tokenizer_files = ["tokenizer.json"]
    if tokenizer.get("huggingface"):
        with pinned_tokenizer(transformers, tokenizer, os.environ["WALDO_HF_TOKENIZER_DIR"]) as (_, snapshot):
            tokenizer_files = sorted(tokenizer["huggingface"]["files"])
            for name in tokenizer_files:
                (directory / name).write_bytes((snapshot / name).read_bytes())
        write_json(directory / "waldo_tokenizer.json", tokenizer)
        tokenizer_files.append("waldo_tokenizer.json")
    else:
        write_json(directory / "tokenizer.json", tokenizer)
    executed = dict(optimizer_steps=trainer.state.global_step, micro_batches=execution["micro_batches"], sequences=execution["sequences"],
                    per_device_train_batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
                    global_batch_size=args.per_device_train_batch_size * args.gradient_accumulation_steps,
                    parameter_dtype=dtype, compute_dtype=sorted(execution["compute_dtypes"]), fp16=args.fp16, bf16=args.bf16,
                    gradient_checkpointing=bool(getattr(model, "is_gradient_checkpointing", False)),
                    finite_gradient_norm_checks=execution["gradient_norms"])
    write_json(directory / "runtime.json", dict(runtime_facts(device), package=spec["package"], hardware=device_facts, execution=executed, **parameter_evidence(model, config)))
    artifacts = [dict(path=f"{prefix}/{name}", sha256=digest(directory / name), bytes=(directory / name).stat().st_size)
                 for name in ["model.safetensors", "config.json", "training_args.json", "runtime.json"] + tokenizer_files]
    emit("complete", observation=dict(simulated=False, steps=trainer.state.global_step, consumed_tokens=sum(consumption.values()), final_loss=result.training_loss,
                                     checkpoints=checkpoints, evaluations=evaluations, selected_checkpoint=selection, artifacts=artifacts,
                                     consumption=[dict(corpus=key, token_targets=value) for key, value in sorted(consumption.items())]))


if __name__ == "__main__":
    try:
        if sys.argv[1] == "--tokenizer":
            with contextlib.redirect_stdout(sys.stderr):
                tokenizer_service()
        elif sys.argv[1] == "--resolve-trainer":
            with contextlib.redirect_stdout(sys.stderr):
                resolution = resolve_trainer()
            print(json.dumps(resolution, allow_nan=False))
        elif sys.argv[1] == "--probe":
            spec = json.loads(sys.argv[3])
            package = verify_package(sys.argv[2], spec["package"])
            configuration(package, spec)
            # Import the actual training classes, not just the package shell.
            package.Trainer
            package.TrainingArguments
            print(json.dumps({"runtime": json.dumps(runtime_facts(), sort_keys=True)}))
        else:
            with contextlib.redirect_stdout(sys.stderr):
                run()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        if sys.argv[1] not in ("--probe", "--resolve-trainer"):
            # Integrity failures are terminal; WALDO must not retry into them.
            error_class = "artifact-integrity" if isinstance(error, ArtifactIntegrityError) else "numerical-integrity" if isinstance(error, NumericalIntegrityError) else ""
            emit("error", error=str(error), error_class=error_class)
        sys.exit(1)
