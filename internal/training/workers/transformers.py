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

REVISION = "builtin-transformers-worker-schema-1-r3"
PROTOCOL_OUTPUT = sys.stdout


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
    torch.tensor([1.0], device=device).sum().item()
    return device


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
    if begin["tokenizer"].get("huggingface"):
        for name in ("pad", "bos", "eos"):
            expected = begin["tokenizer"][name + "_id"]
            if getattr(config, name + "_token_id", None) != (None if expected == -1 else expected):
                raise ValueError(f"model config {name} ID differs from tokenizer framing")
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
    evaluation_totals = [0.0, 0]
    class WaldoTrainer(transformers.Trainer):
        def _get_num_items_in_batch(self, batch_samples, device):
            # Our labels are already shifted, unlike standard causal-LM labels.
            return sum(batch["labels"].ne(-100).sum() for batch in batch_samples).to(device) if batch_samples else None

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
                    raise ValueError("Mixtral did not return a finite router auxiliary loss")
                # Scale the per-microbatch mean consistently with WALDO's
                # target-weighted gradient accumulation. Evaluation metrics
                # below remain pure held-out cross entropy, not router loss.
                weight = targets / (num_items_in_batch if num_items_in_batch is not None else targets)
                loss = loss + config.router_aux_loss_coef * auxiliary * weight
            if model.training:
                for item in counts:
                    for corpus, count in item.items():
                        consumption[corpus] = consumption.get(corpus, 0) + count
            else:
                evaluation_totals[0] += total.detach().item()
                evaluation_totals[1] += targets.item()
            return (loss, outputs) if return_outputs else loss

    class Progress(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            emit("event", event=dict(kind="progress", step=state.global_step, tokens=sum(consumption.values()), message=f"Transformers step {state.global_step}/{parameters['steps']}"))

    arguments = dict(trainer_spec["arguments"])
    arguments.update(output_dir=str(directory / "trainer"), max_steps=parameters["steps"], seed=parameters["seed"], data_seed=parameters["seed"],
                     use_cpu=device == "cpu", report_to="none", push_to_hub=False, save_strategy="no", eval_strategy="no", disable_tqdm=True,
                     dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=False, prediction_loss_only=True)
    args = transformers.TrainingArguments(**arguments)
    if args.device.type != device or args.world_size != 1 or (device == "cuda" and args.n_gpu != 1):
        raise ValueError("Trainer device/topology differs from WALDO runtime resolution")
    if args.per_device_train_batch_size * args.gradient_accumulation_steps != parameters["batch_size"]:
        raise ValueError("Trainer effective batch differs from WALDO's resolved plan")
    if args.learning_rate != parameters["learning_rate"]:
        raise ValueError("Trainer learning rate differs from WALDO's resolved plan")
    trainer = WaldoTrainer(model=model, args=args, train_dataset=Dataset(), data_collator=collate, callbacks=[Progress()])
    trainer.model_accepts_loss_kwargs = True
    result = trainer.train()
    if trainer.state.global_step != parameters["steps"]:
        raise ValueError("training stream exhausted before planned optimizer steps")
    if not math.isfinite(result.training_loss):
        raise ValueError("Transformers reported a non-finite training loss")
    evaluations = []
    evaluation = list(pack(evaluation_records, continuous=False))
    if evaluation:
        trainer.evaluate(eval_dataset=evaluation)
        loss = evaluation_totals[0] / evaluation_totals[1]
        if not math.isfinite(loss):
            raise ValueError("Transformers reported a non-finite held-out loss")
        evaluations.append(dict(step=trainer.state.global_step, tokens=sum(consumption.values()), metrics={"heldout_loss": loss}))
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "model.safetensors.tmp"
    save_model(model, str(temporary), metadata={"format": "pt", "waldo_backend": "huggingface-transformers", "waldo_architecture_sha256": begin["architecture_sha256"]})
    os.replace(temporary, directory / "model.safetensors")
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
    write_json(directory / "runtime.json", dict(runtime_facts(device), package=spec["package"], hardware=device_facts, **parameter_evidence(model, config)))
    artifacts = [dict(path=f"{prefix}/{name}", sha256=digest(directory / name), bytes=(directory / name).stat().st_size)
                 for name in ["model.safetensors", "config.json", "training_args.json", "runtime.json"] + tokenizer_files]
    emit("complete", observation=dict(simulated=False, steps=trainer.state.global_step, consumed_tokens=sum(consumption.values()), final_loss=result.training_loss,
                                     evaluations=evaluations, artifacts=artifacts, consumption=[dict(corpus=key, token_targets=value) for key, value in sorted(consumption.items())]))


if __name__ == "__main__":
    try:
        if sys.argv[1] == "--tokenizer":
            with contextlib.redirect_stdout(sys.stderr):
                tokenizer_service()
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
        if sys.argv[1] != "--probe":
            emit("error", error=str(error))
        sys.exit(1)
