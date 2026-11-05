# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-process Qwen3 from scratch; WALDO owns every input token."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import traceback

import numpy as np
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainerCallback, TrainingArguments
from transformers import enable_full_determinism, set_seed
from transformers.trainer_callback import PrinterCallback
from . import schemas
from .tokenizer_cli import ASSETS
from .wire import read, frame as wire_frame

LOCAL_STREAM = None

def frame(kind, **fields):
    if int(os.environ.get("RANK", "0")) == 0:
        wire_frame(kind, **fields)

def protocol_read():
    return LOCAL_STREAM.read() if LOCAL_STREAM else read()


def devices():
    if torch.cuda.is_available():
        return [dict(name=torch.cuda.get_device_name(i), kind="cuda",
                     memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                for i in range(torch.cuda.device_count())]
    return [dict(name="CPU", kind="cpu", memory_bytes=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))]


def resolve(request):
    config = schemas.architecture(request["architecture"])
    Qwen3Config(**config)
    supplied = request["parameters"]
    schemas.validate(supplied, schemas.PARAMETERS)
    parameters = schemas.PARAMETER_DEFAULTS | supplied
    world = request["world"]["world_size"]
    global_batch = request["data"]["global_batch_sequences"]
    if world < 1 or global_batch % world:
        raise ValueError("global batch must be divisible by world size")
    cuda = any(d["kind"] == "cuda" for d in request["world"].get("devices", devices()))
    if parameters["bf16"] and (not cuda or not torch.cuda.is_bf16_supported()):
        raise ValueError("requested BF16 requires a supported CUDA device")
    if (parameters["fp16"] or parameters["tf32"]) and not cuda:
        raise ValueError("requested CUDA precision is unavailable")
    if parameters["bf16"] and parameters["fp16"]:
        raise ValueError("BF16 and FP16 are mutually exclusive")
    local = global_batch // world
    default_micro = min(local, 4 if cuda else 2)
    if "gradient_accumulation_steps" in supplied and "per_device_train_batch_size" not in supplied:
        default_micro = local // supplied["gradient_accumulation_steps"]
        if default_micro < 1:
            raise ValueError("accumulation exceeds local batch")
    micro = supplied.get("per_device_train_batch_size", default_micro)
    while "per_device_train_batch_size" not in supplied and "gradient_accumulation_steps" not in supplied and local % micro:
        micro -= 1
    accumulation = supplied.get("gradient_accumulation_steps", local // micro)
    if micro * accumulation * world != global_batch:
        raise ValueError("micro-batch x accumulation x world size must equal global batch")
    parameters.update(per_device_train_batch_size=micro, gradient_accumulation_steps=accumulation)
    parameters["attn_implementation"] = supplied.get("attn_implementation", "sdpa" if cuda else "eager")
    return dict(parameters=parameters, batch=dict(micro_batch_sequences=micro,
                accumulation_steps=accumulation, world_size=world))


def declaration(path, root, published=False):
    suffix = path.suffix
    role = "weights" if suffix == ".safetensors" else "configuration"
    if path.name.startswith("tokenizer"):
        role = "tokenizer"
    fmt = "safetensors" if suffix == ".safetensors" else "json" if suffix == ".json" else "pickle"
    if not published and fmt == "pickle":
        role = "resume"
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return dict(path=path.relative_to(root).as_posix(), sha256=digest,
                bytes=path.stat().st_size, role=role, format=fmt)


class TokenStream(torch.utils.data.IterableDataset):
    def __init__(self, begin):
        self.begin = begin
        self.evaluations = []
        self.pending = None
        self.ended = False
        self.distributed = begin["world"]["world_size"] > 1
        self.rank = int(os.environ.get("RANK", "0"))
        self.ordinal = (begin["resume"]["step"] if begin.get("resume") else 0) * begin["data"]["global_batch_sequences"]
        while True:
            value = self.next_frame()
            if value["kind"] != "evaluation_record":
                self.pending = value
                break
            self.evaluations.append(value["record"])

    def next_frame(self):
        value = protocol_read()
        if value.get("protocol") != 1:
            raise ValueError("unsupported protocol")
        return value

    def __iter__(self):
        if self.distributed:
            yield from self.distributed_records()
            return
        while not self.ended:
            value = self.pending or self.next_frame()
            self.pending = None
            if value["kind"] == "end":
                self.ended = True
                if self.ordinal != self.begin["data"]["steps"] * self.begin["data"]["global_batch_sequences"]:
                    raise ValueError("training stream ended before exact budget")
                return
            if value["kind"] != "record":
                raise ValueError("expected token record")
            record = value["record"]
            if record["ordinal"] != self.ordinal:
                raise ValueError("noncontiguous sequence ordinal")
            self.ordinal += 1
            if len(record["loss_mask"]) != self.begin["data"]["sequence_length"]:
                raise ValueError("wrong packed sequence length")
            yield encoded(record)

    def distributed_records(self):
        batch = []
        world = self.begin["world"]["world_size"]
        micro = self.begin["batch"]["micro_batch_sequences"]
        while not self.ended:
            value = self.pending or self.next_frame()
            self.pending = None
            if value["kind"] == "end":
                if batch or self.ordinal != self.begin["data"]["steps"] * self.begin["data"]["global_batch_sequences"]:
                    raise ValueError("incomplete distributed stream")
                self.ended = True
                return
            if value["kind"] == "micro_batch_end":
                if value["ordinal"] != self.ordinal + micro * world or len(batch) != micro:
                    raise ValueError("invalid global micro-batch boundary")
                self.ordinal = value["ordinal"]
                # Every local reader reaches the boundary before any rank enters
                # model collectives; otherwise the leader can strand a sibling.
                yield from batch
                batch = []
                continue
            if value["kind"] != "record":
                raise ValueError("expected distributed record or boundary")
            record = value["record"]
            if not self.ordinal <= record["ordinal"] < self.ordinal + micro * world:
                raise ValueError("record outside micro-batch")
            if record["ordinal"] % world == self.rank:
                if record["ordinal"] != self.ordinal + len(batch)*world + self.rank:
                    raise ValueError("invalid rank sequence ordinal")
                batch.append(encoded(record))

    def finish(self):
        if self.distributed:
            for _ in self:
                raise ValueError("extra records after budget")
            return
        if self.ordinal != self.begin["data"]["steps"] * self.begin["data"]["global_batch_sequences"]:
            raise ValueError("training did not consume the exact sequence budget")
        if not self.ended:
            value = self.pending or self.next_frame()
            self.pending = None
            if value["kind"] != "end":
                raise ValueError("extra records after budget")
            self.ended = True
        if sys.stdin.read(1):
            raise ValueError("frames after end")


def encoded(record):
    tokens, mask = record["tokens"], record["loss_mask"]
    if len(tokens) != len(mask) + 1 or not any(mask):
        raise ValueError("invalid tokens or loss mask")
    return dict(input_ids=tokens, labels=[-100] + [t if m else -100 for t, m in zip(tokens[1:], mask)],
                consumption=record.get("consumption", {}))


def collate(records):
    counts = {}
    for record in records:
        for corpus, count in record["consumption"].items():
            counts[corpus] = counts.get(corpus, 0) + count
    return dict(input_ids=torch.tensor([r["input_ids"] for r in records]),
                labels=torch.tensor([r["labels"] for r in records]), consumption=counts)


class WaldoTrainer(Trainer):
    def __init__(self, *args, history, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = history
        self.step_loss = 0.0
        self.model_accepts_loss_kwargs = True
        self.remove_callback(PrinterCallback)

    def get_train_dataloader(self):
        # The single-process input is already partitioned and ordered by WALDO.
        # A separate generator prevents DataLoader construction from consuming
        # the model RNG again on resume. No dispatch, padding or lookahead.
        return torch.utils.data.DataLoader(self.train_dataset,
            batch_size=self.args.per_device_train_batch_size, collate_fn=collate,
            num_workers=0, generator=torch.Generator().manual_seed(self.args.seed))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        counts = inputs.pop("consumption")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Trainer supplies the number of supervised labels across ALL accumulated
        # micro-batches. labels[:, 0] is always -100, matching the causal shift.
        denominator = num_items_in_batch if num_items_in_batch is not None else labels[:, 1:].ne(-100).sum()
        distributed = torch.distributed.is_initialized()
        if distributed:
            denominator = torch.as_tensor(denominator, device=outputs.logits.device).clone()
            torch.distributed.all_reduce(denominator)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits[:, :-1].float().reshape(-1, outputs.logits.shape[-1]),
            labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum",
            label_smoothing=self.args.label_smoothing_factor) / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        if distributed:
            observed_loss = loss.detach().clone()
            torch.distributed.all_reduce(observed_loss)
            self.step_loss += observed_loss.item()
            loss = loss * torch.distributed.get_world_size()
            all_counts = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(all_counts, counts)
            counts = {}
            for rank_counts in all_counts:
                for corpus, count in rank_counts.items():
                    counts[corpus] = counts.get(corpus, 0) + count
        else:
            self.step_loss += loss.detach().item()
        for corpus, count in counts.items():
            self.history["consumption"][corpus] = self.history["consumption"].get(corpus, 0) + count
        return (loss, outputs) if return_outputs else loss

    def _save_rng_state(self, output_dir):
        numpy_state = np.random.get_state()
        state = dict(python=random.getstate(), numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                     cpu=torch.get_rng_state())
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        if torch.distributed.is_initialized():
            states = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(states, state)
            if torch.distributed.get_rank() == 0:
                torch.save(states, Path(output_dir) / "rng_state.pth")
        else:
            torch.save(state, Path(output_dir) / "rng_state.pth")

    def _load_rng_state(self, checkpoint):
        state = torch.load(Path(checkpoint) / "rng_state.pth", map_location="cpu", weights_only=True)
        if torch.distributed.is_initialized():
            state = state[torch.distributed.get_rank()]
        random.setstate(state["python"])
        ns = state["numpy"]
        np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
        torch.set_rng_state(state["cpu"])
        if "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def evaluate(model, records, step, tokens):
    was_training = model.training
    model.eval()
    nll, count, size = 0.0, 0, 0
    for record in records:
        data = encoded(record)
        ids = torch.tensor([data["input_ids"]], device=model.device)
        labels = torch.tensor(data["labels"][1:], device=model.device)
        logits = model(input_ids=ids).logits[0, :-1].float()
        nll += torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100, reduction="sum").item()
        count += sum(record["loss_mask"])
        size += record["text_bytes"]
    model.train(was_training)
    if count <= 0 or size <= 0 or not math.isfinite(nll):
        raise FloatingPointError("invalid held-out NLL/counts")
    return dict(step=step, tokens=tokens, nll_sum=nll, target_tokens=count,
                text_bytes=size, loss=nll/count, bits_per_byte=nll/(size*math.log(2)))


class Events(TrainerCallback):
    def __init__(self, begin, dataset, trainer, best_path):
        self.begin, self.dataset, self.trainer, self.best_path = begin, dataset, trainer, best_path
        self.pending_evaluation = None

    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("non-finite gradients")

    def on_step_end(self, args, state, control, model, **kwargs):
        h = self.trainer.history
        step, tokens = state.global_step, sum(h["consumption"].values())
        frame("event", event=dict(kind="progress", step=step, tokens=tokens, loss=self.trainer.step_loss))
        self.trainer.step_loss = 0.0
        data = self.begin["data"]
        due = step == 1 or step == data["steps"] or (data.get("evaluate_every", 0) > 0 and step % data["evaluate_every"] == 0)
        if self.dataset.evaluations and due:
            metric = evaluate(model, self.dataset.evaluations, step, tokens)
            h["evaluations"].append(metric)
            self.pending_evaluation = metric
            if h["selected"] is None or metric["loss"] < h["selected"]["value"]:
                h["selected"] = dict(step=step, tokens=tokens, metric="heldout_loss", value=metric["loss"])
                if args.process_index == 0:
                    model.save_pretrained(self.best_path, safe_serialization=True)
            control.should_save = True
        if step == data["steps"]:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        root = Path(args.output_dir)
        checkpoint = root / f"checkpoint-{state.global_step}"
        h = self.trainer.history
        if h["selected"] is not None:
            shutil.copyfile(self.best_path / "model.safetensors", checkpoint / "best.safetensors")
        (checkpoint / "waldo-history.json").write_text(json.dumps(h, allow_nan=False))
        files = [declaration(p, Path(self.begin["paths"]["checkpoints"])) for p in sorted(checkpoint.iterdir()) if p.is_file()]
        tokens = sum(h["consumption"].values())
        frame("event", event=dict(kind="checkpoint", step=state.global_step, tokens=tokens,
              checkpoint=dict(step=state.global_step, tokens=tokens, artifacts=files)))
        if self.pending_evaluation:
            frame("event", event=dict(kind="evaluation", step=state.global_step, tokens=tokens,
                                      evaluation=self.pending_evaluation))
            self.pending_evaluation = None


def train(begin):
    world = begin["world"]["world_size"]
    if "RANK" not in os.environ and world == 1 and torch.cuda.device_count() > 1:
        raise ValueError("single-process requires one visible CUDA device")
    if "RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group("nccl")
        if torch.distributed.get_world_size() != world:
            raise ValueError("launcher world differs from begin")
    params = begin["parameters"]
    schemas.validate(params, schemas.PARAMETERS)
    data = begin["data"]
    if params["per_device_train_batch_size"] * params["gradient_accumulation_steps"] * world != data["global_batch_sequences"]:
        raise ValueError("resolved batch does not match WALDO global batch")
    torch.set_num_threads(1)
    if begin["determinism"] == "required":
        enable_full_determinism(params["init_seed"])
    else:
        set_seed(params["init_seed"])
    config = Qwen3Config(**schemas.architecture(begin["architecture"]))
    config._attn_implementation = params["attn_implementation"]
    if config.vocab_size != begin["tokenizer"]["vocabulary_size"]:
        raise ValueError("architecture and tokenizer vocabulary differ")
    model = Qwen3ForCausalLM(config)
    if begin.get("initialization"):
        from safetensors.torch import load_model
        load_model(model, begin["initialization"]["path"], strict=True)
    paths = {k: Path(v) for k, v in begin["paths"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    # Each attempt gets its own output directory, so interrupted pending files
    # and previously declared checkpoint bytes can never be overwritten.
    import tempfile
    output = Path(tempfile.mkdtemp(prefix="attempt-", dir=paths["checkpoints"]))
    best_path = paths["scratch"] / "best"
    history = dict(consumption={}, evaluations=[], selected=None)
    resume_dir = None
    if begin.get("resume"):
        resume_dir = Path(tempfile.mkdtemp(prefix="resume-", dir=paths["scratch"]))
        for name, source in begin["resume"]["paths"].items():
            destination = resume_dir / Path(name).name
            if destination.exists():
                raise ValueError("duplicate resume basename")
            shutil.copyfile(source, destination)
        history = json.loads((resume_dir / "waldo-history.json").read_text())
        if history["consumption"] != begin["resume"]["checkpoint_manifest"]["consumption"]:
            raise ValueError("checkpoint consumption mismatch")
        if history["selected"] and int(os.environ.get("RANK", "0")) == 0:
            best_path.mkdir(exist_ok=True)
            shutil.copyfile(resume_dir / "best.safetensors", best_path / "model.safetensors")
    dataset = TokenStream(begin)
    kwargs = {k: v for k, v in params.items() if k not in ("init_seed", "attn_implementation")}
    # WALDO owns all step/data/cadence values. Evaluation is performed by the
    # callback to retain exact NLL sums and byte counts instead of Trainer means.
    arguments = TrainingArguments(output_dir=str(output), max_steps=data["steps"],
        save_strategy="steps" if data.get("checkpoint_every", 0) else "no",
        save_steps=max(1, data.get("checkpoint_every", 0)), eval_steps=max(1, data.get("evaluate_every", 0)),
        eval_strategy="no", seed=data["seed"], full_determinism=begin["determinism"] == "required",
        ignore_data_skip=True, remove_unused_columns=False, dataloader_num_workers=0,
        dataloader_pin_memory=False, disable_tqdm=True, save_safetensors=True,
        logging_strategy="no", use_cpu=not torch.cuda.is_available(), **kwargs)
    trainer = WaldoTrainer(model=model, args=arguments, train_dataset=dataset, data_collator=collate, history=history)
    events = Events(begin, dataset, trainer, best_path)
    trainer.add_callback(events)
    if begin.get("resume") and begin["resume"]["step"] == data["steps"]:
        trainer._load_from_checkpoint(str(resume_dir))
    else:
        # Third-party print output must never contaminate protocol stdout.
        trainer.train(resume_from_checkpoint=str(resume_dir) if resume_dir else None)
    dataset.finish()
    if world > 1 and torch.distributed.get_rank() != 0:
        torch.distributed.barrier()
        return
    tokens = sum(history["consumption"].values())
    selected = history["selected"] or dict(step=data["steps"], tokens=tokens, metric="terminal", value=0.0)
    if history["selected"]:
        from safetensors.torch import load_model
        load_model(model, best_path / "model.safetensors", strict=True)
    model.save_pretrained(paths["artifacts"], safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(ASSETS / name, paths["artifacts"] / name)
    published = Qwen3ForCausalLM.from_pretrained(paths["artifacts"], local_files_only=True,
                                              attn_implementation=params["attn_implementation"]).to(model.device)
    result = dict(steps=data["steps"], consumed_tokens=tokens,
                  artifacts=[declaration(p, paths["artifacts"], True) for p in sorted(paths["artifacts"].iterdir())],
                  artifact_profile="hf-causal-lm", consumption=[dict(corpus=k, token_targets=v) for k,v in sorted(history["consumption"].items())],
                  evaluations=history["evaluations"], selected_checkpoint=selected,
                  environment=dict(torch=torch.__version__, transformers=transformers.__version__, device=str(model.device)))
    if dataset.evaluations:
        result["artifact_evaluation"] = evaluate(published, dataset.evaluations, selected["step"], selected["tokens"])
    frame("complete", complete=result)
    if world > 1:
        torch.distributed.barrier()


def main():
    global LOCAL_STREAM
    command = sys.argv[1]
    if command == "train" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from .distributed import LocalStream
        LOCAL_STREAM = LocalStream()
    request = protocol_read()
    if request.get("protocol") != 1 or request.get("kind") != ("begin" if command == "train" else command):
        raise ValueError("unexpected protocol command")
    if command == "probe":
        import subprocess
        driver = "unavailable"
        if torch.cuda.is_available():
            driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        frame("probe", capabilities=dict(objectives=["causal-language-modeling", "assistant-response-modeling"],
              launch_profiles=["single-process", "distributed"], resume=True),
              frameworks=dict(torch=torch.__version__, transformers=transformers.__version__), devices=devices(),
              driver=driver, cuda=torch.version.cuda or "unavailable",
              schemas=dict(architecture=schemas.ARCHITECTURE, parameters=schemas.PARAMETERS))
    elif command == "resolve":
        frame("resolve", **resolve(request))
    elif command == "train":
        train(request["begin"])
    else:
        raise ValueError("unknown command")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        frame("error", error=str(error), error_class="numerical-integrity" if isinstance(error, FloatingPointError) else "configuration")
        sys.exit(1)
    finally:
        if LOCAL_STREAM:
            LOCAL_STREAM.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
