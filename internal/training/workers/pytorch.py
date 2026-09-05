# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as functional


PROTOCOL_SCHEMA = 1
WORKER_REVISION = "builtin-pytorch-worker-schema-1-r7"
TORCHTITAN_REVISION = "builtin-torchtitan-worker-schema-1-r9"
IS_PRIMARY = True


def emit(kind, **payload):
    if not IS_PRIMARY:
        return
    frame = {"kind": kind, "schema": PROTOCOL_SCHEMA}
    frame.update(payload)
    print(json.dumps(frame, separators=(",", ":")), flush=True)


def artifact(path, logical_path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return {"path": logical_path, "sha256": digest.hexdigest(), "bytes": size}


def write_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def commit_directory(temporary, destination):
    for root, _, files in os.walk(temporary):
        for name in files:
            descriptor = os.open(os.path.join(root, name), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    descriptor = os.open(os.path.dirname(destination), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


SAFE_DTYPES = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
}
SAFE_TORCH_DTYPES = {value: key for key, value in SAFE_DTYPES.items()}


def save_safetensors(path, tensors, metadata):
    names = sorted(tensors)
    header = {"__metadata__": metadata}
    payloads = []
    offset = 0
    for name in names:
        tensor = tensors[name].detach().to("cpu").contiguous()
        dtype = SAFE_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"cannot write Safetensors dtype {tensor.dtype} for {name}")
        payload = tensor.view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    temporary = path + ".tmp"
    with open(temporary, "wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for payload in payloads:
            stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_safetensors(path):
    with open(path, "rb") as stream:
        header_length_data = stream.read(8)
        if len(header_length_data) != 8:
            raise ValueError("initialization Safetensors header is truncated")
        header_length = struct.unpack("<Q", header_length_data)[0]
        if header_length == 0 or header_length > 1024 * 1024 * 1024:
            raise ValueError(f"invalid initialization Safetensors header length {header_length}")
        header = json.loads(stream.read(header_length))
        payload = bytearray(stream.read())
    tensors = {}
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        dtype = SAFE_TORCH_DTYPES.get(descriptor["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported initialization Safetensors dtype {descriptor['dtype']}")
        start, end = descriptor["data_offsets"]
        if start < 0 or end < start or end > len(payload):
            raise ValueError(f"invalid initialization offsets for {name}")
        value = torch.frombuffer(payload[start:end], dtype=dtype).clone()
        tensors[name] = value.reshape(descriptor["shape"])
    return tensors


class RMSNorm(nn.Module):
    def __init__(self, hidden, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.epsilon = epsilon

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.epsilon)
        return normalized.to(value.dtype) * self.weight


def rotate_half(value):
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


class Attention(nn.Module):
    def __init__(self, hidden, heads, kv_heads):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        kv_width = self.head_dim * kv_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_width, bias=False)
        self.v_proj = nn.Linear(hidden, kv_width, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def rope(self, value):
        length = value.shape[2]
        positions = torch.arange(length, device=value.device, dtype=torch.float32)
        frequencies = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=value.device, dtype=torch.float32) / self.head_dim))
        angles = torch.outer(positions, frequencies)
        angles = torch.cat((angles, angles), dim=-1).to(value.dtype)[None, None, :, :]
        return value * angles.cos() + rotate_half(value) * angles.sin()

    def forward(self, value):
        batch, length, _ = value.shape
        query = self.q_proj(value).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(value).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        val = self.v_proj(value).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        query = self.rope(query)
        key = self.rope(key)
        if self.heads != self.kv_heads:
            repeats = self.heads // self.kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            val = val.repeat_interleave(repeats, dim=1)
        attended = functional.scaled_dot_product_attention(query, key, val, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, value):
        return self.down(functional.silu(self.gate(value)) * self.up(value))


class DecoderBlock(nn.Module):
    def __init__(self, hidden, intermediate, heads, kv_heads, dropout):
        super().__init__()
        self.attention_norm = RMSNorm(hidden)
        self.attention = Attention(hidden, heads, kv_heads)
        self.ffn_norm = RMSNorm(hidden)
        self.feed_forward = FeedForward(hidden, intermediate)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, value):
        value = value + self.residual_dropout(self.attention(self.attention_norm(value)))
        return value + self.residual_dropout(self.feed_forward(self.ffn_norm(value)))


class DecoderLM(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        vocabulary = architecture["vocabulary_size"]
        hidden = architecture["hidden_size"]
        self.tie_embeddings = architecture["tie_embeddings"]
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.layers = nn.ModuleList([
            DecoderBlock(
                hidden,
                architecture["intermediate_size"],
                architecture["attention_heads"],
                architecture["key_value_heads"],
                architecture.get("dropout", 0.0),
            )
            for _ in range(architecture["layers"])
        ])
        self.norm = RMSNorm(hidden)
        if not self.tie_embeddings:
            self.output = nn.Linear(hidden, vocabulary, bias=False)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, tokens):
        value = self.embedding(tokens)
        for layer in self.layers:
            value = layer(value)
        value = self.norm(value)
        if self.tie_embeddings:
            return functional.linear(value, self.embedding.weight)
        return self.output(value)


class FramingTokenizer:
    def __init__(self, specification):
        self.name = specification["name"]
        self.revision = specification["revision"]
        self.pad_id = int(specification["pad_id"])
        self.bos_id = int(specification["bos_id"])
        self.eos_id = int(specification["eos_id"])

    def encode_record(self, record):
        if "tokens" in record:
            return [int(token) for token in record["tokens"]] + [self.eos_id]
        if self.name == "byte":
            return [byte + 3 for byte in record["text"].encode("utf-8")] + [self.eos_id]
        raise ValueError(f"record is missing pre-tokenized IDs for {self.name}")


class Trainer:
    def __init__(self, begin, artifact_directory, artifact_prefix, device_name):
        self.begin = begin
        self.architecture = begin["architecture"]
        self.parameters = begin["parameters"]
        self.artifact_directory = artifact_directory
        self.artifact_prefix = artifact_prefix.replace(os.sep, "/").strip("/")
        self.distributed = device_name == "torchtitan"
        self.rank = torch.distributed.get_rank() if self.distributed else 0
        self.world_size = torch.distributed.get_world_size() if self.distributed else 1
        if self.distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
            from torchtitan.distributed import ParallelDims

            self.parallel_dims = ParallelDims(
                dp_replicate=1,
                dp_shard=self.world_size,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                etp=1,
                world_size=self.world_size,
            )
            self.parallel_dims.build_mesh()
        else:
            self.device = torch.device(device_name)
        self.sequence_length = self.parameters["sequence_length"]
        self.batch_size = self.parameters["batch_size"]
        self.target_steps = self.parameters["steps"]
        self.step_number = 0
        self.replay_steps = 0
        self.consumed_tokens = 0
        self.token_buffer = []
        self.loss_buffer = []
        self.corpus_buffer = []
        self.batch = []
        self.consumed_by_corpus = {}
        self.checkpoints = []
        self.evaluations = []
        self.evaluation_sequences = []
        self.evaluation_record_count = 0
        self.evaluation_token_targets = 0
        self.final_loss = None
        self.started = time.perf_counter()

        tokenizer = begin["tokenizer"]
        architecture_tokenizer = self.architecture["tokenizer"]
        if tokenizer["name"] != architecture_tokenizer["name"] or tokenizer["revision"] != architecture_tokenizer["revision"] or tokenizer["vocabulary_size"] != self.architecture["vocabulary_size"]:
            raise ValueError("PyTorch worker tokenizer framing does not match the architecture")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("PyTorch worker selected CUDA but torch.cuda.is_available() is false")
        self.tokenizer = FramingTokenizer(tokenizer)
        torch.manual_seed(self.parameters["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.parameters["seed"])
        self.model = DecoderLM(self.architecture)
        self.model.initialize()
        self.initialization = begin.get("initialization")
        if self.initialization is not None:
            if self.rank == 0 and not self.initialization.get("path"):
                raise ValueError("initialization weights frame is missing a path")
            if not self.distributed or self.rank == 0:
                missing, unexpected = self.model.load_state_dict(load_safetensors(self.initialization["path"]), strict=False)
                if missing or unexpected:
                    raise ValueError(f"initialization weights do not match architecture: missing={missing}, unexpected={unexpected}")
        self.resume = begin.get("resume")
        self.resume_paths = None
        if self.resume is not None:
            self.resume_paths = {os.path.basename(path): path for path in self.resume["paths"]}
            required = {"model.safetensors", "runtime.pt", "state.json"}
            if set(self.resume_paths) != required:
                raise ValueError(f"PyTorch checkpoint requires {sorted(required)}, found {sorted(self.resume_paths)}")
            missing, unexpected = self.model.load_state_dict(load_safetensors(self.resume_paths["model.safetensors"]), strict=False)
            if missing or unexpected:
                raise ValueError(f"resume weights do not match architecture: missing={missing}, unexpected={unexpected}")
        dtype_name = self.architecture["parameter_dtype"]
        self.parameter_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
        if self.device.type == "cpu" and self.parameter_dtype == torch.float16:
            raise ValueError("float16 training is not supported by the PyTorch CPU adapter; use bfloat16 or float32")
        # Keep master weights and AdamW state in FP32. Reduced precision is a
        # compute and portable-artifact format, not optimizer state.
        self.model.to(device=self.device, dtype=torch.float32)
        if self.distributed:
            if self.initialization is not None:
                for parameter in self.model.parameters():
                    torch.distributed.broadcast(parameter.data, src=0)
            from torch.distributed._composable.fsdp import fully_shard

            fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
            for layer in self.model.layers:
                fully_shard(layer, mesh=fsdp_mesh)
            fully_shard(self.model, mesh=fsdp_mesh)
        optimizer_parameters = self.parameters["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.parameters["learning_rate"],
            betas=(optimizer_parameters["beta1"], optimizer_parameters["beta2"]),
            eps=optimizer_parameters["epsilon"],
            weight_decay=optimizer_parameters["weight_decay"],
        )
        if self.resume is not None:
            self.restore_checkpoint()

    def forward_logits(self, model, tokens, mixed_precision=True):
        enabled = mixed_precision and self.parameter_dtype != torch.float32
        with torch.autocast(device_type=self.device.type, dtype=self.parameter_dtype, enabled=enabled):
            return model(tokens)

    def logical(self, name):
        return "/".join(part for part in (self.artifact_prefix, name) if part)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def learning_rate(self, step):
        schedule = self.parameters["schedule"]
        base = self.parameters["learning_rate"]
        warmup = schedule["warmup_steps"]
        if warmup > 0 and step <= warmup:
            return base * step / warmup
        decay_steps = max(1, self.target_steps - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
        ratio = schedule["minimum_rate_ratio"] + (1.0 - schedule["minimum_rate_ratio"]) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return base * ratio

    def add_record(self, record):
        if self.step_number >= self.target_steps:
            return
        encoded = self.tokenizer.encode_record(record)
        loss_mask = record.get("loss_mask", [True] * len(encoded))
        if len(loss_mask) != len(encoded):
            raise ValueError("record loss_mask does not match framed token count")
        self.token_buffer.extend(encoded)
        self.loss_buffer.extend(loss_mask)
        self.corpus_buffer.extend([record.get("corpus", "")] * len(encoded))
        window = self.sequence_length + 1
        while len(self.token_buffer) >= window and self.step_number < self.target_steps:
            self.add_sequence(self.token_buffer[:window], self.loss_buffer[1:window], self.corpus_buffer[1:window])
            del self.token_buffer[: self.sequence_length]
            del self.loss_buffer[: self.sequence_length]
            del self.corpus_buffer[: self.sequence_length]

    def add_evaluation_record(self, record):
        self.evaluation_record_count += 1
        tokens = self.tokenizer.encode_record(record)
        loss_mask = record.get("loss_mask", [True] * len(tokens))
        if len(loss_mask) != len(tokens):
            raise ValueError("evaluation record loss_mask does not match framed token count")
        window = self.sequence_length + 1
        while len(tokens) > 1:
            piece = tokens[:window]
            target_mask = loss_mask[1:len(piece)]
            padded = piece + [self.tokenizer.pad_id] * (window - len(piece))
            mask = [float(value) for value in target_mask] + [0.0] * (self.sequence_length - len(target_mask))
            self.evaluation_sequences.append((padded, mask))
            self.evaluation_token_targets += sum(target_mask)
            del tokens[: self.sequence_length]
            del loss_mask[: self.sequence_length]

    def add_sequence(self, tokens, target_mask, target_corpora=None):
        if not any(target_mask) or self.step_number >= self.target_steps:
            return
        window = self.sequence_length + 1
        padded = tokens + [self.tokenizer.pad_id] * (window - len(tokens))
        mask = [float(value) for value in target_mask] + [0.0] * (self.sequence_length - len(target_mask))
        corpus_counts = {}
        for corpus, supervised in zip(target_corpora or [], target_mask):
            if supervised:
                corpus_counts[corpus] = corpus_counts.get(corpus, 0) + 1
        self.batch.append((padded, mask, corpus_counts))
        if len(self.batch) >= self.batch_size:
            self.train_batch()

    def train_batch(self):
        if not self.batch or self.step_number >= self.target_steps:
            self.batch = []
            return
        if self.replay_steps > 0:
            self.replay_steps -= 1
            self.batch = []
            return
        tokens = torch.tensor([item[0] for item in self.batch], dtype=torch.long, device=self.device)
        mask = torch.tensor([item[1] for item in self.batch], dtype=torch.float32, device=self.device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        next_step = self.step_number + 1
        current_learning_rate = self.learning_rate(next_step)
        for group in self.optimizer.param_groups:
            group["lr"] = current_learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.forward_logits(self.model, inputs)
        losses = functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
        loss = (losses.reshape_as(mask) * mask).sum() / mask.sum()
        loss.backward()
        self.optimizer.step()
        self.synchronize()
        loss_value = float(loss.detach().cpu().item())
        valid_tokens = int(mask.sum().detach().cpu().item())
        self.step_number = next_step
        self.consumed_tokens += valid_tokens
        for item in self.batch:
            for corpus, count in item[2].items():
                self.consumed_by_corpus[corpus] = self.consumed_by_corpus.get(corpus, 0) + count
        self.final_loss = loss_value
        self.batch = []
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        throughput = self.consumed_tokens / elapsed
        eta = int(max(0.0, (self.target_steps - self.step_number) * elapsed / self.step_number))
        report_every = max(1, self.target_steps // 100)
        if self.step_number == 1 or self.step_number == self.target_steps or self.step_number % report_every == 0:
            emit(
                "event",
                event={
                    "kind": "progress",
                    "message": f"step {self.step_number}/{self.target_steps}, loss {loss_value:.4f}, {throughput:.0f} tokens/s",
                    "step": self.step_number,
                    "tokens": self.consumed_tokens,
                    "loss": loss_value,
                    "learning_rate": current_learning_rate,
                    "tokens_per_second": throughput,
                    "eta_seconds": eta,
                },
            )
        checkpoint_every = self.parameters["checkpoint_every"]
        if checkpoint_every > 0 and self.step_number % checkpoint_every == 0:
            self.save_checkpoint()
        evaluate_every = self.parameters["evaluate_every"]
        if evaluate_every > 0 and self.step_number % evaluate_every == 0:
            self.record_evaluation(loss_value)

    def save_weights(self, path, kind, step):
        if self.distributed:
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict,
                StateDictOptions,
            )

            tensors = get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        else:
            tensors = self.model.state_dict()
        if IS_PRIMARY:
            tensors = {
                name: value.to(dtype=self.parameter_dtype) if value.is_floating_point() else value
                for name, value in tensors.items()
            }
            backend = "torchtitan" if self.distributed else "pytorch"
            revision = TORCHTITAN_REVISION if self.distributed else WORKER_REVISION
            save_safetensors(
                path,
                tensors,
                {
                    "format": "openwaldo",
                    "kind": kind,
                    "schema": "1",
                    "backend": backend,
                    "backend_revision": revision,
                    "architecture_sha256": self.begin["architecture_sha256"],
                    "run_id": self.begin["run_id"],
                    "step": str(step),
                },
            )
        if self.distributed:
            torch.distributed.barrier()

    def save_checkpoint(self):
        name = f"checkpoints/step-{self.step_number:08d}"
        path = os.path.join(self.artifact_directory, *name.split("/"))
        temporary = None
        if IS_PRIMARY:
            temporary = path + f".tmp-{os.getpid()}"
            if os.path.exists(temporary):
                shutil.rmtree(temporary)
            os.makedirs(temporary)
        if self.distributed:
            values = [temporary]
            torch.distributed.broadcast_object_list(values, src=0, device=self.device)
            temporary = values[0]
        weights_path = os.path.join(temporary, "model.safetensors")
        runtime_path = os.path.join(temporary, "runtime.pt")
        state_path = os.path.join(temporary, "state.json")
        backend_name = "torchtitan" if self.distributed else "pytorch"
        self.save_weights(weights_path, f"waldo-{backend_name}-checkpoint", self.step_number)
        random_state = {
            "cpu": torch.get_rng_state().cpu(),
            "cuda": torch.cuda.get_rng_state(self.device).cpu() if self.device.type == "cuda" else None,
        }
        if self.distributed:
            random_states = [None for _ in range(self.world_size)]
            torch.distributed.all_gather_object(random_states, random_state)
            from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, StateDictOptions

            optimizer_state = get_optimizer_state_dict(
                self.model,
                self.optimizer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        else:
            random_states = [random_state]
            optimizer_state = self.optimizer.state_dict()
        if IS_PRIMARY:
            torch.save({"optimizer": optimizer_state, "random_states": random_states}, runtime_path)
            write_json(
                state_path,
                {
                    "kind": "waldo-training-checkpoint",
                    "schema": 1,
                    "backend": backend_name,
                    "backend_revision": TORCHTITAN_REVISION if self.distributed else WORKER_REVISION,
                    "run_id": self.begin["run_id"],
                    "architecture_sha256": self.begin["architecture_sha256"],
                    "step": self.step_number,
                    "consumed_tokens": self.consumed_tokens,
                    "consumption": self.consumed_by_corpus,
                    "world_size": self.world_size,
                },
            )
            commit_directory(temporary, path)
        if self.distributed:
            torch.distributed.barrier()
        weights_path = os.path.join(path, "model.safetensors")
        runtime_path = os.path.join(path, "runtime.pt")
        state_path = os.path.join(path, "state.json")
        item = {
            "step": self.step_number,
            "tokens": self.consumed_tokens,
            "artifacts": [
                artifact(weights_path, self.logical(name + "/model.safetensors")),
                artifact(runtime_path, self.logical(name + "/runtime.pt")),
                artifact(state_path, self.logical(name + "/state.json")),
            ] if IS_PRIMARY else [],
        }
        self.checkpoints.append(item)
        emit(
            "event",
            event={
                "kind": "checkpoint",
                "message": f"checkpoint step {self.step_number} persisted",
                "step": self.step_number,
                "tokens": self.consumed_tokens,
                "checkpoint": item,
            },
        )

    def restore_checkpoint(self):
        if self.resume["step"] <= 0 or self.resume["step"] > self.target_steps:
            raise ValueError(f"resume step {self.resume['step']} must be in 1..{self.target_steps}")
        with open(self.resume_paths["state.json"], "r", encoding="utf-8") as stream:
            state = json.load(stream)
        backend_name = "torchtitan" if self.distributed else "pytorch"
        revision = TORCHTITAN_REVISION if self.distributed else WORKER_REVISION
        if (
            state.get("kind") != "waldo-training-checkpoint"
            or state.get("schema") != 1
            or state.get("backend") != backend_name
            or state.get("backend_revision") != revision
            or state.get("run_id") != self.begin["run_id"]
            or state.get("architecture_sha256") != self.begin["architecture_sha256"]
            or state.get("step") != self.resume["step"]
            or state.get("consumed_tokens") != self.resume["tokens"]
            or state.get("world_size") != self.world_size
        ):
            raise ValueError("PyTorch checkpoint state does not match the requested run, backend, and resume point")
        runtime = torch.load(self.resume_paths["runtime.pt"], map_location="cpu", weights_only=True)
        if self.distributed:
            from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict, StateDictOptions

            set_optimizer_state_dict(
                self.model,
                self.optimizer,
                optim_state_dict=runtime["optimizer"],
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        else:
            self.optimizer.load_state_dict(runtime["optimizer"])
        random_state = runtime["random_states"][self.rank]
        torch.set_rng_state(random_state["cpu"])
        if self.device.type == "cuda" and random_state["cuda"] is not None:
            torch.cuda.set_rng_state(random_state["cuda"], self.device)
        self.step_number = self.resume["step"]
        self.consumed_tokens = self.resume["tokens"]
        self.consumed_by_corpus = state.get("consumption", {})
        self.replay_steps = self.resume["step"]
        self.checkpoints = [self.resume["checkpoint"]]

    def evaluate_model(self, model, mixed_precision):
        model.eval()
        total_loss = 0.0
        total_tokens = 0.0
        with torch.no_grad():
            for offset in range(0, len(self.evaluation_sequences), self.batch_size):
                batch = self.evaluation_sequences[offset : offset + self.batch_size]
                tokens = torch.tensor([item[0] for item in batch], dtype=torch.long, device=self.device)
                mask = torch.tensor([item[1] for item in batch], dtype=torch.float32, device=self.device)
                logits = self.forward_logits(model, tokens[:, :-1], mixed_precision=mixed_precision)
                losses = functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction="none")
                total_loss += float((losses.reshape_as(mask) * mask).sum().detach().cpu().item())
                total_tokens += float(mask.sum().detach().cpu().item())
        return total_loss / total_tokens

    def record_evaluation(self, _training_loss):
        if not self.evaluation_sequences:
            return
        loss_value = self.evaluate_model(self.model, mixed_precision=True)
        self.model.train()
        item = {
            "step": self.step_number,
            "tokens": self.consumed_tokens,
            "metrics": {"heldout_loss": loss_value, "heldout_perplexity": math.exp(min(loss_value, 80.0))},
        }
        self.evaluations.append(item)
        emit(
            "event",
            event={
                "kind": "evaluation",
                "message": f"step {self.step_number} held-out loss {loss_value:.4f}",
                "step": self.step_number,
                "tokens": self.consumed_tokens,
                "evaluation": item,
            },
        )

    def finish(self):
        evaluation_set = self.begin["evaluation_set"]
        if self.evaluation_record_count != evaluation_set["records"] or self.evaluation_token_targets != evaluation_set["token_targets"]:
            raise ValueError(
                f"evaluation stream has {self.evaluation_record_count} records and {self.evaluation_token_targets} targets; "
                f"run BOM pins {evaluation_set['records']} records and {evaluation_set['token_targets']} targets"
            )
        if self.step_number < self.target_steps and len(self.token_buffer) > 1:
            target_mask = self.loss_buffer[1 : self.sequence_length + 1]
            self.add_sequence(self.token_buffer[: self.sequence_length + 1], target_mask, self.corpus_buffer[1 : self.sequence_length + 1])
        if self.step_number < self.target_steps and self.batch:
            self.train_batch()
        if self.step_number != self.target_steps:
            raise ValueError(
                f"canonical stream produced only {self.step_number} training steps; profile requires {self.target_steps}"
            )
        if self.parameters["checkpoint_every"] > 0 and (
            not self.checkpoints or self.checkpoints[-1]["step"] != self.step_number
        ):
            self.save_checkpoint()
        if self.parameters["evaluate_every"] > 0 and (
            not self.evaluations or self.evaluations[-1]["step"] != self.step_number
        ):
            self.record_evaluation(self.final_loss)

        weights_name = "model.safetensors"
        weights_path = os.path.join(self.artifact_directory, weights_name)
        backend_name = "torchtitan" if self.distributed else "pytorch"
        self.save_weights(weights_path, f"waldo-{backend_name}-model", self.step_number)
        if IS_PRIMARY and self.evaluation_sequences:
            artifact_model = DecoderLM(self.architecture)
            missing, unexpected = artifact_model.load_state_dict(load_safetensors(weights_path), strict=False)
            if missing or unexpected:
                raise ValueError(f"saved artifact weights do not match architecture: missing={missing}, unexpected={unexpected}")
            artifact_model.to(device=self.device, dtype=torch.float32)
            # Evaluate the artifact exactly as the live model was evaluated. Under
            # a different compute precision this comparison measures autocast
            # rounding rather than the reload, and that error grows with depth
            # until it exceeds the tolerance for every sufficiently large model.
            artifact_loss = self.evaluate_model(artifact_model, mixed_precision=True)
            del artifact_model
            live_loss = self.evaluations[-1]["metrics"]["heldout_loss"]
            tolerance = max(0.02, abs(live_loss) * 0.01)
            if not math.isfinite(artifact_loss) or abs(artifact_loss - live_loss) > tolerance:
                raise ValueError(
                    f"saved artifact held-out loss {artifact_loss:.6f} does not match live loss "
                    f"{live_loss:.6f} within tolerance {tolerance:.6f}"
                )
            self.evaluations[-1]["metrics"]["artifact_heldout_loss"] = artifact_loss
            self.evaluations[-1]["metrics"]["artifact_heldout_perplexity"] = math.exp(min(artifact_loss, 80.0))
            self.evaluations[-1]["metrics"]["artifact_loss_delta"] = artifact_loss - live_loss
            emit(
                "event",
                event={
                    "kind": "log",
                    "message": f"reloaded model artifact verified at held-out loss {artifact_loss:.4f}",
                    "step": self.step_number,
                    "tokens": self.consumed_tokens,
                },
            )
        config_name = "config.json"
        config_path = os.path.join(self.artifact_directory, config_name)
        backend_revision = TORCHTITAN_REVISION if self.distributed else WORKER_REVISION
        if IS_PRIMARY:
            write_json(
                config_path,
                {
                    "kind": f"waldo-{backend_name}-model-config",
                    "schema": 1,
                    "architecture_sha256": self.begin["architecture_sha256"],
                    "architecture": self.architecture,
                    "training_profile": self.parameters,
                    "initialization": None if self.initialization is None else {
                        "source_type": self.initialization.get("source_type", "run"),
                        "source_id": self.initialization.get("source_id", self.initialization.get("source_run_id")),
                        "source_run_id": self.initialization.get("source_run_id"),
                        "artifact": self.initialization["artifact"],
                    },
                    "backend": {"name": backend_name, "revision": backend_revision, "version": torch.__version__, "device": str(self.device), "world_size": self.world_size},
                },
            )
            write_json(
                os.path.join(self.artifact_directory, "tokenizer.json"),
                {
                    "kind": "waldo-tokenizer",
                    "schema": 1,
                    **self.begin["tokenizer"],
                },
            )
        tokenizer_name = "tokenizer.json"
        tokenizer_path = os.path.join(self.artifact_directory, tokenizer_name)
        if self.distributed:
            torch.distributed.barrier()
        outputs = []
        if IS_PRIMARY:
            outputs = [
                artifact(weights_path, self.logical(weights_name)),
                artifact(config_path, self.logical(config_name)),
                artifact(tokenizer_path, self.logical(tokenizer_name)),
            ]
        emit(
            "complete",
            observation={
                "simulated": False,
                "steps": self.step_number,
                "consumed_tokens": self.consumed_tokens,
                "final_loss": self.final_loss,
                "checkpoints": self.checkpoints,
                "evaluations": self.evaluations,
                "artifacts": outputs,
                "consumption": [
                    {"corpus": corpus, "token_targets": targets}
                    for corpus, targets in sorted(self.consumed_by_corpus.items())
                ],
            },
        )


def stream_lines(distributed):
    """Yield the canonical input stream's lines on every rank.

    Global rank zero alone receives the stream on its torchrun parent's stdin.
    It broadcasts each frame to every rank. Secondary hosts therefore need no
    corpus checkout, object cache, tokenizer, or duplicate record stream.
    """
    if not distributed:
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            yield line

    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    sizes = [None] * world
    torch.distributed.all_gather_object(sizes, local_world)
    if len(set(sizes)) != 1:
        raise ValueError(f"local world sizes differ across nodes: {sorted(set(sizes))}")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    while True:
        values = [sys.stdin.readline() if rank == 0 else None]
        torch.distributed.broadcast_object_list(values, src=0, device=device)
        if not values[0]:
            return
        yield values[0]


def run():
    if len(sys.argv) != 4:
        raise ValueError("worker requires artifact directory, artifact prefix, and device")
    artifact_directory = os.path.abspath(sys.argv[1])
    artifact_prefix = sys.argv[2]
    device = sys.argv[3]
    global IS_PRIMARY
    distributed = device == "torchtitan"
    if distributed:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        IS_PRIMARY = torch.distributed.get_rank() == 0
        emit("event", event={"kind": "log", "message": f"rank {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} process group ready on cuda:{local_rank}"})
    os.makedirs(artifact_directory, exist_ok=True)
    trainer = None
    ended = False
    for line in stream_lines(distributed):
        frame = json.loads(line)
        if frame.get("schema") != PROTOCOL_SCHEMA:
            raise ValueError(f"unsupported worker input schema {frame.get('schema')}")
        kind = frame.get("kind")
        if kind == "begin":
            if trainer is not None:
                raise ValueError("worker received duplicate begin frame")
            if distributed:
                run_ids = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(run_ids, frame["begin"].get("run_id"))
                if len(set(run_ids)) != 1:
                    raise ValueError(f"nodes joined the rendezvous with different runs: {sorted(set(run_ids))}")
            trainer = Trainer(frame["begin"], artifact_directory, artifact_prefix, device)
        elif kind == "record":
            if trainer is None or ended:
                raise ValueError("worker received record outside stream")
            trainer.add_record(frame["record"])
        elif kind == "evaluation_record":
            if trainer is None or ended:
                raise ValueError("worker received evaluation record outside stream")
            trainer.add_evaluation_record(frame["record"])
        elif kind == "end":
            if trainer is None or ended:
                raise ValueError("worker received invalid end frame")
            ended = True
        else:
            raise ValueError(f"unsupported worker input kind {kind!r}")
    if trainer is None or not ended:
        raise ValueError("worker input ended without begin/end framing")
    trainer.finish()


try:
    run()
except Exception as error:
    traceback.print_exc(file=sys.stderr)
    emit("error", error=str(error))
    sys.exit(1)
finally:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
