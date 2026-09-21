# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

import datetime
import hashlib
import inspect
import json
import math
import os
import shutil
import socket
import struct
import sys
import tempfile
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint


PROTOCOL_SCHEMA = 1
WORKER_REVISION = "builtin-pytorch-worker-schema-1-r13"
TORCHTITAN_REVISION = "builtin-torchtitan-worker-schema-1-r24"
IS_PRIMARY = True


class ArtifactIntegrityError(ValueError):
    pass


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


def rendezvous_barrier(run_id, phase, rank, world_size):
    """Synchronize slow control-plane work without consuming an NCCL collective."""
    store = torch.distributed.distributed_c10d._get_default_store()
    prefix = f"waldo-control/{run_id}/{phase}"
    own_key = f"{prefix}/{rank}"
    keys = [f"{prefix}/{peer}" for peer in range(world_size)]
    store.set(own_key, "ready")
    store.wait(keys, datetime.timedelta(hours=6))


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
    def __init__(self, hidden, heads, kv_heads, qk_normalization=False):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        self.qk_normalization = qk_normalization
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
        if self.qk_normalization:
            query = query * torch.rsqrt(query.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(query.dtype)
            key = key * torch.rsqrt(key.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(key.dtype)
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
    def __init__(self, hidden, intermediate, heads, kv_heads, dropout, qk_normalization):
        super().__init__()
        self.attention_norm = RMSNorm(hidden)
        self.attention = Attention(hidden, heads, kv_heads, qk_normalization)
        self.ffn_norm = RMSNorm(hidden)
        self.feed_forward = FeedForward(hidden, intermediate)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, value):
        value = value + self.residual_dropout(self.attention(self.attention_norm(value)))
        return value + self.residual_dropout(self.feed_forward(self.ffn_norm(value)))


class DecoderLM(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        vocabulary = architecture["vocabulary_size"]
        hidden = architecture["hidden_size"]
        self.tie_embeddings = architecture["tie_embeddings"]
        self.activation_checkpointing = False
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.layers = nn.ModuleList([
            DecoderBlock(
                hidden,
                architecture["intermediate_size"],
                architecture["attention_heads"],
                architecture["key_value_heads"],
                architecture.get("dropout", 0.0),
                architecture.get("qk_normalization", False),
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
        if self.architecture.get("initialization", "normal") == "depth-scaled":
            residual_std = 0.02 / math.sqrt(2 * len(self.layers))
            for layer in self.layers:
                nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=residual_std)
                nn.init.normal_(layer.feed_forward.down.weight, mean=0.0, std=residual_std)

    def forward(self, tokens):
        value = self.embedding(tokens)
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                value = checkpoint(layer, value, use_reentrant=False)
            else:
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


def zeropower_via_newton_schulz5(gradient, steps=5):
    """Approximate the nearest semi-orthogonal matrix in reduced precision."""
    if gradient.ndim != 2:
        raise ValueError("Muon requires matrix parameters")
    transposed = gradient.shape[0] > gradient.shape[1]
    value = gradient.mT if transposed else gradient
    value = value.to(torch.bfloat16)
    value = value / (value.norm() + 1e-7)
    for _ in range(steps):
        gram = value @ value.mT
        value = 3.4445 * value + (-4.7750 * gram + 2.0315 * gram @ gram) @ value
    return value.mT if transposed else value


class MuonAdamW(torch.optim.Optimizer):
    """Muon for hidden matrices and AdamW for embeddings, heads, and vectors."""
    def __init__(self, matrix_parameters, adam_parameters, lr, betas, eps, weight_decay):
        groups = []
        if matrix_parameters:
            groups.append({"params": matrix_parameters, "optimizer_kind": "muon", "lr": lr, "momentum": 0.95, "weight_decay": weight_decay})
        if adam_parameters:
            groups.append({"params": adam_parameters, "optimizer_kind": "adamw", "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if group["optimizer_kind"] == "muon":
                    momentum = state.setdefault("momentum_buffer", torch.zeros_like(parameter.grad))
                    momentum.lerp_(parameter.grad, 1 - group["momentum"])
                    update = zeropower_via_newton_schulz5(parameter.grad.lerp(momentum, group["momentum"]))
                    update = update.to(parameter.dtype) * math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
                    continue
                step = state.get("step", 0) + 1
                state["step"] = step
                average = state.setdefault("exp_avg", torch.zeros_like(parameter))
                square = state.setdefault("exp_avg_sq", torch.zeros_like(parameter))
                beta1, beta2 = group["betas"]
                average.lerp_(parameter.grad, 1 - beta1)
                square.mul_(beta2).addcmul_(parameter.grad, parameter.grad, value=1 - beta2)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                denominator = square.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(group["eps"])
                parameter.addcdiv_(average, denominator, value=-group["lr"] / (1 - beta1 ** step))
        return loss


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
        self.parallelism = begin.get("parallelism", {})
        self.parallelism_strategy = self.parallelism.get("strategy", "fully-sharded-data-parallel")
        if self.distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
            self.parallel_dims = None
            if self.parallelism_strategy != "data-parallel":
                # ParallelDims uses the PyTorch fake process-group backend for
                # singleton mesh axes. Importing its registration module is
                # part of TorchTitan's normal runtime initialization contract.
                from torch.testing._internal.distributed import fake_pg as _fake_pg  # noqa: F401
                from torchtitan.distributed import ParallelDims

                nodes = int(self.parallelism.get("nodes", 1))
                gpus_per_node = int(self.parallelism.get("gpus_per_node", self.world_size))
                hybrid = self.parallelism_strategy == "hybrid-sharded-data-parallel"
                parallel_arguments = dict(
                    dp_replicate=nodes if hybrid else 1,
                    dp_shard=gpus_per_node if hybrid else self.world_size,
                    cp=1,
                    tp=1,
                    pp=1,
                    ep=1,
                    world_size=self.world_size,
                )
                # TorchTitan development releases exposed an experimental
                # `etp` dimension; stable 0.3 removed it.
                if "etp" in inspect.signature(ParallelDims).parameters:
                    parallel_arguments["etp"] = 1
                if "spmd_backend" in inspect.signature(ParallelDims).parameters:
                    parallel_arguments["spmd_backend"] = "partial_dtensor"
                self.parallel_dims = ParallelDims(**parallel_arguments)
                self.parallel_dims.build_mesh()
        else:
            self.device = torch.device(device_name)
        self.sequence_length = self.parameters["sequence_length"]
        self.global_batch_size = self.parameters["batch_size"]
        self.gradient_accumulation_steps = self.parameters["gradient_accumulation_steps"]
        self.global_micro_batch_size = self.global_batch_size // self.gradient_accumulation_steps
        if self.distributed:
            if self.global_micro_batch_size < self.world_size or self.global_micro_batch_size % self.world_size != 0:
                raise ValueError(
                    f"global micro-batch size {self.global_micro_batch_size} must be divisible by distributed world size {self.world_size}"
                )
            self.batch_size = self.global_micro_batch_size // self.world_size
        else:
            self.batch_size = self.global_micro_batch_size
        self.target_steps = self.parameters["steps"]
        self.step_number = 0
        self.replay_micro_batches = 0
        self.replay_total_micro_batches = 0
        self.replay_report_micro_batches = 0
        self.consumed_tokens = 0
        self.token_buffer = []
        self.loss_buffer = []
        self.corpus_buffer = []
        self.batch = []
        self.sequence_number = 0
        self.consumed_by_corpus = {}
        self.checkpoints = []
        self.evaluations = []
        self.evaluation_sequences = []
        self.evaluation_record_count = 0
        self.evaluation_token_targets = 0
        self.final_loss = None
        self.started = time.perf_counter()
        self.last_step_finished = self.started
        self.data_wait_seconds = 0.0
        self.skipped_steps = 0
        self.accumulation_number = 0
        self.accumulated_loss_sum = 0.0
        self.accumulated_tokens = 0
        self.accumulated_consumption = {}
        self.accumulated_compute_seconds = 0.0

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
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
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
        self.parameter_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[self.architecture["parameter_dtype"]]
        compute_name = self.parameters.get("compute_precision", "auto")
        if compute_name == "auto":
            compute_name = self.architecture["parameter_dtype"]
        self.compute_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[compute_name]
        if self.device.type == "cpu" and self.compute_dtype == torch.float16:
            raise ValueError("float16 training is not supported by the PyTorch CPU adapter; use bfloat16 or float32")
        # Keep master weights and AdamW state in FP32. Reduced precision is a
        # compute and portable-artifact format, not optimizer state.
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.activation_checkpointing = bool(self.parameters.get("activation_checkpointing", False))
        if self.distributed:
            if self.initialization is not None:
                for parameter in self.model.parameters():
                    torch.distributed.broadcast(parameter.data, src=0)
            if self.parallelism_strategy == "data-parallel":
                from torch.nn.parallel import DistributedDataParallel

                local_rank = int(os.environ["LOCAL_RANK"])
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                    static_graph=True,
                )
            else:
                from torch.distributed._composable.fsdp import fully_shard

                fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
                for layer in self.model.layers:
                    fully_shard(layer, mesh=fsdp_mesh)
                fully_shard(self.model, mesh=fsdp_mesh)
        self.compiled_model = torch.compile(self.model, dynamic=False) if self.parameters.get("compile", False) else self.model
        optimizer_parameters = self.parameters["optimizer"]
        if optimizer_parameters["name"] == "muon-adamw":
            if self.distributed and self.parallelism_strategy != "data-parallel":
                raise ValueError("muon-adamw currently requires data-parallel placement; sharded Muon is not yet implemented")
            matrix_parameters = []
            adam_parameters = []
            for name, parameter in self.model.named_parameters():
                if parameter.ndim == 2 and "embedding" not in name and "output" not in name:
                    matrix_parameters.append(parameter)
                else:
                    adam_parameters.append(parameter)
            self.optimizer = MuonAdamW(
                matrix_parameters,
                adam_parameters,
                lr=self.parameters["learning_rate"],
                betas=(optimizer_parameters["beta1"], optimizer_parameters["beta2"]),
                eps=optimizer_parameters["epsilon"],
                weight_decay=optimizer_parameters["weight_decay"],
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.parameters["learning_rate"],
                betas=(optimizer_parameters["beta1"], optimizer_parameters["beta2"]),
                eps=optimizer_parameters["epsilon"],
                weight_decay=optimizer_parameters["weight_decay"],
            )
        scaler_enabled = self.device.type == "cuda" and self.compute_dtype == torch.float16
        if scaler_enabled and self.distributed and self.parallelism_strategy != "data-parallel":
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(enabled=True)
        else:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        if self.resume is not None:
            self.restore_checkpoint()
            if self.distributed:
                # Checkpoint restoration includes loading and repartitioning a
                # potentially multi-gigabyte optimizer state. One node can
                # finish substantially earlier than another. Do not let a
                # faster node enter stream replay while another node is still
                # restoring; its first replay barrier would otherwise be
                # mistaken for a failed training collective.
                print(
                    f"rank {self.rank}/{self.world_size} restored checkpoint step {self.resume['step']}; waiting for all ranks",
                    file=sys.stderr,
                    flush=True,
                )
                emit(
                    "event",
                    event={
                        "kind": "log",
                        "message": f"rank 0 restored checkpoint step {self.resume['step']}; waiting for {self.world_size - 1} other ranks",
                    },
                )
                # Optimizer restoration itself uses NCCL state-dict
                # collectives. Flush that work on the training process group
                # before switching to the control plane for stream replay.
                torch.distributed.barrier()
                emit(
                    "event",
                    event={
                        "kind": "log",
                        "message": (
                            f"checkpoint step {self.resume['step']} restored on all {self.world_size} ranks; "
                            + (
                                "positioning each node's deterministic input stream at the checkpoint boundary"
                                if self.parallelism.get("data_plane") == "node-local-cache"
                                else "replaying the deterministic input stream"
                            )
                        ),
                    },
                )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.last_step_finished = time.perf_counter()
        if self.distributed:
            emit(
                "event",
                event={
                    "kind": "log",
                    "message": (
                        f"global batch {self.global_batch_size} split into {self.gradient_accumulation_steps} micro-batches; "
                        f"each rank processes {self.batch_size} distinct sequences per micro-batch"
                    ),
                },
            )

    def forward_logits(self, model, tokens, mixed_precision=True):
        enabled = mixed_precision and self.compute_dtype != torch.float32
        with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=enabled):
            target = self.compiled_model if model is self.model else model
            return target(tokens)

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
        if schedule["name"] == "warmup-stable-warmdown":
            warmdown = schedule.get("warmdown_steps", 0)
            stable_end = self.target_steps - warmdown
            if warmdown == 0 or step <= stable_end:
                return base
            progress = min(1.0, max(0.0, (step - stable_end) / warmdown))
            return base * (1.0 - progress * (1.0 - schedule["minimum_rate_ratio"]))
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
        if self.distributed:
            owner = self.sequence_number % self.world_size
            self.sequence_number += 1
            if owner == self.rank:
                self.batch.append((padded, mask, corpus_counts))
            # Every rank observes every sequence. Enter the optimizer step only
            # at a complete global-batch boundary so distributed collectives
            # remain in identical order while each rank computes a distinct
            # local slice.
            if self.sequence_number % self.global_micro_batch_size == 0:
                if len(self.batch) != self.batch_size:
                    raise ValueError(
                        f"rank {self.rank} assembled {len(self.batch)} sequences; expected {self.batch_size}"
                    )
                self.train_batch()
        else:
            self.batch.append((padded, mask, corpus_counts))
            if len(self.batch) >= self.batch_size:
                self.train_batch()

    def add_prepared_sequence(self, sequence):
        if not self.distributed:
            raise ValueError("prepared rank sequence requires distributed training")
        ordinal = int(sequence["ordinal"])
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        node_rank = int(os.environ.get("GROUP_RANK", "0"))
        owner = ordinal % self.world_size
        if owner // local_world != node_rank:
            raise ValueError(f"prepared sequence {ordinal} was delivered to the wrong node")
        if owner == self.rank:
            tokens = [int(token) for token in sequence["tokens"]]
            target_mask = [float(value) for value in sequence["loss_mask"]]
            window = self.sequence_length + 1
            if len(tokens) > window or len(target_mask) != len(tokens) - 1:
                raise ValueError(f"prepared sequence {ordinal} has invalid dimensions")
            padded = tokens + [self.tokenizer.pad_id] * (window - len(tokens))
            mask = target_mask + [0.0] * (self.sequence_length - len(target_mask))
            self.batch.append((padded, mask, sequence.get("consumption", {})))

    def finish_prepared_micro_batch(self):
        if not self.distributed or len(self.batch) != self.batch_size:
            raise ValueError(
                f"rank {self.rank} assembled {len(self.batch)} prepared sequences; expected {self.batch_size}"
            )
        self.train_batch()

    def train_batch(self, final=False):
        if not self.batch or self.step_number >= self.target_steps:
            self.batch = []
            return
        if self.replay_micro_batches > 0:
            self.replay_micro_batches -= 1
            self.batch = []
            self.last_step_finished = time.perf_counter()
            replayed = self.replay_total_micro_batches - self.replay_micro_batches
            if self.distributed and self.replay_micro_batches == 0:
                # Node-local preparation can progress at very different rates.
                # Let every rank replay independently, then use the long-lived
                # CPU control group to synchronize before any new NCCL gradient
                # collective. Periodic NCCL barriers can time out merely because
                # another node is still reading or tokenizing its local stream.
                rendezvous_barrier(self.begin["run_id"], "checkpoint-replayed", self.rank, self.world_size)
            if IS_PRIMARY and (
                replayed % self.replay_report_micro_batches == 0 or self.replay_micro_batches == 0
            ):
                replayed_steps = replayed // self.gradient_accumulation_steps
                target_steps = self.replay_total_micro_batches // self.gradient_accumulation_steps
                if self.replay_micro_batches == 0:
                    message = f"checkpoint replay {replayed_steps}/{target_steps} optimizer steps complete and synchronized across all ranks"
                else:
                    message = f"checkpoint replay {replayed_steps}/{target_steps} optimizer steps complete on rank 0"
                emit(
                    "event",
                    event={
                        "kind": "log",
                        "message": message,
                    },
                )
            return
        step_started = time.perf_counter()
        self.data_wait_seconds += max(0.0, step_started - self.last_step_finished)
        if self.accumulation_number == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.accumulated_loss_sum = 0.0
            self.accumulated_tokens = 0
            self.accumulated_consumption = {}
            self.accumulated_compute_seconds = 0.0
        tokens = torch.tensor([item[0] for item in self.batch], dtype=torch.long)
        mask = torch.tensor([item[1] for item in self.batch], dtype=torch.float32)
        if self.device.type == "cuda":
            tokens = tokens.pin_memory().to(self.device, non_blocking=True)
            mask = mask.pin_memory().to(self.device, non_blocking=True)
        else:
            tokens = tokens.to(self.device)
            mask = mask.to(self.device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        next_step = self.step_number + 1
        report_every = max(1, self.target_steps // 100)
        should_report = next_step == 1 or next_step == self.target_steps or next_step % report_every == 0
        current_learning_rate = self.learning_rate(next_step)
        for group in self.optimizer.param_groups:
            group["lr"] = current_learning_rate
        logits = self.forward_logits(self.model, inputs)
        losses = functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
        loss_sum = (losses.reshape_as(mask) * mask).sum()
        local_valid_tokens = mask.sum()
        if self.distributed:
            global_valid_tokens = local_valid_tokens.detach().clone()
            torch.distributed.all_reduce(global_valid_tokens, op=torch.distributed.ReduceOp.SUM)
            # Distributed wrappers average gradients across ranks. Scaling the
            # unnormalized local sum by world size yields the global token-loss
            # sum; accumulated gradients are normalized once at optimizer step.
            loss = loss_sum * self.world_size
        else:
            global_valid_tokens = local_valid_tokens
            loss = loss_sum
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if self.distributed:
            global_loss_sum = loss_sum.detach().clone()
            torch.distributed.all_reduce(global_loss_sum, op=torch.distributed.ReduceOp.SUM)
        else:
            global_loss_sum = loss_sum.detach()
        valid_tokens = int(global_valid_tokens.cpu().item())
        if valid_tokens <= 0 and self.accumulated_tokens == 0:
            raise ValueError("gradient accumulation micro-batch has no supervised token targets")
        self.accumulated_loss_sum += float(global_loss_sum.cpu().item())
        self.accumulated_tokens += valid_tokens
        for item in self.batch:
            for corpus, count in item[2].items():
                self.accumulated_consumption[corpus] = self.accumulated_consumption.get(corpus, 0) + count
        self.batch = []
        self.accumulation_number += 1
        self.accumulated_compute_seconds += max(time.perf_counter() - step_started, 1e-9)
        if self.accumulation_number < self.gradient_accumulation_steps and not final:
            self.last_step_finished = time.perf_counter()
            return

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(self.accumulated_tokens)
        gradient_norm = None
        if should_report:
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), math.inf).detach().cpu().item())
        update_started = time.perf_counter()
        if self.scaler.is_enabled():
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < scale_before:
                self.skipped_steps += 1
        else:
            self.optimizer.step()
        self.synchronize()
        self.accumulated_compute_seconds += max(time.perf_counter() - update_started, 1e-9)
        loss_value = self.accumulated_loss_sum / self.accumulated_tokens
        valid_tokens = self.accumulated_tokens
        step_seconds = max(self.accumulated_compute_seconds, 1e-9)
        self.step_number = next_step
        self.consumed_tokens += valid_tokens
        for corpus, count in self.accumulated_consumption.items():
            self.consumed_by_corpus[corpus] = self.consumed_by_corpus.get(corpus, 0) + count
        self.final_loss = loss_value
        self.accumulation_number = 0
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        throughput = self.consumed_tokens / elapsed
        eta = int(max(0.0, (self.target_steps - self.step_number) * elapsed / self.step_number))
        if should_report:
            peak_memory = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
            data_wait_seconds = self.data_wait_seconds
            if self.distributed:
                peak_tensor = torch.tensor(peak_memory, dtype=torch.int64, device=self.device)
                wait_tensor = torch.tensor(data_wait_seconds, dtype=torch.float64, device=self.device)
                torch.distributed.all_reduce(peak_tensor, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(wait_tensor, op=torch.distributed.ReduceOp.MAX)
                peak_memory = int(peak_tensor.cpu().item())
                data_wait_seconds = float(wait_tensor.cpu().item())
            step_flops = 6 * self.parameter_count * valid_tokens
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
                    "duration_seconds": step_seconds,
                    "data_wait_seconds": data_wait_seconds,
                    "peak_memory_bytes": peak_memory,
                    "training_flops": 6 * self.parameter_count * self.consumed_tokens,
                    "achieved_tflops": step_flops / step_seconds / 1e12,
                    "gradient_norm": gradient_norm,
                    "skipped_steps": self.skipped_steps,
                    "eta_seconds": eta,
                },
            )
        evaluate_every = self.parameters["evaluate_every"]
        evaluate_due = evaluate_every > 0 and (self.step_number == 1 or self.step_number % evaluate_every == 0)
        if evaluate_due:
            self.record_evaluation(loss_value)
        checkpoint_every = self.parameters["checkpoint_every"]
        if checkpoint_every > 0 and self.step_number % checkpoint_every == 0 and not any(checkpoint["step"] == self.step_number for checkpoint in self.checkpoints):
            self.save_checkpoint()
        self.last_step_finished = time.perf_counter()

    def model_state_dict(self):
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
        return tensors

    def save_weights(self, path, kind, step, portable):
        tensors = self.model_state_dict()
        if IS_PRIMARY:
            self.save_weight_tensors(path, tensors, kind, step, self.parameter_dtype if portable else None)
        if self.distributed:
            torch.distributed.barrier()

    def save_weight_tensors(self, path, tensors, kind, step, storage_dtype):
        if storage_dtype is not None:
            tensors = {
                name: value.to(dtype=storage_dtype) if value.is_floating_point() else value
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
                "storage_dtype": "float32" if storage_dtype is None else self.architecture["parameter_dtype"],
            },
        )

    def gather_consumption(self):
        local = dict(self.consumed_by_corpus)
        if not self.distributed:
            return local, [local]
        states = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(states, local)
        total = {}
        for state in states:
            for corpus, count in state.items():
                total[corpus] = total.get(corpus, 0) + count
        return total, states

    def report_checkpoint(self, item):
        emit(
            "event",
            event={
                "kind": "checkpoint",
                "message": f"checkpoint step {item['step']} persisted",
                "step": item["step"],
                "tokens": item["tokens"],
                "checkpoint": item,
            },
        )

    def save_checkpoint(self, report=True):
        consumption, consumption_states = self.gather_consumption()
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
        # Checkpoints are exact continuation state. Keep FP32 master weights;
        # parameter_dtype applies to the portable terminal artifact, not resume.
        self.save_weights(weights_path, f"waldo-{backend_name}-checkpoint", self.step_number, portable=False)
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
            torch.save(
                {
                    "optimizer": optimizer_state,
                    "scaler": self.scaler.state_dict(),
                    "skipped_steps": self.skipped_steps,
                    "random_states": random_states,
                    "consumption_states": consumption_states,
                },
                runtime_path,
            )
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
                    "consumption": consumption,
                    "world_size": self.world_size,
                    "parallelism_strategy": self.parallelism_strategy,
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
        if report:
            self.report_checkpoint(item)
        return item

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
            or state.get("parallelism_strategy", "fully-sharded-data-parallel") != self.parallelism_strategy
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
        self.scaler.load_state_dict(runtime.get("scaler", {}))
        self.skipped_steps = int(runtime.get("skipped_steps", 0))
        random_state = runtime["random_states"][self.rank]
        torch.set_rng_state(random_state["cpu"])
        if self.device.type == "cuda" and random_state["cuda"] is not None:
            torch.cuda.set_rng_state(random_state["cuda"], self.device)
        self.step_number = self.resume["step"]
        self.consumed_tokens = self.resume["tokens"]
        if self.distributed:
            self.consumed_by_corpus = runtime["consumption_states"][self.rank]
        else:
            self.consumed_by_corpus = state.get("consumption", {})
        self.replay_micro_batches = self.resume["step"] * self.gradient_accumulation_steps
        if self.parallelism.get("data_plane") == "node-local-cache":
            # WALDO's prepared stream starts at the exact checkpoint boundary,
            # so Python must not consume the already-trained prefix again.
            self.replay_micro_batches = 0
        self.replay_total_micro_batches = self.replay_micro_batches
        report_steps = max(256, ((self.resume["step"] + 19) // 20 + 255) // 256 * 256)
        self.replay_report_micro_batches = report_steps * self.gradient_accumulation_steps
        self.checkpoints = list(self.resume.get("checkpoints") or [self.resume["checkpoint"]])
        self.evaluations = list(self.resume.get("evaluations") or [])

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

    def evaluate_weight_file(self, path, quantize):
        loss_value = 0.0
        failure = None
        if IS_PRIMARY:
            try:
                tensors = load_safetensors(path)
                if quantize:
                    tensors = {
                        name: value.to(dtype=self.parameter_dtype) if value.is_floating_point() else value
                        for name, value in tensors.items()
                    }
                artifact_model = DecoderLM(self.architecture)
                missing, unexpected = artifact_model.load_state_dict(tensors, strict=False)
                del tensors
                if missing or unexpected:
                    raise ValueError(f"weights do not match portable architecture: missing={missing}, unexpected={unexpected}")
                artifact_model.to(device=self.device, dtype=torch.float32)
                loss_value = self.evaluate_model(artifact_model, mixed_precision=True)
                del artifact_model
            except Exception as error:
                failure = str(error)
        if self.distributed:
            failures = [failure]
            torch.distributed.broadcast_object_list(failures, src=0, device=self.device)
            failure = failures[0]
        if failure is not None:
            raise ValueError(f"evaluate persisted weights: {failure}")
        if self.distributed:
            value = torch.tensor(loss_value, dtype=torch.float64, device=self.device)
            torch.distributed.broadcast(value, src=0)
            loss_value = float(value.cpu().item())
        return loss_value

    def record_evaluation(self, _training_loss):
        if not self.evaluation_sequences:
            return
        live_loss = self.evaluate_model(self.model, mixed_precision=True)
        self.model.train()
        checkpoint = next((item for item in self.checkpoints if item["step"] == self.step_number), None)
        unpublished_checkpoint = checkpoint is None
        if unpublished_checkpoint:
            checkpoint = self.save_checkpoint(report=False)
        checkpoint_path = os.path.join(
            self.artifact_directory,
            "checkpoints",
            f"step-{self.step_number:08d}",
            "model.safetensors",
        )
        artifact_loss = self.evaluate_weight_file(checkpoint_path, quantize=True)
        tolerance = max(0.02, abs(live_loss) * 0.01)
        if not math.isfinite(artifact_loss):
            raise ArtifactIntegrityError("publishable checkpoint held-out loss is not finite")
        if abs(artifact_loss - live_loss) > tolerance:
            raise ArtifactIntegrityError(
                f"publishable {self.architecture['parameter_dtype']} checkpoint held-out loss {artifact_loss:.6f} "
                f"does not match live FP32-master loss {live_loss:.6f} within tolerance {tolerance:.6f}; "
                "use parameter_dtype float32 or correct target-dtype training before spending more compute"
            )
        if unpublished_checkpoint:
            self.report_checkpoint(checkpoint)
        item = {
            "step": self.step_number,
            "tokens": self.consumed_tokens,
            "metrics": {
                "heldout_loss": artifact_loss,
                "heldout_perplexity": math.exp(min(artifact_loss, 80.0)),
                "live_compiled_heldout_loss": live_loss,
                "publishable_checkpoint_heldout_loss": artifact_loss,
                "artifact_loss_delta": artifact_loss - live_loss,
            },
        }
        self.evaluations.append(item)
        emit(
            "event",
            event={
                "kind": "evaluation",
                "message": f"step {self.step_number} publishable held-out loss {artifact_loss:.4f} (live {live_loss:.4f})",
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
        if self.step_number < self.target_steps:
            if self.distributed:
                pending = [None for _ in range(self.world_size)]
                torch.distributed.all_gather_object(pending, len(self.batch))
                if max(pending) > 0:
                    missing = self.batch_size - len(self.batch)
                    if missing < 0:
                        raise ValueError(
                            f"rank {self.rank} has {len(self.batch)} sequences in the final batch; "
                            f"local batch capacity is {self.batch_size}"
                        )
                    padding = [self.tokenizer.pad_id] * (self.sequence_length + 1)
                    zero_mask = [0.0] * self.sequence_length
                    self.batch.extend((padding, zero_mask, {}) for _ in range(missing))
                    if IS_PRIMARY:
                        real_sequences = sum(pending)
                        padded_slots = self.global_micro_batch_size - real_sequences
                        emit(
                            "event",
                            event={
                                "kind": "log",
                                "message": (
                                    f"final partial global batch has {real_sequences} training sequences and "
                                    f"{padded_slots} empty slots; empty slots contribute zero loss"
                                ),
                            },
                        )
                    self.train_batch(final=True)
                else:
                    self.batch = []
            elif self.batch:
                self.train_batch(final=True)
        if self.step_number < self.target_steps and self.accumulation_number > 0:
            padding = [self.tokenizer.pad_id] * (self.sequence_length + 1)
            zero_mask = [0.0] * self.sequence_length
            self.batch = [(padding, zero_mask, {}) for _ in range(self.batch_size)]
            self.train_batch(final=True)
        if self.step_number != self.target_steps:
            raise ValueError(
                f"canonical stream produced only {self.step_number} training steps; profile requires {self.target_steps}"
            )
        if self.parameters["evaluate_every"] > 0 and (
            not self.evaluations or self.evaluations[-1]["step"] != self.step_number
        ):
            self.record_evaluation(self.final_loss)
        if self.parameters["checkpoint_every"] > 0 and (
            not self.checkpoints or self.checkpoints[-1]["step"] != self.step_number
        ):
            self.save_checkpoint()

        final_consumption, _ = self.gather_consumption()

        weights_name = "model.safetensors"
        weights_path = os.path.join(self.artifact_directory, weights_name)
        backend_name = "torchtitan" if self.distributed else "pytorch"
        evaluated_checkpoints = {checkpoint["step"]: checkpoint for checkpoint in self.checkpoints}
        candidates = [evaluation for evaluation in self.evaluations if evaluation["step"] in evaluated_checkpoints]
        selected_evaluation = min(candidates, key=lambda evaluation: evaluation["metrics"]["heldout_loss"]) if candidates else None
        selected_checkpoint = None if selected_evaluation is None else evaluated_checkpoints[selected_evaluation["step"]]
        if IS_PRIMARY and selected_checkpoint is not None:
            selected_path = os.path.join(self.artifact_directory, "checkpoints", f"step-{selected_checkpoint['step']:08d}", "model.safetensors")
            self.save_weight_tensors(
                weights_path,
                load_safetensors(selected_path),
                f"waldo-{backend_name}-model",
                selected_checkpoint["step"],
                self.parameter_dtype,
            )
        elif selected_checkpoint is None:
            self.save_weights(weights_path, f"waldo-{backend_name}-model", self.step_number, portable=True)
        if self.distributed and selected_checkpoint is not None:
            torch.distributed.barrier()
        selection = None
        artifact_loss = self.evaluate_weight_file(weights_path, quantize=False) if self.evaluation_sequences else None
        if IS_PRIMARY and artifact_loss is not None:
            selected_evaluation = self.evaluations[-1] if selected_evaluation is None else selected_evaluation
            candidate_loss = selected_evaluation["metrics"]["heldout_loss"]
            tolerance = max(0.02, abs(candidate_loss) * 0.01)
            if not math.isfinite(artifact_loss):
                raise ArtifactIntegrityError("saved artifact held-out loss is not finite")
            if abs(artifact_loss - candidate_loss) > tolerance:
                raise ArtifactIntegrityError(
                    f"saved artifact held-out loss {artifact_loss:.6f} does not match selected publishable "
                    f"checkpoint loss {candidate_loss:.6f} within tolerance {tolerance:.6f}"
                )
            metrics = selected_evaluation["metrics"]
            metrics["artifact_heldout_loss"] = artifact_loss
            metrics["artifact_heldout_perplexity"] = math.exp(min(artifact_loss, 80.0))
            metrics["serialization_loss_delta"] = artifact_loss - candidate_loss
            selection = {
                "step": selected_evaluation["step"],
                "tokens": selected_evaluation["tokens"],
                "metric": "heldout_loss",
                "value": candidate_loss,
            }
            emit(
                "event",
                event={
                    "kind": "log",
                    "message": f"selected checkpoint step {selected_evaluation['step']} and verified the model artifact at held-out loss {artifact_loss:.4f}",
                    "step": selected_evaluation["step"],
                    "tokens": selected_evaluation["tokens"],
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
                "selected_checkpoint": selection,
                "artifacts": outputs,
                "consumption": [
                    {"corpus": corpus, "token_targets": targets}
                    for corpus, targets in sorted(final_consumption.items())
                ],
            },
        )


def receive_exact(connection, size):
    value = bytearray()
    while len(value) < size:
        block = connection.recv(size - len(value))
        if not block:
            raise ValueError("node-local input stream ended unexpectedly")
        value.extend(block)
    return bytes(value)


def node_local_stream_path(artifact_directory, node_rank):
    identity = hashlib.sha256(
        f"{os.path.abspath(artifact_directory)}:{node_rank}".encode("utf-8")
    ).hexdigest()[:24]
    return os.path.join(tempfile.gettempdir(), f"waldo-stream-{identity}.sock")


def stream_lines(distributed, artifact_directory):
    """Yield the canonical input stream's lines on every rank.

    The default multi-node path gives each node one identical stream from its
    verified node-local cache. The node's local rank zero broadcasts frames
    only to sibling ranks, keeping record traffic off the training network.
    Launcher-stream compatibility falls back to global-rank-zero broadcast.
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
    node_local = os.environ.get("WALDO_TORCH_DATA_PLANE") == "node-local-cache"
    source_rank = 0
    if node_local:
        node_rank = int(os.environ.get("GROUP_RANK", str(rank // local_world)))
        source_rank = node_rank * local_world
        socket_path = node_local_stream_path(artifact_directory, node_rank)
        if rank == source_rank:
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(socket_path)
            os.chmod(socket_path, 0o600)
            server.listen(local_world - 1)
            siblings = [server.accept()[0] for _ in range(local_world - 1)]
            try:
                while True:
                    line = sys.stdin.readline()
                    encoded = line.encode("utf-8") if line else b""
                    header = struct.pack("!q", len(encoded) if line else -1)
                    for connection in siblings:
                        connection.sendall(header)
                        if encoded:
                            connection.sendall(encoded)
                    if not line:
                        return
                    yield line
            finally:
                for connection in siblings:
                    connection.close()
                server.close()
                try:
                    os.unlink(socket_path)
                except FileNotFoundError:
                    pass
        else:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            deadline = time.monotonic() + 60
            while True:
                try:
                    connection.connect(socket_path)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out joining node-local input stream")
                    time.sleep(0.05)
            try:
                while True:
                    byte_count = struct.unpack("!q", receive_exact(connection, 8))[0]
                    if byte_count < 0:
                        return
                    yield receive_exact(connection, byte_count).decode("utf-8")
            finally:
                connection.close()
        return
    device = torch.device(f"cuda:{local_rank}")
    while True:
        line = sys.stdin.readline() if rank == source_rank else ""
        values = [line if rank == source_rank else None]
        torch.distributed.broadcast_object_list(values, src=source_rank, device=device)
        line = values[0]
        if not line:
            return
        yield line


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
    for line in stream_lines(distributed, artifact_directory):
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
        elif kind == "sequence":
            if trainer is None or ended:
                raise ValueError("worker received prepared sequence outside stream")
            trainer.add_prepared_sequence(frame["sequence"])
        elif kind == "micro_batch_end":
            if trainer is None or ended:
                raise ValueError("worker received micro-batch boundary outside stream")
            trainer.finish_prepared_micro_batch()
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
    emit("error", error=str(error), error_class="artifact-integrity" if isinstance(error, ArtifactIntegrityError) else "")
    sys.exit(1)
finally:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
