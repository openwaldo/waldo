# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

"""Reproducible WALDO DDP training throughput benchmark.

The benchmark uses fixed synthetic global batches, excludes input generation,
and prints one JSON result per optimization. Run it on four GPUs with:

  torchrun --standalone --nproc-per-node=4 \
    testing/python/pytorch_throughput_benchmark.py \
    --model-source internal/pytorchruntime/model.py

For two hosts, use torchrun's normal --nnodes/--node-rank/--master-* options.
The default architecture is WALDO's 1.034B rung architecture. Use --tiny only
for a quick functional smoke test.
"""

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import os
import time

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel


VARIANTS = (
    ("legacy-accum2", 2, False, False, "plain", False),
    ("legacy-accum1", 1, False, False, "plain", False),
    ("no-sync", 2, True, False, "plain", False),
    ("boundary-scalars", 2, True, True, "plain", False),
    ("foreach", 2, True, True, "foreach", False),
    ("fused-adamw", 2, True, True, "fused", False),
    ("compiled-accum1", 1, True, True, "foreach", True),
)


def load_model(path):
    spec = importlib.util.spec_from_file_location("waldo_benchmark_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load model source {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def architecture(tiny):
    if tiny:
        return {
            "vocabulary_size": 257,
            "hidden_size": 128,
            "intermediate_size": 352,
            "layers": 4,
            "attention_heads": 4,
            "key_value_heads": 2,
            "dropout": 0.0,
            "qk_normalization": False,
            "initialization": "depth-scaled",
            "tie_embeddings": True,
        }
    return {
        "vocabulary_size": 32000,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "layers": 22,
        "attention_heads": 32,
        "key_value_heads": 4,
        "dropout": 0.0,
        "qk_normalization": False,
        "initialization": "depth-scaled",
        "tie_embeddings": True,
    }


def synchronize():
    torch.cuda.synchronize()


def measured(call):
    synchronize()
    started = time.perf_counter()
    value = call()
    synchronize()
    return value, time.perf_counter() - started


class CommunicationTimer:
    """Measures DDP bucket all-reduces without changing their reduction result."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.seconds = 0.0

    def hook(self, _, bucket):
        started = time.perf_counter()
        future = distributed.all_reduce(bucket.buffer(), async_op=True).get_future()

        def complete(result):
            value = result.value()[0]
            value.div_(self.world_size)
            self.seconds += time.perf_counter() - started
            return value

        return future.then(complete)


def fixed_local_microbatches(step, accumulation, rank, world_size, sequence_length, vocabulary_size, device):
    global_microbatch = 8 // accumulation
    local_size = global_microbatch // world_size
    result = []
    for microbatch in range(accumulation):
        rows = []
        for local_index in range(local_size):
            ordinal = microbatch * global_microbatch + rank * local_size + local_index
            generator = torch.Generator().manual_seed(170_003 + step * 97 + ordinal)
            rows.append(torch.randint(0, vocabulary_size, (sequence_length + 1,), generator=generator))
        result.append(torch.stack(rows).pin_memory().to(device, non_blocking=True))
    synchronize()
    return result


def optimizer_for(model, implementation):
    options = {}
    if implementation == "foreach":
        options["foreach"] = True
    elif implementation == "fused":
        options["fused"] = True
    return torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, **options)


def train_step(model, execution_model, optimizer, batches, accumulation, use_no_sync, boundary_scalars, gradient_mode, world_size):
    phases = {name: 0.0 for name in ("forward", "backward", "communication", "gradient_processing", "optimizer")}
    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = torch.zeros((), dtype=torch.float64, device=batches[0].device)
    local_tokens = torch.zeros((), dtype=torch.float64, device=batches[0].device)
    global_loss_sum = 0.0
    global_tokens = 0
    for index, tokens in enumerate(batches):
        final = index + 1 == accumulation
        synchronization = model.no_sync() if use_no_sync and not final else contextlib.nullcontext()
        with synchronization:
            def forward():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = execution_model(tokens[:, :-1])
                losses = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction="none"
                )
                return losses.sum()

            loss_sum, elapsed = measured(forward)
            phases["forward"] += elapsed
            _, elapsed = measured(lambda: (loss_sum * world_size).backward())
            phases["backward"] += elapsed
        token_count = torch.tensor(tokens[:, 1:].numel(), dtype=torch.float64, device=tokens.device)
        if boundary_scalars:
            local_loss_sum.add_(loss_sum.detach().double())
            local_tokens.add_(token_count)
        else:
            metrics = torch.stack((loss_sum.detach().double(), token_count))
            _, elapsed = measured(lambda: distributed.all_reduce(metrics))
            phases["communication"] += elapsed
            global_loss_sum += float(metrics[0].item())
            global_tokens += int(metrics[1].item())
    if boundary_scalars:
        metrics = torch.stack((local_loss_sum, local_tokens))
        _, elapsed = measured(lambda: distributed.all_reduce(metrics))
        phases["communication"] += elapsed
        global_loss_sum = float(metrics[0].item())
        global_tokens = int(metrics[1].item())

    def process_gradients():
        parameters = list(model.parameters())
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if gradient_mode in ("foreach", "fused"):
            torch._foreach_div_(gradients, global_tokens)
            norm = torch.nn.utils.clip_grad_norm_(parameters, math.inf, foreach=True)
        else:
            for gradient in gradients:
                gradient.div_(global_tokens)
            norm = torch.nn.utils.clip_grad_norm_(parameters, math.inf)
        if not torch.isfinite(norm):
            raise RuntimeError("non-finite benchmark gradient")

    _, phases["gradient_processing"] = measured(process_gradients)
    _, phases["optimizer"] = measured(optimizer.step)
    return global_loss_sum / global_tokens, phases


def model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.module.state_dict().items()}


def maximum_state_error(left, right):
    return max(float((left[name].double() - right[name].double()).abs().max()) for name in left)


def run_variant(module, architecture_value, sequence_length, warmup, steps, variant, baseline):
    name, accumulation, use_no_sync, boundary_scalars, gradient_mode, compile_model = variant
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    base = module.DecoderLM(architecture_value)
    base.initialize()
    base.to(device=torch.device(f"cuda:{local_rank}"), dtype=torch.float32)
    model = DistributedDataParallel(
        base,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    communication = CommunicationTimer(world_size)
    model.register_comm_hook(communication, communication.hook)
    execution_model = torch.compile(model, dynamic=False) if compile_model else model
    optimizer = optimizer_for(model, gradient_mode)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    totals = {key: 0.0 for key in ("forward", "backward", "communication", "ddp_communication", "gradient_processing", "optimizer")}
    measured_seconds = 0.0
    losses = []
    for step in range(warmup + steps):
        batches = fixed_local_microbatches(
            step, accumulation, rank, world_size, sequence_length, architecture_value["vocabulary_size"], torch.device(f"cuda:{local_rank}")
        )
        communication_before = communication.seconds
        synchronize()
        started = time.perf_counter()
        loss, phases = train_step(
            model, execution_model, optimizer, batches, accumulation, use_no_sync, boundary_scalars, gradient_mode, world_size
        )
        synchronize()
        step_seconds = time.perf_counter() - started
        ddp_communication = communication.seconds - communication_before
        phases["communication"] += ddp_communication
        phases["ddp_communication"] = ddp_communication
        if step >= warmup:
            measured_seconds += step_seconds
            losses.append(loss)
            for key in totals:
                totals[key] += phases[key]
    state = model_state(model) if rank == 0 else None
    seconds_per_step = measured_seconds / steps
    tokens_per_step = 8 * sequence_length
    result = {
        "variant": name,
        "world_size": world_size,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(local_rank),
        "parameters": parameter_count,
        "sequence_length": sequence_length,
        "global_batch": 8,
        "gradient_accumulation_steps": accumulation,
        "local_sequences_per_microbatch": (8 // accumulation) // world_size,
        "optimizer_step_seconds": seconds_per_step,
        "tokens_per_second": tokens_per_step / seconds_per_step,
        "achieved_tflops": 6 * parameter_count * tokens_per_step / seconds_per_step / 1e12,
        "phase_seconds_per_step": {
            **{key: value / steps for key, value in totals.items()},
            "backward_compute_excluding_ddp_communication": max(
                0.0, (totals["backward"] - totals["ddp_communication"]) / steps
            ),
        },
        "final_loss": losses[-1],
        "numerical_comparison": None,
    }
    if rank == 0 and baseline is not None:
        error = maximum_state_error(baseline, state)
        tolerance = 2e-4 if compile_model or accumulation == 1 else 2e-5
        result["numerical_comparison"] = {
            "baseline": "legacy-accum2",
            "maximum_weight_error": error,
            "absolute_tolerance": tolerance,
            "passed": error <= tolerance,
        }
        if error > tolerance:
            raise AssertionError(f"{name} maximum weight error {error} exceeds {tolerance}")
    distributed.barrier()
    return result, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--variants", default=",".join(item[0] for item in VARIANTS))
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("throughput benchmark requires CUDA")
    distributed.init_process_group("nccl")
    rank = distributed.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    world_size = distributed.get_world_size()
    if 8 % world_size != 0:
        raise RuntimeError(f"global batch 8 must be divisible by world size {world_size}")
    selected = set(arguments.variants.split(","))
    variants = [item for item in VARIANTS if item[0] in selected]
    if not variants or variants[0][0] != "legacy-accum2":
        raise RuntimeError("the selected variants must include legacy-accum2 as the numerical baseline")
    module = load_model(arguments.model_source)
    architecture_value = architecture(arguments.tiny)
    baseline = None
    for variant in variants:
        result, state = run_variant(
            module, architecture_value, arguments.sequence_length, arguments.warmup_steps, arguments.steps, variant, baseline
        )
        if baseline is None:
            baseline = state
        if rank == 0:
            print(json.dumps(result, sort_keys=True), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
