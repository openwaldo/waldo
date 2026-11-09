# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

"""Distributed update conformance for WALDO's DDP accumulation contract.

Run with:
  torchrun --standalone --nproc-per-node=2 pytorch_distributed_conformance.py
"""

import contextlib
import copy
import io
import json
import math
import os

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel


ABSOLUTE_TOLERANCE = 2e-6
RELATIVE_TOLERANCE = 2e-5


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 8)
        self.projection = torch.nn.Linear(8, 17, bias=False)

    def forward(self, tokens):
        return self.projection(self.embedding(tokens))


def initialize(model):
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(0.03 * torch.sin(values * 0.17 + index))


def fixed_global_batch(step):
    rows = []
    for ordinal in range(8):
        generator = torch.Generator().manual_seed(91_003 + step * 101 + ordinal)
        rows.append(torch.randint(0, 17, (7,), generator=generator))
    return torch.stack(rows)


def local_microbatches(step, accumulation, rank, world_size):
    batch = fixed_global_batch(step)
    global_microbatch = 8 // accumulation
    local_size = global_microbatch // world_size
    result = []
    for microbatch in range(accumulation):
        start = microbatch * global_microbatch + rank * local_size
        result.append(batch[start : start + local_size])
    return result


def update(model, optimizer, step, accumulation, use_no_sync, consolidate_metrics):
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = torch.zeros((), dtype=torch.float64)
    local_tokens = torch.zeros((), dtype=torch.float64)
    seen = []
    for index, tokens in enumerate(local_microbatches(step, accumulation, rank, world_size)):
        seen.extend(tokens[:, 0].tolist())
        final = index + 1 == accumulation
        synchronization = model.no_sync() if use_no_sync and not final else contextlib.nullcontext()
        with synchronization:
            logits = model(tokens[:, :-1])
            loss_sum = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction="sum"
            )
            (loss_sum * world_size).backward()
        token_count = torch.tensor(tokens[:, 1:].numel(), dtype=torch.float64)
        if consolidate_metrics:
            local_loss_sum.add_(loss_sum.detach().double())
            local_tokens.add_(token_count)
        else:
            metrics = torch.stack((loss_sum.detach().double(), token_count))
            distributed.all_reduce(metrics)
            local_loss_sum.add_(metrics[0])
            local_tokens.add_(metrics[1])
    metrics = torch.stack((local_loss_sum, local_tokens))
    if consolidate_metrics:
        distributed.all_reduce(metrics)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    torch._foreach_div_(gradients, int(metrics[1].item()))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, foreach=True)
    if not torch.isfinite(norm):
        raise AssertionError("non-finite gradient norm")
    optimizer.step()
    return metrics, seen


def state(model):
    return {name: value.detach().cpu().clone() for name, value in model.module.state_dict().items()}


def maximum_error(left, right):
    return max(float((left[name] - right[name]).abs().max()) for name in left)


def assert_state_close(name, left, right):
    error = maximum_error(left, right)
    if any(not torch.allclose(left[key], right[key], atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE) for key in left):
        raise AssertionError(f"{name} differs by {error}")
    return error


def make_model_and_optimizer(initial):
    model = TinyLM()
    model.load_state_dict(initial)
    ddp = DistributedDataParallel(model)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=0.002, betas=(0.9, 0.95), weight_decay=0.1, foreach=True)
    return ddp, optimizer


def main():
    distributed.init_process_group("gloo")
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"distributed conformance requires 2 ranks, got {world_size}")
    torch.manual_seed(7)
    seed_model = TinyLM()
    initialize(seed_model)
    initial = copy.deepcopy(seed_model.state_dict())

    legacy, legacy_optimizer = make_model_and_optimizer(initial)
    legacy_metrics, legacy_seen = update(legacy, legacy_optimizer, 0, 2, False, False)
    legacy_state = state(legacy)

    optimized, optimized_optimizer = make_model_and_optimizer(initial)
    optimized_metrics, optimized_seen = update(optimized, optimized_optimizer, 0, 2, True, True)
    no_sync_error = assert_state_close("no_sync and consolidated metrics", legacy_state, state(optimized))
    if not torch.equal(legacy_metrics, optimized_metrics):
        raise AssertionError(f"metric consolidation changed values: {legacy_metrics} != {optimized_metrics}")

    one_microbatch, one_microbatch_optimizer = make_model_and_optimizer(initial)
    update(one_microbatch, one_microbatch_optimizer, 0, 1, True, True)
    accumulation_error = assert_state_close("accumulation 2 versus 1", legacy_state, state(one_microbatch))

    # The global batch must contain every sequence exactly once. Recover the
    # deterministic first token and compare across ranks with the source batch.
    gathered = [None for _ in range(world_size)]
    distributed.all_gather_object(gathered, optimized_seen)
    observed = sorted(token for rank_tokens in gathered for token in rank_tokens)
    expected = sorted(fixed_global_batch(0)[:, 0].tolist())
    if observed != expected:
        raise AssertionError(f"distributed input omitted or duplicated sequences: {observed} != {expected}")

    # A runtime checkpoint must resume to the exact same next update and logits.
    payload = io.BytesIO()
    torch.save({"model": optimized.module.state_dict(), "optimizer": optimized_optimizer.state_dict()}, payload)
    update(optimized, optimized_optimizer, 1, 2, True, True)
    uninterrupted_state = state(optimized)
    test_tokens = fixed_global_batch(2)[:1, :-1]
    uninterrupted_logits = optimized.module(test_tokens).detach()
    payload.seek(0)
    checkpoint = torch.load(payload, map_location="cpu", weights_only=True)
    resumed, resumed_optimizer = make_model_and_optimizer(initial)
    resumed.module.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    update(resumed, resumed_optimizer, 1, 2, True, True)
    resume_error = assert_state_close("checkpoint resume", uninterrupted_state, state(resumed))
    resumed_logits = resumed.module(test_tokens).detach()
    if not torch.equal(uninterrupted_logits, resumed_logits):
        raise AssertionError("checkpoint save/resume logits differ")

    # Compare DDP's result with a single-process reference over the same global batch.
    reference = TinyLM()
    reference.load_state_dict(initial)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.002, betas=(0.9, 0.95), weight_decay=0.1, foreach=True)
    reference_optimizer.zero_grad(set_to_none=True)
    tokens = fixed_global_batch(0)
    reference_loss = torch.nn.functional.cross_entropy(
        reference(tokens[:, :-1]).reshape(-1, 17), tokens[:, 1:].reshape(-1), reduction="mean"
    )
    reference_loss.backward()
    reference_optimizer.step()
    reference_state = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    reference_error = assert_state_close("single-rank versus distributed", reference_state, legacy_state)

    if rank == 0:
        print(json.dumps({
            "status": "pass",
            "world_size": world_size,
            "tolerances": {"absolute": ABSOLUTE_TOLERANCE, "relative": RELATIVE_TOLERANCE},
            "maximum_weight_errors": {
                "single_rank_vs_distributed": reference_error,
                "accumulation_sync_vs_no_sync": no_sync_error,
                "accumulation_2_vs_1": accumulation_error,
                "checkpoint_resume": resume_error,
            },
            "sequence_coverage": "exact",
            "checkpoint_reload_logits": "exact",
        }, indent=2, sort_keys=True))
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
