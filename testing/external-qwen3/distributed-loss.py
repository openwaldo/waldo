# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Four-rank CPU/Gloo loss and gradient regression; run with torchrun.

Uses the real pinned Transformers count helper, WALDO loss, and DDP reducer.
TRAINER_SOURCE optionally selects a checkout for before/after qualification.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, os.environ.get('TRAINER_SOURCE', str(
    Path(__file__).resolve().parents[2] / 'examples/trainers/qwen3-hf')))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import Trainer, TrainingArguments
from qwen3_hf.train_cli import WaldoTrainer


class Accelerator:
    parallelism_config = None

    def gather(self, value):
        values = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(values, value)
        return torch.stack(values)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.1]))

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.logits.expand(1, 5, 2))


def labels_for(rank, micro):
    count = 1 + (rank + micro) % 4
    return torch.tensor([[-100] + [rank % 2] * count + [-100] * (4 - count)])


def check(average, accumulation, fallback=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    holder = SimpleNamespace(
        args=SimpleNamespace(average_tokens_across_devices=average,
                             world_size=world, n_gpu=1, label_smoothing_factor=0.0),
        model_accepts_loss_kwargs=True, compute_loss_func=None,
        accelerator=Accelerator(), history={'consumption': {}}, step_loss=0.0)
    batches = [{'labels': labels_for(rank, micro)} for micro in range(accumulation)]
    denominator = None if fallback else Trainer._get_num_items_in_batch(
        holder, batches, torch.device('cpu'))
    model = DistributedDataParallel(Model())
    for batch in batches:
        loss = WaldoTrainer.compute_loss(holder, model,
            {'input_ids': torch.zeros(1, 5, dtype=torch.long),
             'labels': batch['labels'], 'consumption': {}},
            num_items_in_batch=denominator)
        loss.backward()
    targets = torch.cat([labels_for(r, m).flatten() for r in range(world)
                         for m in range(accumulation)])
    targets = targets[targets != -100]
    reference = torch.tensor([0.2, -0.1], requires_grad=True)
    expected_loss = torch.nn.functional.cross_entropy(reference.expand(len(targets), 2), targets)
    expected_loss.backward()
    torch.testing.assert_close(torch.tensor(holder.step_loss), expected_loss.detach(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model.module.logits.grad, reference.grad, atol=1e-6, rtol=1e-6)
    if rank == 0:
        print(f'PASS average={average} accumulation={accumulation} fallback={fallback}: '
              f'loss={holder.step_loss:.8f}; DDP gradient matches global reference', flush=True)


def main():
    dist.init_process_group('gloo')
    try:
        assert dist.get_world_size() == 4, 'requires four ranks'
        default = TrainingArguments.__dataclass_fields__['average_tokens_across_devices'].default
        assert default is True
        for average, accumulation, fallback in [(default, 1, False), (default, 2, False),
                                                (False, 1, False), (False, 2, False),
                                                (True, 1, True)]:
            check(average, accumulation, fallback)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
