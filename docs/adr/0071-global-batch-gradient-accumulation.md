# ADR 0071: Keep global optimizer batches stable with gradient accumulation

Status: accepted

## Context

The compose's `batch_size` is a global optimizer batch. The workers previously
required that complete batch to fit in one forward/backward pass, coupling the
learning recipe to accelerator memory and making large logical batches
impractical.

## Decision

Schema-1 composes add optional `gradient_accumulation_steps`, defaulting to 1.
It divides `batch_size` into equal global micro-batches. Each micro-batch must
divide evenly across the selected world size, so every rank processes the same
number of unique sequences.

Gradients accumulate over the micro-batches and are normalized by the exact
number of supervised token targets before one optimizer and scheduler step.
Training steps, token capacity, checkpoints, evaluation cadence, and resume
positions continue to count optimizer steps. The resolved accumulation count
is pinned in the run BOM.

## Consequences

The same logical batch can run with smaller physical batches without changing
the declared learning horizon. Forecasts use per-rank micro-batch memory.
Invalid batch, accumulation, and world-size combinations fail before training.
Checkpoint formats and worker revisions change when accumulation is enabled.
