# ADR 0073: Pin training precision, recomputation, and compilation

Status: accepted

## Context

Mixed precision, activation recomputation, and graph compilation materially
change memory use, throughput, numerical behavior, and checkpoint state. They
cannot remain unrecorded environment switches in a reproducible comparison.
CUDA FP16 without loss scaling can also spend an entire run applying
underflowed gradients.

## Decision

Training parameters add `compute_precision`, `activation_checkpointing`, and
`compile`, all resolved and pinned in the run BOM. `compute_precision` defaults
to `auto`, which follows the architecture's portable parameter dtype.

PyTorch and TorchTitan keep FP32 master parameters and optimizer state, use
autocast for the selected compute dtype, and use dynamic gradient scaling for
CUDA FP16. The scaler and skipped-step count are part of every checkpoint.
Layer activation checkpointing uses non-reentrant recomputation. Compilation
wraps only the live forward path so portable state dictionaries retain stable
names. MLX rejects unsupported recomputation and compile requests instead of
silently ignoring them.

## Consequences

- BF16 remains the reference path and requires no scaler.
- FP16 overflow skips are visible in telemetry and survive resume.
- Resource forecasts reduce their activation estimate when recomputation is
  selected, but the estimate remains conservative until measured on G1 GPUs.
- Each optimization must pass an isolated throughput, learning-curve, and
  checkpoint-resume comparison before use in a promoted run.
