# ADR 0083: Infer from explicitly selected interrupted checkpoints

Status: accepted
- Date: 2026-10-29

## Context

Long training runs can have durable checkpoints worth evaluating before the
run completes. A checkpoint is not a completed model run: it must not become
`current_run_id`, and ordinary inference must continue to reject it. Existing
checkpoint bundles also lack the final configuration and tokenizer sidecars
used by local inference.

## Decision

Checkpoint inference requires both an exact run ID and checkpoint step. WALDO
accepts only interrupted, non-simulated runs; verifies every recorded artifact
size and SHA-256; binds checkpoint state to the model ID, run ID, step, token
position, architecture, backend, and worker revision; and verifies the weight
tensors against the immutable architecture. It never selects a latest
checkpoint implicitly.

Inference configuration and tokenizer sidecars are materialized temporarily
and deterministically from the immutable model and run metadata. They are not
added to the run, BOM, or compose transaction. New checkpoint state records
include `model_id`, and MLX, PyTorch, and TorchTitan worker revisions are
incremented so resume and inference fail closed across the state-contract
change.

## Consequences

- Ordinary chat remains limited to complete real runs.
- Interrupted checkpoints can be evaluated without changing lifecycle state.
- Checkpoints created before the model-ID state binding are not eligible for
  this inference path.
- JSON inference output identifies the interrupted run and exact checkpoint.
