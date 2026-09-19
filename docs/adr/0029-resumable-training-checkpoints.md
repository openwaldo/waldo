# ADR 0029: Resume interrupted runs from complete checkpoint bundles

Status: accepted
- Date: 2026-08-05

## Context

Weight-only checkpoints cannot continue AdamW correctly. Restarting from those
weights resets momentum, variance, schedule position, random state, and the
exact data position. Creating a new completed run after an interruption also
misstates what actually happened.

## Decision

Every resumable checkpoint is committed as one directory. It contains model
weights, backend optimizer state, runtime random state, and strict schema-1
state metadata pinning the run, architecture, backend revision, world size,
step, and consumed tokens. Files are synchronized before the directory is
atomically renamed. WALDO hashes and durably records every member only after
that commit and verifies all members again before backend handoff.

Repeating an exact `model train` invocation resumes the newest interrupted run
when its stage, corpus BOM, resolved parameters, evaluation set, backend, and
execution environment still match. The run ID and immutable `RUN-BOM.json`
do not change. `RUN.json` records each execution attempt and keeps verified
partial progress distinct from a terminal observation. The deterministic
record stream is positioned at the checkpoint boundary, then training continues
with restored state. Backends without a seekable prepared stream replay the
prefix without optimization. Multi-node node-local prepared streams skip the
already-trained prefix before worker handoff.

Any attempt that ends after WALDO has durably recorded a complete compatible
checkpoint is recoverable, regardless of whether the immediate cause was an
operator interruption, worker failure, transport failure, or terminal
bookkeeping failure. WALDO retains the failed attempt and error in history but
classifies the run as interrupted so the exact invocation can resume it. A
failure without a verified checkpoint remains terminal.

Checkpoint runtime state is backend-specific and can resume only under the
same pinned backend revision. Terminal model weights remain portable through
WALDO's shared Safetensors contract.

## Consequences

- Ctrl-C and other context interruptions retain useful, auditable work.
- Resume cannot silently reset optimizer or scheduler behavior.
- The real-backend lifecycle gate interrupts MLX immediately after a durable
  checkpoint and requires resumed terminal tensors to be bit-identical to an
  uninterrupted control run.
- Corrupt, incomplete, mismatched, or path-escaping checkpoint bundles fail
  before a trainer starts.
- ADR 0031 applies this same-run recovery contract to durable model-compose
  transactions without exposing a partial model at its published name.
