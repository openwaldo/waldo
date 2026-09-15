# ADR 0076: Make competitive training recipes explicit

Status: accepted

## Context

WALDO previously fixed AdamW, cosine decay, ordinary normal initialization,
tied embeddings, and unnormalized queries and keys in backend code. That made
controlled capability-per-FLOP comparisons impossible and encouraged copying
large token budgets before testing the recipe.

## Decision

The portable architecture now declares optional QK normalization and either
normal or depth-scaled residual initialization. Training parameters declare
AdamW or combined Muon/AdamW and cosine or warmup-stable-warmdown scheduling,
including the warmdown duration and final learning-rate ratio.

Defaults preserve existing model behavior. Muon uses orthogonalized momentum
updates for hidden matrices and AdamW for embeddings, output heads, and vector
parameters. The current implementation is limited to PyTorch data parallelism;
MLX and sharded TorchTitan placements reject it explicitly rather than silently
falling back to AdamW.

The `conversation2` candidate uses untied embeddings, QK normalization,
depth-scaled initialization, zero dropout, and a warmup-stable-warmdown AdamW
baseline. Muon remains a measured G1/G2 experiment until it wins at equal
FLOPs.

## Consequences

- Recipe changes alter immutable architecture or run identities.
- Checkpoints persist the selected optimizer and schedule state.
- The final optimizer is selected from small runs, not assumed from nanochat.
- The 758M candidate uses roughly 6.1B planned training tokens instead of the
  previous 21.4B-token curriculum.
