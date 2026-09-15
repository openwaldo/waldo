# ADR 0044: Verify persisted training artifacts

Status: accepted

## Decision

Before a PyTorch run completes, the worker reloads `model.safetensors` through
the portable inference representation and evaluates the pinned held-out set.
The run fails when the artifact result is non-finite. The persisted artifact's
loss is the authoritative final metric because that artifact, rather than the
FP32 master weights behind a compiled training graph, is what inference and
distribution consume. The live compiled loss is retained separately. A delta
larger than the greater of 0.02 loss or one percent emits a warning rather than
discarding an otherwise valid, fully trained artifact.

The final evaluation records `live_compiled_heldout_loss`,
`artifact_heldout_loss`, `artifact_heldout_perplexity`, and
`artifact_loss_delta`. WALDO rejects a successful PyTorch observation without
artifact evidence when held-out evaluation was configured. A final checkpoint
from an older worker that failed only on the former delta threshold is
recoverable without replaying its training corpus.

The current worker identities are `builtin-pytorch-worker-schema-1-r11` and
`builtin-torchtitan-worker-schema-1-r22`.
