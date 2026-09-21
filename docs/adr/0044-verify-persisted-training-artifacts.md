# ADR 0044: Verify persisted training artifacts

Status: accepted

## Decision

PyTorch and TorchTitan keep FP32 master parameters in every resumable
checkpoint. Reduced `parameter_dtype` applies only to the portable terminal
artifact; otherwise a resumed run starts from rounded weights while retaining
optimizer state derived from the unrounded weights.

At step 1 and every configured evaluation boundary, the worker evaluates both
the live FP32-master model and the exact parameter representation that could be
published. Candidate selection uses `publishable_checkpoint_heldout_loss`,
also recorded as the authoritative `heldout_loss`. The worker fails immediately
when that value is non-finite or differs from the live value by more than the
greater of 0.02 loss or one percent. It recommends `parameter_dtype: float32`
when reduced-precision publication is not numerically safe.
Artifact-integrity failures are terminal even when an older checkpoint exists;
WALDO does not label them interrupted or repeatedly resume into the same
deterministic failure.

At completion, every real backend writes, reloads, and evaluates the selected
terminal `model.safetensors`. That result must agree with the selected
candidate within the same tolerance. WALDO rejects a completion that does not
name the globally best eligible persisted checkpoint or provide artifact
verification for it.

PyTorch evaluation records `live_compiled_heldout_loss`,
`publishable_checkpoint_heldout_loss`, and `artifact_loss_delta`. The selected
evaluation additionally records `artifact_heldout_loss`,
`artifact_heldout_perplexity`, and `serialization_loss_delta`. MLX records the
same terminal artifact evidence; its training and portable parameter
representation are already identical.

The current worker identities are `builtin-pytorch-worker-schema-1-r13`,
`builtin-torchtitan-worker-schema-1-r24`, and
`builtin-mlx-worker-schema-1-r14`.
