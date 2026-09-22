# ADR 0044: Verify persisted training artifacts

Status: accepted

## Decision

PyTorch and TorchTitan keep FP32 master parameters in every resumable
checkpoint. Reduced `parameter_dtype` applies only to the portable terminal
artifact; otherwise a resumed run starts from rounded weights while retaining
optimizer state derived from the unrounded weights.

At step 1, every configured evaluation boundary, and the terminal step, the
worker separately evaluates the compiled model under
the requested compute precision, the eager model under that same precision,
the eager FP32 master weights, and the exact reloaded parameter representation
under the FP32 semantics used by chat inference. This distinguishes compiled
execution drift, compute-precision drift, checkpoint serialization, and dtype
conversion. Candidate selection uses
`publishable_checkpoint_heldout_loss`, also recorded as the authoritative
`heldout_loss`. The worker fails immediately when any value is non-finite or
when compiled/eager-compute, eager-compute/eager-FP32, or
eager-FP32/publishable loss differs by more than the greater of 0.02 loss or
one percent. Non-finite training loss and every optimizer-step gradient norm also fail
closed before another optimizer update. Numerical- and artifact-integrity
failures are terminal even when an older checkpoint exists.
WALDO does not label them interrupted or repeatedly resume into the same
deterministic failure.

When compiled execution is enabled, additional safety checks at steps 100 and
1000 catch graph drift before a long interval consumes substantial compute.
They are omitted when compilation is disabled so the four-way full held-out
comparison does not add unnecessary accelerator work to reference runs.

At completion, every real backend writes, reloads, and evaluates the selected
terminal `model.safetensors`. That result must agree with the selected
candidate within the same tolerance. WALDO rejects a completion that does not
name the globally best eligible persisted checkpoint or provide artifact
verification for it.

PyTorch evaluation records `live_compiled_heldout_loss`,
`live_eager_compute_heldout_loss`, `live_eager_heldout_loss`,
`compile_loss_delta`, `compute_precision_loss_delta`,
`publishable_checkpoint_heldout_loss`, and `artifact_loss_delta`. The selected
evaluation additionally records `artifact_heldout_loss`,
`artifact_heldout_perplexity`, and `serialization_loss_delta`. MLX records the
same terminal artifact evidence; its training and portable parameter
representation are already identical.

The PyTorch/TorchTitan trainer, artifact evaluator, and PyTorch chat worker use
the same embedded model definition. This is required because architectural
features such as QK normalization can change logits without changing tensor
names or shapes. Runtime configuration is checked against the immutable model
architecture before inference.

The current worker identities are `builtin-pytorch-worker-schema-1-r15`,
`builtin-torchtitan-worker-schema-1-r26`, and
`builtin-mlx-worker-schema-1-r15`.
