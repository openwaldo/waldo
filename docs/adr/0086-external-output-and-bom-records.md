# ADR 0086: Verify external publications and persist external BOM records

Status: accepted

Amended by [ADR 0089](0089-portable-external-plan-identity.md): portable plan
identity is separate from the exact execution config required for resume.

## Decision

Implement ADR 0084's provenance boundary with separate schema-2 external
plan, run and model BOM records. Native schema-1 types and hash encodings do
not change. Opaque external architectures never pass through the native
architecture decoder. The portable `identity` object is repeated unchanged in
PLAN.json, RUN-BOM.json, MODEL-BOM.json and exported BOM.json. Execution values,
launch allowlists and host observations appear only in the run BOM.

TRAINER-SCHEMAS.json preserves exact probe bytes and is published together
with a new plan directory. Run BOM writes refuse replacement. Aggregate BOMs
are written atomically after checking referenced run files and published files.
These are developer persistence APIs; CLI integration remains planned.

Trainer completion verification runs after successful process exit and input
reconciliation. Evaluated candidates must have authenticated, persisted
checkpoint bundles. Completion must preserve WALDO's observed evaluations;
selection cannot hide an earlier better candidate. A caller resuming a run
supplies the independently persisted prior checkpoint references and evaluation
history. An `artifact_evaluation` reports the reloaded published weights on the
same held-out set and selected step. Its counts and NLL arithmetic are checked,
and its loss must agree with selection within max(0.02, 1%). This is a trainer
attestation, not an independent WALDO model evaluation.

The PoC publication allowlist is Safetensors weights and JSON configuration
and tokenizer assets. Unknown roles/formats fail closed, including pickle and
framework archives. Checkpoint pickle remains private resume state. Rooted
opens prevent escaping through symlinks. Hashes, sizes, tensor layouts and all
supported floating tensor values are checked without loading whole tensors.
Non-finite tensors use the numerical-integrity error class; other output
integrity failures use artifact-integrity. Both are non-retryable.

EU-BOM.json stays the existing corpus/regulatory projection. Training-tool
identity is authoritative in the model and run BOMs, rather than duplicated
into regulatory corpus categories. No new compliance claim is implied.
