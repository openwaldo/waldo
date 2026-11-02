# External trainer checkpoints and conformance

The `trainerapp` developer API implements ADRs 0029 and 0084 for external
checkpoints. CLI/lifecycle wiring, durable run BOM creation and reference-trainer
CPU/GPU qualification remain planned. Native worker checkpoints are unchanged.

## Lifecycle integration

Before `Client.Train`, supply `Begin.CheckpointIdentity` from the immutable run
BOM: its SHA-256 and trainer, tokenizer and selected environment identity hashes.
The run BOM must pin the stage, resolved corpus BOMs, parameters, evaluation set
and execution evidence. Supply an absolute `Begin.CheckpointDirectory`, reserved
for WALDO's durable bundles, separate from the trainer's output directories.
These two Go fields are not sent to the trainer.

Every checkpoint event is reconciled against WALDO's cumulative per-corpus
supervised-target counts at that exact optimizer step. A temporary disk ledger
retains prefixes even when input streaming runs ahead of output; memory depends
on corpus count, not step count. Trainers must report immutable checkpoint
files: later steps must use different filenames, not overwrite earlier files.

WALDO opens files relative to the checkpoint root with traversal containment,
requires regular files, checks sizes and SHA-256 hashes, and copies the verified
bytes into a private temporary bundle. It writes `MANIFEST.json`, syncs files
and directories, and atomically renames the bundle. Only then does the event
sink receive `Event.DurableCheckpoint`, containing a relative manifest path and
SHA-256. The sink must persist this reference independently in lifecycle state
before considering a checkpoint accepted. A sink error stops the process; a
bundle without a persisted reference is not an accepted resume point.

## Manifest schema 1

`CheckpointManifest` is the authoritative Go type. Its JSON fields are:

- `kind: waldo-trainer-checkpoint`, `schema: 1`;
- `identity`: run BOM, trainer, tokenizer and environment SHA-256 hashes;
- `begin_sha256`: hash of the normalized begin request excluding paths/resume;
- `parameters_sha256`: complete resolved parameter hash;
- `execution`: schema-role-split resolved execution parameters;
- `world`: complete launch world shape;
- `step`, `tokens`, and WALDO-derived `consumption` keyed by logical corpus;
- `files`: role, relative path, SHA-256, byte size and format for every member.

The manifest hash is pinned outside the bundle. Neither a trainer-provided hash
nor a digest stored inside a manifest authenticates that manifest. Bundles are
named by step and manifest digest. Repeated identical events reuse a bundle
only after verification. Unknown fields, missing fields, unsupported schemas,
invalid consumption and incompatible pins fail closed.

## Resume

Call `LoadCheckpoint(begin, reference)` using the saved lifecycle reference.
It verifies the manifest hash, exact begin/run identity and world, and every
file. Set the returned value as `Begin.Resume`; arbitrary resume JSON fails.
`Train` repeats verification before launching and checks the execution split.
The protocol resume object contains `step`, `tokens`, `checkpoint_manifest` and
`paths` (manifest-relative filenames mapped to verified absolute bundle paths).
The trainer restores weights, optimizer, scheduler and RNG state from those
files. Private checkpoint pickle is allowed; use `weights_only=True` where the
framework supports it. Published artifacts must never contain pickle.

The source must start at `resume.step * global_batch_sequences`. A final-step
resume supplies no training sequences and still completes artifact/evaluation
publication. Historical per-corpus counts are merged with this attempt's sent
counts and checked against completion, including zero remaining steps.

## Reusable conformance harness

`CheckConformance(ctx, factory, absoluteTolerance, relativeTolerance)` accepts a
factory returning isolated clients/begins, evaluation records and a `SourceAt`
function taking a sequence ordinal. Each run needs at least two steps and
`determinism: required`. The factory's alternate flag selects another valid
micro-batch/accumulation decomposition with the same global batch and recipe.

The harness runs an uninterrupted control, interrupts another process on its
first durable intermediate checkpoint, verifies and resumes it, and compares
verified terminal weight files byte-for-byte. This is stricter than tensor
comparison because metadata bytes must also match. It then compares progress
losses at matching steps across the two decompositions, using
`abs(a-b) <= absoluteTolerance + relativeTolerance * max(abs(a), abs(b))`.
The returned report records bit identity, interruption step, comparison count,
maximum observed delta and both tolerances. Errors fail qualification.

Use unequal supervised-target counts across micro-batches to detect incorrect
averaging of micro-batch means. The standard-library fake trainer exercises
multiple corpora, varying masks, optimizer momentum, schedule position and RNG
restoration, and writes a real scalar Safetensors file. Negative controls reset
optimizer state or average micro-batch losses incorrectly; both must fail.

Run the fake harness with `go test -v ./internal/model/trainerapp -run Conformance`.
The reference `examples/trainers/qwen3-hf` is not present yet. No reference CPU,
CUDA, or DGX Spark numerical conformance claim is made. Its future factory must
select the pinned environment and enforce deterministic algorithms in trainer
code. WALDO's determinism string alone is not enforcement.

The launch allowlist passes `CUBLAS_WORKSPACE_CONFIG` when explicitly set (for
example `:4096:8`) and records it in launch evidence. It does not inherit
arbitrary environment variables or set deterministic framework algorithms.
