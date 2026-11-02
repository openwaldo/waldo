# ADR 0084: External trainers

Status: accepted
- Date: 2026-11-02

This ADR records the design of external trainers. Until the implementation
lands and the maintained contracts are updated, the behavior described here is
planned, not available.

## Context

Today the trainer is an interchangeable executor of a pinned recipe. ADR 0017
resolves the backend from host policy outside the compose. ADR 0011 makes the
immutable architecture plan the model identity, and the run BOM pins the
backend only as `backend {name, revision}`, with a hand-bumped revision label,
a free-text runtime, the host, and accelerators. Nothing records a worker
source hash, a WALDO binary digest, or a dependency lock. That is sound only
while WALDO writes every trainer itself and treats the built-in worker as a
constant.

Users want to bring their own training program: a different model family, a
different framework, or their own optimizer and parallelism. Once the trainer
is user supplied it is no longer a constant. Two runs with the same compose
and different trainer commits produce different models, so the trainer, its
tokenizer, and its environment become part of what the model *is*.

WALDO makes no byte-for-byte training-output claim today. The maintained docs
make byte-exact claims only for corpus objects, recipes, and export tensor
copies. ADR 0029's bit-identical resume gate is a conformance test for the
built-in MLX worker, not a published guarantee. External trainers keep that
boundary.

Held-out loss is per-token cross-entropy on a deterministic, pinned sample of
the training corpora excluded from training (ADR 0030, ADR 0080). It is used
for checkpoint selection (ADR 0079) and telemetry, not promotion (ADR 0075).

## Decision

A model compose may name an external trainer and an external tokenizer. Each
is a program pinned by git commit and run in a pinned environment. When
present, the trainer replaces the built-in worker and becomes part of the
model's identity and BOM. A compose without a `trainer` block behaves exactly
as today.

### Model identity

The plan hash covers:

- trainer source: repository, full commit SHA, git tree hash, path, and
  manifest SHA-256;
- tokenizer source, with the same shape;
- each selected named environment: lock file hash and interpreter version;
- the SHA-256 of each schema document the trainer and tokenizer report;
- the resolved recipe parameters of every stage and the resolved tokenizer
  configuration;
- the architecture; and
- the corpus BOMs.

WALDO runs the trainer's `resolve` command for every stage before it computes
the plan hash, so the hash covers the configuration the trainer will actually
use. Resolved execution parameters are excluded, as described in
[Parameter roles](#parameter-roles).

Host facts (GPU, driver, CUDA, kernel, launch profile, launcher, world shape)
and resolved execution parameters go in the run BOM only. They scope
reproducibility claims but do not change the model's identity.

Rationale: a model is what produced it. A commit SHA alone is not enough
because the path inside the repository and the manifest select what actually
runs, and the tree hash makes the checkout verifiable without trusting the
remote. Host facts change how fast and how exactly a run executes, not what
was asked for, so putting them in the plan would fork identity on every
hardware change.

### Data ownership

WALDO owns selection, ordering, the held-out split, corpus balancing, token
budgets resolved to exact steps, packing, loss masks, and the resume position.
Trainers never read the index or Parquet. Trainers receive token IDs only;
there is no text mode.

Rationale: the corpus BOM, consumption accounting, held-out pins, and
contamination gates are WALDO's guarantees (ADR 0002, ADR 0018, ADR 0030,
ADR 0045). They hold for any trainer only if no trainer can change what data
it sees. Token IDs give WALDO one input representation to verify.

### Tokenizer application

A tokenizer has its own manifest, `waldo-tokenizer.yaml`, declaring `kind`,
`schema`, `name`, `protocol`, `entrypoint`, `commands` (`probe` and `serve`),
and named environments. WALDO drives it as a
long-lived NDJSON codec, tokenizer protocol version 1, for all tokenization:
training records, conversation role segments, and held-out records. `probe`
reports the vocabulary size, special token IDs, and the tokenizer's
configuration schema, handled as described in
[Parameter schemas](#parameter-schemas). WALDO still renders
conversations with the ADR 0060 templates and derives loss masks from roles;
the tokenizer only encodes the rendered segments. A tokenizer may share a
repository and commit with the trainer.

Rationale: tokenization is part of data preparation, which WALDO owns, but
the vocabulary belongs to the model family. A separate process lets WALDO keep
tokenization in its pipeline, and its counts in its pins, while the model
author supplies the codec. Because it is independent of the trainer process,
WALDO can tokenize during preflight, before any training process starts.

### Trainer manifest

The trainer manifest is `waldo-trainer.yaml` at the trainer path. It declares
`kind`, `schema`, `name`, `protocol`, `entrypoint` (`module` or `script`;
`executable` is reserved for later), `commands` (`probe`, `resolve`,
`train`), `launch_profiles`, `objectives`, `artifact_profile`, optional
`schemas` (paths to committed `architecture` and `parameters` schema files),
and named `environments`. Environments of `kind: python-lock` are supported.

Rationale: WALDO must know how to launch the trainer, which profiles and
objectives it supports, and how to interpret its output before running any
trainer code. A declarative manifest at a pinned commit makes those facts part
of the hashed source.

### Parameter schemas

The trainer reports its architecture and parameter schemas, as JSON Schema
2020-12 documents, in its `probe` response. The tokenizer reports its
configuration schema the same way. This response is the authoritative source.
A trainer may generate the documents from code, for example from Python
classes.

WALDO records every received trainer and tokenizer schema document byte for
byte, unfiltered, in `TRAINER-SCHEMAS.json` next to `PLAN.json`, and puts each document's SHA-256 in
the plan hash. When the manifest names a committed schema file, the file must
be byte-identical to the reported document, or the run fails. WALDO validates
the compose, and later every `resolve` result, against the full documents.

Rationale: the schema is a deterministic function of the trainer source and
its locked environment, which identity already covers, so a schema generated
at run time gives the same guarantee as a committed file. Recording the
received documents keeps the plan auditable without re-running the trainer
and makes changes caused by a generator dependency visible. A committed file
remains useful for offline validation and review, and the byte-equality check
keeps it from drifting.

### Parameter roles

Each property in the parameter schema has a role. A property annotated
`"x-waldo-role": "execution"` is an execution parameter. Every other property
is a recipe parameter. A property is an execution parameter only if two runs
that differ only in that property perform the same sequence of optimizer
updates in exact arithmetic, given the same data in the same order. Batch
decomposition (micro-batch size and accumulation steps), activation
recomputation, compilation, attention kernel selection, gradient communication
settings, and logging qualify. Learning rate, schedule, optimizer settings,
gradient clipping, numeric precision, loss shaping, and initialization seed
do not.

- WALDO reads roles from the received schema, never from `resolve` output.
- The only allowed role value is `execution`. Any other value fails
  validation.
- A role on an object applies to its whole subtree. WALDO follows `$ref`
  before reading roles.
- The architecture schema must not contain `x-waldo-role`. Architecture is
  always identity.

`resolve` receives the world shape (world size, processes per node, and the
device facts from `probe`) and returns every parameter. WALDO validates each
result against the full schema and splits it by role. Recipe values enter the
plan hash. Execution values go to the run BOM. WALDO calls `resolve` twice,
for the actual world shape and for a single process, and fails if the recipe
values differ. A precision or other recipe value the host cannot honor makes
`resolve` fail rather than fall back.

Trainers must normalize loss by the exact number of supervised target tokens
across the whole global batch, as ADR 0071 requires of the built-in workers,
so that batch decomposition does not change the gradient. The conformance
test runs the same steps under two different decompositions and compares the
losses within a tolerance.

Rationale: the global batch is fixed by the compose, but its decomposition
depends on the number of GPUs and their memory. Without roles, the same
compose run on one GPU and on eight would produce two model identities for
the same sequence of updates. The default role is recipe, so an unannotated
property makes identity too strict, never too loose. Roles live in the hashed
schema, so changing a role is itself an identity change, and the two-shape
`resolve` check catches a trainer that derives a recipe value, such as a
learning rate, from the host.

### Process launch

For every command, WALDO sets `cwd` to the read-only trainer or tokenizer
checkout, starts from an empty environment plus a recorded allowlist
(`env -i`), and runs `$ENV/bin/python -E -s -m <module> <command>` or the
script path. Stdout carries protocol frames only. Stderr is captured.

Two launch profiles exist. A trainer lists the ones it supports in its
manifest, and WALDO refuses a profile the trainer does not declare.

- `single-process` runs one trainer process.
- `distributed` runs the trainer under torchrun
  (`$ENV/bin/python -E -s -m torch.distributed.run` followed by the trainer
  entrypoint), with one launch per node and one process per local GPU. A
  single node uses `--standalone --nproc-per-node=N`. Multiple nodes add
  `--nnodes`, `--node-rank`, `--master-addr`, `--master-port`, and
  `--max-restarts=0`. The allowlist adds only the rank and rendezvous
  variables and the NCCL settings WALDO already configures.

Distributed runs reuse WALDO's existing multi-node machinery rather than a
second one: `--hostfile` and Fuzzball host selection (ADR 0025, ADR 0070),
remote launch from the primary, and the same preflight checks and fail-fast
errors. Before rendezvous, every node verifies the same trainer checkout
(commit and tree hash) and environment (lock hash). A mismatch fails the run.
Data flows as described in [Distributed data plane](#distributed-data-plane).

Rationale: the built-in workers inherit the full WALDO environment and a
candidate-searched host Python (ADR 0020, ADR 0024). For a hashed external
trainer that would let unrecorded state (user site-packages, `PYTHONPATH`,
stray variables) change the run. A pinned interpreter in a verified
environment, isolated flags, and an allowlist make the recorded inputs the
whole input. Reusing the multi-node launcher gives external trainers the same
cluster contract as the built-in TorchTitan path.

### Distributed data plane

Distributed external training uses the `node-local-cache` data plane of
ADR 0072 without change. Every node runs its own WALDO coordinator. Each
coordinator materializes the corpus in its node-local verified cache, drives
its own tokenizer application, builds the identical canonical stream, and
sends its node's local rank 0 only the sequences owned by that node's ranks,
with explicit global micro-batch boundaries. Sequence ordinals assign each
sequence to exactly one global rank. The trainer's local rank 0 distributes
the stream to the other ranks on the same node. Training data never crosses
the inter-node network. The `rank-zero-broadcast` data plane remains a
built-in compatibility mode and is not offered to external trainers.

Rationale: ADR 0072 removed rank zero as a decoding and broadcast bottleneck
and kept corpus traffic off the collective network. External trainers should
not reintroduce either. Keeping data delivery in WALDO also preserves
[Data ownership](#data-ownership) across nodes.

### Trainer protocol

External trainers speak trainer protocol version 1 over stdin and stdout
NDJSON. The built-in workers' schema-1 protocol (ADR 0018) stays internal and
is never declared by an external trainer.

- `probe` reports versions, devices, capabilities, and the schema documents
  of [Parameter schemas](#parameter-schemas).
- `resolve` receives a stage's `trainer` block, the WALDO-owned data
  parameters, and the world shape, and returns the fully resolved parameters.
- `train` runs one stage attempt:
  - `begin` carries paths (artifacts, checkpoints, scratch), WALDO-owned data
    parameters, the opaque resolved trainer parameters and architecture with
    their hashes, tokenizer facts, the evaluation set, initialization, resume
    state with its checkpoint manifest, the determinism mode, and world facts
    (world size, node count, processes per node, node rank). The trainer
    receives both recipe and execution parameters.
  - WALDO then sends evaluation records and training sequences as token IDs and
    loss masks, followed by an end frame.
  - The trainer emits `event` frames (`progress`, `log`, `checkpoint`, and
    `evaluation` with held-out NLL sums and counts), then one `complete` frame
    (artifacts with roles and the artifact profile, consumption per corpus,
    evaluations, the selected checkpoint, and environment evidence) or one
    `error` frame with an error class.

Under the `distributed` profile, every node's local rank 0 receives `begin`
and that node's data frames. Only global rank 0 emits `event`, `complete`,
and `error` frames and writes checkpoints and artifacts into the WALDO paths.
Secondary coordinators report only their process exit.

Rationale: the internal protocol carries built-in assumptions (text frames for
the byte tokenizer, paths in argv, worker-owned `state.json`) and changes with
worker revisions. A separately versioned public protocol can be stable for
third parties while the built-in one keeps evolving. NLL sums and counts,
rather than only a loss value, let WALDO check the reported metrics.

### Compose schema 2

Schema-2 composes add top-level `trainer` and `tokenizer` blocks. Both carry
`source: {repository, commit, path}` and `environment`; the trainer adds
`determinism` and the tokenizer adds `configuration`. `architecture` is opaque
to WALDO and validated by the trainer's architecture schema. Each stage has a
`data` block, owned by WALDO (budget, sequence length,
`global_batch_sequences`, seed, shuffle, evaluation selection, checkpoint and
evaluation cadence), and a `trainer` block, opaque to WALDO, validated by the
trainer's parameter schema and completed by `resolve`. Execution parameters
are normally omitted from the stage `trainer` block. A value written there is
validated against the world shape but still stays out of identity. WALDO
validates the
batch topology, and the trainer must verify that micro-batch size times
accumulation steps times world size equals the global batch. YAML anchors and
aliases are allowed; `yaml.v3` expands them before decoding.

The `determinism` value is recorded in the plan and passed to the trainer in
`begin`. WALDO does not yet enforce it.

A compose without `trainer` and `tokenizer` blocks uses the built-in trainer
and tokenizer transparently.

Rationale: the split between `data` and `trainer` is the ownership line from
[Data ownership](#data-ownership) written into the file format.
`global_batch_sequences` stays on the WALDO side because it fixes optimizer
steps and therefore consumption (ADR 0071). The trainer decides how to realize
it on its hardware.

### Resume

Resume is required. WALDO writes and validates a checkpoint manifest recording
step, tokens, consumption, identity hashes, and file roles, and streams data
from the recorded resume position. The manifest also records the world shape
and the resolved execution parameters, and WALDO refuses to resume under a
different world shape. This applies to both launch profiles: an
interrupted distributed run resumes on every node from the same verified
checkpoint, and each node's coordinator starts at its sequence boundary. A
conformance test compares an interrupted and resumed run with an
uninterrupted one.

Rationale: ADR 0029's guarantees (no silent optimizer reset, auditable
interruption, same run ID) must hold for any trainer. Moving the manifest from
the worker's `state.json` to WALDO means WALDO, not trainer code, decides
whether a checkpoint is compatible.

### Pickle

Pickle is allowed only in checkpoint files, which are private, hash-verified
resume state. It is never allowed in published or exported artifacts.
Trainers should load checkpoints with `weights_only=True`.

Rationale: optimizer and RNG state in common frameworks is pickled, and
forbidding it would make resume impractical. Checkpoints are written and read
by the same pinned trainer under hash verification. Published artifacts are
loaded by other people and tools, so they stay Safetensors.

### Output verification

WALDO checks that reported paths are under the artifacts directory, hashes and
sizes, monotonic counters, finite metrics, consumption against tokens sent,
held-out target-token and byte counts against the pins, loss and bits-per-byte
against the reported NLL sums, Safetensors headers, NaN and Inf in tensors,
and the absence of pickle in published files. Beyond those checks the
trainer's held-out metrics are trusted. There is no separate `evaluate`
command. ADR 0075 promotion gates are unchanged.

Rationale: WALDO can check everything it can derive from its own inputs
without re-running the model. A WALDO-side evaluator would need the trainer's
model code, which is exactly what is being made external. Promotion already
uses independent gates, so trusting the held-out metric affects checkpoint
selection only.

### Objectives

Objectives stay `causal-language-modeling` and
`assistant-response-modeling`. The objective controls which record kinds a
stage accepts, how records are rendered, and which tokens carry loss. Packing
is the same for both. A trainer declares which objectives it supports; it
cannot add one.

Rationale: objectives are data semantics, which WALDO owns.

### Reference trainer

The reference trainer lives in the WALDO repository at
`examples/trainers/qwen3-hf`. It trains Qwen3 from scratch with Hugging Face
Transformers and supports both launch profiles. Trainer and tokenizer share
one directory, and the tokenizer files are committed. Its schemas are defined
in Python and reported through `probe`, and the generated schema files are
also committed so they can be reviewed and checked for drift.

Rationale: one maintained trainer exercises the whole contract, serves as the
conformance target, and shows third parties a complete working layout.

## Amendments to existing ADRs

These apply only when a compose has a `trainer` block. Every built-in contract
below still holds unchanged for composes without one.

- **ADR 0011** (immutable model plans). For external trainers the plan hash
  covers the facts in [Model identity](#model-identity), not only the
  architecture plan. Adding runs to an existing model still does not change
  its identity. The run BOM and model BOM additionally record the trainer,
  tokenizer, environment, and host facts.
- **ADR 0017** (backends resolved outside the compose). For external trainers
  the compose names the trainer, and the trainer is identified by source and
  environment digests, not a revision string. Host policy no longer chooses
  the trainer. It still describes the host, and host facts still go to the run
  BOM. The built-in trainer remains resolved from host policy with its
  revision label.
- **ADR 0018** (one worker protocol). External trainers use trainer protocol
  version 1, not the internal schema-1 worker protocol. Resolution of stage
  parameters moves to the trainer's `resolve` command for the opaque
  `trainer` block. WALDO still resolves and pins every data parameter, and
  framework code still never reads the index or Parquet.
- **ADR 0025** (TorchTitan for distributed training). External trainers reuse
  its hostfile launch, remote staging, and preflight, but not its data path:
  rank zero does not read, tokenize, or broadcast data for external trainers.
  Parallelism strategy belongs to the trainer.
- **ADR 0029** (resume from complete checkpoint bundles). The checkpoint
  manifest moves from the worker's `state.json` to a WALDO-written manifest.
  Compatibility is checked against trainer, tokenizer, and environment
  identity instead of the backend revision. Atomic directory commits, hash
  verification before handoff, same-run resume, and the interrupted
  classification still hold.
- **ADR 0044** (verify persisted training artifacts). The four-way loss
  comparison and the shared embedded model definition are built-in trainer
  requirements and are not imposed on external trainers. External trainers
  are verified as in [Output verification](#output-verification). Selection
  of the best held-out checkpoint (ADR 0079) still applies.
- **ADR 0046** (tokenization in Go). With an external tokenizer, WALDO drives
  the tokenizer application instead of its Go codec. Workers still receive
  token IDs only. The Go tokenizers remain the built-in tokenizers.
- **ADR 0071** (gradient accumulation). Unchanged in substance: as with the
  built-in `gradient_accumulation_steps`, an external trainer's batch
  decomposition is pinned in the run BOM, not in identity, and the global
  batch stays fixed.
- **ADR 0072** (node-local prepared streams). Unchanged, and the only data
  plane offered to external trainers.
- **ADR 0073** (pinned precision, recomputation, compilation). For external
  trainers, recomputation and compilation are execution parameters, as the
  built-in path treats them, but numeric precision is a recipe parameter and
  part of identity.

## Out of scope

The following are not covered by this decision and remain future work:

- launchers other than torchrun, such as `mpirun` and `srun`;
- `executable` (non-Python) entrypoints;
- OCI container environments and user-provided virtual environments for
  `python-lock` environments;
- enforcement of determinism modes and a `model reproduce` command, and with
  them any byte-for-byte or cross-rank training-output claim;
- artifact profiles and exports beyond the reference profile, and inference
  for external artifact profiles;
- tokenizer training through the tokenizer application (ADR 0077), tokenizer
  assets pinned by hash and fetched from their origin, and ownership of chat
  template rendering;
- new objectives and data types;
- a Python helper package for trainers and tokenizer applications; and
- moving the built-in trainers and tokenizers onto the external identity
  model.

## Consequences

- A model's identity names the exact program, tokenizer, and environment that
  produced it. Two composes that differ only in trainer commit are different
  models.
- WALDO's data guarantees (corpus BOM, held-out pins, consumption, loss masks)
  hold for any trainer because no trainer can see anything WALDO did not send.
- External trainers can train on one GPU or across nodes with the same cluster
  contract and data plane as the built-in distributed path.
- Third parties can train model families WALDO does not implement without
  changing WALDO.
- Trust shifts: WALDO verifies everything derivable from its own inputs and
  trusts the trainer's held-out metric for checkpoint selection only.
- Existing composes, runs, checkpoints, and BOMs without a `trainer` block
  keep their behavior and identity.
