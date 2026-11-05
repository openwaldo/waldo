# Architecture

WALDO is one Go binary with bounded internal domains. The binary is the
distribution boundary; it is not a shared service container.

```text
CLI
├── index ───────────────┐
├── ingest ── index ─────┼── corpus BOM ── model ── training/inference
├── corpus ─ lookaside ──┤
└── shard ───────────────┘
                                  └── provenance and exports
```

Dependencies point from model workflows toward resolved corpus contracts.
Index, record, shard, ingest, lookaside, and corpus packages must not depend on
model or training packages.

## Domains

### Index

`internal/index` owns index and manifest schemas, canonical metadata,
traversal, structural verification, summaries, and Git revision identity.
`internal/git` owns safe managed-checkout synchronization.

The index records meaning. It does not store large corpus bytes or inspect
model state.

### Record and shard

`internal/record` owns canonical document and license semantics.
`internal/shard` owns canonical Parquet encoding, decoding, embedded shard
BOMs, and record validation. These definitions are shared by ingestion,
audit, export, and training input.

### Ingestion

`internal/ingest` owns input probing, profiles, conversion, deterministic
deduplication and packing, publication, contribution overlays, and recovery.

Acquisition remains outside WALDO. Fetchers produce a manifest-backed recursive
raw directory and stop. WALDO never executes a fetcher or converter named by
the manifest; it verifies the complete raw tree and applies a reviewed built-in
adapter selected by the declarative input format. See
[INGESTION.md](INGESTION.md).

### Lookaside

`internal/lookaside` owns content-addressed object transport, verification,
caching, S3 credentials, mirroring, inventory, and explicit removal. An
object URL is a transport location, not provenance.

### Corpus and provenance

`internal/corpus` resolves an index selection, license policy, and verified
objects into an immutable corpus BOM or export. `internal/provenance` owns
serialized provenance projections shared across exports.

Model code consumes resolved BOMs. It must not independently traverse index
trees, choose mirrors, or normalize corpus licenses.

### Model, training, and inference

`internal/model` owns model identity, architecture, versioned interaction
contracts, compose transactions, run history, origin pulls, forecasting
inputs, and lifecycle state.

`internal/training` owns portable backend requests and MLX, PyTorch, and
TorchTitan adapters. A backend receives an explicit request and returns
observations; it does not own model persistence or corpus selection.

`internal/join` contributes to distributed training runs kept in a git
repository: it reads a run's plan and signed records, trains, verifies, and
merges units through the model domain, and commits signed records. It runs the
external `git`, `ssh-keygen`, `gh` (GitHub runs repositories), and `hf`
(Hugging Face stores) commands; see [model join](MODEL-JOIN.md).

`internal/inference` loads verified model artifacts for supported local
runtimes and enforces generation stop sequences supplied by the interaction
contract. Export conversion is split across `internal/modelexport`,
`internal/modelweights`, and `internal/modelquant`.

### Cross-cutting packages

- `internal/config`: machine-local transport and execution preferences.
- `internal/host`: framework-independent local CPU, memory, and filesystem facts.
- `internal/status`: read-only index, lookaside, and training readiness orchestration.
- `internal/canon`: deterministic serialization helpers.
- `internal/calibration`: forecast and quantization calibration evidence.
- `internal/disclosure`: EU GPAI disclosure projection.
- `internal/signing`: release BOM signing.
- `internal/ai`: optional advisor provider boundary.

## Process boundaries

Framework-specific Python workers receive a versioned request and stream. They
do not parse the Git index or define a second model configuration format.
External fetchers populate an explicit manifest-backed raw directory and stop
before conversion. Git remains the review mechanism for index metadata.

## Implementation rules

- Durable formats have explicit `kind` and `schema` fields.
- A durable format change requires an ADR, fixtures, and compatibility tests.
- Domain types do not depend on CLI types.
- Interfaces live near their consumer.
- State-establishing writes use temporary files and atomic rename.
- Large data paths stream and remain bounded in memory.
- Errors identify the failed path, object, model, or run and suggest the next
  useful action when one exists.
- Network and destructive operations remain explicit and testable.

## Durable boundaries

The compatibility promises are limited to the surfaces in
[COMPATIBILITY.md](COMPATIBILITY.md). Internal Go packages and managed-state
layouts are not public APIs.
