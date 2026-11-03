# ADR 0087: Integrate external applications through a separate lifecycle record

Status: accepted

## Decision

Schema-2 composes with a trainer block dispatch to `Builder.ComposeExternal`.
Native composes retain their existing lifecycle, architecture decoder and plan
hash. External model identity includes all resolved stages and corpus BOMs;
a changed identity requires a new model name. An exact completed invocation
verifies artifacts and performs no training.

External `MODEL.json` and `RUN.json` use schema 2. The immutable plan supplies
ordered stage names; each stage has one stable run ID equal to its name beneath
`runs/<stage>`. `COMPOSE.json` retains the original request for `model continue`.
The per-model compose lock excludes concurrent native and external compose executions.
The external model state and ordered run states are the compose transaction;
no native architecture or native backend identity is synthesized.

Before launch, WALDO verifies application sources/environments, resolves corpus
BOMs, selects held-out records, tokenizes through the application and resolves
trainer parameters. WALDO passes only token IDs. Shared EOS packing and compact
prepared-cache readers preserve native behavior. The external cache is bound to
the run BOM and can replay a verified suffix; an incomplete cache falls back to
deterministic prefix replay. Protocol 1 requires exactly the declared sequence
count; missing supervised batch slots fail rather than inventing observations.

Each run retains immutable `RUN-BOM.json`, atomic `RUN.json`, attempt history,
independently persisted checkpoint references and evaluation history. A durable
checkpoint is saved before progress callbacks run. Recoverable interruptions
reuse the run ID, run BOM and verified bundle. Numerical/artifact-integrity
errors remain terminal. The initial run BOM records framework/device/driver/CUDA
and environment evidence; each attempt additionally records its current probe
and launch evidence. Successful earlier stages initialize later stages from
verified published weights.

External exports copy only verified published artifacts, run BOMs, the plan and
exact schemas. The final stage's `hf-causal-lm` files are loadable at the package
root; earlier stage artifacts retain relative provenance paths. Private resume
state is excluded. Publication is atomic after verification and the configured
signer succeeds. The existing provider/disclosure gate and signing settings
apply, including the explicit `--allow-incomplete` gate. With no signer configured,
the CLI warns that the export is unsigned. Application identity remains in the
model BOM; EU-BOM stays the shared corpus disclosure projection.

`WALDO_APPLICATION_PYTHON` selects the trusted bootstrap interpreter (default
`python3`), and `WALDO_APPLICATION_CACHE` overrides the source/environment/prepared
cache (default `<model.root>/.applications`). These locations are host policy,
not portable plan identity. Distributed external launch, conversion/quantization
and external inference remain unsupported.
