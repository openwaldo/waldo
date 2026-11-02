# External application loading (PoC)

ADR [0084](adr/0084-external-trainers.md) defines the external trainer design.
The declarative reader, pinned source cache, manifest checks and probe-schema
validation, managed Python environment verification and single-process launch
are implemented as developer APIs.
Tokenizer probe/serve execution is implemented as a developer API. Trainer
probe/resolve execution, external model identity persistence and training
lifecycle integration are **planned**.
The CLI does not yet run or forecast external composes.

## Developer entry points

`model.LoadComposeDocument` reads either a native `Compose` or an
`ExternalCompose`. The native branch uses the existing `LoadCompose` and its
strict decoder. Schema 2 without trainer/tokenizer blocks is normalized to the
schema-1 native representation, preserving compose and architecture hashes.
A tokenizer block without a trainer is rejected. `LoadCompose` remains the
native lifecycle entry point and rejects external fields; callers must not
convert an external architecture into a native one.

Reading has no network or process side effects. Call
`ExternalCompose.ResolveSources(ctx, cacheDirectory)` explicitly to acquire
both applications. After probe supplies schema
bytes, compile them with `application.CompileSchema` and call
`ExternalCompose.ValidateSchemas`. No schema is loaded from a manifest.

## External compose fields

The following is a parser example, not a runnable trainer. Its repository and
commit are placeholders.

```yaml
kind: waldo-model-compose
schema: 2
trainer:
  source: &application
    repository: https://example.invalid/trainer.git
    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    path: examples/trainer
  environment: cpu
  determinism: best-effort
tokenizer:
  source: *application
  environment: cpu
  configuration: {}
architecture: {model_type: qwen3, width: 64}
stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    corpora: [books]
    data: {tokens: 128, global_batch_sequences: 2, sequence_length: 32, seed: 42}
    trainer: {learning_rate: 0.001}
```

The external compose requires:

- `tokenizer`: `source` (which may alias `*application`), `environment`, and an
  object `configuration`, possibly `{}`.
- `architecture`: an opaque object checked against the probe architecture
  schema. Native architecture fields have no special meaning here.
- `stages`: nonempty ordered declarations with the existing `name`, `type`,
  `objective`, `corpora`, optional `filter` and `conversation` fields. Each
  stage requires objects `data` and `trainer` (the latter may be empty).
- Optional `interaction`, with the existing conversation contract.

`data` accepts `tokens`, `epochs`, or `steps` using the existing budget rules;
`global_batch_sequences`, `sequence_length`, `seed`; optional `profile`,
`checkpoint_every`, `evaluate_every`, `shuffle_buffer_records`,
`shuffle_buffer_bytes`, `evaluation_fraction`, `evaluation_selection`,
`evaluation_max_records`, and `evaluation_max_bytes`. Inline corpus weights
retain the existing weighted-profile rules. WALDO resolves these through the
native data rules, but `ResolvedExternalData` exposes no native optimizer or
trainer defaults. Trainer-specific values belong only in `stages[].trainer`.

Both sources require a repository, a full 40-hex commit, and a clean relative
`path` (`.` for the repository root). Both applications select a named
environment. `trainer.determinism` is required and accepts `required`,
`best-effort`, or `off`; it is only recorded. External `base` initialization is
not implemented yet. Unknown fields and trailing documents fail; YAML anchors
and aliases are accepted.

## Manifests

At the selected directory, a trainer declares `waldo-trainer.yaml`:

```yaml
kind: waldo-trainer
schema: 1
name: example
protocol: 1
entrypoint: {module: trainer.main}
commands: {probe: probe, resolve: resolve, train: train}
launch_profiles: [single-process]
objectives: [causal-language-modeling]
artifact_profile: hf-causal-lm
environments:
  cpu: {kind: python-lock, lock: pylock.toml, python: 3.12.3}
```

A tokenizer declares `waldo-tokenizer.yaml` with `kind: waldo-tokenizer`,
`schema: 1`, `protocol: 1`, `name`, `entrypoint`, `environments`, and
`commands: {probe: probe, serve: serve}`. Trainer-only capabilities are rejected
on tokenizers. `entrypoint` contains exactly one of `module` or `script`;
`executable` is not implemented. Command values are single command arguments,
not shell strings. A script and lock must name regular files within the
selected application directory. Environments require `kind: python-lock`, a
relative `lock` and an exact `python` version (`3.x.y`). Environment creation
and lock verification use the explicit API below.

Trainer launch profiles are `single-process` and `distributed`; this reader
only checks declarations. The launcher implements `single-process` and rejects
`distributed` with an explicit not-implemented error. Objectives are restricted to
`causal-language-modeling` and `assistant-response-modeling`. Artifact profiles
are declared here; output verification is a later lifecycle responsibility.
Manifests are strict: in particular, `schemas` is an unknown field.

## Source cache

The cache key is SHA-256 of the JSON pair `[repository, lowercase commit]`.
Different application paths in the same commit share one checkout. go-git
fetches into a temporary bare repository; WALDO exports the pinned tree into a
separate checkout and atomically publishes a read-only entry. It never runs
Git hooks, filters, submodules, conversion tools or application code.

Source facts include the selected directory's Git tree hash and SHA-256 of the
exact manifest bytes. Reuse checks the pinned commit and tree object hashes,
every exported file's Git blob hash, executable bit and read-only permissions,
and rejects added files. Symlinks, submodules, special files and escaping paths
are unsupported and rejected. A damaged cache entry fails rather than being
silently repaired. Read-only permissions prevent accidental changes; they are
not a sandbox against the cache owner or root.

## Probe schemas and roles

`application.ParameterSchema` validates JSON Schema 2020-12 and retains the
original bytes and SHA-256. Unknown `x-*` keywords survive as annotations both
in the original document and in compiled schemas. WALDO does not fetch schema
references: references must resolve within the received document, including
local JSON pointers, anchors and embedded resources. Dynamic references are
not supported by role analysis yet and fail closed.

`x-waldo-role` accepts only `recipe` and `execution`. An omitted role defaults
to recipe unless inherited from an annotated ancestor. Explicit roles constrain
the subtree, including referenced schemas; conflicting nested roles fail.
Architecture schemas accept only recipe roles. Probe schemas should make
execution properties optional in compose input when resolve supplies them;
WALDO does not remove `required`, inject defaults, or weaken the schema.

`ValidateSchemas` checks all opaque sections after probe. `ParameterSchema.Split`
validates a complete resolve result and separates recipe values from execution
values. Execution overrides in a stage remain execution values. The future
resolve caller must validate batch topology and world shape, compare recipe
results across world shapes, and record execution values only in the run BOM.
Schema byte persistence and final plan hashing belong to that lifecycle work.
Role projection handles references, object properties and patterns, arrays,
matching combinator branches and dependencies. Role annotations beneath
`unevaluatedProperties` or `unevaluatedItems` currently fail projection because
it cannot safely derive their evaluated-instance set. Unannotated assertions
at those locations still validate normally.

## Managed Python environments

After source resolution, call `ResolvedSource.EnsureEnvironment(ctx, name,
python, cache)` explicitly. `python` selects a trusted host interpreter with
Python 3.11+ (including `venv` and `tomllib`) and pip 22.3+. Its exact version
must match the selected manifest. The embedded helper uses pip's vendored
packaging implementation for PEP 508 markers, version specifiers and wheel
compatibility tags; no additional Go or target-environment dependency is added.
The helper runs isolated outside the source checkout with an empty environment
plus `PATH=/usr/bin:/bin`, a temporary `HOME`, and `LANG=C.UTF-8`.

The input is PEP 751 `pylock.toml` (or `pylock.<name>.toml`), version 1.0.
This PoC accepts single-use locks, package/environment markers, Python version
constraints, and HTTPS wheel URLs with SHA-256. It chooses the highest-ranked
compatible wheel for each selected package. Multiple matching entries fail.
Dependency edges are informational; they never trigger resolution or downloads.
Locks requiring extras/groups, VCS, directories, local paths, archives, or
builds fail closed. An sdist entry may accompany wheels but is never used.
Wheel `.data` schemes other than purelib, platlib and scripts are currently
unsupported. These restrictions are explicit errors, not fallback behavior.

Creation uses an unseeded venv with system site-packages disabled. WALDO
verifies downloaded artifact hashes before pip installs from an offline
wheelhouse using `--require-hashes --only-binary=:all: --no-deps --no-compile`.
Installer configuration files are disabled with `PIP_CONFIG_FILE=/dev/null`.
The environment contains no pip unless pip itself is locked. Installed package
names and versions must equal the selected lock entries. Every installed
`RECORD` must account for its files with SHA-256/384/512 and size checks (except
its own row and bytecode caches). Baseline venv files are separately attested.
Extra files anywhere in the venv, including `bin/`, fail; regular `.pyc` files
inside `__pycache__` are excepted. Editable/local installs, unsafe RECORD paths,
symlinks outside the venv baseline and overwritten baseline files fail.

Before publishing, WALDO relocates generated scripts/activation paths, updates
the affected RECORD hashes, verifies again, and writes a schema-1
`waldo-python-environment` receipt. It includes the full verified file inventory
so changing a file and its RECORD together still fails cache verification.
Entries become read-only and are atomically renamed into a cache keyed by lock
SHA-256, interpreter executable SHA-256 (after resolving symlinks), and the
highest-priority interpreter/platform wheel tag. Concurrent creators verify the
winner. Failed/interrupted creation publishes no partial entry; abrupt process
or host termination may leave hidden temporary directories, never cache hits.
Published entries are never repaired or reinstalled on reuse.

`pythonenv.Verify` is separate from creation and checks an existing managed
entry without installing anything. Every use repeats package, RECORD, inventory,
configuration and interpreter checks. Verification does not import target
packages or run their `.pth` files; the target interpreter is only queried with
`-I -S` after its files have been checked. This is integrity checking, not a
sandbox against the cache owner/root rewriting both evidence and files, or
concurrent mutation after verification. The base interpreter's stdlib is
covered only by the interpreter realpath executable digest, not hashed file by
file (known PoC limit).

`pythonenv.Result.Identity` exposes name, kind, lock SHA-256, interpreter version
and executable digest, and platform tag to the future external BOM builder.
`Result.Run` holds the cache path, original installer/version and fresh host
observations: kernel, NVIDIA GPU models/drivers, CUDA driver capability and CUDA
toolkit version. Missing tooling/query results are recorded as unavailable;
these observations do not claim which CUDA runtime a future framework loads.
Cache paths, installer and host facts are run-only evidence. Native compose
identity and native execution are unchanged. External plan/run BOM persistence
and CLI launch remain planned; callers must preserve this identity/run split.

## External process launcher

`internal/model/applaunch` is the shared process boundary for trainer
`probe`, `resolve`, `train` and tokenizer `probe`, `serve`. Protocol codecs and
their lifecycle callers remain planned. `Prepare` accepts a freshly verified
`application.ResolvedSource` and `pythonenv.Result`, a manifest command role,
arguments, a launch profile, node/world facts and an existing absolute scratch
directory. Callers must resolve sources and verify environments immediately
before use; the launcher does not install or independently attest these inputs.

`Start` executes the verified environment's `bin/python -E -s`, followed by
`-m <module>` or the relative script path, the manifest's command argument and
the supplied arguments. No shell is involved. The working directory is the
read-only selected application directory. Its complete environment is
`WALDO_PROTOCOL=1`, `PATH=/usr/bin:/bin`, `LANG=C.UTF-8`, `LC_ALL=C.UTF-8`,
`HOME=<scratch>`, plus `CUDA_VISIBLE_DEVICES` only when present in WALDO's
environment (including an explicitly empty value). No other host variables
are inherited. `Process.Evidence()` returns the exact argv, directory,
environment values, launcher, profile and world facts for the future run BOM
builder; these are run evidence, never model identity. This is process isolation
from accidental host configuration, not an OS security sandbox.

The returned process exposes stdin/stdout pipes. Callers write protocol frames
while concurrently reading stdout, close stdin at protocol EOF, always call
`Wait`, and close stdout when finished. Stderr is continuously drained, retains
at most 64 KiB plus a truncation marker, and is separate from protocol output.
`Wait` preserves exit status and cancellation errors and includes captured
stderr on failure. Callers map Ctrl-C/SIGTERM to context cancellation; cancellation
sends SIGTERM to the process group, allows five seconds to drain, then kills
remaining processes. Leader exit also cleans up descendants holding pipes open.

`World` describes a per-node launch, including node rank, processes per node,
world size and rendezvous address/port. Single-process requires the one-node,
one-process shape. Distributed execution is **planned** for WALDO-54 with
torchrun and ADR 0072's node-local-cache data plane: each node's local rank zero
receives begin/data from its own WALDO coordinator; only global rank zero
reports results. This API does not introduce a global-rank-zero data stream.
Built-in worker launch and native compose behavior are unchanged.

## Tokenizer protocol 1

`tokenizerapp.Open(ctx, launchRequest, configuration)` runs a tokenizer probe,
validates the configuration against the returned schema, and starts one `serve`
process. The launch request must contain freshly verified source/environment
results and run scratch. Callers must supply a cancellation/deadline context
and close the client. External lifecycle/CLI orchestration and writing plan/run
BOM files remain planned; this API supplies their tokenizer identity and launch
evidence.

Both commands receive one initial line `{"configuration":{...}}`. Probe writes
exactly one JSON line and exits successfully:

```json
{"name":"example","vocabulary_size":259,"special_tokens":{"pad":0,"bos":1,"eos":2},"deterministic":true,"schemas":{"configuration":{"type":"object"}}}
```

Pad, BOS and EOS are required, may alias, and must be within the vocabulary.
Additional named special IDs are accepted and checked. Determinism must be
reported explicitly. The configuration schema contains recipe roles only and
is retained byte for byte with its SHA-256; schemas never come from manifests.
Probe validates the supplied configuration without inserting defaults.

After its configuration line, serve reads `{"id":1,"text":"hello"}` and
writes `{"id":1,"tokens":[107,104,111,111,114]}`. Encode ordinary text only:
never insert BOS/EOS or recognize special-token spellings. WALDO owns framing
and renders conversation segments separately to preserve loss masks. Requests
are independently identified; responses may arrive out of order. `EncodeBatch`
uses this same wire format, pipelines at most 64 requests, and returns results
in input order. Each NDJSON frame is limited to 16 MiB. Empty token arrays are
valid; missing/null arrays, invalid IDs, out-of-vocabulary tokens, malformed
frames, process exit, and `{"id":1,"error":"reason"}` fail the session.
Pending errors identify their request IDs; data preparation adds record IDs.

The client implements `training.TokenCodec` and its fallible `EncodeChecked`
extension. WALDO checks that extension in capacity planning, held-out counts,
corpus balancing and mask rendering. Legacy `Encode`/`Count` callers must check
`Err`; decoding is unsupported and fails the session. Concurrent corpus workers
share the bounded pipeline. Pass the client to existing tokenizer-taking
partition/source APIs; use `Spec()` for vocabulary and special IDs. Native
codec resolution remains unchanged.

`Identity()` provides source, environment, configuration and its digest,
probe facts, and schema digest for plan/run BOM builders. `Spec().Revision` is
the digest of this identity, so the existing prepared-sequence cache identity
(which hashes the tokenizer specification) changes with every tokenizer pin.
Host launch evidence from `Evidence()` stays outside that digest. The identity
snapshot omits source/cache paths and can be serialized into the future plan.
`training.NewTokenizedRecordSource` wraps a canonical record stream with this
codec for token-only trainer input, clearing text/conversation data after
WALDO renders segments and builds masks.
