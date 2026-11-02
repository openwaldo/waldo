# External application loading (PoC)

ADR [0084](adr/0084-external-trainers.md) defines the external trainer design.
The declarative reader, pinned source cache, manifest checks and probe-schema
validation are implemented. Environment creation, probe/resolve execution,
external model identity and training lifecycle integration are **planned**.
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
both applications. After the future probe implementation supplies schema
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
and lock verification are a separate forthcoming step.

Trainer launch profiles are `single-process` and `distributed`; this reader
only checks declarations and launches neither. Objectives are restricted to
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
