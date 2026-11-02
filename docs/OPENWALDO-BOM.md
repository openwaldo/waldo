# OpenWALDO corpus BOM contract

`EXPORT.json` is WALDO's first stable provenance interchange document. It is a
`waldo-corpus-export` schema-1 envelope containing an OpenWALDO BOM and the
files produced from it.

The envelope and BOM have different jobs:

- The OpenWALDO BOM identifies the selected corpus meaning: index context,
  requested paths, license policy, manifest and source pins, verified
  submanifest tree, fully resolved shard facts, and exact selected totals.
- The export envelope records one materialization: generation time, output
  format, and the identity of every exported file.

## Identity

```json
{
  "kind": "waldo-corpus-export",
  "schema": 1,
  "generated": "2026-08-04T12:00:00Z",
  "format": "native",
  "bom": {
    "kind": "openwaldo-bom",
    "schema": 1,
    "subject": "corpus"
  },
  "files": []
}
```

`generated` is an observation, not part of corpus identity. Re-exporting the
same BOM may produce a later value while preserving every pinned input and
file hash.

## BOM fields

- `index` records the checkout's remote, Git commit, and whether its working
  tree was dirty. Manifest hashes remain the precise pins if Git identity is
  unavailable or dirty.
- `paths` is the sorted, de-duplicated user selection.
- `license_policy` records include and exclude globs; excludes take precedence.
  Globs apply to each term of a compound license expression: any excluded term
  rejects the license, and includes must cover every term (see the record
  filter rules in `MODEL-COMPOSE.md`). Malformed compound expressions are
  rejected when an explicit license policy is configured.
- `manifests` records each selected manifest's hash and resolved corpus-level
  facts, including format, record schema, conversion recipe, upstream sources,
  structured processing declarations, modality measures, and exact totals
  after policy filtering.
- `sub_manifests` records every verified external manifest node, its parent,
  aggregate totals, and encoded object size. It is absent for inline corpora.
- `shards` is the ordered leaf sequence after inheritance and policy filtering.
  Every entry is self-contained: object identity, format, record schema,
  license, source references, conversion recipe, containing submanifest when
  applicable, optional record Merkle root, modality measures, and declared
  measures. After object validation, `attestation` pins the embedded shard BOM
  and its SHA-256; writer-v4 and deep-validated legacy shards record their
  explicit validation status without inventing an embedded BOM.
- `totals`, `modalities`, and `licenses` are exact sums of the selected shards,
  not estimates.

## Exported files

Every file entry distinguishes the selected lookaside object from the derived
export:

- `object_sha256` and `object_bytes` identify the verified input object.
- `sha256` and `bytes` identify the file beneath the export directory.
- `path` is a normalized relative path beneath `data/`.

For `native`, input and output identity must be equal. For `jsonl`, WALDO
streams the native Parquet object into canonical schema-1 records, validates
every record and text hash, and requires emitted document and token totals to
equal the shard declaration before publishing the file atomically.

## Verification and limits

```bash
waldo index bom /path/to/export
waldo index verify /path/to/export
```

`index verify` validates identities, references, policy, totals, safe file paths,
native/interchange relationships, and then hashes every exported file. It is
offline and does not require the original index or lookaside.

This proves internal consistency and possession of the exported bytes. It does
not re-fetch upstream sources, make a legal judgment about license assertions,
prove that a remote Git repository still serves the recorded commit, or prove
that a later trainer consumed every byte.

## Evolution

Readers of schema 1 must accept unknown fields so compatible facts can be
added. A change that removes a field, changes its meaning, or relaxes an
identity invariant requires a new schema. Writers always emit resolved values
for fields with index-level inheritance; consumers must not reconstruct those
values from machine-local configuration.

## External model provenance (developer API)

External trainers use schema-2 `waldo-model-plan` and `openwaldo-bom` records
under ADRs 0084 and 0086. Native schema-1 plan hashes and BOMs are unchanged.
The implemented persistence APIs in `internal/model/external_bom.go` are
`WriteExternalPlan`, `ReadExternalPlan`, `WriteExternalRunBOM` and
`WriteExternalModelBOM`. CLI training/export integration remains planned.

`PLAN.json` contains `kind`, `schema`, `name` and `identity`. Its SHA-256 is
computed from compact Go JSON encoding, without a trailing newline. Identity
contains:

- `trainer`: resolved repository, full commit, selected path, Git tree hash
  and manifest SHA-256, using `application.ResolvedSource` (no cache directory).
- `environment`: named trainer environment, kind, lock SHA-256, interpreter
  version and executable SHA-256, platform tag (`pythonenv.Identity`).
- `tokenizer`: the existing `tokenizerapp.Identity`, including resolved source,
  environment, configuration and configuration/schema hashes and probe facts.
- `architecture`, `architecture_sha256`, `architecture_schema_sha256` and
  `parameters_schema_sha256`.
- `determinism`, optional `interaction`, and ordered `stages`: name, type,
  objective, resolved corpus BOM SHA-256, WALDO-owned data, conversation
  transform, resolved recipe and recipe SHA-256. Execution parameters never
  enter this object. Callers must use WALDO's validated resolve role split.

`TRAINER-SCHEMAS.json` sits beside the plan and preserves the exact probe
schema documents. Its document hashes must equal identity pins. A new plan and
schemas are published together; an existing model directory is not replaced.

`RUN-BOM.json` (`subject: training-run`) repeats identity and records model/run
IDs, stage, corpus BOM, full resolved parameters and their hash, execution
projection, batch decomposition, evaluation-set pins, trainer/tokenizer host
evidence, trainer probe (frameworks, devices, driver, CUDA), and launch evidence
(profile, launcher, world, working directory, argv and exact environment
allowlist). Run BOM files are immutable. Kernel and environment installer/cache
observations are run-only facts. Callers prepare launch evidence before launch;
later attempt observations belong in lifecycle history.

`MODEL-BOM.json` and exported `BOM.json` (`subject: model`) contain the same
identity, model/plan hash, name, `path_base: model-root`, and run references.
Each run records its ID, run BOM file path/byte hash/size, and published artifact
paths/hashes/sizes/roles/formats. Paths resolve relative to that BOM's directory.
The shared aggregate writer verifies those files before writing; an exporter
must copy files and exact schemas into the destination before calling it with
`exported: true`. A JSON file's byte hash is distinct from the canonical JSON
identity hash returned by the run writer.

EU-BOM.json remains the corpus/regulatory projection: trainer/tool identity
belongs to the model and run BOMs. This does not change promotion gates or
assert regulatory compliance.
