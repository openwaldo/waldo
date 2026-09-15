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
