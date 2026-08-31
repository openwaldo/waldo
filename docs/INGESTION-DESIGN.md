# Ingestion architecture

Status: current implementation reference.

The user-facing workflow and trust boundary are defined in
[INGESTION.md](INGESTION.md). The exact handoff schema is defined in
[INGESTION-MANIFEST.md](INGESTION-MANIFEST.md).

## Ownership

Acquisition tools retrieve upstream material and create an immutable,
manifest-backed raw directory. They do not interpret content, invoke WALDO,
publish canonical shards, or choose an index destination.

WALDO owns:

- strict manifest parsing and raw-tree verification;
- physical-format probing;
- declarative mapping to logical records;
- privacy redaction and content assessment;
- content hashing and deduplication;
- token measurement and deterministic Parquet packing;
- shard audit, lookaside publication, and verification; and
- index contribution generation and application.

## Data flow

```
upstream material
    -> acquisition tool
    -> manifest-backed raw directory
    -> WALDO probe and plan
    -> logical records
    -> redaction and assessment
    -> canonical Parquet shards
    -> verified lookaside objects
    -> applied Git index contribution
```

The raw-file inventory and logical-record count are separate facts. One JSONL
file may contain many records. A directory of source files may produce one
record per retained file. A future tree-aware adapter may combine several raw
files into one logical document.

## Supported inputs

The current built-in adapters support:

- UTF-8 text and Markdown;
- mbox, including gzip and Zstandard compression;
- JSON and JSONL with a declared record mapping;
- Parquet with a declared mapping or text column; and
- XML with the bounded `xml-record` mapping;
- PDF text layers in page order; and
- EPUB linear spine content in reading order.

PDF and EPUB each produce one logical document per file. They use versioned
built-in adapters, require no logical profile, and never invoke an external
converter. PDF OCR is deliberately not implicit. EPUB containers are parsed
in place with bounded expansion and no network access.

LaTeX and other dependency-aware document trees are specified as future
tree-aware adapters. They are not implemented. A new adapter must be built
into WALDO, remain inside the verified manifest boundary, and fail closed on
unclaimed files, ambiguous roots, unsafe paths, or unsupported dependencies.

## Canonical records

WALDO currently writes two logical record kinds:

- `pretrain`: canonical text record schema 2.
- `conversation`: tokenizer-neutral structured conversation schema 1.

Every write records the record kind, record schema, writer recipe, adapter
recipe, tokenizer used for reference counts, source and license facts, and
aggregate totals. Existing schema-1 text shards remain readable.

Conversation ingestion preserves ordered roles, message content, optional
context, and tools as canonical JSON. Prompt rendering and supervised-role
selection happen in the model training view, not during ingestion.

## Safety and determinism

- Manifests cannot name executables or runtime adapters.
- Input paths must remain inside their declared source boundary.
- WALDO verifies raw bytes before interpretation and rejects later changes.
- Redaction runs before canonical identity, deduplication, and packing.
- Unsupported or ambiguous inputs fail before publication.
- Complete local assembly and audit finish before remote publication.
- Published objects are addressed and verified by SHA-256.
- The index change is applied only after the complete operation verifies.

## Updates and recovery

`waldo index ingest --update <input> <destination>` treats the supplied input
as a complete authoritative replacement for the existing corpus. It rebuilds
the shard and source set; it is not an append operation.

Ingestion journals local work so an unchanged retry can reuse verified state.
Successful staging and cache objects are purged after publication. WALDO keeps
the contribution copy for review and recovery, applies its metadata changes to
the selected working tree, and never commits or pushes them.

## Change requirements

A durable ingestion change requires:

1. an ADR when it changes a public contract or artifact identity;
2. strict validation and clear failure messages;
3. deterministic fixtures for persistent bytes;
4. focused unit tests and an end-to-end lifecycle; and
5. updates to the maintained contracts in this directory.
