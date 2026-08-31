# Ingestion contract

This is the canonical contract for bringing raw material into WALDO. It
applies whether the raw directory was produced by an OpenWALDO fetcher, another
acquisition tool, or prepared locally.

The normal ingestion command is:

```sh
waldo index ingest /path/to/raw-directory core/example
```

The supplied directory has a `manifest.json` at its root. The manifest owns
the corpus and source metadata, declares the input format and mapping, and
binds the complete recursive raw tree with file count, byte count, and a
deterministic tree SHA-256. The exact schema is defined in
[INGESTION-MANIFEST.md](INGESTION-MANIFEST.md).

## Canonical workflow

1. Acquisition writes raw upstream material beneath a new directory.
2. Acquisition validates the material and writes `manifest.json` last.
3. WALDO strictly parses the manifest and recursively inventories the declared
   source boundaries.
4. WALDO independently hashes every regular raw file and verifies the raw-tree
   evidence.
5. The declared built-in adapter maps the verified tree to logical records.
6. WALDO applies canonical redaction, assessment, deduplication, measurement,
   packing, publication, and index-contribution logic.

Acquisition and ingestion are separate trust boundaries. A fetcher never
chooses an index destination, invokes WALDO, converts to canonical Parquet, or
publishes objects. WALDO never executes a fetcher named by the manifest.

## Raw tree versus logical records

A raw directory may contain one file or thousands of recursively nested files.
The raw inventory exists to bind the complete input bytes; it does not define
record cardinality.

The selected adapter defines the mapping:

- A source-code tree declared as `text` may produce one record per retained
  text file.
- A JSONL file produces one record per line.
- A Parquet file produces one record per selected row.
- A future tree-aware format may combine multiple dependent files into one
  logical document.

Every raw file must be accounted for by the adapter as a logical input, a
dependency, or an explicitly supported non-content resource. Unsupported,
ambiguous, or unclaimed files fail closed.

The current built-in adapters support text, Markdown, mbox, JSON, JSONL,
Parquet, XML, PDF, and EPUB. PDF and EPUB each produce one logical document per
file. PDF requires an embedded text layer; WALDO does not perform OCR. EPUB is
read directly from its container in package-spine order without unpacking it
or fetching external resources. LaTeX and other dependency-aware document
trees require a future built-in adapter and are not accepted today.

## Manifest authority

The root manifest is the reviewed, durable handoff. It owns:

- corpus identity, title, and description;
- source identity, URL, category, effective license, and license evidence;
- content languages, selection, and other source provenance;
- the declared input format and corpus-neutral logical mapping;
- acquisition artifact evidence; and
- recursive raw-tree evidence.

WALDO verifies declarations against the bytes. The manifest cannot instruct
WALDO to execute a program, load an adapter from the raw directory, bypass
format detection, or accept mismatched evidence.

Direct ingestion of ordinary files with command-line metadata remains a
convenience for local use. Reproducible and reviewable corpus contributions
should use the manifest-backed directory contract.
