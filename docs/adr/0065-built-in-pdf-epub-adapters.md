# 0065: Ingest PDF and EPUB with built-in document adapters

Status: accepted
- Date: 2026-08-29

## Context

PDF and EPUB are common upstream publication formats. Acquisition should
retain those general raw formats, while WALDO remains responsible for the
deterministic conversion into canonical records. An executable converter or
corpus-specific profile would violate the manifest-backed ingestion boundary.

## Decision

`pdf` and `epub` are general physical input formats. Their manifests require
only the format declaration; they do not introduce logical profile types.
Each input file produces exactly one canonical text document.

The versioned PDF adapter uses a pinned pure-Go parser to extract an existing
text layer in page order. The initial adapter supports PDF 1.0 through 1.7 and
rejects malformed, encrypted, PDF 2.x, and textless PDFs. WALDO does not
perform implicit OCR, execute active content, or invoke an external program.

The versioned EPUB adapter reads the ZIP/OCF container in place, resolves the
package document, follows supported local resources in linear spine order,
and extracts visible text. Navigation and `linear="no"` resources are omitted.
It never unpacks the archive or fetches external resources, and it enforces
entry-count, expanded-size, compression-ratio, path, and record-size limits.

Embedded metadata is retained as canonical row metadata and passes through
the normal privacy-redaction policy. The ingestion manifest remains
authoritative for effective licensing; embedded rights declarations remain
raw evidence and do not silently override that license.

Adapter names include their extraction implementation version. The durable
conversion profile incorporates those adapter identities so changing a parser
or extraction algorithm changes provenance and plan identity without changing
the canonical Parquet row schema.

## Consequences

- Acquisition can preserve PDF and EPUB without a conversion sidecar.
- Existing Parquet shards remain valid and require no rebuild.
- Scanned PDFs require a future explicit OCR policy or upstream source in a
  different supported raw format.
- Extraction quality is bounded by the embedded PDF text layer and EPUB
  document structure; unsupported content fails closed.
