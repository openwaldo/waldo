# 0086: Select deterministic hashed slices of a corpus

Status: accepted

## Context

Dividing one corpus among several trainings, so that each trains a disjoint,
reproducible share, otherwise needs a separately ingested corpus per share,
which breaks the link to the published corpus. Parties that train separate
shares must agree on exactly which records each share holds, without running
WALDO's code to find out.

## Decision

A record filter may declare `slice: {index, count}`; both are required, with
`0 <= index < count`. A record belongs to the slice when the first 8 bytes of
the SHA-256 of its lowercase hex record ID, read as a big-endian integer, modulo
`count` equal `index`. The filter is persisted in the corpus BOM and therefore
participates in training and preflight identity.

## Consequences

The `count` slices of a corpus are disjoint, cover it, and are near-equal in
size. A record ID is the SHA-256 of its text, so a record stays in its slice
for as long as its text is unchanged; ingestion changes that alter text may
move records, while a run pinned to an index revision is unaffected. Stage and
corpus slices hash the same value, so nesting them intersects the two
selections instead of subdividing a slice.
Older binaries reject composes and BOMs that declare a slice rather than
training the whole corpus.
