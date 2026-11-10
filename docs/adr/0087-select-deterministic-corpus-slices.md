# ADR 0087: Select deterministic hashed slices of a corpus

Status: accepted
- Date: 2026-11-04

## Context

Dividing one corpus among several trainings, so that each trains a disjoint,
reproducible share, otherwise needs a separately ingested corpus per share,
which breaks the link to the published corpus. Several parties must also be
able to agree on exactly which records each share holds.

## Decision

A record filter may declare `slice: {index, count}`. A record belongs to slice
`index` of `count` when the first 8 bytes of the SHA-256 of its canonical record
ID, read as a big-endian integer, modulo `count` equal `index`. Record IDs are
the SHA-256 of a record's final text, so a slice is stable for as long as each
record's text is unchanged. The filter is part of the compose and of the corpus
BOM's record filter, so every run records exactly which share it trained.

## Consequences

- `count` slices are disjoint, together cover the corpus, and are near-equal in
  size for any `count`.
- Changes to ingestion that change records' text change their IDs and may move
  them between slices; a run pinned to an index revision is unaffected.

Relates to [0049](0049-configured-compose-corpora.md) and [0030](0030-deterministic-heldout-evaluation.md).
