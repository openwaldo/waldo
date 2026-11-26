# ADR 0086: Select deterministic hashed slices of a corpus

Status: accepted
- Date: 2026-11-15

## Context

Distributed training requires giving different pieces of a single corpus to
varied trainers. This ADR specifies how to divide a corpus into distinct pieces
in order to be distributed to various trainers, while maintaining lineage and provenance.

## Decision

A record filter may declare `slice: {index, count}`, stage-wide or per corpus.
Both fields are required, with `0 <= index < count`. A record belongs to the
slice when the first 8 bytes of the SHA-256 of its lowercase hex record ID,
read as a big-endian integer, modulo `count` equal `index`. The filter is
persisted in the corpus BOM, so it is part of training and preflight identity.

## Consequences

- The `count` slices of a corpus don't overlap, cover it, and are near-equal.
- A record ID is the SHA-256 of its text, so a record keeps its slice while its
  text is unchanged. Ingestion changes that alter text may move records; a run
  pinned to an index revision is unaffected.
- Each slice of a corpus trains into its own model, and a model that completed
  one slice skips the corpus as already trained.
- A stage slice and a corpus slice hash the same value, so nesting them
  intersects the selections instead of subdividing one.
- Older binaries reject a compose that declares a slice rather than training the
  whole corpus, and can't read a model trained on one: they report its run as
  not matching its model pin and refuse to list the model store that holds it.
