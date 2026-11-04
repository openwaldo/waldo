# ADR 0086: Import exported WALDO packages from a local directory

Status: accepted
- Date: 2026-11-04

## Context

`model pull` imports Hugging Face repositories only. Tools that move exported
packages between machines without a hub, such as `model join` with a file
store, otherwise have to imitate the Hugging Face API, and the resulting origin
then names a repository that does not exist.

## Decision

The importer that verifies and normalizes pulled packages also accepts a local
directory holding an exported Hugging Face package. The origin's provider is
`file`; it pins every imported file by hash and takes the weights hash as its
revision. Hugging Face and directory imports share file selection,
architecture and tokenizer validation, normalization, and the keeping of a
release's lineage node, run BOMs, and training closure (ADR 0084).

## Consequences

- An origin always names the source the weights actually came from.
- The origin currently records the directory's absolute path as its URL, so the
  path is part of the origin BOM and its hash. A relocatable source identity is
  open work.

Relates to [0026](0026-pinned-model-origin.md) and [0051](0051-direct-compose-model-origins.md).
