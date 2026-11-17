# ADR 0087: Import exported WALDO packages from a local directory

Status: accepted
- Date: 2026-11-04

## Context

`model pull` imports Hugging Face repositories only. Tools that move exported
packages between machines without a hub, such as `model join` with a file
store, otherwise have to imitate the Hugging Face API, and the resulting origin
then names a repository that does not exist.

## Decision

The importer that verifies and normalizes pulled packages also accepts a local
directory holding an exported Hugging Face package. It pins every imported file
by hash and takes the weights hash as its revision. The caller names the
package's source, never by local path: `model join` names it by its run and
record name in the runs repository, with provider `waldo-join`. Hugging Face
and directory imports share file selection, architecture and tokenizer
validation, normalization, and the keeping of a release's history (ADR 0084),
run BOMs, and training closure.

## Consequences

- Everyone who imports the same package records the same origin, wherever they
  keep it, and no origin publishes a local path.

Relates to [0026](0026-pinned-model-origin.md) and [0051](0051-direct-compose-model-origins.md).
