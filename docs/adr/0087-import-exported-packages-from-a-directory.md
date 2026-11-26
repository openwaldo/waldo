# ADR 0087: Import exported WALDO packages from a local directory

Status: accepted
- Date: 2026-11-04

## Context

WALDO can only import a package by pulling it from Hugging Face (pinning the
origin). A WALDO model can already be exported as a package: its weights, its
tokenizer, and the files that record its history and the data it trained on.
Since the provenance is in the files, WALDO should be able to check from the
files alone that its history is intact.

WALDO records where every imported model came from: for Hugging Face, the
repository and revision. A filesystem path would be the wrong record for a
package: it differs from one machine to the next, and it's private.

## Decision

The model importer also accepts a local directory holding an exported Hugging
Face package. Directory and Hugging Face imports share file selection,
architecture and tokenizer validation, normalization, and the keeping of a
release's history (ADR 0084), run BOMs, and training closure.

The origin pins every imported file by hash and takes the weights hash as its
revision. The caller names the package's source, never by local path:
`model join` names it by its run and record name in the runs repository, with
provider `waldo-join`.

## Consequences

- Everyone who imports the same package records the same origin, wherever they
  keep it, and no origin publishes a local path.

Relates to [0026](0026-pinned-model-origin.md) and
[0051](0051-direct-compose-model-origins.md).
