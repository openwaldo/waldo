# ADR 0084: Record model lineage as content-addressed nodes

Status: accepted
- Date: 2026-11-01

## Context

WALDO records a model's own training runs and a single parent pin or origin,
so one model traces back to its immediate parent. Weights that pass through
several models, such as a fine-tune of a fine-tune or a merge of fine-tunes,
lose that trail, and nothing can tell whether two models share the weights they
started from. Copying every ancestor's facts forward instead would make each
record grow with the length of its history.

## Decision

A lineage node is the SHA-256 of some weights, their direct parents, each by
weights hash and by the hash of the parent's own node, and the run BOM that
trained these weights from those parents. Like a git commit, a node names only
its direct parents, so its size does not depend on the history behind it.
Each complete real run of a model has its own node, whose parent is the node of
the weights the run started from: an earlier run, the parent pin, or the
origin. A model's node is its newest run's, or its origin's if it has none. A
fine-tune's parent pin records the node of the run it starts from, which never
changes as the parent trains on; a merged origin (ADR 0085) records its own
node; a derived release's `BOM.json` (Hugging Face, MLX, GGUF) carries the
released model's node; a pull keeps the release's node in the origin.

Ancestry is found the way `git merge-base` finds it: by walking each node's
parents through every node the model store holds. Where a chain leaves the
store, the walk stops there.

## Consequences

- Whether two models share ancestor weights is answered from recorded nodes,
  never inferred from names or shapes.
- Records stay bounded: a node names only its direct parents.
- Parent pins recorded before lineage nodes keep matching; their history ends
  at the immediate parent.
- Declared lineage from an unsigned release is recorded as the release declared
  it, and a merge that relies on it says so (ADR 0085). Verifying it against
  signed releases is future work.

Amends [0026](0026-pinned-model-origin.md), [0066](0066-pin-trained-model-parents.md),
and [0022](0022-model-release-exports.md).
