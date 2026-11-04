# ADR 0084: Record model lineage as content-addressed nodes

Status: accepted
- Date: 2026-11-03

## Context

WALDO records a model's own training runs and a single parent pin or origin,
so one model traces back to its immediate parent. Weights that pass through
several models, such as a fine-tune of a fine-tune, or a release pulled and
trained again, lose that trail: a release says nothing about the weights it
descends from, and a fine-tune of a fine-tune cannot show where the first one
started. Copying every ancestor's facts forward instead would make each record
grow with the length of its history.

## Decision

A lineage node is the SHA-256 of some weights, their direct parents, each by
weights hash and by the hash of the parent's own node, and the run BOM that
trained these weights from those parents. Like a git commit, a node names only
its direct parents, so its size does not depend on the history behind it.

Each complete real run of a model has its own node, whose parent is the node of
the weights the run started from: an earlier run, the parent pin, or the
origin. A pulled origin's node is a root: it also names the origin BOM's hash,
so the chain ends at the release, revision, and license the weights came from.
A model's node is its newest run's, or its origin's if it has none. A
fine-tune's parent pin records the node of the run it starts from, which never
changes as the parent trains on. A derived release's `BOM.json` (Hugging Face,
MLX, GGUF) carries the node of the weights it was converted from, and a pull
keeps it in the origin, reading the `lineage` key exactly and refusing a node
not in the form WALDO writes, so the chain continues across publication.

Nodes name weights by content hash and runs by run BOM hash, whatever produced
them, so models trained by other means can join a chain later without changing
the node.

## Consequences

- Records commit to their whole history through hashes, and stay bounded: a
  node names only its direct parents.
- Nodes make it possible to tell whether two models started from the same
  weights by walking parents, like `git merge-base`; nothing walks them yet.
- Deleting a model leaves its descendants valid, but the chain can be walked
  past it only where its small node and origin records are kept.
- Parent pins and interrupted composes recorded before lineage nodes keep
  matching; their history ends at the immediate parent.
- A pulled release's node is what the release declared. WALDO records it as
  declared and cannot check it; verifying it against signed releases is future
  work.

Amends [0026](0026-pinned-model-origin.md), [0066](0066-pin-trained-model-parents.md),
and [0022](0022-model-release-exports.md).
