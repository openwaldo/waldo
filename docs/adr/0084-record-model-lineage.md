# ADR 0084: Record model lineage as content-addressed nodes

Status: accepted
- Date: 2026-11-03

## Context

WALDO records a model's own training runs and a single parent pin or origin,
so one model traces back to its immediate parent. Weights that pass through
several models, such as a fine-tune of a fine-tune, or a release pulled and
trained again, lose that trail: a release says nothing about the weights it
descends from, and a fine-tune of a fine-tune cannot show where the first one
started. Keeping every intermediate model to answer that is too much to store
and to ship with a finished model.

## Decision

A lineage node is the SHA-256 of some weights, their direct parents, each by
weights hash and by the hash of the parent's own node, and the run BOM that
trained these weights from those parents. Like a git commit, a node names only
its direct parents, and the hash of the newest node commits to the whole chain
behind it.

Each complete real run of a model has its own node, computed from its records,
whose parent is the node of the weights the run started from: an earlier run,
the parent's, or the origin's. A pulled origin's node names the release, by
the hash of its repository, resolved revision, license, and files, so the same
release is one node however the pull was written or which mirror served it. Its
parent is the node the release declared, when the pull keeps that lineage;
otherwise it is a root, where the chain ends. A model's node is its newest
run's, or its origin's if it has none.

Nodes live beside a model's records, never in them, so the records stay what
earlier WALDO versions write and read:

- A model composed from a parent's run writes `LINEAGE.json`
  (`waldo-model-lineage`, schema 1): the parent run's node and every node behind
  it the parent held. It is read with exact keys, and its parent node must
  name the weights and run BOM the parent pin names and its ancestors must be
  exactly the chain behind it; a file that does not is an error.
- A pull keeps the release's `BOM.json`, whose hash the origin already pins
  among its source files, as `origin/release/BOM.json`; a kept file that no
  longer has that hash is an error.
- A derived release's `BOM.json` (Hugging Face, MLX, GGUF, Ollama) carries its
  weights' node as `lineage` and the nodes behind it as `lineage_ancestors`. A
  pull keeps that lineage only when every reader reads it the same way and the
  release's `artifacts` list each pulled weights file once by hash and size;
  otherwise it notes why and leaves it out. A release's `BOM.json` never fails a
  pull.

Nodes name weights by content hash and runs by run BOM hash, whatever produced
them, so models trained by other means can join a chain later without changing
the node.

## Consequences

- A model carries its whole known history in one small file, so deleting or
  never fetching the models that made it loses nothing, and a release ships its
  history with it.
- Hashes prove a history consistent, not true: anyone can write a consistent
  chain. Trust in what a node claims comes from signed records above it.
- Fewer ancestors make a shorter history, not a different one: the chain ends at
  the first node a model does not hold. A missing `LINEAGE.json` or kept
  release file shortens a history the same way; an altered one is an error.
- Nodes make it possible to tell whether two models started from the same
  weights by walking parents, like `git merge-base`; nothing walks them yet.
- Earlier WALDO versions read every model this one writes; they ignore
  `LINEAGE.json` and the kept release file. Models made before lineage, or
  composed by an earlier version, start their history at their parent's weights
  hash or at their origin.
- Lineage from a later WALDO never blocks this one: a `LINEAGE.json` or release
  of a later schema is left out, giving a shorter history. A later WALDO that
  changes either bumps its schema.
- A WALDO-format export copies `LINEAGE.json` but does not sign it; release
  formats carry lineage in their signed `BOM.json`.
- A pulled release's node is what the release declared. WALDO keeps it as
  declared and cannot check it; verifying it against signed releases is future
  work.

Amends [0026](0026-pinned-model-origin.md), [0066](0066-pin-trained-model-parents.md),
and [0022](0022-model-release-exports.md).
