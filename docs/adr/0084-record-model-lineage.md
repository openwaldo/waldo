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

A model's identity commits to its history, and the history itself travels in a
small file beside the records, so the models that made it can be deleted:

- A fine-tune's parent pin records the hash of the node of the run it starts
  from (`lineage_sha256`), which never changes as the parent trains on. The
  model also writes `LINEAGE.json` (`waldo-model-lineage`, schema 1): that node
  and every node behind it the parent held, read with exact keys. The parent
  node must hash to the pinned value and the ancestors must be exactly the
  chain behind it; a file that does not is an error.
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
  history with it. Without the file, the pinned hash still names the history;
  it is only not held.
- Hashes prove a history consistent, not true: anyone can write a consistent
  chain. Trust in what a node claims comes from signed records above it.
- Fewer ancestors make a shorter history, not a different one: the chain ends at
  the first node a model does not hold, where the link still names its hash. An
  altered `LINEAGE.json` or kept release file is an error.
- A pulled origin without its release's lineage, because the release file is
  missing, unreadable, or of a later schema, is a root: a different node, so new
  fine-tunes of it pin a different lineage than those of the same release
  pulled with it.
- Nodes make it possible to tell whether two models started from the same
  weights by walking parents, like `git merge-base`; nothing walks them yet.
- The upgrade is one-way for fine-tunes: earlier WALDO versions refuse a model
  whose parent pin records `lineage_sha256`, and composes that assert it, as an
  invalid model plan; they cannot `model rm` one, and one such model stops their
  `model list` and `lookaside cache` over the whole store. They still read
  pulled, untrained, and from-scratch models, and ignore `LINEAGE.json` and the
  kept release file.
- Fine-tunes from before lineage keep their pins without lineage, so both
  versions still read and extend them; their history starts at their parent's
  weights hash, and a `LINEAGE.json` beside one is ignored. A model this WALDO
  makes pins its lineage, even for a compose an earlier WALDO was interrupted
  in before making the model; one it had made keeps its pin without lineage.
- Only a model's records hold its pin, computed from its base's history when
  the model is made; composes carry one only when asserted, which must match.
  Models already made keep the lineage they pinned, even if their base's
  recorded history later shortens.
- Lineage from a later WALDO never blocks this one: a `LINEAGE.json` of a later
  schema is left out, giving a shorter history, and a release of one leaves its
  pulled origin a root. A later WALDO that
  changes either bumps its schema.
- A WALDO-format export copies `LINEAGE.json` unsigned, but its signed
  `BOM.json` pins the parent node hash, which commits to that chain; release
  formats carry lineage in their signed `BOM.json`.
- A pulled release's node is what the release declared. WALDO keeps it as
  declared and cannot check it; verifying it against signed releases is future
  work.

Amends [0026](0026-pinned-model-origin.md), [0066](0066-pin-trained-model-parents.md),
and [0022](0022-model-release-exports.md).
