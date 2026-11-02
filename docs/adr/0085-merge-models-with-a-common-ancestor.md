# ADR 0085: Merge models with a recorded common ancestor into a merged origin

Status: accepted
- Date: 2026-11-01

## Context

Averaging the weights of models fine-tuned from common weights is an
established way to combine them. Merges made outside WALDO discard provenance.
A merged model also fits none of WALDO's existing model kinds: it has no
training run, and ADR 0026 forbids recording a synthetic one.

## Decision

A compose base may be `merge`: two or more managed models, each with an
optional weight given for every parent or none. WALDO resolves each parent's
current weights, requires identical architectures and the byte tokenizer, and
fails closed unless the parents' lineage chains (ADR 0084) reach a common
weights hash. It walks each chain through every lineage node the model store
holds, like `git merge-base`. There is no override: models with no common
ancestor in that lineage are never averaged. That pulled weights came from the
node their release names is only what the release declared, whatever node it
names, and the store cannot check it, so when the common ancestor is shown only
through such claims the merge records `declared_ancestry: true`.

The merge is an origin. WALDO computes the mean in float32 in parent order,
rounds to nearest-even bfloat16 or float32, and writes it under
`<model.root>/.merges/<hash of the merge>` with an origin BOM recording the
versioned method (`mean-f32-v1` or `weighted-mean-f32-v1`), every parent's
model ID, weights hash, and lineage node hash in order, the weights, and its
own lineage node. The same merge is built once. A compose with `base.merge` may
have no stages: the destination is then the merged model, servable, exportable,
and usable as another compose's base; with stages, the merge initializes
training like any origin. `model train` needs no training backend for a
merge-only compose.

## Consequences

- Anyone holding the parents can recompute a merge and compare its weights hash;
  two independent merges of the same parents agree byte for byte.
- A merge never invents a training run.
- Changing the arithmetic requires a new method name.
- Parents trained from random weights, or pulled without recorded lineage,
  cannot be merged, except copies of the same weights.
- A merge's record says whether its common ancestor rests on this store's own
  records or only on what pulled releases declared. A release can declare any
  parent; the record never presents that claim as shown.

Relates to [0026](0026-pinned-model-origin.md), [0051](0051-direct-compose-model-origins.md),
[0043](0043-unified-model-training.md), and [0066](0066-pin-trained-model-parents.md).
