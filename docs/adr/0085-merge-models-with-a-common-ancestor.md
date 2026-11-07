# ADR 0085: Merge models of one family into a merged origin

Status: accepted
- Date: 2026-11-05

## Context

Averaging the weights of models fine-tuned from the same starting weights is
an established way to combine what each learned. Done outside WALDO, the
merged model loses its provenance: nothing records which models went into it,
or how. A merge has no training run either, and ADR 0026 forbids inventing one.

## Decision

A compose can start from a merge of two or more models instead of one base.
WALDO averages their weights into a new origin, so a merged model starts like
any other, and records how it was made.

- WALDO finds where the parents' histories meet (ADR 0084), as git finds a
  merge-base. It refuses the merge if any step above that point is a gap, or
  starts from random weights or a release pulled without its history.
- `MERGE.json` (`waldo-model-merge`, schema 1) records the method and each
  parent's run or origin, weights hash, and `share` (its weight). It names no
  models or places; licenses come from the parents' BOMs in their history.
- The merged origin is kept under `<model.root>/.merges`. Its `LINEAGE.json`
  holds `MERGE.json` and every parent's history, so the parents can be deleted.
- A compose with only a merge needs no training; with stages, training starts
  from the merged weights.

## Consequences

- Parents recorded the same way, by the same run or origin BOM and weights,
  give the same `MERGE.json` and the same weights, so independent merges of
  them compare byte for byte.
- A merge never invents a training run; changing the arithmetic needs a new
  method name.
- Earlier merges in a history are checked by their documents, not recomputed.
- A merged model needs its `.merges` entry to be composed again.
