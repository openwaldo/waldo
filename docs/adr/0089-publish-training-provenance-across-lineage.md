# ADR 0089: Publish training provenance across a model's lineage

Status: accepted
- Date: 2026-11-04

## Context

A model trained from other models, by fine-tuning or merging, also carries what
they were trained on. Anyone using it needs to know all of its training data
and licences, not only what the last training step used. WALDO answered for
the last step only, and nothing for a merged model.

Where part of that history is missing, the answer must say so rather than look
complete.

## Decision

A release publishes, beside its weights:

- `provenance-<sha256>.json`: each run BOM behind the weights, named by its
  hash. A trained model's own runs; a merge's parents' runs.
- `TRAINING-CLOSURE.json`: every corpus in the history, with use counts and
  whether the history is complete.

Pulls verify and keep both; merges keep their parents'.

`EU-BOM.json`, `ATTRIBUTION.md` with its distributable review, and
`waldo model gate` cover every corpus in the history. A non-distributable
corpus anywhere blocks a distributable release, and the gate refuses an
incomplete history.

Run BOMs record the Python runtime by name and version, not by path, since they
are now published.

## Consequences

- A released model shows which data trained it, and under what licences, from
  its published files.
- A merge's package proves its round on its own, so submission packages can be
  pruned after a round closes.
- History recorded before ADR 0084 is reported as incomplete.

Builds on [0084](0084-record-model-lineage.md) and
[0085](0085-merge-models-with-a-common-ancestor.md); amends
[0022](0022-model-release-exports.md); relates to
[0011](0011-model-plan-and-boms.md) and
[0074](0074-fail-closed-distributable-corpus-gate.md).
