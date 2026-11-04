# ADR 0090: Publish training provenance across a model's lineage

Status: accepted
- Date: 2026-11-04

## Context

A model's run BOMs pin the data it trained on, but they stay on the trainer's
disk, and a merge has no runs of its own. Weights that pass through several
models, such as rounds of distributed training that merge and restart every
round, lose the trail from released weights back to their data, so the EU
disclosure and the distributable licence review cover only a model's own runs.
Lineage nodes (ADR 0084) name the run BOMs behind each set of weights by hash;
this decision publishes them.

## Decision

Releases publish the evidence beside the weights. A package includes the run
BOMs behind its weights as `provenance-<sha256>.json`, named by the hash of the
run BOM's canonical JSON, which is how models already pin runs: a trained
model's own runs, a merged model's parents' runs, or a pulled release's runs.
A pulled release keeps its run BOMs only when its node names a run, so it came
from a trained model, which publishes every run it trained; older runs stay
published by the releases they came from. A package also includes
`TRAINING-CLOSURE.json`: every distinct corpus BOM the lineage trained on, with
use counts, and whether that history is complete. It is keyed by corpus BOM, so
it grows with the corpora a lineage uses rather than its length. Pulls verify
and keep both with the origin, checking each run BOM against the hash it is
named by and refusing a closure that is malformed or leaves out the release's
own runs; merges keep their parents' run BOMs and the union of their closures.

Disclosure and attribution cover the whole lineage. `EU-BOM.json` adds the
inherited corpora and reports unrecorded history as a gap. `ATTRIBUTION.md` and
its distributable review take the inherited corpus BOMs too, so a
non-distributable corpus anywhere in a lineage blocks a distributable release.
`waldo model gate` scans every corpus in the closure for contamination, and
refuses a model whose history is incomplete.

Run BOMs record the Python runtime by executable name and version only, never
an absolute local path, because they are now published.

## Consequences

- Any released model answers which data trained it, and under what licences,
  from published artifacts: its weights, its run BOMs, their corpus BOMs, and
  the index commit and shards they pin.
- A merge's package proves its round on its own, so contributors' submission
  packages can be pruned after a round closes.
- Records stay bounded: a package republishes one generation of run BOMs, and
  a closure grows only with distinct corpora.
- History recorded before lineage is reported as incomplete where it matters.

Builds on [0084](0084-record-model-lineage.md) and
[0085](0085-merge-models-with-a-common-ancestor.md); amends
[0022](0022-model-release-exports.md); relates to
[0011](0011-model-plan-and-boms.md) and
[0074](0074-fail-closed-distributable-corpus-gate.md).
