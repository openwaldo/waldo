# ADR 0075: Keep evaluation and contamination gates immutable

Status: accepted

## Context

Training held-out loss proves only that the training objective generalizes to a
small deterministic sample. It does not establish benchmark capability, prove
that benchmark rows stayed out of training, or define a promotion threshold.
Loading an unversioned benchmark bundle at evaluation time also makes a model
result impossible to reproduce or audit.

## Decision

An evaluation is identified by an `openwaldo-evaluation-bom` that pins its
complete corpus OpenWALDO BOM, task, split, metrics, thresholds, and
contamination policy. Evaluation records remain outside training selections.

Before evaluation, WALDO streams the training corpus once and compares it with
the bounded evaluation set. Normalized full-record hashes detect exact overlap.
Word-shingle Jaccard similarity detects fuzzy overlap. The implementation
indexes only evaluation shingles, not the training corpus, so memory use is
bounded by the evaluation set. The resulting report pins both BOM identities,
every detected pair, the pass decision, and its own digest.

Promotion fails closed when contamination exceeds its declared limit, a metric
is missing or non-finite, or any minimum/maximum threshold is missed.

## Consequences

- Held-out loss remains useful training telemetry but cannot promote a model.
- Benchmark sources need the same revision, license, and object verification as
  training sources.
- Exact and fuzzy policy changes create a different evaluation definition.
- Model lifecycle and CLI integration can consume this package without making
  the training adapter aware of index layout or benchmark semantics.
