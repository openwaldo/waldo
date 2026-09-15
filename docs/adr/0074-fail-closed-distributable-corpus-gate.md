# ADR 0074: Fail closed for distributable corpus claims

Status: accepted

## Context

A normalized license string is not enough to establish that WALDO may
redistribute normalized training shards. Aggregated datasets can contain
constituent terms, attribution duties, noncommercial restrictions, or missing
upstream evidence. The planned nanochat-class baseline must be distributable,
so silently accepting an unknown or convenient repository-level label is not
an option.

## Decision

Training parameters may select `distribution_policy: distributable`. WALDO
then applies a record-level filter backed by a conservative reviewed license
allowlist. Rows with noncommercial, unknown, custom, or no-redistribution
licenses are skipped while approved rows in the same corpus remain eligible.
Every source contributing eligible rows must carry upstream license evidence
plus a pinned version or lowercase SHA-256 source digest. The approved license
set and resulting notice, attribution, and share-alike obligations are pinned
as `distribution_review` in the run BOM and revalidated whenever the model is
inspected. A stage fails if the policy selects no eligible records.

This is an engineering policy, not legal advice. Adding a license to the
allowlist requires a reviewed code change and tests; a compose cannot override
the gate.

## Consequences

- Private training remains possible when the policy is omitted, but it cannot
  claim the distributable review.
- The holding nanochat-class compose selects the strict policy.
- Mixed-license corpora remain complete; distribution policy changes training
  selection rather than destructively rebuilding the corpus.
- Attribution and share-alike duties are machine-readable run facts rather
  than prose discovered after training.
