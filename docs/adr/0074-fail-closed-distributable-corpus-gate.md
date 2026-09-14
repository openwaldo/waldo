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
then requires every effective corpus license to be on a conservative reviewed
allowlist and every manifest source to carry upstream license evidence plus a
pinned version or lowercase SHA-256 source digest. Noncommercial, unknown,
custom, and no-redistribution licenses fail before preflight or accelerator
selection. The approved license set and resulting notice, attribution, and
share-alike obligations are pinned as `distribution_review` in the run BOM and
revalidated whenever the model is inspected.

This is an engineering policy, not legal advice. Adding a license to the
allowlist requires a reviewed code change and tests; a compose cannot override
the gate.

## Consequences

- Private training remains possible when the policy is omitted, but it cannot
  claim the distributable review.
- The holding nanochat-class compose selects the strict policy.
- The current Gutenberg and Wikimedia index entries are excluded from that
  compose until their upstream and per-record rights evidence is repaired.
- Attribution and share-alike duties are machine-readable run facts rather
  than prose discovered after training.
