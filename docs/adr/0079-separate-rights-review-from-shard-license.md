# ADR 0079: Separate rights review from immutable shard license assertions

Status: accepted

## Context

Canonical Parquet shards embed the license asserted when they were built. A
later audit can discover that an aggregation is mixed, that a package license
does not cover individual contents, or that attribution was not preserved.
Rewriting the Git manifest's shard license would then contradict the immutable
embedded BOM without changing the object.

## Decision

Corpus manifests may add `rights_review` with an ISO date, a reason, HTTPS
evidence URLs, and separate statuses for training, canonical-corpus
redistribution, and model-weight distribution. The field is copied into the
OpenWALDO corpus BOM. A distributable review fails unless every present status
is `approved`.

The allowed status values are `approved`, `private-research`, `unresolved`, and
`excluded`, except corpus redistribution does not use `private-research`.
Absence retains compatibility with previously reviewed manifests and leaves
the existing SPDX allowlist and source-evidence checks in force.

`rights_review` does not alter or excuse an embedded shard license. A corrected
effective license still requires reingestion and new content-addressed shards.

## Consequences

- A rights audit can immediately block a misleading distributable claim without
  corrupting immutable shard provenance.
- Training, corpus redistribution, and model-weight distribution are explicit
  independent conclusions.
- New `LicenseRef-*` identifiers remain fail-closed and cannot be approved only
  by writing a manifest field.
