# ADR 0080: Support a pinned contiguous-tail evaluation view

Status: accepted

## Context

Some reference language-model corpora are intentionally one continuous text
stream. Canonical ingestion correctly preserves such a source as one record,
but record-level held-out selection cannot reserve validation text without
removing the entire training stream. Repackaging the stream as lines would add
artificial EOS boundaries and change the training problem.

## Decision

Compose parameters may select `contiguous-tail-v1` evaluation. The policy is
limited to causal language modeling with the shuffled profile and exactly one
eligible canonical text record. WALDO places the split at the UTF-8 boundary
nearest the leading `1 - evaluation_fraction` byte fraction, trains on the
prefix, and evaluates the suffix.

Stage preflight pins the canonical row identity and byte offset. Its evaluation
digest includes both values. Reconstruction verifies the offset and reproduces
the same two views without rewriting the indexed record. The suffix must fit
the declared evaluation byte limit. Continuous packing adds EOS only at the
end of each view, never at source lines or internal storage boundaries.

## Consequences

Tiny Shakespeare and similar single-stream controls can retain their natural
ordering and still produce an honest held-out loss. The policy is deliberately
not a general multi-record splitting or sampling mechanism. Existing
record-level policies and preflight artifacts remain valid.
