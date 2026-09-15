# ADR 0077: Train tokenizers from pinned corpus samples

Status: accepted

## Context

Using a large generic tokenizer wastes embedding parameters in compact models,
while adopting an unpinned locally trained tokenizer makes the model impossible
to reproduce. Conventional vocabulary training can also waste substantial CPU
by rescanning the complete corpus once per merge.

## Decision

`waldo model train-tokenizer` accepts an index selection only after the strict
distributable review passes. It pins the corpus BOM digest, takes a bounded
deterministic record sample, and creates a content-identified
`waldo/bytepiece` artifact.

The trainer makes one pass over the sample, counts UTF-8 lexical runs, orders
pieces by frequency with deterministic length and lexical tie breaks, and
retains all 256 bytes as fallback tokens. Encoding uses longest-prefix matching
and therefore round-trips arbitrary bytes without unknown tokens. The command
reports bytes per token for both the candidate and `r50k_base` on the identical
sample.

The artifact is not selected by a model merely because it exists. Its revision
must pass domain-specific compression, round-trip, and downstream G1/G2 gates
before a compose adopts it.

## Consequences

- Tokenizer training is bounded by `--sample-bytes` and does not perform tens
  of thousands of complete corpus rescans.
- The exact corpus BOM, sample identity, ordered vocabulary, and special-token
  IDs are immutable artifact facts.
- The `conversation2` compose retains `r50k_base` until the admitted corpus is
  finalized and the candidate wins its promotion tests.
