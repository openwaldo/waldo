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
deterministic sample balanced across selected corpus paths, and creates a content-identified
`waldo/byte-bpe` artifact.

The trainer makes one pass over the sample to count bounded, whitespace-aware
byte chunks, retains the most frequent 100,000 unique chunks as a weighted
training vocabulary, and learns ordered byte-pair merges with deterministic
frequency and token-ID tie breaks. Pair statistics are updated only for chunk
types affected by each merge rather than rescanning the complete sample. All
256 bytes remain fallback tokens. Encoding replays the learned merge ranks and
therefore round-trips arbitrary bytes without unknown tokens. The command
reports bytes per token and token counts for both the candidate and
`r50k_base` on the identical sample.

The artifact is not selected by a model merely because it exists. Its revision
must pass domain-specific compression and round-trip checks before a compose
adopts it through `tokenizer.artifact_path`. Compose loading validates the
artifact and embeds its complete content in the immutable model architecture,
run plan, and emitted tokenizer artifact.

A model compose may also declare `architecture.tokenizer.training`. This is a
pre-model compose phase: forecast validates it without materialization, while
training resolves the pinned corpus BOM, creates the artifact on rank 0,
enforces the declared token-inflation bound, and only then persists model identity
or distributes a multi-host plan. The resolved architecture retains both the
training declaration and resulting artifact.

Schema-2 `waldo/byte-bpe` artifacts contain ordered merge rules. WALDO retains
a read-only codec for existing schema-1 `waldo/bytepiece` artifacts, but the
CLI and compose phase no longer create them. Compose training uses
`max_token_inflation`, measured directly as candidate tokens divided by r50k
tokens on the same sample minus one.

## Consequences

- Tokenizer training is bounded by `--sample-bytes` and a fixed 100,000-type
  weighted training vocabulary; it does not perform thousands of complete
  corpus rescans.
- The exact corpus BOM, sample identity, ordered BPE merges, and special-token
  IDs are immutable artifact facts.
- Secondary hosts and later inference receive the embedded tokenizer; they do
  not rely on the original machine-local `artifact_path`.
- Candidate tokenizers still require capability comparison before promotion.
