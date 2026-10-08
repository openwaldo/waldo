# ADR 0081: Version exact intrinsic-loss telemetry

Status: accepted
- Date: 2026-10-05

## Context

Mean token loss and perplexity cannot be compared when tokenizers differ.
WALDO already computes the summed held-out negative log likelihood and pins
the exact evaluated token and UTF-8 byte counts, but older workers retained
only the mean and perplexity. `TELEMETRY.csv` also has deployed schema-1 files
whose fixed header cannot be expanded in place.

## Decision

Every new held-out evaluation persists `heldout_nll_sum`,
`heldout_target_tokens`, `heldout_utf8_bytes`, and
`heldout_bits_per_byte = NLL / (ln(2) * bytes)` in addition to token loss and
perplexity. These are additive evaluation metrics, so existing schema-1 run
JSON remains readable.

New telemetry files use schema 2, identified by the `telemetry_schema` column,
and append the four exact intrinsic fields. When resuming a run whose telemetry
header is schema 1, WALDO validates that header and continues writing schema-1
rows with the original column count. It never mixes row widths or rewrites
historical telemetry. Unknown headers fail closed.

The MLX, PyTorch, and TorchTitan worker revision identities are incremented so
checkpoint resume cannot silently cross the metric-producing implementation
change.

## Consequences

- New runs support exact cross-tokenizer BPB calculations.
- Old telemetry remains appendable and readable without migration.
- BPB remains an intrinsic held-out-distribution metric, not a semantic
  capability score.
