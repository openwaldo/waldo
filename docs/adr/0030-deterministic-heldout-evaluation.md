# ADR 0030: Pin a bounded deterministic held-out set

Status: accepted

## Decision

Training selects held-out records through a named deterministic policy. The
default indexed policies derive a bounded sample of row positions from the
pinned shard counts and seed, apply record filters to those candidates, and
read only the selected complete records. The stratified form distributes
candidates across logical corpus paths. The target is one percent of declared
records, capped at 256 records and 1 MiB of source content while leaving at
least one declared training record.

The original `lowest-sha256-v1` policies remain available for exact selection
by the lowest seeded SHA-256 scores over every eligible canonical shard and row
identity. They require a complete eligible-row scan.

The run BOM pins the selection algorithm, seed, record count, selected-model
token targets, content bytes, and digest of the ordered identities. Selected
records are excluded from every training epoch.

The seed that selects held-out records is the training seed unless a stage
declares `evaluation_seed`, so runs that shuffle differently can hold out the
same records and compare their held-out loss. The run BOM pins whichever seed
selected them.

Evaluation uses the model's configured tokenizer and per-record EOS packing.
The worker records no-gradient held-out loss and perplexity at the resolved
cadence.
