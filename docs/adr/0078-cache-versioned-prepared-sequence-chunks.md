# ADR 0078: Cache versioned prepared-sequence chunks per node

Status: accepted

## Context

Node-local preparation removes rank-zero and inter-node data bottlenecks, but a
retry still repeats Parquet decoding, record transforms, tokenization, packing,
and rank assignment on every node. Large stages can waste substantial CPU and
delay accelerators even though all preparation inputs are immutable.

## Decision

For the node-local data plane, each coordinator streams its prepared sequences
to the worker and simultaneously writes schema-1 cache chunks capped at 16 MiB.
The cache identity hashes the architecture, objective, conversation transform,
tokenizer, complete corpus BOM, resolved parameters, topology, and held-out
selection. Run identity is excluded so an otherwise identical retry can reuse
the work.

Each node stores only the sequences assigned to its local ranks. A manifest
pins topology, global micro-batch, sequence count, chunk order, byte sizes, and
every chunk SHA-256. Publication is directory-atomic. Replay verifies the
manifest, every chunk hash and size, sequence ordering and ownership, tensor
dimensions, per-corpus consumption, and micro-batch boundaries before feeding
the worker. Corruption fails closed instead of silently rebuilding or training
on changed bytes.

The primary and secondary nodes use their configured lookaside scratch area.
Launcher-stream compatibility runs do not have corpus objects and retain
rank-zero broadcast, so they do not create these artifacts.

## Consequences

- First attempts retain streaming behavior and do not wait for full preparation.
- Identical retries avoid Parquet decoding and tokenization CPU work.
- Any corpus, recipe, tokenizer, evaluation split, architecture, or topology
  change selects a different cache identity.
- Chunks remain an internal optimization; immutable corpus and run BOMs remain
  the reproducibility authority.
