# ADR 0078: Cache versioned prepared-sequence chunks per node

Status: accepted

## Context

Node-local preparation removes rank-zero and inter-node data bottlenecks, but a
retry still repeats Parquet decoding, record transforms, tokenization, packing,
and rank assignment on every node. Large stages can waste substantial CPU and
delay accelerators even though all preparation inputs are immutable.

## Decision

For the node-local data plane, each coordinator starts an independent prepared
stream builder while the ordinary on-the-fly stream begins GPU training. The
builder writes schema-3 compact binary chunks capped at 16 MiB, so preparation
and accelerator work overlap instead of placing a preprocessing barrier before
training. The cache identity hashes the architecture, objective, conversation
transform, tokenizer, complete corpus BOM, resolved parameters, topology, and
held-out selection. Its manifest additionally binds the exact immutable Run
BOM SHA-256 used by the attempt.

Each node stores only the sequences assigned to its local ranks. Token IDs use
fixed-width binary integers, masks are bit-packed, and corpus consumption is
canonicalized. A manifest pins topology, global micro-batch, sequence count,
chunk order, first and last global sequence ordinal, byte sizes, and every
chunk SHA-256. Publication is directory-atomic. Resume calculates the exact
ordinal as `checkpoint step × global batch size`, skips chunks ending before
that ordinal without decoding them, and verifies every suffix chunk before its
bytes reach a worker. It also verifies ordering, ownership, tensor dimensions,
per-corpus consumption, and micro-batch boundaries. Corruption fails closed.

Schema-1 and schema-2 NDJSON caches remain readable for compatibility. New
caches use the compact indexed format and the verified Run BOM is the semantic
authority; the cache remains a replaceable derivative optimization.

The primary and secondary nodes use their configured lookaside scratch area.
Launcher-stream compatibility runs do not have corpus objects and retain
rank-zero broadcast, so they do not create these artifacts.

## Consequences

- First attempts retain streaming behavior and do not wait for full preparation;
  cache construction runs concurrently and reports when direct resume is ready.
- Checkpoint resume seeks directly to its indexed suffix instead of tokenizing,
  packing, and discarding the already-trained prefix.
- Any corpus, recipe, tokenizer, evaluation split, architecture, or topology
  change selects a different cache identity.
- Chunks remain an internal optimization; immutable corpus and run BOMs remain
  the reproducibility authority.
