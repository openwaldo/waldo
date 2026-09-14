# ADR 0072: Prepare and partition training streams once per node

Status: accepted

## Context

TorchTitan previously made global rank zero decode every JSON record and
broadcast it to every rank over the training process group. That serialized
input behind one CPU and placed corpus traffic on the same inter-node network
used by gradient collectives. Secondary nodes already downloaded and verified
the corpus into node-local cache, but their workers did not consume it.

## Decision

The normal shared-plan multi-node path materializes each content-addressed
corpus shard once in each node's existing verified cache. One WALDO
coordinator per node tokenizes and packs the identical canonical stream, emits
only the sequences owned by ranks on that node, and sends explicit global
micro-batch boundaries. Local rank zero distributes that prepared stream only
to ranks on the same node. A bounded record queue overlaps canonical decoding
and tokenization with worker-protocol encoding. CUDA workers stage each local
micro-batch in pinned host memory and use non-blocking device transfers.
Sequence ordinals deterministically assign each sequence to exactly one global
rank and are reconstructed from the beginning when resuming.

The hostfile launcher passes the primary's absolute cache root, scratch root,
byte limit, and configured mirrors to every remote coordinator. Launcher plans
therefore use the same `node-local-cache` data plane. Initialization and resume
artifacts are staged and verified separately. The selected data plane is pinned
in execution provenance; it is not allowed to change during resume.

## Consequences

- Training records no longer cross the inter-node collective network in the
  default shared-plan path.
- Corpus objects are mirrored once per node, never once per rank. Ordinary NFS
  is unnecessary for training data; it may still carry the small shared plan.
- On the first attempt, every node deterministically tokenizes and packs its
  local share while training consumes it. ADR 0078 adds bounded verified chunks
  so an identical retry can replay that work without reopening the shards.
- Hostfile runs no longer make rank zero a corpus decoding, tokenization, or
  inter-node record-broadcast bottleneck.
