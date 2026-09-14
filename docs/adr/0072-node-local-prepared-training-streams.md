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
to ranks on the same node. Sequence ordinals deterministically assign each
sequence to exactly one global rank and are reconstructed from the beginning
when resuming.

The launcher-stream compatibility path remains `rank-zero-broadcast` because
those remote workers currently receive a plan without object-store
credentials. The selected data plane is pinned in execution provenance; it is
not allowed to change during resume.

## Consequences

- Training records no longer cross the inter-node collective network in the
  default shared-plan path.
- Corpus objects are mirrored once per node, never once per rank. Ordinary NFS
  is unnecessary for training data; it may still carry the small shared plan.
- Every node currently repeats deterministic tokenization and packing. A later
  content-addressed prepared-stage cache can remove that remaining CPU work
  without changing rank assignments or the worker protocol.
- Launcher-stream runs remain correct but retain the old data-plane bottleneck
  and must identify it in their run BOM and efficiency report.
