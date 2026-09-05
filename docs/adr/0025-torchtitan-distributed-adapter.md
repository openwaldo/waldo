# ADR 0025: Use TorchTitan for distributed training

Status: accepted

## Decision

TorchTitan is the distributed Linux adapter. It requires a compatible
TorchTitan and PyTorch installation and visible CUDA or ROCm devices. WALDO
records every worker rank in the execution topology.

Single-node execution launches one local rank per visible GPU. Multi-host
execution uses a hostfile containing only host names. Rank 0 stages its exact
binary and launches secondary workers through non-interactive SSH. It verifies
a homogeneous runtime, GPU count, and accelerator topology before materializing
data. Manual node-rank and rendezvous flags remain a compatibility interface.

Rank zero alone reads and tokenizes WALDO's deterministic worker stream and
broadcasts compact token frames through NCCL. Secondary hosts do not need the
index, corpus objects, lookaside credentials, shared model storage, or NFS.
FSDP2 shards model state. Rank zero commits checkpoints, terminal Safetensors,
and observations.

Multi-node interrupted runs cannot currently resume. They must be restarted
with the same topology and a fresh run.
