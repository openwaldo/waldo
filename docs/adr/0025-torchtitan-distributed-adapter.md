# ADR 0025: Use TorchTitan for distributed training

Status: accepted

## Decision

TorchTitan is the distributed Linux adapter. It requires a compatible
TorchTitan and PyTorch installation and visible CUDA or ROCm devices. WALDO
records every worker rank in the execution topology.

Single-node execution launches one local rank per visible GPU. Multi-node
execution uses `--nodes`, `--rendezvous`, and `--rendezvous-id`; secondary nodes
join with `waldo model train-worker`. The primary owns the run BOM and publishes
a verified per-stage handoff plan for secondaries through a writable model root
mounted at the same absolute path on every host. Corpus caches and training
scratch remain node-local. The initial contract requires a homogeneous GPU
count and accelerator topology across hosts.

Rank zero receives WALDO's deterministic worker stream and broadcasts it.
FSDP2 shards model state. Rank zero commits checkpoints, terminal Safetensors,
and observations.

Multi-node interrupted runs cannot currently resume. They must be restarted
with the same topology and a fresh run.
