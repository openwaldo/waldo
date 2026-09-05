# Multi-host TorchTitan training

WALDO supports homogeneous, non-elastic multi-host training through TorchTitan
and PyTorch FSDP2. The normal interface is one rank-0 command with a hostfile.

## Hostfile

The hostfile contains one SSH host per line. Empty lines and `#` comments are
ignored. The first host is rank 0 and must be the host running the command:

```text
train-0
train-1
train-2
train-3
```

Hostfile entries contain no slots or GPU counts. WALDO uses every visible GPU
and rejects hosts whose GPU count, model, memory, Python, PyTorch, or TorchTitan
versions differ. Names may be SSH aliases, but the first name must also be
resolvable by every training host as the rank-0 rendezvous address.

`--hostfile` selects TorchTitan. It accepts `model.backend=auto` or
`model.backend=torchtitan` and rejects explicit MLX, PyTorch, or fake backends.

## Host requirements

Every host needs:

- Linux with compatible GPU drivers, PyTorch, TorchTitan, and NCCL;
- passwordless, non-interactive SSH from rank 0;
- the same number and class of visible GPUs; and
- unrestricted node-to-node traffic on the selected training interface.

WALDO copies the exact rank-0 binary to a SHA-256-addressed directory under
`/tmp/waldo-launch/` on each secondary and verifies it before use. WALDO does
not install or modify Python, PyTorch, TorchTitan, GPU drivers, or NCCL.

NFS and shared paths are not required. Rank 0 alone uses the index, lookaside
credentials, corpus cache, and durable `model.root`. Secondary hosts use only
launcher-managed local scratch.

Rank 0 resolves, verifies, filters, orders, and tokenizes the corpus. It sends
compact token frames and masks through the NCCL process group; raw Parquet
objects are not copied to secondary hosts. Every rank participates in the
globally FSDP2-sharded model step. The compose `batch_size` remains one logical
batch and is not multiplied by the host count.

## Network configuration

When automatic NCCL interface selection is unsuitable, configure rank 0. The
launcher passes the resolved settings to every worker:

```console
waldo config set model.nccl.interface ib0
waldo config set model.nccl.hca mlx5_0
```

The HCA setting is only appropriate for InfiniBand or RoCE. Allow the selected
rendezvous port and NCCL peer traffic through every firewall.

## Start a run

Run one command on the first host:

```console
waldo model train my-model /path/to/compose.yaml --hostfile /path/to/hosts
```

The default rendezvous port is 29500. Override it when necessary:

```console
waldo model train my-model /path/to/compose.yaml \
  --hostfile /path/to/hosts --rendezvous-port 29600
```

WALDO performs all host and runtime checks before corpus materialization. It
then launches and supervises every secondary, relays their output with host
labels, and publishes each compose stage directly over the launcher channel.

## Failure behavior

This implementation is deliberately non-elastic. A secondary failure cancels
rank 0; a rank-0 failure terminates the SSH workers. Multi-host checkpoint
resume is not yet supported. Remove an incomplete model explicitly or choose a
new model name, then restart with a fresh command.

Rank 0 owns the model record, run BOM, telemetry, checkpoints, and terminal
Safetensors. Secondary hosts never author durable lifecycle records.

## Acceptance test

Before a long run, use a disposable model and a tiny compose with two optimizer
steps and checkpoint/evaluation intervals of one. Verify:

1. every host passes topology preflight and reports its process group joined;
2. the run BOM records the intended node count and global world size;
3. rank 0 writes both checkpoints and terminal Safetensors;
4. every secondary exits successfully; and
5. no secondary downloads a corpus object.

The repository's opt-in multi-node hardware test exercises rendezvous and FSDP2
on two GPUs in one Linux host. A real hostfile smoke test additionally validates
SSH launch, routing, firewall, and inter-host NCCL transport.
