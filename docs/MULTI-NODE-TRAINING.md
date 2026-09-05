# Multi-node TorchTitan training

WALDO supports homogeneous, non-elastic multi-node training through TorchTitan
and PyTorch FSDP2. This is model sharding across all ranks, not a scheduler.

## Cluster contract

Every host must have:

- Linux, the same WALDO binary, and compatible PyTorch and TorchTitan installs;
- the same number and class of visible GPUs;
- a shared, writable `model.root` mounted at the same absolute path;
- access to every corpus object and the same lookaside credentials; and
- unrestricted node-to-node traffic on the selected training interface.

The primary writes stage plans beneath
`<model.root>/.multinode/<rendezvous-id>/`. Secondary hosts read those plans,
independently materialize and verify the pinned corpus shards, reproduce the
held-out split, and join the TorchTitan rendezvous. The shared model root also
makes initialization weights available to every host. Lookaside cache and
scratch paths should be fast node-local storage; they do not need to be shared.

All hosts must expose the same GPU count. The current worker checks that local
world sizes agree before training. Use equivalent GPU models and memory sizes;
the run BOM records the primary host's detected accelerator topology as the
homogeneous topology for the cluster.

The compose `batch_size` is one logical batch processed by the globally sharded
model. It is not multiplied by the number of nodes.

## Configure each host

Use the same shared model path and backend everywhere. Give each host its own
local cache and scratch paths:

```console
waldo config set model.root /shared/waldo/models
waldo config set model.backend torchtitan
waldo config set lookaside s3://openwaldo/lookaside
waldo config set lookaside.cache /local/waldo/cache
waldo config set lookaside.scratch /local/waldo/scratch
```

When automatic NCCL interface selection is unsuitable, set the interface on
every host. Set the HCA only when using InfiniBand or RoCE and use the value
appropriate to that host:

```console
waldo config set model.nccl.interface ib0
waldo config set model.nccl.hca mlx5_0
```

The primary's rendezvous address must be reachable from every secondary. Allow
the rendezvous port and NCCL peer traffic through host and network firewalls.

## Start a run

For four hosts, choose one unique rendezvous ID. Start ranks 1 through 3 first,
one command on each secondary host:

```console
# host 1
waldo model train-worker --nodes 4 --node-rank 1 \
  --rendezvous train-0.example:29500 --rendezvous-id experiment-001 \
  --plan-wait 24h

# host 2
waldo model train-worker --nodes 4 --node-rank 2 \
  --rendezvous train-0.example:29500 --rendezvous-id experiment-001 \
  --plan-wait 24h

# host 3
waldo model train-worker --nodes 4 --node-rank 3 \
  --rendezvous train-0.example:29500 --rendezvous-id experiment-001 \
  --plan-wait 24h
```

Then start rank 0 on the primary host:

```console
waldo model train my-model /path/to/compose.yaml \
  --nodes 4 \
  --rendezvous train-0.example:29500 \
  --rendezvous-id experiment-001
```

The values of `--nodes`, `--rendezvous`, and `--rendezvous-id` must match on
every host. Node ranks must be unique and cover `0..nodes-1`; `model train`
owns rank 0. A multi-stage compose needs only these commands: each secondary
waits for and joins every stage in order.

The primary resolves and verifies every stage before publishing the first
plan. Set `--plan-wait` long enough for that preflight; the default is 24 hours.
Secondaries validate their TorchTitan runtime immediately, before waiting for a
plan or downloading corpus objects.

## Failure and restart behavior

This first implementation is deliberately non-elastic. If any host exits, stop
the remaining commands. WALDO does not currently resume a multi-node run from
its distributed checkpoint. Remove the incomplete model explicitly or choose a
new model name, then start every host again with a fresh rendezvous ID.

The primary owns the model record, run BOM, telemetry, checkpoints, and terminal
Safetensors. Secondary hosts never author model lifecycle records. A successful
stage removes its temporary plan handoff.

## Acceptance test

Before a long run, use a tiny compose with two optimizer steps, checkpoint and
evaluation intervals of one, and disposable model and rendezvous names. Verify:

1. every host reports that its process group joined;
2. the run BOM records the intended node count and total world size;
3. the primary writes two complete checkpoints and terminal Safetensors;
4. every secondary exits successfully; and
5. `<model.root>/.multinode/<rendezvous-id>` is absent after completion.

The repository's hardware test validates this lifecycle on one host with two
GPUs. A real two-host smoke test must additionally confirm the shared mount,
rendezvous route, firewall, and NCCL transport.
