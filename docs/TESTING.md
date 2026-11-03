# Testing

Focused Go tests live beside their packages. Process-level tests live under
`testing/`. Python worker and environment tests use stdlib `unittest` under
`testing/python/`. Environment tests also require pip and venv (see below).

## Local checks

```bash
./testing/unit.sh
./testing/vet.sh
./testing/docs.sh
./testing/python.sh
```

Run the complete local suite with:

```bash
./testing/all.sh
```

Run the independent PyTorch numerical conformance gate with:

```bash
./testing/training-conformance.sh
```

This loads WALDO's production shared PyTorch model and compares it with a
separate functional oracle for logits, masked causal loss, every gradient, one
AdamW update and optimizer state, saved/reloaded logits, and next-token argmax.
It always runs the FP64 CPU comparison and additionally runs FP32 CUDA when a
CUDA device is available. The test requires PyTorch when invoked directly;
`testing/all.sh` reports a skip on development hosts without PyTorch.

The complete suite runs unit tests, static analysis, ingestion lifecycles, the
structured-conversation ingestion and training lifecycle, the general fake
model lifecycle, and hardware-dependent MLX, PyTorch, and TorchTitan
lifecycles. Hardware tests report a skip when their runtime or device is not
available.

Individual end-to-end tests are available under `testing/e2e/`:

```bash
./testing/e2e/ingest-direct.sh
./testing/e2e/structured-conversation.sh
./testing/e2e/model-fake.sh
./testing/e2e/model-mlx.sh
./testing/e2e/model-pytorch.sh
./testing/e2e/model-pytorch-memorization.sh
./testing/e2e/model-torchtitan.sh
./testing/e2e/model-torchtitan-multinode.sh
./testing/e2e/model-external-trainer.sh
```

`model-torchtitan-multinode.sh` is opt-in and uses two GPUs on one Linux host
to exercise the internal node-rank compatibility path without needing two
machines:

```bash
WALDO_E2E_MULTINODE=1 ./testing/e2e/model-torchtitan-multinode.sh
```

It is not a substitute for the two-host acceptance test in
[Multi-host training](MULTI-NODE-TRAINING.md), which additionally exercises
hostfile parsing, SSH staging, routing, firewall, and inter-host NCCL.

Before a production multi-host run, execute the required acceptance gate with
the real hostfile and a deliberately small existing structured-conversation
corpus:

```bash
./testing/training-acceptance.sh \
  --hostfile ~/hostfile \
  --corpus post-train/sft/waldo-project-v1
```

The hostfile test uses the configured index, lookaside, caches, credentials,
NCCL settings, and model root. It creates a uniquely named two-stage model with
10 optimizer steps per stage,
removes it after success, and preserves it plus its temporary compose after a
failure. It does not ingest or publish test corpus objects.

The acceptance script runs numerical conformance before it starts any training
lifecycle, so a reference-math failure stops the GPU tests immediately. It then
runs a disposable compose-driven memorization control through the normal
`waldo model train` interface. That control must reduce held-out loss by at
least 90%, reach a held-out loss no greater than 0.25, reproduce the fixed
held-out continuation, and terminate it with EOS. This distinguishes a backend
that can merely execute from one that can demonstrably learn and publish the
learned weights.

## External trainer end-to-end

`testing/e2e/model-external-trainer.sh` drives the real `waldo` CLI with the
pinned Qwen3 reference trainer (`examples/trainers/qwen3-hf/compose.tiny.yaml`).
It is opt-in because it clones the pinned commit and installs locked wheels
through pip (Python 3.12.3 with pip and venv; `WALDO_APPLICATION_PYTHON`
selects it, `WALDO_APPLICATION_CACHE` keeps the downloads between runs):

```bash
WALDO_E2E_EXTERNAL=1 ./testing/e2e/model-external-trainer.sh
```

Each variant ingests a disposable corpus, trains, interrupts after the first
durable checkpoint, resumes with the same command, and asserts byte-identical
weights against an uninterrupted run, a no-op completed rerun, `model continue`,
a verified `hf-causal-lm` export, and signing dispatch (a mock `cosign`; this
is not a signature qualification). The script runs:

- **CPU** (`env/cpu`): always, for hosts and CI without a GPU.
- **GPU** (`env/cuda`, same compose with the environment rewritten): only when
  `nvidia-smi -L` lists a GPU. Otherwise it prints that the GPU run was skipped,
  which is not a pass.
- **~280M GPU** (`compose.yaml`), when `WALDO_E2E_EXTERNAL_LARGE=1`: one
  budget-capped run (10 steps, sequence length 512, global batch 8) that prints
  wall time and sampled host memory in use (GB10 unified memory; `nvidia-smi`
  reports N/A).
- the existing schema-1 lifecycle (`model-fake.sh`) to show native behavior is
  unchanged. This step needs Go on the host.

The script builds `waldo` with `go build` unless `WALDO_BIN` names a binary.
Run a single variant directly with
`python3 testing/external-qwen3/cli-smoke.py --help`.

## Live tests

Live tests are never run by `testing/all.sh`. Their environment variables are
deliberate write or network authorization.

Audit a small corpus from a local public-index checkout:

```bash
WALDO_LIVE_ALLOW_PUBLIC_INDEX=1 \
./testing/live/public-index-audit.sh /path/to/waldo-index/core/example
```

Exercise S3 ingestion only against a disposable `waldo-e2e` prefix with a
lifecycle policy:

```bash
WALDO_LIVE_ALLOW_S3=1 \
WALDO_E2E_AWS_REGION=us-west-2 \
./testing/live/s3-ingest.sh s3://example-test-bucket/waldo-e2e
```

Credentials come from the AWS SDK chain or `waldo lookaside login`; tests must
not write credentials to fixtures.

External environment tests require host Python 3.11+ with `venv` and pip 22.3+.
They install locally generated test wheels through pip into unseeded temporary
venvs; they require no package-index access or GPUs. The standard Python suite
includes these tests and fails (rather than skips) when tooling is missing.
The Go `pythonenv` integration test exercises the embedded helper through its
public API. On Debian test images install `python3-pip` and `python3-venv`.
