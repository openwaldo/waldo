# Testing

Focused Go tests live beside their packages. Process-level tests live under
`testing/`. Python worker unit tests (stdlib `unittest`, no pip installs)
live under `testing/python/`.

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

For an explicitly CPU-only suite, use `WALDO_TEST_CPU_ONLY=1 ./testing/all.sh`.
This hides CUDA devices from the conformance test and skips accelerator lifecycle
scripts, including MLX. Skipped hardware coverage is not a hardware qualification.

## Fixed-global-batch FP32 qualification

`testing/training-fixed-batch.sh` exercises the production PyTorch worker's
sequence ownership, loss normalization, accumulation, and AdamW updates. It
compares every step with the independent functional oracle in
`testing/python/pytorch_conformance.py` and with the singleton worker baseline.
It does not replace CLI ingestion/export acceptance or test checkpoint resume.

Prepare a tiny synthetic fixture and initial FP32 Safetensors on CPU:

```bash
./testing/training-fixed-batch.sh prepare --output /path/to/new-fixture
./testing/training-fixed-batch.sh verify --fixture /path/to/new-fixture
./testing/training-fixed-batch.sh run \
  --fixture /path/to/new-fixture --output /path/to/new-cpu-results --device cpu
./testing/training-fixed-batch.sh verify-results \
  --fixture /path/to/new-fixture --output /path/to/new-cpu-results
```

Set `WALDO_PYTHON` to the explicit PyTorch interpreter. Preparation requires
PyTorch and NumPy but no GPU or external corpus. Reuse the staged fixture bytes
on other hosts; do not regenerate weights independently for each matrix cell.
`verify` uses only the Python standard library. `verify-results` recomputes
the oracle on CPU, checks inventory hashes and every required rank/cell, and
does not rewrite evidence. Hashes are integrity checks, not signer attestation.

The default eight-update fixture uses global batch eight, sequence length
eight, dropout disabled, and an eager FP32 model. Both all-valid and uneven
loss masks are tested. World sizes one, two, and four each run accumulation
one, and additionally accumulation four/world-size where distinct: ten cells.
The accumulation variant keeps per-rank microbatch two. The test records actual
sample ownership and rejects omissions, duplication, reordering, nonfinite
tensors, optimizer-step mismatches, and missing tensor names. Tensor tolerances
are fixed to the existing CUDA FP32 oracle thresholds in the test source.

CPU uses a test-only Gloo/DDP placement adapter because production distributed
placement requires CUDA. The adapter leaves Trainer's sampling and optimizer
methods intact. A CPU pass explicitly reports `cuda_qualified: false`.

Only after allocating otherwise-unused GPUs, run the CUDA matrix:

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID-1,GPU-UUID-2,GPU-UUID-3,GPU-UUID-4 \
WALDO_FIXED_BATCH_ALLOW_CUDA=1 \
./testing/training-fixed-batch.sh run \
  --fixture /path/to/new-fixture --output /path/to/new-cuda-results \
  --device cuda --strategy data-parallel
```

CUDA uses the actual TorchTitan Trainer constructor and NCCL. Four visible GPUs
are required for the full matrix. `fully-sharded-data-parallel` selects FSDP2
instead of DDP and requires the compatible TorchTitan runtime; report the two
strategies separately. CUDA, FSDP2, BF16, and resume coverage must not be inferred
from CPU results. Smaller `--world-sizes` matrices are diagnostic and cannot
produce a complete qualification receipt.

Each cell has a 600-second default timeout; the default aggregate CUDA budget
is two GPU-hours. Overrides are bounded to at most eight GPU-hours. A failed
cell stops the matrix and retains its log. Timeouts terminate only that test's
process group. Output directories must be new. Preserve the fixture manifest,
source revision and patch, exact runtime lock, per-rank tensor snapshots,
comparison reports, logs, `summary.json`, and `SHA256SUMS` together. Tests do not
reserve hardware, inspect serving workloads, modify drivers, or publish results.
Instrumentation copies tensors and changes timing: its telemetry is not a
performance benchmark. Durable storage and signer trust are operator choices.

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

## BF16 and committed-checkpoint resume qualification

`testing/training-precision-resume.sh` extends the fixed-batch fixture without
modifying the FP32 runner or production training code. Its `run` command accepts
`--mode bf16` or `--mode resume`, `--device cpu|cuda`, and the same explicit
fixture, new output directory, and DDP/FSDP2 strategy as the FP32 test. CUDA still
requires `WALDO_FIXED_BATCH_ALLOW_CUDA=1` and explicitly allocated visible GPUs.

```bash
./testing/training-precision-resume.sh run --mode bf16 --device cpu \
  --fixture /path/to/fixture --output /path/to/new-bf16-results
./testing/training-precision-resume.sh run --mode resume --device cpu \
  --world-sizes 1 --fixture /path/to/fixture --output /path/to/new-resume-results
./testing/training-precision-resume.sh verify \
  --fixture /path/to/fixture --output /path/to/results
```

The BF16 test confirms BF16 forward outputs and FP32 master weights, gradients,
and moments. At each update, logits/loss/gradients are compared with the
independent FP32 functional oracle at the same current weights. AdamW is
independently recomputed from its measured input gradients and prior moments,
using the existing strict FP32 optimizer tolerance. Comparing Adam updates
against a different FP32 gradient trajectory is unsound near zero gradients.
The original full FP32 oracle trajectory is also retained for investigation.

A separate topology gate compares complete BF16 trajectories to the singleton
baseline. Its proposed (not upstream-approved) atol/rtol pairs are frozen in
`POLICY`: logits .01/.02, loss .005/.002, gradients .002/.05, weights .002/.02,
first moments .0005/.05, second moments .00005/.05. Every tensor is checked.
Arithmetic conformance and topology drift are reported separately. Independent
matrix cells continue after a topology-only mismatch to characterize coverage;
an arithmetic failure stops the matrix. Any topology mismatch still fails the
qualification and returns nonzero. Thresholds must not be loosened after seeing
qualification failures. The verifier can authenticate a complete negative
verdict as well as a pass; it does not turn a negative verdict into a pass.

Resume tests use native Trainer checkpoint save/restore with dropout .1,
uneven masks, eight total updates, and checkpoint step four. For each FP32/BF16
and world-size combination, the runner executes an uninterrupted control,
then a separate prefix. Once every rank has saved its evidence and the native
checkpoint has been committed/fsynced, rank zero terminates itself with SIGKILL and torchrun shuts down its peers.
The runner verifies that no experiment workers remain before launching fresh
processes at the same topology. The
resume input retains the original total-step target and initialization, and
replays the original sample stream through the production cursor logic.

The test requires bitwise-equal continuation logits, gradients, weights,
optimizer state, and native final weights; exact restored/final CPU and CUDA
RNG states; and exact sample/corpus/token accounting. It checks FP32 and BF16
at one/two/four GPUs for each selected strategy. CPU resume tests use the native
serial CPU worker and support world size one only. This is committed optimizer-
step resume, not mid-microbatch, changed-world-size, CLI transaction, or audited
corpus/export acceptance. Prefix termination is an expected recorded event;
other process errors remain failures. Preserve checkpoints, receipts, logs,
source/policy hashes, and all failed attempts. Verification runs on CPU.
Completed resume comparison failures retain a negative verdict and allow the
remaining independent cells to run. Launch, checkpoint-interruption, or worker
cleanup errors still stop the matrix. The verifier recomputes each positive or
negative verdict; numerical thresholds and the bitwise resume requirement are
unchanged.
