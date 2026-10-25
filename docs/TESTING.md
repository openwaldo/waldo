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

### Transformers experiment

The stdlib Python suite tests packing/loss masks and package-tampering rejection
without installing Transformers. The real Go lifecycle test is opt-in:

```bash
WALDO_TRANSFORMERS_PYTHON=/absolute/path/to/venv/bin/python \
WALDO_TRANSFORMERS_WHEEL=/absolute/path/to/transformers-5.16.1-py3-none-any.whl \
go test ./internal/model -run TestTransformersRealLifecycle -v -count=1
```

Prepare the isolated runtime as described in [MODEL-COMPOSE.md](MODEL-COMPOSE.md).
The test uses generated local Parquet/corpus-BOM fixtures and tiny random
Llama, Qwen2, and Qwen3 models. It exercises Trainer for two optimizer steps,
verifies durable artifacts, and continues the Llama from saved weights with
gradient accumulation. It performs no model downloads or external publication.
Tests skip this path unless both variables are set.

#### Runtime setup for acceptance tests

These instructions provision a developer test runtime, not a WALDO-managed
installation. Start with an isolated Python environment whose PyTorch build
already works for the intended CPU/CUDA/ROCm device. Driver, CPU architecture,
PyTorch wheel, and GPU runtime compatibility are machine-local prerequisites;
installing Transformers does not provision a driver or guarantee GPU support.
Do not reuse a macOS virtual environment on Linux or an x86 environment on ARM64.

The examples currently pin Transformers 5.16.1. Keep the wheel as well as the
installed package. With `hf_python` pointing at that chosen isolated interpreter:

```bash
hf_python=/absolute/path/to/isolated-runtime/bin/python
hf_wheels=$(mktemp -d)
"$hf_python" -m pip download --no-deps --dest "$hf_wheels" transformers==5.16.1
export WALDO_TRANSFORMERS_WHEEL="$hf_wheels/transformers-5.16.1-py3-none-any.whl"
"$hf_python" -c 'import hashlib, os, pathlib; p = pathlib.Path(os.environ["WALDO_TRANSFORMERS_WHEEL"]); assert hashlib.sha256(p.read_bytes()).hexdigest() == "2f2d5b98a5ad3718713653734298fa620754ed683702a635ebb587df3ed29c7e", "wheel hash mismatch"'
"$hf_python" -m pip install "$WALDO_TRANSFORMERS_WHEEL" accelerate==1.15.0
export WALDO_TRANSFORMERS_PYTHON="$hf_python"
```

Package installation can change transitive dependencies. Recheck the installed
environment and selected device afterward; keep the dependency inventory with
the test logs. For CPU, explicitly set `WALDO_TRANSFORMERS_DEVICE=cpu`. For CUDA,
expose one GPU and verify a real allocation before running WALDO:

```bash
export CUDA_VISIBLE_DEVICES=0
export WALDO_TRANSFORMERS_DEVICE=cuda
"$WALDO_TRANSFORMERS_PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(4, device="cuda").sum().item())'
"$WALDO_TRANSFORMERS_PYTHON" -m pip freeze
```

The recorded DGX Spark acceptance environment was Linux ARM64, Python 3.12,
NVIDIA GB10, driver 580.173.02, PyTorch 2.13.0+cu129, Transformers 5.16.1,
Accelerate 1.15.0, Tokenizers 0.23.2, and Safetensors 0.8.0. It used an existing
`vllm/vllm-openai:v0.28.0-aarch64-cu129` container with GPU access and an isolated
`python3 -m venv --system-site-packages` overlay. This records a tested setup,
not a requirement to install vLLM or a claim that the image tag is immutable.
The host driver's advertised CUDA 13.0 capability is distinct from the PyTorch
CUDA 12.9 runtime used by these tests. Revalidate other runtime combinations.

HF tokenizer tests additionally require the local hash-pinned assets described
below; byte-tokenizer tests do not. WALDO never downloads models, tokenizer
assets, or dependencies during these tests. For offline execution, provision
the runtime/wheel and tokenizer assets before disconnecting.

With the same runtime variables set, run the dedicated custom Qwen3 compose test:

```bash
go test ./internal/model -run 'TestTransformersQwen3Smoke$' -v -count=1
```

This loads `docs/examples/transformers-qwen3-smoke.yaml`, trains a randomly
initialized two-layer Qwen3 using local fixtures, and verifies the saved custom
dimensions, measured parameter count, evaluation, and artifact hashes.

For the pinned standard Qwen tokenizer smoke, acquire `tokenizer.json` and
`tokenizer_config.json` from `Qwen/Qwen3-0.6B` at commit
`c1899de289a04d12100db370d81485cdf75e47ca`. The example compose records their
SHA-256 hashes. With the same Python/wheel variables set:

```bash
WALDO_HF_TOKENIZER_DIR=/absolute/path/to/tokenizer-files \
go test ./internal/model -run TestTransformersHuggingFaceTokenizerSmoke -v -count=1
```

This trains random weights for two steps on local fixtures and verifies the
tokenizer artifacts. No model weights or datasets are downloaded by the test.

To exercise CPU chat against a completed, disposable Transformers smoke model
without modifying it, set the Python/wheel variables above and run:

```bash
WALDO_HF_CHAT_TEST_ROOT=/absolute/path/to/models \
WALDO_HF_CHAT_TEST_MODEL=qwen3-tokenizer-smoke \
go test ./internal/inference -run TestTransformersRealChat -v -count=1
```

The test checks streamed output and repeated deterministic generation in one
session. Run it with both byte and pinned fast-tokenizer smoke models.

The expanded family matrix covers Mistral, Gemma 2, Phi-3, OLMo 2, Mixtral,
and text-only Qwen3.5. Each test trains a tiny random model, continues from saved
weights with gradient accumulation, runs chat, and exports verified artifacts:

```bash
WALDO_TRANSFORMERS_DEVICE=cpu \
go test ./internal/model -run '^TestTransformersFamilyLifecycle$' -v -count=1
```

For GPU acceptance, use a Linux host with the matching Transformers wheel and
a working PyTorch CUDA/ROCm runtime. Keep the Python/wheel variables set and
expose exactly one GPU. For CUDA:

```bash
CUDA_VISIBLE_DEVICES=0 bash testing/hf-gpu.sh fp32
CUDA_VISIBLE_DEVICES=0 bash testing/hf-gpu.sh fp16
CUDA_VISIBLE_DEVICES=0 bash testing/hf-gpu.sh bf16
```

The script forces GPU execution and fails rather than silently using CPU. BF16
requires supporting hardware; a rejection is not a passing BF16 test. For ROCm,
use that runtime's device-visibility control. No model weights, datasets, or
tokenizers are downloaded. Test-managed models and exported releases are local
temporary files. Attach command output and GPU/runtime details to PR feedback.
The new-family matrix honors the selected mixed precision; existing Llama and
Qwen smoke tests also run as baseline FP32 regressions.
For both the initial run (accumulation 1) and the continuation (per-device
batch 1, accumulation 2), the family test requires the run BOM's resolved
global batch, accumulation, compute precision, and pinned Trainer resolution
to equal the worker's observed `runtime.json` execution evidence: optimizer
steps, micro-batches, sequences, and output-head compute dtype.
CPU and mocked hardware-policy tests do not establish GPU correctness.

`TestTransformersRealTrainerResolution` uses the same runtime variables on CPU.
It requires `fp16` on CPU to fail during pinned Trainer resolution without
creating run state, and requires gradient checkpointing with accumulation 2 to
be recorded in the run BOM and observed in execution evidence.

For the byte-versus-HF tokenizer lifecycle audit, additionally set
`WALDO_HF_TOKENIZER_DIR` to the pinned Qwen3 tokenizer files used by
`docs/examples/transformers-qwen3-tokenizer-smoke.yaml`, then run:

```bash
WALDO_TRANSFORMERS_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
go test ./internal/model -run '^TestTransformersTokenizerLifecycle$' -v -count=1
```

This FP32 test trains and continues tiny random Qwen3 models for both tokenizer
types, checks accounting/package evidence, streams deterministic chat, exercises
stop strings and context sliding, and exports without access to the original
tokenizer directory. An independent offline Transformers process reloads the
source and export, checks tokenizer pins and encoding parity, compares exact
weights/logits, and reproduces WALDO's greedy output. The byte tokenizer is a
WALDO descriptor, not an `AutoTokenizer` asset; its documented byte mapping is
used explicitly. Corrupt retained tokenizer artifacts must fail chat,
continuation, and export. This complements the worker unit tests for EOS framing,
special-ID validation, package integrity, and stop-prefix buffering; random
generation does not guarantee that the EOS generation branch is exercised.

The DGX Spark audit on 2026-10-01 exposed and verified a fix for continuation
accepting a corrupt retained tokenizer artifact. Continuation now verifies every
retained source-run artifact, requires config/tokenizer evidence, and checks HF
tokenizer hashes against architecture pins. Both tokenizer lifecycle tests passed
on CUDA after the fix, including independent export parity and tamper rejection.

Failure-path coverage includes byte/HF special-ID validation, package filename
and installed-version mismatches, unavailable CUDA/allocation failures, and
subprocess-backed HF chat cancellation, callback failure, malformed/oversized
output, abrupt exits, diagnostics, and process reaping. Run the opt-in
`TestTransformersRejectsByteSpecialIDs` model test with the pinned runtime to
verify that incompatible byte configurations fail before producing weights.

`TestTransformersRejectsDivergentTraining` uses the pinned runtime with a
deliberately divergent FP32 CPU learning rate and no held-out evaluation. It
requires a non-finite loss or optimizer-step gradient-norm error, the terminal
`numerical-integrity` failure class, and no published weights. With the pinned
5.16.1 runtime, the step-2 gradient norm is the first non-finite value, so the
run fails before that optimizer update. The worker additionally
checks final floating-point parameters and buffers before saving, since the last
optimizer update may corrupt them after the last loss calculation. Unicode tests
check bounded, lossless output frames through both the Python emitter and Go
reader, plus stop-prefix and incomplete UTF-8 handling.

`TestTransformersPublishesVerifiedBestCandidate` trains a tiny FP32 Llama on CPU
for four steps with `evaluate_every: 1` and a high constant learning rate. With
this seed the final step regresses: held-out loss is 2.97 at step 3 and 3.19 at
step 4. The test requires the following:

- Every step is a persisted, evaluated candidate with live, compute-precision,
  and reloaded publishable losses.
- The full budget completes.
- The earlier best step is selected, and its reloaded artifact verification is
  recorded.
- The published tensors equal the selected candidate's tensors, not the final
  step's.
- `runtime.json` records one finite gradient-norm check per optimizer step.
- A continuation stage initializes from the published weights' hash.

`TestTransformersCompletionRequiresVerifiedBestCandidate` runs without a
runtime. A mock Transformers backend reports a regressed final step, and the
test requires completion to fail when any of these is true:

- The final candidate is selected instead of the earlier best.
- Selection or artifact verification is missing.
- The selected candidate was not persisted.
- Evaluations are absent.
- The reloaded artifact loss drifts beyond tolerance.

The worker unit tests cover earliest-tie selection, the 0.02-or-one-percent
tolerance, non-finite gradient norms, boundary scheduling, and separating
autocast from the live parameters.

#### Transformers validation scope

Recorded September/October 2026 acceptance results:

| Check | Evidence | Limitation |
| --- | --- | --- |
| Family lifecycle | CPU and GB10 CUDA FP32; six-family CUDA FP16/BF16 matrix | Llama/Qwen2/Qwen3 baseline tests stay FP32; not a production-scale benchmark |
| Tokenizer lifecycle | Byte and pinned Qwen HF tokenizer: train, continue, stream, export/reload, corruption rejection | Custom Qwen3 FP32 fixtures, not every tokenizer/model/precision combination |
| Resolved execution facts | Pinned Trainer resolution in the run BOM matches observed micro-batches, sequences, and compute dtype; CPU FP32 and GB10 CUDA FP32/FP16/BF16; FP16-on-CPU rejected before run creation | Tiny fixtures with accumulation 1 and 2; gradient checkpointing observed on CPU only |
| Numerical safety | Deliberately divergent no-held-out run rejected at a non-finite optimizer-step gradient norm as terminal `numerical-integrity`; final NaN parameters/Inf buffers rejected | Does not prove convergence or gradient/update parity |
| Publication safety (ADR 0044/0079) | Step-1/periodic/terminal candidates persisted and reload-verified; regressed final step not published; continuation from published weights; CPU FP32 regression fixture; GB10 CUDA FP32 and BF16 lifecycle runs verify the artifact | Serialization- and precision-drift rejection exercised by unit and mocked-completion tests, not induced in a real run. With FP16, the first GradScaler overflow at the default initial scale is a non-finite unscaled gradient norm and fails closed, as in the native PyTorch worker. Gemma 2, OLMo 2, and Qwen3.5 fixtures fail at step 1 for this reason |
| Stream and cleanup | Large valid Unicode output, bounded frames, stop prefixes, cancellation, crash/protocol failure and reaping | Large-output runtime probe used the pinned Qwen tokenizer |
| Export evidence | Artifact hashes and training-run package provenance enter the signing callback | Not an external signing-service integration test |
| ROCm | Device path implemented | No real ROCm hardware validation yet |

The scoped audit found no further confirmed defect after the fixes, but the
following numerical/evidence coverage remains open before claiming a complete
audit: unequal-mask and partial-accumulation gradient/update parity, explicit
Mixtral auxiliary-gradient checks, Qwen3.5 layer-update and context-rollover
parity, complete signing-evidence field assertions and HF finalizer-failure
cleanup, and independent Mixtral/Qwen3.5 export reload parity. Do not replace
these checks with a successful smoke run or mark them complete based only on
the implemented-feature matrix.

### External services

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
