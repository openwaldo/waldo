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
CPU and mocked hardware-policy tests do not establish GPU correctness.

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
