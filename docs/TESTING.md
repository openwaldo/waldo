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
