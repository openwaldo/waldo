# WALDO

WALDO is a command-line tool for building auditable AI training datasets and
models. It connects a Git-governed corpus index, content-addressed Parquet
objects, and model artifacts with verifiable bills of materials (BOMs).

WALDO is under active development. It is being opened early so users and
contributors can help shape the project through frequent, incremental
releases. Expect interfaces and formats outside the documented compatibility
contract to evolve.

## What works today

- Inspect, verify, audit, ingest, update, and export indexed corpora.
- Read and publish canonical Parquet objects through local or S3 lookaside
  storage.
- Create corpus, training-run, model, and release provenance records.
- Forecast and train models through MLX, PyTorch, or single- and multi-host
  TorchTitan when the required runtime is installed.
- Export native WALDO, Hugging Face, MLX, GGUF, and Ollama packages.

WALDO does not prove that a license assertion is legally correct, that model
output is safe, or that a generated disclosure alone establishes regulatory
compliance. It records attributable facts and verifies artifact identity.

## Build

WALDO requires Go 1.25 or newer.

```
git clone https://github.com/openwaldo/waldo.git
cd waldo
go build -o ./waldo ./cmd/waldo
sudo install -m 0755 ./waldo /usr/local/bin/waldo
waldo --help
```

## First steps

With no configured index, read-only index commands use a managed checkout at
`~/.waldo/index`:

```
waldo status
waldo index list
waldo index summary
waldo index verify --offline
```

Contributing data requires a separate writable index checkout:

```
git clone https://github.com/openwaldo/waldo-index.git
waldo config set index /path/to/waldo-index
waldo config set lookaside file:///tmp/waldo-lookaside
waldo index ingest --help
```

Use `waldo <command> --help` as the authoritative command reference.

## Documentation

Start with the [documentation index](docs/README.md). In particular:

- [Ingestion contract](docs/INGESTION.md)
- [Command guide](docs/UX.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Testing](docs/TESTING.md)

Early feedback, testing, documentation improvements, and focused code
contributions are welcome. See [Contributing](docs/CONTRIBUTING.md).

## Development

```
./testing/unit.sh
./testing/vet.sh
```

The full end-to-end suite is described in [docs/TESTING.md](docs/TESTING.md).

## License

WALDO is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE)
for attribution notices.
