# Command guide

This guide describes the current CLI. `waldo <command> --help` is authoritative
for flags and detailed behavior.

## Command tree

```text
waldo
├── advisor
├── status
├── config
│   ├── show
│   ├── get
│   ├── set
│   └── unset
├── index
│   ├── init
│   ├── pull
│   ├── list
│   ├── show
│   ├── summary
│   ├── verify
│   ├── audit
│   ├── bom
│   ├── ingest
│   └── export
├── shard
│   ├── summary
│   ├── audit
│   ├── bom
│   ├── list-records
│   └── export-record
├── lookaside
│   ├── login
│   ├── logout
│   ├── list
│   ├── status
│   ├── verify
│   ├── mirror (not available)
│   └── rm
├── model
│   ├── advisor
│   ├── init
│   ├── pull
│   ├── list
│   ├── summary
│   ├── bom
│   ├── forecast
│   ├── train
│   ├── continue
│   ├── export
│   ├── chat
│   └── rm
└── completion
```

There is no top-level `waldo bom` command. Corpus BOM operations are under
`waldo index bom`; model BOM operations are under `waldo model bom`.

## Local readiness

```bash
waldo status
```

`status` inspects local resources, the selected index, lookaside paths, and the
training harness chosen by the same environment resolver used by `model
train`. It performs no network requests and changes no state. Each unavailable
workflow reports `Ready: no` and explains why. Global `--json` returns the
same structured facts for automation.

## Index selection

When `config.index` is unset, read-only index commands use a managed checkout
at `~/.waldo/index`. WALDO creates it on first use and safely fast-forwards it
when the checkout is clean and behind its tracking branch. If upstream history
was rewritten, WALDO also resets this clean, read-only managed checkout to the
new canonical branch. It never applies that recovery to a configured
contributor checkout.

Set `config.index` to use a writable contributor checkout:

```bash
waldo config set index /path/to/waldo-index
```

Logical index paths resolve beneath the selected checkout. Existing filesystem
paths and paths beginning with `./`, `../`, `/`, or `~/` explicitly select the
exact local checkout and never trigger Git network access. Omitting a path
selects the whole resolved index. `waldo index verify --offline` additionally
disables object-storage verification and performs local structural validation.

## Inspect and verify data

```bash
waldo index list
waldo index show core/example
waldo index summary
waldo index verify --offline
waldo index verify core/example
waldo index verify core/example --objects
waldo index audit core/example --workers 8
```

Default verification checks metadata and canonical object availability without
downloading object bodies. `--objects` downloads and hashes every selected
object. `audit` additionally validates canonical shard contents. Its worker
count controls concurrent fetch-and-audit operations.

Local Parquet files can be inspected without an index:

```bash
waldo shard summary shard.parquet
waldo shard audit shard.parquet
waldo shard bom shard.parquet
waldo shard list-records shard.parquet
```

## Contribute data

Authoring commands refuse the managed read-only checkout. Clone the index and
configure a writable checkout first:

```bash
git clone https://github.com/openwaldo/waldo-index.git
waldo config set index /path/to/waldo-index
waldo config set lookaside file:///tmp/waldo-lookaside
```

Inspect the complete ingest interface before use:

```bash
waldo index ingest --help
```

The canonical ingestion input is a raw directory containing a root
`manifest.json`; see the [ingestion contract](INGESTION.md). Direct supported
files remain a local convenience. After all objects are audited and published,
WALDO retains a contribution overlay and atomically applies those metadata
changes to the selected index working tree for normal Git review. WALDO does
not commit, push, or open a pull request.

## Export data and inspect its BOM

```bash
waldo index export core/example /path/to/export
waldo index bom /path/to/export
waldo index verify /path/to/export
```

An export contains `EXPORT.json` plus the selected data files. `index verify`
checks the persisted BOM and hashes each exported file.

## Lookaside storage

```bash
waldo lookaside status
waldo lookaside cache status
waldo lookaside cache clean
waldo lookaside list
waldo lookaside verify
waldo lookaside rm --help
```

`lookaside cache clean` removes unused verified objects while protecting those
referenced by running or resumable model runs. `--all` also removes protected
objects and can make a later resume download them again. The original
`lookaside status` command remains available as a compatibility alias for
`lookaside cache status`.

Successful commands purge their used objects by default. Repeated experiments
can retain them within the configured LRU bound:

```bash
waldo config set lookaside.cache.retain-completed true
waldo config set lookaside.cache.max-size 30GiB
```

`waldo lookaside mirror` is reserved but not implemented.

For S3 publication, configure the destination and store bucket-scoped
credentials outside corpus metadata:

```bash
waldo config set lookaside s3://bucket/prefix
waldo config set lookaside.region us-west-2
waldo lookaside login
```

Removal requires explicit object names. Review command help before any
consequential lookaside operation.

## Models

The model lifecycle can start from an index selection, a compose file, or a
supported open-weight origin. Available training backends depend on the host.

```bash
waldo model forecast composes/0000-canary.yaml
waldo model train canary composes/0000-canary.yaml
waldo model summary canary
waldo model bom canary
waldo model chat canary
waldo model export --help
```

`forecast` reports readiness and runtime for the current host. Add
`--compare-hosts` to include the versioned hardware comparison. Run `forecast`
before allocating substantial compute. Training and generation
fail when the selected host lacks a compatible runtime or artifacts.
Models with a declared `interaction.template` automatically receive the
matching prompt format and multi-turn history in `model chat`; models without
one remain raw causal-continuation models.

Reference composes under `composes/` are test and experiment inputs, not model
quality guarantees. See the [model compose guide](MODEL-COMPOSE.md) for the
complete schema, defaults, profiles, and validation rules.

## Output and help

Human-readable output is the default. Global `--json` emits structured results
while progress remains on standard error.

```bash
waldo --json config get
waldo index verify --help
waldo completion --help
```
