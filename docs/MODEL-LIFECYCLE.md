# Model lifecycle

The model lifecycle separates a stable architecture from an append-only history
of explicit training runs. On Apple Silicon, WALDO automatically discovers a
Metal-capable Python installation of MLX and performs real decoder training.
It never silently substitutes simulation for training.

All durable formats in this document use schema 1.

## Machine configuration

Logical corpus paths use one configured index checkout. Model state and the
verified shard cache have independent locations:

```bash
waldo config set index /path/to/waldo-index
waldo config set model.root /fast-disk/waldo-models
waldo config set lookaside.cache /fast-disk/waldo-cache
```

Defaults are `~/.waldo/models` and a user-scoped lookaside cache beneath the
operating system's temporary directory. Verified objects remain available
while an operation is active and across a failure or interruption. After a
successful operation commits, WALDO removes every cache object that operation
used. `lookaside.cache.max-size` bounds recovery objects left by incomplete
operations; it is not a post-success retention target.
`model.backend` defaults to `auto`. On macOS it selects MLX and requires Apple
Silicon. On Linux it probes Python environments in deterministic order,
preferring an installed TorchTitan and then an installed PyTorch. It never
falls back to simulation. `mlx`, `torchtitan`, and `pytorch` are explicit
machine-local overrides; `fake` is an explicit simulation mode for development
and automated lifecycle tests whose artifacts are permanently marked as
simulated.

Backend resolution happens before corpus materialization or a run record is
created. A missing or unusable selected backend therefore fails immediately
with platform-appropriate official
installation guidance. On Linux, WALDO reports the detected distribution,
Python candidates, and NVIDIA/AMD tooling. It gives the matching
package-manager command for Python prerequisites and directs the operator to
the official CUDA, ROCm, or CPU PyTorch selector; it does not assume a distro
package named `pytorch` is correct. MLX and single-process PyTorch are
executable adapters. PyTorch verifies a real operation on its selected Linux
CPU, NVIDIA CUDA, or AMD ROCm device before a run is created. TorchTitan
verifies its current package APIs and every visible GPU, launches one rank per
GPU on a single Linux node, and records the complete device set.

## Basic commands

```bash
waldo model init small --preset 10m
waldo model list 'small*'
waldo model summary small
waldo model train small core/books science/papers
waldo model bom small
waldo model export small ./small-export
waldo model chat small
waldo model rm small
```

## Downloaded open-weight origins

`model pull` is the explicit command for importing a supported external model
into WALDO's managed model store. Schema 1 acquires training-quality Hugging
Face Safetensors, resolves
the requested reference to an immutable repository commit, hashes every source
artifact, validates the architecture, tokenizer, tensor names, shapes, and
precision, and streams the tensor bytes into WALDO's canonical names:

```bash
waldo model pull llama-base huggingface://organization/model@main
waldo model train llama-base core/books --epochs 1
```

Private or gated repositories use `HF_TOKEN` or the standard Hugging Face token
file. Source downloads live only in private staging and are removed after the
managed artifact has been verified. `ORIGIN-BOM.json` retains the repository,
requested reference, resolved commit, license when declared, and hashes of all
acquired files; it does not retain a redundant copy of the source weights.
`MODEL-BOM.json` selects the origin until a complete real training run produces
new current weights.

Schema 1 initially accepts standard bias-free Llama weights with
`OpenWALDOByteTokenizer`, vocabulary size 259, and F32, F16, or BF16 tensors.
That is the format produced by WALDO's Hugging Face export. Other tokenizers or
architectural variants fail before publication rather than being numerically
converted or silently retokenized.

A compose can use this same verified importer without first creating a visible
base model:

```yaml
base:
  source: huggingface://organization/model@0123456789abcdef0123456789abcdef01234567
```

Direct compose sources must name an immutable commit. WALDO inherits the
verified architecture when `architecture` is omitted, or treats a supplied
architecture as an exact assertion. Acquired origins are content-addressed
beneath `<model.root>/.origins`, hidden from `model list`, and reused by later
composes. Target models hard-link cached artifacts when the filesystem permits
and copy them otherwise. The resulting model retains the same `ORIGIN-BOM.json`
provenance as an explicit `model pull`.

`init` creates an untrained immutable architecture. `train` resolves one or
more recursive index selections, deduplicates them into an OpenWALDO BOM,
materializes size- and SHA-256-verified Parquet through the shared cache, and
appends one run. `--audit` additionally verifies shard structure, embedded
attestations, and declared totals; legacy or unattested shards then receive the
normal deep record scan. Before training, WALDO deterministically
holds out one percent of records, capped at 256 records and 1 MiB of text, and
pins that exact selection in the run BOM. Held-out records never enter a
training epoch. A one-record corpus records no holdout because a real split is
impossible. The current compact default is one pass,
batch size 8, the architecture context length, learning rate 0.0003, and seed
42. `--epochs <n>` controls complete passes over the selected records and
defaults to 1. For the built-in byte tokenizer, WALDO counts exact packed
byte-token targets for all epochs rather than reusing the manifest's reference
token estimate. It then reports and persists the derived optimizer-step count:

```text
steps = ceil(byte targets / (batch size × sequence length))
```

The byte-target count excludes the held-out set. Backends evaluate its
per-record EOS-packed sequences without gradients and report `heldout_loss`
and `heldout_perplexity`; the old current-training-batch loss is not presented
as evaluation.

Compose-driven models may instead pin either
`tiktoken/r50k_base@tiktoken-r50k-base` with vocabulary size 50259 or
`tiktoken/cl100k_base@tiktoken-cl100k-base` with vocabulary size 100259. WALDO
encodes canonical text in Go from bundled offline vocabularies and streams
identical token IDs to MLX or PyTorch. For `r50k_base`, pad, BOS, and EOS use
IDs 50256 through 50258; for `cl100k_base`, they use IDs 100256 through 100258.
Chat uses the persisted tokenizer identity and the same codec for prompts and
generated IDs.

Real backends commit checkpoint directories containing model weights,
optimizer state, runtime random state, and state metadata before reporting the
checkpoint to WALDO. Repeating the exact `model train` command after Ctrl-C
resumes the same run ID and immutable run BOM. WALDO verifies every checkpoint
member, restores the pinned backend revision, replays the deterministic input
stream without optimization to the saved step, and continues. `RUN.json`
records each attempt. A changed corpus, epoch count, profile, backend, or
execution environment is a new run rather than an unsafe resume.

Epoch boundaries remain part of one continuous-EOS token stream, while each
epoch gets a deterministic seed-derived shuffle. Exact low-level or multi-stage
parameters belong in a model compose.

`model bom` writes JSON to standard output unless an output file is supplied.
`model export` requires a new destination directory because a model contains
multiple artifacts and provenance records. Its default `waldo` package exposes
the portable aggregate as `BOM.json` and always adds `EU-BOM.json`; the managed
model retains the internal `MODEL-BOM.json` name. Configure the provider once
with `waldo config set disclosure.provider provider.json`. A normal export
fails before publication if required disclosure facts are absent, while
`--allow-incomplete` writes a conspicuously marked development draft. When
`signing.method` is configured, export automatically signs both BOMs and fails
if signing fails. Otherwise it succeeds with an unsigned warning. `model rm`
accepts only exact model names.

The complete package layouts, BOM layers, conversion rules, signing behavior,
failure modes, and consumer examples are in
[the model export guide](MODEL-EXPORTS.md).

`--format huggingface` exports the current verified origin or complete,
non-simulated run as a standalone Transformers package. WALDO rewrites only the Safetensors
header: tensor bytes remain unchanged while names move to the standard Llama
layout and container metadata identifies PyTorch. The package includes
`architecture.py`, the schema-1 byte tokenizer implementation and
configuration, `BOM.json`, and `EU-BOM.json`. The tokenizer is custom code, so
Transformers callers load it with `trust_remote_code=True`; the model itself
uses the standard Llama configuration. A model without a usable origin or real
run is rejected rather than exporting simulated or incomplete artifacts.

`--format mlx` emits the same standard Llama tensor names with Safetensors
metadata for MLX, an executable binding to MLX-LM's Llama model, and the same
explicit byte tokenizer. It is a separate package, not a second copy bundled
into the Hugging Face export. Tensor data is again copied byte-for-byte; only
the container header and surrounding runtime files differ.

`--format gguf` streams the current weights into one GGUF v3 `model.gguf` for
llama.cpp-compatible runtimes such as LM Studio. It preserves BF16 or F16
matrix precision, promotes one-dimensional normalization weights to F32,
applies the Llama Q/K head layout required by GGUF, and embeds WALDO's byte
tokenizer. `--format ollama` produces the same GGUF representation plus a
portable `Modelfile` referencing `./model.gguf` and the architecture context
length. Both packages contain `BOM.json` and `EU-BOM.json`; they do not include
a redundant Safetensors copy.

```bash
waldo model export small ./small-gguf --format gguf
waldo model export small ./small-ollama --format ollama
ollama create small -f ./small-ollama/Modelfile
```

All derived exports preserve the model's immutable interaction contract
automatically. Raw models remain raw; conversational models receive their
pinned template; models declaring `interaction.tools: true` also receive tool
schema, call, argument, and result handling. Hugging Face and MLX use tokenizer
chat-template metadata, GGUF embeds `tokenizer.chat_template`, and Ollama emits
the equivalent `Modelfile` template. No format-specific tool flag is required.

`model chat` opens the BOM-selected current origin or newest complete
real-weight run, verifies its weights, configuration, and tokenizer, and uses
the compatible runtime for an origin or the backend recorded by a run. MLX
sessions use incremental key/value caching; PyTorch and TorchTitan-produced
weights use a single-process PyTorch session on the selected Linux device:

```bash
waldo model chat small
waldo model chat small "Once upon a time"
printf 'Once upon a time' | waldo model chat small
waldo --json model chat small "Once" --max-tokens 64 --temperature 0 --seed 7
```

No generation option is required. Defaults are 256 maximum tokens,
temperature 0.8, and top-p 0.95. A zero temperature is deterministic; `seed`
makes sampling reproducible. Interactive sessions support `/clear`, `/help`,
and `/exit`. On a terminal, WALDO periodically re-renders the accumulated
response so Markdown appears while generation is still running. Control and
invalid UTF-8 bytes are escaped so model output cannot emit terminal control
sequences. Redirected output is rendered once without cursor controls. JSON is
one-shot and includes model and run identity, prompt, text, token count, finish
reason, and generation duration.

The built-in byte-tokenizer models are causal pretraining models and carry no
chat template. Interactive mode therefore performs raw continuation;
instruction-following behavior requires later supervised fine-tuning and a
pinned chat template. Generation is ephemeral and does not mutate lifecycle
state or claim a new BOM observation.

The MLX, PyTorch, and TorchTitan adapters support `decoder-transformer`
architectures pinned to the byte, `r50k_base`, or `cl100k_base` tokenizer
contracts described above. Unsupported tokenizer identities, revisions, or
vocabulary sizes fail during backend resolution before a run record is
created. WALDO records the selected Python path, framework version, host,
device, accelerator identity, and accelerator memory when applicable in the
run BOM.

## Model compose

A model compose is strict YAML or JSON. The command supplies the local model
name, so the portable file contains no name and can be reused. The
[model compose guide](MODEL-COMPOSE.md) is the authoritative field reference;
this section describes its place in the model lifecycle:

```yaml
kind: waldo-model-compose
schema: 1

# Optional. Omit this block to initialize a blank architecture. Use either
# model or source, never both.
base:
  model: llama-base
  # Optional assertion; WALDO always pins the resolved value into PLAN.json.
  origin_sha256: <origin-bom-sha256>

architecture:
  family: decoder-transformer
  context_tokens: 2048
  vocabulary_size: 259
  hidden_size: 384
  intermediate_size: 1024
  layers: 6
  attention_heads: 6
  key_value_heads: 2
  tie_embeddings: true
  parameter_dtype: bfloat16
  tokenizer:
    name: byte
    revision: builtin-byte-schema-1

stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    corpora:
      - core/books
      - core/common-pile/foodista
    parameters:
      profile: causal-pretrain-shuffled
      tokens: 20480000
      batch_size: 2
      sequence_length: 1024
      learning_rate: 0.0003
      seed: 7

      # Optional overrides; omitted values resolve from the profile.
      weight_decay: 0.1
      warmup_steps: 100
      checkpoint_every: 500
      evaluate_every: 500
      evaluation_fraction: 0.01
      evaluation_max_records: 256
      evaluation_max_bytes: 1048576
      shuffle_buffer_records: 1024
      shuffle_buffer_bytes: 67108864
```

Use `architecture.dropout` for recorded residual dropout. Multi-corpus runs
that need unequal exposure use `profile: causal-pretrain-weighted` and declare an
integer weight for every resolved corpus path:

```yaml
architecture:
  dropout: 0.1
stages:
  - name: pretrain
    corpora:
      - path: core/books/gutenberg
        weight: 1
      - path: core/common-pile/wikimedia
        weight: 2
    parameters:
      profile: causal-pretrain-weighted
```

Weights define relative tokenizer-target exposure, not sampling probabilities
or duplicated source records. They are preserved in the run BOM and exact
consumption remains backend-validated. Built-in MLX and PyTorch workers apply
dropout to attention and feed-forward residual branches during training and
disable it during evaluation and inference.

Scalar corpus paths and the older `parameters.corpus_weights` map remain
accepted. A configured corpus entry can also carry license, language, source,
and date filters; stage-wide and corpus-local filters are combined and pinned
in the corpus BOM before held-out selection and training.

Run it with:

```bash
waldo model train example model.yaml
```

Unknown fields, additional YAML documents, incomplete architecture, unsupported
objectives, empty or duplicate corpus selections, incomplete weighted corpus
maps, duplicate stage names, and invalid parameters are rejected. Corpus values
are index paths, never raw directories or corpus exports. Explicit paths
discover their checkout; logical paths use the current or configured checkout.

When `base.model` is present, WALDO selects its explicitly requested completed
run or its latest completed run with real weights. If it has no such run, WALDO
uses the model's verified pulled origin. WALDO pins the parent model ID, run ID,
run BOM hash, weight hash, and size, then hard-links or copies trained parent
weights into the destination as `base/model.safetensors`. `base.source` instead
accepts a pinned supported external source and uses the same verified importer
as `model pull`. In every mode WALDO verifies artifacts and exact architecture
equality, and a compose never mutates its base model.

WALDO resolves and hash-verifies every stage and creates the active model at
`<model.root>/<name>` when absent. When the name already exists, its normalized
architecture and tokenizer hash must exactly match the compose; a mismatch is
refused with guidance to use a new model name. WALDO removes corpus paths
already present in that model's completed run BOMs, then appends stages for
the remaining paths without replacing the model. If no paths remain, the
model is unchanged. This is path-level reuse, not record- or shard-level delta
detection. Durable transaction metadata beneath
`<model.root>/.waldo-compose` pins the compose, every corpus BOM, the model ID,
and the starting run ordinal.
Passing `--audit` audits every materialized stage before the transaction starts.
Interactive terminals receive byte-level materialization progress; redirected
logs receive one completion line for every shard. The optional audit is shown
as a separate phase. Deterministic held-out selection enumerates the row counts
pinned by the corpus BOM, reports shard and record progress, and reads only the
bounded candidate rows before backend selection. While a compose is running,
ordinary `model list` and `model summary` operations see its current state at
the standard model path.
After Ctrl-C or process loss, repeating the exact command discovers the active
model, marks an abandoned running attempt interrupted, and resumes the same
stage and run from its newest verified checkpoint. Different inputs are refused
while that transaction is unfinished. Completed-path skipping is disabled for
an unfinished transaction so its checkpoint selection remains exact. A failed
stage is cleared; interrupted work is retained.

## Durable layout

```text
<model.root>/<name>/
├── PLAN.json
├── MODEL.json
├── MODEL-BOM.json
├── ORIGIN-BOM.json        # pulled/external-origin models only
├── origin/artifacts/      # one normalized, verified starting checkpoint
├── base/model.safetensors # trained managed-model parent only
└── runs/
    └── 0001-<stage>-<run-id>/
        ├── RUN-BOM.json
        ├── RUN.json
        ├── TELEMETRY.csv
        └── artifacts/
            └── checkpoints/
                └── step-00000500/
                    ├── model.safetensors
                    ├── optimizer.safetensors  # MLX
                    ├── runtime.pt              # PyTorch/TorchTitan
                    └── state.json
```

- `PLAN.json` content-identifies the immutable architecture and local model
  name plus either an external origin BOM or a trained-parent pin. The parent
  pin names the model and run and hashes both the run BOM and copied weight
  artifact. `MODEL.json` and `MODEL-BOM.json` repeat that lineage. Adding
  training does not change model identity.
- `RUN-BOM.json` embeds the hash-pinned corpus OpenWALDO BOM and pins
  architecture, backend, objective, parameters, and execution environment
  before launch. With `--audit`, it additionally carries each embedded shard
  BOM and its independent SHA-256, or explicit legacy/deep validation status.
- `RUN.json` moves atomically through `planned`, `running`, and exactly one of
  `complete`, `failed`, or `interrupted`. It separates verified partial
  progress from a complete observation and records every resume attempt.
- Once a model is published beneath `model.root`, failed and interrupted work
  remains visible and inspectable. Training never replaces or removes it;
  removal requires explicit `model rm`.
- `TELEMETRY.csv` is an append-only, spreadsheet-ready event and scalar time
  series. Its stable columns record UTC observation time, attempt-relative
  elapsed time, run and stage identity, event and state, optimizer progress,
  loss, held-out loss and perplexity, throughput, ETA, and the human message.
  Resume attempts append to the same file with a new attempt number.
- `model summary` snapshots this durable state for a deterministic, read-only
  health assessment. `advisor` is an interactive assistant grounded in
  the saved compose, run history, refreshed telemetry, and a compact inventory
  of every corpus in the configured index. It can propose a follow-up compose,
  but WALDO validates all corpus paths against that index and asks before
  updating the separate advisor draft. For a model name that does not exist,
  it interviews the operator, creates a validated compose, and separately asks
  before starting the build. It does not send weights, training records, or
  corpus text or alter an existing source model.
- Every distinct compose is archived as ordered YAML beneath `composes/`, using
  `0000-<name>.yaml`, `0001-<name>.yaml`, and so on. `COMPOSE.json` remains the
  canonical compatibility record. Exact transaction resume deduplicates the
  archive rather than recording the same compose twice.
- Advisor edits to an existing working compose require a new-revision versus
  update choice. New revision is the default for architecture/base changes and
  whenever the working compose matches an archived compose. Archived model
  composes are never edited.
- `model continue <name>` is valid only when `.waldo-compose` retains an
  interrupted transaction for that model. It loads the latest archived compose
  (falling back to legacy `COMPOSE.json`) and enters the normal transaction and
  verified-checkpoint resume path.
- Advisor sessions append schema-1 JSONL records to `advisor/CHAT.jsonl`.
  Advisor-started builds enqueue provider analysis at checkpoint boundaries;
  provider latency does not block training. Completed assessments are persisted
  and included with compact run and compose history in later chats.
- `MODEL-BOM.json` aggregates run-BOM hashes, terminal states, backend and
  simulation identity, observation hashes, and artifact hashes. Its
  `path_base` is `model-root`: every `run_bom` and artifact `path` resolves
  from the directory containing `MODEL-BOM.json` in a managed model, or
  `BOM.json` in a model export. Paths are portable and never contain a
  machine-specific model root.
  Those run-BOM hashes transitively pin every shard BOM without duplicating
  the full shard evidence in the aggregate model document.
  `current_run_id` selects the newest complete, non-simulated run containing
  real weight artifacts; earlier simulated and real runs remain visible as
  provenance. Before such a run exists, `current_origin_sha256` selects the
  verified downloaded starting checkpoint.

Every aggregate artifact has a role such as `weights`, `configuration`,
`tokenizer`, or `simulation`. `model export` rewrites any accepted legacy
schema-1 aggregate BOM into this unambiguous form and verifies the bytes,
sizes, and SHA-256 hashes of terminal and checkpoint artifacts before
publishing the exported directory.

Machine-local index roots and cache paths never enter identity. Run BOMs retain
logical index paths, manifest and shard hashes, licenses, source evidence, and
the index Git identity when available.

## Resource forecast

```bash
waldo model forecast model.yaml
waldo model forecast /path/to/waldo-index/core/books
waldo model forecast core/books science/papers
waldo model forecast model.yaml --compare-hosts
```

A compose supplies exact architecture and training budgets. Direct index paths
resolve a deduplicated selection, recommend the largest model rung supported by
roughly 20 tokens per parameter, and forecast one pass. Forecast creates no
model or run state. Human forecasts report the derived approximate parameter
count in both compact and exact form; JSON exposes the same value as
`approximate_parameters`. WALDO reports missing or empty compose files as input
errors rather than treating their filenames as corpus paths. Logical corpus
selections use the normal configured-index refresh before resolution.

By default the forecast resolves the same MLX, PyTorch, or TorchTitan harness
that training would use, compares the workload with the current host's memory,
and reports `Ready: yes` or explains why it is not ready. A matching catalog or
observed-run profile also supplies an approximate duration. Backend discovery
or local readiness failure does not discard the workload forecast. When the
workload exceeds local memory, WALDO recommends remote compute with sufficient
usable memory. A workload larger than every dated catalog configuration still
returns its conservative resource forecast instead of failing.

`--compare-hosts` additionally shows fitting catalog configurations from
slowest to fastest:

```text
PARAMETERS:  9.5M (9,543,210)
TOKENS:      1.0B

GPUS  MFR     ACCELERATOR                    MEMORY/GPU  APPROX. TIME
   1  Apple   M4 Max 40-core GPU                 128 GB       48 days
   8  NVIDIA  H100 SXM                           80 GB       44 hours
```

The estimate uses planned tokens, approximate parameters, sharded optimizer
state, full physical-batch activations and vocabulary logits, device headroom,
and conservative effective throughput from a versioned hardware catalog. It
does not assume activation checkpointing or split a declared batch across FSDP
ranks. WALDO also verifies completed, non-simulated
runs beneath `model.root` and measures their active attempt time. Evidence is
aggregated only for the exact accelerator model and GPU count observed; that
row uses measured throughput, while every unmatched row retains its dated
catalog value and overhead. Human output states when local calibration is
applied. JSON includes local readiness and omits comparison configurations
unless `--compare-hosts` is given. The comparison includes the formula, source
per row, unrounded inputs, aggregate run count, measured seconds and FLOPs, and
a hash of the contributing evidence.

## Backend boundary

Model composes never select MLX, PyTorch, TensorFlow, or TorchTitan. Before a
run is written, the environment-aware resolver chooses an adapter and records
its immutable identity, framework, runtime, host, accelerator, node count, and
world size in the run BOM. Every adapter receives the same architecture,
hash-pinned BOM, resolved training profile, deterministic canonical-record stream,
artifact directory, and progress sink. Adapters never parse Parquet or choose
record order. An embedded worker communicates through schema-1 NDJSON: a begin
frame, record frames from a shuffle bounded by both record count and retained
bytes, an end frame, then typed progress,
checkpoint, evaluation, completion, or error output frames.
Human-readable training progress includes a remaining-time ETA after the
startup sample; JSON progress exposes the underlying `eta_seconds` value.

The MLX, PyTorch, and TorchTitan adapters embed their worker source in the
WALDO binary while using the machine's explicit Python framework runtime. They
implement rotary
grouped-query attention, RMS normalization, gated feed-forward blocks, byte
tokenization, continuous-EOS packing, AdamW with warmup and cosine decay, real
loss/gradient updates, checkpoint and terminal Safetensors, progress,
training-loss metrics, deterministic held-out loss/perplexity, complete
optimizer/runtime checkpoint bundles, exact resume, and observed token totals. Their internal tensor names
and shapes are identical, so verified WALDO weights can continue across those
frameworks. A later run verifies and initializes from the most recent
non-simulated terminal `model.safetensors`; its source run and weight hash are
pinned in the new run BOM. Instruction tuning and chat templates remain
separate work.

TorchTitan uses WALDO's same PyTorch model and optimizer implementation but
launches it through `torch.distributed.run`. Global rank zero receives WALDO's
deterministic, pre-tokenized schema-1 stream and broadcasts it to every rank
through NCCL. TorchTitan constructs a global device mesh and PyTorch FSDP2 shards
model parameters across every rank. All ranks participate when gathering
checkpoints, optimizer state, runtime random state, and terminal weights; only
global rank zero writes and reports the portable full artifacts. Multi-node
rendezvous and rank-0 hostfile launching are supported. Scheduler integration,
elastic restart, and tensor/pipeline parallelism remain later orchestration
work. See [Multi-host training](MULTI-NODE-TRAINING.md) for the current cluster contract
and operator procedure.

NeMo/Megatron and native sparse-MoE execution are planned, not current
behavior. WALDO will remain the lifecycle and provenance control plane while a
separate adapter owns native architecture construction, distributed topology,
training, and distributed checkpoints. The prepared-data boundary, native
artifact sets, reference B200 host, and acceptance gates are defined in the
[Foundation, post-training, and sparse-MoE plan](FOUNDATION-MOE-PLAN.md).
