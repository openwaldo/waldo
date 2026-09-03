# Model compose guide

A model compose is a strict, portable YAML or JSON document that declares a
model architecture and one or more ordered training stages. Use one when WALDO
must create a model, train several stages as one resumable transaction, or
preserve an exact experiment for later review.

The compose does not select MLX, PyTorch, TorchTitan, GPUs, paths, credentials,
or other machine-local policy. WALDO selects a compatible training backend on
the machine that runs the compose.

```bash
waldo model forecast composes/0001-babble.yaml
waldo model train babble composes/0001-babble.yaml
```

`forecast` is read-only. `train` creates the named model when it does not
exist, or appends the declared stages when the existing model has exactly the
same architecture.

Forecasts calculate fixed-token stages immediately. Epoch-driven stage sizes
depend on the selected index revision, filters, held-out records, tokenizer,
and objective, so their exact tokens, steps, and added runtime are reported
during training preflight rather than guessed.

## Why the profile is called causal pretraining

The name is **causal**, not casual. In causal language modeling, the model
predicts each next token using only the tokens that precede it. Future tokens
are masked. This left-to-right objective is what makes the resulting base model
able to continue a prompt.

`pretrain` in the profile name describes the general data-ordering and training
contract. It does not make the stage label `type: pre-training` mandatory; the
currently executable objective is the same for every accepted stage type.

## Complete annotated compose

This example shows every schema-1 field. The optional `base` block is commented
out because it is valid only when the named model has compatible verified
weights.

```yaml
kind: waldo-model-compose
schema: 1

# Optional: choose one base form, never both.
# base:
#   model: conversation
#   model_id: <optional-expected-model-id>
#   run_id: <optional-completed-run-id>
#   run_bom_sha256: <optional-run-bom-sha256>
#   artifact_sha256: <optional-checkpoint-sha256>
#   artifact_bytes: <optional-checkpoint-size>
#   origin_sha256: <expected-origin-bom-sha256> # optional assertion
#
# Or acquire a supported external origin directly. With this form only,
# architecture may be omitted and inherited from the verified origin.
# base:
#   source: huggingface://organization/model@<commit>

# Optional: declare the exact inference-time dialogue format learned by the
# model. Omit this block for raw causal continuation.
interaction:
  template: user-assistant-v1
  tools: true # optional; enables the model-level tool interaction contract

architecture:
  family: decoder-transformer
  context_tokens: 2048
  vocabulary_size: 50259
  hidden_size: 768
  intermediate_size: 2048
  layers: 12
  attention_heads: 12
  key_value_heads: 4
  dropout: 0.1
  tie_embeddings: true
  parameter_dtype: bfloat16
  tokenizer:
    name: tiktoken/r50k_base
    revision: tiktoken-r50k-base

stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    filter:
      main_content: true
      exclude:
        repetitive_content: true
        boilerplate_content: true
        licenses: [CC-BY-NC-*]
    corpora:
      - path: core/books/gutenberg
        weight: 1
        filter:
          languages:
            include: [en]
          date:
            from: "1900"
      - path: core/common-pile/wikimedia
        weight: 2
        filter:
          sources:
            exclude: [deprecated-*]
    parameters:
      profile: causal-pretrain-weighted
      tokens: 1966080000
      batch_size: 32
      sequence_length: 1024
      learning_rate: 0.0002
      seed: 42
      weight_decay: 0.15
      warmup_steps: 600
      checkpoint_every: 6000
      evaluate_every: 6000
      shuffle_buffer_records: 32768
      shuffle_buffer_bytes: 1073741824
      evaluation_fraction: 0.01
      evaluation_max_records: 512
      evaluation_max_bytes: 16777216
```

Unknown fields and additional YAML documents are rejected. JSON uses the same
field names and structure.

## Top-level fields

| Field | Required | Value | Meaning |
| --- | --- | --- | --- |
| `kind` | yes | `waldo-model-compose` | Identifies the document as a model compose. |
| `schema` | yes | `1` | Selects the compose schema. |
| `base` | no | object | Optionally initializes a new model from verified managed or external weights. |
| `architecture` | normally | object | Defines immutable model structure and tokenizer identity. It may be omitted with `base.source`, in which case WALDO inherits the verified source architecture. |
| `interaction` | no | object | Declares a versioned inference-time prompt contract. Omit it for raw causal continuation. |
| `stages` | yes | non-empty list | Ordered training stages. Stage names must be unique. |

### Interaction fields

Schema 1 supports `interaction.template: user-assistant-v1` and
`interaction.template: chatml-v1`. The former renders each turn as:

```text
User: <message>

Assistant: <response>
```

`chatml-v1` uses versioned `<|im_start|>` and `<|im_end|>` textual framing.
Conversation ingestion does not select either template. `waldo model chat`
maintains the selected transcript across interactive turns and
stops generation before the model begins a new `User:` turn. The interaction
contract is stored in the immutable model plan, model record, and model BOM.
It changes model identity but not parameter count. Models without an
interaction block retain raw causal-continuation behavior.

`interaction.tools: true` declares tool use once for the model, independently
of any corpus path. WALDO then accepts tool definitions in structured
conversation records, records the effective capability in training BOMs, and
automatically translates the same interaction contract into every supported
release format. Tool-enabled interaction requires an assistant-response stage
that supervises assistant messages. Omit `tools` for an ordinary conversational
model; never repeat it under an individual stage.

For schema-1 compatibility, WALDO still accepts the former
`stages[].conversation.tools: true` spelling and normalizes it to the same
model-level contract. New composes should use only `interaction.tools`.

### Base fields

| Field | Required | Value | Meaning |
| --- | --- | --- | --- |
| `model` | exactly one of `model` or `source` | `^[a-z0-9][a-z0-9._-]{0,63}$` | Names a managed model with a verified completed checkpoint or pulled origin. |
| `source` | exactly one of `model` or `source` | pinned model source | Acquires a supported external model through the same verified importer as `model pull`. Schema 1 accepts `huggingface://organization/model@<commit>`. |
| `origin_sha256` | no | SHA-256 | Asserts the expected origin BOM. WALDO always resolves and pins the actual value. |
| `model_id` | no | model ID | Asserts the identity of a managed base. WALDO fills it during resolution. |
| `run_id` | no | run ID | Selects a completed run. When omitted, WALDO resolves the latest completed run with real weights. |
| `run_bom_sha256` | no | SHA-256 | Asserts the selected run BOM. WALDO fills it during resolution. |
| `artifact_sha256` | no | SHA-256 | Asserts the selected checkpoint bytes. WALDO fills it during resolution. |
| `artifact_bytes` | no | non-negative bytes | Asserts the selected checkpoint size. WALDO fills it during resolution. |

## Base initialization

`base` controls the weights used to initialize a model before its first stage.
It does not name the destination model; the destination is the first argument
to `waldo model train`. A compose supports three initialization modes:

| Compose declaration | Initial weights | Architecture rule |
| --- | --- | --- |
| no `base` | Newly initialized weights | `architecture` is required. |
| `base.model` | Verified completed checkpoint from a named managed model, falling back to its origin when it has no completed run | `architecture` is required and must exactly match the managed model. |
| `base.source` | Verified origin weights acquired from an external source | `architecture` may be omitted and inherited; when present, it must exactly match. |

`model` and `source` are mutually exclusive. A base initializes the destination
model and is never mutated by its training. For a trained managed base, WALDO
resolves the latest completed run containing real weights unless `run_id` is
specified. It verifies and persists the parent model ID, run ID, run BOM hash,
artifact hash, and artifact size. The destination hard-links or copies those
weights to `base/model.safetensors`, so it remains usable if the parent is later
changed or removed. For an origin base, `origin_sha256` is an optional
fail-closed assertion against the canonical origin BOM hash; WALDO always pins
the resolved hash.

### Named managed base

```yaml
base:
  model: conversation
  # run_id: <optional-completed-run-id>
  # artifact_sha256: <optional-checkpoint-sha256>
```

The compose must contain a complete architecture exactly matching the named
model. Use this form when the base should be visible to `waldo model list` and
independently inspectable with `waldo model summary conversation`. Resolution
turns omitted pins into immutable values before training starts.

### Direct external base

The smallest complete source-based compose is:

```yaml
kind: waldo-model-compose
schema: 1

base:
  source: huggingface://organization/model@0123456789abcdef0123456789abcdef01234567

stages:
  - name: adapt
    type: fine-tuning
    objective: causal-language-modeling
    corpora: [core/books/gutenberg]
    parameters:
      profile: causal-pretrain-shuffled
      epochs: 1
      batch_size: 8
      sequence_length: 512
      learning_rate: 0.00005
      seed: 42
```

Run it normally:

```bash
waldo model forecast model.yaml
waldo model train adapted-model model.yaml
```

The source must pin a 40- to 64-character hexadecimal commit. Branches and tags
are rejected because they can move. `forecast` downloads only repository
metadata, `config.json`, and `tokenizer_config.json`; it does not download model
weights. `train` acquires and verifies the complete origin through the same
importer as `waldo model pull`.

Private or gated Hugging Face repositories use `HF_TOKEN` or the standard
Hugging Face token file. Verified direct origins are cached beneath
`<model.root>/.origins`, reused by later composes, and excluded from
`waldo model list`. The destination hard-links cached artifacts when possible
and copies them otherwise. Its `ORIGIN-BOM.json` records the repository,
requested and resolved revisions, declared license when available, source-file
hashes, and normalized artifact hashes.

Schema 1 currently accepts standard bias-free Llama Safetensors with
`OpenWALDOByteTokenizer`, vocabulary size 259, and F32, F16, or BF16 tensors—the
format produced by WALDO's Hugging Face export. Other tokenizers, architectures,
tensor layouts, and source providers fail before the destination model is
published. `base.source` and `model pull` intentionally share this same
compatibility boundary.

## Architecture fields

| Field | Required | Value | Meaning |
| --- | --- | --- | --- |
| `family` | yes | `decoder-transformer` | The only schema-1 architecture family. |
| `context_tokens` | yes | positive integer | Maximum token context. Every stage sequence length must fit within it. |
| `vocabulary_size` | yes | positive integer | Must match the selected tokenizer contract. |
| `hidden_size` | yes | positive integer | Transformer residual width. Must be divisible by `attention_heads`. |
| `intermediate_size` | yes | positive integer | Gated feed-forward intermediate width. |
| `layers` | yes | positive integer | Number of decoder blocks. |
| `attention_heads` | yes | positive integer | Query-head count. Must divide `hidden_size`. |
| `key_value_heads` | yes | positive integer | Key/value-head count. Must divide `attention_heads`. |
| `dropout` | no | `0 <= value < 1`; default `0` | Residual dropout applied during training and disabled during evaluation and inference. |
| `tie_embeddings` | no | boolean; default `false` | Reuses input embeddings as the output projection when true. False adds a separate output matrix. Reference composes set it explicitly. |
| `parameter_dtype` | yes | `float32`, `float16`, or `bfloat16` | Portable parameter and mixed-precision artifact declaration. Backend support is checked before training. |
| `tokenizer.name` | yes | supported name | Selects WALDO's offline tokenizer implementation. |
| `tokenizer.revision` | yes | immutable revision | Pins exact tokenizer behavior. |

The architecture determines the model parameter count. WALDO derives and
reports that count in forecasts and model summaries; it is not an independent
compose field.

### Supported tokenizer contracts

The tokenizer name, revision, and vocabulary size are one exact contract:

| Name | Revision | `vocabulary_size` | Intended use |
| --- | --- | ---: | --- |
| `byte` | `builtin-byte-schema-1` | 259 | Legacy and very small byte-token models. |
| `tiktoken/r50k_base` | `tiktoken-r50k-base` | 50259 | Compact English-oriented subword models. |
| `tiktoken/cl100k_base` | `tiktoken-cl100k-base` | 100259 | Larger multilingual and code-capable subword vocabulary. |

WALDO performs tokenization before the framework worker, ensuring supported
backends receive identical token IDs.

## Stage fields

| Field | Required | Value | Meaning |
| --- | --- | --- | --- |
| `name` | yes | `^[a-z0-9][a-z0-9._-]{0,63}$` | Unique durable stage and run label. |
| `type` | yes | `pre-training`, `fine-tuning`, `alignment`, or `other` | Records the stage's intended role in provenance. |
| `objective` | yes | `causal-language-modeling` or `assistant-response-modeling` | Causal loss covers every next token; assistant-response loss supervises roles selected by a structured conversation transformation. |
| `conversation` | for assistant-response modeling and conversation shards | object | Pins `template` (`user-assistant-v1` or `chatml-v1`) and a non-empty `supervised_roles` list. It must match `interaction.template`; tool capability is declared only by `interaction.tools`. |
| `filter` | no | record filter | Applies one record-level condition to every selected corpus. |
| `corpora` | yes | non-empty list of unique scalar paths or configured corpus objects | Selects canonical corpus records for the stage. |
| `parameters` | yes | object | Declares the portable training budget and controls. |

Relative corpus values are logical paths beneath the selected WALDO index.
Absolute paths may identify another index checkout. Values select indexed
corpora, never raw source directories or exported corpus files.

Before downloading any shard, `model train` refreshes the selected index and
checks every corpus path in every stage. A failed sanity check reports all
unavailable paths with their stage names and performs no shard materialization.

Stages execute in listed order. Each completed stage produces the current
weights used to initialize the next stage. If a stage fails, later stages do
not run. Repeating the exact command after interruption resumes the durable
transaction and its latest verified checkpoint.

Stage `type` currently records intent; it does not select a different loss or
framework algorithm. `objective` selects executable behavior, and schema 1
supports only causal language modeling.

## Corpus selection and record filters

A `corpora` entry may remain a path string, preserving every existing
schema-1 compose:

```yaml
corpora:
  - core/books/gutenberg
```

Use the object form when a corpus needs configuration. The canonical exclusion
form is a single deny list whose conditions are ORed:

```yaml
filter:                         # stage-wide
  main_content: true
  exclude:
    repetitive_content: true
    boilerplate_content: true
    licenses: [CC-BY-NC-*, LicenseRef-Restricted-*]
corpora:
  - path: core/books/gutenberg
    weight: 2
    filter:                     # only this corpus
      languages:
        include: [en]
      sources:
        exclude: [deprecated-*]
      date:
        from: "1900"
        to: "2025-06-30"
```

| Field | Required | Meaning |
| --- | --- | --- |
| `path` | yes in object form | Logical index path. |
| `weight` | only for `causal-pretrain-weighted` | Positive integer relative token exposure. It replaces the legacy map entry for this corpus. |
| `filter` | no | Record filter local to this corpus. |
| `filter.main_content` | no | Requires the canonical main-content boolean to equal the declared value. Normally `true`; older schemas default to `true`. |
| `filter.exclude.repetitive_content` | no | Excludes rows whose schema-2 repeated-token flag equals the declared boolean. The normal policy is `true`. |
| `filter.exclude.boilerplate_content` | no | Excludes rows whose schema-2 duplicated-structure flag equals the declared boolean. The normal policy is `true`. |
| `filter.exclude.licenses` | no | Excludes rows whose normalized license matches any listed shell-style pattern. |
| `licenses` | no | Matches the canonical row's normalized license. |
| `languages` | no | Matches the canonical row's language. |
| `sources` | no | Matches either the canonical source identifier or source name. |
| `date` | no | Selects canonical dates that overlap the inclusive `from`/`to` interval. |
| `include` | no | At least one shell-style, case-sensitive pattern must match. |
| `exclude` | no | Any matching pattern rejects the record and takes precedence over `include`. |
| `from` | no | Inclusive lower date bound: `YYYY`, `YYYY-MM`, `YYYY-MM-DD`, or RFC 3339. |
| `to` | no | Inclusive upper date bound in the same formats. |

Every declared filter must contain at least one condition. Within the canonical
`filter.exclude` object, a match on any declared boolean or license pattern
rejects the row. A stage-wide
`filter` and a corpus-local `filter` are combined with AND; the local filter
cannot loosen the global one. Other filter fields are ANDed with the exclusion
decision. Missing or malformed row values do not satisfy an include or date
condition.

The older `filter.licenses.include`/`exclude` representation remains accepted.
It cannot be combined with `filter.exclude.licenses` in the same filter.
Content-assessment filtering is applied wherever record schema 2 supplies the
declared facts. Schema-1 rows are unassessed rather than clean: WALDO retains
them, ignores only the unavailable assessment conditions, and emits a warning
that names the affected stage, shard count, and fields. Other conditions in the
same filter still apply. Assessment filters exclude complete rows; they never
redact or rewrite their text.

Filtering happens while WALDO streams canonical rows, before deterministic
held-out selection and training shuffle. The versioned effective policy is
pinned in the corpus OpenWALDO BOM, so a resume or distributed node cannot
silently train on a different subset. The BOM's manifest totals remain the
indexed reference totals; run and evaluation evidence describe actual training
consumption.

For `causal-pretrain-weighted`, prefer inline `weight` fields. Existing
`parameters.corpus_weights` maps remain valid for compatibility, but a stage
must use one representation or the other, never both.

## Training parameter fields

| Field | Required | Default or range | Meaning |
| --- | --- | --- | --- |
| `profile` | no | `causal-pretrain-shuffled` | Selects versioned record ordering, corpus exposure, and held-out selection. |
| `tokens` | one training budget | positive integer | Fixed pretraining target budget. WALDO rounds it up to a complete optimizer step and persists the derived step count. Cannot be combined with `epochs` or `steps`. |
| `epochs` | one training budget | `1..1000000` | Complete deterministic passes over every selected canonical record. When `steps` is omitted, WALDO derives the exact optimizer-step count after filtering and held-out selection. |
| `steps` | legacy/fixed-step budget | positive integer | Explicit optimizer steps and learning-rate schedule length. Retained for existing composes and exact fixed-step experiments; it may be combined with `epochs` as a repetition limit. |
| `batch_size` | yes | positive integer | Number of packed sequences in each optimizer step. |
| `sequence_length` | yes | positive integer, at most `context_tokens` | Number of predicted token targets per packed sequence. |
| `learning_rate` | yes | finite positive number | Peak AdamW learning rate. |
| `seed` | no | default `0` | Controls deterministic shuffling, evaluation selection, initialization, and training randomness. Reference composes set it explicitly. |
| `weight_decay` | no | default `0.1`; `0..1` | AdamW weight decay. Explicit zero disables it. |
| `warmup_steps` | no | `min(100, steps/10)`; `0..steps` | Linear warmup duration. For runs longer than one step, the default is at least one. Explicit zero disables warmup. |
| `checkpoint_every` | no | `min(500, steps)`; `0..steps` | Checkpoint interval. Explicit zero disables periodic checkpoints. |
| `evaluate_every` | no | `min(500, steps)`; `0..steps` | Held-out evaluation interval. Explicit zero disables periodic evaluation. |
| `shuffle_buffer_records` | no | default `1024`; `1..1000000` | Maximum records retained by deterministic bounded shuffle. |
| `shuffle_buffer_bytes` | no | default 64 MiB; `1 B..16 GiB` | Maximum record text retained by deterministic bounded shuffle. |
| `corpus_weights` | only for `causal-pretrain-weighted`; legacy form | each weight `1..1000000` | Integer relative token exposure keyed by every selected corpus path. Configured corpus `weight` fields are preferred. |
| `evaluation_fraction` | no | default `0.01`; `0 <= value < 1` | Candidate fraction for deterministic held-out selection. |
| `evaluation_max_records` | no | default `256`; `0..1000000` | Held-out record cap. |
| `evaluation_max_bytes` | no | default 1 MiB; `0 B..16 GiB` | Held-out text-byte cap. |

Exactly one of `tokens`, `epochs`, or legacy `steps` is normally declared. A
legacy compose may declare both `steps` and `epochs`; steps remains the exact
stop while epochs limits the finite source passes available to reach it.

For fixed-token and fixed-step stages, the planned token capacity is:

```text
derived_steps * batch_size * sequence_length
```

Records are continuously packed with an EOS token between records; document
boundaries do not force padding to a new sequence. Epoch-driven stages scan the
finite filtered stream and derive their exact steps before creating a run.
Fixed-token stages derive steps without a full scan and retain a single source
pass. Legacy stages declaring both fields verify that their epochs contain
enough packed targets to reach the requested steps. A run fails rather than
silently shortening its declared budget.

Setting any one of `evaluation_fraction`, `evaluation_max_records`, or
`evaluation_max_bytes` to zero disables the held-out set and resolves all
three values to zero.

### Fixed profile behavior

All profiles resolve to AdamW with betas `0.9` and `0.95`, epsilon `1e-8`, and
a cosine schedule ending at 10% of the peak learning rate. Those values and
continuous EOS packing are versioned profile facts, not compose fields.

A schema-1 compose has no fields for arbitrary chat-template expressions,
optimizer choice, gradient accumulation, activation checkpointing,
mixture-of-experts routing, or distributed topology. The optional built-in
interaction contract controls inference formatting; it does not change the
causal training objective. Hardware and backend topology remain machine-local
policy; other training behaviors require a separately versioned portable
contract rather than an ignored compose field.

| Profile | Training record order | Held-out selection | Corpus weights |
| --- | --- | --- | --- |
| `causal-pretrain-shuffled` | One bounded deterministic shuffle across the selection. | Deterministic lowest SHA-256 candidates across the selection. | Not accepted. |
| `causal-pretrain-balanced` | Balances emitted tokenizer targets equally across logical corpus paths, with bounded shuffle within each. | Deterministically stratified across corpus paths. | Not accepted. |
| `causal-pretrain-weighted` | Selects the corpus with the lowest emitted-target-to-declared-weight ratio, with bounded shuffle within each. | Deterministically stratified across corpus paths. | Required for every selected corpus. |

Each named profile currently has `profile_schema: 1` in the resolved run BOM.
That field versions the behavior contract independently; it is not a model
architecture version. The former `causal-pretrain-v1`, `-v2`, and `-v3` names
remain accepted as deprecated input aliases and resolve respectively to
`shuffled`, `balanced`, and `weighted`. New resolved run BOMs use the
behavior-named identities.

Use `causal-pretrain-balanced` when every selected corpus should receive equal
token exposure. Use `causal-pretrain-weighted` when the intended mixture is
unequal. Weights are relative—for example, `2` and `1` target approximately
twice as many emitted training tokens from the first corpus while it remains
available. They do not duplicate canonical records or alter corpus provenance.
Consequently, weights determine total mixture exposure for a fixed-token stage
that stops early; an epoch-driven stage still consumes every filtered record
from every selected corpus once per epoch, with weights affecting interleaving
rather than final totals.

## Common compose patterns

### Minimal model from scratch

```yaml
kind: waldo-model-compose
schema: 1
architecture:
  family: decoder-transformer
  context_tokens: 512
  vocabulary_size: 50259
  hidden_size: 512
  intermediate_size: 1536
  layers: 8
  attention_heads: 8
  key_value_heads: 2
  tie_embeddings: true
  parameter_dtype: bfloat16
  tokenizer:
    name: tiktoken/r50k_base
    revision: tiktoken-r50k-base
stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    corpora: [core/books/gutenberg, science/plos]
    parameters:
      profile: causal-pretrain-balanced
      tokens: 1048576000
      batch_size: 64
      sequence_length: 512
      learning_rate: 0.0003
      seed: 42
```

### Initialize from managed weights

```yaml
base:
  model: conversation
```

Add this block to a complete compose whose architecture exactly matches
`conversation`. WALDO selects and pins its latest completed real checkpoint,
falling back to a verified pulled origin if no completed checkpoint exists.

To acquire the origin directly from a supported external source, pin its
immutable commit and omit `architecture` to inherit the verified definition:

```yaml
base:
  source: huggingface://organization/model@0123456789abcdef0123456789abcdef01234567
```

See [Base initialization](#base-initialization) for the complete source-based
compose, acquisition behavior, authentication, cache, provenance, and current
compatibility boundary.

### Multiple ordered stages

```yaml
stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    corpora: [core/books/gutenberg, science/plos]
    parameters:
      profile: causal-pretrain-balanced
      tokens: 1048576000
      batch_size: 64
      sequence_length: 512
      learning_rate: 0.0003
      seed: 42

  - name: domain-tune
    type: fine-tuning
    objective: causal-language-modeling
    corpora: [core/common-pile/python-enhancement-proposals/peps]
    parameters:
      profile: causal-pretrain-shuffled
      epochs: 1
      batch_size: 32
      sequence_length: 512
      learning_rate: 0.00005
      seed: 43
```

The second stage starts from the first stage's completed weights. The
`fine-tuning` type is provenance; both stages currently use the causal
language-modeling objective.

## Validation and failure behavior

WALDO fails before training when a compose has:

- an unknown field, extra YAML document, unsupported kind, or schema;
- an unsupported architecture, tokenizer contract, dtype, or objective;
- zero architecture dimensions or incompatible attention-head divisibility;
- dropout outside `0..<1` or a sequence longer than the architecture context;
- no stages, duplicate stage names, no corpus selection, or duplicate corpora;
- invalid parameter ranges or an overflowing planned token capacity;
- corpus weights outside `causal-pretrain-weighted`, missing weighted-profile
  weights, or weights for unselected corpora;
- a base whose source is mutable or whose origin, architecture, or current weights do not match; or
- an existing destination model with a different immutable architecture.

During training, WALDO fails rather than accepting incomplete steps,
unaccounted corpus exposure, corrupt checkpoints, or artifacts that do not
match their recorded hashes.

## Portability and identity

Architecture, tokenizer, resolved profile, corpus BOMs, backend identity,
evaluation selection, checkpoints, telemetry, and output artifacts are
persisted in model and run records. Changing architecture fields creates a
different architecture identity. Changing stage parameters or corpus weights
creates a different run contract and prevents an incompatible checkpoint from
being resumed.

The compose remains portable because backend selection and hardware are not
part of it. Portability means WALDO can select any backend that implements the
declared architecture and objective; it does not promise that every compose
fits every machine. Run `waldo model forecast` before allocating substantial
compute.

Schema 1 is limited to WALDO's dense decoder architecture and current training
strategies. The planned contract for foundation adaptation, native imported
architectures, parameter-efficient tuning, stage lineages, and sparse-MoE
models is maintained in the
[Foundation, post-training, and sparse-MoE plan](FOUNDATION-MOE-PLAN.md). That
plan is not accepted schema-1 compose syntax.

See the measured examples in [`composes/`](../composes/), the broader
[model lifecycle](MODEL-LIFECYCLE.md), and the
[model export guide](MODEL-EXPORTS.md).
