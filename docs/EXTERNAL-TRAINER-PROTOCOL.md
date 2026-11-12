# External trainer protocol 1

This is the implemented developer API contract in `internal/model/trainerapp`,
under ADR [0084](adr/0084-external-trainers.md). Terminal tensor/file verification and developer plan/run/model BOM persistence
are implemented, and the CLI lifecycle drives them (ADR 0087). See
[OpenWALDO BOM](OPENWALDO-BOM.md#external-model-provenance).
Checkpoint verification and conformance are implemented; see
[external checkpoints](EXTERNAL-CHECKPOINTS.md). The built-in workers'
internal schema-1 protocol is unchanged and is not this protocol.

## Transport and ownership

WALDO launches each command through `applaunch` with a freshly verified source
and environment, the manifest's command argument, and a caller-owned deadline
or cancellation context. Stdout is NDJSON only; stderr is separately captured.
Each line is one JSON object, smaller than 16 MiB including its newline.
Unknown fields, unknown frame kinds, unsupported protocol versions, missing or
null required fields, malformed JSON and extra responses fail closed. Fields
inside opaque schema-defined objects and environment evidence are not filtered.
A successful command must exit zero. An error kills and reaps the process;
there are no detached children or outstanding producer goroutines on return.
Call a client sequentially; it records launch evidence across its commands.

All examples below are individual frames, formatted on one line. Hash strings
and paths are illustrative; real hashes must match the actual bytes/values.

WALDO owns corpus selection, order, held-out selection, tokenization, packing,
mask construction, exact step budgets and resume positioning. The trainer must
never read an index or Parquet. The training API takes **already packed**
`training.PreparedSequence` values from a cancellation-aware `SequenceSource`;
it does not take raw text. Lifecycle integration must connect WALDO's prepared
stream, not delegate packing to the trainer. Distributed launch uses node-local sequence ownership and explicit global
micro-batch boundaries; see [Writing external trainers](WRITING-EXTERNAL-TRAINERS.md#distributed-trainers).

## Probe

WALDO sends one line, closes stdin, reads exactly one response, then waits for
successful process exit:

```json
{"kind":"probe","protocol":1}
```

```json
{"kind":"probe","protocol":1,"capabilities":{"objectives":["causal-language-modeling"],"launch_profiles":["single-process"],"resume":true},"frameworks":{"torch":"2.9.0"},"devices":[{"name":"CPU","kind":"cpu","memory_bytes":8589934592}],"driver":"unavailable","cuda":"unavailable","schemas":{"architecture":{"type":"object","properties":{"width":{"type":"integer"}},"required":["width"]},"parameters":{"type":"object","properties":{"learning_rate":{"type":"number"},"micro_batch_sequences":{"type":"integer","x-waldo-role":"execution"}}}}}
```

Every shown envelope field is required. Devices require name, kind and positive
memory bytes; framework versions are a nonempty string map. Use `unavailable`
for absent driver/CUDA facts. Framework/device/driver/CUDA facts and launch
observations are run evidence, never model identity. Resume support must be
true. Every requested objective must occur in both manifest and probe, and the
selected launch profile must occur in both. These checks happen before any
caller creates model/run state.

Both schema documents come exclusively from probe. They are self-contained
JSON Schema 2020-12 documents; architecture allows only recipe roles.
`application.CompileSchema` and `ParameterSchema.Split` enforce full validation
and `x-waldo-role` rules. Unknown vendor annotations survive unchanged.
`Schemas()` returns the original bytes, and `SchemaHashes()` hashes each exact
document. `SchemaDocument(tokenizerSchema)` creates the schema-1
`waldo-trainer-schemas` envelope containing `architecture`, `parameters` and
`tokenizer` **without re-encoding their bytes**. The lifecycle writer must
atomically persist those returned bytes as `TRAINER-SCHEMAS.json` beside
`PLAN.json`, with all three document hashes in identity. Do not pass that file
through a JSON pretty-printer or marshaler that compacts raw documents.

## Resolve

For every stage, before computing any plan hash, call `Client.Resolve` with
architecture, stage trainer parameters, WALDO-resolved data and actual world
shape. WALDO validates compose values against both full schemas first; schema
validation does not insert defaults or remove required constraints.

```json
{"kind":"resolve","protocol":1,"architecture":{"width":64},"parameters":{"learning_rate":0.001},"data":{"steps":1,"global_batch_sequences":2,"sequence_length":2,"seed":42,"order":{},"checkpoint_every":1,"evaluate_every":1,"planned_token_capacity":4},"world":{"nodes":1,"node_rank":0,"processes_per_node":1,"world_size":1,"devices":[{"name":"CPU","kind":"cpu","memory_bytes":8589934592}]}}
```

```json
{"kind":"resolve","protocol":1,"parameters":{"learning_rate":0.001,"micro_batch_sequences":2},"batch":{"micro_batch_sequences":2,"accumulation_steps":1,"world_size":1}}
```

`parameters` is the complete object, with every trainer default explicit.
`batch` is required, separate protocol metadata because the opaque parameter
schema may use any names/nesting for execution settings. The trainer must make
this metadata agree with its execution parameters. WALDO checks positive
integers and `micro_batch_sequences × accumulation_steps × world_size ==
global_batch_sequences` without multiplication overflow.

WALDO calls resolve again with one node/process and the same probe devices,
architecture, data and recipe inputs, even when the actual shape is already
single-process. Each response is validated and split by schema role. Different
recipe hashes fail. An unsupported recipe value must produce an error; a
trainer must not silently downgrade precision or otherwise change the recipe.
Stage `trainer` and `execution` inputs are checked against schema roles before
merging into the wire `parameters`. The comparison resolve omits execution
inputs so a world-specific decomposition does not prevent the one-process check.
Only the resolved recipe and its hash enter plan identity. Execution values,
batch decomposition and the complete parameter hash are run evidence. The
complete resolved object and its hash are passed to `begin`.

## Training input

WALDO sends one `begin`, the pinned evaluation records, the remaining packed
training sequences, then one `end`, and closes stdin. Output is read concurrently.

```json
{"kind":"begin","protocol":1,"begin":{"paths":{"artifacts":"/run/artifacts","checkpoints":"/run/checkpoints","scratch":"/run/scratch"},"model_id":"model","run_id":"run","stage":"pretrain","objective":"causal-language-modeling","data":{"steps":1,"global_batch_sequences":2,"sequence_length":2,"seed":42,"order":{},"checkpoint_every":1,"evaluate_every":1,"planned_token_capacity":4},"architecture":{"width":64},"architecture_sha256":"<architecture digest>","parameters":{"learning_rate":0.001,"micro_batch_sequences":2},"parameters_sha256":"<complete parameters digest>","batch":{"micro_batch_sequences":2,"accumulation_steps":1,"world_size":1},"tokenizer":{"name":"example","revision":"<tokenizer identity>","vocabulary_size":259,"pad_id":0,"bos_id":1,"eos_id":2},"evaluation_set":{"selection":"pinned","seed":42,"records":1,"token_targets":2,"text_bytes":2,"sha256":"<selection digest>"},"initialization":null,"resume":null,"determinism":"best-effort","world":{"nodes":1,"node_rank":0,"processes_per_node":1,"world_size":1}}}
```

Paths must be absolute; scratch and world must match launch evidence. Architecture
and parameters use SHA-256 of Go's JSON encoding of normalized JSON objects
(sorted object keys, no indentation/newline). The resolver supplies the parameter
hash. Initialization, when present, uses the existing WALDO initialization facts
(source type/ID/run, artifact and verified absolute path). Resume, when present,
is `{step, tokens, checkpoint_manifest, paths}`. WALDO owns the manifest
format, compatibility and file verification, and maps each manifest filename
to a verified absolute bundle path in `paths`. `LoadCheckpoint` supplies the
only accepted resume value; `Train` revalidates it before launch. The
trainer restores optimizer, scheduler and RNG state, not just weights.

```json
{"kind":"evaluation_record","protocol":1,"record":{"id":"heldout-1","tokens":[1,107,108],"loss_mask":[true,true],"text_bytes":2}}
```

```json
{"kind":"record","protocol":1,"record":{"ordinal":0,"tokens":[1,107,108],"loss_mask":[true,true],"consumption":{"books":2}}}
```

A sequence has `sequence_length + 1` token IDs. Mask entry `i` supervises the
prediction of token `i+1` from the preceding tokens. No EOS insertion, shifting
of masks, repacking, shuffling or tokenizer calls belong in the trainer. IDs
must be in vocabulary; at least one target is supervised. Per-corpus consumption
sums to supervised targets. Ordinals are contiguous, starting at
`resume.step × global_batch_sequences` (zero for a fresh run). Exactly
`steps × global_batch_sequences` sequences, including the resumed prefix, make
the budget; too few or too many fail. Evaluation masks have the same target
semantics, with arbitrary record lengths; counts and UTF-8 byte totals must
match the pinned evaluation set before launch.

Trainers **must normalize loss by supervised target tokens across the whole
global batch**, not average per-micro-batch means. This is necessary for recipe
invariance under changed batch decomposition.

```json
{"kind":"end","protocol":1}
```

## Training output

An `event` contains exactly one typed event. Step and token counters are required
and nonnegative. Progress/checkpoint counters may repeat but never regress.
`log` requires a message, `progress` requires a finite loss, `checkpoint` requires
its matching checkpoint payload, and `evaluation` requires its matching metrics.

```json
{"kind":"event","protocol":1,"event":{"kind":"log","step":0,"tokens":0,"message":"initialized"}}
{"kind":"event","protocol":1,"event":{"kind":"progress","step":1,"tokens":4,"loss":1.0}}
{"kind":"event","protocol":1,"event":{"kind":"checkpoint","step":1,"tokens":4,"checkpoint":{"step":1,"tokens":4,"artifacts":[{"path":"step-1/model.safetensors","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bytes":16,"role":"weights","format":"safetensors"}]}}}
{"kind":"event","protocol":1,"event":{"kind":"evaluation","step":1,"tokens":4,"evaluation":{"step":1,"tokens":4,"nll_sum":2.0,"target_tokens":2,"text_bytes":2,"loss":1.0,"bits_per_byte":1.4426950408889634}}}
```

NLL uses natural logarithms. WALDO checks finite nonnegative NLL, pinned target
and byte counts, `loss = nll_sum / target_tokens`, and
`bits_per_byte = nll_sum / (text_bytes × ln(2))`, with relative tolerance 1e-8
and absolute floor 1e-8. Evaluation may report an earlier checkpoint; it does
not move the current training cursor backwards. `Event.Telemetry()` adapts to
existing `training.Event` progress/telemetry sinks while the original event
retains artifact roles and all evaluation evidence. A sink error stops training.

Exactly one terminal `complete` is required on success, followed by EOF and
zero exit. Every field shown below is required; arrays may be empty only where
semantically allowed (e.g. evaluations when no evaluation set exists).

```json
{"kind":"complete","protocol":1,"complete":{"steps":1,"consumed_tokens":4,"artifacts":[{"path":"model.safetensors","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bytes":16,"role":"weights","format":"safetensors"}],"artifact_profile":"hf-causal-lm","consumption":[{"corpus":"books","token_targets":4}],"evaluations":[{"step":1,"tokens":4,"nll_sum":2.0,"target_tokens":2,"text_bytes":2,"loss":1.0,"bits_per_byte":1.4426950408889634}],"selected_checkpoint":{"step":1,"tokens":4,"metric":"heldout_loss","value":1.0},"artifact_evaluation":{"step":1,"tokens":4,"nll_sum":2.0,"target_tokens":2,"text_bytes":2,"loss":1.0,"bits_per_byte":1.4426950408889634},"environment":{"torch":"2.9.0"}}}
```

Artifact declarations require a unique clean relative path, lowercase SHA-256,
nonnegative size, role and format. Completion paths are relative to artifacts;
checkpoint paths are relative to checkpoints. Checkpoint events verify and
commit filesystem contents before the event sink is called. Terminal artifacts
are verified after successful process exit: rooted containment, sizes/hashes,
Safetensors headers and finite floating tensors. Published roles are `weights`
with format `safetensors`, or `configuration`/`tokenizer` with format `json`
(JSON files are limited to 16 MiB). Other roles/formats fail closed.
Checkpoint pickle is private resume state; trainers should use
`weights_only=True` when loading it.

Completion must reach the exact step budget, retain monotonic counters, match
the manifest artifact profile and reconcile consumption totals. Fresh attempts
also reconcile each corpus against sent masks. Resumed attempts merge verified
checkpoint consumption with current per-corpus sent counts and check the full
cumulative result. When held-out evaluation exists, selection
must name the lowest-loss reported candidate, ties choosing the earliest step.
Without evaluation, select the terminal step. `Train` additionally
verifies that selected and evaluated candidates exist as complete authenticated
checkpoint bundles. On resume, callers supply independently persisted
`Begin.PriorCheckpoints` and `Begin.PriorEvaluations`; neither is sent to the
trainer. Completion must preserve observed evaluation history.
When a held-out set exists, completion also requires `artifact_evaluation`,
with the same fields as an evaluation, describing the reloaded published
weights at the selected step/tokens. Its NLL arithmetic/counts must match pins
and its loss must agree with selection within max(0.02, 1%). Without a held-out
set this field is omitted. WALDO trusts the reported loss beyond these checks.
Successful `Train` verifies output bytes but does not copy or publish them;
callers must reverify files at publication/export time.

## Errors

Every command may instead emit:

```json
{"kind":"error","protocol":1,"error":"requested precision unavailable","error_class":"configuration"}
```

The message and class must be nonempty. Classes are extensible strings;
`artifact-integrity` and `numerical-integrity` reuse WALDO's non-retryable worker
error classification. Process failures retain exit status and captured stderr.
Cancellation, producer failure, missing completion, duplicate completion,
frames after completion and protocol violations all fail the attempt.
