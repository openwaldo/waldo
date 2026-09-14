# Training robustness and compute-efficiency plan

This plan targets capability per training FLOP, not feature parity for its own
sake. The comparison baseline is nanochat commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` (2026-07-03). WALDO keeps its
stronger immutable BOM, provenance, replay, and artifact contracts while
adopting the training practices that make nanochat efficient and measurable.

The comparison target is the numbered
[`composes/0003-conversation.yaml`](../composes/0003-conversation.yaml), trained
as model `conversation2`. The existing `0000` and `0001` composes provide its
canary and systems gates, and `0002-conversation.yaml` remains the known-good
comparison model.

## Definition of done

WALDO is competitive when one pinned WALDO run and one pinned nanochat run,
using the same accelerator class and measured training FLOPs, show:

- WALDO reaches at least the nanochat validation BPB and normalized evaluation
  score, with no material regression on conversation or code evaluation.
- Median model FLOP utilization, tokens/second, step time, peak accelerator
  memory, data-wait time, and optimizer-step count are recorded in the run
  artifact.
- Resuming from a checkpoint reproduces the uninterrupted run within a declared
  numeric tolerance.
- Every training and evaluation input is revision-pinned, license-approved,
  contamination-checked, and represented by an immutable BOM.
- A failed experiment is stopped by a small-run gate before the full token
  budget is spent.

Held-out loss alone is not a promotion gate.

## Work sequence

### 0. Establish a trustworthy baseline

Before another multi-day model run:

1. Fix the resource forecast so global batch is divided across workers before
   estimating per-device activation memory.
2. Add a deterministic one-GPU and multi-GPU golden run with checkpoint/resume
   comparison, including sample order and consumed-token accounting.
3. Record wall time, tokens/second, step time, data-wait time, peak memory,
   estimated training FLOPs, MFU, gradient norm, and skipped/non-finite steps.
4. Pin the hardware, software, compose, corpus BOM, tokenizer, seed, and
   evaluation BOM in one comparison report.

Exit gate: estimates agree with observed peak memory within 15%, token counts
are exact, and resumed and uninterrupted golden runs pass.

### 1. Remove data-plane waste

The current PyTorch worker has rank 0 decode JSON records and synchronously
broadcast each Python object to all ranks. Replace this with a prepared,
tokenized stage artifact that is deterministically partitioned by rank and read
with asynchronous prefetch. Preserve the original document and shard lineage
in the artifact manifest.

Add bounded queues, pinned-memory transfer where useful, and separate metrics
for object-store read, decode/tokenize, host wait, and device wait.

The planned distribution design is:

1. Canonical tokenized shards remain immutable, content-addressed objects in
   lookaside/object storage.
2. One cache coordinator per node downloads and verifies each required shard
   once into a bounded node-local NVMe cache.
3. All local ranks read the shared read-only node cache, but receive
   deterministic, non-overlapping shard or byte-range assignments.
4. Small stages may be mirrored completely once per node. Large stages use a
   rolling prefetch window and evict only shards outside the resume window.
5. On restart, the saved sampler position reconstructs the same rank
   assignments independent of cache contents.

Do not mirror the complete corpus per rank. Ordinary NFS is not the primary
training data path because metadata contention and synchronized reads can
stall every accelerator. A high-throughput parallel filesystem may be a
measured fallback, and NFS may carry small metadata, but either must pass the
same data-wait gate as node-local caching.

Exit gate: accelerators wait for input less than 2% of training wall time and
adding ranks does not multiply rank-0 CPU or network work.

### 2. Decouple logical batch from device batch

Add gradient accumulation to the portable training contract and backend. The
compose must express a global token batch; the backend derives a safe
micro-batch per rank and accumulation count from world size and observed
memory. Accumulated loss scaling, clipping, scheduler steps, checkpoint state,
and token accounting must be tested together.

Add activation checkpointing as an explicit, reported execution option. Add
CUDA FP16 gradient scaling or reject FP16 on CUDA; BF16 remains the default.

Exit gate: changing world size or micro-batch while holding the global token
batch fixed produces equivalent short-run learning curves.

### 3. Make the training recipe tunable and measurable

Version a new compose contract for optimizer and schedule choices. Keep AdamW
as the correctness baseline, then compare nanochat's mixed Muon/AdamW approach
as a controlled experiment. Add linear warmup, constant middle, linear
warmdown, depth-scaled initialization, zero-dropout, QK normalization, and
untied embeddings as explicit choices rather than hidden backend behavior.

Run small scaling sweeps over depth, token-to-parameter ratio, global token
batch, peak learning rate, and weight decay. Select the full run from measured
validation BPB and task score per FLOP, not a copied hyperparameter.

Exit gate: at least three model sizes produce a coherent scaling curve and the
chosen recipe beats the current WALDO AdamW baseline at equal FLOPs.

### 4. Add capability gates

Create immutable evaluation BOMs, kept out of training unless a split is
explicitly part of supervised training. The minimum gate is:

- base: validation BPB plus a broad normalized multiple-choice/continuation
  suite comparable to CORE;
- chat: ARC-Easy, ARC-Challenge, MMLU test, GSM8K test, and HumanEval;
- WALDO regressions: fixed generation, conversation, and code behavior cases;
- contamination: exact and fuzzy overlap reports against every training BOM.

Evaluation inputs must use reviewed upstream releases, not nanochat's
unversioned downloadable evaluation bundle.

Exit gate: evaluation is reproducible from BOMs and automatically blocks model
promotion when any declared threshold or contamination limit fails.

### 5. Train the tokenizer on the admitted corpus

Train and version a roughly 32K-token tokenizer on a deterministic sample of
the approved pretraining mixture. Compare fertility, bytes per token, and
downstream BPB against `r50k_base`, including prose, code, mathematics, and
conversation data. The tokenizer artifact and sample BOM become part of the
model identity.

Exit gate: the tokenizer improves or preserves compression on every critical
domain and round-trip, special-token, and distributed-training tests pass.

### 6. Run the parity ladder

Run canary, then approximately 100M-, 300M-, and 700M-parameter experiments.
Each rung gets a fixed FLOP budget and must pass throughput, loss, capability,
resume, and contamination gates. Only then run the 24-layer `conversation2`
candidate.

Do not start the current `0003-conversation.yaml` multi-day run before phases
0 through 4 pass. Its 22B-token curriculum is too expensive to use as the
systems benchmark.

## Durable contract work

The following need reviewed ADRs and versioned schemas before implementation:

- prepared tokenized stage artifacts and deterministic rank partitioning;
- global token batch, micro-batch, and gradient accumulation semantics;
- optimizer, learning-rate schedule, activation checkpointing, and precision
  execution fields;
- evaluation BOMs, metric definitions, contamination reports, and promotion
  thresholds;
- tokenizer-training inputs and tokenizer artifact identity.

Backend-only tuning that affects reproducibility must still be recorded in the
run artifact.

## Integration and verification ladder

Every implementation change moves through the same test funnel. A failure at
one level stops progression; it is not compensated for by a larger run.

### Test levels

| Level | Purpose | Required checks |
| --- | --- | --- |
| Unit | Prove the isolated calculation or state transition | boundary cases, invalid configurations, deterministic serialization |
| Component | Prove two cooperating subsystems | sampler/cache, accumulation/scheduler, checkpoint/RNG, evaluator/BOM |
| Local end-to-end | Prove a complete lifecycle cheaply | ingest or resolve, train, checkpoint, resume, evaluate, export |
| Distributed golden | Detect rank-dependent errors | 1 versus 2 GPUs initially; identical global batch, sample coverage, optimizer steps, and equivalent loss |
| Model rung | Verify learning and efficiency | fixed corpus BOM, evaluation BOM, seed set, hardware description, and FLOP budget |

Use at least three fixed seeds for model-quality comparisons. Correctness tests
may use one fixed seed when they compare exact saved state.

### Model rungs

| Rung | Approximate size | Initial budget | What it proves | Promotion gate |
| --- | ---: | ---: | --- | --- |
| G0 canary | 14M | 100M tokens maximum | complete pipeline correctness | falling loss; exact token/sample accounting; uninterrupted and resumed state match; no non-finite steps |
| G1 systems | 75M | 600M tokens maximum | sustained data and distributed efficiency | data wait below 2%; forecast within 15%; useful multi-GPU scaling; no resume drift |
| G2 capability | 300M | 2.4B tokens maximum | tokenizer, optimizer, schedule, and evaluation choices | beats G1 scaling prediction and current AdamW/r50k baseline at equal FLOPs across three seeds |
| G3 parity | approximately 757M | approximately 5.84B tokens | nanochat-class comparison | meets or beats the pinned nanochat reference on BPB and normalized capability per FLOP without WALDO regression failures |

The token ceilings roughly follow the nanochat target of eight training tokens
per scaling parameter. Early stopping is mandatory when a run cannot meet its
promotion projection.

For day-to-day integration, run G0 for only 1M-5M tokens after each change and
G1 for roughly 25M-50M tokens after each major phase. Spend the full rung budget
only once when qualifying that rung for promotion.

### Change-by-change rollout

1. **Benchmark harness and telemetry.** Capture the current G0 result before
   changing training. Add step, input-wait, checkpoint, evaluation, memory,
   FLOP, and MFU measurements. Re-run G0 to prove measurement overhead is
   bounded and results are unchanged.
2. **Forecast correction.** Add formula tests for global batch, world size,
   accumulation, dtype, optimizer state, and activation checkpointing. Confirm
   with observed G0 memory, then with G1 on one and two GPUs.
3. **Prepared tokenized artifacts and node cache.** Test hash verification,
   interrupted downloads, eviction, deterministic rank partitioning, and
   restart. Compare the old and new G0 sample stream exactly. Advance to G1
   only after input wait and scaling gates pass.
4. **Gradient accumulation.** Compare one large physical batch with several
   accumulated micro-batches using the same global token batch. Test clipping,
   scheduler steps, partial accumulation rejection/recovery, and checkpoint
   resume. Verify on G0, then G1 across different world sizes.
5. **Deterministic distributed resume.** Interrupt G0 and G1 at checkpoints and
   during shard transitions. Verify model, optimizer, scheduler, scaler,
   sampler, RNG, and consumed-token state against uninterrupted runs.
6. **Precision, compilation, and memory controls.** Establish BF16 as the
   reference. Test FP16 scaling and overflow recovery, activation
   checkpointing, `torch.compile`, and later FP8 independently. Promote an
   optimization only when G1 improves throughput without quality or resume
   regression.
7. **Evaluation and contamination gates.** Build approved evaluation BOMs and
   prove split isolation with synthetic overlaps. Run the complete gate on G1;
   scores may be low, but execution and normalization must be reproducible.
8. **Tokenizer and recipe sweeps.** At G1, compare the admitted-corpus 32K
   tokenizer, optimizer, learning-rate schedule, batch, and initialization one
   variable at a time. Confirm the selected combination at G2 across three
   seeds.
9. **Parity run.** Freeze code, environment, compose, corpus/evaluation BOMs,
   tokenizer, seed set, and hardware. Run G3 only after every earlier gate is
   green, then publish both quality and efficiency results, including failed
   or stopped runs.

Each accepted change gets its own signed-off commit with its unit/component
tests. Model rung evidence is attached to the run artifacts and referenced by
the subsequent commit or decision record.
