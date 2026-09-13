# Training robustness and compute-efficiency plan

This plan targets capability per training FLOP, not feature parity for its own
sake. The comparison baseline is nanochat commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` (2026-07-03). WALDO keeps its
stronger immutable BOM, provenance, replay, and artifact contracts while
adopting the training practices that make nanochat efficient and measurable.

The companion holding compose is
[`composes/holding/nanochat-open-baseline.yaml`](../composes/holding/nanochat-open-baseline.yaml).
It is a target experiment, not a claim that the current trainer can execute the
recipe efficiently.

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
resume, and contamination gates. Only then run the holding 24-layer baseline.

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
