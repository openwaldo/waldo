# Reference-model training ladder

This directory contains the active, gated path from a reproducible narrow
language model toward broader capability. Every rung has one hypothesis, a
pinned compose, fixed prompts, and explicit promotion criteria. A failed rung
stops the ladder; it does not justify changing several variables at once.

Historical composes remain under [`archive`](archive). The `experiments`,
`general-foundation`, and `tinystories` subdirectories preserve earlier
diagnostics and are not active ladder rungs.

## Rung 0001: Tiny Shakespeare reference — passed

[`0001-tiny-shakespeare.yaml`](0001-tiny-shakespeare.yaml) is WALDO's first
reference model. It uses the exact 1,115,394-byte Tiny Shakespeare text, the
built-in byte tokenizer, a 10.7M-parameter decoder, and a deterministic 90/10
contiguous split that does not introduce artificial line-level EOS tokens.

Validated result on 2026-10-05:

- model `tiny-shakespeare-control-01`, ID `b4f8477a55ad`;
- 5,000 optimizer steps and 81.92M consumed tokens;
- held-out loss improved from 5.3814 to 1.5117;
- step 1,250 was correctly selected and reloaded after terminal loss rose to
  2.0940;
- all eight temperature-0.8 samples preserved play formatting and produced
  locally plausible Shakespeare-like text without immediate loop collapse;
- greedy decoding exposed a repeat attractor around "season/state/seas"; and
- training completed in under 15 minutes on two H200 GPUs.

The full 81.92M-token compose is retained because it reproduces the learning
curve, overtraining evidence, and selected checkpoint. Changing its horizon to
20.48M would also change the cosine schedule and would not reproduce the same
checkpoint.

Run and evaluate it with:

```console
go run ./cmd/waldo/ model forecast composes/0001-tiny-shakespeare.yaml
go run ./cmd/waldo/ model train tiny-shakespeare-control-01 \
  composes/0001-tiny-shakespeare.yaml
./composes/evaluate-tiny-shakespeare.sh \
  tiny-shakespeare-control-01 /tmp/tiny-shakespeare-control-01-eval.jsonl
```

This rung proves that WALDO's ingestion, byte tokenization, packing, optimizer,
held-out evaluation, checkpoint selection, reload, and raw generation paths can
learn a real language distribution. EOS is not a gate because this corpus is
one continuous document.

## Rung 0002: TinyStories byte control — diagnostic complete

[`0002-tinystories-byte.yaml`](0002-tinystories-byte.yaml) asks whether the
same proven 10.7M model can move from one play-like stream to many short,
simple stories. It keeps the architecture, tokenizer, context, batch,
optimizer, learning rate, dropout, initialization, and seed from rung 0001.
The intentional changes are the corpus, record-level evaluation, shuffle
capacity, and the longer 15,000-step horizon needed to encounter diverse
stories.

The corpus is the first pinned training Parquet shard from the original
TinyStories release. This bounded quarter-corpus makes the rung practical on a
Mac while retaining hundreds of thousands of complete story records. It is a
WALDO-shaped learning control, not yet an exact reproduction of the paper's
alternating GPT-Neo attention or pruned tokenizer. The paper and published
prompts are available from the
[TinyStories project](https://huggingface.co/datasets/roneneldan/TinyStories)
and [paper](https://arxiv.org/abs/2305.07759).

Hypothesis: changing only to a constrained simple-English story distribution
will preserve grammatical generation while improving entity, causal, and
short-plot consistency beyond the Shakespeare style control.

Promotion gates:

1. Training completes without non-finite loss, and the selected checkpoint's
   reloaded held-out loss agrees with its persisted evaluation.
2. Best held-out loss improves by at least 50% from the initial evaluation.
3. At least six of eight temperature-0.8 samples remain grammatical and retain
   the prompt's people, objects, and causal setup for at least 150 generated
   byte tokens.
4. No more than two samples collapse into an immediate repeated sentence or
   phrase loop.
5. EOS is measured at a horizon calibrated to the corpus record-length
   distribution. Unlike rung 0001, every TinyStories record teaches a real
   document boundary.
6. Greedy output is recorded as a degeneration diagnostic, but it is not the
   sole promotion decision.

Validated results on 2026-10-05:

- the four-GPU, two-host run reached held-out loss 0.5782 after 245.8M tokens;
- the two-GPU, single-host control reached 0.5794 with the same architecture,
  data, token budget, global batch, optimizer steps, and seed;
- the 0.21% final-loss difference passes the 3% topology-equivalence gate and
  rules out multi-host data parallelism as the cause of the generation issues;
- both runs produced grammatical simple-story prose at temperature 0.8, but
  lost prompt entities and causal details, exhibited greedy repetition, and
  rarely emitted EOS within the evaluation horizon; and
- rung 0002 therefore passes learning and execution controls but fails its
  prompt-retention and repetition capability gates.

The failure is useful evidence rather than a reason to tune several settings:
with byte tokenization, the 256-token context is exactly 256 UTF-8 bytes. The
published prompts consume much of that window before generation begins.

Run it only after `core/synthetic/tinystories-reference` has been ingested:

```console
cd ../fetchers
go run ./cmd/fetcher corpora/tinystories-reference.ini \
  /tmp/tinystories-reference
cd ../waldo
go run ./cmd/waldo/ index ingest /tmp/tinystories-reference \
  core/synthetic/tinystories-reference

go run ./cmd/waldo/ model forecast composes/0002-tinystories-byte.yaml
go run ./cmd/waldo/ model train tinystories-byte-01 \
  composes/0002-tinystories-byte.yaml
./composes/evaluate-tinystories.sh \
  tinystories-byte-01 /tmp/tinystories-byte-01-eval.jsonl
./composes/evaluate-tinystories.sh \
  tinystories-byte-01 /tmp/tinystories-byte-01-greedy.jsonl 0 42
```

## Rung 0003: TinyStories 512-byte context — passed

[`0003-tinystories-context-512.yaml`](0003-tinystories-context-512.yaml)
isolates the next hypothesis: rung 0002 failed because a 256-byte window cannot
hold the prompt and enough continuation to preserve its setup. It changes only
the architecture and training sequence lengths from 256 to 512 and reduces the
global batch from 64 to 32. Both rungs therefore retain 16,384 tokens per
optimizer update, 15,000 optimizer steps, and 245.76M total training tokens.

The comparison asks:

1. Does held-out loss remain stable or improve without changing the token
   budget or optimizer-step count?
2. Do at least six of eight temperature-0.8 samples retain the people,
   objects, and causal setup for 150 generated bytes?
3. Are immediate repetition and prompt retention materially better than rung
   0002?

Validated result on 2026-10-06:

- held-out loss improved from rung 0002's 0.5782 to 0.5375, or 7.0%, with the
  same 245.76M tokens and 15,000 optimizer steps;
- temperature-0.8 prompt retention improved, with four clear passes, two
  borderline continuations, and two failures; no sample immediately collapsed
  into a phrase loop;
- greedy decoding remained strongly repetitive, so context alone did not fix
  the remaining capability limitation;
- an audit of all 509,625 source stories measured mean 900 bytes, P50 789,
  P90 1,381, and P95 1,757; only 0.016% are 256 bytes or shorter; and
- after correcting the invalid 256-token EOS horizon, six of eight generations
  emitted EOS within 1,536 tokens. EOS training works; the two remaining
  max-token responses exposed the same long-generation degeneration.

Rung 0003 passes the context hypothesis. The next controlled question is
whether additional core model capacity reduces semantic drift and repetition.

Run and evaluate it with:

```console
go run ./cmd/waldo/ model forecast composes/0003-tinystories-context-512.yaml
go run ./cmd/waldo/ model train tinystories-context-512-01 \
  composes/0003-tinystories-context-512.yaml \
  --hostfile ~/hostfile
./composes/evaluate-tinystories.sh \
  tinystories-context-512-01 /tmp/tinystories-context-512-01-eval.jsonl
./composes/evaluate-tinystories.sh \
  tinystories-context-512-01 /tmp/tinystories-context-512-01-greedy.jsonl 0 42
./composes/evaluate-tinystories.sh \
  tinystories-context-512-01 /tmp/tinystories-context-512-01-eos.jsonl \
  0.8 42 1536
```

## Rung 0004: TinyStories capacity pilot — passed

[`0004-tinystories-capacity-pilot.yaml`](0004-tinystories-capacity-pilot.yaml)
keeps the byte tokenizer, 512-byte context, corpus, batch, optimizer-step count,
and 245.76M-token budget from rung 0003. It increases only core capacity to a
30.8M-parameter, 9-layer, width-512 decoder with full multi-head attention.
The peak learning rate follows square-root model-size scaling from 0.001 to
0.0006; all other training controls remain fixed.

This is intentionally a capacity pilot at about 8 tokens per parameter, not a
compute-optimal qualification run. Promote it only if held-out loss and the
fixed samples materially improve. If capacity helps, a later qualification
rung can extend the same architecture toward 20 tokens per parameter. If it
does not help, do not spend the larger token budget.

Validated result on 2026-10-06:

- held-out loss improved another 9.0%, from rung 0003's 0.5375 to 0.4892;
- loss continued improving through the terminal checkpoint, including from
  0.5004 at step 12,000 to 0.4892 at step 15,000;
- all eight temperature-0.8 samples emitted EOS within the calibrated
  1,536-token horizon, improving from six of eight;
- obvious greedy loop collapse fell from roughly seven of eight samples to
  three of eight; and
- entity and causal fidelity remained inconsistent, so the pilot establishes
  scaling direction rather than completing the capability gate.

Run it with:

```console
go run ./cmd/waldo/ model forecast composes/0004-tinystories-capacity-pilot.yaml
go run ./cmd/waldo/ model train tinystories-capacity-pilot-01 \
  composes/0004-tinystories-capacity-pilot.yaml \
  --hostfile ~/hostfile
./composes/evaluate-tinystories.sh \
  tinystories-capacity-pilot-01 /tmp/tinystories-capacity-pilot-01-eval.jsonl
./composes/evaluate-tinystories.sh \
  tinystories-capacity-pilot-01 /tmp/tinystories-capacity-pilot-01-greedy.jsonl \
  0 42
./composes/evaluate-tinystories.sh \
  tinystories-capacity-pilot-01 /tmp/tinystories-capacity-pilot-01-eos.jsonl \
  0.8 42 1536
```

## Rung 0005: TinyStories capacity qualification — behavioral near-miss

[`0005-tinystories-capacity-20tpp.yaml`](0005-tinystories-capacity-20tpp.yaml)
changes only the training horizon from rung 0004. Its requested 613,611,520
tokens resolve to 613,613,568 packed tokens: 37,452 optimizer steps and 20.00
tokens per core parameter. It trains from initialization so the cosine schedule
covers the complete qualification horizon.

Here "tokens" means built-in byte-tokenizer tokens. The ratio is exact for the
compose but is not interchangeable with 20 tokens per parameter measured by a
subword tokenizer. The 613.6M byte tokens represent about 1.34 passes over the
458.7M source bytes; at the earlier 4.26 bytes/subword measurement they are only
about 4.7 subword-equivalent tokens per core parameter.

Promotion requires all of the following:

1. Reloaded held-out loss is at least 3% below the pilot, or no more than
   0.4745, with the selected checkpoint near the end rather than an early
   overtraining reversal.
2. At least six of eight temperature-0.8 samples clearly retain the prompt's
   people, objects, and causal setup for the first 150 generated bytes.
3. No more than one temperature sample and no more than two greedy samples
   collapse into an immediate repeated sentence or phrase loop.
4. At least seven of eight temperature-0.8 samples emit EOS within the
   corpus-calibrated 1,536-token horizon.

Validated result on 2026-10-07:

- held-out loss improved 9.1%, from 0.4892 to 0.4447, passing the 0.4745
  numerical gate; the terminal step 37,452 was selected and reloaded;
- five of eight temperature samples clearly retained the prompt's entities,
  objects, and causal setup, short of the six-of-eight gate;
- no temperature sample immediately collapsed into an exact phrase loop, but
  about four of eight greedy samples did, missing the two-of-eight limit; and
- seven of eight long-horizon samples emitted EOS, passing the calibrated
  stopping gate.

Rung 0005 therefore confirms that additional capacity and exposure lower loss,
improve prose, and teach stopping, but it does not pass causal fidelity or
greedy-repetition promotion. Do not extend the same byte-token recipe merely
because its loss was still improving.

Run it with:

```console
go run ./cmd/waldo/ model forecast composes/0005-tinystories-capacity-20tpp.yaml
go run ./cmd/waldo/ model train tinystories-capacity-20tpp-01 \
  composes/0005-tinystories-capacity-20tpp.yaml \
  --hostfile ~/hostfile
./composes/evaluate-tinystories.sh \
  tinystories-capacity-20tpp-01 /tmp/tinystories-capacity-20tpp-01-eval.jsonl
./composes/evaluate-tinystories.sh \
  tinystories-capacity-20tpp-01 /tmp/tinystories-capacity-20tpp-01-greedy.jsonl \
  0 42
./composes/evaluate-tinystories.sh \
  tinystories-capacity-20tpp-01 /tmp/tinystories-capacity-20tpp-01-eos.jsonl \
  0.8 42 1536
```

## Later rungs

Rung 0006 must be a compose-native compact byte-BPE source-exposure control,
not another byte-token extension. The measurement compose
[`experiments/0003-tinystories-bpe-preflight.yaml`](experiments/0003-tinystories-bpe-preflight.yaml)
trains a 4K BPE on distributable PressBooks, then measures it on the pinned
TinyStories corpus. This keeps tokenizer training disjoint from the model
evaluation corpus and avoids weakening the distribution gate. It must be run
only with `model forecast --preflight`, never `model train`.

Use the measured TinyStories bytes/token, record lengths, and unique targets to
choose token context and a 1.34-source-pass budget matching rung 0005. Do not
create the training compose until those measurements are available; token
counts from different tokenizers are not equivalent. General-corpus mixtures
come only after these narrow reference
controls establish stable grammar, consistency, EOS, repetition, and held-out
behavior.

## Rung 0006: TinyStories BPE source-exposure control — behavioral near-miss

[`0006-tinystories-bpe-source-control.yaml`](0006-tinystories-bpe-source-control.yaml)
uses the measured PressBooks-trained 4K BPE while retaining rung 0005's
9-layer, width-512 core, 512-token context, TinyStories revision, optimizer,
learning rate, dropout, initialization, seed, and evaluation partition.

The preflight measured:

- 3.013 TinyStories bytes/token in training and 3.020 held out;
- a 1,543-byte effective 512-token context;
- token-length P50/P90/P95 of 258/475/604;
- 91.6% of records fitting one sequence, versus 3.8% with byte tokens;
- 152,610,804 unique packed BPE targets per source pass; and
- 1.708 expected document/EOS boundaries per sequence.

Rung 0005 consumed 613,613,568 packed byte targets over 458,762,511
unique packed targets, or 1.337541 effective passes. Applying that exact ratio
to the BPE source gives a requested budget of 204,123,174 tokens; WALDO rounds
this to 204,128,256 packed tokens and 1.337574 passes.

The global batch is 12 sequences with three accumulation steps. On four GPUs,
each micro-batch contains four sequences, one per rank. At measured fertility,
one optimizer update covers about 18.5K source bytes, close to rung 0005's
16.4K, and resolves to 33,224 optimizer steps instead of allowing token
compression to reduce the run to roughly one-third as many updates.

Promotion requires all of the following:

1. Training and artifact reload complete without non-finite loss.
2. Held-out bits per byte, computed as `loss / (3.020 * ln(2))`, is no worse
   than rung 0005's 0.6416 byte-token BPB. Raw token losses must not be compared
   across the two tokenizers.
3. At least six of eight temperature-0.8 samples clearly retain the prompt's
   people, objects, and causal setup for the first 150 generated bytes.
4. No more than one temperature sample and no more than two greedy samples
   collapse into an immediate repeated sentence or phrase loop.
5. At least seven of eight temperature-0.8 samples emit EOS within 1,536
   generated tokens. Record token lengths are now shorter, but the same horizon
   is retained for a conservative stopping comparison.

Run and evaluate it with:

```console
go run ./cmd/waldo/ model forecast \
  composes/0006-tinystories-bpe-source-control.yaml
go run ./cmd/waldo/ model train tinystories-bpe-source-control-01 \
  composes/0006-tinystories-bpe-source-control.yaml \
  --hostfile ~/hostfile
./composes/evaluate-tinystories.sh \
  tinystories-bpe-source-control-01 \
  /tmp/tinystories-bpe-source-control-01-eval.jsonl
./composes/evaluate-tinystories.sh \
  tinystories-bpe-source-control-01 \
  /tmp/tinystories-bpe-source-control-01-greedy.jsonl 0 42
./composes/evaluate-tinystories.sh \
  tinystories-bpe-source-control-01 \
  /tmp/tinystories-bpe-source-control-01-eos.jsonl 0.8 42 1536
```

Validated result on 2026-10-08:

- model `tinystories-bpe-source-control-01`, ID `e6ddce038dfd`, completed all
  33,224 steps and selected step 33,000;
- held-out loss 1.1500 normalizes to 0.54943 bits per byte, improving 14.4%
  over rung 0005's 0.6416 BPB;
- the final quarter still improved held-out loss by 3.5%, while late gradient
  norms remained stable and the final checkpoint showed no material reversal;
- all eight long-horizon samples emitted EOS;
- clear greedy collapse fell to two of eight, at the promotion limit; and
- only five of eight temperature samples clearly preserved prompt entities and
  causality, missing the six-of-eight promotion gate.

Rung 0006 proves that compact BPE fixes effective context, normalized language
loss, stopping, and much of the repetition failure. Semantic binding remains
the limiting capability. Because it used only 6.65 BPE tokens/core parameter,
ended with measurable learning headroom, and changed several behavioral
diagnostics in the right direction, one full 20-token/core qualification is
justified.

## Rung 0007: TinyStories BPE 20-TPP qualification — passed

[`0007-tinystories-bpe-20tpp.yaml`](0007-tinystories-bpe-20tpp.yaml) changes
only the training horizon from rung 0006. It trains from initialization so the
cosine schedule spans all 613,611,520 requested BPE tokens. WALDO resolves
99,872 optimizer steps, 613,613,568 packed targets, and 4.0208 effective source
passes.

Promotion requires:

1. Selected held-out BPB no greater than 0.5329, a 3% improvement from rung
   0006, with no material late overtraining reversal.
2. At least six of eight temperature-0.8 samples clearly retain prompt people,
   objects, and causal setup for the first 150 generated bytes.
3. No more than one temperature sample and no more than two greedy samples
   collapse into an immediate repeated sentence or phrase loop.
4. At least seven of eight temperature-0.8 samples emit EOS within 1,536
   generated tokens.

Run and evaluate it with:

```console
go run ./cmd/waldo/ model forecast composes/0007-tinystories-bpe-20tpp.yaml
go run ./cmd/waldo/ model train tinystories-bpe-20tpp-01 \
  composes/0007-tinystories-bpe-20tpp.yaml \
  --hostfile ~/hostfile
./composes/evaluate-tinystories.sh \
  tinystories-bpe-20tpp-01 /tmp/tinystories-bpe-20tpp-01-eval.jsonl
./composes/evaluate-tinystories.sh \
  tinystories-bpe-20tpp-01 /tmp/tinystories-bpe-20tpp-01-greedy.jsonl 0 42
./composes/evaluate-tinystories.sh \
  tinystories-bpe-20tpp-01 /tmp/tinystories-bpe-20tpp-01-eos.jsonl \
  0.8 42 1536
```

Every result must retain the compose, model summary, run ID, telemetry,
consumption report, temperature samples, greedy samples, and a written gate
decision.

Validated result on 2026-10-08:

- model `tinystories-bpe-20tpp-01`, ID `5d18938fa102`, run
  `3ee94d2f3e466ce2`, completed all 99,872 steps and selected the terminal
  checkpoint;
- exact held-out loss 1.042866 normalizes to 0.498234 bits per byte, improving
  9.3% over rung 0006 and passing the 0.5329 gate;
- held-out loss improved through the final quarter, with no late reversal and
  exact agreement between live, serialized, and reloaded FP32 evaluation;
- seven of eight temperature samples clearly retained the prompt's people,
  objects, and causal setup through the first 150 generated bytes;
- no temperature sample and one greedy sample collapsed into a clear repeated
  phrase or sentence loop; and
- all eight long-horizon samples emitted EOS.

Rung 0007 passes every promotion gate. It is the first qualified TinyStories
reference for this ladder: compact BPE, sufficient effective context, stable
multi-host optimization, prompt retention, bounded repetition, and learned
document stopping all work together.

## Rung 0008: PressBooks real-text control — ready

Do not jump directly from this narrow reference to the retired broad-mixture
recipe. First isolate distribution transfer while preserving the qualified
architecture, tokenizer, context, optimizer, and seed. The measurement compose
[`experiments/0004-pressbooks-bpe-preflight.yaml`](experiments/0004-pressbooks-bpe-preflight.yaml)
measures one filtered PressBooks pass with the exact rung-0007 tokenizer.
PressBooks is the tokenizer's training source and is structured, distributable
real educational prose, so this is the smallest defensible bridge away from
synthetic stories.

The completed preflight measured:

- 50.8K included training records and 503.9 MiB of UTF-8 text;
- 3.463 training bytes/token and 3.500 held-out bytes/token;
- 1,773 effective context bytes per 512-token sequence;
- token-length P50/P90/P95 of 1,466/6,907/10,117, with 20.3% of records
  fitting one sequence;
- 0.170 expected document/EOS boundaries per sequence; and
- 152.6M unique packed targets per source pass.

[`0008-pressbooks-bpe-20tpp.yaml`](0008-pressbooks-bpe-20tpp.yaml) therefore
trains for four source passes. Preflight resolves that to about 610.5M BPE targets,
essentially the same 20-token/core-parameter horizon and four-source-pass
exposure as rung 0007. It trains from initialization and changes only the
filtered model-training distribution; the architecture, tokenizer, context,
optimizer, learning rate, dropout, initialization, seed, and evaluation policy
remain fixed.

Promotion requires:

1. Training and artifact reload complete without non-finite loss, selected
   held-out loss improves at least 70% from initialization, and selection does
   not expose a material late overtraining reversal.
2. At least 12 of 15 deterministic general-continuation probes are grammatical
   and remain on topic through their first sentence.
3. No more than two deterministic probes and no more than two temperature-0.7
   probes collapse into an immediate repeated sentence or phrase loop.
4. At least eight of ten factual stems produce a relevant first sentence.
   Exact factual recall is scored and retained, but is not yet a hard gate for
   a 32.8M raw pretraining control.

Run and evaluate it with:

```console
go run ./cmd/waldo/ model forecast composes/0008-pressbooks-bpe-20tpp.yaml \
  --preflight
go run ./cmd/waldo/ model train pressbooks-bpe-20tpp-01 \
  composes/0008-pressbooks-bpe-20tpp.yaml --hostfile ~/hostfile
./composes/general-foundation/evaluate-general.sh \
  pressbooks-bpe-20tpp-01 /tmp/pressbooks-bpe-20tpp-01-greedy.jsonl 0 42
./composes/general-foundation/evaluate-general.sh \
  pressbooks-bpe-20tpp-01 /tmp/pressbooks-bpe-20tpp-01-temp07.jsonl 0.7 42
```

Do not add Wikimedia, Stack Exchange, or PLOS until this real-text control
passes. This keeps corpus mixture, source exposure, and architecture from
changing in one experiment.

## Forecast before training

`waldo model forecast <compose>` now reports the exact WALDO parameter
decomposition, core/token-I/O allocation, GQA projections, SwiGLU matrices,
context fitness, optimizer-step arithmetic, conventional and
architecture-aware compute, memory components, advisory warnings, and JSON
fields under `forecast.fitness`.

Add `--preflight` to materialize the compose corpus and make the missing
measurements without initializing model weights or running an optimizer:

```console
go run ./cmd/waldo/ model forecast \
  composes/0005-tinystories-capacity-20tpp.yaml \
  --preflight
```

The report includes training and model-held-out fertility, record-length
percentiles, one-sequence fit rate, packed document/EOS density, unique target
count, effective corpus passes, and per-corpus exposure. The held-out partition
is disjoint from model training, but is not claimed to be disjoint from a
compose-declared tokenizer-training sample.

For a byte tokenizer, token context is also exact byte context. The current
reference models therefore expose 256 tokens as exactly 256 UTF-8 bytes; after
150 generated bytes, no more than about 106 prompt bytes can remain visible.
For BPE and multi-corpus recipes, bytes/token, record percentiles, fit rate,
packed boundaries, EOS targets, and per-corpus fertility are marked as
requiring corpus preflight rather than estimated from unrelated manifest token
counts.

Warnings do not promote, reject, or launch a run. In particular,
tokens-per-parameter and the Chinchilla allocation are not capability gates.
Loss prediction is refused until WALDO has a sufficiently large cohort with
the same corpus/held-out revision, tokenizer, architecture family, context,
objective, and optimizer recipe. See
[`docs/MODEL-TRAINING-FITNESS.md`](../docs/MODEL-TRAINING-FITNESS.md).
