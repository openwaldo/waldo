# Training validation and capability plan

This is the authoritative plan for WALDO language-model training. It replaces
the experimental compose ladders as the source of project direction. Existing
composes remain reproducible evidence, but no larger foundation run is
authorized until the validation gates below pass.

The immediate decision is **stop scaling the current ladder**. The evidence
does not show that another corpus-weight change or a larger model will resolve
the observed failures. It shows that four different questions have been
conflated:

1. Is WALDO's training implementation numerically correct?
2. Does a particular corpus and recipe learn efficiently at small scale?
3. Can a model of a given size and token budget acquire broad capability?
4. Can a pretrained model be post-trained into a useful assistant?

Each question now has a separate control, metric, and exit gate.

## Executive findings

### What is established

WALDO has demonstrated valuable systems behavior:

- deterministic corpus resolution and exact weighted token accounting;
- two-host, four-GPU data parallel training;
- checkpointing and distributed resume machinery;
- FP32 master weights and persisted-artifact reload verification;
- one shared PyTorch model definition for training and inference;
- trained byte-level BPE tokenizers with immutable identities; and
- assistant-only supervision that can teach EOS behavior.

These are necessary properties. They do not establish language-model
correctness or capability.

### What is not established

WALDO does not yet have an independent known-answer test for model logits,
causal loss, gradients, an AdamW update, or a complete learning curve. Most
worker tests verify source invariants; normal Go end-to-end tests use a fake
worker. The real PyTorch worker has substantial integrity checking, but it is
checking itself. A shared implementation eliminates train/inference drift; it
does not prove that the shared math is correct.

The project also lacks a stable quantitative evaluation baseline. Aggregate
held-out loss, fifteen hand-scored continuations, one seed, and changing prompt
sets cannot distinguish implementation error, undertraining, architecture,
tokenizer, or data quality. Losses from different tokenizers and domain mixes
are not directly comparable.

### What the experiments actually show

| Experiment | Scale | Result | Defensible conclusion |
| --- | ---: | --- | --- |
| r50k tiny foundation | 16.0M, 50M and 160M tokens | severe repetition and factual failure | token I/O consumed too much tiny-model capacity; capability did not emerge |
| r50k small foundation | 76.4M, 760M and 1.5B | loss fell to 3.1515 and 3.0751; generations still repeated | lower aggregate loss was not a capability gate |
| byte-BPE prose foundation | 76.6M, 760M and 1.5B | loss fell to 2.9694 and 2.7984; behavior still failed | the tokenizer improved efficiency but did not isolate the remaining cause |
| assistant EOS SFT | 76.6M base plus 10M tokens | EOS rose to 7--8/10; answers became canned | stopping can be taught; SFT cannot repair a weak foundation |
| Cosmopedia proxy | 8.6M/10M, 8.6M/500M, 32.3M/1B | loss reached 2.1102; long generations still drifted and repeated | Cosmopedia is not TinyStories and the proxy did not validate the TinyStories result |
| general mixture | 32.3M, 1B | loss 2.9481; 2/15 EOS and 13/15 max-token terminations | the 55/25/15/5 mixture failed its behavioral gate and exposed source artifacts |

The general-mixture run consumed the requested shares accurately: 55.01%
Wikimedia, 25.00% Stack Exchange, 15.00% PLOS, and 4.99% PressBooks. The
sampler therefore did what the compose requested. Its outputs included a
Stack Exchange username, repeated PLOS boilerplate, a PLOS table DOI, and
recurring source-like templates. That is evidence against the selected data
view and gate, not evidence for another arbitrary mixture.

The loss curves fell smoothly and persisted correctly. This makes a grossly
broken run less likely, but it cannot rule out a subtle model, optimizer,
packing, or recipe problem.

## Why the old loop failed

### Twenty tokens per parameter was used for the wrong purpose

The Chinchilla result estimates how to allocate model size and data under a
fixed training-compute budget. It is not a minimum training requirement, a
convergence theorem, or a promise that a 32M or 125M model will answer factual
questions after 20 tokens per parameter.

Contemporary small models are commonly trained far beyond that frontier when
inference cost or fixed deployment size matters. Pythia trained every size,
including 31M, on about 300B tokens. SmolLM trained its 135M and 360M models on
600B tokens. WALDO's 32.3M/1B run saw about 31 tokens per parameter; Pythia-31M
saw roughly 9,700. These are different optimization objectives.

The hard-coded `tokens / 20` preset selector must therefore be treated as an
obsolete compute-allocation heuristic, not product policy. It should be
removed or renamed before it influences another model choice.

### The expected behavior did not match the experiment

TinyStories demonstrates that sub-10M models can produce coherent stories on
a deliberately constrained synthetic language distribution. It does not show
that the same model can learn broad factual and technical knowledge from a
heterogeneous web mixture. WALDO trained on Cosmopedia, not TinyStories, so the
experiment was a useful data point but not a recapitulation.

Similarly, raw greedy completion is not assistant evaluation. A base model may
continue documents rather than answer questions, and greedy decoding exposes
loops. It remains a useful regression diagnostic, but it cannot be the primary
promotion metric.

### Too many variables changed between runs

Across the ladders, architecture, vocabulary, tokenizer training corpus,
pretraining mix, token budget, prompt format, and evaluation prompts changed.
Some failed gates were followed by larger runs. The compose directory became
an experiment notebook, and a new numbered YAML often replaced a written
causal conclusion.

There are 103 commits touching `composes/` in the current history. That is a
process signal: immutable composes are useful run inputs, but they are not a
substitute for an experiment registry, comparable metrics, and enforced stop
conditions.

## Technical audit

### Corrected historical failures

The September artifact-boundary audit found real defects that are now useful
warnings. A compiled training graph reported losses that did not survive the
saved-model boundary: one pretraining checkpoint changed from 2.6724 live to
3.1388 after reload, and a later FP32 experiment still changed from 3.5102
compiled to 3.8389 after reload. The former inference worker also duplicated
the model and omitted QK normalization. Structured conversation stages were
using full-sequence rather than assistant-only loss.

The current worker addresses those defects with FP32 master weights, separate
compiled/eager/persisted evaluation, fail-closed artifact checks, best persisted
checkpoint selection, a shared model implementation, and assistant-only
supervision. Those repairs demonstrate why boundary tests matter. They also
reinforce the need for an external numerical oracle: internal agreement alone
would not have caught every historical defect.

### Model and optimizer

The PyTorch worker implements a decoder-only pre-norm transformer with RMSNorm,
RoPE, SwiGLU, grouped-query attention, optional QK normalization, tied
embeddings, BF16 compute, and FP32 master weights. The 32.3M model's 10K
vocabulary uses about 5.1M parameters, so token I/O is no longer the dominant
80% seen in the earlier r50k tiny model.

The current AdamW path applies weight decay to every parameter, including
normalization and embedding parameters. It records gradient norm but calls
`clip_grad_norm_` with infinity, so it does not clip. Neither choice is proven
wrong for this model, but both must be explicit sweep variables rather than
unexamined defaults.

The general run used 32,768 tokens per optimizer update: global batch 64 times
sequence length 512. Pythia used a roughly 2M-token batch. Learning rate,
schedule, and batch interact, so copying a plausible peak learning rate without
a batch/learning-rate sweep is not a validated recipe.

### Distribution and packing

The distributed loss path appears internally coherent: local unnormalized
losses are reduced, DDP averaging is compensated, and gradients are normalized
once by global valid-token count. Sequence ownership and observed mixture
accounting also look correct.

Continuous packing appends EOS at each record boundary and permits attention
across packed documents. This is a common design, but WALDO has not compared
it with a reference implementation or document-isolated attention. Held-out
evaluation resets at record boundaries, so train and evaluation packing are
not identical. This needs a controlled test, not a presumption of fault.

### Tokenizer

The byte-BPE implementation is deterministic and byte-complete. It reads up to
the declared sample-byte budget, reduces text to weighted lexical chunks, and
retains the 100,000 most frequent unique chunks for merge training. The current
compression check is evaluated on the same sample used to train the tokenizer.

The general model's 10K tokenizer was trained only on PressBooks even though
the model trained mostly on Wikimedia, Stack Exchange, and PLOS. Future gates
must report held-out fertility and bits per byte by domain. Tokenizer training
and tokenizer evaluation need disjoint, immutable samples.

### Data quality

The selected Common Pile inputs already include upstream filtering. WALDO adds
main-content, repetition, and boilerplate filters, but its content assessment
primarily detects repeated lines, paragraphs, and n-grams. It does not detect
source-specific table fragments, forum handles, citation templates, navigation,
or low-value scientific boilerplate. The failed general outputs demonstrate
that those artifacts remain learnable.

Exact deduplication and provenance are strong, but there is no complete
cross-corpus near-deduplication and template-cluster control for a composed
training view. Data selection must become an audited derived dataset, not a
weight-only compose decision.

### Evaluation

The current 512-record aggregate held-out set can select a checkpoint, but it
cannot explain domain behavior. It needs per-domain metrics and a stable
external evaluation set. Cross-tokenizer comparison must use bytes or characters
as the common denominator; raw token NLL and perplexity change when tokenization
changes.

Generation needs multiple declared decoding settings and automatic repetition,
prompt-retention, and EOS measures. Factual capability decisions need
low-variance log-likelihood or multiple-choice tasks with confidence intervals,
not only subjective continuation scoring.

## The new validation sequence

Every phase has a blocking exit gate. Failure creates an issue or code change,
not a larger compose.

### Phase 0: freeze and separate the goals

Before training, every proposed run must identify exactly one track:

- **trainer validation:** prove implementation correctness;
- **research:** learn which data, architecture, or recipe performs better;
- **from-scratch foundation:** produce a reusable base model; or
- **product:** produce a useful assistant or domain model.

The current foundation ladders are frozen. Their YAML files and results remain
evidence. No 125M promotion, post-training expansion, or new weight mixture is
authorized by the 32M general run.

Exit gate: the run proposal names one track, one hypothesis, one independent
baseline, one changed variable, a budget, and a stop condition.

### Phase 1: prove numerical conformance

Build tests around a tiny fixed architecture with QK normalization disabled so
an independent Hugging Face-compatible implementation can be used as oracle.
With fixed weights and token IDs, compare:

1. logits before loss;
2. masked causal cross-entropy;
3. every parameter gradient;
4. one AdamW update and optimizer state;
5. saved, reloaded, and inference-worker logits; and
6. uninterrupted versus checkpoint-resumed state.

Then hold the global batch and sample order constant and compare one, two, and
four GPU execution. Add a tiny-corpus memorization test that drives loss near
zero and reproduces the training continuations including EOS.

Exit gate: agreed FP32 and BF16 tolerances pass for every tensor; a resumed run
matches uninterrupted execution within tolerance; world-size changes preserve
the learning curve and sample accounting; the memorization control succeeds.

Implemented: `testing/training-conformance.sh` compares the production shared
PyTorch model with an independent functional oracle for logits, masked causal
loss, every gradient, one AdamW update and optimizer state, save/reload logits,
and next-token argmax in FP64 CPU and, when available, FP32 CUDA. It is a
required first step of `testing/training-acceptance.sh`.
`testing/e2e/model-pytorch-memorization.sh` adds the real-worker learning
control through an ordinary generated compose: it requires at least a 90%
held-out-loss reduction, a final best loss no greater than 0.25, exact learned
continuation, and EOS termination. BF16 and fixed-global-batch distributed
equivalence remain blocking work in this phase.

### Phase 2: reproduce known learning controls

Run two controls before general-corpus research:

1. **Tiny Shakespeare systems control.** Match a pinned nanoGPT-style character
   recipe closely enough to compare loss and samples. This cheaply exercises
   the complete real worker and optimizer.
2. **Actual TinyStories learning control.** Ingest the released TinyStories
   corpus with its exact provenance and use a pinned published architecture,
   tokenizer, split, and budget. Do not substitute Cosmopedia. If exact
   architectural reproduction is outside WALDO's current model contract, say
   so and first compare an independently implemented WALDO-shaped reference on
   the same data.

For broad-text calibration, evaluate public Pythia-31M and Pythia-70M
checkpoints at matched token counts and at completion using the same external
evaluation harness. Pythia checkpoints around 1B and 2B tokens provide the
critical answer: whether WALDO's raw behavior is abnormal for this size and
exposure, or whether the expectation was abnormal.

Exit gate: WALDO matches the independent reference's loss or bits-per-byte
curve within a predeclared band and produces the expected narrow-domain
behavior. If it does not, stop in trainer/recipe validation.

### Phase 3: establish quantitative evaluation

Add or adapt a token-level log-likelihood interface and create immutable
evaluation BOMs. The standard report must contain:

- aggregate and per-domain bits per byte;
- a stable low-variance cloze or multiple-choice suite appropriate to model
  scale, including a BabyLM-style low-resource suite where licensing permits;
- bootstrap confidence intervals or another declared uncertainty estimate;
- exact and fuzzy training/evaluation overlap reports;
- automatic repetition, prompt-retention, EOS, and length diagnostics;
- random/untrained, public-reference, and previous-WALDO baselines; and
- results for at least three fixed seeds before a research decision is
  promoted.

Raw continuation samples remain in the report, but they are diagnostic, not a
standalone capability gate.

Exit gate: one command produces a versioned report in which every score has a
dataset revision, split, metric definition, tokenizer treatment, and baseline.

### Phase 4: build a data observatory

Do not rewrite canonical source records in place. Produce immutable derived
views with complete source lineage. Before another mixture run, report for
each corpus:

- language, document length, tokenizer fertility, and bits-per-byte profiles;
- exact and near-duplicate clusters within and across corpora;
- template, citation, table, navigation, handle, markup, and dialogue rates;
- educational/factual quality labels from an audited classifier or sampled
  human rubric;
- effective epochs at every proposed token budget; and
- per-domain held-out loss from a shared reference model.

Deduplication must preserve the union of provenance and rights facts even when
one retained text represents multiple sources. A license difference should not
force duplicate text into the training stream.

Model-based quality filtering is appropriate only after its labels, thresholds,
and domain effects are measured. Synthetic rewrites are separate corpora with
generator, prompt, filtering, and rights provenance; they are not silent
canonicalization.

Exit gate: a reviewed data card and immutable derived-view BOM explain every
included source, filter, duplicate decision, measured artifact rate, and
evaluation overlap.

### Phase 5: calibrate the recipe at fixed data

Use a single frozen data view, tokenizer, architecture, and evaluation BOM.
Run short, budget-matched sweeps that change one variable at a time:

- global token batch, including materially larger batches than 32K;
- peak learning rate and schedule;
- gradient clipping and weight-decay exclusions;
- depth versus width and QK normalization;
- vocabulary size and held-out domain fertility; and
- packing with and without cross-document attention.

Use learning curves, bits per byte per FLOP, and task metrics. Use three seeds
for finalists. A 32M model may rank cheap choices after Phases 1--3, but proxy
ranking at this size is evidence with uncertainty, not a capability promise.
DoReMi used a 280M proxy for an 8B target; DataComp-LM's standardized data
experiments began at substantially larger scale. Transfer from 32M must be
demonstrated rather than assumed.

Exit gate: one recipe beats the reference at equal FLOPs with confidence, and
the ranking survives at the next model size.

### Phase 6: choose an honest delivery track

#### From-scratch research track

After all earlier gates, use a validated 125M--160M architecture and high-quality
data view. Treat 2.5B and 5B as early checkpoints, not final capability budgets.
A defensible first qualification horizon is approximately 10B--15B tokens
(about 80--120 tokens per parameter), with quantitative gates at 1B, 2.5B, 5B,
10B, and 15B. Stop when projected improvement cannot justify the next interval.

This is still much less exposure than Pythia or SmolLM and should be described
as an economical research baseline, not state of the art. Any claim of a
useful general model must come from the evaluation suite, not the ratio.

#### Product track

If the near-term goal is a useful assistant, import a well-characterized open
pretrained checkpoint in the desired deployment range and use WALDO for
auditable post-training, EOS, tool, safety, evaluation, and packaging stages.
This tests WALDO's differentiating lifecycle features without paying to
rediscover hundreds of billions of pretraining tokens.

#### Narrow-domain track

If the goal is a small model trained from scratch under a limited budget,
define a narrow language and task distribution, as TinyStories did, and build
an evaluation matched to it. Do not use broad factual prompts as the success
criterion for a constrained model.

## Experiment and compose governance

Composes are immutable executable inputs, not the project notebook. Introduce
one experiment registry whose entries contain:

- question and falsifiable hypothesis;
- parent/baseline run and exact changed variable;
- code revision, compose digest, corpus/tokenizer/evaluation BOMs, seed, and
  hardware;
- predicted result, budget, early-stop rule, and promotion threshold;
- summaries, telemetry, quantitative report, and sample artifacts; and
- final decision: pass, fail, inconclusive, or invalid.

Only promoted recipes belong in the top-level compose ladder. Exploratory
composes live under a dated experiment directory and are closed with a result.
No new compose is added merely because a run failed. A failed gate blocks its
dependent runs in documentation and automation.

The immediate compose cleanup is documentation-only: mark the existing
general-foundation, TinyStories-proxy, and assistant-EOS sequences complete or
failed and frozen. Move files only after the validation harness has a stable
home, so history is not rearranged while the plan is still being implemented.

## Ordered implementation backlog

1. Add the independent logits/loss/gradient/AdamW conformance test.
2. Add real-worker tiny memorization and save/reload/inference parity tests.
   Implemented by the required training-acceptance gates.
3. Add one/two/four-GPU fixed-global-batch equivalence and resume acceptance.
4. Add log-likelihood evaluation with per-domain bits per byte.
5. Add a public-checkpoint baseline runner for matched Pythia checkpoints.
6. Ingest actual TinyStories and pin a reproducible reference control.
7. Add the experiment registry and make promotion depend on its declared gate.
8. Build corpus artifact and near-duplicate reports, then freeze a derived view.
9. Run the fixed-data recipe sweep.
10. Select the research, product, or narrow-domain delivery track.

Items 1--7 precede any new general-foundation compose. Items 8--9 precede any
new from-scratch foundation training. Item 10 is a project decision, not an
automatic consequence of a loss value.

## Research calibration

- [Chinchilla](https://arxiv.org/abs/2203.15556) studies compute-optimal
  allocation; it does not define a capability threshold.
- [Pythia](https://github.com/EleutherAI/pythia) provides fixed-data-order
  models and intermediate checkpoints; every released size saw about 300B
  tokens and used a roughly 2M-token batch.
- [SmolLM](https://huggingface.co/blog/smollm) trained 135M and 360M models on
  600B curated tokens and reports continued gains beyond the Chinchilla point.
- [TinyStories](https://arxiv.org/abs/2305.07759) demonstrates tiny-model
  coherence on an intentionally constrained synthetic distribution, not broad
  web knowledge; its [released training and validation data](https://huggingface.co/datasets/roneneldan/TinyStories)
  make an actual control possible.
- [MobileLLM](https://arxiv.org/abs/2402.14905) shows that deep/thin design,
  embedding sharing, and grouped-query attention materially affect sub-billion
  models.
- [DataComp-LM](https://arxiv.org/abs/2406.11794) uses standardized training
  and 53 evaluations to compare data curation; model-based quality filtering
  was a central result.
- [DoReMi](https://arxiv.org/abs/2305.10429) learns mixture weights from
  per-domain excess loss with a 280M proxy rather than selecting weights from
  a handful of generations.
- [Paloma](https://arxiv.org/abs/2312.10523) motivates stable per-domain
  language-model evaluation rather than one aggregate held-out number.
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
  supports near-deduplication as both a quality and memorization control.
- [BabyLM](https://babylm.github.io/) provides controlled 10M- and 100M-word
  tracks and a quantitative low-resource evaluation framework.

## Definition of done

WALDO training is trustworthy when:

- numerical conformance and real-worker learning controls pass;
- distributed and resumed execution match at fixed global batch;
- data, tokenizer, training, and evaluation inputs are immutable and audited;
- evaluation is quantitative, per-domain, uncertainty-aware, and comparable to
  public checkpoints;
- each run tests one hypothesis and failed gates actually stop dependent work;
- capability claims are tied to measured tasks and a declared use case; and
- efficiency reports include tokens, FLOPs, throughput, data wait, memory,
  optimizer steps, and effective corpus epochs.

Until then, a completed run with falling held-out loss proves only that the
current system optimized its current objective.
