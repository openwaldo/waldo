# Validated capability ladder

This directory is a catalog of reproducible models that passed every declared
gate. It is not a roadmap and it is not an experiment log. At present the
validated ladder ends at rung 0007.

Only a passed recipe may have a numbered rung directory. Candidate composes
belong on an experiment branch or in a temporary workspace. A failed candidate
is removed; its compose, telemetry, evaluation output, and decision remain in
Git and the persisted model artifact.

## Current ladder

| Rung | Proven capability | Model | Evidence | Status |
| --- | --- | --- | --- | --- |
| [0001](0001-tiny-shakespeare/) | End-to-end learning and checkpoint-selection control on one continuous literary distribution | 10.7M parameters, byte tokenizer | Plausible play continuation; best pre-overtraining checkpoint selected and reloaded exactly | Passed |
| [0002](0002-tinystories-bpe/) | Narrow simple-story language model | 30.7M core parameters, 4K byte BPE | Entity/causal retention, bounded repetition, and 8/8 long-horizon EOS | Passed |
| [0003](0003-narrow-grounded-responder/) | Narrow grounded responder inside the TinyStories language domain | Same 30.7M core model plus bounded assistant-response training | 20/20 bounded EOS, 14/15 exact grounded answers, zero interaction loops, and retained story behavior | Passed |
| [0004](0004-small-simple-english/) | Small base model spanning simple narrative, description, dialogue, and elementary explanation | 59.5M core parameters, 8K byte BPE | Significant broad-form BPB gain over rung 0002, strong prompt/continuation discrimination, complete-sentence generation, and zero loops | Passed |
| [0005](0005-general-base-125m/) | 125M general-language base model | 106.2M core parameters, 32K byte BPE | 16/20 locally relevant two-sentence continuations, bounded repetition, and exact reload | Passed |
| [0006](0006-general-base-350m/) | 350M general-language base model with longer coherent continuation | 314.6M core parameters, 32K byte BPE | BPB 0.8009, 18/20 locally relevant three-sentence continuations, bounded repetition, and exact reload | Passed |
| [0007](0007-interaction-assistant-350m/) | Bounded interaction-contract assistant derived from rung 0006 | Same 350M model plus 75-step contract SFT and 5-step formatting correction | 42/42 EOS, 37/42 exact, valid JSON/YAML, zero interaction loops/leaks, and retained raw generation | Passed |

A cell-biology candidate previously occupying rung 0003 failed every
behavioral promotion gate and was removed. The promoted rung instead adds a
narrow grounded interaction contract to the passed TinyStories foundation.
Later definitions do not authorize training or create compose directories.

## Defined ladder

The ladder separates four different abilities:

1. learning a text distribution;
2. producing coherent language inside a constrained distribution;
3. following a narrow interaction contract; and
4. acquiring increasingly broad language and knowledge.

A rung never claims the abilities of a later rung. In particular, a raw base
model is not graded as an assistant, and a model below 500M parameters is not
expected to be a reliable open-domain factual system.

| Rung | Scale | Architecture | Honest target | Status |
| --- | ---: | --- | --- | --- |
| 0001 | 10.7M | 6 layers, width 384, byte tokenizer, 256 context | Learn one literary stream and prove the training system | Passed |
| 0002 | 30.7M core | 9 layers, width 512, 8 attention heads, 4K tied byte-BPE, 512 context | Generate coherent simple stories with boundaries | Passed |
| 0003 | same 30.7M core | Inherit rung 0002 exactly; add only a supervised interaction stage | Respond to simple, grounded prompts inside the TinyStories language domain | Passed |
| 0004 | 64.7M total / 59.5M core | 12 layers, width 640, SwiGLU 1,728, 10/10 heads, 8K tied vocabulary, 1,024 context | Short coherent simple-English prose across several everyday forms | Passed |
| 0005 | 124.6M total / about 106.2M core | 30 layers, width 576, SwiGLU 1,536, 9/3 grouped-query heads, 32K tied vocabulary, 2,048 context | Grammatical, locally relevant general-language continuation without assistant claims | Passed |
| 0006 | 345.4M total / 314.6M core | 32 layers, width 960, SwiGLU 2,560, 15/5 grouped-query heads, 32K tied vocabulary, 2,048 context | Improve distribution fit and sustain longer coherent general-language continuation | Passed at 20 core TPP |
| 0007 | inherit passed 0006 | No architecture change; 75-step supervised contract stage plus 5-step formatting-only correction | Bounded assistant behavior across the frozen interaction contract | Passed |

The 125M and 350M designs follow the deep-and-thin, tied-embedding, grouped-
query direction validated by MobileLLM at those scales. Rung 0004 is an
empirical bridge: its conventional 12-layer shape beat a matched-compute
24-layer pilot, so the deeper candidate was rejected rather than promoted on
architectural fashion.

### Rung 0001: systems learning control

Purpose: prove ingestion, byte tokenization, causal training, checkpoint
selection, serialization, reload, and recognizable local generation.

Promotion criteria:

- finite loss and exact selected-artifact reload;
- at least 50% held-out-loss improvement;
- selected checkpoint may precede the final step when later training degrades;
- 6/8 sampled continuations preserve play dialogue and character formatting;
- no requirement for EOS, factual answers, instruction following, or general
  English.

This rung is passed. Its criteria intentionally describe a systems control,
not a useful language model.

### Rung 0002: narrow simple-story base

Purpose: prove that a roughly 30M-core model can learn coherent English inside
one intentionally simple distribution.

Promotion criteria:

- finite metrics, exact reload, and held-out BPB no worse than 0.52;
- 7/8 sampled prompts retain the named people or objects and the prompt's main
  causal direction through the first 150 generated tokens;
- no sampled continuation and no more than one greedy continuation enters an
  obvious immediate repetition loop; and
- 8/8 sampled long-horizon generations emit EOS.

This rung is passed. It proves simple-story generation only. It does not prove
question answering, factual recall, broad knowledge, or assistant behavior.

### Rung 0003: narrow grounded responder

Purpose: test the user's reasonable expectation for a 30M model: it can barely
understand a simple request, produce a relevant response, and stop. This is
post-training of rung 0002, not a new foundation model.

The prompt must contain every fact needed to answer. Allowed tasks are simple
story questions, extraction, yes/no classification, and one-step rewriting in
the TinyStories vocabulary. Open-domain facts, arithmetic reasoning, coding,
and multi-step instructions are explicitly out of scope.

Promotion criteria over immutable held-out prompts:

- 18/20 responses obey the user/assistant boundary and emit EOS within 128
  generated tokens;
- 14/20 begin with a relevant answer rather than continuing or restating the
  prompt;
- 12/15 grounded questions return the answer explicitly present in the supplied
  text;
- no more than 1/20 responses enters a phrase or sentence loop; and
- the original rung-0002 suite retains at least 6/8 entity/causal passes and
  7/8 long-horizon EOS.

These are interaction and reading gates, not knowledge gates. The promoted
checkpoint is now the frozen 30M-scale reference for this suite. Any future
data, optimizer, or architecture change must compare against it rather than
inventing a new threshold after training.

### Rung 0004: small simple-English base

Purpose: expand from one story form to several constrained forms: short
narratives, descriptions, simple dialogue, and elementary explanatory prose.
It remains a base model and is not expected to answer questions on command.

Architecture: 64.7M total and 59.5M core parameters, 12 decoder layers, width
640, SwiGLU intermediate size 1,728, ten full attention heads, tied 8K
embeddings, RoPE, RMSNorm, and 1,024-token context. A matched-compute pilot
rejected the proposed 24-layer, width-448 shape: at roughly 315M/346M training
tokens its held-out loss was 2.9966 versus 2.9279 for the conventional control.
Full attention is retained at this bridge scale so grouped-query attention is
not introduced in the same experiment as capacity growth.

Promotion criteria:

- beat the preserved rung-0002 base on the frozen 20-item broad-form
  conditional-BPB suite, with a paired bootstrap 95% interval below zero;
- assign the correct prompt/continuation pairing at least 0.25 BPB less than a
  within-form rotated pairing, while the deterministic untrained control's gap
  remains within 0.05 BPB of zero;
- improve explanation BPB by at least 25%, with no individual form regressing
  more than 0.15 BPB from rung 0002;
- at least 18/20 sampled continuations produce two complete sentences;
- no more than 2/20 sampled or 3/20 greedy outputs enter repetition loops; and
- no factual-accuracy, question-answering, instruction-following, or arbitrary
  fixed-horizon EOS requirement.

The last exclusion is deliberate. This is a base model trained on documents,
not an assistant trained to return bounded responses. EOS is still reported as
a diagnostic, but a long elementary explanation reaching a generation limit
does not invalidate conditional language modeling. Training may expose at most
four effective passes over any source view.

This rung passed at 20 core TPP. Its paired BPB delta versus rung 0002 was
-0.1775 with a 95% interval of [-0.4021, -0.0017]. The correct-context
advantage was 0.4701 BPB versus 0.0034 for the untrained control; explanation
BPB improved from 1.8259 to 1.0909. All 20 sampled outputs produced two
complete sentences, and neither the sampled nor greedy suite looped.

### Rung 0005: 125M general-language base

Purpose: establish an honest small general base model—not a capable assistant.
The promoted architecture has 30 layers, width 576, SwiGLU intermediate size
1,536, nine query heads, three key/value heads, tied 32K embeddings, RoPE,
RMSNorm, and 2,048 context. It contains 124.6M total and approximately 106.2M
non-embedding parameters.

Promotion criteria:

- preserve finite training and evaluation metrics, select and exactly reload a
  verified checkpoint, and select that checkpoint from the final 20% of the
  current exposure phase;
- keep every source view at or below four effective passes;
- 16/20 sampled prose continuations are grammatical and remain on topic for
  two sentences;
- at least 18/20 sampled outputs contain two complete sentences;
- no more than 2/20 sampled or 3/20 greedy outputs enter an obvious loop; and
- no open-domain short-answer, chat, coding, or multi-step reasoning gate.

EOS is diagnostic for this base model, not a gate. The corpus is fixed at 45%
Wikimedia, 30% filtered young-child
Cosmopedia textbook narrative, 15% StackExchange, and 10% GPT-4 TinyStories.
This is broad enough to test generalization but still restricted to explanatory,
encyclopedic, question/answer, and simple narrative English.

This rung passed at 20 core TPP. Held-out loss improved from 10.4318 to 2.4652,
and the selected step-114,000 artifact reloaded exactly. The largest source
exposure was 3.40 effective passes. Native WALDO evaluation produced two
complete sentences for 20/20 sampled prompts; 16/20 remained grammatical and
on topic for both sentences, with zero sampled loops and two greedy loops.
Supplementary public-task results are retained in the rung evidence packet,
but they do not replace the ladder's native `model chat` evaluation.

### Rung 0006: 350M general-language base

Purpose: test whether increased capacity improves general-language distribution
fit and supports longer coherent continuation. It remains a raw base model.
Reliable factual recall, assistant behavior, coding, and reasoning are not
promotion claims.

Architecture target: the MobileLLM-350M scale—32 layers, width 960, SwiGLU
intermediate size 2,560, 15 query heads, five key/value heads, tied 32K
embeddings, RoPE, RMSNorm, and 2,048 context. WALDO's parameter formula
estimates 345.4M total and 314.6M non-embedding parameters.

The capacity pilot changes only the architecture. It retains rung 0005's
tokenizer, corpus views and weights, 2,048-token context, optimizer, schedule,
seed, held-out selection, and native generation suite. At 1.573B tokens, the
passed rung-0005 architecture recorded held-out BPB 0.9243; that is the matched
token-exposure baseline.

The 5-core-TPP pilot may advance only if:

- training and evaluation remain finite, the artifact reloads exactly, and the
  selected checkpoint lies in the final 20%;
- effective passes are recorded for every source view so repeated exposure is
  explicit, but pass count alone is not a failure;
- selected held-out BPB is no worse than 0.9243;
- at least 16/20 sampled outputs remain grammatical and on topic for two
  sentences, and at least 18/20 contain two complete sentences; and
- no more than 2/20 sampled or 3/20 greedy outputs loop.

The 5-core-TPP pilot passed: selected held-out BPB was 0.8442, the step-96,000
artifact reloaded exactly, 20/20 sampled outputs contained two complete
sentences, 17/20 remained grammatical and on topic for both sentences, and the
sampled/greedy suites contained one/two loops respectively.

The controlled 10-core-TPP phase also passed its advancement checks, but it is
not the promoted rung. It consumed 3.146B tokens over six packing passes and
selected step 190,000 of 192,017. Selected held-out BPB improved from 0.8442
to 0.8178, the artifact reloaded exactly, all 20 sampled outputs contained two
complete sentences, and neither the sampled nor greedy suite looped. Manual
review found 16/20 grammatical and on topic for two sentences, a one-item
change from the 5-TPP phase rather than a material regression. Only 15/20
remained grammatical and on topic for three sentences, so the 10-TPP artifact
does not meet the final-promotion gate. Sampled EOS was 17/20 and remains a
diagnostic rather than a gate.

Ten and twenty core TPP are controlled follow-on exposure phases, not automatic
full runs. They retain the exact tokenizer, source views, mixture weights,
architecture, optimizer, schedule, seed, and evaluation suite; only token
exposure changes. Effective source passes are evidence, not a hard ceiling.
Advance only while held-out BPB does not regress from the preceding phase, the
selected checkpoint remains in the final 20%, and native generation does not
materially regress. An earlier selected checkpoint or worsening held-out BPB is
evidence to stop for overtraining rather than change the corpus mid-comparison.

Final promotion requires held-out BPB no worse than 0.85, at least 18/20
sampled outputs grammatical and on topic for three sentences, no more than one
sampled or two greedy loops, and exact reload. EOS and public benchmark results
remain diagnostics. Beyond 20 TPP requires a new evidence review, not an
automatic extension.

The controlled 20-core-TPP phase passed and is the promoted rung. It consumed
6.292B tokens and selected step 382,000 of 384,034. Selected held-out BPB
improved from 0.8178 at 10 TPP to 0.8009, and the artifact reloaded exactly.
All 20 sampled outputs contained two complete sentences; manual review found
18/20 grammatical and on topic through three sentences. The sampled/greedy
suites contained zero/one loops, and sampled EOS was 17/20. Only exposure
changed across the 5-, 10-, and 20-TPP phases.

### Rung 0007: interaction-contract assistant

Purpose: post-train the passed rung 0006 so it can recognize a user request,
respond directly, respect elementary operating boundaries, and stop. This is
not a new foundation run: the frozen rung-0006 architecture, tokenizer, and
weights are the exact parent, followed by one supervised epoch.

The training contract contains 45,000 examples across 21 bounded behaviors:
directness, constraints, clarification, assumptions, corrections, context,
uncertainty, evidence, tool boundaries, tool success and failure,
authorization, memory, safety, narrow refusal, planning, status communication,
error recovery, user control, instruction priority, and formatting. It does
not grant tools or external authority. Open-domain expertise, dependable
coding, and multi-step mathematical reasoning remain out of scope.

Promotion is decided against a frozen 42-item suite containing two held-out,
deterministically selected examples from each behavior, plus the unchanged
rung-0006 native generation suite. The criteria are:

- finite metrics, exact reload, and a selected checkpoint in the final 20% of
  the supervised run;
- at least 38/42 responses emit EOS without a role-boundary leak;
- at least 34/42 satisfy the requested interaction contract on manual review;
- both formatting cases produce valid output in the requested format;
- no more than one interaction response contains a repetition loop or
  role-boundary leak;
- at least 16/20 sampled raw continuations remain grammatical and on topic for
  two sentences, and at least 18/20 contain two complete sentences; and
- no more than two sampled or three greedy raw continuations enter an obvious
  loop.

Exact match and token F1 against the reference response are reported as
diagnostics, not universal gates: many contract cases allow multiple correct
wordings. The frozen base suite is run in raw mode because it measures
foundation preservation, while the interaction suite uses the inherited
`user-assistant-v1` interface.

No assistant rung can be proposed until its base rung passes. Post-training is
not permitted to conceal a failed foundation model.

The first candidate, `interaction-assistant-350m-01`, was rejected. Its single
supervised epoch was numerically healthy and the frozen interaction suite
scored 42/42 exact responses, 42/42 EOS, zero role leaks, and zero loops. That
result did not survive the foundation-retention gate: only 15/20 sampled raw
continuations contained two complete sentences versus the required 18/20, and
manual review found interaction-contract language intruding into unrelated
stories and elementary explanations. The behavior was therefore narrow
template acquisition with material catastrophic forgetting, not a promotable
assistant. Rung 0006 remains the frozen parent; any next candidate must add an
explicit retention strategy and must use the same two frozen suites.

The promoted recipe found that retention boundary without changing the parent,
corpus scope, or gates. A 75-step contract stage preserved rung-0006 behavior
and passed every interaction check except YAML. A five-step, `5e-6`
formatting-only correction then passed the complete frozen suite: 42/42 EOS,
37/42 exact reference responses, at least 37/42 conservative manual contract
passes, valid JSON and YAML, zero interaction loops or role leaks, 18/20 raw
outputs with two complete sentences, 16/20 conservative grammatical/on-topic
passes, one sampled loop, and zero greedy loops. Both selected artifacts
reloaded exactly. Ten formatting steps were rejected because retention fell to
17/20 two-sentence outputs; the promoted five-step stage is therefore the
smallest tested correction that passed both objectives.

## What the failed expansion proved

The failed `cell-biology-30m-80tpp-01` run was numerically healthy: it consumed
2.454B tokens, improved held-out loss from 8.4060 to 2.1062, selected and
exactly reloaded its best checkpoint, and finished four-GPU training normally.
It nevertheless produced approximately 1/10 correct factual continuations,
failed the explanation and retained-story gates, entered repetition loops on
nearly every factual probe, and emitted EOS on 0/10 long factual generations.

That result does **not** establish that WALDO's transformer implementation is
broken. The same architecture family learns TinyStories, and independent
numerical and memorization controls pass. It establishes that the proposed
capability experiment was unsupported:

- a TinyStories-proven 30.7M-core architecture was assumed to support factual
  domain acquisition without a matched public or internal baseline;
- elementary textbook facts were evaluated after training mostly on research
  articles, which optimize a different continuation distribution;
- only 269.5M unique targets were replayed to fill 2.454B training tokens;
- PLOS Genetics was repeated about 15.6 times and the small Wikimedia view
  about 35.2 times; and
- raw loss on the training distribution was treated as evidence for semantic
  capability even though no such relationship had been calibrated.

The failure therefore rejects that experiment. It does not authorize another
mixture, a larger token budget, or a larger model.

## Why TinyStories succeeded and factual rungs did not

TinyStories intentionally constrains vocabulary, syntax, subject matter, and
document form. Its evaluation asks the model to continue the same distribution
it learned. The TinyStories paper specifically demonstrates that very small
models can become coherent under this simplified distribution; it does not
claim that the same capacity can absorb broad or research-grade factual text.

Chinchilla's often-quoted token/parameter relationship is a compute-allocation
result, not a capability threshold. Conversely, very high token/parameter
ratios do not create capability when a small unique corpus is replayed many
times. Work on data-constrained scaling finds that repeated data remains useful
for a limited number of epochs, after which the value of more compute decays.
The failed cell-biology run exceeded that evidence for its smallest views.

References:

- [TinyStories](https://arxiv.org/abs/2305.07759)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
- [BabyLM](https://arxiv.org/abs/2301.11796)

## Hard stop before any future rung

No new candidate compose—foundation, domain, or post-training—may be added or
trained until all of the following exist:

1. **A declared track.** The proposal is explicitly trainer validation,
   research, a from-scratch foundation model, or product post-training.
2. **A quantitative evaluator.** The report uses immutable evaluation data,
   low-variance log-likelihood or multiple-choice metrics, automatic EOS and
   repetition diagnostics, and confidence intervals where applicable. Raw
   generations are supporting diagnostics, not the primary score.
3. **Independent baselines.** The same evaluator is run on an untrained model
   and the previous WALDO rung. General-model rungs 0005 and later also require
   a public model at comparable size and exposure. A proposed threshold must be
   attainable by an identified baseline.
4. **A data-support report.** Before training, the exact derived view reports
   unique tokens, effective source passes, duplicates, artifacts, tokenizer
   fertility, evaluation overlap, and direct support for every demanded
   capability.
5. **A scaling pilot.** Existing checkpoints or a deliberately tiny run show
   that the chosen metric improves with exposure or capacity. The pilot changes
   one causal variable and has a written early-stop rule.
6. **A bounded qualification run.** Architecture, tokenizer, data, optimizer,
   evaluator, seed, checkpoints, maximum budget, and promotion gate are frozen.
   Failure ends the candidate; it does not trigger an improvised extension.

Until these prerequisites pass, the correct next task is measurement and
baseline implementation—not training.

## Required evidence packet

Every proposed rung must have a reviewable packet before its compose is
eligible to run:

```text
track and intended use
one falsifiable hypothesis
parent or public baseline
one changed variable
immutable data and tokenizer identities
data-support and contamination report
evaluation BOM and scoring implementation
expected metric range with supporting evidence
pilot budget and early-stop rule
maximum qualification budget
promotion and rejection thresholds
```

Claims such as "30M should know elementary biology" or "80 tokens per
parameter should be enough" are not evidence and cannot populate this packet.

## Parameter and loss accounting

WALDO reports core parameters separately from token input/output weights.
Training exposure, unique source exposure, and effective source passes are
separate quantities. A tokens-per-parameter ratio is descriptive; it is never
a promotion criterion by itself.

Raw token loss is comparable only with the same tokenizer and held-out bytes.
When tokenizers differ, use exact bits per UTF-8 byte:

```text
BPB = held_out_nll_sum / (held_out_utf8_bytes * ln(2))
```

BPB measures distribution fit, not factual correctness, reasoning, stopping,
or usefulness. Every capability claim requires an external task metric.

## Architecture policy

The 30.7M-core architecture is validated for TinyStories and its narrow
grounded-response extension. The 59.5M-core architecture is validated for the
rung-0004 simple-English distribution. Neither result proves open-domain
factual ability. Architecture changes must be comparative experiments at fixed
data, tokenizer, FLOPs, optimizer, and evaluation—not guesses based on
parameter count alone.

Depth, width, attention layout, vocabulary allocation, context, batch size,
learning rate, weight decay, clipping, and packing are all controlled variables.
At least two matched configurations are required before an architecture shape
is selected. The selected qualification run uses one frozen seed; a later
replication seed strengthens evidence but is not silently substituted for a
failed qualification run.

## Next authorized work

Rungs 0001 through 0007 are passed and frozen. The first rung-0008 capacity
pilot increased the frozen general-base architecture from 314.6M to 969.0M
core parameters while retaining the same 1.573B-token exposure and all other
training controls. It was rejected: selected BPB was 0.8487 versus the
predeclared rung-0006 baseline of 0.8442. The frozen behavioral suite passed,
but behavioral non-regression cannot replace the failed distribution-fit gate.
The complete decision is recorded in
[`docs/RUNG-0008-CAPACITY-PILOT.md`](../docs/RUNG-0008-CAPACITY-PILOT.md).

The matched learning-rate calibration selected `4e-4`, and its full
1.573B-token qualification selected BPB `0.8125`, reloaded exactly, completed
20/20 sampled responses, and stayed within the frozen repetition limits. Its
longer educational continuations were not yet reliable enough for final rung
promotion at only 1.62 core TPP.

The authorized next experiment is therefore the unnumbered
[`5-TPP qualification`](experiments/rung-0008-general-base-1b-5tpp/). It keeps
the qualified architecture, `4e-4` learning rate, data, tokenizer, seed,
batch, and evaluator fixed while increasing only the cosine exposure horizon.
It has a stop/go behavioral gate at 2.5 core TPP and the final rung gate at 5
core TPP. Passing 5 TPP authorizes 10 TPP; each later exposure phase must pass
before the next one is run. No numbered rung-0008 directory is created until
a candidate passes every promotion gate.

## Rung directory contract

Each promoted rung contains:

```text
README.md       purpose, immutable evidence, gates, and commands
compose.yaml    exact promoted training recipe
compose_test.go guards for architecture, data, and training contract
evaluate.sh     fixed behavioral diagnostics
suite.json      immutable conditional-likelihood prompts and continuations
likelihood.py   exact BPB evaluator where the rung requires cross-tokenizer comparison
```

The authoritative project-wide validation sequence is documented in
[`docs/TRAINING-ROBUSTNESS-PLAN.md`](../docs/TRAINING-ROBUSTNESS-PLAN.md).
