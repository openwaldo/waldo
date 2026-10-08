# Model architecture and training-fitness analysis

Status: implemented structural forecast and exact intrinsic-loss telemetry;
corpus profiling and empirical curve fitting remain gated work.

WALDO exposes this analysis through `waldo model forecast <compose>` and its
JSON `forecast.fitness` object. It does not create another training command or
ladder. Forecast warnings are review evidence, not claims that a formula can
predict semantic capability.

## Research basis and limits

The design follows primary sources:

- [Kaplan et al., *Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361)
  fit power laws using **non-embedding** parameters and show diminishing
  returns when model or data scale is held fixed.
- [Hoffmann et al., *Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556)
  fit compute-optimal model/data allocations. The often-repeated roughly 20:1
  token/parameter point is a compute-allocation result for that experiment,
  not a minimum, convergence theorem, or capability guarantee.
- [DeepSeek LLM](https://arxiv.org/abs/2401.02954) replaces parameter count
  with non-embedding FLOPs/token for scaling work because `6ND` omits
  attention/context cost while total `N` includes vocabulary work that does
  not represent transformer capacity. Its validation metric is bits per byte.
- [PALOMA](https://arxiv.org/abs/2312.10523) evaluates fit across many domains,
  controls the held-out distribution, and recommends bits per byte when
  tokenizers differ. BPB improves intrinsic comparability; it does not make
  different corpora or validation constructions equivalent.
- [OLMo](https://arxiv.org/abs/2402.00838) combines in-loop evaluation with
  fixed offline intrinsic and downstream evaluation and explicitly
  decontaminates its PALOMA evaluation data.
- [TinyStories](https://arxiv.org/abs/2305.07759) demonstrates small models on
  a deliberately narrow synthetic distribution and evaluates grammar,
  consistency, creativity, and reasoning—not loss alone. WALDO's Cosmopedia
  proxies are not reproductions of that experiment.
- [Tensor Programs V / μTransfer](https://arxiv.org/abs/2203.03466) supports
  learning-rate transfer only when the model is intentionally parameterized
  for μP. WALDO's current architecture is not μP, so forecast does not claim
  that a learning rate transfers across width, depth, GQA, or vocabulary.

## Exact WALDO parameter decomposition

For vocabulary `V`, hidden width `H`, intermediate width `I`, layers `L`, query
heads `A`, and key/value heads `K`, WALDO uses head width `H/A` and GQA key/value
width `G=(H/A)K`.

The implementation and forecast both use:

- token input: `VH`;
- token output: zero when tied, otherwise another `VH`;
- learned position embedding: zero, because WALDO uses parameter-free RoPE;
- per layer: `Q=H²`, `K=HG`, `V=HG`, `O=H²`;
- per layer SwiGLU: gate `HI`, up `HI`, down `IH`, for `3HI` total;
- RMSNorm: `(2L+1)H` total; and
- bias: zero. QK normalization is also parameter-free.

Thus total parameters are

```text
token I/O + L(2H² + 2HG + 3HI) + (2L+1)H
```

`forecast.fitness.architecture` reports every term and refuses an internal
decomposition that does not exactly equal WALDO's architecture forecast.
"Core" or "non-embedding" means total parameters minus both input and untied
output token matrices.

## Historical architecture audit

All token matrices below are tied; "other" is RMSNorm because these models
have no biases or learned positions.

| Architecture used by runs | Total | Token I/O | Core | Attention | MLP | Other | Allocation (token / attention / MLP / other) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r50k tiny: `foundation-canary-02`, `foundation-tiny-pilot-01/02` | 16,014,336 | 12,866,304 | 3,148,032 | 786,432 | 2,359,296 | 2,304 | 80.34% / 4.91% / 14.73% / 0.01% |
| r50k small: `foundation-small-pilot-01`, `foundation-small-01` | 76,416,000 | 32,165,760 | 44,250,240 | 9,830,400 | 34,406,400 | 13,440 | 42.09% / 12.86% / 45.03% / 0.02% |
| 16K BPE tiny: `foundation-pipeline-canary-bpe-01`, `foundation-mixture-canary-bpe-01` | 7,244,032 | 4,096,000 | 3,148,032 | 786,432 | 2,359,296 | 2,304 | 56.54% / 10.86% / 32.57% / 0.03% |
| 16K BPE small: `foundation-small-language-bpe-01/full-bpe-01` | 76,615,040 | 10,240,000 | 66,375,040 | 14,745,600 | 51,609,600 | 19,840 | 13.37% / 19.25% / 67.36% / 0.03% |
| 10K BPE 8.6M: `tinystories-canary-01`, `tinystories-8m-01` | 8,593,664 | 2,560,000 | 6,033,664 | 1,310,720 | 4,718,592 | 4,352 | 29.79% / 15.25% / 54.91% / 0.05% |
| 10K BPE 32.3M: `cosmopedia-32m-01`, `general-mixture-32m-01` | 32,261,632 | 5,120,000 | 27,141,632 | 5,898,240 | 21,233,664 | 9,728 | 15.87% / 18.28% / 65.82% / 0.03% |
| byte 10.7M: both validated byte references | 10,721,280 | 99,456 | 10,621,824 | 3,538,944 | 7,077,888 | 4,992 | 0.93% / 33.01% / 66.02% / 0.05% |

This answers how much was actual transformer capacity. The 16M r50k model had
only 3.15M core parameters. The 76.6M/16K model had 66.38M. The current byte
model has 10.62M. That is strong evidence of embedding starvation in the old
tiny design, but it is not evidence that architecture alone caused the later
TinyStories result.

## Tokenizer, context, and corpus fitness

The portable compose proves vocabulary and token context. Corpus-dependent
facts require the selected immutable corpus revision and tokenizer, so compose
forecast marks them `requires-corpus-preflight` rather than inventing values.
The preflight extension must persist, overall and per corpus:

- exact UTF-8 bytes divided by emitted tokenizer tokens;
- effective context bytes and an observed character distribution;
- p50, p90, and p95 record lengths in tokens and bytes;
- percentage of records fitting in one context;
- expected packed record boundaries and supervised EOS targets; and
- fertility (tokens/byte and tokens/character) for every selected corpus.

For the built-in byte tokenizer, one token is exactly one UTF-8 byte. Therefore
the current 256-token context is exactly 256 bytes of input capacity and at
most 256 characters. With 150 generated bytes, no more than approximately 106
prompt bytes remain visible at the end. A long TinyStories setup therefore
cannot remain fully in context, which explains part—but not necessarily all—of
the observed entity and plot drift.

Average bytes/token can estimate effective context for a BPE tokenizer, but it
cannot produce exact BPB or a record fit distribution. Averages hide tails,
per-domain fertility, special tokens, and record boundaries.

## Data and optimization report

For every fixed-budget stage forecast reports requested and rounded planned
tokens, `batch_size * sequence_length` global target capacity per optimizer
step, optimizer steps, tokens per total and core parameter, and warmup/warmdown
percentages. Valid 1/2/4/8-rank proposals report global batch, global physical
micro-batch, and rank-local micro-batch.

WALDO's global `batch_size` is invariant across world sizes. With a fixed seed,
compose, pinned corpus preflight, and global batch, its prepared sequence
ordinal and sample order are invariant; only rank ownership changes. A world
size is omitted if it does not divide the global physical micro-batch. Current
checkpoint state still pins topology, so changing world size while resuming a
checkpoint is not promised.

Unique available training tokens, effective corpus passes, records, EOS
boundaries, and per-corpus exposure require preflight's tokenizer-specific
stream scan. Manifest token totals are not silently substituted because they
may have been counted with another tokenizer.

## Compute and memory approximations

Forecast retains conventional `6 * total parameters * planned tokens`. It also
reports an architecture-aware estimate. Per token, WALDO counts forward
matrix work as:

```text
L(4H² + 4HG + 6HI + 4HS) + 2HV
```

where `S` is each stage's declared sequence length (not merely the architecture
maximum) and multiply-add is two FLOPs. Training is approximated as three times
forward work. This includes GQA projection savings,
SwiGLU's three matrices, causal attention score/value work, and the vocabulary
projection. Tying eliminates a second parameter matrix but does not eliminate
the vocabulary projection computation. Embedding lookup is excluded.

This is still an estimate: it omits elementwise operations, RMSNorm, RoPE,
optimizer updates, padding, recomputation, kernel behavior, and communication.
Forecast labels both estimates accordingly.

Memory JSON separates parameter artifact bytes, gradients, FP32 Adam moments,
FP32 master weights when training lower-precision parameters, approximate
checkpoint state, and topology-specific activation/logit workspace. The
hardware `required_per_gpu_bytes` additionally includes a conservative 4 GiB
runtime allowance. It is a capacity forecast, not an allocator guarantee.

## Comparable intrinsic loss

Cross-entropy is measured per tokenizer token. For nearly uniform initial
logits, expected token loss is `ln(V)`:

| Tokenizer | `ln(V)` | Observed initial loss |
| --- | ---: | ---: |
| r50k, 50,259 | 10.825 | 10.8531 (`foundation-canary-02`) |
| 16K BPE | 9.680 | not retained in the supplied case-study evidence |
| byte, 259 | 5.557 | 5.3814 Shakespeare; 5.3819 TinyStories |

Vocabulary size therefore explains most of the roughly 5.47-nat difference
between the r50k canary and byte reference initial losses. It does not explain
the entire difference: initialization, corpus, context positions, and the
held-out construction also contribute.

Raw token loss and perplexity cannot be compared across tokenizers because the
unit/event space changes. The workers now persist:

- `heldout_nll_sum`;
- `heldout_target_tokens`;
- `heldout_utf8_bytes`;
- `heldout_loss` and `heldout_perplexity`; and
- `heldout_bits_per_byte = NLL / (ln(2) * UTF-8 bytes)`.

These fields are in each evaluation and telemetry schema 2. WALDO continues to
append schema-1 telemetry files using their original header. Exact BPB uses the
summed NLL and exact selected text bytes; multiplying average token loss by an
average bytes/token value is only an approximation.

Even exact BPB is intrinsic distribution fit, not semantic capability. Corpus
entropy, duplication, templates, domain breadth, held-out selection, and
contamination can all change it. A model can obtain low local loss by learning
short-range and repeated patterns while still looping, losing prompts beyond
its context, failing EOS, or emitting false facts.

## Warnings and gates

Forecast warnings never block training. Current structural defaults are:

- above 20% token I/O: advisory design review;
- above 50% token I/O: warning because a mathematical majority of parameters
  is outside the transformer blocks;
- warmup above 20%: warning; and
- fewer than ten global optimizer batches: warning.

The 20% value is an explicit project heuristic, not a research law. The 50%
value is a majority boundary, not a quality threshold. A future compose-level
policy may override advisory values without changing model identity.

Corpus-aware warnings remain preflight work: record/context fit, excessive
effective passes, held-out size and contamination, EOS density, and corpus
fertility. A PALOMA-scale 100K-token/domain or 1M-token/source sample is useful
evidence for broad-domain stability, but WALDO must not turn it into an
unexplained hard failure for narrow controls. Cross-tokenizer reports must
reject raw-loss comparisons and require exact BPB plus matching evaluation
distribution.

## Empirical loss prediction gate

Forecast currently returns `refused-insufficient-evidence`. A prediction is
permitted only after a run registry can form a cohort with the same corpus BOM
and held-out SHA-256, tokenizer revision/artifact, architecture family,
context, objective, optimizer, schedule, batch semantics, and other declared
recipe fields.

The first supported fit should use BPB and a form such as

```text
L(N,D) = E + A/N^alpha + B/D^beta
```

where `N` is core parameters or, preferably, architecture-aware non-embedding
FLOPs/token. Require at least 12 completed real runs, at least three distinct
model scales and three distinct data scales, repeated points/seeds for residual
variance, finite exact BPB, and no contaminated held-out set. Fit with
positivity constraints, compare against simpler nested baselines, and report
bootstrap confidence intervals, held-out fit error, residual diagnostics,
sample count, parameter/data range, and log-space extrapolation distance.
Refuse if identifiability, fit quality, or extrapolation checks fail. Never
translate predicted intrinsic loss, tokens/parameter, or Chinchilla allocation
into a semantic capability promise.

## WALDO tiny-model case study

| Family and run | Tokens | Best token loss | Interpretation |
| --- | ---: | ---: | --- |
| r50k tiny `foundation-canary-02` | 5M | 6.2503 | 80.3% token I/O; repetitive/incoherent |
| r50k tiny `foundation-tiny-pilot-01` | 50M | 4.9693 | lower same-tokenizer loss; factual loops remained |
| r50k tiny `foundation-tiny-pilot-02` | 160M | 4.1755 | further loss gain; still incoherent |
| r50k small `foundation-small-pilot-01` | 760M | 3.1515 | more structure, still repetitive/inaccurate |
| r50k small `foundation-small-01` | 1.5B | 3.0751 | diminishing loss gain and poor generation |
| 16K BPE tiny pipeline / mixture canaries | 10M / 50M | 5.6343 / 4.8192 | the mixture and evaluation size also changed, so not a budget-only ablation |
| 16K BPE small / full | 760M / 1.5B | 2.9694 / 2.7984 | same architecture/tokenizer/data recipe; budget and schedule horizon changed |
| Cosmopedia proxy 8.6M canary / full | 10M / 500M | 4.9915 / 2.6408 | same architecture/tokenizer/corpus; budget and schedule horizon changed |
| `cosmopedia-32m-01` | 1B | 2.1102 | learned educational style but failed coherence/EOS/repetition |
| `general-mixture-32m-01` | 1B | 2.9481 | correct 55/25/15/5 accounting; breadth overwhelmed the model; 2/15 EOS |
| byte Shakespeare | 81.92M | 1.5117 | passed narrow style gate; terminal 2.0940 and greedy repeat attractor show overtraining/degeneration |
| byte TinyStories | 245.76M | 0.5782 | grammatical narrow English; context drift, 1/8 EOS, and greedy repetition remain |
| byte memorization control | small exact distribution | about 0.0002 | exact continuation and EOS prove the objective/optimizer/save-reload path can memorize the control |

The raw final-loss comparison that remains invalid is all comparison across
tokenizers and all comparison across different evaluation distributions. Even
BPB would not make Shakespeare, TinyStories, Cosmopedia, and a broad mixture
the same task.

The cleanest historical budget comparisons are the paired tiny r50k, small
BPE, and 8.6M proxy runs, subject to their deliberately longer schedule
horizons and checkpoint cadence. The r50k small pair also changed effective
mixture exposure when PressBooks exhausted, so it is not a clean token-only
ablation. The canary-to-mixture BPE pair changes corpus composition. Every
cross-family comparison confounds some combination of architecture,
tokenizer, corpus, context, validation view, or execution history.

Required causal ablations are same corpus BOM and held-out records, same
tokenizer, same context/objective, same optimizer and global sample order,
multiple seeds, and one architecture change at a time: vocabulary/allocation,
width, depth, GQA, or context. A context ablation must keep evaluated byte
content and reporting comparable. Only those controls can estimate an
architecture contribution.

The supported conclusion is limited: the original tiny models were severely
embedding-starved; the byte model assigns much more capacity to transformer
layers; TinyStories is narrower and more predictable than the general corpora;
its raw token loss is partly lower because its vocabulary is smaller; and its
256-byte context explains part of its drift. Architecture alone has not been
shown to cause the successful TinyStories behavior.

## Information still required

Genuine loss prediction needs tokenizer-specific corpus profiles, exact BPB
from immutable held-out sets, contamination evidence, and a deliberately
designed matrix of at least 12 comparable multi-seed runs spanning model and
data scale. WALDO's current history is valuable diagnostic evidence, but its
confounds make it insufficient for a defensible fitted loss curve.
