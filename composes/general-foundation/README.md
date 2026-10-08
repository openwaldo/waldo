# General-foundation ladder (frozen)

**Status: failed at Gate 1 and frozen. Do not run Gate 2 or Gate 3.** The
32.3M/1B general-mixture run accurately consumed the requested mixture and
reached held-out loss 2.9481, but failed the behavioral gate with severe
repetition, factual errors, and source artifacts. The full audit concluded
that another compose cannot distinguish trainer correctness, recipe, data,
undertraining, and unrealistic capability expectations.

The next work is defined by the
[training validation and capability plan](../../docs/TRAINING-ROBUSTNESS-PLAN.md).
This file preserves the original hypothesis and procedure as historical
evidence.

This was the active foundation experiment. It starts with a controlled data
ablation at 32.3M parameters, then promotes one fixed broad mixture to a 125.6M
pilot and qualification run. Run one gate at a time and retain every summary,
consumption report, telemetry file, and evaluation output.

## Why this ladder exists

The prior experiments established four facts:

1. Training, multi-host data parallelism, checkpoint selection, publication,
   and reload are working.
2. Lower held-out loss is not sufficient evidence of useful generation.
3. An 8.6M model trained on 500M Cosmopedia tokens reached loss 2.6408 but
   retained prompts poorly, repeated, and emitted EOS only once in ten probes.
4. Scaling the same distribution to 32.3M parameters and 1B tokens reduced loss
   to 2.1102 and improved local grammar, but all ten greedy and all ten
   temperature-0.7 probes still hit the token limit. Sampling reduced exact
   loops without fixing semantic drift.

The next experiment therefore changes data before spending on a larger model.
Gate 1 keeps the exact 32.3M architecture, tokenizer recipe, 1B-token horizon,
optimizer, and seed from the Cosmopedia run. It changes only the pretraining
mixture. A failure stops the ladder before the 125.6M expense.

## Data recipe

The selected existing-index mixture is:

| Corpus | Weight | Target share | Purpose |
| --- | ---: | ---: | --- |
| `core/common-pile/wikimedia` | 11 | 55% | Broad factual and expository prose |
| `core/common-pile/stackexchange` | 5 | 25% | Questions, answers, technical explanations, and varied registers |
| `science/plos` | 3 | 15% | Long-form scientific reasoning and evidence-oriented prose |
| `core/common-pile/pressbooks` | 1 | 5% | Structured textbook and instructional prose |

Every stage selects main-content records and excludes records already assessed
as repetitive or boilerplate. Cosmopedia is deliberately absent: the controlled
proxy showed that its recurring child-explanation templates dominated a model
of this size. The stage has no `distribution_policy` because the current rights
audit does not approve publication of every selected canonical corpus artifact;
treat trained models as research artifacts pending rights review.

The 125.6M tokenizer reuses the proven 16K training recipe over Wikimedia,
PressBooks, and PLOS. Stack Exchange participates in model pretraining but is
not needed to change the tokenizer contract, avoiding an unnecessary second
variable and keeping tokenizer rights review on the previously exercised BOM.

At the 5B-token horizon, PressBooks receives about 250M target tokens, roughly
two passes over its indexed token count before record filtering. The larger
corpora remain well below one pass. This keeps repetition within the range that
data-constrained scaling work found useful while avoiding a tiny corpus
dominating the stream.

## Rungs

| Gate | Compose | Parameters | Tokens | Question |
| --- | --- | ---: | ---: | --- |
| 1 | `0001-general-mixture-32m.yaml` | 32.3M | 1B | Does a broad clean mixture beat Cosmopedia-only training at identical capacity and compute? |
| 2 | `0002-general-foundation-125m-pilot.yaml` | 125.6M | 2.5B | Does increased capacity acquire stable basic knowledge and non-repetitive prose? |
| 3 | `0003-general-foundation-125m.yaml` | 125.6M | 5B | Does extending the unchanged recipe improve quality rather than loss alone? |

Gate 2 is near 20 tokens per parameter and Gate 3 is near 40. The ratios are
measurement points, not automatic promotion criteria. Compute-optimal scaling
allocates a budget; it does not promise capability or convergence.

## Run procedure

Start only Gate 1:

```console
go run ./cmd/waldo/ model forecast \
  composes/general-foundation/0001-general-mixture-32m.yaml

go run ./cmd/waldo/ model train general-mixture-32m-01 \
  composes/general-foundation/0001-general-mixture-32m.yaml \
  --hostfile ~/hostfile

./composes/general-foundation/evaluate-general.sh \
  general-mixture-32m-01 /tmp/general-mixture-32m-01-eval.jsonl
```

Run a sampled diagnostic only after preserving the deterministic result:

```console
./composes/general-foundation/evaluate-general.sh \
  MODEL /tmp/MODEL-temp07-seed43.jsonl 0.7 43
```

## Scoring

Score every deterministic response from 0 to 2:

- **0:** incoherent, unrelated, a tautological loop, or materially wrong.
- **1:** relevant and understandable but incomplete, vague, or partly wrong.
- **2:** coherent, directly relevant, and materially correct.

Separately mark a repetition failure when a sentence or four-word phrase loops
more than twice or successive sentences make no semantic progress. EOS is
recorded diagnostically, but raw pretraining documents are often longer than
128 tokens, so EOS is not a foundation promotion threshold.

## Promotion gates

### Gate 1: controlled data ablation

- The fixed-prompt score is at least 12/30.
- At least 10/15 outputs avoid repetition failure.
- At least 8/10 factual prompts remain relevant through the first sentence.
- Both deterministic and sampled results materially improve over the 32.3M
  Cosmopedia model; lower held-out loss alone is not a pass.
- Observed exposure is within 0.5 percentage points of 55/25/15/5.

### Gate 2: 125M pilot

- The fixed-prompt score is at least 18/30.
- At least 12/15 outputs avoid repetition failure.
- At least 8/10 factual prompts score at least 1.
- At least 3/4 sampled probes are relevant and non-repetitive.
- Loss and prompt quality both improve over Gate 1.

### Gate 3: 125M qualification

- The fixed-prompt score is at least 22/30.
- At least 13/15 outputs avoid repetition failure.
- At least 8/10 factual prompts score at least 1 and at least 5 score 2.
- The deterministic and sampled suites improve over Gate 2 without a
  repetition regression.
- Repeat Gate 3 with seed 43 only after seed 42 passes.

## Questions recorded at every gate

1. Did held-out loss and fixed-prompt quality improve together?
2. Which failure class dominates: factual error, prompt drift, repetition,
   malformed language, or truncation?
3. Did observed corpus consumption match 55/25/15/5 after filtering?
4. Which domain has the highest held-out loss, and is aggregate loss hiding it?
5. Did temperature 0.7 reveal useful alternatives or merely different errors?
6. Did the selected checkpoint and reloaded FP32 artifact agree?
7. How many effective passes did each corpus contribute?
8. Does this evidence justify the next gate's cost?

## Corpus engineering plan

Yes, corpus packaging should evolve. It must produce auditable selection facts,
not silently rewrite upstream meaning.

### During acquisition and canonicalization

1. Preserve immutable raw inputs, source revision, license evidence, and hashes.
2. Normalize encoding and structural markup while preserving document and
   paragraph boundaries. Remove navigation, repeated headers, and extraction
   artifacts only with versioned detectors.
3. Attach record-level language, length, document-kind, educational-quality,
   repetition, boilerplate, synthetic/template, and safety assessments.
4. Perform exact and near deduplication across selected corpora, not merely
   within one shard. Keep the retained-record decision and duplicate cluster ID.
5. Detect overlap with every evaluation prompt and benchmark before training.
6. Publish immutable derived index views whose manifest pins the source BOM,
   detector versions, thresholds, and selection counts.

Do not paraphrase, summarize, fact-correct, or splice canonical source records
in place. Any model-generated textbook, exercise, answer, or rewrite is a new
synthetic corpus with generator identity, prompt recipe, filtering evidence,
and its own rights review.

### Data ablations before expensive runs

Use 32M proxy runs to change one selection variable at a time: near-deduplication,
quality threshold, domain weights, or synthetic contribution. Compare fixed
prompt quality and per-domain held-out loss, not only aggregate loss. A
proxy-selected mixture becomes a new pinned compose; it is never substituted
under an existing compose name.

## Post-training plan

Post-training begins only after Gate 3 passes raw causal-continuation gates.
Our assistant-EOS experiments proved that assistant-only supervision can teach
stopping, but contract-heavy SFT also produced canned deployment language and
did not repair foundation knowledge.

The first post-training corpus should therefore:

1. contain concise factual answers, explanations, transformations, and safe
   procedural responses rather than mostly API-contract examples;
2. preserve explicit user/assistant structure and supervise assistant tokens,
   including the final EOS;
3. balance response lengths so the model learns both concise stopping and
   longer explanations;
4. deduplicate prompts and answers against pretraining and evaluation sets;
5. retain source-level licenses and generator provenance;
6. reserve `interaction-contract-v1` for a small format-control share, not the
   dominant semantic corpus.

SFT may teach response shape, instruction following, and stopping. It must not
be used to hide factual collapse, repetition, or incoherence in the foundation.
The child compose must pin the qualified base model ID and run ID, so it will be
written only after Gate 3 succeeds.

## Research basis

- [Chinchilla](https://arxiv.org/abs/2203.15556): model size and token budget
  must be allocated together under fixed compute.
- [DataComp-LM](https://arxiv.org/abs/2406.11794): controlled dataset design and
  model-based filtering materially improve capability per unit of compute.
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499):
  near-deduplication reduces memorization and reaches equal or better accuracy
  in fewer steps.
- [DoReMi](https://arxiv.org/abs/2305.10429): proxy models can improve larger
  runs by selecting domain weights rather than treating mixtures as arbitrary.
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264):
  up to roughly four repeated epochs can retain value, after which returns decay.
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644): carefully
  selected and separately generated textbook-quality data can improve small
  model efficiency, but synthetic provenance and evaluation remain essential.
