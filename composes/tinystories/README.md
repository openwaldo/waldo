# TinyStories-style controlled proxy

This ladder asks whether WALDO can reproduce the *behavioral lesson* of
TinyStories—small models learning coherent short-form text from synthetic,
relatively simple prose—using only corpora already present in the OpenWALDO
index.

It is not a TinyStories dataset reproduction. The OpenWALDO index does not
currently contain `roneneldan/TinyStories`, so the ladder uses the already
ingested `core/synthetic/cosmopedia-v2` corpus as the nearest synthetic prose
proxy. No download or index ingestion step is required.

## Controlled recipe

- Pretraining corpus: `core/synthetic/cosmopedia-v2` only.
- Tokenizer training corpus: `core/common-pile/pressbooks`, which is already
  indexed and passes WALDO's distributable-tokenizer policy.
- Tokenizer: deterministic 10,000-entry byte BPE.
- Context: 512 tokens.
- Quality filter: main-content records, excluding records assessed as
  repetitive or boilerplate.
- Optimizer: AdamW, 5e-4 peak learning rate, proportional
  warmup-stable-warmdown schedule.
- Evaluation: the same ten story beginnings at every gate.

Cosmopedia v2 mixes educational articles, synthetic textbooks, and stories;
WALDO's current record filter cannot isolate its story subset. That makes this
a broader and harder proxy than TinyStories. A failure does not falsify the
TinyStories result, but a pass demonstrates that WALDO's small-model pipeline
can learn coherent generation from an existing controlled synthetic corpus.

The Cosmopedia rights audit remains unresolved for distributed model weights,
so the pretraining stage intentionally does not declare
`distribution_policy: distributable`. Treat resulting models as research
artifacts. The separately trained PressBooks tokenizer does carry its normal
distributable review.

## Ladder

| Gate | Compose | Parameters | Tokens | Purpose |
| --- | --- | ---: | ---: | --- |
| 1 | `0001-tinystories-canary.yaml` | 8.6M | 10M | Existing-index, tokenizer, training, publication, and inference smoke test |
| 2 | `0002-tinystories-8m.yaml` | 8.6M | 500M | Small-model coherent-generation qualification |
| 3 | `0003-tinystories-32m.yaml` | 32.3M | 1B | Capacity scaling on the identical corpus and controls |

The horizons are controlled budgets, not a claim that 20 tokens per parameter
is optimal. Record WALDO's observed record count, corpus passes, and consumed
tokens for every run.

## Run one gate at a time

Start with Gate 1:

```console
go run ./cmd/waldo/ model forecast \
  composes/tinystories/0001-tinystories-canary.yaml

go run ./cmd/waldo/ model train tinystories-canary-01 \
  composes/tinystories/0001-tinystories-canary.yaml \
  --hostfile ~/hostfile

./composes/tinystories/evaluate-tinystories.sh \
  tinystories-canary-01 /tmp/tinystories-canary-01-eval.jsonl
```

After Gate 1 passes, repeat with `0002-tinystories-8m.yaml` and a fresh model
name. Do not train Gate 3 until Gate 2 passes.

## Scoring

For each story, score these three dimensions from 0 to 2:

1. **Grammar:** 0 unreadable; 1 understandable with substantial errors; 2
   consistently grammatical simple English.
2. **Prompt consistency:** 0 unrelated or contradictory; 1 partly follows the
   setup; 2 preserves the named characters, objects, and situation.
3. **Progression:** 0 loops or makes no progress; 1 adds relevant events; 2
   develops and resolves a simple passage or story.

Separately mark repetition failure if a sentence or four-word phrase repeats
more than twice, and record whether generation ended with EOS before 256 tokens.
Do not truncate repeated text before scoring; repetition is part of the result.

## Promotion gates

### Gate 1: pipeline canary

- Tokenizer, training, checkpointing, evaluation, publication, reload, and
  inference pass using only existing index paths.
- Held-out loss is finite and below initialization.
- All ten prompts produce non-empty output.
- No generation-quality or EOS threshold applies at 10M tokens.

### Gate 2: 8M qualification

- At least 8/10 generations score 2 for grammar.
- At least 7/10 score 2 for prompt consistency.
- At least 6/10 score at least 1 for progression.
- At least 8/10 avoid repetition failure.
- At least 6/10 emit EOS before 256 tokens.
- Held-out loss is finite and improves through the selected checkpoint.

### Gate 3: 32M scaling

- At least 9/10 generations score 2 for grammar.
- At least 8/10 score 2 for prompt consistency.
- At least 8/10 score at least 1 for progression.
- At least 9/10 avoid repetition failure.
- At least 8/10 emit EOS before 256 tokens.
- It improves on Gate 2's mean score without worse repetition or EOS behavior.

Once a deterministic gate passes, run a secondary diversity check at
temperature 1 with several seeds. Those samples diagnose breadth; they do not
replace the deterministic promotion result:

```console
./composes/tinystories/evaluate-tinystories.sh \
  MODEL /tmp/MODEL-temp1-seed43.jsonl 1 43
```

## Questions recorded at every gate

1. Did held-out loss and generation quality improve together?
2. Did the selected checkpoint and reloaded artifact agree?
3. How much of Cosmopedia did WALDO consume for the requested budget?
4. Did the model learn coherent progression rather than only local grammar?
5. Did it learn EOS without post-training?
6. Were failures caused by looping, contradiction, grammar, or truncation?
7. Does the evidence justify the next gate's cost?

Reference: [TinyStories paper](https://arxiv.org/abs/2305.07759). The paper is
the experimental inspiration; Cosmopedia v2 is the actual training corpus.

## Recorded results

This proxy is complete and did not meet its coherence hypothesis:

| Model | Parameters | Tokens | Held-out loss | Greedy EOS | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `tinystories-canary-01` | 8.6M | 10M | 4.9915 | 0/10 | Pipeline passed; generation collapsed into markup and repeated templates |
| `tinystories-8m-01` | 8.6M | 500M | 2.6408 | 1/10 | Better local English; poor prompt retention and repetition in all probes |
| `cosmopedia-32m-01` | 32.3M | 1B | 2.1102 | 0/10 | Better grammar; semantic drift and repetition remained |

The 32.3M model also produced 0/10 EOS at temperature 0.7. Sampling reduced
exact phrase loops but did not restore the ball, rabbit, dog, boat, or bird
prompt subjects. WALDO's continuous-EOS packing appends and supervises EOS at
every document boundary, so the stopping result is not missing EOS training;
Cosmopedia documents are long and the model learned their recurring explanatory
templates.

Do not scale this recipe further. The active experiment moves to a broad data
ablation in [`../general-foundation`](../general-foundation/README.md).
