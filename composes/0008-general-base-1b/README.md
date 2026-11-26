# Rung 0008: 1B general-language base

This is the promoted 1B general-language base. It proves that WALDO can train
a billion-parameter decoder from initialization across the frozen 9:6:3:2
Wikimedia, child-filtered Cosmopedia, Stack Exchange, and TinyStories mixture.
It remains a raw causal base model, not an assistant or a reliable factual,
coding, or reasoning system.

## Promoted result

- Model: `general-base-1b-coherence-5tpp-01`
- Model ID: `112a481abdde5d9870d6cfb14d8855e10b98f45edff8d4770d0d2b3319036d33`
- Run ID: `e361dbbe010dcd43`
- Capacity: 1,034,512,384 total and 968,976,384 core parameters
- Tokenizer: frozen 32K byte BPE
- Context: 2,048 tokens
- Exposure: 4,844,896,256 packed targets, or 5.00 core TPP
- Selected checkpoint: final step 295,709
- Held-out loss: 10.7068 initially and 2.1941 selected
- Held-out BPB: 0.7989 selected
- Reload: selected artifact reproduced held-out loss 2.1941 exactly
- Generation: 20/20 sampled outputs contained two complete sentences; 18/20
  were conservatively judged grammatical and on topic through three sentences
- Repetition: 1/20 sampled and 1/20 greedy outputs looped
- EOS diagnostic: 17/20 sampled outputs emitted EOS within 512 tokens

## Promotion gates

- Training and evaluation metrics remain finite, and the selected artifact
  reloads exactly.
- The selected checkpoint lies in the final 20% of training.
- At least 18/20 sampled outputs contain two complete sentences.
- At least 18/20 sampled outputs remain grammatical and on topic through three
  sentences.
- No more than 1/20 sampled or 2/20 greedy outputs enter a phrase loop.

Every gate passed. EOS remains diagnostic. The two manual failures were
pronoun or identity-continuity errors, which bound the capability claim.

## Qualification history

The original matched-exposure capacity pilot was rejected because its BPB
regressed from the rung-0006 baseline. A matched learning-rate qualification
selected `4e-4`, but its longer continuations were not reliable at 1.62 core
TPP. A bounded source probe then demonstrated that the frozen 9:6:3:2 mixture
removed the severe repetition failure. The final fresh run changed only that
qualified source mixture and exposure horizon.

At the mandatory 2.5-core-TPP checkpoint, held-out loss had improved to 2.4225;
the frozen suite produced 20/20 two-sentence outputs, zero sampled loops, one
greedy loop, and 15/20 conservative three-sentence coherence passes. That
authorized completion to 5 core TPP without weakening the final gate.

## Run and evaluate

From the repository root:

```console
go run ./cmd/waldo/ model forecast \
  composes/0008-general-base-1b/compose.yaml
go run ./cmd/waldo/ model train general-base-1b-coherence-5tpp-01 \
  composes/0008-general-base-1b/compose.yaml \
  --hostfile ~/hostfile

./composes/0008-general-base-1b/evaluate.sh \
  general-base-1b-coherence-5tpp-01 /tmp/rung0008-sampled.jsonl 0.8 42 512
./composes/0008-general-base-1b/evaluate.sh \
  general-base-1b-coherence-5tpp-01 /tmp/rung0008-greedy.jsonl 0 42 512
python3 composes/0008-general-base-1b/score.py \
  /tmp/rung0008-sampled.jsonl /tmp/rung0008-greedy.jsonl
```

Omit `--hostfile` for single-host training.

## Directory contents

- `compose.yaml`: exact promoted training recipe
- `tokenizer.json`: immutable trained tokenizer used by the promoted model
- `compose_test.go`: architecture, tokenizer, data, and training guards
- `suite.json`: immutable native generation prompts
- `evaluate.sh`: native `waldo model chat` runner
- `score.py`: sentence-completion, EOS, and repetition diagnostics
- `evidence.json`: immutable promoted result
- `README.md`: purpose, evidence, gates, and commands
