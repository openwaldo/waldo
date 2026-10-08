# Controlled experiments

## TinyStories BPE source measurement

[`0003-tinystories-bpe-preflight.yaml`](0003-tinystories-bpe-preflight.yaml)
trains a compact 4K byte-level BPE on distributable PressBooks and measures it
against the pinned TinyStories reference corpus. It is not a training rung and
must not be passed to `model train`.

Run only:

```console
go run ./cmd/waldo/ model forecast \
  composes/experiments/0003-tinystories-bpe-preflight.yaml \
  --preflight |
  tee /tmp/tinystories-bpe-preflight.txt
```

The promoted rung will retain the 9-layer, width-512 core and use the measured
unique BPE targets to match rung 0005's 1.34 effective source passes. It is not
valid to copy the 613.6M byte-token budget into the BPE compose.

## Assistant EOS canaries

These experiments test whether assistant-response post-training teaches the
76.6M Gate 3B checkpoint to terminate answers and whether a broad conversation
mixture improves answer quality. They do not reopen or promote the foundation
ladder.

## What is controlled

- The base model, model ID, and run ID are pinned.
- Baseline and child use the same textual prompt: `User: ...\n\nAssistant:`.
- Decoding is greedy with a 128-token ceiling.
- Each child receives one 10M-token assistant-response stage.
- WALDO masks user text, supervises assistant text, and supervises the packed
  EOS target when the final message is an assistant message.

This is a practical intervention test, not an EOS-only ablation: each child
also learns answer content and conversational formatting from its selected
data.

The first child weights the narrow interaction-contract corpus heavily. It
improved EOS completion from 0/10 to 7/10 but retained severe repetition and
inserted unrelated operational language. The second child changes only the
corpus mixture to 50% OASST2, 25% HelpSteer2, and 25% Dolly. The base, token
budget, optimizer, learning rate, schedule, seed, and evaluation remain fixed.

## Train

```console
go run ./cmd/waldo/ model forecast \
  composes/experiments/0001-assistant-eos-canary.yaml

go run ./cmd/waldo/ model train foundation-small-assistant-eos-01 \
  composes/experiments/0001-assistant-eos-canary.yaml \
  --hostfile ~/hostfile
```

Broad-mixture follow-up:

```console
go run ./cmd/waldo/ model forecast \
  composes/experiments/0002-assistant-eos-broad-canary.yaml

go run ./cmd/waldo/ model train foundation-small-assistant-broad-01 \
  composes/experiments/0002-assistant-eos-broad-canary.yaml \
  --hostfile ~/hostfile
```

## Fixed questions

1. `What is Linux?`
2. `Who created the Linux kernel?`
3. `What is the capital of France?`
4. `At what temperature does water freeze at sea level?`
5. `What does a CPU do?`
6. `Who created the Python programming language?`
7. `What is two plus two?`
8. `Why are backups useful?`
9. `How do plants use sunlight?`
10. `Give one safe way to install software on Linux.`

## Evidence and decision rule

Run the script in this directory for the base, contract-heavy child, or broad
child. It writes JSONL containing the generated token count and `finish_reason`
for every question.

```console
./composes/experiments/evaluate-assistant-eos.sh baseline /tmp/assistant-eos-baseline.jsonl
./composes/experiments/evaluate-assistant-eos.sh contract /tmp/assistant-eos-contract.jsonl
./composes/experiments/evaluate-assistant-eos.sh broad /tmp/assistant-eos-broad.jsonl
```

The theory is supported enough to continue the conversation-tuning ladder when:

- at least 8 of 10 child responses finish with `eos` before 128 tokens;
- at least 8 of 10 avoid a phrase or sentence looping more than twice;
- at least 6 of 10 begin with a relevant, materially correct answer; and
- the child materially improves EOS completion over the pinned base baseline.

Failure does not justify more foundation pre-training. Inspect the SFT records,
EOS target accounting, learning rate, and checkpoint trajectory before spending
more compute.
