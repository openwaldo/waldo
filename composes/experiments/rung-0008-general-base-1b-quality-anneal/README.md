# Rung 0008 1B quality-anneal experiment

This controlled experiment starts from the immutable selected artifact of the
completed 5-TPP near miss. It changes neither architecture nor tokenizer and
does not repeat foundation training.

The intervention adds 0.25 core tokens per parameter at one tenth of the
original peak learning rate. Its corpus is deliberately restricted to
Wikimedia and StackExchange at 3:1. TinyStories and Cosmopedia are excluded to
test the specific diagnosis from the frozen suite: excessive story style,
entity drift, and factual continuations that become synthetic textbook prose.

## Acceptance gates

The exact frozen rung-0006 sampled and greedy suites remain authoritative:

- finite metrics, exact artifact reload, and a selected checkpoint in the
  final 20%;
- held-out BPB no worse than the parent model's `0.798612` by more than `0.01`;
- at least 18/20 sampled outputs grammatical and on topic through three
  sentences;
- at least 18/20 sampled outputs with two complete sentences;
- no more than one sampled loop and two greedy loops.

EOS remains diagnostic. Failure rejects the anneal; it does not weaken the
5-TPP gate or authorize 10 TPP.

## Commands

```console
go run ./cmd/waldo/ model forecast \
  composes/experiments/rung-0008-general-base-1b-quality-anneal/compose.yaml

go run ./cmd/waldo/ model train general-base-1b-quality-anneal-01 \
  composes/experiments/rung-0008-general-base-1b-quality-anneal/compose.yaml \
  --hostfile ~/hostfile
```
