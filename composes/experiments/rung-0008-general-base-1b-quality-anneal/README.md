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

## Completed result

The experiment is rejected. Model `general-base-1b-quality-anneal-01` selected
step 14,784 of 14,786 with exact artifact reload, but behavioral quality
regressed materially from the 5-TPP parent:

- 20/20 sampled outputs contain two complete sentences;
- sampled loops are 1/20;
- greedy loops are 3/20, failing the frozen maximum of two;
- sampled EOS fell from 18/20 to 5/20; and
- conservative manual review found only 8/20 outputs grammatical and on topic
  through three sentences, versus 13/20 for the parent and 18/20 required.

The restricted held-out loss is not directly comparable to the parent's
mixed-corpus held-out loss. Its internal trend improved from `2.5695` to
`2.4640`, but that did not translate into better frozen-suite behavior.

This result rejects late Wikimedia/StackExchange annealing as the correction.
It does not support changing the validated mixture. Any 10-TPP experiment must
retain the original mixture and use an intermediate stop/go gate rather than
assuming that additional exposure will repair coherence.

## Commands

```console
go run ./cmd/waldo/ model forecast \
  composes/experiments/rung-0008-general-base-1b-quality-anneal/compose.yaml

go run ./cmd/waldo/ model train general-base-1b-quality-anneal-01 \
  composes/experiments/rung-0008-general-base-1b-quality-anneal/compose.yaml \
  --hostfile ~/hostfile
```
