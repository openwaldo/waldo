# Rung 0008 fresh 1B data-rebalance qualification

This is a fresh-from-initialization correction of the rejected rung-0008
lineage. It changes only the corpus views and weights. Architecture, tokenizer,
context, optimizer, learning rate, schedule, batch, seed, and five-core-TPP
horizon remain identical to the prior 1B run.

The previous mixture assigned 30% to child-filtered Cosmopedia and 10% to
TinyStories. Its selected model passed mechanical gates but only 13/20 manual
coherence checks. Continuing that checkpoint with either a restricted quality
anneal or another 0.5 core TPP made behavior worse. This experiment therefore
does not resume that lineage.

The replacement mixture is:

- 45% Wikimedia;
- 30% Stack Exchange;
- 15% unfiltered Cosmopedia v2;
- 5% Pressbooks; and
- 5% LibreTexts.

This removes TinyStories and the child-audience filter while retaining a
bounded amount of synthetic explanatory text. The two open-textbook sources
add edited educational prose without allowing either small source to dominate.
Evaluation and resumable checkpoints occur every 49,286 steps, including the
exact 2.5-core-TPP midpoint, to avoid retaining redundant 12 GiB states.

## Stop/go gate

At the first persisted checkpoint at or above 2.5 core TPP:

- metrics must be finite and checkpoint inference must reload exactly;
- held-out loss must still be improving over the preceding evaluation;
- at least 18/20 sampled outputs must contain two complete sentences;
- sampled phrase loops must be at most 1/20 and greedy loops at most 2/20; and
- conservative manual review must find at least 13/20 outputs grammatical and
  on topic through three sentences.

A failed gate stops this run, preserves evidence, and triggers another bounded
recipe iteration. It never disables the persistent monitoring schedule.

## Final gate

- the selected checkpoint reloads exactly and lies in the final 20%;
- at least 18/20 sampled outputs contain two complete sentences;
- sampled phrase loops are at most 1/20 and greedy loops at most 2/20; and
- at least 18/20 sampled outputs are grammatical and on topic through three
  sentences.

EOS is diagnostic for this raw base model. A passing five-TPP result authorizes
a fresh ten-TPP qualification using the same mixture; it does not authorize a
restarted optimizer continuation.

## Commands

```console
go run ./cmd/waldo/ model forecast \
  composes/experiments/rung-0008-general-base-1b-data-rebalance/compose.yaml

go run ./cmd/waldo/ model train general-base-1b-rebalanced-5tpp-01 \
  composes/experiments/rung-0008-general-base-1b-data-rebalance/compose.yaml \
  --hostfile ~/hostfile
```
