# Rung 0008 1B general-base 10-TPP qualification

This controlled continuation tests whether the remaining rung-0008 coherence
miss is exposure-limited. It starts from the immutable selected artifact of
`general-base-1b-5tpp-02` and restores the exact qualified 9:6:3:2 corpus
mixture. The rejected Wikimedia/StackExchange anneal is not part of this
lineage.

The selected parent artifact contains 4,441,260,032 trained tokens, or 4.5835
tokens per 968,976,384 core parameters. This compose adds 5,248,503,808 tokens,
bringing the selected lineage to exactly 10 core TPP. The peak learning rate is
reduced from `4e-4` to `1e-4` because this is continuation training after the
parent cosine schedule, not a fresh initialization.

## Mandatory +1-TPP stop/go gate

Interrupt and evaluate the exact run at the first checkpoint at or above
59,142 steps, one additional core TPP. Continue toward 10 TPP only if:

- metrics are finite and checkpoint reload is exact;
- the frozen objective suite has at least 18/20 two-sentence outputs, no more
  than one sampled loop, and no more than two greedy loops;
- conservative manual review improves from the parent's 13/20 result to at
  least 15/20 grammatical and on-topic outputs through three sentences; and
- no parent capability category shows material regression.

This is a trend gate, not promotion. Failure stops the run and rejects the
additional-exposure hypothesis before most of the budget is spent.

## Final gate

At 10 core TPP, the original rung-0008 final behavioral gates remain frozen:

- exact reload and selected checkpoint in the final 20%;
- at least 18/20 sampled outputs grammatical and on topic through three
  sentences;
- at least 18/20 sampled outputs with two complete sentences;
- sampled loops at most 1/20 and greedy loops at most 2/20.

Held-out BPB must improve on a comparable mixed-corpus partition. EOS remains
diagnostic for this raw base model.

## Probe result

The additional-exposure hypothesis is rejected before the +1-TPP gate. The
run was cleanly interrupted after the persisted +0.5-TPP checkpoint at step
29,571 because both comparable held-out loss and frozen-suite behavior moved
in the wrong direction:

- held-out loss worsened from `2.2166` at continuation start to `2.3016`;
- 20/20 sampled outputs still contain two complete sentences;
- sampled loops are 0/20 and greedy loops are 2/20;
- sampled EOS is 17/20; but
- conservative manual review is only 11/20 grammatical and on topic through
  three sentences, below both the parent's 13/20 and the 15/20 trend gate.

The exact checkpoint therefore does not justify another half-TPP, much less
the full continuation to 10 TPP. The interrupted run is not resumed. These
results show that additional exposure under a restarted cosine schedule is not
the correction for rung 0008; future work must revisit the model/data design
rather than spend more tokens on this lineage.

## Commands

```console
go run ./cmd/waldo/ model forecast \
  composes/experiments/rung-0008-general-base-1b-10tpp/compose.yaml

go run ./cmd/waldo/ model train general-base-1b-10tpp-01 \
  composes/experiments/rung-0008-general-base-1b-10tpp/compose.yaml \
  --hostfile ~/hostfile
```
