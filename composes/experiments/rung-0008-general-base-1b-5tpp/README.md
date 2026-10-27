# Rung 0008 1B general-base 5-TPP qualification

This unnumbered experiment tests whether the qualified 1B architecture becomes
a stable general-language base at five core tokens per parameter. It is not a
validated ladder rung until every final gate passes.

The architecture, tokenizer, corpus identities and weights, context, seed,
batch, optimizer, and evaluator are frozen from the successful `4e-4`
matched-token qualification. Only the cosine training horizon increases from
1.62 to 5 core TPP.

The 295,716-step plan evaluates every 24,643 steps and checkpoints every
49,286 steps. Step 147,858 is therefore both an evaluation and a resumable
checkpoint at exactly 2.5 core TPP. This sparse cadence bounds checkpoint
storage without sacrificing the midpoint decision.

## Midpoint gate

At the first publishable checkpoint at or above 2.5 core TPP:

- metrics are finite and checkpoint reload is exact;
- held-out BPB is below the 1.62-TPP result of `0.812535`;
- at least 18/20 sampled outputs contain two complete sentences;
- sampled phrase loops are at most 1/20 and greedy loops at most 2/20; and
- conservative manual review finds at least 12/20 outputs grammatical and on
  topic through three sentences.

Passing resumes the exact interrupted run. Failure stops it before spending
the remaining exposure.

## Final 5-TPP gate

- metrics are finite, artifact reload is exact, and the selected checkpoint
  lies in the final 20%;
- held-out BPB improves over the midpoint and is no worse than `0.80`;
- at least 18/20 sampled outputs are grammatical and on topic through three
  sentences;
- at least 18/20 sampled outputs contain two complete sentences; and
- sampled phrase loops are at most 1/20 and greedy loops at most 2/20.

EOS remains diagnostic for this raw base model. Passing authorizes a controlled
10-TPP exposure phase. Ten TPP must pass before 20 TPP is authorized. Forty
TPP is considered only if the 20-TPP phase still shows meaningful held-out and
behavioral gains without evidence of overtraining.

## Commands

```console
go run ./cmd/waldo/ model forecast \
  composes/experiments/rung-0008-general-base-1b-5tpp/compose.yaml

go run ./cmd/waldo/ model train general-base-1b-5tpp-01 \
  composes/experiments/rung-0008-general-base-1b-5tpp/compose.yaml \
  --hostfile ~/hostfile
```

Use the frozen rung-0006 generation suite for midpoint and final evaluation.
