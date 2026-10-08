# Failed tiny r50k capability experiment

These composes are preserved as negative evidence. Do not promote or rerun
them as capability models.

The architecture has 16,014,336 parameters, but its tied 50,259 by 256 token
embedding contains 12,866,304 parameters. Only 3,148,032 parameters remain for
four transformer blocks and norms. The embedding therefore consumes 80.3% of
the advertised model capacity.

Observed runs:

| Model | Run ID | Tokens | Held-out loss | Deterministic generation |
| --- | --- | ---: | ---: | --- |
| `foundation-tiny-pilot-01` | `246cdae7ed71a425` | 50M | 4.9693 | Off-topic, severe phrase and sentence loops |
| `foundation-tiny-pilot-02` | `0a3bb4417957c9d0` | 160M | 4.1755 | Still off-topic, severe phrase and sentence loops |

Both runs completed correctly. Four-GPU accounting, 5:2:1 corpus exposure,
checkpoint selection, and reloaded-artifact loss were sound. Loss improved and
the final checkpoint was best in both runs, but capability remained unusable.
This matches the failure mode already documented in ADR 0047: a compact model
dominated by a large tokenizer embedding can show healthy loss while generation
collapses.

The 5M-token canary remains useful only as a systems test. The active ladder
starts capability evaluation at 76M parameters, where the r50k embedding is
42.1% rather than 80.3% of the model.
