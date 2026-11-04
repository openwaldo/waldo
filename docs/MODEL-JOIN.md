# Distributed training with `waldo model join`

`waldo model join` lets many machines train one model with no coordinator. A
run lives in a git repository: a plan plus signed records of every unit of
work. Every model it produces traces back to its corpora. Design:
[ADR 0089](adr/0089-distributed-training-through-signed-records.md).

```bash
waldo model join <runs-repo> <run-dir> --identity <name> [--once] [--rounds <n>] [--email <address>] [--key <path>] [--wait <seconds>]
```

Each pass does the next piece of work and commits its record. `--once` does one
piece; `--rounds <n>` stops after n rounds merge; otherwise join loops until the
run completes. An identity keeps its key, clone, and models in
`~/.waldo/join/<identity>/`; interrupted training resumes from its checkpoint.

## Runs repository

```
runs/<run>/
  run.yaml                               # the run file: a compose with a join block
  rounds/round-NNNN/
    claims/<slot>[.<attempt>].yaml(.sig) # slot: <unit>-<n> or close-<n>
    submissions/<unit>-<identity>.yaml(.sig)
    merges/<identity>.yaml(.sig)
```

A run file is an ordinary compose ([Model compose](MODEL-COMPOSE.md)) with a
`join:` block, so `waldo model train baseline run.yaml` trains the same model
alone. For each unit and round, join cuts a compose from it: one stage, the
round's merge as `base`, the unit's data, seed, and budget.

```yaml
join:
  name: python-v1
  units: {u0: [TheAlgorithms/Python], u1: [pallets/flask, httpie/cli]}
  replicas: 2
  store: file:///srv/waldo-join
  identities: {alice: {key: ssh-ed25519 AAAA...}}
  rounds: {bootstrap_tokens: 2000000, pretrain: 4, tokens_per_round: 8192000, posttrain: 10, epochs_per_round: 8}
```

| `join` key | Meaning |
| --- | --- |
| `name`, `units` | Run name; unit to the sources its pre-training reads. Fine-tuning units take slices of the first corpus. |
| `replicas`, `closers` | Identities per unit per round; identities closing each round (default 3, majority decides). |
| `claim_minutes` | How long a claim holds its slot. Default 60. |
| `store`, `identities` | `file:///path`; name to `{key: <ssh public key>}`. |
| `rounds` | `bootstrap_tokens`, `pretrain` rounds of `tokens_per_round`, `posttrain` rounds of `epochs_per_round`. |
| `base` | Optional starting model instead of a bootstrap: `model`, `url` (`file://`), `model_sha256`, `files`. |
| `gates` | Optional bar per stage; see [Gates](#gates). |
| `replica_tolerance`, `replica_floor` | Agreement thresholds; defaults 0.25 and 0.001. |
| `min_steps_per_round` | Minimum steps per unit. |

## Rounds

Round 0 bootstraps on every unit's sources (training the tokenizer, if the run
file asks), or starts from `base`. Each later round starts from the previous
merge. An identity claims a slot, then trains a unit or closes the round; a
claim not followed by its record within `claim_minutes` can be retaken.

Every record pins a hash of the run file and its gate files, so editing the run
starts it afresh. Each unit holds out the same records every round, so its
losses compare across rounds.

## Judging and closing

A closer sets aside replicas that started from other weights, trained other
tokens, parameters, or corpora, or whose provenance doesn't check; scores the
rest with `--evaluate-only` ([ADR 0088](adr/0088-score-weights-without-training.md));
and accepts the replica most others agree with. Accepted units merge by
token-weighted mean ([ADR 0085](adr/0085-merge-models-with-a-common-ancestor.md)).

The round is decided by the first merge a majority of closing slots sign.
Each merge publishes its accepted units' run BOMs, the union of their training
closures ([ADR 0090](adr/0090-publish-training-provenance-across-lineage.md)), and
`JOIN-ROUND.json`, linking the weights back to the signed records.

## Gates

```yaml
  gates:
    - {after: pretrain, heldout_loss_at_most: 1.6, extra_rounds: 3}
    - {after: posttrain, evaluation: eval-bom.json, evaluator: [python3, evaluate_python.py], extra_rounds: 4}
```

A stage's last round checks its gate; a failed gate adds rounds up to
`extra_rounds`, and one that never passes stops the run. Evaluators that run
model-written code should only be used on models you would run anyway.

## Testing

`testing/e2e/model-join.sh` (opt-in, `WALDO_E2E_JOIN=1`) trains three
identities through a local runs repository; one free-rides, another closes
dishonestly, and the test asserts both are caught.

## Limits

- A colluding majority of closers or replicas is not defended against.
- A copied replica passes as agreement (commit-reveal would stop it).
- The plan's author is trusted; records pin the plan.
