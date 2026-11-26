# Rung 0007: interaction-contract assistant

This rung post-trains the passed 350M general-language base to recognize a
user request, respond directly, respect elementary operating boundaries, and
stop. It does not claim dependable open-domain expertise, coding, tool use, or
multi-step reasoning.

## Design

- The immutable parent is rung 0006, `general-base-350m-20tpp-01`.
- Stage one runs 75 supervised steps over the pinned 21-category interaction
  contract with assistant-only loss and a `2e-5` peak learning rate.
- Stage two runs only five steps at `5e-6` over the 2,142 formatting examples
  from that same pinned training split.
- The final model uses the `user-assistant-v1` interface and learns to emit EOS
  at the end of an assistant response.
- Both frozen interaction and raw rung-0006 generation suites decide promotion.

The exposure boundary is empirical. Fifty interaction steps preserved the base
but missed both formatting cases. Seventy-five steps passed every interaction
gate except YAML. One hundred steps passed interaction but materially damaged
foundation retention. A ten-step formatting booster fixed YAML but produced
only 17/20 two-sentence raw continuations. The promoted five-step booster is
the minimum tested intervention that passed both suites.

## Promoted result

- Model: `interaction-assistant-350m-rung7-01`
- Model ID: `d35954500ac3`
- Parent: `general-base-350m-20tpp-01`
- Interaction stage: 75 steps and 136,013 packed targets
- Formatting stage: 5 steps and 6,075 packed targets
- Interaction behavior: 42/42 EOS, 37/42 exact reference responses, at least
  37/42 conservative manual contract passes, valid JSON and YAML, zero role
  leaks, and zero loops
- Foundation retention: 18/20 outputs with two complete sentences, 16/20
  conservative grammatical/on-topic passes, one sampled loop, and zero greedy
  loops
- Both selected artifacts reloaded with zero serialization loss delta
- A separate two-stage compose reproduction matched every tensor in both
  stages; safetensors container hashes differ only because their `run_id`
  metadata is expectedly unique

The exact artifact identities and metrics are recorded in `evidence.json`.

## Promotion gates

- finite metrics, exact reload, and selected checkpoints in the final 20%;
- at least 38/42 interaction responses emit EOS without a role leak;
- at least 34/42 manually satisfy the requested interaction contract;
- both formatting cases are valid in the requested format;
- no more than one interaction loop or role leak;
- at least 16/20 sampled raw continuations remain grammatical and on topic for
  two sentences, with at least 18/20 containing two complete sentences; and
- no more than two sampled or three greedy raw-continuation loops.

Every frozen gate passed.

## Corpus preparation

Check out `openwaldo/post-training-data` with commit
`8d2454f2625a85d911cd7810816e489f0ebb1448` available, then materialize and
ingest both manifest-backed views:

```console
python3 composes/0007-interaction-assistant-350m/prepare-corpora.py \
  /path/to/post-training-data /tmp/rung-0007-corpora
go run ./cmd/waldo/ index ingest \
  /tmp/rung-0007-corpora/interaction-contract-v1 \
  ~/.waldo/index/post-train/sft/interaction-contract-v1
go run ./cmd/waldo/ index ingest \
  /tmp/rung-0007-corpora/interaction-formatting-v1 \
  ~/.waldo/index/post-train/sft/interaction-formatting-v1
```

The full view ingests the source repository's two-turn `prompt`/`response`
projection with the `dialogue-pair` adapter. That is the representation of the
published `post-train/sft/interaction-contract-v1` shard: `waldo index export
--format jsonl` of this locally prepared corpus and of the published shard
produce the same 45,000 records (JSONL SHA-256
`b46d3b97237e4df1774ba5765dfb4dba150ab973926a4a6ff265c1d4918f0450`). The typed
multi-turn `messages`, including tool calls, stay in the source rows and are
not part of this rung's training representation. The local Parquet object has
the published shard's size and differs from it only in the `plan_sha256` of
the embedded shard BOM, so its container SHA-256 is not the published one. The
formatting view is unchanged.

## Training and evaluation

The pinned rung-0006 parent must exist in the configured model store.

```console
go run ./cmd/waldo/ model forecast \
  composes/0007-interaction-assistant-350m/compose.yaml
go run ./cmd/waldo/ model train interaction-assistant-350m-rung7-01 \
  composes/0007-interaction-assistant-350m/compose.yaml
./composes/0007-interaction-assistant-350m/validate.sh \
  interaction-assistant-350m-rung7-01 /tmp/interaction-assistant-350m-rung7-01-eval
```

The compose deliberately contains two separately scheduled stages so the
optimizer resets between the broad interaction contract and the minimal
formatting correction, matching the promoted lineage.
