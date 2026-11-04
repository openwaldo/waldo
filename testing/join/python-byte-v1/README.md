# python-byte-v1: does the byte tokenizer fix copying?

python-bpe-v1 learned which program to write but garbled the values it copied from the request:
"says hello to Frank" became `print("Hello, flu!")`, and "the string apple in uppercase" became
`"icppercacy"`. The byte-level python-basics-v1 (9.5M parameters) copied the same values exactly
(`"apple"`, `"pbbvyaipcaxg"`, `range(1, 72 + 1)`) but knew fewer programs. This run keeps
python-bpe-v1's recipe and changes the tokenizer to bytes.

## What changes from python-bpe-v1

- **Tokenizer:** the built-in byte tokenizer (259 entries) instead of a trained 16K BPE.
- **Base:** trained from scratch (a 20M-byte bootstrap, then 12 pretraining rounds), since there is
  no byte code-m0. python-bpe-v1 started from code-m0's 756M tokens.
- **Size:** the same transformer body (8 layers, width 512), about 25M parameters; python-bpe-v1's
  extra 8M sat in its BPE vocabulary.

Unchanged: the three Python units, 6.5M per unit per pretraining round (bytes now, tokens before),
post-training on python-basics-v4 and python-programs-v1, the 0.9 gate on 120 held-out programs,
and the 30-request demo.

## Prediction

Taught task types should pass almost all of their demo requests, including hello-to-a-name and
reversing a string, because values are copied byte by byte. Unseen task types (0/20 for both
earlier models) should stay low: the training data doesn't teach them.

## Launch

The plan is this directory's `run.yaml`, `eval.bom.json`, `eval.jsonl` and `evaluate_python.py`,
committed to the local runs repository as `runs/python-byte-v1/`. It runs on `join-local` stacked on
the lineage (#41) and merge (#42) PRs: the `waldo` binary was built from `ef3318c`. Three identities
share one M2 Max, each throttled to 0.10, with a 20-minute cooldown between rounds (lid open):

    cd ~/.waldo/join/local/seed
    caffeinate -s env ROUNDS=21 RUNS_REPO=$HOME/.waldo/join/local/runs.git \
      JOIN_IDENTITY_PREFIX=mdagosta-bot- IDENTITIES=3 JOIN_THROTTLE=0.10 \
      JOIN_EMAIL=346140425+mdagosta-bot@users.noreply.github.com \
      ~/src/openwaldo/waldo-waldito/testing/join/run-identities.sh runs/python-byte-v1

21 rounds is the most the plan can take: the bootstrap, 12 pretraining rounds, 4 post-training rounds
and the gate's 4 extra rounds; the run stops early when the gate passes. `measure.sh` then scores the
final merge on the demo and the full held-out set. Besides pass_rate, the evaluator prints how often
the answer had the reference's program shape and how many of the request's values it copied.

Results: TBD.
