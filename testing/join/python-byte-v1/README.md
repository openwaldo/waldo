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

The plan goes into the runs repository as `runs/python-byte-v1/` (this directory's `run.yaml`,
`eval.bom.json`, `eval.jsonl` and `evaluate_python.py`). It runs on `join-local` stacked on the
lineage (#41) and merge (#42) PRs; record the commit of the `waldo` binary used here before round 0.

Results: TBD.
