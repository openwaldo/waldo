# python-bpe-v1: a Python model trained with `waldo model join`

A 34M-parameter Python model, trained from scratch by three volunteer identities on one M2 Max in
about 14.5 full-GPU hours. Every round is a signed record, and every training is checked by a second replica.
It writes 70% of 218 held-out programs correctly, but none of 20 tasks it was never taught; its gate
(0.9) refused it. Its training history, down to each corpus and license, ships inside the model.

## Compute

| | code-m0 (base) | python-bpe-v1 | total |
|---|---|---|---|
| rounds | bootstrap + 5 pretrain | 6 pretrain + 8 posttrain | |
| trainings (each unit twice) | 43 | 84 | 127 |
| wall-clock hours, GPU in use | 38.7 | 14.4 | 53.1 |
| GPU throttle per identity | 0.12 | 0.10 | |
| full-GPU hours (training hours × throttle) | 10.3 | 4.2 | 14.5 |
| per volunteer machine (÷ 3) | 3.4 | 1.4 | 4.8 |
| tokens processed, both replicas | 756M | 266M | 1,022M |

Half of the compute is verification: every unit trains on two identities, and a round merges only
units whose replicas agree. Numbers come from each training's RUN.json and TELEMETRY.csv.

## Provenance

What the model learned from, in order. Each step is a merge whose package records the step before it.

1. **Tokenizer**: a 16K byte-level BPE trained on 64 MB sampled from `code/permissive` and five
   `core/common-pile` English collections (LibreTexts, Pressbooks, OER Commons, news, Python PEPs).
2. **code-m0 bootstrap**: 50M tokens of the same mix, trained from random weights on one shared unit.
3. **code-m0 pretraining**, 5 rounds, 4 units: Python, web, systems code, and English.
4. **python-bpe-v1 pretraining**, 6 rounds, 3 units (py-a, py-b, py-c) over the same 40 Python
   repositories, split by repository and balanced to about 42M tokens each.
5. **python-bpe-v1 posttraining**, 8 rounds on `python-basics-v4` (3,539 prompt and program pairs)
   and `python-programs-v1` (2,270 whole scripts with a main function, a `__main__` guard, and logging).

The final package's TRAINING-CLOSURE.json is complete: it lists 11 corpus selections pinned to
waldo-index commit 33d08473, with their licenses (0BSD, Apache-2.0, BSD-2/3-Clause, CC0, CC-BY,
CC-BY-SA, GFDL, MIT, public domain, and the permissive BSD-family terms). ATTRIBUTION.md credits
every source. JOIN-ROUND.json names the run, the round, the plan's hash, and each accepted
contribution's identity, weights hash, and signed record.

## Results

Python held-out loss fell from 3.3–3.7 to about 2.7 over pretraining. All 14 rounds merged every unit; no replica pair disagreed.
The posttraining gate scores 120 held-out programs by running them: 0.55 after round 9, 0.69 after
round 11, 0.75 after round 13. It needed 0.9, so the run stopped there, as its plan says.

On all 218 held-out programs the final model scores 152 (70%). `demo.jsonl` holds 30 requests:
one from each of the 10 task types posttraining teaches, and 20 it never shows. A request passes
when the program runs and prints exactly what the reference prints, with the same structure.

| | passed |
|---|---|
| taught task types (10) | 5: divide, factorial, fibonacci, absolute value, sum |
| never-taught tasks (20) | 0 |

It picks the right program and gets it right, more often than not:

    > write a python script that prints the first 89 fibonacci numbers
    def main():
        a, b = 0, 1
        for _ in range(89):
            print(a)
            a, b = b, a + b

When it misses a taught task, it usually mangles a value it should copy:

    > write a python script that prints Hello world
        print("Hello, go, World!")
    > write a python program with a main function that says hello to Frank
        print("Hello, flu!")

Asked for something new, it answers with the nearest template it knows:

    > write a python script that multiplies 12 and 7
        print(7 + 7)
    > write a python script that counts down from 5 to 1
        for number in range(1, 3 + 1):

The mechanism held: 14 rounds, 127 trainings, every one replicated, recorded, and traceable.
The model learned the taught program families but not programming; more varied posttraining
data is the next lever. Every answer is in `results/`.

## Reproduce

`run.yaml` is the run's plan, byte for byte (its hash is in every record). `measure.sh` waits for a
run to end, then scores its final merge on `demo.jsonl` and `eval.jsonl` and totals the GPU time:

    WALDO=<waldo> WALDO_CONFIG=<config> uv run --with mlx python3 evaluate_python.py --show \
      --records demo.jsonl <model> eval.bom.json results.json
