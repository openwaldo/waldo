# python-byte-v3: python-byte-v2 from scratch on the vetted stack

[python-byte-v2](../python-byte-v2/README.md)'s recipe, with the pre-training rate at 0.0003, trained from
scratch like [python-byte-v1](../python-byte-v1/README.md): bootstrap, 12 pretraining rounds, 4
post-training rounds and the gate. It replaces v2, which ran on code from before the lineage (#41) and merge
(#42) PRs were reworked: the binary built from them can't read v2's models. v3 uses its own
identities, so their model stores start empty.

## Launch

The plan is this directory's `run.yaml`, `eval.bom.json`, `eval.jsonl` and `evaluate_python.py`,
committed to the local runs repository as `runs/python-byte-v3/`. ROUNDS is however many fit the day; it
resumes at the next round (stop during a cooldown). Keep the lid open while it trains.

    cd ~/.waldo/join/local/seed
    caffeinate -s env ROUNDS=4 RUNS_REPO=$HOME/.waldo/join/local/runs.git \
      JOIN_IDENTITY_PREFIX=mdagosta-v3- IDENTITIES=3 JOIN_THROTTLE=0.08 \
      JOIN_EMAIL=346140425+mdagosta-bot@users.noreply.github.com \
      ~/src/openwaldo/waldo-waldito/testing/join/run-identities.sh runs/python-byte-v3

Results: TBD.
