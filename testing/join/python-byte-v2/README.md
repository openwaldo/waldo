# python-byte-v2: python-byte-v1 at half the learning rate

[python-byte-v1](../python-byte-v1/README.md) pretrained at learning rate 0.0006 and diverged in 3 of
its first 13 units. In round 4, py-c's loss fell to about 1.6, then jumped to 5.1 at step 140 and ended
worse than the round started (held-out 2.39 to 3.26). Both replicas diverged at the same step, so they
agreed and the unit was merged. Round 3 was the last merge whose units all improved.

A run's records pin its plan, so changing the rate starts a new plan. This one starts from v1's round-3
merge (`base:`, pinned by every file's SHA-256) with the pre-training rate at 0.0003 and the 9
pretraining rounds left of v1's 12. Everything else is v1's recipe. Lineage runs through the base back
to v1's bootstrap.

## Launch

The plan is this directory's `run.yaml`, `eval.bom.json`, `eval.jsonl` and `evaluate_python.py`,
committed to the local runs repository as `runs/python-byte-v2/`. ROUNDS is however many fit the day; it
resumes at the next round (stop during a cooldown). 17 rounds is the most it can take: 9 pretraining, 4
post-training and the gate's 4 extra.

    cd ~/.waldo/join/local/seed
    caffeinate -s env ROUNDS=4 RUNS_REPO=$HOME/.waldo/join/local/runs.git \
      JOIN_IDENTITY_PREFIX=mdagosta-bot- IDENTITIES=3 JOIN_THROTTLE=0.08 \
      JOIN_EMAIL=346140425+mdagosta-bot@users.noreply.github.com \
      ~/src/openwaldo/waldo-waldito/testing/join/run-identities.sh runs/python-byte-v2

Results: TBD.
