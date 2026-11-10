# ADR 0089: Coordinate distributed training through signed records in a git repository

Status: accepted
- Date: 2026-11-04

## Context

Training an open model with many contributors' machines needs a way to divide
the work, check each contribution, combine them, and keep the record of how the
model was built, without trusting any contributor and without running a
coordinator. The initial scope in [VISION](../VISION.md) excluded automatic pull
requests and multi-node orchestration; synchronous multi-host training
([0025](0025-torchtitan-distributed-adapter.md)) requires trusted, homogeneous
hosts.

## Decision

`waldo model join <runs-repo> <run-dir> --identity <name>` contributes to a
training run kept in a git repository, expanding WALDO's scope to asynchronous,
untrusted distributed training.

- A run is one file, `run.yaml`: a compose whose `join` block (strict) sets
  units, replicas, closers, rounds, gates, a store, and the identities allowed to
  contribute, each an SSH key. Join cuts each
  unit's work from the compose; the same file trains a single-model baseline.
- Work is claimed before it is done. Records (`waldo-join-claim`,
  `-submission`, `-merge`, schema 1) are JSON signed with `ssh-keygen -Y sign`
  in the `waldo-join` namespace, and count only if they verify, match their file
  name, run, round, stage, and the pinned run, and, for work, come from the
  slot's claimant.
- Closers set aside replicas one at a time (wrong base, odd tokens, provenance
  that does not rebuild, another unit's corpus, short budget), score the rest
  with `--evaluate-only` (0088), and merge the agreed ones with
  `base.merge` (0085). A round is decided by the first decision, in slot order,
  that a majority of closing slots sign, including the published provenance.
- Gates make runs training ladders: a stage's last round checks its merge
  against a held-out loss, or an evaluation through `waldo model gate`; a failed
  gate adds rounds of the stage up to a limit, and a gate that never passes stops
  the run. The result is part of the closers' decision.
- Weights and their provenance (0084) are published to a file store, pinned by
  hash. Remote stores and runs repositories come later, through a relay.
- Each identity lives in `~/.waldo/join/<identity>`. Join runs `git` and `ssh-keygen`.

## Consequences

- No process coordinates a run, and no single contributor or closer is trusted;
  the repository's history is the record of how the model was built.
- Every model traces back to its corpora: closers rebuild each submission's
  training closure, and the majority's decision commits to it. Each merge
  publishes `JOIN-ROUND.json`, linking its weights back to the signed records.
- Not defended: a colluding majority of closers or replicas, a copied replica
  (identical weights count as agreement, since honest training can reproduce
  them; commit-reveal would stop copies), and the plan's author.

`internal/join` is a bounded domain of the one binary ([0001](0001-single-binary.md)) that trains,
measures, and merges only through the model domain. Relates to [0035](0035-local-s3-credentials.md),
[0075](0075-immutable-evaluation-and-contamination-gates.md), and
[0025](0025-torchtitan-distributed-adapter.md).
