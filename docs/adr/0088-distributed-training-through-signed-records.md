# ADR 0088: Coordinate untrusted training runs through signed records in git

Status: accepted
- Date: 2026-11-04

## Context

Many people who don't know each other have a GPU that sits idle much of the
time, and want models they can control and check rather than depend on access
to a large cluster. One machine is too small to train a useful model; many
together could. To do so without trusting any of them, the work would need to
be divided, each contribution checked, the results combined, and a record kept
of how the model was built.

Synchronous multi-host training ([0025](0025-torchtitan-distributed-adapter.md))
needs trusted, identical hosts, and [VISION](../VISION.md)'s initial scope
excluded multi-node orchestration.

## Decision

`waldo model join <runs-repo> <run-dir> --identity <name>` contributes to a
training run kept in a git repository, extending WALDO's scope to untrusted
distributed training.

- A run is one file, `run.yaml`: a compose with a `join` block naming units,
  replicas, closers, rounds, gates, a store, and the contributors' SSH keys.
- Contributors claim work, submit it, and close rounds through JSON records
  (`waldo-join-*`, schema 1) signed with `ssh-keygen -Y sign` in the
  `waldo-join` namespace. A record counts only if it verifies, fits its place
  in the run, and, for work, comes from the slot's claimant.
- Closers reject replicas that fail a check, score the rest with
  `--evaluate-only`, and merge them with `base.merge` (0085). A majority of
  closers decides each round.
- A gate can check a stage's final merge; failing adds rounds, up to a limit.
- Weights and their provenance are published to a file store, pinned by hash.

## Consequences

- No process coordinates a run and no contributor is trusted; the repository's
  history records how the model was built.
- Every merge traces to its corpora: closers rebuild each submission's training
  closure, and each merge publishes `JOIN-ROUND.json`, linking its weights to
  the signed records.
- Not defended: a colluding majority, a copied replica (identical weights count
  as agreement), and the plan's author.

`internal/join` trains, measures and merges only through the model domain,
inside the one binary ([0001](0001-single-binary.md)). Relates to
[0035](0035-local-s3-credentials.md) and
[0075](0075-immutable-evaluation-and-contamination-gates.md).
