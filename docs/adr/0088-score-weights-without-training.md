# ADR 0088: Score current weights on a compose's held-out records without training

Status: accepted
- Date: 2026-11-04

## Context

Comparing models on equal terms needs each one's loss on the same held-out
records. Held-out loss is otherwise only measured inside a training run, so
measuring an existing model meant training it: even one step at a negligible
learning rate changes its weights, and a template that hardcodes its learning
rate can silently turn that step into real training.

## Decision

`waldo model train <name> <compose> --evaluate-only` prepares the compose's
first stage exactly as training would, selects its held-out records with the
same selection code and cache, and scores the named model's current weights on
them through the selected backend's worker in an evaluate mode: load the
weights, score the held-out records once, return. It trains nothing, writes no
weights, records no run, and refuses simulated backends. It prints the loss,
the held-out set's size and SHA-256, and the weights' SHA-256.

Held-out selection depends on the corpus, filters, objective, evaluation
parameters, and seed, not on the training budget, so two machines scoring
different weights with the same compose and seed score the same records. With
the seed a training stage used, they score exactly the records that training
held out.

This is a measurement for comparing weights, such as replicas of the same
training. It does not promote a model; promotion remains governed by ADR 0075's
pinned evaluation corpora and contamination gates.

## Consequences

- The MLX and PyTorch workers gain an evaluate mode and new revisions.
- `model train` stays the single entry point for stage execution; the flag
  makes measurement explicit rather than adding a command.

Relates to [0043](0043-unified-model-training.md), [0030](0030-deterministic-heldout-evaluation.md),
and [0075](0075-immutable-evaluation-and-contamination-gates.md).
