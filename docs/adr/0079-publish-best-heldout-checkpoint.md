# ADR 0079: Publish the best held-out checkpoint

Status: accepted

## Context

A fixed-budget training stage can improve and then regress on its immutable
held-out set. Previously, WALDO always published the weights from the last
optimizer step. A regressed terminal checkpoint could therefore become the
starting point for the next stage even when the same run had already produced
and measured a better model.

## Decision

When a stage has held-out evaluation, its terminal artifact is the evaluated
checkpoint with the lowest finite `heldout_loss`. Ties select the earliest
checkpoint. Each backend saves every newly best evaluated candidate even when
it falls between the configured periodic checkpoint boundaries.

Checkpoint and evaluation histories are durable resume state. At completion,
the backend reloads the selected checkpoint, writes the terminal weights from
it, and evaluates the serialized artifact on the same pinned held-out set.
WALDO records the selected step, consumed tokens, metric, and value and rejects
a PyTorch completion that lacks artifact verification for that selection.

The requested optimizer-step budget still completes. Checkpoint selection is
not automatic early stopping, and it does not replace independent downstream
quality evaluation.

## Consequences

- A later stage starts from the best measured checkpoint rather than blindly
  inheriting the final optimizer step.
- Interrupted runs retain enough history to select an earlier checkpoint after
  resume.
- Summaries can expose last-step regression and the checkpoint actually
  published.
- New-best checkpoints may use more temporary storage than cadence alone would
  require.
