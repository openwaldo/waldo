# ADR 0041: Record chartable training telemetry as CSV

Status: accepted
- Date: 2026-08-09

The schema-1 header below remains supported. ADR 0081 defines the additive
schema-2 header used for new files and the append-only compatibility rule.

## Context

`RUN.json` preserves verified checkpoints, evaluations, and terminal outcome,
but intentionally does not retain every training sample. Console messages are
useful live and unsuitable as a stable source for charts, comparisons, or an
independent monitoring process.

## Decision

Every real or simulated training run appends `TELEMETRY.csv` beside `RUN.json`.
The schema has this fixed header:

```text
observed_utc,elapsed_seconds,run_id,stage,attempt,event,state,step,planned_steps,tokens,planned_tokens,loss,heldout_loss,heldout_perplexity,learning_rate,tokens_per_second,eta_seconds,message
```

WALDO writes a row when an attempt starts and ends and for every backend
progress, checkpoint, and evaluation event. Each row is flushed and synced
before it is considered recorded. A resumed attempt appends to the same file
with a new attempt number and a new attempt-relative elapsed clock.

CSV contains scalar operational observations only. Immutable configuration,
execution and corpus identity remain in `RUN-BOM.json`; verified checkpoints,
evaluations, artifacts, and terminal state remain in `RUN.json`. Native model
exports retain all three files.

## Consequences

- Spreadsheets, plotting tools, scripts, and future advisory processes can
  consume live training data without parsing terminal text.
- Telemetry failure is a training failure rather than silently producing an
  incomplete operational record.
- Monitoring and optional LLM advice can remain outside the backend and use
  the same provider-neutral data contract.
