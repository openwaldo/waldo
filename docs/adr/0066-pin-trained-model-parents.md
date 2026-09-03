# 0066: Pin trained managed-model parents

Status: accepted

## Decision

A schema-1 compose using `base.model` may initialize from a completed,
non-simulated training run with a terminal `model.safetensors` artifact. WALDO
selects the requested `run_id`, or the latest eligible run when it is omitted,
and resolves the declaration to the parent model ID, run ID, run-BOM SHA-256,
weight SHA-256, and byte size.

The child plan, model record, and aggregate model BOM carry the same parent
pin. The verified parent weights are hard-linked or copied into the child at
`base/model.safetensors`; later parent training or removal cannot change the
child's initialization. The child architecture must exactly match its parent.

If a managed model has no eligible trained run, its verified pulled origin
remains a valid base. Simulated runs are never usable as model weights.

## Consequences

- Foundation, conversation, tool-use, and later model rungs can form an
  auditable compose ladder without repeating earlier training stages.
- A resolved compose is reproducible and can explicitly select an older run.
- Child storage may consume another copy of the parent weights when hard links
  are unavailable, in exchange for self-contained lifecycle integrity.
- The schema-1 additions are optional; existing plans, records, BOMs, and
  origin-based composes remain readable.
