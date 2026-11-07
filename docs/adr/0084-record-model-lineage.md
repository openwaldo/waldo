# ADR 0084: Keep model history as documents beside the records

Status: accepted
- Date: 2026-11-05

## Context

A model records only one step of its history: the parent run it was trained
from (ADR 0066), or the pinned weights it started from (ADR 0026). Everything
earlier is known only through the models in between. If an intermediate
model is deleted or lost, the intermediate history is lost too, breaking the
provenance that shows how, and on what data, a model was trained.

## Decision

When WALDO makes a model from another model, by compose or by pull, it saves
the documents behind it in a new file, `LINEAGE.json` (`waldo-model-lineage`,
schema 1). The file holds plans, run BOMs, origin BOMs, model BOMs, and release
`BOM.json` files, each as its exact bytes under its SHA-256. It is written
once, when the model is made. No existing record changes, and nothing pins it.

- WALDO follows history only through pins that already exist. A document
  counts only when its hash matches and it names what its pin names.
- A missing or mismatched document is a gap: WALDO warns and carries on.
- Exports ship `LINEAGE.json` beside `BOM.json`, filling what gaps they can.
  A pull keeps it out of the origin, so a model's ID doesn't depend on it.

## Consequences

- The models in a history can be deleted. Their documents travel with every
  model and release made from them.
- As in [VISION.md](../VISION.md), hashes prove a history's identity and
  integrity, not that it is true: its documents say what their makers said,
  and signing says who said it.
- A release's history is public: every ancestor's plan, model name, and run
  BOMs with their corpus BOMs.
- A model keeps the gaps it had when it was made; an export fills what it can.
