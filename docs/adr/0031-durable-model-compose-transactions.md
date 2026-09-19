# ADR 0031: Resume durable model-compose transactions

Status: accepted

## Decision

WALDO content-identifies a compose transaction from the target model, strict
compose, ordered corpus BOM hashes, and current model identity. The active
model remains at `<model.root>/<name>` while transaction metadata lives beneath
`<model.root>/.waldo-compose`.

Each invocation takes a non-blocking per-model lock. Completed stages are
verified and skipped. An interrupted stage resumes the same run from its newest
verified compatible checkpoint. A failed attempt with a complete verified
checkpoint is retained as interrupted and resumes from that checkpoint; a
failed attempt without one is terminal and is not silently replayed.

The transaction is removed after every stage completes and the model BOM
commits. Failures without a checkpoint remain terminal. A storage failure can prevent
both the terminal run state and transaction from being persisted atomically.
For the narrow resulting case—a model still marked `running` with no retained
transaction—WALDO checks the non-blocking per-model lock. If no process owns
it, WALDO marks the abandoned attempt interrupted and reconstructs the
transaction from the compatible archived compose and run BOM. The same
training command or `waldo model continue <name>` then resumes the newest
verified checkpoint. A held lock remains an active run and fails closed.
