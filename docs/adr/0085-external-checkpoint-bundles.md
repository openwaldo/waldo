# ADR 0085: Commit and authenticate external checkpoint bundles in WALDO

Status: accepted
- Date: 2026-11-02

## Decision

Implement ADR 0084's external resume boundary with the schema-1
`waldo-trainer-checkpoint` manifest. WALDO derives per-corpus consumption from
its prepared stream at the reported step, copies hash-verified regular files
into a temporary bundle, synchronizes it, then atomically publishes it.
Trainer files must remain immutable after their event. WALDO's lifecycle
persists the manifest path and SHA-256 independently; a self-reported digest
inside a bundle is not an integrity anchor.

The manifest pins the immutable run BOM, trainer, tokenizer, selected
environment, complete normalized begin request (excluding paths and resume),
complete parameter hash, resolved execution object, world shape, position,
consumption and file roles. Resume requires exact compatibility and rechecks
the independent manifest pin and every file immediately before launch.

The wire resume includes a filename-to-absolute-path map for the verified
bundle. The trainer restores all optimizer, scheduler and random state there.
The lifecycle positions the source at the verified sequence ordinal. Native
checkpoint formats and behavior are unchanged.

The maintained contract is [External checkpoints](../EXTERNAL-CHECKPOINTS.md).
CLI/lifecycle persistence remains a separate integration; this decision does
not claim that external composes already run through the CLI.

## Consequences

- Corruption and incompatible run/world changes fail before trainer launch.
- WALDO-owned copies survive later trainer output changes.
- The event sink receives only a committed checkpoint reference; it must save
  that reference before accepting the checkpoint in lifecycle state.
- Temporary disk usage scales with step ledger length and checkpoint size;
  consumption tracking memory scales only with corpus count.
- Conformance can interrupt exactly at a durable checkpoint and compare a
  resumed run with a control, independently of the trainer framework.
