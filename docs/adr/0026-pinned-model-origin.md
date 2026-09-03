# 0026: Normalize pinned open weights behind an origin BOM

Status: accepted

## Context

Continued training needs full-precision or training-precision weights, not an
inference-oriented quantized package. A mutable repository name is not a model
identity. Retaining both an upstream Safetensors checkpoint and a renamed WALDO
copy would waste storage, while treating an external model as a completed WALDO
training run would invent observations that WALDO never made.

## Decision

`waldo model pull` defaults to a Hugging Face Safetensors repository. It
resolves the requested reference to an immutable commit, pulls into private
staging, hashes every selected source artifact, and validates architecture,
tokenizer, tensor names, shapes, and precision before publishing anything.

WALDO streams tensor payloads unchanged into one managed Safetensors checkpoint
whose header uses WALDO's tensor names. Source files are then removed. A
separate schema-1 `ORIGIN-BOM.json` records the provider, repository, requested
reference, resolved commit, declared license, source-artifact hashes, and the
normalized artifacts. The immutable model plan pins the origin-BOM hash. The
aggregate model BOM selects that origin until a later complete, non-simulated
run produces current weights.

A model compose may name a local pulled base and optionally assert its origin
hash. The architecture must match exactly. The new plan pins the resolved
origin and the base is never mutated. ADR 0066 extends `base.model` to prefer a
verified completed training run when one exists.

Compatibility is explicit and fail-closed. The first profile accepts standard
bias-free Llama Safetensors using the OpenWALDO byte tokenizer. Supporting a new
tokenizer or architecture requires a named adapter and tests; implicit
retokenization, quantization, dtype conversion, and unrecorded tensor surgery
are forbidden.

## Consequences

- Pulled weights can be trained, composed, inspected, chatted with where a
  runtime supports them, and exported through the normal lifecycle.
- The exact upstream bytes remain auditable by hash and immutable repository
  URI without storing redundant source weights locally.
- An origin is lineage, not a synthetic training run.
- Native WALDO export remains the complete transfer/archive representation;
  GGUF and Ollama remain derived inference formats.
