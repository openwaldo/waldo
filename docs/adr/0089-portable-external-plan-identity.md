# ADR 0089: Separate external training specification from execution

Status: accepted
- Date: 2026-11-12

## Decision

An external plan identifies a training specification, not the resulting weights.
Weights are identified by artifact digests in output and model BOMs. The plan
contains trainer and tokenizer sources (repository, commit, tree, path and
manifest digest), named environments (name, kind, lock digest and interpreter
version), schema documents, resolved recipe parameters, architecture, interaction,
WALDO-owned data parameters and corpus BOMs. Hardware and launch choices never
enter its hash. Native formats, behavior and identities are unchanged.

The environment installer still verifies interpreter bytes, the selected wheel
hashes, installed files and cache receipts on every use. Interpreter digest,
platform tag and the sorted selected wheel filenames move to run evidence.
The reference CUDA lock merges the existing aarch64 and x86_64 wheel sets with
unchanged package versions; its header records the reproducible merge command.

The existing `x-waldo-role: execution` annotation on the trainer's probe
parameter schema is authoritative. The manifest selects the pinned probe
command; its schema bytes and source are identity-bearing. Unannotated values
default to recipe (identity). WALDO rejects execution annotations on WALDO-owned
data keys, including global batch size and seed. Unknown classification questions
default to identity. The trainer is trusted to classify its own opaque math;
WALDO cannot infer arbitrary program semantics from a property name.

Stage `trainer` contains recipe inputs; a separate optional stage `execution`
object contains execution inputs. Values in the wrong block fail before resolve.
Resolve receives their union. WALDO compares recipe resolution against a
single-process resolution with execution inputs omitted, since a fixed
decomposition may only fit the requested world shape. Resolved recipe values
alone enter `recipe_sha256`; all resolved values enter the run BOM. Global batch
size stays identity and micro-batch × accumulation × world size must equal it.

For qwen3-hf, micro-batch, accumulation, attention implementation, gradient
checkpointing and compilation are execution choices. Its existing logging and
reporting annotations also remain execution. Precision (`bf16`, `fp16`, `tf32`),
optimizer, schedule, initialization seed, learning rate and warmup remain recipe.
The compose retains `trainer.determinism`, passed unchanged in `begin`, but it
is run evidence only.

## Reproducibility and resume

Run evidence distinguishes two possible reproducibility claims:

- **Bitwise:** same plan, same execution config and `determinism: required`.
- **Statistical:** same plan on any hardware, with equivalent held-out NLL and
  bits-per-byte within an explicitly stated tolerance.

These are claim prerequisites, not verified guarantees. The PoC verifies neither
level; `model reproduce`, enforcement of deterministic kernels and cross-hardware
output qualification remain follow-ups. Plan equality never proves equal weights.

Resume remains exact: schema-2 checkpoint manifests authenticate the plan through
an explicit `plan_sha256`, normalized begin/model ID and original run BOM, and record the
execution config. That config comprises profile, launcher, world shape,
resolved execution parameters, determinism, and trainer/tokenizer platform tags,
interpreter digests and selected wheel sets. Differences fail before train launch
with field names. GPU model, driver, CUDA and kernel are recorded but not compared.
Every attempt writes an immutable `RUN-BOM-ATTEMPT-*.json`; `RUN.json` references
it and the model/export BOM authenticates the segment files. The original run BOM
and checkpoint integrity anchors remain immutable. Resume across execution
configurations is outside the PoC.

This amends ADRs 0084 and 0086's identity/environment/determinism wording and
clarifies ADRs 0085 and 0088: exact compatibility means identity plus execution
config, and topology remains preserved. ADR 0017's phrase placing execution facts
in “the model plan and run BOM” does not describe native code: native execution
facts are in the run BOM. This decision changes no native behavior.

## Compatibility

This corrects the experimental external identity encoding. Earlier external plans
are not migrated in place or claimed to share the new identity; use a new model
name. Schema-1 external checkpoint fixtures remain historical evidence but cannot
resume through the schema-2 writer/reader. Built-in checkpoint formats are untouched.
