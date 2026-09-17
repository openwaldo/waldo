# ADR 0071: Version the compose execution and architecture envelope

Status: accepted

## Decision

Compose schema 2 introduces explicit `training.engine` and
`architecture.provider` discriminators and places architecture values beneath
`architecture.config`. Schema 1 remains readable with its existing meaning.

The native path supports `waldo-native` for both discriminators.
The reader strictly validates the envelope, then translates it into the existing
authoritative native compose type. Native architecture hashes, parent matching,
resolved profile behavior, checkpoint compatibility, and run BOMs are unchanged.
Retained compose history currently contains this normalized schema-1 form.

The experimental `huggingface-transformers` engine uses the same lifecycle and
worker framing, with a distinct architecture family and pinned provider spec.
Config dimensions project into lifecycle resource fields; construction uses only
the provider config. The package pin is part of config interpretation identity.
The normalized retained compose is an internal schema-1 representation with
provider evidence; native schema-1 identities are unchanged. JSON canonicalization
keeps YAML and persisted JSON argument/config maps comparable.

Trainer owns optimizer/scheduler behavior. WALDO owns corpus BOM resolution,
ordering, token framing, loss masks, stopping rules, filesystem destinations,
and release publication. Only supported TrainingArguments pass through;
reserved and unknown fields fail closed. Runs retain requested arguments and
hash resolved arguments, model config, runtime inventory, and provider-native
weights. Provider artifacts are never fed through native inference/converters.
Dedicated Transformers inference reuses the training package verifier and
tokenizer loader, consuming verified saved run files without downloads.
The existing WALDO interaction contract remains the prompt-format authority.

Wheel SHA-256 and installed source verification establish Transformers package
identity before probe and execution. Verified source loading bypasses bytecode
caches. The local Python interpreter is trusted. Other dependencies are
version-inventoried, not claimed to be hash-locked. Byte-tokenizer Hugging Face
exports include the run evidence in the normal signed release inventory.

## Consequences

The experiment can test random initialization, Trainer execution,
managed-weight continuation, loss masking, and provenance without a downloaded
checkpoint. Forecasts are rough decoder estimates; observed parameter counts
come from the constructed model. Native schema-1/schema-2 behavior remains
covered by compatibility tests.

Pinned data-only fast tokenizers use local snapshots isolated from unpinned
files, loaded offline through the verified Transformers package. Source commit,
file hashes, and explicit special IDs belong to architecture identity. Planning
and training share fallible encoding semantics, without automatic BOS/EOS;
WALDO owns record EOS framing. Provider files survive training and export
byte-for-byte. WALDO does not acquire tokenizer assets or execute custom code.

A single embedded registry owns allowed model/config pairs. Mixtral explicitly
retains router balancing loss and total versus active-parameter evidence.
Text-only Qwen3.5 exercises hybrid full/linear-attention layers without requiring
external kernels. CPU and single-GPU execution reuse WALDO's PyTorch hardware
probe; explicit GPU requests fail closed rather than falling back. Execution
evidence records hardware and precision settings. Real GPU validation remains an
operator acceptance requirement, separate from CPU-only test success.

Upstream chat-template rendering, arbitrary model
families and Trainer subclasses, distributed execution, periodic checkpoint
resume, tiktoken release packaging, and transitive dependency locks remain
planned. Supported pairs and exact limits live in MODEL-COMPOSE.md.
