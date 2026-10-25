# ADR 0083: Version the compose envelope and separate training engines from host policy

Status: accepted
- Date: 2026-10-22
- Amends: [0017](0017-portable-training-backend-contract.md) and
  [0020](0020-host-backend-policy.md)

## Context

ADR 0017 keeps backend selection out of model composes so a compose stays
portable and machine policy stays on the machine. Training models defined by
Hugging Face Transformers needs a compose to say which implementation defines
the architecture and training semantics. Read literally, an engine field in a
compose is a backend field.

These are different facts. MLX, PyTorch, and TorchTitan execute one WALDO
architecture and training definition; choosing between them changes where and
how the same model trains, not what model it is. A Transformers
`config_class`, its config values, and Trainer optimizer and scheduler
semantics define a different model and objective. They cannot be derived from
host policy, and two hosts that resolved different engines would not be
training the same compose.

## Decision

### Schema-2 envelope

Compose schema 2 introduces explicit `training.engine` and
`architecture.provider` discriminators and places architecture values beneath
`architecture.config`. Schema 1 remains readable with its existing meaning.

The native path supports `waldo-native` for both discriminators. The reader
strictly validates the envelope, then translates it into the existing
authoritative native compose type. Native architecture hashes, parent matching,
resolved profile behavior, checkpoint compatibility, and run BOMs are unchanged.
Retained compose history contains this normalized schema-1 form.

The experimental `huggingface-transformers` engine uses the same lifecycle and
worker framing, with a distinct architecture family and pinned provider spec.
Config dimensions project into lifecycle resource fields; construction uses only
the provider config. The package pin is part of config interpretation identity.
The normalized retained compose is an internal schema-1 representation with
provider evidence; native schema-1 identities are unchanged. JSON
canonicalization keeps YAML and persisted JSON argument/config maps comparable.

### Engines, backends, and devices

WALDO distinguishes three layers. Only the first may appear in a compose.

1. **Engine and provider (portable).** The implementation that gives the
   architecture and training parameters their meaning: `waldo-native` or
   `huggingface-transformers`, with the provider's pinned package, model and
   config classes, config, and allowlisted Trainer arguments. These are part of
   architecture and run identity. They do not name a host, device, framework
   build, path, or credential.
2. **Backend adapter (host policy).** The framework adapter that executes an
   engine on this machine, selected by `model.backend` under ADR 0020. The
   native engine resolves to MLX, PyTorch, TorchTitan, or explicit `fake`. The
   Transformers engine resolves only to its PyTorch-based Transformers worker.
   `auto` and `pytorch` select it; `mlx`, `torchtitan`, and `fake` conflict with
   the engine and fail before a run ID or run directory exists.
3. **Devices and topology (host policy).** Device kind and selection,
   visible accelerators, node count, world size, rank, and launch mechanism.
   These come from the shared PyTorch hardware probe and host environment,
   such as `WALDO_TRANSFORMERS_DEVICE` and `CUDA_VISIBLE_DEVICES`, never from a
   compose. Trainer arguments that select devices or distributed execution
   remain reserved and fail closed.

Portable execution semantics that happen to carry device-flavored provider
names, such as `per_device_train_batch_size`, `fp16`, `bf16`, and
`gradient_checkpointing`, are engine semantics, comparable to the native
precision and recomputation controls of ADR 0073. While the Transformers engine
is single-process, the per-device batch times gradient accumulation is the
global optimizer batch of ADR 0071. That resolved global batch, accumulation,
and precision are recorded as immutable run facts before training. A future
multi-process Transformers adapter must preserve that global batch rather than
scale it with the device count.

Schema-1 composes have no engine field and always mean `waldo-native`.

ADR 0017's resolver contract applies to every engine: resolution returns the
adapter and immutable execution facts, including engine, provider package pin,
backend revision, framework, runtime, host, accelerators, node count, and world
size, before model or run state is created. A host that cannot execute the
compose's engine fails during resolution with the same reporting rules as a
missing framework.

### Ownership and provider identity

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

Pinned data-only fast tokenizers use local snapshots isolated from unpinned
files, loaded offline through the verified Transformers package. Source commit,
file hashes, and explicit special IDs belong to architecture identity. Planning
and training share fallible encoding semantics, without automatic BOS/EOS;
WALDO owns record EOS framing. Provider files survive training and export
byte-for-byte. WALDO does not acquire tokenizer assets or execute custom code.

A single embedded registry owns allowed model/config pairs. Mixtral explicitly
retains router balancing loss and total versus active-parameter evidence.
Text-only Qwen3.5 exercises hybrid full/linear-attention layers without
requiring external kernels.

### Lifecycle contracts

Engine choice does not exempt a run from lifecycle contracts. ADR 0029,
0044, and 0079 apply to Transformers runs as written. Where the Transformers
engine does not yet meet them, the gap is unimplemented work, not an accepted
exception. Any engine-specific exception requires its own ADR.

## Consequences

- A compose remains portable across hosts that implement its engine. Selecting
  the Transformers engine narrows which hosts qualify; it never adds machine
  facts to the compose.
- `model.backend` keeps one meaning on every engine: a host preference among
  adapters that can execute the compose's engine.
- The experiment can test random initialization, Trainer execution,
  managed-weight continuation, loss masking, and provenance without a
  downloaded checkpoint. Observed parameter counts come from the constructed
  model.
- Real GPU validation remains an operator acceptance requirement, separate
  from CPU-only test success.
- Transformers runs adopt ADR 0029 resume rather than a separate mechanism:
  checkpoint bundles carry Trainer optimizer, scheduler, scaler, and random
  state, and WALDO positions the record stream. Until that is implemented, an
  interrupted Transformers stage restarts that stage, and resume is not
  advertised.
- Upstream chat-template rendering, arbitrary model families and Trainer
  subclasses, distributed execution, tiktoken release packaging, and transitive
  dependency locks remain planned. Supported pairs and exact limits live in
  [MODEL-COMPOSE.md](../MODEL-COMPOSE.md), and the maintained compose and
  [lifecycle](../MODEL-LIFECYCLE.md) contracts state the three layers.
