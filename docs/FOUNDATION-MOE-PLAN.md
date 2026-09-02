# Foundation, post-training, and sparse-MoE plan

Status: **planned; not implemented**.

This document defines the intended model-development path beyond WALDO's
current schema-1 dense decoder training. It covers models trained from random
initialization, continued pretraining of imported foundation models,
behavioral post-training, specialization, and sparse mixture-of-experts (MoE)
architectures. Current executable behavior remains defined by
[Model compose](MODEL-COMPOSE.md) and [Model lifecycle](MODEL-LIFECYCLE.md).

## Model lineage

WALDO should represent model development as an ordered lineage of independently
reproducible stages:

1. **Foundation training** starts from random initialization and learns from a
   general-purpose corpus mixture.
2. **Foundation adaptation** continues pretraining a pinned base checkpoint on
   a reviewed WALDO knowledge mixture.
3. **Behavioral post-training** teaches conversation, instruction following,
   reasoning presentation, tool protocols, and safety behavior.
4. **Specialization** produces domain- or task-specific adapters or checkpoints
   without rebuilding the foundation.

Every stage consumes a verified artifact set, emits a new immutable artifact
set, records its exact corpus and execution BOMs, and has an evaluation gate.
A base compose therefore describes a foundation-producing operation, whether
it starts from random weights or continues a pinned foundation checkpoint.
Fine-tuning composes describe intentional behavioral or domain changes.

The stages should remain separate composes joined by a lineage declaration,
rather than becoming one large machine-specific compose. Hardware and backend
selection remain execution policy outside the portable model definition.

## First sparse-MoE proof

The first target is the pinned
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` checkpoint. It is a hybrid
Mamba/Transformer sparse-MoE model with approximately 30 billion total
parameters and 3.5 billion parameters active per token. The intended proof is:

```text
Pinned Nemotron-3 Nano 30B-A3B Base
        |
        +-- WALDO continued pretraining (initially 1B, then up to 5B tokens)
        |
        +-- General conversation and instruction SFT
        |
        +-- Normalized tool-use SFT
        |
        +-- Optional domain adapters
```

This sequence proves more than tool and conversation tuning: it exercises
foundation adaptation, sparse routing, full-parameter training, parameter-
efficient training, native model artifacts, and stage lineage.

Tool records must remain canonical structured conversations in WALDO and be
rendered into Nemotron's native tool-call representation during preparation.
Hermes JSON, bare call arrays, terminal-agent traces, and Nemotron native tool
markup must not be mixed as assistant targets without an explicit normalization
step.

## Sparse-MoE model contract

Dense parameter count is not an adequate resource description for an MoE
model. WALDO should record and forecast at least:

- total parameters, which drive weight, checkpoint, and optimizer storage;
- active parameters per token, which help characterize computation;
- trainable parameters, which distinguish full training from adapters;
- expert count, shared experts, and experts selected per token;
- router and load-balancing configuration;
- tensor, expert, pipeline, context, and data-parallel topology; and
- activation, optimizer, checkpoint, host-memory, local-storage, and network
  requirements.

Training observations should include expert utilization, router auxiliary loss,
load balance, and dropped or overflowed tokens where the implementation exposes
them. A run is not considered healthy merely because aggregate loss decreases.

## NeMo/Megatron backend boundary

NeMo/Megatron should be a new execution adapter beside MLX, PyTorch, and
TorchTitan. WALDO remains the control plane and owns:

- compose and lineage resolution;
- corpus selection, filtering, weighting, and provenance;
- deterministic training and evaluation preparation;
- run BOMs, lifecycle state, progress, and cancellation;
- artifact verification and checkpoint lineage; and
- evaluation gates and release records.

The backend owns:

- native Nemotron construction and configuration;
- tensor, expert, pipeline, context, and data parallelism;
- optimizer sharding, mixed precision, and activation checkpointing;
- the distributed training loop and efficient checkpoint writing; and
- translation of native metrics into WALDO training events.

Production runs should use a pinned NVIDIA NeMo container by immutable digest.
The recorded execution identity must include the container, NeMo, Megatron,
CUDA, NCCL, host, accelerator, and topology versions. The compose declares
required capabilities but does not name NeMo or any other backend.

## Prepared data boundary

Streaming individual JSON records over the current worker protocol is not the
intended high-throughput boundary. WALDO should materialize immutable, packed,
tokenized shards on local or shared storage before launching NeMo. A prepared
data manifest should bind:

- token IDs and loss masks;
- sequence boundaries and packing policy;
- training and evaluation membership;
- canonical record and corpus identities;
- exact token counts; and
- content digests for every shard.

For conversational stages, only declared supervised roles contribute to the
loss. System prompts, user prompts, tool definitions, and tool results remain
context unless the stage explicitly declares otherwise.

## Native artifact sets

An imported or trained native model cannot be represented as a single assumed
`model.safetensors` file. WALDO should verify and address an artifact set that
can contain:

- sharded model weights;
- distributed optimizer and scheduler state;
- random and data-loader state required for exact resume;
- tokenizer and native model configuration;
- NeMo distributed-checkpoint metadata;
- a Hugging Face-compatible inference export;
- a PEFT adapter for parameter-efficient stages; and
- an optional merged deployment artifact.

The artifact-set manifest becomes the initialization reference for the next
lineage stage. Native distributed checkpoints are retained for resume; portable
exports are separate release artifacts.

## Initial hardware profile

The reference execution target is one 8-GPU NVIDIA B200 SXM system with
NVSwitch, 180 GB HBM per GPU, 2 TB host memory, and at least 8 TB of local NVMe
workspace. Initial validation uses BF16, a 4,096-token sequence length,
microbatch size one, sharded optimizer state, and activation checkpointing.

This host is expected to accommodate LoRA SFT, full-parameter SFT, and bounded
continued pretraining of the 30B-A3B model. Longer contexts are promoted only
after the 4K configuration is stable. Additional hosts are a throughput and
scale decision, not an initial model-fit requirement.

The following are reservation estimates to be replaced by measured WALDO run
evidence:

| Workload | Initial reservation estimate |
| --- | ---: |
| Environment, conversion, and packing | 2-4 hours |
| 1B-token continued-pretraining pilot | 2-4 hours |
| 5B-token continued-pretraining candidate | 10-18 hours |
| Conversation SFT | 1-2 hours |
| Tool SFT | 20-60 minutes |
| Evaluation and export | 1-3 hours |
| Complete initial 1B-token proof | 8-12 hours |
| Complete 5B-token candidate | 18-30 hours |

These estimates describe accelerator reservations, not implementation
schedules. Forecast output must identify assumptions and must not treat these
numbers as measured performance.

## Delivery order and acceptance gates

Implementation follows this dependency order without attaching calendar
estimates:

1. **External proof runner:** WALDO exports a pinned dataset and resolved
   configuration that a pinned NeMo container can train on one B200 host.
2. **Compose and lineage contract:** native origins, sparse architecture facts,
   adaptation strategies, prepared-data manifests, and artifact sets receive
   durable schemas.
3. **Native backend:** WALDO resolves, launches, observes, cancels, resumes, and
   verifies NeMo/Megatron runs through the normal lifecycle.
4. **Forecast and evaluation:** resource calculations become MoE-aware and
   stage promotion depends on recorded quality and routing evidence.
5. **Reference lineage:** reviewed foundation-adaptation, conversation, and
   tool-use composes form the first Nemotron lineage.

The external proof is accepted when a pinned input produces a resumable native
checkpoint and measured throughput on the reference host. The native backend
is accepted when the same run is fully represented by WALDO lifecycle records,
can resume after interruption, and produces verified native and inference
artifacts. The reference lineage is accepted only when each stage improves its
declared evaluation gate without unacceptable regression in earlier gates.
