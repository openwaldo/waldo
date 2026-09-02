# Reference model ladder

These composes are reviewed experiments, not hardware benchmarks. Each rung
must demonstrate a capability before a higher rung is added.

| Compose | Intended result | Readiness |
| --- | --- | --- |
| `0000-canary.yaml` | Exercise training and inference cheaply | Ready |
| `0001-babble.yaml` | Produce coherent text and basic interaction | Ready |
| `0002-conversation.yaml` | Sustain a simple user/assistant exchange | Ready |
| `0003-tool-use.yaml` | Select tools, construct calls, consume results, and answer | Revised after first tool-use run; ready for validation |

Run `waldo model forecast` before allocating a training host. A reference
compose is ready only when every corpus path exists, required record semantics
survive ingestion, the forecast fits the target system, and its predecessor's
evaluation gate has passed.

## Practices learned from completed runs

1. **Size the base from its pretraining budget.** For a new dense model, start
   near 20 high-quality pretraining tokens per parameter. Count pretraining
   tokens, not later instruction repetitions. WALDO forecasts the exact
   parameter count; it is deliberately derived rather than declared.
2. **Use capability names.** A rung describes what the result should do, not a
   GPU model or elapsed time. Hardware and backend policy stay outside YAML.
3. **Match the tokenizer to the job.** Use `r50k_base` for compact English
   experiments. Use `cl100k_base` when code, multilingual text, long structured
   syntax, or tools materially matter.
4. **Build knowledge before behavior.** Pretraining uses diverse, clean prose
   and code. Conversation and tool stages then use structured conversations
   with `assistant-response-modeling`; prompts and tool results are context,
   while assistant turns are targets.
5. **Bound synthetic post-training.** More instruction tokens can erase base
   knowledge and increase repetition. Use a lower learning rate, a small fixed
   budget or few epochs, and compare every evaluation checkpoint. Do not assume
   the final checkpoint is the best one.
6. **Treat weights as exposure, not coverage.** In a fixed-token stage, weights
   choose the approximate mixture until the budget ends; they do not guarantee
   that every record is seen. Use an epoch-driven stage only when a complete
   pass over every selected record is intentional.
7. **Require structured tool evidence.** Tool training records must retain tool
   definitions, assistant calls, tool results, and final answers. Ordinary
   question/answer data cannot substitute for this contract.
8. **Use row filters only with assessed rows.** Schema-1 shards are retained
   with a warning because they lack per-row content assessments. Reingest the
   corpora used by an important run as schema 2 before relying on `main_content`,
   repetition, or boilerplate filters.
9. **Do not mistake discussion for reference material.** Source code and public
   engineering mail are valuable domain evidence, but raw headers, quotations,
   patches, and repeated threads can dominate a small model. Normalize general
   message structure and keep discussion data subordinate to explanatory prose.
10. **Promote measured results, not plausible YAML.** Record held-out loss,
    checkpoint behavior, representative prompts, corpus consumption, runtime,
    and failure modes. Remove or revise a rung that does not improve its stated
    capability.
11. **Teach when not to call a tool.** Tool-call traces must be mixed with
    ordinary assistant responses rendered through the same interaction
    contract. A corpus where every prompt requires a call teaches call syntax,
    not tool selection.
12. **Do not mix textual call protocols.** Bare JSON arrays, tagged JSON calls,
    and terminal-agent transcripts are different output contracts. Select one
    reviewed protocol for a reference run; add another corpus only after its
    calls have been normalized to that contract.

## Tool-use gate

`0003-tool-use.yaml` uses the same `user-assistant-v1` interaction contract as
the validated conversation rung, with structured system, user, assistant, and
tool roles preserved by the canonical conversation records. Its approximately
650M parameters receive 16B pretraining tokens. Cosmopedia supplies clean
explanatory and procedural text; it improves the foundation but does not itself
teach tool calling.

The first tool-use experiment used approximately 1.18B parameters, 24B
pretraining tokens, and a complete pass over six heterogeneous tool and agent
corpora. It took about ten days on one H200. The resulting model could emit
recognizable call-like text, but it also invented tools for ordinary questions,
mixed tagged and bare-array call syntax, and repeated calls when no tool registry
was present. That run demonstrated syntax exposure, not reliable tool selection.

The revised rung is intentionally an iteration model rather than a capability
maximum. Its final 20M-token stage uses Hermes as the sole tool-call protocol and
mixes it with ordinary Interaction Contract and HelpSteer responses. xLAM,
ToolACE, SmolTalk2 tool traces, and OpenThoughts Agent remain valuable source
material, but they are excluded from this reference compose until their call
representations are normalized and their effect can be evaluated independently.

Before training this rung:

- ingest Cosmopedia-v2 and confirm its logical path;
- inspect Hermes samples to confirm tool definitions, strict JSON assistant
  calls, tool results, and final answers all survived conversion;
- confirm the ordinary-response corpora contain no implicit tool-call targets;
  and
- forecast the compose and verify memory and runtime on the intended host.

The current `waldo model chat` interface can exercise learned textual tool-call
syntax, but it does not yet accept a tool registry or execute a tool loop. That
runtime boundary must be implemented and tested before WALDO describes the
model as an end-to-end tool-using assistant.

## Planned foundation and sparse-MoE lineage

The numbered schema-1 ladder remains the executable reference for WALDO's
current dense models. The planned path for continued pretraining of imported
foundation models, behavioral post-training, specialization, sparse-MoE
metadata, native artifact sets, and a NeMo/Megatron backend is documented in
the
[Foundation, post-training, and sparse-MoE plan](../docs/FOUNDATION-MOE-PLAN.md).
Planned schema-2 examples must not be added to this directory as executable
YAML until the corresponding schema and lifecycle contracts are implemented.
