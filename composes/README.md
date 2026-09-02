# Reference model-building strategy

These composes are reviewed experiments, not hardware benchmarks. The
directory is a capability ladder: each rung must reach a declared, testable
endpoint before work proceeds to the next capability.

The current numbered files are not a weight lineage. They use different model
dimensions, context lengths, and tokenizers, so each starts a new model. They
build on experimental evidence, not on the preceding weights. Once an
architecture is large enough for the next behavior, later behavioral composes
should initialize from its promoted checkpoint instead of repeating
pretraining. A change to architecture or tokenizer starts a new weight lineage
and must replay all earlier gates.

| Compose | Endpoint | Current disposition |
| --- | --- | --- |
| `0000-canary.yaml` | Prove the complete lifecycle mechanically | Release gate; no model-quality claim |
| `0001-babble.yaml` | Produce stable short-form language and retain simple facts from its corpus | Foundation pilot |
| `0002-conversation.yaml` | Sustain ordinary multi-turn conversation without emitting tool syntax | Conversation baseline; observed as useful, formal gate still required |
| `0003-tool-use.yaml` | Decide whether a tool is needed, call only a provided tool correctly, consume its result, and preserve the conversation baseline | Revised candidate; not promoted until the tool gates pass |

Run `waldo model forecast` before allocating a training host. A reference
compose is ready only when every corpus path exists, required record semantics
survive ingestion, the forecast fits the target system, and its predecessor's
evaluation gate has passed.

## Required revisions before the next candidate runs

- Preserve completed runs as historical evidence; do not reinterpret them
  after changing a compose.
- Change the two conversational stages in `0002-conversation.yaml` from
  `causal-language-modeling` to `assistant-response-modeling` before its next
  run. Under current WALDO semantics, the causal objective makes every rendered
  conversation token a target and therefore overrides `supervised_roles`.
- Establish and freeze the conversation behavioral set before selecting a new
  `0002` checkpoint.
- Add selectable tool-decision and tool-execution corpus categories before the
  next `0003` reference run. Until then, keep the revised compose a candidate.
- When a promoted conversation checkpoint has the architecture and tokenizer
  intended for tools, initialize tool training from that exact artifact rather
  than repeating foundation training.

## Promotion rules

A compose is promoted only when its run records contain all of the following:

1. A verified corpus and run BOM, measured corpus consumption, and a complete
   resumable checkpoint.
2. A fixed evaluation set selected before training and excluded from training.
3. Held-out loss and capability-specific behavioral results for the selected
   checkpoint, not merely the final checkpoint.
4. Regression results for every earlier capability in the same lineage.
5. Representative successful and failed examples, including the reason each
   failed case is acceptable or blocks promotion.
6. Measured runtime, peak memory, throughput, and the exact backend and host.

Thresholds are written before the candidate run. They are not lowered after
examining its output. A failed rung is revised and rerun; its intended next
capability is not added to compensate for the failure.

## Capability ladder and testable endpoints

### 0000: lifecycle canary

The canary tests WALDO, not language quality. It passes only when training,
checkpointing, interruption and resume, evaluation, export, and inference all
complete with verified artifacts. Loss must remain finite and consumed-token
accounting must match the run contract. Generated text only needs to prove that
the saved artifact can execute.

Required corpus properties are small, redistributable raw text and structured
conversation records that exercise both training views. The existing Public
Domain Review, PEP, Dolly, and OASST1 selection is sufficient.

### 0001: language foundation pilot

The foundation pilot tests whether the chosen architecture, tokenizer, corpus
mixture, and token budget learn stable language. It is not expected to be a
general assistant. Its fixed evaluation should cover continuation, elementary
factual recall from held-out source documents, short summarization, syntax,
and repetition. Promotion requires improving held-out loss, coherent bounded
completions, and no systematic collapse into repetition or markup.

Required corpus roles are explanatory reference prose, sustained edited prose,
and scientific exposition. Gutenberg, Wikimedia, and PLOS cover the minimum
roles for this pilot. They do not constitute a balanced production foundation
mixture.

### 0002: conversation baseline

The conversation model must first retain the language-pilot gate. It then has
to answer the latest request directly, follow simple format constraints, use
prior-turn context, accept a correction, ask for clarification only when
needed, and stop after one assistant response. It must not emit tool-call
markup because no tool contract has been taught.

The initial behavioral gate is:

- at least 80% direct-answer and explicit-constraint success;
- at least 75% prior-context and correction success;
- at least 99% of no-tool prompts contain no invented tool-call syntax; and
- no material regression in the language-pilot evaluation.

These thresholds are initial promotion requirements and may be changed only
before a candidate run, with the reason recorded.

Conversation training needs three distinct data roles:

- natural multi-turn dialogue for turn-taking, repair, clarification, and
  context retention;
- broad instruction/response examples for task coverage; and
- a small reviewed interaction contract for precise behavior.

OASST1/2, Taskmaster, Schema-Guided Dialogue, and CCPE provide human or natural
dialogue. Tulu 3 and UltraChat provide broader instruction coverage but contain
substantial synthetic material and must remain separately measurable.
HelpSteer2 supplies quality-filtered responses. Interaction Contract v1
supplies reviewed behavior cases and must remain bounded so its templates do
not dominate model voice.

### 0003: tool-use candidate

Tool use is two capabilities and must be evaluated in order:

1. **Tool decision:** determine whether any call is necessary and select only
   from the tools supplied for that request.
2. **Tool execution:** construct schema-valid arguments, consume success,
   empty, and error results, and produce a grounded final answer.

The current compose combines these in one SFT stage because schema 1 cannot
select the required behavioral views independently. Before its next reference
run, the corpus needs explicit selectable categories so decision examples can
be trained and evaluated separately from call construction.

The initial behavioral gate is:

- at least 90% balanced accuracy on tool-required versus no-tool prompts;
- at least 90% correct selection from the provided tool registry;
- at least 95% syntactically and schema-valid calls when a call is required;
- at least 90% final-answer grounding in the returned tool result;
- no invented tool names in at least 99% of no-tool or unavailable-tool cases;
- correct clarification or bounded failure for missing arguments and tool
  errors in at least 80% of cases; and
- no material regression in the complete conversation gate.

Hermes Function Calling is the current positive-call source. Interaction
Contract v1 and ordinary HelpSteer2 responses provide some contrast, but they
are not yet a sufficient matched decision set. ToolACE, xLAM, SmolTalk2 tool
traces, and OpenThoughts Agent must remain excluded until their protocols are
normalized and their behavioral categories can be selected independently.

## Corpus requirements and gaps

The index contains enough material to run the current experiments, but not yet
enough reviewed balance to claim a broadly trained foundation or reliable tool
assistant.

| Training role | Available candidates | Requirement before a promoted larger model |
| --- | --- | --- |
| Edited and reference prose | Gutenberg, Wikimedia, PressBooks, PLOS | Measure age, domain, language, and duplication balance |
| Science | PLOS, peS2o, PubMed, arXiv papers and abstracts | Prefer full explanatory text; partition licenses and deduplicate papers |
| Technical prose | Stack Exchange, GitHub Archive, Stack V2 HTML | Add and validate Stack V2 Edu; prevent discussions and markup from dominating |
| Source code | Permissive code and Cloud Native Core | Retain language/project balance and pair code with explanatory material |
| Mathematics | mathlib and limited open texts | Add redistributable exposition, worked solutions, proofs, and verified reasoning seeds |
| General education | PressBooks and selected open books | Add license-qualified Open Textbook Library material; resolve the OpenStax policy decision |
| Multilingual foundation | Multilingual portions of Wikimedia and existing corpora | Measure language coverage and acquire high-quality prose before making multilingual claims |
| Natural conversation | OASST1/2, Taskmaster, Schema-Guided Dialogue, CCPE | Add balanced open-domain dialogue and prevent task-oriented call-center voice from dominating |
| General instruction SFT | Tulu 3, UltraChat, Dolly, Aya, HelpSteer2 | Preserve source labels, cap synthetic repetition, and maintain human-quality anchors |
| Interaction behavior | Interaction Contract v1 | Complete category review, near-duplicate analysis, and clean train/evaluation separation |
| Tool decision | Partial coverage in Interaction Contract and ordinary SFT | Add matched positive, no-tool, unavailable-tool, ambiguous, and clarification examples |
| Tool execution | Hermes and typed Interaction Contract records | Normalize one protocol and add result grounding, empty results, errors, and multi-step cases |
| Behavioral evaluation | Interaction Contract validation/evaluation splits | Make immutable non-training evaluation selections first-class in model runs |

Corpus acquisition should prioritize gaps that unblock a gate, in this order:

1. Create reviewed, selectable tool-decision and tool-execution views with
   matched negative examples.
2. Expose immutable behavioral evaluation splits that cannot enter training.
3. Complete the natural-conversation mixture and measure its domain and voice
   balance.
4. Add open educational and mathematical exposition suitable for foundation
   training.
5. Add Stack V2 Edu and measure overlap with existing technical corpora.
6. Expand multilingual data only alongside an explicit language evaluation.

Adding more undifferentiated SFT data is not a remedy for tool confusion. The
next tool experiment should change one variable at a time: decision balance,
protocol normalization, or execution/error coverage. It should retain the same
promoted conversation base and replay the same evaluation sets.

## Compose design going forward

Within a weight-compatible lineage, use separate stages or composes for:

```text
foundation checkpoint
    -> conversation adaptation
    -> tool-decision adaptation
    -> tool-execution adaptation
    -> optional domain specialization
```

Each arrow must name its exact parent artifact. Conversation examples remain
in later mixtures as regression anchors, but later stages receive smaller
budgets and lower learning rates. Reasoning, preference optimization, safety
specialization, and domain adapters are separate future branches; they should
not be mixed into the tool experiment before its gate passes.

When a larger dense or sparse-MoE architecture is introduced, its foundation
stage inherits the reviewed corpus roles and evaluation gates, not incompatible
weights. The planned native-model and Nemotron path is defined in the
[foundation and sparse-MoE plan](../docs/FOUNDATION-MOE-PLAN.md).

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
