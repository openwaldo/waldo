# Model-building ladder

Each model must reach a testable endpoint before the next rung begins. A model
inherits the previous checkpoint when its architecture and tokenizer remain
compatible. A larger or different architecture starts new weights, but reuses
the proven corpus recipe, training process, and evaluation gates.

## Canary / smoke test (`0000-canary.yaml`)

- **Builds from:** Random initialization.
- **Success criteria:** The model may be unusable, but training, checkpointing,
  resume, evaluation, export, and inference all complete correctly.
- **Model type:** Small dense monolithic model.
- **Corpus requirements:** Small raw-text and structured-conversation samples.
- **WALDO requirements:** Basic ingestion, compose, training, lifecycle,
  artifact verification, and inference pipelines.

## Babbling model (`0001-babble.yaml`)

- **Builds from:** Random initialization using the canary-proven pipeline.
- **Success criteria:** Produces stable short-form language, improves held-out
  loss, recalls simple corpus facts, and does not collapse into repetition.
- **Model type:** Small dense monolithic foundation model.
- **Corpus requirements:** Edited prose, reference text, and scientific
  exposition. Gutenberg, Wikimedia, and PLOS are sufficient for this rung.
- **WALDO requirements:** Fixed generation tests in addition to held-out loss.

## Conversation model (`0002-conversation.yaml`)

- **Builds from:** A newly initialized larger dense model using the babbling
  model's proven foundation recipe and tests.
- **Success criteria:** Answers directly, follows simple constraints, uses
  prior-turn context, accepts corrections, asks necessary clarifying questions,
  and does not emit tool-call syntax.
- **Model type:** Dense monolithic foundation plus conversation SFT.
- **Corpus requirements:** Natural multi-turn dialogue, broad instruction data,
  quality-filtered answers, and a bounded Interaction Contract mixture.
- **WALDO requirements:** Use `assistant-response-modeling` for SFT, preserve
  assistant-only loss masks, and add fixed conversation and regression tests.

## Tool-use model (`0003-tool-use.yaml`, to be revised)

- **Builds from:** The promoted `0002` conversation checkpoint, using the same
  architecture and tokenizer. It should not repeat foundation or conversation
  training.
- **Success criteria:** Decides whether a tool is needed, calls only a provided
  tool with valid arguments, handles results and errors, produces a grounded
  answer, and retains the complete conversation capability.
- **Model type:** Dense conversation model plus tool-use SFT.
- **Corpus requirements:** One normalized call protocol; matched tool and
  no-tool cases; unavailable-tool, clarification, invalid-argument,
  empty-result, error, and result-grounding examples.
- **WALDO requirements:** Selectable tool-data categories, fixed behavioral and
  regression tests, tool metrics, and eventually an inference tool registry
  and execution loop.

## Capable dense foundation model

- **Builds from:** New larger-model initialization using the proven dense
  recipes and all earlier foundation tests. It does not inherit the smaller
  model's incompatible weights.
- **Success criteria:** Demonstrates useful general language, factual knowledge,
  summarization, technical understanding, code, and mathematical competence
  before any assistant tuning.
- **Model type:** Production-oriented dense foundation model.
- **Corpus requirements:** Balanced reference prose, books, education, science,
  technical documentation, code, mathematics, law, and measured multilingual
  material. Add open textbooks, stronger mathematical exposition and verified
  problems, Stack V2 Edu, and explicit cross-corpus deduplication.
- **WALDO requirements:** Corpus-mixture reporting, contamination checks,
  domain-specific held-out evaluations, scaling forecasts, and checkpoint
  comparison.

## Capable dense assistant

- **Builds from:** The promoted capable dense foundation checkpoint.
- **Success criteria:** Passes the conversation model's tests at materially
  higher quality while retaining the foundation evaluations.
- **Model type:** Dense foundation model plus conversation and instruction SFT.
- **Corpus requirements:** Human and natural dialogue anchors, filtered broad
  instruction data, high-quality scored responses, and the reviewed Interaction
  Contract.
- **WALDO requirements:** Explicit parent artifacts, immutable behavioral
  evaluation splits, assistant-only loss, and regression reporting.

## Reasoning assistant

- **Builds from:** The promoted capable dense assistant checkpoint.
- **Success criteria:** Solves multi-step mathematical, scientific, coding, and
  planning problems with verifiable answers while retaining conversation and
  foundation quality.
- **Model type:** Dense assistant plus reasoning post-training.
- **Corpus requirements:** Redistributable worked problems, proofs, executable
  code tasks, scientific reasoning, and OpenWALDO-generated examples with
  independently verified answers and complete provenance.
- **WALDO requirements:** Reasoning-specific data types, answer verification,
  code/test execution where applicable, contamination controls, and benchmark
  regression gates.

## Reliable tool and agent assistant

- **Builds from:** The promoted reasoning assistant checkpoint.
- **Success criteria:** Preserves the earlier tool-use gate and adds multi-step
  planning, multiple-tool selection, recovery, bounded retries, and correct
  stopping behavior.
- **Model type:** Dense reasoning assistant plus agentic tool post-training.
- **Corpus requirements:** Normalized multi-step tool traces, alternate plans,
  partial results, failures, retries, permission boundaries, and ordinary
  no-tool conversation anchors.
- **WALDO requirements:** Stateful tool-loop evaluation, sandboxed executable
  environments, trajectory metrics, and end-to-end agent regression tests.

## Small sparse-MoE proof

- **Builds from:** Random initialization using the proven dense pipeline,
  corpus contracts, and evaluations. It starts a new weight lineage.
- **Success criteria:** Trains and resumes reliably, uses experts without
  collapse or severe imbalance, and matches a comparable dense control on a
  bounded language task.
- **Model type:** Small sparse mixture-of-experts foundation model.
- **Corpus requirements:** The babbling-model foundation mixture is sufficient;
  this rung tests architecture and routing rather than additional knowledge.
- **WALDO requirements:** Sparse-MoE architecture declarations, total/active/
  trainable parameter accounting, expert parallelism, router metrics,
  distributed checkpoints, and MoE-aware forecasting.

## OpenWALDO sparse-MoE foundation and assistant

- **Builds from:** A scaled sparse-MoE configuration after the small MoE proof.
  The foundation starts new weights; conversation, reasoning, and tool stages
  inherit its promoted checkpoints in order.
- **Success criteria:** Meets the capable dense foundation and assistant gates,
  demonstrates useful compute efficiency, and retains healthy expert routing
  through each post-training stage.
- **Model type:** Sparse-MoE foundation with successive conversation, reasoning,
  and tool adapters or checkpoints.
- **Corpus requirements:** The complete capable-model mixture, with enough
  domain and language diversity to exercise expert specialization without
  allowing one source to dominate routing.
- **WALDO requirements:** Packed training data, distributed topology planning,
  native artifact sets, expert-level telemetry, and a NeMo/Megatron backend.

## Nemotron 30B foundation adaptation

- **Builds from:** A pinned Nemotron-3 Nano 30B-A3B Base checkpoint after WALDO's
  sparse-MoE pipeline has passed its smaller proof. This starts an external
  model lineage rather than inheriting OpenWALDO weights.
- **Success criteria:** Improves the selected WALDO knowledge domains without
  unacceptable regression in the original base model or unhealthy routing.
- **Model type:** Native hybrid Mamba/Transformer sparse-MoE model using
  full-parameter continued pretraining.
- **Corpus requirements:** The reviewed capable-foundation mixture, initially
  bounded to a measurable 1B-token proof before a larger continuation.
- **WALDO requirements:** Pinned native-model import, Nemotron configuration and
  tokenizer support, packed data, NeMo/Megatron execution, native distributed
  checkpoints, MoE telemetry, and base-model regression evaluation.

## Nemotron 30B post-training

- **Builds from:** The promoted Nemotron foundation-adaptation checkpoint.
  Conversation, reasoning, and tool training produce separate ordered
  checkpoints rather than one undifferentiated fine-tuning run.
- **Success criteria:** Each stage passes the corresponding capable-model gate
  while retaining every earlier foundation and behavior gate.
- **Model type:** Native sparse-MoE foundation plus full SFT or LoRA adapters.
- **Corpus requirements:** The same reviewed conversation, reasoning, and
  normalized tool mixtures used by the smaller models, rendered through
  Nemotron's native interaction protocol.
- **WALDO requirements:** Native chat and tool templates, adapter lineage,
  stage-specific evaluation, native and merged exports, and the inference tool
  loop.

## Next steps

- Freeze the language, conversation, and tool evaluation sets.
- Correct and rerun `0002`, then promote a conversation checkpoint.
- Revise `0003` to fine-tune that exact checkpoint on normalized tool data.
- Build the capable dense foundation, assistant, reasoning, and agent rungs.
- Fill the textbook, mathematics, technical, and tool-corpus gaps identified
  by those rungs.
- Implement and validate the small sparse-MoE proof before scaling MoE work.
- Build the OpenWALDO sparse-MoE lineage before adapting Nemotron 30B.
- Run Nemotron foundation adaptation first, followed by separate conversation,
  reasoning, and tool post-training stages.

The supporting native-model and backend design is in the
[foundation and sparse-MoE plan](../docs/FOUNDATION-MOE-PLAN.md).
