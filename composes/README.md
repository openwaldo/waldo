# Model-building ladder

Each model must reach a testable endpoint before the next rung begins. A model
inherits the previous checkpoint when its architecture and tokenizer remain
compatible. A different architecture starts new weights but reuses the proven
corpus recipe, training process, and evaluation gates.

Runtime estimates below cover training after data and the environment are
ready. They are planning ranges, not measured promises. Replace them with
observed WALDO run evidence as each rung completes.

## Canary / smoke test (`0000-canary.yaml`)

| Field | Plan |
| --- | --- |
| **Status** | Existing compose; ready |
| **Builds from** | Random initialization |
| **Success criteria** | <ul><li>The model may be unusable</li><li>Training and evaluation complete</li><li>Checkpoint and resume work</li><li>Exported artifact runs inference</li></ul> |
| **Model type** | Small dense monolithic model; approximately 14M parameters |
| **Corpus requirements** | <ul><li>Small raw-text sample</li><li>Small structured-conversation sample</li></ul>Current selection is sufficient. |
| **WALDO requirements** | <ul><li>Basic ingestion and compose</li><li>Training lifecycle</li><li>Artifact verification</li><li>Inference</li></ul>Current support is sufficient. |
| **Recommended hardware** | Apple M4 Max with 128 GB unified memory |
| **Approximate runtime** | 1-3 minutes |

## Babbling model (`0001-babble.yaml`)

| Field | Plan |
| --- | --- |
| **Status** | Existing compose; ready for a formally evaluated run |
| **Builds from** | Random initialization using the canary-proven pipeline |
| **Success criteria** | <ul><li>Stable short-form language</li><li>Improving held-out loss</li><li>Simple corpus recall</li><li>No repetition collapse</li></ul> |
| **Model type** | Small dense monolithic foundation; approximately 76M parameters |
| **Corpus requirements** | <ul><li>Edited prose: Gutenberg</li><li>Reference text: Wikimedia</li><li>Scientific exposition: PLOS</li></ul>Current selection is sufficient for this rung. |
| **WALDO requirements** | <ul><li>Current dense training</li><li>Fixed generation tests in addition to loss</li></ul> |
| **Recommended hardware** | 1x NVIDIA H100 80 GB |
| **Approximate runtime** | 1-2 hours for 1.57B tokens |

## Conversation model (`0002-conversation.yaml`)

| Field | Plan |
| --- | --- |
| **Status** | Existing compose; promising prior result, but not formally promoted |
| **Builds from** | New larger initialization using the babbling model's proven recipe and tests |
| **Success criteria** | <ul><li>Direct answers</li><li>Simple constraint following</li><li>Prior-turn context</li><li>Correction handling</li><li>Necessary clarification</li><li>No tool-call syntax</li></ul> |
| **Model type** | Dense monolithic foundation plus conversation SFT; approximately 337M parameters |
| **Corpus requirements** | <ul><li>Natural multi-turn dialogue</li><li>Broad instruction data</li><li>Quality-filtered responses</li><li>Bounded Interaction Contract examples</li></ul> |
| **WALDO requirements** | <ul><li>Change SFT to `assistant-response-modeling`</li><li>Preserve assistant-only loss masks</li><li>Add fixed conversation tests</li><li>Replay foundation regression tests</li></ul> |
| **Recommended hardware** | 1x 8-GPU NVIDIA H100 SXM system |
| **Approximate runtime** | 4-8 hours for approximately 12B pretraining tokens plus SFT |

## Tool-use model (`0003-tool-use.yaml`, revision required)

| Field | Plan |
| --- | --- |
| **Status** | Existing replacement candidate is not approved; redesign before running |
| **Builds from** | Promoted `0002` checkpoint with the same architecture and tokenizer; do not repeat foundation or conversation training |
| **Success criteria** | <ul><li>Decides whether a tool is needed</li><li>Calls only a provided tool</li><li>Produces schema-valid arguments</li><li>Handles results and errors</li><li>Grounds the final answer in results</li><li>Retains conversation quality</li></ul> |
| **Model type** | Dense conversation model plus tool-use SFT; approximately 337M parameters after revision |
| **Corpus requirements** | <ul><li>One normalized call protocol</li><li>Matched tool and no-tool cases</li><li>Unavailable-tool cases</li><li>Clarification and invalid-argument cases</li><li>Empty and failed results</li><li>Result-grounding examples</li></ul> |
| **WALDO requirements** | <ul><li>Selectable tool-data categories</li><li>Fixed tool and conversation regression tests</li><li>Tool-specific metrics</li><li>Inference tool registry and execution loop</li></ul> |
| **Recommended hardware** | 1x NVIDIA H200 141 GB |
| **Approximate runtime** | 1-2 hours for the revised tool-only stage; the current from-scratch YAML would take materially longer and should not be run |

## Capable dense foundation model

| Field | Plan |
| --- | --- |
| **Status** | Planned; corpus and evaluation work required |
| **Builds from** | New larger initialization using all proven dense recipes and foundation tests |
| **Success criteria** | <ul><li>Useful general language and factual knowledge</li><li>Summarization</li><li>Technical understanding</li><li>Code completion</li><li>Mathematical competence</li></ul>All are evaluated before assistant tuning. |
| **Model type** | Dense foundation; initial target approximately 3B parameters |
| **Corpus requirements** | <ul><li>Reference prose and books</li><li>Education and science</li><li>Technical documentation and code</li><li>Mathematics and law</li><li>Measured multilingual material</li><li>Open textbooks, stronger math, and Stack V2 Edu additions</li></ul> |
| **WALDO requirements** | <ul><li>Corpus-mixture reporting</li><li>Cross-corpus deduplication</li><li>Contamination checks</li><li>Domain evaluations</li><li>Scaling forecasts</li><li>Checkpoint comparison</li></ul> |
| **Recommended hardware** | 1x 8-GPU NVIDIA B200 SXM system |
| **Approximate runtime** | 3-6 days for an initial 3B-parameter, 60B-token candidate |

## Capable dense assistant

| Field | Plan |
| --- | --- |
| **Status** | Planned; follows the capable dense foundation |
| **Builds from** | Promoted capable dense foundation checkpoint |
| **Success criteria** | <ul><li>Passes the complete conversation gate at higher quality</li><li>Retains foundation knowledge and skills</li><li>Avoids excessive refusal, verbosity, and template repetition</li></ul> |
| **Model type** | Dense foundation plus conversation and instruction SFT |
| **Corpus requirements** | <ul><li>Human and natural dialogue anchors</li><li>Filtered broad instruction data</li><li>High-quality scored responses</li><li>Bounded reviewed Interaction Contract examples</li></ul> |
| **WALDO requirements** | <ul><li>Explicit parent artifacts</li><li>Immutable behavioral evaluation splits</li><li>Assistant-only loss</li><li>Regression reporting</li></ul> |
| **Recommended hardware** | 1x 8-GPU NVIDIA B200 SXM system |
| **Approximate runtime** | 2-6 hours for approximately 200M-500M SFT tokens |

## Reasoning assistant

| Field | Plan |
| --- | --- |
| **Status** | Planned; training corpus is incomplete |
| **Builds from** | Promoted capable dense assistant checkpoint |
| **Success criteria** | <ul><li>Multi-step mathematics</li><li>Scientific problem solving</li><li>Code generation validated by tests</li><li>Planning with verifiable outcomes</li><li>No regression in conversation or foundation gates</li></ul> |
| **Model type** | Dense assistant plus reasoning post-training |
| **Corpus requirements** | <ul><li>Redistributable worked problems and proofs</li><li>Executable code tasks</li><li>Scientific reasoning</li><li>OpenWALDO-generated examples with independently verified answers and provenance</li></ul> |
| **WALDO requirements** | <ul><li>Reasoning-specific record types</li><li>Answer verification</li><li>Sandboxed code and test execution</li><li>Contamination controls</li><li>Benchmark regression gates</li></ul> |
| **Recommended hardware** | 1x 8-GPU NVIDIA B200 SXM system |
| **Approximate runtime** | 4-12 hours for approximately 500M-2B verified post-training tokens |

## Reliable tool and agent assistant

| Field | Plan |
| --- | --- |
| **Status** | Planned; depends on the reasoning and basic tool gates |
| **Builds from** | Promoted reasoning assistant checkpoint |
| **Success criteria** | <ul><li>Retains basic tool selection and execution</li><li>Plans multi-step work</li><li>Selects among multiple tools</li><li>Recovers from failures with bounded retries</li><li>Stops correctly</li></ul> |
| **Model type** | Dense reasoning assistant plus agentic tool post-training |
| **Corpus requirements** | <ul><li>Normalized multi-step tool traces</li><li>Alternate plans and partial results</li><li>Failures and retries</li><li>Permission boundaries</li><li>Ordinary no-tool conversation anchors</li></ul> |
| **WALDO requirements** | <ul><li>Stateful tool-loop evaluation</li><li>Sandboxed executable environments</li><li>Trajectory metrics</li><li>End-to-end agent regression tests</li></ul> |
| **Recommended hardware** | 1x 8-GPU NVIDIA B200 SXM system |
| **Approximate runtime** | 2-8 hours for approximately 200M-1B trajectory tokens |

## Small sparse-MoE proof

| Field | Plan |
| --- | --- |
| **Status** | Planned; WALDO does not yet support sparse-MoE training |
| **Builds from** | Random initialization using the proven dense pipeline, corpus contracts, and evaluations |
| **Success criteria** | <ul><li>Training and resume are reliable</li><li>No expert collapse</li><li>Acceptable load balance</li><li>Matches a comparable dense control on a bounded language task</li></ul> |
| **Model type** | Small sparse-MoE foundation; target 1B-3B total and 300M-700M active parameters |
| **Corpus requirements** | <ul><li>Babbling-model foundation mixture</li><li>No new knowledge corpus required; this rung tests routing</li></ul> |
| **WALDO requirements** | <ul><li>Sparse architecture declarations</li><li>Total, active, and trainable parameter accounting</li><li>Expert parallelism</li><li>Router metrics</li><li>Distributed checkpoints</li><li>MoE-aware forecasting</li></ul> |
| **Recommended hardware** | 1x 8-GPU NVIDIA H200 or B200 SXM system |
| **Approximate runtime** | 4-12 hours for a bounded 2B-5B-token proof |

## OpenWALDO sparse-MoE foundation and assistant

| Field | Plan |
| --- | --- |
| **Status** | Planned; follows the small sparse-MoE proof |
| **Builds from** | A scaled MoE configuration starts new foundation weights; conversation, reasoning, and tools then inherit its promoted checkpoints |
| **Success criteria** | <ul><li>Meets the capable dense foundation and assistant gates</li><li>Shows useful compute efficiency</li><li>Maintains healthy routing through post-training</li></ul> |
| **Model type** | Target 10B-20B total and 2B-4B active sparse-MoE foundation with successive assistant checkpoints |
| **Corpus requirements** | <ul><li>Complete capable-foundation mixture</li><li>Sufficient domain and language diversity to exercise experts</li><li>Mixture controls that prevent one source from dominating routing</li></ul> |
| **WALDO requirements** | <ul><li>Packed training data</li><li>Distributed topology planning</li><li>Native artifact sets</li><li>Expert-level telemetry</li><li>NeMo/Megatron backend</li></ul> |
| **Recommended hardware** | 1x 8-GPU NVIDIA B200 SXM system |
| **Approximate runtime** | 4-10 days for a 50B-100B-token foundation candidate; post-training adds approximately 1 day |

## Nemotron 30B foundation adaptation

| Field | Plan |
| --- | --- |
| **Status** | Planned; begins after the smaller sparse-MoE path is proven |
| **Builds from** | Pinned Nemotron-3 Nano 30B-A3B Base; starts an external model lineage |
| **Success criteria** | <ul><li>Improves selected WALDO knowledge domains</li><li>No unacceptable base-model regression</li><li>Healthy expert routing</li><li>Exact resume and verified native export</li></ul> |
| **Model type** | Native 30B-total, approximately 3.5B-active hybrid Mamba/Transformer sparse-MoE model using full-parameter continued pretraining |
| **Corpus requirements** | <ul><li>Reviewed capable-foundation mixture</li><li>Initial bounded 1B-token proof</li><li>Optional 5B-token candidate after the proof passes</li></ul> |
| **WALDO requirements** | <ul><li>Pinned native-model import</li><li>Nemotron tokenizer and configuration</li><li>Packed data</li><li>NeMo/Megatron execution</li><li>Native distributed checkpoints</li><li>MoE and regression evaluation</li></ul> |
| **Recommended hardware** | 1x 8-GPU NVIDIA B200 SXM system with 2 TB host RAM and 8-16 TB local NVMe |
| **Approximate runtime** | 2-4 hours for 1B training tokens; 10-18 hours for 5B tokens; reserve additional time for preparation, evaluation, and export |

## Nemotron 30B post-training

| Field | Plan |
| --- | --- |
| **Status** | Planned; last rung in this ladder |
| **Builds from** | Promoted Nemotron foundation-adaptation checkpoint; conversation, reasoning, and tools produce separate ordered checkpoints |
| **Success criteria** | <ul><li>Each stage passes its corresponding smaller-model gate</li><li>Every earlier foundation and behavior gate remains passing</li><li>Adapters and merged artifacts are reproducible</li></ul> |
| **Model type** | Native sparse-MoE foundation plus full SFT or LoRA adapters |
| **Corpus requirements** | <ul><li>Reviewed conversation mixture</li><li>Verified reasoning mixture</li><li>Normalized tool mixture</li><li>Nemotron-native rendering</li></ul> |
| **WALDO requirements** | <ul><li>Native chat and tool templates</li><li>Adapter lineage</li><li>Stage-specific evaluation</li><li>Native and merged exports</li><li>Inference tool loop</li></ul> |
| **Recommended hardware** | 1x 8-GPU NVIDIA B200 SXM system |
| **Approximate runtime** | 4-12 hours for the full conversation, reasoning, tool, evaluation, and export sequence |

## Next steps

- Freeze the language, conversation, and tool evaluation sets.
- Correct and rerun `0002`, then promote a conversation checkpoint.
- Revise `0003` to fine-tune that exact checkpoint on normalized tool data.
- Build the capable dense foundation, assistant, reasoning, and agent rungs.
- Fill the textbook, mathematics, technical, and tool-corpus gaps those rungs
  expose.
- Implement and validate the small sparse-MoE proof.
- Build the OpenWALDO sparse-MoE lineage.
- Adapt the Nemotron foundation, then run its separate conversation, reasoning,
  and tool post-training stages.

The supporting native-model and backend design is in the
[foundation and sparse-MoE plan](../docs/FOUNDATION-MOE-PLAN.md).
