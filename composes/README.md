# Model-building ladder

Each model must reach a testable endpoint before the next rung begins. A model
inherits the previous checkpoint when its architecture and tokenizer remain
compatible. A different architecture starts new weights but reuses the proven
corpus recipe, training process, and evaluation gates.

Runtime estimates cover training after data and the environment are ready.
They are planning ranges until replaced by observed WALDO run evidence.

## Canary / smoke test (`0000-canary.yaml`)

| Field | Plan |
| --- | --- |
| Status | Existing compose; ready |
| Builds from | Random initialization |
| Model type | Small dense monolithic model; approximately 14M parameters |
| Recommended hardware | Apple M4 Max with 128 GB unified memory |
| Approximate runtime | 1-3 minutes |

Success criteria:

- The model may be unusable.
- Training and evaluation complete.
- Checkpoint and resume work.
- The exported artifact runs inference.

Corpus requirements:

- Small raw-text sample.
- Small structured-conversation sample.
- Current selection is sufficient.

WALDO requirements:

- Basic ingestion and compose.
- Training lifecycle and artifact verification.
- Inference.
- Current support is sufficient.

## Babbling model (`0001-babble.yaml`)

| Field | Plan |
| --- | --- |
| Status | Existing compose; ready for a formally evaluated run |
| Builds from | Random initialization using the canary-proven pipeline |
| Model type | Small dense monolithic foundation; approximately 76M parameters |
| Recommended hardware | 1x NVIDIA H100 80 GB |
| Approximate runtime | 1-2 hours for 1.57B tokens |

Success criteria:

- Stable short-form language.
- Improving held-out loss.
- Simple corpus recall.
- No repetition collapse.

Corpus requirements:

- Edited prose from Gutenberg.
- Reference text from Wikimedia.
- Scientific exposition from PLOS.
- Current selection is sufficient for this rung.

WALDO requirements:

- Current dense training.
- Fixed generation tests in addition to held-out loss.

## Conversation level 1 (`0002-conversation1.yaml`)

| Field | Plan |
| --- | --- |
| Status | Existing known-good compose preserved |
| Builds from | New larger initialization using the babbling model's proven recipe and tests |
| Model type | Dense monolithic foundation plus conversation SFT; approximately 337M parameters |
| Recommended hardware | 1x 8-GPU NVIDIA H100 SXM system |
| Approximate runtime | 4-8 hours for approximately 12B pretraining tokens plus SFT |

Success criteria:

- Direct answers and simple constraint following.
- Prior-turn context and correction handling.
- Necessary clarification.
- No tool-call syntax.

Corpus requirements:

- Natural multi-turn dialogue.
- Broad instruction data.
- Quality-filtered responses.
- Bounded Interaction Contract examples.

WALDO requirements:

- Assistant-response modeling and assistant-only loss masks (supported).
- Add fixed conversation tests.
- Replay foundation regression tests.

## Conversation level 2 (`0002-conversation2.yaml`)

| Field | Plan |
| --- | --- |
| Status | Ready after conversation1 is trained and evaluated |
| Builds from | Continues the same managed `conversation` model; completed conversation1 corpus paths are skipped |
| Model type | Same approximately 337M-parameter dense model with English software midtraining and expanded conversation SFT |
| Recommended hardware | 1x NVIDIA H200 141 GB |
| Approximate runtime | 2-4 hours for 400M technical tokens plus 100M conversation tokens |

Success criteria:

- Improves instruction following and multi-turn coherence over `conversation`.
- Improves familiarity with software development, systems, debugging, review, and technical documentation.
- Preserves the baseline's directness, correction handling, and no-tool behavior.
- Passes the baseline conversation and foundation regression tests.

Corpus requirements:

- English-only development mailing lists spanning Linux, Git, Python, Apache,
  GCC, glibc, GNU, QEMU, Alpine, and other open-source communities.
- English technical issue, pull-request, review, and repository-documentation text.
- Smol-SmolTalk for compact-model instruction breadth.
- UltraChat 200k for additional multi-turn dialogue.

WALDO requirements:

- Append-only continuation with completed-path skipping (supported).
- Fixed side-by-side conversation evaluations.
- Promote only when it beats the previous rung without material regression.

## Tool-use model (`0003-tool-use.yaml`)

| Field | Plan |
| --- | --- |
| Status | Compose updated; run after the desired conversation level is promoted |
| Builds from | Current verified checkpoint of the managed `conversation` model |
| Model type | Dense conversation model plus tool-use SFT; approximately 337M parameters after revision |
| Recommended hardware | 1x NVIDIA H200 141 GB |
| Approximate runtime | 1-2 hours for the 20M-token tool-only stage |

Success criteria:

- Decides whether a tool is needed.
- Calls only a provided tool with schema-valid arguments.
- Handles results and errors.
- Grounds the final answer in tool results.
- Retains conversation quality.

Corpus requirements:

- One normalized call protocol.
- Matched tool and no-tool cases.
- Unavailable-tool and clarification cases.
- Invalid-argument, empty-result, and error cases.
- Result-grounding examples.

WALDO requirements:

- Verified trained-parent initialization and lineage (supported).
- Selectable tool-data categories.
- Fixed tool and conversation regression tests.
- Tool-specific metrics.
- Inference tool registry and execution loop.

## Capable dense foundation model

| Field | Plan |
| --- | --- |
| Status | Planned; corpus and evaluation work required |
| Builds from | New larger initialization using all proven dense recipes and foundation tests |
| Model type | Dense foundation; initial target approximately 3B parameters |
| Recommended hardware | 1x 8-GPU NVIDIA B200 SXM system |
| Approximate runtime | 3-6 days for an initial 3B-parameter, 60B-token candidate |

Success criteria:

- Useful general language and factual knowledge.
- Summarization and technical understanding.
- Code completion and mathematical competence.
- All capabilities are evaluated before assistant tuning.

Corpus requirements:

- Reference prose, books, and education.
- Science, technical documentation, and code.
- Mathematics, law, and measured multilingual material.
- Add open textbooks, stronger mathematics, and Stack V2 Edu.

WALDO requirements:

- Corpus-mixture reporting and cross-corpus deduplication.
- Contamination checks and domain evaluations.
- Scaling forecasts and checkpoint comparison.

## Capable dense assistant

| Field | Plan |
| --- | --- |
| Status | Planned; follows the capable dense foundation |
| Builds from | Promoted capable dense foundation checkpoint |
| Model type | Dense foundation plus conversation and instruction SFT |
| Recommended hardware | 1x 8-GPU NVIDIA B200 SXM system |
| Approximate runtime | 2-6 hours for approximately 200M-500M SFT tokens |

Success criteria:

- Passes the complete conversation gate at higher quality.
- Retains foundation knowledge and skills.
- Avoids excessive refusal, verbosity, and template repetition.

Corpus requirements:

- Human and natural dialogue anchors.
- Filtered broad instruction data.
- High-quality scored responses.
- Bounded reviewed Interaction Contract examples.

WALDO requirements:

- Explicit parent artifacts.
- Immutable behavioral evaluation splits.
- Assistant-only loss and regression reporting.

## Reasoning assistant

| Field | Plan |
| --- | --- |
| Status | Planned; training corpus is incomplete |
| Builds from | Promoted capable dense assistant checkpoint |
| Model type | Dense assistant plus reasoning post-training |
| Recommended hardware | 1x 8-GPU NVIDIA B200 SXM system |
| Approximate runtime | 4-12 hours for approximately 500M-2B verified post-training tokens |

Success criteria:

- Multi-step mathematical and scientific problem solving.
- Code generation validated by tests.
- Planning with verifiable outcomes.
- No regression in conversation or foundation gates.

Corpus requirements:

- Redistributable worked problems and proofs.
- Executable code tasks and scientific reasoning.
- OpenWALDO-generated examples with independently verified answers and complete
  provenance.

WALDO requirements:

- Reasoning-specific record types and answer verification.
- Sandboxed code and test execution.
- Contamination controls and benchmark regression gates.

## Reliable tool and agent assistant

| Field | Plan |
| --- | --- |
| Status | Planned; depends on the reasoning and basic tool gates |
| Builds from | Promoted reasoning assistant checkpoint |
| Model type | Dense reasoning assistant plus agentic tool post-training |
| Recommended hardware | 1x 8-GPU NVIDIA B200 SXM system |
| Approximate runtime | 2-8 hours for approximately 200M-1B trajectory tokens |

Success criteria:

- Retains basic tool selection and execution.
- Plans multi-step work and selects among multiple tools.
- Recovers from failures with bounded retries.
- Stops correctly.

Corpus requirements:

- Normalized multi-step tool traces.
- Alternate plans, partial results, failures, and retries.
- Permission boundaries.
- Ordinary no-tool conversation anchors.

WALDO requirements:

- Stateful tool-loop evaluation.
- Sandboxed executable environments.
- Trajectory metrics and end-to-end agent regression tests.

## Small sparse-MoE proof

| Field | Plan |
| --- | --- |
| Status | Planned; WALDO does not yet support sparse-MoE training |
| Builds from | Random initialization using the proven dense pipeline, corpus contracts, and evaluations |
| Model type | Small sparse-MoE foundation; target 1B-3B total and 300M-700M active parameters |
| Recommended hardware | 1x 8-GPU NVIDIA H200 or B200 SXM system |
| Approximate runtime | 4-12 hours for a bounded 2B-5B-token proof |

Success criteria:

- Training and resume are reliable.
- No expert collapse and acceptable load balance.
- Matches a comparable dense control on a bounded language task.

Corpus requirements:

- Babbling-model foundation mixture.
- No new knowledge corpus is required; this rung tests routing.

WALDO requirements:

- Sparse architecture declarations.
- Total, active, and trainable parameter accounting.
- Expert parallelism and router metrics.
- Distributed checkpoints and MoE-aware forecasting.

## OpenWALDO sparse-MoE foundation and assistant

| Field | Plan |
| --- | --- |
| Status | Planned; follows the small sparse-MoE proof |
| Builds from | A scaled MoE configuration starts new foundation weights; conversation, reasoning, and tools then inherit promoted checkpoints |
| Model type | Target 10B-20B total and 2B-4B active sparse-MoE foundation with successive assistant checkpoints |
| Recommended hardware | 1x 8-GPU NVIDIA B200 SXM system |
| Approximate runtime | 4-10 days for a 50B-100B-token foundation candidate; post-training adds approximately 1 day |

Success criteria:

- Meets the capable dense foundation and assistant gates.
- Shows useful compute efficiency.
- Maintains healthy routing through post-training.

Corpus requirements:

- Complete capable-foundation mixture.
- Enough domain and language diversity to exercise experts.
- Mixture controls that prevent one source from dominating routing.

WALDO requirements:

- Packed training data and distributed topology planning.
- Native artifact sets and expert-level telemetry.
- NeMo/Megatron backend.

## Nemotron 30B foundation adaptation

| Field | Plan |
| --- | --- |
| Status | Planned; begins after the smaller sparse-MoE path is proven |
| Builds from | Pinned Nemotron-3 Nano 30B-A3B Base; starts an external model lineage |
| Model type | Native 30B-total, approximately 3.5B-active hybrid Mamba/Transformer sparse-MoE using full-parameter continued pretraining |
| Recommended hardware | 1x 8-GPU NVIDIA B200 SXM system with 2 TB host RAM and 8-16 TB local NVMe |
| Approximate runtime | 2-4 hours for 1B training tokens; 10-18 hours for 5B tokens, plus preparation and evaluation |

Success criteria:

- Improves selected WALDO knowledge domains.
- Avoids unacceptable base-model regression.
- Maintains healthy expert routing.
- Resumes exactly and produces a verified native export.

Corpus requirements:

- Reviewed capable-foundation mixture.
- Initial bounded 1B-token proof.
- Optional 5B-token candidate after the proof passes.

WALDO requirements:

- Pinned native-model import and Nemotron tokenizer/configuration.
- Packed data and NeMo/Megatron execution.
- Native distributed checkpoints.
- MoE and base-model regression evaluation.

## Nemotron 30B post-training

| Field | Plan |
| --- | --- |
| Status | Planned; last rung in this ladder |
| Builds from | Promoted Nemotron foundation-adaptation checkpoint; conversation, reasoning, and tools produce separate ordered checkpoints |
| Model type | Native sparse-MoE foundation plus full SFT or LoRA adapters |
| Recommended hardware | 1x 8-GPU NVIDIA B200 SXM system |
| Approximate runtime | 4-12 hours for conversation, reasoning, tools, evaluation, and export |

Success criteria:

- Each stage passes its corresponding smaller-model gate.
- Every earlier foundation and behavior gate remains passing.
- Adapters and merged artifacts are reproducible.

Corpus requirements:

- Reviewed conversation mixture.
- Verified reasoning mixture.
- Normalized tool mixture.
- Nemotron-native rendering.

WALDO requirements:

- Native chat and tool templates.
- Adapter lineage and stage-specific evaluation.
- Native and merged exports.
- Inference tool loop.

## Next steps

- Freeze the language, conversation, and tool evaluation sets.
- Run `0002-conversation1` as the managed model named `conversation`.
- Apply `0002-conversation2` to that same model and compare its new run with the previous checkpoint.
- Define each later conversation compose cumulatively and continue the same model.
- Run `0003` after the desired `conversation` checkpoint is current.
- Build the capable dense foundation, assistant, reasoning, and agent rungs.
- Fill the textbook, mathematics, technical, and tool-corpus gaps.
- Implement and validate the small sparse-MoE proof.
- Build the OpenWALDO sparse-MoE lineage.
- Adapt the Nemotron foundation, then run its separate conversation, reasoning,
  and tool post-training stages.

The supporting native-model and backend design is in the
[foundation and sparse-MoE plan](../docs/FOUNDATION-MOE-PLAN.md).
