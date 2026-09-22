# Model-building ladder

Each model must reach a testable endpoint before the next rung begins. A model
inherits the previous checkpoint when its architecture and tokenizer remain
compatible. A different architecture starts new weights but reuses the proven
corpus recipe, training process, and evaluation gates.

Runtime estimates cover training after data and the environment are ready.
They are planning ranges until replaced by observed WALDO run evidence.

The numbered ladder also owns the measured capability-per-FLOP comparison.
`0000-canary.yaml` and `0001-babble.yaml` are the systems gates,
`0002-conversation.yaml` records the undertrained 2.4B-token experiment,
`0003-conversation.yaml` is the corrected 12B-token comparison baseline, and
`0004-conversation.yaml` is the larger `conversation3` candidate. Do not run
the larger candidate until the correctness, data-plane, batch-semantics, and
evaluation gates in the [training robustness plan](../docs/TRAINING-ROBUSTNESS-PLAN.md)
pass. Corpus and license differences are tracked in the
[compose corpus licensing audit](../docs/COMPOSE-CORPUS-LICENSE-AUDIT.md) and
[nanochat coverage audit](../docs/NANOCHAT-CORPUS-COVERAGE.md).

Every conversational rung includes the compact `waldo-project-v1` corpus so
models learn stable facts about WALDO and the responsibilities of open-source
AI without treating WALDO as the assistant's identity.

Stages are weight-changing operations and execute strictly in YAML order. Keep
broad foundation data first, domain or technical adaptation next, conversation
training after that, and narrow assistant, alignment, or tool-use training
last. If the relative order of completed stages is corrected, use a new model;
replaying an existing model cannot retroactively change its training order.
See [Stage ordering is part of the model design](../docs/MODEL-COMPOSE.md#stage-ordering-is-part-of-the-model-design).

For domain knowledge, lead with explanatory reference material and question/
answer text. Source code and expert discussions are valuable supporting data,
but a mixture dominated by patches, issue traffic, or mailing-list replies can
teach domain vocabulary without reliably teaching basic facts. Keep enough
general instruction data after domain training to make the knowledge usable,
then finish with the narrowest validated behavior stage.

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
| Approximate runtime | Measure from promoted G0 evidence for the 600M-token ceiling |

This systems-gate compose is for private research while Gutenberg and PLOS
record-level rights are under review. It intentionally omits the
`distributable` policy, retains full corpus provenance, and must not be used as
evidence that its resulting model can be distributed.

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

## Conversation level 2 (`0002-conversation.yaml`)

| Field | Plan |
| --- | --- |
| Status | Completed 2.4B-token experiment; materially undertrained and not the known-good baseline |
| Builds from | New larger initialization using the babbling model's proven recipe and tests |
| Model type | Dense monolithic foundation plus conversation SFT; approximately 337M parameters |
| Recommended hardware | 1x 8-GPU NVIDIA H100 SXM system |
| Approximate runtime | Measure from promoted G1 evidence for the 2.4B-token ceiling |

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

- Causal conversation modeling is retained in 0002 as the historical comparison point.
- Add fixed conversation tests.
- Replay foundation regression tests.

## Conversation level 3 (`0003-conversation.yaml`)

| Field | Plan |
| --- | --- |
| Status | Corrected 12B-token comparison baseline; rerun under a fresh model name after artifact-integrity fixes |
| Builds from | Fresh random initialization; same core architecture and first three corpus stages as 0002, with lossless float32 portable weights |
| Model type | Dense monolithic foundation plus conversation SFT; approximately 337M parameters |
| Recommended hardware | 1x 8-GPU NVIDIA H100 SXM system |
| Approximate runtime | Measure directly; approximately five times the pretraining exposure of 0002 |

Success criteria:

- Restore direct answers, basic factual grounding, and simple constraint following.
- Preserve prior-turn context and correction handling.
- Avoid the repetition collapse observed in the 2.4B-token run.
- Beat 0002 on fixed foundation and conversation evaluations.

Corpus requirements:

- The exact 0002 corpus selection and weights for its first three stages.
- 12B pretraining tokens, one bounded pass over each broad conversation stage,
  then five low-rate passes over the compact WALDO project grounding corpus.

WALDO requirements:

- Train under a fresh model name; do not append pretraining after 0002 post-training.
- Apply assistant-only response loss during both conversation stages.
- Use lower conversation-stage learning rates and one pass to limit the held-out-loss regression observed in the initial 0003 run.
- Keep the portable artifact in float32 until reduced-precision publication
  passes WALDO's live-versus-publishable loss check for this architecture.
- Keep compiled execution disabled until compiled and eager held-out losses
  agree throughout a representative run.
- Select project grounding against its own held-out set rather than allowing
  the much larger broad SFT mixture to hide failure to learn WALDO facts.
- Add fixed side-by-side generation and held-out evaluations.
- Pass `./testing/training-acceptance.sh` on the target Linux GPU software
  stack before starting the full run.

## Conversation level 4 (`0004-conversation.yaml`)

| Field | Plan |
| --- | --- |
| Status | Larger candidate; blocked on systems, evaluation, and corpus gates |
| Builds from | Random initialization with the restored 0003 curriculum embedded first |
| Model type | Approximately 758M-parameter dense model, 4,096-token context, technical knowledge midtraining, expanded assistant-only SFT, and isolated WALDO grounding |
| Recommended hardware | 4x NVIDIA H200 GPUs; one or two nodes |
| Approximate runtime | Determine from promoted G1/G2 evidence for the roughly 6.1B-token curriculum |

Success criteria:

- Clearly improves instruction following, knowledge, and multi-turn coherence over the restored 0003 model.
- Correctly answers basic factual questions about operating systems, Linux, programming, and systems administration.
- Improves familiarity with software development, systems, debugging, review, and technical documentation.
- Preserves the baseline's directness, correction handling, and no-tool behavior.
- Passes the baseline conversation and foundation regression tests.

Corpus requirements:

- Cosmopedia v2 educational material, Stack Exchange technical Q&A, PLOS, and
  Wikimedia form the majority of the 5B-token foundation mixture.
- Linux/GNU and cloud-native source, repository documentation, and a bounded
  amount of Linux, Git, and Python development discussion provide concrete
  systems vocabulary in a separate 1B-token stage. Known non-English rows are
  excluded; legacy rows without language metadata are retained.
- Tulu 3, Smol-SmolTalk, and UltraChat provide broader assistant supervision.
- The validated Interaction Contract and HelpSteer2 behavior anchor follows
  broad SFT and immediately precedes the narrow project-grounding stage.

WALDO requirements:

- Use assistant-only loss for every structured conversation stage; role
  markers, user prompts, and system context remain conditioning input.
- Keep portable parameters in float32 until this larger architecture has
  passed the reduced-precision artifact-integrity gate.
- Keep compiled execution disabled until it passes the compiled/eager
  equivalence gate.
- Evaluate the final WALDO grounding stage on its own held-out records.
- A fresh model is required because conversation3 has more than twice the
  parameter capacity and context length of the 0003 conversation model as well as a corrected
  stage order.
- Fixed side-by-side conversation evaluations.
- Promote only when it beats the previous rung without material regression.
- Train this compose under a fresh model name; the numeric compose prefix
  describes its ladder position, not its model artifact name.

## Tool-use model (`holding/tool-use.yaml`)

| Field | Plan |
| --- | --- |
| Status | On hold until the next conversation model is trained, evaluated, and promoted |
| Builds from | Placeholder `conversation` model; update the base and architecture before use |
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
- Preserve `conversation1` as the 2.4B-token diagnostic result.
- Train `0003-conversation` as `conversation2` and compare it with conversation1.
- Train `0004-conversation` as `conversation3` only after the restored baseline
  passes. Its larger architecture cannot reuse the earlier weights.
- Keep tool-use training on hold until a conversation checkpoint is promoted,
  then update and revalidate `holding/tool-use.yaml` against that parent.
- Build the capable dense foundation, assistant, reasoning, and agent rungs.
- Fill the textbook, mathematics, technical, and tool-corpus gaps.
- Implement and validate the small sparse-MoE proof.
- Build the OpenWALDO sparse-MoE lineage.
- Adapt the Nemotron foundation, then run its separate conversation, reasoning,
  and tool post-training stages.

The supporting native-model and backend design is in the
[foundation and sparse-MoE plan](../docs/FOUNDATION-MOE-PLAN.md).
