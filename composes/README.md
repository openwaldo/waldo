# Reference model ladder

Each compose must end at a capability that can be tested before the next model
starts. When architecture and tokenizer are unchanged, the next compose should
initialize from the promoted checkpoint. When either changes, a new weight
lineage begins and must repeat the earlier capability tests.

The four current composes build on lessons from the prior rung, but not its
weights: their architectures differ. The immediate goal is to stop restarting
at `0003` and prove tool tuning directly on the working conversation model.

## Current ladder

| Compose | Builds from | Successful endpoint | Corpus requirements | WALDO requirements |
| --- | --- | --- | --- | --- |
| `0000-canary.yaml` | Random initialization | Training, checkpoint, resume, evaluation, export, and inference complete with verified artifacts | Small raw-text and structured-conversation samples; current selection is sufficient | Current lifecycle and backends are sufficient |
| `0001-babble.yaml` | Random initialization; canary-proven pipeline | Stable short language, improving held-out loss, simple corpus recall, and no repetition collapse | Edited prose, reference text, and scientific exposition; Gutenberg, Wikimedia, and PLOS are sufficient for this pilot | Add a fixed generation test set so promotion is not based only on loss |
| `0002-conversation.yaml` | New architecture; replays the language gate | Direct answers, constraint following, prior-turn context, correction handling, and no tool-call syntax | Natural multi-turn dialogue, broad instruction data, quality-filtered answers, and the bounded Interaction Contract | Change its SFT objectives to `assistant-response-modeling`; add a fixed behavioral evaluation set and regression gate |
| `0003-tool-use.yaml` | Currently another new architecture; should instead be redone from the promoted `0002` checkpoint | Decide whether a tool is needed, select only a provided tool, produce valid arguments, consume results and errors, and retain conversation quality | One normalized call protocol; matched tool/no-tool cases; unavailable-tool, clarification, invalid-argument, empty-result, error, and result-grounding examples | Select corpus categories and fixed evaluation splits; report behavioral metrics; eventually add an inference tool registry and execution loop |

### Redoing tool use from conversation

We should revise `0003` to use the exact architecture, tokenizer, and promoted
checkpoint from `0002`, then train only the tool-specific stage. This isolates
the effect of tool data, avoids repeating foundation and conversation training,
and makes conversation regression measurable.

The existing `0003` architecture has a larger tokenizer and context, so its
weights cannot inherit from `0002`. If 2,048 tokens prove insufficient for the
controlled tool experiment, start a new 4K conversation lineage first and only
add tools after that conversation checkpoint passes. Do not change architecture
and behavior in the same experiment.

## Next full-size lineage

The 30B Nemotron sparse-MoE work starts a new weight lineage, but reuses the
corpus lessons and evaluation gates established above.

| Planned compose | Builds from | Successful endpoint | Corpus requirements | WALDO requirements |
| --- | --- | --- | --- | --- |
| Nemotron foundation adaptation | Pinned Nemotron-3 Nano 30B-A3B Base | Improved WALDO-domain knowledge without unacceptable regression in the original base | Balanced reference prose, science, code, technical documentation, law, education, and math; add open textbooks, stronger math exposition, and Stack V2 Edu | Native checkpoint import, sparse-MoE model facts and forecasting, packed datasets, artifact sets, and a NeMo/Megatron backend |
| Nemotron conversation | Promoted Nemotron foundation checkpoint | Pass the same conversation gate as `0002` at full scale | OASST1/2, natural dialogue, filtered Tulu 3 and UltraChat, HelpSteer2, and bounded Interaction Contract examples | Native Nemotron tokenizer/chat template, assistant-only loss, full SFT or LoRA, and behavioral evaluation |
| Nemotron tool use | Promoted Nemotron conversation checkpoint | Pass the complete tool gate while retaining foundation and conversation quality | Normalized Hermes-style calls plus matched negatives, clarification, tool errors, multi-step results, and ordinary conversation anchors | Nemotron-native tool rendering, selectable tool categories, tool evaluation, adapter lineage, and the inference tool loop |

Later reasoning, preference, safety, and domain models should branch from a
promoted foundation or conversation checkpoint. They should not be mixed into
the first tool-use proof.

The supporting native-model and backend design is in the
[foundation and sparse-MoE plan](../docs/FOUNDATION-MOE-PLAN.md).

## Next steps

- Freeze small, non-training language, conversation, and tool evaluation sets.
- Correct `0002` to use assistant-only SFT, rerun it, and select a promoted
  conversation checkpoint.
- Revise `0003` to initialize from that checkpoint and remove its repeated
  pretraining and general conversation stages.
- Split tool data into selectable decision, call, result, error, clarification,
  and no-tool categories; normalize all retained calls to one protocol.
- Run the revised tool stage and reject it if conversation behavior regresses or
  tool/no-tool selection remains confused.
- Fill the foundation corpus gaps: open textbooks, mathematical exposition and
  verified reasoning seeds, Stack V2 Edu, and measured multilingual coverage.
- Add the native artifact, sparse-MoE, packed-data, and NeMo/Megatron support
  required for the three-stage Nemotron lineage.
