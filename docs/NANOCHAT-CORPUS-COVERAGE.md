# Nanochat corpus coverage and license gate

This inventory compares WALDO with nanochat commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`. It separates training inputs from
evaluation inputs and applies a conservative distribution gate.

This is an engineering review, not legal advice. A dataset is admitted only
when its pinned source includes terms that allow WALDO to redistribute the
normalized shards and release the intended model without a noncommercial or
field-of-use restriction. Attribution and share-alike duties must be carried
into the corpus BOM and release materials. A repository-level license tag is
not enough when a dataset aggregates independently licensed sources.

## Training inputs

| Nanochat input | Use | Declared terms | WALDO index | Decision |
| --- | --- | --- | --- | --- |
| `nvidia/Nemotron-ClimbMix` via `karpathy/climbmix-400b-shuffle` | tokenizer and pretraining | CC-BY-NC-4.0; research/development only | absent | **Reject from the distributable default.** Do not ingest or disguise the repack as MIT. It may only be considered in a separately isolated, noncommercial benchmark after explicit approval. |
| `HuggingFaceTB/smol-smoltalk`, `train` | SFT | Apache-2.0 | present at `post-train/sft/smol-smoltalk`; 460,222 records | **Admit.** Keep the pinned revision and existing license evidence. |
| `cais/mmlu`, `auxiliary_train`, repeated 3x | SFT | repository declares MIT, but the split aggregates ARC, MC_TEST, OpenBookQA, RACE, and other material | absent | **Hold.** Audit every constituent and transformation before planning ingestion. |
| `openai/gsm8k`, `main/train`, repeated 4x | SFT | MIT | absent | **Admit for ingestion.** Pin an upstream commit, preserve notices, and exclude `test` from training. |

The exact ClimbMix pretraining match is therefore intentionally impossible for
the default distributable WALDO lineage. WALDO should compare training systems
and capability per FLOP using a license-approved mixture, not import a
noncommercial corpus merely to match a benchmark.

## Evaluation-only inputs

| Nanochat evaluation | Declared terms | WALDO index | Decision |
| --- | --- | --- | --- |
| Smol-SmolTalk `test` | Apache-2.0 | excluded from the training entry | Create a separate immutable evaluation BOM; never add it to a training corpus. |
| MMLU `test` | MIT at repository level | absent | Hold for the same constituent audit; evaluation BOM only if approved. |
| GSM8K `main/test` | MIT | absent | Add as an evaluation BOM, separate from `main/train`. |
| AI2 ARC Easy/Challenge | CC-BY-SA-4.0 | absent | Admissible with attribution/share-alike handling; evaluation BOM only. |
| HumanEval | MIT | absent | Admissible; evaluation BOM only and execute tests in a sandbox. |
| Nanochat CORE bundle | no complete, pinned upstream license BOM is supplied with the download | absent | Do not redistribute or ingest the bundle. Reconstruct a comparable suite from individually pinned, approved upstream datasets. |

Evaluation records do not belong in `waldo-index` training paths. They need a
separate evaluation-BOM namespace/contract so contamination checks can prove
that test rows were excluded from training.

## Approved open pretraining baseline already indexed

The holding compose uses only indexed corpora with redistribution-capable
declared terms:

| WALDO path | Declared terms | Role |
| --- | --- | --- |
| `science/plos` | CC-BY-4.0 | scientific exposition |
| `core/synthetic/cosmopedia-v2` | ODC-BY-1.0 | educational synthetic text |
| `core/common-pile/stackexchange` | CC-BY-SA-4.0 | technical questions and answers |

These corpora overlap ClimbMix's goals, not its exact rows, filtering, topic
weights, or quality classifiers. The baseline must be described as
WALDO-open, never as a ClimbMix reproduction.

The holding compose retains `core/books/gutenberg` and
`core/common-pile/wikimedia` so the intended mixture is not silently narrowed.
The strict gate currently rejects both index entries. Gutenberg is labeled
corpus-wide `CC0-1.0`, but Project Gutenberg's actual terms distinguish U.S.
public-domain works, permission-only works, and trademark/license material;
its manifest carries no per-work rights proof. Wikimedia carries a
`CC-BY-SA-4.0` label but no pinned source version or license evidence, and its
upstream dataset warns that license metadata can be incorrect. The compose
cannot run under `distribution_policy: distributable` until that evidence and
the attribution path are repaired or explicitly resolved by review.

## Index completion work

1. Add GSM8K train through the normal fetcher/manifest/ingest flow, with a
   pinned commit, MIT evidence, and a training-only split policy.
2. Add separate GSM8K test, ARC, HumanEval, and Smol-SmolTalk test evaluation
   BOMs after the evaluation contract exists.
3. Complete a constituent-level MMLU audit. Add only splits whose complete
   provenance chain passes the redistribution gate.
4. Record attribution/share-alike obligations in generated release notices.
5. Add an automated policy check that rejects missing, unknown,
   noncommercial, no-redistribution, or unreviewed aggregate licenses from a
   distributable model BOM.
6. Run exact and fuzzy contamination checks between admitted training and
   evaluation BOMs before training.

Corpus parity is complete when every admitted nanochat input is pinned and
indexed, every rejected input has a documented replacement, and the resolved
compose and evaluation BOM pass the automated license and contamination gates.

## Evidence reviewed

- [nanochat speedrun at the pinned commit](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/speedrun.sh)
- [nanochat SFT mixture at the pinned commit](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_sft.py)
- [NVIDIA ClimbMix dataset card](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)
- [Smol-SmolTalk dataset card](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
- [MMLU dataset card](https://huggingface.co/datasets/cais/mmlu)
- [GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k)
- [AI2 ARC dataset card](https://huggingface.co/datasets/allenai/ai2_arc)
- [HumanEval dataset card](https://huggingface.co/datasets/openai/openai_humaneval)
