# Compose corpus licensing audit

Status: engineering review completed 2026-09-14. This is not legal advice.

This audit covers every corpus selected by `composes/0000-canary.yaml` through
`0004-conversation.yaml`. A source license, a dataset/database license, and a
license or public-domain status for each contained work are separate facts.
WALDO implements `distribution_policy: distributable` as one simple,
record-level rule: include a document when its effective license is on the
reviewed distributable allowlist, and otherwise skip it. Package and source
rights are recorded as evidence and obligations, not as separate training-gate
facets.

The index preserves the license assertion embedded in each already-published
Parquet shard. Where this audit identifies a corrected `LicenseRef-*`, the
fetcher recipe uses it for the next reingestion; existing objects are not
silently relabeled. Mixed-license corpora remain intact. A distributable compose
selects only rows carrying a reviewed distributable effective license and skips
the others.

## Conclusions

| Index corpus | Pinned upstream and selected files | Rights evidence | Engineering classification |
| --- | --- | --- | --- |
| `core/books/gutenberg` | Project Gutenberg, 2026-07-31 catalog; fetched plain-text ebooks except ID 673; raw tree SHA-256 is in the manifest | [Publisher permission policy](https://www.gutenberg.org/policy/permission.html): most, not all, ebooks are US public domain; thousands are copyrighted and status outside the US differs. Project Gutenberg does not apply CC0. | **Unresolved; legal review.** The current CC0 assertion is not supported. Future ingestion uses `LicenseRef-Project-Gutenberg-Mixed`; require item-level rights filtering before distributable training or corpus publication. |
| `core/common-pile/wikimedia` | Common Pile `wikimedia_filtered` revision `0641bb84bd9b7162bcddf8be7836822161a9a342`; March 2025 official Wikimedia dump JSON gzip files | [Pinned dataset card](https://huggingface.co/datasets/common-pile/wikimedia_filtered/tree/0641bb84bd9b7162bcddf8be7836822161a9a342) and [Wikimedia Terms §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) | **Approved for distributable model training.** Corpus redistribution must preserve page/source attribution, CC-BY-SA-4.0, change indication, and share-alike. |
| `government/regulations` | Common Pile `regulations_filtered` revision `3327364490dfc7929009226ad667eceb2441d93a`; `regulations-0000.json.gz` and `regulations-0001.json.gz` | [Pinned dataset card](https://huggingface.co/datasets/common-pile/regulations_filtered/tree/3327364490dfc7929009226ad667eceb2441d93a) says the aggregation includes agency documents and public comments. [17 U.S.C. §105](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title17-section105) covers US Government works, not third-party submissions. | **Unresolved; legal review.** Future ingestion uses `LicenseRef-US-Federal-Rulemaking-Mixed`; separate government works from comments and other third-party documents. Do not approve the current generic public-domain assertion. |
| `science/plos` | PLOS All of PLOS ZIP acquired under raw tree SHA-256 `71d1ca1e20db3c538587eece5813ce92f27030925126a833e22eebe5fb810cf9`; article XML only | [PLOS open-access policy](https://plos.org/open-science/open-access/) licenses research articles under CC BY; each XML article carries its version, authors, and attribution data. | **Approved for training but not redistribution of the current WALDO shards.** Future ingestion uses `LicenseRef-PLOS-Article-Level-CC-BY` as its fallback and records each article's normalized license; reingest with authors, DOI/source URL, and attribution before publishing canonical corpus artifacts. |
| `post-train/sft/helpsteer2` | NVIDIA HelpSteer2 revision `cece6cab0b0e2851523fd3664220d36a63337b47`; `train.jsonl.gz` | [Pinned publisher card](https://huggingface.co/datasets/nvidia/HelpSteer2/tree/cece6cab0b0e2851523fd3664220d36a63337b47) labels the package CC-BY-4.0 but says most prompts are user-contributed ShareGPT prompts. | **Unresolved; legal review.** Future ingestion uses `LicenseRef-HelpSteer2-ShareGPT-Origin-Unresolved`; establish underlying prompt rights or use the independently authored subset. |
| `post-train/sft/interaction-contract-v1` | OpenWALDO `post-training-data` commit `8d2454f2625a85d911cd7810816e489f0ebb1448`; 18 training JSONL shards | [Pinned repository license](https://github.com/openwaldo/post-training-data/blob/8d2454f2625a85d911cd7810816e489f0ebb1448/LICENSE) | **Approved for distributable model training.** Preserve Apache-2.0 license and NOTICE material for corpus redistribution. |
| `post-train/conversation/schema-guided-dialogue` | Google dataset commit `e852981ae34990f4358979625854259302feaa78`; original SGD train/dev/test dialogue JSON | [Pinned publisher license](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/e852981ae34990f4358979625854259302feaa78/LICENSE.txt) | **Approved for distributable model training.** Attribution, CC-BY-SA-4.0, modification notice, and share-alike apply to redistributed corpus adaptations. |
| `post-train/conversation/ccpe` | Google CCPE commit `2c9cd30f33f3a154b5a27d015333679262ff36f5`; complete `data.json` (502 dialogues) | [Pinned publisher README](https://github.com/google-research-datasets/ccpe/blob/2c9cd30f33f3a154b5a27d015333679262ff36f5/README.md) | **Approved for distributable model training.** Preserve CC-BY-4.0 attribution and modification notice for corpus redistribution. |
| `post-train/sft/oasst1` | OpenAssistant revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`; `2023-04-12_oasst_ready.trees.jsonl.gz` | [Pinned publisher card](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/fdf72ae0827c1cda404aff25b6603abec9e3399b) identifies the crowdsourced content and Apache-2.0 license. | **Approved for distributable model training.** Preserve Apache-2.0 license and notices for corpus redistribution. |
| `post-train/sft/oasst2` | OpenAssistant revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`; `2023-11-05_oasst2_ready.trees.jsonl.gz` | [Pinned publisher card](https://huggingface.co/datasets/OpenAssistant/oasst2/tree/179dd21fc55192153d94adb0e0ce8f69e222bf75) identifies the crowdsourced content and Apache-2.0 license. | **Approved for distributable model training.** Preserve Apache-2.0 license and notices for corpus redistribution. |
| `post-train/sft/dolly` | Databricks revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`; published JSONL | [Pinned publisher card](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a) expressly allows academic and commercial use under CC-BY-SA-3.0 and identifies Wikipedia-derived categories. | **Approved for distributable model training.** Attribute Databricks and Wikipedia contributors; preserve CC-BY-SA-3.0 and share-alike for redistributed corpus adaptations. |
| `post-train/sft/aya` | Cohere For AI revision `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`; `data/*.parquet` | [Pinned publisher card](https://huggingface.co/datasets/CohereForAI/aya_dataset/tree/f9ea04583f02a8f86404ff6c58bf75fe637df8a2) says the human-authored dataset may be used for any academic or commercial purpose under Apache-2.0. | **Approved for distributable model training.** Preserve Apache-2.0 license and notices for corpus redistribution. |
| `core/synthetic/cosmopedia-v2` | Hugging Face SmolLM corpus revision `3ba9d605774198c5868892d7a8deda78031a781f`; all 104 Cosmopedia v2 Parquet shards | [Pinned publisher card](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/3ba9d605774198c5868892d7a8deda78031a781f) applies ODC-BY to the collection and says generated documents use web pages as seed samples. [ODC-BY](https://opendatacommons.org/licenses/by/1-0/) distinguishes database rights from rights in individual contents. | **Unresolved; legal review.** The collection-level ODC-BY assertion is retained as history; future remediation should use `LicenseRef-Cosmopedia-V2-Contents-Unresolved` until seed/content rights and generation terms are verified. |
| `core/common-pile/stackexchange` | Common Pile revision `c0ac7373830c688a43fc12d1988c4b19ccd884ab`; filtered JSON gzip shards | [Publisher licensing timeline](https://stackoverflow.com/help/licensing) applies CC-BY-SA-2.5, 3.0, or 4.0 by contribution date; the [pinned Common Pile card](https://huggingface.co/datasets/common-pile/stackexchange_filtered/tree/c0ac7373830c688a43fc12d1988c4b19ccd884ab) says each record carries its license. | **Approved for training but not redistribution of the current WALDO shards.** Future ingestion starts with `LicenseRef-Stack-Exchange-Mixed-CC-BY-SA`, then records each row's `metadata.license`; retain author/post URLs before publishing corpus artifacts. |
| `code/copyleft/linux-core` | 22 official Git repositories at the exact commits and raw-tree hashes recorded in its manifest | Each source has a pinned official COPYING/LICENSE URL; examples: [Linux](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/COPYING?id=038d61fd642278bab63ee8ef722c50d10ab01e8f) and [GCC](https://github.com/gcc-mirror/gcc/blob/5115c7e447fc07457443df874bf57840e8316d5f/COPYING3). | **Unresolved; legal review for model distribution.** Training is not prohibited by the recorded GPL/LGPL terms, and source/corpus redistribution is possible only while satisfying exact license, source, notice, and exception obligations. WALDO continues to fail closed for model-weight distribution. |
| `code/permissive/linux-core` | 11 official Git repositories at exact commits and raw-tree hashes recorded in its manifest | Pinned official evidence is recorded per source; examples: [musl](https://git.musl-libc.org/cgit/musl/tree/COPYRIGHT?id=0784374d561435f7c787a555aeab8ede699ed298), [OpenBSD policy](https://www.openbsd.org/policy.html), and [NetBSD redistribution](https://www.netbsd.org/about/redistribution.html). | **Approved for training but not redistribution of the current combined shards.** Per-file copyright/license notices and several non-uniform BSD/ISC-style terms must be preserved and audited before corpus publication; custom `LicenseRef-*` values remain fail-closed. |
| `community/linux-kernel-mailing-list` | Official lore.kernel.org public-inbox epochs 15–18; messages dated 2025; exact raw-tree hash in manifest | [Kernel.org lore documentation](https://www.kernel.org/doc/projects/korg/lore.html) documents public archival access but grants no blanket copyright license from message authors. | **Unresolved; legal review.** Public accessibility is not training or redistribution permission; `LicenseRef-Publicly-Archived-Forum` must remain fail-closed. |
| `code/stack-v2-html` | Common Pile `stackv2_html_filtered` revision `92c9fa898d6af643db9c7d73e75219f862be8ca0`; five filtered JSONL gzip shards | [Pinned dataset card](https://huggingface.co/datasets/common-pile/stackv2_html_filtered/tree/92c9fa898d6af643db9c7d73e75219f862be8ca0) and the manifest's per-shard license expressions identify hundreds of SPDX combinations plus unresolved records. | **Unresolved; legal review.** Per-file license/notice obligations and `LicenseRef-Mixed` records prevent a distributable-model approval. |
| `code/cloud-native-core` | containerd `aad11006…`, etcd `5e7fd0de…`, and Prometheus `bb5dff00…`; project-owned files selected by the source-code profile | Pinned official Apache-2.0 files: [containerd](https://github.com/containerd/containerd/blob/aad11006b869517fcd3009450b6f82da282e1a9b/LICENSE), [etcd](https://github.com/etcd-io/etcd/blob/5e7fd0de9a57db03ecc11794dc40403a734c07bb/LICENSE), [Prometheus](https://github.com/prometheus/prometheus/blob/bb5dff00cf8fdfbf5c65e0531aa835fa238a43a2/LICENSE). | **Approved for distributable model training.** Preserve Apache-2.0 license, copyright, and NOTICE material for corpus redistribution. |
| `community/git-mailing-list` | Official lore.kernel.org Git public-inbox epoch 1; messages dated 2025; exact raw-tree hash in manifest | [Kernel.org lore documentation](https://www.kernel.org/doc/projects/korg/lore.html) grants archive access but no blanket author license. | **Unresolved; legal review.** `LicenseRef-Publicly-Archived-Forum` remains fail-closed. |
| `community/python-mailing-lists` | Official monthly mbox exports: python-list and tutor 2015–2025, python-dev 2015–2022; exact hashes per list | [Python.org mailing-list documentation](https://www.python.org/community/lists/) establishes public archives, not a blanket content license. | **Unresolved; legal review.** `LicenseRef-Publicly-Archived-Forum` remains fail-closed. |
| `post-train/sft/tulu3` | Ai2 revision `b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`; all six training Parquet shards | [Pinned publisher card](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/tree/b14afda60f1bbebe55d5d2fa1e4df5042f97f8be) says ODC-BY covers the collection, subsets have different licenses, some are noncommercial, and third-party model outputs have separate terms. | **Approved only for private/research training as currently assembled.** The existing ODC-BY collection assertion is not a content BOM; build a source/license allowlisted derivative and use `LicenseRef-Tulu3-Mixed-Including-Noncommercial` until then. |
| `post-train/sft/smol-smoltalk` | Hugging Face revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`; four training Parquet shards | The [child card](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/tree/f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc) labels the subset Apache-2.0, but the [parent SmolTalk card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) says existing components retain their original licenses and supplies no record-level license BOM for this adapted subset. | **Unresolved; legal review.** The current Apache package assertion remains recorded; use `LicenseRef-Smol-SmolTalk-Mixed-Upstream` for future ingestion unless an exact component-rights BOM is obtained. |
| `post-train/sft/ultrachat-200k` | Hugging Face H4 revision `8049631c405ae6576f93f445c6b8166f76f5505a`; three `train_sft` Parquet shards | [Pinned derivative card](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/8049631c405ae6576f93f445c6b8166f76f5505a) and [original publisher repository](https://github.com/thunlp/UltraChat#data) identify automatically generated dialogue and MIT distribution. | **Approved for distributable model training.** Preserve MIT copyright and license notice for corpus redistribution. |

Every index source retains its full raw-tree SHA-256. Git sources additionally
retain immutable commit IDs, and downloaded datasets retain immutable publisher
revisions where one exists. A mutable publisher artifact such as All of PLOS is
identified by the exact SHA-256 captured at acquisition.

## Obligations

- Apache-2.0: retain the license, copyright notices, attribution notices, and
  any upstream NOTICE material when redistributing corpus artifacts.
- MIT/BSD/ISC-style licenses: retain copyright and license notices with
  redistributed corpus material.
- CC-BY: identify creators and source, link the license, indicate changes, and
  retain any supplied attribution statement.
- CC-BY-SA: meet all CC-BY duties and license redistributed adaptations under
  the applicable same-license/compatible-license terms. WALDO records this as
  a corpus-artifact obligation; it does not silently declare model weights to
  be an adaptation.
- ODC-BY covers database extraction/reuse and requires database attribution;
  it does not erase separate rights in individual contents.
- Public-domain status is jurisdictional and item-specific. It is not CC0 and
  does not remove trademark, privacy, publicity, contract, or database issues.
- Public archival access supplies provenance and availability, not a blanket
  copyright license.

Generated model releases should always ship WALDO's generated training-data
attribution and provenance BOM. That preservation step does not by itself cure
an unresolved input or decide whether a model is a derivative work.

## Required remediation and legal questions

1. Gutenberg: retain and evaluate each ebook's rights header before bounded-text
   stripping; define the release jurisdictions and treatment of copyrighted
   permission titles and the Project Gutenberg trademark.
2. Regulations: identify document submitter/type and partition official federal
   works from public comments and incorporated third-party material.
3. PLOS: map article license URI/version, authors, title, DOI, source URL, and
   supplied attribution into canonical records, then partition homogeneous
   shards and reingest.
4. Stack Exchange: ingest `metadata.license`, author attribution, post URL, and
   modification state; reingest before corpus redistribution.
5. HelpSteer2: obtain the ShareGPT prompt rights chain or select only prompts
   independently created for HelpSteer2.
6. Tulu 3: construct a per-source license BOM and exclude noncommercial or
   separately restricted subsets from any distributable compose.
7. Cosmopedia v2 and Smol-SmolTalk: obtain record-level source/component rights
   and applicable model-output terms rather than relying on collection licenses.
8. Copyleft, BSD-family aggregate, and Stack V2 code: prove preservation of
   per-file notices, source/offer obligations, exceptions, and all unresolved
   license expressions; obtain legal review for released model weights.
9. Public mailing lists: obtain author/list terms or a legal basis for training
   and redistribution. Do not infer permission from archive availability.

Until those items are closed, the four numbered reference composes intentionally
omit `distribution_policy: distributable`; their resulting model artifacts are
private/research artifacts and must not be represented as distributable.
