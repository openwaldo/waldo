# EU GPAI training-content disclosure

Status: implemented mapping and gap-analysis contract; official Word-template
rendering remains pending. This document maps WALDO provenance to the European
Commission's current public-summary template; it is not legal advice and WALDO
must not claim that generated output alone establishes compliance.

## Regulatory target

Article 53(1)(d) of Regulation (EU) 2024/1689 requires providers of
general-purpose AI models to publish a sufficiently detailed summary of the
content used for training according to the AI Office template.

The format audited here is the Commission's English "Template for the Public
Summary of Training Content for General-Purpose AI models," contained in
C(2025) 8311 final of 5 December 2025. The Commission download page was last
updated on 26 March 2026:

- [Official template and explanatory notice](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models)
- [Commission FAQ](https://digital-strategy.ec.europa.eu/en/faqs/template-general-purpose-ai-model-providers-summarise-their-training-content)
- [Article 53](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689)

The template is model-specific and covers content used in pre-training and all
later training stages, including alignment and fine-tuning. It is not merely a
catalog of available corpus objects. Consequently, a corpus OpenWALDO BOM is a
necessary input but a complete export requires the model/run lineage that says
what was actually used.

## Compatibility with the current public index

The current `waldo-index` uses schema 1 for both directory indexes and corpus
manifests.
Its 20 manifests and 1,087 shards verify successfully with `waldo`.

The proposed ingestion design fits the existing structural model:

- one Git manifest defines one corpus;
- one shard entry refers to one content-addressed object;
- `format` and `record_schema` may be declared at manifest level and overridden
  where supported;
- sources, conversion recipe, effective license, document/token/object totals,
  and per-shard source references are already represented; and
- a large shard list may use the existing verified submanifest tree.

Every current source is a third-party publicly available dataset. Every current
object uses the default Parquet format and schema-1 record interpretation. The
observed encoded shard size has a median of approximately 156 MB and a maximum
of approximately 282 MB, so a future 256 MiB target and 512 MiB maximum are
compatible with the current object and manifest structure.

No directory-index schema change is required. The Parquet-only lookaside rule
also requires no compound-object or sidecar extension: a multimodal Parquet
file remains one normal shard URL and SHA-256.

## Missing durable facts

The current manifests are sufficient to identify today's text corpora but not
to populate every data section of the Commission template reliably.

### Modality measures

`docs`, `tokens`, and encoded `bytes` do not express:

- number of images;
- hours of audio or video;
- multimodal composition; or
- uncompressed content size after preprocessing.

New shards and rollups need exact per-modality measures. Existing totals remain
for compatibility: `docs` is the number of logical rows, `tokens` is the named
token-count recipe's text total, and `bytes` is the encoded Parquet object size.

```json
{
  "modalities": {
    "text":  { "samples": 1200, "tokens": 450000, "content_bytes": 1800000 },
    "image": { "samples": 300, "items": 420, "content_bytes": 125000000 },
    "audio": { "samples": 40, "items": 40, "duration_ms": 7200000 },
    "video": { "samples": 12, "items": 12, "duration_ms": 3600000 }
  }
}
```

All measures are non-negative integers. An image-only shard legitimately has
zero tokens. Validators must require positive object bytes and logical rows,
but must not require positive text tokens for a non-text shard.

The same `modalities` aggregate belongs on rollup references so offline index
summaries and OpenWALDO BOM construction do not need to fetch every leaf solely
to calculate disclosure ranges.

### Source classification and usage

The legacy source category `public` is too broad. New contributions use one of:

- `public-dataset`;
- `commercially-licensed`;
- `private-third-party`;
- `web-crawl`;
- `user-data`;
- `synthetic`;
- `other`.

`public` remains a readable legacy alias for `public-dataset`; existing
manifests do not need rewriting.

Each source needs its post-processing usage by modality. This allows the GPAI
exporter to determine which public datasets exceed the template's three-percent
"large dataset" threshold. A shard's current `sources` list cannot apportion
measures when several sources share one shard.

The compact schema-1 text manifests produced by the initial ingest path do not
persist this richer optional evidence. A fail-closed EU GPAI export must report
the resulting gap unless another pinned input supplies it; it must not infer a
legal disclosure fact from `docs`, `tokens`, or compressed object bytes.

When present in a richer manifest, source usage is an aggregate, not a second
object:

```json
{
  "name": "example-dataset",
  "category": "public-dataset",
  "usage": {
    "text": { "samples": 1200, "tokens": 450000 }
  }
}
```

Schema-1 canonical records have one primary source identity. Additional parent
or transformation evidence may be recorded separately, preventing double
counting when source aggregates are reconciled with shard totals.

### Source content and acquisition evidence

Future sources need additive, generally useful provenance fields for:

- modalities and plain-language content types;
- languages, especially EU official languages where known;
- geographical or demographic characteristics when known and relevant;
- start/end dates of the underlying content, distinct from acquisition dates;
- selection approach when only part of a dataset was used;
- whether content is known to include personal, copyright-protected, or
  machine-generated material;
- acquisition start/end dates;
- for direct web collection, crawler identity, purpose and behaviour, including
  respected access controls and protocols;
- for direct web collection, domain-level acquired content bytes and
  post-processing retained content bytes so the required top-domain summary is
  based on content actually used for training;
- for user data, the collecting product/service and interaction type;
- for synthetic data, generator model identity, version and public-summary
  link where available; and
- for commercial/private data, the acquisition basis without publishing
  confidential agreement terms.

Ingestion manifests declare these facts directly on each source. Existing
`collected_from`/`collected_to` are the acquisition period;
`content.from`/`content.to` are the distinct underlying content period, and
`content.selection` preserves the declared subset rule. `content` and
`acquisition` use the same durable structures in ingestion manifests, plans,
index manifests, and OpenWALDO BOMs.

The normalized effective/default source license remains `source.license`.
`source.license_evidence.declaration` preserves upstream wording verbatim and
`source.license_evidence.url` preserves its evidence location. Per-record raw
license values remain in canonical `license_raw`; none of these evidence fields
is silently substituted for the normalized effective license.

Unknown must be distinguishable from false. The wire format should use explicit
states or omit a fact and let the exporter report the gap; it must never infer
`no` from absence.

Large domain summaries are Git metadata. They may be embedded in the corpus
manifest or, if they would make it impractical, in a hash-pinned Git disclosure
file referenced by the manifest. They are not lookaside objects. The manifest
remains the authoritative link and the index verifier must hash any referenced
Git disclosure file.

The fetcher supplies acquired-domain counts. WALDO recomputes retained-domain
counts after filtering and normalization. The exporter uses retained content;
it must not report a domain merely because it was crawled when none of its
content entered the selected corpus.

### Processing evidence

The conversion recipe currently identifies the software but does not describe
the measures needed for Sections 3.1 and 3.2 of the template. Ingestion must
record structured processing declarations containing:

- normalization, filtering, deduplication and selection steps;
- measures used during collection and before training to respect reservations
  of rights from text and data mining;
- opt-out protocols and solutions honoured by WALDO or the upstream acquirer;
  and
- measures used to avoid or remove illegal content.

These are observed or declared process facts, not WALDO's legal conclusion.
They belong in the Git manifest and pass through the OpenWALDO BOM.

## Ownership of disclosure fields

| Template section | Authoritative WALDO source |
| --- | --- |
| Summary version and update date | generated disclosure record |
| 1.1 provider and representative | provider profile supplied at export |
| 1.2 model identity and dependencies | model BOM and lineage |
| EU market-placement date | provider/model release profile |
| 1.3 modalities and sizes | observed model-run inputs plus OpenWALDO BOM modality measures |
| Content types, languages and characteristics | source/manifest provenance aggregated through the BOM |
| Latest acquisition date | source acquisition facts |
| 2.1 public datasets and three-percent threshold | source category and per-source modality usage |
| 2.2 licensed/private datasets | source category, acquisition basis and public-description policy |
| 2.3 direct crawling | fetcher acquisition evidence and domain-size aggregates |
| 2.4 user data | source category and provider service description |
| 2.5 synthetic data | generator model provenance and model-summary link |
| 2.6 other data | source narrative |
| Code of Practice signatory status | provider profile supplied at export |
| TDM reservation measures | acquisition/manifest processing facts plus provider copyright profile |
| Illegal-content measures | manifest processing facts plus provider-level measures |
| Optional processing comments | provider profile and model/run provenance |

Provider identity, EU representative, Code of Practice status, copyright-policy
link, and market-placement date do not belong in corpus manifests because they
describe the provider or a particular model release, not the corpus. The
global `disclosure.provider` profile contains provider-level facts only;
model-specific release facts live with the model.

## OpenWALDO BOM and run requirements

OpenWALDO BOMs must preserve the additive source, modality, acquisition and
processing facts from their pinned manifests. They must not force a later
exporter to reopen a mutable index checkout.

The model run record must distinguish:

- planned corpus selection from observed consumption;
- pre-training, fine-tuning, alignment and other stages;
- unique source content from repeated epochs or sampling exposure;
- consumption by source and modality; and
- base-model lineage from the additional content used for a modification.

For a modified or fine-tuned GPAI model, the Commission template normally
describes the additional training content used for the modification and links
to the original model's summary. Model lineage therefore cannot be reconstructed
from corpus manifests alone.

WALDO reads both from the model's history (`LINEAGE.json`): a model is modified
when its history reaches a pulled model, and the nearest one supplies the
original summary's URL; a model trained and merged entirely here is new. The
report lists the corpora of the model's own runs and of the ancestors trained
on this side of that pull, such as a merge's parents. What the pulled model's
maker trained on stays out, for their summary to describe, though
`ATTRIBUTION.md` still credits it.

## Export command

The implemented audit UX is:

```bash
waldo model bom <model-name-or-path> \
  --format eu-gpai
```

Configure the reusable provider profile first with `waldo config set
disclosure.provider provider.json`. `--provider provider.json` is an explicit
one-command override, not a second home for model-release facts.

With no destination, the converted JSON document is written to standard
output, so ordinary shell redirection works. A final positional destination
writes the same document atomically instead:

```bash
waldo model bom <model-name-or-path> training-content.json \
  --format eu-gpai
```

`model bom --format eu-gpai` derives the disclosure from a model or aggregate
model BOM, not only a corpus OpenWALDO BOM.

The current exporter:

1. pins a supported Commission template version;
2. aggregates all observed training stages and de-duplicates corpus BOMs;
3. preserves sources in the mutually exclusive template categories;
4. carries exact modality and source-usage measures needed for the size ranges
   and public-dataset three-percent calculation;
5. carries retained web-domain aggregates and reports missing crawler/domain
   evidence;
6. merges provider and model-release declarations that cannot belong to corpus
   metadata while keeping their ownership separate;
7. emits a machine-readable, versioned mapping pinned to the official English
   template URL and SHA-256; and
8. reports every required, review, and optional gap before writing the output.

The official English editable artifact at Commission resource `118578` was
verified on 4 August 2026 as SHA-256
`ec803008a5263a485146b24497a3445e2ea32f8b73f818e67652ad70de40f09b`.
The later Word renderer must transform that exact pinned artifact (or a newly
reviewed version), preserve its structure, and emit a separate generation
record containing the rendered output hash. WALDO will not generate a
similar-looking substitute and call it the official template.

Normal export fails closed if a required field is missing. An explicit
`--allow-incomplete` creates a marked machine-readable draft; it does not call
the draft compliant. Review and optional notes remain visible without being
misrepresented as mandatory legal requirements.

The exporter must not hard-code one template forever. Its renderer and field
mapping are versioned because the Commission states that the explanatory notice
and template may be reviewed.

## Schema evolution

Most disclosure fields remain additive. Generated corpus manifests now use
schema 2 for multi-source and mixed-license facts; directory indexes and the
OpenWALDO BOM remain schema 1:

- add modality measures to shards and rollups;
- add source category vocabulary, content/acquisition facts and source usage;
- add structured processing evidence;
- carry those fields unchanged into the unreleased schema-1 OpenWALDO BOM; and
- permit zero tokens for non-text shards while continuing to require positive
  logical rows and encoded bytes.

Old manifests remain readable and produce the same existing totals. New writers
declare `format: "parquet"`, the versioned record schema, the exact writer recipe, and the additive provenance
facts. Unknown additive fields continue to be accepted by older readers.
New canonical Parquet records use record schema 2 while schema-1 shards remain
readable and unassessed. Current schema-2 ingestion applies the mandatory
`waldo/privacy-redaction-v1` policy before canonical identity, retains names,
and records replacement/removal counts in rows, manifests, and BOMs. This is
risk-reduction evidence rather than a claim of anonymity or GDPR compliance.
Schema 2 records also carry versioned, immutable row facts
for detected email-shaped strings, token repetition, and structural boilerplate;
the flags do not themselves establish a legal basis, personal-data status, or
overall content quality. Detector identities and aggregate counts survive in
manifests and OpenWALDO BOMs. OpenWALDO BOMs remain schema 1. Ingestion
manifests support a single source at the root or explicitly bounded grouped
sources beneath it.

The remaining disclosure fixtures should cover:

- one current legacy public text manifest unchanged;
- one new mixed text/image manifest with zero text tokens in an image shard;
- one multi-source manifest whose source usage reconciles exactly;
- one direct-web source with crawler and domain evidence;
- one synthetic source with generator lineage; and
- one model with pre-training plus fine-tuning whose exported summary includes
  only the appropriate content for each model lineage case.
