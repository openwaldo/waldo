// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package index reads and verifies WALDO's Git metadata tree.
package index

import (
	"bytes"
	"encoding/json"
	"fmt"
)

const (
	DirectorySchema      = 1
	ManifestSchema       = 2
	LegacyManifestSchema = 1
)

// SupportedDirectorySchema reports whether WALDO can read a directory index
// schema. New directory indexes use schema 1; schema 2 remains readable for
// compatibility with the existing public JSON index.
func SupportedDirectorySchema(schema int) bool {
	return schema == DirectorySchema || schema == 2
}

// Directory is one index.yaml/index.yml/index.json file. Directory indexes are
// generated navigation data; manifests remain the authority for corpus meaning.
type Directory struct {
	Kind    string  `json:"kind"`
	Schema  int     `json:"schema"`
	Path    string  `json:"path"`
	Entries []Entry `json:"entries"`
}

type Entry struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// Manifest is the schema-1 corpus metadata needed for read compatibility. The
// decoder deliberately permits unknown fields so additive metadata does not
// make an older reader reject an otherwise compatible index.
type Manifest struct {
	Kind         string                `json:"kind"`
	Schema       int                   `json:"schema"`
	Name         string                `json:"name"`
	Title        string                `json:"title"`
	Description  string                `json:"description"`
	Content      *Content              `json:"content,omitempty"`
	License      string                `json:"license,omitempty"`
	Licenses     []string              `json:"licenses,omitempty"`
	Format       string                `json:"format,omitempty"`
	Sources      []Source              `json:"sources"`
	ConvertedBy  Conversion            `json:"converted_by"`
	RecordSchema int                   `json:"record_schema,omitempty"`
	RecordKind   string                `json:"record_kind,omitempty"`
	Assessment   *ContentAssessment    `json:"assessment,omitempty"`
	Redaction    *ContentRedaction     `json:"redaction,omitempty"`
	Processing   *Processing           `json:"processing,omitempty"`
	ComposedBy   *IngestRecipeEvidence `json:"composed_by,omitempty"`
	Shards       []Shard               `json:"-"`
	Rollup       *Rollup               `json:"-"`
}

// ContentAssessment summarizes deterministic row-level flags. Records counts
// are additive; Detector pins the exact classifier used during ingestion.
type ContentAssessment struct {
	EmailAddresses     *DetectionMeasure `json:"email_addresses,omitempty" yaml:"email_addresses,omitempty"`
	RepetitiveContent  *DetectionMeasure `json:"repetitive_content,omitempty" yaml:"repetitive_content,omitempty"`
	BoilerplateContent *DetectionMeasure `json:"boilerplate_content,omitempty" yaml:"boilerplate_content,omitempty"`
}

type DetectionMeasure struct {
	Detector string `json:"detector" yaml:"detector"`
	Records  int64  `json:"records" yaml:"records"`
}

// ContentRedaction records irreversible, deterministic transformations made
// before canonical identity and packing. Names are deliberately retained.
type ContentRedaction struct {
	Policy             string `json:"policy" yaml:"policy"`
	NamesRetained      bool   `json:"names_retained" yaml:"names_retained"`
	EmailAddresses     int64  `json:"email_addresses" yaml:"email_addresses"`
	IPAddresses        int64  `json:"ip_addresses" yaml:"ip_addresses"`
	PhoneNumbers       int64  `json:"phone_numbers" yaml:"phone_numbers"`
	MailRoutingHeaders int64  `json:"mail_routing_headers" yaml:"mail_routing_headers"`
	Credentials        int64  `json:"credentials" yaml:"credentials"`
}

// UnmarshalJSON accepts the schema-1 polymorphic shards field: an inline
// array, or a content-addressed sub-manifest reference.
func (manifest *Manifest) UnmarshalJSON(data []byte) error {
	type plain Manifest
	var wire struct {
		*plain
		Shards json.RawMessage `json:"shards"`
	}
	wire.plain = (*plain)(manifest)
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	manifest.Shards = nil
	manifest.Rollup = nil
	raw := bytes.TrimSpace(wire.Shards)
	if len(raw) == 0 {
		return nil
	}
	switch raw[0] {
	case '[':
		return json.Unmarshal(raw, &manifest.Shards)
	case '{':
		manifest.Rollup = &Rollup{}
		return json.Unmarshal(raw, manifest.Rollup)
	default:
		return fmt.Errorf("manifest %q: shards must be an array or sub-manifest reference", manifest.Name)
	}
}

// MarshalJSON emits the schema-1 polymorphic shards field in the same shape
// accepted by UnmarshalJSON.
func (manifest Manifest) MarshalJSON() ([]byte, error) {
	type plain Manifest
	wire := struct {
		plain
		Shards any `json:"shards"`
	}{plain: plain(manifest)}
	if manifest.Rollup != nil {
		wire.Shards = manifest.Rollup
	} else {
		wire.Shards = manifest.Shards
	}
	return json.Marshal(wire)
}

type Source struct {
	Name            string           `json:"name"`
	Source          string           `json:"source"`
	Version         string           `json:"version,omitempty"`
	InputFormats    []string         `json:"input_formats,omitempty"`
	License         string           `json:"license,omitempty"`
	LicenseEvidence *LicenseEvidence `json:"license_evidence,omitempty"`
	URL             string           `json:"url"`
	Category        string           `json:"category,omitempty"`
	CollectedFrom   string           `json:"collected_from,omitempty"`
	CollectedTo     string           `json:"collected_to,omitempty"`
	SHA256          string           `json:"sha256"`
	Files           []SourceFile     `json:"files,omitempty"`
	Usage           Modalities       `json:"usage,omitempty"`
	Content         *Content         `json:"content,omitempty"`
	Acquisition     *Acquisition     `json:"acquisition,omitempty"`
}

// LicenseEvidence preserves what the upstream said about licensing separately
// from Source.License, which is WALDO's normalized effective/default license.
type LicenseEvidence struct {
	Declaration string `json:"declaration,omitempty" yaml:"declaration,omitempty"`
	URL         string `json:"url,omitempty" yaml:"url,omitempty"`
}

// Modalities contains exact, additive measures keyed by a stable modality
// name such as text, image, audio, or video. A multimodal sample may contribute
// to Samples in more than one entry, so modality sample counts are not summed
// to derive a manifest's logical document count.
type Modalities map[string]ModalityMeasure

type ModalityMeasure struct {
	Samples      int64 `json:"samples,omitempty"`
	Items        int64 `json:"items,omitempty"`
	Tokens       int64 `json:"tokens,omitempty"`
	DurationMS   int64 `json:"duration_ms,omitempty"`
	ContentBytes int64 `json:"content_bytes,omitempty"`
}

// Content describes source material independently of how WALDO acquired or
// encoded it. Tri-state declarations use "yes", "no", or "unknown"; absence
// means the contributor did not make a declaration.
type Content struct {
	Types                []string `json:"types,omitempty" yaml:"types,omitempty"`
	Languages            []string `json:"languages,omitempty" yaml:"languages,omitempty"`
	ProgrammingLanguages []string `json:"programming_languages,omitempty" yaml:"programming_languages,omitempty"`
	Geographies          []string `json:"geographies,omitempty" yaml:"geographies,omitempty"`
	Demographics         []string `json:"demographics,omitempty" yaml:"demographics,omitempty"`
	From                 string   `json:"from,omitempty" yaml:"from,omitempty"`
	To                   string   `json:"to,omitempty" yaml:"to,omitempty"`
	Selection            string   `json:"selection,omitempty" yaml:"selection,omitempty"`
	PersonalData         string   `json:"personal_data,omitempty" yaml:"personal_data,omitempty"`
	Copyrighted          string   `json:"copyrighted,omitempty" yaml:"copyrighted,omitempty"`
	MachineGenerated     string   `json:"machine_generated,omitempty" yaml:"machine_generated,omitempty"`
}

// Acquisition records how an upstream source was obtained. Variant details
// are present only for the applicable source category.
type Acquisition struct {
	Basis     string          `json:"basis,omitempty" yaml:"basis,omitempty"`
	Crawler   *Crawler        `json:"crawler,omitempty" yaml:"crawler,omitempty"`
	UserData  *UserData       `json:"user_data,omitempty" yaml:"user_data,omitempty"`
	Synthetic *SyntheticData  `json:"synthetic,omitempty" yaml:"synthetic,omitempty"`
	Domains   []DomainMeasure `json:"domains,omitempty" yaml:"domains,omitempty"`
}

type Crawler struct {
	Name      string   `json:"name" yaml:"name"`
	Purpose   string   `json:"purpose" yaml:"purpose"`
	Behaviour string   `json:"behaviour" yaml:"behaviour"`
	Protocols []string `json:"protocols,omitempty" yaml:"protocols,omitempty"`
}

type UserData struct {
	Service     string `json:"service" yaml:"service"`
	Interaction string `json:"interaction" yaml:"interaction"`
}

type SyntheticData struct {
	Model       string `json:"model" yaml:"model"`
	Version     string `json:"version,omitempty" yaml:"version,omitempty"`
	SummaryURL  string `json:"summary_url,omitempty" yaml:"summary_url,omitempty"`
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

type DomainMeasure struct {
	Domain        string `json:"domain" yaml:"domain"`
	AcquiredBytes int64  `json:"acquired_bytes,omitempty" yaml:"acquired_bytes,omitempty"`
	RetainedBytes int64  `json:"retained_bytes,omitempty" yaml:"retained_bytes,omitempty"`
}

type Processing struct {
	Steps                     []ProcessingStep `json:"steps,omitempty"`
	RightsReservationMeasures []string         `json:"rights_reservation_measures,omitempty"`
	IllegalContentMeasures    []string         `json:"illegal_content_measures,omitempty"`
}

type ProcessingStep struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

type SourceFile struct {
	Name       string `json:"name"`
	URL        string `json:"url"`
	SHA256     string `json:"sha256"`
	Bytes      int64  `json:"bytes,omitempty"`
	Format     string `json:"format,omitempty"`
	Adapter    string `json:"adapter,omitempty"`
	TextColumn string `json:"text_column,omitempty"`
}

// IngestRecipeEvidence pins an explicitly supplied historical ingest recipe
// and every external command WALDO executed before probing its local artifacts.
// Arguments remain pinned by the recipe hash and are not repeated here.
type IngestRecipeEvidence struct {
	Path       string               `json:"path"`
	SHA256     string               `json:"sha256"`
	Repository string               `json:"repository,omitempty"`
	Commit     string               `json:"commit,omitempty"`
	Dirty      bool                 `json:"dirty"`
	Steps      []RecipeStepEvidence `json:"steps"`
}

type RecipeStepEvidence struct {
	Name       string `json:"name"`
	Executable string `json:"script"`
	SHA256     string `json:"sha256"`
}

type Conversion struct {
	Tool      string `json:"tool"`
	Version   string `json:"version"`
	Commit    string `json:"commit,omitempty"`
	Collector string `json:"collector,omitempty"`
	Profile   string `json:"profile"`
	Recipe    string `json:"recipe"`
	Tokenizer string `json:"tokenizer,omitempty"`
}

type Shard struct {
	URL          string              `json:"url"`
	SHA256       string              `json:"sha256"`
	Format       string              `json:"format,omitempty"`
	License      string              `json:"license,omitempty"`
	Licenses     []string            `json:"licenses,omitempty"`
	LicenseUsage map[string]Measures `json:"license_usage,omitempty"`
	Sources      []string            `json:"sources,omitempty"`
	ConvertedBy  *Conversion         `json:"converted_by,omitempty"`
	Docs         int64               `json:"docs"`
	Tokens       int64               `json:"tokens"`
	Bytes        int64               `json:"bytes"`
	RecordsRoot  string              `json:"records_root,omitempty"`
	Modalities   Modalities          `json:"modalities,omitempty"`
	Assessment   *ContentAssessment  `json:"assessment,omitempty"`
	Redaction    *ContentRedaction   `json:"redaction,omitempty"`
}

// Rollup describes an external submanifest tree. Its aggregate counts are
// enough for offline summaries; object-level verification belongs to the
// network-enabled verification slice.
type Rollup struct {
	URL        string     `json:"url"`
	SHA256     string     `json:"sha256"`
	Count      int64      `json:"count"`
	Docs       int64      `json:"docs"`
	Tokens     int64      `json:"tokens"`
	Bytes      int64      `json:"bytes"`
	Modalities Modalities `json:"modalities,omitempty"`
}

// SubManifest is the content-addressed overflow form of a manifest's shard
// list. Children may nest to any depth.
type SubManifest struct {
	Kind     string   `json:"kind"`
	Schema   int      `json:"schema"`
	Shards   []Shard  `json:"shards,omitempty"`
	Children []Rollup `json:"children,omitempty"`
}

type Corpus struct {
	Path     string
	Manifest Manifest
}

type Measures struct {
	Shards int64 `json:"shards"`
	Docs   int64 `json:"docs"`
	Tokens int64 `json:"tokens"`
	Bytes  int64 `json:"bytes"`
}

// Totals are exact integer aggregates. Human formatting belongs to the CLI.
type Totals struct {
	Corpora int64 `json:"corpora"`
	Measures
	Licenses map[string]Measures `json:"licenses,omitempty"`
}

func (m Manifest) EffectiveLicense(shard Shard) string {
	if shard.License != "" {
		return shard.License
	}
	return m.License
}

func (m Manifest) EffectiveLicenses(shard Shard) []string {
	if len(shard.Licenses) > 0 {
		return append([]string(nil), shard.Licenses...)
	}
	if shard.License != "" {
		return []string{shard.License}
	}
	if len(m.Licenses) > 0 {
		return append([]string(nil), m.Licenses...)
	}
	if m.License != "" {
		return []string{m.License}
	}
	return nil
}
