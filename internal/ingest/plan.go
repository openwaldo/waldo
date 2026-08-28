// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/record"
	"github.com/openwaldo/waldo/internal/shard"
)

type Plan struct {
	Kind           string                      `json:"kind"`
	Schema         int                         `json:"schema"`
	Destination    string                      `json:"destination"`
	Title          string                      `json:"title"`
	Description    string                      `json:"description"`
	License        string                      `json:"license"`
	Source         PlanSource                  `json:"source"`
	Sources        []PlanSource                `json:"sources,omitempty"`
	Mode           string                      `json:"mode"`
	MemoryBytes    int64                       `json:"memory_bytes"`
	Writer         WriterPlan                  `json:"writer"`
	Inputs         []PlanInput                 `json:"inputs"`
	TextFallbacks  []TextFallback              `json:"text_fallbacks,omitempty"`
	RecipeEvidence *index.IngestRecipeEvidence `json:"ingest_recipe,omitempty"`
	Update         *UpdatePlan                 `json:"update,omitempty"`
}

type TextFallback struct {
	DetectedFormat string `json:"detected_format"`
	Adapter        string `json:"adapter"`
	Artifacts      int64  `json:"artifacts"`
	Bytes          int64  `json:"bytes"`
}

type UpdatePlan struct {
	Manifest       string `json:"manifest"`
	ManifestSHA256 string `json:"manifest_sha256"`
	Mode           string `json:"mode"`
}

type PlanSource struct {
	ID              string                 `json:"id,omitempty"`
	Name            string                 `json:"name"`
	License         string                 `json:"license,omitempty"`
	Version         string                 `json:"version,omitempty"`
	InputFormats    []string               `json:"input_formats,omitempty"`
	URL             string                 `json:"url"`
	Category        string                 `json:"category"`
	CollectedFrom   string                 `json:"collected_from,omitempty"`
	CollectedTo     string                 `json:"collected_to,omitempty"`
	LicenseEvidence *index.LicenseEvidence `json:"license_evidence,omitempty"`
	Content         *index.Content         `json:"content,omitempty"`
	Acquisition     *index.Acquisition     `json:"acquisition,omitempty"`
}

type WriterPlan struct {
	Format               string `json:"format"`
	RecordKind           string `json:"record_kind"`
	RecordSchema         int    `json:"record_schema"`
	Recipe               string `json:"recipe"`
	AdapterRecipe        string `json:"adapter_recipe"`
	CompressedTarget     int64  `json:"compressed_target_bytes"`
	CompressedMaximum    int64  `json:"compressed_maximum_bytes"`
	RowGroupLogicalBytes int64  `json:"row_group_logical_bytes"`
	PageBytes            int64  `json:"page_bytes"`
	AdapterBatchBytes    int64  `json:"adapter_batch_bytes"`
	RecordMaximumBytes   int64  `json:"record_maximum_bytes"`
	Compression          string `json:"compression"`
}

type PlanInput struct {
	Artifact       Artifact     `json:"artifact"`
	Adapter        string       `json:"adapter"`
	DetectedFormat string       `json:"detected_format,omitempty"`
	TextColumn     string       `json:"text_column,omitempty"`
	SourcePath     string       `json:"source_path,omitempty"`
	SourceID       string       `json:"source_id,omitempty"`
	Profile        InputProfile `json:"profile,omitempty"`
}

type PlanRequest struct {
	Destination        string
	Title              string
	Description        string
	License            string
	Source             PlanSource
	Mode               string
	MemoryBytes        int64
	TextColumn         string
	ForceFormat        string
	RecordMaximumBytes int64
	Profile            InputProfile
	InputRoot          string
	Sources            []PlanSourceRequest
	RecipeEvidence     *index.IngestRecipeEvidence
	Update             *UpdatePlan
}

type PlanSourceRequest struct {
	ID                 string
	License            string
	Source             PlanSource
	InputRoot          string
	TextColumn         string
	RecordMaximumBytes int64
	Profile            InputProfile
}

func NewPlan(probe Probe, request PlanRequest) (Plan, error) {
	if probe.Kind != "waldo-ingest-probe" || probe.Schema != 1 || len(probe.Artifacts) == 0 {
		return Plan{}, fmt.Errorf("invalid or empty ingestion probe")
	}
	if strings.TrimSpace(request.Destination) == "" || strings.TrimSpace(request.Title) == "" {
		return Plan{}, fmt.Errorf("destination and title are required")
	}
	if request.ForceFormat != "" && !slices.Contains([]string{"text", "markdown", "mbox", "json", "jsonl", "parquet", "xml"}, request.ForceFormat) {
		return Plan{}, fmt.Errorf("unsupported --force-format %q; use text, markdown, mbox, json, jsonl, parquet, or xml", request.ForceFormat)
	}
	if len(request.Sources) == 0 {
		if strings.TrimSpace(request.License) == "" {
			return Plan{}, fmt.Errorf("license is required")
		}
		if request.Profile.Format != "" {
			request.Source.InputFormats = []string{request.Profile.Format}
		}
		if err := normalizePlanSource(&request.Source); err != nil {
			return Plan{}, err
		}
		if request.TextColumn != "" && request.Profile.Type != "" {
			return Plan{}, fmt.Errorf("text column and input profile cannot both be set")
		}
	} else {
		seen := map[string]bool{}
		for position := range request.Sources {
			source := &request.Sources[position]
			if !recipeStepName.MatchString(source.ID) || seen[source.ID] {
				return Plan{}, fmt.Errorf("source %d has invalid or duplicate id %q", position+1, source.ID)
			}
			seen[source.ID] = true
			source.Source.ID = source.ID
			source.Source.License = record.NormalizeLicense(source.License)
			if source.Profile.Format != "" {
				source.Source.InputFormats = []string{source.Profile.Format}
			}
			if source.Source.License == "" {
				return Plan{}, fmt.Errorf("source %q license is required", source.ID)
			}
			if err := normalizePlanSource(&source.Source); err != nil {
				return Plan{}, fmt.Errorf("source %q: %w", source.ID, err)
			}
			if source.TextColumn != "" && source.Profile.Type != "" {
				return Plan{}, fmt.Errorf("source %q text column and input profile cannot both be set", source.ID)
			}
		}
	}
	mode := request.Mode
	if mode == "" {
		mode = "streaming"
	}
	if mode != "streaming" && mode != "canonical" {
		return Plan{}, fmt.Errorf("ingestion mode must be streaming or canonical")
	}
	memory := request.MemoryBytes
	if memory == 0 {
		memory = 2 << 30
	}
	if memory < 256<<20 {
		return Plan{}, fmt.Errorf("ingestion memory budget must be at least 256 MiB")
	}
	plan := Plan{
		Kind: "waldo-ingest-plan", Schema: 1,
		Destination: request.Destination, Title: request.Title, Description: request.Description, License: record.NormalizeLicense(request.License),
		Source: request.Source, Mode: mode, MemoryBytes: memory,
		RecipeEvidence: request.RecipeEvidence,
		Update:         request.Update,
		Writer: WriterPlan{
			Format: "parquet", RecordKind: record.KindPretrain, RecordSchema: shard.TextRecordSchema, Recipe: shard.TextWriterRecipe,
			AdapterRecipe:    "canonical-adapters-2",
			CompressedTarget: 256 << 20, CompressedMaximum: 512 << 20,
			RowGroupLogicalBytes: 64 << 20, PageBytes: 1 << 20,
			AdapterBatchBytes: 16 << 20, RecordMaximumBytes: 64 << 20,
			Compression: "zstd-level-6",
		},
	}
	for _, source := range request.Sources {
		plan.Sources = append(plan.Sources, source.Source)
		if source.RecordMaximumBytes > plan.Writer.RecordMaximumBytes {
			plan.Writer.RecordMaximumBytes = source.RecordMaximumBytes
		}
	}
	if request.RecordMaximumBytes != 0 {
		plan.Writer.RecordMaximumBytes = request.RecordMaximumBytes
	}
	if plan.Description == "" {
		if len(plan.Sources) == 0 {
			plan.Description = "Training corpus acquired from " + request.Source.Name + "."
		} else {
			plan.Description = fmt.Sprintf("Training corpus acquired from %d sources.", len(plan.Sources))
		}
	}
	for _, artifact := range probe.Artifacts {
		// Acquisition recipes may select empty tracked files. They carry no
		// trainable content, so omit them before adapter selection. Direct input
		// remains strict and reports the unsupported empty artifact.
		if artifact.Format == "empty" && request.RecipeEvidence != nil {
			continue
		}
		profile, textColumn, inputRoot, sourceID := request.Profile, request.TextColumn, request.InputRoot, ""
		sourceCode := contentIncludesSourceCode(request.Source.Content)
		if len(request.Sources) > 0 {
			source, err := sourceRequestForArtifact(request.Sources, artifact.Path)
			if err != nil {
				return Plan{}, err
			}
			profile, textColumn, inputRoot, sourceID = source.Profile, source.TextColumn, source.InputRoot, source.ID
			sourceCode = contentIncludesSourceCode(source.Source.Content)
		}
		if request.ForceFormat == "" && profile.Format != "" {
			if !declaredFormatMatches(profile.Format, artifact, sourceCode) {
				return Plan{}, fmt.Errorf("%s: manifest declares input format %q but WALDO detected %q; correct the fetcher INI and refetch, or correct the raw data", artifact.Path, profile.Format, artifact.Format)
			}
			if artifact.Format != profile.Format {
				detected := artifact.Format
				artifact.Format = profile.Format
				artifact.Evidence = append(artifact.Evidence, "manifest-format:"+detected+"->"+profile.Format)
			}
		} else if request.ForceFormat == "" && sourceCode && sourceCodeTextFormat(artifact) {
			artifact.Format = "text"
			artifact.Evidence = append(artifact.Evidence, "source-code-context")
		}
		detectedFormat := ""
		if request.ForceFormat != "" {
			detectedFormat = artifact.Format
			artifact.Format = request.ForceFormat
			artifact.Evidence = append(artifact.Evidence, "forced-format:"+detectedFormat+"->"+request.ForceFormat)
		}
		profile = profile.withDefaults()
		input := PlanInput{Artifact: artifact, Profile: profile, SourceID: sourceID, DetectedFormat: detectedFormat}
		if inputRoot != "" {
			root, err := filepath.Abs(inputRoot)
			if err != nil {
				return Plan{}, err
			}
			relative, err := filepath.Rel(root, artifact.Path)
			if err != nil || relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
				return Plan{}, fmt.Errorf("input %s is outside recipe output %s", artifact.Path, root)
			}
			input.SourcePath = filepath.ToSlash(relative)
		}
		if err := input.Profile.Validate(); err != nil {
			return Plan{}, fmt.Errorf("%s: %w", artifact.Path, err)
		}
		if input.Profile.recordProfile() {
			if input.Profile.Type == ProfileRankedConversationTree && artifact.Format == "parquet" {
				return Plan{}, fmt.Errorf("%s: ranked-conversation-tree requires JSON or JSONL", artifact.Path)
			}
			switch artifact.Format {
			case "json", "jsonl", "parquet":
				input.Adapter = artifact.Format
			default:
				return Plan{}, fmt.Errorf("%s: profile %s requires JSON, JSONL, or Parquet, not %q", artifact.Path, input.Profile.Type, artifact.Format)
			}
			plan.Inputs = append(plan.Inputs, input)
			continue
		}
		switch input.Profile.Type {
		case ProfileBoundedText:
			if artifact.Format != "text" {
				return Plan{}, fmt.Errorf("%s: bounded-text requires text input, not %q", artifact.Path, artifact.Format)
			}
			input.Adapter = ProfileBoundedText
		case ProfileXMLRecord:
			if artifact.Format != "xml" {
				return Plan{}, fmt.Errorf("%s: xml-record requires XML input, not %q", artifact.Path, artifact.Format)
			}
			input.Adapter = ProfileXMLRecord
		default:
			switch artifact.Format {
		case "text", "markdown", "mbox", "latex":
			input.Adapter = artifact.Format
			case "parquet":
				if textColumn == "" {
					return Plan{}, fmt.Errorf("%s: Parquet input requires a record input profile or an explicit text column; use a manifest [input] mapping or --input-profile/--text-column", artifact.Path)
				}
				column, err := chooseTextColumn(artifact, textColumn)
				if err != nil {
					return Plan{}, fmt.Errorf("%s: %w", artifact.Path, err)
				}
				input.Adapter = "parquet"
				input.TextColumn = column
			case "json", "jsonl":
				return Plan{}, fmt.Errorf("%s: %s input requires a record input profile; use a manifest [input] mapping or --input-profile, or deliberately override it with --force-format text", artifact.Path, strings.ToUpper(artifact.Format))
			case "xml":
				return Plan{}, fmt.Errorf("%s: XML input requires an xml-record input profile; use a manifest [input] mapping or --input-profile, or deliberately override it with --force-format text", artifact.Path)
			default:
				return Plan{}, fmt.Errorf("%s: unsupported raw format %q; add a general ingestion adapter or deliberately select an existing one with --force-format", artifact.Path, artifact.Format)
			}
		}
		plan.Inputs = append(plan.Inputs, input)
	}
	if len(request.Sources) > 0 {
		planned := map[string]int{}
		for _, input := range plan.Inputs {
			planned[input.SourceID]++
		}
		for _, source := range request.Sources {
			if planned[source.ID] == 0 {
				return Plan{}, fmt.Errorf("source %q produced no non-empty supported inputs", source.ID)
			}
		}
	}
	logicalKind := ""
	for _, input := range plan.Inputs {
		kind := record.KindPretrain
		if input.Profile.Type == ProfileDialoguePair || input.Profile.Type == ProfileChatMessages || input.Profile.Type == ProfileRankedConversationTree {
			kind = record.KindConversation
		}
		if logicalKind != "" && logicalKind != kind {
			return Plan{}, fmt.Errorf("one ingestion plan cannot mix %s and %s logical records", logicalKind, kind)
		}
		logicalKind = kind
		if kind == record.KindConversation {
			plan.Writer.RecordKind = record.KindConversation
			plan.Writer.RecordSchema = shard.ConversationRecordSchema
			plan.Writer.Recipe = shard.ConversationWriterRecipe
			plan.Writer.AdapterRecipe = "canonical-conversation-adapters-1"
		}
	}
	if err := plan.Validate(); err != nil {
		return Plan{}, err
	}
	return plan, nil
}

func sourceCodeTextFormat(artifact Artifact) bool {
	if artifact.Compression != "" {
		return false
	}
	return slices.Contains([]string{"text", "markdown", "json", "jsonl", "html", "xml", "warc", "mbox"}, artifact.Format)
}

func declaredFormatMatches(expected string, artifact Artifact, sourceCode bool) bool {
	if expected == artifact.Format {
		return true
	}
	if artifact.Compression != "" {
		return false
	}
	switch expected {
	case "text":
		return artifact.Format == "markdown" || sourceCode && sourceCodeTextFormat(artifact)
	case "markdown":
		return artifact.Format == "text" || artifact.Format == "markdown"
	default:
		return false
	}
}

func recordTextFallback(plan *Plan, format, adapter string, bytes int64) {
	for position := range plan.TextFallbacks {
		if plan.TextFallbacks[position].DetectedFormat == format && plan.TextFallbacks[position].Adapter == adapter {
			plan.TextFallbacks[position].Artifacts++
			plan.TextFallbacks[position].Bytes += bytes
			return
		}
	}
	plan.TextFallbacks = append(plan.TextFallbacks, TextFallback{DetectedFormat: format, Adapter: adapter, Artifacts: 1, Bytes: bytes})
}

func contentIncludesSourceCode(content *index.Content) bool {
	if content == nil {
		return false
	}
	for _, contentType := range content.Types {
		if strings.EqualFold(strings.TrimSpace(contentType), "source code") {
			return true
		}
	}
	return false
}

func normalizePlanSource(source *PlanSource) error {
	if source.Name == "" || source.URL == "" || source.Category == "" {
		return fmt.Errorf("source name, URL, and category are required")
	}
	category, ok := index.CanonicalSourceCategory(source.Category)
	if !ok {
		return fmt.Errorf("unsupported source category %q", source.Category)
	}
	source.Category = category
	return index.ValidateSourceProvenance(index.Source{
		Category: category, CollectedFrom: source.CollectedFrom, CollectedTo: source.CollectedTo,
		InputFormats: source.InputFormats, LicenseEvidence: source.LicenseEvidence, Content: source.Content, Acquisition: source.Acquisition,
	})
}

func sourceRequestForArtifact(sources []PlanSourceRequest, artifactPath string) (PlanSourceRequest, error) {
	for _, source := range sources {
		root, err := filepath.Abs(source.InputRoot)
		if err != nil {
			return PlanSourceRequest{}, err
		}
		relative, err := filepath.Rel(root, artifactPath)
		if err == nil && relative != "." && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			return source, nil
		}
	}
	return PlanSourceRequest{}, fmt.Errorf("input %s is outside every declared source", artifactPath)
}

func (plan Plan) sourceFor(input PlanInput) (PlanSource, string, error) {
	if input.SourceID == "" {
		return plan.Source, plan.License, nil
	}
	for _, source := range plan.Sources {
		if source.ID == input.SourceID {
			return source, source.License, nil
		}
	}
	return PlanSource{}, "", fmt.Errorf("input %s references unknown source %q", input.Artifact.Path, input.SourceID)
}

func chooseTextColumn(artifact Artifact, requested string) (string, error) {
	if artifact.Parquet == nil {
		return "", fmt.Errorf("Parquet footer information is missing")
	}
	if requested != "" {
		if slices.Contains(artifact.Parquet.Columns, requested) {
			if strings.Contains(requested, ".") {
				return "", fmt.Errorf("nested text column %q is not enabled yet", requested)
			}
			return requested, nil
		}
		return "", fmt.Errorf("requested text column %q is absent", requested)
	}
	var candidates []string
	for _, candidate := range []string{"text", "content", "document"} {
		if slices.Contains(artifact.Parquet.Columns, candidate) {
			candidates = append(candidates, candidate)
		}
	}
	if len(candidates) == 1 {
		return candidates[0], nil
	}
	if len(candidates) == 0 {
		return "", fmt.Errorf("cannot infer a text column; columns are %s", strings.Join(artifact.Parquet.Columns, ", "))
	}
	return "", fmt.Errorf("text column is ambiguous (%s); specify it explicitly", strings.Join(candidates, ", "))
}

func (plan Plan) Validate() error {
	textWriter := plan.Writer.RecordKind == record.KindPretrain && plan.Writer.RecordSchema == shard.TextRecordSchema && plan.Writer.Recipe == shard.TextWriterRecipe && plan.Writer.AdapterRecipe == "canonical-adapters-2"
	conversationWriter := plan.Writer.RecordKind == record.KindConversation && plan.Writer.RecordSchema == shard.ConversationRecordSchema && plan.Writer.Recipe == shard.ConversationWriterRecipe && plan.Writer.AdapterRecipe == "canonical-conversation-adapters-1"
	if plan.Kind != "waldo-ingest-plan" || plan.Schema != 1 || plan.Writer.Format != "parquet" || (!textWriter && !conversationWriter) {
		return fmt.Errorf("unsupported ingestion plan identity or writer")
	}
	cleanDestination := filepath.ToSlash(filepath.Clean(plan.Destination))
	if plan.Destination == "" || plan.Destination == "." || filepath.IsAbs(plan.Destination) || strings.HasPrefix(cleanDestination, "..") || plan.Destination != cleanDestination {
		return fmt.Errorf("destination must be a relative index path")
	}
	if plan.Title == "" || plan.Description == "" {
		return fmt.Errorf("ingestion plan is missing corpus or source identity")
	}
	if len(plan.Sources) == 0 {
		if plan.License == "" || plan.Source.Name == "" || plan.Source.URL == "" || plan.Source.Category == "" {
			return fmt.Errorf("ingestion plan is missing corpus or source identity")
		}
		if err := validatePlanSource(plan.Source); err != nil {
			return fmt.Errorf("ingestion plan source: %w", err)
		}
	} else {
		seen := map[string]bool{}
		for _, source := range plan.Sources {
			if source.ID == "" || seen[source.ID] || source.Name == "" || source.License == "" || source.URL == "" || source.Category == "" {
				return fmt.Errorf("ingestion plan has an invalid multi-source identity")
			}
			seen[source.ID] = true
			if err := validatePlanSource(source); err != nil {
				return fmt.Errorf("ingestion plan source %q: %w", source.ID, err)
			}
		}
	}
	if plan.Mode != "streaming" && plan.Mode != "canonical" {
		return fmt.Errorf("unsupported ingestion mode %q", plan.Mode)
	}
	if plan.Update != nil {
		cleanManifest := filepath.ToSlash(filepath.Clean(filepath.FromSlash(plan.Update.Manifest)))
		if cleanManifest == "." || cleanManifest != plan.Update.Manifest || filepath.IsAbs(filepath.FromSlash(plan.Update.Manifest)) || strings.HasPrefix(cleanManifest, "../") || !validSHA256(plan.Update.ManifestSHA256) || plan.Update.Mode != "rebuild-shards" {
			return fmt.Errorf("ingestion update has invalid manifest identity or mode")
		}
	}
	if plan.MemoryBytes < 256<<20 || plan.Writer.CompressedTarget <= 0 || plan.Writer.CompressedMaximum < plan.Writer.CompressedTarget || plan.Writer.RowGroupLogicalBytes <= 0 || plan.Writer.PageBytes <= 0 || plan.Writer.AdapterBatchBytes <= 0 || plan.Writer.RecordMaximumBytes < plan.Writer.AdapterBatchBytes || plan.Writer.RecordMaximumBytes > plan.MemoryBytes/2 {
		return fmt.Errorf("ingestion plan has invalid resource or writer limits")
	}
	for _, fallback := range plan.TextFallbacks {
		if fallback.DetectedFormat == "" || (fallback.Adapter != "text" && fallback.Adapter != "opaque-base64") || fallback.Artifacts <= 0 || fallback.Bytes < 0 {
			return fmt.Errorf("ingestion plan has invalid text fallback counts")
		}
	}
	previous := ""
	for _, input := range plan.Inputs {
		artifact := input.Artifact
		if !filepath.IsAbs(artifact.Path) || artifact.Path <= previous || artifact.Bytes < 0 || !validSHA256(artifact.SHA256) {
			return fmt.Errorf("plan inputs must have sorted absolute paths, sizes, and hashes")
		}
		if input.SourcePath != "" {
			cleanSource := filepath.ToSlash(filepath.Clean(filepath.FromSlash(input.SourcePath)))
			if cleanSource == "." || cleanSource != input.SourcePath || strings.HasPrefix(cleanSource, "../") || filepath.IsAbs(filepath.FromSlash(input.SourcePath)) {
				return fmt.Errorf("input %s has invalid source path %q", artifact.Path, input.SourcePath)
			}
		}
		if _, license, err := plan.sourceFor(input); err != nil || license == "" {
			return fmt.Errorf("input %s has invalid source assignment", artifact.Path)
		}
		if input.DetectedFormat != "" && !slices.Contains([]string{"empty", "text", "markdown", "mbox", "json", "jsonl", "parquet", "xml", "html", "warc", "compressed", "unknown"}, input.DetectedFormat) {
			return fmt.Errorf("input %s has invalid overridden detected format %q", artifact.Path, input.DetectedFormat)
		}
		if err := input.Profile.Validate(); err != nil {
			return fmt.Errorf("input %s: %w", artifact.Path, err)
		}
		if input.Profile.recordProfile() {
			if input.Adapter != artifact.Format || (input.Adapter != "json" && input.Adapter != "jsonl" && input.Adapter != "parquet") || input.TextColumn != "" {
				return fmt.Errorf("input %s has an inconsistent record-profile adapter", artifact.Path)
			}
			previous = artifact.Path
			continue
		}
		switch input.Adapter {
		case "text", "markdown":
			if artifact.Format != input.Adapter || input.TextColumn != "" {
				return fmt.Errorf("input %s has an inconsistent text adapter", artifact.Path)
			}
		case "mbox":
			if artifact.Format != "mbox" || input.TextColumn != "" || (artifact.Compression != "" && artifact.Compression != "gzip" && artifact.Compression != "zstd") {
				return fmt.Errorf("input %s has an inconsistent mbox adapter", artifact.Path)
			}
		case "latex":
			if artifact.Format != "latex" {
					return fmt.Errorf("input %s has an inconsistent latex adapter", artifact.Path)
			}
		case "parquet":
			if artifact.Format != "parquet" || input.TextColumn == "" {
				return fmt.Errorf("Parquet input %s has no valid text column mapping", artifact.Path)
			}
		case "jsonl":
			if artifact.Format != "jsonl" || input.TextColumn != "" || (artifact.Compression != "" && artifact.Compression != "gzip" && artifact.Compression != "zstd") {
				return fmt.Errorf("input %s has an inconsistent JSONL adapter", artifact.Path)
			}
		case "opaque-base64":
			if input.TextColumn != "" {
				return fmt.Errorf("input %s has an inconsistent opaque adapter", artifact.Path)
			}
		case ProfileBoundedText:
			if artifact.Format != "text" || input.Profile.Type != ProfileBoundedText {
				return fmt.Errorf("input %s has an inconsistent bounded-text adapter", artifact.Path)
			}
		case ProfileXMLRecord:
			if artifact.Format != "xml" || input.Profile.Type != ProfileXMLRecord {
				return fmt.Errorf("input %s has an inconsistent XML-record adapter", artifact.Path)
			}
		default:
			return fmt.Errorf("input %s has unsupported adapter %q", artifact.Path, input.Adapter)
		}
		previous = artifact.Path
	}
	if len(plan.Inputs) == 0 {
		return fmt.Errorf("ingestion plan has no inputs")
	}
	return nil
}

func validatePlanSource(source PlanSource) error {
	category, ok := index.CanonicalSourceCategory(source.Category)
	if !ok || category != source.Category {
		return fmt.Errorf("source category %q is not canonical", source.Category)
	}
	return index.ValidateSourceProvenance(index.Source{
		Category: category, CollectedFrom: source.CollectedFrom, CollectedTo: source.CollectedTo,
		InputFormats: source.InputFormats, LicenseEvidence: source.LicenseEvidence, Content: source.Content, Acquisition: source.Acquisition,
	})
}

func validSHA256(value string) bool {
	if len(value) != sha256.Size*2 || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

// Identity hashes the complete accepted plan. Execution journals pin this
// value and refuse to resume if any input, mapping, recipe, or limit changes.
func (plan Plan) Identity() (string, error) {
	if err := plan.Validate(); err != nil {
		return "", err
	}
	data, err := json.Marshal(plan)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}
