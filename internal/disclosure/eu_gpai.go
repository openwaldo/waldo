// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package disclosure maps verified model provenance into versioned public
// disclosure inputs. It reports absent facts; it does not make legal findings.
package disclosure

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/model"
)

const (
	EUGPAIReportSchema          = 1
	ProviderSchema              = 1
	EUGPAITemplate              = "C(2025) 8311 final"
	EUGPAITemplateDate          = "2025-12-05"
	EUGPAITemplateSource        = "https://ec.europa.eu/newsroom/dae/redirection/document/118578"
	EUGPAIEnglishTemplateSHA256 = "ec803008a5263a485146b24497a3445e2ea32f8b73f818e67652ad70de40f09b"
)

type ProviderProfile struct {
	Kind                 string           `json:"kind"`
	Schema               int              `json:"schema"`
	Provider             Organization     `json:"provider"`
	EURepresentative     *Organization    `json:"eu_representative,omitempty"`
	CodeOfPracticeStatus string           `json:"code_of_practice_status"`
	CopyrightPolicyURL   string           `json:"copyright_policy_url"`
	AdditionalMeasures   ProviderMeasures `json:"additional_measures,omitempty"`
}

type Organization struct {
	Name            string `json:"name"`
	Address         string `json:"address"`
	Contact         string `json:"contact"`
	EstablishedInEU string `json:"established_in_eu,omitempty"`
	Size            string `json:"size,omitempty"`
}

type ReleaseProfile struct {
	SummaryVersion      string   `json:"summary_version"`
	PreviousSummaryURLs []string `json:"previous_summary_urls,omitempty"`
	PublicName          string   `json:"public_name"`
	Version             string   `json:"version"`
	MarketPlacementDate string   `json:"market_placement_date"`
	Origin              string   `json:"origin"`
	OriginalSummaryURL  string   `json:"original_summary_url,omitempty"`
	DocumentationURL    string   `json:"documentation_url,omitempty"`
	ContinuousTraining  string   `json:"continuous_training"`
}

type ProviderMeasures struct {
	RightsReservation []string `json:"rights_reservation,omitempty"`
	IllegalContent    []string `json:"illegal_content,omitempty"`
	Comments          []string `json:"comments,omitempty"`
}

type EUGPAIReport struct {
	Kind       string           `json:"kind"`
	Schema     int              `json:"schema"`
	Status     string           `json:"status"`
	Generated  string           `json:"generated"`
	Template   TemplatePin      `json:"template"`
	Model      ModelSummary     `json:"model"`
	Provider   *ProviderProfile `json:"provider,omitempty"`
	Release    ReleaseProfile   `json:"release"`
	Training   TrainingSummary  `json:"training"`
	Gaps       []Gap            `json:"gaps,omitempty"`
	Disclaimer string           `json:"disclaimer"`
}

type TemplatePin struct {
	Authority    string `json:"authority"`
	Document     string `json:"document"`
	Version      string `json:"version"`
	Language     string `json:"language"`
	Source       string `json:"source"`
	SourceSHA256 string `json:"source_sha256"`
}

type ModelSummary struct {
	ID                 string              `json:"id"`
	Name               string              `json:"name"`
	PlanSHA256         string              `json:"plan_sha256"`
	ArchitectureSHA256 string              `json:"architecture_sha256"`
	BOMSHA256          string              `json:"bom_sha256"`
	Runs               int                 `json:"runs"`
	Origin             *model.OriginSource `json:"origin,omitempty"`
}

type TrainingSummary struct {
	Stages        []StageUse  `json:"stages"`
	UniqueCorpora []CorpusUse `json:"unique_corpora"`
}

type StageUse struct {
	Ordinal                int            `json:"ordinal"`
	Name                   string         `json:"name"`
	Type                   string         `json:"type"`
	Objective              string         `json:"objective"`
	State                  model.RunState `json:"state"`
	CorpusBOMSHA256        string         `json:"corpus_bom_sha256"`
	PlannedTokens          int64          `json:"planned_tokens"`
	ObservedTokens         int64          `json:"observed_tokens,omitempty"`
	ObservationIsSimulated bool           `json:"observation_is_simulated,omitempty"`
}

type CorpusUse struct {
	BOMSHA256  string                    `json:"bom_sha256"`
	Uses       int                       `json:"uses"`
	Stages     []string                  `json:"stages"`
	Paths      []string                  `json:"paths"`
	Totals     index.Measures            `json:"totals"`
	Modalities index.Modalities          `json:"modalities,omitempty"`
	Licenses   map[string]index.Measures `json:"licenses"`
	Manifests  []ManifestEvidence        `json:"manifests"`
}

type ManifestEvidence struct {
	Path       string                      `json:"path"`
	SHA256     string                      `json:"sha256"`
	Name       string                      `json:"name"`
	Title      string                      `json:"title"`
	Sources    []index.Source              `json:"sources"`
	Processing *index.Processing           `json:"processing,omitempty"`
	ComposedBy *index.IngestRecipeEvidence `json:"composed_by,omitempty"`
}

type Gap struct {
	Severity string `json:"severity"`
	Section  string `json:"section"`
	Field    string `json:"field"`
	Context  string `json:"context,omitempty"`
	Message  string `json:"message"`
}

func LoadProvider(path string) (ProviderProfile, error) {
	file, err := os.Open(path)
	if err != nil {
		return ProviderProfile{}, err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var profile ProviderProfile
	if err := decoder.Decode(&profile); err != nil {
		return ProviderProfile{}, fmt.Errorf("%s: %w", path, err)
	}
	if err := requireEOF(decoder); err != nil {
		return ProviderProfile{}, fmt.Errorf("%s: %w", path, err)
	}
	if profile.Kind != "waldo-disclosure-provider" || profile.Schema != ProviderSchema {
		return ProviderProfile{}, fmt.Errorf("%s has unsupported provider profile identity %q schema %d", path, profile.Kind, profile.Schema)
	}
	return profile, nil
}

func BuildEUGPAIReport(inspection model.Inspection, provider *ProviderProfile, release ReleaseProfile, now time.Time) (EUGPAIReport, error) {
	modelBOMHash, err := hashJSON(inspection.BOM)
	if err != nil {
		return EUGPAIReport{}, err
	}
	report := EUGPAIReport{
		Kind: "waldo-eu-gpai-training-content", Schema: EUGPAIReportSchema,
		Status: "complete", Generated: now.UTC().Format(time.RFC3339),
		Template:   TemplatePin{Authority: "European Commission", Document: EUGPAITemplate, Version: EUGPAITemplateDate, Language: "en", Source: EUGPAITemplateSource, SourceSHA256: EUGPAIEnglishTemplateSHA256},
		Model:      ModelSummary{ID: inspection.Model.ID, Name: inspection.Model.Name, PlanSHA256: inspection.Model.PlanSHA256, ArchitectureSHA256: inspection.Model.ArchitectureSHA256, BOMSHA256: modelBOMHash, Runs: len(inspection.Runs)},
		Provider:   provider,
		Release:    release,
		Disclaimer: "WALDO maps recorded provenance to disclosure fields. This report is not legal advice and does not by itself establish compliance.",
	}
	if inspection.Origin != nil {
		source := inspection.Origin.Source
		report.Model.Origin = &source
	}
	report.addProviderGaps(provider)
	report.addReleaseGaps(release)
	// Training behind the starting weights counts too: a merge or a model built
	// on a release discloses its whole history, not only its own runs.
	inherited, err := model.InheritedClosure(inspection)
	if err != nil {
		return EUGPAIReport{}, err
	}
	if !inherited.Complete {
		report.gap("1.3", "training.lineage", inspection.Model.Name, "training behind the model's starting weights is not fully recorded")
	}
	if len(inspection.RunBOMs) == 0 && len(inherited.Corpora) == 0 {
		report.gap("1.3", "training.stages", inspection.Model.Name, "the model records no training stages")
	}
	if len(inspection.RunBOMs) != len(inspection.Runs) {
		return EUGPAIReport{}, fmt.Errorf("model inspection has %d run records and %d run BOMs", len(inspection.Runs), len(inspection.RunBOMs))
	}
	corpora := map[string]*CorpusUse{}
	for position, runBOM := range inspection.RunBOMs {
		run := inspection.Runs[position]
		stage := StageUse{Ordinal: runBOM.Ordinal, Name: runBOM.Stage, Type: runBOM.StageType, Objective: runBOM.Objective, State: run.State, CorpusBOMSHA256: runBOM.CorpusBOMSHA256, PlannedTokens: plannedTokens(runBOM)}
		if run.Observation != nil {
			stage.ObservedTokens = run.Observation.ConsumedTokens
			stage.ObservationIsSimulated = run.Observation.Simulated
		}
		report.Training.Stages = append(report.Training.Stages, stage)
		context := fmt.Sprintf("run %04d %s", runBOM.Ordinal, runBOM.Stage)
		if run.State != model.RunComplete {
			report.gap("1.3", "training.completed", context, "the run is not complete")
		}
		if run.Observation == nil || run.Observation.Simulated {
			report.gap("1.3", "training.observed-consumption", context, "actual training consumption is not recorded")
		}
		if runBOM.StageType == "" {
			report.gap("1.3", "training.stage-type", context, "pre-training, fine-tuning, alignment, or other stage type is not declared")
		}
		report.gap("1.3", "training.consumption-breakdown", context, "actual consumption by source and modality is not recorded")

		entry := corpora[runBOM.CorpusBOMSHA256]
		if entry == nil {
			entry = corpusUse(runBOM.CorpusBOMSHA256, runBOM.CorpusBOM)
			corpora[runBOM.CorpusBOMSHA256] = entry
			report.addCorpusGaps(*entry)
		}
		entry.Uses++
		entry.Stages = append(entry.Stages, runBOM.Stage)
	}
	for _, ancestor := range inherited.Corpora {
		entry := corpora[ancestor.CorpusBOMSHA256]
		if entry == nil {
			entry = corpusUse(ancestor.CorpusBOMSHA256, ancestor.CorpusBOM)
			corpora[ancestor.CorpusBOMSHA256] = entry
			report.addCorpusGaps(*entry)
		}
		entry.Uses += ancestor.Uses
		entry.Stages = append(entry.Stages, "inherited")
	}
	keys := make([]string, 0, len(corpora))
	for key := range corpora {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		report.Training.UniqueCorpora = append(report.Training.UniqueCorpora, *corpora[key])
	}
	sort.Slice(report.Gaps, func(i, j int) bool {
		if report.Gaps[i].Section != report.Gaps[j].Section {
			return report.Gaps[i].Section < report.Gaps[j].Section
		}
		if report.Gaps[i].Field != report.Gaps[j].Field {
			return report.Gaps[i].Field < report.Gaps[j].Field
		}
		return report.Gaps[i].Context < report.Gaps[j].Context
	})
	if report.BlockingGaps() > 0 {
		report.Status = "incomplete-draft"
	} else if len(report.Gaps) > 0 {
		report.Status = "complete-with-notes"
	}
	return report, nil
}

func (report *EUGPAIReport) addProviderGaps(profile *ProviderProfile) {
	if profile == nil {
		report.gap("1.1", "provider.profile", "", "a provider profile was not supplied")
		return
	}
	require := func(section, field, value, message string) {
		if strings.TrimSpace(value) == "" {
			report.gap(section, field, "provider profile", message)
		}
	}
	require("1.1", "provider.name", profile.Provider.Name, "provider name is missing")
	require("1.1", "provider.address", profile.Provider.Address, "provider address is missing")
	require("1.1", "provider.contact", profile.Provider.Contact, "provider contact is missing")
	if profile.Provider.EstablishedInEU != "yes" && profile.Provider.EstablishedInEU != "no" {
		report.gap("1.1", "provider.established-in-eu", "provider profile", "EU establishment must be declared as yes or no")
	}
	if profile.Provider.EstablishedInEU == "no" && (profile.EURepresentative == nil || strings.TrimSpace(profile.EURepresentative.Name) == "" || strings.TrimSpace(profile.EURepresentative.Contact) == "") {
		report.gap("1.1", "provider.eu-representative", "provider profile", "a provider established outside the Union requires authorised representative details")
	}
	require("3.1", "provider.code-of-practice-status", profile.CodeOfPracticeStatus, "Code of Practice status is missing")
	if profile.CodeOfPracticeStatus != "" && profile.CodeOfPracticeStatus != "yes" && profile.CodeOfPracticeStatus != "no" {
		report.gap("3.1", "provider.code-of-practice-status", "provider profile", "Code of Practice status must be yes or no")
	}
	if strings.TrimSpace(profile.CopyrightPolicyURL) == "" {
		report.note("optional", "3.1", "provider.copyright-policy-url", "provider profile", "a public copyright policy link is encouraged by the template")
	}
}

func (report *EUGPAIReport) addReleaseGaps(profile ReleaseProfile) {
	require := func(section, field, value, message string) {
		if strings.TrimSpace(value) == "" {
			report.gap(section, field, "model release", message)
		}
	}
	require("General", "summary.version", profile.SummaryVersion, "summary version is missing")
	require("1.2", "model.public-name", profile.PublicName, "public model name is missing")
	require("1.2", "model.version", profile.Version, "public model version is missing")
	require("1.2", "model.market-placement-date", profile.MarketPlacementDate, "EU market-placement date is missing")
	if profile.MarketPlacementDate != "" {
		if _, err := time.Parse("2006-01-02", profile.MarketPlacementDate); err != nil {
			report.gap("1.2", "model.market-placement-date", "model release", "market-placement date must use YYYY-MM-DD")
		}
	}
	if profile.Origin != "new" && profile.Origin != "modified" {
		report.gap("1.2", "model.origin", "model release", "model origin must be new or modified")
	}
	if profile.Origin == "modified" && strings.TrimSpace(profile.OriginalSummaryURL) == "" {
		report.gap("1.2", "model.original-summary", "model release", "a modified model requires the original model public-summary URL")
	}
	if profile.ContinuousTraining != "yes" && profile.ContinuousTraining != "no" {
		report.gap("1.3", "model.continuous-training", "model release", "continuous training must be declared as yes or no")
	}
}

func ReleaseFromModel(inspection model.Inspection) ReleaseProfile {
	release := ReleaseProfile{
		SummaryVersion: "1",
		PublicName:     inspection.Model.Name,
		Version:        inspection.Model.ID,
		Origin:         "new",
	}
	if inspection.Origin != nil {
		release.Origin = "modified"
		release.OriginalSummaryURL = inspection.Origin.Source.URL
	}
	return release
}

func (report *EUGPAIReport) addCorpusGaps(item CorpusUse) {
	context := shortHash(item.BOMSHA256)
	if len(item.Modalities) == 0 {
		report.gap("1.3", "content.modalities", context, "exact per-modality measures are absent")
	}
	for _, manifest := range item.Manifests {
		manifestContext := manifest.Path
		if manifest.Processing == nil || len(manifest.Processing.Steps) == 0 {
			report.note("review", "3.3", "processing.steps", manifestContext, "general structured processing steps are absent")
		}
		if manifest.Processing == nil || len(manifest.Processing.RightsReservationMeasures) == 0 {
			report.gap("3.1", "processing.rights-reservation", manifestContext, "rights-reservation measures are absent")
		}
		if manifest.Processing == nil || len(manifest.Processing.IllegalContentMeasures) == 0 {
			report.gap("3.2", "processing.illegal-content", manifestContext, "illegal-content measures are absent")
		}
		for _, source := range manifest.Sources {
			sourceContext := manifest.Path + ":" + source.Name
			category, ok := index.CanonicalSourceCategory(source.Category)
			if !ok {
				report.gap("2", "source.category", sourceContext, "a supported source category is absent")
			}
			if len(source.Usage) == 0 {
				report.gap("1.3", "source.usage", sourceContext, "per-source modality usage is absent")
			}
			if source.Content == nil || len(source.Content.Types) == 0 {
				report.gap("1.3", "source.content-types", sourceContext, "plain-language content types are absent")
			}
			if source.Content == nil || len(source.Content.Languages) == 0 {
				report.note("review", "1.3", "source.languages", sourceContext, "language information is absent; confirm whether it is applicable")
			}
			if source.CollectedTo == "" {
				report.gap("1.3", "source.acquisition-date", sourceContext, "latest acquisition date is absent")
			}
			switch category {
			case index.SourceCommerciallyLicensed:
				if source.Acquisition == nil || strings.TrimSpace(source.Acquisition.Basis) == "" {
					report.gap("2.2", "source.acquisition-basis", sourceContext, "licensed/private acquisition basis is absent")
				}
			case index.SourceWebCrawl:
				if source.Acquisition == nil || source.Acquisition.Crawler == nil || len(source.Acquisition.Domains) == 0 {
					report.gap("2.3", "source.web-evidence", sourceContext, "crawler and retained-domain evidence is incomplete")
				}
			case index.SourceUserData:
				if source.Acquisition == nil || source.Acquisition.UserData == nil {
					report.gap("2.4", "source.user-data", sourceContext, "collecting service and interaction are absent")
				}
			case index.SourceSynthetic:
				if source.Acquisition == nil || source.Acquisition.Synthetic == nil {
					report.gap("2.5", "source.synthetic", sourceContext, "generator model provenance is absent")
				}
			}
		}
	}
}

func corpusUse(hash string, bom corpus.BOM) *CorpusUse {
	result := &CorpusUse{BOMSHA256: hash, Paths: append([]string(nil), bom.Paths...), Totals: bom.Totals, Modalities: cloneModalities(bom.Modalities), Licenses: bom.Licenses}
	for _, manifest := range bom.Manifests {
		result.Manifests = append(result.Manifests, ManifestEvidence{Path: manifest.Path, SHA256: manifest.SHA256, Name: manifest.Name, Title: manifest.Title, Sources: manifest.Sources, Processing: manifest.Processing, ComposedBy: manifest.ComposedBy})
	}
	return result
}

func plannedTokens(run model.RunBOM) int64 {
	return run.Parameters.Steps * run.Parameters.BatchSize * run.Parameters.SequenceLength
}

func (report *EUGPAIReport) gap(section, field, context, message string) {
	report.note("required", section, field, context, message)
}

func (report *EUGPAIReport) note(severity, section, field, context, message string) {
	report.Gaps = append(report.Gaps, Gap{Severity: severity, Section: section, Field: field, Context: context, Message: message})
}

func (report EUGPAIReport) BlockingGaps() int {
	count := 0
	for _, gap := range report.Gaps {
		if gap.Severity == "required" {
			count++
		}
	}
	return count
}

func WriteEUGPAIReport(path string, report EUGPAIReport) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".waldo-eu-gpai-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	committed := false
	defer func() {
		_ = temporary.Close()
		if !committed {
			_ = os.Remove(temporaryPath)
		}
	}()
	if _, err := temporary.Write(data); err != nil {
		return err
	}
	if err := temporary.Chmod(0o644); err != nil {
		return err
	}
	if err := temporary.Sync(); err != nil {
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return err
	}
	committed = true
	return nil
}

func hashJSON(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func requireEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values are not allowed")
		}
		return err
	}
	return nil
}

func cloneModalities(source index.Modalities) index.Modalities {
	if len(source) == 0 {
		return nil
	}
	result := make(index.Modalities, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func shortHash(value string) string {
	if len(value) > 12 {
		return value[:12]
	}
	return value
}
