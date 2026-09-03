// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/training"
)

const (
	PlanSchema     = 1
	ModelSchema    = 1
	RunSchema      = 1
	ModelBOMSchema = 1
	RunBOMSchema   = 1
)

type Plan struct {
	Kind               string               `json:"kind"`
	Schema             int                  `json:"schema"`
	Name               string               `json:"name"`
	ArchitectureSHA256 string               `json:"architecture_sha256"`
	Architecture       Architecture         `json:"architecture"`
	Interaction        Interaction          `json:"interaction,omitzero"`
	Forecast           ArchitectureForecast `json:"forecast"`
	Stages             []PlannedStage       `json:"stages,omitempty"`
	OriginBOMSHA256    string               `json:"origin_bom_sha256,omitempty"`
	Parent             *ModelParent         `json:"parent,omitempty"`
}

type ModelParent struct {
	Model        string            `json:"model"`
	ModelID      string            `json:"model_id"`
	RunID        string            `json:"run_id"`
	RunBOMSHA256 string            `json:"run_bom_sha256"`
	Artifact     training.Artifact `json:"artifact"`
}

type OriginSource struct {
	Provider          string `json:"provider"`
	Repository        string `json:"repository"`
	RequestedRevision string `json:"requested_revision"`
	Revision          string `json:"revision"`
	URL               string `json:"url"`
	License           string `json:"license,omitempty"`
}

type OriginArtifact struct {
	Role   string `json:"role"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type OriginBOM struct {
	Kind               string           `json:"kind"`
	Schema             int              `json:"schema"`
	Subject            string           `json:"subject"`
	Source             OriginSource     `json:"source"`
	ArchitectureSHA256 string           `json:"architecture_sha256"`
	SourceArtifacts    []OriginArtifact `json:"source_artifacts"`
	Artifacts          []OriginArtifact `json:"artifacts"`
}

type PlannedStage struct {
	Name            string              `json:"name"`
	Type            string              `json:"type"`
	Objective       string              `json:"objective"`
	CorpusBOMSHA256 string              `json:"corpus_bom_sha256"`
	Files           int                 `json:"files"`
	Docs            int64               `json:"docs"`
	Tokens          int64               `json:"tokens"`
	Bytes           int64               `json:"bytes"`
	Parameters      training.Parameters `json:"parameters"`
	PlannedTokens   int64               `json:"planned_token_capacity"`
}

type ModelRecord struct {
	Kind               string               `json:"kind"`
	Schema             int                  `json:"schema"`
	ID                 string               `json:"id"`
	Name               string               `json:"name"`
	PlanSHA256         string               `json:"plan_sha256"`
	ArchitectureSHA256 string               `json:"architecture_sha256"`
	Architecture       Architecture         `json:"architecture"`
	Interaction        Interaction          `json:"interaction,omitzero"`
	Forecast           ArchitectureForecast `json:"forecast"`
	Created            string               `json:"created"`
	Updated            string               `json:"updated"`
	Runs               []RunPin             `json:"runs"`
	OriginBOMSHA256    string               `json:"origin_bom_sha256,omitempty"`
	OriginArtifacts    []OriginArtifact     `json:"origin_artifacts,omitempty"`
	Parent             *ModelParent         `json:"parent,omitempty"`
}

type RunState string

const (
	RunPlanned     RunState = "planned"
	RunRunning     RunState = "running"
	RunComplete    RunState = "complete"
	RunFailed      RunState = "failed"
	RunInterrupted RunState = "interrupted"
)

type RunPin struct {
	ID                string                `json:"id"`
	Stage             string                `json:"stage"`
	Ordinal           int                   `json:"ordinal"`
	BOMSHA256         string                `json:"bom_sha256"`
	State             RunState              `json:"state"`
	Backend           training.Identity     `json:"backend,omitempty"`
	Simulated         bool                  `json:"simulated"`
	ObservationSHA256 string                `json:"observation_sha256,omitempty"`
	Artifacts         []training.Artifact   `json:"artifacts,omitempty"`
	Resume            *training.ResumePoint `json:"resume,omitempty"`
}

type RunBOM struct {
	Kind               string                         `json:"kind"`
	Schema             int                            `json:"schema"`
	Subject            string                         `json:"subject"`
	ID                 string                         `json:"id"`
	ModelID            string                         `json:"model_id"`
	Stage              string                         `json:"stage"`
	StageType          string                         `json:"stage_type"`
	Ordinal            int                            `json:"ordinal"`
	Objective          string                         `json:"objective"`
	Conversation       training.ConversationTransform `json:"conversation,omitzero"`
	Execution          training.Execution             `json:"execution"`
	ArchitectureSHA256 string                         `json:"architecture_sha256"`
	CorpusBOMSHA256    string                         `json:"corpus_bom_sha256"`
	CorpusBOM          corpus.BOM                     `json:"corpus_bom"`
	Parameters         training.ResolvedParameters    `json:"parameters"`
	EvaluationSet      *training.EvaluationSet        `json:"evaluation_set,omitempty"`
	Initialization     *training.Initialization       `json:"initialization,omitempty"`
}

// EffectiveInteraction returns the immutable model interaction plus the
// equivalent tool declaration recorded by older schema-1 training runs.
func (inspection Inspection) EffectiveInteraction() Interaction {
	interaction := inspection.Model.Interaction
	if interaction.Tools {
		return interaction
	}
	for index, bom := range inspection.RunBOMs {
		if index >= len(inspection.Runs) || inspection.Runs[index].State != RunComplete || bom.Objective != "assistant-response-modeling" || !bom.Conversation.Tools || bom.Conversation.Template != interaction.Template {
			continue
		}
		for _, role := range bom.Conversation.SupervisedRoles {
			if role == "assistant" {
				interaction.Tools = true
				return interaction
			}
		}
	}
	return interaction
}

const MultiNodePlanSchema = 1

const MultiNodePlanKind = "openwaldo-multinode-plan"

type MultiNodePlan struct {
	Kind               string                         `json:"kind"`
	Schema             int                            `json:"schema"`
	RunID              string                         `json:"run_id"`
	Stage              string                         `json:"stage"`
	StageOrdinal       int                            `json:"stage_ordinal"`
	StageCount         int                            `json:"stage_count"`
	Nodes              int                            `json:"nodes"`
	Objective          string                         `json:"objective"`
	Conversation       training.ConversationTransform `json:"conversation,omitzero"`
	ArchitectureSHA256 string                         `json:"architecture_sha256"`
	Architecture       json.RawMessage                `json:"architecture"`
	Parameters         training.ResolvedParameters    `json:"parameters"`
	CorpusBOM          corpus.BOM                     `json:"corpus_bom"`
	EvaluationSet      *training.EvaluationSet        `json:"evaluation_set,omitempty"`
	Initialization     *training.Initialization       `json:"initialization,omitempty"`
	InitializationPath string                         `json:"initialization_path,omitempty"`
}

func MultiNodePlanPath(root, rendezvousID string) string {
	return filepath.Join(root, ".multinode", rendezvousID, "plan.json")
}

type RunRecord struct {
	Kind        string                `json:"kind"`
	Schema      int                   `json:"schema"`
	ID          string                `json:"id"`
	State       RunState              `json:"state"`
	BOMSHA256   string                `json:"bom_sha256"`
	Planned     string                `json:"planned"`
	Started     string                `json:"started,omitempty"`
	Finished    string                `json:"finished,omitempty"`
	Observation *training.Observation `json:"observation,omitempty"`
	Progress    *training.Progress    `json:"progress,omitempty"`
	Attempts    []RunAttempt          `json:"attempts,omitempty"`
	Error       string                `json:"error,omitempty"`
}

type RunAttempt struct {
	Ordinal    int      `json:"ordinal"`
	Started    string   `json:"started"`
	Finished   string   `json:"finished,omitempty"`
	State      RunState `json:"state"`
	Error      string   `json:"error,omitempty"`
	ResumeStep int64    `json:"resume_step,omitempty"`
}

type ModelBOM struct {
	Kind                string          `json:"kind"`
	Schema              int             `json:"schema"`
	Subject             string          `json:"subject"`
	ModelID             string          `json:"model_id"`
	Name                string          `json:"name"`
	PlanSHA256          string          `json:"plan_sha256"`
	ArchitectureSHA256  string          `json:"architecture_sha256"`
	Interaction         Interaction     `json:"interaction,omitzero"`
	PathBase            string          `json:"path_base"`
	CurrentRunID        string          `json:"current_run_id,omitempty"`
	CurrentOriginSHA256 string          `json:"current_origin_sha256,omitempty"`
	Origin              *ModelBOMOrigin `json:"origin,omitempty"`
	Parent              *ModelParent    `json:"parent,omitempty"`
	Runs                []ModelBOMRun   `json:"runs"`
	Generated           string          `json:"generated"`
}

type ModelBOMOrigin struct {
	BOM       string             `json:"bom"`
	SHA256    string             `json:"sha256"`
	Artifacts []ModelBOMArtifact `json:"artifacts"`
}

type ModelBOMRun struct {
	ID                string             `json:"id"`
	Stage             string             `json:"stage"`
	Ordinal           int                `json:"ordinal"`
	RunBOM            string             `json:"run_bom"`
	BOMSHA256         string             `json:"bom_sha256"`
	State             RunState           `json:"state"`
	Backend           training.Identity  `json:"backend"`
	Simulated         bool               `json:"simulated"`
	ObservationSHA256 string             `json:"observation_sha256,omitempty"`
	Artifacts         []ModelBOMArtifact `json:"artifacts,omitempty"`
}

type ModelBOMArtifact struct {
	Role   string `json:"role"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type Inspection struct {
	Path    string      `json:"path"`
	Plan    Plan        `json:"plan"`
	Model   ModelRecord `json:"model"`
	BOM     ModelBOM    `json:"bom"`
	Runs    []RunRecord `json:"runs"`
	RunBOMs []RunBOM    `json:"run_boms"`
	Origin  *OriginBOM  `json:"origin,omitempty"`
}

func Inspect(root, nameOrPath string) (Inspection, error) {
	directory, err := modelDirectory(root, nameOrPath)
	if err != nil {
		return Inspection{}, err
	}
	var record ModelRecord
	if err := readJSON(filepath.Join(directory, "MODEL.json"), &record); err != nil {
		return Inspection{}, err
	}
	if record.Kind != "waldo-model" || record.Schema != ModelSchema || !validName.MatchString(record.Name) || record.ID == "" {
		return Inspection{}, fmt.Errorf("%s has an invalid model record", directory)
	}
	architectureHash, err := hashJSON(record.Architecture)
	if err != nil {
		return Inspection{}, err
	}
	forecast, err := record.Architecture.Forecast()
	if err != nil || architectureHash != record.ArchitectureSHA256 || !reflect.DeepEqual(forecast, record.Forecast) {
		return Inspection{}, fmt.Errorf("%s has inconsistent architecture identity or forecast", directory)
	}
	var plan Plan
	if err := readJSON(filepath.Join(directory, "PLAN.json"), &plan); err != nil {
		return Inspection{}, err
	}
	planHash, err := hashJSON(plan)
	if err != nil {
		return Inspection{}, err
	}
	if plan.Kind != "waldo-model-plan" || plan.Schema != PlanSchema || planHash != record.PlanSHA256 || record.ID != planHash || plan.Name != record.Name || plan.ArchitectureSHA256 != record.ArchitectureSHA256 || plan.OriginBOMSHA256 != record.OriginBOMSHA256 || !reflect.DeepEqual(plan.Parent, record.Parent) || !reflect.DeepEqual(plan.Architecture, record.Architecture) || !reflect.DeepEqual(plan.Interaction, record.Interaction) || !reflect.DeepEqual(plan.Forecast, record.Forecast) {
		return Inspection{}, fmt.Errorf("%s has an invalid immutable model plan", directory)
	}
	if record.Parent != nil && record.OriginBOMSHA256 != "" {
		return Inspection{}, fmt.Errorf("%s cannot have both parent-run and origin initialization", directory)
	}
	if record.Parent != nil {
		parent := record.Parent
		if parent.Model == "" || parent.ModelID == "" || parent.RunID == "" || parent.RunBOMSHA256 == "" || parent.Artifact.Path != "base/model.safetensors" {
			return Inspection{}, fmt.Errorf("%s has an invalid parent-run pin", directory)
		}
		if err := VerifyArtifactFile(filepath.Join(directory, filepath.FromSlash(parent.Artifact.Path)), parent.Artifact); err != nil {
			return Inspection{}, fmt.Errorf("model parent: %w", err)
		}
	}
	var origin *OriginBOM
	if record.OriginBOMSHA256 != "" {
		var value OriginBOM
		if err := readJSON(filepath.Join(directory, "ORIGIN-BOM.json"), &value); err != nil {
			return Inspection{}, err
		}
		digest, err := hashJSON(value)
		if err != nil || digest != record.OriginBOMSHA256 || value.Kind != "openwaldo-bom" || value.Schema != 1 || value.Subject != "model-origin" || value.ArchitectureSHA256 != record.ArchitectureSHA256 {
			return Inspection{}, fmt.Errorf("%s has an invalid model origin BOM", directory)
		}
		for _, artifact := range value.Artifacts {
			if err := verifyOriginArtifact(directory, artifact); err != nil {
				return Inspection{}, fmt.Errorf("model origin: %w", err)
			}
		}
		if !reflect.DeepEqual(value.Artifacts, record.OriginArtifacts) {
			return Inspection{}, fmt.Errorf("%s model origin artifacts do not match their model pin", directory)
		}
		origin = &value
	}
	var bom ModelBOM
	bomPath := filepath.Join(directory, "MODEL-BOM.json")
	if _, err := os.Stat(bomPath); os.IsNotExist(err) {
		bomPath = filepath.Join(directory, "BOM.json")
	}
	if err := readJSON(bomPath, &bom); err != nil {
		return Inspection{}, err
	}
	if bom.Kind != "openwaldo-bom" || bom.Schema != ModelBOMSchema || bom.Subject != "model" || bom.ModelID != record.ID || bom.Name != record.Name || bom.PlanSHA256 != record.PlanSHA256 || bom.ArchitectureSHA256 != record.ArchitectureSHA256 || !reflect.DeepEqual(bom.Interaction, record.Interaction) || bom.Generated != record.Updated {
		return Inspection{}, fmt.Errorf("%s has an invalid model OpenWALDO BOM", directory)
	}
	originalPins := append([]RunPin(nil), record.Runs...)
	inspection := Inspection{Path: directory, Plan: plan, Model: record, Origin: origin}
	for _, pin := range record.Runs {
		position := len(inspection.Runs)
		if pin.Ordinal != position+1 {
			return Inspection{}, fmt.Errorf("model run %s has ordinal %d at position %d", pin.ID, pin.Ordinal, position+1)
		}
		var run RunRecord
		runDirectory := filepath.Join(directory, "runs", runDirectoryName(pin))
		if err := readJSON(filepath.Join(runDirectory, "RUN.json"), &run); err != nil {
			return Inspection{}, err
		}
		var runBOM RunBOM
		if err := readJSON(filepath.Join(runDirectory, "RUN-BOM.json"), &runBOM); err != nil {
			return Inspection{}, err
		}
		runBOMHash, err := hashJSON(runBOM)
		if err != nil {
			return Inspection{}, err
		}
		if run.Kind != "waldo-training-run" || run.Schema != RunSchema || run.ID != pin.ID || run.State != pin.State || run.BOMSHA256 != pin.BOMSHA256 || runBOMHash != pin.BOMSHA256 || runBOM.ID != pin.ID || runBOM.ModelID != record.ID || runBOM.Stage != pin.Stage || runBOM.Ordinal != pin.Ordinal {
			return Inspection{}, fmt.Errorf("run %s does not match its model pin", pin.ID)
		}
		persistedBackend := pin.Backend
		persistedSimulated := pin.Simulated
		if persistedBackend != (training.Identity{}) && persistedBackend != runBOM.Execution.Backend {
			return Inspection{}, fmt.Errorf("run %s backend does not match its model pin", pin.ID)
		}
		pin.Backend = runBOM.Execution.Backend
		if persistedBackend == (training.Identity{}) {
			pin.Simulated = runBOM.Execution.Backend.Name == training.BackendFake || run.Observation != nil && run.Observation.Simulated
		} else if run.Observation != nil && persistedSimulated != run.Observation.Simulated {
			return Inspection{}, fmt.Errorf("run %s simulation state does not match its model pin", pin.ID)
		}
		if runBOM.ArchitectureSHA256 != plan.ArchitectureSHA256 || runBOM.ModelID != record.ID || runBOM.Stage == "" || runBOM.StageType == "" || runBOM.Objective == "" {
			return Inspection{}, fmt.Errorf("run %s does not match its immutable model architecture", pin.ID)
		}
		corpusHash, err := hashJSON(runBOM.CorpusBOM)
		if err != nil || corpusHash != runBOM.CorpusBOMSHA256 || runBOM.ArchitectureSHA256 != record.ArchitectureSHA256 {
			return Inspection{}, fmt.Errorf("run %s has inconsistent corpus or architecture identity", pin.ID)
		}
		if err := runBOM.CorpusBOM.Validate(); err != nil {
			return Inspection{}, fmt.Errorf("run %s corpus OpenWALDO BOM: %w", pin.ID, err)
		}
		if err := validateEvaluationSet(runBOM); err != nil {
			return Inspection{}, fmt.Errorf("run %s evaluation set: %w", pin.ID, err)
		}
		if run.Progress != nil {
			for _, checkpoint := range run.Progress.Checkpoints {
				for _, artifact := range checkpoint.Artifacts {
					clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(artifact.Path)))
					if artifact.Path == "" || filepath.IsAbs(filepath.FromSlash(artifact.Path)) || clean != artifact.Path || !strings.HasPrefix(clean, "artifacts/checkpoints/") {
						return Inspection{}, fmt.Errorf("run %s checkpoint artifact path %q is invalid", pin.ID, artifact.Path)
					}
					if err := VerifyArtifactFile(filepath.Join(runDirectory, filepath.FromSlash(artifact.Path)), artifact); err != nil {
						return Inspection{}, fmt.Errorf("run %s checkpoint: %w", pin.ID, err)
					}
				}
			}
		}
		if err := validateRunState(run, pin); err != nil {
			return Inspection{}, err
		}
		inspection.Runs = append(inspection.Runs, run)
		inspection.RunBOMs = append(inspection.RunBOMs, runBOM)
		inspection.Model.Runs[position] = pin
	}
	normalized := modelBOM(inspection.Model)
	if !reflect.DeepEqual(bom, normalized) && !legacyModelBOMMatches(bom, record, originalPins) {
		return Inspection{}, fmt.Errorf("%s has an invalid model OpenWALDO BOM", directory)
	}
	inspection.BOM = normalized
	return inspection, nil
}

func validateEvaluationSet(runBOM RunBOM) error {
	policy := runBOM.Parameters.Evaluation
	if policy == nil {
		if runBOM.EvaluationSet != nil {
			return fmt.Errorf("legacy profile has unexpected held-out evidence")
		}
		return nil
	}
	set := runBOM.EvaluationSet
	if set == nil || set.Selection != policy.Selection || set.Seed != runBOM.Parameters.Seed || set.Records < 0 || set.Records > runBOM.CorpusBOM.Totals.Docs || set.TokenTargets < 0 || set.TextBytes < 0 {
		return fmt.Errorf("evidence does not match the resolved evaluation policy or corpus")
	}
	decoded, err := hex.DecodeString(set.SHA256)
	if err != nil || len(decoded) != sha256.Size || set.SHA256 != strings.ToLower(set.SHA256) {
		return fmt.Errorf("evidence SHA-256 is invalid")
	}
	if policy.Fraction == 0 || policy.MaxRecords == 0 || policy.MaxBytes == 0 {
		if set.Records != 0 || set.TokenTargets != 0 || set.TextBytes != 0 {
			return fmt.Errorf("disabled evaluation policy selected records")
		}
	} else if runBOM.CorpusBOM.Totals.Docs > 1 && set.Records == 0 {
		return fmt.Errorf("enabled evaluation policy selected no records")
	}
	return nil
}

func validateRunState(run RunRecord, pin RunPin) error {
	if err := validateAttempts(run); err != nil {
		return err
	}
	if run.Progress != nil {
		if run.Observation != nil {
			return fmt.Errorf("run %s contains both progress and a complete observation", run.ID)
		}
		if err := validateProgress(*run.Progress, pin.Resume); err != nil {
			return fmt.Errorf("run %s progress: %w", run.ID, err)
		}
	} else if pin.Resume != nil {
		return fmt.Errorf("run %s has a resume pin without durable progress", run.ID)
	}
	switch run.State {
	case RunPlanned:
		if run.Started != "" || run.Finished != "" || run.Observation != nil || run.Progress != nil || run.Error != "" {
			return fmt.Errorf("planned run %s contains observations or terminal state", run.ID)
		}
	case RunRunning:
		if run.Started == "" || run.Finished != "" || run.Observation != nil || run.Error != "" {
			return fmt.Errorf("running run %s has inconsistent state", run.ID)
		}
	case RunComplete:
		if run.Started == "" || run.Finished == "" || run.Observation == nil || run.Progress != nil || run.Error != "" || len(run.Observation.Artifacts) == 0 || pin.Resume != nil {
			return fmt.Errorf("complete run %s has incomplete observations", run.ID)
		}
		observationHash, err := hashJSON(run.Observation)
		if err != nil {
			return err
		}
		if observationHash != pin.ObservationSHA256 || !reflect.DeepEqual(run.Observation.Artifacts, pin.Artifacts) {
			return fmt.Errorf("complete run %s observations do not match its model pin", run.ID)
		}
	case RunFailed, RunInterrupted:
		if run.Started == "" || run.Finished == "" || run.Observation != nil || run.Error == "" {
			return fmt.Errorf("terminal run %s has inconsistent failure state", run.ID)
		}
	default:
		return fmt.Errorf("run %s has unsupported state %q", run.ID, run.State)
	}
	return nil
}

func validateAttempts(run RunRecord) error {
	for index, attempt := range run.Attempts {
		if attempt.Ordinal != index+1 || attempt.Started == "" {
			return fmt.Errorf("run %s attempt %d has invalid identity", run.ID, index+1)
		}
		switch attempt.State {
		case RunRunning:
			if index != len(run.Attempts)-1 || run.State != RunRunning || attempt.Finished != "" || attempt.Error != "" {
				return fmt.Errorf("run %s attempt %d has inconsistent running state", run.ID, index+1)
			}
		case RunComplete:
			if index != len(run.Attempts)-1 || run.State != RunComplete || attempt.Finished == "" || attempt.Error != "" {
				return fmt.Errorf("run %s attempt %d has inconsistent completion", run.ID, index+1)
			}
		case RunFailed, RunInterrupted:
			if attempt.Finished == "" || attempt.Error == "" {
				return fmt.Errorf("run %s attempt %d has incomplete terminal state", run.ID, index+1)
			}
		default:
			return fmt.Errorf("run %s attempt %d has unsupported state %q", run.ID, index+1, attempt.State)
		}
	}
	return nil
}

func validateProgress(progress training.Progress, resume *training.ResumePoint) error {
	if progress.Steps < 0 || progress.ConsumedTokens < 0 {
		return fmt.Errorf("negative step or token count")
	}
	if progress.LastLoss != nil && (*progress.LastLoss < 0 || math.IsNaN(*progress.LastLoss) || math.IsInf(*progress.LastLoss, 0)) {
		return fmt.Errorf("invalid last loss")
	}
	previous := int64(0)
	for _, checkpoint := range progress.Checkpoints {
		if checkpoint.Step <= previous || checkpoint.Tokens < 0 || len(checkpoint.Artifacts) == 0 {
			return fmt.Errorf("invalid checkpoint at step %d", checkpoint.Step)
		}
		previous = checkpoint.Step
	}
	if resume == nil {
		// RUN.json is committed before the aggregate model pin. A process may
		// stop between those two atomic writes; the verified newest checkpoint
		// remains recoverable directly from this run-local evidence.
	} else if len(progress.Checkpoints) == 0 || !reflect.DeepEqual(resume.Checkpoint, progress.Checkpoints[len(progress.Checkpoints)-1]) || resume.Step != resume.Checkpoint.Step || resume.Tokens != resume.Checkpoint.Tokens {
		return fmt.Errorf("resume pin is not the newest checkpoint")
	}
	previous = 0
	for _, evaluation := range progress.Evaluations {
		if evaluation.Step <= previous || evaluation.Tokens < 0 || len(evaluation.Metrics) == 0 {
			return fmt.Errorf("invalid evaluation at step %d", evaluation.Step)
		}
		for name, value := range evaluation.Metrics {
			if name == "" || math.IsNaN(value) || math.IsInf(value, 0) {
				return fmt.Errorf("invalid evaluation metric %q", name)
			}
		}
		previous = evaluation.Step
	}
	return nil
}

func modelDirectory(root, value string) (string, error) {
	if value == "~" || strings.HasPrefix(value, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		if value == "~" {
			value = home
		} else {
			value = filepath.Join(home, strings.TrimPrefix(value, "~/"))
		}
	}
	if filepath.IsAbs(value) || strings.HasPrefix(value, ".") || strings.ContainsRune(value, filepath.Separator) {
		return filepath.Abs(value)
	}
	if !validName.MatchString(value) {
		return "", fmt.Errorf("invalid model name or path %q", value)
	}
	return filepath.Join(root, value), nil
}

func runDirectoryName(pin RunPin) string {
	return fmt.Sprintf("%04d-%s-%s", pin.Ordinal, pin.Stage, pin.ID)
}

func modelBOM(record ModelRecord) ModelBOM {
	bom := ModelBOM{
		Kind: "openwaldo-bom", Schema: ModelBOMSchema, Subject: "model",
		ModelID: record.ID, Name: record.Name, PlanSHA256: record.PlanSHA256,
		ArchitectureSHA256: record.ArchitectureSHA256, Interaction: record.Interaction, PathBase: "model-root", Generated: record.Updated,
	}
	if record.OriginBOMSHA256 != "" {
		bom.Origin = &ModelBOMOrigin{BOM: "ORIGIN-BOM.json", SHA256: record.OriginBOMSHA256}
		for _, artifact := range record.OriginArtifacts {
			bom.Origin.Artifacts = append(bom.Origin.Artifacts, ModelBOMArtifact{Role: artifact.Role, Path: artifact.Path, SHA256: artifact.SHA256, Bytes: artifact.Bytes})
		}
		bom.CurrentOriginSHA256 = record.OriginBOMSHA256
	}
	if record.Parent != nil {
		parent := *record.Parent
		bom.Parent = &parent
	}
	for _, pin := range record.Runs {
		directory := filepath.ToSlash(filepath.Join("runs", runDirectoryName(pin)))
		run := ModelBOMRun{
			ID: pin.ID, Stage: pin.Stage, Ordinal: pin.Ordinal,
			RunBOM: directory + "/RUN-BOM.json", BOMSHA256: pin.BOMSHA256,
			State: pin.State, Backend: pin.Backend, Simulated: pin.Simulated,
			ObservationSHA256: pin.ObservationSHA256,
		}
		for _, artifact := range pin.Artifacts {
			run.Artifacts = append(run.Artifacts, ModelBOMArtifact{
				Role: artifactRole(artifact.Path), Path: directory + "/" + artifact.Path,
				SHA256: artifact.SHA256, Bytes: artifact.Bytes,
			})
		}
		bom.Runs = append(bom.Runs, run)
		if pin.State == RunComplete && !pin.Simulated && hasWeightArtifact(pin.Artifacts) {
			bom.CurrentRunID = pin.ID
			bom.CurrentOriginSHA256 = ""
		}
	}
	return bom
}

func verifyOriginArtifact(root string, artifact OriginArtifact) error {
	if artifact.Path == "" || filepath.IsAbs(filepath.FromSlash(artifact.Path)) {
		return fmt.Errorf("artifact path %q is not model-root-relative", artifact.Path)
	}
	clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(artifact.Path)))
	if clean != artifact.Path || clean == "." || strings.HasPrefix(clean, "../") {
		return fmt.Errorf("artifact path %q escapes the model root", artifact.Path)
	}
	return VerifyArtifactFile(filepath.Join(root, filepath.FromSlash(clean)), training.Artifact{Path: artifact.Path, SHA256: artifact.SHA256, Bytes: artifact.Bytes})
}

func artifactRole(path string) string {
	switch {
	case strings.Contains(path, "/checkpoints/") && strings.HasSuffix(path, ".safetensors"):
		return "checkpoint"
	case strings.HasSuffix(path, "/model.safetensors"):
		return "weights"
	case strings.HasSuffix(path, "/config.json"):
		return "configuration"
	case strings.HasSuffix(path, "/tokenizer.json"):
		return "tokenizer"
	case strings.HasSuffix(path, "/fake-model.json"):
		return "simulation"
	default:
		return "artifact"
	}
}

func hasWeightArtifact(artifacts []training.Artifact) bool {
	for _, artifact := range artifacts {
		if artifactRole(artifact.Path) == "weights" {
			return true
		}
	}
	return false
}

func legacyModelBOMMatches(bom ModelBOM, record ModelRecord, pins []RunPin) bool {
	if record.Parent != nil || bom.PathBase != "" || bom.CurrentRunID != "" || len(bom.Runs) != len(pins) {
		return false
	}
	for index, pin := range pins {
		run := bom.Runs[index]
		if run.ID != pin.ID || run.Stage != pin.Stage || run.Ordinal != pin.Ordinal || run.RunBOM != "" || run.BOMSHA256 != pin.BOMSHA256 || run.State != pin.State || run.Backend != (training.Identity{}) || run.Simulated || run.ObservationSHA256 != pin.ObservationSHA256 || len(run.Artifacts) != len(pin.Artifacts) {
			return false
		}
		for artifactIndex, artifact := range pin.Artifacts {
			candidate := run.Artifacts[artifactIndex]
			if candidate.Role != "" || candidate.Path != artifact.Path || candidate.SHA256 != artifact.SHA256 || candidate.Bytes != artifact.Bytes {
				return false
			}
		}
	}
	return record.ID == bom.ModelID
}

func writeJSONAtomic(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".waldo-state-*")
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

func readJSON(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func sha256Hex(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func hashJSON(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return sha256Hex(data), nil
}

func formatTime(value time.Time) string { return value.UTC().Format(time.RFC3339Nano) }

func sortPins(pins []RunPin) {
	sort.Slice(pins, func(i, j int) bool { return pins[i].Ordinal < pins[j].Ordinal })
}
