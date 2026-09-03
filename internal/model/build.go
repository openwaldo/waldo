// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/openwaldo/waldo/internal/training"
)

type Progress struct {
	Phase    string          `json:"phase"`
	Stage    string          `json:"stage,omitempty"`
	RunID    string          `json:"run_id,omitempty"`
	State    RunState        `json:"state,omitempty"`
	Message  string          `json:"message"`
	Training *training.Event `json:"training,omitempty"`
}

type Builder struct {
	Root         string
	Now          func() time.Time
	NewID        func() (string, error)
	Resolver     training.Resolver
	OriginPuller *Puller
	Progress     func(Progress)
	ComposeName  string
	MultiNode    MultiNodeHandoff
}

type MultiNodeHandoff struct {
	RendezvousID string
	Nodes        int
	StageOrdinal int
	StageCount   int
}

// ResolveBackend selects and validates the same training harness used by a
// build without creating model or run state.
func (builder Builder) ResolveBackend(ctx context.Context, architecture Architecture, objectives []string) (training.Selection, error) {
	architectureJSON, err := json.Marshal(architecture)
	if err != nil {
		return training.Selection{}, err
	}
	resolver := builder.Resolver
	if resolver == nil {
		resolver = builtinResolver()
	}
	selection, err := resolver.Resolve(ctx, training.ResolveRequest{Architecture: architectureJSON, Objectives: objectives})
	if err != nil {
		return training.Selection{}, fmt.Errorf("resolve training backend: %w", err)
	}
	if err := validateSelection(selection, objectives); err != nil {
		return training.Selection{}, err
	}
	return selection, nil
}

// CheckBackend validates that this host can execute the requested portable
// architecture and objectives before callers materialize any corpus objects.
func (builder Builder) CheckBackend(ctx context.Context, architecture Architecture, objectives []string) error {
	_, err := builder.ResolveBackend(ctx, architecture, objectives)
	return err
}

func ValidateName(name string) error {
	if !validName.MatchString(name) {
		return fmt.Errorf("model name must match %s", validName.String())
	}
	return nil
}

// Initialize creates an untrained model with an immutable architecture.
func (builder Builder) Initialize(name string, architecture Architecture) (Inspection, error) {
	return builder.initialize(name, architecture, Interaction{})
}

func (builder Builder) initialize(name string, architecture Architecture, interaction Interaction) (Inspection, error) {
	if builder.Root == "" {
		return Inspection{}, fmt.Errorf("model root is required")
	}
	if err := ValidateName(name); err != nil {
		return Inspection{}, err
	}
	if err := architecture.Validate(); err != nil {
		return Inspection{}, err
	}
	if err := interaction.Validate(); err != nil {
		return Inspection{}, err
	}
	modelPath := filepath.Join(builder.Root, name)
	if _, err := os.Stat(modelPath); err == nil {
		return Inspection{}, fmt.Errorf("model %q already exists", name)
	} else if !os.IsNotExist(err) {
		return Inspection{}, err
	}
	plan, err := composePlan(name, Compose{Architecture: architecture, Interaction: interaction})
	if err != nil {
		return Inspection{}, err
	}
	planHash, err := hashJSON(plan)
	if err != nil {
		return Inspection{}, err
	}
	now := builder.clock()
	created := now()
	record := ModelRecord{
		Kind: "waldo-model", Schema: ModelSchema, ID: planHash, Name: name,
		PlanSHA256: planHash, ArchitectureSHA256: plan.ArchitectureSHA256,
		Architecture: plan.Architecture, Interaction: plan.Interaction, Forecast: plan.Forecast,
		Created: formatTime(created), Updated: formatTime(created),
	}
	if err := initializeModel(builder.Root, modelPath, plan, record); err != nil {
		return Inspection{}, err
	}
	builder.report(Progress{Phase: "model", Message: "persisted immutable model architecture and OpenWALDO BOM"})
	return Inspect(builder.Root, name)
}

// Train appends one explicit run to an existing model. The model architecture
// and identity remain unchanged; the aggregate model BOM gains a run pin.
func (builder Builder) Train(ctx context.Context, name string, prepared PreparedStage) (Inspection, error) {
	inspection, err := Inspect(builder.Root, name)
	if err != nil {
		return Inspection{}, err
	}
	stage := prepared.Stage
	stage = stageWithInteraction(stage, inspection.Model.Interaction)
	prepared.Stage = stage
	prepared, err = PrepareStage(stage, prepared.BOM, prepared.Inputs)
	if err != nil {
		return Inspection{}, err
	}
	if err := validateStage(stage, inspection.Model.Architecture); err != nil {
		return Inspection{}, err
	}
	if err := prepared.BOM.Validate(); err != nil {
		return Inspection{}, fmt.Errorf("stage %s corpus OpenWALDO BOM: %w", stage.Name, err)
	}
	if len(prepared.Inputs) == 0 {
		return Inspection{}, fmt.Errorf("stage %s has no verified shard inputs", stage.Name)
	}
	resolvedParameters, err := stage.ResolvePlanningParameters()
	if err != nil {
		return Inspection{}, fmt.Errorf("stage %s training profile: %w", stage.Name, err)
	}
	if resolvedParameters.Data.Order == "corpus-weighted-shuffle-v1" {
		resolvedParameters.Data.CorpusWeights, err = resolveCorpusWeights(resolvedParameters.Data.CorpusWeights, prepared.BOM.Paths)
		if err != nil {
			return Inspection{}, fmt.Errorf("stage %s %w", stage.Name, err)
		}
	}
	codec, err := training.ResolveTokenizerCodec(inspection.Model.Architecture.Tokenizer.Name)
	if err != nil {
		return Inspection{}, fmt.Errorf("stage %s tokenizer: %w", stage.Name, err)
	}
	builder.report(Progress{Phase: "preflight", Stage: stage.Name, Message: fmt.Sprintf("selecting deterministic held-out records across %d shards", len(prepared.Inputs))})
	conversation := training.ConversationTransform{}
	if stage.Conversation != nil {
		conversation = *stage.Conversation
	}
	partition, err := training.NewRecordPartitionContextWithTransform(ctx, prepared.Inputs, resolvedParameters, codec, stage.Objective, conversation, func(event training.PartitionProgress) {
		builder.report(Progress{Phase: "preflight", Stage: stage.Name, Message: fmt.Sprintf("evaluation selection %d/%d shards, %d records indexed", event.CurrentShard, event.TotalShards, event.Records)})
	})
	if err != nil {
		return Inspection{}, fmt.Errorf("stage %s held-out evaluation partition: %w", stage.Name, err)
	}
	builder.report(Progress{Phase: "preflight", Stage: stage.Name, Message: fmt.Sprintf("selected %d held-out records (%s text)", partition.Evaluation.Records, byteCount(partition.Evaluation.TextBytes))})
	if stage.Parameters.Steps == 0 && stage.Parameters.Tokens == 0 {
		builder.report(Progress{Phase: "preflight", Stage: stage.Name, Message: fmt.Sprintf("deriving optimizer steps from %d epochs", resolvedParameters.Epochs)})
		derivedSteps, err := partition.TrainingSteps(ctx)
		if err != nil {
			return Inspection{}, fmt.Errorf("stage %s derive epoch training steps: %w", stage.Name, err)
		}
		resolvedParameters, err = stage.ResolveParametersForSteps(derivedSteps)
		if err != nil {
			return Inspection{}, fmt.Errorf("stage %s resolve epoch training parameters: %w", stage.Name, err)
		}
		builder.report(Progress{Phase: "preflight", Stage: stage.Name, Message: fmt.Sprintf("planned %d optimizer steps from %d epochs", derivedSteps, resolvedParameters.Epochs)})
	} else if stage.Parameters.Epochs > 0 {
		builder.report(Progress{Phase: "preflight", Stage: stage.Name, Message: fmt.Sprintf("validating capacity for %d optimizer steps across %d explicit epochs", resolvedParameters.Steps, resolvedParameters.Epochs)})
		availableSteps, sufficient, err := partition.TrainingStepCapacity(ctx, resolvedParameters.Steps)
		if err != nil {
			return Inspection{}, fmt.Errorf("stage %s training capacity: %w", stage.Name, err)
		}
		if !sufficient {
			return Inspection{}, fmt.Errorf("stage %s requests %d optimizer steps, but its filtered training stream provides only %d across %d epochs; reduce steps, increase epochs, or select more data", stage.Name, resolvedParameters.Steps, availableSteps, resolvedParameters.Epochs)
		}
		builder.report(Progress{Phase: "preflight", Stage: stage.Name, Message: fmt.Sprintf("verified capacity for %d optimizer steps", resolvedParameters.Steps)})
	}
	records, err := partition.TrainingRecords()
	if err != nil {
		return Inspection{}, fmt.Errorf("stage %s training record stream: %w", stage.Name, err)
	}
	evaluationRecords := partition.EvaluationRecords()
	architectureJSON, err := json.Marshal(inspection.Model.Architecture)
	if err != nil {
		return Inspection{}, err
	}
	resolver := builder.Resolver
	if resolver == nil {
		resolver = builtinResolver()
	}
	builder.report(Progress{Phase: "backend", Stage: stage.Name, Message: "verifying the selected training backend"})
	selection, err := resolver.Resolve(ctx, training.ResolveRequest{
		ArchitectureSHA256: inspection.Model.ArchitectureSHA256,
		Architecture:       architectureJSON,
		Objectives:         []string{stage.Objective},
	})
	if err != nil {
		return Inspection{}, fmt.Errorf("resolve training backend: %w", err)
	}
	if err := validateSelection(selection, []string{stage.Objective}); err != nil {
		return Inspection{}, err
	}
	builder.report(Progress{Phase: "backend", Stage: stage.Name, Message: fmt.Sprintf("selected %s@%s", selection.Execution.Backend.Name, selection.Execution.Backend.Revision)})
	var initialization *training.Initialization
	if selection.Execution.Backend.Name != training.BackendFake {
		initialization, err = resolveInitialization(inspection)
		if err != nil {
			return Inspection{}, err
		}
	}

	bomHash, err := hashJSON(prepared.BOM)
	if err != nil {
		return Inspection{}, err
	}
	if candidate, ok := resumableRun(inspection, stage, resolvedParameters, partition.Evaluation, bomHash, selection.Execution); ok {
		return builder.resumeTraining(ctx, name, inspection, candidate, stage, prepared, records, evaluationRecords, architectureJSON, selection)
	}

	runID, err := builder.identifier()()
	if err != nil {
		return Inspection{}, err
	}
	ordinal := len(inspection.Model.Runs) + 1
	pin := RunPin{ID: runID, Stage: stage.Name, Ordinal: ordinal, State: RunPlanned, Backend: selection.Execution.Backend, Simulated: selection.Execution.Backend.Name == training.BackendFake}
	runDirectory := filepath.Join(inspection.Path, "runs", runDirectoryName(pin))
	runBOM := RunBOM{
		Kind: "openwaldo-bom", Schema: RunBOMSchema, Subject: "training-run",
		ID: runID, ModelID: inspection.Model.ID, Stage: stage.Name, StageType: stage.Type,
		Ordinal: ordinal, Objective: stage.Objective, Execution: selection.Execution,
		ArchitectureSHA256: inspection.Model.ArchitectureSHA256,
		CorpusBOMSHA256:    bomHash, CorpusBOM: prepared.BOM, Parameters: resolvedParameters,
		EvaluationSet: &partition.Evaluation, Initialization: initialization,
	}
	if stage.Conversation != nil {
		runBOM.Conversation = *stage.Conversation
	}
	runBOMHash, err := hashJSON(runBOM)
	if err != nil {
		return Inspection{}, err
	}
	pin.BOMSHA256 = runBOMHash
	now := builder.clock()
	run := RunRecord{Kind: "waldo-training-run", Schema: RunSchema, ID: runID, State: RunPlanned, BOMSHA256: runBOMHash, Planned: formatTime(now())}
	if err := writeJSONAtomic(filepath.Join(runDirectory, "RUN-BOM.json"), runBOM); err != nil {
		return Inspection{}, err
	}
	if err := writeJSONAtomic(filepath.Join(runDirectory, "RUN.json"), run); err != nil {
		return Inspection{}, err
	}
	record := inspection.Model
	record.Runs = append(record.Runs, pin)
	if err := persistModel(inspection.Path, &record, now()); err != nil {
		return Inspection{}, err
	}
	builder.report(Progress{Phase: "run", Stage: pin.Stage, RunID: runID, State: RunPlanned, Message: "persisted run OpenWALDO BOM"})
	return builder.executeTrainingAttempt(ctx, name, inspection.Path, &record, pin, run, runBOM, stage, prepared, records, evaluationRecords, architectureJSON, selection, nil)
}

func byteCount(value int64) string {
	const unit = int64(1024)
	if value < unit {
		return fmt.Sprintf("%d B", value)
	}
	units := []string{"KiB", "MiB", "GiB", "TiB"}
	amount := float64(value)
	for _, label := range units {
		amount /= 1024
		if amount < 1024 || label == units[len(units)-1] {
			return fmt.Sprintf("%.1f %s", amount, label)
		}
	}
	return fmt.Sprintf("%d B", value)
}

func resumableRun(inspection Inspection, stage Stage, parameters training.ResolvedParameters, evaluation training.EvaluationSet, corpusHash string, execution training.Execution) (int, bool) {
	if len(inspection.Runs) == 0 {
		return 0, false
	}
	index := len(inspection.Runs) - 1
	run := inspection.Runs[index]
	bom := inspection.RunBOMs[index]
	conversation := training.ConversationTransform{}
	if stage.Conversation != nil {
		conversation = *stage.Conversation
	}
	if !resumableRunState(run, parameters) || bom.Stage != stage.Name || bom.StageType != stage.Type || bom.Objective != stage.Objective || !reflect.DeepEqual(bom.Conversation, conversation) || bom.CorpusBOMSHA256 != corpusHash || !equivalentTrainingParameters(bom.Parameters, parameters) || bom.EvaluationSet == nil || !reflect.DeepEqual(*bom.EvaluationSet, evaluation) || !reflect.DeepEqual(bom.Execution, execution) {
		return 0, false
	}
	return index, true
}

func resumableRunState(run RunRecord, parameters training.ResolvedParameters) bool {
	if run.State == RunInterrupted {
		return true
	}
	// Releases before the duplicate-evaluation fix could successfully verify a
	// final artifact and then mark the run failed while persisting the repeated
	// evaluation at its resume step. Permit only that exact, checkpoint-backed
	// bookkeeping failure to resume; ordinary failed runs remain terminal.
	if run.State != RunFailed || !strings.HasPrefix(run.Error, "persist training progress: evaluation step ") || !strings.HasSuffix(run.Error, " does not advance durable progress") || run.Progress == nil || len(run.Progress.Checkpoints) == 0 {
		return false
	}
	checkpoint := run.Progress.Checkpoints[len(run.Progress.Checkpoints)-1]
	return checkpoint.Step == parameters.Steps && checkpoint.Tokens == parameters.PlannedTokenCapacity
}

// HasRecoverableFinalizationFailure identifies the narrowly scoped failure
// produced by WALDO releases that rejected a repeated final-step evaluation.
func HasRecoverableFinalizationFailure(inspection Inspection) bool {
	if len(inspection.Runs) == 0 || len(inspection.RunBOMs) != len(inspection.Runs) {
		return false
	}
	last := len(inspection.Runs) - 1
	return resumableRunState(inspection.Runs[last], inspection.RunBOMs[last].Parameters)
}

func (builder Builder) resumeTraining(ctx context.Context, name string, inspection Inspection, index int, stage Stage, prepared PreparedStage, records, evaluationRecords training.RecordSource, architectureJSON json.RawMessage, selection training.Selection) (Inspection, error) {
	pin := inspection.Model.Runs[index]
	run := inspection.Runs[index]
	runBOM := inspection.RunBOMs[index]
	if run.Progress != nil && len(run.Progress.Checkpoints) > 0 && !selection.Backend.Descriptor().Capabilities.CheckpointResume {
		return Inspection{}, fmt.Errorf("stage %s has a resumable checkpoint, but backend %s@%s cannot restore optimizer state", stage.Name, pin.Backend.Name, pin.Backend.Revision)
	}
	var resume *training.ResumePoint
	if pin.Resume != nil {
		resume = cloneResumePoint(pin.Resume)
	} else if run.Progress != nil && len(run.Progress.Checkpoints) > 0 {
		checkpoint := run.Progress.Checkpoints[len(run.Progress.Checkpoints)-1]
		resume = &training.ResumePoint{Step: checkpoint.Step, Tokens: checkpoint.Tokens, Checkpoint: checkpoint}
	}
	if resume != nil {
		runDirectory := filepath.Join(inspection.Path, "runs", runDirectoryName(pin))
		for _, artifact := range resume.Checkpoint.Artifacts {
			path := filepath.Join(runDirectory, filepath.FromSlash(artifact.Path))
			if err := VerifyArtifactFile(path, artifact); err != nil {
				return Inspection{}, fmt.Errorf("resume run %s: %w", pin.ID, err)
			}
			resume.Paths = append(resume.Paths, path)
		}
	}
	if resume == nil && runBOM.Initialization != nil {
		resolved, err := resolveInitialization(inspection)
		if err != nil {
			return Inspection{}, err
		}
		if resolved == nil || resolved.SourceType != runBOM.Initialization.SourceType || resolved.SourceID != runBOM.Initialization.SourceID || resolved.SourceRunID != runBOM.Initialization.SourceRunID || !reflect.DeepEqual(resolved.Artifact, runBOM.Initialization.Artifact) {
			return Inspection{}, fmt.Errorf("resume run %s: pinned initialization is no longer the current verified source", pin.ID)
		}
		runBOM.Initialization = resolved
	}
	record := inspection.Model
	builder.report(Progress{Phase: "run", Stage: pin.Stage, RunID: pin.ID, State: RunInterrupted, Message: fmt.Sprintf("resuming existing run from step %d", resumeStep(resume))})
	return builder.executeTrainingAttempt(ctx, name, inspection.Path, &record, pin, run, runBOM, stage, prepared, records, evaluationRecords, architectureJSON, selection, resume)
}

func (builder Builder) executeTrainingAttempt(ctx context.Context, name, modelPath string, record *ModelRecord, pin RunPin, run RunRecord, runBOM RunBOM, stage Stage, prepared PreparedStage, records, evaluationRecords training.RecordSource, architectureJSON json.RawMessage, selection training.Selection, resume *training.ResumePoint) (Inspection, error) {
	tokenizerSpec := training.TokenizerSpec{Name: record.Architecture.Tokenizer.Name, Revision: record.Architecture.Tokenizer.Revision, VocabularySize: int(record.Architecture.VocabularySize), PadID: 0, BOSID: 1, EOSID: 2}
	if selection.Execution.Backend.Name == training.BackendPyTorch || selection.Execution.Backend.Name == training.BackendTorchTitan || selection.Execution.Backend.Name == training.BackendMLX {
		var err error
		tokenizerSpec, _, err = training.ResolveTokenizer(record.Architecture.Tokenizer.Name, record.Architecture.Tokenizer.Revision, record.Architecture.VocabularySize)
		if err != nil {
			return Inspection{}, fmt.Errorf("stage %s tokenizer: %w", stage.Name, err)
		}
	}
	if builder.MultiNode.RendezvousID != "" {
		if err := builder.publishMultiNodePlan(pin, runBOM, prepared, architectureJSON, stage, resume); err != nil {
			return Inspection{}, err
		}
		defer os.RemoveAll(filepath.Dir(MultiNodePlanPath(builder.Root, builder.MultiNode.RendezvousID)))
	}
	now := builder.clock()
	runDirectory := filepath.Join(modelPath, "runs", runDirectoryName(pin))
	telemetryPath := filepath.Join(runDirectory, TelemetryFilename)
	attemptStarted := now()
	run.State = RunRunning
	run.Finished = ""
	run.Error = ""
	if run.Started == "" {
		run.Started = formatTime(attemptStarted)
	}
	run.Attempts = append(run.Attempts, RunAttempt{Ordinal: len(run.Attempts) + 1, Started: formatTime(attemptStarted), State: RunRunning, ResumeStep: resumeStep(resume)})
	if err := persistRunAndPin(modelPath, runDirectory, record, pin, run, now()); err != nil {
		return Inspection{}, err
	}
	attemptOrdinal := len(run.Attempts)
	if err := appendTelemetry(telemetryPath, telemetryRow{Observed: attemptStarted, Started: attemptStarted, RunID: pin.ID, Stage: stage.Name, Attempt: attemptOrdinal, Event: "run", State: RunRunning, PlannedSteps: runBOM.Parameters.Steps, PlannedTokens: runBOM.Parameters.PlannedTokenCapacity, Message: "training backend started"}); err != nil {
		return Inspection{}, telemetryError(telemetryPath, err)
	}
	builder.report(Progress{Phase: "run", Stage: pin.Stage, RunID: pin.ID, State: RunRunning, Message: "training backend started"})

	var progressMutex sync.Mutex
	var progressErr error
	report := func(event training.Event) {
		progressMutex.Lock()
		defer progressMutex.Unlock()
		if progressErr == nil {
			observed := now()
			if err := appendTelemetry(telemetryPath, telemetryRow{Observed: observed, Started: attemptStarted, RunID: pin.ID, Stage: stage.Name, Attempt: attemptOrdinal, Event: event.Kind, State: RunRunning, PlannedSteps: runBOM.Parameters.Steps, PlannedTokens: runBOM.Parameters.PlannedTokenCapacity, Training: &event, Message: event.Message}); err != nil {
				progressErr = telemetryError(telemetryPath, err)
			} else {
				progressErr = persistTrainingEvent(modelPath, runDirectory, record, pin, &run, event, observed)
			}
		}
		builder.report(Progress{Phase: "training", Stage: stage.Name, RunID: pin.ID, State: RunRunning, Message: event.Message, Training: &event})
	}
	artifactPrefix := "artifacts"
	observation, backendErr := selection.Backend.Run(ctx, training.Request{
		RunID: pin.ID, Stage: stage.Name, Objective: stage.Objective,
		Conversation:       runBOM.Conversation,
		ArchitectureSHA256: record.ArchitectureSHA256,
		Architecture:       architectureJSON, Tokenizer: tokenizerSpec, BOM: prepared.BOM, Inputs: prepared.Inputs,
		Parameters: runBOM.Parameters, Records: records, EvaluationRecords: evaluationRecords, EvaluationSet: EvaluationSetValue(runBOM.EvaluationSet), Initialization: initializationForAttempt(runBOM.Initialization, resume), Resume: resume,
		ArtifactDirectory: filepath.Join(runDirectory, artifactPrefix), ArtifactPrefix: artifactPrefix, Report: report,
	})
	progressMutex.Lock()
	if progressErr != nil {
		backendErr = errors.Join(backendErr, fmt.Errorf("persist training progress: %w", progressErr))
	}
	progressMutex.Unlock()
	if backendErr == nil {
		observation = mergeProgress(run.Progress, observation)
		plannedParameters := stage.Parameters
		plannedParameters.Steps = runBOM.Parameters.Steps
		planned := PlannedStage{Name: stage.Name, Parameters: plannedParameters, PlannedTokens: runBOM.Parameters.PlannedTokenCapacity}
		if err := validateBackendObservation(runDirectory, planned, observation); err != nil {
			backendErr = fmt.Errorf("invalid backend observation: %w", err)
		} else if set := runBOM.EvaluationSet; set != nil && set.Records > 0 && runBOM.Parameters.EvaluateEvery > 0 {
			if len(observation.Evaluations) == 0 {
				backendErr = fmt.Errorf("invalid backend observation: held-out evaluation was configured but no metrics were reported")
			} else {
				for index, evaluation := range observation.Evaluations {
					if _, ok := evaluation.Metrics["heldout_loss"]; !ok {
						backendErr = fmt.Errorf("invalid backend observation: evaluation %d does not report heldout_loss", index+1)
						break
					}
				}
				if backendErr == nil && selection.Execution.Backend.Name == training.BackendPyTorch {
					metrics := observation.Evaluations[len(observation.Evaluations)-1].Metrics
					if _, ok := metrics["artifact_heldout_loss"]; !ok {
						backendErr = fmt.Errorf("invalid backend observation: final PyTorch evaluation does not verify the persisted model artifact")
					}
				}
			}
		}
		if backendErr == nil && (runBOM.Parameters.Data.Order == "corpus-balanced-shuffle-v1" || runBOM.Parameters.Data.Order == "corpus-weighted-shuffle-v1") {
			var consumed int64
			seen := map[string]bool{}
			for _, item := range observation.Consumption {
				if item.Corpus == "" || item.TokenTargets <= 0 || seen[item.Corpus] {
					backendErr = fmt.Errorf("invalid backend observation: invalid corpus consumption evidence")
					break
				}
				seen[item.Corpus] = true
				consumed += item.TokenTargets
			}
			if backendErr == nil && (len(seen) != len(runBOM.CorpusBOM.Paths) || consumed != observation.ConsumedTokens) {
				backendErr = fmt.Errorf("invalid backend observation: corpus consumption accounts for %d corpora and %d of %d token targets", len(seen), consumed, observation.ConsumedTokens)
			}
		}
	}
	run.Finished = formatTime(now())
	attempt := &run.Attempts[len(run.Attempts)-1]
	attempt.Finished = run.Finished
	if backendErr != nil {
		run.State = RunFailed
		if errors.Is(backendErr, context.Canceled) || errors.Is(backendErr, context.DeadlineExceeded) {
			run.State = RunInterrupted
		}
		run.Error = backendErr.Error()
		attempt.State = run.State
		attempt.Error = run.Error
		if err := appendTelemetry(telemetryPath, telemetryRow{Observed: now(), Started: attemptStarted, RunID: pin.ID, Stage: stage.Name, Attempt: attemptOrdinal, Event: "run", State: run.State, PlannedSteps: runBOM.Parameters.Steps, PlannedTokens: runBOM.Parameters.PlannedTokenCapacity, Message: run.Error}); err != nil {
			backendErr = errors.Join(backendErr, telemetryError(telemetryPath, err))
		}
		if err := persistRunAndPin(modelPath, runDirectory, record, pin, run, now()); err != nil {
			return Inspection{}, errors.Join(backendErr, err)
		}
		builder.report(Progress{Phase: "run", Stage: pin.Stage, RunID: pin.ID, State: run.State, Message: run.Error})
		return Inspection{}, fmt.Errorf("stage %s: %w", pin.Stage, backendErr)
	}
	run.State = RunComplete
	attempt.State = RunComplete
	run.Observation = &observation
	run.Progress = nil
	clearResumePin(record, pin.ID)
	if err := appendTelemetry(telemetryPath, telemetryRow{Observed: now(), Started: attemptStarted, RunID: pin.ID, Stage: stage.Name, Attempt: attemptOrdinal, Event: "run", State: RunComplete, PlannedSteps: runBOM.Parameters.Steps, PlannedTokens: runBOM.Parameters.PlannedTokenCapacity, Message: "training complete"}); err != nil {
		return Inspection{}, telemetryError(telemetryPath, err)
	}
	if err := persistRunAndPin(modelPath, runDirectory, record, pin, run, now()); err != nil {
		return Inspection{}, err
	}
	builder.report(Progress{Phase: "run", Stage: pin.Stage, RunID: pin.ID, State: RunComplete, Message: "persisted training observations and artifact hashes"})
	return Inspect(builder.Root, name)
}

func resumeStep(resume *training.ResumePoint) int64 {
	if resume == nil {
		return 0
	}
	return resume.Step
}

func EvaluationSetValue(value *training.EvaluationSet) training.EvaluationSet {
	if value == nil {
		return training.EvaluationSet{}
	}
	return *value
}

func initializationForAttempt(initialization *training.Initialization, resume *training.ResumePoint) *training.Initialization {
	if resume != nil {
		return nil
	}
	return initialization
}

func (builder Builder) publishMultiNodePlan(pin RunPin, runBOM RunBOM, prepared PreparedStage, architectureJSON json.RawMessage, stage Stage, resume *training.ResumePoint) error {
	if resume != nil {
		return fmt.Errorf("stage %s: multi-node training cannot resume an interrupted run; restart it fresh", stage.Name)
	}
	if builder.MultiNode.StageOrdinal < 1 || builder.MultiNode.StageCount < builder.MultiNode.StageOrdinal {
		return fmt.Errorf("stage %s: multi-node stage accounting %d/%d is invalid", stage.Name, builder.MultiNode.StageOrdinal, builder.MultiNode.StageCount)
	}
	if builder.MultiNode.Nodes < 2 {
		return fmt.Errorf("stage %s: multi-node plan would publish %d nodes; a multi-node run needs at least two", stage.Name, builder.MultiNode.Nodes)
	}
	path := MultiNodePlanPath(builder.Root, builder.MultiNode.RendezvousID)
	if _, err := os.Stat(path); err == nil {
		return fmt.Errorf("stage %s: a multi-node plan already exists at %s (leftover from a previous run); remove it or use a fresh --rendezvous-id", stage.Name, path)
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("stage %s: check for an existing multi-node plan: %w", stage.Name, err)
	}
	plan := MultiNodePlan{
		Kind: MultiNodePlanKind, Schema: MultiNodePlanSchema,
		RunID: pin.ID, Stage: stage.Name, StageOrdinal: builder.MultiNode.StageOrdinal, StageCount: builder.MultiNode.StageCount,
		Nodes: builder.MultiNode.Nodes, Objective: stage.Objective,
		ArchitectureSHA256: runBOM.ArchitectureSHA256, Architecture: architectureJSON,
		Parameters: runBOM.Parameters, CorpusBOM: prepared.BOM,
		EvaluationSet: runBOM.EvaluationSet, Initialization: runBOM.Initialization,
	}
	if stage.Conversation != nil {
		plan.Conversation = *stage.Conversation
	}
	if runBOM.Initialization != nil {
		if runBOM.Initialization.Path == "" {
			return fmt.Errorf("stage %s: initialization weights have no path to share with secondary nodes", stage.Name)
		}
		relative, err := filepath.Rel(builder.Root, runBOM.Initialization.Path)
		if err != nil || !filepath.IsLocal(relative) {
			return fmt.Errorf("stage %s: initialization weights at %s are outside the shared model root %s; secondary nodes cannot reach them", stage.Name, runBOM.Initialization.Path, builder.Root)
		}
		plan.InitializationPath = filepath.ToSlash(relative)
	}
	if err := writeJSONAtomic(path, plan); err != nil {
		return fmt.Errorf("publish multi-node plan for secondary nodes: %w", err)
	}
	builder.report(Progress{Phase: "run", Stage: pin.Stage, RunID: pin.ID, State: RunRunning, Message: "published multi-node plan for secondary nodes"})
	return nil
}

func cloneResumePoint(value *training.ResumePoint) *training.ResumePoint {
	if value == nil {
		return nil
	}
	clone := *value
	clone.Checkpoint.Artifacts = append([]training.Artifact(nil), value.Checkpoint.Artifacts...)
	clone.Paths = append([]string(nil), value.Paths...)
	return &clone
}

func clearResumePin(record *ModelRecord, runID string) {
	for index := range record.Runs {
		if record.Runs[index].ID == runID {
			record.Runs[index].Resume = nil
			return
		}
	}
}

func persistTrainingEvent(modelPath, runDirectory string, record *ModelRecord, pin RunPin, run *RunRecord, event training.Event, now time.Time) error {
	if event.Kind != "checkpoint" && event.Kind != "evaluation" {
		return nil
	}
	if run.Progress == nil {
		run.Progress = &training.Progress{}
	}
	run.Progress.Steps = max(run.Progress.Steps, event.Step)
	run.Progress.ConsumedTokens = max(run.Progress.ConsumedTokens, event.Tokens)
	if event.Loss != nil {
		loss := *event.Loss
		run.Progress.LastLoss = &loss
	}
	if event.Checkpoint != nil {
		checkpoint := *event.Checkpoint
		checkpoint.Artifacts = append([]training.Artifact(nil), event.Checkpoint.Artifacts...)
		if len(run.Progress.Checkpoints) > 0 && checkpoint.Step <= run.Progress.Checkpoints[len(run.Progress.Checkpoints)-1].Step {
			if reflect.DeepEqual(checkpoint, run.Progress.Checkpoints[len(run.Progress.Checkpoints)-1]) {
				return nil
			}
			return fmt.Errorf("checkpoint step %d does not advance durable progress", checkpoint.Step)
		}
		for _, artifact := range checkpoint.Artifacts {
			clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(artifact.Path)))
			if artifact.Path == "" || filepath.IsAbs(filepath.FromSlash(artifact.Path)) || clean != artifact.Path || !strings.HasPrefix(clean, "artifacts/checkpoints/") {
				return fmt.Errorf("checkpoint step %d artifact path %q is not canonical beneath artifacts/checkpoints/", checkpoint.Step, artifact.Path)
			}
			if err := VerifyArtifactFile(filepath.Join(runDirectory, filepath.FromSlash(artifact.Path)), artifact); err != nil {
				return fmt.Errorf("checkpoint step %d: %w", checkpoint.Step, err)
			}
		}
		run.Progress.Checkpoints = append(run.Progress.Checkpoints, checkpoint)
		resume := &training.ResumePoint{Step: checkpoint.Step, Tokens: checkpoint.Tokens, Checkpoint: checkpoint}
		for index := range record.Runs {
			if record.Runs[index].ID == pin.ID {
				record.Runs[index].Resume = resume
				break
			}
		}
	}
	if event.Evaluation != nil {
		evaluation := *event.Evaluation
		evaluation.Metrics = cloneMetrics(event.Evaluation.Metrics)
		changed, err := updateDurableEvaluation(run, evaluation)
		if err != nil {
			return err
		}
		if !changed {
			return nil
		}
	}
	return persistRunAndPin(modelPath, runDirectory, record, pin, *run, now)
}

func updateDurableEvaluation(run *RunRecord, evaluation training.Evaluation) (bool, error) {
	if len(run.Progress.Evaluations) == 0 {
		run.Progress.Evaluations = append(run.Progress.Evaluations, evaluation)
		return true, nil
	}
	last := len(run.Progress.Evaluations) - 1
	previous := run.Progress.Evaluations[last]
	if evaluation.Step > previous.Step {
		run.Progress.Evaluations = append(run.Progress.Evaluations, evaluation)
		return true, nil
	}
	if reflect.DeepEqual(evaluation, previous) {
		return false, nil
	}
	if evaluation.Step == previous.Step && len(run.Attempts) > 0 && run.Attempts[len(run.Attempts)-1].ResumeStep == evaluation.Step {
		// Re-evaluation after restoring the same checkpoint can differ slightly
		// because of accelerator arithmetic. It replaces the prior live metric;
		// the completion observation subsequently enriches it with artifact metrics.
		run.Progress.Evaluations[last] = evaluation
		return true, nil
	}
	return false, fmt.Errorf("evaluation step %d does not advance durable progress", evaluation.Step)
}

func mergeProgress(progress *training.Progress, observation training.Observation) training.Observation {
	if progress == nil {
		return observation
	}
	checkpoints := append([]training.Checkpoint(nil), progress.Checkpoints...)
	for _, checkpoint := range observation.Checkpoints {
		if len(checkpoints) == 0 || checkpoint.Step > checkpoints[len(checkpoints)-1].Step {
			checkpoints = append(checkpoints, checkpoint)
		}
	}
	evaluations := append([]training.Evaluation(nil), progress.Evaluations...)
	for _, evaluation := range observation.Evaluations {
		if len(evaluations) > 0 && evaluation.Step == evaluations[len(evaluations)-1].Step {
			// The completion observation may enrich the already-durable event at
			// the same step with post-save artifact verification metrics.
			evaluations[len(evaluations)-1] = evaluation
		} else if len(evaluations) == 0 || evaluation.Step > evaluations[len(evaluations)-1].Step {
			evaluations = append(evaluations, evaluation)
		}
	}
	observation.Checkpoints = checkpoints
	observation.Evaluations = evaluations
	return observation
}

func cloneMetrics(metrics map[string]float64) map[string]float64 {
	result := make(map[string]float64, len(metrics))
	for name, value := range metrics {
		result[name] = value
	}
	return result
}

func resolveInitialization(inspection Inspection) (*training.Initialization, error) {
	if initialization, err := resolveRunInitialization(inspection, ""); initialization != nil || err != nil {
		return initialization, err
	}
	if inspection.Model.Parent != nil {
		parent := inspection.Model.Parent
		path := filepath.Join(inspection.Path, filepath.FromSlash(parent.Artifact.Path))
		if err := VerifyArtifactFile(path, parent.Artifact); err != nil {
			return nil, fmt.Errorf("initialize from parent run %s: %w", parent.RunID, err)
		}
		return &training.Initialization{SourceType: "run", SourceID: parent.ModelID, SourceRunID: parent.RunID, Artifact: parent.Artifact, Path: path}, nil
	}
	if inspection.Origin != nil {
		for _, artifact := range inspection.Origin.Artifacts {
			if artifact.Role != "weights" {
				continue
			}
			path := filepath.Join(inspection.Path, filepath.FromSlash(artifact.Path))
			trainingArtifact := training.Artifact{Path: artifact.Path, SHA256: artifact.SHA256, Bytes: artifact.Bytes}
			if err := VerifyArtifactFile(path, trainingArtifact); err != nil {
				return nil, fmt.Errorf("initialize from model origin %s: %w", inspection.Model.OriginBOMSHA256, err)
			}
			return &training.Initialization{SourceType: "origin", SourceID: inspection.Model.OriginBOMSHA256, Artifact: trainingArtifact, Path: path}, nil
		}
		return nil, fmt.Errorf("model origin %s has no weights artifact", inspection.Model.OriginBOMSHA256)
	}
	return nil, nil
}

func resolveRunInitialization(inspection Inspection, requestedRunID string) (*training.Initialization, error) {
	for index := len(inspection.Runs) - 1; index >= 0; index-- {
		run := inspection.Runs[index]
		if requestedRunID != "" && run.ID != requestedRunID {
			continue
		}
		if run.State != RunComplete || run.Observation == nil || run.Observation.Simulated {
			continue
		}
		for _, artifact := range run.Observation.Artifacts {
			if artifact.Path != "artifacts/model.safetensors" {
				continue
			}
			var pin RunPin
			for _, candidate := range inspection.Model.Runs {
				if candidate.ID == run.ID {
					pin = candidate
					break
				}
			}
			if pin.ID == "" {
				return nil, fmt.Errorf("initialize from run %s: model run pin is missing", run.ID)
			}
			path := filepath.Join(inspection.Path, "runs", runDirectoryName(pin), filepath.FromSlash(artifact.Path))
			if err := VerifyArtifactFile(path, artifact); err != nil {
				return nil, fmt.Errorf("initialize from run %s: %w", run.ID, err)
			}
			return &training.Initialization{SourceType: "run", SourceID: run.ID, SourceRunID: run.ID, Artifact: artifact, Path: path}, nil
		}
	}
	if requestedRunID != "" {
		return nil, fmt.Errorf("model %q has no complete real weights for run %s", inspection.Model.Name, requestedRunID)
	}
	return nil, nil
}

type composeTransaction struct {
	Kind       string                    `json:"kind"`
	Schema     int                       `json:"schema"`
	Name       string                    `json:"name"`
	Compose    Compose                   `json:"compose"`
	CorpusBOMs []composeTransactionStage `json:"corpus_boms"`
	ModelID    string                    `json:"model_id,omitempty"`
	StartRun   int                       `json:"start_run,omitempty"`
	// Legacy replacement fields remain readable so an interrupted transaction
	// created by an earlier WALDO can fail safely instead of being misread.
	Replace       bool   `json:"replace,omitempty"`
	TargetModelID string `json:"target_model_id,omitempty"`
	TargetSHA256  string `json:"target_sha256,omitempty"`
}

type composeTransactionStage struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
}

// CheckComposeTarget rejects incompatible existing targets before expensive
// corpus preparation. Compose repeats the check while holding its name lock.
func (builder Builder) CheckComposeTarget(name string, compose Compose) error {
	if err := ValidateName(name); err != nil {
		return err
	}
	if err := compose.Validate(); err != nil {
		return err
	}
	if compose.Architecture == (Architecture{}) {
		return fmt.Errorf("compose base source must be resolved before checking its target")
	}
	pending, err := pendingComposeTransaction(builder.Root, name)
	if err != nil {
		return err
	}
	if pending != nil {
		if !reflect.DeepEqual(pending.Compose, compose) {
			return fmt.Errorf("model %q has an unfinished compose with different inputs; repeat the exact command to resume it", name)
		}
		return nil
	}
	if _, err := os.Stat(filepath.Join(builder.Root, name)); os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return err
	}
	target, err := Inspect(builder.Root, name)
	if err != nil {
		return err
	}
	return validateComposeTarget(target, compose)
}

func validateComposeTarget(target Inspection, compose Compose) error {
	if len(target.Runs) > 0 {
		state := target.Runs[len(target.Runs)-1].State
		if state == RunRunning || state == RunInterrupted {
			return fmt.Errorf("model %q has an unfinished %s run; resume that training before starting another compose", target.Model.Name, state)
		}
	}
	architectureHash, err := canonicalHash(compose.Architecture)
	if err != nil {
		return err
	}
	if target.Model.ArchitectureSHA256 != architectureHash {
		return fmt.Errorf("compose architecture does not match existing model %q; use a new model name", target.Model.Name)
	}
	if !reflect.DeepEqual(target.Model.Interaction, compose.Interaction) {
		return fmt.Errorf("compose interaction template does not match existing model %q; use a new model name", target.Model.Name)
	}
	if compose.Base != nil && compose.Base.OriginSHA256 != "" && target.Model.OriginBOMSHA256 != compose.Base.OriginSHA256 {
		return fmt.Errorf("compose base does not match existing model %q; use a new model name", target.Model.Name)
	}
	if compose.Base != nil && compose.Base.RunID != "" && !reflect.DeepEqual(target.Model.Parent, composeParent(compose)) {
		return fmt.Errorf("compose parent does not match existing model %q; use a new model name", target.Model.Name)
	}
	return nil
}

// ResolveCompose resolves an inherited base architecture. When acquire is
// true, external origin weights are acquired through the same importer used by
// model pull and cached outside the visible managed-model namespace.
func (builder Builder) ResolveCompose(ctx context.Context, compose Compose, acquire bool) (Compose, error) {
	resolved, _, err := builder.resolveCompose(ctx, compose, acquire)
	return resolved, err
}

func (builder Builder) resolveCompose(ctx context.Context, compose Compose, acquire bool) (Compose, *Inspection, error) {
	if err := compose.Validate(); err != nil {
		return Compose{}, nil, err
	}
	if compose.Base == nil {
		return compose, nil, nil
	}
	declaration := *compose.Base
	compose.Base = &declaration
	if compose.Base.Source != "" {
		repository, revision, err := parseHuggingFaceSource(compose.Base.Source)
		if err != nil {
			return Compose{}, nil, fmt.Errorf("resolve compose base source: %w", err)
		}
		compose.Base.Source = "huggingface://" + repository + "@" + strings.ToLower(revision)
	}
	var base Inspection
	if compose.Base.Model != "" {
		resolved, err := Inspect(builder.Root, compose.Base.Model)
		if err != nil {
			return Compose{}, nil, fmt.Errorf("resolve compose base model %q: %w", compose.Base.Model, err)
		}
		base = resolved
	} else if acquire {
		originRoot := filepath.Join(builder.Root, ".origins")
		digest := sha256.Sum256([]byte(compose.Base.Source))
		originName := hex.EncodeToString(digest[:])
		originPath := filepath.Join(originRoot, originName)
		if _, err := os.Stat(originPath); os.IsNotExist(err) {
			puller := Puller{Root: originRoot}
			if builder.OriginPuller != nil {
				puller = *builder.OriginPuller
				puller.Root = originRoot
			}
			resolved, pullErr := puller.Pull(ctx, originName, compose.Base.Source)
			if pullErr != nil {
				return Compose{}, nil, fmt.Errorf("acquire compose base %s: %w", compose.Base.Source, pullErr)
			}
			base = resolved
		} else if err != nil {
			return Compose{}, nil, err
		} else {
			resolved, err := Inspect(originRoot, originName)
			if err != nil {
				return Compose{}, nil, err
			}
			base = resolved
		}
	} else {
		puller := Puller{Root: filepath.Join(builder.Root, ".origins")}
		if builder.OriginPuller != nil {
			puller = *builder.OriginPuller
			puller.Root = filepath.Join(builder.Root, ".origins")
		}
		architecture, err := puller.Probe(ctx, compose.Base.Source)
		if err != nil {
			return Compose{}, nil, fmt.Errorf("resolve compose base %s: %w", compose.Base.Source, err)
		}
		if compose.Architecture != (Architecture{}) && !reflect.DeepEqual(compose.Architecture, architecture) {
			return Compose{}, nil, fmt.Errorf("compose architecture does not match base source %q", compose.Base.Source)
		}
		compose.Architecture = architecture
		if err := compose.Validate(); err != nil {
			return Compose{}, nil, err
		}
		return compose, nil, nil
	}
	if compose.Base.Model != "" {
		if compose.Base.ModelID != "" && compose.Base.ModelID != base.Model.ID {
			return Compose{}, nil, fmt.Errorf("compose base model ID is %s, not requested %s", base.Model.ID, compose.Base.ModelID)
		}
		compose.Base.ModelID = base.Model.ID
	}
	initialization, err := resolveRunInitialization(base, compose.Base.RunID)
	if err != nil {
		return Compose{}, nil, err
	}
	if initialization != nil {
		var pin RunPin
		for _, candidate := range base.Model.Runs {
			if candidate.ID == initialization.SourceRunID {
				pin = candidate
				break
			}
		}
		if compose.Base.OriginSHA256 != "" {
			return Compose{}, nil, fmt.Errorf("trained compose base cannot assert origin_sha256; pin run_id and artifact_sha256 instead")
		}
		if compose.Base.RunBOMSHA256 != "" && compose.Base.RunBOMSHA256 != pin.BOMSHA256 {
			return Compose{}, nil, fmt.Errorf("compose base run BOM is %s, not requested %s", pin.BOMSHA256, compose.Base.RunBOMSHA256)
		}
		if compose.Base.ArtifactSHA256 != "" && compose.Base.ArtifactSHA256 != initialization.Artifact.SHA256 {
			return Compose{}, nil, fmt.Errorf("compose base artifact is %s, not requested %s", initialization.Artifact.SHA256, compose.Base.ArtifactSHA256)
		}
		if compose.Base.ArtifactBytes != 0 && compose.Base.ArtifactBytes != initialization.Artifact.Bytes {
			return Compose{}, nil, fmt.Errorf("compose base artifact has %d bytes, not requested %d", initialization.Artifact.Bytes, compose.Base.ArtifactBytes)
		}
		compose.Base.RunID = initialization.SourceRunID
		compose.Base.RunBOMSHA256 = pin.BOMSHA256
		compose.Base.ArtifactSHA256 = initialization.Artifact.SHA256
		compose.Base.ArtifactBytes = initialization.Artifact.Bytes
	} else {
		if base.Origin == nil || base.BOM.CurrentOriginSHA256 == "" {
			return Compose{}, nil, fmt.Errorf("compose base model has no complete real weights")
		}
		if compose.Base.RunID != "" || compose.Base.RunBOMSHA256 != "" || compose.Base.ArtifactSHA256 != "" || compose.Base.ArtifactBytes != 0 {
			return Compose{}, nil, fmt.Errorf("origin compose base cannot assert managed-model run pins")
		}
		if compose.Base.OriginSHA256 != "" && compose.Base.OriginSHA256 != base.Model.OriginBOMSHA256 {
			return Compose{}, nil, fmt.Errorf("compose base origin is %s, not requested %s", base.Model.OriginBOMSHA256, compose.Base.OriginSHA256)
		}
		compose.Base.OriginSHA256 = base.Model.OriginBOMSHA256
	}
	if compose.Architecture != (Architecture{}) && !reflect.DeepEqual(compose.Architecture, base.Model.Architecture) {
		return Compose{}, nil, fmt.Errorf("compose architecture does not match base model")
	}
	compose.Architecture = base.Model.Architecture
	if err := compose.Validate(); err != nil {
		return Compose{}, nil, err
	}
	return compose, &base, nil
}

// Compose creates a model when absent or appends stages to an existing model
// with the same immutable architecture. Hidden transaction metadata preserves
// exact-command resume in both cases.
func (builder Builder) Compose(ctx context.Context, name string, compose Compose, stages []PreparedStage) (Inspection, error) {
	var base *Inspection
	var err error
	compose, base, err = builder.resolveCompose(ctx, compose, true)
	if err != nil {
		return Inspection{}, err
	}
	if err := ValidateName(name); err != nil {
		return Inspection{}, err
	}
	if len(stages) != len(compose.Stages) {
		return Inspection{}, fmt.Errorf("compose has %d stages but %d prepared stages", len(compose.Stages), len(stages))
	}
	for index := range stages {
		if !reflect.DeepEqual(stages[index].Stage, compose.Stages[index]) {
			return Inspection{}, fmt.Errorf("prepared stage %d does not match model compose", index+1)
		}
	}
	if err := os.MkdirAll(builder.Root, 0o755); err != nil {
		return Inspection{}, err
	}
	destination := filepath.Join(builder.Root, name)
	transaction := composeTransaction{Kind: "waldo-model-compose-transaction", Schema: 1, Name: name, Compose: compose}
	for _, stage := range stages {
		digest, err := hashJSON(stage.BOM)
		if err != nil {
			return Inspection{}, err
		}
		transaction.CorpusBOMs = append(transaction.CorpusBOMs, composeTransactionStage{Name: stage.Stage.Name, SHA256: digest})
	}
	stagingRoot := filepath.Join(builder.Root, ".waldo-compose")
	if err := os.MkdirAll(stagingRoot, 0o755); err != nil {
		return Inspection{}, err
	}
	lock, err := lockComposeTransaction(filepath.Join(stagingRoot, name+".lock"))
	if err != nil {
		return Inspection{}, fmt.Errorf("compose model %q: %w", name, err)
	}
	defer unlockComposeTransaction(lock)

	workspace, existing, err := findComposeTransaction(stagingRoot, transaction)
	if err != nil {
		return Inspection{}, err
	}
	resumingTransaction := existing != nil
	if resumingTransaction {
		transaction = *existing
		if transaction.Replace || transaction.TargetModelID != "" || transaction.TargetSHA256 != "" {
			return Inspection{}, fmt.Errorf("model %q has an unfinished legacy replacement compose; continue it with the WALDO version that created it or start a new model", name)
		}
	} else {
		if _, err := os.Stat(destination); err == nil {
			target, inspectErr := Inspect(builder.Root, name)
			if inspectErr != nil {
				return Inspection{}, inspectErr
			}
			if err := validateComposeTarget(target, compose); err != nil {
				return Inspection{}, err
			}
			transaction.ModelID = target.Model.ID
			transaction.StartRun = len(target.Runs)
			if start, ok := recoverableComposeStart(target, stages); ok {
				transaction.StartRun = start
			}
		} else if err != nil && !os.IsNotExist(err) {
			return Inspection{}, err
		}
		transactionID, err := hashJSON(transaction)
		if err != nil {
			return Inspection{}, err
		}
		workspace = filepath.Join(stagingRoot, transactionID)
		if err := os.Mkdir(workspace, 0o755); err != nil {
			return Inspection{}, err
		}
		if err := writeJSONAtomic(filepath.Join(workspace, "COMPOSE.json"), transaction); err != nil {
			_ = os.RemoveAll(workspace)
			return Inspection{}, err
		}
	}
	transactionID, err := hashJSON(transaction)
	if err != nil {
		return Inspection{}, err
	}
	if resumingTransaction {
		builder.report(Progress{Phase: "compose", Message: fmt.Sprintf("resuming durable transaction %s", transactionID[:12])})
		legacyModel := filepath.Join(workspace, name)
		if _, legacyErr := os.Stat(legacyModel); legacyErr == nil {
			if _, err := os.Stat(destination); err == nil {
				return Inspection{}, fmt.Errorf("cannot migrate legacy compose model %q because its destination already exists", name)
			} else if !os.IsNotExist(err) {
				return Inspection{}, err
			}
			if err := os.Rename(legacyModel, destination); err != nil {
				return Inspection{}, fmt.Errorf("migrate composing model %q to its standard path: %w", name, err)
			}
		} else if legacyErr != nil && !os.IsNotExist(legacyErr) {
			return Inspection{}, legacyErr
		}
	}
	if _, err := os.Stat(destination); os.IsNotExist(err) {
		if compose.Base == nil {
			if _, err := builder.initialize(name, compose.Architecture, compose.Interaction); err != nil {
				finishFailedCompose(workspace)
				return Inspection{}, err
			}
		} else if _, err := builder.initializeFromBase(name, compose, *base); err != nil {
			finishFailedCompose(workspace)
			return Inspection{}, err
		}
	} else if err != nil {
		return Inspection{}, err
	}
	composePath := filepath.Join(destination, "COMPOSE.json")
	if data, err := os.ReadFile(composePath); err == nil {
		var persisted Compose
		if err := json.Unmarshal(data, &persisted); err != nil {
			return Inspection{}, fmt.Errorf("read persisted compose for model %s: %w", destination, err)
		}
		if !reflect.DeepEqual(persisted, compose) && transaction.ModelID == "" {
			return Inspection{}, fmt.Errorf("composing model %s has a different persisted compose", destination)
		}
		if !reflect.DeepEqual(persisted, compose) {
			if err := writeJSONAtomic(composePath, compose); err != nil {
				return Inspection{}, fmt.Errorf("persist latest model compose: %w", err)
			}
		}
	} else if os.IsNotExist(err) {
		if err := writeJSONAtomic(composePath, compose); err != nil {
			return Inspection{}, fmt.Errorf("persist model compose: %w", err)
		}
	} else {
		return Inspection{}, err
	}
	if _, err := ArchiveCompose(destination, compose, builder.ComposeName); err != nil {
		return Inspection{}, fmt.Errorf("archive model compose: %w", err)
	}
	staged, err := Inspect(builder.Root, name)
	if err != nil {
		return Inspection{}, fmt.Errorf("inspect composing model %s: %w", destination, err)
	}
	architectureHash, err := canonicalHash(compose.Architecture)
	if err != nil {
		return Inspection{}, err
	}
	if staged.Model.ArchitectureSHA256 != architectureHash {
		return Inspection{}, fmt.Errorf("composing model %s does not match the requested architecture", destination)
	}
	if transaction.ModelID == "" && (staged.Model.OriginBOMSHA256 != composeOriginHash(compose) || !reflect.DeepEqual(staged.Model.Parent, composeParent(compose))) {
		return Inspection{}, fmt.Errorf("composing model %s does not match the requested architecture or origin", destination)
	}
	if transaction.ModelID != "" && staged.Model.ID != transaction.ModelID {
		return Inspection{}, fmt.Errorf("existing model %q changed while its compose transaction was pending", name)
	}
	if len(staged.Runs) > 0 && staged.Runs[len(staged.Runs)-1].State == RunRunning {
		if err := recoverAbandonedComposeRun(builder, &staged); err != nil {
			return Inspection{}, err
		}
	}
	for index, stage := range stages {
		staged, err = Inspect(builder.Root, name)
		if err != nil {
			return Inspection{}, err
		}
		runIndex := transaction.StartRun + index
		if runIndex < len(staged.Runs) {
			if err := validateStagedComposeRun(staged, runIndex, stage); err != nil {
				return Inspection{}, fmt.Errorf("composing model %s: %w", destination, err)
			}
			if staged.Runs[runIndex].State == RunComplete {
				continue
			}
			if staged.Runs[runIndex].State != RunInterrupted && !resumableRunState(staged.Runs[runIndex], staged.RunBOMs[runIndex].Parameters) {
				finishFailedCompose(workspace)
				return Inspection{}, fmt.Errorf("compose stage %s ended %s and cannot be resumed", stage.Stage.Name, staged.Runs[runIndex].State)
			}
		}
		stageBuilder := builder
		if stageBuilder.MultiNode.RendezvousID != "" {
			stageBuilder.MultiNode.StageOrdinal = index + 1
			stageBuilder.MultiNode.StageCount = len(stages)
		}
		if _, err := stageBuilder.Train(ctx, name, stage); err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				builder.report(Progress{Phase: "compose", Message: fmt.Sprintf("retained transaction %s; repeat the exact command to resume", transactionID[:12])})
			} else {
				finishFailedCompose(workspace)
			}
			return Inspection{}, err
		}
	}
	staged, err = Inspect(builder.Root, name)
	if err != nil {
		return Inspection{}, err
	}
	wantRuns := transaction.StartRun + len(stages)
	if len(staged.Runs) != wantRuns {
		return Inspection{}, fmt.Errorf("composing model %s has %d runs, expected %d", destination, len(staged.Runs), wantRuns)
	}
	if err := os.RemoveAll(workspace); err != nil {
		return Inspection{}, fmt.Errorf("remove completed compose staging %s: %w", workspace, err)
	}
	builder.report(Progress{Phase: "compose", Message: fmt.Sprintf("completed durable transaction %s", transactionID[:12])})
	return Inspect(builder.Root, name)
}

func recoverableComposeStart(inspection Inspection, stages []PreparedStage) (int, bool) {
	if !HasRecoverableFinalizationFailure(inspection) {
		return 0, false
	}
	failed := len(inspection.Runs) - 1
	for stageIndex := len(stages) - 1; stageIndex >= 0; stageIndex-- {
		start := failed - stageIndex
		if start < 0 {
			continue
		}
		matches := true
		for position := 0; position <= stageIndex; position++ {
			runIndex := start + position
			if err := validateStagedComposeRun(inspection, runIndex, stages[position]); err != nil {
				matches = false
				break
			}
			if position < stageIndex && inspection.Runs[runIndex].State != RunComplete {
				matches = false
				break
			}
		}
		if matches {
			return start, true
		}
	}
	return 0, false
}

func pendingComposeTransaction(root, name string) (*composeTransaction, error) {
	entries, err := os.ReadDir(filepath.Join(root, ".waldo-compose"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		var transaction composeTransaction
		if err := readJSON(filepath.Join(root, ".waldo-compose", entry.Name(), "COMPOSE.json"), &transaction); err != nil {
			continue
		}
		if transaction.Kind == "waldo-model-compose-transaction" && transaction.Schema == 1 && transaction.Name == name {
			return &transaction, nil
		}
	}
	return nil, nil
}

func findComposeTransaction(root string, requested composeTransaction) (string, *composeTransaction, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return "", nil, err
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		var existing composeTransaction
		workspace := filepath.Join(root, entry.Name())
		if err := readJSON(filepath.Join(workspace, "COMPOSE.json"), &existing); err != nil {
			continue
		}
		candidate := existing
		candidate.ModelID = ""
		candidate.StartRun = 0
		candidate.TargetModelID = ""
		candidate.TargetSHA256 = ""
		if reflect.DeepEqual(candidate, requested) {
			return workspace, &existing, nil
		}
		if existing.Kind == requested.Kind && existing.Schema == requested.Schema && existing.Name == requested.Name {
			return "", nil, fmt.Errorf("model %q has an unfinished compose with different inputs; repeat the exact command to resume it", requested.Name)
		}
	}
	return "", nil, nil
}

// finishFailedCompose removes only transaction metadata. The model and all
// durable run evidence remain available for inspection and explicit action.
func finishFailedCompose(workspace string) {
	_ = os.RemoveAll(workspace)
}

func composeOriginHash(compose Compose) string {
	if compose.Base == nil {
		return ""
	}
	return compose.Base.OriginSHA256
}

func composeParent(compose Compose) *ModelParent {
	if compose.Base == nil || compose.Base.RunID == "" {
		return nil
	}
	return &ModelParent{
		Model: compose.Base.Model, ModelID: compose.Base.ModelID,
		RunID: compose.Base.RunID, RunBOMSHA256: compose.Base.RunBOMSHA256,
		Artifact: training.Artifact{Path: "base/model.safetensors", SHA256: compose.Base.ArtifactSHA256, Bytes: compose.Base.ArtifactBytes},
	}
}

func lockComposeTransaction(path string) (*os.File, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("another process owns this compose; wait for it to finish")
	}
	return file, nil
}

func unlockComposeTransaction(file *os.File) {
	if file == nil {
		return
	}
	_ = syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
	_ = file.Close()
}

func recoverAbandonedComposeRun(builder Builder, inspection *Inspection) error {
	position := len(inspection.Runs) - 1
	run := inspection.Runs[position]
	pin := inspection.Model.Runs[position]
	now := builder.clock()()
	run.State = RunInterrupted
	run.Finished = formatTime(now)
	run.Error = "previous compose process ended without terminal state"
	if len(run.Attempts) > 0 && run.Attempts[len(run.Attempts)-1].State == RunRunning {
		attempt := &run.Attempts[len(run.Attempts)-1]
		attempt.State = RunInterrupted
		attempt.Finished = run.Finished
		attempt.Error = run.Error
	}
	record := inspection.Model
	runDirectory := filepath.Join(inspection.Path, "runs", runDirectoryName(pin))
	if err := persistRunAndPin(inspection.Path, runDirectory, &record, pin, run, now); err != nil {
		return fmt.Errorf("recover abandoned compose run %s: %w", run.ID, err)
	}
	updated, err := Inspect(builder.Root, inspection.Model.Name)
	if err != nil {
		return err
	}
	*inspection = updated
	return nil
}

func validateStagedComposeRun(inspection Inspection, index int, prepared PreparedStage) error {
	prepared.Stage = stageWithInteraction(prepared.Stage, inspection.Model.Interaction)
	if index >= len(inspection.RunBOMs) || inspection.Model.Runs[index].Stage != prepared.Stage.Name {
		return fmt.Errorf("run %d does not match stage %s", index+1, prepared.Stage.Name)
	}
	bom := inspection.RunBOMs[index]
	corpusHash, err := hashJSON(prepared.BOM)
	if err != nil {
		return err
	}
	parameters, err := prepared.Stage.ResolvePlanningParameters()
	if err != nil {
		return err
	}
	if prepared.Stage.Parameters.Steps == 0 && prepared.Stage.Parameters.Tokens == 0 {
		parameters, err = prepared.Stage.ResolveParametersForSteps(bom.Parameters.Steps)
		if err != nil {
			return err
		}
	}
	if parameters.Data.Order == "corpus-weighted-shuffle-v1" {
		parameters.Data.CorpusWeights, err = resolveCorpusWeights(parameters.Data.CorpusWeights, prepared.BOM.Paths)
		if err != nil {
			return err
		}
	}
	conversation := training.ConversationTransform{}
	if prepared.Stage.Conversation != nil {
		conversation = *prepared.Stage.Conversation
	}
	if bom.Stage != prepared.Stage.Name || bom.StageType != prepared.Stage.Type || bom.Objective != prepared.Stage.Objective || !reflect.DeepEqual(bom.Conversation, conversation) || bom.CorpusBOMSHA256 != corpusHash || !equivalentTrainingParameters(bom.Parameters, parameters) {
		return fmt.Errorf("run %d immutable facts do not match stage %s", index+1, prepared.Stage.Name)
	}
	return nil
}

func equivalentTrainingParameters(left, right training.ResolvedParameters) bool {
	return reflect.DeepEqual(training.NormalizeResolvedParameters(left), training.NormalizeResolvedParameters(right))
}

func (builder Builder) initializeFromBase(name string, compose Compose, base Inspection) (Inspection, error) {
	if compose.Base.RunID != "" {
		return builder.initializeFromRun(name, compose, base)
	}
	return builder.initializeFromOrigin(name, compose, base)
}

func (builder Builder) initializeFromRun(name string, compose Compose, base Inspection) (Inspection, error) {
	initialization, err := resolveRunInitialization(base, compose.Base.RunID)
	if err != nil {
		return Inspection{}, err
	}
	if initialization == nil {
		return Inspection{}, fmt.Errorf("compose base model %q has no complete real weights", base.Model.Name)
	}
	plan, err := composePlan(name, compose)
	if err != nil {
		return Inspection{}, err
	}
	planHash, err := hashJSON(plan)
	if err != nil {
		return Inspection{}, err
	}
	now := builder.clock()()
	parent := composeParent(compose)
	record := ModelRecord{
		Kind: "waldo-model", Schema: ModelSchema, ID: planHash, Name: name,
		PlanSHA256: planHash, ArchitectureSHA256: plan.ArchitectureSHA256,
		Architecture: plan.Architecture, Interaction: plan.Interaction, Forecast: plan.Forecast,
		Created: formatTime(now), Updated: formatTime(now), Parent: parent,
	}
	destination := filepath.Join(builder.Root, name)
	if err := initializeModel(builder.Root, destination, plan, record); err != nil {
		return Inspection{}, err
	}
	target := filepath.Join(destination, filepath.FromSlash(parent.Artifact.Path))
	if err := copyArtifactFile(initialization.Path, target); err != nil {
		return Inspection{}, err
	}
	return Inspect(builder.Root, name)
}

func (builder Builder) initializeFromOrigin(name string, compose Compose, base Inspection) (Inspection, error) {
	plan, err := composePlan(name, compose)
	if err != nil {
		return Inspection{}, err
	}
	planHash, err := hashJSON(plan)
	if err != nil {
		return Inspection{}, err
	}
	now := builder.clock()()
	record := ModelRecord{
		Kind: "waldo-model", Schema: ModelSchema, ID: planHash, Name: name,
		PlanSHA256: planHash, ArchitectureSHA256: plan.ArchitectureSHA256,
		Architecture: plan.Architecture, Interaction: plan.Interaction, Forecast: plan.Forecast,
		Created: formatTime(now), Updated: formatTime(now),
		OriginBOMSHA256: base.Model.OriginBOMSHA256,
		OriginArtifacts: append([]OriginArtifact(nil), base.Model.OriginArtifacts...),
	}
	destination := filepath.Join(builder.Root, name)
	if err := initializeModel(builder.Root, destination, plan, record); err != nil {
		return Inspection{}, err
	}
	for _, artifact := range base.Origin.Artifacts {
		source := filepath.Join(base.Path, filepath.FromSlash(artifact.Path))
		target := filepath.Join(destination, filepath.FromSlash(artifact.Path))
		if err := copyArtifactFile(source, target); err != nil {
			return Inspection{}, err
		}
	}
	if err := writeJSONAtomic(filepath.Join(destination, "ORIGIN-BOM.json"), base.Origin); err != nil {
		return Inspection{}, err
	}
	return Inspect(builder.Root, name)
}

func copyArtifactFile(source, destination string) error {
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return err
	}
	if err := os.Link(source, destination); err == nil {
		return nil
	}
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(output, input)
	closeErr := output.Close()
	return errors.Join(copyErr, closeErr)
}

func initializeModel(root, destination string, plan Plan, record ModelRecord) error {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}
	temporary, err := os.MkdirTemp(root, ".waldo-model-*")
	if err != nil {
		return err
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(temporary)
		}
	}()
	if err := writeJSONAtomic(filepath.Join(temporary, "PLAN.json"), plan); err != nil {
		return err
	}
	if err := writeJSONAtomic(filepath.Join(temporary, "MODEL.json"), record); err != nil {
		return err
	}
	if err := writeJSONAtomic(filepath.Join(temporary, "MODEL-BOM.json"), modelBOM(record)); err != nil {
		return err
	}
	if err := os.Rename(temporary, destination); err != nil {
		return fmt.Errorf("create model %s: %w", record.Name, err)
	}
	committed = true
	return nil
}

func persistRunAndPin(modelPath, runDirectory string, record *ModelRecord, original RunPin, run RunRecord, now time.Time) error {
	if err := writeJSONAtomic(filepath.Join(runDirectory, "RUN.json"), run); err != nil {
		return err
	}
	observationHash := ""
	var artifacts []training.Artifact
	if run.Observation != nil {
		var err error
		observationHash, err = hashJSON(run.Observation)
		if err != nil {
			return err
		}
		artifacts = append([]training.Artifact(nil), run.Observation.Artifacts...)
	}
	for i := range record.Runs {
		if record.Runs[i].ID == original.ID {
			record.Runs[i].State = run.State
			record.Runs[i].ObservationSHA256 = observationHash
			record.Runs[i].Artifacts = artifacts
			break
		}
	}
	return persistModel(modelPath, record, now)
}

func persistModel(modelPath string, record *ModelRecord, now time.Time) error {
	record.Updated = formatTime(now)
	sortPins(record.Runs)
	if err := writeJSONAtomic(filepath.Join(modelPath, "MODEL.json"), record); err != nil {
		return err
	}
	return writeJSONAtomic(filepath.Join(modelPath, "MODEL-BOM.json"), modelBOM(*record))
}

func builtinResolver() training.Resolver {
	return training.NewEnvironmentResolver(training.BackendAuto)
}

func validateSelection(selection training.Selection, objectives []string) error {
	if selection.Backend == nil {
		return fmt.Errorf("resolved training backend is nil")
	}
	descriptor := selection.Backend.Descriptor()
	if descriptor.Identity.Name == "" || descriptor.Identity.Revision == "" || descriptor.Framework == "" {
		return fmt.Errorf("resolved training backend has an incomplete descriptor")
	}
	if selection.Execution.Backend != descriptor.Identity || selection.Execution.Framework != descriptor.Framework {
		return fmt.Errorf("resolved execution does not match backend %s@%s", descriptor.Identity.Name, descriptor.Identity.Revision)
	}
	if selection.Execution.Host.OS == "" || selection.Execution.Host.Architecture == "" || selection.Execution.Nodes <= 0 || selection.Execution.WorldSize <= 0 {
		return fmt.Errorf("resolved execution has incomplete host or topology facts")
	}
	supported := make(map[string]bool, len(descriptor.Capabilities.Objectives))
	for _, objective := range descriptor.Capabilities.Objectives {
		supported[objective] = true
	}
	for _, objective := range objectives {
		if !supported[objective] {
			return fmt.Errorf("training backend %s@%s does not support objective %s", descriptor.Identity.Name, descriptor.Identity.Revision, objective)
		}
	}
	return nil
}

func (builder Builder) clock() func() time.Time {
	if builder.Now != nil {
		return builder.Now
	}
	return time.Now
}

func (builder Builder) identifier() func() (string, error) {
	if builder.NewID != nil {
		return builder.NewID
	}
	return randomID
}

func (builder Builder) report(progress Progress) {
	if builder.Progress != nil {
		builder.Progress(progress)
	}
}

func randomID() (string, error) {
	var value [8]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(value[:]), nil
}

func planObjectives(plan Plan) []string {
	seen := make(map[string]bool)
	for _, stage := range plan.Stages {
		seen[stage.Objective] = true
	}
	objectives := make([]string, 0, len(seen))
	for objective := range seen {
		objectives = append(objectives, objective)
	}
	sort.Strings(objectives)
	return objectives
}
