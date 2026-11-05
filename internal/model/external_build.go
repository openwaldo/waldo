// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package model

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/model/tokenizerapp"
	"github.com/openwaldo/waldo/internal/model/trainerapp"
	"github.com/openwaldo/waldo/internal/training"
)

// ExternalOptions contains machine-local policy and resolved corpus inputs.
// Applications receive only prepared tokens, never these input paths.
type ExternalOptions struct {
	NodeLaunches          map[string][]applaunch.Evidence
	PreparedCacheMaxBytes int64
	Hosts                 []string
	Publish               func(ExternalNodePlan) error
	World                 applaunch.World
	Profile               string
	NCCLInterface         string
	NCCLHCA               string
	Cache                 string
	Python                string
	Prepare               func(context.Context, ExternalStage) (corpus.BOM, []training.Input, error)
}

type ExternalAttempt struct {
	NodeLaunches []applaunch.Evidence `json:"node_launches,omitempty"`
	Started      string               `json:"started"`
	Finished     string               `json:"finished,omitempty"`
	Error        string               `json:"error,omitempty"`
	Launch       applaunch.Evidence   `json:"launch"`
	Probe        trainerapp.Probe     `json:"probe"`
}

type ExternalRunRecord struct {
	Kind           string                           `json:"kind"`
	Schema         int                              `json:"schema"`
	ID             string                           `json:"id"`
	ModelID        string                           `json:"model_id"`
	Stage          string                           `json:"stage"`
	State          RunState                         `json:"state"`
	BOMSHA256      string                           `json:"bom_sha256"`
	Checkpoints    []trainerapp.CheckpointReference `json:"checkpoints"`
	Evaluations    []trainerapp.Evaluation          `json:"evaluations"`
	Attempts       []ExternalAttempt                `json:"attempts"`
	Complete       *trainerapp.Complete             `json:"complete,omitempty"`
	Initialization *training.WorkerInitialization   `json:"initialization,omitempty"`
}

type ExternalModelRecord struct {
	Kind    string   `json:"kind"`
	Schema  int      `json:"schema"`
	ID      string   `json:"id"`
	Name    string   `json:"name"`
	State   RunState `json:"state"`
	Created string   `json:"created"`
	Updated string   `json:"updated"`
}

type ExternalInspection struct {
	Path    string              `json:"path"`
	Model   ExternalModelRecord `json:"model"`
	Plan    ExternalPlan        `json:"plan"`
	Runs    []ExternalRunRecord `json:"runs"`
	RunBOMs []ExternalRunBOM    `json:"run_boms"`
	BOM     ExternalModelBOM    `json:"bom"`
}

// IsExternalModel examines only the versioned plan envelope, without attempting
// to decode an opaque architecture into the native contract.
func IsExternalModel(root, name string) (bool, error) {
	directory, err := modelDirectory(root, name)
	if err != nil {
		return false, err
	}
	var header struct {
		Schema int `json:"schema"`
	}
	err = readJSON(filepath.Join(directory, "PLAN.json"), &header)
	if os.IsNotExist(err) {
		return false, nil
	}
	return header.Schema == 2, err
}

func InspectExternal(root, name string) (ExternalInspection, error) {
	var out ExternalInspection
	directory, err := modelDirectory(root, name)
	if err != nil {
		return out, err
	}
	out.Path = directory
	p, id, err := ReadExternalPlan(out.Path)
	if err != nil {
		return out, err
	}
	out.Plan = p
	if err = readJSON(filepath.Join(out.Path, "MODEL.json"), &out.Model); err != nil {
		return out, err
	}
	if out.Model.Kind != "waldo-model" || out.Model.Schema != 2 || out.Model.ID != id || out.Model.Name != p.Name {
		return out, fmt.Errorf("external model record differs from plan")
	}
	for _, stage := range p.Identity.Stages {
		var r ExternalRunRecord
		err = readJSON(filepath.Join(out.Path, "runs", stage.Name, "RUN.json"), &r)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return out, err
		}
		if r.Kind != "waldo-model-run" || r.Schema != 2 || r.ModelID != id || r.Stage != stage.Name || r.ID != stage.Name {
			return out, fmt.Errorf("external run record differs from plan")
		}
		var bom ExternalRunBOM
		if err = readJSON(filepath.Join(out.Path, "runs", stage.Name, "RUN-BOM.json"), &bom); err != nil {
			return out, err
		}
		if err = checkExternalHash(bom, r.BOMSHA256); err != nil {
			return out, err
		}
		if bom.ModelID != id || bom.ID != r.ID || bom.Stage != r.Stage || !sameExternalJSON(bom.Identity, p.Identity) {
			return out, fmt.Errorf("external run identity mismatch")
		}
		if r.State == RunComplete {
			if r.Complete == nil {
				return out, fmt.Errorf("completed external run lacks completion")
			}
			if err = trainerapp.VerifyPublished(filepath.Join(out.Path, "runs", stage.Name, "artifacts"), r.Complete.Artifacts); err != nil {
				return out, err
			}
		}
		out.Runs = append(out.Runs, r)
		out.RunBOMs = append(out.RunBOMs, bom)
	}
	if err = readJSON(filepath.Join(out.Path, "MODEL-BOM.json"), &out.BOM); err != nil {
		return out, err
	}
	if out.BOM.ModelID != id || !sameExternalJSON(out.BOM.Identity, p.Identity) {
		return out, fmt.Errorf("external aggregate identity mismatch")
	}
	var expected []ExternalBOMRun
	for _, r := range out.Runs {
		if r.State != RunComplete {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(out.Path, "runs", r.ID, "RUN-BOM.json"))
		if err != nil {
			return out, err
		}
		ref := ExternalBOMRun{ID: r.ID, BOM: training.Artifact{Path: filepath.Join("runs", r.ID, "RUN-BOM.json"), SHA256: application.Digest(raw), Bytes: int64(len(raw))}}
		for _, a := range r.Complete.Artifacts {
			a.Path = filepath.Join("runs", r.ID, "artifacts", a.Path)
			ref.Artifacts = append(ref.Artifacts, a)
		}
		expected = append(expected, ref)
	}
	if !sameExternalJSON(expected, out.BOM.Runs) {
		return out, fmt.Errorf("external aggregate run references differ from completed runs")
	}

	return out, nil
}

type externalPreparedStage struct {
	bom         corpus.BOM
	partition   training.RecordPartition
	data        training.ExternalDataParameters
	resolved    trainerapp.Resolved
	evaluations []trainerapp.EvaluationRecord
}

// DataStage projects only WALDO-owned stage fields for shared corpus policy and
// data resolution. The placeholder learning rate never enters external identity
// or trainer parameters.
func (s ExternalStage) DataStage() Stage {
	d := s.Data
	return Stage{Name: s.Name, Type: s.Type, Objective: s.Objective, Corpora: s.Corpora, Filter: s.Filter, Conversation: s.Conversation, Parameters: training.Parameters{Profile: d.Profile, Tokens: d.Tokens, Epochs: d.Epochs, Steps: d.Steps, BatchSize: d.GlobalBatchSequences, SequenceLength: d.SequenceLength, Seed: d.Seed, LearningRate: 1, CheckpointEvery: d.CheckpointEvery, EvaluateEvery: d.EvaluateEvery, ShuffleBufferRecords: d.ShuffleBufferRecords, ShuffleBufferBytes: d.ShuffleBufferBytes, EvaluationFraction: d.EvaluationFraction, EvaluationSelection: d.EvaluationSelection, EvaluationMaxRecords: d.EvaluationMaxRecords, EvaluationMaxBytes: d.EvaluationMaxBytes}}
}

func prepareExternalStage(ctx context.Context, s ExternalStage, interaction Interaction, o ExternalOptions, codec *tokenizerapp.Client) (externalPreparedStage, error) {
	var out externalPreparedStage
	var err error
	var inputs []training.Input
	out.bom, inputs, err = o.Prepare(ctx, s)
	if err != nil {
		return out, err
	}
	if err = out.bom.Validate(); err != nil {
		return out, err
	}
	selected := map[string]bool{}
	for _, path := range out.bom.Paths {
		selected[path] = true
	}
	for _, input := range inputs {
		if !selected[input.Corpus] {
			return out, fmt.Errorf("external input corpus %q is absent from resolved BOM", input.Corpus)
		}
	}
	native := s.DataStage()
	if _, err = PrepareStage(native, out.bom, inputs); err != nil {
		return out, err
	}
	parameters, err := native.ResolvePlanningParameters()
	if err != nil {
		return out, err
	}
	if parameters.Data.Order == "corpus-weighted-shuffle-v1" {
		parameters.Data.CorpusWeights, err = resolveCorpusWeights(parameters.Data.CorpusWeights, out.bom.Paths)
		if err != nil {
			return out, err
		}
	}
	transform := training.ConversationTransform{}
	if s.Conversation != nil {
		transform = *s.Conversation
		transform.Tools = interaction.Tools
	}
	out.partition, err = training.NewRecordPartitionContextWithTransform(ctx, inputs, parameters, codec, s.Objective, transform, nil)
	if err != nil {
		return out, err
	}
	out.data, err = s.ResolveData()
	if err != nil {
		return out, err
	}
	out.data.Order = parameters.Data
	if s.Data.Epochs > 0 && s.Data.Steps == 0 && s.Data.Tokens == 0 {
		out.data.Steps, err = out.partition.TrainingSteps(ctx)
		if err != nil {
			return out, err
		}
		out.data.PlannedTokenCapacity = out.data.Steps * out.data.GlobalBatchSequences * out.data.SequenceLength
		raw, err := out.partition.TrainingRecords()
		if err != nil {
			return out, err
		}
		tokens, err := training.NewTokenizedRecordSource(raw, codec, s.Objective, transform)
		if err != nil {
			return out, err
		}
		exact, err := training.NewPreparedSequenceSource(tokens, out.data, codec.Spec(), 0)
		if err != nil {
			return out, err
		}
		if err = exact(ctx, func(training.PreparedSequence) error { return nil }); err != nil {
			return out, fmt.Errorf("finite epoch budget cannot fill exact external batches: %w", err)
		}

	} else if !parameters.Data.RepeatUntilSteps {
		out.partition, out.data.Epochs, err = out.partition.WithMinimumEpochsForSteps(ctx, out.data.Steps)
		if err != nil {
			return out, err
		}
	}
	// Capture original byte counts before tokenization clears the text view.
	var raw []training.Record
	err = out.partition.EvaluationRecords().Stream(ctx, func(r training.Record) error { raw = append(raw, r); return nil })
	if err != nil {
		return out, err
	}
	tokenized, err := training.NewTokenizedRecordSource(out.partition.EvaluationRecords(), codec, s.Objective, transform)
	if err != nil {
		return out, err
	}
	n := 0
	err = tokenized.Stream(ctx, func(r training.Record) error {
		bytes := int64(len(raw[n].Text))
		n++
		out.evaluations = append(out.evaluations, trainerapp.EvaluationRecord{ID: r.ID, Tokens: append(r.Tokens, codec.Spec().EOSID), LossMask: r.LossMask[1:], TextBytes: bytes})
		return nil
	})
	return out, err
}

// ComposeExternal runs an immutable external compose under the same per-model
// transaction lock as native composes. Repeating it resumes the saved run IDs.
func (b Builder) ComposeExternal(ctx context.Context, name string, c ExternalCompose, o ExternalOptions) (ExternalInspection, error) {
	var zero ExternalInspection
	if err := ValidateName(name); err != nil {
		return zero, err
	}
	if err := c.Validate(); err != nil {
		return zero, err
	}
	if o.Prepare == nil {
		return zero, fmt.Errorf("external compose requires resolved corpus provider")
	}
	root, err := filepath.Abs(b.Root)
	if err != nil {
		return zero, err
	}
	b.Root = root
	cache, err := filepath.Abs(o.Cache)
	if err != nil {
		return zero, err
	}
	if o.Python == "" {
		o.Python = "python3"
	}
	lockDir := filepath.Join(root, ".waldo-compose")
	if err = os.MkdirAll(lockDir, 0755); err != nil {
		return zero, err
	}
	lock, err := lockComposeTransaction(filepath.Join(lockDir, name+".lock"))
	if err != nil {
		return zero, err
	}
	defer unlockComposeTransaction(lock)
	directory := filepath.Join(root, name)
	b.report(Progress{Phase: "applications", Message: "verifying pinned trainer, tokenizer and Python environments"})
	sources, err := c.ResolveSources(ctx, filepath.Join(cache, "sources"))
	if err != nil {
		return zero, err
	}
	trainerEnv, err := sources.Trainer.EnsureEnvironment(ctx, c.Trainer.Environment, o.Python, filepath.Join(cache, "environments"))
	if err != nil {
		return zero, err
	}
	if o.Profile == "distributed" {
		host, err := training.InspectExternalDistributedHost(ctx, filepath.Join(trainerEnv.Run.CachePath, "bin", "python"))
		if err != nil {
			return zero, err
		}
		if o.World.ProcessesPerNode > len(host.Accelerators) {
			return zero, fmt.Errorf("distributed process count exceeds visible GPUs")
		}
		if err = training.ValidateTorchTitanHostConfiguration(host, training.Cluster{Nodes: o.World.Nodes, Interface: o.NCCLInterface, HCA: o.NCCLHCA}); err != nil {
			return zero, err
		}
	}
	tokenEnv, err := sources.Tokenizer.EnsureEnvironment(ctx, c.Tokenizer.Environment, o.Python, filepath.Join(cache, "environments"))
	if err != nil {
		return zero, err
	}
	scratch, err := os.MkdirTemp(lockDir, name+"-probe-")
	if err != nil {
		return zero, err
	}
	defer os.RemoveAll(scratch)
	tokenRequest := applaunch.Request{Source: sources.Tokenizer, Environment: tokenEnv, Scratch: scratch, Command: "serve"}
	codec, err := tokenizerapp.Open(ctx, tokenRequest, c.Tokenizer.Configuration)
	if err != nil {
		return zero, err
	}
	defer codec.Close()
	world := o.World
	if world == (applaunch.World{}) {
		world = applaunch.World{Nodes: 1, ProcessesPerNode: 1, WorldSize: 1}
	}
	trainerRequest := applaunch.Request{Source: sources.Trainer, Environment: trainerEnv, Scratch: scratch, World: world, Profile: o.Profile, NCCLInterface: o.NCCLInterface, NCCLHCA: o.NCCLHCA, Command: "train"}
	var objectives []string
	for _, s := range c.Stages {
		objectives = append(objectives, s.Objective)
	}
	client, err := trainerapp.Open(ctx, trainerRequest, objectives)
	if err != nil {
		return zero, err
	}
	schemas, err := client.SchemaDocument(codec.Identity().Probe.Schemas.Configuration)
	if err != nil {
		return zero, err
	}
	archHash, err := externalHash(c.Architecture)
	if err != nil {
		return zero, err
	}
	plan := ExternalPlan{Kind: "waldo-model-plan", Schema: 2, Name: name, Identity: ExternalIdentity{Trainer: sources.Trainer, Environment: trainerEnv.Identity, Tokenizer: codec.Identity(), Architecture: c.Architecture, ArchitectureSHA256: archHash, ArchitectureSchemaSHA256: client.SchemaHashes()["architecture"], ParametersSchemaSHA256: client.SchemaHashes()["parameters"], Determinism: c.Trainer.Determinism, Interaction: c.Interaction}}
	var stages []externalPreparedStage
	var resolutions []trainerapp.Resolved
	for _, s := range c.Stages {
		b.report(Progress{Phase: "preflight", Stage: s.Name, Message: "preparing corpus partition with tokenizer application"})
		p, err := prepareExternalStage(ctx, s, c.Interaction, o, codec)
		if err != nil {
			return zero, fmt.Errorf("stage %s: %w", s.Name, err)
		}
		p.resolved, err = client.Resolve(ctx, c.Architecture, s.Trainer, p.data, trainerapp.World{World: world})
		if err != nil {
			return zero, err
		}
		bomHash, err := externalHash(p.bom)
		if err != nil {
			return zero, err
		}
		transform := training.ConversationTransform{}
		if s.Conversation != nil {
			transform = *s.Conversation
			transform.Tools = c.Interaction.Tools
		}
		plan.Identity.Stages = append(plan.Identity.Stages, ExternalStageIdentity{Name: s.Name, Type: s.Type, Objective: s.Objective, CorpusBOMSHA256: bomHash, Data: p.data, Conversation: transform, Recipe: p.resolved.Recipe, RecipeSHA256: p.resolved.RecipeSHA256})
		stages = append(stages, p)
		resolutions = append(resolutions, p.resolved)
	}
	id, err := plan.SHA256()
	if err != nil {
		return zero, err
	}
	if _, err = os.Stat(directory); os.IsNotExist(err) {
		if _, err = WriteExternalPlan(directory, plan, schemas, resolutions); err != nil {
			return zero, err
		}
	} else if err != nil {
		return zero, err
	} else {
		_, saved, err := ReadExternalPlan(directory)
		if err != nil {
			return zero, err
		}
		if saved != id {
			return zero, fmt.Errorf("external compose differs from immutable plan for %s; choose a new model name", name)
		}
	}
	now := formatTime(b.clock()())
	modelRecord := ExternalModelRecord{Kind: "waldo-model", Schema: 2, ID: id, Name: name, State: RunRunning, Created: now, Updated: now}
	var prior ExternalModelRecord
	if err = readJSON(filepath.Join(directory, "MODEL.json"), &prior); err == nil {
		if prior.Kind != "waldo-model" || prior.Schema != 2 || prior.ID != id || prior.Name != name {
			return zero, fmt.Errorf("external model record differs from plan")
		}
		modelRecord = prior
		modelRecord.Updated = now
	} else if !os.IsNotExist(err) {
		return zero, err
	}
	if _, statErr := os.Stat(filepath.Join(directory, "COMPOSE.json")); os.IsNotExist(statErr) {
		if err = writeJSONAtomic(filepath.Join(directory, "COMPOSE.json"), c); err != nil {
			return zero, err
		}
	} else if statErr != nil {
		return zero, statErr
	}
	if err = writeJSONAtomic(filepath.Join(directory, "MODEL.json"), modelRecord); err != nil {
		return zero, err
	}
	var refs []ExternalBOMRun
	if _, statErr := os.Stat(filepath.Join(directory, "MODEL-BOM.json")); os.IsNotExist(statErr) {
		if _, err = WriteExternalModelBOM(directory, false, refs); err != nil {
			return zero, err
		}
	} else if statErr != nil {
		return zero, statErr
	}
	var initialization *training.WorkerInitialization
	for index, s := range c.Stages {
		p := stages[index]
		runDir := filepath.Join(directory, "runs", s.Name)
		for _, dir := range []string{runDir, filepath.Join(runDir, "artifacts"), filepath.Join(runDir, "checkpoints"), filepath.Join(runDir, "scratch")} {
			if err = os.MkdirAll(dir, 0700); err != nil {
				return zero, err
			}
		}
		request := trainerRequest
		request.Scratch = filepath.Join(runDir, "scratch")
		current, err := trainerapp.Open(ctx, request, []string{s.Objective})
		if err != nil {
			return zero, err
		}
		if current.SchemaHashes()["architecture"] != plan.Identity.ArchitectureSchemaSHA256 || current.SchemaHashes()["parameters"] != plan.Identity.ParametersSchemaSHA256 {
			return zero, fmt.Errorf("trainer probe schemas changed from immutable plan")
		}
		evidence, err := applaunch.Prepare(request)
		if err != nil {
			return zero, err
		}
		tokenEvidence, err := applaunch.Prepare(tokenRequest)
		if err != nil {
			return zero, err
		}
		bom := ExternalRunBOM{Hosts: o.Hosts, DataPlane: training.DataPlaneNodeLocal, Initialization: initialization, Kind: "openwaldo-bom", Schema: 2, Subject: "training-run", ID: s.Name, ModelID: id, Stage: s.Name, Identity: plan.Identity, CorpusBOM: p.bom, Resolved: p.resolved, EvaluationSet: p.partition.Evaluation, TrainerHost: trainerEnv.Run, TokenizerHost: tokenEnv.Run, Probe: current.Probe(), Launches: append(append(current.Evidence(), evidence, tokenEvidence), o.NodeLaunches[s.Name]...)}
		record := ExternalRunRecord{Kind: "waldo-model-run", Schema: 2, ID: s.Name, ModelID: id, Stage: s.Name, State: RunPlanned, Initialization: initialization}
		recordPath := filepath.Join(runDir, "RUN.json")
		if err = readJSON(recordPath, &record); err == nil {
			if record.ModelID != id || record.ID != s.Name || record.Stage != s.Name || record.Kind != "waldo-model-run" || record.Schema != 2 {
				return zero, fmt.Errorf("invalid external run record")
			}
			var saved ExternalRunBOM
			if err = readJSON(filepath.Join(runDir, "RUN-BOM.json"), &saved); err != nil {
				return zero, err
			}
			if err = checkExternalHash(saved, record.BOMSHA256); err != nil {
				return zero, err
			}
			// Host probe values may vary; the resolved execution and evaluation pins may not.
			if !sameExternalJSON(saved.Hosts, o.Hosts) || saved.Resolved.Batch.WorldSize != int64(world.WorldSize) || !sameExternalJSON(saved.Identity, plan.Identity) || !sameExternalJSON(saved.Resolved, p.resolved) || !sameExternalJSON(saved.EvaluationSet, p.partition.Evaluation) || !sameExternalJSON(saved.Initialization, initialization) || !sameExternalJSON(record.Initialization, initialization) {
				return zero, fmt.Errorf("external resume execution or evaluation pins changed")
			}
			if len(record.Attempts) > 0 && (record.Attempts[0].Launch.World != world || record.Attempts[0].Launch.Profile != evidence.Profile) {
				return zero, fmt.Errorf("external resume topology/profile changed")
			}
			bom = saved
		} else if os.IsNotExist(err) {
			record.BOMSHA256, err = WriteExternalRunBOM(directory, filepath.Join(runDir, "RUN-BOM.json"), bom)
			if err != nil {
				return zero, err
			}
			if err = writeJSONAtomic(recordPath, record); err != nil {
				return zero, err
			}
		} else {
			return zero, err
		}
		begin := trainerapp.Begin{Paths: trainerapp.Paths{Artifacts: filepath.Join(runDir, "artifacts"), Checkpoints: filepath.Join(runDir, "checkpoints"), Scratch: request.Scratch}, ModelID: id, RunID: s.Name, Stage: s.Name, Objective: s.Objective, Data: p.data, Architecture: c.Architecture, ArchitectureSHA256: archHash, Parameters: p.resolved.Parameters, ParametersSHA256: p.resolved.ParametersSHA256, Batch: p.resolved.Batch, Tokenizer: codec.Spec(), EvaluationSet: p.partition.Evaluation, Initialization: record.Initialization, Determinism: c.Trainer.Determinism, World: world, CheckpointDirectory: filepath.Join(runDir, "bundles"), PriorCheckpoints: record.Checkpoints, PriorEvaluations: record.Evaluations}
		trainerHash, _ := externalHash(sources.Trainer)
		envHash, _ := externalHash(trainerEnv.Identity)
		begin.CheckpointIdentity = trainerapp.CheckpointIdentity{RunBOMSHA256: record.BOMSHA256, TrainerSHA256: trainerHash, TokenizerSHA256: codec.Identity().SHA256(), EnvironmentSHA256: envHash}
		if record.State != RunComplete {
			if record.State == RunFailed {
				return zero, fmt.Errorf("external run %s failed without a recoverable checkpoint", record.ID)
			}
			step := int64(0)
			if len(record.Checkpoints) > 0 {
				begin.Resume, err = trainerapp.LoadCheckpoint(begin, record.Checkpoints[len(record.Checkpoints)-1])
				if err != nil {
					return zero, err
				}
				step = begin.Resume.Step
			}
			raw, err := p.partition.TrainingRecords()
			if err != nil {
				return zero, err
			}
			tokens, err := training.NewTokenizedRecordSource(raw, codec, s.Objective, plan.Identity.Stages[index].Conversation)
			if err != nil {
				return zero, err
			}
			source, err := training.CachedPreparedSequenceSource(tokens, p.data, codec.Spec(), step, filepath.Join(cache, "prepared"), record.BOMSHA256, record.BOMSHA256, b.PreparedCacheMaxBytes)
			if err != nil {
				return zero, err
			}
			if record.State == RunRunning && len(record.Attempts) > 0 {
				previous := &record.Attempts[len(record.Attempts)-1]
				if previous.Finished == "" {
					previous.Finished = formatTime(b.clock()())
					previous.Error = "prior attempt ended without terminal bookkeeping"
				}
			}
			record.State = RunRunning
			modelRecord.State = RunRunning
			if err = writeJSONAtomic(filepath.Join(directory, "MODEL.json"), modelRecord); err != nil {
				return zero, err
			}
			started := b.clock()()
			record.Attempts = append(record.Attempts, ExternalAttempt{NodeLaunches: o.NodeLaunches[s.Name], Started: formatTime(started), Launch: evidence, Probe: current.Probe()})
			if err = writeJSONAtomic(recordPath, record); err != nil {
				return zero, err
			}
			if o.Publish != nil {
				nodePlan := ExternalNodePlan{Compose: c, Stage: s, BOM: bom, Begin: begin, CheckpointIdentity: begin.CheckpointIdentity, CheckpointDirectory: begin.CheckpointDirectory, Python: o.Python}
				if len(record.Checkpoints) > 0 {
					ref := record.Checkpoints[len(record.Checkpoints)-1]
					nodePlan.Checkpoint = &ref
				}
				if err = o.Publish(nodePlan); err != nil {
					return zero, err
				}
			}
			complete, trainErr := current.Train(ctx, begin, p.evaluations, source, func(event trainerapp.Event) error {
				if event.DurableCheckpoint != nil {
					record.Checkpoints = append(record.Checkpoints, *event.DurableCheckpoint)
				}
				if event.Evaluation != nil {
					record.Evaluations = append(record.Evaluations, *event.Evaluation)
				}
				if event.DurableCheckpoint != nil || event.Evaluation != nil {
					if err := writeJSONAtomic(recordPath, record); err != nil {
						return err
					}
				}
				telemetry := event.Telemetry()
				if err := appendTelemetry(filepath.Join(directory, TelemetryFilename), telemetryRow{Observed: b.clock()(), Started: started, RunID: record.ID, Stage: s.Name, Attempt: len(record.Attempts), Event: event.Kind, State: RunRunning, PlannedSteps: p.data.Steps, PlannedTokens: p.data.PlannedTokenCapacity, Training: &telemetry, Message: event.Message}); err != nil {
					return err
				}
				b.report(Progress{Phase: "training", Stage: s.Name, RunID: record.ID, Training: &telemetry, Message: event.Message})
				return nil
			})
			attempt := &record.Attempts[len(record.Attempts)-1]
			attempt.Finished = formatTime(b.clock()())
			if trainErr != nil {
				record.State = RunFailed
				if len(record.Checkpoints) > 0 && !training.IsNonRetryableWorkerError(trainErr) {
					record.State = RunInterrupted
				}
				attempt.Error = trainErr.Error()
				modelRecord.State = record.State
				modelRecord.Updated = formatTime(b.clock()())
				return zero, errors.Join(trainErr, writeJSONAtomic(recordPath, record), writeJSONAtomic(filepath.Join(directory, "MODEL.json"), modelRecord))
			}
			record.State = RunComplete
			record.Complete = &complete
			if err = writeJSONAtomic(recordPath, record); err != nil {
				return zero, err
			}
		}
		if record.Complete == nil {
			return zero, fmt.Errorf("external completion missing")
		}
		if err = trainerapp.VerifyPublished(begin.Paths.Artifacts, record.Complete.Artifacts); err != nil {
			return zero, err
		}
		runBytes, err := os.ReadFile(filepath.Join(runDir, "RUN-BOM.json"))
		if err != nil {
			return zero, err
		}
		ref := ExternalBOMRun{ID: record.ID, BOM: training.Artifact{Path: filepath.Join("runs", s.Name, "RUN-BOM.json"), SHA256: application.Digest(runBytes), Bytes: int64(len(runBytes))}}
		for _, a := range record.Complete.Artifacts {
			original := a
			a.Path = filepath.Join("runs", s.Name, "artifacts", a.Path)
			ref.Artifacts = append(ref.Artifacts, a)
			if a.Role == "weights" {
				initialization = &training.WorkerInitialization{SourceType: "run", SourceID: id, SourceRunID: record.ID, Artifact: original.Artifact, Path: filepath.Join(directory, a.Path)}
			}
		}
		refs = append(refs, ref)
		if _, err = WriteExternalModelBOM(directory, false, refs); err != nil {
			return zero, err
		}
	}
	modelRecord.State = RunComplete
	modelRecord.Updated = formatTime(b.clock()())
	if err = writeJSONAtomic(filepath.Join(directory, "MODEL.json"), modelRecord); err != nil {
		return zero, err
	}
	return InspectExternal(root, name)
}
