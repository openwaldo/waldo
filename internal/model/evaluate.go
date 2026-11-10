// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/openwaldo/waldo/internal/training"
)

// Evaluation is a model's held-out loss on a stage's held-out records: a
// measurement of its current weights, not a run in its history.
type Evaluation struct {
	Loss            float64                `json:"heldout_loss"`
	Set             training.EvaluationSet `json:"evaluation_set"`
	WeightsSHA256   string                 `json:"weights_sha256"`
	CorpusBOMSHA256 string                 `json:"corpus_bom_sha256"` // the corpus selection scored, as a training run's BOM records it
}

// Evaluate scores the named model's current weights on the held-out records
// that training the prepared stage would select, using the same selection,
// tokenization, and backend as Train, without training or recording a run.
func (builder Builder) Evaluate(ctx context.Context, name string, prepared PreparedStage) (Evaluation, error) {
	inspection, err := Inspect(builder.Root, name)
	if err != nil {
		return Evaluation{}, err
	}
	stage := stageWithInteraction(prepared.Stage, inspection.Model.Interaction)
	prepared, err = PrepareStage(stage, prepared.BOM, prepared.Inputs)
	if err != nil {
		return Evaluation{}, err
	}
	if err := validateStage(stage, inspection.Model.Architecture); err != nil {
		return Evaluation{}, err
	}
	if err := prepared.BOM.Validate(); err != nil {
		return Evaluation{}, fmt.Errorf("stage %s corpus OpenWALDO BOM: %w", stage.Name, err)
	}
	if len(prepared.Inputs) == 0 {
		return Evaluation{}, fmt.Errorf("stage %s has no verified shard inputs", stage.Name)
	}
	parameters, err := stage.ResolvePlanningParameters()
	if err != nil {
		return Evaluation{}, fmt.Errorf("stage %s training profile: %w", stage.Name, err)
	}
	if parameters.Data.Order == "corpus-weighted-shuffle-v1" {
		if parameters.Data.CorpusWeights, err = resolveCorpusWeights(parameters.Data.CorpusWeights, prepared.BOM.Paths); err != nil {
			return Evaluation{}, fmt.Errorf("stage %s %w", stage.Name, err)
		}
	}
	tokenizer, codec, err := inspection.Model.Architecture.ResolveTokenizer()
	if err != nil {
		return Evaluation{}, fmt.Errorf("stage %s tokenizer: %w", stage.Name, err)
	}
	conversation := training.ConversationTransform{}
	if stage.Conversation != nil {
		conversation = *stage.Conversation
	}
	bomHash, err := hashJSON(prepared.BOM)
	if err != nil {
		return Evaluation{}, err
	}
	// The same identity Train caches its held-out selection under, so either can reuse the other's.
	selectionIdentity, err := hashJSON(stageSelectionIdentity{
		Schema: 1, ArchitectureSHA256: inspection.Model.ArchitectureSHA256, CorpusBOMSHA256: bomHash,
		Objective: stage.Objective, Conversation: conversation, Seed: parameters.Seed, Evaluation: parameters.Evaluation,
	})
	if err != nil {
		return Evaluation{}, err
	}
	var partition training.RecordPartition
	if cached, ok, loadErr := builder.loadStageSelection(selectionIdentity); loadErr != nil {
		return Evaluation{}, fmt.Errorf("stage %s cached evaluation selection: %w", stage.Name, loadErr)
	} else if ok {
		partition, err = training.NewRecordPartitionFromPreflight(ctx, prepared.Inputs, parameters, codec, stage.Objective, conversation, cached)
	} else {
		builder.report(Progress{Phase: "evaluate", Stage: stage.Name, Message: fmt.Sprintf("selecting deterministic held-out records across %d shards", len(prepared.Inputs))})
		partition, err = training.NewRecordPartitionContextWithTransform(ctx, prepared.Inputs, parameters, codec, stage.Objective, conversation, nil)
		if err == nil {
			err = builder.storeStageSelection(partition.Preflight(selectionIdentity, parameters, false))
		}
	}
	if err != nil {
		return Evaluation{}, fmt.Errorf("stage %s held-out evaluation partition: %w", stage.Name, err)
	}
	if partition.Evaluation.Records == 0 {
		return Evaluation{}, fmt.Errorf("stage %s selects no held-out records to evaluate", stage.Name)
	}
	records, err := partition.TrainingRecords()
	if err != nil {
		return Evaluation{}, fmt.Errorf("stage %s record stream: %w", stage.Name, err)
	}
	architectureJSON, err := json.Marshal(inspection.Model.Architecture)
	if err != nil {
		return Evaluation{}, err
	}
	resolver := builder.Resolver
	if resolver == nil {
		resolver = builtinResolver()
	}
	selection, err := resolver.Resolve(ctx, training.ResolveRequest{
		ArchitectureSHA256: inspection.Model.ArchitectureSHA256, Architecture: architectureJSON,
		Objectives: []string{stage.Objective}, ApproximateParameters: inspection.Model.Forecast.ApproximateParameters,
	})
	if err != nil {
		return Evaluation{}, fmt.Errorf("resolve evaluation backend: %w", err)
	}
	if err := validateSelection(selection, []string{stage.Objective}); err != nil {
		return Evaluation{}, err
	}
	if selection.Execution.WorldSize > 1 {
		return Evaluation{}, fmt.Errorf("evaluation runs on a single process; %s selected %d", selection.Execution.Backend.Name, selection.Execution.WorldSize)
	}
	selection.Execution.Parallelism.DataPlane = training.DataPlaneLocal
	initialization, err := resolveInitialization(inspection)
	if err != nil {
		return Evaluation{}, err
	}
	if initialization == nil {
		return Evaluation{}, fmt.Errorf("model %q has no weights to evaluate", name)
	}
	scratch, err := os.MkdirTemp("", "waldo-evaluate-*")
	if err != nil {
		return Evaluation{}, err
	}
	defer os.RemoveAll(scratch)
	builder.report(Progress{Phase: "evaluate", Stage: stage.Name, Message: fmt.Sprintf("scoring %s on %d held-out records with %s", name, partition.Evaluation.Records, selection.Execution.Backend.Name)})
	observation, err := selection.Backend.Run(ctx, training.Request{
		RunID: "evaluate", Stage: stage.Name, Objective: stage.Objective, Conversation: conversation,
		ArchitectureSHA256: inspection.Model.ArchitectureSHA256, Architecture: architectureJSON, Tokenizer: tokenizer,
		BOM: prepared.BOM, Inputs: prepared.Inputs, Parameters: parameters, Parallelism: selection.Execution.Parallelism,
		Records: records, EvaluationRecords: partition.EvaluationRecords(), EvaluationSet: partition.Evaluation,
		Initialization: initialization, ArtifactDirectory: scratch, ArtifactPrefix: "artifacts", EvaluateOnly: true,
	})
	if err != nil {
		return Evaluation{}, err
	}
	if observation.Simulated {
		return Evaluation{}, fmt.Errorf("backend %s only simulates; evaluation needs real weights scored by a real backend", selection.Execution.Backend.Name)
	}
	if len(observation.Evaluations) != 1 {
		return Evaluation{}, fmt.Errorf("evaluation of %s returned %d results, not one", name, len(observation.Evaluations))
	}
	loss, ok := observation.Evaluations[0].Metrics["heldout_loss"]
	if !ok {
		return Evaluation{}, fmt.Errorf("evaluation of %s returned no held-out loss", name)
	}
	return Evaluation{Loss: loss, Set: partition.Evaluation, WeightsSHA256: initialization.Artifact.SHA256, CorpusBOMSHA256: bomHash}, nil
}
