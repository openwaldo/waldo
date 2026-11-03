// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package model

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/tokenizerapp"
	"github.com/openwaldo/waldo/internal/model/trainerapp"
	"github.com/openwaldo/waldo/internal/training"
)

// ExternalNodePlan carries the primary's immutable pins to a node-local coordinator.
type ExternalNodePlan struct {
	Compose             ExternalCompose                 `json:"compose"`
	Stage               ExternalStage                   `json:"stage"`
	BOM                 ExternalRunBOM                  `json:"bom"`
	Begin               trainerapp.Begin                `json:"begin"`
	CheckpointIdentity  trainerapp.CheckpointIdentity   `json:"checkpoint_identity"`
	CheckpointDirectory string                          `json:"checkpoint_directory"`
	Checkpoint          *trainerapp.CheckpointReference `json:"checkpoint,omitempty"`
	Python              string                          `json:"python"`
}

// RunExternalSecondary verifies the same source, environment, resolved recipe and
// data pins before acknowledging readiness. It owns no primary lifecycle state.
func RunExternalSecondary(ctx context.Context, plan ExternalNodePlan, o ExternalOptions, scratch string, ready func() error) error {
	sources, err := plan.Compose.ResolveSources(ctx, filepath.Join(o.Cache, "sources"))
	if err != nil {
		return err
	}
	if !sameExternalJSON(sources.Trainer, plan.BOM.Identity.Trainer) {
		return fmt.Errorf("secondary trainer source mismatch")
	}
	env, err := sources.Trainer.EnsureEnvironment(ctx, plan.Compose.Trainer.Environment, o.Python, filepath.Join(o.Cache, "environments"))
	if err != nil {
		return err
	}
	if !sameExternalJSON(env.Identity, plan.BOM.Identity.Environment) {
		return fmt.Errorf("secondary trainer environment mismatch")
	}
	tokenEnv, err := sources.Tokenizer.EnsureEnvironment(ctx, plan.Compose.Tokenizer.Environment, o.Python, filepath.Join(o.Cache, "environments"))
	if err != nil {
		return err
	}
	codec, err := tokenizerapp.Open(ctx, applaunch.Request{Source: sources.Tokenizer, Environment: tokenEnv, Scratch: scratch, Command: "serve"}, plan.Compose.Tokenizer.Configuration)
	if err != nil {
		return err
	}
	defer codec.Close()
	if !sameExternalJSON(codec.Identity(), plan.BOM.Identity.Tokenizer) {
		return fmt.Errorf("secondary tokenizer mismatch")
	}
	prepared, err := prepareExternalStage(ctx, plan.Stage, plan.Compose.Interaction, o, codec)
	if err != nil {
		return err
	}
	if !sameExternalJSON(prepared.bom, plan.BOM.CorpusBOM) || !sameExternalJSON(prepared.partition.Evaluation, plan.BOM.EvaluationSet) || !sameExternalJSON(prepared.data, plan.Begin.Data) {
		return fmt.Errorf("secondary corpus/data/evaluation mismatch")
	}
	request := applaunch.Request{Source: sources.Trainer, Environment: env, Scratch: scratch, Command: "train", Profile: "distributed", World: o.World, NCCLInterface: o.NCCLInterface, NCCLHCA: o.NCCLHCA}
	client, err := trainerapp.Open(ctx, request, []string{plan.Stage.Objective})
	if err != nil {
		return err
	}
	resolved, err := client.Resolve(ctx, plan.Compose.Architecture, plan.Stage.Trainer, prepared.data, trainerapp.World{World: o.World})
	if err != nil {
		return err
	}
	if !sameExternalJSON(resolved, plan.BOM.Resolved) {
		return fmt.Errorf("secondary resolved trainer parameters mismatch")
	}
	begin := plan.Begin
	begin.CheckpointIdentity = plan.CheckpointIdentity
	begin.CheckpointDirectory = plan.CheckpointDirectory
	// Verify the primary checkpoint against its original begin before changing
	// coordinator-local paths and node rank.
	if plan.Checkpoint != nil {
		begin.Resume, err = trainerapp.LoadCheckpoint(begin, *plan.Checkpoint)
		if err != nil {
			return err
		}
	}
	begin.World = o.World
	begin.Paths = trainerapp.Paths{Artifacts: filepath.Join(scratch, "artifacts"), Checkpoints: filepath.Join(scratch, "checkpoints"), Scratch: scratch}
	for _, path := range []string{begin.Paths.Artifacts, begin.Paths.Checkpoints} {
		if err = os.MkdirAll(path, 0700); err != nil {
			return err
		}
	}
	raw, err := prepared.partition.TrainingRecords()
	if err != nil {
		return err
	}
	transform := training.ConversationTransform{}
	if plan.Stage.Conversation != nil {
		transform = *plan.Stage.Conversation
		transform.Tools = plan.Compose.Interaction.Tools
	}
	tokens, err := training.NewTokenizedRecordSource(raw, codec, plan.Stage.Objective, transform)
	if err != nil {
		return err
	}
	step := int64(0)
	if begin.Resume != nil {
		step = begin.Resume.Step
	}
	source, err := training.CachedPreparedSequenceSource(tokens, begin.Data, codec.Spec(), step, filepath.Join(o.Cache, "prepared"), plan.CheckpointIdentity.RunBOMSHA256, plan.CheckpointIdentity.RunBOMSHA256, o.PreparedCacheMaxBytes)
	if err != nil {
		return err
	}
	if err = ready(); err != nil {
		return err
	}
	_, err = client.Train(ctx, begin, prepared.evaluations, source, nil)
	return err
}

// ExternalCorpusProvider binds a secondary to the resolved BOM without index traversal.
func ExternalCorpusProvider(bom corpus.BOM, inputs []training.Input) func(context.Context, ExternalStage) (corpus.BOM, []training.Input, error) {
	return func(context.Context, ExternalStage) (corpus.BOM, []training.Input, error) { return bom, inputs, nil }
}
