// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package generalbase1b_test

import (
	"reflect"
	"testing"

	"github.com/openwaldo/waldo/internal/model"
)

func TestPromotedReference(t *testing.T) {
	compose, _, err := model.LoadCompose("compose.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if compose.Base != nil || compose.Interaction.Template != "" || len(compose.Stages) != 1 {
		t.Fatalf("reference is not fresh one-stage pretraining: %+v", compose)
	}
	forecast, err := model.ForecastCompose(compose)
	if err != nil {
		t.Fatal(err)
	}
	if forecast.ApproximateParameters != 1034512384 || forecast.PlannedTokens != 4844896256 {
		t.Fatalf("forecast = %d parameters/%d tokens", forecast.ApproximateParameters, forecast.PlannedTokens)
	}

	architecture := compose.Architecture
	if architecture.ContextTokens != 2048 || architecture.VocabularySize != 32000 || architecture.HiddenSize != 2048 || architecture.IntermediateSize != 5632 || architecture.Layers != 22 || architecture.AttentionHeads != 32 || architecture.KeyValueHeads != 4 || architecture.Dropout != 0 || architecture.QKNormalization || architecture.Initialization != "depth-scaled" || !architecture.TieEmbeddings || architecture.ParameterDType != "float32" {
		t.Fatalf("architecture changed: %+v", architecture)
	}
	if architecture.Tokenizer.Name != "waldo/byte-bpe" || architecture.Tokenizer.Revision != "sha256:a1ce9b055192b69c22b6be077ecd10aaf230ca94d5ce2bf244d10a871fecdab4" || architecture.Tokenizer.Artifact == nil {
		t.Fatalf("tokenizer changed: %+v", architecture.Tokenizer)
	}

	stage := compose.Stages[0]
	if stage.Name != "general-base-1b-coherence-5tpp" || stage.Type != "pre-training" || stage.Objective != "causal-language-modeling" || !reflect.DeepEqual(corpusPaths(stage.Corpora), []string{"core/common-pile/wikimedia", "core/synthetic/cosmopedia-v2", "core/common-pile/stackexchange", "core/synthetic/tinystories-gpt4-v2"}) {
		t.Fatalf("stage changed: %+v", stage)
	}
	wantWeights := []uint64{9, 6, 3, 2}
	for index, selection := range stage.Corpora {
		if selection.Weight == nil || *selection.Weight != wantWeights[index] {
			t.Fatalf("corpus %d weight = %v", index, selection.Weight)
		}
	}
	filter := stage.Corpora[1].Filter
	if filter == nil || !reflect.DeepEqual(filter.Metadata["audience"].Include, []string{"young_children"}) || !reflect.DeepEqual(filter.Metadata["format"].Include, []string{"textbook_narrative"}) {
		t.Fatalf("Cosmopedia filter changed: %+v", filter)
	}
	parameters := stage.Parameters
	if parameters.Profile != "causal-pretrain-weighted" || parameters.Tokens != 4844881920 || parameters.BatchSize != 8 || parameters.GradientAccumulation != 2 || parameters.SequenceLength != 2048 || parameters.LearningRate != 0.0004 || parameters.Optimizer != "adamw" || parameters.Schedule != "cosine" || parameters.Seed != 42 || parameters.CheckpointEvery == nil || *parameters.CheckpointEvery != 147855 || parameters.EvaluateEvery == nil || *parameters.EvaluateEvery != 147855 || parameters.EvaluationSelection != "stratified-lowest-sha256-v1" {
		t.Fatalf("training controls changed: %+v", parameters)
	}
}

func corpusPaths(selections []model.CorpusSelection) []string {
	paths := make([]string, len(selections))
	for index, selection := range selections {
		paths[index] = selection.Path
	}
	return paths
}
