// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"path/filepath"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

// Expected totals are sum(p.numel()) of models constructed by the pinned
// Transformers 5.16.1 wheel with vocab 259, hidden 32, intermediate 64, two
// layers, four query heads, and two key/value heads.
func TestTransformersStructureMatchesPinnedProviderCounts(t *testing.T) {
	variants := []struct {
		name  string
		extra map[string]any
	}{
		{"base", nil},
		{"untied", map[string]any{"tie_word_embeddings": false}},
		{"tied", map[string]any{"tie_word_embeddings": true}},
		{"attention_bias", map[string]any{"attention_bias": true}},
		{"mlp_bias", map[string]any{"mlp_bias": true}},
		{"head_dim12", map[string]any{"head_dim": 12}},
	}
	expected := map[string][]uint64{
		"LlamaForCausalLM":   {35168, 35168, 26880, 35360, 35488, 38240},
		"Qwen2ForCausalLM":   {35296, 35296, 27008, 35296, 35296, 38432},
		"Qwen3ForCausalLM":   {127840, 127840, 119552, 129952, 127840, 38288},
		"MistralForCausalLM": {35168, 35168, 26880, 35168, 35168, 38240},
		"Gemma2ForCausalLM":  {217472, 225760, 217472, 221632, 217472, 30080},
		"Phi3ForCausalLM":    {35168, 35168, 26880, 35168, 35168, 38240},
		"Olmo2ForCausalLM":   {35264, 35264, 26976, 35456, 35264, 38384},
		"MixtralForCausalLM": {72288, 72288, 64000, 72288, 72288, 75360},
	}
	classes := training.TransformersModelClasses()
	for class, totals := range expected {
		for index, variant := range variants {
			t.Run(class+"/"+variant.name, func(t *testing.T) {
				config := map[string]any{"vocab_size": 259, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 128}
				if class == "MixtralForCausalLM" {
					config["num_local_experts"], config["num_experts_per_tok"] = 4, 2
				}
				for key, value := range variant.extra {
					config[key] = value
				}
				spec := &training.TransformersModel{ModelClass: class, ConfigClass: classes[class], Config: config}
				architecture, err := transformersArchitecture(spec, Tokenizer{Name: "byte", Revision: training.ByteTokenizerRevision})
				if err != nil {
					t.Fatal(err)
				}
				forecast, err := architecture.Forecast()
				if err != nil {
					t.Fatal(err)
				}
				if forecast.ApproximateParameters != totals[index] {
					t.Fatalf("forecast %d parameters, pinned provider measured %d", forecast.ApproximateParameters, totals[index])
				}
				a, err := AnalyzeArchitecture(architecture)
				if err != nil {
					t.Fatal(err)
				}
				if a.ParameterCountStatus != FitnessEstimated || a.DecompositionStatus != FitnessEstimated || a.ActiveParameterStatus != FitnessEstimated {
					t.Fatalf("provider statuses = %s/%s/%s", a.ParameterCountStatus, a.DecompositionStatus, a.ActiveParameterStatus)
				}
				if a.TokenIOParameters+a.AttentionParameters+a.MLPParameters+a.OtherParameters != a.TotalParameters {
					t.Fatalf("components do not sum: %+v", a)
				}
			})
		}
	}
}

func TestShippedTransformersExamplesForecast(t *testing.T) {
	measured := map[string]uint64{
		"transformers-smoke.yaml": 17600, "transformers-mistral-smoke.yaml": 26880, "transformers-phi3-smoke.yaml": 26880,
		"transformers-gemma2-smoke.yaml": 27008, "transformers-olmo2-smoke.yaml": 26976, "transformers-qwen3-smoke.yaml": 102976,
		"transformers-qwen3-tokenizer-smoke.yaml": 4871280, "transformers-mixtral-smoke.yaml": 64000,
	}
	paths, err := filepath.Glob("../../docs/examples/transformers-*.yaml")
	if err != nil || len(paths) != len(measured)+1 {
		t.Fatalf("examples = %v, %v", paths, err)
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			compose, _, err := LoadCompose(path)
			if err != nil {
				t.Fatal(err)
			}
			report, err := ForecastCompose(compose)
			if err != nil {
				t.Fatal(err)
			}
			a, compute := report.Fitness.Architecture, report.Fitness.Compute
			if a.ParameterCountStatus != FitnessEstimated || a.Basis == "" {
				t.Fatalf("parameter status %q basis %q", a.ParameterCountStatus, a.Basis)
			}
			if total, ok := measured[filepath.Base(path)]; ok {
				if report.ApproximateParameters != total || a.DecompositionStatus != FitnessEstimated || compute.ArchitectureAwareStatus != FitnessEstimated {
					t.Fatalf("total %d (measured %d), decomposition %s, compute %s", report.ApproximateParameters, total, a.DecompositionStatus, compute.ArchitectureAwareStatus)
				}
			} else if compose.Architecture.Transformers.ModelClass != transformersHybridModelClass ||
				a.DecompositionStatus != FitnessUnavailable || a.ActiveParameterStatus != FitnessUnavailable || a.ActiveParametersPerToken != nil ||
				compute.ArchitectureAwareStatus != FitnessUnavailable || compute.ArchitectureAwareTrainingFLOPs != 0 || compute.ActiveTrainingFLOPs != nil {
				t.Fatalf("hybrid proxy must not present a decomposition: %+v %+v", a, compute)
			}
			for _, stage := range report.Fitness.Stages {
				if len(stage.Topologies) != 1 || stage.Topologies[0].WorldSize != 1 || stage.TopologyNote == "" {
					t.Fatalf("stage %s topologies %+v note %q", stage.Name, stage.Topologies, stage.TopologyNote)
				}
			}
			if len(report.Configurations) == 0 || report.CatalogNote == "" {
				t.Fatalf("configurations %d note %q", len(report.Configurations), report.CatalogNote)
			}
			for _, configuration := range report.Configurations {
				if configuration.GPUs != 1 || configuration.Manufacturer == "Apple" {
					t.Fatalf("non-executable Transformers configuration %+v", configuration)
				}
			}
		})
	}
}

func TestMixtralSeparatesResidentAndActiveParameters(t *testing.T) {
	compose, _, err := LoadCompose("../../docs/examples/transformers-mixtral-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	report, err := ForecastCompose(compose)
	if err != nil {
		t.Fatal(err)
	}
	a, compute := report.Fitness.Architecture, report.Fitness.Compute
	// runtime.json defines active as total - experts + experts * k / E.
	if a.TotalParameters != 64000 || a.ExpertParameters != 49152 || a.RouterParameters != 256 || a.ExpertsPerLayer != 4 || a.ExpertsPerToken != 2 ||
		a.ActiveParametersPerToken == nil || *a.ActiveParametersPerToken != 64000-49152+49152*2/4 {
		t.Fatalf("mixture decomposition %+v", a)
	}
	if compute.ActiveTrainingFLOPs == nil || *compute.ActiveTrainingFLOPs >= compute.ConventionalTrainingFLOPs {
		t.Fatalf("active compute %v must be below resident compute %f", compute.ActiveTrainingFLOPs, compute.ConventionalTrainingFLOPs)
	}
	dense := compose
	dense.Architecture.Transformers = &training.TransformersModel{ModelClass: "MistralForCausalLM", ConfigClass: "MistralConfig", Config: map[string]any{}}
	densePerToken, _, err := transformersTrainingFLOPsPerToken(dense.Architecture, 32)
	if err != nil {
		t.Fatal(err)
	}
	if compute.ArchitectureAwareFLOPsPerToken <= densePerToken {
		t.Fatalf("two routed experts %f must exceed one dense MLP %f", compute.ArchitectureAwareFLOPsPerToken, densePerToken)
	}
}
