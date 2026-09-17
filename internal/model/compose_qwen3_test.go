// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

const qwen3SmokeCompose = "../../docs/examples/transformers-qwen3-smoke.yaml"

func TestQwen3SmokeCompose(t *testing.T) {
	compose, _, err := LoadCompose(qwen3SmokeCompose)
	if err != nil {
		t.Fatal(err)
	}
	spec := compose.Architecture.Transformers
	if compose.Base != nil || spec == nil || spec.ModelClass != "Qwen3ForCausalLM" || spec.ConfigClass != "Qwen3Config" {
		t.Fatal("expected a random-initialized Qwen3 model")
	}
	if compose.Architecture.Layers != 2 || compose.Architecture.HiddenSize != 64 || compose.Architecture.IntermediateSize != 160 || spec.Config["head_dim"] != float64(16) {
		t.Fatal("custom Qwen3 dimensions were not preserved")
	}
	path, err := ArchiveCompose(t.TempDir(), compose, "qwen3-smoke")
	if err != nil {
		t.Fatal(err)
	}
	archived, _, err := LoadCompose(path)
	if err != nil || !reflect.DeepEqual(compose, archived) {
		t.Fatalf("Qwen3 compose round trip changed the specification: %v", err)
	}
}

func TestTransformersQwen3Smoke(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" {
		t.Skip("set WALDO_TRANSFORMERS_PYTHON and WALDO_TRANSFORMERS_WHEEL for real CPU training")
	}
	compose, _, err := LoadCompose(qwen3SmokeCompose)
	if err != nil {
		t.Fatal(err)
	}
	// Use generated local corpus-BOM/Parquet fixtures instead of downloading books.
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
	builder := Builder{Root: t.TempDir()}
	inspection, err := builder.Compose(context.Background(), "qwen3-smoke", compose, []PreparedStage{preparedFixture(t, compose.Stages[0])})
	if err != nil {
		t.Fatal(err)
	}
	if len(inspection.Runs) != 1 || inspection.Runs[0].State != RunComplete {
		t.Fatal("Qwen3 compose did not complete one run")
	}
	observation := inspection.Runs[0].Observation
	if observation == nil || observation.Simulated || observation.Steps != 2 || observation.ConsumedTokens != 128 || observation.FinalLoss == nil || len(observation.Evaluations) != 1 {
		t.Fatalf("unexpected Qwen3 training evidence: %+v", observation)
	}
	run := inspection.RunBOMs[0]
	if run.Initialization != nil || run.Execution.Backend.Name != training.BackendTransformers || !reflect.DeepEqual(run.Execution.Package, &compose.Architecture.Transformers.Package) {
		t.Fatal("Qwen3 execution lost initialization or package provenance")
	}
	if len(observation.Artifacts) != 5 {
		t.Fatal("Qwen3 run is missing artifacts")
	}
	directory := filepath.Join(inspection.Path, "runs", runDirectoryName(inspection.Model.Runs[0]))
	for _, artifact := range observation.Artifacts {
		path := filepath.Join(directory, artifact.Path)
		if err := VerifyArtifactFile(path, artifact); err != nil {
			t.Fatal(err)
		}
		if filepath.Base(path) == "config.json" {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var config map[string]any
			if err := json.Unmarshal(data, &config); err != nil {
				t.Fatal(err)
			}
			for _, key := range []string{"model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim"} {
				if !reflect.DeepEqual(config[key], compose.Architecture.Transformers.Config[key]) {
					t.Fatalf("worker changed architecture.config.%s", key)
				}
			}
		}
		if filepath.Base(path) == "runtime.json" {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var evidence struct {
				Parameters int `json:"parameters"`
			}
			if err := json.Unmarshal(data, &evidence); err != nil {
				t.Fatal(err)
			}
			if evidence.Parameters != 102976 {
				t.Fatalf("custom Qwen3 has %d parameters, expected 102976", evidence.Parameters)
			}
			t.Logf("Qwen3: %d measured parameters; %d optimizer steps; %d token targets; loss %.4f", evidence.Parameters, observation.Steps, observation.ConsumedTokens, *observation.FinalLoss)
		}
	}
}
