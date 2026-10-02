// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

func TestTransformersComposeRoundTrip(t *testing.T) {
	compose, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if compose.Base != nil || compose.Architecture.Transformers == nil {
		t.Fatal("expected explicit random Transformers initialization")
	}
	parameters, err := compose.Stages[0].ResolveParameters()
	if err != nil {
		t.Fatal(err)
	}
	if parameters.BatchSize != 2 || parameters.LearningRate != .001 || parameters.Trainer.Class != "Trainer" {
		t.Fatalf("incorrect resolution: %+v", parameters)
	}
	path, err := ArchiveCompose(t.TempDir(), compose, "smoke")
	if err != nil {
		t.Fatal(err)
	}
	loaded, _, err := LoadCompose(path)
	if err != nil || !reflect.DeepEqual(compose, loaded) {
		t.Fatalf("history changed config or trainer: %v", err)
	}
	before, _ := canonicalHash(compose.Architecture)
	compose.Architecture.Transformers.Package.Version = "5.16.2"
	after, _ := canonicalHash(compose.Architecture)
	if before == after {
		t.Fatal("package pin does not affect architecture interpretation identity")
	}
}

func TestTransformersComposeRejectsUnsafeOrAmbiguousInputs(t *testing.T) {
	data, err := os.ReadFile("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string]string{
		"schema":                     strings.Replace(string(data), "schema: 2", "schema: 1", 1),
		"provider":                   strings.Replace(string(data), "provider: huggingface-transformers", "provider: waldo-native", 1),
		"package version":            strings.Replace(string(data), "version: 5.16.1", "version: latest", 1),
		"hash":                       strings.Replace(string(data), "sha256: 2f2d", "sha256: 3g2d", 1),
		"unknown trainer":            strings.Replace(string(data), "logging_steps: 1", "logging_step_typo: 1", 1),
		"output override":            strings.Replace(string(data), "logging_steps: 1", "output_dir: /tmp/escape", 1),
		"publication":                strings.Replace(string(data), "logging_steps: 1", "push_to_hub: true", 1),
		"duplicate batch owner":      strings.Replace(string(data), "steps: 2", "steps: 2\n      batch_size: 8", 1),
		"unimplemented architecture": strings.Replace(string(data), "LlamaForCausalLM", "Qwen3_5ForCausalLM", 1),
		"extra document":             string(data) + "\n---\n{}",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bad.yaml")
			if err := os.WriteFile(path, []byte(input), 0600); err != nil {
				t.Fatal(err)
			}
			if _, _, err := LoadCompose(path); err == nil {
				t.Fatal("invalid compose accepted")
			}
		})
	}
}

// Opt-in real end-to-end test: Go compose -> resolved corpus BOM -> Trainer ->
// verified artifacts and durable run evidence -> managed continuation.
func TestTransformersRealLifecycle(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" {
		t.Skip("set WALDO_TRANSFORMERS_PYTHON and WALDO_TRANSFORMERS_WHEEL for real CPU training")
	}
	compose, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
	builder := Builder{Root: t.TempDir()}
	prepared := preparedFixture(t, compose.Stages[0])
	inspection, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{prepared})
	if err != nil {
		t.Fatal(err)
	}
	if len(inspection.Runs) != 1 || inspection.Runs[0].State != RunComplete || inspection.Runs[0].Observation.Simulated {
		t.Fatal("expected real complete Transformers run")
	}
	run := inspection.RunBOMs[0]
	if run.Execution.Backend.Name != training.BackendTransformers || !reflect.DeepEqual(run.Execution.Package, &compose.Architecture.Transformers.Package) || run.Parameters.Trainer == nil {
		t.Fatal("missing package/trainer provenance")
	}
	if len(inspection.Runs[0].Observation.Artifacts) != 5 || inspection.Runs[0].Observation.ConsumedTokens != 128 {
		t.Fatalf("unexpected observation: %+v", inspection.Runs[0].Observation)
	}
	for _, artifact := range inspection.Runs[0].Observation.Artifacts {
		if strings.HasSuffix(artifact.Path, "/runtime.json") {
			data, err := os.ReadFile(filepath.Join(inspection.Path, "runs", runDirectoryName(inspection.Model.Runs[0]), artifact.Path))
			if err != nil {
				t.Fatal(err)
			}
			var evidence struct {
				Parameters int `json:"parameters"`
			}
			if err := json.Unmarshal(data, &evidence); err != nil || evidence.Parameters <= 0 {
				t.Fatalf("missing measured parameter count: %v", err)
			}
		}
	}
	prepared.Stage.Name = "continuation"
	prepared.Stage.Parameters.Trainer.Arguments["gradient_accumulation_steps"] = float64(2)
	prepared.Stage.Parameters.Trainer.Arguments["per_device_train_batch_size"] = float64(1)
	continued, err := builder.Train(context.Background(), "smoke", prepared)
	if err != nil {
		t.Fatal(err)
	}
	if continued.RunBOMs[1].Initialization == nil {
		t.Fatal("continuation did not use previous verified weights")
	}
	if continued.Runs[1].Observation.ConsumedTokens != 128 {
		t.Fatalf("gradient accumulation observation: %+v; resolved: %+v", continued.Runs[1].Observation, continued.RunBOMs[1].Parameters)
	}
	for _, family := range []struct{ name, modelClass, configClass string }{{"qwen2", "Qwen2ForCausalLM", "Qwen2Config"}, {"qwen3", "Qwen3ForCausalLM", "Qwen3Config"}} {
		t.Run(family.name, func(t *testing.T) {
			other, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
			if err != nil {
				t.Fatal(err)
			}
			other.Architecture.Transformers.ModelClass = family.modelClass
			other.Architecture.Transformers.ConfigClass = family.configClass
			other.Architecture.Transformers.Config["model_type"] = family.name
			if _, err := builder.Initialize(family.name, other.Architecture); err != nil {
				t.Fatal(err)
			}
			trained, err := builder.Train(context.Background(), family.name, preparedFixture(t, other.Stages[0]))
			if err != nil {
				t.Fatal(err)
			}
			if trained.Runs[0].Observation.Steps != 2 || trained.Runs[0].Observation.Simulated {
				t.Fatal("expected two real provider training steps")
			}
		})
	}
}
