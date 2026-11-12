// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/pythonenv"
)

func TestResolveComposeHardwareDecompositionIdentity(t *testing.T) {
	code, err := os.ReadFile("testdata/trainer.py")
	if err != nil {
		t.Fatal(err)
	}
	script := strings.Replace(string(code), "'lr': {'type': 'number'},", "'lr': {'type': 'number'}, 'accumulation': {'type': 'integer', 'x-waldo-role': 'execution'}, 'attn': {'type': 'string', 'x-waldo-role': 'execution'},", 1)
	script = strings.Replace(script, "parameters = dict(lr=0.1, micro=global_batch // world)", "parameters = dict(lr=request['parameters'].get('lr', 0.1), micro=request['parameters'].get('micro', global_batch // world), accumulation=request['parameters'].get('accumulation', 1), attn=request['parameters'].get('attn', 'eager'))", 1)
	script = strings.Replace(script, "micro_batch_sequences=global_batch // world", "micro_batch_sequences=parameters['micro']", 1)
	script = strings.Replace(script, "accumulation_steps=2 if MODE == 'batch' else 1", "accumulation_steps=parameters['accumulation']", 1)
	c, err := Open(t.Context(), fixture(t, script), nil)
	if err != nil {
		t.Fatal(err)
	}
	d := data()
	d.GlobalBatchSequences = 32
	var results []Resolved
	for i, world := range []int{1, 4} {
		resolved, err := c.ResolveCompose(t.Context(), map[string]any{}, map[string]any{"lr": 0.1}, map[string]any{"micro": 8, "accumulation": []int{4, 1}[i], "attn": []string{"sdpa", "eager"}[i]}, d, World{World: applaunch.World{Nodes: 1, ProcessesPerNode: world, WorldSize: world}})
		if err != nil {
			t.Fatal(err)
		}
		results = append(results, resolved)
	}
	if results[0].RecipeSHA256 != results[1].RecipeSHA256 || results[0].ParametersSHA256 == results[1].ParametersSHA256 {
		t.Fatal("recipe/execution identity boundary failed")
	}
}

func TestCheckpointV2Fixture(t *testing.T) {
	raw, err := os.ReadFile("testdata/checkpoint-v2.json")
	if err != nil {
		t.Fatal(err)
	}
	var m CheckpointManifest
	if err := strict(raw, &m); err != nil {
		t.Fatal(err)
	}
	if m.Schema != 2 || m.ExecutionConfig == nil || m.ExecutionConfig.Determinism != "required" {
		t.Fatal("missing exact execution config")
	}
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var a, b any
	json.Unmarshal(raw, &a)
	json.Unmarshal(out, &b)
	if digest(a) != digest(b) {
		t.Fatal("schema-2 fixture changed on roundtrip")
	}
}

func TestCheckpointExecutionBoundary(t *testing.T) {
	ctx, req := setup(t, "ok")
	c, err := Open(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	b := begin(t, c)
	b.ExecutionConfig = ExecutionConfig{Profile: "single-process", Launcher: "python", TrainerEnvironment: pythonenv.ExecutionEvidence{InterpreterSHA256: digest("python"), PlatformTag: "linux-aarch64", Wheels: []string{"torch.whl"}}, TokenizerEnvironment: pythonenv.ExecutionEvidence{InterpreterSHA256: digest("tokenizer-python"), PlatformTag: "linux-aarch64", Wheels: []string{"tokenizer.whl"}}}
	var ref CheckpointReference
	_, err = c.Train(ctx, b, evals(), source, func(e Event) error {
		if e.DurableCheckpoint != nil {
			ref = *e.DurableCheckpoint
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(b, ref); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*Begin){
		"world.world_size":                       func(b *Begin) { b.World.WorldSize++ },
		"world.processes_per_node":               func(b *Begin) { b.World.ProcessesPerNode++ },
		"parameters.micro":                       func(b *Begin) { b.Parameters = map[string]any{"micro": 1} },
		"determinism":                            func(b *Begin) { b.Determinism = "off" },
		"profile":                                func(b *Begin) { b.ExecutionConfig.Profile = "distributed" },
		"launcher":                               func(b *Begin) { b.ExecutionConfig.Launcher = "torchrun" },
		"trainer_environment.platform_tag":       func(b *Begin) { b.ExecutionConfig.TrainerEnvironment.PlatformTag = "linux-x86_64" },
		"trainer_environment.interpreter_sha256": func(b *Begin) { b.ExecutionConfig.TrainerEnvironment.InterpreterSHA256 = digest("other") },
		"trainer_environment.wheels":             func(b *Begin) { b.ExecutionConfig.TrainerEnvironment.Wheels = []string{"other.whl"} },
		"tokenizer_environment.platform_tag":     func(b *Begin) { b.ExecutionConfig.TokenizerEnvironment.PlatformTag = "linux-x86_64" },
	} {
		t.Run(name, func(t *testing.T) {
			changed := b
			mutate(&changed)
			if _, err := LoadCheckpoint(changed, ref); err == nil || !strings.Contains(err.Error(), name) {
				t.Fatalf("missing field error: %v", err)
			}
		})
	}
	changed := b
	changed.ModelID = digest("different plan")
	if _, err := LoadCheckpoint(changed, ref); err == nil {
		t.Fatal("changed plan accepted")
	}
}
