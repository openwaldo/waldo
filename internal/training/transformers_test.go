// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestTransformersUnavailableDeviceProbe(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("Python required for subprocess probe test")
	}
	directory := t.TempDir()
	// Execute the actual probe with a minimal CPU-only torch stand-in.
	if err := os.WriteFile(filepath.Join(directory, "torch.py"), []byte("__version__ = 'test'\nclass CUDA:\n    def is_available(self): return False\ncuda = CUDA()\n"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PYTHONPATH", directory)
	t.Setenv("WALDO_TRANSFORMERS_DEVICE", "cuda")
	if _, err := ProbeTransformersDevice(t.Context(), python); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("unavailable CUDA: %v", err)
	}
}

func TestTransformersDoesNotOverrideExplicitHostPolicy(t *testing.T) {
	request := ResolveRequest{Architecture: json.RawMessage(`{"family":"huggingface-transformers"}`)}
	for _, preference := range []string{BackendFake, BackendMLX, BackendTorchTitan} {
		_, err := (EnvironmentResolver{Preference: preference}).Resolve(context.Background(), request)
		if err == nil || !strings.Contains(err.Error(), "conflicts") {
			t.Fatalf("%s: %v", preference, err)
		}
	}
	for _, cluster := range []Cluster{{Nodes: 2}, {WorldSize: 2}, {NodeRank: 1}} {
		_, err := (EnvironmentResolver{Cluster: cluster}).Resolve(context.Background(), request)
		if err == nil || !strings.Contains(err.Error(), "single-process") {
			t.Fatalf("%+v: %v", cluster, err)
		}
	}
}

func TestTransformersTrainerIsSoleOptimizerAuthority(t *testing.T) {
	parameters := Parameters{Steps: 2, BatchSize: 2, SequenceLength: 32, LearningRate: .001, Trainer: &TransformersTrainer{Class: "Trainer", Arguments: map[string]any{"learning_rate": .001, "per_device_train_batch_size": 2, "optim": "sgd"}}}
	resolved, err := ResolveParameters(parameters)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(resolved)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), `"optimizer"`) || strings.Contains(string(data), `"schedule"`) {
		t.Fatalf("native optimizer claims leaked into Trainer provenance: %s", data)
	}
	if !strings.Contains(string(data), `"optim":"sgd"`) {
		t.Fatal("Trainer optimizer was not preserved")
	}
}

func trainerParameters(arguments map[string]any) Parameters {
	return Parameters{Steps: 4, SequenceLength: 32, Trainer: &TransformersTrainer{Class: "Trainer", Arguments: arguments}}
}

func TestTransformersTrainerIsSoleExecutionAuthority(t *testing.T) {
	for name, test := range map[string]struct {
		arguments     map[string]any
		batch, accum  int64
		precision     string
		checkpointing bool
	}{
		"defaults":     {map[string]any{"per_device_train_batch_size": 2, "learning_rate": .001}, 2, 1, "auto", false},
		"accumulation": {map[string]any{"per_device_train_batch_size": 1, "gradient_accumulation_steps": 2, "learning_rate": .001}, 2, 2, "auto", false},
		"fp16":         {map[string]any{"per_device_train_batch_size": 2, "learning_rate": .001, "fp16": true}, 2, 1, "float16", false},
		"bf16":         {map[string]any{"per_device_train_batch_size": 2, "learning_rate": .001, "bf16": true, "fp16": false}, 2, 1, "bfloat16", false},
		"recompute":    {map[string]any{"per_device_train_batch_size": 3, "gradient_accumulation_steps": 3, "learning_rate": .001, "gradient_checkpointing": true}, 9, 3, "auto", true},
	} {
		t.Run(name, func(t *testing.T) {
			resolved, err := ResolveParameters(trainerParameters(test.arguments))
			if err != nil {
				t.Fatal(err)
			}
			if resolved.BatchSize != test.batch || resolved.GradientAccumulation != test.accum || resolved.ComputePrecision != test.precision || resolved.ActivationCheckpointing != test.checkpointing || resolved.LearningRate != .001 {
				t.Fatalf("resolved %+v", resolved)
			}
			if resolved.BatchSize/resolved.GradientAccumulation != test.batch/test.accum {
				t.Fatal("global micro-batch does not equal the Trainer per-device batch")
			}
		})
	}
	// Retained normalized composes may restate the projection exactly.
	restated := trainerParameters(map[string]any{"per_device_train_batch_size": 1, "gradient_accumulation_steps": 2, "learning_rate": .001})
	restated.BatchSize, restated.LearningRate = 2, .001
	if _, err := ResolveParameters(restated); err != nil {
		t.Fatal(err)
	}
}

func TestTransformersRejectsConflictingOrUnsupportedKnobs(t *testing.T) {
	base := func() Parameters {
		return trainerParameters(map[string]any{"per_device_train_batch_size": 1, "gradient_accumulation_steps": 2, "learning_rate": .001, "fp16": true})
	}
	weight, steps, ratio, every := .1, int64(1), .5, int64(2)
	for name, mutate := range map[string]func(*Parameters){
		"batch":           func(p *Parameters) { p.BatchSize = 1 },
		"learning rate":   func(p *Parameters) { p.LearningRate = .01 },
		"accumulation":    func(p *Parameters) { p.GradientAccumulation = 1 },
		"precision":       func(p *Parameters) { p.ComputePrecision = "bfloat16" },
		"auto precision":  func(p *Parameters) { p.ComputePrecision = "auto" },
		"recomputation":   func(p *Parameters) { p.ActivationCheckpointing = true },
		"compile":         func(p *Parameters) { p.Compile = true },
		"optimizer":       func(p *Parameters) { p.Optimizer = "adamw" },
		"schedule":        func(p *Parameters) { p.Schedule = "cosine" },
		"weight decay":    func(p *Parameters) { p.WeightDecay = &weight },
		"warmup":          func(p *Parameters) { p.WarmupSteps = &steps },
		"warmdown":        func(p *Parameters) { p.WarmdownSteps = &steps },
		"minimum rate":    func(p *Parameters) { p.MinimumRateRatio = &ratio },
		"checkpoints":     func(p *Parameters) { p.CheckpointEvery = &every },
		"fp16 and bf16":   func(p *Parameters) { p.Trainer.Arguments["bf16"] = true },
		"fractional":      func(p *Parameters) { p.Trainer.Arguments["per_device_train_batch_size"] = 1.5 },
		"string bool":     func(p *Parameters) { p.Trainer.Arguments["fp16"] = "yes" },
		"missing rate":    func(p *Parameters) { delete(p.Trainer.Arguments, "learning_rate") },
		"zero accumulate": func(p *Parameters) { p.Trainer.Arguments["gradient_accumulation_steps"] = 0 },
		"negative warmup": func(p *Parameters) { p.Trainer.Arguments["warmup_steps"] = -1 },
	} {
		t.Run(name, func(t *testing.T) {
			parameters := base()
			mutate(&parameters)
			if _, err := ResolvePlanningParameters(parameters); err == nil {
				t.Fatal("conflicting or unsupported knob accepted")
			}
		})
	}
	if err := ValidateTransformersPrecision("float16", "bfloat16"); err == nil {
		t.Fatal("fp16 accepted without float32 parameters")
	}
	if err := ValidateTransformersPrecision("bfloat16", "float16"); err == nil {
		t.Fatal("bf16 accepted with float16 parameters")
	}
}

func TestTransformersEvaluationCadenceIsOptIn(t *testing.T) {
	parameters := trainerParameters(map[string]any{"per_device_train_batch_size": 1, "learning_rate": .001})
	resolved, err := ResolveParameters(parameters)
	if err != nil || resolved.EvaluateEvery != 0 || resolved.CheckpointEvery != 0 {
		t.Fatalf("default Transformers cadence = evaluate %d checkpoint %d: %v", resolved.EvaluateEvery, resolved.CheckpointEvery, err)
	}
	every := int64(1)
	parameters.EvaluateEvery = &every
	if resolved, err = ResolveParameters(parameters); err != nil || resolved.EvaluateEvery != 1 || resolved.CheckpointEvery != 0 {
		t.Fatalf("periodic Transformers evaluation = %d checkpoint %d: %v", resolved.EvaluateEvery, resolved.CheckpointEvery, err)
	}
}

func TestTransformersNativeParametersAreNotProjected(t *testing.T) {
	native := Parameters{Steps: 4, BatchSize: 8, GradientAccumulation: 2, ComputePrecision: "bfloat16", ActivationCheckpointing: true, SequenceLength: 32, LearningRate: .001, Optimizer: "muon-adamw", Schedule: "warmup-stable-warmdown"}
	projected, err := ProjectTrainerParameters(native)
	if err != nil || !reflect.DeepEqual(projected, native) {
		t.Fatalf("native parameters changed: %+v %v", projected, err)
	}
}

func pinnedResolution(requested map[string]any, warmup int64) TransformersTrainerResolution {
	// Transformers 5.16.1 TrainingArguments defaults for the allowlist.
	arguments := map[string]any{"per_device_train_batch_size": 8, "per_device_eval_batch_size": 8, "gradient_accumulation_steps": 1, "learning_rate": 5e-05,
		"weight_decay": 0.0, "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-08, "lr_scheduler_type": "linear", "lr_scheduler_kwargs": nil,
		"warmup_steps": 0, "max_grad_norm": 1.0, "optim": "adamw_torch_fused", "optim_args": nil, "bf16": false, "fp16": false,
		"gradient_checkpointing": false, "gradient_checkpointing_kwargs": nil, "use_cache": false, "logging_steps": 500, "full_determinism": false}
	for key, value := range requested {
		arguments[key] = value
	}
	return TransformersTrainerResolution{Arguments: arguments, WarmupSteps: warmup}
}

func TestTransformersTrainerResolutionMustAgree(t *testing.T) {
	requested := map[string]any{"per_device_train_batch_size": 1, "gradient_accumulation_steps": 2, "learning_rate": .001, "bf16": true, "warmup_steps": .5}
	resolved, err := ResolveParameters(trainerParameters(requested))
	if err != nil {
		t.Fatal(err)
	}
	pinned, err := resolved.WithTrainerResolution(pinnedResolution(requested, 2))
	if err != nil {
		t.Fatal(err)
	}
	if pinned.TrainerResolution == nil || pinned.TrainerResolution.Arguments["optim"] != "adamw_torch_fused" {
		t.Fatal("pinned defaults were not recorded")
	}
	for name, mutate := range map[string]func(*TransformersTrainerResolution){
		"normalized request": func(r *TransformersTrainerResolution) { r.Arguments["learning_rate"] = .002 },
		"missing default":    func(r *TransformersTrainerResolution) { delete(r.Arguments, "optim") },
		"extra argument":     func(r *TransformersTrainerResolution) { r.Arguments["output_dir"] = "/tmp" },
		"precision":          func(r *TransformersTrainerResolution) { r.Arguments["bf16"] = false },
		"bool as number":     func(r *TransformersTrainerResolution) { r.Arguments["bf16"] = 1 },
		"recomputation":      func(r *TransformersTrainerResolution) { r.Arguments["gradient_checkpointing"] = true },
		"warmup semantics":   func(r *TransformersTrainerResolution) { r.WarmupSteps = 0 },
	} {
		t.Run(name, func(t *testing.T) {
			resolution := pinnedResolution(requested, 2)
			mutate(&resolution)
			if _, err := resolved.WithTrainerResolution(resolution); err == nil {
				t.Fatal("inconsistent pinned resolution accepted")
			}
		})
	}
	if _, err := (ResolvedParameters{}).WithTrainerResolution(pinnedResolution(requested, 2)); err == nil {
		t.Fatal("resolution accepted without a trainer")
	}
}

func TestTransformersRunRequiresPinnedResolution(t *testing.T) {
	resolved, err := ResolveParameters(trainerParameters(map[string]any{"per_device_train_batch_size": 2, "learning_rate": .001}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := (Transformers{}).Run(context.Background(), Request{Parameters: resolved}); err == nil || !strings.Contains(err.Error(), "pinned resolution") {
		t.Fatalf("run without pinned Trainer resolution: %v", err)
	}
}
