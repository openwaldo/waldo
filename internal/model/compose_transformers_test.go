// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
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
	if resolved := continued.RunBOMs[1].Parameters; resolved.BatchSize != 2 || resolved.GradientAccumulation != 2 || resolved.TrainerResolution == nil {
		t.Fatalf("accumulation was not recorded as an immutable run fact: %+v", resolved)
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

func TestTransformersComposeRejectsNativeExecutionKnobs(t *testing.T) {
	data, err := os.ReadFile("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string]string{
		"learning rate":          strings.Replace(string(data), "steps: 2", "steps: 2\n      learning_rate: 0.001", 1),
		"accumulation":           strings.Replace(string(data), "steps: 2", "steps: 2\n      gradient_accumulation_steps: 1", 1),
		"compute precision":      strings.Replace(string(data), "steps: 2", "steps: 2\n      compute_precision: bfloat16", 1),
		"activation checkpoints": strings.Replace(string(data), "steps: 2", "steps: 2\n      activation_checkpointing: true", 1),
		"compile":                strings.Replace(string(data), "steps: 2", "steps: 2\n      compile: true", 1),
		"optimizer":              strings.Replace(string(data), "steps: 2", "steps: 2\n      optimizer: adamw", 1),
		"schedule":               strings.Replace(string(data), "steps: 2", "steps: 2\n      schedule: cosine", 1),
		"warmdown":               strings.Replace(string(data), "steps: 2", "steps: 2\n      warmdown_steps: 1", 1),
		"minimum rate":           strings.Replace(string(data), "steps: 2", "steps: 2\n      minimum_learning_rate_ratio: 0", 1),
		"checkpoint cadence":     strings.Replace(string(data), "steps: 2", "steps: 2\n      checkpoint_every: 0", 1),
		"fp16 and bf16":          strings.Replace(string(data), "logging_steps: 1", "logging_steps: 1\n        fp16: true\n        bf16: true", 1),
		"fp16 without fp32":      strings.Replace(strings.Replace(string(data), "dtype: float32", "dtype: bfloat16", 1), "logging_steps: 1", "logging_steps: 1\n        fp16: true", 1),
		"bf16 with fp16 weights": strings.Replace(strings.Replace(string(data), "dtype: float32", "dtype: float16", 1), "logging_steps: 1", "logging_steps: 1\n        bf16: true", 1),
		"fractional batch":       strings.Replace(string(data), "per_device_train_batch_size: 2", "per_device_train_batch_size: 1.5", 1),
		"missing learning rate":  strings.Replace(string(data), "learning_rate: 0.001", "", 1),
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bad.yaml")
			if err := os.WriteFile(path, []byte(input), 0600); err != nil {
				t.Fatal(err)
			}
			if _, _, err := LoadCompose(path); err == nil {
				t.Fatal("ambiguous or unsupported execution knob accepted")
			}
		})
	}
}

func transformersAccumulationCompose(t *testing.T) Compose {
	t.Helper()
	data, err := os.ReadFile("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	input := strings.Replace(strings.Replace(string(data), "per_device_train_batch_size: 2", "per_device_train_batch_size: 1", 1), "gradient_accumulation_steps: 1", "gradient_accumulation_steps: 2\n        bf16: true\n        gradient_checkpointing: true", 1)
	input = strings.Replace(input, "warmup_steps: 0", "warmup_steps: 0.5", 1)
	path := filepath.Join(t.TempDir(), "accumulation.yaml")
	if err := os.WriteFile(path, []byte(input), 0600); err != nil {
		t.Fatal(err)
	}
	compose, _, err := LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	return compose
}

func TestTransformersAccumulationReachesResolvedFactsAndForecast(t *testing.T) {
	compose := transformersAccumulationCompose(t)
	stage := compose.Stages[0]
	if stage.Parameters.BatchSize != 0 || stage.Parameters.GradientAccumulation != 0 || stage.Parameters.ComputePrecision != "" {
		t.Fatalf("normalized compose stores native duplicates: %+v", stage.Parameters)
	}
	resolved, err := stage.ResolveParameters()
	if err != nil {
		t.Fatal(err)
	}
	if resolved.BatchSize != 2 || resolved.GradientAccumulation != 2 || resolved.ComputePrecision != "bfloat16" || !resolved.ActivationCheckpointing {
		t.Fatalf("resolved %+v", resolved)
	}
	forecast, err := ForecastCompose(compose)
	if err != nil {
		t.Fatal(err)
	}
	fitness := forecast.Fitness.Stages[0]
	if len(fitness.Topologies) != 1 || fitness.Topologies[0].GlobalBatchSequences != 2 || fitness.Topologies[0].GlobalMicroBatchSequences != 1 || fitness.Topologies[0].RankLocalMicroBatchSequences != 1 {
		t.Fatalf("forecast topology %+v", fitness.Topologies)
	}
	if fitness.GlobalTokensPerOptimizerStep != 64 || fitness.WarmupPercent != 50 {
		t.Fatalf("forecast stage %+v", fitness)
	}
	// One sequence per micro-batch with recomputation must forecast less
	// activation memory than two sequences without it.
	single, err := ForecastCompose(func() Compose {
		other, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
		if err != nil {
			t.Fatal(err)
		}
		return other
	}())
	if err != nil {
		t.Fatal(err)
	}
	if forecast.Fitness.Memory.ActivationAndLogitsBytes*2 > single.Fitness.Memory.ActivationAndLogitsBytes {
		t.Fatalf("activation forecast ignored accumulation/recomputation: %d vs %d", forecast.Fitness.Memory.ActivationAndLogitsBytes, single.Fitness.Memory.ActivationAndLogitsBytes)
	}
}

type trainerResolvingBackend struct {
	backendFunc
	resolve func(training.ResolvedParameters) (training.ResolvedParameters, error)
}

func (backend trainerResolvingBackend) Descriptor() training.Descriptor {
	descriptor := backend.backendFunc.Descriptor()
	descriptor.Capabilities.ActivationCheckpointing = true
	return descriptor
}

func (backend trainerResolvingBackend) ResolveTrainer(_ context.Context, _ json.RawMessage, parameters training.ResolvedParameters) (training.ResolvedParameters, error) {
	return backend.resolve(parameters)
}

func pinnedTrainerResolution(parameters training.ResolvedParameters, warmup int64) training.TransformersTrainerResolution {
	arguments := map[string]any{"per_device_train_batch_size": 8, "per_device_eval_batch_size": 8, "gradient_accumulation_steps": 1, "learning_rate": 5e-05,
		"weight_decay": 0.0, "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-08, "lr_scheduler_type": "linear", "lr_scheduler_kwargs": nil,
		"warmup_steps": 0, "max_grad_norm": 1.0, "optim": "adamw_torch_fused", "optim_args": nil, "bf16": false, "fp16": false,
		"gradient_checkpointing": false, "gradient_checkpointing_kwargs": nil, "use_cache": false, "logging_steps": 500, "full_determinism": false}
	for key, value := range parameters.Trainer.Arguments {
		arguments[key] = value
	}
	return training.TransformersTrainerResolution{Arguments: arguments, WarmupSteps: warmup}
}

func TestTransformersRunBOMPinsTrainerResolutionBeforeExecution(t *testing.T) {
	compose := transformersAccumulationCompose(t)
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
	var executed training.ResolvedParameters
	backend := trainerResolvingBackend{
		backendFunc: func(_ context.Context, request training.Request) (training.Observation, error) {
			executed = request.Parameters
			return training.Observation{}, fmt.Errorf("stop after capturing the request")
		},
		resolve: func(parameters training.ResolvedParameters) (training.ResolvedParameters, error) {
			return parameters.WithTrainerResolution(pinnedTrainerResolution(parameters, 1))
		},
	}
	root := t.TempDir()
	builder := Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
	if _, err := builder.Compose(context.Background(), "accumulation", compose, []PreparedStage{preparedFixture(t, compose.Stages[0])}); err == nil {
		t.Fatal("expected captured backend failure")
	}
	inspection, err := Inspect(root, "accumulation")
	if err != nil {
		t.Fatal(err)
	}
	if len(inspection.RunBOMs) != 1 {
		t.Fatalf("runs = %d", len(inspection.RunBOMs))
	}
	recorded := inspection.RunBOMs[0].Parameters
	if recorded.TrainerResolution == nil || recorded.TrainerResolution.WarmupSteps != 1 || recorded.BatchSize != 2 || recorded.GradientAccumulation != 2 || recorded.ComputePrecision != "bfloat16" || !recorded.ActivationCheckpointing {
		t.Fatalf("run BOM parameters %+v", recorded)
	}
	if !reflect.DeepEqual(executed, recorded) {
		t.Fatalf("executed parameters differ from the immutable run BOM:\n%+v\n%+v", executed, recorded)
	}
}

func TestTransformersTrainerConflictFailsBeforeRunCreation(t *testing.T) {
	compose := transformersAccumulationCompose(t)
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
	for name, resolve := range map[string]func(training.ResolvedParameters) (training.ResolvedParameters, error){
		"host rejects": func(training.ResolvedParameters) (training.ResolvedParameters, error) {
			return training.ResolvedParameters{}, fmt.Errorf("mixed precision requires a GPU in this adapter")
		},
		"pinned drift": func(parameters training.ResolvedParameters) (training.ResolvedParameters, error) {
			resolution := pinnedTrainerResolution(parameters, 1)
			resolution.Arguments["gradient_accumulation_steps"] = 1
			return parameters.WithTrainerResolution(resolution)
		},
	} {
		t.Run(name, func(t *testing.T) {
			called := false
			backend := trainerResolvingBackend{
				backendFunc: func(context.Context, training.Request) (training.Observation, error) {
					called = true
					return training.Observation{}, nil
				},
				resolve: resolve,
			}
			root := t.TempDir()
			builder := Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
				return testSelection(backend), nil
			})}
			if _, err := builder.Compose(context.Background(), "conflict", compose, []PreparedStage{preparedFixture(t, compose.Stages[0])}); err == nil || !strings.Contains(err.Error(), "resolve Trainer settings") {
				t.Fatalf("conflict: %v", err)
			}
			if called {
				t.Fatal("backend executed despite Trainer conflict")
			}
			inspection, err := Inspect(root, "conflict")
			if err == nil && len(inspection.Model.Runs) != 0 {
				t.Fatal("run created before Trainer resolution failed")
			}
		})
	}
	compose.Stages[0].Parameters.Trainer.Arguments["gradient_checkpointing"] = false
	backend := backendFunc(func(context.Context, training.Request) (training.Observation, error) {
		return training.Observation{}, nil
	})
	builder := Builder{Root: t.TempDir(), Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
	if _, err := builder.Compose(context.Background(), "unresolvable", compose, []PreparedStage{preparedFixture(t, compose.Stages[0])}); err == nil || !strings.Contains(err.Error(), "cannot resolve Transformers Trainer settings") {
		t.Fatalf("backend without Trainer resolution: %v", err)
	}
}

// Opt-in real checks that the pinned Trainer resolution gates run creation
// and that resolved recomputation is what executes.
func TestTransformersRealTrainerResolution(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" {
		t.Skip("set WALDO_TRANSFORMERS_PYTHON and WALDO_TRANSFORMERS_WHEEL for real Trainer resolution")
	}
	t.Setenv("WALDO_TRANSFORMERS_DEVICE", "cpu")
	load := func(arguments map[string]any) (Compose, PreparedStage) {
		compose, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
		if err != nil {
			t.Fatal(err)
		}
		compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
		for key, value := range arguments {
			compose.Stages[0].Parameters.Trainer.Arguments[key] = value
		}
		return compose, preparedFixture(t, compose.Stages[0])
	}
	t.Run("host rejects before run", func(t *testing.T) {
		compose, prepared := load(map[string]any{"fp16": true})
		root := t.TempDir()
		_, err := (Builder{Root: root}).Compose(context.Background(), "cpu-fp16", compose, []PreparedStage{prepared})
		if err == nil || !strings.Contains(err.Error(), "resolve Trainer settings") || !strings.Contains(err.Error(), "mixed precision requires a GPU") {
			t.Fatalf("expected pre-run Trainer rejection: %v", err)
		}
		if entries, err := filepath.Glob(filepath.Join(root, "*", "runs", "*")); err != nil || len(entries) != 0 {
			t.Fatalf("run state created before Trainer rejection: %v %v", entries, err)
		}
	})
	t.Run("recomputation", func(t *testing.T) {
		compose, prepared := load(map[string]any{"gradient_checkpointing": true, "gradient_accumulation_steps": 2, "per_device_train_batch_size": 1, "warmup_steps": 0.5})
		inspection, err := (Builder{Root: t.TempDir()}).Compose(context.Background(), "recompute", compose, []PreparedStage{prepared})
		if err != nil {
			t.Fatal(err)
		}
		parameters := inspection.RunBOMs[0].Parameters
		if !parameters.ActivationCheckpointing || parameters.GradientAccumulation != 2 || parameters.TrainerResolution == nil || parameters.TrainerResolution.WarmupSteps != 1 || parameters.TrainerResolution.Arguments["optim"] != "adamw_torch_fused" {
			t.Fatalf("run BOM parameters %+v", parameters)
		}
		var evidence struct {
			Execution struct {
				Checkpointing bool  `json:"gradient_checkpointing"`
				MicroBatches  int64 `json:"micro_batches"`
			} `json:"execution"`
		}
		for _, artifact := range inspection.Runs[0].Observation.Artifacts {
			if strings.HasSuffix(artifact.Path, "/runtime.json") {
				data, err := os.ReadFile(filepath.Join(inspection.Path, "runs", runDirectoryName(inspection.Model.Runs[0]), artifact.Path))
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(data, &evidence); err != nil {
					t.Fatal(err)
				}
			}
		}
		if !evidence.Execution.Checkpointing || evidence.Execution.MicroBatches != 4 {
			t.Fatalf("executed %+v", evidence.Execution)
		}
	})
}

type transformersIdentityBackend struct{ trainerResolvingBackend }

func (backend transformersIdentityBackend) Descriptor() training.Descriptor {
	descriptor := backend.trainerResolvingBackend.Descriptor()
	descriptor.Identity = training.Identity{Name: training.BackendTransformers, Revision: training.TransformersRevision}
	descriptor.Framework = training.BackendTransformers
	return descriptor
}

// ADR 0044/0079 completion evidence is enforced for Transformers exactly as for
// the native real backends: the published artifact must be the verified best
// persisted candidate, even when the final step regressed.
// transformersIntrinsicMetrics returns the ADR 0081 metrics a worker reports
// for a summed held-out NLL over the run BOM's pinned evaluation view.
func transformersIntrinsicMetrics(set training.EvaluationSet, loss float64) map[string]float64 {
	nll := loss * float64(set.TokenTargets)
	metrics := map[string]float64{
		"heldout_loss": nll / float64(set.TokenTargets), "heldout_perplexity": math.Exp(loss), "heldout_nll_sum": nll,
		"heldout_target_tokens": float64(set.TokenTargets), "heldout_utf8_bytes": float64(set.TextBytes),
	}
	if set.TextBytes > 0 {
		metrics["heldout_bits_per_byte"] = nll / (math.Ln2 * float64(set.TextBytes))
	}
	return metrics
}

func writeTransformersArtifact(t *testing.T, directory, name string) training.Artifact {
	t.Helper()
	path := filepath.Join(directory, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	data := []byte(name)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	return training.Artifact{Path: "artifacts/" + name, SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}
}

// transformersCandidateObservation is a two-step Transformers completion
// whose final step (2) regressed, so step 1 is the best persisted candidate.
// It reports each evaluation as the worker does, then completes.
func transformersCandidateObservation(t *testing.T, request training.Request) training.Observation {
	t.Helper()
	tokens := request.Parameters.PlannedTokenCapacity
	evaluations := []training.Evaluation{
		{Step: 1, Tokens: tokens / 2, Metrics: transformersIntrinsicMetrics(request.EvaluationSet, 1.0)},
		{Step: 2, Tokens: tokens, Metrics: transformersIntrinsicMetrics(request.EvaluationSet, 1.5)},
	}
	if request.Report != nil {
		for index := range evaluations {
			evaluation := evaluations[index]
			evaluation.Metrics = cloneMetrics(evaluation.Metrics)
			request.Report(training.Event{Kind: "evaluation", Step: evaluation.Step, Tokens: evaluation.Tokens, Evaluation: &evaluation, Message: "candidate evaluated"})
		}
	}
	evaluations[0].Metrics["artifact_heldout_loss"] = evaluations[0].Metrics["heldout_loss"] + 0.001
	return training.Observation{
		Steps: request.Parameters.Steps, ConsumedTokens: tokens,
		Checkpoints: []training.Checkpoint{
			{Step: 1, Tokens: tokens / 2, Artifacts: []training.Artifact{writeTransformersArtifact(t, request.ArtifactDirectory, "checkpoints/step-00000001/model.safetensors")}},
			{Step: 2, Tokens: tokens, Artifacts: []training.Artifact{writeTransformersArtifact(t, request.ArtifactDirectory, "checkpoints/step-00000002/model.safetensors")}},
		},
		Evaluations:        evaluations,
		SelectedCheckpoint: &training.CheckpointSelection{Step: 1, Tokens: tokens / 2, Metric: "heldout_loss", Value: evaluations[0].Metrics["heldout_loss"]},
		Artifacts:          []training.Artifact{writeTransformersArtifact(t, request.ArtifactDirectory, "model.safetensors")},
	}
}

func transformersTestBuilder(root string, run backendFunc) Builder {
	backend := transformersIdentityBackend{trainerResolvingBackend{
		backendFunc: run,
		resolve: func(parameters training.ResolvedParameters) (training.ResolvedParameters, error) {
			return parameters.WithTrainerResolution(pinnedTrainerResolution(parameters, 0))
		},
	}}
	return Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
}

func TestTransformersCompletionRequiresVerifiedBestCandidate(t *testing.T) {
	observation := transformersCandidateObservation
	for _, test := range []struct {
		name   string
		mutate func(*training.Observation)
		want   string
	}{
		{name: "valid"},
		{name: "final replaces best", mutate: func(value *training.Observation) {
			value.SelectedCheckpoint = &training.CheckpointSelection{Step: 2, Tokens: value.ConsumedTokens, Metric: "heldout_loss", Value: 1.5}
			value.Evaluations[1].Metrics["artifact_heldout_loss"] = 1.5
		}, want: "not the best"},
		{name: "missing selection", mutate: func(value *training.Observation) { value.SelectedCheckpoint = nil }, want: "did not identify"},
		{name: "missing verification", mutate: func(value *training.Observation) { delete(value.Evaluations[0].Metrics, "artifact_heldout_loss") }, want: "does not verify"},
		{name: "serialization drift", mutate: func(value *training.Observation) { value.Evaluations[0].Metrics["artifact_heldout_loss"] = 1.5 }, want: "does not verify"},
		{name: "unpersisted selection", mutate: func(value *training.Observation) { value.Checkpoints = value.Checkpoints[1:] }, want: "persisted checkpoint"},
		{name: "missing evaluations", mutate: func(value *training.Observation) {
			value.Evaluations, value.SelectedCheckpoint = nil, nil
		}, want: "no metrics"},
		{name: "missing exact NLL", mutate: func(value *training.Observation) { delete(value.Evaluations[1].Metrics, "heldout_nll_sum") }, want: "does not report heldout_nll_sum"},
		{name: "missing bits per byte", mutate: func(value *training.Observation) { delete(value.Evaluations[0].Metrics, "heldout_bits_per_byte") }, want: "does not report heldout_bits_per_byte"},
		{name: "unpinned targets", mutate: func(value *training.Observation) { value.Evaluations[1].Metrics["heldout_target_tokens"]++ }, want: "run BOM pins"},
		{name: "unpinned bytes", mutate: func(value *training.Observation) { value.Evaluations[1].Metrics["heldout_utf8_bytes"]-- }, want: "run BOM pins"},
		{name: "loss is not summed NLL", mutate: func(value *training.Observation) { value.Evaluations[1].Metrics["heldout_nll_sum"] *= 1.01 }, want: "not its summed NLL over the pinned targets"},
		{name: "bits per byte uses another denominator", mutate: func(value *training.Observation) {
			metrics := value.Evaluations[1].Metrics
			metrics["heldout_bits_per_byte"] = metrics["heldout_nll_sum"] / (math.Ln2 * (metrics["heldout_utf8_bytes"] + 1))
		}, want: "pinned UTF-8 bytes"},
	} {
		t.Run(test.name, func(t *testing.T) {
			compose, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
			if err != nil {
				t.Fatal(err)
			}
			compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
			root := t.TempDir()
			builder := transformersTestBuilder(root, func(_ context.Context, request training.Request) (training.Observation, error) {
				if request.EvaluationSet.Records == 0 || request.EvaluationSet.TextBytes == 0 {
					t.Fatal("fixture has no held-out set")
				}
				// Report no durable events, so the completion alone is judged.
				request.Report = nil
				value := observation(t, request)
				if test.mutate != nil {
					test.mutate(&value)
				}
				return value, nil
			})
			inspection, err := builder.Compose(context.Background(), "selection", compose, []PreparedStage{preparedFixture(t, compose.Stages[0])})
			if test.want == "" {
				if err != nil {
					t.Fatal(err)
				}
				if selected := inspection.Runs[0].Observation.SelectedCheckpoint; selected == nil || selected.Step != 1 {
					t.Fatalf("published selection %+v, want the earlier best step 1", selected)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
			failed, inspectErr := Inspect(root, "selection")
			if inspectErr != nil {
				t.Fatal(inspectErr)
			}
			if run := failed.Runs[len(failed.Runs)-1]; run.State == RunComplete || run.Observation != nil {
				t.Fatalf("run completed without valid evidence: %+v", run)
			}
		})
	}
}

func transformersSelectionCompose(t *testing.T, steps int, learningRate string) Compose {
	t.Helper()
	data, err := os.ReadFile("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	input := strings.Replace(string(data), "steps: 2\n", fmt.Sprintf("steps: %d\n      evaluate_every: 1\n", steps), 1)
	input = strings.Replace(input, "sequence_length: 32", "sequence_length: 16", 1)
	input = strings.Replace(input, "learning_rate: 0.001", "learning_rate: "+learningRate, 1)
	input = strings.Replace(input, "lr_scheduler_type: linear", "lr_scheduler_type: constant", 1)
	path := filepath.Join(t.TempDir(), "selection.yaml")
	if err := os.WriteFile(path, []byte(input), 0600); err != nil {
		t.Fatal(err)
	}
	compose, _, err := LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
	return compose
}

// Opt-in real CPU test of ADR 0044/0079 publication: every evaluated
// candidate is persisted and verified, the full budget runs, a regressed final
// step does not replace the earlier best, and continuation starts from the
// verified published weights.
func TestTransformersPublishesVerifiedBestCandidate(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" {
		t.Skip("set WALDO_TRANSFORMERS_PYTHON and WALDO_TRANSFORMERS_WHEEL for real Transformers training")
	}
	// FP32 CPU is deterministic for this seed; the high constant rate makes the
	// tiny model overshoot on its final step (held-out 2.97 at step 3, 3.19 at 4).
	t.Setenv("WALDO_TRANSFORMERS_DEVICE", "cpu")
	const steps = 4
	compose := transformersSelectionCompose(t, steps, "0.1")
	builder := Builder{Root: t.TempDir()}
	prepared := preparedFixture(t, compose.Stages[0])
	inspection, err := builder.Compose(context.Background(), "selection", compose, []PreparedStage{prepared})
	if err != nil {
		t.Fatal(err)
	}
	run, pin := inspection.Runs[0], inspection.Model.Runs[0]
	observation := run.Observation
	if run.State != RunComplete || observation.Steps != steps || inspection.RunBOMs[0].Parameters.EvaluateEvery != 1 {
		t.Fatalf("expected the complete %d-step budget with periodic evaluation: %+v", steps, observation)
	}
	if len(observation.Evaluations) != steps || len(observation.Checkpoints) != steps {
		t.Fatalf("evaluations %d / candidates %d, want %d each", len(observation.Evaluations), len(observation.Checkpoints), steps)
	}
	best := observation.Evaluations[0]
	for index, evaluation := range observation.Evaluations {
		metrics := evaluation.Metrics
		t.Logf("step %d: publishable %.6f, live %.6f, compute %.6f", evaluation.Step, metrics["heldout_loss"], metrics["live_eager_heldout_loss"], metrics["live_eager_compute_heldout_loss"])
		if evaluation.Step != int64(index+1) || observation.Checkpoints[index].Step != evaluation.Step || observation.Checkpoints[index].Tokens != evaluation.Tokens {
			t.Fatalf("candidate %d is not a persisted evaluated checkpoint", index)
		}
		for _, name := range []string{"publishable_checkpoint_heldout_loss", "live_eager_heldout_loss", "live_eager_compute_heldout_loss", "artifact_loss_delta", "compute_precision_loss_delta"} {
			if _, ok := metrics[name]; !ok {
				t.Fatalf("step %d lacks %s", evaluation.Step, name)
			}
		}
		if metrics["heldout_loss"] != metrics["publishable_checkpoint_heldout_loss"] {
			t.Fatal("selection metric is not the publishable candidate loss")
		}
		if metrics["heldout_loss"] < best.Metrics["heldout_loss"] {
			best = evaluation
		}
	}
	final := observation.Evaluations[steps-1]
	selected := observation.SelectedCheckpoint
	if selected == nil || selected.Step != best.Step || selected.Value != best.Metrics["heldout_loss"] {
		t.Fatalf("selected %+v, want earliest best step %d", selected, best.Step)
	}
	if selected.Step == steps || final.Metrics["heldout_loss"] <= selected.Value {
		t.Fatalf("fixture no longer regresses on the final step; selected %+v final %.6f", selected, final.Metrics["heldout_loss"])
	}
	verification := best.Metrics
	if _, ok := verification["artifact_heldout_loss"]; !ok || math.Abs(verification["serialization_loss_delta"]) > 1e-6 {
		t.Fatalf("published artifact verification %+v", verification)
	}
	if _, ok := final.Metrics["artifact_heldout_loss"]; ok {
		t.Fatal("the regressed final candidate was verified as the published artifact")
	}
	runDirectory := filepath.Join(inspection.Path, "runs", runDirectoryName(pin))
	published := map[string]training.Artifact{}
	for _, artifact := range observation.Artifacts {
		published[artifact.Path] = artifact
	}
	weights := published["artifacts/model.safetensors"]
	selectedCandidate := observation.Checkpoints[selected.Step-1].Artifacts[0]
	finalCandidate := observation.Checkpoints[steps-1].Artifacts[0]
	tensors := func(artifact training.Artifact) map[string]string {
		return safetensorsTensorDigests(t, filepath.Join(runDirectory, filepath.FromSlash(artifact.Path)))
	}
	// The terminal file is re-serialized from the selected candidate.
	if publishedTensors := tensors(weights); !reflect.DeepEqual(publishedTensors, tensors(selectedCandidate)) || reflect.DeepEqual(publishedTensors, tensors(finalCandidate)) {
		t.Fatalf("published tensors are not the selected step-%d candidate", selected.Step)
	}
	data, err := os.ReadFile(filepath.Join(runDirectory, "artifacts", "runtime.json"))
	if err != nil {
		t.Fatal(err)
	}
	var runtime struct {
		Execution struct {
			Checks int64 `json:"finite_gradient_norm_checks"`
		} `json:"execution"`
	}
	if err := json.Unmarshal(data, &runtime); err != nil || runtime.Execution.Checks != steps {
		t.Fatalf("gradient norm checks %d, want one per optimizer step: %v", runtime.Execution.Checks, err)
	}
	prepared.Stage.Name = "continuation"
	continued, err := builder.Train(context.Background(), "selection", prepared)
	if err != nil {
		t.Fatal(err)
	}
	initialization := continued.RunBOMs[1].Initialization
	if initialization == nil || initialization.SourceRunID != pin.ID || initialization.Artifact.SHA256 != weights.SHA256 {
		t.Fatalf("continuation initialization %+v, want verified published step-%d weights %s", initialization, selected.Step, weights.SHA256)
	}
	t.Logf("continuation initialized from the verified step-%d weights %s", selected.Step, weights.SHA256)
}

// safetensorsTensorDigests maps tensor names to dtype, shape, and data digests,
// independent of header key order.
func safetensorsTensorDigests(t *testing.T, path string) map[string]string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) < 8 {
		t.Fatalf("%s is not safetensors", path)
	}
	size := binary.LittleEndian.Uint64(data[:8])
	var header map[string]json.RawMessage
	if size > uint64(len(data)-8) || json.Unmarshal(data[8:8+size], &header) != nil {
		t.Fatalf("%s has an invalid safetensors header", path)
	}
	body := data[8+size:]
	result := make(map[string]string, len(header))
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var tensor struct {
			DType   string   `json:"dtype"`
			Shape   []int64  `json:"shape"`
			Offsets []uint64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &tensor); err != nil || len(tensor.Offsets) != 2 || tensor.Offsets[0] > tensor.Offsets[1] || tensor.Offsets[1] > uint64(len(body)) {
			t.Fatalf("%s tensor %s has invalid metadata", path, name)
		}
		digest := sha256.Sum256(body[tensor.Offsets[0]:tensor.Offsets[1]])
		result[name] = fmt.Sprintf("%s%v:%x", tensor.DType, tensor.Shape, digest)
	}
	return result
}

func TestTransformersEvaluationCadenceCompose(t *testing.T) {
	smoke, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	periodic := transformersSelectionCompose(t, 4, "0.1")
	for _, test := range []struct {
		compose Compose
		want    int64
	}{{smoke, 0}, {periodic, 1}} {
		resolved, err := test.compose.Stages[0].ResolveParameters()
		if err != nil || resolved.EvaluateEvery != test.want || resolved.CheckpointEvery != 0 {
			t.Fatalf("evaluate_every %d checkpoint_every %d, want %d and 0: %v", resolved.EvaluateEvery, resolved.CheckpointEvery, test.want, err)
		}
	}
	path, err := ArchiveCompose(t.TempDir(), periodic, "periodic")
	if err != nil {
		t.Fatal(err)
	}
	if loaded, _, err := LoadCompose(path); err != nil || !reflect.DeepEqual(loaded, periodic) {
		t.Fatalf("archived periodic evaluation changed: %v", err)
	}
}
