// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
	"time"
)

const BackendTransformers = "huggingface-transformers"
const TransformersRevision = "builtin-transformers-worker-schema-1-r8"

//go:embed transformers_models.json
var transformersModelsJSON string

func TransformersModelClasses() map[string]string {
	var classes map[string]string
	if err := json.Unmarshal([]byte(transformersModelsJSON), &classes); err != nil {
		panic(err)
	}
	return classes
}

// PackagePin identifies distribution bytes, not just an import's version string.
type PackagePin struct {
	Distribution string `json:"distribution" yaml:"distribution"`
	Version      string `json:"version" yaml:"version"`
	Artifact     string `json:"artifact" yaml:"artifact"`
	SHA256       string `json:"sha256" yaml:"sha256"`
}

func (pin PackagePin) Validate() error {
	if pin.Distribution != "transformers" || !regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+$`).MatchString(pin.Version) || pin.Artifact != "transformers-"+pin.Version+"-py3-none-any.whl" || !regexp.MustCompile(`^[a-f0-9]{64}$`).MatchString(pin.SHA256) {
		return fmt.Errorf("Transformers requires an exact distribution version, matching wheel filename, and lowercase SHA-256")
	}
	return nil
}

// TransformersModel is also the immutable interpretation dependency for config.
// Native lifecycle projections are derived from Config, never user-supplied.
type TransformersModel struct {
	Package     PackagePin     `json:"package" yaml:"package"`
	ModelClass  string         `json:"model_class" yaml:"model_class"`
	ConfigClass string         `json:"config_class" yaml:"config_class"`
	Config      map[string]any `json:"config" yaml:"config"`
}

func (model TransformersModel) Validate() error {
	if err := model.Package.Validate(); err != nil {
		return err
	}
	classes := TransformersModelClasses()
	if classes[model.ModelClass] != model.ConfigClass || model.ConfigClass == "" {
		return fmt.Errorf("unsupported Transformers model/config pair %s / %s", model.ModelClass, model.ConfigClass)
	}
	if len(model.Config) == 0 {
		return fmt.Errorf("Transformers architecture.config is required")
	}
	if model.ModelClass == "MixtralForCausalLM" {
		if _, _, err := model.ExpertCounts(); err != nil {
			return err
		}
		if model.Config["output_router_logits"] != true {
			return fmt.Errorf("Mixtral requires output_router_logits: true")
		}
	}
	return nil
}

func (model TransformersModel) ExpertCounts() (uint64, uint64, error) {
	data, err := json.Marshal(model.Config)
	if err != nil {
		return 0, 0, err
	}
	var counts struct {
		Experts uint64 `json:"num_local_experts"`
		Active  uint64 `json:"num_experts_per_tok"`
	}
	if err := json.Unmarshal(data, &counts); err != nil {
		return 0, 0, err
	}
	if counts.Experts == 0 || counts.Active == 0 || counts.Active > counts.Experts {
		return 0, 0, fmt.Errorf("Mixtral requires explicit positive num_local_experts and num_experts_per_tok <= num_local_experts")
	}
	return counts.Experts, counts.Active, nil
}

type TransformersTrainer struct {
	Class     string         `json:"class" yaml:"class"`
	Arguments map[string]any `json:"arguments" yaml:"arguments"`
}

// Arguments affecting WALDO's data stream, filesystem, topology, publication,
// or stopping rules are deliberately not passthroughs. Unknown options fail.
var transformersTrainerArguments = strings.Fields("per_device_train_batch_size per_device_eval_batch_size gradient_accumulation_steps learning_rate weight_decay adam_beta1 adam_beta2 adam_epsilon lr_scheduler_type lr_scheduler_kwargs warmup_steps max_grad_norm optim optim_args bf16 fp16 gradient_checkpointing gradient_checkpointing_kwargs use_cache logging_steps full_determinism")

// TransformersTrainerArguments lists the allowlisted TrainingArguments fields.
func TransformersTrainerArguments() []string {
	return append([]string(nil), transformersTrainerArguments...)
}

func (trainer TransformersTrainer) Validate() error {
	if trainer.Class != "Trainer" {
		return fmt.Errorf("Transformers trainer.class must be Trainer")
	}
	allowed := map[string]bool{}
	for _, key := range transformersTrainerArguments {
		allowed[key] = true
	}
	for key := range trainer.Arguments {
		if !allowed[key] {
			return fmt.Errorf("trainer.arguments.%s is unsupported or owned by WALDO", key)
		}
	}
	_, err := json.Marshal(trainer.Arguments)
	return err
}

// TransformersExecution projects the Trainer arguments that carry WALDO's
// portable execution facts (ADR 0071, 0073, 0083). Omitted values use the
// Trainer defaults: accumulation 1, no fp16/bf16 autocast, no gradient
// checkpointing, and no warmup.
type TransformersExecution struct {
	PerDeviceBatchSize      int64
	GradientAccumulation    int64
	LearningRate            float64
	ComputePrecision        string
	ActivationCheckpointing bool
	Warmup                  float64
}

func (trainer TransformersTrainer) Execution() (TransformersExecution, error) {
	data, err := json.Marshal(trainer.Arguments)
	if err != nil {
		return TransformersExecution{}, err
	}
	var arguments struct {
		Batch         *int64   `json:"per_device_train_batch_size"`
		Accumulation  *int64   `json:"gradient_accumulation_steps"`
		Rate          *float64 `json:"learning_rate"`
		FP16          *bool    `json:"fp16"`
		BF16          *bool    `json:"bf16"`
		Checkpointing *bool    `json:"gradient_checkpointing"`
		Warmup        *float64 `json:"warmup_steps"`
	}
	if err := json.Unmarshal(data, &arguments); err != nil {
		return TransformersExecution{}, fmt.Errorf("trainer.arguments: %w", err)
	}
	execution := TransformersExecution{GradientAccumulation: 1, ComputePrecision: "auto"}
	if arguments.Batch == nil || arguments.Rate == nil {
		return TransformersExecution{}, fmt.Errorf("trainer requires per_device_train_batch_size and learning_rate")
	}
	execution.PerDeviceBatchSize, execution.LearningRate = *arguments.Batch, *arguments.Rate
	if arguments.Accumulation != nil {
		execution.GradientAccumulation = *arguments.Accumulation
	}
	if execution.PerDeviceBatchSize <= 0 || execution.GradientAccumulation <= 0 || execution.PerDeviceBatchSize > math.MaxInt64/execution.GradientAccumulation || execution.LearningRate <= 0 || math.IsInf(execution.LearningRate, 0) {
		return TransformersExecution{}, fmt.Errorf("trainer requires positive per_device_train_batch_size, gradient_accumulation_steps, and learning_rate")
	}
	fp16, bf16 := arguments.FP16 != nil && *arguments.FP16, arguments.BF16 != nil && *arguments.BF16
	switch {
	case fp16 && bf16:
		return TransformersExecution{}, fmt.Errorf("trainer fp16 and bf16 are mutually exclusive")
	case fp16:
		execution.ComputePrecision = "float16"
	case bf16:
		execution.ComputePrecision = "bfloat16"
	}
	execution.ActivationCheckpointing = arguments.Checkpointing != nil && *arguments.Checkpointing
	if arguments.Warmup != nil {
		execution.Warmup = *arguments.Warmup
	}
	if execution.Warmup < 0 || math.IsInf(execution.Warmup, 0) {
		return TransformersExecution{}, fmt.Errorf("trainer warmup_steps must be finite and non-negative")
	}
	return execution, nil
}

// GlobalBatchSize is the ADR 0071 optimizer batch while the engine is
// single-process; a multi-process adapter must keep it fixed.
func (execution TransformersExecution) GlobalBatchSize() int64 {
	return execution.PerDeviceBatchSize * execution.GradientAccumulation
}

// WarmupSteps follows TrainingArguments.get_warmup_steps: values of at least 1
// are step counts and smaller values are a fraction of the optimizer steps.
func (execution TransformersExecution) WarmupSteps(steps int64) int64 {
	if execution.Warmup >= 1 {
		return int64(execution.Warmup)
	}
	return int64(math.Ceil(float64(steps) * execution.Warmup))
}

// ValidateTransformersPrecision rejects mixed-precision requests that do not
// match the model's parameter dtype.
func ValidateTransformersPrecision(computePrecision, parameterDType string) error {
	if computePrecision == "float16" && parameterDType != "float32" {
		return fmt.Errorf("trainer fp16 scaling requires float32 model parameters")
	}
	if computePrecision == "bfloat16" && parameterDType != "float32" && parameterDType != "bfloat16" {
		return fmt.Errorf("trainer bf16 requires float32 or bfloat16 model parameters")
	}
	return nil
}

// TransformersTrainerResolution is the pinned package's effective value for
// every allowlisted TrainingArguments field, resolved on the selected host
// before a run exists. The worker must reproduce it exactly.
type TransformersTrainerResolution struct {
	Arguments   map[string]any `json:"arguments"`
	WarmupSteps int64          `json:"warmup_steps"`
}

// WithTrainerResolution records a pinned Trainer resolution after checking
// that it agrees with the requested arguments and the resolved native facts.
func (parameters ResolvedParameters) WithTrainerResolution(resolution TransformersTrainerResolution) (ResolvedParameters, error) {
	if parameters.Trainer == nil {
		return ResolvedParameters{}, fmt.Errorf("Trainer resolution requires a trainer specification")
	}
	if len(resolution.Arguments) != len(transformersTrainerArguments) {
		return ResolvedParameters{}, fmt.Errorf("Trainer resolution must report every allowlisted argument")
	}
	for _, key := range transformersTrainerArguments {
		effective, exists := resolution.Arguments[key]
		if !exists {
			return ResolvedParameters{}, fmt.Errorf("Trainer resolution is missing %s", key)
		}
		if requested, exists := parameters.Trainer.Arguments[key]; exists {
			same, err := sameJSON(requested, effective)
			if err != nil {
				return ResolvedParameters{}, err
			}
			if !same {
				return ResolvedParameters{}, fmt.Errorf("pinned Trainer resolved %s to %v, not the requested %v", key, effective, requested)
			}
		}
	}
	execution, err := (TransformersTrainer{Class: parameters.Trainer.Class, Arguments: resolution.Arguments}).Execution()
	if err != nil {
		return ResolvedParameters{}, fmt.Errorf("pinned Trainer resolution: %w", err)
	}
	if execution.GlobalBatchSize() != parameters.BatchSize || execution.GradientAccumulation != parameters.GradientAccumulation || execution.LearningRate != parameters.LearningRate || execution.ComputePrecision != parameters.ComputePrecision || execution.ActivationCheckpointing != parameters.ActivationCheckpointing {
		return ResolvedParameters{}, fmt.Errorf("pinned Trainer resolution differs from resolved batch, accumulation, learning rate, precision, or recomputation")
	}
	if resolution.WarmupSteps != execution.WarmupSteps(parameters.Steps) || resolution.WarmupSteps > parameters.Steps {
		return ResolvedParameters{}, fmt.Errorf("pinned Trainer resolved %d warmup steps, not %d of %d", resolution.WarmupSteps, execution.WarmupSteps(parameters.Steps), parameters.Steps)
	}
	parameters.TrainerResolution = &resolution
	return parameters, nil
}

func sameJSON(left, right any) (bool, error) {
	a, err := json.Marshal(left)
	if err != nil {
		return false, err
	}
	b, err := json.Marshal(right)
	if err != nil {
		return false, err
	}
	return bytes.Equal(a, b), nil
}

//go:embed workers/transformers.py
var transformersWorkerBody string

var transformersWorker = "MODEL_CLASSES = " + transformersModelsJSON + "\n" + transformersWorkerBody

// TransformersPythonSupport supplies the same wheel verifier and pinned
// tokenizer loader to inference. Execute with a non-main module name.
func TransformersPythonSupport() string { return transformersWorker }

type Transformers struct {
	Python, Wheel string
	Device        PyTorchDeviceFacts
}

func (backend Transformers) Descriptor() Descriptor {
	return Descriptor{Identity: Identity{Name: BackendTransformers, Revision: TransformersRevision}, Framework: BackendTransformers, Capabilities: Capabilities{Objectives: []string{"causal-language-modeling", "assistant-response-modeling"}, Safetensors: true, ActivationCheckpointing: true}}
}

func (backend Transformers) Run(ctx context.Context, request Request) (Observation, error) {
	if request.Resume != nil {
		return Observation{}, fmt.Errorf("Transformers checkpoint resume is not supported yet")
	}
	if request.Parameters.Trainer == nil || request.Parameters.TrainerResolution == nil {
		return Observation{}, fmt.Errorf("Transformers run requires a trainer specification and its pinned resolution")
	}
	if err := request.Parameters.Trainer.Validate(); err != nil {
		return Observation{}, err
	}
	request.PreTokenize = true
	deviceJSON, err := json.Marshal(backend.Device)
	if err != nil {
		return Observation{}, err
	}
	return runPythonWorker(ctx, "Transformers", backend.Python, transformersWorker, request, backend.Wheel, string(deviceJSON))
}

// ResolveTrainer asks the pinned package on this host for its effective
// TrainingArguments before a run exists, so the run BOM records what executes.
func (backend Transformers) ResolveTrainer(ctx context.Context, architecture json.RawMessage, parameters ResolvedParameters) (ResolvedParameters, error) {
	if parameters.Trainer == nil {
		return ResolvedParameters{}, fmt.Errorf("Transformers run requires a trainer specification")
	}
	request, err := json.Marshal(map[string]any{"architecture": architecture, "parameters": parameters, "arguments": transformersTrainerArguments})
	if err != nil {
		return ResolvedParameters{}, err
	}
	deviceJSON, err := json.Marshal(backend.Device)
	if err != nil {
		return ResolvedParameters{}, err
	}
	probeCtx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()
	command := exec.CommandContext(probeCtx, backend.Python, "-c", transformersWorker, "--resolve-trainer", backend.Wheel, string(request), string(deviceJSON))
	var stderr cappedBuffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		return ResolvedParameters{}, fmt.Errorf("resolve pinned Trainer arguments: %w%s", err, workerStderr(stderr.String()))
	}
	var resolution TransformersTrainerResolution
	decoder := json.NewDecoder(bytes.NewReader(output))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&resolution); err != nil {
		return ResolvedParameters{}, fmt.Errorf("invalid pinned Trainer resolution: %w", err)
	}
	return parameters.WithTrainerResolution(resolution)
}

type TransformersResolver struct{ Candidates []string }

func (resolver TransformersResolver) Resolve(ctx context.Context, request ResolveRequest) (Selection, error) {
	var architecture struct {
		Transformers *TransformersModel `json:"transformers"`
	}
	if err := json.Unmarshal(request.Architecture, &architecture); err != nil {
		return Selection{}, err
	}
	if architecture.Transformers == nil {
		return Selection{}, fmt.Errorf("missing Transformers architecture")
	}
	if err := architecture.Transformers.Validate(); err != nil {
		return Selection{}, err
	}
	if request.Parallelism != "" && request.Parallelism != ParallelismAuto {
		return Selection{}, fmt.Errorf("Transformers currently supports one process on CPU or one GPU")
	}
	wheel := os.Getenv("WALDO_TRANSFORMERS_WHEEL")
	if wheel == "" {
		return Selection{}, fmt.Errorf("set WALDO_TRANSFORMERS_WHEEL to the local wheel matching training.package; WALDO does not install packages automatically")
	}
	candidates := resolver.Candidates
	if python := os.Getenv("WALDO_TRANSFORMERS_PYTHON"); python != "" {
		candidates = []string{python}
	}
	if len(candidates) == 0 {
		candidates = pythonCandidates()
	}
	modelJSON, err := json.Marshal(architecture.Transformers)
	if err != nil {
		return Selection{}, err
	}
	var failures []string
	for _, python := range candidates {
		probeCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
		command := exec.CommandContext(probeCtx, python, "-c", transformersWorker, "--probe", wheel, string(modelJSON))
		var stderr cappedBuffer
		command.Stderr = &stderr
		output, err := command.Output()
		cancel()
		if err != nil {
			failures = append(failures, python+": "+err.Error()+workerStderr(stderr.String()))
			continue
		}
		var facts struct {
			Runtime string `json:"runtime"`
		}
		if err := json.Unmarshal(output, &facts); err != nil || facts.Runtime == "" {
			failures = append(failures, python+": invalid Transformers probe output")
			continue
		}
		device, err := ProbeTransformersDevice(ctx, python)
		if err != nil {
			failures = append(failures, python+": "+err.Error())
			continue
		}
		backend := Transformers{Python: python, Wheel: wheel, Device: device}
		descriptor := backend.Descriptor()
		runtimeJSON, _ := json.Marshal(map[string]any{"software": json.RawMessage(facts.Runtime), "hardware": device})
		execution := Execution{Backend: descriptor.Identity, Framework: descriptor.Framework, Runtime: string(runtimeJSON), Package: &architecture.Transformers.Package, Host: Host{OS: runtime.GOOS, Architecture: runtime.GOARCH}, Nodes: 1, WorldSize: 1}
		if device.Device == "cuda" {
			execution.Accelerators = []Accelerator{{Manufacturer: device.Manufacturer, Model: device.Accelerator, MemoryBytes: device.MemoryBytes}}
		}
		return Selection{Backend: backend, Execution: execution}, nil
	}
	return Selection{}, fmt.Errorf("no verified Transformers runtime: %s", strings.Join(failures, "; "))
}
