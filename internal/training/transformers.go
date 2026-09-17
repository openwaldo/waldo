// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
	"time"
)

const BackendTransformers = "huggingface-transformers"
const TransformersRevision = "builtin-transformers-worker-schema-1-r3"

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
func (trainer TransformersTrainer) Validate() error {
	if trainer.Class != "Trainer" {
		return fmt.Errorf("Transformers trainer.class must be Trainer")
	}
	allowed := map[string]bool{}
	for _, key := range strings.Fields("per_device_train_batch_size per_device_eval_batch_size gradient_accumulation_steps learning_rate weight_decay adam_beta1 adam_beta2 adam_epsilon lr_scheduler_type lr_scheduler_kwargs warmup_steps max_grad_norm optim optim_args bf16 fp16 gradient_checkpointing gradient_checkpointing_kwargs use_cache logging_steps full_determinism") {
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
	return Descriptor{Identity: Identity{Name: BackendTransformers, Revision: TransformersRevision}, Framework: BackendTransformers, Capabilities: Capabilities{Objectives: []string{"causal-language-modeling", "assistant-response-modeling"}, Safetensors: true}}
}

func (backend Transformers) Run(ctx context.Context, request Request) (Observation, error) {
	if request.Resume != nil {
		return Observation{}, fmt.Errorf("Transformers checkpoint resume is not supported yet")
	}
	if request.Parameters.Trainer == nil {
		return Observation{}, fmt.Errorf("Transformers run requires a trainer specification")
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
