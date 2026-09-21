// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

const PyTorchRevision = "builtin-pytorch-worker-schema-1-r13"

//go:embed workers/pytorch.py
var pyTorchWorker []byte

type PyTorch struct {
	Python  string
	Version string
	Device  string
}

func (backend PyTorch) Descriptor() Descriptor {
	return Descriptor{
		Identity:  Identity{Name: BackendPyTorch, Revision: PyTorchRevision},
		Framework: BackendPyTorch,
		Capabilities: Capabilities{
			Objectives: []string{"causal-language-modeling", "assistant-response-modeling"}, CheckpointResume: true, Safetensors: true,
			ActivationCheckpointing: true, Compile: true,
		},
	}
}

func (backend PyTorch) Run(ctx context.Context, request Request) (Observation, error) {
	device := backend.Device
	if device == "" {
		device = "cpu"
	}
	return runPythonWorker(ctx, "PyTorch", backend.Python, string(pyTorchWorker), request, device)
}

type pyTorchProbe struct {
	PythonVersion string `json:"python_version"`
	TorchVersion  string `json:"torch_version"`
	Device        string `json:"device"`
	Manufacturer  string `json:"manufacturer"`
	Accelerator   string `json:"accelerator"`
	MemoryBytes   uint64 `json:"memory_bytes"`
}

type PyTorchResolver struct {
	Candidates []string
	Probe      func(context.Context, string) (pyTorchProbe, error)
	OS         string
	Arch       string
}

func NewPyTorchResolver() Resolver { return PyTorchResolver{} }

func (resolver PyTorchResolver) Resolve(ctx context.Context, request ResolveRequest) (Selection, error) {
	hostOS, hostArch := resolver.OS, resolver.Arch
	if hostOS == "" {
		hostOS = runtime.GOOS
	}
	if hostArch == "" {
		hostArch = runtime.GOARCH
	}
	if hostOS != "linux" {
		return Selection{}, fmt.Errorf("PyTorch training currently requires Linux; this host is %s/%s", hostOS, hostArch)
	}
	var architecture struct {
		Family         string `json:"family"`
		VocabularySize uint64 `json:"vocabulary_size"`
		Tokenizer      struct {
			Name     string `json:"name"`
			Revision string `json:"revision"`
		} `json:"tokenizer"`
	}
	if err := json.Unmarshal(request.Architecture, &architecture); err != nil {
		return Selection{}, fmt.Errorf("decode architecture for PyTorch: %w", err)
	}
	if architecture.Family != "decoder-transformer" {
		return Selection{}, fmt.Errorf("PyTorch backend does not support architecture family %q", architecture.Family)
	}
	if _, _, err := ResolveTokenizer(architecture.Tokenizer.Name, architecture.Tokenizer.Revision, architecture.VocabularySize); err != nil {
		return Selection{}, fmt.Errorf("PyTorch backend: %w", err)
	}
	candidates := resolver.Candidates
	if len(candidates) == 0 {
		candidates = pythonCandidates()
	}
	probe := resolver.Probe
	if probe == nil {
		probe = probePyTorch
	}
	var failures []string
	for _, candidate := range candidates {
		facts, err := probe(ctx, candidate)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", candidate, err))
			continue
		}
		backend := PyTorch{Python: candidate, Version: facts.TorchVersion, Device: facts.Device}
		descriptor := backend.Descriptor()
		execution := Execution{
			Backend: descriptor.Identity, Framework: descriptor.Framework,
			Runtime: fmt.Sprintf("%s; Python %s; PyTorch %s; device %s", candidate, facts.PythonVersion, facts.TorchVersion, facts.Device),
			Host:    Host{OS: hostOS, Architecture: hostArch}, Nodes: 1, WorldSize: 1,
		}
		if facts.Device == "cuda" && facts.Accelerator != "" {
			execution.Accelerators = []Accelerator{{Manufacturer: facts.Manufacturer, Model: facts.Accelerator, MemoryBytes: facts.MemoryBytes}}
		}
		return Selection{Backend: backend, Execution: execution}, nil
	}
	detail := strings.Join(failures, "; ")
	if detail != "" {
		detail = ": " + detail
	}
	return Selection{}, fmt.Errorf("no usable PyTorch runtime found; choose the CUDA, ROCm, or CPU install for this Linux host at https://pytorch.org/get-started/locally/%s", detail)
}

const pyTorchProbeProgram = `
import json
import platform
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
manufacturer = ""
accelerator = ""
memory_bytes = 0
if device == "cuda":
    properties = torch.cuda.get_device_properties(0)
    accelerator = properties.name
    memory_bytes = properties.total_memory
    manufacturer = "AMD" if torch.version.hip else "NVIDIA"
else:
    accelerator = platform.processor() or platform.machine() or "CPU"
    manufacturer = "CPU"

# Allocate and execute a real operation on the selected device. Import success
# alone does not establish that the installed runtime is usable.
value = torch.tensor([1.0], device=device)
torch.sum(value).item()
print(json.dumps({
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "device": device,
    "manufacturer": manufacturer,
    "accelerator": accelerator,
    "memory_bytes": memory_bytes,
}))
`

func probePyTorch(ctx context.Context, python string) (pyTorchProbe, error) {
	probeCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	command := exec.CommandContext(probeCtx, python, "-c", pyTorchProbeProgram)
	var stderr cappedBuffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		if probeCtx.Err() != nil {
			return pyTorchProbe{}, probeCtx.Err()
		}
		return pyTorchProbe{}, fmt.Errorf("probe failed%s", workerStderr(stderr.String()))
	}
	var facts pyTorchProbe
	if err := json.Unmarshal(bytes.TrimSpace(output), &facts); err != nil {
		return pyTorchProbe{}, fmt.Errorf("invalid probe output: %w", err)
	}
	if facts.PythonVersion == "" || facts.TorchVersion == "" || facts.Device == "" || facts.Accelerator == "" {
		return pyTorchProbe{}, fmt.Errorf("incomplete probe output")
	}
	if facts.Device != "cpu" && facts.Device != "cuda" {
		return pyTorchProbe{}, fmt.Errorf("unsupported PyTorch device %q", facts.Device)
	}
	return facts, nil
}
