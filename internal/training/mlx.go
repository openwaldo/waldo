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
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/openwaldo/waldo/internal/mlxruntime"
)

const MLXRevision = "builtin-mlx-worker-schema-1-r8"

//go:embed workers/mlx.py
var mlxWorker []byte

type MLX struct {
	Python  string
	Version string
}

func (backend MLX) Descriptor() Descriptor {
	return Descriptor{
		Identity:  Identity{Name: "mlx", Revision: MLXRevision},
		Framework: "mlx",
		Capabilities: Capabilities{
			Objectives: []string{"causal-language-modeling", "assistant-response-modeling"}, CheckpointResume: true, Safetensors: true,
		},
	}
}

func (backend MLX) Run(ctx context.Context, request Request) (Observation, error) {
	return runPythonWorker(ctx, "MLX", backend.Python, mlxruntime.WithModel(mlxWorker), request)
}

type mlxProbe struct {
	PythonVersion string `json:"python_version"`
	MLXVersion    string `json:"mlx_version"`
	Accelerator   string `json:"accelerator"`
	MemoryBytes   uint64 `json:"memory_bytes"`
}

type MLXResolver struct {
	Candidates []string
	Probe      func(context.Context, string) (mlxProbe, error)
	OS         string
	Arch       string
}

func NewMLXResolver() Resolver { return MLXResolver{} }

func (resolver MLXResolver) Resolve(ctx context.Context, request ResolveRequest) (Selection, error) {
	hostOS, hostArch := resolver.OS, resolver.Arch
	if hostOS == "" {
		hostOS = runtime.GOOS
	}
	if hostArch == "" {
		hostArch = runtime.GOARCH
	}
	if hostOS != "darwin" || hostArch != "arm64" {
		return Selection{}, fmt.Errorf("no real training backend is available for %s/%s; MLX requires Apple Silicon", hostOS, hostArch)
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
		return Selection{}, fmt.Errorf("decode architecture for MLX: %w", err)
	}
	if architecture.Family != "decoder-transformer" {
		return Selection{}, fmt.Errorf("MLX backend does not support architecture family %q", architecture.Family)
	}
	if _, _, err := ResolveTokenizer(architecture.Tokenizer.Name, architecture.Tokenizer.Revision, architecture.VocabularySize); err != nil {
		return Selection{}, fmt.Errorf("MLX backend: %w", err)
	}
	candidates := resolver.Candidates
	if len(candidates) == 0 {
		candidates = pythonCandidates()
	}
	probe := resolver.Probe
	if probe == nil {
		probe = probeMLX
	}
	var failures []string
	for _, candidate := range candidates {
		facts, err := probe(ctx, candidate)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", candidate, err))
			continue
		}
		backend := MLX{Python: candidate, Version: facts.MLXVersion}
		descriptor := backend.Descriptor()
		return Selection{Backend: backend, Execution: Execution{
			Backend: descriptor.Identity, Framework: descriptor.Framework,
			Runtime:      fmt.Sprintf("%s; Python %s; MLX %s", candidate, facts.PythonVersion, facts.MLXVersion),
			Host:         Host{OS: hostOS, Architecture: hostArch},
			Accelerators: []Accelerator{{Manufacturer: "Apple", Model: facts.Accelerator, MemoryBytes: facts.MemoryBytes}},
			Nodes:        1, WorldSize: 1,
		}}, nil
	}
	detail := strings.Join(failures, "; ")
	if detail != "" {
		detail = ": " + detail
	}
	return Selection{}, fmt.Errorf("no usable MLX runtime found; install MLX into a Python 3 environment on PATH%s", detail)
}

func FakeResolver() Resolver {
	return ResolverFunc(func(_ context.Context, _ ResolveRequest) (Selection, error) {
		backend := Fake{}
		descriptor := backend.Descriptor()
		return Selection{Backend: backend, Execution: Execution{
			Backend: descriptor.Identity, Framework: descriptor.Framework, Runtime: "explicit-test-simulation",
			Host: Host{OS: runtime.GOOS, Architecture: runtime.GOARCH}, Nodes: 1, WorldSize: 1,
		}}, nil
	})
}

func pythonCandidates() []string {
	var candidates []string
	for _, name := range []string{"python3", "python"} {
		if path, err := exec.LookPath(name); err == nil {
			candidates = append(candidates, path)
		}
	}
	candidates = append(candidates, "/opt/homebrew/bin/python3", "/usr/local/bin/python3")
	seen := map[string]bool{}
	result := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate == "" || seen[candidate] {
			continue
		}
		if info, err := os.Stat(candidate); err != nil || info.IsDir() || info.Mode()&0o111 == 0 {
			continue
		}
		seen[candidate] = true
		result = append(result, candidate)
	}
	return result
}

const mlxProbeProgram = `
import importlib.metadata
import json
import platform
import subprocess
import sys
import mlx.core as mx
mx.eval(mx.array([1], dtype=mx.int32))
def sysctl(name, default):
    try:
        return subprocess.check_output(["/usr/sbin/sysctl", "-n", name], text=True).strip()
    except Exception:
        return default
print(json.dumps({
    "python_version": platform.python_version(),
    "mlx_version": importlib.metadata.version("mlx"),
    "accelerator": sysctl("machdep.cpu.brand_string", "Apple Silicon GPU"),
    "memory_bytes": int(sysctl("hw.memsize", "0")),
}))
`

func probeMLX(ctx context.Context, python string) (mlxProbe, error) {
	probeCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	command := exec.CommandContext(probeCtx, python, "-c", mlxProbeProgram)
	var stderr cappedBuffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		if probeCtx.Err() != nil {
			return mlxProbe{}, probeCtx.Err()
		}
		return mlxProbe{}, fmt.Errorf("probe failed%s", workerStderr(stderr.String()))
	}
	var facts mlxProbe
	if err := json.Unmarshal(bytes.TrimSpace(output), &facts); err != nil {
		return mlxProbe{}, fmt.Errorf("invalid probe output: %w", err)
	}
	if facts.PythonVersion == "" || facts.MLXVersion == "" || facts.Accelerator == "" {
		return mlxProbe{}, fmt.Errorf("incomplete probe output")
	}
	return facts, nil
}

type cappedBuffer struct {
	mutex     sync.Mutex
	data      []byte
	truncated bool
}

func (buffer *cappedBuffer) Write(data []byte) (int, error) {
	buffer.mutex.Lock()
	defer buffer.mutex.Unlock()
	const limit = 64 * 1024
	written := len(data)
	if len(data) > limit {
		data = data[len(data)-limit:]
		buffer.truncated = true
	}
	buffer.data = append(buffer.data, data...)
	if overflow := len(buffer.data) - limit; overflow > 0 {
		buffer.data = append(buffer.data[:0], buffer.data[overflow:]...)
		buffer.truncated = true
	}
	return written, nil
}

func (buffer *cappedBuffer) String() string {
	buffer.mutex.Lock()
	defer buffer.mutex.Unlock()
	if buffer.truncated {
		return "[earlier output truncated] " + string(buffer.data)
	}
	return string(buffer.data)
}

func workerSkipped(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	return "; non-protocol worker output: " + value
}

func workerStderr(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	return "; stderr: " + value
}
