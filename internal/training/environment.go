// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

const (
	BackendAuto       = "auto"
	BackendMLX        = "mlx"
	BackendTorchTitan = "torchtitan"
	BackendPyTorch    = "pytorch"
	BackendFake       = "fake"
)

var errPythonPackageNotFound = errors.New("Python package not found")

type pythonPackage struct {
	Python  string
	Version string
}

type Cluster struct {
	Nodes        int
	NodeRank     int
	WorldSize    int
	Rendezvous   string
	RendezvousID string
	Interface    string
	HCA          string
}

// EnvironmentResolver keeps host policy separate from framework adapters.
// A new adapter only needs to be registered in Resolvers; automatic host and
// installation selection remains here.
type EnvironmentResolver struct {
	Preference string
	Cluster    Cluster
	OS         string
	Arch       string
	Candidates []string
	Resolvers  map[string]Resolver
	Probe      func(context.Context, []string, string, string) (pythonPackage, error)
}

func NewEnvironmentResolver(preference string) Resolver {
	return NewEnvironmentResolverForCluster(preference, Cluster{})
}

func NewEnvironmentResolverForCluster(preference string, cluster Cluster) Resolver {
	return EnvironmentResolver{
		Preference: preference,
		Cluster:    cluster,
		Resolvers: map[string]Resolver{
			BackendMLX:        NewMLXResolver(),
			BackendTorchTitan: NewTorchTitanResolverForCluster(cluster),
			BackendPyTorch:    NewPyTorchResolver(),
			BackendFake:       FakeResolver(),
		},
	}
}

func (resolver EnvironmentResolver) Resolve(ctx context.Context, request ResolveRequest) (Selection, error) {
	var envelope struct {
		Family string `json:"family"`
	}
	if err := json.Unmarshal(request.Architecture, &envelope); len(request.Architecture) > 0 && err != nil {
		return Selection{}, err
	}
	if envelope.Family == BackendTransformers {
		if resolver.Preference != "" && resolver.Preference != BackendAuto && resolver.Preference != BackendPyTorch {
			return Selection{}, fmt.Errorf("Transformers engine conflicts with model.backend=%s; use auto for this experimental adapter", resolver.Preference)
		}
		if resolver.Cluster.Nodes > 1 || resolver.Cluster.WorldSize > 1 || resolver.Cluster.NodeRank > 0 {
			return Selection{}, fmt.Errorf("Transformers engine currently supports single-process training only")
		}
		return (TransformersResolver{Candidates: resolver.candidates()}).Resolve(ctx, request)
	}
	preference := strings.ToLower(strings.TrimSpace(resolver.Preference))
	if preference == "" {
		preference = BackendAuto
	}
	hostOS, hostArch := resolver.OS, resolver.Arch
	if hostOS == "" {
		hostOS = runtime.GOOS
	}
	if hostArch == "" {
		hostArch = runtime.GOARCH
	}

	selected := preference
	var detected *pythonPackage
	if preference == BackendAuto {
		var err error
		selected, detected, err = resolver.automaticBackend(ctx, hostOS, hostArch)
		if err != nil {
			return Selection{}, err
		}
	}
	if resolver.Cluster.Nodes > 1 && selected != BackendTorchTitan {
		return Selection{}, fmt.Errorf("multi-node training (--nodes %d) requires the TorchTitan backend, but model.backend=%s resolved to %s; set model.backend=torchtitan or run single-node", resolver.Cluster.Nodes, preference, backendDisplayName(selected))
	}
	if selected == BackendMLX && (hostOS != "darwin" || hostArch != "arm64") {
		return Selection{}, fmt.Errorf("model.backend=%s selected MLX, but MLX training requires macOS on Apple Silicon; use model.backend=auto on this %s/%s host", preference, hostOS, hostArch)
	}
	if selected == BackendTorchTitan || selected == BackendPyTorch {
		if detected == nil {
			distribution, module := selected, selected
			if selected == BackendPyTorch {
				distribution, module = "torch", "torch"
			}
			installation, err := resolver.probe()(ctx, resolver.candidates(), distribution, module)
			if err != nil {
				if errors.Is(err, errPythonPackageNotFound) {
					return Selection{}, backendInstallError(preference, selected, hostOS, hostArch)
				}
				return Selection{}, fmt.Errorf("probe selected %s backend: %w", selected, err)
			}
			detected = &installation
		}
		adapter := resolver.resolver(selected)
		if adapter == nil {
			return Selection{}, fmt.Errorf("model.backend=%s selected %s %s in %s, but this WALDO build does not yet include the %s execution adapter", preference, backendDisplayName(selected), detected.Version, detected.Python, backendDisplayName(selected))
		}
		return adapter.Resolve(ctx, request)
	}
	adapter := resolver.resolver(selected)
	if adapter == nil {
		return Selection{}, fmt.Errorf("model.backend=%s selected unsupported backend %q", preference, selected)
	}
	selection, err := adapter.Resolve(ctx, request)
	if err != nil && selected == BackendMLX && strings.Contains(err.Error(), "no usable MLX runtime") {
		return Selection{}, fmt.Errorf("model.backend=%s selected MLX, but it is not installed or usable; install it with `python3 -m pip install mlx` (requires macOS 14+, Apple Silicon, and native Python 3.10+): %w", preference, err)
	}
	return selection, err
}

func (resolver EnvironmentResolver) automaticBackend(ctx context.Context, hostOS, hostArch string) (string, *pythonPackage, error) {
	switch hostOS {
	case "darwin":
		// MLX is the sole automatic macOS choice. An Intel Mac receives the
		// platform-specific error in Resolve instead of an unrelated fallback.
		return BackendMLX, nil, nil
	case "linux":
		for _, candidate := range []struct {
			backend      string
			distribution string
			module       string
		}{
			{BackendTorchTitan, "torchtitan", "torchtitan"},
			{BackendPyTorch, "torch", "torch"},
		} {
			installation, err := resolver.probe()(ctx, resolver.candidates(), candidate.distribution, candidate.module)
			if err == nil {
				return candidate.backend, &installation, nil
			}
			if !errors.Is(err, errPythonPackageNotFound) {
				return "", nil, fmt.Errorf("probe %s: %w", backendDisplayName(candidate.backend), err)
			}
		}
		return "", nil, fmt.Errorf("model.backend=auto found neither TorchTitan nor PyTorch\n%s", linuxBackendInstallGuidance(ctx, hostOS, hostArch, resolver.candidates()))
	default:
		return "", nil, fmt.Errorf("model.backend=auto has no training backend policy for %s/%s; configure a supported backend explicitly", hostOS, hostArch)
	}
}

func (resolver EnvironmentResolver) resolver(name string) Resolver {
	if resolver.Resolvers != nil {
		return resolver.Resolvers[name]
	}
	if name == BackendMLX {
		return NewMLXResolver()
	}
	if name == BackendPyTorch {
		return NewPyTorchResolver()
	}
	if name == BackendTorchTitan {
		return NewTorchTitanResolverForCluster(resolver.Cluster)
	}
	if name == BackendFake {
		return FakeResolver()
	}
	return nil
}

func (resolver EnvironmentResolver) candidates() []string {
	if len(resolver.Candidates) > 0 {
		return resolver.Candidates
	}
	return pythonCandidates()
}

func (resolver EnvironmentResolver) probe() func(context.Context, []string, string, string) (pythonPackage, error) {
	if resolver.Probe != nil {
		return resolver.Probe
	}
	return probePythonPackage
}

func backendDisplayName(name string) string {
	switch name {
	case BackendMLX:
		return "MLX"
	case BackendTorchTitan:
		return "TorchTitan"
	case BackendPyTorch:
		return "PyTorch"
	default:
		return name
	}
}

func backendInstallError(preference, selected, hostOS, hostArch string) error {
	switch selected {
	case BackendTorchTitan:
		return fmt.Errorf("model.backend=%s selected TorchTitan, but the `torchtitan` Python package is not installed on %s/%s; follow https://github.com/pytorch/torchtitan#installation", preference, hostOS, hostArch)
	case BackendPyTorch:
		return fmt.Errorf("model.backend=%s selected PyTorch, but the `torch` Python package is not installed on %s/%s; choose the CUDA, ROCm, or CPU install for this host at https://pytorch.org/get-started/locally/", preference, hostOS, hostArch)
	default:
		return fmt.Errorf("model.backend=%s selected %s, but it is not installed on %s/%s", preference, selected, hostOS, hostArch)
	}
}

func linuxBackendInstallGuidance(ctx context.Context, hostOS, hostArch string, candidates []string) string {
	distribution := "Linux"
	if hostOS == "linux" {
		if data, err := os.ReadFile("/etc/os-release"); err == nil {
			distribution = linuxDistribution(data)
		}
	}
	python := "none found on PATH"
	if len(candidates) > 0 {
		python = strings.Join(candidates, ", ")
	}
	accelerator := detectLinuxAccelerator(ctx)
	prerequisite := "install Python 3.10+ and pip using this distribution's package manager"
	lower := strings.ToLower(distribution)
	switch {
	case strings.Contains(lower, "ubuntu"), strings.Contains(lower, "debian"), strings.Contains(lower, "mint"):
		prerequisite = "sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip"
	case strings.Contains(lower, "fedora"), strings.Contains(lower, "rhel"), strings.Contains(lower, "rocky"), strings.Contains(lower, "alma"), strings.Contains(lower, "centos"):
		prerequisite = "sudo dnf install -y python3 python3-pip"
	case strings.Contains(lower, "suse"):
		prerequisite = "sudo zypper install python3 python3-pip"
	}
	compute := "the CPU build"
	if strings.HasPrefix(accelerator, "NVIDIA") {
		compute = "the CUDA build matching this NVIDIA host"
	} else if strings.HasPrefix(accelerator, "AMD") {
		compute = "the ROCm build matching this AMD host"
	}
	return fmt.Sprintf(`detected host: %s (%s/%s)
detected Python: %s
detected accelerator: %s

Install prerequisites:
  %s

Create or activate a Python environment, then install %s using the command
generated by the official PyTorch selector:
  https://pytorch.org/get-started/locally/

Verify it in that same environment:
  python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'

For distributed TorchTitan instead, follow:
  https://github.com/pytorch/torchtitan#installation`, distribution, hostOS, hostArch, python, accelerator, prerequisite, compute)
}

func linuxDistribution(data []byte) string {
	values := map[string]string{}
	for _, line := range strings.Split(string(data), "\n") {
		key, value, found := strings.Cut(line, "=")
		if !found {
			continue
		}
		values[key] = strings.Trim(strings.TrimSpace(value), `"`)
	}
	if values["PRETTY_NAME"] != "" {
		return values["PRETTY_NAME"]
	}
	if values["NAME"] != "" {
		return strings.TrimSpace(values["NAME"] + " " + values["VERSION_ID"])
	}
	return "Linux"
}

func detectLinuxAccelerator(ctx context.Context) string {
	if command, err := exec.LookPath("nvidia-smi"); err == nil {
		probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		output, runErr := exec.CommandContext(probeCtx, command, "--query-gpu=name", "--format=csv,noheader").Output()
		cancel()
		if runErr == nil && strings.TrimSpace(string(output)) != "" {
			return "NVIDIA " + strings.Join(strings.Fields(strings.TrimSpace(string(output))), " ")
		}
		return "NVIDIA tooling present (GPU query failed)"
	}
	if _, err := exec.LookPath("rocminfo"); err == nil {
		return "AMD ROCm tooling present"
	}
	return "no NVIDIA or AMD GPU tooling detected"
}

const pythonPackageProbeProgram = `
import importlib.metadata
import importlib.util
import json
import sys
distribution, module = sys.argv[1], sys.argv[2]
spec = importlib.util.find_spec(module)
if spec is None:
    print(json.dumps({"installed": False}))
else:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    print(json.dumps({"installed": True, "version": version}))
`

func probePythonPackage(ctx context.Context, candidates []string, distribution, module string) (pythonPackage, error) {
	type result struct {
		Installed bool   `json:"installed"`
		Version   string `json:"version"`
	}
	var failures []string
	for _, python := range candidates {
		probeCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		command := exec.CommandContext(probeCtx, python, "-c", pythonPackageProbeProgram, distribution, module)
		var stderr cappedBuffer
		command.Stderr = &stderr
		output, err := command.Output()
		cancel()
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s%s", python, workerStderr(stderr.String())))
			continue
		}
		var facts result
		if err := json.Unmarshal(bytes.TrimSpace(output), &facts); err != nil {
			failures = append(failures, fmt.Sprintf("%s: invalid probe output", python))
			continue
		}
		if facts.Installed {
			return pythonPackage{Python: python, Version: facts.Version}, nil
		}
	}
	if len(failures) > 0 && len(failures) == len(candidates) {
		return pythonPackage{}, fmt.Errorf("no Python runtime could be probed: %s", strings.Join(failures, "; "))
	}
	return pythonPackage{}, errPythonPackageNotFound
}
