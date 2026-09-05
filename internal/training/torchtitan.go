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
	"net"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"syscall"
	"time"
)

const TorchTitanRevision = "builtin-torchtitan-worker-schema-1-r9"

type TorchTitan struct {
	Python     string
	Version    string
	LocalProcs int
	Nodes      int
	NodeRank   int
	Rendezvous string
	Interface  string
	HCA        string
	Secondary  bool
}

func (backend TorchTitan) Descriptor() Descriptor {
	return Descriptor{
		Identity:  Identity{Name: BackendTorchTitan, Revision: TorchTitanRevision},
		Framework: BackendTorchTitan,
		Capabilities: Capabilities{
			Objectives: []string{"causal-language-modeling", "assistant-response-modeling"}, CheckpointResume: true, Distributed: true, Safetensors: true,
		},
	}
}

func (backend TorchTitan) Run(ctx context.Context, request Request) (Observation, error) {
	if backend.Python == "" {
		return Observation{}, fmt.Errorf("TorchTitan Python runtime is required")
	}
	if backend.LocalProcs < 1 {
		return Observation{}, fmt.Errorf("TorchTitan requires at least one local process")
	}
	if backend.Nodes > 1 {
		if _, _, err := net.SplitHostPort(backend.Rendezvous); err != nil {
			return Observation{}, fmt.Errorf("multi-node TorchTitan rendezvous %q must be host:port: %w", backend.Rendezvous, err)
		}
		if backend.NodeRank < 0 || backend.NodeRank >= backend.Nodes {
			return Observation{}, fmt.Errorf("TorchTitan node rank %d is out of range for %d nodes", backend.NodeRank, backend.Nodes)
		}
		if !backend.Secondary && backend.NodeRank != 0 {
			return Observation{}, fmt.Errorf("primary TorchTitan node must be rank 0, not %d", backend.NodeRank)
		}
	}
	if err := os.MkdirAll(request.ArtifactDirectory, 0o755); err != nil {
		return Observation{}, fmt.Errorf("create TorchTitan artifact directory: %w", err)
	}
	request.PreTokenize = true
	worker, err := os.CreateTemp(request.ArtifactDirectory, ".waldo-torchtitan-worker-*.py")
	if err != nil {
		return Observation{}, fmt.Errorf("stage embedded TorchTitan worker: %w", err)
	}
	workerPath := worker.Name()
	defer os.Remove(workerPath)
	if _, err := worker.Write(pyTorchWorker); err != nil {
		_ = worker.Close()
		return Observation{}, err
	}
	if err := worker.Sync(); err != nil {
		_ = worker.Close()
		return Observation{}, err
	}
	if err := worker.Close(); err != nil {
		return Observation{}, err
	}
	command := exec.CommandContext(ctx, backend.Python, backend.launchArguments(workerPath, request)...)
	command.Env = backend.environment()
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	command.Cancel = func() error {
		return syscall.Kill(-command.Process.Pid, syscall.SIGKILL)
	}
	if backend.Secondary {
		return runWorkerStreamJoin(ctx, "TorchTitan", command, request)
	}
	return runWorkerCommand(ctx, "TorchTitan", command, request)
}

func (backend TorchTitan) launchArguments(workerPath string, request Request) []string {
	arguments := []string{"-m", "torch.distributed.run"}
	if backend.Nodes > 1 {
		host, port, _ := net.SplitHostPort(backend.Rendezvous)
		arguments = append(arguments,
			fmt.Sprintf("--nnodes=%d", backend.Nodes),
			fmt.Sprintf("--node-rank=%d", backend.NodeRank),
			fmt.Sprintf("--master-addr=%s", host),
			fmt.Sprintf("--master-port=%s", port),
			"--max-restarts=0",
		)
	} else {
		arguments = append(arguments, "--standalone")
	}
	return append(arguments,
		fmt.Sprintf("--nproc-per-node=%d", backend.LocalProcs),
		workerPath, request.ArtifactDirectory, request.ArtifactPrefix, "torchtitan",
	)
}

func (backend TorchTitan) environment() []string {
	environment := append(os.Environ(), "PYTHONUNBUFFERED=1")
	if backend.Interface != "" {
		environment = append(environment, "NCCL_SOCKET_IFNAME="+backend.Interface)
	}
	if backend.HCA != "" {
		environment = append(environment, "NCCL_IB_HCA="+backend.HCA, "NCCL_IB_DISABLE=0")
	}
	return environment
}

type torchTitanDevice struct {
	Manufacturer string `json:"manufacturer"`
	Model        string `json:"model"`
	MemoryBytes  uint64 `json:"memory_bytes"`
}

type torchTitanProbe struct {
	PythonVersion     string             `json:"python_version"`
	TorchVersion      string             `json:"torch_version"`
	TorchTitanVersion string             `json:"torchtitan_version"`
	Devices           []torchTitanDevice `json:"devices"`
}

type TorchTitanHost struct {
	Python            string        `json:"python"`
	PythonVersion     string        `json:"python_version"`
	TorchVersion      string        `json:"torch_version"`
	TorchTitanVersion string        `json:"torchtitan_version"`
	Accelerators      []Accelerator `json:"accelerators"`
}

type TorchTitanResolver struct {
	Candidates []string
	Probe      func(context.Context, string) (torchTitanProbe, error)
	OS         string
	Arch       string
	Cluster    Cluster
}

func NewTorchTitanResolverForCluster(cluster Cluster) Resolver {
	return TorchTitanResolver{Cluster: cluster}
}

func backendForCluster(python string, facts torchTitanProbe, cluster Cluster, secondary bool) TorchTitan {
	nodes := cluster.Nodes
	if nodes < 1 {
		nodes = 1
	}
	return TorchTitan{
		Python: python, Version: facts.TorchTitanVersion, LocalProcs: len(facts.Devices),
		Nodes: nodes, NodeRank: cluster.NodeRank, Rendezvous: cluster.Rendezvous,
		Interface: cluster.Interface, HCA: cluster.HCA, Secondary: secondary,
	}
}

func firstUsableTorchTitan(ctx context.Context, candidates []string, probe func(context.Context, string) (torchTitanProbe, error)) (string, torchTitanProbe, []string) {
	var failures []string
	for _, candidate := range candidates {
		facts, err := probe(ctx, candidate)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", candidate, err))
			continue
		}
		return candidate, facts, nil
	}
	return "", torchTitanProbe{}, failures
}

func (resolver TorchTitanResolver) Resolve(ctx context.Context, request ResolveRequest) (Selection, error) {
	hostOS, hostArch := resolver.OS, resolver.Arch
	if hostOS == "" {
		hostOS = runtime.GOOS
	}
	if hostArch == "" {
		hostArch = runtime.GOARCH
	}
	if hostOS != "linux" {
		return Selection{}, fmt.Errorf("TorchTitan training requires Linux; this host is %s/%s", hostOS, hostArch)
	}
	if err := validateTorchArchitecture(request.Architecture, "TorchTitan"); err != nil {
		return Selection{}, err
	}
	candidates := resolver.Candidates
	if len(candidates) == 0 {
		candidates = pythonCandidates()
	}
	probe := resolver.Probe
	if probe == nil {
		probe = probeTorchTitan
	}
	python, facts, failures := firstUsableTorchTitan(ctx, candidates, probe)
	if python == "" {
		detail := strings.Join(failures, "; ")
		if detail != "" {
			detail = ": " + detail
		}
		return Selection{}, fmt.Errorf("no usable TorchTitan runtime found; install a matching TorchTitan and PyTorch build from https://github.com/pytorch/torchtitan#installation%s", detail)
	}
	backend := backendForCluster(python, facts, resolver.Cluster, false)
	nodes, localProcs := backend.Nodes, backend.LocalProcs
	descriptor := backend.Descriptor()
	execution := Execution{
		Backend: descriptor.Identity, Framework: descriptor.Framework,
		Runtime: fmt.Sprintf("%s; Python %s; TorchTitan %s; PyTorch %s", python, facts.PythonVersion, facts.TorchTitanVersion, facts.TorchVersion),
		Host:    Host{OS: hostOS, Architecture: hostArch}, Nodes: nodes, WorldSize: nodes * localProcs,
	}
	for node := 0; node < nodes; node++ {
		for _, device := range facts.Devices {
			execution.Accelerators = append(execution.Accelerators, Accelerator{Manufacturer: device.Manufacturer, Model: device.Model, MemoryBytes: device.MemoryBytes})
		}
	}
	return Selection{Backend: backend, Execution: execution}, nil
}

func validateTorchArchitecture(raw json.RawMessage, label string) error {
	var architecture struct {
		Family         string `json:"family"`
		VocabularySize uint64 `json:"vocabulary_size"`
		Tokenizer      struct {
			Name     string `json:"name"`
			Revision string `json:"revision"`
		} `json:"tokenizer"`
	}
	if err := json.Unmarshal(raw, &architecture); err != nil {
		return fmt.Errorf("decode architecture for %s: %w", label, err)
	}
	if architecture.Family != "decoder-transformer" {
		return fmt.Errorf("%s backend does not support architecture family %q", label, architecture.Family)
	}
	if _, _, err := ResolveTokenizer(architecture.Tokenizer.Name, architecture.Tokenizer.Revision, architecture.VocabularySize); err != nil {
		return fmt.Errorf("%s backend: %w", label, err)
	}
	return nil
}

const torchTitanProbeProgram = `
import importlib.metadata
import json
import platform
import torch
import torchtitan
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torchtitan.distributed import ParallelDims

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError("TorchTitan requires at least one visible CUDA or ROCm GPU")
manufacturer = "AMD" if torch.version.hip else "NVIDIA"
devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    value = torch.tensor([1.0], device=f"cuda:{index}")
    torch.sum(value).item()
    devices.append({"manufacturer": manufacturer, "model": properties.name, "memory_bytes": properties.total_memory})
print(json.dumps({
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torchtitan_version": importlib.metadata.version("torchtitan"),
    "devices": devices,
}))
`

func probeTorchTitan(ctx context.Context, python string) (torchTitanProbe, error) {
	probeCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	command := exec.CommandContext(probeCtx, python, "-c", torchTitanProbeProgram)
	var stderr cappedBuffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		if probeCtx.Err() != nil {
			return torchTitanProbe{}, probeCtx.Err()
		}
		return torchTitanProbe{}, fmt.Errorf("probe failed%s", workerStderr(stderr.String()))
	}
	var facts torchTitanProbe
	if err := json.Unmarshal(bytes.TrimSpace(output), &facts); err != nil {
		return torchTitanProbe{}, fmt.Errorf("invalid probe output: %w", err)
	}
	if facts.PythonVersion == "" || facts.TorchVersion == "" || facts.TorchTitanVersion == "" || len(facts.Devices) == 0 {
		return torchTitanProbe{}, fmt.Errorf("incomplete probe output")
	}
	for _, device := range facts.Devices {
		if device.Manufacturer == "" || device.Model == "" || device.MemoryBytes == 0 {
			return torchTitanProbe{}, fmt.Errorf("incomplete accelerator probe output")
		}
	}
	return facts, nil
}

func RunSecondaryTorchTitan(ctx context.Context, cluster Cluster, request Request) error {
	backend, err := resolveSecondaryTorchTitan(ctx, cluster)
	if err != nil {
		return err
	}
	if err := validateTorchArchitecture(request.Architecture, "TorchTitan"); err != nil {
		return err
	}
	_, err = backend.Run(ctx, request)
	return err
}

// CheckSecondaryTorchTitan verifies the secondary host runtime before WALDO
// waits for a primary plan or materializes any corpus objects.
func CheckSecondaryTorchTitan(ctx context.Context, cluster Cluster) error {
	_, err := resolveSecondaryTorchTitan(ctx, cluster)
	return err
}

// InspectTorchTitanHost probes the runtime and every visible accelerator on a
// prospective TorchTitan node without resolving a model architecture.
func InspectTorchTitanHost(ctx context.Context) (TorchTitanHost, error) {
	if runtime.GOOS != "linux" {
		return TorchTitanHost{}, fmt.Errorf("TorchTitan training requires Linux; this host is %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	python, facts, failures := firstUsableTorchTitan(ctx, pythonCandidates(), probeTorchTitan)
	if python == "" {
		detail := strings.Join(failures, "; ")
		if detail != "" {
			detail = ": " + detail
		}
		return TorchTitanHost{}, fmt.Errorf("no usable TorchTitan runtime found%s", detail)
	}
	host := TorchTitanHost{
		Python: python, PythonVersion: facts.PythonVersion, TorchVersion: facts.TorchVersion,
		TorchTitanVersion: facts.TorchTitanVersion,
	}
	for _, device := range facts.Devices {
		host.Accelerators = append(host.Accelerators, Accelerator{
			Manufacturer: device.Manufacturer, Model: device.Model, MemoryBytes: device.MemoryBytes,
		})
	}
	return host, nil
}

func resolveSecondaryTorchTitan(ctx context.Context, cluster Cluster) (TorchTitan, error) {
	if runtime.GOOS != "linux" {
		return TorchTitan{}, fmt.Errorf("TorchTitan training requires Linux; this host is %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	if cluster.Nodes < 2 {
		return TorchTitan{}, fmt.Errorf("secondary TorchTitan requires at least two nodes")
	}
	if cluster.NodeRank < 1 || cluster.NodeRank >= cluster.Nodes {
		return TorchTitan{}, fmt.Errorf("secondary TorchTitan node rank %d is out of range for %d nodes", cluster.NodeRank, cluster.Nodes)
	}
	if cluster.Rendezvous == "" || cluster.RendezvousID == "" {
		return TorchTitan{}, fmt.Errorf("secondary TorchTitan requires a rendezvous endpoint and id")
	}
	host, err := InspectTorchTitanHost(ctx)
	if err != nil {
		return TorchTitan{}, fmt.Errorf("secondary node: %w", err)
	}
	facts := torchTitanProbe{
		PythonVersion: host.PythonVersion, TorchVersion: host.TorchVersion,
		TorchTitanVersion: host.TorchTitanVersion,
	}
	for _, device := range host.Accelerators {
		facts.Devices = append(facts.Devices, torchTitanDevice{
			Manufacturer: device.Manufacturer, Model: device.Model, MemoryBytes: device.MemoryBytes,
		})
	}
	return backendForCluster(host.Python, facts, cluster, true), nil
}
