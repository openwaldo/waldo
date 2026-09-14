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

const (
	TorchTitanRevision           = "builtin-torchtitan-worker-schema-1-r18"
	recommendedTorchVersion      = "2.15.0.dev20260905+cu130"
	recommendedTorchTitanVersion = "0.3.0"
	recommendedTorchIndex        = "https://download.pytorch.org/whl/nightly/cu130"
)

var selectRendezvousInterface = rendezvousInterface

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
	nodes := backend.Nodes
	if nodes < 1 {
		nodes = 1
	}
	worldSize := nodes * backend.LocalProcs
	if err := ValidateBatchTopology(request.Parameters, worldSize); err != nil {
		return Observation{}, fmt.Errorf("TorchTitan: %w", err)
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
	environment, err := backend.environment()
	if err != nil {
		return Observation{}, err
	}
	if backend.Nodes > 1 && request.Report != nil {
		transport := fmt.Sprintf("NCCL socket transport via %s", environmentSetting(environment, "NCCL_SOCKET_IFNAME"))
		if backend.HCA == "" {
			transport += " (RDMA disabled; configure model.nccl.hca to enable it)"
		} else {
			transport += fmt.Sprintf(" with RDMA HCA %s", backend.HCA)
		}
		request.Report(Event{Kind: "log", Message: transport})
	}
	command := exec.CommandContext(ctx, backend.Python, backend.launchArguments(workerPath, request)...)
	command.Env = environment
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	configureGracefulCancellation(command)
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

func (backend TorchTitan) environment() ([]string, error) {
	environment := append(os.Environ(), "PYTHONUNBUFFERED=1")
	interfaceName := backend.Interface
	if interfaceName == "" && backend.Nodes > 1 {
		var err error
		interfaceName, err = selectRendezvousInterface(backend.Rendezvous)
		if err != nil {
			return nil, fmt.Errorf("select NCCL interface for rendezvous %s: %w", backend.Rendezvous, err)
		}
	}
	if interfaceName != "" {
		environment = append(environment, "NCCL_SOCKET_IFNAME="+interfaceName)
	}
	if backend.HCA != "" {
		environment = append(environment, "NCCL_IB_HCA="+backend.HCA, "NCCL_IB_DISABLE=0")
	} else {
		// RDMA must be explicitly selected. NCCL can otherwise prefer an HCA
		// that is present but not routable between the training hosts.
		environment = append(environment, "NCCL_IB_DISABLE=1")
	}
	return environment, nil
}

func rendezvousInterface(address string) (string, error) {
	connection, err := net.Dial("udp", address)
	if err != nil {
		return "", err
	}
	local, ok := connection.LocalAddr().(*net.UDPAddr)
	_ = connection.Close()
	if !ok {
		return "", fmt.Errorf("local route has address type %T", connection.LocalAddr())
	}
	interfaces, err := net.Interfaces()
	if err != nil {
		return "", err
	}
	for _, candidate := range interfaces {
		addresses, err := candidate.Addrs()
		if err != nil {
			continue
		}
		for _, candidateAddress := range addresses {
			ip, _, err := net.ParseCIDR(candidateAddress.String())
			if err == nil && ip.Equal(local.IP) {
				return candidate.Name, nil
			}
		}
	}
	return "", fmt.Errorf("no interface owns routed local address %s", local.IP)
}

func environmentSetting(environment []string, name string) string {
	prefix := name + "="
	value := ""
	for _, item := range environment {
		if strings.HasPrefix(item, prefix) {
			value = strings.TrimPrefix(item, prefix)
		}
	}
	return value
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
	MemlockSoftBytes  int64              `json:"memlock_soft_bytes"`
	NetworkInterfaces []HostInterface    `json:"network_interfaces"`
	RDMADevices       []RDMADevice       `json:"rdma_devices"`
	LocalInterconnect string             `json:"local_interconnect"`
}

type HostInterface struct {
	Name       string `json:"name"`
	State      string `json:"state"`
	HasAddress bool   `json:"has_address"`
}

type RDMADevice struct {
	Name   string `json:"name"`
	Active bool   `json:"active"`
}

type TorchTitanHost struct {
	Python            string          `json:"python"`
	PythonVersion     string          `json:"python_version"`
	TorchVersion      string          `json:"torch_version"`
	TorchTitanVersion string          `json:"torchtitan_version"`
	Accelerators      []Accelerator   `json:"accelerators"`
	MemlockSoftBytes  int64           `json:"memlock_soft_bytes"`
	NetworkInterfaces []HostInterface `json:"network_interfaces"`
	RDMADevices       []RDMADevice    `json:"rdma_devices"`
	LocalInterconnect string          `json:"local_interconnect"`
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
		return Selection{}, fmt.Errorf("no usable TorchTitan runtime found%s\n%s", detail, torchTitanInstallGuidance())
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
	parallelism, err := resolveTorchTitanParallelism(request, facts, resolver.Cluster, nodes, localProcs)
	if err != nil {
		return Selection{}, err
	}
	execution.Parallelism = parallelism
	return Selection{Backend: backend, Execution: execution}, nil
}

func resolveTorchTitanParallelism(request ResolveRequest, facts torchTitanProbe, cluster Cluster, nodes, GPUsPerNode int) (Parallelism, error) {
	requested := strings.TrimSpace(request.Parallelism)
	if requested == "" {
		requested = ParallelismAuto
	}
	if err := ValidateParallelismRequest(requested); err != nil {
		return Parallelism{}, err
	}
	worldSize := nodes * GPUsPerNode
	if worldSize < 1 || len(facts.Devices) == 0 {
		return Parallelism{}, fmt.Errorf("TorchTitan parallelism requires at least one visible GPU")
	}
	memoryPerGPU := facts.Devices[0].MemoryBytes
	for _, device := range facts.Devices[1:] {
		if device.MemoryBytes < memoryPerGPU {
			memoryPerGPU = device.MemoryBytes
		}
	}
	if request.ApproximateParameters > ^uint64(0)/16 {
		return Parallelism{}, fmt.Errorf("estimated model state size overflows uint64")
	}
	stateBytes := request.ApproximateParameters * 16
	// Keep 40% of each GPU free for activations, logits, allocator overhead,
	// and framework workspaces.
	stateBudget := memoryPerGPU / 10 * 6
	strategy := requested
	if requested == ParallelismAuto && request.ApproximateParameters == 0 {
		strategy = ParallelismFullySharded
	} else if requested == ParallelismAuto {
		switch {
		case stateBytes <= stateBudget:
			strategy = ParallelismData
		case nodes > 1 && GPUsPerNode > 1 && divideRoundUpBytes(stateBytes, uint64(GPUsPerNode)) <= stateBudget:
			strategy = ParallelismHybridSharded
		default:
			strategy = ParallelismFullySharded
		}
	}
	if strategy == ParallelismHybridSharded && (nodes < 2 || GPUsPerNode < 2) {
		return Parallelism{}, fmt.Errorf("hybrid-sharded-data-parallel requires at least two hosts with at least two GPUs each")
	}
	if strategy == ParallelismData && stateBytes > stateBudget {
		return Parallelism{}, fmt.Errorf("data-parallel model state requires approximately %d bytes per GPU, exceeding WALDO's %d-byte safe state budget; use auto or a sharded strategy", stateBytes, stateBudget)
	}
	if strategy == ParallelismHybridSharded && divideRoundUpBytes(stateBytes, uint64(GPUsPerNode)) > stateBudget {
		return Parallelism{}, fmt.Errorf("hybrid-sharded-data-parallel model state does not fit safely when divided across %d GPUs per host; use auto or fully-sharded-data-parallel", GPUsPerNode)
	}
	if strategy == ParallelismFullySharded && divideRoundUpBytes(stateBytes, uint64(worldSize)) > stateBudget {
		return Parallelism{}, fmt.Errorf("model state does not fit safely even when divided across all %d GPUs", worldSize)
	}
	sharing, copies := 1, worldSize
	if strategy == ParallelismHybridSharded {
		sharing, copies = GPUsPerNode, nodes
	} else if strategy == ParallelismFullySharded {
		sharing, copies = worldSize, 1
	}
	interNode := ""
	if nodes > 1 {
		if cluster.HCA != "" {
			interNode = "rdma"
		} else {
			interNode = "tcp"
		}
	}
	return Parallelism{
		Requested: requested, Strategy: strategy, WorldSize: worldSize, Nodes: nodes, GPUsPerNode: GPUsPerNode,
		CompleteModelCopies: copies, GPUsSharingEachModelCopy: sharing,
		LocalInterconnect: facts.LocalInterconnect, InterNodeInterconnect: interNode,
		EstimatedModelStateBytes: stateBytes, MemoryPerGPUBytes: memoryPerGPU,
	}, nil
}

func divideRoundUpBytes(value, divisor uint64) uint64 {
	result := value / divisor
	if value%divisor != 0 {
		result++
	}
	return result
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
import fcntl
import json
import platform
import resource
import subprocess
import socket
import struct
from pathlib import Path
import torch
import torchtitan
import torch.testing._internal.distributed.fake_pg
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

def read_text(path, default="unknown"):
    try:
        return path.read_text().strip()
    except OSError:
        return default

network_interfaces = []
network_root = Path("/sys/class/net")
ipv6_interfaces = set()
try:
    ipv6_interfaces = {line.split()[5] for line in Path("/proc/net/if_inet6").read_text().splitlines()}
except OSError:
    pass

def has_ipv4_address(name):
    try:
        request = struct.pack("256s", name.encode()[:15])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as descriptor:
            fcntl.ioctl(descriptor.fileno(), 0x8915, request)  # SIOCGIFADDR
        return True
    except OSError:
        return False

if network_root.is_dir():
    for path in sorted(network_root.iterdir(), key=lambda value: value.name):
        network_interfaces.append({
            "name": path.name,
            "state": read_text(path / "operstate"),
            "has_address": has_ipv4_address(path.name) or path.name in ipv6_interfaces,
        })

rdma_devices = []
rdma_root = Path("/sys/class/infiniband")
if rdma_root.is_dir():
    for path in sorted(rdma_root.iterdir(), key=lambda value: value.name):
        active = any(read_text(port / "state").startswith("4:") for port in (path / "ports").glob("*"))
        rdma_devices.append({"name": path.name, "active": active})

def local_interconnect():
    if len(devices) < 2:
        return "single-gpu"
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"], check=True, capture_output=True,
            text=True, timeout=5,
        )
        rows = [line.split() for line in result.stdout.splitlines() if line.startswith("GPU")]
        links = []
        for row_index, row in enumerate(rows):
            for column_index in range(len(rows)):
                if row_index != column_index and column_index + 1 < len(row):
                    links.append(row[column_index + 1])
        if links and all(link.startswith("NV") for link in links):
            return "nvlink"
    except (OSError, subprocess.SubprocessError):
        pass
    if all(torch.cuda.can_device_access_peer(left, right)
           for left in range(len(devices)) for right in range(len(devices))
           if left != right):
        return "gpu-peer-to-peer"
    return "pcie"

memlock_soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
print(json.dumps({
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torchtitan_version": importlib.metadata.version("torchtitan"),
    "devices": devices,
    "memlock_soft_bytes": memlock_soft,
    "network_interfaces": network_interfaces,
    "rdma_devices": rdma_devices,
    "local_interconnect": local_interconnect(),
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
		return TorchTitanHost{}, fmt.Errorf("no usable TorchTitan runtime found%s\n%s", detail, torchTitanInstallGuidance())
	}
	host := TorchTitanHost{
		Python: python, PythonVersion: facts.PythonVersion, TorchVersion: facts.TorchVersion,
		TorchTitanVersion: facts.TorchTitanVersion, MemlockSoftBytes: facts.MemlockSoftBytes,
		NetworkInterfaces: facts.NetworkInterfaces, RDMADevices: facts.RDMADevices,
		LocalInterconnect: facts.LocalInterconnect,
	}
	for _, device := range facts.Devices {
		host.Accelerators = append(host.Accelerators, Accelerator{
			Manufacturer: device.Manufacturer, Model: device.Model, MemoryBytes: device.MemoryBytes,
		})
	}
	return host, nil
}

// ValidateTorchTitanHostConfiguration checks host-local settings that would
// otherwise fail only when the distributed process group starts.
func ValidateTorchTitanHostConfiguration(host TorchTitanHost, cluster Cluster) error {
	if cluster.Interface != "" {
		found := false
		for _, candidate := range host.NetworkInterfaces {
			if candidate.Name != cluster.Interface {
				continue
			}
			found = true
			if candidate.State != "up" && candidate.State != "unknown" {
				return fmt.Errorf("configured NCCL interface %q is not up (state %s)", cluster.Interface, candidate.State)
			}
			if !candidate.HasAddress {
				return fmt.Errorf("configured NCCL interface %q has no IP address", cluster.Interface)
			}
			break
		}
		if !found {
			return fmt.Errorf("configured NCCL interface %q does not exist", cluster.Interface)
		}
	}
	if cluster.HCA == "" {
		return nil
	}
	found := false
	for _, device := range host.RDMADevices {
		if device.Name != cluster.HCA {
			continue
		}
		found = true
		if !device.Active {
			return fmt.Errorf("configured RDMA HCA %q has no active port", cluster.HCA)
		}
		break
	}
	if !found {
		return fmt.Errorf("configured RDMA HCA %q does not exist", cluster.HCA)
	}
	if host.MemlockSoftBytes != -1 {
		return fmt.Errorf("configured RDMA HCA %q requires an unlimited memlock soft limit; current limit is %d bytes; configure /etc/security/limits.d/90-waldo-rdma.conf for both '*' and 'root', then log out and reconnect", cluster.HCA, host.MemlockSoftBytes)
	}
	return nil
}

func torchTitanInstallGuidance() string {
	distribution := "Linux"
	if data, err := os.ReadFile("/etc/os-release"); err == nil {
		distribution = linuxDistribution(data)
	}
	return torchTitanInstallGuidanceForDistribution(distribution)
}

func torchTitanInstallGuidanceForDistribution(distribution string) string {
	prerequisite := `# Install Python 3.11 or newer and pip using this host's package manager.
# Then ensure python3 resolves to that interpreter.`
	lower := strings.ToLower(distribution)
	if strings.Contains(lower, "rocky") || strings.Contains(lower, "rhel") || strings.Contains(lower, "red hat") || strings.Contains(lower, "alma") || strings.Contains(lower, "centos") || strings.Contains(lower, "fedora") {
		prerequisite = `sudo dnf install -y python3.11 python3.11-pip
mkdir -p "$HOME/.local/bin"
ln -sfn /usr/bin/python3.11 "$HOME/.local/bin/python3"
export PATH="$HOME/.local/bin:$PATH"
hash -r`
	}
	return fmt.Sprintf(`detected distribution: %s
TorchTitan runtime is unavailable. Copy and run this installation block as your normal user:

%s
%s

Python must be 3.11 or newer. If python3 still reports an older interpreter,
run `+"`hash -r`"+` or start a new login shell, then repeat the block.`, distribution, prerequisite, TorchTitanPythonInstallScript())
}

// TorchTitanPythonInstallScript returns the tested user-local Python runtime
// installation and verification commands used by hostfile preflight errors.
func TorchTitanPythonInstallScript() string {
	return fmt.Sprintf(`python3 --version
python3 -m pip install --user --upgrade pip setuptools wheel
python3 -m pip install --user 'torch==%s' --index-url %s
python3 -m pip install --user 'torchtitan==%s'
python3 -c 'import torch, torchtitan; print(torch.__version__, torchtitan.__version__, torch.cuda.is_available(), torch.cuda.device_count(), torch.distributed.is_nccl_available())'`, recommendedTorchVersion, recommendedTorchIndex, recommendedTorchTitanVersion)
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
		TorchTitanVersion: host.TorchTitanVersion, LocalInterconnect: host.LocalInterconnect,
	}
	for _, device := range host.Accelerators {
		facts.Devices = append(facts.Devices, torchTitanDevice{
			Manufacturer: device.Manufacturer, Model: device.Model, MemoryBytes: device.MemoryBytes,
		})
	}
	return backendForCluster(host.Python, facts, cluster, true), nil
}
