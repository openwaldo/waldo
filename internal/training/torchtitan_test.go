// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTorchTitanResolverRecordsEveryVisibleAccelerator(t *testing.T) {
	architecture := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`)
	resolver := TorchTitanResolver{
		OS: "linux", Arch: "amd64", Candidates: []string{"missing", "titan-python"},
		Probe: func(_ context.Context, candidate string) (torchTitanProbe, error) {
			if candidate == "missing" {
				return torchTitanProbe{}, errors.New("not installed")
			}
			return torchTitanProbe{
				PythonVersion: "3.13", TorchVersion: "2.12", TorchTitanVersion: "0.2.2",
				Devices: []torchTitanDevice{
					{Manufacturer: "NVIDIA", Model: "NVIDIA H100", MemoryBytes: 80 << 30},
					{Manufacturer: "NVIDIA", Model: "NVIDIA H100", MemoryBytes: 80 << 30},
				},
			}, nil
		},
	}
	selection, err := resolver.Resolve(context.Background(), ResolveRequest{Architecture: architecture})
	if err != nil {
		t.Fatal(err)
	}
	backend, ok := selection.Backend.(TorchTitan)
	if !ok || backend.LocalProcs != 2 || selection.Execution.WorldSize != 2 || selection.Execution.Nodes != 1 || len(selection.Execution.Accelerators) != 2 || !selection.Backend.Descriptor().Capabilities.Distributed || !strings.Contains(selection.Execution.Runtime, "TorchTitan 0.2.2") {
		t.Fatalf("selection = %+v", selection)
	}
}

func TestTorchTitanSelectsTopologyAwareParallelism(t *testing.T) {
	facts := torchTitanProbe{
		Devices: []torchTitanDevice{
			{Manufacturer: "NVIDIA", Model: "H200", MemoryBytes: 140 << 30},
			{Manufacturer: "NVIDIA", Model: "H200", MemoryBytes: 140 << 30},
		},
		LocalInterconnect: "nvlink",
	}
	cluster := Cluster{Nodes: 2, HCA: "mlx5_0"}
	tests := []struct {
		name       string
		parameters uint64
		requested  string
		want       string
		copies     int
		sharing    int
	}{
		{"small model is complete on every GPU", 337_000_000, "", ParallelismData, 4, 1},
		{"larger model is divided within each host", 6_000_000_000, "", ParallelismHybridSharded, 2, 2},
		{"largest model is divided across every GPU", 12_000_000_000, "", ParallelismFullySharded, 1, 4},
		{"explicit full sharding", 337_000_000, ParallelismFullySharded, ParallelismFullySharded, 1, 4},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan, err := resolveTorchTitanParallelism(ResolveRequest{
				ApproximateParameters: test.parameters, Parallelism: test.requested,
			}, facts, cluster, 2, 2)
			if err != nil {
				t.Fatal(err)
			}
			if plan.Strategy != test.want || plan.CompleteModelCopies != test.copies || plan.GPUsSharingEachModelCopy != test.sharing || plan.LocalInterconnect != "nvlink" || plan.InterNodeInterconnect != "rdma" {
				t.Fatalf("parallelism = %+v", plan)
			}
		})
	}
}

func TestDescribeParallelismUsesPlainLanguage(t *testing.T) {
	plan := Parallelism{
		Requested: ParallelismAuto, Strategy: ParallelismData,
		WorldSize: 4, Nodes: 2, GPUsPerNode: 2,
		CompleteModelCopies: 4, GPUsSharingEachModelCopy: 1,
		LocalInterconnect: "nvlink", InterNodeInterconnect: "rdma",
	}
	text := strings.Join(DescribeParallelism(plan, 32), "\n")
	for _, expected := range []string{
		"automatically selected data parallelism",
		"each global batch is sharded across 4 GPUs",
		"each GPU processes 8 unique sequences",
		"no sequence is duplicated between GPUs",
		"each of the 4 GPUs holds a synchronized complete copy",
		"the run still produces one model", "NVLink", "RDMA", "2 hosts",
	} {
		if !strings.Contains(text, expected) {
			t.Fatalf("description %q omits %q", text, expected)
		}
	}
	if strings.Contains(strings.ToLower(text), "replica") {
		t.Fatalf("description uses unexplained replica terminology: %s", text)
	}
}

func TestDescribeShardedModelSeparatesDataAndModelPlacement(t *testing.T) {
	plan := Parallelism{
		Requested: ParallelismAuto, Strategy: ParallelismFullySharded,
		WorldSize: 4, Nodes: 2, GPUsPerNode: 2,
		CompleteModelCopies: 1, GPUsSharingEachModelCopy: 4,
	}
	text := strings.Join(DescribeParallelism(plan, 8), "\n")
	for _, expected := range []string{
		"each global batch is sharded across 4 GPUs",
		"each GPU processes 2 unique sequences",
		"no sequence is duplicated between GPUs",
		"one model is divided across all 4 GPUs",
		"no GPU or host holds a complete copy",
	} {
		if !strings.Contains(text, expected) {
			t.Fatalf("description %q omits %q", text, expected)
		}
	}
}

func TestDescribeExplicitParallelismDoesNotInventAutomaticRationale(t *testing.T) {
	plan := Parallelism{
		Requested: ParallelismFullySharded, Strategy: ParallelismFullySharded,
		WorldSize: 4, Nodes: 2, GPUsPerNode: 2,
		CompleteModelCopies: 1, GPUsSharingEachModelCopy: 4,
	}
	text := strings.Join(DescribeParallelism(plan, 8), "\n")
	if !strings.Contains(text, "selected full model sharding as requested by the compose") || strings.Contains(text, "because") {
		t.Fatalf("explicit description = %q", text)
	}
}

func TestTorchTitanResolverFailsClosed(t *testing.T) {
	valid := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`)
	if _, err := (TorchTitanResolver{OS: "darwin", Arch: "arm64"}).Resolve(context.Background(), ResolveRequest{Architecture: valid}); err == nil || !strings.Contains(err.Error(), "requires Linux") {
		t.Fatalf("platform error = %v", err)
	}
	unsupported := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":100277,"tokenizer":{"name":"other","revision":"one"}}`)
	if _, err := (TorchTitanResolver{OS: "linux", Arch: "amd64"}).Resolve(context.Background(), ResolveRequest{Architecture: unsupported}); err == nil || !strings.Contains(err.Error(), "unsupported tokenizer") {
		t.Fatalf("tokenizer error = %v", err)
	}
}

func TestTorchTitanInstallGuidanceExplainsPythonResolution(t *testing.T) {
	guidance := torchTitanInstallGuidanceForDistribution("Rocky Linux 9")
	for _, expected := range []string{"Copy and run this installation block", "gcc", "python3.11-devel", "python3.11-pip", "hash -r", recommendedTorchVersion, recommendedTorchIndex, recommendedTorchTitanVersion, "torch.cuda.is_available()", "torch.distributed.is_nccl_available()"} {
		if !strings.Contains(guidance, expected) {
			t.Fatalf("installation guidance omits %q: %s", expected, guidance)
		}
	}
	if strings.Contains(guidance, "openssh") {
		t.Fatalf("installation guidance must not install SSH packages: %s", guidance)
	}
	if strings.Count(guidance, "python3 --version") != 1 {
		t.Fatalf("installation guidance should contain one Python version check: %s", guidance)
	}
}

func TestValidateTorchTitanHostConfiguration(t *testing.T) {
	ready := TorchTitanHost{
		MemlockSoftBytes:  -1,
		NetworkInterfaces: []HostInterface{{Name: "ens10f1", State: "up", HasAddress: true}},
		RDMADevices:       []RDMADevice{{Name: "mlx5_0", Active: true}},
	}
	cluster := Cluster{Interface: "ens10f1", HCA: "mlx5_0"}
	if err := ValidateTorchTitanHostConfiguration(ready, cluster); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		host TorchTitanHost
		want string
	}{
		{"missing interface", TorchTitanHost{MemlockSoftBytes: -1, RDMADevices: ready.RDMADevices}, "interface \"ens10f1\" does not exist"},
		{"down interface", TorchTitanHost{MemlockSoftBytes: -1, NetworkInterfaces: []HostInterface{{Name: "ens10f1", State: "down", HasAddress: true}}, RDMADevices: ready.RDMADevices}, "interface \"ens10f1\" is not up"},
		{"unaddressed interface", TorchTitanHost{MemlockSoftBytes: -1, NetworkInterfaces: []HostInterface{{Name: "ens10f1", State: "up"}}, RDMADevices: ready.RDMADevices}, "interface \"ens10f1\" has no IP address"},
		{"missing HCA", TorchTitanHost{MemlockSoftBytes: -1, NetworkInterfaces: ready.NetworkInterfaces}, "HCA \"mlx5_0\" does not exist"},
		{"inactive HCA", TorchTitanHost{MemlockSoftBytes: -1, NetworkInterfaces: ready.NetworkInterfaces, RDMADevices: []RDMADevice{{Name: "mlx5_0"}}}, "HCA \"mlx5_0\" has no active port"},
		{"low memlock", TorchTitanHost{MemlockSoftBytes: 8 << 20, NetworkInterfaces: ready.NetworkInterfaces, RDMADevices: ready.RDMADevices}, "unlimited memlock soft limit"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateTorchTitanHostConfiguration(test.host, cluster)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestTorchTitanProbeCollectsNetworkAndRDMASanityFacts(t *testing.T) {
	for _, expected := range []string{"resource.RLIMIT_MEMLOCK", `Path("/sys/class/net")`, `Path("/sys/class/infiniband")`, `"has_address"`, `"memlock_soft_bytes"`, `"network_interfaces"`, `"rdma_devices"`, `"nvidia-smi", "topo", "-m"`, `"local_interconnect"`, "triton_driver.active.get_current_target()"} {
		if !strings.Contains(torchTitanProbeProgram, expected) {
			t.Fatalf("TorchTitan probe omits %q", expected)
		}
	}
}

func TestTorchTitanWorkerAdaptsParallelDimsAPI(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`from torch.testing._internal.distributed import fake_pg`,
		`inspect.signature(ParallelDims).parameters`,
		`if "etp" in`,
		`parallel_arguments["spmd_backend"] = "partial_dtensor"`,
		`ParallelDims(**parallel_arguments)`,
		`DistributedDataParallel`,
		`nodes if hybrid else 1`,
		`gpus_per_node if hybrid else self.world_size`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("TorchTitan worker omits ParallelDims compatibility logic %q", expected)
		}
	}
}

func TestTorchTitanBackendLaunchesTorchrunThroughSharedProtocol(t *testing.T) {
	worker := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
case "$*" in
  *'torch.distributed.run'*'--standalone'*'--nproc-per-node=2'*'torchtitan'*) ;;
  *) printf '%s\n' '{"kind":"error","schema":1,"error":"unexpected torchrun arguments"}'; exit 1;;
esac
seen_tokens=0
while IFS= read -r line; do
  case "$line" in *'"kind":"record"'*'"tokens"'*) seen_tokens=1;; esac
  case "$line" in *'"kind":"record"'*'"text":"hello"'*) exit 5;; esac
done
[ "$seen_tokens" -eq 1 ] || exit 6
printf '%s\n' '{"kind":"event","schema":1,"event":{"kind":"progress","message":"torchtitan worker event","step":1,"tokens":2}}'
printf '%s\n' '{"kind":"complete","schema":1,"observation":{"simulated":false,"steps":1,"consumed_tokens":2,"final_loss":1.0,"artifacts":[]}}'
`
	if err := os.WriteFile(worker, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 2, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	artifactDirectory := t.TempDir()
	reported := false
	observation, err := (TorchTitan{Python: worker, LocalProcs: 2}).Run(context.Background(), Request{
		RunID: "run", Stage: "pretrain", Objective: "causal-language-modeling",
		ArchitectureSHA256: strings.Repeat("a", 64), Architecture: json.RawMessage(`{"family":"decoder-transformer"}`),
		Parameters: parameters, ArtifactDirectory: artifactDirectory, ArtifactPrefix: "artifacts",
		Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
			return consume(Record{ID: "one", Text: "hello"})
		}),
		Report: func(event Event) { reported = event.Message == "torchtitan worker event" },
	})
	if err != nil {
		t.Fatal(err)
	}
	if observation.Simulated || observation.Steps != 1 || !reported {
		t.Fatalf("observation = %+v, reported = %v", observation, reported)
	}
	matches, err := filepath.Glob(filepath.Join(artifactDirectory, ".waldo-torchtitan-worker-*.py"))
	if err != nil || len(matches) != 0 {
		t.Fatalf("staged workers after run = %v, error = %v", matches, err)
	}
}

func TestTorchTitanMultiNodeLaunchUsesRendezvous(t *testing.T) {
	worker := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
case "$*" in
  *'--standalone'*) printf '%s\n' '{"kind":"error","schema":1,"error":"standalone leaked into a multi-node launch"}'; exit 1;;
esac
case "$*" in
  *'--nnodes=4'*'--node-rank=0'*'--master-addr=primary'*'--master-port=29500'*'--max-restarts=0'*'--nproc-per-node=1'*'torchtitan'*) ;;
  *) printf '%s\n' '{"kind":"error","schema":1,"error":"unexpected torchrun arguments"}'; exit 1;;
esac
while IFS= read -r line; do :; done
printf '%s\n' '{"kind":"complete","schema":1,"observation":{"simulated":false,"steps":1,"consumed_tokens":2,"final_loss":1.0,"artifacts":[]}}'
`
	if err := os.WriteFile(worker, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 4, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	backend := TorchTitan{
		Python: worker, LocalProcs: 1, Nodes: 4, NodeRank: 0,
		Rendezvous: "primary:29500", Interface: "eth0",
	}
	if _, err := backend.Run(context.Background(), Request{
		RunID: "run", Stage: "pretrain", Objective: "causal-language-modeling",
		ArchitectureSHA256: strings.Repeat("a", 64), Architecture: json.RawMessage(`{"family":"decoder-transformer"}`),
		Parameters: parameters, ArtifactDirectory: t.TempDir(), ArtifactPrefix: "artifacts",
		Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
			return consume(Record{ID: "one", Text: "hello"})
		}),
	}); err != nil {
		t.Fatal(err)
	}
}

func TestTorchTitanSelectsRendezvousRouteAndDisablesUnconfiguredRDMA(t *testing.T) {
	previous := selectRendezvousInterface
	selectRendezvousInterface = func(address string) (string, error) {
		if address != "primary:29500" {
			t.Fatalf("rendezvous address = %q", address)
		}
		return "route0", nil
	}
	defer func() { selectRendezvousInterface = previous }()
	backend := TorchTitan{Nodes: 2, Rendezvous: "primary:29500"}
	environment, err := backend.environment()
	if err != nil {
		t.Fatal(err)
	}
	interfaceName := environmentSetting(environment, "NCCL_SOCKET_IFNAME")
	if interfaceName == "" {
		t.Fatal("NCCL socket interface was not derived from the rendezvous route")
	}
	if interfaceName != "route0" {
		t.Fatalf("derived NCCL interface = %q, want route0", interfaceName)
	}
	if value := environmentSetting(environment, "NCCL_IB_DISABLE"); value != "1" {
		t.Fatalf("NCCL_IB_DISABLE = %q, want 1", value)
	}
}

func TestTorchTitanRejectsGlobalBatchThatCannotBePartitioned(t *testing.T) {
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 6, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	backend := TorchTitan{Python: "unused", LocalProcs: 2, Nodes: 2, Rendezvous: "primary:29500"}
	_, err = backend.Run(context.Background(), Request{Parameters: parameters})
	if err == nil || !strings.Contains(err.Error(), "global micro-batch 6 (batch_size 6 / gradient_accumulation_steps 1) must be divisible by world size 4") {
		t.Fatalf("batch partition error = %v", err)
	}
}

func TestTorchTitanHonorsExplicitNCCLTransport(t *testing.T) {
	backend := TorchTitan{Nodes: 2, Rendezvous: "unresolved.invalid:29500", Interface: "eth9", HCA: "mlx5_4"}
	environment, err := backend.environment()
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{
		"NCCL_SOCKET_IFNAME": "eth9",
		"NCCL_IB_HCA":        "mlx5_4",
		"NCCL_IB_DISABLE":    "0",
	} {
		if got := environmentSetting(environment, name); got != want {
			t.Fatalf("%s = %q, want %q", name, got, want)
		}
	}
}

func TestTorchTitanRejectsPrimaryWithNonZeroRank(t *testing.T) {
	backend := TorchTitan{Python: "unused", LocalProcs: 1, Nodes: 4, NodeRank: 2, Rendezvous: "primary:29500"}
	_, err := backend.Run(context.Background(), Request{ArtifactDirectory: t.TempDir(), ArtifactPrefix: "artifacts"})
	if err == nil || !strings.Contains(err.Error(), "primary TorchTitan node must be rank 0") {
		t.Fatalf("primary rank guard error = %v", err)
	}
}

func TestTorchTitanResolverAggregatesClusterNodes(t *testing.T) {
	architecture := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`)
	resolver := TorchTitanResolver{
		OS: "linux", Arch: "arm64", Candidates: []string{"titan-python"},
		Cluster: Cluster{Nodes: 4, NodeRank: 0, Rendezvous: "primary:29500", RendezvousID: "run-42", Interface: "roce0", HCA: "mlx5_0"},
		Probe: func(_ context.Context, _ string) (torchTitanProbe, error) {
			return torchTitanProbe{
				PythonVersion: "3.13", TorchVersion: "2.12", TorchTitanVersion: "0.2.2",
				Devices: []torchTitanDevice{{Manufacturer: "NVIDIA", Model: "GB10", MemoryBytes: 128 << 30}},
			}, nil
		},
	}
	selection, err := resolver.Resolve(context.Background(), ResolveRequest{Architecture: architecture})
	if err != nil {
		t.Fatal(err)
	}
	backend, ok := selection.Backend.(TorchTitan)
	if !ok {
		t.Fatalf("backend = %T", selection.Backend)
	}
	if backend.Nodes != 4 || backend.LocalProcs != 1 || backend.Rendezvous != "primary:29500" || backend.Interface != "roce0" || backend.HCA != "mlx5_0" {
		t.Fatalf("backend = %+v", backend)
	}
	if selection.Execution.Nodes != 4 || selection.Execution.WorldSize != 4 || len(selection.Execution.Accelerators) != 4 {
		t.Fatalf("execution = %+v", selection.Execution)
	}
}

func TestTorchTitanSecondaryLauncherStreamReceivesRecordsFromGlobalRankZero(t *testing.T) {
	worker := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
case "$*" in
  *'--standalone'*) exit 3;;
esac
case "$*" in
  *'--nnodes=2'*'--node-rank=1'*'--master-addr=primary'*'--master-port=29500'*'--max-restarts=0'*'--nproc-per-node=1'*'torchtitan'*) ;;
  *) exit 3;;
esac
if IFS= read -r line; then exit 4; fi
exit 0
`
	if err := os.WriteFile(worker, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 2, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	backend := TorchTitan{
		Python: worker, LocalProcs: 1, Nodes: 2, NodeRank: 1,
		Rendezvous: "primary:29500", Interface: "eth0", Secondary: true,
	}
	observation, err := backend.Run(context.Background(), Request{
		ArtifactDirectory: t.TempDir(), ArtifactPrefix: "artifacts",
		Parameters: parameters, Architecture: json.RawMessage(`{"family":"decoder-transformer"}`),
	})
	if err != nil {
		t.Fatalf("secondary run: %v", err)
	}
	if len(observation.Artifacts) != 0 || observation.Steps != 0 {
		t.Fatalf("secondary observation = %+v", observation)
	}
}

func TestTorchTitanSecondaryStreamsVerifiedNodeLocalRecords(t *testing.T) {
	worker := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
[ "$WALDO_TORCH_DATA_PLANE" = "node-local-cache" ] || exit 5
IFS= read -r begin || exit 6
IFS= read -r record || exit 7
IFS= read -r boundary || exit 8
IFS= read -r end || exit 9
case "$begin" in *'"kind":"begin"'*) ;; *) exit 9;; esac
case "$record" in *'"kind":"sequence"'*'"tokens"'*) ;; *) exit 10;; esac
case "$boundary" in *'"kind":"micro_batch_end"'*) ;; *) exit 11;; esac
case "$end" in *'"kind":"end"'*) ;; *) exit 12;; esac
exit 0
`
	if err := os.WriteFile(worker, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 2, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	backend := TorchTitan{
		Python: worker, LocalProcs: 1, Nodes: 2, NodeRank: 1,
		Rendezvous: "primary:29500", Interface: "eth0", Secondary: true,
	}
	observation, err := backend.Run(context.Background(), Request{
		ArtifactDirectory: t.TempDir(), ArtifactPrefix: "artifacts",
		Parameters: parameters, Architecture: json.RawMessage(`{"family":"decoder-transformer"}`),
		Parallelism: Parallelism{WorldSize: 2, GPUsPerNode: 1, DataPlane: DataPlaneNodeLocal},
		Records:     staticRecordSource{{ID: "one", Text: "node-local", Corpus: "test"}},
	})
	if err != nil {
		t.Fatalf("secondary node-local run: %v", err)
	}
	if len(observation.Artifacts) != 0 || observation.Steps != 0 {
		t.Fatalf("secondary observation = %+v", observation)
	}
}
