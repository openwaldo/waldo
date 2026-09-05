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
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
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
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	backend := TorchTitan{
		Python: worker, LocalProcs: 1, Nodes: 4, NodeRank: 0,
		Rendezvous: "primary:29500",
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

func TestTorchTitanSecondaryNodeReceivesRecordsFromGlobalRankZero(t *testing.T) {
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
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	backend := TorchTitan{
		Python: worker, LocalProcs: 1, Nodes: 2, NodeRank: 1,
		Rendezvous: "primary:29500", Secondary: true,
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
