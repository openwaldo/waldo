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

func TestPyTorchResolverSelectsFirstUsableRuntime(t *testing.T) {
	architecture := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`)
	resolver := PyTorchResolver{
		OS: "linux", Arch: "amd64", Candidates: []string{"missing", "/home/me/.venv/bin/torch-python"},
		Probe: func(_ context.Context, candidate string) (pyTorchProbe, error) {
			if candidate == "missing" {
				return pyTorchProbe{}, errors.New("not installed")
			}
			return pyTorchProbe{
				PythonVersion: "3.13.0", TorchVersion: "2.8.0", Device: "cuda",
				Manufacturer: "NVIDIA", Accelerator: "NVIDIA H100", MemoryBytes: 80 << 30,
			}, nil
		},
	}
	selection, err := resolver.Resolve(context.Background(), ResolveRequest{Architecture: architecture, Objectives: []string{"causal-language-modeling"}})
	if err != nil {
		t.Fatal(err)
	}
	backend, ok := selection.Backend.(PyTorch)
	if !ok || backend.Python != "/home/me/.venv/bin/torch-python" || backend.Device != "cuda" || selection.Execution.Framework != BackendPyTorch || selection.Execution.Accelerators[0].Manufacturer != "NVIDIA" || !strings.HasPrefix(selection.Execution.Runtime, "torch-python; ") || !strings.Contains(selection.Execution.Runtime, "PyTorch 2.8.0") {
		t.Fatalf("selection = %+v", selection)
	}
}

func TestPyTorchResolverRecordsCPUExecutionWithoutClaimingGPUMemory(t *testing.T) {
	architecture := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`)
	resolver := PyTorchResolver{
		OS: "linux", Arch: "arm64", Candidates: []string{"python3"},
		Probe: func(context.Context, string) (pyTorchProbe, error) {
			return pyTorchProbe{PythonVersion: "3.12", TorchVersion: "2.7", Device: "cpu", Manufacturer: "CPU", Accelerator: "aarch64"}, nil
		},
	}
	selection, err := resolver.Resolve(context.Background(), ResolveRequest{Architecture: architecture})
	if err != nil {
		t.Fatal(err)
	}
	if selection.Execution.WorldSize != 1 || len(selection.Execution.Accelerators) != 0 || !strings.Contains(selection.Execution.Runtime, "device cpu") {
		t.Fatalf("execution = %+v", selection.Execution)
	}
}

func TestPyTorchResolverFailsClosed(t *testing.T) {
	valid := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`)
	if _, err := (PyTorchResolver{OS: "darwin", Arch: "arm64"}).Resolve(context.Background(), ResolveRequest{Architecture: valid}); err == nil || !strings.Contains(err.Error(), "requires Linux") {
		t.Fatalf("platform error = %v", err)
	}
	unsupported := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":100277,"tokenizer":{"name":"tiktoken/cl100k_base","revision":"tiktoken-cl100k-base"}}`)
	if _, err := (PyTorchResolver{OS: "linux", Arch: "amd64"}).Resolve(context.Background(), ResolveRequest{Architecture: unsupported}); err == nil || !strings.Contains(err.Error(), "unsupported tokenizer") {
		t.Fatalf("tokenizer error = %v", err)
	}
}

func TestPyTorchBackendStreamsSharedProtocol(t *testing.T) {
	worker := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
if [ "$5" != "cpu" ]; then
  printf '%s\n' '{"kind":"error","schema":1,"error":"missing device argument"}'
  exit 1
fi
while IFS= read -r line; do
  case "$line" in *'"kind":"record"'*) records=$((records + 1));; esac
done
printf '%s\n' '{"kind":"event","schema":1,"event":{"kind":"progress","message":"pytorch worker event","step":1,"tokens":2}}'
printf '%s\n' '{"kind":"complete","schema":1,"observation":{"simulated":false,"steps":1,"consumed_tokens":2,"final_loss":1.0,"artifacts":[]}}'
`
	if err := os.WriteFile(worker, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	reported := false
	observation, err := (PyTorch{Python: worker, Device: "cpu"}).Run(context.Background(), Request{
		RunID: "run", Stage: "pretrain", Objective: "causal-language-modeling",
		ArchitectureSHA256: strings.Repeat("a", 64), Architecture: json.RawMessage(`{"family":"decoder-transformer"}`),
		Parameters: parameters, ArtifactDirectory: t.TempDir(), ArtifactPrefix: "artifacts",
		Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
			return consume(Record{ID: "one", Text: "hello"})
		}),
		Report: func(event Event) { reported = event.Message == "pytorch worker event" },
	})
	if err != nil {
		t.Fatal(err)
	}
	if observation.Simulated || observation.Steps != 1 || !reported {
		t.Fatalf("observation = %+v, reported = %v", observation, reported)
	}
}
