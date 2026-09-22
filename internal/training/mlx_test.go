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

func TestMLXResolverSelectsFirstUsableRuntime(t *testing.T) {
	architecture := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`)
	resolver := MLXResolver{
		OS: "darwin", Arch: "arm64", Candidates: []string{"missing", "mlx-python"},
		Probe: func(_ context.Context, candidate string) (mlxProbe, error) {
			if candidate == "missing" {
				return mlxProbe{}, errors.New("not installed")
			}
			return mlxProbe{PythonVersion: "3.14.0", MLXVersion: "0.31.2", Accelerator: "Apple M4 Max", MemoryBytes: 128 << 30}, nil
		},
	}
	selection, err := resolver.Resolve(context.Background(), ResolveRequest{Architecture: architecture, Objectives: []string{"causal-language-modeling"}})
	if err != nil {
		t.Fatal(err)
	}
	backend, ok := selection.Backend.(MLX)
	if !ok || backend.Python != "mlx-python" || selection.Execution.Framework != "mlx" || selection.Execution.Accelerators[0].MemoryBytes != 128<<30 || !strings.Contains(selection.Execution.Runtime, "MLX 0.31.2") {
		t.Fatalf("selection = %+v", selection)
	}
}

func TestMLXWorkerPublishesBestEvaluatedCheckpoint(t *testing.T) {
	source := string(mlxWorker)
	if !strings.Contains(source, `WORKER_REVISION = "`+MLXRevision+`"`) {
		t.Fatalf("embedded MLX worker revision does not match Go adapter %q", MLXRevision)
	}
	for _, expected := range []string{
		`selected_evaluation = min(candidates, key=lambda evaluation: evaluation["metrics"]["heldout_loss"])`,
		`self.model.load_weights(selected_path)`,
		`"selected_checkpoint": selection`,
		`selected checkpoint step {selected_step}`,
		`self.model.load_weights(weights_path)`,
		`if abs(artifact_loss - live_loss) > tolerance:`,
		`"value": live_loss`,
		`safety_steps = {1}`,
		`if self.evaluation_sequences and (not self.evaluations or self.evaluations[-1]["step"] != self.step_number):`,
		`error_class = "artifact-integrity"`,
		`error_class = "numerical-integrity"`,
		`non-finite training loss at optimizer step`,
		`non-finite gradient norm at optimizer step`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("MLX worker omits best-checkpoint publication behavior %q", expected)
		}
	}
}

func TestMLXResolverFailsClosed(t *testing.T) {
	valid := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`)
	if _, err := (MLXResolver{OS: "linux", Arch: "amd64"}).Resolve(context.Background(), ResolveRequest{Architecture: valid}); err == nil || !strings.Contains(err.Error(), "Apple Silicon") {
		t.Fatalf("platform error = %v", err)
	}
	unsupported := json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":100277,"tokenizer":{"name":"tiktoken/cl100k_base","revision":"tiktoken-cl100k-base"}}`)
	if _, err := (MLXResolver{OS: "darwin", Arch: "arm64"}).Resolve(context.Background(), ResolveRequest{Architecture: unsupported}); err == nil || !strings.Contains(err.Error(), "unsupported tokenizer") {
		t.Fatalf("tokenizer error = %v", err)
	}
}

func TestMLXBackendStreamsProtocolAndConsumesCompletion(t *testing.T) {
	worker := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
while IFS= read -r line; do
  case "$line" in *'"kind":"record"'*) records=$((records + 1));; esac
done
printf '%s\n' '{"kind":"event","schema":1,"event":{"kind":"progress","message":"real worker event","step":1,"tokens":2}}'
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
	observation, err := (MLX{Python: worker}).Run(context.Background(), Request{
		RunID: "run", Stage: "pretrain", Objective: "causal-language-modeling",
		ArchitectureSHA256: strings.Repeat("a", 64), Architecture: json.RawMessage(`{"family":"decoder-transformer"}`),
		Parameters: parameters, ArtifactDirectory: t.TempDir(), ArtifactPrefix: "artifacts",
		Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
			return consume(Record{ID: "one", Text: "hello"})
		}),
		Report: func(event Event) { reported = event.Message == "real worker event" },
	})
	if err != nil {
		t.Fatal(err)
	}
	if observation.Simulated || observation.Steps != 1 || !reported {
		t.Fatalf("observation = %+v, reported = %v", observation, reported)
	}
}

func TestInitializationOmitsMachineLocalPathFromDurableJSON(t *testing.T) {
	initialization := Initialization{
		SourceRunID: "prior", Path: "/private/machine/model.safetensors",
		Artifact: Artifact{Path: "artifacts/model.safetensors", SHA256: strings.Repeat("a", 64), Bytes: 12},
	}
	encoded, err := json.Marshal(initialization)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "/private/machine") || !strings.Contains(string(encoded), `"source_run_id":"prior"`) {
		t.Fatalf("durable initialization = %s", encoded)
	}
}

type recordSourceFunc func(context.Context, func(Record) error) error

func (function recordSourceFunc) Stream(ctx context.Context, consume func(Record) error) error {
	return function(ctx, consume)
}
