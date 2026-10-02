// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestTransformersUnavailableDeviceProbe(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("Python required for subprocess probe test")
	}
	directory := t.TempDir()
	// Execute the actual probe with a minimal CPU-only torch stand-in.
	if err := os.WriteFile(filepath.Join(directory, "torch.py"), []byte("__version__ = 'test'\nclass CUDA:\n    def is_available(self): return False\ncuda = CUDA()\n"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PYTHONPATH", directory)
	t.Setenv("WALDO_TRANSFORMERS_DEVICE", "cuda")
	if _, err := ProbeTransformersDevice(t.Context(), python); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("unavailable CUDA: %v", err)
	}
}

func TestTransformersDoesNotOverrideExplicitHostPolicy(t *testing.T) {
	request := ResolveRequest{Architecture: json.RawMessage(`{"family":"huggingface-transformers"}`)}
	for _, preference := range []string{BackendFake, BackendMLX, BackendTorchTitan} {
		_, err := (EnvironmentResolver{Preference: preference}).Resolve(context.Background(), request)
		if err == nil || !strings.Contains(err.Error(), "conflicts") {
			t.Fatalf("%s: %v", preference, err)
		}
	}
	for _, cluster := range []Cluster{{Nodes: 2}, {WorldSize: 2}, {NodeRank: 1}} {
		_, err := (EnvironmentResolver{Cluster: cluster}).Resolve(context.Background(), request)
		if err == nil || !strings.Contains(err.Error(), "single-process") {
			t.Fatalf("%+v: %v", cluster, err)
		}
	}
}

func TestTransformersTrainerIsSoleOptimizerAuthority(t *testing.T) {
	parameters := Parameters{Steps: 2, BatchSize: 2, SequenceLength: 32, LearningRate: .001, Trainer: &TransformersTrainer{Class: "Trainer", Arguments: map[string]any{"learning_rate": .001, "per_device_train_batch_size": 2, "optim": "sgd"}}}
	resolved, err := ResolveParameters(parameters)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(resolved)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), `"optimizer"`) || strings.Contains(string(data), `"schedule"`) {
		t.Fatalf("native optimizer claims leaked into Trainer provenance: %s", data)
	}
	if !strings.Contains(string(data), `"optim":"sgd"`) {
		t.Fatal("Trainer optimizer was not preserved")
	}
}
