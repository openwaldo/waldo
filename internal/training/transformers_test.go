// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

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
