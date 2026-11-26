// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/training"
)

func hostfileWrapper(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "wrapper")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// The removed worker.done branch is replaced by closing worker.ready when the
// worker's stdout ends. A secondary that reads its plan and exits without an
// acknowledgement must fail the stage promptly, not wait for the 24h timeout.
func TestHostfilePublishFailsWhenWorkerExitsWithoutAcknowledgement(t *testing.T) {
	previousListener := listenHostfileRendezvous
	listenHostfileRendezvous = func(string) (io.Closer, error) { return io.NopCloser(strings.NewReader("")), nil }
	t.Cleanup(func() { listenHostfileRendezvous = previousListener })
	previous := hostfileStageReadyTimeout
	hostfileStageReadyTimeout = 20 * time.Second
	t.Cleanup(func() { hostfileStageReadyTimeout = previous })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	session := &hostfileSession{
		ctx: ctx, cancel: cancel, output: io.Discard,
		hostfile:   trainingHostfile{Hosts: []string{"train-0", "train-1"}, Wrapper: hostfileWrapper(t, "IFS= read -r plan\nexit 0")},
		cluster:    training.Cluster{Nodes: 2, Rendezvous: "127.0.0.1:0", RendezvousID: "exit-test"},
		remoteRoot: t.TempDir(),
	}
	started := time.Now()
	err := session.publish(model.MultiNodePlan{RunID: "run-1", Stage: "pretrain", StageOrdinal: 1, StageCount: 1})
	if err == nil || !strings.Contains(err.Error(), "train-1 closed output before acknowledging stage 1/1") {
		t.Fatalf("publish error = %v", err)
	}
	if elapsed := time.Since(started); elapsed > 10*time.Second {
		t.Fatalf("publish waited %s for an exited worker", elapsed)
	}
	for _, worker := range session.workers {
		<-worker.done
	}
}

// Rank 0 records each node's distributed launch; a node must report torchrun
// evidence for every stage with its own rank and the session's node count.
func TestHostfileExternalProbeRejectsInconsistentLaunchEvidence(t *testing.T) {
	compose, err := json.Marshal(model.ExternalCompose{Stages: []model.ExternalStage{{Name: "pretrain"}, {Name: "sft"}}})
	if err != nil {
		t.Fatal(err)
	}
	good := applaunch.Evidence{Profile: "distributed", Launcher: "torchrun", World: applaunch.World{Nodes: 2, NodeRank: 1, ProcessesPerNode: 1, WorldSize: 2}}
	for name, mutate := range map[string]func(map[string]applaunch.Evidence){
		"valid":         func(map[string]applaunch.Evidence) {},
		"wrong rank":    func(l map[string]applaunch.Evidence) { e := l["sft"]; e.World.NodeRank = 0; l["sft"] = e },
		"wrong nodes":   func(l map[string]applaunch.Evidence) { e := l["sft"]; e.World.Nodes = 3; l["sft"] = e },
		"python":        func(l map[string]applaunch.Evidence) { e := l["sft"]; e.Launcher = "python"; l["sft"] = e },
		"single":        func(l map[string]applaunch.Evidence) { e := l["sft"]; e.Profile = "single-process"; l["sft"] = e },
		"missing stage": func(l map[string]applaunch.Evidence) { delete(l, "sft") },
		"renamed stage": func(l map[string]applaunch.Evidence) { l["other"] = l["sft"]; delete(l, "sft") },
		"extra stage":   func(l map[string]applaunch.Evidence) { l["other"] = l["sft"] },
	} {
		launches := map[string]applaunch.Evidence{"pretrain": good, "sft": good}
		mutate(launches)
		raw, err := json.Marshal(externalHostPreflight{TorchTitanHost: training.TorchTitanHost{PythonVersion: "3.12", TorchVersion: "2.8", TorchTitanVersion: "external"}, Launches: launches})
		if err != nil {
			t.Fatal(err)
		}
		output := filepath.Join(t.TempDir(), "preflight.json")
		if err := os.WriteFile(output, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		session := &hostfileSession{
			ctx: context.Background(), externalCompose: string(compose), externalLaunches: map[string][]applaunch.Evidence{},
			hostfile:   trainingHostfile{Wrapper: hostfileWrapper(t, "cat "+shellQuote(output))},
			cluster:    training.Cluster{Nodes: 2, Rendezvous: "127.0.0.1:29500", RendezvousID: "probe"},
			remoteRoot: t.TempDir(),
		}
		host, err := session.probeHost("train-1", 1)
		if name == "valid" {
			if err != nil || host.TorchTitanVersion != "external" || len(session.externalLaunches["pretrain"]) != 1 || len(session.externalLaunches["sft"]) != 1 {
				t.Fatalf("valid evidence: %v %+v", err, session.externalLaunches)
			}
			continue
		}
		if err == nil || !strings.Contains(err.Error(), "train-1") {
			t.Fatalf("%s accepted: %v", name, err)
		}
	}
}

// Moving the backend check behind compose dispatch must keep the native
// --hostfile refusal for non-TorchTitan backends, with and without a compose.
func TestNativeHostfileStillRequiresTorchTitanBackend(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Model: config.Model{Root: filepath.Join(t.TempDir(), "models"), Backend: "fake"}}); err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	hostfile := filepath.Join(directory, "hosts")
	compose := filepath.Join(directory, "compose.yaml")
	native := "kind: waldo-model-compose\nschema: 2\n" +
		"architecture:\n  family: decoder-transformer\n  context_tokens: 128\n  vocabulary_size: 256\n  hidden_size: 64\n  intermediate_size: 192\n  layers: 2\n  attention_heads: 4\n  key_value_heads: 2\n  tie_embeddings: true\n  parameter_dtype: float32\n  tokenizer:\n    name: byte\n    revision: sha256:example\n" +
		"stages:\n  - name: pretrain\n    type: pre-training\n    objective: causal-language-modeling\n    corpora: [core/books]\n    parameters: {steps: 2, batch_size: 1, sequence_length: 64, learning_rate: 0.001, seed: 7}\n"
	for path, data := range map[string]string{hostfile: "train-0\ntrain-1\n", compose: native} {
		if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, args := range [][]string{{"model", "train", "native", "--hostfile", hostfile}, {"model", "train", "native", compose, "--hostfile", hostfile}} {
		var stdout, stderr bytes.Buffer
		if code := Run(args, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "--hostfile requires TorchTitan, but model.backend=fake") {
			t.Fatalf("%v: code %d, stderr %q", args, code, stderr.String())
		}
	}
}
