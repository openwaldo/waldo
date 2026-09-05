// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
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

	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

func writeHostfile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "hosts")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadTrainingHostfile(t *testing.T) {
	path := writeHostfile(t, "# rank zero first\ntrain-0\n\ntrain-1 # worker\ntrain-2\n")
	hostfile, err := loadTrainingHostfile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(hostfile.Hosts, ",") != "train-0,train-1,train-2" {
		t.Fatalf("hosts = %v", hostfile.Hosts)
	}
}

func TestLoadTrainingHostfileRejectsTopologyOptions(t *testing.T) {
	for _, content := range []string{
		"train-0 slots=8\ntrain-1\n",
		"train-0\ntrain-0\n",
		"train-0\n",
		"-oProxyCommand=bad\ntrain-1\n",
	} {
		if _, err := loadTrainingHostfile(writeHostfile(t, content)); err == nil {
			t.Fatalf("hostfile %q unexpectedly passed", content)
		}
	}
}

func TestCompareTorchTitanHosts(t *testing.T) {
	primary := training.TorchTitanHost{
		PythonVersion: "3.12", TorchVersion: "2.8", TorchTitanVersion: "0.2",
		Accelerators: []training.Accelerator{{Manufacturer: "NVIDIA", Model: "H200", MemoryBytes: 140 << 30}},
	}
	if err := compareTorchTitanHosts(primary, primary); err != nil {
		t.Fatal(err)
	}
	secondary := primary
	secondary.Accelerators = []training.Accelerator{{Manufacturer: "NVIDIA", Model: "H100", MemoryBytes: 80 << 30}}
	if err := compareTorchTitanHosts(primary, secondary); err == nil || !strings.Contains(err.Error(), "topology differs") {
		t.Fatalf("topology error = %v", err)
	}
}

func TestRunSecondaryStreamPlansNeedsNoCorpusData(t *testing.T) {
	parameters, err := training.ResolveParameters(training.Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	evaluation := training.EvaluationSet{Selection: "lowest-sha256-v1", SHA256: strings.Repeat("a", 64)}
	plan := model.MultiNodePlan{
		Kind: model.MultiNodePlanKind, Schema: model.MultiNodePlanSchema,
		RunID: "run-1", Stage: "pretrain", StageOrdinal: 1, StageCount: 1,
		Nodes: 2, Objective: "causal-language-modeling",
		ArchitectureSHA256: strings.Repeat("b", 64),
		Architecture:       json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`),
		Parameters:         parameters, EvaluationSet: &evaluation,
	}
	var stream bytes.Buffer
	if err := json.NewEncoder(&stream).Encode(plan); err != nil {
		t.Fatal(err)
	}
	called := false
	runner := func(_ context.Context, _ training.Cluster, request training.Request) error {
		called = true
		if request.Records != nil || request.EvaluationRecords != nil || len(request.BOM.Shards) != 0 || len(request.Inputs) != 0 {
			t.Fatalf("secondary received corpus data: %+v", request)
		}
		return nil
	}
	cluster := training.Cluster{Nodes: 2, NodeRank: 1, Rendezvous: "train-0:29500", RendezvousID: "test"}
	if err := runSecondaryStreamPlansWithRunner(Context{Execution: context.Background()}, cluster, t.TempDir(), &stream, runner, io.Discard, io.Discard); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("secondary runner was not called")
	}
}

func TestHostfileSessionLaunchesAndPublishesPlans(t *testing.T) {
	bin := t.TempDir()
	ssh := filepath.Join(bin, "ssh")
	script := `#!/bin/sh
case "$*" in
  *--check*)
    printf '%s\n' '{"python":"python3","python_version":"3.12","torch_version":"2.8","torchtitan_version":"0.2","accelerators":[{"manufacturer":"NVIDIA","model":"H200","memory_bytes":150323855360}]}'
    exit 0
    ;;
  *--plan-stdin*)
    IFS= read -r plan || exit 3
    printf '%s\n' 'worker accepted launcher plan'
    exit 0
    ;;
  *)
    cat >/dev/null
    exit 0
    ;;
esac
`
	if err := os.WriteFile(ssh, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	capabilities := training.TorchTitanHost{
		Python: "python3", PythonVersion: "3.12", TorchVersion: "2.8", TorchTitanVersion: "0.2",
		Accelerators: []training.Accelerator{{Manufacturer: "NVIDIA", Model: "H200", MemoryBytes: 140 << 30}},
	}
	previous := inspectHostfileTorchTitan
	inspectHostfileTorchTitan = func(context.Context) (training.TorchTitanHost, error) { return capabilities, nil }
	t.Cleanup(func() { inspectHostfileTorchTitan = previous })
	cluster := training.Cluster{Nodes: 2, Rendezvous: "train-0:29500", RendezvousID: "session-test"}
	var output bytes.Buffer
	session, err := startHostfileSession(context.Background(), trainingHostfile{Hosts: []string{"train-0", "train-1"}}, cluster, &output)
	if err != nil {
		t.Fatal(err)
	}
	evaluation := training.EvaluationSet{Selection: "lowest-sha256-v1", SHA256: strings.Repeat("a", 64)}
	if err := session.publish(model.MultiNodePlan{
		Kind: model.MultiNodePlanKind, Schema: model.MultiNodePlanSchema,
		RunID: "run-1", Stage: "pretrain", StageOrdinal: 1, StageCount: 1,
		Nodes: 2, EvaluationSet: &evaluation,
	}); err != nil {
		t.Fatal(err)
	}
	if err := session.finish(nil); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "[train-1] worker accepted launcher plan") {
		t.Fatalf("worker output = %q", output.String())
	}
}

func TestHostfileWorkerArgumentsCarryNCCLSettings(t *testing.T) {
	session := hostfileSession{
		remoteBinary: "/tmp/waldo-launch/build/waldo",
		remoteRoot:   "/tmp/waldo-launch/build",
		cluster: training.Cluster{
			Nodes: 2, Rendezvous: "train-0:29500", RendezvousID: "session-test",
			Interface: "ib0", HCA: "mlx5_0",
		},
	}
	arguments := strings.Join(session.workerArguments(1, false), " ")
	for _, expected := range []string{"--nccl-interface ib0", "--nccl-hca mlx5_0"} {
		if !strings.Contains(arguments, expected) {
			t.Fatalf("worker arguments %q omit %q", arguments, expected)
		}
	}
}
