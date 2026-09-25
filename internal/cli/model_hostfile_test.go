// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/openwaldo/waldo/internal/lookaside"
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

func TestTorchTitanHostSummaryNamesLocalGPUConnection(t *testing.T) {
	host := training.TorchTitanHost{
		PythonVersion: "3.11", TorchVersion: "2.15", TorchTitanVersion: "0.3",
		LocalInterconnect: "nvlink",
		Accelerators:      []training.Accelerator{{Model: "H200"}, {Model: "H200"}},
	}
	summary := torchTitanHostSummary(host)
	if !strings.Contains(summary, "2 GPUs (NVLink between local GPUs)") {
		t.Fatalf("host summary = %q", summary)
	}
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

func TestLoadFuzzballTrainingHostlist(t *testing.T) {
	wrapper := filepath.Join(t.TempDir(), "ssh-wrapper")
	if err := os.WriteFile(wrapper, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	previousHostname := currentTrainingHostname
	currentTrainingHostname = func() (string, error) { return "train-0", nil }
	t.Cleanup(func() { currentTrainingHostname = previousHostname })
	t.Setenv("MULTINODE_HOSTLIST_NOSLOTS", "train-1, train-0,train-2")
	t.Setenv("MULTINODE_NODE_IP", "10.42.0.10")
	t.Setenv("MULTINODE_SSH_WRAPPER", wrapper)
	t.Setenv("MULTINODE_RSH_WRAPPER", "")
	hostfile, err := loadFuzzballTrainingHostlist()
	if err != nil {
		t.Fatal(err)
	}
	if hostfile.Source != "fuzzball" || strings.Join(hostfile.Hosts, ",") != "train-0,train-1,train-2" || hostfile.RendezvousHost != "10.42.0.10" || hostfile.Wrapper != wrapper {
		t.Fatalf("Fuzzball topology = %+v", hostfile)
	}
}

func TestLoadFuzzballTrainingHostlistRejectsInvalidEnvironment(t *testing.T) {
	wrapper := filepath.Join(t.TempDir(), "ssh-wrapper")
	if err := os.WriteFile(wrapper, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	previousHostname := currentTrainingHostname
	currentTrainingHostname = func() (string, error) { return "train-0", nil }
	t.Cleanup(func() { currentTrainingHostname = previousHostname })
	for _, test := range []struct {
		hosts, nodeIP, sshWrapper, rshWrapper, want string
	}{
		{"train-0", "", wrapper, "", "at least two hosts"},
		{"train-0:2,train-1:2", "", wrapper, "", "without slots"},
		{"train-0,train-0", "", wrapper, "", "repeats host"},
		{"train-0,train-1", "head", wrapper, "", "must be an IP address"},
		{"other-0,other-1", "", wrapper, "", "does not contain local rank-0 host"},
		{"train-0,train-1", "", "", "", "MULTINODE_SSH_WRAPPER is required"},
		{"train-0,train-1", "", wrapper, "/different/wrapper", "different launchers"},
	} {
		t.Setenv("MULTINODE_HOSTLIST_NOSLOTS", test.hosts)
		t.Setenv("MULTINODE_NODE_IP", test.nodeIP)
		t.Setenv("MULTINODE_SSH_WRAPPER", test.sshWrapper)
		t.Setenv("MULTINODE_RSH_WRAPPER", test.rshWrapper)
		if _, err := loadFuzzballTrainingHostlist(); err == nil || !strings.Contains(err.Error(), test.want) {
			t.Fatalf("hosts %q error = %v, want %q", test.hosts, err, test.want)
		}
	}
}

func TestLoadFuzzballSSHWrapperAcceptsRshAlias(t *testing.T) {
	wrapper := filepath.Join(t.TempDir(), "rsh-wrapper")
	if err := os.WriteFile(wrapper, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("MULTINODE_SSH_WRAPPER", "")
	t.Setenv("MULTINODE_RSH_WRAPPER", wrapper)
	got, err := loadFuzzballSSHWrapper()
	if err != nil || got != wrapper {
		t.Fatalf("wrapper = %q, err = %v", got, err)
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

func TestTorchTitanHostMismatchNamesInstallationTarget(t *testing.T) {
	err := torchTitanHostMismatchError("reno-gpu-02", io.ErrUnexpectedEOF)
	for _, expected := range []string{"host reno-gpu-02 is incompatible", "install the tested Python runtime on host reno-gpu-02", "torchtitan=="} {
		if !strings.Contains(err.Error(), expected) {
			t.Fatalf("mismatch error %q omits %q", err, expected)
		}
	}
}

func TestProbeHostNamesMissingRuntimeInstallationTarget(t *testing.T) {
	bin := t.TempDir()
	ssh := filepath.Join(bin, "ssh")
	if err := os.WriteFile(ssh, []byte("#!/bin/sh\nprintf '%s\\n' 'TorchTitan runtime is unavailable. Copy and run this installation block:' >&2\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	session := hostfileSession{
		ctx: context.Background(), remoteBinary: "/tmp/waldo", pythonDir: "/usr/bin",
		cluster: training.Cluster{Nodes: 2, Rendezvous: "train-0:29500", RendezvousID: "test"},
	}
	_, err := session.probeHost("reno-gpu-02", 1)
	if err == nil {
		t.Fatal("missing remote runtime unexpectedly passed")
	}
	for _, expected := range []string{"host reno-gpu-02 is not ready", "correct the reported condition on host reno-gpu-02", "TorchTitan runtime is unavailable"} {
		if !strings.Contains(err.Error(), expected) {
			t.Fatalf("preflight error %q omits %q", err, expected)
		}
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
	var stdout bytes.Buffer
	if err := runSecondaryStreamPlansWithRunner(Context{Execution: context.Background()}, cluster, t.TempDir(), nil, &stream, nil, runner, &stdout, io.Discard); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("secondary runner was not called")
	}
	var ready hostfileStageReady
	if err := json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &ready); err != nil {
		t.Fatalf("decode secondary readiness %q: %v", stdout.String(), err)
	}
	if ready.Kind != hostfileStageReadyKind || ready.Schema != 1 || ready.RunID != plan.RunID || ready.Stage != plan.Stage || ready.StageOrdinal != plan.StageOrdinal || ready.StageCount != plan.StageCount {
		t.Fatalf("secondary readiness = %+v", ready)
	}
}

func TestRunSecondaryStreamPlansAcceptsNoWork(t *testing.T) {
	runner := func(context.Context, training.Cluster, training.Request) error {
		t.Fatal("secondary runner was called for an empty launcher plan stream")
		return nil
	}
	cluster := training.Cluster{Nodes: 2, NodeRank: 1, Rendezvous: "train-0:29500", RendezvousID: "test"}
	var stdout bytes.Buffer
	if err := runSecondaryStreamPlansWithRunner(Context{Execution: context.Background()}, cluster, t.TempDir(), nil, strings.NewReader(""), nil, runner, &stdout, io.Discard); err != nil {
		t.Fatalf("empty launcher plan stream: %v", err)
	}
	if stdout.Len() != 0 {
		t.Fatalf("empty launcher plan stream emitted control output %q", stdout.String())
	}
}

func TestRunSecondaryStreamPlansRejectsTruncatedStages(t *testing.T) {
	parameters, err := training.ResolveParameters(training.Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	plan := model.MultiNodePlan{
		Kind: model.MultiNodePlanKind, Schema: model.MultiNodePlanSchema,
		RunID: "run-1", Stage: "pretrain", StageOrdinal: 1, StageCount: 2,
		Nodes: 2, Objective: "causal-language-modeling",
		ArchitectureSHA256: strings.Repeat("b", 64),
		Architecture:       json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`),
		Parameters:         parameters,
		EvaluationSet:      &training.EvaluationSet{Selection: "lowest-sha256-v1", SHA256: strings.Repeat("a", 64)},
	}
	var stream bytes.Buffer
	if err := json.NewEncoder(&stream).Encode(plan); err != nil {
		t.Fatal(err)
	}
	runner := func(context.Context, training.Cluster, training.Request) error { return nil }
	cluster := training.Cluster{Nodes: 2, NodeRank: 1, Rendezvous: "train-0:29500", RendezvousID: "test"}
	err = runSecondaryStreamPlansWithRunner(Context{Execution: context.Background()}, cluster, t.TempDir(), nil, &stream, nil, runner, io.Discard, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "launcher plan stream ended before the final stage") {
		t.Fatalf("truncated launcher plan error = %v", err)
	}
}

func TestRunSecondaryStreamPlansDoesNotAcknowledgeFailedReadiness(t *testing.T) {
	parameters, err := training.ResolveParameters(training.Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	plan := model.MultiNodePlan{
		Kind: model.MultiNodePlanKind, Schema: model.MultiNodePlanSchema,
		RunID: "run-1", Stage: "pretrain", StageOrdinal: 1, StageCount: 2,
		Nodes: 2, Objective: "causal-language-modeling",
		ArchitectureSHA256: strings.Repeat("b", 64),
		Architecture:       json.RawMessage(`{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`),
		Parameters:         parameters,
		EvaluationSet:      &training.EvaluationSet{Selection: "lowest-sha256-v1", SHA256: strings.Repeat("a", 64)},
	}
	var stream bytes.Buffer
	if err := json.NewEncoder(&stream).Encode(plan); err != nil {
		t.Fatal(err)
	}
	prepare := func(context.Context, training.Cluster) error { return errors.New("rendezvous unreachable") }
	runner := func(context.Context, training.Cluster, training.Request) error {
		t.Fatal("training started after readiness failed")
		return nil
	}
	var stdout bytes.Buffer
	cluster := training.Cluster{Nodes: 2, NodeRank: 1, Rendezvous: "train-0:29500", RendezvousID: "test"}
	err = runSecondaryStreamPlansWithRunner(Context{Execution: context.Background()}, cluster, t.TempDir(), nil, &stream, prepare, runner, &stdout, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "stage 1/2 secondary readiness") || !strings.Contains(err.Error(), "rendezvous unreachable") {
		t.Fatalf("readiness error = %v", err)
	}
	if stdout.Len() != 0 {
		t.Fatalf("failed readiness emitted acknowledgement %q", stdout.String())
	}
}

func TestRunSecondaryStreamPlansMaterializesNodeLocalCorpus(t *testing.T) {
	bom := seedMultiNodeCorpus(t)
	plan := multiNodePlanForTest(t, bom, `{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`)
	plan.Parallelism = training.Parallelism{WorldSize: 2, GPUsPerNode: 1, DataPlane: training.DataPlaneNodeLocal}
	var stream bytes.Buffer
	if err := json.NewEncoder(&stream).Encode(plan); err != nil {
		t.Fatal(err)
	}
	cache, err := lookaside.NewCache(t.TempDir(), nil, lookaside.WithPersistentStorage(t.TempDir(), 1<<20))
	if err != nil {
		t.Fatal(err)
	}
	cluster := training.Cluster{Nodes: 2, NodeRank: 1, WorldSize: 2, Rendezvous: "train-0:29500", RendezvousID: "test"}
	called := false
	runner := func(_ context.Context, _ training.Cluster, request training.Request) error {
		called = true
		if request.Records == nil || request.EvaluationRecords == nil || len(request.Inputs) != 1 || len(request.BOM.Shards) != 1 {
			t.Fatalf("node-local request omitted corpus data: %+v", request)
		}
		if request.DataNodeRank != 1 || request.PreparedCacheDirectory != filepath.Join(cache.Scratch(), "prepared") || request.PreparedCacheMaxBytes != cache.MaxBytes() {
			t.Fatalf("node-local cache request = %+v", request)
		}
		return nil
	}
	if err := runSecondaryStreamPlansWithRunner(Context{Execution: context.Background()}, cluster, t.TempDir(), cache, &stream, nil, runner, io.Discard, io.Discard); err != nil {
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
    printf '%s\n' '{"kind":"waldo-hostfile-stage-ready","schema":1,"run_id":"run-1","stage":"pretrain","stage_ordinal":1,"stage_count":2}'
    printf '%s\n' 'worker accepted launcher stage 1'
    IFS= read -r plan || exit 4
    printf '%s\n' '{"kind":"waldo-hostfile-stage-ready","schema":1,"run_id":"run-2","stage":"post-train","stage_ordinal":2,"stage_count":2}'
    printf '%s\n' 'worker accepted launcher stage 2'
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
	previousListener := listenHostfileRendezvous
	listenHostfileRendezvous = func(string) (io.Closer, error) {
		return io.NopCloser(strings.NewReader("")), nil
	}
	t.Cleanup(func() { listenHostfileRendezvous = previousListener })
	cluster := training.Cluster{Nodes: 2, Rendezvous: "127.0.0.1:0", RendezvousID: "session-test"}
	scratch := t.TempDir()
	cacheRoot := t.TempDir()
	cache, err := lookaside.NewCache(cacheRoot, nil, lookaside.WithPersistentStorage(scratch, 1<<20))
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	session, err := startHostfileSession(context.Background(), trainingHostfile{Hosts: []string{"train-0", "train-1"}}, cluster, cache, &output)
	if err != nil {
		t.Fatal(err)
	}
	if session.workersStarted || len(session.workers) != 0 {
		t.Fatalf("training workers started before the first stage plan: %+v", session.workers)
	}
	if session.cluster.WorldSize != 2 {
		t.Fatalf("discovered world size = %d, want 2", session.cluster.WorldSize)
	}
	if !strings.HasPrefix(session.remoteRoot, scratch+string(os.PathSeparator)) {
		t.Fatalf("remote launch root = %q, want beneath configured scratch %q", session.remoteRoot, scratch)
	}
	evaluation := training.EvaluationSet{Selection: "lowest-sha256-v1", SHA256: strings.Repeat("a", 64)}
	if err := session.publish(model.MultiNodePlan{
		Kind: model.MultiNodePlanKind, Schema: model.MultiNodePlanSchema,
		RunID: "run-1", Stage: "pretrain", StageOrdinal: 1, StageCount: 2,
		Nodes: 2, EvaluationSet: &evaluation,
	}); err != nil {
		t.Fatal(err)
	}
	if !session.workersStarted || len(session.workers) != 1 {
		t.Fatalf("first stage plan did not start one secondary worker: %+v", session.workers)
	}
	if err := session.publish(model.MultiNodePlan{
		Kind: model.MultiNodePlanKind, Schema: model.MultiNodePlanSchema,
		RunID: "run-2", Stage: "post-train", StageOrdinal: 2, StageCount: 2,
		Nodes: 2, EvaluationSet: &evaluation,
	}); err != nil {
		t.Fatal(err)
	}
	if err := session.finish(nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(session.remoteRoot); !os.IsNotExist(err) {
		t.Fatalf("launch staging remains after finish: %v", err)
	}
	if !strings.Contains(output.String(), "[train-1] worker accepted launcher stage 1") || !strings.Contains(output.String(), "[train-1] worker accepted launcher stage 2") {
		t.Fatalf("worker output = %q", output.String())
	}
	if strings.Contains(output.String(), hostfileStageReadyKind) {
		t.Fatalf("worker control frames leaked into output: %q", output.String())
	}
	for _, expected := range []string{"multi-host preflight  rank 0 ready", "multi-host preflight  staging WALDO on train-1", "multi-host preflight  train-1 ready"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("preflight output %q omits %q", output.String(), expected)
		}
	}
}

func TestHostfileSessionNoWorkDoesNotStartTrainingWorkers(t *testing.T) {
	bin := t.TempDir()
	ssh := filepath.Join(bin, "ssh")
	marker := filepath.Join(t.TempDir(), "worker-started")
	script := `#!/bin/sh
case "$*" in
  *--check*)
    printf '%s\n' '{"python":"python3","python_version":"3.12","torch_version":"2.8","torchtitan_version":"0.2","accelerators":[{"manufacturer":"NVIDIA","model":"H200","memory_bytes":150323855360}]}'
    ;;
  *--plan-stdin*)
    : > '` + marker + `'
    cat >/dev/null
    ;;
  *) cat >/dev/null ;;
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
	previousListener := listenHostfileRendezvous
	listenHostfileRendezvous = func(string) (io.Closer, error) { return io.NopCloser(strings.NewReader("")), nil }
	t.Cleanup(func() { listenHostfileRendezvous = previousListener })
	cache, err := lookaside.NewCache(t.TempDir(), nil, lookaside.WithPersistentStorage(t.TempDir(), 1<<20))
	if err != nil {
		t.Fatal(err)
	}
	session, err := startHostfileSession(context.Background(), trainingHostfile{Hosts: []string{"train-0", "train-1"}}, training.Cluster{Nodes: 2, Rendezvous: "127.0.0.1:0", RendezvousID: "no-work"}, cache, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.finish(nil); err != nil {
		t.Fatal(err)
	}
	if session.workersStarted || len(session.workers) != 0 {
		t.Fatalf("no-work session started training workers: %+v", session.workers)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("remote training worker was invoked for no work: %v", err)
	}
}

func TestHostfileWorkerMaterializeProgressReusesTerminalLine(t *testing.T) {
	var output bytes.Buffer
	session := hostfileSession{output: &output, outputTerminal: true}
	session.copyWorkerStdout(&hostfileWorker{host: "train-1", ready: make(chan hostfileStageReady, 1)}, strings.NewReader(
		"  materialize [===================     ]  82%  107/131  28.0 GiB/33.8 GiB  verified one\n"+
			"  materialize [====================    ]  83%  108/131  28.3 GiB/33.8 GiB  verified two\n"+
			"  materialize [========================] 100%  131/131  33.8 GiB/33.8 GiB  verified final\n"+
			"joining rendezvous as node 1 of 2\n"))

	got := output.String()
	if strings.Count(got, "\r\x1b[K[train-1]   materialize") != 3 {
		t.Fatalf("terminal materialize output = %q", got)
	}
	if strings.Contains(got, "verified one\n") || strings.Contains(got, "verified two\n") {
		t.Fatalf("intermediate terminal updates created lines: %q", got)
	}
	if !strings.Contains(got, "verified final\n[train-1] joining rendezvous") {
		t.Fatalf("completed terminal update was not finalized: %q", got)
	}
}

func TestHostfileWorkerMaterializeProgressRemainsLineOrientedInLogs(t *testing.T) {
	var output bytes.Buffer
	session := hostfileSession{output: &output}
	session.copyWorkerOutput("train-1", strings.NewReader(
		"  materialize [===================     ]  82%  107/131  28.0 GiB/33.8 GiB  verified one\n"+
			"  materialize [========================] 100%  131/131  33.8 GiB/33.8 GiB  verified final\n"))

	if strings.Count(output.String(), "\n") != 2 || strings.Contains(output.String(), "\r") {
		t.Fatalf("non-terminal materialize output = %q", output.String())
	}
}

func TestHostfilePublishRejectsWrongStageAcknowledgement(t *testing.T) {
	previousListener := listenHostfileRendezvous
	listenHostfileRendezvous = func(string) (io.Closer, error) { return io.NopCloser(strings.NewReader("")), nil }
	t.Cleanup(func() { listenHostfileRendezvous = previousListener })
	inputReader, inputWriter := io.Pipe()
	defer inputReader.Close()
	go func() { _, _ = io.Copy(io.Discard, inputReader) }()
	worker := &hostfileWorker{
		host: "train-1", stdin: inputWriter,
		ready: make(chan hostfileStageReady, 1), done: make(chan struct{}),
	}
	worker.ready <- hostfileStageReady{Kind: hostfileStageReadyKind, Schema: 1, RunID: "wrong-run", Stage: "post-train", StageOrdinal: 2, StageCount: 3}
	session := hostfileSession{ctx: context.Background(), cluster: training.Cluster{Rendezvous: "127.0.0.1:0"}, workers: []*hostfileWorker{worker}, workersStarted: true}
	err := session.publish(model.MultiNodePlan{RunID: "run-2", Stage: "post-train", StageOrdinal: 2, StageCount: 3})
	_ = inputWriter.Close()
	if err == nil || !strings.Contains(err.Error(), "train-1 acknowledged unexpected stage plan") || !strings.Contains(err.Error(), "wrong-run") {
		t.Fatalf("wrong acknowledgement error = %v", err)
	}
}

func TestHostfilePublishTimesOutNamingHostAndStage(t *testing.T) {
	previousListener := listenHostfileRendezvous
	listenHostfileRendezvous = func(string) (io.Closer, error) { return io.NopCloser(strings.NewReader("")), nil }
	t.Cleanup(func() { listenHostfileRendezvous = previousListener })
	inputReader, inputWriter := io.Pipe()
	defer inputReader.Close()
	go func() { _, _ = io.Copy(io.Discard, inputReader) }()
	worker := &hostfileWorker{
		host: "train-2", stdin: inputWriter,
		ready: make(chan hostfileStageReady, 1), done: make(chan struct{}),
	}
	previous := hostfileStageReadyTimeout
	hostfileStageReadyTimeout = 10 * time.Millisecond
	t.Cleanup(func() { hostfileStageReadyTimeout = previous })
	session := hostfileSession{ctx: context.Background(), cluster: training.Cluster{Rendezvous: "127.0.0.1:0"}, workers: []*hostfileWorker{worker}, workersStarted: true}
	err := session.publish(model.MultiNodePlan{RunID: "run-3", StageOrdinal: 3, StageCount: 5})
	_ = inputWriter.Close()
	if err == nil || !strings.Contains(err.Error(), "train-2 did not acknowledge stage 3/5") || !strings.Contains(err.Error(), "10ms") {
		t.Fatalf("readiness timeout error = %v", err)
	}
}

func TestHostfileFinishPreservesPrimaryAndSecondaryFailureSemantics(t *testing.T) {
	primaryFailure := errors.New("primary failed")
	secondaryFailure := errors.New("secondary failed")
	for _, test := range []struct {
		name       string
		primaryErr error
		workerErr  error
		want       string
	}{
		{name: "success"},
		{name: "secondary failure", workerErr: secondaryFailure, want: "secondary training workers failed: train-1: secondary failed"},
		{name: "primary failure", primaryErr: primaryFailure, want: "primary failed"},
		{name: "primary failure remains authoritative", primaryErr: primaryFailure, workerErr: secondaryFailure, want: "primary failed"},
		{name: "secondary failure that canceled primary", primaryErr: context.Canceled, workerErr: secondaryFailure, want: "secondary training workers failed: train-1: secondary failed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			done := make(chan struct{})
			close(done)
			reader, writer := io.Pipe()
			defer reader.Close()
			worker := &hostfileWorker{host: "train-1", stdin: writer, done: done, err: test.workerErr}
			session := hostfileSession{
				ctx: context.Background(), cancel: func() {}, output: io.Discard,
				hostfile: trainingHostfile{Hosts: []string{"train-0", "train-1"}, Wrapper: "/usr/bin/true"},
				workers:  []*hostfileWorker{worker}, workersStarted: true,
			}
			err := session.finish(test.primaryErr)
			if test.want == "" {
				if err != nil {
					t.Fatalf("finish error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("finish error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestTrainingRendezvousReachability(t *testing.T) {
	previous := dialTrainingRendezvous
	t.Cleanup(func() { dialTrainingRendezvous = previous })
	dialTrainingRendezvous = func(_ context.Context, address string) error {
		if address != "train-0:29500" {
			t.Fatalf("rendezvous address = %q", address)
		}
		return nil
	}
	if err := checkTrainingRendezvous(context.Background(), "train-0:29500"); err != nil {
		t.Fatal(err)
	}

	dialTrainingRendezvous = func(context.Context, string) error {
		return errors.New("no route to host")
	}
	err := checkTrainingRendezvous(context.Background(), "train-0:29500")
	if err == nil || !strings.Contains(err.Error(), "verify name resolution, routing, and firewall rules") {
		t.Fatalf("closed rendezvous error = %v", err)
	}
}

func TestHostfileWorkerArgumentsCarryNCCLSettings(t *testing.T) {
	session := hostfileSession{
		remoteBinary:  "/tmp/waldo-launch/build/waldo",
		remoteRoot:    "/tmp/waldo-launch/build",
		cacheRoot:     "/home/gmk/.waldo/cache",
		cacheScratch:  "/home/gmk/.waldo/scratch",
		cacheMaxBytes: 20 << 30,
		cacheMirrors:  []string{"https://mirror.example/lookaside/v1"},
		cluster: training.Cluster{
			Nodes: 2, Rendezvous: "train-0:29500", RendezvousID: "session-test",
			Interface: "ib0", HCA: "mlx5_0",
		},
	}
	arguments := strings.Join(session.workerArguments(1, false), " ")
	for _, expected := range []string{"--nccl-interface ib0", "--nccl-hca mlx5_0", "--cache-root /home/gmk/.waldo/cache", "--cache-scratch /home/gmk/.waldo/scratch", "--cache-max-bytes 21474836480", "--cache-mirror https://mirror.example/lookaside/v1"} {
		if !strings.Contains(arguments, expected) {
			t.Fatalf("worker arguments %q omit %q", arguments, expected)
		}
	}
	if !strings.Contains(arguments, "--scratch /tmp/waldo-launch/build/runs/session-test/node-1") {
		t.Fatalf("worker arguments %q omit session-scoped scratch", arguments)
	}
}

func TestHostfileRemoteInvocationSelectsRankZeroPythonDirectory(t *testing.T) {
	session := hostfileSession{pythonDir: "/opt/waldo-python/bin"}
	invocation := session.remoteInvocation([]string{"/tmp/waldo", "model", "train-worker"})
	if !strings.HasPrefix(invocation, "PATH='/opt/waldo-python/bin:/usr/local/bin:/usr/bin:/bin' ") {
		t.Fatalf("remote invocation = %q", invocation)
	}
}

func TestHostfileRemoteWorkerInvocationPublishesPIDAndForwardsTermination(t *testing.T) {
	session := hostfileSession{pythonDir: "/opt/waldo-python/bin", remoteRoot: "/tmp/waldo/session"}
	invocation := session.remoteWorkerInvocation(2, []string{"/tmp/waldo", "model", "train-worker"})
	for _, expected := range []string{
		`env PATH='/opt/waldo-python/bin:/usr/local/bin:/usr/bin:/bin' '/tmp/waldo' 'model' 'train-worker' <&0 & child=$!`,
		`printf '%s\n' "$child" > '/tmp/waldo/session/worker-2.pid'`,
		`trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child"; exit 143' HUP INT TERM`,
		`rm -f -- '/tmp/waldo/session/worker-2.pid'`,
	} {
		if !strings.Contains(invocation, expected) {
			t.Fatalf("remote worker invocation %q omits %q", invocation, expected)
		}
	}
}

func TestHostfileRemoteWorkerInvocationPreservesPlanStdin(t *testing.T) {
	directory := t.TempDir()
	helper := filepath.Join(directory, "worker")
	result := filepath.Join(directory, "result")
	if err := os.WriteFile(helper, []byte("#!/bin/sh\nIFS= read -r line\nprintf '%s\\n' \"$line\" > \"$1\"\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	session := hostfileSession{pythonDir: "/usr/bin", remoteRoot: directory}
	command := exec.Command("sh", "-c", session.remoteWorkerInvocation(1, []string{helper, result}))
	command.Stdin = strings.NewReader("stage-plan\n")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("remote worker invocation: %v: %s", err, output)
	}
	data, err := os.ReadFile(result)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "stage-plan\n" {
		t.Fatalf("worker stdin = %q, want stage plan", data)
	}
	if _, err := os.Stat(session.workerPIDPath(1)); !os.IsNotExist(err) {
		t.Fatalf("worker PID file remains after normal exit: %v", err)
	}
}

func TestHostfileResumeStagesVerifiedCheckpointUnderConfiguredScratch(t *testing.T) {
	sourceDirectory := t.TempDir()
	source := filepath.Join(sourceDirectory, "runtime.pt")
	data := []byte("synthetic checkpoint")
	if err := os.WriteFile(source, data, 0o600); err != nil {
		t.Fatal(err)
	}
	digest, err := fileSHA256(source)
	if err != nil {
		t.Fatal(err)
	}
	scratch := t.TempDir()
	session := hostfileSession{
		ctx:        context.Background(),
		resumeRoot: filepath.Join(scratch, "multinode", "binary", "session", "resume"),
		output:     io.Discard,
	}
	resume := &training.ResumePoint{
		Step: 42,
		Checkpoint: training.Checkpoint{Artifacts: []training.Artifact{{
			Path: "artifacts/checkpoints/step-00000042/runtime.pt", SHA256: digest, Bytes: int64(len(data)),
		}}},
		Paths: []string{source},
	}
	if err := session.prepareResume("run-42", resume); err != nil {
		t.Fatal(err)
	}
	if len(resume.Paths) != 1 || !strings.HasPrefix(resume.Paths[0], session.resumeRoot+string(os.PathSeparator)) {
		t.Fatalf("staged paths = %v, want beneath %s", resume.Paths, session.resumeRoot)
	}
	if err := model.VerifyArtifactFile(resume.Paths[0], resume.Checkpoint.Artifacts[0]); err != nil {
		t.Fatal(err)
	}
	session.cleanupResumeStaging()
	if _, err := os.Stat(session.resumeRoot); !os.IsNotExist(err) {
		t.Fatalf("resume staging remains after cleanup: %v", err)
	}
}

func TestHostfileSSHUsesGracefulBoundedCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	session := hostfileSession{ctx: ctx}
	command := session.remoteCommand("worker", "true")
	if command.Cancel == nil {
		t.Fatal("SSH command has no graceful cancellation function")
	}
	if command.WaitDelay != hostfileWorkerExitGrace {
		t.Fatalf("SSH wait delay = %v, want %v", command.WaitDelay, hostfileWorkerExitGrace)
	}
}

func TestHostfileRemoteCommandUsesFuzzballWrapperContract(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	wrapper := filepath.Join(t.TempDir(), "ssh-wrapper")
	session := hostfileSession{ctx: ctx, hostfile: trainingHostfile{Wrapper: wrapper}}
	command := session.remoteCommand("worker-1", "printf '%s' ready")
	if command.Path != wrapper {
		t.Fatalf("launcher = %q, want %q", command.Path, wrapper)
	}
	if got := strings.Join(command.Args[1:], "|"); got != "worker-1|printf '%s' ready" {
		t.Fatalf("wrapper arguments = %q", got)
	}
	if command.Cancel == nil || command.WaitDelay != hostfileWorkerExitGrace {
		t.Fatal("Fuzzball wrapper lacks bounded graceful cancellation")
	}
}
