// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/training"
)

func TestPublishMultiNodePlanRoundTrips(t *testing.T) {
	root := t.TempDir()
	builder := Builder{Root: root, MultiNode: MultiNodeHandoff{RendezvousID: "run-42", Nodes: 4, StageOrdinal: 1, StageCount: 1}}
	evaluation := &training.EvaluationSet{Selection: "lowest-sha256-v1", Seed: 7, Records: 3, SHA256: strings.Repeat("a", 64)}
	runBOM := RunBOM{
		ArchitectureSHA256: strings.Repeat("b", 64),
		CorpusBOMSHA256:    strings.Repeat("c", 64),
		Parameters:         training.ResolvedParameters{Seed: 7, BatchSize: 2, SequenceLength: 8},
		Execution: training.Execution{Parallelism: training.Parallelism{
			Requested: training.ParallelismAuto, Strategy: training.ParallelismData,
			WorldSize: 4, Nodes: 2, GPUsPerNode: 2,
			CompleteModelCopies: 4, GPUsSharingEachModelCopy: 1,
		}},
		EvaluationSet: evaluation,
		Initialization: &training.Initialization{
			SourceType: "run", SourceRunID: "run0000",
			Artifact: training.Artifact{Path: "artifacts/model.safetensors", SHA256: strings.Repeat("d", 64), Bytes: 4},
			Path:     filepath.Join(root, "smoke", "runs", "0001-train-run0000", "artifacts", "model.safetensors"),
		},
	}
	prepared := PreparedStage{BOM: corpus.BOM{Kind: "openwaldo-bom"}}
	pin := RunPin{ID: "run0001", Stage: "train-0001", BOMSHA256: strings.Repeat("e", 64)}
	stage := Stage{Name: "train-0001", Type: "pre-training", Objective: "causal-language-modeling"}
	architecture := json.RawMessage(`{"family":"decoder-transformer"}`)

	if err := builder.publishMultiNodePlan(pin, runBOM, prepared, architecture, stage, nil); err != nil {
		t.Fatalf("publish: %v", err)
	}
	data, err := os.ReadFile(MultiNodePlanPath(root, "run-42"))
	if err != nil {
		t.Fatalf("read plan: %v", err)
	}
	var plan MultiNodePlan
	if err := json.Unmarshal(data, &plan); err != nil {
		t.Fatalf("decode plan: %v", err)
	}
	if plan.Kind != MultiNodePlanKind || plan.Schema != MultiNodePlanSchema || plan.RunBOMSHA256 != pin.BOMSHA256 || plan.RunID != "run0001" || plan.Stage != "train-0001" || plan.Objective != "causal-language-modeling" {
		t.Fatalf("plan identity = %+v", plan)
	}
	var got, want bytes.Buffer
	if err := json.Compact(&got, plan.Architecture); err != nil {
		t.Fatalf("compact plan architecture: %v", err)
	}
	if err := json.Compact(&want, architecture); err != nil {
		t.Fatalf("compact architecture: %v", err)
	}
	if plan.ArchitectureSHA256 != runBOM.ArchitectureSHA256 || got.String() != want.String() {
		t.Fatalf("plan corpus/arch = %+v", plan)
	}
	if plan.EvaluationSet == nil || plan.EvaluationSet.SHA256 != evaluation.SHA256 || plan.Parameters.Seed != 7 {
		t.Fatalf("plan split/params = %+v", plan)
	}
	if plan.Parallelism.Strategy != training.ParallelismData || plan.Parallelism.WorldSize != 4 {
		t.Fatalf("plan parallelism = %+v", plan.Parallelism)
	}
	if plan.InitializationPath != "smoke/runs/0001-train-run0000/artifacts/model.safetensors" {
		t.Fatalf("plan initialization path = %q", plan.InitializationPath)
	}
	if plan.Initialization == nil || plan.Initialization.SourceRunID != "run0000" {
		t.Fatalf("plan initialization = %+v", plan.Initialization)
	}
}

func TestPublishMultiNodePlanUsesLauncherCallback(t *testing.T) {
	root := t.TempDir()
	var published MultiNodePlan
	builder := Builder{Root: root, MultiNode: MultiNodeHandoff{
		RendezvousID: "hostfile-42", Nodes: 2, StageOrdinal: 1, StageCount: 1,
		Publish: func(plan MultiNodePlan) error {
			published = plan
			return nil
		},
	}}
	stage := Stage{Name: "pretrain", Type: "pre-training", Objective: "causal-language-modeling"}
	if err := builder.publishMultiNodePlan(
		RunPin{ID: "run0001", Stage: "pretrain"},
		RunBOM{EvaluationSet: &training.EvaluationSet{Selection: "lowest-sha256-v1"}},
		PreparedStage{}, json.RawMessage(`{"family":"decoder-transformer"}`), stage, nil,
	); err != nil {
		t.Fatal(err)
	}
	if published.RunID != "run0001" || published.Nodes != 2 {
		t.Fatalf("published plan = %+v", published)
	}
	if _, err := os.Stat(MultiNodePlanPath(root, "hostfile-42")); !os.IsNotExist(err) {
		t.Fatalf("launcher callback must not create a shared plan file: %v", err)
	}
}

func TestPublishMultiNodePlanRejectsUnportableInitialization(t *testing.T) {
	root := t.TempDir()
	stage := Stage{Name: "train-0001", Type: "pre-training", Objective: "causal-language-modeling"}
	for name, path := range map[string]string{
		"empty path":   "",
		"outside root": filepath.Join(t.TempDir(), "elsewhere", "model.safetensors"),
	} {
		t.Run(name, func(t *testing.T) {
			builder := Builder{Root: root, MultiNode: MultiNodeHandoff{RendezvousID: "run-" + strings.ReplaceAll(name, " ", "-"), Nodes: 4, StageOrdinal: 1, StageCount: 1}}
			runBOM := RunBOM{Initialization: &training.Initialization{SourceType: "run", Path: path}}
			err := builder.publishMultiNodePlan(RunPin{ID: "run0001"}, runBOM, PreparedStage{}, nil, stage, nil)
			if err == nil {
				t.Fatal("expected unportable initialization to fail closed")
			}
			if _, statErr := os.Stat(MultiNodePlanPath(root, builder.MultiNode.RendezvousID)); !os.IsNotExist(statErr) {
				t.Fatalf("plan must not be written; stat err = %v", statErr)
			}
		})
	}
}

func TestPublishMultiNodePlanRejectsLeftoverPlan(t *testing.T) {
	builder := Builder{Root: t.TempDir(), MultiNode: MultiNodeHandoff{RendezvousID: "run-42", Nodes: 4, StageOrdinal: 1, StageCount: 1}}
	stage := Stage{Name: "train-0001", Type: "pre-training", Objective: "causal-language-modeling"}
	if err := builder.publishMultiNodePlan(RunPin{ID: "run0001"}, RunBOM{}, PreparedStage{}, nil, stage, nil); err != nil {
		t.Fatalf("first publish: %v", err)
	}
	err := builder.publishMultiNodePlan(RunPin{ID: "run0002"}, RunBOM{}, PreparedStage{}, nil, stage, nil)
	if err == nil || !strings.Contains(err.Error(), "already exists") || !strings.Contains(err.Error(), "--rendezvous-id") {
		t.Fatalf("leftover plan guard error = %v", err)
	}
}

func TestPublishMultiNodePlanRequiresResumeStaging(t *testing.T) {
	builder := Builder{Root: t.TempDir(), MultiNode: MultiNodeHandoff{RendezvousID: "run-42", Nodes: 4, StageOrdinal: 1, StageCount: 1}}
	stage := Stage{Name: "train-0001"}
	resume := &training.ResumePoint{Step: 5}
	err := builder.publishMultiNodePlan(RunPin{ID: "run0001"}, RunBOM{}, PreparedStage{}, nil, stage, resume)
	if err == nil || !strings.Contains(err.Error(), "requires a launcher") {
		t.Fatalf("resume guard error = %v", err)
	}
	if _, statErr := os.Stat(MultiNodePlanPath(builder.Root, "run-42")); !os.IsNotExist(statErr) {
		t.Fatalf("plan must not be written when a multi-node run is asked to resume; stat err = %v", statErr)
	}
}

func TestPublishMultiNodePlanCarriesPreparedResume(t *testing.T) {
	var published MultiNodePlan
	builder := Builder{Root: t.TempDir(), MultiNode: MultiNodeHandoff{
		RendezvousID: "run-42", Nodes: 2, StageOrdinal: 1, StageCount: 1,
		PrepareResume: func(runID string, resume *training.ResumePoint) error {
			if runID != "run0001" || resume.Step != 5 {
				t.Fatalf("prepare resume = %q, %+v", runID, resume)
			}
			resume.Paths = []string{"/staged/state.json"}
			return nil
		},
		Publish: func(plan MultiNodePlan) error {
			published = plan
			return nil
		},
	}}
	resume := &training.ResumePoint{Step: 5, Tokens: 40, Checkpoint: training.Checkpoint{Step: 5, Tokens: 40, Artifacts: []training.Artifact{{Path: "artifacts/checkpoints/step-00000005/state.json", SHA256: strings.Repeat("a", 64), Bytes: 1}}}, Paths: []string{"/original/state.json"}}
	runBOM := RunBOM{Initialization: &training.Initialization{}}
	if err := builder.publishMultiNodePlan(RunPin{ID: "run0001"}, runBOM, PreparedStage{}, nil, Stage{Name: "train-0001"}, resume); err != nil {
		t.Fatal(err)
	}
	if published.Resume == nil || published.Resume.Step != 5 || !reflect.DeepEqual(published.ResumePaths, []string{"/staged/state.json"}) {
		t.Fatalf("published resume = %+v, paths %v", published.Resume, published.ResumePaths)
	}
	if published.Initialization != nil || published.InitializationPath != "" {
		t.Fatalf("resumed plan retained superseded initialization: %+v, %q", published.Initialization, published.InitializationPath)
	}
}

func TestComposePublishesPerStagePlans(t *testing.T) {
	root := t.TempDir()
	compose := validCompose()
	compose.Stages = append(compose.Stages, testStage("refine"))
	stages := []PreparedStage{preparedFixture(t, compose.Stages[0]), preparedFixture(t, compose.Stages[1])}
	nextID := 0
	type published struct {
		runID   string
		ordinal int
		count   int
	}
	var seen []published
	builder := Builder{
		Root:      root,
		MultiNode: MultiNodeHandoff{RendezvousID: "compose-42", Nodes: 4},
		NewID: func() (string, error) {
			nextID++
			return fmt.Sprintf("run%04d", nextID), nil
		},
		Resolver: training.FakeResolver(),
		Progress: func(event Progress) {
			if event.Message != "published multi-node plan for secondary nodes" {
				return
			}
			data, err := os.ReadFile(MultiNodePlanPath(root, "compose-42"))
			if err != nil {
				t.Errorf("read plan during publish event: %v", err)
				return
			}
			var plan MultiNodePlan
			if err := json.Unmarshal(data, &plan); err != nil {
				t.Errorf("decode plan during publish event: %v", err)
				return
			}
			seen = append(seen, published{runID: plan.RunID, ordinal: plan.StageOrdinal, count: plan.StageCount})
		},
	}
	if _, err := builder.Compose(context.Background(), "smoke", compose, stages); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 || seen[0].ordinal != 1 || seen[0].count != 2 || seen[1].ordinal != 2 || seen[1].count != 2 {
		t.Fatalf("published plans = %+v", seen)
	}
	if seen[0].runID == seen[1].runID || seen[0].runID == "" {
		t.Fatalf("stage runIDs = %+v", seen)
	}
	if _, err := os.Stat(filepath.Join(root, ".multinode", "compose-42")); !os.IsNotExist(err) {
		t.Fatalf("plan directory must be removed after compose; stat err = %v", err)
	}
}
