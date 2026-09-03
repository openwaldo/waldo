// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/record"
	"github.com/openwaldo/waldo/internal/shard"
	"github.com/openwaldo/waldo/internal/training"
	"github.com/parquet-go/parquet-go"
	"gopkg.in/yaml.v3"
)

type backendFunc func(context.Context, training.Request) (training.Observation, error)

func (backendFunc) Descriptor() training.Descriptor {
	return training.Descriptor{
		Identity: training.Identity{Name: "test", Revision: "test-schema-1"}, Framework: "test",
		Capabilities: training.Capabilities{Objectives: []string{"causal-language-modeling"}, CheckpointResume: true},
	}
}

func (function backendFunc) Run(ctx context.Context, request training.Request) (training.Observation, error) {
	return function(ctx, request)
}

func TestLoadComposeIsStrictAndKeepsIndexPathsLogical(t *testing.T) {
	path := filepath.Join(t.TempDir(), "smoke.yaml")
	if err := os.WriteFile(path, []byte(composeYAML("")), 0o644); err != nil {
		t.Fatal(err)
	}
	compose, loaded, err := LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded != path || !reflect.DeepEqual(CorpusPaths(compose.Stages[0].Corpora), []string{"core/books", "science/papers"}) {
		t.Fatalf("loaded = %q, corpora = %v", loaded, compose.Stages[0].Corpora)
	}
	if err := os.WriteFile(path, []byte(composeYAML("backend:\n  name: fake\n")), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadCompose(path); err == nil || !strings.Contains(err.Error(), "field backend not found") {
		t.Fatalf("LoadCompose backend error = %v", err)
	}
}

func TestLoadComposeExplainsMissingAndEmptyFiles(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing.yaml")
	if _, _, err := LoadCompose(missing); err == nil || !strings.Contains(err.Error(), "model compose") || !strings.Contains(err.Error(), "does not exist") {
		t.Fatalf("missing compose error = %v", err)
	}
	empty := filepath.Join(t.TempDir(), "empty.yaml")
	if err := os.WriteFile(empty, []byte(" \n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadCompose(empty); err == nil || !strings.Contains(err.Error(), "model compose") || !strings.Contains(err.Error(), "is empty") {
		t.Fatalf("empty compose error = %v", err)
	}
}

func TestComposeConversationInteractionIsStrictAndChangesPlanIdentity(t *testing.T) {
	path := filepath.Join(t.TempDir(), "conversation.yaml")
	document := strings.Replace(composeYAML(""), "architecture:\n", "interaction:\n  template: user-assistant-v1\narchitecture:\n", 1)
	if err := os.WriteFile(path, []byte(document), 0o644); err != nil {
		t.Fatal(err)
	}
	compose, _, err := LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	if !compose.Interaction.Conversational() || compose.Interaction.Prompt("", "Hello") != "User: Hello\n\nAssistant:" {
		t.Fatalf("interaction = %+v", compose.Interaction)
	}
	raw := compose
	raw.Interaction = Interaction{}
	conversationPlan, err := composePlan("example", compose)
	if err != nil {
		t.Fatal(err)
	}
	rawPlan, err := composePlan("example", raw)
	if err != nil {
		t.Fatal(err)
	}
	conversationHash, _ := hashJSON(conversationPlan)
	rawHash, _ := hashJSON(rawPlan)
	if conversationHash == rawHash {
		t.Fatal("interaction did not change immutable plan identity")
	}
	encodedRaw, err := json.Marshal(rawPlan)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encodedRaw, []byte(`"interaction"`)) {
		t.Fatalf("raw plan changed its legacy JSON identity: %s", encodedRaw)
	}

	unknown := strings.Replace(document, "template: user-assistant-v1", "unknown: true", 1)
	if err := os.WriteFile(path, []byte(unknown), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadCompose(path); err == nil || !strings.Contains(err.Error(), "field unknown not found") {
		t.Fatalf("interaction unknown-field error = %v", err)
	}
}

func TestComposeDeclaresToolsOnceAtModelInteraction(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tools.yaml")
	document := strings.Replace(composeYAML(""), "architecture:\n", "interaction:\n  template: user-assistant-v1\n  tools: true\narchitecture:\n", 1)
	document = strings.Replace(document, "objective: causal-language-modeling\n", "objective: assistant-response-modeling\n    conversation:\n      template: user-assistant-v1\n      supervised_roles: [assistant]\n", 1)
	if err := os.WriteFile(path, []byte(document), 0o644); err != nil {
		t.Fatal(err)
	}
	compose, _, err := LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	if !compose.Interaction.Tools || compose.Stages[0].Conversation == nil || compose.Stages[0].Conversation.Tools {
		t.Fatalf("tool contract = %+v / %+v", compose.Interaction, compose.Stages[0].Conversation)
	}
	plain := compose
	plain.Interaction.Tools = false
	toolPlan, err := composePlan("tools", compose)
	if err != nil {
		t.Fatal(err)
	}
	plainPlan, err := composePlan("tools", plain)
	if err != nil {
		t.Fatal(err)
	}
	toolHash, _ := hashJSON(toolPlan)
	plainHash, _ := hashJSON(plainPlan)
	if toolHash == plainHash {
		t.Fatal("interaction.tools did not change immutable model identity")
	}

	legacy := strings.Replace(document, "  tools: true\narchitecture:", "architecture:", 1)
	legacy = strings.Replace(legacy, "supervised_roles: [assistant]", "supervised_roles: [assistant]\n      tools: true", 1)
	if err := os.WriteFile(path, []byte(legacy), 0o644); err != nil {
		t.Fatal(err)
	}
	legacyCompose, _, err := LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	if !legacyCompose.Interaction.Tools || legacyCompose.Stages[0].Conversation.Tools {
		t.Fatalf("legacy tool contract was not normalized: %+v / %+v", legacyCompose.Interaction, legacyCompose.Stages[0].Conversation)
	}
	legacyPlan, err := composePlan("tools", legacyCompose)
	if err != nil {
		t.Fatal(err)
	}
	legacyHash, _ := hashJSON(legacyPlan)
	if legacyHash != toolHash {
		t.Fatalf("legacy tool declaration changed model identity: %s != %s", legacyHash, toolHash)
	}
}

func TestInteractionTrimsGeneratedNextTurn(t *testing.T) {
	interaction := Interaction{Template: InteractionUserAssistantV1}
	value := " I can help.\n\nUser: ignored\n\nAssistant: ignored"
	if got := interaction.TrimResponse(value); got != " I can help." {
		t.Fatalf("trimmed response = %q", got)
	}
}

func TestLoadComposeAcceptsPinnedSourceWithInheritedArchitecture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "source.yaml")
	revision := strings.Repeat("a", 40)
	document := "kind: waldo-model-compose\nschema: 1\nbase:\n  source: huggingface://org/model@" + revision + "\n" +
		"stages:\n  - name: pretrain\n    type: pre-training\n    objective: causal-language-modeling\n    corpora: [core/books]\n    parameters:\n      steps: 2\n      batch_size: 1\n      sequence_length: 64\n      learning_rate: 0.001\n      seed: 7\n"
	if err := os.WriteFile(path, []byte(document), 0o644); err != nil {
		t.Fatal(err)
	}
	compose, _, err := LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	if compose.Base == nil || compose.Base.Source == "" || compose.Architecture != (Architecture{}) {
		t.Fatalf("compose = %+v", compose)
	}
	if err := os.WriteFile(path, []byte(strings.Replace(document, "  source:", "  unknown: true\n  source:", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadCompose(path); err == nil || !strings.Contains(err.Error(), "field unknown not found") {
		t.Fatalf("base unknown-field error = %v", err)
	}
}

func TestComposeAcceptsConfiguredCorporaWithoutBreakingScalarForm(t *testing.T) {
	path := filepath.Join(t.TempDir(), "configured.yaml")
	configured := strings.Replace(composeYAML(""), "      - core/books\n      - science/papers", `      - path: core/books
        weight: 2
        filter:
          licenses:
            include: [CC-BY-*]
      - path: science/papers
        weight: 1`, 1)
	configured = strings.Replace(configured, "    corpora:\n", "    filter:\n      languages:\n        include: [en]\n    corpora:\n", 1)
	configured = strings.Replace(configured, "    parameters:\n", "    parameters:\n      profile: causal-pretrain-weighted\n", 1)
	if err := os.WriteFile(path, []byte(configured), 0o644); err != nil {
		t.Fatal(err)
	}
	compose, _, err := LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	stage := compose.Stages[0]
	resolved, err := stage.ResolveParameters()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resolved.Data.CorpusWeights, map[string]uint64{"core/books": 2, "science/papers": 1}) {
		t.Fatalf("inline weights = %v", resolved.Data.CorpusWeights)
	}
	policy, err := stage.RecordFilterPolicy([]string{"core/books.yaml", "science/papers"})
	if err != nil {
		t.Fatal(err)
	}
	if policy.Global == nil || policy.Corpora["core/books.yaml"].Licenses == nil {
		t.Fatalf("resolved filters = %+v", policy)
	}
	encoded, err := yaml.Marshal(compose)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(encoded, []byte("- path: core/books")) || !bytes.Contains(encoded, []byte("- path: science/papers")) {
		t.Fatalf("configured corpus form was not preserved:\n%s", encoded)
	}

	bad := strings.Replace(configured, "include: [CC-BY-*]", "unknown: true", 1)
	if err := os.WriteFile(path, []byte(bad), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadCompose(path); err == nil || !strings.Contains(err.Error(), "field unknown not found") {
		t.Fatalf("nested unknown field error = %v", err)
	}
}

func TestComposeAcceptsUnifiedExclusionFilterStrictly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "exclude.yaml")
	document := strings.Replace(composeYAML(""), "    corpora:\n", "    filter:\n      main_content: true\n      exclude:\n        repetitive_content: true\n        boilerplate_content: true\n        licenses: [CC-BY-NC-*, LicenseRef-Restricted-*]\n    corpora:\n", 1)
	if err := os.WriteFile(path, []byte(document), 0o644); err != nil {
		t.Fatal(err)
	}
	compose, _, err := LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	exclude := compose.Stages[0].Filter.Exclude
	if compose.Stages[0].Filter.MainContent == nil || !*compose.Stages[0].Filter.MainContent || exclude == nil || exclude.RepetitiveContent == nil || !*exclude.RepetitiveContent || exclude.BoilerplateContent == nil || !*exclude.BoilerplateContent || len(exclude.Licenses) != 2 {
		t.Fatalf("exclude filter = %+v", exclude)
	}
	unknown := strings.Replace(document, "repetitive_content: true", "unknown: true", 1)
	if err := os.WriteFile(path, []byte(unknown), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadCompose(path); err == nil || !strings.Contains(err.Error(), "field unknown not found") {
		t.Fatalf("unknown exclusion error = %v", err)
	}
}

func TestCorpusSelectionJSONIsStrictAndAcceptsScalarWhitespace(t *testing.T) {
	var scalar CorpusSelection
	if err := json.Unmarshal([]byte(`  "core/books" `), &scalar); err != nil || scalar.Path != "core/books" {
		t.Fatalf("scalar JSON selection = %+v, err = %v", scalar, err)
	}
	var configured CorpusSelection
	if err := json.Unmarshal([]byte(`{"path":"core/books","unknown":true}`), &configured); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("configured JSON unknown-field error = %v", err)
	}
}

func TestComposeRejectsMixedInlineAndLegacyWeights(t *testing.T) {
	compose := validCompose()
	weight := uint64(2)
	compose.Stages[0].Corpora[0].Weight = &weight
	compose.Stages[0].Parameters.CorpusWeights = map[string]uint64{"example": 2}
	if err := compose.Validate(); err == nil || !strings.Contains(err.Error(), "cannot be combined") {
		t.Fatalf("mixed weight error = %v", err)
	}
}

func TestArchitectureRejectsInvalidDropout(t *testing.T) {
	architecture := testArchitecture()
	architecture.Dropout = 1
	if err := architecture.Validate(); err == nil || !strings.Contains(err.Error(), "dropout") {
		t.Fatalf("invalid dropout error = %v", err)
	}
}

func TestComposeRejectsDuplicateAndMismatchedWeightedCorpora(t *testing.T) {
	compose := validCompose()
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example", "example"})
	if err := compose.Validate(); err == nil || !strings.Contains(err.Error(), "duplicate corpus") {
		t.Fatalf("duplicate corpus error = %v", err)
	}

	compose = validCompose()
	compose.Stages[0].Parameters.Profile = training.WeightedProfile
	compose.Stages[0].Parameters.CorpusWeights = map[string]uint64{"other": 1}
	if err := compose.Validate(); err == nil || !strings.Contains(err.Error(), "does not declare corpus") {
		t.Fatalf("missing corpus weight error = %v", err)
	}

	compose.Stages[0].Parameters.CorpusWeights = map[string]uint64{"example": 1, "other": 1}
	if err := compose.Validate(); err == nil || !strings.Contains(err.Error(), "unselected corpus") {
		t.Fatalf("extra corpus weight error = %v", err)
	}
}

func TestResolveCorpusWeightsUsesLogicalManifestPaths(t *testing.T) {
	resolved, err := resolveCorpusWeights(map[string]uint64{"core/peps": 2, "science/plos": 1}, []string{"core/peps.yaml", "science/plos"})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]uint64{"core/peps.yaml": 2, "science/plos": 1}
	if !reflect.DeepEqual(resolved, want) {
		t.Fatalf("resolved weights = %v, want %v", resolved, want)
	}
}

func TestInitializeAndTrainKeepStableModelIdentity(t *testing.T) {
	root := t.TempDir()
	clock := time.Date(2026, 8, 4, 12, 0, 0, 0, time.UTC)
	builder := Builder{Root: root, Now: func() time.Time { return clock }, NewID: func() (string, error) { return "run0001", nil }, Resolver: training.FakeResolver()}
	initialized, err := builder.Initialize("smoke", testArchitecture())
	if err != nil {
		t.Fatal(err)
	}
	if len(initialized.Model.Runs) != 0 || initialized.Plan.Kind != "waldo-model-plan" {
		t.Fatalf("initialized = %+v", initialized)
	}
	trained, err := builder.Train(context.Background(), "smoke", preparedFixture(t, testStage("pretrain")))
	if err != nil {
		t.Fatal(err)
	}
	if trained.Model.ID != initialized.Model.ID || trained.Model.PlanSHA256 != initialized.Model.PlanSHA256 || len(trained.Model.Runs) != 1 || trained.Model.Runs[0].State != RunComplete {
		t.Fatalf("trained model = %+v", trained.Model)
	}
	if trained.RunBOMs[0].Execution.Backend.Name != "fake" || trained.RunBOMs[0].Execution.Host.OS == "" || trained.Runs[0].Observation == nil || !trained.Runs[0].Observation.Simulated {
		t.Fatalf("run = %+v, BOM = %+v", trained.Runs[0], trained.RunBOMs[0])
	}
	if trained.RunBOMs[0].EvaluationSet == nil || trained.RunBOMs[0].EvaluationSet.Records != 1 || len(trained.Runs[0].Observation.Evaluations) != 1 || trained.Runs[0].Observation.Evaluations[0].Metrics["heldout_loss"] <= 0 {
		t.Fatalf("held-out evidence = %+v / %+v", trained.RunBOMs[0].EvaluationSet, trained.Runs[0].Observation.Evaluations)
	}
	telemetryPath := filepath.Join(trained.Path, "runs", "0001-pretrain-run0001", TelemetryFilename)
	telemetryFile, err := os.Open(telemetryPath)
	if err != nil {
		t.Fatal(err)
	}
	telemetry, err := csv.NewReader(telemetryFile).ReadAll()
	_ = telemetryFile.Close()
	if err != nil || len(telemetry) < 4 || !reflect.DeepEqual(telemetry[0], telemetryHeader) {
		t.Fatalf("telemetry rows = %v, err = %v", telemetry, err)
	}
	last := telemetry[len(telemetry)-1]
	if last[2] != "run0001" || last[5] != "run" || last[6] != string(RunComplete) || last[8] == "" || last[10] == "" {
		t.Fatalf("terminal telemetry = %v", last)
	}
	foundEvaluation := false
	for _, row := range telemetry[1:] {
		if row[5] == "evaluation" && row[12] != "" && row[13] != "" {
			foundEvaluation = true
		}
	}
	if !foundEvaluation {
		t.Fatalf("telemetry has no chartable held-out evaluation: %v", telemetry)
	}
	bomRun := trained.BOM.Runs[0]
	if trained.BOM.PathBase != "model-root" || trained.BOM.CurrentRunID != "" || bomRun.Backend.Name != "fake" || !bomRun.Simulated || bomRun.RunBOM != "runs/0001-pretrain-run0001/RUN-BOM.json" || bomRun.Artifacts[0].Role != "simulation" || bomRun.Artifacts[0].Path != "runs/0001-pretrain-run0001/artifacts/fake-model.json" {
		t.Fatalf("aggregate BOM = %+v", trained.BOM)
	}
	artifact := trained.Model.Runs[0].Artifacts[0]
	data, err := os.ReadFile(filepath.Join(trained.Path, "runs", runDirectoryName(trained.Model.Runs[0]), filepath.FromSlash(artifact.Path)))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "no trained model weights") {
		t.Fatalf("artifact = %q", data)
	}
}

func TestComposeInitializesFromCompletedManagedModelRun(t *testing.T) {
	root := t.TempDir()
	ids := []string{"parentrun", "childrun"}
	calls := 0
	backend := backendFunc(func(_ context.Context, request training.Request) (training.Observation, error) {
		calls++
		if calls == 1 && request.Initialization != nil {
			return training.Observation{}, fmt.Errorf("parent unexpectedly has initialization: %+v", request.Initialization)
		}
		if calls == 2 {
			if request.Initialization == nil || request.Initialization.SourceType != "run" || request.Initialization.SourceRunID != "parentrun" {
				return training.Observation{}, fmt.Errorf("child initialization = %+v", request.Initialization)
			}
		}
		if err := os.MkdirAll(request.ArtifactDirectory, 0o755); err != nil {
			return training.Observation{}, err
		}
		data := []byte("weights-" + request.RunID)
		path := filepath.Join(request.ArtifactDirectory, "model.safetensors")
		if err := os.WriteFile(path, data, 0o644); err != nil {
			return training.Observation{}, err
		}
		digest := sha256.Sum256(data)
		return training.Observation{
			Steps: 2, ConsumedTokens: 128,
			Evaluations: []training.Evaluation{{Step: 2, Tokens: 128, Metrics: map[string]float64{"heldout_loss": 1}}},
			Artifacts:   []training.Artifact{{Path: "artifacts/model.safetensors", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}},
		}, nil
	})
	builder := Builder{
		Root: root,
		NewID: func() (string, error) {
			id := ids[0]
			ids = ids[1:]
			return id, nil
		},
		Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
			return testSelection(backend), nil
		}),
	}
	if _, err := builder.Initialize("conversation", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	parent, err := builder.Train(context.Background(), "conversation", preparedFixture(t, testStage("conversation")))
	if err != nil {
		t.Fatal(err)
	}
	stage := testStage("tool-use")
	compose := Compose{
		Kind: "waldo-model-compose", Schema: 1,
		Base:         &ComposeBase{Model: "conversation"},
		Architecture: testArchitecture(),
		Stages:       []Stage{stage},
	}
	resolved, err := builder.ResolveCompose(context.Background(), compose, false)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Base.ModelID != parent.Model.ID || resolved.Base.RunID != "parentrun" || resolved.Base.RunBOMSHA256 != parent.Model.Runs[0].BOMSHA256 || resolved.Base.ArtifactSHA256 == "" || resolved.Base.ArtifactBytes == 0 {
		t.Fatalf("resolved parent pins = %+v", resolved.Base)
	}
	compose = resolved
	child, err := builder.Compose(context.Background(), "tool-use", compose, []PreparedStage{preparedFixture(t, stage)})
	if err != nil {
		t.Fatal(err)
	}
	if child.Model.Parent == nil || child.Model.Parent.Model != "conversation" || child.Model.Parent.ModelID != parent.Model.ID || child.Model.Parent.RunID != "parentrun" || child.Model.Parent.RunBOMSHA256 != parent.Model.Runs[0].BOMSHA256 {
		t.Fatalf("parent pin = %+v", child.Model.Parent)
	}
	if child.RunBOMs[0].Initialization == nil || child.RunBOMs[0].Initialization.SourceRunID != "parentrun" || child.RunBOMs[0].Initialization.Artifact.Path != "base/model.safetensors" {
		t.Fatalf("child initialization = %+v", child.RunBOMs[0].Initialization)
	}
	if child.BOM.Parent == nil || !reflect.DeepEqual(child.BOM.Parent, child.Model.Parent) {
		t.Fatalf("child BOM parent = %+v", child.BOM.Parent)
	}
	if err := os.RemoveAll(parent.Path); err != nil {
		t.Fatal(err)
	}
	if _, err := Inspect(root, "tool-use"); err != nil {
		t.Fatalf("child depends on parent files after initialization: %v", err)
	}
}

func TestTrainRejectsExhaustedStreamBeforeBackend(t *testing.T) {
	root := t.TempDir()
	called := false
	backend := backendFunc(func(context.Context, training.Request) (training.Observation, error) {
		called = true
		return training.Observation{}, nil
	})
	builder := Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
	if _, err := builder.Initialize("smoke", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	stage := testStage("post-train")
	stage.Type = "fine-tuning"
	stage.Parameters.Epochs = 1
	stage.Parameters.Steps = 100
	_, err := builder.Train(context.Background(), "smoke", preparedFixture(t, stage))
	if err == nil || !strings.Contains(err.Error(), "requests 100 optimizer steps") || !strings.Contains(err.Error(), "provides only") {
		t.Fatalf("capacity error = %v", err)
	}
	if called {
		t.Fatal("training backend ran after capacity preflight failed")
	}
	inspection, err := Inspect(root, "smoke")
	if err != nil {
		t.Fatal(err)
	}
	if len(inspection.Runs) != 0 {
		t.Fatalf("capacity failure persisted %d runs", len(inspection.Runs))
	}
}

func TestTrainingRunPinsConfiguredRecordFilter(t *testing.T) {
	root := t.TempDir()
	builder := Builder{Root: root, NewID: func() (string, error) { return "filtered1", nil }, Resolver: training.FakeResolver()}
	if _, err := builder.Initialize("filtered", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	stage := testStage("pretrain")
	stage.Filter = &corpus.RecordFilter{Licenses: &corpus.ValueFilter{Include: []string{"CC0-*"}}}
	prepared := preparedFixture(t, testStage("pretrain"))
	policy, err := stage.RecordFilterPolicy(prepared.BOM.Paths)
	if err != nil {
		t.Fatal(err)
	}
	prepared.Stage = stage
	prepared.BOM.RecordFilter = policy
	for index := range prepared.Inputs {
		prepared.Inputs[index].RecordFilter = policy
	}
	inspection, err := builder.Train(context.Background(), "filtered", prepared)
	if err != nil {
		t.Fatal(err)
	}
	if len(inspection.RunBOMs) != 1 || !reflect.DeepEqual(inspection.RunBOMs[0].CorpusBOM.RecordFilter, policy) {
		t.Fatalf("run corpus filter = %+v", inspection.RunBOMs)
	}
	runBOMData, err := os.ReadFile(filepath.Join(inspection.Path, "runs", "0001-pretrain-filtered1", "RUN-BOM.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(runBOMData, []byte(`"record_filter"`)) || !bytes.Contains(runBOMData, []byte(`"CC0-*"`)) {
		t.Fatalf("persisted run BOM omits filter: %s", runBOMData)
	}
}

func TestMergeProgressKeepsTerminalArtifactEvaluation(t *testing.T) {
	progress := &training.Progress{Evaluations: []training.Evaluation{{Step: 10, Tokens: 100, Metrics: map[string]float64{"heldout_loss": 2}}}}
	observation := training.Observation{Evaluations: []training.Evaluation{{Step: 10, Tokens: 100, Metrics: map[string]float64{"heldout_loss": 2, "artifact_heldout_loss": 2.001}}}}
	merged := mergeProgress(progress, observation)
	if len(merged.Evaluations) != 1 || merged.Evaluations[0].Metrics["artifact_heldout_loss"] != 2.001 {
		t.Fatalf("merged evaluations = %+v", merged.Evaluations)
	}
}

func TestResumeReplacesEvaluationAtCheckpointStep(t *testing.T) {
	previous := training.Evaluation{Step: 10, Tokens: 100, Metrics: map[string]float64{"heldout_loss": 2}}
	run := RunRecord{
		Progress: &training.Progress{Evaluations: []training.Evaluation{previous}},
		Attempts: []RunAttempt{{Ordinal: 2, State: RunRunning, ResumeStep: 10}},
	}
	replacement := training.Evaluation{Step: 10, Tokens: 100, Metrics: map[string]float64{"heldout_loss": 1.99}}
	changed, err := updateDurableEvaluation(&run, replacement)
	if err != nil || !changed || !reflect.DeepEqual(run.Progress.Evaluations, []training.Evaluation{replacement}) {
		t.Fatalf("updated evaluation = %+v, changed %v, err %v", run.Progress.Evaluations, changed, err)
	}
}

func TestOnlyFinalCheckpointBookkeepingFailureIsResumable(t *testing.T) {
	requested := training.Parameters{Steps: 10, BatchSize: 1, SequenceLength: 10, LearningRate: 0.001, Seed: 1}
	parameters, err := training.ResolveParameters(requested)
	if err != nil {
		t.Fatal(err)
	}
	run := RunRecord{
		State:    RunFailed,
		Error:    "persist training progress: evaluation step 10 does not advance durable progress",
		Progress: &training.Progress{Checkpoints: []training.Checkpoint{{Step: 10, Tokens: 100}}},
	}
	if !resumableRunState(run, parameters) {
		t.Fatal("final-checkpoint evaluation bookkeeping failure is not resumable")
	}
	stage := testStage("pretrain")
	stage.Parameters = requested
	prepared := preparedFixture(t, stage)
	corpusHash, err := hashJSON(prepared.BOM)
	if err != nil {
		t.Fatal(err)
	}
	inspection := Inspection{
		Model:   ModelRecord{Runs: []RunPin{{Stage: stage.Name}}},
		Runs:    []RunRecord{run},
		RunBOMs: []RunBOM{{Stage: stage.Name, StageType: stage.Type, Objective: stage.Objective, CorpusBOMSHA256: corpusHash, Parameters: parameters}},
	}
	if start, ok := recoverableComposeStart(inspection, []PreparedStage{prepared}); !ok || start != 0 {
		t.Fatalf("recoverable compose start = %d, %v", start, ok)
	}
	run.Error = "trainer exited"
	if resumableRunState(run, parameters) {
		t.Fatal("ordinary failed run became resumable")
	}
}

func TestNumberedProfileRemainsResumeCompatible(t *testing.T) {
	current, err := training.ResolveParameters(training.Parameters{Profile: training.WeightedProfile, Steps: 10, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, CorpusWeights: map[string]uint64{"example": 1}})
	if err != nil {
		t.Fatal(err)
	}
	legacy := current
	legacy.Profile = "causal-pretrain-v3"
	legacy.ProfileSchema = 3
	if !equivalentTrainingParameters(legacy, current) {
		t.Fatalf("legacy parameters are not resume-compatible: %+v / %+v", legacy, current)
	}
}

func TestModelBOMIdentifiesLatestRealWeights(t *testing.T) {
	record := ModelRecord{
		ID: "model", Name: "example", PlanSHA256: "plan", ArchitectureSHA256: "architecture", Updated: "now",
		Runs: []RunPin{
			{ID: "fake", Stage: "first", Ordinal: 1, State: RunComplete, Backend: training.Identity{Name: "fake", Revision: "fake-v1"}, Simulated: true, Artifacts: []training.Artifact{{Path: "artifacts/fake-model.json", SHA256: "fake", Bytes: 1}}},
			{ID: "real", Stage: "second", Ordinal: 2, State: RunComplete, Backend: training.Identity{Name: "mlx", Revision: "mlx-v1"}, Artifacts: []training.Artifact{{Path: "artifacts/model.safetensors", SHA256: "weights", Bytes: 2}, {Path: "artifacts/config.json", SHA256: "config", Bytes: 3}}},
		},
	}
	bom := modelBOM(record)
	if bom.CurrentRunID != "real" || bom.Runs[1].Artifacts[0].Role != "weights" || bom.Runs[1].Artifacts[1].Role != "configuration" || !strings.HasPrefix(bom.Runs[1].Artifacts[0].Path, "runs/0002-second-real/") {
		t.Fatalf("BOM = %+v", bom)
	}
}

func TestInspectNormalizesLegacySchemaOneModelBOM(t *testing.T) {
	root := t.TempDir()
	builder := Builder{Root: root, NewID: func() (string, error) { return "legacy", nil }, Resolver: training.FakeResolver()}
	trained, err := builder.Initialize("smoke", testArchitecture())
	if err != nil {
		t.Fatal(err)
	}
	trained, err = builder.Train(context.Background(), "smoke", preparedFixture(t, testStage("pretrain")))
	if err != nil {
		t.Fatal(err)
	}
	record := trained.Model
	record.Runs[0].Backend = training.Identity{}
	record.Runs[0].Simulated = false
	if err := writeJSONAtomic(filepath.Join(trained.Path, "MODEL.json"), record); err != nil {
		t.Fatal(err)
	}
	legacyBOM := map[string]any{
		"kind": "openwaldo-bom", "schema": 1, "subject": "model", "model_id": record.ID,
		"name": record.Name, "plan_sha256": record.PlanSHA256, "architecture_sha256": record.ArchitectureSHA256,
		"runs": []RunPin{record.Runs[0]}, "generated": record.Updated,
	}
	if err := writeJSONAtomic(filepath.Join(trained.Path, "MODEL-BOM.json"), legacyBOM); err != nil {
		t.Fatal(err)
	}
	inspection, err := Inspect(root, "smoke")
	if err != nil {
		t.Fatal(err)
	}
	if inspection.BOM.PathBase != "model-root" || inspection.BOM.Runs[0].Backend.Name != "fake" || !inspection.BOM.Runs[0].Simulated || inspection.BOM.Runs[0].Artifacts[0].Role != "simulation" {
		t.Fatalf("normalized BOM = %+v", inspection.BOM)
	}
}

func TestInspectAcceptsLegacyRunBOMWithoutEpochs(t *testing.T) {
	root := t.TempDir()
	builder := Builder{Root: root, NewID: func() (string, error) { return "legacy-epochs", nil }, Resolver: training.FakeResolver()}
	if _, err := builder.Initialize("smoke", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	trained, err := builder.Train(context.Background(), "smoke", preparedFixture(t, testStage("pretrain")))
	if err != nil {
		t.Fatal(err)
	}
	pin := trained.Model.Runs[0]
	runDirectory := filepath.Join(trained.Path, "runs", runDirectoryName(pin))
	runBOM := trained.RunBOMs[0]
	runBOM.Parameters.Epochs = 0
	legacyHash, err := hashJSON(runBOM)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeJSONAtomic(filepath.Join(runDirectory, "RUN-BOM.json"), runBOM); err != nil {
		t.Fatal(err)
	}
	legacyData, err := os.ReadFile(filepath.Join(runDirectory, "RUN-BOM.json"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(legacyData, []byte(`"epochs"`)) {
		t.Fatalf("legacy run BOM unexpectedly contains epochs: %s", legacyData)
	}
	run := trained.Runs[0]
	run.BOMSHA256 = legacyHash
	if err := writeJSONAtomic(filepath.Join(runDirectory, "RUN.json"), run); err != nil {
		t.Fatal(err)
	}
	record := trained.Model
	record.Runs[0].BOMSHA256 = legacyHash
	if err := writeJSONAtomic(filepath.Join(trained.Path, "MODEL.json"), record); err != nil {
		t.Fatal(err)
	}
	if err := writeJSONAtomic(filepath.Join(trained.Path, "MODEL-BOM.json"), modelBOM(record)); err != nil {
		t.Fatal(err)
	}
	inspection, err := Inspect(root, "smoke")
	if err != nil {
		t.Fatal(err)
	}
	if inspection.RunBOMs[0].Parameters.Epochs != 0 {
		t.Fatalf("legacy epochs = %d, want omitted zero", inspection.RunBOMs[0].Parameters.Epochs)
	}
}

func TestExportRejectsCorruptModelArtifact(t *testing.T) {
	root := t.TempDir()
	builder := Builder{Root: root, NewID: func() (string, error) { return "run0001", nil }, Resolver: training.FakeResolver()}
	if _, err := builder.Initialize("smoke", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	trained, err := builder.Train(context.Background(), "smoke", preparedFixture(t, testStage("pretrain")))
	if err != nil {
		t.Fatal(err)
	}
	pin := trained.Model.Runs[0]
	artifact := filepath.Join(trained.Path, "runs", runDirectoryName(pin), filepath.FromSlash(pin.Artifacts[0].Path))
	if err := os.WriteFile(artifact, []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Export(root, "smoke", filepath.Join(t.TempDir(), "export")); err == nil || !strings.Contains(err.Error(), "verify exported model artifacts") {
		t.Fatalf("Export error = %v", err)
	}
}

func TestTrainRejectsResolverMismatchBeforeAddingRun(t *testing.T) {
	root := t.TempDir()
	builder := Builder{Root: root}
	if _, err := builder.Initialize("smoke", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	builder.Resolver = training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		selection := testSelection(training.Fake{})
		selection.Execution.Framework = "pytorch"
		return selection, nil
	})
	if _, err := builder.Train(context.Background(), "smoke", preparedFixture(t, testStage("pretrain"))); err == nil || !strings.Contains(err.Error(), "does not match backend") {
		t.Fatalf("Train error = %v", err)
	}
	inspection, err := Inspect(root, "smoke")
	if err != nil {
		t.Fatal(err)
	}
	if len(inspection.Model.Runs) != 0 {
		t.Fatalf("runs = %+v", inspection.Model.Runs)
	}
}

func TestTrainPersistsFailureAndInterruption(t *testing.T) {
	for _, test := range []struct {
		name  string
		err   error
		state RunState
	}{{"failed", errors.New("trainer exited"), RunFailed}, {"interrupted", context.Canceled, RunInterrupted}} {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			backend := backendFunc(func(context.Context, training.Request) (training.Observation, error) {
				return training.Observation{}, test.err
			})
			builder := Builder{Root: root, NewID: func() (string, error) { return "run0001", nil }, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
				return testSelection(backend), nil
			})}
			if _, err := builder.Initialize("smoke", testArchitecture()); err != nil {
				t.Fatal(err)
			}
			if _, err := builder.Train(context.Background(), "smoke", preparedFixture(t, testStage("pretrain"))); err == nil {
				t.Fatal("Train succeeded")
			}
			inspection, err := Inspect(root, "smoke")
			if err != nil {
				t.Fatal(err)
			}
			if inspection.Runs[0].State != test.state || inspection.Runs[0].Error != test.err.Error() {
				t.Fatalf("run = %+v", inspection.Runs[0])
			}
		})
	}
}

func TestTrainResumesInterruptedRunFromVerifiedCheckpoint(t *testing.T) {
	root := t.TempDir()
	attempts := 0
	backend := backendFunc(func(_ context.Context, request training.Request) (training.Observation, error) {
		attempts++
		if attempts == 1 {
			path := filepath.Join(request.ArtifactDirectory, "checkpoints", "step-00000001", "state.json")
			data := []byte("{\"kind\":\"waldo-test-checkpoint\",\"schema\":1,\"step\":1}\n")
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return training.Observation{}, err
			}
			if err := os.WriteFile(path, data, 0o644); err != nil {
				return training.Observation{}, err
			}
			digest := sha256.Sum256(data)
			checkpoint := training.Checkpoint{Step: 1, Tokens: 64, Artifacts: []training.Artifact{{Path: "artifacts/checkpoints/step-00000001/state.json", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}}}
			request.Report(training.Event{Kind: "checkpoint", Message: "test checkpoint", Step: 1, Tokens: 64, Checkpoint: &checkpoint})
			return training.Observation{}, context.Canceled
		}
		if request.Resume == nil || request.Resume.Step != 1 || request.Resume.Tokens != 64 || len(request.Resume.Paths) != 1 {
			return training.Observation{}, fmt.Errorf("missing resume point: %+v", request.Resume)
		}
		data := []byte("resumed model weights")
		path := filepath.Join(request.ArtifactDirectory, "model.safetensors")
		if err := os.WriteFile(path, data, 0o644); err != nil {
			return training.Observation{}, err
		}
		digest := sha256.Sum256(data)
		loss := 0.5
		return training.Observation{Steps: 2, ConsumedTokens: 128, FinalLoss: &loss, Evaluations: []training.Evaluation{{Step: 2, Tokens: 128, Metrics: map[string]float64{"heldout_loss": 0.75, "heldout_perplexity": 2.117}}}, Artifacts: []training.Artifact{{Path: "artifacts/model.safetensors", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}}}, nil
	})
	ids := 0
	builder := Builder{Root: root, NewID: func() (string, error) { ids++; return "resume0001", nil }, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
	if _, err := builder.Initialize("smoke", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	stage := preparedFixture(t, testStage("train-0001"))
	if _, err := builder.Train(context.Background(), "smoke", stage); !errors.Is(err, context.Canceled) {
		t.Fatalf("first Train error = %v", err)
	}
	interrupted, err := Inspect(root, "smoke")
	if err != nil {
		t.Fatal(err)
	}
	if len(interrupted.Model.Runs) != 1 || interrupted.Runs[0].State != RunInterrupted || interrupted.Model.Runs[0].Resume == nil || len(interrupted.Runs[0].Attempts) != 1 {
		t.Fatalf("interrupted run = %+v / %+v", interrupted.Model.Runs, interrupted.Runs)
	}
	completed, err := builder.Train(context.Background(), "smoke", stage)
	if err != nil {
		t.Fatal(err)
	}
	if ids != 1 || attempts != 2 || len(completed.Model.Runs) != 1 || completed.Runs[0].State != RunComplete || len(completed.Runs[0].Attempts) != 2 || completed.Model.Runs[0].Resume != nil {
		t.Fatalf("completed run = ids %d, attempts %d, model %+v, run %+v", ids, attempts, completed.Model.Runs, completed.Runs[0])
	}
	if len(completed.Runs[0].Observation.Checkpoints) != 1 || completed.Runs[0].Observation.Checkpoints[0].Step != 1 {
		t.Fatalf("completed observation = %+v", completed.Runs[0].Observation)
	}
}

func TestTrainDerivesAndPersistsEpochSteps(t *testing.T) {
	root := t.TempDir()
	builder := Builder{Root: root, Resolver: training.FakeResolver()}
	if _, err := builder.Initialize("epoch-model", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	stage := testStage("epoch-train")
	stage.Parameters.Steps = 0
	stage.Parameters.Epochs = 2
	result, err := builder.Train(context.Background(), "epoch-model", preparedFixture(t, stage))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.RunBOMs) != 1 || result.RunBOMs[0].Parameters.Steps <= 0 || result.RunBOMs[0].Parameters.Epochs != 2 {
		t.Fatalf("epoch-derived run BOM = %+v", result.RunBOMs)
	}
}

func TestComposeAppendsToCompatibleModelAndRejectsDifferentArchitecture(t *testing.T) {
	root := t.TempDir()
	compose := validCompose()
	compose.Interaction = Interaction{Template: InteractionUserAssistantV1}
	stage := preparedFixture(t, compose.Stages[0])
	nextID := 0
	builder := Builder{Root: root, NewID: func() (string, error) {
		nextID++
		return fmt.Sprintf("run%04d", nextID), nil
	}, Resolver: training.FakeResolver()}
	first, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{stage})
	if err != nil {
		t.Fatal(err)
	}
	second, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{stage})
	if err != nil {
		t.Fatal(err)
	}
	if second.Model.ID != first.Model.ID || len(second.Runs) != 2 || second.Runs[1].State != RunComplete || second.Plan.Interaction != compose.Interaction || second.Model.Interaction != compose.Interaction || second.BOM.Interaction != compose.Interaction {
		t.Fatalf("compatible compose did not append: first = %+v, second = %+v", first.Model, second.Model)
	}
	differentInteraction := compose
	differentInteraction.Interaction = Interaction{}
	if _, err := builder.Compose(context.Background(), "smoke", differentInteraction, []PreparedStage{stage}); err == nil || !strings.Contains(err.Error(), "interaction template") {
		t.Fatalf("different interaction error = %v", err)
	}
	different := compose
	different.Architecture.Layers++
	if _, err := builder.Compose(context.Background(), "smoke", different, []PreparedStage{stage}); err == nil || !strings.Contains(err.Error(), "use a new model name") {
		t.Fatalf("different architecture error = %v", err)
	}
}

func TestFailedNewComposeRemainsListed(t *testing.T) {
	root := t.TempDir()
	compose := validCompose()
	stage := preparedFixture(t, compose.Stages[0])
	backend := backendFunc(func(context.Context, training.Request) (training.Observation, error) {
		return training.Observation{}, errors.New("trainer failed")
	})
	builder := Builder{Root: root, NewID: func() (string, error) { return "failed0001", nil }, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
	if _, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{stage}); err == nil {
		t.Fatal("failed compose succeeded")
	}
	inspection, err := Inspect(root, "smoke")
	if err != nil || len(inspection.Runs) != 1 || inspection.Runs[0].State != RunFailed {
		t.Fatalf("failed compose model = %+v, err = %v", inspection, err)
	}
	listed, err := List(root, nil)
	if err != nil || len(listed) != 1 || listed[0].Name != "smoke" || listed[0].State != string(RunFailed) {
		t.Fatalf("failed compose listing = %+v, err = %v", listed, err)
	}
}

func TestComposeResumesDurableTransactionAfterInterruption(t *testing.T) {
	root := t.TempDir()
	compose := validCompose()
	stage := preparedFixture(t, compose.Stages[0])
	attempts := 0
	backend := backendFunc(func(_ context.Context, request training.Request) (training.Observation, error) {
		attempts++
		if attempts == 1 {
			return training.Observation{}, context.Canceled
		}
		data := []byte("resumed compose weights")
		path := filepath.Join(request.ArtifactDirectory, "model.safetensors")
		if err := os.MkdirAll(request.ArtifactDirectory, 0o755); err != nil {
			return training.Observation{}, err
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			return training.Observation{}, err
		}
		digest := sha256.Sum256(data)
		loss := 0.5
		return training.Observation{
			Steps: 2, ConsumedTokens: 128, FinalLoss: &loss,
			Evaluations: []training.Evaluation{{Step: 2, Tokens: 128, Metrics: map[string]float64{"heldout_loss": 0.75}}},
			Artifacts:   []training.Artifact{{Path: "artifacts/model.safetensors", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}},
		}, nil
	})
	ids := 0
	builder := Builder{Root: root, ComposeName: "babble.yaml", NewID: func() (string, error) {
		ids++
		return "compose0001", nil
	}, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
	if _, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{stage}); !errors.Is(err, context.Canceled) {
		t.Fatalf("first Compose error = %v", err)
	}
	if pending, err := HasPendingCompose(root, "smoke"); err != nil || !pending {
		t.Fatalf("pending compose = %v, err = %v", pending, err)
	}
	latest, err := LatestComposePath(filepath.Join(root, "smoke"))
	if err != nil || filepath.Base(latest) != "0000-babble.yaml" {
		t.Fatalf("latest compose = %q, err = %v", latest, err)
	}
	interrupted, err := Inspect(root, "smoke")
	if err != nil || len(interrupted.Runs) != 1 || interrupted.Runs[0].State != RunInterrupted {
		t.Fatalf("interrupted compose model = %+v, err = %v", interrupted, err)
	}
	entries, err := os.ReadDir(filepath.Join(root, ".waldo-compose"))
	if err != nil || len(entries) < 2 {
		t.Fatalf("durable staging entries = %v, err = %v", entries, err)
	}
	listed, err := List(root, nil)
	if err != nil || len(listed) != 1 || listed[0].Name != "smoke" || listed[0].State != string(RunInterrupted) || listed[0].Path != filepath.Join(root, "smoke") {
		t.Fatalf("active compose listing = %+v, err = %v", listed, err)
	}
	completed, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{stage})
	if err != nil {
		t.Fatal(err)
	}
	if pending, err := HasPendingCompose(root, "smoke"); err != nil || pending {
		t.Fatalf("completed pending compose = %v, err = %v", pending, err)
	}
	if attempts != 2 || ids != 1 || len(completed.Runs) != 1 || completed.Runs[0].State != RunComplete || len(completed.Runs[0].Attempts) != 2 {
		t.Fatalf("completed compose: attempts %d ids %d run %+v", attempts, ids, completed.Runs)
	}
	entries, err = os.ReadDir(filepath.Join(root, ".waldo-compose"))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			t.Fatalf("completed compose retained workspace %s", entry.Name())
		}
	}
}

func TestComposeResumesWhenAppendingToExistingModel(t *testing.T) {
	root := t.TempDir()
	compose := validCompose()
	stage := preparedFixture(t, compose.Stages[0])
	ids := 0
	builder := Builder{Root: root, NewID: func() (string, error) {
		ids++
		return fmt.Sprintf("append%04d", ids), nil
	}, Resolver: training.FakeResolver()}
	if _, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{stage}); err != nil {
		t.Fatal(err)
	}
	attempts := 0
	backend := backendFunc(func(ctx context.Context, request training.Request) (training.Observation, error) {
		attempts++
		if attempts == 1 {
			return training.Observation{}, context.Canceled
		}
		observation, err := (training.Fake{}).Run(ctx, request)
		observation.Simulated = false
		return observation, err
	})
	builder.Resolver = training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})
	if _, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{stage}); !errors.Is(err, context.Canceled) {
		t.Fatalf("append Compose error = %v", err)
	}
	if pending, err := HasPendingCompose(root, "smoke"); err != nil || !pending {
		t.Fatalf("pending append = %v, err = %v", pending, err)
	}
	if err := builder.CheckComposeTarget("smoke", compose); err != nil {
		t.Fatalf("matching pending compose target = %v", err)
	}
	different := compose
	different.Stages = append([]Stage(nil), compose.Stages...)
	different.Stages[0].Parameters.Steps++
	if err := builder.CheckComposeTarget("smoke", different); err == nil || !strings.Contains(err.Error(), "repeat the exact command") {
		t.Fatalf("different pending compose target = %v", err)
	}
	completed, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{stage})
	if err != nil {
		t.Fatal(err)
	}
	if attempts != 2 || ids != 2 || len(completed.Runs) != 2 || completed.Runs[0].State != RunComplete || completed.Runs[1].State != RunComplete || len(completed.Runs[1].Attempts) != 2 {
		t.Fatalf("completed append: attempts %d ids %d runs %+v", attempts, ids, completed.Runs)
	}
}

func TestComposeHistoryOrdersDistinctRecipes(t *testing.T) {
	modelPath := t.TempDir()
	first := validCompose()
	path, err := ArchiveCompose(modelPath, first, "first compose.yaml")
	if err != nil || filepath.Base(path) != "0000-first-compose.yaml" {
		t.Fatalf("first archive = %q, err = %v", path, err)
	}
	if duplicate, err := ArchiveCompose(modelPath, first, "renamed.yaml"); err != nil || duplicate != path {
		t.Fatalf("duplicate archive = %q, err = %v", duplicate, err)
	}
	second := validCompose()
	second.Stages[0].Parameters.Steps++
	path, err = ArchiveCompose(modelPath, second, "second.json")
	if err != nil || filepath.Base(path) != "0001-second.yaml" {
		t.Fatalf("second archive = %q, err = %v", path, err)
	}
	latest, err := LatestComposePath(modelPath)
	if err != nil || latest != path {
		t.Fatalf("latest archive = %q, err = %v", latest, err)
	}
	third := validCompose()
	third.Stages[0].Parameters.Steps += 2
	path, err = ArchiveCompose(modelPath, third, "0042-third.yaml")
	if err != nil || filepath.Base(path) != "0002-third.yaml" {
		t.Fatalf("prefixed source archive = %q, err = %v", path, err)
	}
}

func TestComposeCancellationDuringPreflightRemainsListed(t *testing.T) {
	root := t.TempDir()
	compose := validCompose()
	stage := preparedFixture(t, compose.Stages[0])
	builder := Builder{Root: root, NewID: func() (string, error) { return "preflight0001", nil }, Resolver: training.FakeResolver()}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := builder.Compose(ctx, "smoke", compose, []PreparedStage{stage}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Compose error = %v", err)
	}
	listed, err := List(root, nil)
	if err != nil || len(listed) != 1 || listed[0].Name != "smoke" || listed[0].Path != filepath.Join(root, "smoke") {
		t.Fatalf("preflight-canceled compose listing = %+v, err = %v", listed, err)
	}
	inspection, err := Inspect(root, "smoke")
	if err != nil || len(inspection.Runs) != 0 {
		t.Fatalf("preflight-canceled model = %+v, err = %v", inspection, err)
	}
	completed, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{stage})
	if err != nil || len(completed.Runs) != 1 || completed.Runs[0].State != RunComplete {
		t.Fatalf("resumed preflight compose = %+v, err = %v", completed, err)
	}
}

func TestListExportAndRemoveModels(t *testing.T) {
	root := t.TempDir()
	builder := Builder{Root: root}
	if _, err := builder.Initialize("alpha", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Initialize("beta", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	listed, err := List(root, []string{"a*"})
	if err != nil || len(listed) != 1 || listed[0].Name != "alpha" {
		t.Fatalf("listed = %+v, err = %v", listed, err)
	}
	destination := filepath.Join(t.TempDir(), "alpha-export")
	if _, err := Export(root, "alpha", destination); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(destination, "BOM.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(destination, "MODEL-BOM.json")); !os.IsNotExist(err) {
		t.Fatalf("native export retained internal BOM name: %v", err)
	}
	if _, err := Export(root, "alpha", filepath.Join(root, "alpha", "recursive-export")); err == nil {
		t.Fatal("Export accepted a destination inside its source model")
	}
	if _, err := Inspect(t.TempDir(), destination); err != nil {
		t.Fatal(err)
	}
	if _, err := Remove(root, []string{"alpha", "beta"}); err != nil {
		t.Fatal(err)
	}
	listed, err = List(root, nil)
	if err != nil || len(listed) != 0 {
		t.Fatalf("listed = %+v, err = %v", listed, err)
	}
}

func validCompose() Compose {
	return Compose{Kind: "waldo-model-compose", Schema: 1, Architecture: testArchitecture(), Stages: []Stage{testStage("pretrain")}}
}

func testArchitecture() Architecture {
	return Architecture{Family: "decoder-transformer", ContextTokens: 128, VocabularySize: 256, HiddenSize: 64, IntermediateSize: 192, Layers: 2, AttentionHeads: 4, KeyValueHeads: 2, TieEmbeddings: true, ParameterDType: "float32", Tokenizer: Tokenizer{Name: "byte", Revision: "sha256:example"}}
}

func testStage(name string) Stage {
	return Stage{Name: name, Type: "pre-training", Objective: "causal-language-modeling", Corpora: NewCorpusSelections([]string{"example"}), Parameters: training.Parameters{Steps: 2, BatchSize: 1, SequenceLength: 64, LearningRate: 0.001, Seed: 7}}
}

func preparedFixture(t *testing.T, stage Stage) PreparedStage {
	t.Helper()
	text := strings.Repeat("canonical parquet fixture ", 8)
	var encoded bytes.Buffer
	writer := parquet.NewGenericWriter[shard.Row](&encoded)
	secondText := text + " second"
	if _, err := writer.Write([]shard.Row{
		{SHA256: record.TextHash(text), Kind: record.KindPretrain, Text: text, Source: "fixture", License: "CC0-1.0", Tokens: 128},
		{SHA256: record.TextHash(secondText), Kind: record.KindPretrain, Text: secondText, Source: "fixture", License: "CC0-1.0", Tokens: 128},
	}); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	data := encoded.Bytes()
	digestArray := sha256.Sum256(data)
	digest := hex.EncodeToString(digestArray[:])
	path := filepath.Join(t.TempDir(), digest)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	conversion := index.Conversion{Tool: "test", Version: "1", Profile: "text", Recipe: "test/v1", Tokenizer: "byte"}
	measures := index.Measures{Shards: 1, Docs: 2, Tokens: 256, Bytes: int64(len(data))}
	bom := corpus.BOM{
		Kind: "openwaldo-bom", Schema: 1, Subject: "corpus", Paths: []string{"example"}, Licenses: map[string]index.Measures{"CC0-1.0": measures}, Totals: measures,
		Manifests: []corpus.ManifestPin{{Path: "example/example.json", SHA256: strings.Repeat("a", 64), Name: "example", Title: "Example", Description: "Model fixture.", License: "CC0-1.0", Format: "parquet", RecordSchema: 1, ConvertedBy: conversion, Sources: []index.Source{{Name: "fixture", Source: "Fixture", URL: "https://example.test", SHA256: strings.Repeat("b", 64)}}, Totals: measures, Licenses: map[string]index.Measures{"CC0-1.0": measures}}},
		Shards:    []corpus.ShardPin{{Manifest: "example/example.json", URL: "https://objects.example/" + digest, SHA256: digest, Format: "parquet", RecordSchema: 1, License: "CC0-1.0", ConvertedBy: conversion, Docs: 2, Tokens: 256, Bytes: int64(len(data))}},
	}
	prepared, err := PrepareStage(stage, bom, []training.Input{{Path: path, SHA256: digest, Bytes: int64(len(data)), Records: 2}})
	if err != nil {
		t.Fatal(err)
	}
	return prepared
}

func composeYAML(suffix string) string {
	return "kind: waldo-model-compose\nschema: 1\n" +
		"architecture:\n  family: decoder-transformer\n  context_tokens: 128\n  vocabulary_size: 256\n  hidden_size: 64\n  intermediate_size: 192\n  layers: 2\n  attention_heads: 4\n  key_value_heads: 2\n  tie_embeddings: true\n  parameter_dtype: float32\n  tokenizer:\n    name: byte\n    revision: sha256:example\n" +
		"stages:\n  - name: pretrain\n    type: pre-training\n    objective: causal-language-modeling\n    corpora:\n      - core/books\n      - science/papers\n    parameters:\n      steps: 2\n      batch_size: 1\n      sequence_length: 64\n      learning_rate: 0.001\n      seed: 7\n" + suffix
}

func testSelection(backend training.Backend) training.Selection {
	descriptor := backend.Descriptor()
	return training.Selection{Backend: backend, Execution: training.Execution{Backend: descriptor.Identity, Framework: descriptor.Framework, Runtime: "test", Host: training.Host{OS: "test-os", Architecture: "test-arch"}, Nodes: 1, WorldSize: 1}}
}
