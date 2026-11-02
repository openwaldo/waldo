// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

// pullRelease pulls a release whose BOM records a lineage node descending
// from parent; salt changes one weight byte so releases hold different weights.
func pullRelease(t *testing.T, root, name, parent string, salt byte) {
	t.Helper()
	pullReleaseOf(t, root, name, parent, salt, mergeArchitecture())
}

func mergeArchitecture() Architecture {
	return Architecture{Family: "decoder-transformer", ContextTokens: 32, VocabularySize: 259, HiddenSize: 4, IntermediateSize: 8, Layers: 1, AttentionHeads: 2, KeyValueHeads: 1, TieEmbeddings: true, ParameterDType: "bfloat16", Tokenizer: Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"}}
}

func pullReleaseOf(t *testing.T, root, name, parent string, salt byte, architecture Architecture) {
	t.Helper()
	files := huggingFaceFixture(t, architecture, HuggingFaceTokenizerClass)
	weights := files["model.safetensors"]
	// Keep each bf16 value below 2, so averaging them stays finite.
	for index := 8 + int(binary.LittleEndian.Uint64(weights)) + 1; index < len(weights); index += 2 {
		weights[index] &= 0x3f
	}
	weights[len(weights)-1] ^= salt
	node := LineageNode{WeightsSHA256: strings.Repeat(string("0123456789"[salt]), 64), Parents: []LineageLink{{WeightsSHA256: parent}}}
	bom, err := json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": node})
	if err != nil {
		t.Fatal(err)
	}
	files["BOM.json"] = bom
	client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("a", 40))}
	if _, err := (Puller{Root: root, Endpoint: "https://hub.test", Client: client}).Pull(t.Context(), name, "huggingface://org/tiny"); err != nil {
		t.Fatal(err)
	}
}

func TestMergeComposeRecordsLineageAndFailsClosed(t *testing.T) {
	common, other := strings.Repeat("a", 64), strings.Repeat("b", 64)
	root := t.TempDir()
	isolateHuggingFaceToken(t)
	pullRelease(t, root, "left", common, 0)
	pullRelease(t, root, "right", common, 1)
	pullRelease(t, root, "stranger", other, 2)
	builder := Builder{Root: root}
	merged, err := builder.Compose(context.Background(), "merged", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left", Weight: 1}, {Model: "right", Weight: 3}}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	origin := merged.Origin
	if origin == nil || origin.Source.Provider != "merge" || origin.Merge == nil || origin.Merge.Method != "weighted-mean-f32-v1" || len(origin.Merge.Parents) != 2 || origin.Merge.Parents[1].Weight != 3 {
		t.Fatalf("origin = %+v", origin)
	}
	// Only the releases say they share a parent, so the merge records that it rests on their word.
	if !origin.Merge.DeclaredAncestry {
		t.Fatal("ancestry shown only by release BOMs was not marked declared")
	}
	// Like a git merge commit, the merged node names each parent's node and nothing older.
	if origin.Lineage == nil || len(origin.Lineage.Parents) != 2 || origin.Lineage.Parents[0].LineageSHA256 != origin.Merge.Parents[0].LineageSHA256 || origin.Merge.Parents[0].LineageSHA256 == "" {
		t.Fatalf("lineage = %+v", origin.Lineage)
	}
	for _, artifact := range origin.Artifacts {
		if artifact.Role == "weights" && artifact.SHA256 != origin.Lineage.WeightsSHA256 {
			t.Fatalf("merged node weights = %s, artifact %s", origin.Lineage.WeightsSHA256, artifact.SHA256)
		}
	}
	// The common ancestor is two links up the chain, through each release's own node.
	index, _ := builder.lineageIndex()
	if reached := ancestry(*origin.Lineage, index, nil); !reached[common] || reached[other] {
		t.Fatalf("ancestry = %v", reached)
	}
	if merged.BOM.CurrentOriginSHA256 == "" || len(merged.Runs) != 0 {
		t.Fatalf("merged model is not usable from its origin: %+v", merged.BOM)
	}
	if state := listingFromInspection(merged).State; state != "merged" {
		t.Fatalf("merged model lists as %q", state)
	}
	// A release of the merge records the merged node, so the chain continues past publication.
	lineage, err := ModelLineage(merged, "")
	if err != nil || !reflect.DeepEqual(lineage, *origin.Lineage) {
		t.Fatalf("export lineage = %+v, %v", lineage, err)
	}
	_, err = builder.Compose(context.Background(), "unrelated", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "stranger"}}}}, nil)
	if err == nil || !strings.Contains(err.Error(), "share no recorded ancestor") {
		t.Fatalf("unrelated merge error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "unrelated")); !os.IsNotExist(err) {
		t.Fatalf("unrelated merge left a model: %v", err)
	}
}

// Anyone holding the same parents gets the same merged weights, in their own store or again in one.
func TestMergeComposeIsReproducible(t *testing.T) {
	isolateHuggingFaceToken(t)
	var weights []string
	for range 2 {
		root := t.TempDir()
		pullRelease(t, root, "left", strings.Repeat("a", 64), 0)
		pullRelease(t, root, "right", strings.Repeat("a", 64), 1)
		builder := Builder{Root: root}
		for _, name := range []string{"merged", "again"} {
			merged, err := builder.Compose(t.Context(), name, Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}, nil)
			if err != nil {
				t.Fatal(err)
			}
			weights = append(weights, merged.Origin.Lineage.WeightsSHA256)
			// Drop the cached merge so the second one is computed again, not reused.
			if err := os.RemoveAll(filepath.Join(root, ".merges")); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, hash := range weights {
		if hash != weights[0] {
			t.Fatalf("merged weights differ: %v", weights)
		}
	}
}

func TestMergeComposeRefusesDifferentArchitectures(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	wider := mergeArchitecture()
	wider.HiddenSize, wider.IntermediateSize = 8, 16
	pullRelease(t, root, "left", strings.Repeat("a", 64), 0)
	pullReleaseOf(t, root, "wider", strings.Repeat("a", 64), 1, wider)
	_, err := (Builder{Root: root}).Compose(t.Context(), "merged", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "wider"}}}}, nil)
	if err == nil || !strings.Contains(err.Error(), "does not have the architecture of") {
		t.Fatalf("mismatched merge error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "merged")); !os.IsNotExist(err) {
		t.Fatalf("mismatched merge left a model: %v", err)
	}
}

// With stages, the merge is the origin training starts from.
func TestMergeComposeTrainsFromTheMerge(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullRelease(t, root, "left", strings.Repeat("a", 64), 0)
	pullRelease(t, root, "right", strings.Repeat("a", 64), 1)
	stage := testStage("continue")
	stage.Parameters.SequenceLength = 32
	compose := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}, Architecture: mergeArchitecture(), Stages: []Stage{stage}}
	trained, err := (Builder{Root: root, Resolver: training.FakeResolver()}).Compose(t.Context(), "trained", compose, []PreparedStage{preparedFixture(t, stage)})
	if err != nil {
		t.Fatal(err)
	}
	if trained.Origin == nil || trained.Origin.Merge == nil || len(trained.Runs) != 1 || trained.Plan.OriginBOMSHA256 != trained.Model.OriginBOMSHA256 {
		t.Fatalf("trained = %+v", trained)
	}
}

// A release can claim any parent. The merge goes ahead, but its record says the
// shared ancestor is declared, never that this store showed it; a model trained
// here from the same parent shares a recorded one.
func TestMergeComposeMarksAncestryOnlyAReleaseDeclares(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullRelease(t, root, "victim", strings.Repeat("a", 64), 0)
	victim, err := Inspect(root, "victim")
	if err != nil {
		t.Fatal(err)
	}
	node, err := ModelLineage(victim, "")
	if err != nil {
		t.Fatal(err)
	}
	pullRelease(t, root, "claimant", node.WeightsSHA256, 3)
	// Training flips one bit of its initial weights, so "trained" holds weights of its own.
	backend := backendFunc(func(_ context.Context, request training.Request) (training.Observation, error) {
		data, err := os.ReadFile(request.Initialization.Path)
		if err != nil {
			return training.Observation{}, err
		}
		data[len(data)-1] ^= 1
		if err := os.MkdirAll(request.ArtifactDirectory, 0o755); err != nil {
			return training.Observation{}, err
		}
		if err := os.WriteFile(filepath.Join(request.ArtifactDirectory, "model.safetensors"), data, 0o644); err != nil {
			return training.Observation{}, err
		}
		digest := sha256.Sum256(data)
		return training.Observation{
			Steps: 2, ConsumedTokens: 128,
			Evaluations: []training.Evaluation{{Step: 2, Tokens: 128, Metrics: map[string]float64{"heldout_loss": 1}}},
			Artifacts:   []training.Artifact{{Path: "artifacts/model.safetensors", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}},
		}, nil
	})
	builder := Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
	stage := testStage("continue")
	stage.Parameters.SequenceLength = 32
	stage.Parameters.Steps *= 2
	if _, err := builder.Compose(t.Context(), "trained", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "victim"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	for parent, declared := range map[string]bool{"claimant": true, "trained": false} {
		merged, err := builder.Compose(t.Context(), "with-"+parent, Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "victim"}, {Model: parent}}}}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if merged.Origin.Merge.DeclaredAncestry != declared {
			t.Fatalf("%s: declared ancestry = %v, want %v", parent, merged.Origin.Merge.DeclaredAncestry, declared)
		}
	}
}

// realWeightsBuilder trains with a backend that writes distinct, non-simulated
// weights for every run, so each run has its own lineage node.
func realWeightsBuilder(root string) Builder {
	ids := 0
	backend := backendFunc(func(_ context.Context, request training.Request) (training.Observation, error) {
		if err := os.MkdirAll(request.ArtifactDirectory, 0o755); err != nil {
			return training.Observation{}, err
		}
		data := []byte("weights-" + request.RunID)
		if err := os.WriteFile(filepath.Join(request.ArtifactDirectory, "model.safetensors"), data, 0o644); err != nil {
			return training.Observation{}, err
		}
		digest := sha256.Sum256(data)
		return training.Observation{
			Steps: 2, ConsumedTokens: 128,
			Evaluations: []training.Evaluation{{Step: 2, Tokens: 128, Metrics: map[string]float64{"heldout_loss": 1}}},
			Artifacts:   []training.Artifact{{Path: "artifacts/model.safetensors", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}},
		}, nil
	})
	return Builder{Root: root, NewID: func() (string, error) { ids++; return fmt.Sprintf("run%04d", ids), nil },
		Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
			return testSelection(backend), nil
		})}
}

// trainTwice initializes model p and trains it twice, returning its first run's ID.
func trainTwice(t *testing.T, builder Builder) string {
	t.Helper()
	if _, err := builder.Initialize("p", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	p, err := builder.Train(t.Context(), "p", preparedFixture(t, testStage("one")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Train(t.Context(), "p", preparedFixture(t, testStage("two"))); err != nil {
		t.Fatal(err)
	}
	return p.Model.Runs[0].ID
}

// Pinning an earlier run records that run's node: its weights and only the run that made them.
func TestComposePinnedToEarlierRunRecordsThatRunsLineage(t *testing.T) {
	builder := realWeightsBuilder(t.TempDir())
	first := trainTwice(t, builder)
	resolved, err := builder.ResolveCompose(t.Context(), Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p", RunID: first}, Stages: []Stage{testStage("child")}}, false)
	if err != nil {
		t.Fatal(err)
	}
	index, _ := builder.lineageIndex()
	node, ok := index[resolved.Base.LineageSHA256]
	if !ok || node.WeightsSHA256 != resolved.Base.ArtifactSHA256 || len(node.Runs) != 1 || node.Runs[0] != resolved.Base.RunBOMSHA256 {
		t.Fatalf("pin weights %s, run %s; lineage node %+v", resolved.Base.ArtifactSHA256, resolved.Base.RunBOMSHA256, node)
	}
}

// A run's node never changes, so a child pinned to it keeps accepting stages after its parent trains on.
func TestChildAcceptsStagesAfterItsParentTrainsOn(t *testing.T) {
	builder := realWeightsBuilder(t.TempDir())
	if _, err := builder.Initialize("p", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	p, err := builder.Train(t.Context(), "p", preparedFixture(t, testStage("one")))
	if err != nil {
		t.Fatal(err)
	}
	stage := testStage("child")
	compose := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p", RunID: p.Model.Runs[0].ID}, Stages: []Stage{stage}}
	if _, err := builder.Compose(t.Context(), "child", compose, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Train(t.Context(), "p", preparedFixture(t, testStage("two"))); err != nil {
		t.Fatal(err)
	}
	compose.Stages = []Stage{stage, testStage("more")}
	if _, err := builder.Compose(t.Context(), "child", compose, []PreparedStage{preparedFixture(t, stage), preparedFixture(t, testStage("more"))}); err != nil {
		t.Fatal(err)
	}
}

// A child of p's first run and p after its second run meet at the first run's weights.
func TestMergeFindsAncestryThroughEarlierRuns(t *testing.T) {
	builder := realWeightsBuilder(t.TempDir())
	if _, err := builder.Initialize("p", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Train(t.Context(), "p", preparedFixture(t, testStage("one"))); err != nil {
		t.Fatal(err)
	}
	stage := testStage("child")
	if _, err := builder.Compose(t.Context(), "child", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Train(t.Context(), "p", preparedFixture(t, testStage("two"))); err != nil {
		t.Fatal(err)
	}
	if _, declared, err := builder.mergeParents(ComposeBase{Merge: []ComposeMergeParent{{Model: "child"}, {Model: "p"}}}); err != nil || declared {
		t.Fatalf("merge of child and p: declared %v, %v", declared, err)
	}
}

// A run its pin marks simulated, as a legacy fake-backend run is, has no
// lineage node and its weights are never current, whatever its observation says.
func TestLineageAndCurrentWeightsSkipRunsThePinMarksSimulated(t *testing.T) {
	root := t.TempDir()
	trainTwice(t, realWeightsBuilder(root))
	p, err := Inspect(root, "p")
	if err != nil {
		t.Fatal(err)
	}
	p.Model.Runs[1].Simulated = true
	first := p.Model.Runs[0].Artifacts[0].SHA256
	node, err := ModelLineage(p, "")
	if err != nil {
		t.Fatal(err)
	}
	_, current, err := currentWeights(p)
	if err != nil {
		t.Fatal(err)
	}
	if node.WeightsSHA256 != first || current != first {
		t.Fatalf("lineage %s and current %s, want the first run's %s", node.WeightsSHA256, current, first)
	}
}

// A pulled release that trains on keeps the ancestry its release recorded.
func TestMergeFindsAncestryOfAPulledReleaseThatTrained(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullRelease(t, root, "left", strings.Repeat("a", 64), 0)
	pullRelease(t, root, "right", strings.Repeat("a", 64), 1)
	builder := realWeightsBuilder(root)
	stage := testStage("continue")
	stage.Parameters.SequenceLength = 32
	stage.Parameters.Steps *= 2
	if _, err := builder.Train(t.Context(), "left", preparedFixture(t, stage)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := builder.mergeParents(ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}); err != nil {
		t.Fatal(err)
	}
}

// Pulling back a release of this store's own model declares nodes the store
// already records, so their ancestry stays recorded.
func TestPullingOwnReleaseKeepsAncestryRecorded(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	builder := realWeightsBuilder(root)
	if _, err := builder.Initialize("p", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Train(t.Context(), "p", preparedFixture(t, testStage("one"))); err != nil {
		t.Fatal(err)
	}
	stage := testStage("child")
	if _, err := builder.Compose(t.Context(), "child", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	p, err := Inspect(root, "p")
	if err != nil {
		t.Fatal(err)
	}
	node, err := ModelLineage(p, "")
	if err != nil {
		t.Fatal(err)
	}
	files := huggingFaceFixture(t, mergeArchitecture(), HuggingFaceTokenizerClass)
	if files["BOM.json"], err = json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": node}); err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("a", 40))}
	if _, err := (Puller{Root: root, Endpoint: "https://hub.test", Client: client}).Pull(t.Context(), "republished", "huggingface://org/tiny"); err != nil {
		t.Fatal(err)
	}
	if _, declared, err := builder.mergeParents(ComposeBase{Merge: []ComposeMergeParent{{Model: "child"}, {Model: "p"}}}); err != nil || declared {
		t.Fatalf("merge of child and p: declared %v, %v", declared, err)
	}
}

// Only a trained base has a lineage node to pin; any other pin is refused, not recorded.
func TestComposeRefusesLineagePinOnUntrainedBase(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullRelease(t, root, "left", strings.Repeat("a", 64), 0)
	stage := testStage("continue")
	stage.Parameters.SequenceLength = 32
	compose := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "left", LineageSHA256: strings.Repeat("f", 64)}, Stages: []Stage{stage}}
	if _, err := (Builder{Root: root, Resolver: training.FakeResolver()}).Compose(t.Context(), "child", compose, []PreparedStage{preparedFixture(t, stage)}); err == nil {
		t.Fatal("origin base accepted a lineage pin")
	}
	source := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Source: "huggingface://org/tiny@" + strings.Repeat("a", 40), LineageSHA256: strings.Repeat("f", 64)}, Stages: []Stage{stage}}
	if err := source.Validate(); err == nil || !strings.Contains(err.Error(), "base.source") {
		t.Fatalf("source base with a lineage pin: %v", err)
	}
}

func TestMergeComposeValidation(t *testing.T) {
	for name, base := range map[string]ComposeBase{
		"one parent":     {Merge: []ComposeMergeParent{{Model: "a"}}},
		"duplicate":      {Merge: []ComposeMergeParent{{Model: "a"}, {Model: "a"}}},
		"partial weight": {Merge: []ComposeMergeParent{{Model: "a", Weight: 2}, {Model: "b"}}},
		"with model":     {Model: "c", Merge: []ComposeMergeParent{{Model: "a"}, {Model: "b"}}},
		"run pin":        {RunID: "r", Merge: []ComposeMergeParent{{Model: "a"}, {Model: "b"}}},
		"tiny weight":    {Merge: []ComposeMergeParent{{Model: "a", Weight: 1e-50}, {Model: "b", Weight: 1}}},
		"huge weights":   {Merge: []ComposeMergeParent{{Model: "a", Weight: 1e300}, {Model: "b", Weight: 1e300}}},
	} {
		if err := (Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &base}).Validate(); err == nil {
			t.Fatalf("%s: compose accepted", name)
		}
	}
	if err := (Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "a"}}).Validate(); err == nil {
		t.Fatal("stageless compose without base.merge accepted")
	}
}

func TestSameParentAcceptsPinsRecordedBeforeLineage(t *testing.T) {
	pin := ModelParent{Model: "base", ModelID: "id", RunID: "run", RunBOMSHA256: "bom", Artifact: training.Artifact{Path: "base/model.safetensors", SHA256: "weights"}}
	withLineage := pin
	withLineage.LineageSHA256 = strings.Repeat("a", 64)
	if !sameParent(&pin, &withLineage) || !sameParent(&withLineage, &withLineage) {
		t.Fatal("matching parents rejected")
	}
	other := withLineage
	other.LineageSHA256 = strings.Repeat("b", 64)
	if sameParent(&withLineage, &other) {
		t.Fatal("a recorded lineage must match exactly")
	}
	changed := pin
	changed.RunID = "another"
	if sameParent(&changed, &withLineage) || sameParent(nil, &withLineage) {
		t.Fatal("a different parent matched")
	}
}
