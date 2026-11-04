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

	waldotokenizer "github.com/openwaldo/waldo/internal/tokenizer"
	"github.com/openwaldo/waldo/internal/training"
)

// pullMergeable pulls a release whose BOM records a lineage node descending
// from parent; salt changes one weight byte so releases hold different weights.
func pullMergeable(t *testing.T, root, name, parent string, salt byte) {
	t.Helper()
	pullMergeableOf(t, root, name, parent, salt, releaseArchitecture())
}

func pullMergeableOf(t *testing.T, root, name, parent string, salt byte, architecture Architecture) {
	t.Helper()
	node := LineageNode{WeightsSHA256: strings.Repeat(string("0123456789"[salt]), 64), Parents: []LineageLink{{WeightsSHA256: parent}}}
	pullClaiming(t, root, name, mergeableFixture(t, architecture, salt), node)
}

// mergeableFixture is a release's files whose weights average to finite
// values; salt changes one weight byte so releases hold different weights.
func mergeableFixture(t *testing.T, architecture Architecture, salt byte) map[string][]byte {
	t.Helper()
	files := huggingFaceFixture(t, architecture, HuggingFaceTokenizerClass)
	weights := files["model.safetensors"]
	// Keep each bf16 value below 2, so averaging them stays finite.
	for index := 8 + int(binary.LittleEndian.Uint64(weights)) + 1; index < len(weights); index += 2 {
		weights[index] &= 0x3f
	}
	weights[len(weights)-1] ^= salt
	return files
}

// pullClaiming pulls files as a release whose BOM.json declares node as the
// lineage its weights came from, or with no BOM.json when node is empty.
func pullClaiming(t *testing.T, root, name string, files map[string][]byte, node LineageNode) {
	t.Helper()
	if node.WeightsSHA256 != "" {
		bom, err := json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": node, "artifacts": inventory(files)})
		if err != nil {
			t.Fatal(err)
		}
		files["BOM.json"] = bom
	}
	client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("a", 40))}
	if _, err := (Puller{Root: root, Endpoint: "https://hub.test", Client: client}).Pull(t.Context(), name, "huggingface://org/tiny"); err != nil {
		t.Fatal(err)
	}
}

// flipWeightsBuilder trains with a backend that flips one bit of the starting
// weights, so each run holds real, mergeable weights of its own.
func flipWeightsBuilder(root string) Builder {
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
	return Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
}

// assertMerges checks how many merged origins the store holds.
func assertMerges(t *testing.T, root string, want int) {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(root, ".merges"))
	if err != nil || len(entries) != want {
		t.Fatalf(".merges holds %d entries, want %d: %v", len(entries), want, err)
	}
}

func TestMergeComposeRecordsLineageAndFailsClosed(t *testing.T) {
	common, other := strings.Repeat("a", 64), strings.Repeat("b", 64)
	root := t.TempDir()
	isolateHuggingFaceToken(t)
	pullMergeable(t, root, "left", common, 0)
	pullMergeable(t, root, "right", common, 1)
	pullMergeable(t, root, "stranger", other, 2)
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
	node, history, err := LineageHistory(merged, "")
	if err != nil || len(node.Parents) != 2 || node.Parents[0].LineageSHA256 != origin.Merge.Parents[0].LineageSHA256 || origin.Merge.Parents[0].LineageSHA256 == "" {
		t.Fatalf("lineage = %+v, %v", node, err)
	}
	for _, artifact := range origin.Artifacts {
		if artifact.Role == "weights" && artifact.SHA256 != node.WeightsSHA256 {
			t.Fatalf("merged node weights = %s, artifact %s", node.WeightsSHA256, artifact.SHA256)
		}
	}
	// The common ancestor is two links up the chain, through each release's own node, held in LINEAGE.json.
	if reached := ancestry(node, history, false); !reached[common] || reached[other] {
		t.Fatalf("ancestry = %v", reached)
	}
	if merged.BOM.CurrentOriginSHA256 == "" || len(merged.Runs) != 0 {
		t.Fatalf("merged model is not usable from its origin: %+v", merged.BOM)
	}
	if state := listingFromInspection(merged).State; state != "merged" {
		t.Fatalf("merged model lists as %q", state)
	}
	// A release of the merge records the merged node, so the chain continues past publication.
	lineage, err := nodeOf(merged, "")
	if err != nil || !reflect.DeepEqual(lineage, node) {
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
		pullMergeable(t, root, "left", strings.Repeat("a", 64), 0)
		pullMergeable(t, root, "right", strings.Repeat("a", 64), 1)
		builder := Builder{Root: root}
		for _, name := range []string{"merged", "again"} {
			merged, err := builder.Compose(t.Context(), name, Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}, nil)
			if err != nil {
				t.Fatal(err)
			}
			weights = append(weights, merged.Origin.Artifacts[0].SHA256)
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
	wider := releaseArchitecture()
	wider.HiddenSize, wider.IntermediateSize = 8, 16
	pullMergeable(t, root, "left", strings.Repeat("a", 64), 0)
	pullMergeableOf(t, root, "wider", strings.Repeat("a", 64), 1, wider)
	_, err := (Builder{Root: root}).Compose(t.Context(), "merged", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "wider"}}}}, nil)
	if err == nil || !strings.Contains(err.Error(), "does not have the architecture of") {
		t.Fatalf("mismatched merge error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "merged")); !os.IsNotExist(err) {
		t.Fatalf("mismatched merge left a model: %v", err)
	}
}

// Merges average bfloat16 and float32 weights; a plan must not accept
// float16 parents that the merge itself will refuse.
func TestMergeComposeRefusesFloat16Parents(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	architecture := releaseArchitecture()
	architecture.ParameterDType = "float16"
	pullMergeableOf(t, root, "left", strings.Repeat("a", 64), 0, architecture)
	pullMergeableOf(t, root, "right", strings.Repeat("a", 64), 1, architecture)
	if _, _, err := (Builder{Root: root}).mergeParents(ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}); err == nil || !strings.Contains(err.Error(), "float16") {
		t.Fatalf("float16 merge error = %v", err)
	}
}

// With stages, the merge is the origin training starts from.
func TestMergeComposeTrainsFromTheMerge(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullMergeable(t, root, "left", strings.Repeat("a", 64), 0)
	pullMergeable(t, root, "right", strings.Repeat("a", 64), 1)
	stage := testStage("continue")
	stage.Parameters.SequenceLength = 32
	compose := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}, Architecture: releaseArchitecture(), Stages: []Stage{stage}}
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
	pullMergeable(t, root, "victim", strings.Repeat("a", 64), 0)
	victim, err := Inspect(root, "victim")
	if err != nil {
		t.Fatal(err)
	}
	node, err := nodeOf(victim, "")
	if err != nil {
		t.Fatal(err)
	}
	pullMergeable(t, root, "claimant", node.WeightsSHA256, 3)
	builder := flipWeightsBuilder(root)
	stage := releaseStage("continue")
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

// A model built on a merge keeps that merge as its base: re-running its
// compose, or appending stages after a parent trains on, reuses the recorded
// merge instead of averaging the parents' newer weights.
func TestMergeComposeKeepsItsRecordedMerge(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullMergeable(t, root, "left", strings.Repeat("a", 64), 0)
	pullMergeable(t, root, "right", strings.Repeat("a", 64), 1)
	builder := flipWeightsBuilder(root)
	merge := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}
	merged, err := builder.Compose(t.Context(), "merged", merge, nil)
	if err != nil {
		t.Fatal(err)
	}
	stage := releaseStage("one")
	trained := merge
	trained.Stages = []Stage{stage}
	if _, err := builder.Compose(t.Context(), "trained", trained, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Train(t.Context(), "left", preparedFixture(t, releaseStage("more"))); err != nil {
		t.Fatal(err)
	}
	// The merge-only model is complete as it stands.
	resolved, err := builder.ResolveCompose(t.Context(), "merged", merge, true)
	if err != nil {
		t.Fatal(err)
	}
	if completed, err := ComposeCompleted(resolved, merged); err != nil || !completed {
		t.Fatalf("merge-only compose completed = %v, %v", completed, err)
	}
	// A failed run leaves the weights as they were, so the merge-only model is still complete.
	failing := backendFunc(func(context.Context, training.Request) (training.Observation, error) {
		return training.Observation{}, fmt.Errorf("interrupted")
	})
	failer := Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(failing), nil
	})}
	if _, err := failer.Train(t.Context(), "merged", preparedFixture(t, releaseStage("failed"))); err == nil {
		t.Fatal("failing backend trained")
	}
	if merged, err = Inspect(root, "merged"); err != nil {
		t.Fatal(err)
	}
	if completed, err := ComposeCompleted(resolved, merged); err != nil || !completed {
		t.Fatalf("merge-only compose after a failed run completed = %v, %v", completed, err)
	}
	two := releaseStage("two")
	trained.Stages = []Stage{stage, two}
	if _, err := builder.Compose(t.Context(), "trained", trained, []PreparedStage{preparedFixture(t, stage), preparedFixture(t, two)}); err != nil {
		t.Fatal(err)
	}
	assertMerges(t, root, 1)
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

// A pulled release that trains on keeps the ancestry its release recorded.
func TestMergeFindsAncestryOfAPulledReleaseThatTrained(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullMergeable(t, root, "left", strings.Repeat("a", 64), 0)
	pullMergeable(t, root, "right", strings.Repeat("a", 64), 1)
	builder := realWeightsBuilder(root)
	stage := testStage("continue")
	stage.Parameters.SequenceLength = 32
	stage.Parameters.Steps *= 2
	if _, err := builder.Train(t.Context(), "left", preparedFixture(t, stage)); err != nil {
		t.Fatal(err)
	}
	// The two releases meet only through what they declared.
	if _, declared, err := builder.mergeParents(ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}); err != nil || !declared {
		t.Fatalf("merge of left and right: declared %v, %v", declared, err)
	}
}

// A release can name a node this store records: the plain node of weights
// anyone can pull, or a run node any export publishes. Its weights descending
// from that node is still only its claim.
func TestReleaseNamingARecordedNodeIsStillDeclared(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullClaiming(t, root, "plain", mergeableFixture(t, releaseArchitecture(), 0), LineageNode{})
	stage := releaseStage("local")
	if _, err := flipWeightsBuilder(root).Compose(t.Context(), "trained", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "plain"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	for salt, recorded := range map[byte]string{1: "plain", 2: "trained"} {
		inspection, err := Inspect(root, recorded)
		if err != nil {
			t.Fatal(err)
		}
		node, err := nodeOf(inspection, "")
		if err != nil {
			t.Fatal(err)
		}
		name := "claims-" + recorded
		pullClaiming(t, root, name, mergeableFixture(t, releaseArchitecture(), salt), node)
		if _, declared, err := (Builder{Root: root}).mergeParents(ComposeBase{Merge: []ComposeMergeParent{{Model: recorded}, {Model: name}}}); err != nil || !declared {
			t.Fatalf("%s: declared %v, %v", name, declared, err)
		}
	}
}

// Pulling back a release that republishes a node this store records leaves
// the chain through that node recorded: two fine-tunes of different runs of p
// still meet at p's first run, past the republished node.
func TestPullingOwnReleaseKeepsAncestryRecorded(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	builder := realWeightsBuilder(root)
	first := trainTwice(t, builder)
	for name, base := range map[string]ComposeBase{"early": {Model: "p", RunID: first}, "late": {Model: "p"}} {
		stage := testStage(name)
		if _, err := builder.Compose(t.Context(), name, Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &base, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)}); err != nil {
			t.Fatal(err)
		}
	}
	p, err := Inspect(root, "p")
	if err != nil {
		t.Fatal(err)
	}
	node, err := nodeOf(p, "")
	if err != nil {
		t.Fatal(err)
	}
	pullClaiming(t, root, "republished", huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass), node)
	if _, declared, err := builder.mergeParents(ComposeBase{Merge: []ComposeMergeParent{{Model: "early"}, {Model: "late"}}}); err != nil || declared {
		t.Fatalf("merge of early and late: declared %v, %v", declared, err)
	}
}

// A recorded merge belongs to the models it averaged: once a parent's name
// holds a different model, the compose no longer names that merge.
func TestRecordedMergeRefusesARecreatedParent(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullMergeable(t, root, "left", strings.Repeat("a", 64), 0)
	pullMergeable(t, root, "right", strings.Repeat("a", 64), 1)
	builder := flipWeightsBuilder(root)
	merge := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}
	if _, err := builder.Compose(t.Context(), "merged", merge, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Remove(root, []string{"left"}); err != nil {
		t.Fatal(err)
	}
	pullClaiming(t, root, "left", mergeableFixture(t, releaseArchitecture(), 9), LineageNode{WeightsSHA256: strings.Repeat("c", 64)})
	if _, err := builder.ResolveCompose(t.Context(), "merged", merge, true); err == nil {
		t.Fatal("merge compose resolved to the merge of the removed left")
	}
	assertMerges(t, root, 1)
}

// A parent rebuilt from the same base keeps its plan's model ID but not its
// weights; the merge of the old weights is refused before anything is merged.
func TestRecordedMergeRefusesAParentRebuiltUnderItsID(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullMergeable(t, root, "base", strings.Repeat("a", 64), 0)
	builder := flipWeightsBuilder(root)
	child := func(name string, stages ...Stage) {
		var prepared []PreparedStage
		for _, stage := range stages {
			prepared = append(prepared, preparedFixture(t, stage))
		}
		if _, err := builder.Compose(t.Context(), name, Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "base"}, Stages: stages}, prepared); err != nil {
			t.Fatal(err)
		}
	}
	child("left", releaseStage("one"))
	child("right", releaseStage("two"))
	merge := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}
	if _, err := builder.Compose(t.Context(), "merged", merge, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Remove(root, []string{"left"}); err != nil {
		t.Fatal(err)
	}
	child("left", releaseStage("other"), releaseStage("data"))
	if _, err := builder.ResolveCompose(t.Context(), "merged", merge, true); err == nil || !strings.Contains(err.Error(), "no longer holds") {
		t.Fatalf("rebuilt parent error = %v", err)
	}
	assertMerges(t, root, 1)
}

// A merge-only compose describes the merged model only until it runs a stage,
// even one that leaves its weights as they were.
func TestMergeOnlyComposeIsNotCompleteAfterAStage(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullMergeable(t, root, "left", strings.Repeat("a", 64), 0)
	pullMergeable(t, root, "right", strings.Repeat("a", 64), 1)
	merge := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}
	simulated := Builder{Root: root, Resolver: training.FakeResolver()}
	if _, err := simulated.Compose(t.Context(), "merged", merge, nil); err != nil {
		t.Fatal(err)
	}
	stage := releaseStage("simulated")
	grown := merge
	grown.Stages = []Stage{stage}
	if _, err := simulated.Compose(t.Context(), "merged", grown, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	merged, err := Inspect(root, "merged")
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := simulated.ResolveCompose(t.Context(), "merged", merge, true)
	if err != nil {
		t.Fatal(err)
	}
	if completed, err := ComposeCompleted(resolved, merged); err != nil || completed {
		t.Fatalf("merge-only compose after a simulated stage completed = %v, %v", completed, err)
	}
}

// A merge-only compose interrupted after its model is written leaves its
// transaction pending; repeating the compose finishes it.
func TestMergeComposeFinishesAnInterruptedTransaction(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullMergeable(t, root, "left", strings.Repeat("a", 64), 0)
	pullMergeable(t, root, "right", strings.Repeat("a", 64), 1)
	builder := Builder{Root: root}
	merge := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}
	resolved, err := builder.ResolveCompose(t.Context(), "", merge, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Compose(t.Context(), "merged", merge, nil); err != nil {
		t.Fatal(err)
	}
	// The transaction Compose writes before the model exists, left behind by an interruption.
	transaction := composeTransaction{Kind: "waldo-model-compose-transaction", Schema: 1, Name: "merged", Compose: resolved}
	id, err := hashJSON(transaction)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeJSONAtomic(filepath.Join(root, ".waldo-compose", id, "COMPOSE.json"), transaction); err != nil {
		t.Fatal(err)
	}
	if pending, err := HasPendingCompose(root, "merged"); err != nil || !pending {
		t.Fatalf("pending = %v, %v", pending, err)
	}
	if _, err := builder.Compose(t.Context(), "merged", merge, nil); err != nil {
		t.Fatal(err)
	}
	if pending, err := HasPendingCompose(root, "merged"); err != nil || pending {
		t.Fatalf("after repeating the compose, pending = %v, %v", pending, err)
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

// Repeating a finished compose keeps one history entry, for a merge-only model
// and for a fine-tune whose base carries declared ancestry.
func TestRepeatedComposeKeepsOneHistoryEntry(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullClaiming(t, root, "victim", mergeableFixture(t, releaseArchitecture(), 0), LineageNode{})
	victim, err := Inspect(root, "victim")
	if err != nil {
		t.Fatal(err)
	}
	node, err := nodeOf(victim, "")
	if err != nil {
		t.Fatal(err)
	}
	pullClaiming(t, root, "claimant", mergeableFixture(t, releaseArchitecture(), 3), LineageNode{WeightsSHA256: node.WeightsSHA256})
	builder := flipWeightsBuilder(root)
	merge := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "victim"}, {Model: "claimant"}}}}
	stage := releaseStage("mid")
	mid := merge
	mid.Stages = []Stage{stage}
	if _, err := builder.Compose(t.Context(), "mid", mid, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	child := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "mid"}, Stages: []Stage{releaseStage("child")}}
	for name, compose := range map[string]Compose{"merged": merge, "child": child} {
		model, err := builder.Compose(t.Context(), name, compose, preparedStages(t, compose))
		if err != nil {
			t.Fatal(err)
		}
		before, err := os.ReadDir(filepath.Join(model.Path, "composes"))
		if err != nil {
			t.Fatal(err)
		}
		for range 2 {
			resolved, err := builder.ResolveCompose(t.Context(), name, compose, true)
			if err != nil {
				t.Fatal(err)
			}
			if err := PersistCompletedCompose(model.Path, resolved, name+".yaml"); err != nil {
				t.Fatal(err)
			}
		}
		after, err := os.ReadDir(filepath.Join(model.Path, "composes"))
		if err != nil || len(after) != len(before) {
			t.Fatalf("%s: history grew from %d to %d entries: %v", name, len(before), len(after), err)
		}
	}
}

func preparedStages(t *testing.T, compose Compose) []PreparedStage {
	t.Helper()
	var prepared []PreparedStage
	for _, stage := range compose.Stages {
		prepared = append(prepared, preparedFixture(t, stage))
	}
	return prepared
}

// Repeating a merge-only compose verifies the merged weights, which come from
// the origin, not from a run.
func TestCompletedMergeOnlyComposeVerifies(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullMergeable(t, root, "left", strings.Repeat("a", 64), 0)
	pullMergeable(t, root, "right", strings.Repeat("a", 64), 1)
	builder := Builder{Root: root}
	merge := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}
	merged, err := builder.Compose(t.Context(), "merged", merge, nil)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := builder.ResolveCompose(t.Context(), "merged", merge, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyCompletedCompose(resolved, merged); err != nil {
		t.Fatalf("verify merge-only model: %v", err)
	}
}

// A merge-based model whose cached merge is gone is refused before anything is
// merged, rather than averaging its parents' current weights.
func TestMergeComposeRefusesAMissingRecordedMerge(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullMergeable(t, root, "left", strings.Repeat("a", 64), 0)
	pullMergeable(t, root, "right", strings.Repeat("a", 64), 1)
	builder := flipWeightsBuilder(root)
	merge := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}
	if _, err := builder.Compose(t.Context(), "merged", merge, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Join(root, ".merges")); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.ResolveCompose(t.Context(), "merged", merge, true); err == nil || !strings.Contains(err.Error(), ".merges") {
		t.Fatalf("missing recorded merge: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, ".merges")); !os.IsNotExist(err) {
		t.Fatalf("a refused compose wrote a merge: %v", err)
	}
}

func TestMergeComposeMergesTrainedTokenizerModels(t *testing.T) {
	artifact, err := waldotokenizer.TrainByteBPE([]waldotokenizer.Sample{{ID: "one", Text: strings.Repeat("fn main() { println!(\"hello\"); } ", 20)}}, 290, 4096, strings.Repeat("a", 64))
	if err != nil {
		t.Fatal(err)
	}
	architecture := Architecture{Family: "decoder-transformer", ContextTokens: 32, VocabularySize: uint64(artifact.VocabularySize), HiddenSize: 4, IntermediateSize: 8, Layers: 1, AttentionHeads: 2, KeyValueHeads: 1, TieEmbeddings: true, ParameterDType: "bfloat16", Tokenizer: Tokenizer{Name: artifact.Name, Revision: artifact.Revision, Artifact: &artifact}}
	exported, err := json.Marshal(struct {
		Kind   string `json:"kind"`
		Schema int    `json:"schema"`
		training.TokenizerSpec
	}{"waldo-tokenizer", 1, training.TokenizerSpec{Name: artifact.Name, Revision: artifact.Revision, VocabularySize: artifact.VocabularySize, PadID: 0, BOSID: 1, EOSID: 2, Artifact: &artifact}})
	if err != nil {
		t.Fatal(err)
	}
	common := strings.Repeat("a", 64)
	root := t.TempDir()
	extra := map[string][]byte{TokenizerArtifactFile: exported}
	for name, directory := range map[string]string{"left": releasePackageOf(t, architecture, extra, common, 0), "right": releasePackageOf(t, architecture, extra, common, 1), "byte": releasePackage(t, common, 2)} {
		if _, err := (Puller{Root: root}).PullDirectory(name, directory); err != nil {
			t.Fatal(err)
		}
	}
	builder := Builder{Root: root}
	merged, err := builder.Compose(context.Background(), "merged", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(merged.Model.Architecture, architecture) {
		t.Fatalf("merged architecture = %+v", merged.Model.Architecture)
	}
	// The merge's runtime tokenizer is the trained one, so it can be continued and exported.
	if _, _, err := training.ResolveArchitectureTokenizer(mustJSON(t, merged.Model.Architecture)); err != nil {
		t.Fatal(err)
	}
	_, err = builder.Compose(context.Background(), "mixed", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "byte"}}}}, nil)
	if err == nil || !strings.Contains(err.Error(), "does not have the architecture") {
		t.Fatalf("mixed tokenizer merge error = %v", err)
	}
}
