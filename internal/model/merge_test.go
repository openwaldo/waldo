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
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/training"
)

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

// pullPlain pulls files as a release with no WALDO history: an end.
func pullPlain(t *testing.T, root, name string, files map[string][]byte) {
	t.Helper()
	isolateHuggingFaceToken(t)
	client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("a", 40))}
	if _, err := (Puller{Root: root, Endpoint: "https://hub.test", Client: client}).Pull(t.Context(), name, "huggingface://org/tiny"); err != nil {
		t.Fatal(err)
	}
}

// flipWeightsBuilder trains with a backend that flips one bit of the starting
// weights, a different bit each run, so each run holds real, mergeable
// weights of its own.
func flipWeightsBuilder(root string) Builder {
	runs := 0
	backend := backendFunc(func(_ context.Context, request training.Request) (training.Observation, error) {
		data, err := os.ReadFile(request.Initialization.Path)
		if err != nil {
			return training.Observation{}, err
		}
		data[len(data)-1] ^= 1 << (runs % 5)
		runs++
		if err := os.MkdirAll(request.ArtifactDirectory, 0o755); err != nil {
			return training.Observation{}, err
		}
		if err := os.WriteFile(filepath.Join(request.ArtifactDirectory, "model.safetensors"), data, 0o644); err != nil {
			return training.Observation{}, err
		}
		digest := sha256.Sum256(data)
		return training.Observation{
			Steps: 2, ConsumedTokens: 64,
			Evaluations: []training.Evaluation{{Step: 2, Tokens: 64, Metrics: map[string]float64{"heldout_loss": 1}}},
			Artifacts:   []training.Artifact{{Path: "artifacts/model.safetensors", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}},
		}, nil
	})
	return Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
}

// tune composes name from base with one stage per stage name.
func tune(t *testing.T, builder Builder, base ComposeBase, name string, stages ...string) Inspection {
	t.Helper()
	compose := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &base}
	for _, stage := range stages {
		compose.Stages = append(compose.Stages, lineageStage(stage))
	}
	inspection, err := builder.Compose(t.Context(), name, compose, preparedStages(t, compose))
	if err != nil {
		t.Fatal(err)
	}
	return inspection
}

// family pulls a base with no history and fine-tunes left and right from it.
func family(t *testing.T, root string) Builder {
	t.Helper()
	pullPlain(t, root, "base", mergeableFixture(t, lineageArchitecture(), 0))
	builder := flipWeightsBuilder(root)
	tune(t, builder, ComposeBase{Model: "base"}, "left", "left")
	tune(t, builder, ComposeBase{Model: "base"}, "right", "right")
	return builder
}

func mergeOf(parents ...string) Compose {
	base := ComposeBase{}
	for _, parent := range parents {
		base.Merge = append(base.Merge, ComposeMergeParent{Model: parent})
	}
	return Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &base}
}

// assertMerges checks how many merged origins the root holds.
func assertMerges(t *testing.T, root string, want int) {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(root, ".merges"))
	if err != nil || len(entries) != want {
		t.Fatalf(".merges holds %d entries, want %d: %v", len(entries), want, err)
	}
}

func TestMergeComposeRecordsItsParentsAndFailsClosed(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	pullPlain(t, root, "stranger", mergeableFixture(t, lineageArchitecture(), 2))
	merged, err := builder.Compose(t.Context(), "merged", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left", Weight: 1}, {Model: "right", Weight: 3}}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !isMerge(merged.Origin) || merged.Origin.Source.Revision != merged.Origin.SourceArtifacts[0].SHA256 || merged.Origin.Source.URL != "" {
		t.Fatalf("origin = %+v", merged.Origin)
	}
	record, err := ReadMerge(merged)
	if err != nil || record.Method != "weighted-mean-f32-v1" || len(record.Parents) != 2 || record.Parents[0].WeightsSHA256 > record.Parents[1].WeightsSHA256 {
		t.Fatalf("merge = %+v, %v", record, err)
	}
	if merged.BOM.CurrentOriginSHA256 == "" || len(merged.Runs) != 0 || listingFromInspection(merged).State != "merged" {
		t.Fatalf("merged model is not usable from its origin: %+v", merged.BOM)
	}
	// The merge's history holds both parents' runs, so they can be deleted.
	for _, name := range []string{"left", "right"} {
		if err := os.RemoveAll(filepath.Join(root, name)); err != nil {
			t.Fatal(err)
		}
	}
	walk := modelHistory(root, merged, "")
	for _, parent := range record.Parents {
		if _, ok := walk.held[parent.RunBOMSHA256]; !ok || len(walk.gaps) != 0 {
			t.Fatalf("parent run %s isn't in the merge's history: gaps %v", short(parent.RunBOMSHA256), walk.sortedGaps())
		}
	}
	tune(t, builder, ComposeBase{Model: "base"}, "left", "again")
	if _, err := builder.Compose(t.Context(), "unrelated", mergeOf("left", "stranger"), nil); err == nil || !strings.Contains(err.Error(), "share no ancestor") {
		t.Fatalf("unrelated merge error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "unrelated")); !os.IsNotExist(err) {
		t.Fatalf("unrelated merge left a model: %v", err)
	}
}

// The same parents give the same MERGE.json and the same merged weights,
// whatever order a compose names them in, each time they are computed.
func TestMergeComposeIsReproducible(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	var keys []string
	for _, merge := range []Compose{mergeOf("left", "right"), mergeOf("right", "left")} {
		merged, err := builder.Compose(t.Context(), fmt.Sprintf("merged-%d", len(keys)), merge, nil)
		if err != nil {
			t.Fatal(err)
		}
		keys = append(keys, merged.Origin.Source.Revision+" "+merged.Origin.Artifacts[0].SHA256)
		// Drop the cached merge so the next one is computed again, not reused.
		if err := os.RemoveAll(filepath.Join(root, ".merges")); err != nil {
			t.Fatal(err)
		}
	}
	if keys[0] != keys[1] {
		t.Fatalf("the same merge differs by order: %v", keys)
	}
}

func TestMergeComposeRefusesDifferentArchitectures(t *testing.T) {
	root := t.TempDir()
	wider := lineageArchitecture()
	wider.HiddenSize, wider.IntermediateSize = 8, 16
	pullPlain(t, root, "left", mergeableFixture(t, lineageArchitecture(), 0))
	pullPlain(t, root, "wider", mergeableFixture(t, wider, 1))
	_, err := (Builder{Root: root}).Compose(t.Context(), "merged", mergeOf("left", "wider"), nil)
	if err == nil || !strings.Contains(err.Error(), "does not have the architecture of") {
		t.Fatalf("mismatched merge error = %v", err)
	}
}

// Merges average bfloat16 and float32 weights; a plan must not accept
// float16 parents that the merge itself will refuse.
func TestMergeComposeRefusesFloat16Parents(t *testing.T) {
	root := t.TempDir()
	architecture := lineageArchitecture()
	architecture.ParameterDType = "float16"
	pullPlain(t, root, "left", mergeableFixture(t, architecture, 0))
	pullPlain(t, root, "right", mergeableFixture(t, architecture, 1))
	if _, err := (Builder{Root: root}).mergeParents(ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}); err == nil || !strings.Contains(err.Error(), "float16") {
		t.Fatalf("float16 merge error = %v", err)
	}
}

// With stages, the merge is the origin training starts from.
func TestMergeComposeTrainsFromTheMerge(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	trained := tune(t, builder, *mergeOf("left", "right").Base, "trained", "continue")
	if !isMerge(trained.Origin) || len(trained.Runs) != 1 {
		t.Fatalf("trained = %+v", trained)
	}
}

// A gap above where the parents meet is refused: the merge can't show that
// the parents are of one family.
func TestMergeRefusesAGapAboveTheMergeBase(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	tune(t, builder, ComposeBase{Model: "left"}, "deeper", "deeper")
	for _, path := range []string{filepath.Join(root, "left"), filepath.Join(root, "deeper", lineageFile)} {
		if err := os.RemoveAll(path); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := builder.Compose(t.Context(), "merged", mergeOf("deeper", "right"), nil); err == nil || !strings.Contains(err.Error(), "gap") {
		t.Fatalf("merge across a gap = %v", err)
	}
}

func TestMergeAcceptsAGapAtTheMergeBase(t *testing.T) {
	root := t.TempDir()
	trainTwice(t, lineageBuilder(root), "base")
	builder := flipWeightsBuilder(root)
	tune(t, builder, ComposeBase{Model: "base"}, "left", "left")
	tune(t, builder, ComposeBase{Model: "base"}, "right", "right")
	// What's missing now is how the merge-base itself was made, which a merge only records.
	for _, path := range []string{filepath.Join(root, "base"), filepath.Join(root, "left", lineageFile), filepath.Join(root, "right", lineageFile)} {
		if err := os.RemoveAll(path); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := builder.Compose(t.Context(), "merged", mergeOf("left", "right"), nil); err != nil {
		t.Fatalf("a gap at the merge-base refused the merge: %v", err)
	}
}

// A release pulled and trained on meets the model it was exported from.
func TestMergeFindsTheMergeBaseThroughARelease(t *testing.T) {
	root := t.TempDir()
	builder := lineageBuilder(root)
	exporter := trainTwice(t, builder, "exporter")
	pullRelease(t, root, releaseFiles(t, root, exporter, true, nil), nil)
	tune(t, builder, ComposeBase{Model: "pulled"}, "remote", "remote")
	tune(t, builder, ComposeBase{Model: "exporter"}, "local", "local")
	parents, err := builder.mergeParents(*mergeOf("remote", "local").Base)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMergeHistory(root, parents, nil); err != nil {
		t.Fatalf("a release and its exporter's fine-tune don't meet: %v", err)
	}
}

// A child of p's first run and p after its second run meet at the first run's weights.
func TestMergeFindsTheMergeBaseThroughEarlierRuns(t *testing.T) {
	root := t.TempDir()
	builder := lineageBuilder(root)
	if _, err := builder.Initialize("p", lineageArchitecture()); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Train(t.Context(), "p", preparedFixture(t, lineageStage("one"))); err != nil {
		t.Fatal(err)
	}
	tune(t, builder, ComposeBase{Model: "p"}, "child", "child")
	if _, err := builder.Train(t.Context(), "p", preparedFixture(t, lineageStage("two"))); err != nil {
		t.Fatal(err)
	}
	parents, err := builder.mergeParents(*mergeOf("child", "p").Base)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMergeHistory(root, parents, nil); err != nil {
		t.Fatalf("merge of child and p: %v", err)
	}
}

// graph is a hand-made weights graph: each weights to what it was made from.
func graph(edges map[string][]string, merged ...string) *lineageWalk {
	walk := newLineageWalk(newLineageIndex("", nil))
	for weights, parents := range edges {
		walk.parents[weights] = parents
	}
	for _, weights := range merged {
		walk.merged[weights] = true
	}
	return walk
}

// Two merges of the same parents, merged again, meet at both parents, as git
// finds two merge-bases for a criss-cross history.
func TestMergeBaseOfACrissCrossIsBothParents(t *testing.T) {
	walk := graph(map[string][]string{"a": {"x1", "x2"}, "b": {"x1", "x2"}, "x1": {"root"}, "x2": {"root"}}, "a", "b")
	if err := walk.mergeBase(map[string]string{"a": "a", "b": "b"}); err != nil {
		t.Fatalf("criss-cross merge refused: %v", err)
	}
}

// A merge above the merge-base with a parent from outside the family is refused.
// One contributor's earlier model and the group's merge of it stay one family: merged
// again, the merge meets at that model, and the merge's other parent still reaches the base.
func TestAMergeMergesAgainWithItsOwnFamily(t *testing.T) {
	for name, second := range map[string]func(*testing.T, Builder) (string, string){
		"with a parent": func(*testing.T, Builder) (string, string) { return "m1", "left" },
		"with a parent trained on": func(t *testing.T, builder Builder) (string, string) {
			tune(t, builder, ComposeBase{Model: "left"}, "left2", "more")
			return "m1", "left2"
		},
		"trained on, with a parent": func(t *testing.T, builder Builder) (string, string) {
			tune(t, builder, ComposeBase{Model: "m1"}, "m1t", "sft")
			return "m1t", "right"
		},
	} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			builder := family(t, root)
			if _, err := builder.Compose(t.Context(), "m1", mergeOf("left", "right"), nil); err != nil {
				t.Fatal(err)
			}
			first, other := second(t, builder)
			if _, err := builder.Compose(t.Context(), "m2", mergeOf(first, other), nil); err != nil {
				t.Fatalf("merge of %s and %s refused: %v", first, other, err)
			}
		})
	}
}

func TestMergeRefusesAnOutsideParentAboveTheMergeBase(t *testing.T) {
	walk := graph(map[string][]string{"a": {"base"}, "b": {"m"}, "m": {"base", "outsider"}}, "m")
	if err := walk.mergeBase(map[string]string{"a": "a", "b": "b"}); err == nil || !strings.Contains(err.Error(), "outside the family") {
		t.Fatalf("outside parent = %v", err)
	}
	walk.ends["outsider"] = true
	walk.parents["outsider"] = nil
	walk.parents["b"] = []string{"outsider"}
	if err := walk.mergeBase(map[string]string{"a": "a", "b": "b"}); err == nil {
		t.Fatal("parents that don't meet were merged")
	}
}

// A model built on a merge keeps that merge as its base: re-running its
// compose, or appending stages after a parent trains on, reuses the recorded
// merge instead of averaging the parents' newer weights.
func TestMergeComposeKeepsItsRecordedMerge(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	merge := mergeOf("left", "right")
	merged, err := builder.Compose(t.Context(), "merged", merge, nil)
	if err != nil {
		t.Fatal(err)
	}
	stage := lineageStage("one")
	trained := merge
	trained.Stages = []Stage{stage}
	if _, err := builder.Compose(t.Context(), "trained", trained, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Train(t.Context(), "left", preparedFixture(t, lineageStage("more"))); err != nil {
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
	if _, err := failer.Train(t.Context(), "merged", preparedFixture(t, lineageStage("failed"))); err == nil {
		t.Fatal("failing backend trained")
	}
	if merged, err = Inspect(root, "merged"); err != nil {
		t.Fatal(err)
	}
	if completed, err := ComposeCompleted(resolved, merged); err != nil || !completed {
		t.Fatalf("merge-only compose after a failed run completed = %v, %v", completed, err)
	}
	two := lineageStage("two")
	trained.Stages = []Stage{stage, two}
	if _, err := builder.Compose(t.Context(), "trained", trained, []PreparedStage{preparedFixture(t, stage), preparedFixture(t, two)}); err != nil {
		t.Fatal(err)
	}
	assertMerges(t, root, 1)
}

func TestRecordedMergeNeedsADifferentParentForEachOne(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	if _, err := builder.Compose(t.Context(), "merged", mergeOf("base", "right"), nil); err != nil {
		t.Fatal(err)
	}
	// base2 holds base's weights, so alone it could answer for base's place in the merge, but not for right's.
	pullPlain(t, root, "base2", mergeableFixture(t, lineageArchitecture(), 0))
	if _, err := builder.Compose(t.Context(), "merged", mergeOf("base", "base2"), nil); err == nil {
		t.Fatal("a merge of base and right was taken for a merge of base and its copy")
	}
}

// A recorded merge belongs to the weights it averaged: once a parent's name
// holds a different model, the compose no longer names that merge.
func TestRecordedMergeRefusesARecreatedParent(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	merge := mergeOf("left", "right")
	if _, err := builder.Compose(t.Context(), "merged", merge, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Remove(root, []string{"left"}); err != nil {
		t.Fatal(err)
	}
	pullPlain(t, root, "left", mergeableFixture(t, lineageArchitecture(), 9))
	if _, err := builder.ResolveCompose(t.Context(), "merged", merge, true); err == nil || !strings.Contains(err.Error(), "no longer holds") {
		t.Fatalf("recreated parent = %v", err)
	}
	assertMerges(t, root, 1)
}

// A parent rebuilt from the same base keeps its plan's model ID but not its
// weights; the merge of the old weights is refused before anything is merged.
func TestRecordedMergeRefusesAParentRebuiltUnderItsID(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	merge := mergeOf("left", "right")
	if _, err := builder.Compose(t.Context(), "merged", merge, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Remove(root, []string{"left"}); err != nil {
		t.Fatal(err)
	}
	tune(t, builder, ComposeBase{Model: "base"}, "left", "other", "data")
	if _, err := builder.ResolveCompose(t.Context(), "merged", merge, true); err == nil || !strings.Contains(err.Error(), "no longer holds") {
		t.Fatalf("rebuilt parent error = %v", err)
	}
	assertMerges(t, root, 1)
}

// A merge-only compose describes the merged model only until it runs a stage,
// even one that leaves its weights as they were.
func TestMergeOnlyComposeIsNotCompleteAfterAStage(t *testing.T) {
	root := t.TempDir()
	family(t, root)
	merge := mergeOf("left", "right")
	simulated := Builder{Root: root, Resolver: training.FakeResolver()}
	if _, err := simulated.Compose(t.Context(), "merged", merge, nil); err != nil {
		t.Fatal(err)
	}
	stage := lineageStage("simulated")
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
	root := t.TempDir()
	builder := family(t, root)
	merge := mergeOf("left", "right")
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

// Repeating a finished compose keeps one history entry, for a merge-only
// model and for a fine-tune of a merge.
func TestRepeatedComposeKeepsOneHistoryEntry(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	merge := mergeOf("left", "right")
	mid := merge
	mid.Stages = []Stage{lineageStage("mid")}
	if _, err := builder.Compose(t.Context(), "mid", mid, preparedStages(t, mid)); err != nil {
		t.Fatal(err)
	}
	child := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "mid"}, Stages: []Stage{lineageStage("child")}}
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
	root := t.TempDir()
	builder := family(t, root)
	merge := mergeOf("left", "right")
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
	root := t.TempDir()
	builder := family(t, root)
	merge := mergeOf("left", "right")
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

func TestAMergeCarriesItsParentsTrainingData(t *testing.T) {
	root := t.TempDir()
	pullPlain(t, root, "base", mergeableFixture(t, lineageArchitecture(), 0))
	builder := flipWeightsBuilder(root)
	// Each parent trains on a corpus of its own, which only the walk through the merge can find.
	parents := map[string]string{}
	for _, name := range []string{"left", "right"} {
		stage := lineageStage(name)
		compose := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "base"}, Stages: []Stage{stage}}
		prepared := preparedFixture(t, stage)
		bom := prepared.BOM
		bom.Manifests = append([]corpus.ManifestPin(nil), bom.Manifests...)
		bom.Manifests[0].Title = "Corpus of " + name
		own, err := PrepareStage(stage, bom, prepared.Inputs)
		if err != nil {
			t.Fatal(err)
		}
		parent, err := builder.Compose(t.Context(), name, compose, []PreparedStage{own})
		if err != nil {
			t.Fatal(err)
		}
		parents[parent.RunBOMs[0].CorpusBOMSHA256] = name
	}
	if _, err := builder.Compose(t.Context(), "merged", mergeOf("left", "right"), nil); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"left", "right", "base"} {
		if err := os.RemoveAll(filepath.Join(root, name)); err != nil {
			t.Fatal(err)
		}
	}
	next := tune(t, builder, ComposeBase{Model: "merged"}, "next", "next")
	history := TrainingHistory(root, next).Corpora
	for digest, name := range parents {
		if _, own := parents[next.RunBOMs[0].CorpusBOMSHA256]; own {
			t.Fatal("the fine-tune of the merge trained on a parent's corpus; the test can't tell the walk from its own run")
		}
		if used, ok := history[digest]; !ok || used.Model != name {
			t.Fatalf("a model made from a merge lost %s's corpus BOM %s: %+v", name, digest[:12], used)
		}
	}
}

func TestTrainingHistoryFindsThePullItStartedFrom(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	if _, err := builder.Compose(t.Context(), "merged", mergeOf("left", "right"), nil); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"left", "merged"} {
		inspection, err := Inspect(root, name)
		if err != nil {
			t.Fatal(err)
		}
		if upstream := TrainingHistory(root, inspection).Upstream; upstream == nil || upstream.Repository != "org/tiny" {
			t.Fatalf("%s's history doesn't reach the pulled base it started from: %+v", name, upstream)
		}
	}
	scratch := trainTwice(t, lineageBuilder(t.TempDir()), "scratch")
	if upstream := TrainingHistory(t.TempDir(), scratch).Upstream; upstream != nil {
		t.Fatalf("a model trained from scratch has an upstream: %+v", upstream)
	}
}

func TestComposeFromAMergeAfterItsParentsAreDeleted(t *testing.T) {
	root := t.TempDir()
	builder := family(t, root)
	if _, err := builder.Train(t.Context(), "left", preparedFixture(t, lineageStage("left2"))); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Compose(t.Context(), "merged", mergeOf("left", "right"), nil); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"left", "right", "base"} {
		if err := os.RemoveAll(filepath.Join(root, name)); err != nil {
			t.Fatal(err)
		}
	}
	next := tune(t, builder, ComposeBase{Model: "merged"}, "next", "next")
	if walk := modelHistory(root, next, next.BOM.CurrentRunID); len(walk.gaps) != 0 {
		t.Fatalf("history has gaps after the merge's parents were deleted: %v", walk.sortedGaps())
	}
}
