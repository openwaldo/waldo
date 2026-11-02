// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"encoding/json"
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
	if reached := ancestry(*origin.Lineage, builder.lineageIndex()); !reached[common] || reached[other] {
		t.Fatalf("ancestry = %v", reached)
	}
	if merged.BOM.CurrentOriginSHA256 == "" || len(merged.Runs) != 0 {
		t.Fatalf("merged model is not usable from its origin: %+v", merged.BOM)
	}
	// A release of the merge records the merged node, so the chain continues past publication.
	lineage, err := ModelLineage(merged)
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

func TestMergeComposeValidation(t *testing.T) {
	for name, base := range map[string]ComposeBase{
		"one parent":     {Merge: []ComposeMergeParent{{Model: "a"}}},
		"duplicate":      {Merge: []ComposeMergeParent{{Model: "a"}, {Model: "a"}}},
		"partial weight": {Merge: []ComposeMergeParent{{Model: "a", Weight: 2}, {Model: "b"}}},
		"with model":     {Model: "c", Merge: []ComposeMergeParent{{Model: "a"}, {Model: "b"}}},
		"run pin":        {RunID: "r", Merge: []ComposeMergeParent{{Model: "a"}, {Model: "b"}}},
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
