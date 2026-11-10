// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/corpus"
)

// withClosure adds a release training closure to a package: its own run BOMs' corpora and the named inherited ones.
func withClosure(t *testing.T, directory string, corpora ...string) {
	t.Helper()
	closure := emptyClosure(true)
	for _, digest := range corpora {
		closure.Corpora = append(closure.Corpora, ClosureCorpus{CorpusBOMSHA256: digest, Uses: 1, CorpusBOM: corpus.BOM{Kind: "openwaldo-bom", Schema: 1, Subject: "corpus", Paths: []string{"code/" + digest[:4]}}})
	}
	runs, err := filepath.Glob(filepath.Join(directory, "provenance-*.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range runs {
		var run RunBOM
		data, err := os.ReadFile(path)
		if err != nil || json.Unmarshal(data, &run) != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		closure.Corpora = append(closure.Corpora, ClosureCorpus{CorpusBOMSHA256: run.CorpusBOMSHA256, Uses: 1, CorpusBOM: run.CorpusBOM})
	}
	closure = unionClosures(closure)
	data, err := json.Marshal(closure)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, ClosureFile), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestTrainingClosureFollowsTheWholeLineage(t *testing.T) {
	common, first, second := strings.Repeat("a", 64), strings.Repeat("1", 64), strings.Repeat("2", 64)
	left, right, bare := releasePackage(t, common, 0), releasePackage(t, common, 1), releasePackage(t, common, 2)
	withClosure(t, left, first)
	withClosure(t, right, first, second)
	root := t.TempDir()
	for name, directory := range map[string]string{"left": left, "right": right, "bare": bare} {
		if _, err := (Puller{Root: root}).PullDirectory(name, directory); err != nil {
			t.Fatal(err)
		}
	}
	merged, err := (Builder{Root: root}).Compose(context.Background(), "merged", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	closure, err := ModelClosure(merged)
	if err != nil {
		t.Fatal(err)
	}
	// Both parents trained on the first corpus, so the merge counts it twice and lists it once,
	// beside the second corpus and each parent's own run.
	uses := map[string]int{}
	for _, entry := range closure.Corpora {
		uses[entry.CorpusBOMSHA256] = entry.Uses
	}
	if !closure.Complete || len(closure.Corpora) != 4 || uses[first] != 2 || uses[second] != 1 {
		t.Fatalf("merged closure = %+v", closure)
	}
	pulled, err := Inspect(root, "bare")
	if err != nil {
		t.Fatal(err)
	}
	// A release that recorded no closure leaves its history unrecorded, not empty.
	if inherited, err := InheritedClosure(pulled); err != nil || inherited.Complete {
		t.Fatalf("closure of a release without one = %+v, %v", inherited, err)
	}
	if fresh := emptyClosure(true); !unionClosures(fresh).Complete {
		t.Fatal("a model trained from scratch has a complete, empty history")
	}
}

func TestPullRefusesAMalformedTrainingClosure(t *testing.T) {
	directory := releasePackage(t, strings.Repeat("a", 64), 0)
	if err := os.WriteFile(filepath.Join(directory, ClosureFile), []byte(`{"kind":"something-else"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := (Puller{Root: t.TempDir()}).PullDirectory("bad", directory); err == nil || !strings.Contains(err.Error(), "training closure") {
		t.Fatalf("malformed closure error = %v", err)
	}
}

func TestReleaseClosuresMustCoverTheirRuns(t *testing.T) {
	// A closure that leaves out the release's own run is refused on pull.
	directory := releasePackage(t, strings.Repeat("a", 64), 0)
	closure := emptyClosure(true)
	closure.Corpora = append(closure.Corpora, ClosureCorpus{CorpusBOMSHA256: strings.Repeat("1", 64), Uses: 1})
	data, err := json.Marshal(closure)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, ClosureFile), data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := (Puller{Root: t.TempDir()}).PullDirectory("omits", directory); err == nil || !strings.Contains(err.Error(), "leaves out corpus") {
		t.Fatalf("omitting closure error = %v", err)
	}
	// A closure that covers its run but invents an inherited corpus fails CheckClosure against its true start.
	root := t.TempDir()
	honest, padded := releasePackage(t, strings.Repeat("a", 64), 1), releasePackage(t, strings.Repeat("a", 64), 2)
	withClosure(t, honest)
	withClosure(t, padded, strings.Repeat("9", 64))
	for name, directory := range map[string]string{"honest": honest, "padded": padded} {
		if _, err := (Puller{Root: root}).PullDirectory(name, directory); err != nil {
			t.Fatal(err)
		}
	}
	for name, want := range map[string]bool{"honest": true, "padded": false} {
		inspection, err := Inspect(root, name)
		if err != nil {
			t.Fatal(err)
		}
		if err := CheckClosure(inspection, nil); (err == nil) != want {
			t.Fatalf("%s: CheckClosure = %v", name, err)
		}
	}
}

func TestPullKeepsTheRoundRecord(t *testing.T) {
	for name, content := range map[string]string{
		"kept":      `{"kind":"openwaldo-bom","schema":1,"subject":"join-round","run":"r","round":1}`,
		"malformed": `{"kind":"something-else"}`,
	} {
		directory := releasePackage(t, strings.Repeat("a", 64), 0)
		withClosure(t, directory)
		if err := os.WriteFile(filepath.Join(directory, RoundFile), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		pulled, err := (Puller{Root: t.TempDir()}).PullDirectory(name, directory)
		if name == "malformed" {
			if err == nil || !strings.Contains(err.Error(), RoundFile) {
				t.Fatalf("malformed round record error = %v", err)
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		kept := false
		for _, artifact := range pulled.Origin.Artifacts {
			kept = kept || artifact.Role == RoundRole
		}
		if !kept {
			t.Fatalf("round record not kept: %+v", pulled.Origin.Artifacts)
		}
	}
}

// releasePackage writes an exported package whose release BOM records a
// lineage node descending from parent; salt changes one weight byte so
// packages can hold different weights.
func releasePackage(t *testing.T, parent string, salt byte) string {
	t.Helper()
	architecture := Architecture{Family: "decoder-transformer", ContextTokens: 32, VocabularySize: 259, HiddenSize: 4, IntermediateSize: 8, Layers: 1, AttentionHeads: 2, KeyValueHeads: 1, TieEmbeddings: true, ParameterDType: "bfloat16", Tokenizer: Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"}}
	return releasePackageOf(t, architecture, nil, parent, salt)
}

// releasePackageOf is releasePackage for architecture, with extra package files.
func releasePackageOf(t *testing.T, architecture Architecture, extra map[string][]byte, parent string, salt byte) string {
	t.Helper()
	directory := t.TempDir()
	files := huggingFaceFixture(t, architecture, HuggingFaceTokenizerClass)
	for name, data := range extra {
		files[name] = data
	}
	weights := files["model.safetensors"]
	// Keep each bf16 value below 2, so averaging them stays finite.
	for index := 8 + int(binary.LittleEndian.Uint64(weights)) + 1; index < len(weights); index += 2 {
		weights[index] &= 0x3f
	}
	weights[len(weights)-1] ^= salt
	// Published run BOMs are pinned, like a model's runs, by the hash of their canonical JSON.
	runBOM := RunBOM{Kind: "openwaldo-bom", Subject: "training-run", ID: fmt.Sprintf("run-%d", salt),
		CorpusBOM: corpus.BOM{Kind: "openwaldo-bom", Schema: 1, Subject: "corpus", Paths: []string{fmt.Sprintf("code/run-%d", salt)}}}
	corpusDigest, err := hashJSON(runBOM.CorpusBOM)
	if err != nil {
		t.Fatal(err)
	}
	runBOM.CorpusBOMSHA256 = corpusDigest
	runDigest, err := hashJSON(runBOM)
	if err != nil {
		t.Fatal(err)
	}
	runJSON, err := json.MarshalIndent(runBOM, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	files["provenance-"+runDigest+".json"] = runJSON
	node := LineageNode{WeightsSHA256: strings.Repeat(string("0123456789"[salt]), 64), Parents: []LineageLink{{WeightsSHA256: parent}}, Runs: []string{runDigest}}
	bom, err := json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": node, "artifacts": inventory(map[string][]byte{"model.safetensors": files["model.safetensors"]})})
	if err != nil {
		t.Fatal(err)
	}
	files["BOM.json"] = bom
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(directory, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return directory
}

func TestPullRefusesProvenanceThatDoesNotMatchItsName(t *testing.T) {
	directory := releasePackage(t, strings.Repeat("a", 64), 0)
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "provenance-") {
			if err := os.WriteFile(filepath.Join(directory, entry.Name()), []byte(`{"tampered":true}`), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	if _, err := (Puller{Root: t.TempDir()}).PullDirectory("tampered", directory); err == nil || !strings.Contains(err.Error(), "does not hash") {
		t.Fatalf("tampered provenance error = %v", err)
	}
}

// A trained model's release publishes every run it trained, while its node
// names only the newest. Pulling it keeps them all; a release whose node names
// no run keeps none, since those runs belong to the releases it came from.
func TestPullKeepsEveryRunATrainedReleasePublishes(t *testing.T) {
	directory := releasePackage(t, strings.Repeat("a", 64), 0)
	earlier := RunBOM{Kind: "openwaldo-bom", Subject: "training-run", ID: "run-earlier",
		CorpusBOM: corpus.BOM{Kind: "openwaldo-bom", Schema: 1, Subject: "corpus", Paths: []string{"code/run-earlier"}}}
	corpusDigest, err := hashJSON(earlier.CorpusBOM)
	if err != nil {
		t.Fatal(err)
	}
	earlier.CorpusBOMSHA256 = corpusDigest
	earlierDigest, err := hashJSON(earlier)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.MarshalIndent(earlier, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "provenance-"+earlierDigest+".json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]int{"trained": 2, "untrained": 0} {
		var release struct {
			Lineage LineageNode `json:"lineage"`
		}
		bomPath := filepath.Join(directory, "BOM.json")
		original, err := os.ReadFile(bomPath)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(original, &release); err != nil {
			t.Fatal(err)
		}
		if name == "untrained" {
			release.Lineage.Runs = nil
		}
		weights, err := os.ReadFile(filepath.Join(directory, "model.safetensors"))
		if err != nil {
			t.Fatal(err)
		}
		bom, err := json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": release.Lineage, "artifacts": inventory(map[string][]byte{"model.safetensors": weights})})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(bomPath, bom, 0o644); err != nil {
			t.Fatal(err)
		}
		pulled, err := (Puller{Root: t.TempDir()}).PullDirectory(name, directory)
		if err != nil {
			t.Fatal(err)
		}
		records, err := ProvenanceRecords(pulled)
		if err != nil || len(records) != want {
			t.Fatalf("%s release keeps %d run BOMs, want %d: %v", name, len(records), want, err)
		}
		if err := os.WriteFile(bomPath, original, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// A merge carries its parents' published run BOMs, so its release proves the round's data.
func TestMergeCarriesItsParentsProvenance(t *testing.T) {
	common := strings.Repeat("a", 64)
	root := t.TempDir()
	for name, directory := range map[string]string{"left": releasePackage(t, common, 0), "right": releasePackage(t, common, 1)} {
		if _, err := (Puller{Root: root}).PullDirectory(name, directory); err != nil {
			t.Fatal(err)
		}
	}
	merged, err := (Builder{Root: root}).Compose(t.Context(), "merged", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Merge: []ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	carried, err := ProvenanceRecords(merged)
	if err != nil || len(carried) != 2 {
		t.Fatalf("merged provenance = %v, %v", carried, err)
	}
	for _, name := range []string{"left", "right"} {
		parent, err := Inspect(root, name)
		if err != nil {
			t.Fatal(err)
		}
		release, _, err := keptReleaseLineage(parent)
		if err != nil || release == nil {
			t.Fatalf("%s release lineage = %v, %v", name, release, err)
		}
		records, err := ProvenanceRecords(parent)
		if err != nil || len(records) != 1 {
			t.Fatalf("%s provenance = %v, %v", name, records, err)
		}
		for digest := range records {
			if _, ok := carried[digest]; !ok || !slices.Contains(release.Runs, digest) {
				t.Fatalf("%s run %s was not carried", name, digest)
			}
		}
	}
}
