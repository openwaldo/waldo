// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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
	first, second := strings.Repeat("1", 64), strings.Repeat("2", 64)
	left, right, _ := familyPackages(t, lineageArchitecture(), nil)
	bare := releasePackage(t, 2)
	withClosure(t, left, first)
	withClosure(t, right, first, second)
	root := t.TempDir()
	for name, directory := range map[string]string{"left": left, "right": right, "bare": bare} {
		if _, err := pullDirectory(t, root, name, directory); err != nil {
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
	// beside the second corpus and the parents' own runs, which share a corpus.
	uses := map[string]int{}
	for _, entry := range closure.Corpora {
		uses[entry.CorpusBOMSHA256] = entry.Uses
	}
	if !closure.Complete || len(closure.Corpora) != 3 || uses[first] != 2 || uses[second] != 1 {
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
	directory := releasePackage(t, 0)
	if err := os.WriteFile(filepath.Join(directory, ClosureFile), []byte(`{"kind":"something-else"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := pullDirectory(t, t.TempDir(), "bad", directory); err == nil || !strings.Contains(err.Error(), "training closure") {
		t.Fatalf("malformed closure error = %v", err)
	}
}

func TestReleaseClosuresMustCoverTheirRuns(t *testing.T) {
	// A closure that leaves out the release's own run is refused on pull.
	directory := releasePackage(t, 0)
	closure := emptyClosure(true)
	closure.Corpora = append(closure.Corpora, ClosureCorpus{CorpusBOMSHA256: strings.Repeat("1", 64), Uses: 1})
	data, err := json.Marshal(closure)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, ClosureFile), data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := pullDirectory(t, t.TempDir(), "omits", directory); err == nil || !strings.Contains(err.Error(), "leaves out corpus") {
		t.Fatalf("omitting closure error = %v", err)
	}
	// A closure that covers its run but invents an inherited corpus fails CheckClosure against its true start.
	root := t.TempDir()
	honest, padded, _ := familyPackages(t, lineageArchitecture(), nil)
	withClosure(t, honest)
	withClosure(t, padded, strings.Repeat("9", 64))
	for name, directory := range map[string]string{"honest": honest, "padded": padded} {
		if _, err := pullDirectory(t, root, name, directory); err != nil {
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
		directory := releasePackage(t, 0)
		withClosure(t, directory)
		if err := os.WriteFile(filepath.Join(directory, RoundFile), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		pulled, err := pullDirectory(t, t.TempDir(), name, directory)
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

// releasePackage writes an exported package with one published run BOM and
// no history; salt changes one weight byte so packages hold different weights.
func releasePackage(t *testing.T, salt byte) string {
	t.Helper()
	directory := t.TempDir()
	files := mergeableFixture(t, lineageArchitecture(), salt)
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
	if files["provenance-"+runDigest+".json"], err = json.MarshalIndent(runBOM, "", "  "); err != nil {
		t.Fatal(err)
	}
	writePackage(t, directory, files)
	return directory
}

func writePackage(t *testing.T, directory string, files map[string][]byte) {
	t.Helper()
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(directory, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// exportPackage writes a model's release as join publishes a unit: its
// weights, history, and every run BOM it trained, beside extra files.
func exportPackage(t *testing.T, root string, inspection Inspection, extra map[string][]byte) string {
	t.Helper()
	files := releaseFiles(t, root, inspection, true, nil)
	for name, data := range extra {
		files[name] = data
	}
	for _, pin := range inspection.Model.Runs {
		data, err := os.ReadFile(filepath.Join(inspection.Path, "runs", runDirectoryName(pin), "RUN-BOM.json"))
		if err != nil {
			t.Fatal(err)
		}
		files["provenance-"+pin.BOMSHA256+".json"] = data
	}
	directory := t.TempDir()
	writePackage(t, directory, files)
	return directory
}

// familyPackages exports two fine-tunes of one pulled base, as two units of
// a join round; the base package is returned too.
func familyPackages(t *testing.T, architecture Architecture, extra map[string][]byte) (string, string, string) {
	t.Helper()
	exporter := t.TempDir()
	files := mergeableFixture(t, architecture, 0)
	for name, data := range extra {
		files[name] = data
	}
	pullPlain(t, exporter, "base", files)
	builder := flipWeightsBuilder(exporter)
	left := tune(t, builder, ComposeBase{Model: "base"}, "left", "left")
	right := tune(t, builder, ComposeBase{Model: "base"}, "right", "right")
	base, err := Inspect(exporter, "base")
	if err != nil {
		t.Fatal(err)
	}
	return exportPackage(t, exporter, left, extra), exportPackage(t, exporter, right, extra), exportPackage(t, exporter, base, extra)
}

// pullDirectory imports a package as join does, named by its run and name.
func pullDirectory(t *testing.T, root, name, directory string) (Inspection, error) {
	t.Helper()
	return (Puller{Root: root}).PullDirectory(name, directory, OriginSource{Provider: "waldo-join", Repository: "runs/test/" + name})
}

func TestPullRefusesProvenanceThatDoesNotMatchItsName(t *testing.T) {
	directory := releasePackage(t, 0)
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
	if _, err := pullDirectory(t, t.TempDir(), "tampered", directory); err == nil || !strings.Contains(err.Error(), "does not hash") {
		t.Fatalf("tampered provenance error = %v", err)
	}
}

// A trained model's release publishes every run it trained; pulling it
// keeps them all. An untrained origin's release keeps none, since those runs
// belong to the releases it came from.
func TestPullKeepsEveryRunATrainedReleasePublishes(t *testing.T) {
	left, _, base := familyPackages(t, lineageArchitecture(), nil)
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
	if err := os.WriteFile(filepath.Join(left, "provenance-"+earlierDigest+".json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]struct {
		directory string
		runs      int
	}{"trained": {left, 2}, "untrained": {base, 0}} {
		pulled, err := pullDirectory(t, t.TempDir(), name, want.directory)
		if err != nil {
			t.Fatal(err)
		}
		records, err := ProvenanceRecords(pulled)
		if err != nil || len(records) != want.runs {
			t.Fatalf("%s release keeps %d run BOMs, want %d: %v", name, len(records), want.runs, err)
		}
	}
}

// A merge carries its parents' published run BOMs, so its release proves the round's data.
func TestMergeCarriesItsParentsProvenance(t *testing.T) {
	left, right, _ := familyPackages(t, lineageArchitecture(), nil)
	root := t.TempDir()
	for name, directory := range map[string]string{"left": left, "right": right} {
		if _, err := pullDirectory(t, root, name, directory); err != nil {
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
		records, err := ProvenanceRecords(parent)
		if err != nil || len(records) != 1 {
			t.Fatalf("%s provenance = %v, %v", name, records, err)
		}
		for digest := range records {
			if _, ok := carried[digest]; !ok {
				t.Fatalf("%s run %s was not carried", name, digest)
			}
		}
	}
}
