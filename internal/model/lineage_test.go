// Copyright (c) 2026 OpenWALDO Project contributors
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
	"sort"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

// lineageArchitecture is small enough to export and pull back.
func lineageArchitecture() Architecture {
	return Architecture{Family: "decoder-transformer", ContextTokens: 32, VocabularySize: 259, HiddenSize: 4, IntermediateSize: 8, Layers: 1, AttentionHeads: 2, KeyValueHeads: 1, TieEmbeddings: true, ParameterDType: "bfloat16", Tokenizer: Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"}}
}

// waldoWeights writes Safetensors as WALDO's trainers do: WALDO tensor names,
// sorted, with metadata; the payload depends on seed.
func waldoWeights(architecture Architecture, metadata map[string]string, seed string) []byte {
	header := map[string]any{"__metadata__": metadata}
	shapes := expectedTensorShapes(architecture)
	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	sort.Strings(names)
	var offset uint64
	for _, name := range names {
		elements := uint64(1)
		for _, dimension := range shapes[name] {
			elements *= dimension
		}
		header[name] = map[string]any{"dtype": "BF16", "shape": shapes[name], "data_offsets": []uint64{offset, offset + elements*2}}
		offset += elements * 2
	}
	encoded, _ := json.Marshal(header)
	for len(encoded)%8 != 0 {
		encoded = append(encoded, ' ')
	}
	data := binary.LittleEndian.AppendUint64(nil, uint64(len(encoded)))
	data = append(data, encoded...)
	for index := uint64(0); index < offset; index++ {
		data = append(data, seed[index%uint64(len(seed))])
	}
	return data
}

// lineageBuilder trains with a backend that writes real WALDO weights.
func lineageBuilder(root string) Builder {
	next := 0
	backend := backendFunc(func(_ context.Context, request training.Request) (training.Observation, error) {
		data := waldoWeights(lineageArchitecture(), map[string]string{"format": "openwaldo", "run_id": request.RunID, "architecture_sha256": request.ArchitectureSHA256}, request.RunID)
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
	return Builder{
		Root: root,
		NewID: func() (string, error) {
			next++
			return fmt.Sprintf("run%04d", next), nil
		},
		Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
			return testSelection(backend), nil
		}),
	}
}

func lineageStage(name string) Stage {
	stage := testStage(name)
	stage.Parameters.SequenceLength = 32
	return stage
}

// trainTwice makes a model with two runs, the second started from the first.
func trainTwice(t *testing.T, builder Builder, name string) Inspection {
	t.Helper()
	if _, err := builder.Initialize(name, lineageArchitecture()); err != nil {
		t.Fatal(err)
	}
	for _, stage := range []string{"first", "second"} {
		if _, err := builder.Train(context.Background(), name, preparedFixture(t, lineageStage(stage))); err != nil {
			t.Fatal(err)
		}
	}
	inspection, err := Inspect(builder.Root, name)
	if err != nil {
		t.Fatal(err)
	}
	return inspection
}

// fineTune composes a new model from base's current run.
func fineTune(t *testing.T, builder Builder, base, name string, warnings *[]string) Inspection {
	t.Helper()
	builder.Progress = func(progress Progress) {
		if progress.Phase == "lineage" && warnings != nil {
			*warnings = append(*warnings, progress.Message)
		}
	}
	stage := lineageStage(name)
	compose, err := builder.ResolveCompose(context.Background(), Compose{Kind: "waldo-model-compose", Schema: 1, Base: &ComposeBase{Model: base}, Stages: []Stage{stage}}, false)
	if err != nil {
		t.Fatal(err)
	}
	inspection, err := builder.Compose(context.Background(), name, compose, []PreparedStage{preparedFixture(t, stage)})
	if err != nil {
		t.Fatal(err)
	}
	return inspection
}

func modelHistory(root string, inspection Inspection, runID string) *lineageWalk {
	walk := newLineageWalk(newLineageIndex(root, nil))
	walk.model(inspection, runID)
	return walk
}

// holdsRuns fails unless a history holds every run BOM of these models.
func holdsRuns(t *testing.T, walk *lineageWalk, models ...Inspection) {
	t.Helper()
	if len(walk.gaps) != 0 {
		t.Fatalf("history has gaps: %v", walk.sortedGaps())
	}
	for _, model := range models {
		for _, run := range model.BOM.Runs {
			if _, ok := walk.held[run.BOMSHA256]; !ok {
				t.Fatalf("model %s run %s isn't in the history", model.Model.Name, run.ID)
			}
		}
	}
}

func TestFineTuneHistorySurvivesDeletingItsParents(t *testing.T) {
	root := t.TempDir()
	builder := lineageBuilder(root)
	base := trainTwice(t, builder, "base")
	child := fineTune(t, builder, "base", "child", nil)
	grandchild := fineTune(t, builder, "child", "grandchild", nil)
	if _, err := os.Stat(filepath.Join(child.Path, lineageFile)); err != nil {
		t.Fatalf("a fine-tune has no history file: %v", err)
	}
	for _, name := range []string{"base", "child"} {
		if err := os.RemoveAll(filepath.Join(root, name)); err != nil {
			t.Fatal(err)
		}
	}
	holdsRuns(t, modelHistory(root, grandchild, grandchild.BOM.CurrentRunID), base, child, grandchild)
}

func TestDamagedHistoryIsAGapNotAnError(t *testing.T) {
	for name, damage := range map[string]func([]byte) []byte{
		"truncated": func(data []byte) []byte { return data[:len(data)/2] },
		"altered document": func(data []byte) []byte {
			var file LineageFile
			if err := json.Unmarshal(data, &file); err != nil {
				panic(err)
			}
			for digest, document := range file.Documents {
				if strings.Contains(string(document), `"training-run"`) {
					file.Documents[digest] = []byte(strings.Replace(string(document), `"training-run"`, `"training-rum"`, 1))
				}
			}
			data, _ = json.Marshal(file)
			return data
		},
	} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			builder := lineageBuilder(root)
			trainTwice(t, builder, "base")
			child := fineTune(t, builder, "base", "child", nil)
			path := filepath.Join(child.Path, lineageFile)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, damage(data), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.RemoveAll(filepath.Join(root, "base")); err != nil {
				t.Fatal(err)
			}
			var warnings []string
			grandchild := fineTune(t, builder, "child", "grandchild", &warnings)
			if !strings.Contains(strings.Join(warnings, "\n"), "gap") {
				t.Fatalf("no warning about the damaged history: %q", warnings)
			}
			walk := modelHistory(root, grandchild, grandchild.BOM.CurrentRunID)
			if !walk.gaps[LineageGap{Gap: child.Model.Parent.RunBOMSHA256, Link: "run-bom"}] {
				t.Fatalf("the damaged base run isn't a gap: %v", walk.sortedGaps())
			}
		})
	}
}

func TestAParentPinMustNameItsRun(t *testing.T) {
	for name, bom := range map[string]RunBOM{
		"another model": {ModelID: "another-model", ID: "expected-run"},
		"another run":   {ModelID: "expected-model", ID: "another-run"},
	} {
		t.Run(name, func(t *testing.T) {
			walk := newLineageWalk(newLineageIndex("", nil))
			digest, err := walk.index.addRecord(bom)
			if err != nil {
				t.Fatal(err)
			}
			walk.parent(ModelParent{ModelID: "expected-model", RunID: "expected-run", RunBOMSHA256: digest})
			if !walk.gaps[LineageGap{Gap: digest, Link: "run-bom"}] {
				t.Fatalf("a run BOM for %s/%s passed as expected-model/expected-run: %v", bom.ModelID, bom.ID, walk.sortedGaps())
			}
		})
	}
}

func TestAReleaseMustNameItsModelBOM(t *testing.T) {
	walk := newLineageWalk(newLineageIndex("", nil))
	add := func(record any) string {
		digest, err := walk.index.addRecord(record)
		if err != nil {
			t.Fatal(err)
		}
		return digest
	}
	origin := add(OriginBOM{Subject: "shared base"})
	x, y := add(Plan{Name: "x", OriginBOMSHA256: origin}), add(Plan{Name: "y", OriginBOMSHA256: origin})
	other := add(ModelBOM{ModelID: y, Origin: &ModelBOMOrigin{SHA256: origin}})
	release := add(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "model_id": x, "source_bom_sha256": other, "artifacts": []map[string]string{{"role": "weights", "sha256": "w"}}})
	walk.release(release, OriginBOM{SourceArtifacts: []OriginArtifact{{Role: "source-weights", SHA256: "w"}}})
	if len(walk.anchors[x]) != 0 || !walk.gaps[LineageGap{Gap: other, Link: "model-bom"}] {
		t.Fatalf("model y's BOM was linked as model x's: anchors %v, gaps %v", walk.anchors, walk.sortedGaps())
	}
}

func TestAParentPinMustNameItsRunsWeights(t *testing.T) {
	walk := newLineageWalk(newLineageIndex("", nil))
	add := func(record any) string {
		digest, err := walk.index.addRecord(record)
		if err != nil {
			t.Fatal(err)
		}
		return digest
	}
	parent := add(Plan{Name: "parent"})
	parentRun := add(RunBOM{ModelID: parent, ID: "r"})
	parentBOM := add(ModelBOM{ModelID: parent, Runs: []ModelBOMRun{{ID: "r", BOMSHA256: parentRun, Artifacts: []ModelBOMArtifact{{Role: "weights", SHA256: "real"}}}}})
	walk.index.claims[parent] = []string{parentBOM}
	child := add(Plan{Name: "child", Parent: &ModelParent{ModelID: parent, RunID: "r", RunBOMSHA256: parentRun, Artifact: training.Artifact{SHA256: "fake"}}})
	childRun := add(RunBOM{ModelID: child, ID: "c", Initialization: &training.Initialization{SourceType: "run", SourceID: parent, SourceRunID: "r", Artifact: training.Artifact{SHA256: "fake"}}})
	walk.run(childRun, child, "c")
	if len(walk.gaps) == 0 {
		t.Fatalf("a child that started from weights its parent run never made has no gap; held %d documents", len(walk.held))
	}
}

func TestComposeFillsAMissingHistory(t *testing.T) {
	root := t.TempDir()
	builder := lineageBuilder(root)
	trainTwice(t, builder, "base")
	child := fineTune(t, builder, "base", "child", nil)
	path := filepath.Join(child.Path, lineageFile)
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// A compose interrupted before its history was written, or started by
	// an earlier WALDO, writes it when it resumes; an existing one stays.
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := fillLineage(root, child, nil, func(string) {}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != string(want) {
		t.Fatalf("refilled history differs: %v", err)
	}
	if err := os.WriteFile(path, []byte("kept"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := fillLineage(root, child, nil, func(string) {}); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); string(got) != "kept" {
		t.Fatal("an existing history file was rewritten")
	}
}

// releaseFiles makes a Hugging Face release of a model, with lineage when withLineage is set.
func releaseFiles(t *testing.T, root string, inspection Inspection, withLineage bool, change func([]byte) []byte) map[string][]byte {
	t.Helper()
	files := huggingFaceFixture(t, lineageArchitecture(), HuggingFaceTokenizerClass)
	sourceType, sourceID := "origin", inspection.Model.OriginBOMSHA256
	if inspection.BOM.CurrentRunID != "" {
		sourceType, sourceID = "run", inspection.BOM.CurrentRunID
	}
	weights := files["model.safetensors"]
	if change != nil {
		weights = change(weights)
	}
	files["model.safetensors"] = weights
	type artifact struct {
		Path   string `json:"path"`
		Role   string `json:"role"`
		SHA256 string `json:"sha256"`
	}
	artifacts := []artifact{{Path: "model.safetensors", Role: "weights", SHA256: sha256Hex(weights)}}
	if withLineage {
		lineage, err := ReleaseLineage(root, inspection, nil)
		if err != nil {
			t.Fatal(err)
		}
		files[lineageFile] = lineage
		artifacts = append(artifacts, artifact{Path: lineageFile, Role: "lineage", SHA256: sha256Hex(lineage)})
	}
	sourceBOM, err := hashJSON(inspection.BOM)
	if err != nil {
		t.Fatal(err)
	}
	files["BOM.json"], err = json.MarshalIndent(map[string]any{
		"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "format": "huggingface",
		"model_id": inspection.Model.ID, "name": inspection.Model.Name, "source_type": sourceType, "source_id": sourceID,
		"run_id": inspection.BOM.CurrentRunID, "source_bom_sha256": sourceBOM, "artifacts": artifacts,
	}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return files
}

func pullRelease(t *testing.T, root string, files map[string][]byte, warnings *[]string) Inspection {
	t.Helper()
	isolateHuggingFaceToken(t)
	client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("a", 40))}
	puller := Puller{Root: root, Endpoint: "https://hub.test", Client: client, Progress: func(progress PullProgress) {
		if progress.Phase == "lineage" && warnings != nil {
			*warnings = append(*warnings, progress.Message)
		}
	}}
	inspection, err := puller.Pull(context.Background(), "pulled", "huggingface://org/tiny")
	if err != nil {
		t.Fatal(err)
	}
	return inspection
}

func TestPullLinksAReleaseToTheRunThatMadeIt(t *testing.T) {
	exporter := t.TempDir()
	base := trainTwice(t, lineageBuilder(exporter), "base")
	files := releaseFiles(t, exporter, base, true, nil)
	root := t.TempDir()
	var warnings []string
	pulled := pullRelease(t, root, files, &warnings)
	if len(warnings) != 0 {
		t.Fatalf("pull warned: %q", warnings)
	}
	for _, artifact := range pulled.Origin.SourceArtifacts {
		if artifact.Path == lineageFile {
			t.Fatal("the origin pins LINEAGE.json, so its ID differs from an earlier WALDO's pull")
		}
	}
	builder := lineageBuilder(root)
	builder.NewID = func() (string, error) { return "tuned0001", nil }
	tuned := fineTune(t, builder, "pulled", "tuned", nil)
	if err := os.RemoveAll(pulled.Path); err != nil {
		t.Fatal(err)
	}
	walk := modelHistory(root, tuned, tuned.BOM.CurrentRunID)
	holdsRuns(t, walk, base, tuned)

	// An earlier WALDO's release has no LINEAGE.json, so its history is a gap.
	files = releaseFiles(t, exporter, base, false, nil)
	warnings = nil
	plain := pullRelease(t, t.TempDir(), files, &warnings)
	if plain.Model.OriginBOMSHA256 == pulled.Model.OriginBOMSHA256 {
		t.Fatal("fixture: the releases should differ in BOM.json")
	}
	if !strings.Contains(strings.Join(warnings, "\n"), "gap") {
		t.Fatalf("warnings = %q", warnings)
	}
}

func TestFineTuneOfAPulledFineTuneKeepsItsHistory(t *testing.T) {
	exporter := t.TempDir()
	builder := lineageBuilder(exporter)
	base := trainTwice(t, builder, "base")
	child := fineTune(t, builder, "base", "child", nil)
	root := t.TempDir()
	pulled := pullRelease(t, root, releaseFiles(t, exporter, child, true, nil), nil)
	tuned := fineTune(t, lineageBuilder(root), "pulled", "tuned", nil)
	if err := os.RemoveAll(pulled.Path); err != nil {
		t.Fatal(err)
	}
	holdsRuns(t, modelHistory(root, tuned, tuned.BOM.CurrentRunID), base, child, tuned)
}

func TestPullLinksAReleaseOfAnUntrainedOrigin(t *testing.T) {
	exporter := t.TempDir()
	base := trainTwice(t, lineageBuilder(exporter), "base")
	middle := t.TempDir()
	pulled := pullRelease(t, middle, releaseFiles(t, exporter, base, true, nil), nil)
	root := t.TempDir()
	var warnings []string
	again := pullRelease(t, root, releaseFiles(t, middle, pulled, true, nil), &warnings)
	if len(warnings) != 0 {
		t.Fatalf("pull warned: %q", warnings)
	}
	walk := modelHistory(root, again, "")
	holdsRuns(t, walk, base)
}

func TestPullOriginMatchesAPullThatIgnoresLineage(t *testing.T) {
	exporter := t.TempDir()
	base := trainTwice(t, lineageBuilder(exporter), "base")
	files := releaseFiles(t, exporter, base, true, nil)
	with := pullRelease(t, t.TempDir(), files, nil)
	delete(files, lineageFile) // listed in BOM.json, but not fetched, as by an earlier WALDO
	without := pullRelease(t, t.TempDir(), files, nil)
	if with.Model.OriginBOMSHA256 != without.Model.OriginBOMSHA256 || with.Model.ID != without.Model.ID {
		t.Fatal("fetching the release history changed the pulled model's identity")
	}
}

func TestAReleaseBOMLinksOnlyItsOwnWeights(t *testing.T) {
	exporter := t.TempDir()
	base := trainTwice(t, lineageBuilder(exporter), "base")
	files := releaseFiles(t, exporter, base, true, nil)
	root := t.TempDir()
	pullRelease(t, root, files, nil)
	// A second release copies BOM.json byte for byte but ships other weights.
	copied := releaseFiles(t, exporter, base, true, func(weights []byte) []byte {
		changed := append([]byte(nil), weights...)
		changed[len(changed)-1] ^= 1
		return changed
	})
	copied["BOM.json"] = files["BOM.json"]
	isolateHuggingFaceToken(t)
	client := &http.Client{Transport: huggingFaceTransport(t, copied, strings.Repeat("a", 40))}
	other, err := (Puller{Root: root, Endpoint: "https://hub.test", Client: client}).Pull(context.Background(), "other", "huggingface://org/tiny")
	if err != nil {
		t.Fatal(err)
	}
	walk := modelHistory(root, other, "")
	if _, ok := walk.held[base.BOM.Runs[1].BOMSHA256]; ok || len(walk.gaps) == 0 {
		t.Fatalf("other weights were linked through a copied BOM.json: gaps %v", walk.sortedGaps())
	}
}

func TestAReleaseNeverFailsAPull(t *testing.T) {
	exporter := t.TempDir()
	base := trainTwice(t, lineageBuilder(exporter), "base")
	files := releaseFiles(t, exporter, base, true, nil)
	files[lineageFile] = []byte("junk")
	var warnings []string
	pulled := pullRelease(t, t.TempDir(), files, &warnings)
	if !strings.Contains(strings.Join(warnings, "\n"), "doesn't match BOM.json") || pulled.Model.ID == "" {
		t.Fatalf("warnings = %q", warnings)
	}
}
