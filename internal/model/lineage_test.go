// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"crypto/sha256"
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

func releaseArchitecture() Architecture {
	return Architecture{Family: "decoder-transformer", ContextTokens: 32, VocabularySize: 259, HiddenSize: 4, IntermediateSize: 8, Layers: 1, AttentionHeads: 2, KeyValueHeads: 1, TieEmbeddings: true, ParameterDType: "bfloat16", Tokenizer: Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"}}
}

// pullRelease pulls a release whose BOM.json declares node, with ancestors
// behind it, as the lineage its weights came from, or with no BOM.json when
// node is empty.
func pullRelease(t *testing.T, root, name string, node LineageNode, ancestors ...LineageNode) {
	t.Helper()
	files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
	if node.WeightsSHA256 != "" {
		release := map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": node, "artifacts": inventory(files)}
		if len(ancestors) > 0 {
			release["lineage_ancestors"] = ancestors
		}
		bom, err := json.Marshal(release)
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

// inventory is a release BOM's artifacts list for files.
func inventory(files map[string][]byte) []OriginArtifact {
	var artifacts []OriginArtifact
	for path, data := range files {
		digest := sha256.Sum256(data)
		artifacts = append(artifacts, OriginArtifact{Role: "weights", Path: path, SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))})
	}
	return artifacts
}

// releaseStage is a stage that fits releaseArchitecture's context.
func releaseStage(name string) Stage {
	stage := testStage(name)
	stage.Parameters.SequenceLength = 32
	stage.Parameters.Steps *= 2
	return stage
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

func lineageHash(t *testing.T, node LineageNode) string {
	t.Helper()
	digest, err := hashJSON(node)
	if err != nil {
		t.Fatal(err)
	}
	return digest
}

// Each run's node names the weights it produced, the run that produced them,
// and the node of the weights it started from, like a git commit.
func TestEachRunRecordsItsOwnLineageNode(t *testing.T) {
	root := t.TempDir()
	trainTwice(t, realWeightsBuilder(root))
	p, err := Inspect(root, "p")
	if err != nil {
		t.Fatal(err)
	}
	first, err := ModelLineage(p, p.Model.Runs[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ModelLineage(p, "")
	if err != nil {
		t.Fatal(err)
	}
	if first.WeightsSHA256 != p.Model.Runs[0].Artifacts[0].SHA256 || !reflect.DeepEqual(first.Runs, []string{p.Model.Runs[0].BOMSHA256}) {
		t.Fatalf("first run's node = %+v", first)
	}
	want := []LineageLink{{WeightsSHA256: first.WeightsSHA256, LineageSHA256: lineageHash(t, first)}}
	if second.WeightsSHA256 != p.Model.Runs[1].Artifacts[0].SHA256 || !reflect.DeepEqual(second.Parents, want) || !reflect.DeepEqual(second.Runs, []string{p.Model.Runs[1].BOMSHA256}) {
		t.Fatalf("second run's node = %+v, want parent %+v", second, want)
	}
}

// A fine-tune pinned to an earlier run starts from that run's node: its
// weights and only the run that made them.
func TestComposePinnedToEarlierRunStartsFromThatRunsNode(t *testing.T) {
	root := t.TempDir()
	builder := realWeightsBuilder(root)
	first := trainTwice(t, builder)
	stage := testStage("child")
	child, err := builder.Compose(t.Context(), "child", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p", RunID: first}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)})
	if err != nil {
		t.Fatal(err)
	}
	p, err := Inspect(root, "p")
	if err != nil {
		t.Fatal(err)
	}
	pinned, err := ModelLineage(p, first)
	if err != nil {
		t.Fatal(err)
	}
	node, history, err := LineageHistory(child, "")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(node.Parents, []LineageLink{{WeightsSHA256: pinned.WeightsSHA256, LineageSHA256: lineageHash(t, pinned)}}) || !reflect.DeepEqual(history, []LineageNode{pinned}) {
		t.Fatalf("child's node %+v and history %+v, want to start from %+v", node, history, pinned)
	}
}

// A fine-tune holds its parent's whole chain, so its history stays complete
// once the parent model is deleted.
func TestFineTuneKeepsItsHistoryWithoutItsParent(t *testing.T) {
	root := t.TempDir()
	builder := realWeightsBuilder(root)
	trainTwice(t, builder)
	p, err := Inspect(root, "p")
	if err != nil {
		t.Fatal(err)
	}
	second, history, err := LineageHistory(p, "")
	if err != nil {
		t.Fatal(err)
	}
	stage := testStage("child")
	if _, err := builder.Compose(t.Context(), "child", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	if _, err := Remove(root, []string{"p"}); err != nil {
		t.Fatal(err)
	}
	child, err := Inspect(root, "child")
	if err != nil {
		t.Fatal(err)
	}
	node, chain, err := LineageHistory(child, "")
	if err != nil {
		t.Fatal(err)
	}
	want := append([]LineageNode{second}, history...)
	if len(want) != 2 || !reflect.DeepEqual(node.Parents, []LineageLink{{WeightsSHA256: second.WeightsSHA256, LineageSHA256: lineageHash(t, second)}}) || !reflect.DeepEqual(chain, want) {
		t.Fatalf("child's node %+v and history %+v, want %+v", node, chain, want)
	}
}

// Lineage lives beside the records, which stay exactly as earlier WALDO
// versions write and read them.
func TestLineageLeavesRecordsAsEarlierVersionsReadThem(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	builder := realWeightsBuilder(root)
	trainTwice(t, builder)
	stage := testStage("child")
	if _, err := builder.Compose(t.Context(), "child", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatal(err)
	}
	pullRelease(t, root, "pulled", LineageNode{WeightsSHA256: strings.Repeat("a", 64)})
	for _, model := range []string{"child", "pulled"} {
		err := filepath.WalkDir(filepath.Join(root, model), func(path string, entry os.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(path, ".json") || strings.HasSuffix(path, LineageFile) || strings.HasSuffix(path, filepath.FromSlash(ReleaseBOMPath)) {
				return err
			}
			data, err := os.ReadFile(path)
			if err == nil && strings.Contains(string(data), "lineage") {
				t.Errorf("%s records lineage", path)
			}
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "child", LineageFile)); err != nil {
		t.Fatalf("child has no %s: %v", LineageFile, err)
	}
}

// A LINEAGE.json is refused unless its parent node names the weights the
// parent pin names and its ancestors are exactly the chain behind it. Fewer
// ancestors is a shorter history, not a different one; a model made before
// lineage starts from its parent's weights alone.
func TestFineTuneLineageMustMatchItsParent(t *testing.T) {
	root := t.TempDir()
	builder := realWeightsBuilder(root)
	trainTwice(t, builder)
	stage := testStage("child")
	child, err := builder.Compose(t.Context(), "child", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "child", LineageFile)
	original, err := readLineageRecord(path)
	if err != nil {
		t.Fatal(err)
	}
	stray := LineageNode{WeightsSHA256: strings.Repeat("c", 64)}
	for name, change := range map[string]func(*lineageRecord){
		"other weights":    func(record *lineageRecord) { record.Parent.WeightsSHA256 = strings.Repeat("d", 64) },
		"other run":        func(record *lineageRecord) { record.Parent.Runs = []string{strings.Repeat("f", 64)} },
		"stray ancestor":   func(record *lineageRecord) { record.Ancestors = append(record.Ancestors, stray) },
		"altered ancestor": func(record *lineageRecord) { record.Ancestors[0].Runs = []string{strings.Repeat("e", 64)} },
		"another kind":     func(record *lineageRecord) { record.Kind = "waldo-model" },
	} {
		record := *original
		record.Ancestors = append([]LineageNode(nil), original.Ancestors...)
		change(&record)
		if err := writeJSONAtomic(path, record); err != nil {
			t.Fatal(err)
		}
		if _, err := ModelLineage(child, ""); err == nil {
			t.Fatalf("%s: lineage accepted", name)
		}
	}
	// A later schema's LINEAGE.json, like none at all, starts from the parent's weights alone.
	later := map[string]any{"kind": lineageKind, "schema": 2, "parent": map[string]any{"weights": "a later form"}}
	for name, write := range map[string]func() error{
		"later schema": func() error { return writeJSONAtomic(path, later) },
		"none":         func() error { return os.Remove(path) },
	} {
		if err := write(); err != nil {
			t.Fatal(err)
		}
		node, err := ModelLineage(child, "")
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !reflect.DeepEqual(node.Parents, []LineageLink{{WeightsSHA256: child.Model.Parent.Artifact.SHA256}}) {
			t.Fatalf("%s: starts from %+v", name, node.Parents)
		}
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
	current, err := resolveRunInitialization(p, "")
	if err != nil {
		t.Fatal(err)
	}
	if node.WeightsSHA256 != first || current.Artifact.SHA256 != first {
		t.Fatalf("lineage %s and current %s, want the first run's %s", node.WeightsSHA256, current.Artifact.SHA256, first)
	}
}

// LINEAGE.json is read exactly as any reader would read it: keys another
// reader could resolve differently, or a link in place of the file, are refused.
func TestLineageFileMustReadAlike(t *testing.T) {
	root := t.TempDir()
	builder := realWeightsBuilder(root)
	trainTwice(t, builder)
	stage := testStage("child")
	child, err := builder.Compose(t.Context(), "child", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "child", LineageFile)
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	body := strings.TrimSuffix(strings.TrimSpace(string(original)), "}")
	forged := `{"weights_sha256":"` + strings.Repeat("e", 64) + `"}`
	for name, data := range map[string]string{
		"case-variant parent": body + `,"Parent":` + forged + `}`,
		"repeated ancestors":  body + `,"ancestors":[` + forged + `]}`,
		"case-variant schema": body + `,"Schema":2}`,
		"unknown key":         body + `,"note":"x"}`,
		"trailing data":       string(original) + ` x`,
	} {
		if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ModelLineage(child, ""); err == nil || !strings.Contains(err.Error(), `model "child"`) {
			t.Fatalf("%s: %v", name, err)
		}
	}
	target := filepath.Join(t.TempDir(), "elsewhere.json")
	if err := os.WriteFile(target, original, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, path); err != nil {
		t.Fatal(err)
	}
	if _, err := ModelLineage(child, ""); err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("linked LINEAGE.json: %v", err)
	}
	// A later schema may change everything but its kind and schema.
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"kind":"waldo-model-lineage","schema":2,"history":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if node, err := ModelLineage(child, ""); err != nil || !reflect.DeepEqual(node.Parents, []LineageLink{{WeightsSHA256: child.Model.Parent.Artifact.SHA256}}) {
		t.Fatalf("later schema: %+v, %v", node.Parents, err)
	}
}

// A pull keeps the release's BOM.json, so its origin descends from the node
// the release declares and holds the chain behind it, and training the pulled
// model, or a model composed from it, continues that chain.
func TestPulledReleaseKeepsItsLineage(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	older := LineageNode{WeightsSHA256: strings.Repeat("c", 64), Runs: []string{strings.Repeat("d", 64)}}
	declared := LineageNode{WeightsSHA256: strings.Repeat("a", 64), Parents: []LineageLink{{WeightsSHA256: older.WeightsSHA256, LineageSHA256: lineageHash(t, older)}}, Runs: []string{strings.Repeat("b", 64)}}
	pullRelease(t, root, "pulled", declared, older)
	pulled, err := Inspect(root, "pulled")
	if err != nil {
		t.Fatal(err)
	}
	origin, history, err := LineageHistory(pulled, "")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(origin.Parents, []LineageLink{{WeightsSHA256: declared.WeightsSHA256, LineageSHA256: lineageHash(t, declared)}}) || !reflect.DeepEqual(history, []LineageNode{declared, older}) {
		t.Fatalf("pulled origin's node %+v and history %+v", origin, history)
	}
	stage := releaseStage("more")
	composed, err := realWeightsBuilder(root).Compose(t.Context(), "composed", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "pulled"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)})
	if err != nil {
		t.Fatal(err)
	}
	trained, err := realWeightsBuilder(root).Train(t.Context(), "pulled", preparedFixture(t, releaseStage("more")))
	if err != nil {
		t.Fatal(err)
	}
	for _, model := range []Inspection{trained, composed} {
		node, chain, err := LineageHistory(model, "")
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(node.Parents, []LineageLink{{WeightsSHA256: origin.WeightsSHA256, LineageSHA256: lineageHash(t, origin)}}) || !reflect.DeepEqual(chain, []LineageNode{origin, declared, older}) {
			t.Fatalf("%s: node %+v and history %+v do not continue the pulled origin %+v", model.Model.Name, node, chain, origin)
		}
	}
}

// The kept release BOM.json is read only while it has the hash its origin
// pinned.
func TestPulledLineageRefusesAnEditedReleaseBOM(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullRelease(t, root, "pulled", LineageNode{WeightsSHA256: strings.Repeat("a", 64)})
	pulled, err := Inspect(root, "pulled")
	if err != nil {
		t.Fatal(err)
	}
	bom := `{"kind":"openwaldo-bom","schema":1,"subject":"model-release","lineage":{"weights_sha256":"` + strings.Repeat("e", 64) + `"}}`
	if err := os.WriteFile(filepath.Join(root, "pulled", filepath.FromSlash(ReleaseBOMPath)), []byte(bom), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ModelLineage(pulled, ""); err == nil || !strings.Contains(err.Error(), "release BOM.json") {
		t.Fatalf("edited release BOM.json: %v", err)
	}
	stage := releaseStage("child")
	if _, err := realWeightsBuilder(root).Compose(t.Context(), "child", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "pulled"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)}); err == nil || !strings.Contains(err.Error(), "release BOM.json") {
		t.Fatalf("compose from an edited release BOM.json: %v", err)
	}
}

// Lineage another reader could read differently, or that WALDO cannot read
// as schema 1, is left out of the pull with a note rather than recorded or
// fatal, so the origin is a root.
func TestPullLeavesOutLineageItCannotReadExactly(t *testing.T) {
	isolateHuggingFaceToken(t)
	a, c := strings.Repeat("a", 64), strings.Repeat("c", 64)
	node := `{"weights_sha256":"` + a + `"}`
	older := `{"weights_sha256":"` + c + `"}`
	release := func(fields string) string {
		return `{"kind":"openwaldo-bom","schema":1,"subject":"model-release",` + fields + `,"artifacts":ARTIFACTS}`
	}
	for name, bom := range map[string]string{
		"node type":          release(`"lineage":{"weights_sha256":123}`),
		"node parents":       release(`"lineage":{"weights_sha256":"` + a + `","parents":"oops"}`),
		"node digest":        release(`"lineage":{"weights_sha256":"xyz"}`),
		"node key case":      release(`"lineage":{"WEIGHTS_SHA256":"` + a + `"}`),
		"node extra":         release(`"lineage":{"weights_sha256":"` + a + `","declared":true}`),
		"node duplicates":    release(`"lineage":{"weights_sha256":"` + a + `","weights_sha256":"` + c + `"}`),
		"null node":          release(`"lineage":null`),
		"ancestors only":     release(`"lineage_ancestors":[` + older + `]`),
		"stray ancestor":     release(`"lineage":` + node + `,"lineage_ancestors":[` + older + `]`),
		"link weights":       release(`"lineage":{"weights_sha256":"` + a + `","parents":[{"weights_sha256":"` + strings.Repeat("9", 64) + `","lineage_sha256":"` + lineageHash(t, LineageNode{WeightsSHA256: c}) + `"}]},"lineage_ancestors":[` + older + `]`),
		"case variant key":   release(`"lineage":` + node + `,"LINEAGE":` + older),
		"repeated key":       release(`"lineage":` + node + `,"lineage":` + older),
		"kind case":          `{"Kind":"openwaldo-bom","schema":1,"subject":"model-release","lineage":` + node + `,"artifacts":ARTIFACTS}`,
		"trailing data":      release(`"lineage":`+node) + ` x`,
		"escaped key, trail": release(`"\u006cineage":`+node) + ` x`,
		"inventory key case": `{"kind":"openwaldo-bom","schema":1,"subject":"model-release","lineage":` + node + `,"artifacts":[{"role":"weights","path":"model.safetensors","sha256":"` + strings.Repeat("0", 64) + `","bytes":1,"SHA256":"WEIGHTS"}]}`,
		"schema missing":     `{"kind":"openwaldo-bom","subject":"model-release","lineage":` + node + `,"artifacts":ARTIFACTS}`,
		"schema string":      `{"kind":"openwaldo-bom","schema":"1","subject":"model-release","lineage":` + node + `,"artifacts":ARTIFACTS}`,
		"schema later":       `{"kind":"openwaldo-bom","schema":2,"subject":"model-release","lineage":{"weights_sha256":"a later form"},"artifacts":ARTIFACTS}`,
		"third-party key":    `{"bomFormat":"CycloneDX","lineage":"fine-tuned from a base"}`,
		"third-party array":  `[{"description":"data lineage"}]`,
		"third-party text":   `lineage: see the model card`,
	} {
		files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
		artifacts, err := json.Marshal(inventory(files))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(files["model.safetensors"])
		bom = strings.NewReplacer("ARTIFACTS", string(artifacts), "WEIGHTS", hex.EncodeToString(digest[:])).Replace(bom)
		files["BOM.json"] = []byte(bom)
		assertPulledRoot(t, files, name)
	}
}

// Lineage in a release BOM.json saved as UTF-16 is left out with a note, not silently.
func TestPullNotesLineageInUTF16(t *testing.T) {
	isolateHuggingFaceToken(t)
	files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
	bom, err := json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": LineageNode{WeightsSHA256: strings.Repeat("a", 64)}, "artifacts": inventory(files)})
	if err != nil {
		t.Fatal(err)
	}
	utf16 := []byte{0xff, 0xfe}
	for _, char := range bom {
		utf16 = append(utf16, char, 0)
	}
	files["BOM.json"] = utf16
	assertPulledRoot(t, files, "utf-16")
}

// Only the weights entries of a release's artifacts bind its lineage: other
// entries may repeat or carry more fields, while a weights entry must be exact.
func TestPullReadsOnlyWeightsEntriesExactly(t *testing.T) {
	isolateHuggingFaceToken(t)
	declared := LineageNode{WeightsSHA256: strings.Repeat("a", 64)}
	pull := func(extra map[string]any, weightsField bool) (map[string][]byte, []any) {
		files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
		var artifacts []any
		for _, artifact := range inventory(files) {
			entry := map[string]any{"role": artifact.Role, "path": artifact.Path, "sha256": artifact.SHA256, "bytes": artifact.Bytes}
			if artifact.Path == "model.safetensors" && !weightsField {
				artifacts = append(artifacts, artifact)
				continue
			}
			for key, value := range extra {
				entry[key] = value
			}
			artifacts = append(artifacts, entry)
		}
		return files, append(artifacts, map[string]any{"path": "config.json", "note": "listed twice"})
	}
	files, artifacts := pull(map[string]any{"media_type": "application/json"}, false)
	bom, err := json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": declared, "artifacts": artifacts})
	if err != nil {
		t.Fatal(err)
	}
	files["BOM.json"] = bom
	client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("a", 40))}
	pulled, err := (Puller{Root: t.TempDir(), Endpoint: "https://hub.test", Client: client}).Pull(t.Context(), "tiny", "huggingface://org/tiny")
	if err != nil {
		t.Fatal(err)
	}
	if origin, err := ModelLineage(pulled, ""); err != nil || len(origin.Parents) != 1 {
		t.Fatalf("other entries cost the lineage: %+v, %v", origin, err)
	}
	files, artifacts = pull(map[string]any{"media_type": "application/octet-stream"}, true)
	if bom, err = json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": declared, "artifacts": artifacts}); err != nil {
		t.Fatal(err)
	}
	files["BOM.json"] = bom
	assertPulledRoot(t, files, "extra field on weights entry")
}

// A release BOM.json saved with a UTF-8 byte-order mark keeps its lineage.
func TestPullKeepsLineageAfterAByteOrderMark(t *testing.T) {
	isolateHuggingFaceToken(t)
	files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
	declared := LineageNode{WeightsSHA256: strings.Repeat("a", 64)}
	bom, err := json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": declared, "artifacts": inventory(files)})
	if err != nil {
		t.Fatal(err)
	}
	files["BOM.json"] = append([]byte("\xef\xbb\xbf"), bom...)
	client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("a", 40))}
	pulled, err := (Puller{Root: t.TempDir(), Endpoint: "https://hub.test", Client: client}).Pull(t.Context(), "tiny", "huggingface://org/tiny")
	if err != nil {
		t.Fatal(err)
	}
	origin, err := ModelLineage(pulled, "")
	if err != nil || !reflect.DeepEqual(origin.Parents, []LineageLink{{WeightsSHA256: declared.WeightsSHA256, LineageSHA256: lineageHash(t, declared)}}) {
		t.Fatalf("origin %+v, err %v", origin, err)
	}
}

// A release BOM.json whose artifacts do not list the pulled weights, such as
// one left behind when the weights were replaced, lends them no lineage.
func TestPullLeavesOutLineageOfOtherWeights(t *testing.T) {
	isolateHuggingFaceToken(t)
	node := LineageNode{WeightsSHA256: strings.Repeat("a", 64)}
	for name, artifacts := range map[string][]OriginArtifact{
		"other weights": {{Role: "weights", Path: "model.safetensors", SHA256: strings.Repeat("0", 64), Bytes: 1}},
		"no inventory":  nil,
		"listed twice":  nil,
	} {
		files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
		if name == "listed twice" {
			stale := OriginArtifact{Role: "weights", Path: "model.safetensors", SHA256: strings.Repeat("0", 64), Bytes: 1}
			artifacts = append([]OriginArtifact{stale}, inventory(files)...)
		}
		bom, err := json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": node, "artifacts": artifacts})
		if err != nil {
			t.Fatal(err)
		}
		files["BOM.json"] = bom
		assertPulledRoot(t, files, name)
	}
}

// assertPulledRoot pulls files and requires a pull that notes its release's
// lineage was left out and whose origin is a root.
func assertPulledRoot(t *testing.T, files map[string][]byte, name string) {
	t.Helper()
	root := t.TempDir()
	var notes []string
	client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("a", 40))}
	puller := Puller{Root: root, Endpoint: "https://hub.test", Client: client, Progress: func(progress PullProgress) {
		if progress.Phase == "lineage" {
			notes = append(notes, progress.Message)
		}
	}}
	pulled, err := puller.Pull(t.Context(), "tiny", "huggingface://org/tiny")
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	origin, history, err := LineageHistory(pulled, "")
	if err != nil || len(origin.Parents) != 0 || len(history) != 0 || len(notes) != 1 {
		t.Fatalf("%s: origin %+v, history %+v, notes %q, err %v", name, origin, history, notes, err)
	}
}

// A pulled root names the release it came from: the same weights from another
// repository or revision are another root, while the same release is one root
// however the pull was written or which mirror served it.
func TestPulledRootsNameTheirRelease(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
	main, other := strings.Repeat("a", 40), strings.Repeat("b", 40)
	pullRoot := func(name, endpoint, source, revision string) LineageNode {
		t.Helper()
		client := &http.Client{Transport: huggingFaceTransport(t, files, revision)}
		if _, err := (Puller{Root: root, Endpoint: endpoint, Client: client}).Pull(t.Context(), name, source); err != nil {
			t.Fatal(err)
		}
		pulled, err := Inspect(root, name)
		if err != nil {
			t.Fatal(err)
		}
		node, err := ModelLineage(pulled, "")
		if err != nil {
			t.Fatal(err)
		}
		return node
	}
	byBranch := pullRoot("by-branch", "https://hub.test", "huggingface://org/tiny", main)
	byCommit := pullRoot("by-commit", "https://hub.test", "huggingface://org/tiny@"+main, main)
	byMirror := pullRoot("by-mirror", "https://mirror.test", "huggingface://org/tiny", main)
	if byBranch.SourceSHA256 == "" || !reflect.DeepEqual(byBranch, byCommit) || !reflect.DeepEqual(byBranch, byMirror) {
		t.Fatalf("one release, several roots: %+v, %+v, %+v", byBranch, byCommit, byMirror)
	}
	for name, node := range map[string]LineageNode{
		"another revision":   pullRoot("revision", "https://hub.test", "huggingface://org/tiny", other),
		"another repository": pullRoot("repository", "https://hub.test", "huggingface://org/bad", main),
	} {
		if node.WeightsSHA256 != byBranch.WeightsSHA256 || node.SourceSHA256 == byBranch.SourceSHA256 {
			t.Fatalf("%s: same weights should be another root: %+v and %+v", name, node, byBranch)
		}
	}
}
