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

// pullRelease pulls a release whose BOM.json declares node as the lineage its
// weights came from, or with no BOM.json when node is empty.
func pullRelease(t *testing.T, root, name string, node LineageNode) {
	t.Helper()
	files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
	if node.WeightsSHA256 != "" {
		bom, err := json.Marshal(map[string]any{"kind": "openwaldo-bom", "schema": 1, "subject": "model-release", "lineage": node})
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

// Pinning an earlier run records that run's node: its weights and only the run that made them.
func TestComposePinnedToEarlierRunRecordsThatRunsLineage(t *testing.T) {
	root := t.TempDir()
	builder := realWeightsBuilder(root)
	first := trainTwice(t, builder)
	resolved, err := builder.ResolveCompose(t.Context(), Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p", RunID: first}, Stages: []Stage{testStage("child")}}, false)
	if err != nil {
		t.Fatal(err)
	}
	p, err := Inspect(root, "p")
	if err != nil {
		t.Fatal(err)
	}
	node, err := ModelLineage(p, first)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Base.LineageSHA256 != lineageHash(t, node) || node.WeightsSHA256 != resolved.Base.ArtifactSHA256 || !reflect.DeepEqual(node.Runs, []string{resolved.Base.RunBOMSHA256}) {
		t.Fatalf("pin weights %s, run %s, lineage %s; node %+v", resolved.Base.ArtifactSHA256, resolved.Base.RunBOMSHA256, resolved.Base.LineageSHA256, node)
	}
}

// A fine-tune's first run descends from the node its parent pin records, so
// the chain continues from model to model.
func TestFineTuneContinuesItsParentsChain(t *testing.T) {
	root := t.TempDir()
	builder := realWeightsBuilder(root)
	trainTwice(t, builder)
	stage := testStage("child")
	child, err := builder.Compose(t.Context(), "child", Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p"}, Stages: []Stage{stage}}, []PreparedStage{preparedFixture(t, stage)})
	if err != nil {
		t.Fatal(err)
	}
	node, err := ModelLineage(child, "")
	if err != nil {
		t.Fatal(err)
	}
	parent := child.Model.Parent
	if parent == nil || parent.LineageSHA256 == "" || !reflect.DeepEqual(node.Parents, []LineageLink{{WeightsSHA256: parent.Artifact.SHA256, LineageSHA256: parent.LineageSHA256}}) {
		t.Fatalf("child's node %+v does not continue its parent pin %+v", node, parent)
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

// A pull keeps the node its release declares, and training the pulled model
// continues the chain from it.
func TestPulledReleaseKeepsItsLineage(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	declared := LineageNode{WeightsSHA256: strings.Repeat("a", 64), Runs: []string{strings.Repeat("b", 64)}}
	pullRelease(t, root, "pulled", declared)
	pulled, err := Inspect(root, "pulled")
	if err != nil {
		t.Fatal(err)
	}
	if pulled.Origin.Lineage == nil || !reflect.DeepEqual(*pulled.Origin.Lineage, declared) {
		t.Fatalf("origin lineage = %+v", pulled.Origin.Lineage)
	}
	origin, err := ModelLineage(pulled, "")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(origin.Parents, []LineageLink{{WeightsSHA256: declared.WeightsSHA256, LineageSHA256: lineageHash(t, declared)}}) {
		t.Fatalf("pulled origin's node = %+v", origin)
	}
	trained, err := realWeightsBuilder(root).Train(t.Context(), "pulled", preparedFixture(t, releaseStage("more")))
	if err != nil {
		t.Fatal(err)
	}
	node, err := ModelLineage(trained, "")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(node.Parents, []LineageLink{{WeightsSHA256: origin.WeightsSHA256, LineageSHA256: lineageHash(t, origin)}}) {
		t.Fatalf("trained node %+v does not continue the pulled origin %+v", node, origin)
	}
}

// A release whose lineage is not a lineage node is refused, not pulled as if
// it recorded none.
func TestPullRefusesMalformedReleaseLineage(t *testing.T) {
	isolateHuggingFaceToken(t)
	for name, lineage := range map[string]string{"type": `{"weights_sha256":123}`, "parents": `{"weights_sha256":"` + strings.Repeat("a", 64) + `","parents":"oops"}`, "digest": `{"weights_sha256":"xyz"}`} {
		files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
		files["BOM.json"] = []byte(`{"kind":"openwaldo-bom","schema":1,"subject":"model-release","lineage":` + lineage + `}`)
		client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("a", 40))}
		if _, err := (Puller{Root: t.TempDir(), Endpoint: "https://hub.test", Client: client}).Pull(t.Context(), "tiny", "huggingface://org/tiny"); err == nil || !strings.Contains(err.Error(), "lineage") {
			t.Fatalf("%s: pull error = %v", name, err)
		}
	}
}

// Only a trained base has a lineage node to pin; any other pin is refused, not recorded.
func TestComposeRefusesLineagePinOnUntrainedBase(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	pullRelease(t, root, "pulled", LineageNode{})
	stage := releaseStage("continue")
	compose := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "pulled", LineageSHA256: strings.Repeat("f", 64)}, Stages: []Stage{stage}}
	if _, err := (Builder{Root: root, Resolver: training.FakeResolver()}).Compose(t.Context(), "child", compose, []PreparedStage{preparedFixture(t, stage)}); err == nil || !strings.Contains(err.Error(), "cannot assert managed-model run pins") {
		t.Fatalf("origin base with a lineage pin: %v", err)
	}
	source := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Source: "huggingface://org/tiny@" + strings.Repeat("a", 40), LineageSHA256: strings.Repeat("f", 64)}, Stages: []Stage{stage}}
	if err := source.Validate(); err == nil || !strings.Contains(err.Error(), "base.source") {
		t.Fatalf("source base with a lineage pin: %v", err)
	}
}

// Parent pins recorded before lineage still match the same parent, so existing
// fine-tunes keep accepting stages.
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

// An asserted lineage pin must be the base run's own node.
func TestComposeRefusesAWrongLineagePin(t *testing.T) {
	builder := realWeightsBuilder(t.TempDir())
	first := trainTwice(t, builder)
	compose := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p", RunID: first, LineageSHA256: strings.Repeat("f", 64)}, Stages: []Stage{testStage("child")}}
	if _, err := builder.ResolveCompose(t.Context(), compose, false); err == nil || !strings.Contains(err.Error(), "recorded ancestry") {
		t.Fatalf("wrong lineage pin: %v", err)
	}
}

// A fine-tune interrupted before lineage was recorded resumes with the same
// command; its parent pin stays as it was written.
func TestComposeResumesATransactionFromBeforeLineage(t *testing.T) {
	root := t.TempDir()
	builder := realWeightsBuilder(root)
	first := trainTwice(t, builder)
	stage := testStage("child")
	compose := Compose{Kind: "waldo-model-compose", Schema: ComposeSchema, Base: &ComposeBase{Model: "p", RunID: first}, Stages: []Stage{stage}}
	interrupted := builder
	interrupted.Resolver = training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backendFunc(func(context.Context, training.Request) (training.Observation, error) {
			return training.Observation{}, context.Canceled
		})), nil
	})
	if _, err := interrupted.Compose(t.Context(), "child", compose, []PreparedStage{preparedFixture(t, stage)}); err == nil {
		t.Fatal("interrupted compose finished")
	}
	// Rewrite the kept transaction and compose as a WALDO without lineage wrote them.
	paths, err := filepath.Glob(filepath.Join(root, ".waldo-compose", "*", "COMPOSE.json"))
	if err != nil || len(paths) != 1 {
		t.Fatalf("pending transactions %v, %v", paths, err)
	}
	var transaction composeTransaction
	if err := readJSON(paths[0], &transaction); err != nil {
		t.Fatal(err)
	}
	transaction.Compose.Base.LineageSHA256 = ""
	if err := writeJSONAtomic(paths[0], transaction); err != nil {
		t.Fatal(err)
	}
	if err := writeJSONAtomic(filepath.Join(root, "child", "COMPOSE.json"), transaction.Compose); err != nil {
		t.Fatal(err)
	}
	resolved, err := builder.ResolveCompose(t.Context(), compose, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := builder.CheckComposeTarget("child", resolved); err != nil {
		t.Fatalf("check before resuming: %v", err)
	}
	if _, err := builder.Compose(t.Context(), "child", compose, []PreparedStage{preparedFixture(t, stage)}); err != nil {
		t.Fatalf("resume: %v", err)
	}
}

// The pull reads the exact lineage key in WALDO's own form, so the recorded
// node is the one any reader of the release's BOM.json sees.
func TestPullRefusesALineageItsReadersCouldSeeDifferently(t *testing.T) {
	isolateHuggingFaceToken(t)
	node := `{"weights_sha256":"` + strings.Repeat("a", 64) + `"}`
	for name, bom := range map[string]string{
		"case variant":    `{"kind":"openwaldo-bom","subject":"model-release","lineage":` + node + `,"LINEAGE":{"weights_sha256":"` + strings.Repeat("c", 64) + `"}}`,
		"node key case":   `{"kind":"openwaldo-bom","subject":"model-release","lineage":{"WEIGHTS_SHA256":"` + strings.Repeat("a", 64) + `"}}`,
		"node extra":      `{"kind":"openwaldo-bom","subject":"model-release","lineage":{"weights_sha256":"` + strings.Repeat("a", 64) + `","note":"x"}}`,
		"node duplicates": `{"kind":"openwaldo-bom","subject":"model-release","lineage":{"weights_sha256":"` + strings.Repeat("a", 64) + `","weights_sha256":"` + strings.Repeat("c", 64) + `"}}`,
	} {
		files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
		files["BOM.json"] = []byte(bom)
		client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("a", 40))}
		if _, err := (Puller{Root: t.TempDir(), Endpoint: "https://hub.test", Client: client}).Pull(t.Context(), "tiny", "huggingface://org/tiny"); err == nil || !strings.Contains(err.Error(), "lineage") {
			t.Fatalf("%s: pull error = %v", name, err)
		}
	}
}

// A pulled root names the origin record it came from, so the same weights
// from two sources are two roots, and the chain reaches the source.
func TestPulledRootsNameTheirOrigin(t *testing.T) {
	isolateHuggingFaceToken(t)
	root := t.TempDir()
	files := huggingFaceFixture(t, releaseArchitecture(), HuggingFaceTokenizerClass)
	var nodes []LineageNode
	for _, revision := range []string{strings.Repeat("a", 40), strings.Repeat("b", 40)} {
		client := &http.Client{Transport: huggingFaceTransport(t, files, revision)}
		name := "pulled-" + revision[:1]
		if _, err := (Puller{Root: root, Endpoint: "https://hub.test", Client: client}).Pull(t.Context(), name, "huggingface://org/tiny"); err != nil {
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
		if node.OriginSHA256 != pulled.Model.OriginBOMSHA256 {
			t.Fatalf("revision %s: node origin %s, model origin %s", revision, node.OriginSHA256, pulled.Model.OriginBOMSHA256)
		}
		nodes = append(nodes, node)
	}
	if nodes[0].WeightsSHA256 != nodes[1].WeightsSHA256 || lineageHash(t, nodes[0]) == lineageHash(t, nodes[1]) {
		t.Fatalf("same weights from two sources: %+v and %+v", nodes[0], nodes[1])
	}
}
