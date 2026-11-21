// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

func TestParseModelTrainDefaultsAndValidatesEpochs(t *testing.T) {
	context, args, err := parseCobraCommand(t, []string{"model", "train"}, []string{"foo", "core/books", "science/papers"})
	if err != nil {
		t.Fatal(err)
	}
	if args[0] != "foo" || int64Option(context, "epochs") != 1 || len(args[1:]) != 2 {
		t.Fatalf("args = %v, epochs = %d", args, int64Option(context, "epochs"))
	}
	context, _, err = parseCobraCommand(t, []string{"model", "train"}, []string{"foo", "core/books", "--epochs", "3"})
	if err != nil || int64Option(context, "epochs") != 3 {
		t.Fatalf("epochs = %d, error = %v", int64Option(context, "epochs"), err)
	}
	context, _, err = parseCobraCommand(t, []string{"model", "train"}, []string{"foo", "core/books", "--audit"})
	if err != nil || !boolOption(context, "audit") {
		t.Fatalf("audit = %v, error = %v", boolOption(context, "audit"), err)
	}
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"model", "train", "foo", "core/books", "--epochs", "0"}, &stdout, &stderr); code != 1 {
		t.Fatal("zero epochs accepted")
	}
	stderr.Reset()
	if code := Run([]string{"model", "train", "foo", "core/books", "--evaluate-only"}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "--evaluate-only needs a compose file") {
		t.Fatalf("evaluate-only without compose: code %d, stderr %q", code, stderr.String())
	}
}

func TestTrainingComposeInputRequiresOneIdentifiedCompose(t *testing.T) {
	directory := t.TempDir()
	compose := filepath.Join(directory, "training.yaml")
	if err := os.WriteFile(compose, []byte("kind: waldo-model-compose\nschema: 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	index := filepath.Join(directory, "index.yaml")
	if err := os.WriteFile(index, []byte("kind: index\nschema: 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := trainingComposeInput([]string{compose}); err != nil || got != compose {
		t.Fatalf("compose input = %q, err = %v", got, err)
	}
	if got, err := trainingComposeInput([]string{index}); err != nil || got != "" {
		t.Fatalf("index input = %q, err = %v", got, err)
	}
	if _, err := trainingComposeInput([]string{compose, "core/books"}); err == nil || !strings.Contains(err.Error(), "only training input") {
		t.Fatalf("mixed compose inputs error = %v", err)
	}
}

func TestComposeCorpusSanityCheckReportsAllMissingPathsBeforeMaterialization(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "index.yaml"), []byte("kind: index\nschema: 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "present"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Index: root}); err != nil {
		t.Fatal(err)
	}
	compose := model.Compose{Stages: []model.Stage{
		{Name: "pretrain", Corpora: model.NewCorpusSelections([]string{"present", "missing/base"})},
		{Name: "post-train", Corpora: model.NewCorpusSelections([]string{"missing/sft"})},
	}}
	var progress bytes.Buffer
	_, err := sanityCheckComposeCorpora(context.Background(), compose, &progress)
	if err == nil {
		t.Fatal("missing compose corpora passed sanity check")
	}
	for _, want := range []string{"failed before shard download", "stage pretrain: missing/base", "stage post-train: missing/sft", "run `waldo index pull`"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("sanity error = %q, missing %q", err, want)
		}
	}
	if !strings.Contains(progress.String(), "checking 3 corpus paths against index "+root) || strings.Contains(progress.String(), "materialize") {
		t.Fatalf("sanity progress = %q", progress.String())
	}
}

func TestModelProgressMessageIncludesStableETA(t *testing.T) {
	event := model.Progress{
		Phase: "training", Message: "step 319/31984, loss 3.7461, 146497 tokens/s",
		Training: &training.Event{Kind: "progress", Step: 319, ETASeconds: 3661},
	}
	if got := modelProgressMessage(event); got != event.Message+", ETA 1h 1m" {
		t.Fatalf("progress message = %q", got)
	}
	event.Training.Step = 1
	if got := modelProgressMessage(event); got != event.Message {
		t.Fatalf("startup progress message = %q", got)
	}
}

func TestFormatModelProgressBar(t *testing.T) {
	got := formatModelProgressBar(model.ProgressBar{
		Label:   "pack",
		Current: 25,
		Total:   100,
		Detail:  "200 cumulative record visits; corpus pass 2 (1 completed)",
	})
	for _, want := range []string{"pack", "25%", "25/100 sequences", "200 cumulative record visits", "corpus pass 2 (1 completed)"} {
		if !strings.Contains(got, want) {
			t.Fatalf("progress bar %q does not contain %q", got, want)
		}
	}
}

func TestModelMaterializeProgressReportsEveryCompletedShardToLogs(t *testing.T) {
	var output bytes.Buffer
	report := modelMaterializeProgressPrinter(&output)
	for position := 1; position <= 3; position++ {
		report(corpus.MaterializeProgress{
			Phase: "complete", Current: position, Total: 3,
			Bytes: int64(position * 10), TotalBytes: 30,
			Shard: corpus.ShardPin{SHA256: strings.Repeat(string(rune('a'+position-1)), 64)},
		})
	}
	if strings.Count(output.String(), "materialize [") != 3 ||
		!strings.Contains(output.String(), " 66%") ||
		!strings.Contains(output.String(), "100%") ||
		!strings.Contains(output.String(), "30 B/30 B") {
		t.Fatalf("progress output = %q", output.String())
	}
}

func TestEpochStageExplainsMaterializationBeforeResolvingShards(t *testing.T) {
	var progress bytes.Buffer
	stage := model.Stage{Name: "pretrain", Parameters: training.Parameters{Epochs: 3}}
	_, err := materializeModelStage(Context{Execution: context.Background()}, stage, corpus.BOM{}, nil, &progress, false)
	if err == nil {
		t.Fatal("empty BOM unexpectedly materialized")
	}
	got := progress.String()
	note := "epoch-based training requires every selected shard to be materialized (or reused from cache) and hash-verified"
	resolved := "preflight/pretrain          resolving"
	if !strings.Contains(got, note) {
		t.Fatalf("progress = %q, missing epoch materialization explanation", got)
	}
	if strings.Index(got, note) > strings.Index(got, resolved) {
		t.Fatalf("epoch explanation follows shard resolution: %q", got)
	}
}

func TestParseModelTrainDiagnosesOmittedModelName(t *testing.T) {
	for _, path := range []string{"core/common-pile/python-enhancement-proposals/peps", "."} {
		var stdout, stderr bytes.Buffer
		if code := Run([]string{"model", "train", path}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "model name is required") {
			t.Fatalf("path %q: code = %d, error = %q", path, code, stderr.String())
		}
	}
}

func TestModelTrainFormatsOmittedNameDiagnostic(t *testing.T) {
	var stdout, stderr bytes.Buffer
	path := "core/common-pile/python-enhancement-proposals/peps"
	if code := Run([]string{"model", "train", path}, &stdout, &stderr); code != 1 {
		t.Fatalf("code = %d, stderr = %q", code, stderr.String())
	}
	wantError := "Error: model name is required before index path \"" + path + "\"\n"
	if stderr.String() != wantError {
		t.Fatalf("stderr = %q, want %q", stderr.String(), wantError)
	}
	if !strings.Contains(stdout.String(), "Usage:\n  waldo model train <name> [index-path...] | <name> <compose-file> [flags]") {
		t.Fatalf("stdout does not contain Cobra usage: %q", stdout.String())
	}
}

// A pending merge-only compose has no stages left, but only Compose finishes
// its transaction, so it never takes the completed-compose shortcut.
func TestPendingMergeComposeFinishesThroughCompose(t *testing.T) {
	merge := model.Compose{Base: &model.ComposeBase{Merge: []model.ComposeMergeParent{{Model: "left"}, {Model: "right"}}}}
	for name, test := range map[string]struct {
		pending bool
		compose model.Compose
		want    bool
	}{
		"repeated merge":   {false, merge, true},
		"pending merge":    {true, merge, false},
		"pending training": {true, model.Compose{}, true},
	} {
		if got := completesWithoutCompose(test.pending, test.compose); got != test.want {
			t.Fatalf("%s: completes without Compose = %v, want %v", name, got, test.want)
		}
	}
}

const evaluateOnlyCompose = `kind: waldo-model-compose
schema: 1
architecture:
  family: decoder-transformer
  context_tokens: 128
  vocabulary_size: 259
  hidden_size: 64
  intermediate_size: 192
  layers: 2
  attention_heads: 4
  key_value_heads: 2
  tie_embeddings: true
  parameter_dtype: float32
  tokenizer:
    name: byte
    revision: builtin-byte-schema-1
stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    corpora:
      - books
    parameters:
      steps: 2
      batch_size: 1
      sequence_length: 64
      learning_rate: 0.001
      seed: 7
`

// evaluateOnlyFixture configures an empty index and an untrained model the compose describes.
func evaluateOnlyFixture(t *testing.T) string {
	t.Helper()
	index, models := t.TempDir(), t.TempDir()
	writeCLIFile(t, filepath.Join(index, "index.yaml"), "kind: index\nschema: 1\n")
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Index: index, Model: config.Model{Root: models, Backend: "fake"}}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "compose.yaml")
	writeCLIFile(t, path, evaluateOnlyCompose)
	compose, _, err := model.LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := (model.Builder{Root: models}).Initialize("untrained", compose.Architecture); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestEvaluateOnlyRefusesAModelBeforeCheckingTheCorpus(t *testing.T) {
	compose := evaluateOnlyFixture(t)
	for name, want := range map[string]string{"absent": "absent", "untrained": "no weights to evaluate"} {
		var stdout, stderr bytes.Buffer
		if code := Run([]string{"model", "train", name, compose, "--evaluate-only"}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), want) || strings.Contains(stderr.String(), "corpus") {
			t.Fatalf("%s: code %d, stderr %q", name, code, stderr.String())
		}
	}
}

func TestEvaluateOnlyRefusesAComposeForAnotherModel(t *testing.T) {
	compose := evaluateOnlyFixture(t)
	for _, change := range []struct{ from, to string }{{"layers: 2", "layers: 3"}, {"name: byte", "name: other"}} {
		data, err := os.ReadFile(compose)
		if err != nil {
			t.Fatal(err)
		}
		other := filepath.Join(t.TempDir(), "other.yaml")
		writeCLIFile(t, other, strings.Replace(string(data), change.from, change.to, 1))
		var stdout, stderr bytes.Buffer
		if code := Run([]string{"model", "train", "untrained", other, "--evaluate-only"}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "does not match model") {
			t.Fatalf("%s: code %d, stderr %q", change.to, code, stderr.String())
		}
	}
}

func TestEvaluateOnlyRejectsTrainingAndClusterFlags(t *testing.T) {
	for _, flags := range [][]string{{"--hostfile", "hosts"}, {"--nodes", "2"}, {"--rendezvous", "primary:29500"}, {"--rendezvous-id", "r1"}, {"--rendezvous-port", "29501"}, {"--audit"}, {"--epochs", "2"}, {"--seed", "3"}} {
		var stdout, stderr bytes.Buffer
		if code := Run(append([]string{"model", "train", "foo", "compose.yaml", "--evaluate-only"}, flags...), &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), flags[0]+" cannot be used with --evaluate-only") {
			t.Fatalf("%v: code %d, stderr %q", flags, code, stderr.String())
		}
	}
}
