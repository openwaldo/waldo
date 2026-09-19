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
		Detail:  "200 record visits; up to 2 corpus passes",
	})
	for _, want := range []string{"pack", "25%", "25/100 sequences", "200 record visits", "up to 2 corpus passes"} {
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
