// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model_test

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/openwaldo/waldo/internal/inference"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/modelexport"
)

func TestTransformersRejectsByteSpecialIDs(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" {
		t.Skip("set Transformers Python and wheel")
	}
	for _, name := range []string{"pad_token_id", "bos_token_id", "eos_token_id"} {
		t.Run(name, func(t *testing.T) {
			compose, _, err := model.LoadCompose("../../docs/examples/transformers-smoke.yaml")
			if err != nil {
				t.Fatal(err)
			}
			if name == "pad_token_id" {
				delete(compose.Architecture.Transformers.Config, name)
			} else {
				compose.Architecture.Transformers.Config[name] = 3
			}
			compose.Stages[0].Corpora = model.NewCorpusSelections([]string{"example"})
			root := t.TempDir()
			_, err = (model.Builder{Root: root}).Compose(t.Context(), "bad-specials", compose, []model.PreparedStage{hfPrepared(t, compose.Stages[0])})
			if err == nil || !strings.Contains(err.Error(), "differs from tokenizer framing") {
				t.Fatalf("expected special-ID rejection: %v", err)
			}
			if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if entry.Name() == "model.safetensors" {
					t.Errorf("invalid config produced weights: %s", path)
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestTransformersRejectsDivergentTraining(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" {
		t.Skip("set Transformers Python and wheel")
	}
	// The deliberately divergent FP32 CPU case is reproducible independently
	// of CUDA mixed-precision scaling. No evaluation may catch it for us.
	t.Setenv("WALDO_TRANSFORMERS_DEVICE", "cpu")
	for _, steps := range []int64{2, 3} {
		t.Run(fmt.Sprintf("steps-%d", steps), func(t *testing.T) {
			compose, _, err := model.LoadCompose("../../docs/examples/transformers-smoke.yaml")
			if err != nil {
				t.Fatal(err)
			}
			stage := &compose.Stages[0]
			stage.Corpora = model.NewCorpusSelections([]string{"example"})
			zero := 0.0
			stage.Parameters.EvaluationFraction = &zero
			stage.Parameters.Steps = steps
			stage.Parameters.SequenceLength = 8
			stage.Parameters.BatchSize = 1
			stage.Parameters.LearningRate = 1e20
			stage.Parameters.Trainer.Arguments["learning_rate"] = 1e20
			stage.Parameters.Trainer.Arguments["per_device_train_batch_size"] = 1
			root := t.TempDir()
			_, err = (model.Builder{Root: root}).Compose(t.Context(), "divergent", compose, []model.PreparedStage{hfPrepared(t, *stage)})
			if err == nil || !strings.Contains(err.Error(), "non-finite") {
				t.Fatalf("expected non-finite rejection: %v", err)
			}
			t.Logf("rejected: %v", err)
			if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if entry.Name() == "model.safetensors" {
					t.Errorf("divergent training published weights: %s", path)
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// This opt-in test uses only synthetic corpus data and pinned local assets.
func TestTransformersTokenizerLifecycle(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" || os.Getenv("WALDO_HF_TOKENIZER_DIR") == "" {
		t.Skip("set Transformers Python, wheel, and WALDO_HF_TOKENIZER_DIR")
	}
	for _, variant := range []struct {
		name, compose string
		targets       int64
	}{
		{"byte", "transformers-qwen3-smoke.yaml", 128},
		{"huggingface", "transformers-qwen3-tokenizer-smoke.yaml", 16},
	} {
		t.Run(variant.name, func(t *testing.T) {
			tokenizerDirectory := os.Getenv("WALDO_HF_TOKENIZER_DIR")
			compose, _, err := model.LoadCompose("../../docs/examples/" + variant.compose)
			if err != nil {
				t.Fatal(err)
			}
			compose.Stages[0].Corpora = model.NewCorpusSelections([]string{"example"})
			builder := model.Builder{Root: t.TempDir()}
			prepared := hfPrepared(t, compose.Stages[0])
			inspection, err := builder.Compose(t.Context(), variant.name, compose, []model.PreparedStage{prepared})
			if err != nil {
				t.Fatal(err)
			}
			if inspection.RunBOMs[0].Initialization != nil {
				t.Fatal("expected random initialization")
			}
			prepared.Stage.Name = "continue"
			inspection, err = builder.Train(t.Context(), variant.name, prepared)
			if err != nil {
				t.Fatal(err)
			}
			if inspection.RunBOMs[1].Initialization == nil {
				t.Fatal("missing continuation provenance")
			}
			for i, run := range inspection.Runs {
				if run.State != model.RunComplete || run.Observation == nil || run.Observation.Simulated || int64(run.Observation.ConsumedTokens) != variant.targets || run.Observation.Steps != 2 {
					t.Fatalf("run %d accounting: %+v", i, run)
				}
				if !reflect.DeepEqual(inspection.RunBOMs[i].Execution.Package, &compose.Architecture.Transformers.Package) {
					t.Fatal("package provenance mismatch")
				}
				if os.Getenv("WALDO_TRANSFORMERS_DEVICE") == "cuda" && len(inspection.RunBOMs[i].Execution.Accelerators) != 1 {
					t.Fatal("missing CUDA evidence")
				}
			}
			// Inference must use retained assets, not the original tokenizer directory.
			t.Setenv("WALDO_HF_TOKENIZER_DIR", filepath.Join(t.TempDir(), "unavailable"))
			opened, err := inference.Open(t.Context(), inspection)
			if err != nil {
				t.Fatal(err)
			}
			defer opened.Session.Close()
			prompt := "Once upon a time — café 世界"
			options := inference.Options{MaxTokens: 8, Temperature: 0, TopP: 1}
			var streamed strings.Builder
			result, err := opened.Session.Generate(t.Context(), prompt, options, func(token inference.Token) error { streamed.Write(token.Bytes); return nil })
			if err != nil {
				t.Fatal(err)
			}
			repeat, err := opened.Session.Generate(t.Context(), prompt, options, nil)
			if err != nil || result.Text != streamed.String() || result.Text != repeat.Text || result.Tokens != repeat.Tokens || !utf8.ValidString(result.Text) || result.FinishReason == "" {
				t.Fatalf("stream/determinism: %+v %+v %v", result, repeat, err)
			}
			if result.Text == "" {
				t.Fatal("fixture must generate text to exercise stop handling")
			}
			stopOptions := options
			stopOptions.Stop = []string{result.Text}
			stopped, err := opened.Session.Generate(t.Context(), prompt, stopOptions, nil)
			if err != nil || stopped.Text != "" || stopped.FinishReason != "stop" {
				t.Fatalf("stop: %+v %v", stopped, err)
			}
			// The worker's documented context policy keeps the most recent tokens.
			longPrompt := strings.Repeat("word ", 400)
			if _, err := opened.Session.Generate(t.Context(), longPrompt, options, nil); err != nil {
				t.Fatalf("context sliding: %v", err)
			}
			opened.Session.Close()
			destination := filepath.Join(t.TempDir(), "release")
			if _, err := modelexport.ExportHuggingFace(t.Context(), inspection, destination, modelexport.Options{EUBOM: []byte("{}")}); err != nil {
				t.Fatal(err)
			}
			var tokenizerPath string
			for _, run := range inspection.BOM.Runs {
				if run.ID != inspection.BOM.CurrentRunID {
					continue
				}
				for _, artifact := range run.Artifacts {
					name := filepath.Base(artifact.Path)
					if name == "tokenizer.json" {
						tokenizerPath = filepath.Join(inspection.Path, artifact.Path)
					}
					if pin := compose.Architecture.Tokenizer.HuggingFace; pin != nil && pin.Files[name] != "" && pin.Files[name] != artifact.SHA256 {
						t.Fatalf("tokenizer pin changed: %s", name)
					}
				}
			}
			if tokenizerPath == "" {
				t.Fatal("missing retained tokenizer")
			}
			payload, err := json.Marshal(map[string]any{"source": filepath.Dir(tokenizerPath), "export": destination, "prompt": prompt, "text": result.Text, "tokens": result.Tokens, "reason": result.FinishReason})
			if err != nil {
				t.Fatal(err)
			}
			command := exec.CommandContext(t.Context(), os.Getenv("WALDO_TRANSFORMERS_PYTHON"), "../../testing/python/hf_export_parity.py", string(payload))
			output, err := command.CombinedOutput()
			if err != nil {
				t.Fatalf("independent reload: %v\n%s", err, output)
			}
			t.Logf("%s", output)
			// Corrupt only disposable test artifacts, leaving the valid export intact.
			if err := os.WriteFile(tokenizerPath, []byte("tampered"), 0600); err != nil {
				t.Fatal(err)
			}
			if bad, err := inference.Open(t.Context(), inspection); err == nil {
				bad.Session.Close()
				t.Fatal("chat accepted corrupt tokenizer")
			}
			if _, err := modelexport.ExportHuggingFace(t.Context(), inspection, filepath.Join(t.TempDir(), "bad"), modelexport.Options{EUBOM: []byte("{}")}); err == nil {
				t.Fatal("export accepted corrupt tokenizer")
			}
			prepared.Stage.Name = "tampered-continuation"
			t.Setenv("WALDO_HF_TOKENIZER_DIR", tokenizerDirectory)
			if _, err := builder.Train(t.Context(), variant.name, prepared); err == nil {
				t.Fatal("continuation accepted corrupt tokenizer")
			}
			t.Log("training, continuation, streamed chat, stop/context, independent export reload, and tamper rejection passed")
		})
	}
}
