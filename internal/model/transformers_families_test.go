// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/inference"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/modelexport"
	"github.com/openwaldo/waldo/internal/record"
	"github.com/openwaldo/waldo/internal/shard"
	"github.com/openwaldo/waldo/internal/training"
	"github.com/parquet-go/parquet-go"
)

var hfFamilies = []string{"mistral", "gemma2", "phi3", "olmo2", "mixtral", "qwen35"}

func TestTransformersFamilyComposes(t *testing.T) {
	for _, family := range hfFamilies {
		t.Run(family, func(t *testing.T) {
			compose, _, err := model.LoadCompose("../../docs/examples/transformers-" + family + "-smoke.yaml")
			if err != nil {
				t.Fatal(err)
			}
			if compose.Base != nil || compose.Architecture.Layers != 2 {
				t.Fatal("expected custom two-layer random model")
			}
		})
	}
}

// Opt-in, no acquisition/publication: generated Parquet -> train -> continue ->
// chat -> export. Set WALDO_TRANSFORMERS_DEVICE=cuda to make GPU mandatory.
func TestTransformersFamilyLifecycle(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" {
		t.Skip("set Transformers Python and wheel for real lifecycle tests")
	}
	for _, family := range hfFamilies {
		t.Run(family, func(t *testing.T) {
			compose, _, err := model.LoadCompose("../../docs/examples/transformers-" + family + "-smoke.yaml")
			if err != nil {
				t.Fatal(err)
			}
			compose.Stages[0].Corpora = model.NewCorpusSelections([]string{"example"})
			if precision := os.Getenv("WALDO_HF_TEST_PRECISION"); precision != "" {
				if precision != "bf16" && precision != "fp16" {
					t.Fatal("WALDO_HF_TEST_PRECISION must be fp16 or bf16")
				}
				compose.Stages[0].Parameters.Trainer.Arguments[precision] = true
			}
			builder := model.Builder{Root: t.TempDir()}
			prepared := hfPrepared(t, compose.Stages[0])
			inspection, err := builder.Compose(t.Context(), family, compose, []model.PreparedStage{prepared})
			if err != nil {
				t.Fatal(err)
			}
			prepared.Stage.Name = "continue"
			prepared.Stage.Parameters.Trainer.Arguments["gradient_accumulation_steps"] = float64(2)
			prepared.Stage.Parameters.Trainer.Arguments["per_device_train_batch_size"] = float64(1)
			inspection, err = builder.Train(t.Context(), family, prepared)
			if err != nil {
				t.Fatal(err)
			}
			if inspection.RunBOMs[1].Initialization == nil || inspection.Runs[1].Observation.ConsumedTokens != 128 {
				t.Fatal("continuation/accounting failed")
			}
			if os.Getenv("WALDO_TRANSFORMERS_DEVICE") == "cuda" && len(inspection.RunBOMs[1].Execution.Accelerators) != 1 {
				t.Fatal("GPU test ran without accelerator evidence")
			}
			opened, err := inference.Open(t.Context(), inspection)
			if err != nil {
				t.Fatal(err)
			}
			result, err := opened.Session.Generate(t.Context(), "Once upon a time", inference.Options{MaxTokens: 4, Temperature: 0, TopP: 1}, nil)
			opened.Session.Close()
			if err != nil || result.FinishReason == "" {
				t.Fatalf("chat: %+v %v", result, err)
			}
			destination := filepath.Join(t.TempDir(), "release")
			if _, err := modelexport.ExportHuggingFace(t.Context(), inspection, destination, modelexport.Options{EUBOM: []byte("{}")}); err != nil {
				t.Fatal(err)
			}
			var evidence map[string]any
			data, err := os.ReadFile(filepath.Join(destination, "runtime.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(data, &evidence); err != nil {
				t.Fatal(err)
			}
			if evidence["parameters"].(float64) <= 0 {
				t.Fatal("no parameter evidence")
			}
			if family == "mixtral" && (evidence["active_parameters_estimate"].(float64) >= evidence["parameters"].(float64) || evidence["expert_parameters"].(float64) <= 0) {
				t.Fatal("missing total/active expert evidence")
			}
			t.Logf("%s: %v parameters, device %v, continuation + chat + export passed", family, evidence["parameters"], evidence["device"])
		})
	}
}

func hfPrepared(t *testing.T, stage model.Stage) model.PreparedStage {
	t.Helper()
	text := strings.Repeat("canonical parquet fixture ", 8)
	var encoded bytes.Buffer
	writer := parquet.NewGenericWriter[shard.Row](&encoded)
	for _, text := range []string{text, text + " second"} {
		if _, err := writer.Write([]shard.Row{{SHA256: record.TextHash(text), Kind: record.KindPretrain, Text: text, Source: "fixture", License: "CC0-1.0", Tokens: 128}}); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	data := encoded.Bytes()
	hash := sha256.Sum256(data)
	digest := hex.EncodeToString(hash[:])
	path := filepath.Join(t.TempDir(), digest)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	conversion := index.Conversion{Tool: "test", Version: "1", Profile: "text", Recipe: "test/v1", Tokenizer: "byte"}
	measures := index.Measures{Shards: 1, Docs: 2, Tokens: 256, Bytes: int64(len(data))}
	bom := corpus.BOM{Kind: "openwaldo-bom", Schema: 1, Subject: "corpus", Paths: []string{"example"}, Licenses: map[string]index.Measures{"CC0-1.0": measures}, Totals: measures,
		Manifests: []corpus.ManifestPin{{Path: "example/example.json", SHA256: strings.Repeat("a", 64), Name: "example", Title: "Example", Description: "HF test fixture", License: "CC0-1.0", Format: "parquet", RecordSchema: 1, ConvertedBy: conversion, Sources: []index.Source{{Name: "fixture", Source: "Fixture", URL: "https://example.test", SHA256: strings.Repeat("b", 64)}}, Totals: measures, Licenses: map[string]index.Measures{"CC0-1.0": measures}}},
		Shards:    []corpus.ShardPin{{Manifest: "example/example.json", URL: "https://objects.example/" + digest, SHA256: digest, Format: "parquet", RecordSchema: 1, License: "CC0-1.0", ConvertedBy: conversion, Docs: 2, Tokens: 256, Bytes: int64(len(data))}}}
	prepared, err := model.PrepareStage(stage, bom, []training.Input{{Path: path, SHA256: digest, Bytes: int64(len(data)), Records: 2}})
	if err != nil {
		t.Fatal(err)
	}
	return prepared
}
