// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

const hfTokenizerCompose = "../../docs/examples/transformers-qwen3-tokenizer-smoke.yaml"

func TestHuggingFaceTokenizerCompose(t *testing.T) {
	compose, _, err := LoadCompose(hfTokenizerCompose)
	if err != nil {
		t.Fatal(err)
	}
	path, err := ArchiveCompose(t.TempDir(), compose, "hf-tokenizer")
	if err != nil {
		t.Fatal(err)
	}
	archived, _, err := LoadCompose(path)
	if err != nil || !reflect.DeepEqual(compose, archived) {
		t.Fatalf("tokenizer round trip: %v", err)
	}
	compose.Architecture.Tokenizer.Revision = "main"
	if compose.Architecture.Validate() == nil {
		t.Fatal("accepted mutable tokenizer revision")
	}
}

func TestTransformersHuggingFaceTokenizerSmoke(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" || os.Getenv("WALDO_HF_TOKENIZER_DIR") == "" {
		t.Skip("set the Transformers runtime and WALDO_HF_TOKENIZER_DIR for real training")
	}
	compose, _, err := LoadCompose(hfTokenizerCompose)
	if err != nil {
		t.Fatal(err)
	}
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
	builder := Builder{Root: t.TempDir()}
	inspection, err := builder.Compose(t.Context(), "hf-tokenizer", compose, []PreparedStage{preparedFixture(t, compose.Stages[0])})
	if err != nil {
		t.Fatal(err)
	}
	observation := inspection.Runs[0].Observation
	if inspection.Runs[0].State != RunComplete || observation == nil || observation.Simulated || observation.Steps != 2 || observation.ConsumedTokens != 16 || len(observation.Evaluations) != 1 {
		t.Fatalf("unexpected result: %+v", observation)
	}
	directory := filepath.Join(inspection.Path, "runs", runDirectoryName(inspection.Model.Runs[0]))
	found := map[string]bool{}
	for _, artifact := range observation.Artifacts {
		if err := VerifyArtifactFile(filepath.Join(directory, artifact.Path), artifact); err != nil {
			t.Fatal(err)
		}
		name := filepath.Base(artifact.Path)
		found[name] = true
		if expected := compose.Architecture.Tokenizer.HuggingFace.Files[name]; expected != "" && expected != artifact.SHA256 {
			t.Fatalf("tokenizer bytes changed: %s", name)
		}
	}
	if !found["waldo_tokenizer.json"] || !found["tokenizer.json"] || !found["tokenizer_config.json"] {
		t.Fatal("missing tokenizer evidence")
	}
	t.Logf("standard Qwen tokenizer: %d steps, %d targets, loss %.4f", observation.Steps, observation.ConsumedTokens, *observation.FinalLoss)
}
