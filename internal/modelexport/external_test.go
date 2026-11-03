// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package modelexport

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/model/pythonenv"
	"github.com/openwaldo/waldo/internal/model/tokenizerapp"
	"github.com/openwaldo/waldo/internal/model/trainerapp"
	"github.com/openwaldo/waldo/internal/training"
)

func externalExportFixture(t *testing.T) model.ExternalInspection {
	t.Helper()
	hash := func(v any) string {
		raw, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return application.Digest(raw)
	}
	schema := json.RawMessage(`{"type":"object"}`)
	source := application.ResolvedSource{Source: application.Source{Repository: "https://example.invalid/repo", Commit: strings.Repeat("a", 40), Path: "."}, Tree: strings.Repeat("b", 40), ManifestSHA256: strings.Repeat("c", 64)}
	environment := pythonenv.Identity{Name: "cpu", Kind: "python-lock", LockSHA256: strings.Repeat("d", 64), InterpreterVersion: "3.12.3", InterpreterSHA256: strings.Repeat("e", 64), PlatformTag: "linux"}
	empty := map[string]any{}
	token := tokenizerapp.Identity{Source: source, Environment: environment, Configuration: json.RawMessage(`{}`), ConfigurationSHA256: hash(empty), SchemaSHA256: application.Digest(schema)}
	token.Probe.Schemas.Configuration = schema
	plan := model.ExternalPlan{Kind: "waldo-model-plan", Schema: 2, Name: "fixture", Identity: model.ExternalIdentity{Trainer: source, Environment: environment, Tokenizer: token, Architecture: empty, ArchitectureSHA256: hash(empty), ArchitectureSchemaSHA256: application.Digest(schema), ParametersSchemaSHA256: application.Digest(schema), Determinism: "required", Stages: []model.ExternalStageIdentity{{Name: "train", Type: "pre-training", Objective: "causal-language-modeling", CorpusBOMSHA256: hash(corpus.BOM{}), Data: training.ExternalDataParameters{Steps: 1, GlobalBatchSequences: 1, SequenceLength: 2}, Recipe: empty, RecipeSHA256: hash(empty)}}}}
	resolved := trainerapp.Resolved{Parameters: empty, Recipe: empty, Execution: empty, ParametersSHA256: hash(empty), RecipeSHA256: hash(empty), Batch: trainerapp.Batch{MicroBatchSequences: 1, AccumulationSteps: 1, WorldSize: 1}}
	directory := filepath.Join(t.TempDir(), "model")
	schemas := []byte(`{"kind":"waldo-trainer-schemas","schema":1,"architecture":` + string(schema) + `,"parameters":` + string(schema) + `,"tokenizer":` + string(schema) + `}`)
	id, err := model.WriteExternalPlan(directory, plan, schemas, []trainerapp.Resolved{resolved})
	if err != nil {
		t.Fatal(err)
	}
	run := model.ExternalRunBOM{Kind: "openwaldo-bom", Schema: 2, Subject: "training-run", ID: "train", Stage: "train", ModelID: id, Identity: plan.Identity, Resolved: resolved, Probe: trainerapp.Probe{Schemas: trainerapp.Schemas{Architecture: schema, Parameters: schema}}, Launches: []applaunch.Evidence{{Profile: "single-process"}}}
	runPath := "runs/train/RUN-BOM.json"
	if _, err = model.WriteExternalRunBOM(directory, filepath.Join(directory, runPath), run); err != nil {
		t.Fatal(err)
	}
	header := []byte(`{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`)
	payload := make([]byte, 8+len(header)+4)
	binary.LittleEndian.PutUint64(payload, uint64(len(header)))
	copy(payload[8:], header)
	artifactPath := "runs/train/artifacts/model.safetensors"
	if err = os.MkdirAll(filepath.Dir(filepath.Join(directory, artifactPath)), 0700); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(directory, artifactPath), payload, 0600); err != nil {
		t.Fatal(err)
	}
	// Private checkpoint files must never be copied by export.
	if err = os.WriteFile(filepath.Join(directory, "private.pkl"), []byte("private"), 0600); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(directory, runPath))
	if err != nil {
		t.Fatal(err)
	}
	refs := []model.ExternalBOMRun{{ID: "train", BOM: training.Artifact{Path: runPath, SHA256: application.Digest(raw), Bytes: int64(len(raw))}, Artifacts: []trainerapp.Artifact{{Artifact: training.Artifact{Path: artifactPath, SHA256: application.Digest(payload), Bytes: int64(len(payload))}, Role: "weights", Format: "safetensors"}}}}
	bom, err := model.WriteExternalModelBOM(directory, false, refs)
	if err != nil {
		t.Fatal(err)
	}
	return model.ExternalInspection{Path: directory, Plan: plan, Model: model.ExternalModelRecord{ID: id, Name: "fixture", State: model.RunComplete}, BOM: bom, Runs: []model.ExternalRunRecord{{ID: "train", State: model.RunComplete, Complete: &trainerapp.Complete{ArtifactProfile: "hf-causal-lm"}}}}
}

func TestExternalExportCopiesVerifiedPackageAndFinalizes(t *testing.T) {
	inspection := externalExportFixture(t)
	destination := filepath.Join(t.TempDir(), "export")
	called := false
	_, err := ExportExternal(t.Context(), inspection, destination, Options{EUBOM: []byte("{}"), Finalize: func(dir string) error {
		called = true
		for _, name := range []string{"BOM.json", "EU-BOM.json", "PLAN.json", "TRAINER-SCHEMAS.json", "model.safetensors"} {
			if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
				return err
			}
		}
		return os.WriteFile(filepath.Join(dir, "BOM.sigstore.json"), []byte("test signature"), 0600)
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("finalizer not called")
	}
	if _, err = os.Stat(filepath.Join(destination, "private.pkl")); !os.IsNotExist(err) {
		t.Fatal("private checkpoint exported")
	}
	raw, err := os.ReadFile(filepath.Join(destination, "BOM.json"))
	if err != nil {
		t.Fatal(err)
	}
	var bom model.ExternalModelBOM
	if err = json.Unmarshal(raw, &bom); err != nil {
		t.Fatal(err)
	}
	if bom.Runs[0].Artifacts[0].Path != "model.safetensors" || bom.ModelID != inspection.Model.ID {
		t.Fatal("export BOM changed identity or paths")
	}
	if _, err = ExportExternal(t.Context(), inspection, destination, Options{}); err == nil {
		t.Fatal("existing export overwritten")
	}
}

func TestExternalExportRejectsCorruptionAndSigningFailure(t *testing.T) {
	inspection := externalExportFixture(t)
	destination := filepath.Join(t.TempDir(), "export")
	sentinel := errors.New("signing failed")
	if _, err := ExportExternal(t.Context(), inspection, destination, Options{Finalize: func(string) error { return sentinel }}); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatal("failed signature published export")
	}
	if err := os.WriteFile(filepath.Join(inspection.Path, inspection.BOM.Runs[0].Artifacts[0].Path), []byte("corrupt"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := ExportExternal(t.Context(), inspection, destination, Options{}); err == nil {
		t.Fatal("corrupt weights exported")
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatal("corrupt export published")
	}
}
