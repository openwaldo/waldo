// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package modelexport

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

func transformersExportFixture(t *testing.T) model.Inspection {
	t.Helper()
	compose, _, err := model.LoadCompose("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	run := model.ModelBOMRun{ID: "run", State: model.RunComplete, Backend: training.Identity{Name: training.BackendTransformers, Revision: training.TransformersRevision}}
	for name, data := range map[string]string{"model.safetensors": "provider-native-weight-bytes", "config.json": `{"model_type":"llama"}`, "tokenizer.json": `{"name":"byte"}`, "training_args.json": `{"learning_rate":0.001}`, "runtime.json": `{"device":"cpu"}`} {
		path := filepath.Join("runs", "run", "artifacts", name)
		if err := os.MkdirAll(filepath.Dir(filepath.Join(root, path)), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, path), []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256([]byte(data))
		run.Artifacts = append(run.Artifacts, model.ModelBOMArtifact{Path: filepath.ToSlash(path), SHA256: hex.EncodeToString(hash[:]), Bytes: int64(len(data)), Role: "artifact"})
	}
	return model.Inspection{Path: root, Model: model.ModelRecord{ID: "model", Name: "smoke", Architecture: compose.Architecture},
		BOM:     model.ModelBOM{CurrentRunID: "run", Runs: []model.ModelBOMRun{run}},
		RunBOMs: []model.RunBOM{{ID: "run", Execution: training.Execution{Package: &compose.Architecture.Transformers.Package}}}}
}

func TestTransformersExportPreservesBytesAndSignsEvidence(t *testing.T) {
	inspection := transformersExportFixture(t)
	finalized := false
	destination := filepath.Join(t.TempDir(), "release")
	_, err := ExportHuggingFace(context.Background(), inspection, destination, Options{EUBOM: []byte("{}\n"), Finalize: func(root string) error {
		finalized = true
		data, err := os.ReadFile(filepath.Join(root, "BOM.json"))
		if err != nil {
			return err
		}
		var bom releaseBOM
		if err := json.Unmarshal(data, &bom); err != nil {
			return err
		}
		found := false
		for _, artifact := range bom.Artifacts {
			if err := model.VerifyArtifactFile(filepath.Join(root, artifact.Path), training.Artifact{Path: artifact.Path, SHA256: artifact.SHA256, Bytes: artifact.Bytes}); err != nil {
				return err
			}
			found = found || artifact.Path == "training-run.json"
		}
		if !found {
			return fmt.Errorf("package provenance not bound by release BOM")
		}
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !finalized {
		t.Fatal("normal signing callback was not invoked")
	}
	data, err := os.ReadFile(filepath.Join(destination, "model.safetensors"))
	if err != nil || string(data) != "provider-native-weight-bytes" {
		t.Fatalf("provider weights were rewritten: %v", err)
	}
	data, err = os.ReadFile(filepath.Join(destination, "training-run.json"))
	if err != nil {
		t.Fatal(err)
	}
	var run model.RunBOM
	if err := json.Unmarshal(data, &run); err != nil || run.Execution.Package.SHA256 != inspection.RunBOMs[0].Execution.Package.SHA256 {
		t.Fatal("release lost exact wheel pin")
	}
}

func TestTransformersExportPreservesPinnedTokenizer(t *testing.T) {
	inspection := transformersExportFixture(t)
	pin := &training.HuggingFaceTokenizer{Files: map[string]string{}}
	inspection.Model.Architecture.Tokenizer = model.Tokenizer{Name: "huggingface", Revision: "fixture", HuggingFace: pin}
	for name, data := range map[string]string{"tokenizer_config.json": "{}", "waldo_tokenizer.json": "{}"} {
		path := filepath.Join("runs", "run", "artifacts", name)
		if err := os.WriteFile(filepath.Join(inspection.Path, path), []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256([]byte(data))
		inspection.BOM.Runs[0].Artifacts = append(inspection.BOM.Runs[0].Artifacts, model.ModelBOMArtifact{Path: filepath.ToSlash(path), SHA256: hex.EncodeToString(hash[:]), Bytes: int64(len(data)), Role: "tokenizer"})
	}
	for _, artifact := range inspection.BOM.Runs[0].Artifacts {
		name := filepath.Base(artifact.Path)
		if name == "tokenizer.json" || name == "tokenizer_config.json" {
			pin.Files[name] = artifact.SHA256
		}
	}
	destination := filepath.Join(t.TempDir(), "release")
	if _, err := ExportHuggingFace(t.Context(), inspection, destination, Options{EUBOM: []byte("{}")}); err != nil {
		t.Fatal(err)
	}
	for name, hash := range pin.Files {
		data, err := os.ReadFile(filepath.Join(destination, name))
		if err != nil {
			t.Fatal(err)
		}
		actual := sha256.Sum256(data)
		if hex.EncodeToString(actual[:]) != hash {
			t.Fatal("tokenizer changed during export")
		}
	}
	if _, err := os.Stat(filepath.Join(destination, "tokenization_openwaldo.py")); !os.IsNotExist(err) {
		t.Fatal("export injected a byte tokenizer")
	}
	pin.Files["tokenizer.json"] = "wrong"
	if _, err := ExportHuggingFace(t.Context(), inspection, filepath.Join(t.TempDir(), "bad"), Options{EUBOM: []byte("{}")}); err == nil {
		t.Fatal("export ignored tokenizer compose pin")
	}
}

func TestTransformersExportRejectsTampering(t *testing.T) {
	inspection := transformersExportFixture(t)
	path := filepath.Join(inspection.Path, filepath.FromSlash(inspection.BOM.Runs[0].Artifacts[0].Path))
	if err := os.WriteFile(path, []byte("tampered"), 0644); err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(t.TempDir(), "release")
	if _, err := ExportHuggingFace(context.Background(), inspection, destination, Options{EUBOM: []byte("{}")}); err == nil {
		t.Fatal("tampered artifact exported")
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatal("failed export published a destination")
	}
}
