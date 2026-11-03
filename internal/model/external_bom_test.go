// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package model

import (
	"encoding/binary"
	"encoding/json"
	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/model/pythonenv"
	"github.com/openwaldo/waldo/internal/model/tokenizerapp"
	"github.com/openwaldo/waldo/internal/model/trainerapp"
	"github.com/openwaldo/waldo/internal/training"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func externalBOMFixture(t *testing.T) (ExternalPlan, []byte, ExternalRunBOM) {
	t.Helper()
	arch := json.RawMessage(`{ "type": "object" }`)
	params := json.RawMessage(`{ "type":"object", "properties":{"rate":{"type":"number"},"micro":{"type":"integer","x-waldo-role":"execution"}},"required":["rate","micro"] }`)
	tok := json.RawMessage(`{ "type":"object", "x-extra": true }`)
	schemas := []byte(`{"kind":"waldo-trainer-schemas","schema":1,"architecture":` + string(arch) + `,"parameters":` + string(params) + `,"tokenizer":` + string(tok) + `}`)
	source := application.ResolvedSource{Source: application.Source{Repository: "https://example.invalid/trainer", Commit: strings.Repeat("a", 40), Path: "."}, Tree: strings.Repeat("b", 40), ManifestSHA256: strings.Repeat("c", 64)}
	env := pythonenv.Identity{Name: "cpu", Kind: "python-lock", LockSHA256: strings.Repeat("d", 64), InterpreterVersion: "3.12.3", InterpreterSHA256: strings.Repeat("e", 64), PlatformTag: "linux-aarch64"}
	tokenizer := tokenizerapp.Identity{Source: source, Environment: env, Configuration: json.RawMessage(`{}`), ConfigurationSHA256: application.Digest([]byte(`{}`)), SchemaSHA256: application.Digest(tok)}
	tokenizer.Probe.Schemas.Configuration = tok
	tokenizer.Probe.Name = "test"
	tokenizer.Probe.VocabularySize = 10
	corpusBOM := corpus.BOM{}
	ch, _ := externalHash(corpusBOM)
	recipe := map[string]any{"rate": 0.1}
	rh, _ := externalHash(recipe)
	architecture := map[string]any{"width": float64(16)}
	ah, _ := externalHash(architecture)
	p := ExternalPlan{Kind: "waldo-model-plan", Schema: 2, Name: "external", Identity: ExternalIdentity{Trainer: source, Environment: env, Tokenizer: tokenizer, Architecture: architecture, ArchitectureSHA256: ah, ArchitectureSchemaSHA256: application.Digest(arch), ParametersSchemaSHA256: application.Digest(params), Determinism: "best-effort", Stages: []ExternalStageIdentity{{Name: "train", Type: "pre-training", Objective: "causal-language-modeling", CorpusBOMSHA256: ch, Data: training.ExternalDataParameters{Steps: 2, GlobalBatchSequences: 2, SequenceLength: 4}, Recipe: recipe, RecipeSHA256: rh}}}}
	id, _ := p.SHA256()
	resolved := trainerapp.Resolved{Parameters: map[string]any{"rate": 0.1, "micro": float64(2)}, Recipe: recipe, Execution: map[string]any{"micro": float64(2)}, RecipeSHA256: rh, Batch: trainerapp.Batch{MicroBatchSequences: 2, AccumulationSteps: 1, WorldSize: 1}}
	resolved.ParametersSHA256, _ = externalHash(resolved.Parameters)
	r := ExternalRunBOM{Kind: "openwaldo-bom", Schema: 2, Subject: "training-run", ID: "run", ModelID: id, Stage: "train", Identity: p.Identity, CorpusBOM: corpusBOM, Resolved: resolved, Probe: trainerapp.Probe{Schemas: trainerapp.Schemas{Architecture: arch, Parameters: params}}, Launches: []applaunch.Evidence{{Profile: "single-process", Launcher: "python", Environment: []string{"HOME=/scratch"}}}}
	return p, schemas, r
}
func TestExternalBOMRoundTrip(t *testing.T) {
	p, schemas, run := externalBOMFixture(t)
	dir := filepath.Join(t.TempDir(), "model")
	id, err := WriteExternalPlan(dir, p, schemas, []trainerapp.Resolved{run.Resolved})
	if err != nil {
		t.Fatal(err)
	}
	read, got, err := ReadExternalPlan(dir)
	if err != nil || id != got || !sameExternalJSON(read, p) {
		t.Fatalf("round trip: %v", err)
	}
	exact, _ := os.ReadFile(filepath.Join(dir, "TRAINER-SCHEMAS.json"))
	if string(exact) != string(schemas) {
		t.Fatal("schema bytes changed")
	}
	if _, err := WriteExternalPlan(dir, p, schemas, []trainerapp.Resolved{run.Resolved}); err == nil {
		t.Fatal("replaced immutable plan")
	}
	path := filepath.Join(dir, "runs", "run", "RUN-BOM.json")
	if _, err := WriteExternalRunBOM(dir, path, run); err != nil {
		t.Fatal(err)
	}
	if _, err := WriteExternalRunBOM(dir, path, run); err == nil {
		t.Fatal("replaced immutable run")
	}
	rawRun, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	header := []byte(`{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`)
	weight := make([]byte, 8)
	binary.LittleEndian.PutUint64(weight, uint64(len(header)))
	weight = append(weight, header...)
	weight = append(weight, 0, 0, 128, 63)
	weightPath := "runs/run/artifacts/model.safetensors"
	if err := os.MkdirAll(filepath.Dir(filepath.Join(dir, weightPath)), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, weightPath), weight, 0600); err != nil {
		t.Fatal(err)
	}
	runs := []ExternalBOMRun{{ID: run.ID, BOM: training.Artifact{Path: "runs/run/RUN-BOM.json", Bytes: int64(len(rawRun)), SHA256: application.Digest(rawRun)}, Artifacts: []trainerapp.Artifact{{Artifact: training.Artifact{Path: weightPath, Bytes: int64(len(weight)), SHA256: application.Digest(weight)}, Role: "weights", Format: "safetensors"}}}}
	// Managed and export projections carry identical provenance and verified files.
	for _, exported := range []bool{false, true} {
		bom, err := WriteExternalModelBOM(dir, exported, runs)
		if err != nil {
			t.Fatal(err)
		}
		if !sameExternalJSON(bom.Identity, p.Identity) || bom.PlanSHA256 != id {
			t.Fatal("identity lost")
		}
		name := "MODEL-BOM.json"
		if exported {
			name = "BOM.json"
		}
		var restored ExternalModelBOM
		if err := readJSON(filepath.Join(dir, name), &restored); err != nil || !sameExternalJSON(bom, restored) {
			t.Fatal("BOM round trip", err)
		}
	}
	runs[0].BOM.SHA256 = strings.Repeat("f", 64)
	if _, err := WriteExternalModelBOM(dir, true, runs); err == nil {
		t.Fatal("accepted corrupt run BOM pin")
	}
	// The run document's JSON encoding must carry both identity and host allowlist.
	var restored ExternalRunBOM
	if err := readJSON(path, &restored); err != nil || !sameExternalJSON(restored, run) {
		t.Fatal("run round trip", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "TRAINER-SCHEMAS.json"), append(schemas, '!'), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ReadExternalPlan(dir); err == nil {
		t.Fatal("accepted corrupt schemas")
	}
}
func TestExternalIdentitySensitivity(t *testing.T) {
	p, _, r := externalBOMFixture(t)
	original, _ := p.SHA256()
	mutations := []func(*ExternalPlan){
		func(p *ExternalPlan) { p.Identity.Trainer.Source.Commit = strings.Repeat("f", 40) },
		func(p *ExternalPlan) { p.Identity.Trainer.Tree = strings.Repeat("f", 40) },
		func(p *ExternalPlan) { p.Identity.Trainer.ManifestSHA256 = strings.Repeat("f", 64) },
		func(p *ExternalPlan) { p.Identity.Tokenizer.Source.Source.Path = "tokenizer" },
		func(p *ExternalPlan) { p.Identity.Environment.LockSHA256 = strings.Repeat("f", 64) },
		func(p *ExternalPlan) { p.Identity.Environment.InterpreterVersion = "3.13.1" },
		func(p *ExternalPlan) { p.Identity.Tokenizer.ConfigurationSHA256 = strings.Repeat("f", 64) },
		func(p *ExternalPlan) { p.Identity.ParametersSchemaSHA256 = strings.Repeat("f", 64) },
		func(p *ExternalPlan) { p.Identity.Stages[0].RecipeSHA256 = strings.Repeat("f", 64) },
		func(p *ExternalPlan) { p.Identity.Stages[0].CorpusBOMSHA256 = strings.Repeat("f", 64) },
	}
	for n, mutate := range mutations {
		raw, _ := json.Marshal(p)
		var clone ExternalPlan
		json.Unmarshal(raw, &clone)
		mutate(&clone)
		got, _ := clone.SHA256()
		if got == original {
			t.Fatalf("mutation %d did not alter identity", n)
		}
	}
	before, _ := externalHash(r)
	r.TrainerHost.CachePath = "/different/cache"
	r.Probe.Driver = "new driver"
	r.Resolved.Execution["micro"] = float64(1)
	after, _ := externalHash(r)
	unchanged, _ := p.SHA256()
	if before == after || unchanged != original {
		t.Fatal("execution leaked into identity or vanished from run")
	}
}
func TestExternalRunRejectsIdentityChanges(t *testing.T) {
	for _, mode := range []string{"recipe", "parameters-hash", "execution", "batch", "corpus", "probe", "identity", "data-plane"} {
		t.Run(mode, func(t *testing.T) {
			p, schemas, r := externalBOMFixture(t)
			dir := filepath.Join(t.TempDir(), "model")
			if _, err := WriteExternalPlan(dir, p, schemas, []trainerapp.Resolved{r.Resolved}); err != nil {
				t.Fatal(err)
			}
			switch mode {
			case "data-plane":
				r.DataPlane = "rank-zero-broadcast"
			case "recipe":
				r.Resolved.Parameters["rate"] = 0.2
			case "parameters-hash":
				r.Resolved.ParametersSHA256 = strings.Repeat("f", 64)
			case "execution":
				r.Resolved.Execution["micro"] = float64(1)
			case "batch":
				r.Resolved.Batch.WorldSize = 3
			case "corpus":
				r.CorpusBOM.Kind = "changed"
			case "probe":
				r.Probe.Schemas.Parameters = json.RawMessage(`{}`)
			case "identity":
				r.Identity.Environment.Name = "cuda"
			}
			if _, err := WriteExternalRunBOM(dir, filepath.Join(dir, "RUN-BOM.json"), r); err == nil {
				t.Fatal("accepted inconsistent run")
			}
		})
	}
}

func TestExternalPlanRejectsUnresolvedRecipe(t *testing.T) {
	p, schemas, r := externalBOMFixture(t)
	p.Identity.Stages[0].Recipe["micro"] = float64(2)
	p.Identity.Stages[0].RecipeSHA256, _ = externalHash(p.Identity.Stages[0].Recipe)
	dir := filepath.Join(t.TempDir(), "model")
	if _, err := WriteExternalPlan(dir, p, schemas, []trainerapp.Resolved{r.Resolved}); err == nil {
		t.Fatal("execution value entered recipe identity")
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("invalid plan created model state")
	}
}
