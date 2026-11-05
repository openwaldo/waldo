// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package model

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/model/pythonenv"
	"github.com/openwaldo/waldo/internal/model/tokenizerapp"
	"github.com/openwaldo/waldo/internal/model/trainerapp"
	"github.com/openwaldo/waldo/internal/training"
)

// ExternalIdentity contains portable facts only. The existing schema-1 native
// records are deliberately untouched: opaque external architectures must never
// be decoded into Architecture or inherit native optimizer defaults.
type ExternalIdentity struct {
	Trainer                  application.ResolvedSource `json:"trainer"`
	Environment              pythonenv.Identity         `json:"environment"`
	Tokenizer                tokenizerapp.Identity      `json:"tokenizer"`
	Architecture             map[string]any             `json:"architecture"`
	ArchitectureSHA256       string                     `json:"architecture_sha256"`
	ArchitectureSchemaSHA256 string                     `json:"architecture_schema_sha256"`
	ParametersSchemaSHA256   string                     `json:"parameters_schema_sha256"`
	Determinism              string                     `json:"determinism"`
	Interaction              Interaction                `json:"interaction,omitzero"`
	Stages                   []ExternalStageIdentity    `json:"stages"`
}

type ExternalStageIdentity struct {
	Name            string                          `json:"name"`
	Type            string                          `json:"type"`
	Objective       string                          `json:"objective"`
	CorpusBOMSHA256 string                          `json:"corpus_bom_sha256"`
	Data            training.ExternalDataParameters `json:"data"`
	Conversation    training.ConversationTransform  `json:"conversation,omitzero"`
	Recipe          map[string]any                  `json:"recipe"`
	RecipeSHA256    string                          `json:"recipe_sha256"`
}

type ExternalPlan struct {
	Kind     string           `json:"kind"`
	Schema   int              `json:"schema"`
	Name     string           `json:"name"`
	Identity ExternalIdentity `json:"identity"`
}

// ExternalRunBOM embeds the same immutable identity and separates actual
// execution parameters, launch allowlists and host observations from it.
type ExternalRunBOM struct {
	Hosts          []string                       `json:"hosts,omitempty"`
	DataPlane      string                         `json:"data_plane,omitempty"`
	Initialization *training.WorkerInitialization `json:"initialization,omitempty"`
	Kind           string                         `json:"kind"`
	Schema         int                            `json:"schema"`
	Subject        string                         `json:"subject"`
	ID             string                         `json:"id"`
	ModelID        string                         `json:"model_id"`
	Stage          string                         `json:"stage"`
	Identity       ExternalIdentity               `json:"identity"`
	CorpusBOM      corpus.BOM                     `json:"corpus_bom"`
	Resolved       trainerapp.Resolved            `json:"resolved"`
	EvaluationSet  training.EvaluationSet         `json:"evaluation_set"`
	TrainerHost    pythonenv.RunEvidence          `json:"trainer_host"`
	TokenizerHost  pythonenv.RunEvidence          `json:"tokenizer_host"`
	Probe          trainerapp.Probe               `json:"probe"`
	Launches       []applaunch.Evidence           `json:"launches"`
}

type ExternalModelBOM struct {
	Kind       string           `json:"kind"`
	Schema     int              `json:"schema"`
	Subject    string           `json:"subject"`
	ModelID    string           `json:"model_id"`
	Name       string           `json:"name"`
	PlanSHA256 string           `json:"plan_sha256"`
	Identity   ExternalIdentity `json:"identity"`
	PathBase   string           `json:"path_base"`
	Runs       []ExternalBOMRun `json:"runs"`
}
type ExternalBOMRun struct {
	ID        string                `json:"id"`
	BOM       training.Artifact     `json:"bom"`
	Artifacts []trainerapp.Artifact `json:"artifacts"`
}

func sameExternalJSON(a, b any) bool {
	x, e := externalHash(a)
	y, f := externalHash(b)
	return e == nil && f == nil && x == y
}

func externalHash(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return application.Digest(b), nil
}
func checkExternalHash(v any, hash string) error {
	h, err := externalHash(v)
	if err != nil {
		return err
	}
	if h != hash {
		return fmt.Errorf("external identity digest mismatch")
	}
	return nil
}
func checkSource(s application.ResolvedSource) error {
	if err := s.Source.Validate(); err != nil {
		return err
	}
	if len(s.Tree) != 40 || !hexDigest(s.Tree, 40) || !hexDigest(s.ManifestSHA256, 64) {
		return fmt.Errorf("invalid resolved source digests")
	}
	return nil
}
func hexDigest(s string, n int) bool {
	if len(s) != n {
		return false
	}
	for _, r := range s {
		if !(r >= '0' && r <= '9' || r >= 'a' && r <= 'f') {
			return false
		}
	}
	return true
}
func checkEnvironment(e pythonenv.Identity) error {
	if e.Name == "" || e.Kind != "python-lock" || e.InterpreterVersion == "" || e.PlatformTag == "" || !hexDigest(e.LockSHA256, 64) || !hexDigest(e.InterpreterSHA256, 64) {
		return fmt.Errorf("incomplete environment identity")
	}
	return nil
}

// ValidateSchemas verifies the exact byte-preserving TRAINER-SCHEMAS.json and
// validates architecture, tokenizer configuration and role-separated recipes.
func (p ExternalPlan) ValidateSchemas(raw []byte) error {
	if p.Kind != "waldo-model-plan" || p.Schema != 2 || !validName.MatchString(p.Name) {
		return fmt.Errorf("invalid external plan envelope")
	}
	i := p.Identity
	if err := checkSource(i.Trainer); err != nil {
		return err
	}
	if err := checkSource(i.Tokenizer.Source); err != nil {
		return err
	}
	if err := checkEnvironment(i.Environment); err != nil {
		return err
	}
	if err := checkEnvironment(i.Tokenizer.Environment); err != nil {
		return err
	}
	if i.Determinism != "required" && i.Determinism != "best-effort" && i.Determinism != "off" {
		return fmt.Errorf("invalid determinism")
	}
	var schemas struct {
		Kind         string          `json:"kind"`
		Schema       int             `json:"schema"`
		Architecture json.RawMessage `json:"architecture"`
		Parameters   json.RawMessage `json:"parameters"`
		Tokenizer    json.RawMessage `json:"tokenizer"`
	}
	if err := json.Unmarshal(raw, &schemas); err != nil {
		return err
	}
	if schemas.Kind != "waldo-trainer-schemas" || schemas.Schema != 1 || application.Digest(schemas.Architecture) != i.ArchitectureSchemaSHA256 || application.Digest(schemas.Parameters) != i.ParametersSchemaSHA256 || application.Digest(schemas.Tokenizer) != i.Tokenizer.SchemaSHA256 {
		return fmt.Errorf("schema document differs from identity pins")
	}
	a, err := application.CompileSchema(schemas.Architecture, true)
	if err != nil {
		return err
	}
	_, err = application.CompileSchema(schemas.Parameters, false)
	if err != nil {
		return err
	}
	t, err := application.CompileSchema(schemas.Tokenizer, true)
	if err != nil {
		return err
	}
	if i.Architecture == nil {
		return fmt.Errorf("missing architecture")
	}
	if err := a.Validate(i.Architecture); err != nil {
		return err
	}
	if err := checkExternalHash(i.Architecture, i.ArchitectureSHA256); err != nil {
		return err
	}
	var config map[string]any
	if err := json.Unmarshal(i.Tokenizer.Configuration, &config); err != nil {
		return err
	}
	if config == nil {
		return fmt.Errorf("missing tokenizer configuration")
	}
	if err := t.Validate(config); err != nil {
		return err
	}
	if err := checkExternalHash(config, i.Tokenizer.ConfigurationSHA256); err != nil {
		return err
	}
	if len(i.Stages) == 0 {
		return fmt.Errorf("external plan requires stages")
	}
	names := map[string]bool{}
	for _, s := range i.Stages {
		if s.Name == "" || names[s.Name] || s.Type == "" || (s.Objective != "causal-language-modeling" && s.Objective != "assistant-response-modeling") || !hexDigest(s.CorpusBOMSHA256, 64) || s.Data.Steps <= 0 || s.Data.GlobalBatchSequences <= 0 || s.Data.SequenceLength <= 0 || s.Recipe == nil {
			return fmt.Errorf("invalid external stage identity")
		}
		names[s.Name] = true
		if err := checkExternalHash(s.Recipe, s.RecipeSHA256); err != nil {
			return err
		}
		// Full parameter validation needs execution values; run validation below
		// validates and splits the complete object. Recipe hashes are never supplied
		// by the trainer as a substitute for WALDO's role split.
	}
	return nil
}

func (p ExternalPlan) SHA256() (string, error) { return externalHash(p) }

// WriteExternalPlan atomically publishes a new directory containing the plan
// and byte-exact schemas together. It refuses replacement of an existing model.
func WriteExternalPlan(directory string, p ExternalPlan, schemas []byte, resolved []trainerapp.Resolved) (string, error) {
	if err := p.ValidateSchemas(schemas); err != nil {
		return "", err
	}
	if len(resolved) != len(p.Identity.Stages) {
		return "", fmt.Errorf("plan requires one complete resolve result per stage")
	}
	var documents struct{ Parameters json.RawMessage }
	if err := json.Unmarshal(schemas, &documents); err != nil {
		return "", err
	}
	compiled, err := application.CompileSchema(documents.Parameters, false)
	if err != nil {
		return "", err
	}
	for n, r := range resolved {
		recipe, execution, err := compiled.Split(r.Parameters)
		if err != nil {
			return "", err
		}
		if !sameExternalJSON(recipe, p.Identity.Stages[n].Recipe) || !sameExternalJSON(recipe, r.Recipe) || !sameExternalJSON(execution, r.Execution) || r.RecipeSHA256 != p.Identity.Stages[n].RecipeSHA256 {
			return "", fmt.Errorf("plan recipe differs from validated resolved parameters")
		}
		if err := checkExternalHash(r.Parameters, r.ParametersSHA256); err != nil {
			return "", err
		}
	}
	hash, err := p.SHA256()
	if err != nil {
		return "", err
	}
	if _, err := os.Lstat(directory); !os.IsNotExist(err) {
		return "", fmt.Errorf("external model destination already exists or cannot be inspected")
	}
	if err := os.MkdirAll(filepath.Dir(directory), 0755); err != nil {
		return "", err
	}
	temp, err := os.MkdirTemp(filepath.Dir(directory), ".external-plan-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(temp)
	if err := writeJSONAtomic(filepath.Join(temp, "PLAN.json"), p); err != nil {
		return "", err
	}
	f, err := os.OpenFile(filepath.Join(temp, "TRAINER-SCHEMAS.json"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return "", err
	}
	_, err = f.Write(schemas)
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return "", err
	}
	if closeErr != nil {
		return "", closeErr
	}
	if err := os.Rename(temp, directory); err != nil {
		return "", err
	}
	return hash, nil
}
func ReadExternalPlan(directory string) (ExternalPlan, string, error) {
	var p ExternalPlan
	if err := readJSON(filepath.Join(directory, "PLAN.json"), &p); err != nil {
		return p, "", err
	}
	schemas, err := os.ReadFile(filepath.Join(directory, "TRAINER-SCHEMAS.json"))
	if err != nil {
		return p, "", err
	}
	if err := p.ValidateSchemas(schemas); err != nil {
		return p, "", err
	}
	hash, err := p.SHA256()
	return p, hash, err
}

func WriteExternalRunBOM(directory, path string, run ExternalRunBOM) (string, error) {
	p, id, err := ReadExternalPlan(directory)
	if err != nil {
		return "", err
	}
	if run.Kind != "openwaldo-bom" || run.Schema != 2 || run.Subject != "training-run" || run.ID == "" || run.ModelID != id || !sameExternalJSON(run.Identity, p.Identity) {
		return "", fmt.Errorf("run does not match external plan")
	}
	var stage *ExternalStageIdentity
	for n := range p.Identity.Stages {
		if p.Identity.Stages[n].Name == run.Stage {
			stage = &p.Identity.Stages[n]
		}
	}
	if stage == nil {
		return "", fmt.Errorf("run stage absent from plan")
	}
	if err := checkExternalHash(run.CorpusBOM, stage.CorpusBOMSHA256); err != nil {
		return "", err
	}
	schema, err := os.ReadFile(filepath.Join(directory, "TRAINER-SCHEMAS.json"))
	if err != nil {
		return "", err
	}
	var documents struct{ Parameters json.RawMessage }
	if err := json.Unmarshal(schema, &documents); err != nil {
		return "", err
	}
	compiled, err := application.CompileSchema(documents.Parameters, false)
	if err != nil {
		return "", err
	}
	recipe, execution, err := compiled.Split(run.Resolved.Parameters)
	if err != nil {
		return "", err
	}
	if err := checkExternalHash(recipe, stage.RecipeSHA256); err != nil {
		return "", err
	}
	if err := checkExternalHash(run.Resolved.Parameters, run.Resolved.ParametersSHA256); err != nil {
		return "", err
	}
	if !sameExternalJSON(recipe, run.Resolved.Recipe) || !sameExternalJSON(execution, run.Resolved.Execution) || run.Resolved.RecipeSHA256 != stage.RecipeSHA256 || len(run.Launches) == 0 {
		return "", fmt.Errorf("run resolved values or launch evidence missing/inconsistent")
	}
	b := run.Resolved.Batch
	if (run.DataPlane != "" && run.DataPlane != training.DataPlaneNodeLocal) || (b.WorldSize > 1 && run.DataPlane != training.DataPlaneNodeLocal) {
		return "", fmt.Errorf("external distributed data plane must be node-local-cache")
	}
	if b.MicroBatchSequences <= 0 || b.AccumulationSteps <= 0 || b.WorldSize <= 0 || stage.Data.GlobalBatchSequences%b.WorldSize != 0 || stage.Data.GlobalBatchSequences/b.WorldSize%b.AccumulationSteps != 0 || stage.Data.GlobalBatchSequences/b.WorldSize/b.AccumulationSteps != b.MicroBatchSequences {
		return "", fmt.Errorf("run batch topology differs from data plan")
	}
	if application.Digest(run.Probe.Schemas.Architecture) != p.Identity.ArchitectureSchemaSHA256 || application.Digest(run.Probe.Schemas.Parameters) != p.Identity.ParametersSchemaSHA256 {
		return "", fmt.Errorf("run probe schemas differ from plan")
	}
	if _, err := os.Lstat(path); !os.IsNotExist(err) {
		return "", fmt.Errorf("immutable run BOM already exists or cannot be inspected")
	}
	if err := writeJSONAtomic(path, run); err != nil {
		return "", err
	}
	return externalHash(run)
}

// WriteExternalModelBOM is shared by managed MODEL-BOM.json and exported
// BOM.json. Artifact and run paths are relative to their respective model root;
// callers copy files first. Every referenced file is verified before publication.
func WriteExternalModelBOM(directory string, exported bool, runs []ExternalBOMRun) (ExternalModelBOM, error) {
	var bom ExternalModelBOM
	p, id, err := ReadExternalPlan(directory)
	if err != nil {
		return bom, err
	}
	seen := map[string]bool{}
	for _, r := range runs {
		if r.ID == "" || seen[r.ID] || !filepath.IsLocal(r.BOM.Path) || filepath.Clean(r.BOM.Path) != r.BOM.Path {
			return bom, fmt.Errorf("invalid external run reference")
		}
		seen[r.ID] = true
		// Reuse rooted publication checks for run JSON as well as released tensors.
		if err := verifyExternalRunFile(directory, r.BOM, p, id, r.ID); err != nil {
			return bom, err
		}
		if err := trainerapp.VerifyPublished(directory, r.Artifacts); err != nil {
			return bom, err
		}
	}
	bom = ExternalModelBOM{Kind: "openwaldo-bom", Schema: 2, Subject: "model", ModelID: id, Name: p.Name, PlanSHA256: id, Identity: p.Identity, PathBase: "model-root", Runs: runs}
	name := "MODEL-BOM.json"
	if exported {
		name = "BOM.json"
	}
	return bom, writeJSONAtomic(filepath.Join(directory, name), bom)
}
func verifyExternalRunFile(directory string, a training.Artifact, p ExternalPlan, id, runID string) error {
	root, err := os.OpenRoot(directory)
	if err != nil {
		return err
	}
	defer root.Close()
	f, err := root.Open(a.Path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() != a.Bytes || a.Bytes > 16<<20 {
		return fmt.Errorf("invalid run BOM file size")
	}
	raw := make([]byte, a.Bytes)
	if _, err := io.ReadFull(f, raw); err != nil {
		return err
	}
	if application.Digest(raw) != a.SHA256 {
		return fmt.Errorf("run BOM file hash mismatch")
	}
	var r ExternalRunBOM
	if err := json.Unmarshal(raw, &r); err != nil {
		return err
	}
	if r.Kind != "openwaldo-bom" || r.Schema != 2 || r.Subject != "training-run" || r.ID != runID || r.ModelID != id || !sameExternalJSON(r.Identity, p.Identity) {
		return fmt.Errorf("aggregate run identity mismatch")
	}
	return nil
}
