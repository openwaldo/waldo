// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/model/application"
	"gopkg.in/yaml.v3"
)

const externalComposeYAML = `kind: waldo-model-compose
schema: 2
trainer:
  source: &source
    repository: https://example.invalid/trainer.git
    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    path: app
  environment: cpu
  determinism: best-effort
tokenizer:
  source: *source
  environment: cpu
  configuration: {}
architecture: {model_type: qwen3, width: 64}
stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    corpora: [books]
    data: {tokens: 128, global_batch_sequences: 2, sequence_length: 32, seed: 42}
    trainer: {learning_rate: 0.001}
`

func loadExternalFixture(t *testing.T, data string) (ComposeDocument, error) {
	t.Helper()
	p := filepath.Join(t.TempDir(), "compose.yaml")
	if err := os.WriteFile(p, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
	doc, _, err := LoadComposeDocument(p)
	return doc, err
}
func TestExternalComposeParsing(t *testing.T) {
	doc, err := loadExternalFixture(t, externalComposeYAML)
	if err != nil {
		t.Fatal(err)
	}
	if doc.Builtin != nil || doc.External == nil || doc.External.Trainer.Source != doc.External.Tokenizer.Source {
		t.Fatal("bad dispatch or alias expansion")
	}
	c := doc.External
	resolved, err := c.Stages[0].ResolveData()
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Steps != 2 || resolved.GlobalBatchSequences != 2 {
		t.Fatalf("wrong budget: %+v", resolved)
	}
	for _, marshal := range []func(any) ([]byte, error){json.Marshal, yaml.Marshal} {
		data, err := marshal(c)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := loadExternalFixture(t, string(data)); err != nil {
			t.Fatal(err)
		}
	}
	cases := map[string]string{
		"schema 1":          strings.Replace(externalComposeYAML, "schema: 2", "schema: 1", 1),
		"schema 3":          strings.Replace(externalComposeYAML, "schema: 2", "schema: 3", 1),
		"branch":            strings.Replace(externalComposeYAML, strings.Repeat("a", 40), "main", 1),
		"source escape":     strings.Replace(externalComposeYAML, "path: app", "path: ../app", 1),
		"unknown top":       externalComposeYAML + "extra: true\n",
		"unknown source":    strings.Replace(externalComposeYAML, "path: app", "path: app\n    extra: true", 1),
		"unknown data":      strings.Replace(externalComposeYAML, "seed: 42", "seed: 42, learning_rate: 0.01", 1),
		"unknown stage":     strings.Replace(externalComposeYAML, "    trainer:", "    parameters: {}\n    trainer:", 1),
		"determinism":       strings.Replace(externalComposeYAML, "best-effort", "sometimes", 1),
		"no determinism":    strings.Replace(externalComposeYAML, "  determinism: best-effort\n", "", 1),
		"environment":       strings.Replace(externalComposeYAML, "environment: cpu", "environment: ''", 1),
		"bad budget":        strings.Replace(externalComposeYAML, "tokens: 128", "tokens: -1", 1),
		"zero batch":        strings.Replace(externalComposeYAML, "global_batch_sequences: 2", "global_batch_sequences: 0", 1),
		"duplicate corpus":  strings.Replace(externalComposeYAML, "[books]", "[books, books]", 1),
		"trailing":          externalComposeYAML + "---\n{}",
		"null architecture": strings.Replace(externalComposeYAML, "{model_type: qwen3, width: 64}", "null", 1),
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := loadExternalFixture(t, data); err == nil {
				t.Fatal("accepted invalid compose")
			}
		})
	}
}
func TestExternalComposeProbeSchemas(t *testing.T) {
	doc, err := loadExternalFixture(t, externalComposeYAML)
	if err != nil {
		t.Fatal(err)
	}
	compile := func(raw string, architecture bool) *application.ParameterSchema {
		t.Helper()
		s, err := application.CompileSchema([]byte(raw), architecture)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	arch := compile(`{"type":"object","properties":{"model_type":{"const":"qwen3"},"width":{"type":"integer","minimum":1}},"required":["model_type","width"],"additionalProperties":false}`, true)
	params := compile(`{"type":"object","properties":{"learning_rate":{"type":"number","exclusiveMinimum":0},"micro_batch":{"type":"integer","minimum":1,"x-waldo-role":"execution"}},"additionalProperties":false}`, false)
	tok := compile(`{"type":"object","additionalProperties":false}`, false)
	if err := doc.External.ValidateSchemas(arch, params, tok); err != nil {
		t.Fatal(err)
	}
	doc.External.Stages[0].Trainer["micro_batch"] = 2
	if err := doc.External.ValidateSchemas(arch, params, tok); err != nil {
		t.Fatal(err)
	}
	recipe, execution, err := params.Split(doc.External.Stages[0].Trainer)
	if err != nil {
		t.Fatal(err)
	}
	if len(recipe) != 1 || len(execution) != 1 {
		t.Fatal("execution override leaked into recipe")
	}
	doc.External.Stages[0].Trainer["learning_rate"] = -1
	if err := doc.External.ValidateSchemas(arch, params, tok); err == nil || !strings.Contains(err.Error(), "stage pretrain trainer") {
		t.Fatalf("missing contextual error: %v", err)
	}
}
func TestBuiltinSchemaTwoPreservesHashes(t *testing.T) {
	original := composeYAML("")
	first, err := loadExternalFixture(t, original)
	if err != nil {
		t.Fatal(err)
	}
	second, err := loadExternalFixture(t, strings.Replace(original, "schema: 1", "schema: 2", 1))
	if err != nil {
		t.Fatal(err)
	}
	want, err := canonicalHash(first.Builtin)
	if err != nil {
		t.Fatal(err)
	}
	got, err := canonicalHash(second.Builtin)
	if err != nil {
		t.Fatal(err)
	}
	if want != got || second.External != nil {
		t.Fatalf("native compose identity changed: %s != %s", got, want)
	}
	a, _ := canonicalHash(first.Builtin.Architecture)
	b, _ := canonicalHash(second.Builtin.Architecture)
	if a != b {
		t.Fatal("native architecture changed")
	}
	// Freeze the pre-existing schema-1 canonical representation, not just an
	// equality between two paths that could regress together.
	const baseline = "6147a29f748417391baa28842af2a744187199baee90b515c7074e543252944a"
	if want != baseline {
		t.Fatalf("schema-1 hash = %s", want)
	}
}

func TestNativePlanGoldenHash(t *testing.T) {
	doc, err := loadExternalFixture(t, composeYAML(""))
	if err != nil {
		t.Fatal(err)
	}
	architecture := doc.Builtin.Architecture
	ah, err := hashJSON(architecture)
	if err != nil {
		t.Fatal(err)
	}
	forecast, err := architecture.Forecast()
	if err != nil {
		t.Fatal(err)
	}
	plan := Plan{Kind: "waldo-model-plan", Schema: 1, Name: "native", Architecture: architecture, ArchitectureSHA256: ah, Forecast: forecast}
	got, err := hashJSON(plan)
	if err != nil {
		t.Fatal(err)
	}
	const baseline = "85f83122fa673a5b8309b5ff2ef0692a07362d1fcccde118d09db66af285ed68"
	if got != baseline {
		t.Fatalf("native plan hash = %s", got)
	}
}
