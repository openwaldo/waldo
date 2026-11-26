// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package application

import (
	"strings"
	"testing"
)

func TestComposeParameterRoles(t *testing.T) {
	s, err := CompileSchema([]byte(`{"type":"object","additionalProperties":false,"properties":{"bf16":{"type":"boolean"},"learning_rate":{"type":"number"},"micro":{"type":"integer","x-waldo-role":"execution"}}}`), false)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"bf16", "learning_rate"} {
		v := any(true)
		if key == "learning_rate" {
			v = 0.1
		}
		if _, err := s.ComposeParameters(map[string]any{}, map[string]any{key: v}); err == nil || !strings.Contains(err.Error(), key) {
			t.Fatalf("identity misclassified: %v", err)
		}
	}
	if _, err := s.ComposeParameters(map[string]any{"micro": 4}, nil); err == nil || !strings.Contains(err.Error(), "execution parameter micro") {
		t.Fatalf("execution in recipe: %v", err)
	}
	if _, err := s.ComposeParameters(map[string]any{"bf16": true}, map[string]any{"micro": 4}); err != nil {
		t.Fatal(err)
	}
}

func TestWaldoOwnedExecutionDeclarationsRejected(t *testing.T) {
	for _, key := range []string{"global_batch_sequences", "seed", "steps", "sequence_length", "evaluation", "data"} {
		for _, schema := range []string{
			`{"properties":{"` + key + `":{"x-waldo-role":"execution"}}}`,
			`{"x-waldo-role":"execution","properties":{"` + key + `":{}}}`,
			`{"properties":{"` + key + `":{"$ref":"#/$defs/exec"}},"$defs":{"exec":{"x-waldo-role":"execution"}}}`,
		} {
			if _, err := CompileSchema([]byte(schema), false); err == nil || !strings.Contains(err.Error(), key) {
				t.Fatalf("%s: %v", schema, err)
			}
		}
	}
}

func TestComposeNestedParameterRoles(t *testing.T) {
	s, err := CompileSchema([]byte(`{"type":"object","properties":{"optimizer":{"type":"object","properties":{"rate":{"type":"number"},"micro":{"type":"integer","x-waldo-role":"execution"}}}}}`), false)
	if err != nil {
		t.Fatal(err)
	}
	recipe := map[string]any{"optimizer": map[string]any{"rate": 0.1}}
	execution := map[string]any{"optimizer": map[string]any{"micro": 4}}
	merged, err := s.ComposeParameters(recipe, execution)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Validate(merged); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ComposeParameters(merged, nil); err == nil || !strings.Contains(err.Error(), "optimizer.micro") {
		t.Fatalf("nested misclassification: %v", err)
	}
	if _, err := s.ComposeParameters(recipe, recipe); err == nil || !strings.Contains(err.Error(), "optimizer.rate") {
		t.Fatalf("nested duplicate: %v", err)
	}
}
