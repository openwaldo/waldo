// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/config"
)

// A schema-2 PLAN.json marks an external model. Every model command that has an
// external branch must take it (reaching the external plan reader, which needs
// TRAINER-SCHEMAS.json, or the stored external COMPOSE.json) instead of the
// native inspection path, and export must refuse conversion formats up front.
func TestExternalModelCommandsDispatchToExternalPath(t *testing.T) {
	models := filepath.Join(t.TempDir(), "models")
	if err := os.MkdirAll(filepath.Join(models, "ext"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(models, "ext", "PLAN.json"), []byte(`{"kind":"waldo-model-plan","schema":2}`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Model: config.Model{Root: models}}); err != nil {
		t.Fatal(err)
	}
	export := filepath.Join(t.TempDir(), "export")
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"model", "summary", "ext"}, "TRAINER-SCHEMAS.json"},
		{[]string{"model", "bom", "ext"}, "TRAINER-SCHEMAS.json"},
		{[]string{"model", "bom", "ext", "--format", "eu-gpai"}, "TRAINER-SCHEMAS.json"},
		{[]string{"model", "export", "ext", export}, "TRAINER-SCHEMAS.json"},
		{[]string{"model", "export", "ext", export, "--format", "gguf"}, "external hf-causal-lm export supports waldo or huggingface format"},
		{[]string{"model", "export", "ext", export, "--format", "mlx"}, "external hf-causal-lm export supports waldo or huggingface format"},
		{[]string{"model", "continue", "ext"}, "COMPOSE.json"},
	} {
		var stdout, stderr bytes.Buffer
		if code := Run(tc.args, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), tc.want) {
			t.Fatalf("%v: code %d, stderr %q, want %q", tc.args, code, stderr.String(), tc.want)
		}
	}
	if _, err := os.Stat(export); !os.IsNotExist(err) {
		t.Fatalf("refused export created output: %v", err)
	}
}
