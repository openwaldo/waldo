// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/model"
)

// A model a later WALDO wrote, with a field this one doesn't know, is skipped
// with a warning; the rest of the store still lists.
func TestModelListSkipsModelsItCannotRead(t *testing.T) {
	models := filepath.Join(t.TempDir(), "models")
	architecture := model.Architecture{Family: "decoder-transformer", ContextTokens: 128, VocabularySize: 256, HiddenSize: 64, IntermediateSize: 192, Layers: 2, AttentionHeads: 4, KeyValueHeads: 2, TieEmbeddings: true, ParameterDType: "float32", Tokenizer: model.Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"}}
	for _, name := range []string{"current", "future"} {
		if _, err := (&model.Builder{Root: models}).Initialize(name, architecture); err != nil {
			t.Fatal(err)
		}
	}
	// A later WALDO adds a field to the plan and hashes the plan with it; this
	// one drops the field it doesn't know, so the plan's hash no longer matches.
	future := filepath.Join(models, "future")
	plan := rewriteJSON(t, filepath.Join(future, "PLAN.json"), func(fields map[string]any) {
		fields["field_from_a_later_waldo"] = true
	})
	digest := sha256.Sum256(plan)
	rewriteJSON(t, filepath.Join(future, "MODEL.json"), func(fields map[string]any) {
		fields["plan_sha256"] = hex.EncodeToString(digest[:])
		fields["id"] = hex.EncodeToString(digest[:])
	})
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Model: config.Model{Root: models}}); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"model", "list"}, {"--json", "model", "list"}} {
		var stdout, stderr bytes.Buffer
		if code := Run(args, &stdout, &stderr); code != 0 {
			t.Fatalf("%v code = %d, stderr = %q", args, code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "current") || strings.Contains(stdout.String(), "future") {
			t.Fatalf("%v stdout = %q", args, stdout.String())
		}
		if !strings.Contains(stderr.String(), "warning: skipped model future") {
			t.Fatalf("%v stderr = %q", args, stderr.String())
		}
	}
	if _, err := model.List(models, nil); err == nil {
		t.Fatal("List read past a model it cannot read; cache cleanup relies on it failing")
	}
}

func rewriteJSON(t *testing.T, path string, change func(map[string]any)) []byte {
	t.Helper()
	var fields map[string]any
	data, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(data, &fields) != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	change(fields)
	if data, err = json.Marshal(fields); err != nil || os.WriteFile(path, data, 0o644) != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return data
}
