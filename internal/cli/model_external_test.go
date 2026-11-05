// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package cli

import (
	"bytes"
	"github.com/openwaldo/waldo/internal/config"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExternalComposeCLIDispatchBeforeNativeArchitecture(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "index.yaml"), []byte("kind: index\nschema: 1\n"), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Index: root}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../../examples/trainers/qwen3-hf/compose.tiny.yaml")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "compose.yaml")
	if err = os.WriteFile(path, raw, 0644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run([]string{"model", "train", "external", path}, &stdout, &stderr)
	if code != 1 || !strings.Contains(stderr.String(), "unavailable corpus paths") || strings.Contains(stderr.String(), "field trainer not found") {
		t.Fatalf("dispatch: %d %s", code, stderr.String())
	}
}
