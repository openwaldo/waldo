// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFetchPinnedFetchesOnlyThePinnedRelease(t *testing.T) {
	release := t.TempDir()
	files := map[string]string{}
	for name, content := range map[string]string{"model.safetensors": "weights", "BOM.json": "{}"} {
		if err := os.WriteFile(filepath.Join(release, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256([]byte(content))
		files[name] = hex.EncodeToString(digest[:])
	}
	target := filepath.Join(t.TempDir(), "fetched")
	if err := (Puller{}).FetchPinned(context.Background(), "file://"+release, files, target); err != nil {
		t.Fatal(err)
	}
	if data, err := os.ReadFile(filepath.Join(target, "model.safetensors")); err != nil || string(data) != "weights" {
		t.Fatalf("fetched %q, %v", data, err)
	}
	// A file that changed since it was pinned refuses the whole release.
	tampered := map[string]string{"model.safetensors": files["model.safetensors"], "BOM.json": strings.Repeat("0", 64)}
	other := filepath.Join(t.TempDir(), "other")
	if err := (Puller{}).FetchPinned(context.Background(), "file://"+release, tampered, other); err == nil || !strings.Contains(err.Error(), "pinned SHA-256") {
		t.Fatalf("tampered error = %v", err)
	}
	if _, err := os.Stat(other); !os.IsNotExist(err) {
		t.Fatal("a refused release left files behind")
	}
}
