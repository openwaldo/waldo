// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/lookaside"
)

func TestLookasideCacheStatusAndClean(t *testing.T) {
	cacheRoot, scratchRoot, modelRoot := t.TempDir(), t.TempDir(), t.TempDir()
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{
		Lookaside: config.Lookaside{Cache: cacheRoot, Scratch: scratchRoot, CacheMaxBytes: 1 << 20, RetainCompleted: true},
		Model:     config.Model{Root: modelRoot},
	}); err != nil {
		t.Fatal(err)
	}
	cache, err := lookaside.NewCache(cacheRoot, nil)
	if err != nil {
		t.Fatal(err)
	}
	content := []byte("disposable cached object")
	sum := sha256.Sum256(content)
	digest := hex.EncodeToString(sum[:])
	path, err := cache.Path(digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if code := Run([]string{"lookaside", "cache", "status"}, &stdout, &stderr); code != 0 {
		t.Fatalf("status code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), cacheRoot) || !strings.Contains(stdout.String(), "objects        1") || !strings.Contains(stdout.String(), "retain completed true") {
		t.Fatalf("status output = %q", stdout.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"lookaside", "cache", "clean"}, &stdout, &stderr); code != 0 {
		t.Fatalf("clean code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "removed 1 objects") {
		t.Fatalf("clean output = %q", stdout.String())
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("cache object remains: %v", err)
	}
}
