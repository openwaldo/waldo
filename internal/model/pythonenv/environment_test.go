// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package pythonenv

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestEnvironmentHelperRoundTrip(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal("python3 with pip and venv is required:", err)
	}
	version, err := exec.Command(python, "-I", "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))").Output()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	t.Cleanup(func() {
		_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err == nil && info.IsDir() {
				return os.Chmod(path, 0755)
			}
			return err
		})
	})
	lock := filepath.Join(dir, "pylock.toml")
	if err := os.WriteFile(lock, []byte("lock-version = \"1.0\"\ncreated-by = \"waldo-test\"\npackages = []\n"), 0600); err != nil {
		t.Fatal(err)
	}
	request := Request{Name: "cpu", Python: python, PythonVersion: strings.TrimSpace(string(version)), Lock: lock, Cache: filepath.Join(dir, "cache")}
	result, err := Ensure(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Identity.Name != "cpu" || result.Identity.Kind != "python-lock" || len(result.Identity.LockSHA256) != 64 || len(result.Identity.InterpreterSHA256) != 64 || result.Run.Installer != "pip" {
		t.Fatalf("unexpected result: %+v", result)
	}
	verified, err := Verify(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Identity != verified.Identity || result.Run.CachePath != verified.Run.CachePath {
		t.Fatal("verification changed identity or cache")
	}
	if err := os.Chmod(filepath.Join(result.Run.CachePath, "bin"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(result.Run.CachePath, "bin", "extra"), []byte("extra"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Ensure(context.Background(), request); err == nil || !strings.Contains(err.Error(), "unattested extra file") {
		t.Fatalf("cache corruption accepted: %v", err)
	}
}

func TestEnvironmentRequiresExplicitInputs(t *testing.T) {
	if _, err := Ensure(context.Background(), Request{}); err == nil {
		t.Fatal("empty request accepted")
	}
}
