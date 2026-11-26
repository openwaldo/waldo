// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSelectedPathsResolveFromDisk(t *testing.T) {
	paths := []string{"code/permissive", "post-train/sft/python-basics-v4"}
	for declared, want := range map[string]string{
		"post-train/sft/python-basics-v4":                      "post-train/sft/python-basics-v4",
		"/home/me/index-local/post-train/sft/python-basics-v4": "post-train/sft/python-basics-v4",
		"~/index-local/post-train/sft/python-basics-v4/":       "post-train/sft/python-basics-v4",
		"./code/permissive":                                    "code/permissive",
	} {
		if got, err := resolveSelectedPath(declared, paths); err != nil || got != want {
			t.Fatalf("%s: %s, %v", declared, got, err)
		}
	}
	// A path on disk takes the longest logical path it ends with.
	nested := []string{"sft/python-basics-v4", "post-train/sft/python-basics-v4"}
	for declared, want := range map[string]string{"/idx/post-train/sft/python-basics-v4": "post-train/sft/python-basics-v4", "/idx/sft/python-basics-v4": "sft/python-basics-v4"} {
		if got, err := resolveSelectedPath(declared, nested); err != nil || got != want {
			t.Fatalf("%s: %s, %v", declared, got, err)
		}
	}
	// A ./ path resolves from the current directory, as the index resolves it.
	inside := filepath.Join(t.TempDir(), "post-train")
	if err := os.Mkdir(inside, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(inside)
	if got, err := resolveSelectedPath("./sft/python-basics-v4", nested); err != nil || got != "post-train/sft/python-basics-v4" {
		t.Fatalf("./sft/python-basics-v4 from post-train: %s, %v", got, err)
	}
	// Weights given by a path on disk land on the corpus it resolves to.
	weights, err := resolveCorpusWeights(map[string]uint64{"/home/me/index-local/code/permissive": 3, "post-train/sft/python-basics-v4": 1}, paths)
	if err != nil || weights["code/permissive"] != 3 || weights["post-train/sft/python-basics-v4"] != 1 {
		t.Fatalf("weights = %v, %v", weights, err)
	}
	// Only a path on disk matches by suffix; a logical path must match exactly.
	if _, err := resolveSelectedPath("sft/python-basics-v4", paths); err == nil {
		t.Fatal("a partial logical path resolved")
	}
}
