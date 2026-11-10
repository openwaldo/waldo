// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import "testing"

func TestSelectedPathsResolveFromDisk(t *testing.T) {
	paths := []string{"code/permissive", "post-train/sft/python-basics-v4"}
	for declared, want := range map[string]string{
		"post-train/sft/python-basics-v4":                      "post-train/sft/python-basics-v4",
		"/home/me/index-local/post-train/sft/python-basics-v4": "post-train/sft/python-basics-v4",
		"~/index-local/post-train/sft/python-basics-v4/":       "post-train/sft/python-basics-v4",
	} {
		if got, err := resolveSelectedPath(declared, paths); err != nil || got != want {
			t.Fatalf("%s: %s, %v", declared, got, err)
		}
	}
	// Only a path on disk matches by suffix; a logical path must match exactly.
	if _, err := resolveSelectedPath("sft/python-basics-v4", paths); err == nil {
		t.Fatal("a partial logical path resolved")
	}
}
