// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

func TestTransformersContinuationArtifacts(t *testing.T) {
	for _, composePath := range []string{"../../docs/examples/transformers-qwen3-smoke.yaml", hfTokenizerCompose} {
		t.Run(composePath, func(t *testing.T) {
			compose, _, err := LoadCompose(composePath)
			if err != nil {
				t.Fatal(err)
			}
			directory := t.TempDir()
			if err := os.Mkdir(filepath.Join(directory, "artifacts"), 0700); err != nil {
				t.Fatal(err)
			}
			names := []string{"model.safetensors", "config.json", "tokenizer.json"}
			if compose.Architecture.Tokenizer.HuggingFace != nil {
				names = append(names, "waldo_tokenizer.json", "tokenizer_config.json")
			}
			var artifacts []training.Artifact
			for _, name := range names {
				data := []byte(name)
				hash := sha256.Sum256(data)
				artifact := training.Artifact{Path: "artifacts/" + name, Bytes: int64(len(data)), SHA256: hex.EncodeToString(hash[:])}
				if err := os.WriteFile(filepath.Join(directory, artifact.Path), data, 0600); err != nil {
					t.Fatal(err)
				}
				if pin := compose.Architecture.Tokenizer.HuggingFace; pin != nil && pin.Files[name] != "" {
					pin.Files[name] = artifact.SHA256
				}
				artifacts = append(artifacts, artifact)
			}
			verify := func(items []training.Artifact) error {
				return verifyTransformersContinuationArtifacts(directory, compose.Architecture, items)
			}
			if err := verify(artifacts); err != nil {
				t.Fatal(err)
			}
			if err := verify(append(append([]training.Artifact{}, artifacts...), artifacts[0])); err == nil {
				t.Fatal("accepted duplicate")
			}
			for i, artifact := range artifacts {
				missing := append([]training.Artifact{}, artifacts[:i]...)
				missing = append(missing, artifacts[i+1:]...)
				if err := verify(missing); err == nil {
					t.Fatalf("accepted missing %s", artifact.Path)
				}
				path := filepath.Join(directory, artifact.Path)
				original, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				corrupt := append([]byte{}, original...)
				corrupt[0] ^= 1 // Same size: exercise digest verification, not just length.
				if err := os.WriteFile(path, corrupt, 0600); err != nil {
					t.Fatal(err)
				}
				if err := verify(artifacts); err == nil {
					t.Fatalf("accepted corrupt %s", artifact.Path)
				}
				if err := os.WriteFile(path, original, 0600); err != nil {
					t.Fatal(err)
				}
			}
			if pin := compose.Architecture.Tokenizer.HuggingFace; pin != nil {
				pin.Files["tokenizer.json"] = "wrong"
				if err := verify(artifacts); err == nil {
					t.Fatal("accepted changed tokenizer pin")
				}
			}
		})
	}
}
