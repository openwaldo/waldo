// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/openwaldo/waldo/internal/training"
)

// A continuation inherits a verified run, not just a weights file. Fail closed
// before starting another run if its retained provider evidence is incomplete.
func verifyTransformersContinuationArtifacts(directory string, architecture Architecture, artifacts []training.Artifact) error {
	required := map[string]string{"artifacts/model.safetensors": "", "artifacts/config.json": "", "artifacts/tokenizer.json": ""}
	if pin := architecture.Tokenizer.HuggingFace; pin != nil {
		required["artifacts/waldo_tokenizer.json"] = ""
		for name, digest := range pin.Files {
			required["artifacts/"+name] = digest
		}
	}
	seen := make(map[string]bool)
	for _, artifact := range artifacts {
		clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(artifact.Path)))
		if clean != artifact.Path || !strings.HasPrefix(clean, "artifacts/") || seen[clean] {
			return fmt.Errorf("invalid or duplicate continuation artifact %q", artifact.Path)
		}
		seen[clean] = true
		if digest := required[clean]; digest != "" && digest != artifact.SHA256 {
			return fmt.Errorf("continuation tokenizer artifact %s differs from architecture pin", clean)
		}
		if err := VerifyArtifactFile(filepath.Join(directory, filepath.FromSlash(clean)), artifact); err != nil {
			return err
		}
	}
	for name := range required {
		if !seen[name] {
			return fmt.Errorf("missing continuation artifact %s", name)
		}
	}
	return nil
}
