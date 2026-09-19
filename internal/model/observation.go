// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/openwaldo/waldo/internal/training"
)

func validateBackendObservation(runDirectory string, planned PlannedStage, observation training.Observation) error {
	if observation.Steps < 0 || observation.Steps > planned.Parameters.Steps {
		return fmt.Errorf("reported steps %d are outside planned range 0..%d", observation.Steps, planned.Parameters.Steps)
	}
	if observation.ConsumedTokens < 0 || observation.ConsumedTokens > planned.PlannedTokens {
		return fmt.Errorf("reported tokens %d are outside planned range 0..%d", observation.ConsumedTokens, planned.PlannedTokens)
	}
	if len(observation.Artifacts) == 0 {
		return fmt.Errorf("reported no output artifacts")
	}
	if observation.FinalLoss != nil && (*observation.FinalLoss < 0 || math.IsNaN(*observation.FinalLoss) || math.IsInf(*observation.FinalLoss, 0)) {
		return fmt.Errorf("reported final loss is not finite and non-negative")
	}
	seen := make(map[string]bool)
	validateArtifact := func(artifact training.Artifact) error {
		if artifact.Path == "" || filepath.IsAbs(filepath.FromSlash(artifact.Path)) {
			return fmt.Errorf("artifact path %q must be relative", artifact.Path)
		}
		clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(artifact.Path)))
		if clean != artifact.Path || clean == "artifacts" || !strings.HasPrefix(clean, "artifacts/") {
			return fmt.Errorf("artifact path %q must be canonical beneath artifacts/", artifact.Path)
		}
		if seen[clean] {
			return fmt.Errorf("artifact path %q is duplicated", artifact.Path)
		}
		seen[clean] = true
		path := filepath.Join(runDirectory, filepath.FromSlash(clean))
		return VerifyArtifactFile(path, artifact)
	}
	for _, artifact := range observation.Artifacts {
		if err := validateArtifact(artifact); err != nil {
			return err
		}
	}
	previousStep := int64(0)
	for position, checkpoint := range observation.Checkpoints {
		if checkpoint.Step <= previousStep || checkpoint.Step > observation.Steps || checkpoint.Tokens < 0 || checkpoint.Tokens > observation.ConsumedTokens || len(checkpoint.Artifacts) == 0 {
			return fmt.Errorf("checkpoint %d has invalid step, token count, or artifacts", position+1)
		}
		previousStep = checkpoint.Step
		for _, artifact := range checkpoint.Artifacts {
			if err := validateArtifact(artifact); err != nil {
				return fmt.Errorf("checkpoint %d: %w", position+1, err)
			}
		}
	}
	previousStep = 0
	for position, evaluation := range observation.Evaluations {
		if evaluation.Step <= previousStep || evaluation.Step > observation.Steps || evaluation.Tokens < 0 || evaluation.Tokens > observation.ConsumedTokens || len(evaluation.Metrics) == 0 {
			return fmt.Errorf("evaluation %d has invalid step, token count, or metrics", position+1)
		}
		previousStep = evaluation.Step
		for name, value := range evaluation.Metrics {
			if name == "" || math.IsNaN(value) || math.IsInf(value, 0) {
				return fmt.Errorf("evaluation %d has invalid metric %q", position+1, name)
			}
		}
	}
	if selected := observation.SelectedCheckpoint; selected != nil {
		if selected.Step <= 0 || selected.Step > observation.Steps || selected.Tokens < 0 || selected.Tokens > observation.ConsumedTokens || selected.Metric != "heldout_loss" || selected.Value < 0 || math.IsNaN(selected.Value) || math.IsInf(selected.Value, 0) {
			return fmt.Errorf("selected checkpoint metadata is invalid")
		}
		checkpointFound, evaluationFound := false, false
		for _, checkpoint := range observation.Checkpoints {
			checkpointFound = checkpointFound || checkpoint.Step == selected.Step && checkpoint.Tokens == selected.Tokens
		}
		for _, evaluation := range observation.Evaluations {
			value, ok := evaluation.Metrics[selected.Metric]
			evaluationFound = evaluationFound || evaluation.Step == selected.Step && evaluation.Tokens == selected.Tokens && ok && value == selected.Value
		}
		if !checkpointFound || !evaluationFound {
			return fmt.Errorf("selected checkpoint does not match a persisted checkpoint and evaluation")
		}
	}
	return nil
}

func VerifyArtifactFile(path string, artifact training.Artifact) error {
	if err := CheckArtifactFile(path, artifact); err != nil {
		return err
	}
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("artifact %s: %w", artifact.Path, err)
	}
	hasher := sha256.New()
	bytes, copyErr := io.Copy(hasher, file)
	closeErr := file.Close()
	if copyErr != nil {
		return fmt.Errorf("artifact %s: %w", artifact.Path, copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("artifact %s: %w", artifact.Path, closeErr)
	}
	if bytes != artifact.Bytes {
		return fmt.Errorf("artifact %s size is %d, backend reported %d", artifact.Path, bytes, artifact.Bytes)
	}
	digest := hex.EncodeToString(hasher.Sum(nil))
	if digest != artifact.SHA256 {
		return fmt.Errorf("artifact %s SHA-256 is %s, backend reported %s", artifact.Path, digest, artifact.SHA256)
	}
	return nil
}

// CheckArtifactFile validates artifact identity metadata and the inexpensive
// filesystem facts needed during routine model inspection. Call
// VerifyArtifactFile at trust boundaries and immediately before consuming a
// checkpoint to verify its complete content digest.
func CheckArtifactFile(path string, artifact training.Artifact) error {
	digest, err := hex.DecodeString(artifact.SHA256)
	if err != nil || len(digest) != sha256.Size {
		return fmt.Errorf("artifact %s has invalid SHA-256 metadata", artifact.Path)
	}
	if artifact.Bytes < 0 {
		return fmt.Errorf("artifact %s has invalid byte size %d", artifact.Path, artifact.Bytes)
	}
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("artifact %s: %w", artifact.Path, err)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("artifact %s is not a regular file", artifact.Path)
	}
	if info.Size() != artifact.Bytes {
		return fmt.Errorf("artifact %s size is %d, backend reported %d", artifact.Path, info.Size(), artifact.Bytes)
	}
	return nil
}
