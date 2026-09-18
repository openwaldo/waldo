// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

func TestValidateBackendObservationVerifiesBoundsAndArtifacts(t *testing.T) {
	runDirectory := t.TempDir()
	artifactPath := filepath.Join(runDirectory, "artifacts", "weights.safetensors")
	if err := os.MkdirAll(filepath.Dir(artifactPath), 0o755); err != nil {
		t.Fatal(err)
	}
	data := []byte("weights")
	if err := os.WriteFile(artifactPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	checkpointPath := filepath.Join(runDirectory, "artifacts", "checkpoints", "step-2.safetensors")
	if err := os.MkdirAll(filepath.Dir(checkpointPath), 0o755); err != nil {
		t.Fatal(err)
	}
	checkpointData := []byte("checkpoint")
	if err := os.WriteFile(checkpointPath, checkpointData, 0o644); err != nil {
		t.Fatal(err)
	}
	checkpointDigest := sha256.Sum256(checkpointData)
	loss := 1.25
	planned := PlannedStage{PlannedTokens: 128, Parameters: training.Parameters{Steps: 2}}
	valid := training.Observation{
		Steps: 2, ConsumedTokens: 128, FinalLoss: &loss,
		Checkpoints: []training.Checkpoint{{Step: 2, Tokens: 128, Artifacts: []training.Artifact{{Path: "artifacts/checkpoints/step-2.safetensors", SHA256: hex.EncodeToString(checkpointDigest[:]), Bytes: int64(len(checkpointData))}}}},
		Evaluations: []training.Evaluation{{Step: 2, Tokens: 128, Metrics: map[string]float64{"loss": loss}}},
		Artifacts:   []training.Artifact{{Path: "artifacts/weights.safetensors", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}},
	}
	if err := validateBackendObservation(runDirectory, planned, valid); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		mutate func(*training.Observation)
		want   string
	}{
		{name: "steps", mutate: func(value *training.Observation) { value.Steps = 3 }, want: "reported steps"},
		{name: "tokens", mutate: func(value *training.Observation) { value.ConsumedTokens = 129 }, want: "reported tokens"},
		{name: "path", mutate: func(value *training.Observation) { value.Artifacts[0].Path = "../weights" }, want: "canonical beneath"},
		{name: "size", mutate: func(value *training.Observation) { value.Artifacts[0].Bytes++ }, want: "size is"},
		{name: "hash", mutate: func(value *training.Observation) { value.Artifacts[0].SHA256 = strings.Repeat("0", 64) }, want: "SHA-256"},
		{name: "checkpoint", mutate: func(value *training.Observation) { value.Checkpoints[0].Step = 3 }, want: "checkpoint"},
		{name: "evaluation", mutate: func(value *training.Observation) { value.Evaluations[0].Metrics["loss"] = math.Inf(1) }, want: "metric"},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := valid
			candidate.Artifacts = append([]training.Artifact(nil), valid.Artifacts...)
			candidate.Checkpoints = append([]training.Checkpoint(nil), valid.Checkpoints...)
			candidate.Checkpoints[0].Artifacts = append([]training.Artifact(nil), valid.Checkpoints[0].Artifacts...)
			candidate.Evaluations = append([]training.Evaluation(nil), valid.Evaluations...)
			candidate.Evaluations[0].Metrics = map[string]float64{"loss": loss}
			test.mutate(&candidate)
			if err := validateBackendObservation(runDirectory, planned, candidate); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestCheckArtifactFileDefersContentHashing(t *testing.T) {
	path := filepath.Join(t.TempDir(), "runtime.pt")
	data := []byte("checkpoint-state")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	artifact := training.Artifact{Path: "artifacts/checkpoints/step/runtime.pt", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}
	if err := CheckArtifactFile(path, artifact); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("tampered-state!!"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := CheckArtifactFile(path, artifact); err != nil {
		t.Fatalf("structural check unexpectedly hashed content: %v", err)
	}
	if err := VerifyArtifactFile(path, artifact); err == nil || !strings.Contains(err.Error(), "SHA-256") {
		t.Fatalf("full verification did not reject tampered content: %v", err)
	}
	artifact.SHA256 = "invalid"
	if err := CheckArtifactFile(path, artifact); err == nil || !strings.Contains(err.Error(), "invalid SHA-256 metadata") {
		t.Fatalf("invalid digest metadata error = %v", err)
	}
}
