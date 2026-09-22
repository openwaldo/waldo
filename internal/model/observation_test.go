// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
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
	checkpointData := []byte("checkpoint-2")
	if err := os.WriteFile(checkpointPath, checkpointData, 0o644); err != nil {
		t.Fatal(err)
	}
	checkpointDigest := sha256.Sum256(checkpointData)
	firstCheckpointPath := filepath.Join(runDirectory, "artifacts", "checkpoints", "step-1.safetensors")
	firstCheckpointData := []byte("checkpoint-1")
	if err := os.WriteFile(firstCheckpointPath, firstCheckpointData, 0o644); err != nil {
		t.Fatal(err)
	}
	firstCheckpointDigest := sha256.Sum256(firstCheckpointData)
	loss := 1.25
	planned := PlannedStage{PlannedTokens: 128, Parameters: training.Parameters{Steps: 2}}
	valid := training.Observation{
		Steps: 2, ConsumedTokens: 128, FinalLoss: &loss,
		Checkpoints: []training.Checkpoint{
			{Step: 1, Tokens: 64, Artifacts: []training.Artifact{{Path: "artifacts/checkpoints/step-1.safetensors", SHA256: hex.EncodeToString(firstCheckpointDigest[:]), Bytes: int64(len(firstCheckpointData))}}},
			{Step: 2, Tokens: 128, Artifacts: []training.Artifact{{Path: "artifacts/checkpoints/step-2.safetensors", SHA256: hex.EncodeToString(checkpointDigest[:]), Bytes: int64(len(checkpointData))}}},
		},
		Evaluations: []training.Evaluation{
			{Step: 1, Tokens: 64, Metrics: map[string]float64{"heldout_loss": 1.5}},
			{Step: 2, Tokens: 128, Metrics: map[string]float64{"heldout_loss": loss}},
		},
		SelectedCheckpoint: &training.CheckpointSelection{Step: 2, Tokens: 128, Metric: "heldout_loss", Value: loss},
		Artifacts:          []training.Artifact{{Path: "artifacts/weights.safetensors", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}},
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
		{name: "checkpoint", mutate: func(value *training.Observation) { value.Checkpoints[1].Step = 3 }, want: "checkpoint"},
		{name: "evaluation", mutate: func(value *training.Observation) { value.Evaluations[1].Metrics["heldout_loss"] = math.Inf(1) }, want: "metric"},
		{name: "selection", mutate: func(value *training.Observation) { value.SelectedCheckpoint.Step = 1 }, want: "selected checkpoint"},
		{name: "not-best", mutate: func(value *training.Observation) { value.Evaluations[0].Metrics["heldout_loss"] = 1.0 }, want: "not the best"},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := valid
			candidate.Artifacts = append([]training.Artifact(nil), valid.Artifacts...)
			candidate.Checkpoints = append([]training.Checkpoint(nil), valid.Checkpoints...)
			for index := range candidate.Checkpoints {
				candidate.Checkpoints[index].Artifacts = append([]training.Artifact(nil), valid.Checkpoints[index].Artifacts...)
			}
			candidate.Evaluations = append([]training.Evaluation(nil), valid.Evaluations...)
			for index := range candidate.Evaluations {
				candidate.Evaluations[index].Metrics = cloneMetrics(valid.Evaluations[index].Metrics)
			}
			selected := *valid.SelectedCheckpoint
			candidate.SelectedCheckpoint = &selected
			test.mutate(&candidate)
			if err := validateBackendObservation(runDirectory, planned, candidate); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestRealBackendsRequireArtifactVerification(t *testing.T) {
	for _, name := range []string{training.BackendPyTorch, training.BackendTorchTitan, training.BackendMLX} {
		if !backendRequiresArtifactVerification(name) {
			t.Fatalf("real backend %q does not require persisted artifact verification", name)
		}
	}
	if backendRequiresArtifactVerification(training.BackendFake) {
		t.Fatal("fake backend unexpectedly requires real artifact verification")
	}
}

func TestArtifactIntegrityFailureIsNotResumable(t *testing.T) {
	progress := &training.Progress{Checkpoints: []training.Checkpoint{{Step: 1, Tokens: 64}}}
	if !interruptedTrainingError(errors.New("transport failed"), progress) {
		t.Fatal("ordinary failure after a checkpoint should remain resumable")
	}
	integrity := &training.WorkerError{Message: "published weights degraded", Class: training.WorkerErrorArtifactIntegrity}
	if interruptedTrainingError(fmt.Errorf("worker: %w", integrity), progress) {
		t.Fatal("deterministic artifact-integrity failure must fail instead of resuming")
	}
	run := RunRecord{State: RunFailed, Progress: progress, FailureClass: training.WorkerErrorArtifactIntegrity}
	if resumableRunState(run, training.ResolvedParameters{Steps: 2, PlannedTokenCapacity: 128}) {
		t.Fatal("persisted artifact-integrity failure unexpectedly became resumable")
	}
	numerical := &training.WorkerError{Message: "loss became non-finite", Class: training.WorkerErrorNumericalIntegrity}
	if interruptedTrainingError(fmt.Errorf("worker: %w", numerical), progress) {
		t.Fatal("deterministic numerical-integrity failure must fail instead of resuming")
	}
	run.FailureClass = training.WorkerErrorNumericalIntegrity
	if resumableRunState(run, training.ResolvedParameters{Steps: 2, PlannedTokenCapacity: 128}) {
		t.Fatal("persisted numerical-integrity failure unexpectedly became resumable")
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
