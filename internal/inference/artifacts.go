// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package inference

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

type Artifacts struct {
	Model                 string
	SourceType            string
	SourceID              string
	RunID                 string
	CheckpointStep        int64
	InterruptedCheckpoint bool
	Backend               training.Identity
	ContextTokens         int
	Weights               string
	Configuration         string
	Tokenizer             string
	cleanup               func()
}

func ResolveArtifacts(inspection model.Inspection) (Artifacts, error) {
	if inspection.BOM.CurrentRunID == "" && inspection.BOM.CurrentOriginSHA256 == "" {
		return Artifacts{}, fmt.Errorf("model %q has no complete non-simulated run with usable weights", inspection.Model.Name)
	}
	if inspection.BOM.CurrentRunID == "" {
		if inspection.BOM.Origin == nil || inspection.BOM.Origin.SHA256 != inspection.BOM.CurrentOriginSHA256 {
			return Artifacts{}, fmt.Errorf("model %q current origin is invalid", inspection.Model.Name)
		}
		result := Artifacts{Model: inspection.Model.Name, SourceType: "origin", SourceID: inspection.BOM.CurrentOriginSHA256, Backend: training.Identity{Name: "portable-safetensors", Revision: "huggingface-schema-1"}, ContextTokens: int(inspection.Model.Architecture.ContextTokens)}
		for _, artifact := range inspection.BOM.Origin.Artifacts {
			path, err := resolveModelPath(inspection.Path, artifact.Path)
			if err != nil {
				return Artifacts{}, err
			}
			if err := verifyArtifact(path, artifact); err != nil {
				return Artifacts{}, err
			}
			switch artifact.Role {
			case "weights":
				result.Weights = path
			case "configuration":
				result.Configuration = path
			case "tokenizer":
				result.Tokenizer = path
			}
		}
		if result.Weights == "" || result.Configuration == "" || result.Tokenizer == "" {
			return Artifacts{}, fmt.Errorf("model %q origin must provide weights, configuration, and tokenizer artifacts", inspection.Model.Name)
		}
		return result, nil
	}
	var selected *model.ModelBOMRun
	for index := range inspection.BOM.Runs {
		if inspection.BOM.Runs[index].ID == inspection.BOM.CurrentRunID {
			selected = &inspection.BOM.Runs[index]
			break
		}
	}
	if selected == nil || selected.State != model.RunComplete || selected.Simulated {
		return Artifacts{}, fmt.Errorf("model %q current run %q is not a complete real run", inspection.Model.Name, inspection.BOM.CurrentRunID)
	}
	result := Artifacts{
		Model: inspection.Model.Name, SourceType: "run", SourceID: selected.ID, RunID: selected.ID, Backend: selected.Backend,
		ContextTokens: int(inspection.Model.Architecture.ContextTokens),
	}
	for _, artifact := range selected.Artifacts {
		path, err := resolveModelPath(inspection.Path, artifact.Path)
		if err != nil {
			return Artifacts{}, err
		}
		if err := verifyArtifact(path, artifact); err != nil {
			return Artifacts{}, err
		}
		switch artifact.Role {
		case "weights":
			result.Weights = path
		case "configuration":
			result.Configuration = path
		case "tokenizer":
			result.Tokenizer = path
		}
	}
	if result.Weights == "" || result.Configuration == "" || result.Tokenizer == "" {
		return Artifacts{}, fmt.Errorf("model %q current run %q must provide weights, configuration, and tokenizer artifacts", inspection.Model.Name, selected.ID)
	}
	if err := verifyRunConfiguration(result.Configuration, inspection.Model.ArchitectureSHA256, inspection.Model.Architecture); err != nil {
		return Artifacts{}, fmt.Errorf("model %q current run %q: %w", inspection.Model.Name, selected.ID, err)
	}
	return result, nil
}

// resolveCheckpointArtifacts resolves one exact durable checkpoint without
// changing model lifecycle state. Inference-only sidecars are derived into a
// temporary directory from immutable model and run metadata.
func resolveCheckpointArtifacts(inspection model.Inspection, runID string, step int64) (Artifacts, error) {
	if runID == "" || step <= 0 {
		return Artifacts{}, fmt.Errorf("checkpoint inference requires an explicit run ID and positive checkpoint step")
	}
	position := -1
	for index := range inspection.BOM.Runs {
		if inspection.BOM.Runs[index].ID == runID {
			position = index
			break
		}
	}
	if position < 0 || position >= len(inspection.Runs) || position >= len(inspection.RunBOMs) {
		return Artifacts{}, fmt.Errorf("model %q has no run %q", inspection.Model.Name, runID)
	}
	bomRun, run, runBOM := inspection.BOM.Runs[position], inspection.Runs[position], inspection.RunBOMs[position]
	if bomRun.ID != run.ID || runBOM.ID != runID || runBOM.ModelID != inspection.Model.ID {
		return Artifacts{}, fmt.Errorf("checkpoint run %q does not belong to model %q", runID, inspection.Model.Name)
	}
	if bomRun.State != model.RunInterrupted || run.State != model.RunInterrupted {
		return Artifacts{}, fmt.Errorf("model %q run %q is not interrupted", inspection.Model.Name, runID)
	}
	if bomRun.Simulated || bomRun.Backend.Name == training.BackendFake || run.Observation != nil && run.Observation.Simulated {
		return Artifacts{}, fmt.Errorf("model %q run %q is simulated", inspection.Model.Name, runID)
	}
	if bomRun.Backend != runBOM.Execution.Backend || runBOM.ArchitectureSHA256 != inspection.Model.ArchitectureSHA256 {
		return Artifacts{}, fmt.Errorf("model %q run %q backend or architecture does not match its immutable run metadata", inspection.Model.Name, runID)
	}
	if run.Progress == nil {
		return Artifacts{}, fmt.Errorf("model %q run %q has no durable checkpoints", inspection.Model.Name, runID)
	}
	var checkpoint *training.Checkpoint
	for index := range run.Progress.Checkpoints {
		if run.Progress.Checkpoints[index].Step == step {
			checkpoint = &run.Progress.Checkpoints[index]
			break
		}
	}
	if checkpoint == nil {
		return Artifacts{}, fmt.Errorf("model %q run %q has no checkpoint at step %d", inspection.Model.Name, runID, step)
	}
	runDirectory := filepath.Dir(filepath.Join(inspection.Path, filepath.FromSlash(bomRun.RunBOM)))
	checkpointPrefix := fmt.Sprintf("artifacts/checkpoints/step-%08d/", step)
	paths := make(map[string]string, len(checkpoint.Artifacts))
	for _, artifact := range checkpoint.Artifacts {
		if !strings.HasPrefix(artifact.Path, checkpointPrefix) {
			return Artifacts{}, fmt.Errorf("model %q run %q checkpoint step %d artifact %q is outside its exact checkpoint directory", inspection.Model.Name, runID, step, artifact.Path)
		}
		path, err := resolveRunArtifactPath(runDirectory, artifact.Path)
		if err != nil {
			return Artifacts{}, fmt.Errorf("model %q run %q checkpoint step %d: %w", inspection.Model.Name, runID, step, err)
		}
		if err := model.VerifyArtifactFile(path, artifact); err != nil {
			return Artifacts{}, fmt.Errorf("model %q run %q checkpoint step %d: %w", inspection.Model.Name, runID, step, err)
		}
		name := filepath.Base(path)
		if _, exists := paths[name]; exists {
			return Artifacts{}, fmt.Errorf("model %q run %q checkpoint step %d duplicates artifact %s", inspection.Model.Name, runID, step, name)
		}
		paths[name] = path
	}
	required := []string{"model.safetensors", "state.json"}
	switch bomRun.Backend.Name {
	case training.BackendPyTorch, training.BackendTorchTitan:
		required = append(required, "runtime.pt")
	case training.BackendMLX:
		required = append(required, "optimizer.safetensors")
	default:
		return Artifacts{}, fmt.Errorf("model %q run %q checkpoint backend %s@%s does not support inference", inspection.Model.Name, runID, bomRun.Backend.Name, bomRun.Backend.Revision)
	}
	for _, name := range required {
		if paths[name] == "" {
			return Artifacts{}, fmt.Errorf("model %q run %q checkpoint step %d is missing %s", inspection.Model.Name, runID, step, name)
		}
	}
	if err := verifyCheckpointState(paths["state.json"], inspection, runBOM, bomRun.Backend, *checkpoint); err != nil {
		return Artifacts{}, fmt.Errorf("model %q run %q checkpoint step %d: %w", inspection.Model.Name, runID, step, err)
	}
	if err := model.ValidateCheckpointWeights(paths["model.safetensors"], inspection.Model.Architecture); err != nil {
		return Artifacts{}, fmt.Errorf("model %q run %q checkpoint step %d weights: %w", inspection.Model.Name, runID, step, err)
	}
	temporary, configuration, tokenizer, err := materializeCheckpointSidecars(inspection, runBOM, bomRun.Backend)
	if err != nil {
		return Artifacts{}, fmt.Errorf("model %q run %q checkpoint step %d inference sidecars: %w", inspection.Model.Name, runID, step, err)
	}
	return Artifacts{
		Model: inspection.Model.Name, SourceType: "checkpoint", SourceID: runID, RunID: runID,
		CheckpointStep: step, InterruptedCheckpoint: true, Backend: bomRun.Backend,
		ContextTokens: int(inspection.Model.Architecture.ContextTokens), Weights: paths["model.safetensors"],
		Configuration: configuration, Tokenizer: tokenizer, cleanup: func() { _ = os.RemoveAll(temporary) },
	}, nil
}

func resolveRunArtifactPath(runDirectory, logical string) (string, error) {
	if logical == "" || filepath.IsAbs(filepath.FromSlash(logical)) {
		return "", fmt.Errorf("checkpoint artifact path %q is not run-relative", logical)
	}
	clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(logical)))
	if clean != logical || clean == "." || strings.HasPrefix(clean, "../") || !strings.HasPrefix(clean, "artifacts/checkpoints/") {
		return "", fmt.Errorf("checkpoint artifact path %q escapes its run", logical)
	}
	return filepath.Join(runDirectory, filepath.FromSlash(clean)), nil
}

func verifyCheckpointState(path string, inspection model.Inspection, runBOM model.RunBOM, backend training.Identity, checkpoint training.Checkpoint) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read checkpoint state: %w", err)
	}
	var state struct {
		Kind               string `json:"kind"`
		Schema             int    `json:"schema"`
		Backend            string `json:"backend"`
		BackendRevision    string `json:"backend_revision"`
		ModelID            string `json:"model_id"`
		RunID              string `json:"run_id"`
		ArchitectureSHA256 string `json:"architecture_sha256"`
		Step               int64  `json:"step"`
		ConsumedTokens     int64  `json:"consumed_tokens"`
	}
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("decode checkpoint state: %w", err)
	}
	if state.Kind != "waldo-training-checkpoint" || state.Schema != 1 || state.Backend != backend.Name || state.BackendRevision != backend.Revision || state.RunID != runBOM.ID || state.ArchitectureSHA256 != inspection.Model.ArchitectureSHA256 || state.ArchitectureSHA256 != runBOM.ArchitectureSHA256 || state.Step != checkpoint.Step || state.ConsumedTokens != checkpoint.Tokens {
		return fmt.Errorf("checkpoint state does not match the model, run, step, architecture, backend revision, and token position")
	}
	// Historical schema-1 checkpoints predate model_id. Their verified Run BOM
	// provides the immutable model binding. Once present, model_id must match it.
	if state.ModelID != "" && (state.ModelID != inspection.Model.ID || state.ModelID != runBOM.ModelID) {
		return fmt.Errorf("checkpoint state does not match the model, run, step, architecture, backend revision, and token position")
	}
	return nil
}

func materializeCheckpointSidecars(inspection model.Inspection, runBOM model.RunBOM, backend training.Identity) (string, string, string, error) {
	spec, _, err := inspection.Model.Architecture.ResolveTokenizer()
	if err != nil {
		return "", "", "", fmt.Errorf("resolve immutable model tokenizer: %w", err)
	}
	kind := map[string]string{
		training.BackendPyTorch: "waldo-pytorch-model-config", training.BackendTorchTitan: "waldo-torchtitan-model-config", training.BackendMLX: "waldo-mlx-model-config",
	}[backend.Name]
	if kind == "" {
		return "", "", "", fmt.Errorf("unsupported checkpoint backend %s@%s", backend.Name, backend.Revision)
	}
	directory, err := os.MkdirTemp("", "waldo-checkpoint-inference-")
	if err != nil {
		return "", "", "", err
	}
	cleanup := func(err error) (string, string, string, error) {
		_ = os.RemoveAll(directory)
		return "", "", "", err
	}
	configuration := filepath.Join(directory, "config.json")
	tokenizer := filepath.Join(directory, "tokenizer.json")
	configurationData, err := json.Marshal(struct {
		Kind               string                      `json:"kind"`
		Schema             int                         `json:"schema"`
		ArchitectureSHA256 string                      `json:"architecture_sha256"`
		Architecture       model.Architecture          `json:"architecture"`
		TrainingProfile    training.ResolvedParameters `json:"training_profile"`
		Initialization     *training.Initialization    `json:"initialization,omitempty"`
		Backend            training.Identity           `json:"backend"`
	}{kind, 1, inspection.Model.ArchitectureSHA256, inspection.Model.Architecture, runBOM.Parameters, runBOM.Initialization, backend})
	if err != nil {
		return cleanup(err)
	}
	tokenizerData, err := json.Marshal(struct {
		Kind   string `json:"kind"`
		Schema int    `json:"schema"`
		training.TokenizerSpec
	}{"waldo-tokenizer", 1, spec})
	if err != nil {
		return cleanup(err)
	}
	if err := os.WriteFile(configuration, configurationData, 0o600); err != nil {
		return cleanup(err)
	}
	if err := os.WriteFile(tokenizer, tokenizerData, 0o600); err != nil {
		return cleanup(err)
	}
	if err := verifyRunConfiguration(configuration, inspection.Model.ArchitectureSHA256, inspection.Model.Architecture); err != nil {
		return cleanup(err)
	}
	return directory, configuration, tokenizer, nil
}

func verifyRunConfiguration(path, architectureSHA256 string, architecture model.Architecture) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read runtime configuration: %w", err)
	}
	var configuration struct {
		Kind               string             `json:"kind"`
		Schema             int                `json:"schema"`
		ArchitectureSHA256 string             `json:"architecture_sha256"`
		Architecture       model.Architecture `json:"architecture"`
	}
	if err := json.Unmarshal(data, &configuration); err != nil {
		return fmt.Errorf("decode runtime configuration: %w", err)
	}
	if configuration.Schema != 1 || (configuration.Kind != "waldo-mlx-model-config" && configuration.Kind != "waldo-pytorch-model-config" && configuration.Kind != "waldo-torchtitan-model-config") {
		return fmt.Errorf("unsupported runtime configuration %q schema %d", configuration.Kind, configuration.Schema)
	}
	if configuration.ArchitectureSHA256 != architectureSHA256 || !reflect.DeepEqual(configuration.Architecture, architecture) {
		return fmt.Errorf("runtime configuration architecture does not match the immutable model architecture")
	}
	return nil
}

func resolveModelPath(root, logical string) (string, error) {
	if logical == "" || filepath.IsAbs(filepath.FromSlash(logical)) {
		return "", fmt.Errorf("model artifact path %q is not model-root-relative", logical)
	}
	clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(logical)))
	if clean != logical || clean == "." || strings.HasPrefix(clean, "../") {
		return "", fmt.Errorf("model artifact path %q escapes the model root", logical)
	}
	return filepath.Join(root, filepath.FromSlash(clean)), nil
}

func verifyArtifact(path string, artifact model.ModelBOMArtifact) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s artifact %s: %w", artifact.Role, artifact.Path, err)
	}
	hasher := sha256.New()
	bytes, copyErr := io.Copy(hasher, file)
	closeErr := file.Close()
	if copyErr != nil {
		return fmt.Errorf("hash %s artifact %s: %w", artifact.Role, artifact.Path, copyErr)
	}
	if closeErr != nil {
		return closeErr
	}
	if bytes != artifact.Bytes {
		return fmt.Errorf("%s artifact %s has size %d, BOM requires %d", artifact.Role, artifact.Path, bytes, artifact.Bytes)
	}
	digest := hex.EncodeToString(hasher.Sum(nil))
	if digest != artifact.SHA256 {
		return fmt.Errorf("%s artifact %s has SHA-256 %s, BOM requires %s", artifact.Role, artifact.Path, digest, artifact.SHA256)
	}
	return nil
}
