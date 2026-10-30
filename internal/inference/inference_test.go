// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package inference

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/pytorchruntime"
	waldotokenizer "github.com/openwaldo/waldo/internal/tokenizer"
	"github.com/openwaldo/waldo/internal/training"
)

func TestLoadTokenizerUsesEmbeddedTrainedArtifact(t *testing.T) {
	artifact, err := waldotokenizer.TrainByteBPE(
		[]waldotokenizer.Sample{{ID: "one", Text: "portable tokenizer sample"}},
		300,
		1024,
		strings.Repeat("a", 64),
	)
	if err != nil {
		t.Fatal(err)
	}
	payload := struct {
		Kind   string `json:"kind"`
		Schema int    `json:"schema"`
		training.TokenizerSpec
	}{
		Kind: "waldo-tokenizer", Schema: 1,
		TokenizerSpec: training.TokenizerSpec{
			Name: artifact.Name, Revision: artifact.Revision,
			VocabularySize: artifact.VocabularySize,
			PadID:          artifact.PadID, BOSID: artifact.BOSID, EOSID: artifact.EOSID,
			Artifact: &artifact,
		},
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, encoded, 0o644); err != nil {
		t.Fatal(err)
	}
	spec, codec, err := loadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	text := "portable tokenizer"
	if spec.Revision != artifact.Revision || codec.Decode(codec.Encode(text)) != text {
		t.Fatalf("spec = %+v, round trip = %q", spec, codec.Decode(codec.Encode(text)))
	}
}

func TestResolveArtifactsVerifiesCurrentRealRun(t *testing.T) {
	root := t.TempDir()
	architecture := model.Architecture{Family: "decoder-transformer", ContextTokens: 512}
	architectureSHA256 := strings.Repeat("a", 64)
	artifacts := make([]model.ModelBOMArtifact, 0, 3)
	for _, item := range []struct{ role, name string }{{"weights", "model.safetensors"}, {"configuration", "config.json"}, {"tokenizer", "tokenizer.json"}} {
		logical := "runs/0001-train-real/artifacts/" + item.name
		path := filepath.Join(root, filepath.FromSlash(logical))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		data := []byte(item.role)
		if item.role == "configuration" {
			data = []byte(`{"kind":"waldo-mlx-model-config","schema":1,"architecture_sha256":"` + architectureSHA256 + `","architecture":{"family":"decoder-transformer","context_tokens":512}}`)
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(data)
		artifacts = append(artifacts, model.ModelBOMArtifact{Role: item.role, Path: logical, SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))})
	}
	inspection := model.Inspection{Path: root, Model: model.ModelRecord{Name: "example", ArchitectureSHA256: architectureSHA256, Architecture: architecture}, BOM: model.ModelBOM{
		CurrentRunID: "real", Runs: []model.ModelBOMRun{{ID: "fake", State: model.RunComplete, Simulated: true}, {ID: "real", State: model.RunComplete, Backend: training.Identity{Name: "mlx", Revision: "v1"}, Artifacts: artifacts}},
	}}
	resolved, err := ResolveArtifacts(inspection)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.RunID != "real" || resolved.ContextTokens != 512 || !strings.HasSuffix(resolved.Weights, "model.safetensors") {
		t.Fatalf("resolved = %+v", resolved)
	}
	mismatched := inspection
	mismatched.Model.Architecture.ContextTokens = 1024
	if _, err := ResolveArtifacts(mismatched); err == nil || !strings.Contains(err.Error(), "immutable model architecture") {
		t.Fatalf("configuration mismatch error = %v", err)
	}
	if err := os.WriteFile(resolved.Weights, []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ResolveArtifacts(inspection); err == nil || !strings.Contains(err.Error(), "BOM requires") {
		t.Fatalf("corruption error = %v", err)
	}
	inspection.BOM.CurrentRunID = ""
	if _, err := ResolveArtifacts(inspection); err == nil || !strings.Contains(err.Error(), "no complete non-simulated") {
		t.Fatalf("simulation error = %v", err)
	}
}

func TestResolveCheckpointArtifactsRequiresExactVerifiedInterruptedCheckpoint(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		inspection, _ := checkpointInspectionFixture(t, checkpointFixtureOptions{})
		resolved, err := resolveCheckpointArtifacts(inspection, "run1", 10)
		if err != nil {
			t.Fatal(err)
		}
		defer resolved.cleanup()
		if resolved.Model != "example" || resolved.RunID != "run1" || resolved.CheckpointStep != 10 || !resolved.InterruptedCheckpoint || resolved.SourceType != "checkpoint" {
			t.Fatalf("resolved checkpoint = %+v", resolved)
		}
		if _, _, err := loadTokenizer(resolved.Tokenizer); err != nil {
			t.Fatalf("generated tokenizer sidecar: %v", err)
		}
		if err := verifyRunConfiguration(resolved.Configuration, inspection.Model.ArchitectureSHA256, inspection.Model.Architecture); err != nil {
			t.Fatalf("generated configuration sidecar: %v", err)
		}
		if inspection.BOM.CurrentRunID != "" || inspection.Runs[0].State != model.RunInterrupted {
			t.Fatalf("checkpoint selection changed lifecycle: %+v", inspection)
		}
	})

	for _, test := range []struct {
		name    string
		options checkpointFixtureOptions
		step    int64
		want    string
	}{
		{name: "missing step", step: 11, want: "no checkpoint at step 11"},
		{name: "simulated", options: checkpointFixtureOptions{simulated: true}, step: 10, want: "simulated"},
		{name: "state run mismatch", options: checkpointFixtureOptions{stateRunID: "other"}, step: 10, want: "checkpoint state does not match"},
		{name: "state model mismatch", options: checkpointFixtureOptions{stateModelID: "other-model"}, step: 10, want: "checkpoint state does not match"},
		{name: "state architecture mismatch", options: checkpointFixtureOptions{stateArchitectureSHA256: strings.Repeat("b", 64)}, step: 10, want: "checkpoint state does not match"},
		{name: "state backend mismatch", options: checkpointFixtureOptions{stateBackendRevision: "other-backend"}, step: 10, want: "checkpoint state does not match"},
		{name: "state step mismatch", options: checkpointFixtureOptions{stateStep: 9}, step: 10, want: "checkpoint state does not match"},
		{name: "weight architecture mismatch", options: checkpointFixtureOptions{wrongWeightShape: true}, step: 10, want: "model architecture requires"},
	} {
		t.Run(test.name, func(t *testing.T) {
			inspection, _ := checkpointInspectionFixture(t, test.options)
			if _, err := resolveCheckpointArtifacts(inspection, "run1", test.step); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("checkpoint error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestResolveCheckpointArtifactsRejectsTamperedArtifact(t *testing.T) {
	for _, name := range []string{"model.safetensors", "state.json"} {
		t.Run(name, func(t *testing.T) {
			inspection, paths := checkpointInspectionFixture(t, checkpointFixtureOptions{})
			data, err := os.ReadFile(paths[name])
			if err != nil {
				t.Fatal(err)
			}
			data[len(data)-1] ^= 0xff
			if err := os.WriteFile(paths[name], data, 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := resolveCheckpointArtifacts(inspection, "run1", 10); err == nil || !strings.Contains(err.Error(), "SHA-256") {
				t.Fatalf("tampered checkpoint error = %v", err)
			}
		})
	}
}

type checkpointFixtureOptions struct {
	simulated               bool
	stateRunID              string
	stateModelID            string
	stateArchitectureSHA256 string
	stateBackendRevision    string
	stateStep               int64
	wrongWeightShape        bool
}

func checkpointInspectionFixture(t *testing.T, options checkpointFixtureOptions) (model.Inspection, map[string]string) {
	t.Helper()
	root := t.TempDir()
	modelID := "model-id"
	runID := "run1"
	architectureSHA256 := strings.Repeat("a", 64)
	backend := training.Identity{Name: training.BackendPyTorch, Revision: "checkpoint-test-r1"}
	architecture := model.Architecture{
		Family: "decoder-transformer", ContextTokens: 16, VocabularySize: 259,
		HiddenSize: 4, IntermediateSize: 8, Layers: 0, AttentionHeads: 1, KeyValueHeads: 1,
		TieEmbeddings: true, ParameterDType: "float32",
		Tokenizer: model.Tokenizer{Name: "byte", Revision: training.ByteTokenizerRevision},
	}
	runDirectory := filepath.Join(root, "runs", "0001-pretrain-run1")
	checkpointDirectory := filepath.Join(runDirectory, "artifacts", "checkpoints", "step-00000010")
	if err := os.MkdirAll(checkpointDirectory, 0o755); err != nil {
		t.Fatal(err)
	}
	paths := map[string]string{
		"model.safetensors": filepath.Join(checkpointDirectory, "model.safetensors"),
		"runtime.pt":        filepath.Join(checkpointDirectory, "runtime.pt"),
		"state.json":        filepath.Join(checkpointDirectory, "state.json"),
	}
	embeddingRows := uint64(259)
	if options.wrongWeightShape {
		embeddingRows++
	}
	writeCheckpointSafetensors(t, paths["model.safetensors"], embeddingRows, architecture.HiddenSize)
	if err := os.WriteFile(paths["runtime.pt"], []byte("runtime"), 0o644); err != nil {
		t.Fatal(err)
	}
	stateRunID := options.stateRunID
	if stateRunID == "" {
		stateRunID = runID
	}
	stateModelID := options.stateModelID
	if stateModelID == "" {
		stateModelID = modelID
	}
	stateArchitecture := options.stateArchitectureSHA256
	if stateArchitecture == "" {
		stateArchitecture = architectureSHA256
	}
	stateBackendRevision := options.stateBackendRevision
	if stateBackendRevision == "" {
		stateBackendRevision = backend.Revision
	}
	stateStep := options.stateStep
	if stateStep == 0 {
		stateStep = 10
	}
	state, err := json.Marshal(map[string]any{
		"kind": "waldo-training-checkpoint", "schema": 1,
		"backend": backend.Name, "backend_revision": stateBackendRevision,
		"model_id": stateModelID, "run_id": stateRunID, "architecture_sha256": stateArchitecture,
		"step": stateStep, "consumed_tokens": int64(160),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths["state.json"], state, 0o644); err != nil {
		t.Fatal(err)
	}
	var artifacts []training.Artifact
	for _, name := range []string{"model.safetensors", "runtime.pt", "state.json"} {
		data, err := os.ReadFile(paths[name])
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(data)
		artifacts = append(artifacts, training.Artifact{
			Path:   filepath.ToSlash(filepath.Join("artifacts", "checkpoints", "step-00000010", name)),
			SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data)),
		})
	}
	checkpoint := training.Checkpoint{Step: 10, Tokens: 160, Artifacts: artifacts}
	return model.Inspection{
		Path:  root,
		Model: model.ModelRecord{ID: modelID, Name: "example", ArchitectureSHA256: architectureSHA256, Architecture: architecture},
		BOM: model.ModelBOM{Runs: []model.ModelBOMRun{{
			ID: runID, State: model.RunInterrupted, Backend: backend, Simulated: options.simulated,
			RunBOM: filepath.ToSlash(filepath.Join("runs", "0001-pretrain-run1", "RUN-BOM.json")),
		}}},
		Runs:    []model.RunRecord{{ID: runID, State: model.RunInterrupted, Progress: &training.Progress{Checkpoints: []training.Checkpoint{checkpoint}}}},
		RunBOMs: []model.RunBOM{{ID: runID, ModelID: modelID, ArchitectureSHA256: architectureSHA256, Execution: training.Execution{Backend: backend}}},
	}, paths
}

func writeCheckpointSafetensors(t *testing.T, path string, embeddingRows, hidden uint64) {
	t.Helper()
	embeddingBytes := embeddingRows * hidden * 4
	normBytes := hidden * 4
	header := map[string]any{
		"__metadata__":     map[string]string{"format": "openwaldo"},
		"embedding.weight": map[string]any{"dtype": "F32", "shape": []uint64{embeddingRows, hidden}, "data_offsets": []uint64{0, embeddingBytes}},
		"norm.weight":      map[string]any{"dtype": "F32", "shape": []uint64{hidden}, "data_offsets": []uint64{embeddingBytes, embeddingBytes + normBytes}},
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(encoded)%8 != 0 {
		encoded = append(encoded, ' ')
	}
	data := make([]byte, 8+len(encoded)+int(embeddingBytes+normBytes))
	binary.LittleEndian.PutUint64(data[:8], uint64(len(encoded)))
	copy(data[8:], encoded)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestMLXSessionConsumesStreamingProtocol(t *testing.T) {
	python := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
printf '%s\n' '{"kind":"ready","schema":1,"context_tokens":512}'
while IFS= read -r line; do
  printf '%s\n' '{"kind":"token","schema":1,"data":"QQ=="}'
  printf '%s\n' '{"kind":"complete","schema":1,"tokens":1,"finish_reason":"eos","duration_ms":4}'
done
`
	if err := os.WriteFile(python, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	session, contextTokens, err := startMLXSession(context.Background(), python, Artifacts{ContextTokens: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	var streamed string
	result, err := session.Generate(context.Background(), "hello", Options{MaxTokens: 1, Temperature: 0, TopP: 1}, func(token Token) error {
		streamed += string(token.Bytes)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if contextTokens != 512 || streamed != "A" || result.Text != "A" || result.Tokens != 1 || result.FinishReason != "eos" || result.DurationMS != 4 {
		t.Fatalf("context = %d, streamed = %q, result = %+v", contextTokens, streamed, result)
	}
}

func TestMLXInferenceDisablesComposeDropout(t *testing.T) {
	if !strings.Contains(string(mlxChatWorker), "self.model.eval()") {
		t.Fatal("MLX inference must disable training dropout")
	}
}

func TestPyTorchSessionConsumesStreamingProtocol(t *testing.T) {
	python := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
printf '%s\n' '{"kind":"ready","schema":1,"context_tokens":512}'
while IFS= read -r line; do
  printf '%s\n' '{"kind":"token","schema":1,"data":"Qg=="}'
  printf '%s\n' '{"kind":"complete","schema":1,"tokens":1,"finish_reason":"eos","duration_ms":5}'
done
`
	if err := os.WriteFile(python, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	session, contextTokens, err := startPyTorchSession(context.Background(), python, "cuda", Artifacts{ContextTokens: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	var streamed string
	result, err := session.Generate(context.Background(), "hello", Options{MaxTokens: 1, Temperature: 0, TopP: 1}, func(token Token) error {
		streamed += string(token.Bytes)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if contextTokens != 512 || streamed != "B" || result.Text != "B" || result.Tokens != 1 || result.FinishReason != "eos" || result.DurationMS != 5 {
		t.Fatalf("context = %d, streamed = %q, result = %+v", contextTokens, streamed, result)
	}
}

func TestPyTorchInferenceUsesSharedTrainingModel(t *testing.T) {
	source := pytorchruntime.WithModel(pyTorchChatWorker)
	for _, expected := range []string{
		`WALDO_SHARED_DECODER_LM = DecoderLM`,
		`DecoderLM = WALDO_SHARED_DECODER_LM`,
		`architecture.get("qk_normalization", False)`,
		`if self.qk_normalization:`,
		`self.model.eval()`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("PyTorch inference omits shared training-model behavior %q", expected)
		}
	}
}
