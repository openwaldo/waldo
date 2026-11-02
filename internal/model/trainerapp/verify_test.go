// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"encoding/binary"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/training"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func publishedFixture(t *testing.T) (string, []Artifact) {
	t.Helper()
	dir := t.TempDir()
	header := []byte(`{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`)
	raw := make([]byte, 8)
	binary.LittleEndian.PutUint64(raw, uint64(len(header)))
	raw = append(raw, header...)
	raw = append(raw, 0, 0, 128, 63)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	return dir, []Artifact{{Artifact: training.Artifact{Path: "model.safetensors", SHA256: application.Digest(raw), Bytes: int64(len(raw))}, Role: "weights", Format: "safetensors"}}
}
func TestPublishedVerification(t *testing.T) {
	for _, mode := range []string{"ok", "hash", "size", "escape", "symlink", "directory", "pickle-format", "pickle-bytes", "json-pickle", "unknown-role", "nan", "truncated"} {
		t.Run(mode, func(t *testing.T) {
			dir, a := publishedFixture(t)
			path := filepath.Join(dir, a[0].Path)
			switch mode {
			case "hash":
				a[0].SHA256 = digest("wrong")
			case "size":
				a[0].Bytes++
			case "escape":
				a[0].Path = "../model.safetensors"
			case "symlink":
				outside := filepath.Join(t.TempDir(), "weights")
				raw, _ := os.ReadFile(path)
				os.WriteFile(outside, raw, 0600)
				os.Remove(path)
				os.Symlink(outside, path)
			case "directory":
				os.Remove(path)
				os.Mkdir(path, 0700)
			case "pickle-format":
				a[0].Format = "pickle"
			case "unknown-role":
				a[0].Role = "resume-state"
			case "pickle-bytes", "json-pickle", "nan", "truncated":
				raw, _ := os.ReadFile(path)
				switch mode {
				case "pickle-bytes", "json-pickle":
					raw = []byte("cos\nsystem\n(S'echo forbidden'\ntR.")
				case "nan":
					binary.LittleEndian.PutUint32(raw[len(raw)-4:], 0x7fc00000)
				case "truncated":
					raw = raw[:len(raw)-1]
				}
				os.WriteFile(path, raw, 0600)
				a[0].Bytes = int64(len(raw))
				a[0].SHA256 = application.Digest(raw)
				if mode == "json-pickle" {
					a[0].Role = "tokenizer"
					a[0].Format = "json"
				}
			}
			err := VerifyPublished(dir, a)
			if mode == "ok" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if !training.IsNonRetryableWorkerError(err) {
				t.Fatalf("expected nonretryable integrity error: %v", err)
			}
			if mode == "nan" && training.WorkerErrorClass(err) != training.WorkerErrorNumericalIntegrity {
				t.Fatal(err)
			}
		})
	}
}
func TestCompletionEvidenceFailures(t *testing.T) {
	for _, mode := range []string{"missing-candidate", "token-mismatch", "missing-artifact-eval", "artifact-loss", "omitted-best", "corrupt-candidate"} {
		t.Run(mode, func(t *testing.T) {
			ctx, req := setup(t, "ok")
			c, err := Open(ctx, req, nil)
			if err != nil {
				t.Fatal(err)
			}
			b := begin(t, c)
			candidates := map[int64]CheckpointReference{}
			complete, err := c.Train(ctx, b, evals(), source, func(e Event) error {
				if e.DurableCheckpoint != nil {
					candidates[e.Step] = *e.DurableCheckpoint
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			var observed []Evaluation
			switch mode {
			case "missing-candidate":
				delete(candidates, 1)
			case "token-mismatch":
				complete.Evaluations[0].Tokens--
			case "missing-artifact-eval":
				complete.ArtifactEvaluation = nil
			case "artifact-loss":
				complete.ArtifactEvaluation.Loss = 2
				complete.ArtifactEvaluation.NLLSum = 4
				complete.ArtifactEvaluation.BitsPerByte *= 2
			case "omitted-best":
				e := complete.Evaluations[0]
				e.Loss = 0.5
				e.NLLSum = 1
				e.BitsPerByte /= 2
				observed = []Evaluation{e}
			case "corrupt-candidate":
				r := candidates[1]
				r.SHA256 = digest("wrong")
				candidates[1] = r
			}
			if err := verifyCompletionFiles(complete, b, candidates, observed); !training.IsNonRetryableWorkerError(err) {
				t.Fatal("accepted invalid evidence", err)
			}
		})
	}
}

func TestEvaluationAndCounterFailures(t *testing.T) {
	b := Begin{Data: data(), EvaluationSet: training.EvaluationSet{Records: 1, TokenTargets: 2, TextBytes: 2}}
	for _, mode := range []string{"targets", "bytes", "nll", "loss", "bits", "nan", "negative", "future"} {
		t.Run(mode, func(t *testing.T) {
			e := Evaluation{Step: 1, Tokens: 4, NLLSum: 2, TargetTokens: 2, TextBytes: 2, Loss: 1, BitsPerByte: 1 / math.Ln2}
			switch mode {
			case "targets":
				e.TargetTokens = 3
			case "bytes":
				e.TextBytes = 3
			case "nll":
				e.NLLSum = 3
			case "loss":
				e.Loss = 2
			case "bits":
				e.BitsPerByte = 3
			case "nan":
				e.Loss = math.NaN()
			case "negative":
				e.NLLSum = -1
			case "future":
				e.Step = 2
			}
			if err := validateEvaluation(e, b); err == nil {
				t.Fatal("accepted invalid evaluation")
			}
		})
	}
	for _, e := range []Event{{Kind: "progress", Step: 0, Tokens: 4, Loss: new(float64)}, {Kind: "progress", Step: 1, Tokens: 3, Loss: new(float64)}, {Kind: "progress", Step: 2, Tokens: 8, Loss: new(float64)}} {
		if err := validateEvent(e, b, 1, 4); err == nil {
			t.Fatal("accepted invalid counters")
		}
	}
}
