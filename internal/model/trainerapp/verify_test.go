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
	"strings"
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
	for _, mode := range []string{"missing-candidate", "token-mismatch", "missing-artifact-eval", "artifact-loss", "artifact-metrics", "omitted-best", "corrupt-candidate"} {
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
			case "artifact-metrics":
				// Same loss as the selection, but inconsistent with its own NLL sum.
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

// Published files are an allowlist by role, declared format and content. Each
// case keeps valid weights so a rejection cannot come from "missing weights".
func TestPublishedAllowlistWithValidWeights(t *testing.T) {
	for _, mode := range []string{"ok", "json-pickle", "no-weights", "json-weights", "pickle-labelled-json", "backslash"} {
		t.Run(mode, func(t *testing.T) {
			dir, a := publishedFixture(t)
			name, role, format, content := "tokenizer.json", "tokenizer", "json", []byte(`{"vocab":{}}`)
			switch mode {
			case "json-pickle":
				content = []byte("cos\nsystem\n(S'echo forbidden'\ntR.")
			case "no-weights":
				a = nil
			case "json-weights":
				a, role = nil, "weights"
			case "pickle-labelled-json":
				format = "pickle"
			case "backslash":
				name = `sub\tokenizer.json`
			}
			if err := os.WriteFile(filepath.Join(dir, name), content, 0600); err != nil {
				t.Fatal(err)
			}
			a = append(a, Artifact{Artifact: training.Artifact{Path: name, SHA256: application.Digest(content), Bytes: int64(len(content))}, Role: role, Format: format})
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
		})
	}
}

// Counts must equal the pins even when the metrics are self-consistent, and the
// loss/bits-per-byte consistency check must be tight.
func TestEvaluationPinsAndMetricTolerance(t *testing.T) {
	b := Begin{Data: data(), EvaluationSet: training.EvaluationSet{Records: 1, TokenTargets: 2, TextBytes: 2}}
	valid := Evaluation{Step: 1, Tokens: 4, NLLSum: 2, TargetTokens: 2, TextBytes: 2, Loss: 1, BitsPerByte: 1 / math.Ln2}
	if err := validateEvaluation(valid, b); err != nil {
		t.Fatal(err)
	}
	for name, e := range map[string]Evaluation{
		"subset-targets": {Step: 1, Tokens: 4, NLLSum: 4, TargetTokens: 4, TextBytes: 2, Loss: 1, BitsPerByte: 4 / (2 * math.Ln2)},
		"subset-bytes":   {Step: 1, Tokens: 4, NLLSum: 2, TargetTokens: 2, TextBytes: 4, Loss: 1, BitsPerByte: 2 / (4 * math.Ln2)},
		"loss-drift":     {Step: 1, Tokens: 4, NLLSum: 2, TargetTokens: 2, TextBytes: 2, Loss: 1.0001, BitsPerByte: 1 / math.Ln2},
		"bits-drift":     {Step: 1, Tokens: 4, NLLSum: 2, TargetTokens: 2, TextBytes: 2, Loss: 1, BitsPerByte: 1.0001 / math.Ln2},
	} {
		if err := validateEvaluation(e, b); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

func TestEventCounterAndCheckpointConsistency(t *testing.T) {
	b := Begin{Data: data()}
	_, artifacts := publishedFixture(t)
	checkpoint := func(step, tokens int64) *Checkpoint {
		return &Checkpoint{Step: step, Tokens: tokens, Artifacts: artifacts}
	}
	if err := validateEvent(Event{Kind: "checkpoint", Step: 1, Tokens: 4, Checkpoint: checkpoint(1, 4)}, b, 1, 4); err != nil {
		t.Fatal(err)
	}
	for name, e := range map[string]Event{
		// data(): one step of 2 sequences x 2 targets bounds tokens at 4.
		"tokens-exceed-plan":       {Kind: "progress", Step: 1, Tokens: 5, Loss: new(float64)},
		"checkpoint-step-back":     {Kind: "checkpoint", Step: 0, Tokens: 0, Checkpoint: checkpoint(0, 0)},
		"checkpoint-tokens-back":   {Kind: "checkpoint", Step: 1, Tokens: 3, Checkpoint: checkpoint(1, 3)},
		"checkpoint-tokens-differ": {Kind: "checkpoint", Step: 1, Tokens: 4, Checkpoint: checkpoint(1, 3)},
	} {
		if err := validateEvent(e, b, 1, 4); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

func TestArtifactDeclarations(t *testing.T) {
	ok := Artifact{Artifact: training.Artifact{Path: "dir/model.safetensors", SHA256: digest("x"), Bytes: 1}, Role: "weights", Format: "safetensors"}
	if err := validateArtifacts([]Artifact{ok}); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*Artifact){
		"unclean":   func(a *Artifact) { a.Path = "dir/../model.safetensors" },
		"dot":       func(a *Artifact) { a.Path = "./model.safetensors" },
		"absolute":  func(a *Artifact) { a.Path = "/model.safetensors" },
		"parent":    func(a *Artifact) { a.Path = "../model.safetensors" },
		"uppercase": func(a *Artifact) { a.SHA256 = strings.ToUpper(a.SHA256) },
		"short":     func(a *Artifact) { a.SHA256 = a.SHA256[:63] },
		"negative":  func(a *Artifact) { a.Bytes = -1 },
		"role":      func(a *Artifact) { a.Role = "" },
	} {
		a := ok
		change(&a)
		if err := validateArtifacts([]Artifact{a}); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	if err := validateArtifacts([]Artifact{ok, ok}); err == nil {
		t.Fatal("duplicate path accepted")
	}
}

func TestCompleteSelectionValueMustMatchBest(t *testing.T) {
	b := Begin{Data: data(), EvaluationSet: training.EvaluationSet{Records: 1, TokenTargets: 2, TextBytes: 2}}
	_, artifacts := publishedFixture(t)
	e := Evaluation{Step: 1, Tokens: 4, NLLSum: 2, TargetTokens: 2, TextBytes: 2, Loss: 1, BitsPerByte: 1 / math.Ln2}
	c := Complete{Steps: 1, ConsumedTokens: 4, Artifacts: artifacts, ArtifactProfile: "hf-causal-lm", Consumption: []training.CorpusConsumption{{Corpus: "books", TokenTargets: 4}}, Evaluations: []Evaluation{e}, SelectedCheckpoint: &training.CheckpointSelection{Step: 1, Tokens: 4, Metric: "heldout_loss", Value: 1}, Environment: map[string]any{}}
	if err := validateComplete(c, b, "hf-causal-lm", 1, 4); err != nil {
		t.Fatal(err)
	}
	c.SelectedCheckpoint.Value = 0.5
	if err := validateComplete(c, b, "hf-causal-lm", 1, 4); err == nil {
		t.Fatal("selection value differing from evaluated loss accepted")
	}
}
