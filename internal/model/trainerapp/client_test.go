// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/model/pythonenv"
	"github.com/openwaldo/waldo/internal/training"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func fixture(t *testing.T, code string) applaunch.Request {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	source := filepath.Join(root, "source")
	env := filepath.Join(root, "env")
	scratch := filepath.Join(root, "scratch")
	for _, dir := range []string{source, filepath.Join(env, "bin"), scratch} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(python, filepath.Join(env, "bin", "python")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "app.py"), []byte(code), 0444); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "pylock.toml"), []byte("fixture lock"), 0444); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(source, 0555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(source, 0755) })
	return applaunch.Request{Source: application.ResolvedSource{Directory: source, Manifest: application.Manifest{Kind: "waldo-trainer", Schema: 1, Protocol: 1, Name: "test", Entrypoint: application.Entrypoint{Module: "app"}, Commands: application.Commands{Probe: "probe", Resolve: "resolve", Train: "train"}, LaunchProfiles: []string{"single-process"}, Objectives: []string{"causal-language-modeling"}, ArtifactProfile: "hf-causal-lm", Environments: map[string]application.Environment{"cpu": {Kind: "python-lock", Lock: "pylock.toml", Python: "3.11.1"}}}}, Environment: pythonenv.Result{Identity: pythonenv.Identity{Name: "cpu", Kind: "python-lock", InterpreterVersion: "3.11.1", LockSHA256: application.Digest([]byte("fixture lock"))}, Run: pythonenv.RunEvidence{CachePath: env}}, Command: "probe", Scratch: scratch}
}

func setup(t *testing.T, mode string) (context.Context, applaunch.Request) {
	t.Helper()
	code, err := os.ReadFile("testdata/trainer.py")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	return ctx, fixture(t, strings.Replace(string(code), "MODE = 'ok'", fmt.Sprintf("MODE = %q", mode), 1))
}
func data() training.ExternalDataParameters {
	return training.ExternalDataParameters{Steps: 1, SequenceLength: 2, GlobalBatchSequences: 2}
}
func single() World {
	return World{World: applaunch.World{Nodes: 1, ProcessesPerNode: 1, WorldSize: 1}}
}
func begin(t *testing.T, c *Client) Begin {
	t.Helper()
	a := map[string]any{}
	r, err := c.Resolve(context.Background(), a, map[string]any{}, data(), single())
	if err != nil {
		t.Fatal(err)
	}
	return Begin{CheckpointIdentity: CheckpointIdentity{RunBOMSHA256: digest("run"), TrainerSHA256: digest("trainer"), TokenizerSHA256: digest("tokenizer"), EnvironmentSHA256: digest("environment")}, CheckpointDirectory: filepath.Join(c.request.Scratch, "durable"), Paths: Paths{Artifacts: c.request.Scratch, Checkpoints: c.request.Scratch, Scratch: c.request.Scratch}, ModelID: "m", RunID: "r", Stage: "s", Objective: "causal-language-modeling", Data: data(), Architecture: a, ArchitectureSHA256: digest(a), Parameters: r.Parameters, ParametersSHA256: r.ParametersSHA256, Batch: r.Batch, Tokenizer: training.TokenizerSpec{VocabularySize: 10}, EvaluationSet: training.EvaluationSet{Records: 1, TokenTargets: 2, TextBytes: 2}, Determinism: "best-effort", World: single().World}
}
func source(ctx context.Context, consume func(training.PreparedSequence) error) error {
	for i := int64(0); i < 2; i++ {
		if err := consume(training.PreparedSequence{Ordinal: i, Tokens: []int{1, 2, 3}, LossMask: []bool{true, true}, Consumption: map[string]int64{"books": 2}}); err != nil {
			return err
		}
	}
	return nil
}
func evals() []EvaluationRecord {
	return []EvaluationRecord{{ID: "heldout", Tokens: []int{1, 2, 3}, LossMask: []bool{true, true}, TextBytes: 2}}
}
func TestProtocol(t *testing.T) {
	ctx, req := setup(t, "ok")
	c, err := Open(ctx, req, []string{"causal-language-modeling"})
	if err != nil {
		t.Fatal(err)
	}
	r, err := c.Resolve(ctx, map[string]any{}, map[string]any{}, data(), World{World: applaunch.World{Nodes: 1, ProcessesPerNode: 2, WorldSize: 2}})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Recipe["micro"]; ok || fmt.Sprint(r.Execution["micro"]) != "1" {
		t.Fatalf("role split: %+v", r)
	}
	b := begin(t, c)
	kinds := []string{}
	result, err := c.Train(ctx, b, evals(), source, func(e Event) error { kinds = append(kinds, e.Telemetry().Kind); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(kinds, ",") != "log,progress,checkpoint,evaluation" || result.ConsumedTokens != 4 || len(c.Evidence()) != 6 {
		t.Fatalf("result/evidence: %+v %v %d", result, kinds, len(c.Evidence()))
	}
}
func TestProbeFailures(t *testing.T) {
	for _, mode := range []string{"version", "schemas", "role", "capability", "objective", "missing", "extra"} {
		t.Run(mode, func(t *testing.T) {
			ctx, req := setup(t, mode)
			if _, err := Open(ctx, req, []string{"causal-language-modeling"}); err == nil {
				t.Fatal("accepted invalid probe")
			}
		})
	}
}
func TestResolveFailures(t *testing.T) {
	for _, mode := range []string{"batch", "world-recipe", "invalid-resolved"} {
		t.Run(mode, func(t *testing.T) {
			ctx, req := setup(t, mode)
			c, err := Open(ctx, req, []string{"causal-language-modeling"})
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.Resolve(ctx, map[string]any{}, map[string]any{}, data(), World{World: applaunch.World{Nodes: 1, ProcessesPerNode: 2, WorldSize: 2}})
			if err == nil {
				t.Fatal("accepted invalid resolve")
			}
		})
	}
}
func TestTrainFailures(t *testing.T) {
	for _, mode := range []string{"crash", "error", "unknown", "train-version", "nonmonotonic", "bad-metrics", "missing-nll", "null-nll", "missing-complete", "consumption", "escape", "selection", "after-complete"} {
		t.Run(mode, func(t *testing.T) {
			ctx, req := setup(t, mode)
			c, err := Open(ctx, req, []string{"causal-language-modeling"})
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.Train(ctx, begin(t, c), evals(), source, nil)
			if err == nil {
				t.Fatal("accepted invalid train")
			}
			if mode == "error" && training.WorkerErrorClass(err) != "numerical-integrity" {
				t.Fatal(err)
			}
		})
	}
}
func TestInputFailureAndCancellation(t *testing.T) {
	ctx, req := setup(t, "ok")
	c, err := Open(ctx, req, []string{"causal-language-modeling"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Train(ctx, begin(t, c), evals(), func(ctx context.Context, f func(training.PreparedSequence) error) error {
		return fmt.Errorf("source failed")
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "source failed") {
		t.Fatal("source error lost", err)
	}
	ctx, req = setup(t, "hang")
	c, err = Open(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	b := begin(t, c)
	ctx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err = c.Train(ctx, b, evals(), source, nil)
	if err == nil || time.Since(start) > 3*time.Second {
		t.Fatalf("cancellation: %v", err)
	}
}
func TestExactSchemas(t *testing.T) {
	ctx, req := setup(t, "ok")
	c, err := Open(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw := json.RawMessage(`{ "type" : "object", "x-unknown": true }`)
	doc, err := c.SchemaDocument(raw)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Architecture json.RawMessage
		Parameters   json.RawMessage
		Tokenizer    json.RawMessage
	}
	if err = json.Unmarshal(doc, &got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, got.Tokenizer) || !bytes.Equal(c.Schemas().Parameters, got.Parameters) || !bytes.Equal(c.Schemas().Architecture, got.Architecture) {
		t.Fatal("schema bytes changed")
	}
}

func TestResumeFinalBoundary(t *testing.T) {
	ctx, req := setup(t, "ok")
	c, err := Open(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	b := begin(t, c)
	var ref CheckpointReference
	_, err = c.Train(ctx, b, evals(), source, func(e Event) error {
		if e.DurableCheckpoint != nil {
			ref = *e.DurableCheckpoint
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	b.Resume, err = LoadCheckpoint(b, ref)
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Train(ctx, b, evals(), func(context.Context, func(training.PreparedSequence) error) error { return nil }, nil)
	if err != nil {
		t.Fatal(err)
	}
}
func TestInvalidInput(t *testing.T) {
	for _, mode := range []string{"hash", "world", "short", "mask", "ordinal", "budget", "evaluation", "sink"} {
		t.Run(mode, func(t *testing.T) {
			ctx, req := setup(t, "ok")
			c, err := Open(ctx, req, nil)
			if err != nil {
				t.Fatal(err)
			}
			b := begin(t, c)
			ev := evals()
			src := SequenceSource(source)
			var sink func(Event) error
			switch mode {
			case "hash":
				b.ParametersSHA256 = "bad"
			case "world":
				b.World.WorldSize = 2
			case "evaluation":
				ev[0].TextBytes = 3
			case "sink":
				sink = func(Event) error { return fmt.Errorf("sink failed") }
			default:
				src = func(ctx context.Context, f func(training.PreparedSequence) error) error {
					if mode == "short" {
						return nil
					}
					r := training.PreparedSequence{Ordinal: 0, Tokens: []int{1, 2, 3}, LossMask: []bool{true, true}, Consumption: map[string]int64{"books": 2}}
					if mode == "mask" {
						r.LossMask = nil
					}
					if mode == "ordinal" {
						r.Ordinal = 1
					}
					if mode == "budget" {
						for i := int64(0); i < 3; i++ {
							r.Ordinal = i
							if err := f(r); err != nil {
								return err
							}
						}
						return nil
					}
					return f(r)
				}
			}
			if _, err = c.Train(ctx, b, ev, src, sink); err == nil {
				t.Fatal("invalid input accepted")
			}
		})
	}
}
