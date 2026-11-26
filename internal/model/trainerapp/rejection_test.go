// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"bufio"
	"context"
	"errors"
	"io"
	"math"
	"os/exec"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

// Each rejection must fail for its own reason, not because a later check
// (missing complete, file verification) happens to catch the same output.
func TestTrainRejectionReasons(t *testing.T) {
	for mode, want := range map[string]string{
		"non-json":            "invalid character",
		"unknown-field":       `unknown field "surprise"`,
		"unknown-event-field": `unknown field "surprise"`,
		"trailing":            "after top-level value",
		"schema-1":            "missing required field protocol",
		"unknown":             `unknown trainer frame "surprise"`,
		"train-version":       "unsupported trainer protocol 2",
		"error-no-class":      "error frame requires protocol, error and error_class",
		"event-with-error":    "invalid event frame",
		"complete-with-error": "invalid complete frame",
		"unknown-event-kind":  `unknown event kind "bogus"`,
		"empty-log":           "invalid log event",
		"token-overflow":      "invalid event counters/loss",
		"nan-loss":            "invalid character",
		"checkpoint-step":     "invalid checkpoint event",
		"profile":             "complete missing fields or inconsistent counters/profile",
		"escape":              "invalid artifact declaration",
		"selection":           "selected checkpoint is not best held-out candidate",
		"after-complete":      "frame after complete",
		"duplicate-complete":  "frame after complete",
		"no-complete":         "trainer missing complete",
	} {
		t.Run(mode, func(t *testing.T) {
			ctx, req := setup(t, mode)
			c, err := Open(ctx, req, []string{"causal-language-modeling"})
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.Train(ctx, begin(t, c), evals(), source, nil)
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("want %q, got %v", want, err)
			}
		})
	}
}

func TestTrainOversizeFrame(t *testing.T) {
	ctx, req := setup(t, "oversize")
	c, err := Open(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = c.Train(ctx, begin(t, c), evals(), source, nil); !errors.Is(err, bufio.ErrTooLong) {
		t.Fatalf("oversize frame: %v", err)
	}
}

func TestTrainExitAfterComplete(t *testing.T) {
	ctx, req := setup(t, "exit-after-complete")
	c, err := Open(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Train(ctx, begin(t, c), evals(), source, nil)
	var exit *exec.ExitError
	if !errors.As(err, &exit) || exit.ExitCode() != 3 || !strings.Contains(err.Error(), "trainer-stderr-marker") {
		t.Fatalf("nonzero exit after complete must fail with status and stderr: %v", err)
	}
}

func TestCommandRejectionReasons(t *testing.T) {
	ctx, req := setup(t, "probe-error")
	if _, err := Open(ctx, req, nil); training.WorkerErrorClass(err) != "configuration" {
		t.Fatalf("probe error frame class lost: %v", err)
	}
	ctx, req = setup(t, "resolve-kind")
	c, err := Open(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = c.Resolve(ctx, map[string]any{}, map[string]any{}, data(), single()); err == nil || !strings.Contains(err.Error(), "resolve requires kind") {
		t.Fatalf("resolve kind: %v", err)
	}
}

func TestBeginRejectsSchemaInvalidParameters(t *testing.T) {
	ctx, req := setup(t, "ok")
	c, err := Open(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	b := begin(t, c)
	b.Parameters = map[string]any{"lr": "fast", "micro": 1}
	b.ParametersSHA256 = digest(b.Parameters)
	launches := len(c.Evidence())
	if _, err = c.Train(ctx, b, evals(), source, nil); err == nil || len(c.Evidence()) != launches {
		t.Fatalf("schema-invalid begin parameters reached the trainer: %v", err)
	}
}

func TestInvalidInputReasons(t *testing.T) {
	for mode, want := range map[string]string{
		"ordinal":     "sequence ordinal/length exceeds or differs from data plan",
		"length":      "sequence ordinal/length exceeds or differs from data plan",
		"vocabulary":  "token outside vocabulary",
		"consumption": "sequence consumption differs from supervised targets",
	} {
		t.Run(mode, func(t *testing.T) {
			ctx, req := setup(t, "ok")
			c, err := Open(ctx, req, nil)
			if err != nil {
				t.Fatal(err)
			}
			src := func(ctx context.Context, f func(training.PreparedSequence) error) error {
				for i := int64(0); i < 2; i++ {
					r := training.PreparedSequence{Ordinal: i, Tokens: []int{1, 2, 3}, LossMask: []bool{true, true}, Consumption: map[string]int64{"books": 2}}
					switch mode {
					case "ordinal":
						r.Ordinal = i + 1
					case "length":
						r.Tokens, r.LossMask = []int{1, 2, 3, 4}, []bool{true, true, true}
						r.Consumption["books"] = 3
					case "vocabulary":
						r.Tokens[2] = 10
					case "consumption":
						r.Consumption["books"] = 1
					}
					if err := f(r); err != nil {
						return err
					}
				}
				return nil
			}
			if _, err = c.Train(ctx, begin(t, c), evals(), src, nil); err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("want %q, got %v", want, err)
			}
		})
	}
}

func TestValidatorIsolatedChecks(t *testing.T) {
	b := Begin{Data: training.ExternalDataParameters{Steps: 3, SequenceLength: 2, GlobalBatchSequences: 2}, EvaluationSet: training.EvaluationSet{Records: 1, TokenTargets: 2, TextBytes: 2}}
	loss := 1.0
	// Within the token bound for step 1, so only step monotonicity rejects it.
	if err := validateEvent(Event{Kind: "progress", Step: 1, Tokens: 4, Loss: &loss}, b, 2, 4); err == nil {
		t.Fatal("accepted step regression")
	}
	// Internally consistent NLL arithmetic, but counts differ from the pins.
	for _, e := range []Evaluation{
		{Step: 1, Tokens: 4, NLLSum: 3, TargetTokens: 3, TextBytes: 2, Loss: 1, BitsPerByte: 3 / (2 * math.Ln2)},
		{Step: 1, Tokens: 4, NLLSum: 2, TargetTokens: 2, TextBytes: 3, Loss: 1, BitsPerByte: 2 / (3 * math.Ln2)},
	} {
		if err := validateEvaluation(e, b); err == nil {
			t.Fatalf("accepted unpinned evaluation counts: %+v", e)
		}
	}
	// A self-consistent early completion must not satisfy the step budget.
	b.EvaluationSet = training.EvaluationSet{}
	complete := Complete{Steps: 0, ArtifactProfile: "p", Environment: map[string]any{}, Consumption: []training.CorpusConsumption{}, Evaluations: []Evaluation{}, SelectedCheckpoint: &training.CheckpointSelection{}, Artifacts: []Artifact{{Artifact: training.Artifact{Path: "model.safetensors", SHA256: strings.Repeat("a", 64)}, Role: "weights", Format: "safetensors"}}}
	if err := validateComplete(complete, b, "p", 0, 0); err == nil {
		t.Fatal("accepted completion before the step budget")
	}
}

func TestOutputFrameLimit(t *testing.T) {
	if err := write(io.Discard, strings.Repeat("a", FrameLimit)); err == nil {
		t.Fatal("wrote oversize protocol frame")
	}
	if err := write(io.Discard, strings.Repeat("a", FrameLimit-4)); err != nil {
		t.Fatal(err)
	}
}
