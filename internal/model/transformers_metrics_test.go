// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"encoding/csv"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

func transformersMetricsCompose(t *testing.T) Compose {
	t.Helper()
	compose, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
	return compose
}

func evaluationTelemetry(t *testing.T, path string) map[int64]map[string]string {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	rows, err := csv.NewReader(file).ReadAll()
	if err != nil || len(rows) == 0 {
		t.Fatalf("telemetry rows: %v", err)
	}
	result := map[int64]map[string]string{}
	for _, row := range rows[1:] {
		fields := map[string]string{}
		for index, name := range rows[0] {
			fields[name] = row[index]
		}
		if fields["event"] != "evaluation" {
			continue
		}
		step, err := strconv.ParseInt(fields["step"], 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		result[step] = fields
	}
	return result
}

// The Transformers worker reports each held-out evaluation as the standard
// evaluation event. Telemetry and the completed run record must carry the
// same exact intrinsic metrics, pinned to the run BOM's held-out evidence.
func TestTransformersEvaluationTelemetryMatchesRunRecord(t *testing.T) {
	root := t.TempDir()
	builder := transformersTestBuilder(root, func(_ context.Context, request training.Request) (training.Observation, error) {
		return transformersCandidateObservation(t, request), nil
	})
	compose := transformersMetricsCompose(t)
	inspection, err := builder.Compose(context.Background(), "metrics", compose, []PreparedStage{preparedFixture(t, compose.Stages[0])})
	if err != nil {
		t.Fatal(err)
	}
	set := inspection.RunBOMs[0].EvaluationSet
	observation := inspection.Runs[0].Observation
	if set == nil || set.TokenTargets <= 0 || set.TextBytes <= 0 || observation == nil || len(observation.Evaluations) != 2 {
		t.Fatalf("evaluation set %+v, observation %+v", set, observation)
	}
	telemetry := evaluationTelemetry(t, filepath.Join(inspection.Path, "runs", runDirectoryName(inspection.Model.Runs[0]), TelemetryFilename))
	if len(telemetry) != len(observation.Evaluations) {
		t.Fatalf("telemetry has %d evaluation rows, run record has %d evaluations", len(telemetry), len(observation.Evaluations))
	}
	for _, evaluation := range observation.Evaluations {
		row := telemetry[evaluation.Step]
		if row["telemetry_schema"] != "2" {
			t.Fatalf("step %d telemetry schema %q", evaluation.Step, row["telemetry_schema"])
		}
		for _, name := range []string{"heldout_loss", "heldout_perplexity", "heldout_nll_sum", "heldout_target_tokens", "heldout_utf8_bytes", "heldout_bits_per_byte"} {
			value, err := strconv.ParseFloat(row[name], 64)
			if err != nil || value != evaluation.Metrics[name] {
				t.Fatalf("step %d %s: telemetry %q, run record %v", evaluation.Step, name, row[name], evaluation.Metrics[name])
			}
		}
		if evaluation.Metrics["heldout_target_tokens"] != float64(set.TokenTargets) || evaluation.Metrics["heldout_utf8_bytes"] != float64(set.TextBytes) {
			t.Fatalf("step %d metrics are not pinned to %+v", evaluation.Step, *set)
		}
	}
	// The completion enriches the durable selected evaluation without
	// replacing its exact metrics.
	if _, ok := observation.Evaluations[0].Metrics["artifact_heldout_loss"]; !ok {
		t.Fatal("selected evaluation lost its artifact verification")
	}
}

// Transformers candidates are not resume points. An interrupted attempt that
// already reported durable evaluations restarts at step 0 and must not be
// blocked by, or merged with, the abandoned attempt's evaluations.
func TestTransformersInterruptedEvaluationsRestartCleanly(t *testing.T) {
	root := t.TempDir()
	attempts := 0
	builder := transformersTestBuilder(root, func(ctx context.Context, request training.Request) (training.Observation, error) {
		attempts++
		if request.Resume != nil {
			t.Fatalf("attempt %d received resume point %+v", attempts, request.Resume)
		}
		if attempts == 1 {
			for _, step := range []int64{1, 2} {
				evaluation := training.Evaluation{Step: step, Tokens: step, Metrics: transformersIntrinsicMetrics(request.EvaluationSet, 9)}
				request.Report(training.Event{Kind: "evaluation", Step: step, Tokens: step, Evaluation: &evaluation, Message: "abandoned"})
			}
			return training.Observation{}, context.Canceled
		}
		return transformersCandidateObservation(t, request), nil
	})
	compose := transformersMetricsCompose(t)
	prepared := []PreparedStage{preparedFixture(t, compose.Stages[0])}
	if _, err := builder.Compose(context.Background(), "restart", compose, prepared); !errors.Is(err, context.Canceled) {
		t.Fatalf("first attempt error = %v", err)
	}
	interrupted, err := Inspect(root, "restart")
	if err != nil || interrupted.Runs[0].State != RunInterrupted || interrupted.Runs[0].Progress == nil || len(interrupted.Runs[0].Progress.Evaluations) != 2 {
		t.Fatalf("interrupted run %+v: %v", interrupted.Runs, err)
	}
	completed, err := builder.Compose(context.Background(), "restart", compose, prepared)
	if err != nil {
		t.Fatal(err)
	}
	run := completed.Runs[0]
	if attempts != 2 || run.State != RunComplete || len(run.Attempts) != 2 || run.Attempts[1].ResumeStep != 0 {
		t.Fatalf("restart: attempts %d, run %+v", attempts, run)
	}
	for _, evaluation := range run.Observation.Evaluations {
		if evaluation.Metrics["heldout_loss"] == 9 {
			t.Fatalf("abandoned step %d evaluation merged into the completed run", evaluation.Step)
		}
	}
}
