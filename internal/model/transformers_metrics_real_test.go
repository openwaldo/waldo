// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model_test

import (
	"encoding/csv"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/openwaldo/waldo/internal/model"
)

// Opt-in real Transformers test of ADR 0081 metrics. Byte and pinned HF
// tokenizers evaluate the same Unicode held-out text through a whole-record
// view and a contiguous tail. Each run record and its telemetry must agree
// exactly, and the denominators must be the pinned targets and UTF-8 bytes.
func TestTransformersExactHeldoutMetrics(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" {
		t.Skip("set WALDO_TRANSFORMERS_PYTHON and WALDO_TRANSFORMERS_WHEEL for real Transformers training")
	}
	first := strings.Repeat("naïve café ☃ — ünïcödé 日本語 text. ", 20)
	second := strings.Repeat("Ελληνικά and emoji 🦉 stay whole; ", 18)
	const fraction = 0.1
	tail := int64(len(first)) - int64(math.Floor(float64(len(first))*(1-fraction)))
	for offset := int64(len(first)) - tail; !utf8.RuneStart(first[offset]); offset++ {
		tail--
	}
	views := []struct {
		name, selection string
		texts           []string
		bytes           []int64
	}{
		{"records", "", []string{first, second}, []int64{int64(len(first)), int64(len(second))}},
		{"contiguous-tail", "contiguous-tail-v1", []string{first}, []int64{tail}},
	}
	byteEvidence := map[string]int64{}
	for _, variant := range []struct{ name, compose string }{
		{"byte", "transformers-qwen3-smoke.yaml"},
		{"huggingface", "transformers-qwen3-tokenizer-smoke.yaml"},
	} {
		for _, view := range views {
			t.Run(variant.name+"/"+view.name, func(t *testing.T) {
				if variant.name == "huggingface" && os.Getenv("WALDO_HF_TOKENIZER_DIR") == "" {
					t.Skip("set WALDO_HF_TOKENIZER_DIR for the pinned HF tokenizer")
				}
				compose, _, err := model.LoadCompose("../../docs/examples/" + variant.compose)
				if err != nil {
					t.Fatal(err)
				}
				compose.Stages[0].Corpora = model.NewCorpusSelections([]string{"example"})
				compose.Stages[0].Parameters.EvaluationSelection = view.selection
				builder := model.Builder{Root: t.TempDir()}
				inspection, err := builder.Compose(t.Context(), "metrics", compose, []model.PreparedStage{hfPreparedTexts(t, compose.Stages[0], view.texts...)})
				if err != nil {
					t.Fatal(err)
				}
				set := inspection.RunBOMs[0].EvaluationSet
				observation := inspection.Runs[0].Observation
				if set == nil || set.Records != 1 || observation == nil || len(observation.Evaluations) != 2 {
					t.Fatalf("evaluation set %+v, observation %+v", set, observation)
				}
				expectedBytes := false
				for _, value := range view.bytes {
					expectedBytes = expectedBytes || set.TextBytes == value
				}
				if !expectedBytes {
					t.Fatalf("held-out UTF-8 bytes %d, want one of %v", set.TextBytes, view.bytes)
				}
				switch variant.name {
				case "byte":
					// One target per UTF-8 byte: the first byte is context, EOS is scored.
					if set.TokenTargets != set.TextBytes {
						t.Fatalf("byte tokenizer pins %d targets for %d bytes", set.TokenTargets, set.TextBytes)
					}
					byteEvidence[view.name] = set.TextBytes
				case "huggingface":
					if want, ok := byteEvidence[view.name]; ok && set.TextBytes != want {
						t.Fatalf("HF held-out bytes %d differ from byte tokenizer bytes %d for the same text", set.TextBytes, want)
					}
					if set.TokenTargets <= 0 || set.TokenTargets >= set.TextBytes {
						t.Fatalf("HF tokenizer pins %d targets for %d bytes", set.TokenTargets, set.TextBytes)
					}
				}
				telemetry := runTelemetry(t, inspection.Path, inspection.Model.Runs[0].ID)
				if len(telemetry) != len(observation.Evaluations) {
					t.Fatalf("telemetry has %d evaluation rows, run record has %d", len(telemetry), len(observation.Evaluations))
				}
				for _, evaluation := range observation.Evaluations {
					metrics := evaluation.Metrics
					nll := metrics["heldout_nll_sum"]
					if metrics["heldout_target_tokens"] != float64(set.TokenTargets) || metrics["heldout_utf8_bytes"] != float64(set.TextBytes) ||
						metrics["heldout_loss"] != nll/float64(set.TokenTargets) || metrics["heldout_bits_per_byte"] != nll/(math.Ln2*float64(set.TextBytes)) {
						t.Fatalf("step %d metrics %v are not exact over %+v", evaluation.Step, metrics, *set)
					}
					row := telemetry[strconv.FormatInt(evaluation.Step, 10)]
					for _, name := range []string{"heldout_loss", "heldout_perplexity", "heldout_nll_sum", "heldout_target_tokens", "heldout_utf8_bytes", "heldout_bits_per_byte"} {
						if value, err := strconv.ParseFloat(row[name], 64); err != nil || value != metrics[name] {
							t.Fatalf("step %d %s: telemetry %q, run record %v", evaluation.Step, name, row[name], metrics[name])
						}
					}
					t.Logf("step %d: %d targets, %d bytes, loss %.6f nats/target, %.6f bits/byte", evaluation.Step, set.TokenTargets, set.TextBytes, metrics["heldout_loss"], metrics["heldout_bits_per_byte"])
				}
			})
		}
	}
}

// runTelemetry returns a run's evaluation telemetry rows keyed by step.
func runTelemetry(t *testing.T, modelPath, runID string) map[string]map[string]string {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(modelPath, "runs", "*", model.TelemetryFilename))
	if err != nil {
		t.Fatal(err)
	}
	result := map[string]map[string]string{}
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		rows, err := csv.NewReader(file).ReadAll()
		file.Close()
		if err != nil || len(rows) == 0 {
			t.Fatalf("%s: %v", path, err)
		}
		for _, row := range rows[1:] {
			fields := map[string]string{}
			for index, name := range rows[0] {
				fields[name] = row[index]
			}
			if fields["run_id"] == runID && fields["event"] == "evaluation" {
				if fields["telemetry_schema"] != "2" {
					t.Fatalf("telemetry schema %q", fields["telemetry_schema"])
				}
				result[fields["step"]] = fields
			}
		}
	}
	return result
}
