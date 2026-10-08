// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/csv"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/openwaldo/waldo/internal/training"
)

func TestTelemetryRecordIncludesEfficiencyMeasurements(t *testing.T) {
	loss := 1.25
	gradient := 0.75
	event := training.Event{
		Kind: "progress", Step: 2, Tokens: 128, Loss: &loss,
		LearningRate: 0.001, TokensPerSecond: 2048, DurationSeconds: 0.5,
		DataWaitSeconds: 0.02, PeakMemoryBytes: 4096, TrainingFLOPs: 6e9,
		AchievedTFLOPS: 12, ModelFLOPUtilization: 0.5, GradientNorm: &gradient,
		SkippedSteps: 1, ETASeconds: 3,
	}
	started := time.Date(2026, 9, 13, 1, 0, 0, 0, time.UTC)
	record := telemetryRecord(telemetryRow{Observed: started.Add(time.Second), Started: started, Training: &event})
	if len(record) != len(telemetryHeader) {
		t.Fatalf("telemetry record has %d fields, header has %d", len(record), len(telemetryHeader))
	}
	want := map[string]string{
		"duration_seconds": "0.5", "data_wait_seconds": "0.02", "peak_memory_bytes": "4096",
		"training_flops": "6e+09", "achieved_tflops": "12", "model_flop_utilization": "0.5",
		"gradient_norm": "0.75", "skipped_steps": "1", "eta_seconds": "3",
	}
	for name, expected := range want {
		index := telemetryColumn(name)
		if index < 0 || record[index] != expected {
			t.Fatalf("telemetry %s = %q, want %q; record=%v", name, valueAt(record, index), expected, record)
		}
	}
}

func TestTelemetryRecordIncludesExactHeldoutTotals(t *testing.T) {
	evaluation := training.Evaluation{Metrics: map[string]float64{
		"heldout_loss": 1.5, "heldout_perplexity": 4.481689,
		"heldout_nll_sum": 150, "heldout_target_tokens": 100,
		"heldout_utf8_bytes": 75, "heldout_bits_per_byte": 150 / (75 * math.Ln2),
	}}
	record := telemetryRecord(telemetryRow{Training: &training.Event{Kind: "evaluation", Evaluation: &evaluation}})
	for _, name := range []string{"telemetry_schema", "heldout_nll_sum", "heldout_target_tokens", "heldout_utf8_bytes", "heldout_bits_per_byte"} {
		if record[telemetryColumn(name)] == "" {
			t.Fatalf("telemetry %s is empty: %v", name, record)
		}
	}
}

func TestAppendTelemetryPreservesVersionOneSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), TelemetryFilename)
	if err := os.WriteFile(path, []byte(strings.Join(telemetryHeaderV1, ",")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := appendTelemetry(path, telemetryRow{Observed: time.Now(), Started: time.Now(), Event: "run"}); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := csv.NewReader(file).ReadAll()
	_ = file.Close()
	if err != nil || len(rows) != 2 || len(rows[1]) != len(telemetryHeaderV1) {
		t.Fatalf("version-one telemetry rows=%v err=%v", rows, err)
	}
}

func telemetryColumn(name string) int {
	for index, column := range telemetryHeader {
		if column == name {
			return index
		}
	}
	return -1
}

func valueAt(values []string, index int) string {
	if index < 0 || index >= len(values) {
		return "<missing:" + strconv.Itoa(index) + ">"
	}
	return values[index]
}
