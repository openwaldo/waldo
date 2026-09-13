// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"strconv"
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
