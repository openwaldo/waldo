// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"math"
	"regexp"
	"strings"
	"testing"
)

func TestWorkerEventRejectsInvalidEfficiencyTelemetry(t *testing.T) {
	invalid := []Event{
		{Kind: "progress", DurationSeconds: -1},
		{Kind: "progress", DataWaitSeconds: math.Inf(1)},
		{Kind: "progress", TrainingFLOPs: math.NaN()},
		{Kind: "progress", ModelFLOPUtilization: -0.01},
		{Kind: "progress", SkippedSteps: -1},
	}
	for _, event := range invalid {
		if err := event.Validate(); err == nil {
			t.Fatalf("invalid event accepted: %+v", event)
		}
	}
	gradient := math.Inf(1)
	if err := (Event{Kind: "progress", GradientNorm: &gradient}).Validate(); err == nil {
		t.Fatal("invalid gradient norm accepted")
	}
}

func TestEmbeddedWorkersEmitEfficiencyTelemetry(t *testing.T) {
	for name, source := range map[string]string{"pytorch": string(pyTorchWorker), "mlx": string(mlxWorker)} {
		for _, field := range []string{
			`"duration_seconds"`,
			`"data_wait_seconds"`,
			`"peak_memory_bytes"`,
			`"training_flops"`,
			`"achieved_tflops"`,
			`"gradient_norm"`,
			`"skipped_steps"`,
		} {
			if !strings.Contains(source, field) {
				t.Errorf("%s worker omits telemetry field %s", name, field)
			}
		}
	}
}

func TestEmbeddedWorkerEmitsOnlyRecognizedEventKinds(t *testing.T) {
	kindPattern := regexp.MustCompile(`"kind":\s*"([a-z_]+)"`)
	matches := kindPattern.FindAllStringSubmatch(string(pyTorchWorker), -1)
	if len(matches) == 0 {
		t.Fatal("no event kinds found in embedded worker; pattern or worker changed")
	}
	seen := map[string]bool{}
	for _, match := range matches {
		kind := match[1]
		if seen[kind] {
			continue
		}
		seen[kind] = true
		if err := (Event{Kind: kind}).Validate(); err != nil && strings.Contains(err.Error(), "unsupported worker event kind") {
			t.Errorf("embedded worker emits event kind %q that the Go driver rejects: %v", kind, err)
		}
	}
}

func TestEmbeddedWorkerEmitsOnlyRecognizedFrameKinds(t *testing.T) {
	framePattern := regexp.MustCompile(`emit\(\s*"([a-z_]+)"`)
	matches := framePattern.FindAllStringSubmatch(string(pyTorchWorker), -1)
	if len(matches) == 0 {
		t.Fatal("no emit frame kinds found in embedded worker; pattern or worker changed")
	}
	seen := map[string]bool{}
	for _, match := range matches {
		kind := match[1]
		if seen[kind] {
			continue
		}
		seen[kind] = true
		frame := WorkerOutputFrame{Kind: kind, Schema: WorkerProtocolSchema, Error: "x"}
		if err := frame.Validate(); err != nil && strings.Contains(err.Error(), "unsupported worker output kind") {
			t.Errorf("embedded worker emits frame kind %q that the Go driver rejects: %v", kind, err)
		}
	}
}

func TestWorkerArtifactIntegrityErrorIsTypedAndNonRetryable(t *testing.T) {
	var observed error
	err := ReadWorkerOutput(strings.NewReader(`{"kind":"error","schema":1,"error":"artifact degraded","error_class":"artifact-integrity"}`+"\n"), func(frame WorkerOutputFrame) error {
		observed = &WorkerError{Message: frame.Error, Class: frame.ErrorClass}
		return observed
	})
	if err == nil || !IsNonRetryableWorkerError(err) || !IsNonRetryableWorkerError(observed) {
		t.Fatalf("typed worker error = %v / %v", err, observed)
	}
	if err := (WorkerOutputFrame{Kind: "error", Schema: 1, Error: "x", ErrorClass: "unknown"}).Validate(); err == nil {
		t.Fatal("unsupported worker error class was accepted")
	}
}
