// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/openwaldo/waldo/internal/training"
)

func TestBuildAdviceUsesTelemetryToRecommendStop(t *testing.T) {
	root := t.TempDir()
	runPath := filepath.Join("runs", "0001-pretrain-run1")
	if err := os.MkdirAll(filepath.Join(root, runPath), 0o755); err != nil {
		t.Fatal(err)
	}
	telemetry := strings.Join(telemetryHeader, ",") + "\n" +
		strings.Join([]string{"2026-08-09T18:00:00Z", "10", "run1", "pretrain", "1", "evaluation", "running", "25", "100", "250", "1000", "1.2", "1.0", "2.718", "0.001", "1000", "", "", "", "", "", "", "", "", "75", "first evaluation"}, ",") + "\n" +
		strings.Join([]string{"2026-08-09T18:01:00Z", "70", "run1", "pretrain", "1", "evaluation", "running", "50", "100", "500", "1000", "1.4", "1.3", "3.669", "0.0008", "1200", "", "", "", "", "", "", "", "", "60", "second evaluation"}, ",") + "\n"
	if err := os.WriteFile(filepath.Join(root, runPath, TelemetryFilename), []byte(telemetry), 0o644); err != nil {
		t.Fatal(err)
	}
	inspection := Inspection{
		Path:    root,
		Model:   ModelRecord{Name: "example", Runs: []RunPin{{ID: "run1", Stage: "pretrain", Ordinal: 1, State: RunRunning}}},
		Runs:    []RunRecord{{ID: "run1", State: RunRunning, Progress: &training.Progress{Steps: 50, ConsumedTokens: 500}}},
		RunBOMs: []RunBOM{{ID: "run1", Execution: training.Execution{Backend: training.Identity{Name: "pytorch"}}, Parameters: training.ResolvedParameters{Steps: 100, PlannedTokenCapacity: 1000}}},
		BOM:     ModelBOM{Runs: []ModelBOMRun{{ID: "run1", RunBOM: filepath.ToSlash(filepath.Join(runPath, "RUN-BOM.json"))}}},
	}
	report, err := BuildAdvice(inspection, time.Date(2026, 8, 9, 18, 1, 30, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if report.Action != "stop" || report.Run == nil || report.Run.ProgressPercent != 50 || report.Run.HeldoutLoss == nil || *report.Run.HeldoutLoss != 1.3 {
		t.Fatalf("advice = %+v", report)
	}
	if !strings.Contains(strings.Join(report.Findings, "\n"), "worsened 30.0%") {
		t.Fatalf("findings = %v", report.Findings)
	}
}

func TestBuildAdviceForUntrainedModel(t *testing.T) {
	report, err := BuildAdvice(Inspection{Path: t.TempDir(), Model: ModelRecord{Name: "new-model"}}, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if report.State != "untrained" || report.Action != "train" || report.Run != nil {
		t.Fatalf("advice = %+v", report)
	}
}
