// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/openwaldo/waldo/internal/training"
)

const TelemetryFilename = "TELEMETRY.csv"

var telemetryHeader = []string{
	"observed_utc", "elapsed_seconds", "run_id", "stage", "attempt",
	"event", "state", "step", "planned_steps", "tokens", "planned_tokens", "loss",
	"heldout_loss", "heldout_perplexity", "learning_rate", "tokens_per_second",
	"duration_seconds", "data_wait_seconds", "peak_memory_bytes", "training_flops",
	"achieved_tflops", "model_flop_utilization", "gradient_norm", "skipped_steps",
	"eta_seconds", "message",
}

type telemetryRow struct {
	Observed      time.Time
	Started       time.Time
	RunID         string
	Stage         string
	Attempt       int
	Event         string
	State         RunState
	PlannedSteps  int64
	PlannedTokens int64
	Training      *training.Event
	Message       string
}

func appendTelemetry(path string, row telemetryRow) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	writer := csv.NewWriter(file)
	if info.Size() == 0 {
		if err := writer.Write(telemetryHeader); err != nil {
			return err
		}
	}
	if err := writer.Write(telemetryRecord(row)); err != nil {
		return err
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return err
	}
	return file.Sync()
}

func telemetryRecord(row telemetryRow) []string {
	elapsed := row.Observed.Sub(row.Started).Seconds()
	if elapsed < 0 {
		elapsed = 0
	}
	values := make([]string, len(telemetryHeader))
	values[0] = formatTime(row.Observed)
	values[1] = strconv.FormatFloat(elapsed, 'f', 3, 64)
	values[2] = row.RunID
	values[3] = row.Stage
	values[4] = strconv.Itoa(row.Attempt)
	values[5] = row.Event
	values[6] = string(row.State)
	values[8] = strconv.FormatInt(row.PlannedSteps, 10)
	values[10] = strconv.FormatInt(row.PlannedTokens, 10)
	values[25] = row.Message
	if row.Training == nil {
		return values
	}
	event := row.Training
	values[7] = optionalInt(event.Step)
	values[9] = optionalInt(event.Tokens)
	if event.Loss != nil {
		values[11] = strconv.FormatFloat(*event.Loss, 'g', -1, 64)
	}
	if event.Evaluation != nil {
		values[12] = optionalMetric(event.Evaluation.Metrics, "heldout_loss")
		values[13] = optionalMetric(event.Evaluation.Metrics, "heldout_perplexity")
	}
	if event.LearningRate > 0 {
		values[14] = strconv.FormatFloat(event.LearningRate, 'g', -1, 64)
	}
	if event.TokensPerSecond > 0 {
		values[15] = strconv.FormatFloat(event.TokensPerSecond, 'g', -1, 64)
	}
	values[16] = optionalFloat(event.DurationSeconds)
	values[17] = optionalFloat(event.DataWaitSeconds)
	if event.PeakMemoryBytes > 0 {
		values[18] = strconv.FormatUint(event.PeakMemoryBytes, 10)
	}
	values[19] = optionalFloat(event.TrainingFLOPs)
	values[20] = optionalFloat(event.AchievedTFLOPS)
	values[21] = optionalFloat(event.ModelFLOPUtilization)
	if event.GradientNorm != nil {
		values[22] = strconv.FormatFloat(*event.GradientNorm, 'g', -1, 64)
	}
	values[23] = optionalInt(event.SkippedSteps)
	values[24] = optionalInt(event.ETASeconds)
	return values
}

func optionalFloat(value float64) string {
	if value == 0 {
		return ""
	}
	return strconv.FormatFloat(value, 'g', -1, 64)
}

func optionalInt(value int64) string {
	if value == 0 {
		return ""
	}
	return strconv.FormatInt(value, 10)
}

func optionalMetric(metrics map[string]float64, name string) string {
	value, ok := metrics[name]
	if !ok {
		return ""
	}
	return strconv.FormatFloat(value, 'g', -1, 64)
}

func telemetryError(path string, err error) error {
	return fmt.Errorf("append training telemetry %s: %w", path, err)
}
