// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"

	"github.com/openwaldo/waldo/internal/training"
)

const TelemetryFilename = "TELEMETRY.csv"

var telemetryHeaderV1 = []string{
	"observed_utc", "elapsed_seconds", "run_id", "stage", "attempt",
	"event", "state", "step", "planned_steps", "tokens", "planned_tokens", "loss",
	"heldout_loss", "heldout_perplexity", "learning_rate", "tokens_per_second",
	"duration_seconds", "data_wait_seconds", "peak_memory_bytes", "training_flops",
	"achieved_tflops", "model_flop_utilization", "gradient_norm", "skipped_steps",
	"eta_seconds", "message",
}

var telemetryHeader = append(append([]string(nil), telemetryHeaderV1...),
	"telemetry_schema", "heldout_nll_sum", "heldout_target_tokens", "heldout_utf8_bytes", "heldout_bits_per_byte")

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
	header := telemetryHeader
	if info.Size() == 0 {
		if err := writer.Write(header); err != nil {
			return err
		}
	} else {
		header, err = readTelemetryHeader(path)
		if err != nil {
			return err
		}
	}
	if err := writer.Write(telemetryRecordForHeader(row, header)); err != nil {
		return err
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return err
	}
	return file.Sync()
}

func telemetryRecord(row telemetryRow) []string {
	return telemetryRecordForHeader(row, telemetryHeader)
}

func telemetryRecordForHeader(row telemetryRow, header []string) []string {
	elapsed := row.Observed.Sub(row.Started).Seconds()
	if elapsed < 0 {
		elapsed = 0
	}
	fields := map[string]string{
		"observed_utc": formatTime(row.Observed), "elapsed_seconds": strconv.FormatFloat(elapsed, 'f', 3, 64),
		"run_id": row.RunID, "stage": row.Stage, "attempt": strconv.Itoa(row.Attempt), "event": row.Event,
		"state": string(row.State), "planned_steps": strconv.FormatInt(row.PlannedSteps, 10),
		"planned_tokens": strconv.FormatInt(row.PlannedTokens, 10), "message": row.Message, "telemetry_schema": "2",
	}
	if row.Training == nil {
		return telemetryFields(header, fields)
	}
	event := row.Training
	fields["step"] = optionalInt(event.Step)
	fields["tokens"] = optionalInt(event.Tokens)
	if event.Loss != nil {
		fields["loss"] = strconv.FormatFloat(*event.Loss, 'g', -1, 64)
	}
	if event.Evaluation != nil {
		fields["heldout_loss"] = optionalMetric(event.Evaluation.Metrics, "heldout_loss")
		fields["heldout_perplexity"] = optionalMetric(event.Evaluation.Metrics, "heldout_perplexity")
		fields["heldout_nll_sum"] = optionalMetric(event.Evaluation.Metrics, "heldout_nll_sum")
		fields["heldout_target_tokens"] = optionalMetric(event.Evaluation.Metrics, "heldout_target_tokens")
		fields["heldout_utf8_bytes"] = optionalMetric(event.Evaluation.Metrics, "heldout_utf8_bytes")
		fields["heldout_bits_per_byte"] = optionalMetric(event.Evaluation.Metrics, "heldout_bits_per_byte")
	}
	if event.LearningRate > 0 {
		fields["learning_rate"] = strconv.FormatFloat(event.LearningRate, 'g', -1, 64)
	}
	if event.TokensPerSecond > 0 {
		fields["tokens_per_second"] = strconv.FormatFloat(event.TokensPerSecond, 'g', -1, 64)
	}
	fields["duration_seconds"] = optionalFloat(event.DurationSeconds)
	fields["data_wait_seconds"] = optionalFloat(event.DataWaitSeconds)
	if event.PeakMemoryBytes > 0 {
		fields["peak_memory_bytes"] = strconv.FormatUint(event.PeakMemoryBytes, 10)
	}
	fields["training_flops"] = optionalFloat(event.TrainingFLOPs)
	fields["achieved_tflops"] = optionalFloat(event.AchievedTFLOPS)
	fields["model_flop_utilization"] = optionalFloat(event.ModelFLOPUtilization)
	if event.GradientNorm != nil {
		fields["gradient_norm"] = strconv.FormatFloat(*event.GradientNorm, 'g', -1, 64)
	}
	fields["skipped_steps"] = optionalInt(event.SkippedSteps)
	fields["eta_seconds"] = optionalInt(event.ETASeconds)
	return telemetryFields(header, fields)
}

func telemetryFields(header []string, fields map[string]string) []string {
	values := make([]string, len(header))
	for index, name := range header {
		values[index] = fields[name]
	}
	return values
}

func readTelemetryHeader(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	header, err := csv.NewReader(file).Read()
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("read telemetry header: %w", err)
	}
	if !equalStrings(header, telemetryHeaderV1) && !equalStrings(header, telemetryHeader) {
		return nil, fmt.Errorf("unsupported telemetry schema header")
	}
	return header, nil
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
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
