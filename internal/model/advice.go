// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/openwaldo/waldo/internal/training"
)

type Advice struct {
	Model          string       `json:"model"`
	State          string       `json:"state"`
	Action         string       `json:"action"`
	Summary        string       `json:"summary"`
	Architecture   Architecture `json:"architecture"`
	Parameters     uint64       `json:"approximate_parameters"`
	Findings       []string     `json:"findings"`
	Run            *AdviceRun   `json:"run,omitempty"`
	Compose        *Compose     `json:"compose,omitempty"`
	ComposePresent bool         `json:"compose_present"`
}

type AdviceRun struct {
	ID                  string                      `json:"id"`
	Stage               string                      `json:"stage"`
	Ordinal             int                         `json:"ordinal"`
	State               RunState                    `json:"state"`
	Backend             string                      `json:"backend,omitempty"`
	Step                int64                       `json:"step,omitempty"`
	PlannedSteps        int64                       `json:"planned_steps,omitempty"`
	ProgressPercent     float64                     `json:"progress_percent,omitempty"`
	ConsumedTokens      int64                       `json:"consumed_tokens,omitempty"`
	PlannedTokens       int64                       `json:"planned_tokens,omitempty"`
	Loss                *float64                    `json:"loss,omitempty"`
	HeldoutLoss         *float64                    `json:"heldout_loss,omitempty"`
	InitialHeldoutLoss  *float64                    `json:"initial_heldout_loss,omitempty"`
	SelectedStep        int64                       `json:"selected_step,omitempty"`
	SelectedHeldoutLoss *float64                    `json:"selected_heldout_loss,omitempty"`
	LearningRate        float64                     `json:"learning_rate,omitempty"`
	TokensPerSecond     float64                     `json:"tokens_per_second,omitempty"`
	ETASeconds          int64                       `json:"eta_seconds,omitempty"`
	LastObserved        string                      `json:"last_observed_utc,omitempty"`
	Error               string                      `json:"error,omitempty"`
	Parameters          training.ResolvedParameters `json:"parameters"`
	Corpus              AdviceCorpus                `json:"corpus"`
}

type AdviceCorpus struct {
	Paths  []string `json:"paths"`
	Shards int      `json:"shards"`
	Docs   int64    `json:"docs"`
	Tokens int64    `json:"tokens"`
	Bytes  int64    `json:"bytes"`
}

type adviceTelemetry struct {
	Observed        time.Time
	Step            int64
	Tokens          int64
	Loss            *float64
	HeldoutLoss     *float64
	LearningRate    float64
	TokensPerSecond float64
	ETASeconds      int64
}

func BuildAdvice(inspection Inspection, now time.Time) (Advice, error) {
	result := Advice{
		Model: inspection.Model.Name, State: "untrained", Action: "train",
		Architecture: inspection.Model.Architecture,
		Parameters:   inspection.Model.Forecast.ApproximateParameters,
	}
	compose, err := loadPersistedCompose(inspection.Path)
	if err != nil {
		return Advice{}, err
	}
	if compose != nil {
		result.Compose = compose
		result.ComposePresent = true
		result.Findings = append(result.Findings, fmt.Sprintf("saved compose defines %d training stage(s)", len(compose.Stages)))
	} else {
		result.Findings = append(result.Findings, "model has no saved compose; advice is based on recorded runs")
	}
	if len(inspection.Model.Runs) == 0 {
		result.Summary = "The model has no training runs yet."
		return result, nil
	}

	position := len(inspection.Model.Runs) - 1
	pin, run, runBOM := inspection.Model.Runs[position], inspection.Runs[position], inspection.RunBOMs[position]
	current := &AdviceRun{
		ID: pin.ID, Stage: pin.Stage, Ordinal: pin.Ordinal, State: run.State,
		Backend: runBOM.Execution.Backend.Name, PlannedSteps: runBOM.Parameters.Steps,
		PlannedTokens: runBOM.Parameters.PlannedTokenCapacity, Error: run.Error,
		Parameters: runBOM.Parameters,
		Corpus: AdviceCorpus{
			Paths: append([]string(nil), runBOM.CorpusBOM.Paths...), Shards: len(runBOM.CorpusBOM.Shards),
			Docs: runBOM.CorpusBOM.Totals.Docs, Tokens: runBOM.CorpusBOM.Totals.Tokens, Bytes: runBOM.CorpusBOM.Totals.Bytes,
		},
	}
	if run.Progress != nil {
		current.Step, current.ConsumedTokens, current.Loss = run.Progress.Steps, run.Progress.ConsumedTokens, run.Progress.LastLoss
		applyEvaluationAdvice(current, run.Progress.Evaluations)
	}
	if run.Observation != nil {
		current.Step, current.ConsumedTokens, current.Loss = run.Observation.Steps, run.Observation.ConsumedTokens, run.Observation.FinalLoss
		applyEvaluationAdvice(current, run.Observation.Evaluations)
	}
	telemetryPath := filepath.Join(inspection.Path, filepath.Dir(filepath.FromSlash(inspection.BOM.Runs[position].RunBOM)), TelemetryFilename)
	telemetry, err := readAdviceTelemetry(telemetryPath)
	if err != nil {
		return Advice{}, err
	}
	for _, sample := range telemetry {
		if sample.Step > 0 {
			current.Step = sample.Step
		}
		if sample.Tokens > 0 {
			current.ConsumedTokens = sample.Tokens
		}
		if sample.Loss != nil {
			current.Loss = sample.Loss
		}
		if sample.HeldoutLoss != nil {
			if current.InitialHeldoutLoss == nil {
				current.InitialHeldoutLoss = sample.HeldoutLoss
			}
			current.HeldoutLoss = sample.HeldoutLoss
		}
		if sample.LearningRate > 0 {
			current.LearningRate = sample.LearningRate
		}
		if sample.TokensPerSecond > 0 {
			current.TokensPerSecond = sample.TokensPerSecond
		}
		if sample.ETASeconds > 0 {
			current.ETASeconds = sample.ETASeconds
		}
		if !sample.Observed.IsZero() {
			current.LastObserved = sample.Observed.UTC().Format(time.RFC3339Nano)
		}
	}
	if run.Observation != nil && run.Observation.SelectedCheckpoint != nil {
		selected := run.Observation.SelectedCheckpoint
		last := current.HeldoutLoss
		value := selected.Value
		current.SelectedStep = selected.Step
		current.SelectedHeldoutLoss = &value
		current.HeldoutLoss = &value
		if last != nil && *last > value {
			result.Findings = append(result.Findings, fmt.Sprintf("the last optimizer step held-out loss was %.1f%% worse than the selected checkpoint at step %d", 100*(*last/value-1), selected.Step))
		}
	}
	if current.PlannedSteps > 0 {
		current.ProgressPercent = math.Min(100, 100*float64(current.Step)/float64(current.PlannedSteps))
	}
	result.Run, result.State = current, string(run.State)
	result.Action, result.Summary = baseAdvice(run.State)

	if current.Loss != nil && (math.IsNaN(*current.Loss) || math.IsInf(*current.Loss, 0)) {
		result.Action = "stop"
		result.Findings = append(result.Findings, "training loss is non-finite; stop the run and inspect data, precision, and learning rate")
	}
	if current.InitialHeldoutLoss != nil && current.HeldoutLoss != nil && *current.InitialHeldoutLoss > 0 {
		change := (*current.HeldoutLoss / *current.InitialHeldoutLoss) - 1
		switch {
		case change > 0.25:
			result.Action = "stop"
			result.Findings = append(result.Findings, fmt.Sprintf("held-out loss has worsened %.1f%% from its first recorded value", 100*change))
		case change > 0.05 && result.Action == "let-run":
			result.Action = "inspect"
			result.Findings = append(result.Findings, fmt.Sprintf("held-out loss has worsened %.1f%%; verify the trend at the next evaluation", 100*change))
		case change < -0.05:
			result.Findings = append(result.Findings, fmt.Sprintf("held-out loss has improved %.1f%% from its first recorded value", -100*change))
		}
	}
	if run.State == RunRunning && current.LastObserved != "" {
		observed, _ := time.Parse(time.RFC3339Nano, current.LastObserved)
		if now.Sub(observed) > 10*time.Minute && result.Action == "let-run" {
			result.Action = "inspect"
			result.Findings = append(result.Findings, fmt.Sprintf("no telemetry has been recorded for %s", now.Sub(observed).Round(time.Minute)))
		}
	}
	if current.Step > 0 && current.PlannedSteps > 0 {
		result.Findings = append(result.Findings, fmt.Sprintf("latest run is %.1f%% complete at step %d of %d", current.ProgressPercent, current.Step, current.PlannedSteps))
	}
	if current.Error != "" {
		result.Findings = append(result.Findings, "recorded run error: "+current.Error)
	}
	return result, nil
}

func baseAdvice(state RunState) (string, string) {
	switch state {
	case RunRunning:
		return "let-run", "The latest run is active and has no recorded stop condition."
	case RunFailed:
		return "fix", "The latest run failed; address its recorded error before resuming."
	case RunInterrupted:
		return "fix", "The latest run was interrupted and can be inspected for a resumable checkpoint."
	case RunComplete:
		return "complete", "The latest run completed; review evaluation quality before exporting or extending training."
	default:
		return "wait", "The latest run is planned but has not started."
	}
}

func applyEvaluationAdvice(current *AdviceRun, evaluations []training.Evaluation) {
	for _, evaluation := range evaluations {
		loss, ok := evaluation.Metrics["heldout_loss"]
		if !ok {
			continue
		}
		value := loss
		if current.InitialHeldoutLoss == nil {
			current.InitialHeldoutLoss = &value
		}
		current.HeldoutLoss = &value
	}
}

func loadPersistedCompose(directory string) (*Compose, error) {
	data, err := os.ReadFile(filepath.Join(directory, "COMPOSE.json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var compose Compose
	if err := json.Unmarshal(data, &compose); err != nil {
		return nil, fmt.Errorf("read saved model compose: %w", err)
	}
	if err := compose.Validate(); err != nil {
		return nil, fmt.Errorf("saved model compose: %w", err)
	}
	return &compose, nil
}

func readAdviceTelemetry(path string) ([]adviceTelemetry, error) {
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("read training telemetry %s: %w", path, err)
	}
	columns := map[string]int{}
	for index, name := range header {
		columns[name] = index
	}
	var samples []adviceTelemetry
	for {
		record, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read training telemetry %s: %w", path, err)
		}
		sample := adviceTelemetry{
			Observed:        parseAdviceTime(recordValue(record, columns, "observed_utc")),
			Step:            parseAdviceInt(recordValue(record, columns, "step")),
			Tokens:          parseAdviceInt(recordValue(record, columns, "tokens")),
			LearningRate:    parseAdviceFloat(recordValue(record, columns, "learning_rate")),
			TokensPerSecond: parseAdviceFloat(recordValue(record, columns, "tokens_per_second")),
			ETASeconds:      parseAdviceInt(recordValue(record, columns, "eta_seconds")),
		}
		sample.Loss = parseAdviceMetric(recordValue(record, columns, "loss"))
		sample.HeldoutLoss = parseAdviceMetric(recordValue(record, columns, "heldout_loss"))
		samples = append(samples, sample)
	}
	return samples, nil
}

func recordValue(record []string, columns map[string]int, name string) string {
	index, ok := columns[name]
	if !ok || index >= len(record) {
		return ""
	}
	return record[index]
}

func parseAdviceInt(value string) int64 { parsed, _ := strconv.ParseInt(value, 10, 64); return parsed }
func parseAdviceFloat(value string) float64 {
	parsed, _ := strconv.ParseFloat(value, 64)
	return parsed
}
func parseAdviceTime(value string) time.Time {
	parsed, _ := time.Parse(time.RFC3339Nano, value)
	return parsed
}
func parseAdviceMetric(value string) *float64 {
	if value == "" {
		return nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return nil
	}
	return &parsed
}
