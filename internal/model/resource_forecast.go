// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"math"
	"sort"
)

const (
	forecastCatalog           = "openwaldo-training-hardware-2026-08-13"
	forecastFormula           = "6 * parameters * planned_tokens / effective_throughput; catalog estimates include 8% run overhead and exact observed configurations use measured active run time"
	forecastCalibrationSchema = 1
)

type ResourceForecast struct {
	Catalog               string                  `json:"catalog"`
	CatalogNote           string                  `json:"catalog_note,omitempty"`
	Formula               string                  `json:"formula"`
	ApproximateParameters uint64                  `json:"approximate_parameters"`
	PlannedTokens         int64                   `json:"planned_tokens"`
	EpochDerivedStages    []string                `json:"epoch_derived_stages,omitempty"`
	TrainingFLOPs         float64                 `json:"training_flops"`
	Calibrations          []ForecastCalibration   `json:"calibrations,omitempty"`
	Configurations        []HardwareConfiguration `json:"configurations,omitempty"`
}

// ForecastCalibration is aggregate, reproducible evidence from completed
// WALDO runs on one exact accelerator topology.
type ForecastCalibration struct {
	Schema          int     `json:"schema"`
	Manufacturer    string  `json:"manufacturer"`
	Accelerator     string  `json:"accelerator"`
	GPUs            int     `json:"gpus"`
	Runs            int     `json:"runs"`
	TrainingFLOPs   float64 `json:"training_flops"`
	ActiveSeconds   float64 `json:"active_seconds"`
	EffectiveTFLOPS float64 `json:"effective_tflops"`
	EvidenceSHA256  string  `json:"evidence_sha256"`
}

type HardwareConfiguration struct {
	Manufacturer        string  `json:"manufacturer"`
	Accelerator         string  `json:"accelerator"`
	GPUs                int     `json:"gpus"`
	Nodes               int     `json:"nodes"`
	MemoryPerGPUBytes   uint64  `json:"memory_per_gpu_bytes"`
	RequiredPerGPUBytes uint64  `json:"required_per_gpu_bytes"`
	EffectiveTFLOPS     float64 `json:"effective_tflops"`
	ApproximateSeconds  int64   `json:"approximate_seconds"`
	EstimateSource      string  `json:"estimate_source"`
	ObservedRuns        int     `json:"observed_runs,omitempty"`
}

type acceleratorProfile struct {
	manufacturer string
	accelerator  string
	memoryBytes  uint64
	counts       []int
	throughput   float64
	scale2       float64
	scale4       float64
	scale8       float64
	gpusPerNode  int
}

var acceleratorCatalog = []acceleratorProfile{
	{manufacturer: "Apple", accelerator: "M4 Max 40-core GPU", memoryBytes: 128 << 30, counts: []int{1}, throughput: 18},
	{manufacturer: "Apple", accelerator: "M5 Max GPU", memoryBytes: 128 << 30, counts: []int{1}, throughput: 27},
	{manufacturer: "NVIDIA", accelerator: "GeForce RTX 5090", memoryBytes: 32 << 30, counts: []int{1, 4, 8}, throughput: 125, scale4: .60, scale8: .45},
	{manufacturer: "NVIDIA", accelerator: "RTX PRO 6000 Blackwell", memoryBytes: 96 << 30, counts: []int{1, 4, 8}, throughput: 140, scale4: .65, scale8: .50},
	{manufacturer: "NVIDIA", accelerator: "H100 SXM", memoryBytes: 80 << 30, counts: []int{1, 4, 8}, throughput: 300, scale4: .82, scale8: .76},
	{manufacturer: "NVIDIA", accelerator: "H200 SXM", memoryBytes: 141 << 30, counts: []int{1, 4, 8}, throughput: 330, scale4: .84, scale8: .78},
	{manufacturer: "NVIDIA", accelerator: "GB10 Grace Blackwell (DGX Spark)", memoryBytes: 128 << 30, counts: []int{1, 2, 4}, throughput: 40, scale2: .70, scale4: .50, gpusPerNode: 1},
	{manufacturer: "AMD", accelerator: "Instinct MI325X", memoryBytes: 256 << 30, counts: []int{1, 4, 8}, throughput: 280, scale4: .80, scale8: .72},
	{manufacturer: "AMD", accelerator: "Instinct MI350X", memoryBytes: 288 << 30, counts: []int{1, 4, 8}, throughput: 400, scale4: .82, scale8: .75},
}

// ForecastCompose validates the portable compose and estimates its declared
// training budget. Index resolution is unnecessary because the workload is
// steps * batch size * sequence length for each stage.
func ForecastCompose(compose Compose) (ResourceForecast, error) {
	return ForecastComposeWithCalibration(compose, nil)
}

func ForecastComposeWithCalibration(compose Compose, calibrations []ForecastCalibration) (ResourceForecast, error) {
	if err := compose.Validate(); err != nil {
		return ResourceForecast{}, err
	}
	plan, err := forecastPlanForCompose(compose)
	if err != nil {
		return ResourceForecast{}, err
	}
	return forecastPlanWithCalibration(plan, calibrations)
}

func forecastPlan(plan Plan) (ResourceForecast, error) {
	return forecastPlanWithCalibration(plan, nil)
}

func forecastPlanWithCalibration(plan Plan, calibrations []ForecastCalibration) (ResourceForecast, error) {
	var plannedTokens int64
	for _, stage := range plan.Stages {
		if stage.PlannedTokens == 0 {
			continue
		}
		if stage.PlannedTokens < 0 || plannedTokens > math.MaxInt64-stage.PlannedTokens {
			return ResourceForecast{}, fmt.Errorf("planned training tokens overflow int64")
		}
		plannedTokens += stage.PlannedTokens
	}
	parameters := plan.Forecast.ApproximateParameters
	trainingFLOPs := 6 * float64(parameters) * float64(plannedTokens)
	if math.IsInf(trainingFLOPs, 0) || math.IsNaN(trainingFLOPs) {
		return ResourceForecast{}, fmt.Errorf("training workload is too large to forecast")
	}
	report := ResourceForecast{
		Catalog: forecastCatalog, Formula: forecastFormula,
		ApproximateParameters: parameters, PlannedTokens: plannedTokens,
		TrainingFLOPs: trainingFLOPs,
	}
	for _, stage := range plan.Stages {
		if stage.PlannedTokens == 0 {
			report.EpochDerivedStages = append(report.EpochDerivedStages, stage.Name)
		}
	}
	for _, profile := range acceleratorCatalog {
		for _, count := range profile.counts {
			required, err := requiredMemoryPerGPU(plan, count)
			if err != nil {
				return ResourceForecast{}, err
			}
			// Keep 10% free for the runtime, allocator fragmentation, and
			// operating-system use. Configurations that do not fit are omitted.
			if required > profile.memoryBytes-profile.memoryBytes/10 {
				continue
			}
			effective := profile.throughput * float64(count) * scaling(profile, count)
			overhead := 1.08
			source := "catalog"
			observedRuns := 0
			if calibration, ok := matchingCalibration(calibrations, profile, count); ok {
				effective = calibration.EffectiveTFLOPS
				overhead = 1
				source = "observed-runs"
				observedRuns = calibration.Runs
				if !containsCalibration(report.Calibrations, calibration.EvidenceSHA256) {
					report.Calibrations = append(report.Calibrations, calibration)
				}
			}
			secondsFloat := float64(-1)
			if plannedTokens > 0 {
				secondsFloat = math.Ceil(trainingFLOPs / (effective * 1e12) * overhead)
			}
			if secondsFloat > math.MaxInt64 {
				return ResourceForecast{}, fmt.Errorf("training duration exceeds the supported forecast range")
			}
			seconds := int64(secondsFloat)
			if plannedTokens > 0 && seconds < 1 {
				seconds = 1
			}
			report.Configurations = append(report.Configurations, HardwareConfiguration{
				Manufacturer: profile.manufacturer, Accelerator: profile.accelerator,
				GPUs: count, Nodes: nodesFor(profile, count), MemoryPerGPUBytes: profile.memoryBytes,
				RequiredPerGPUBytes: required, EffectiveTFLOPS: effective,
				ApproximateSeconds: seconds, EstimateSource: source, ObservedRuns: observedRuns,
			})
		}
	}
	if len(report.Configurations) == 0 {
		report.CatalogNote = fmt.Sprintf("no configuration in forecast catalog %s has sufficient memory for this workload", forecastCatalog)
	}
	sort.Slice(report.Configurations, func(i, j int) bool {
		left, right := report.Configurations[i], report.Configurations[j]
		if left.ApproximateSeconds != right.ApproximateSeconds {
			return left.ApproximateSeconds > right.ApproximateSeconds
		}
		if left.Manufacturer != right.Manufacturer {
			return left.Manufacturer < right.Manufacturer
		}
		if left.Accelerator != right.Accelerator {
			return left.Accelerator < right.Accelerator
		}
		return left.GPUs < right.GPUs
	})
	return report, nil
}

func containsCalibration(calibrations []ForecastCalibration, digest string) bool {
	for _, calibration := range calibrations {
		if calibration.EvidenceSHA256 == digest {
			return true
		}
	}
	return false
}

func matchingCalibration(calibrations []ForecastCalibration, profile acceleratorProfile, GPUs int) (ForecastCalibration, bool) {
	key := acceleratorKey(profile.manufacturer, profile.accelerator)
	for _, calibration := range calibrations {
		if calibration.Schema == forecastCalibrationSchema && calibration.GPUs == GPUs && calibration.EffectiveTFLOPS > 0 && acceleratorKey(calibration.Manufacturer, calibration.Accelerator) == key {
			return calibration, true
		}
	}
	return ForecastCalibration{}, false
}

func requiredMemoryPerGPU(plan Plan, GPUs int) (uint64, error) {
	if GPUs <= 0 {
		return 0, fmt.Errorf("GPU count must be positive")
	}
	// BF16 parameters and gradients plus FP32 master weights and two Adam
	// moments require approximately 16 bytes per parameter. Model state is
	// assumed fully sharded. The built-in distributed worker partitions the
	// declared global batch across ranks, so activation and logits workspace is
	// estimated from the per-rank batch. Round up here to remain conservative
	// for forecasts even though execution currently requires exact divisibility.
	// The loss retains BF16 logits and an FP32 conversion across the local batch.
	states, err := multiply(plan.Forecast.ApproximateParameters, 16)
	if err != nil {
		return 0, err
	}
	states = divideRoundUp(states, uint64(GPUs))
	var maxActivations uint64
	for _, stage := range plan.Stages {
		batch := divideRoundUp(uint64(stage.Parameters.BatchSize), uint64(GPUs))
		activations, err := multiplyAll(batch, uint64(stage.Parameters.SequenceLength), plan.Architecture.HiddenSize, plan.Architecture.Layers, 72)
		if err != nil {
			return 0, fmt.Errorf("stage %s activation memory: %w", stage.Name, err)
		}
		logits, err := multiplyAll(batch, uint64(stage.Parameters.SequenceLength), plan.Architecture.VocabularySize, 6)
		if err != nil {
			return 0, fmt.Errorf("stage %s logits memory: %w", stage.Name, err)
		}
		activations, err = add(activations, logits)
		if err != nil {
			return 0, fmt.Errorf("stage %s training workspace memory: %w", stage.Name, err)
		}
		if activations > maxActivations {
			maxActivations = activations
		}
	}
	required, err := add(states, maxActivations, 4<<30)
	if err != nil {
		return 0, err
	}
	return required, nil
}

func multiplyAll(values ...uint64) (uint64, error) {
	result := uint64(1)
	for _, value := range values {
		var err error
		result, err = multiply(result, value)
		if err != nil {
			return 0, err
		}
	}
	return result, nil
}

func divideRoundUp(value, divisor uint64) uint64 {
	return value/divisor + boolToUint64(value%divisor != 0)
}

func boolToUint64(value bool) uint64 {
	if value {
		return 1
	}
	return 0
}

func scaling(profile acceleratorProfile, GPUs int) float64 {
	switch GPUs {
	case 2:
		return profile.scale2
	case 4:
		return profile.scale4
	case 8:
		return profile.scale8
	default:
		return 1
	}
}

func nodesFor(profile acceleratorProfile, GPUs int) int {
	if profile.gpusPerNode <= 0 {
		return 1
	}
	return int(divideRoundUp(uint64(GPUs), uint64(profile.gpusPerNode)))
}
