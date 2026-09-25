// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/openwaldo/waldo/internal/training"
)

func TestForecastPlanFiltersNonFitsAndSortsSlowestFirst(t *testing.T) {
	recipe := validCompose()
	architecture, err := recipe.Architecture.Forecast()
	if err != nil {
		t.Fatal(err)
	}
	plan := Plan{
		Architecture: recipe.Architecture, Forecast: architecture,
		Stages: []PlannedStage{{Name: "pretrain", Parameters: recipe.Stages[0].Parameters, PlannedTokens: 10_000_000}},
	}
	report, err := forecastPlan(plan)
	if err != nil {
		t.Fatal(err)
	}
	if report.PlannedTokens != 10_000_000 || report.TrainingFLOPs <= 0 || len(report.Configurations) == 0 {
		t.Fatalf("report = %+v", report)
	}
	for position, configuration := range report.Configurations {
		if configuration.RequiredPerGPUBytes > configuration.MemoryPerGPUBytes-configuration.MemoryPerGPUBytes/10 {
			t.Fatalf("non-fitting configuration returned: %+v", configuration)
		}
		if position > 0 && report.Configurations[position-1].ApproximateSeconds < configuration.ApproximateSeconds {
			t.Fatalf("configurations are not slowest first: %+v", report.Configurations)
		}
	}
}

func TestForecastAccountsForPerRankBatchVocabularyWorkspace(t *testing.T) {
	architecture := Architecture{
		Family: "decoder-transformer", ContextTokens: 2048, VocabularySize: 50259,
		HiddenSize: 1152, IntermediateSize: 3072, Layers: 20,
		AttentionHeads: 16, KeyValueHeads: 4, TieEmbeddings: true,
		ParameterDType: "bfloat16", Tokenizer: Tokenizer{Name: "tiktoken/r50k_base", Revision: "tiktoken-r50k-base"},
	}
	forecast, err := architecture.Forecast()
	if err != nil {
		t.Fatal(err)
	}
	plan := Plan{Architecture: architecture, Forecast: forecast, Stages: []PlannedStage{{Name: "pretrain", Parameters: training.Parameters{BatchSize: 32, SequenceLength: 2048}}}}
	large, err := requiredMemoryPerGPU(plan, 1)
	if err != nil {
		t.Fatal(err)
	}
	plan.Stages[0].Parameters.BatchSize = 16
	safe, err := requiredMemoryPerGPU(plan, 1)
	if err != nil {
		t.Fatal(err)
	}
	h200Limit := uint64(141<<30) - uint64(141<<30)/10
	if large <= h200Limit || safe >= h200Limit || safe >= large {
		t.Fatalf("forecast memory batch32=%d batch16=%d H200 limit=%d", large, safe, h200Limit)
	}
	plan.Stages[0].Parameters.BatchSize = 16
	multiGPU, err := requiredMemoryPerGPU(plan, 8)
	if err != nil {
		t.Fatal(err)
	}
	if multiGPU >= safe/4 {
		t.Fatalf("distributed forecast did not partition model state and global batch: one=%d eight=%d", safe, multiGPU)
	}

	plan.Stages[0].Parameters.BatchSize = 9
	rounded, err := requiredMemoryPerGPU(plan, 8)
	if err != nil {
		t.Fatal(err)
	}
	plan.Stages[0].Parameters.BatchSize = 16
	exact, err := requiredMemoryPerGPU(plan, 8)
	if err != nil {
		t.Fatal(err)
	}
	if rounded != exact {
		t.Fatalf("per-rank batch must round up conservatively: batch9=%d batch16=%d", rounded, exact)
	}
}

func TestForecastUsesPhysicalMicroBatchForActivations(t *testing.T) {
	compose := validCompose()
	forecast, err := compose.Architecture.Forecast()
	if err != nil {
		t.Fatal(err)
	}
	parameters := compose.Stages[0].Parameters
	parameters.BatchSize = 8
	plan := Plan{Architecture: compose.Architecture, Forecast: forecast, Stages: []PlannedStage{{Name: "pretrain", Parameters: parameters}}}
	withoutAccumulation, err := requiredMemoryPerGPU(plan, 1)
	if err != nil {
		t.Fatal(err)
	}
	plan.Stages[0].Parameters.GradientAccumulation = 4
	withAccumulation, err := requiredMemoryPerGPU(plan, 1)
	if err != nil {
		t.Fatal(err)
	}
	if withAccumulation >= withoutAccumulation {
		t.Fatalf("gradient accumulation did not reduce activation memory: without=%d with=%d", withoutAccumulation, withAccumulation)
	}
}

func TestForecastAccountsForActivationCheckpointing(t *testing.T) {
	compose := validCompose()
	forecast, err := compose.Architecture.Forecast()
	if err != nil {
		t.Fatal(err)
	}
	parameters := compose.Stages[0].Parameters
	parameters.BatchSize = 8
	parameters.GradientAccumulation = 1
	plan := Plan{Architecture: compose.Architecture, Forecast: forecast, Stages: []PlannedStage{{Name: "pretrain", Parameters: parameters}}}
	without, err := requiredMemoryPerGPU(plan, 1)
	if err != nil {
		t.Fatal(err)
	}
	plan.Stages[0].Parameters.ActivationCheckpointing = true
	with, err := requiredMemoryPerGPU(plan, 1)
	if err != nil {
		t.Fatal(err)
	}
	if with >= without {
		t.Fatalf("checkpointed activation forecast %d must be below %d", with, without)
	}
}

func TestForecastReportsEpochDerivedStagesWithoutInventingTokens(t *testing.T) {
	compose := validCompose()
	compose.Stages[0].Parameters.Steps = 0
	compose.Stages[0].Parameters.Epochs = 2
	report, err := ForecastCompose(compose)
	if err != nil {
		t.Fatal(err)
	}
	if report.PlannedTokens != 0 || len(report.EpochDerivedStages) != 1 || report.EpochDerivedStages[0] != "pretrain" || len(report.Configurations) == 0 {
		t.Fatalf("epoch-derived forecast = %+v", report)
	}
	for _, configuration := range report.Configurations {
		if configuration.ApproximateSeconds != -1 {
			t.Fatalf("epoch-derived duration = %+v", configuration)
		}
	}
}

func TestForecastUsesExactObservedConfiguration(t *testing.T) {
	recipe := validCompose()
	architecture, err := recipe.Architecture.Forecast()
	if err != nil {
		t.Fatal(err)
	}
	plan := Plan{
		Architecture: recipe.Architecture, Forecast: architecture,
		Stages: []PlannedStage{{Name: "pretrain", Parameters: recipe.Stages[0].Parameters, PlannedTokens: 10_000_000}},
	}
	calibration := ForecastCalibration{Schema: 1, Manufacturer: "Apple", Accelerator: "Apple M4 Max", GPUs: 1, Runs: 2, EffectiveTFLOPS: 9}
	report, err := forecastPlanWithCalibration(plan, []ForecastCalibration{calibration})
	if err != nil {
		t.Fatal(err)
	}
	for _, configuration := range report.Configurations {
		if strings.Contains(configuration.Accelerator, "M4 Max") {
			if configuration.EstimateSource != "observed-runs" || configuration.ObservedRuns != 2 || configuration.EffectiveTFLOPS != 9 {
				t.Fatalf("M4 calibration not applied: %+v", configuration)
			}
			continue
		}
		if configuration.EstimateSource != "catalog" || configuration.ObservedRuns != 0 {
			t.Fatalf("calibration leaked to another topology: %+v", configuration)
		}
	}
}

func TestForecastCalibrationAggregatesRealActiveRunTime(t *testing.T) {
	started := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	inspection := Inspection{
		Model: ModelRecord{ID: "model", Forecast: ArchitectureForecast{ApproximateParameters: 1_000_000}},
		Runs: []RunRecord{{
			ID: "run", State: RunComplete,
			Observation: &training.Observation{ConsumedTokens: 2_000_000},
			Attempts: []RunAttempt{
				{Started: started.Format(time.RFC3339Nano), Finished: started.Add(10 * time.Second).Format(time.RFC3339Nano), State: RunInterrupted},
				{Started: started.Add(time.Minute).Format(time.RFC3339Nano), Finished: started.Add(70 * time.Second).Format(time.RFC3339Nano), State: RunComplete},
			},
		}},
		RunBOMs: []RunBOM{{Execution: training.Execution{Accelerators: []training.Accelerator{{Manufacturer: "NVIDIA", Model: "H100 SXM"}}}}},
	}
	evidence := forecastEvidence(inspection)
	calibrations, err := aggregateForecastEvidence(evidence)
	if err != nil {
		t.Fatal(err)
	}
	if len(calibrations) != 1 || calibrations[0].Runs != 1 || calibrations[0].ActiveSeconds != 20 || calibrations[0].EvidenceSHA256 == "" {
		t.Fatalf("calibrations = %+v", calibrations)
	}
	want := float64(6*1_000_000*2_000_000) / 20 / 1e12
	if math.Abs(calibrations[0].EffectiveTFLOPS-want) > 1e-12 {
		t.Fatalf("effective TFLOPS = %f, want %f", calibrations[0].EffectiveTFLOPS, want)
	}
	inspection.Runs[0].Observation.Simulated = true
	if got := forecastEvidence(inspection); len(got) != 0 {
		t.Fatalf("simulated evidence accepted: %+v", got)
	}
}

func TestLoadForecastCalibrationUsesVerifiedLocalModels(t *testing.T) {
	root := t.TempDir()
	clock := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	backend := backendFunc(func(_ context.Context, request training.Request) (training.Observation, error) {
		if err := os.MkdirAll(request.ArtifactDirectory, 0o755); err != nil {
			return training.Observation{}, err
		}
		data := []byte("observed model weights")
		path := filepath.Join(request.ArtifactDirectory, "model.safetensors")
		if err := os.WriteFile(path, data, 0o644); err != nil {
			return training.Observation{}, err
		}
		digest := sha256.Sum256(data)
		return training.Observation{
			Steps: 2, ConsumedTokens: 128,
			Evaluations: []training.Evaluation{{Step: 2, Tokens: 128, Metrics: map[string]float64{"heldout_loss": 1}}},
			Artifacts:   []training.Artifact{{Path: "artifacts/model.safetensors", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}},
		}, nil
	})
	builder := Builder{
		Root: root,
		Now: func() time.Time {
			clock = clock.Add(time.Second)
			return clock
		},
		NewID: func() (string, error) { return "observed0001", nil },
		Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
			selection := testSelection(backend)
			selection.Execution.Accelerators = []training.Accelerator{{Manufacturer: "Apple", Model: "Apple M4 Max", MemoryBytes: 128 << 30}}
			return selection, nil
		}),
	}
	if _, err := builder.Initialize("observed", testArchitecture()); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Train(context.Background(), "observed", preparedFixture(t, testStage("pretrain"))); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, "broken"), 0o755); err != nil {
		t.Fatal(err)
	}
	calibrations, err := LoadForecastCalibration(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(calibrations) != 1 || calibrations[0].Runs != 1 || calibrations[0].EffectiveTFLOPS <= 0 || calibrations[0].EvidenceSHA256 == "" {
		t.Fatalf("calibrations = %+v", calibrations)
	}
}

func TestForecastPlanReportsModelTooLargeForCatalog(t *testing.T) {
	architecture := Architecture{
		Family: "decoder-transformer", ContextTokens: 2048, VocabularySize: 32000,
		HiddenSize: 65_536, IntermediateSize: 262_144, Layers: 256,
		AttentionHeads: 256, KeyValueHeads: 32, ParameterDType: "bfloat16",
		Tokenizer: Tokenizer{Name: "test", Revision: "sha256:test"},
	}
	forecast, err := architecture.Forecast()
	if err != nil {
		t.Fatal(err)
	}
	report, err := forecastPlan(Plan{
		Architecture: architecture, Forecast: forecast,
		Stages: []PlannedStage{{Name: "pretrain", Parameters: validCompose().Stages[0].Parameters, PlannedTokens: 1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Configurations) != 0 || !strings.Contains(report.CatalogNote, "no configuration") || report.ApproximateParameters == 0 || report.PlannedTokens != 1 {
		t.Fatalf("report = %+v", report)
	}
}
