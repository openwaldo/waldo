// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"math"

	"github.com/openwaldo/waldo/internal/training"
)

const fitnessSchema = 1

// FitnessAnalysis is the compose-only structural analysis. Corpus-dependent
// measurements are deliberately marked unavailable until preflight has read
// the selected immutable corpus revision with the selected tokenizer.
type FitnessAnalysis struct {
	Schema       int                     `json:"schema"`
	Architecture ArchitectureFitness     `json:"architecture"`
	Context      ContextFitness          `json:"tokenizer_context"`
	Stages       []StageFitness          `json:"stages"`
	Compute      ComputeFitness          `json:"compute"`
	Memory       MemoryFitness           `json:"memory"`
	Warnings     []FitnessWarning        `json:"warnings,omitempty"`
	Prediction   LossPredictionReadiness `json:"loss_prediction"`
}

type ArchitectureFitness struct {
	VocabularySize                    uint64  `json:"vocabulary_size"`
	TiedEmbeddings                    bool    `json:"tied_embeddings"`
	TokenEmbeddingParameters          uint64  `json:"token_embedding_parameters"`
	OutputProjectionParameters        uint64  `json:"output_projection_parameters"`
	TokenIOParameters                 uint64  `json:"token_io_parameters"`
	PositionalEmbeddingParameters     uint64  `json:"positional_embedding_parameters"`
	QueryParametersPerLayer           uint64  `json:"query_parameters_per_layer"`
	KeyParametersPerLayer             uint64  `json:"key_parameters_per_layer"`
	ValueParametersPerLayer           uint64  `json:"value_parameters_per_layer"`
	AttentionOutputParametersPerLayer uint64  `json:"attention_output_parameters_per_layer"`
	AttentionParameters               uint64  `json:"attention_parameters"`
	MLPGateParametersPerLayer         uint64  `json:"mlp_gate_parameters_per_layer"`
	MLPUpParametersPerLayer           uint64  `json:"mlp_up_parameters_per_layer"`
	MLPDownParametersPerLayer         uint64  `json:"mlp_down_parameters_per_layer"`
	MLPParameters                     uint64  `json:"mlp_parameters"`
	NormalizationParameters           uint64  `json:"normalization_parameters"`
	BiasParameters                    uint64  `json:"bias_parameters"`
	OtherParameters                   uint64  `json:"other_parameters"`
	TotalParameters                   uint64  `json:"total_parameters"`
	NonEmbeddingParameters            uint64  `json:"non_embedding_parameters"`
	TokenIOPercent                    float64 `json:"token_io_percent"`
	AttentionPercent                  float64 `json:"attention_percent"`
	MLPPercent                        float64 `json:"mlp_percent"`
	EmbeddingPercent                  float64 `json:"embedding_percent"`
	OtherPercent                      float64 `json:"other_percent"`
	GQAQueryHeads                     uint64  `json:"gqa_query_heads"`
	GQAKeyValueHeads                  uint64  `json:"gqa_key_value_heads"`
	GQAKVWidth                        uint64  `json:"gqa_kv_width"`
	Formula                           string  `json:"formula"`
}

type ContextFitness struct {
	TokenizerName                    string            `json:"tokenizer_name"`
	TokenizerRevision                string            `json:"tokenizer_revision"`
	VocabularySize                   uint64            `json:"vocabulary_size"`
	ContextTokens                    uint64            `json:"context_tokens"`
	ExpectedUntrainedTokenLoss       float64           `json:"expected_untrained_token_loss"`
	MeasuredBytesPerToken            *float64          `json:"measured_bytes_per_token"`
	EffectiveContextBytes            *float64          `json:"effective_context_bytes"`
	EffectiveContextCharacters       *float64          `json:"effective_context_characters"`
	RecordLengthP50Tokens            *float64          `json:"record_length_p50_tokens"`
	RecordLengthP90Tokens            *float64          `json:"record_length_p90_tokens"`
	RecordLengthP95Tokens            *float64          `json:"record_length_p95_tokens"`
	RecordLengthP50Bytes             *float64          `json:"record_length_p50_bytes"`
	RecordLengthP90Bytes             *float64          `json:"record_length_p90_bytes"`
	RecordLengthP95Bytes             *float64          `json:"record_length_p95_bytes"`
	RecordsFitOneContextPercent      *float64          `json:"records_fit_one_context_percent"`
	ExpectedPackedDocumentBoundaries *float64          `json:"expected_packed_document_boundaries"`
	ExpectedEOSTargets               *float64          `json:"expected_eos_targets"`
	CorpusFertility                  []CorpusFertility `json:"corpus_fertility"`
	MeasurementStatus                string            `json:"measurement_status"`
	MeasurementReason                string            `json:"measurement_reason,omitempty"`
}

type CorpusFertility struct {
	Corpus             string  `json:"corpus"`
	Tokens             int64   `json:"tokens"`
	UTF8Bytes          int64   `json:"utf8_bytes"`
	BytesPerToken      float64 `json:"bytes_per_token"`
	TokensPerCharacter float64 `json:"tokens_per_character"`
}

type StageFitness struct {
	Name                           string          `json:"name"`
	RequestedTokens                int64           `json:"requested_tokens,omitempty"`
	PlannedTokens                  int64           `json:"planned_tokens,omitempty"`
	GlobalTokensPerOptimizerStep   int64           `json:"global_tokens_per_optimizer_step"`
	OptimizerSteps                 int64           `json:"optimizer_steps"`
	TokensPerTotalParameter        float64         `json:"tokens_per_total_parameter,omitempty"`
	TokensPerNonEmbeddingParameter float64         `json:"tokens_per_non_embedding_parameter,omitempty"`
	UniqueAvailableTrainingTokens  *int64          `json:"unique_available_training_tokens"`
	ExpectedEffectiveCorpusPasses  *float64        `json:"expected_effective_corpus_passes"`
	RecordsEncountered             *float64        `json:"records_encountered"`
	EOSBoundariesEncountered       *float64        `json:"eos_boundaries_encountered"`
	WarmupPercent                  float64         `json:"warmup_percent"`
	WarmdownPercent                float64         `json:"warmdown_percent"`
	Topologies                     []BatchTopology `json:"world_size_topologies"`
	CorpusMeasurementStatus        string          `json:"corpus_measurement_status"`
	CorpusMeasurementReason        string          `json:"corpus_measurement_reason"`
}

type BatchTopology struct {
	WorldSize                    int64 `json:"world_size"`
	GlobalBatchSequences         int64 `json:"global_batch_sequences"`
	GlobalMicroBatchSequences    int64 `json:"global_micro_batch_sequences"`
	RankLocalMicroBatchSequences int64 `json:"rank_local_micro_batch_sequences"`
	PreservesGlobalBatch         bool  `json:"preserves_global_batch"`
	PreservesSampleOrder         bool  `json:"preserves_sample_order"`
}

type ComputeFitness struct {
	ConventionalTrainingFLOPs      float64 `json:"conventional_training_flops"`
	ConventionalFormula            string  `json:"conventional_formula"`
	ArchitectureAwareFLOPsPerToken float64 `json:"architecture_aware_flops_per_token"`
	ArchitectureAwareTrainingFLOPs float64 `json:"architecture_aware_training_flops"`
	ArchitectureAwareFormula       string  `json:"architecture_aware_formula"`
	ApproximationNote              string  `json:"approximation_note"`
}

type MemoryFitness struct {
	ParameterBytes           uint64 `json:"parameter_bytes"`
	GradientBytes            uint64 `json:"gradient_bytes"`
	OptimizerStateBytes      uint64 `json:"optimizer_state_bytes"`
	MasterWeightBytes        uint64 `json:"master_weight_bytes"`
	CheckpointBytes          uint64 `json:"checkpoint_bytes"`
	ActivationAndLogitsBytes uint64 `json:"activation_and_logits_bytes"`
	ActivationStatus         string `json:"activation_status"`
	ApproximationNote        string `json:"approximation_note"`
}

type FitnessWarning struct {
	Code      string `json:"code"`
	Severity  string `json:"severity"`
	Message   string `json:"message"`
	Threshold string `json:"threshold,omitempty"`
	Evidence  string `json:"evidence,omitempty"`
}

type LossPredictionReadiness struct {
	Status                string   `json:"status"`
	ComparableSamples     int      `json:"comparable_samples"`
	MinimumSamples        int      `json:"minimum_samples"`
	Reason                string   `json:"reason"`
	RequiredComparability []string `json:"required_comparability"`
}

func analyzeFitness(plan Plan, plannedTokens int64, conventionalFLOPs float64) (FitnessAnalysis, error) {
	a, err := architectureFitness(plan.Architecture, plan.Forecast.ApproximateParameters)
	if err != nil {
		return FitnessAnalysis{}, err
	}
	context := ContextFitness{
		TokenizerName: plan.Architecture.Tokenizer.Name, TokenizerRevision: plan.Architecture.Tokenizer.Revision,
		VocabularySize: plan.Architecture.VocabularySize, ContextTokens: plan.Architecture.ContextTokens,
		ExpectedUntrainedTokenLoss: math.Log(float64(plan.Architecture.VocabularySize)),
		MeasurementStatus:          "requires-corpus-preflight",
		MeasurementReason:          "bytes per token, record-length percentiles, fit rate, packed boundaries, EOS targets, and per-corpus fertility require tokenizing the selected immutable corpus revision",
		CorpusFertility:            []CorpusFertility{},
	}
	if plan.Architecture.Tokenizer.Name == "byte" && plan.Architecture.Tokenizer.Revision == training.ByteTokenizerRevision {
		value := float64(plan.Architecture.ContextTokens)
		one := float64(1)
		context.MeasuredBytesPerToken = &one
		context.EffectiveContextBytes = &value
		context.EffectiveContextCharacters = &value
		context.MeasurementStatus = "exact-bytes-ascii-character-upper-bound"
		context.MeasurementReason = "the byte tokenizer uses one token per UTF-8 byte; character count equals bytes only for ASCII and is otherwise lower"
	}
	stages := make([]StageFitness, 0, len(plan.Stages))
	warnings := architectureWarnings(a)
	architectureAwareTotal := float64(0)
	for _, stage := range plan.Stages {
		resolved, err := training.ResolvePlanningParameters(stage.Parameters)
		if err != nil {
			return FitnessAnalysis{}, fmt.Errorf("stage %s fitness parameters: %w", stage.Name, err)
		}
		stageFitness := StageFitness{
			Name: stage.Name, RequestedTokens: resolved.RequestedTokens, PlannedTokens: stage.PlannedTokens,
			GlobalTokensPerOptimizerStep: resolved.BatchSize * resolved.SequenceLength, OptimizerSteps: resolved.Steps,
			WarmupPercent:           percent(uint64(resolved.Schedule.WarmupSteps), uint64(resolved.Steps)),
			WarmdownPercent:         percent(uint64(resolved.Schedule.WarmdownSteps), uint64(resolved.Steps)),
			CorpusMeasurementStatus: "requires-corpus-preflight",
			CorpusMeasurementReason: "unique tokenizer tokens, corpus passes, records, EOS boundaries, and record-length distributions are not present in a portable compose",
		}
		if stageFitness.RequestedTokens == 0 && stage.Parameters.Epochs == 0 {
			stageFitness.RequestedTokens = stage.PlannedTokens
		}
		if stage.PlannedTokens > 0 {
			stageFitness.TokensPerTotalParameter = float64(stage.PlannedTokens) / float64(a.TotalParameters)
			stageFitness.TokensPerNonEmbeddingParameter = float64(stage.PlannedTokens) / float64(a.NonEmbeddingParameters)
			architectureAwareTotal += architectureTrainingFLOPsPerToken(plan.Architecture, uint64(resolved.SequenceLength)) * float64(stage.PlannedTokens)
		}
		globalMicro := resolved.BatchSize / resolved.GradientAccumulation
		for _, world := range []int64{1, 2, 4, 8} {
			if globalMicro%world != 0 {
				continue
			}
			stageFitness.Topologies = append(stageFitness.Topologies, BatchTopology{
				WorldSize: world, GlobalBatchSequences: resolved.BatchSize, GlobalMicroBatchSequences: globalMicro,
				RankLocalMicroBatchSequences: globalMicro / world, PreservesGlobalBatch: true, PreservesSampleOrder: true,
			})
		}
		if resolved.Schedule.WarmupSteps*5 > resolved.Steps {
			warnings = append(warnings, FitnessWarning{Code: "warmup-large-fraction", Severity: "warning", Message: fmt.Sprintf("stage %s warmup consumes %.1f%% of optimizer steps", stage.Name, stageFitness.WarmupPercent), Threshold: "warn above 20%"})
		}
		if resolved.BatchSize*resolved.SequenceLength*10 > resolved.PlannedTokenCapacity {
			warnings = append(warnings, FitnessWarning{Code: "short-run-global-batch", Severity: "warning", Message: fmt.Sprintf("stage %s has fewer than 10 global optimizer batches", stage.Name), Threshold: "warn below 10 optimizer steps"})
		}
		stages = append(stages, stageFitness)
	}
	if context.EffectiveContextBytes != nil && *context.EffectiveContextBytes <= 256 {
		warnings = append(warnings, FitnessWarning{Code: "short-byte-context", Severity: "warning", Message: fmt.Sprintf("the byte tokenizer provides only %.0f UTF-8 bytes of context; a prompt plus 150 generated bytes cannot retain more than %.0f prompt bytes", *context.EffectiveContextBytes, math.Max(0, *context.EffectiveContextBytes-150)), Evidence: "one byte token represents exactly one UTF-8 byte"})
	}
	awarePerToken := architectureTrainingFLOPsPerToken(plan.Architecture, plan.Architecture.ContextTokens)
	if plannedTokens > 0 {
		awarePerToken = architectureAwareTotal / float64(plannedTokens)
	}
	parameterBytes := plan.Forecast.ParameterBytes
	gradientBytes := plan.Forecast.ApproximateParameters * 4
	masterBytes := uint64(0)
	if plan.Architecture.ParameterDType != "float32" {
		gradientBytes = plan.Forecast.ApproximateParameters * 2
		masterBytes = plan.Forecast.ApproximateParameters * 4
	}
	optimizerBytes := plan.Forecast.ApproximateParameters * 8
	activationBytes, err := maximumActivationWorkspace(plan, 1)
	if err != nil {
		return FitnessAnalysis{}, err
	}
	return FitnessAnalysis{
		Schema: fitnessSchema, Architecture: a, Context: context, Stages: stages,
		Compute: ComputeFitness{
			ConventionalTrainingFLOPs: conventionalFLOPs, ConventionalFormula: "approximately 6 * total_parameters * planned_tokens",
			ArchitectureAwareFLOPsPerToken: awarePerToken, ArchitectureAwareTrainingFLOPs: architectureAwareTotal,
			ArchitectureAwareFormula: "3 * forward FLOPs: GQA Q/K/V/O projections + SwiGLU gate/up/down + causal attention score/value work + vocabulary projection; embedding lookup excluded",
			ApproximationNote:        "FLOP counts assume multiply-add is two FLOPs, backward is approximately twice forward, full declared sequence length, and omit elementwise operations, normalization, RoPE, optimizer updates, recomputation, padding, and hardware effects",
		},
		Memory: MemoryFitness{
			ParameterBytes: parameterBytes, GradientBytes: gradientBytes, OptimizerStateBytes: optimizerBytes, MasterWeightBytes: masterBytes,
			CheckpointBytes:          parameterBytes + optimizerBytes + masterBytes,
			ActivationAndLogitsBytes: activationBytes,
			ActivationStatus:         "reported per hardware configuration as part of required_per_gpu_bytes",
			ApproximationNote:        "Adam moments use FP32; gradients use parameter precision; lower-precision training retains FP32 master weights; checkpoint estimate excludes framework and RNG metadata",
		},
		Warnings: warnings,
		Prediction: LossPredictionReadiness{
			Status: "refused-insufficient-evidence", ComparableSamples: 0, MinimumSamples: 12,
			Reason:                "a compose contains no verified set of comparable completed runs from which to fit and validate a scaling curve",
			RequiredComparability: []string{"corpus revision and held-out set", "tokenizer", "architecture family", "context and objective", "optimizer recipe"},
		},
	}, nil
}

func architectureFitness(architecture Architecture, total uint64) (ArchitectureFitness, error) {
	h := architecture.HiddenSize
	headWidth := h / architecture.AttentionHeads
	kvWidth := headWidth * architecture.KeyValueHeads
	embedding := architecture.VocabularySize * h
	output := uint64(0)
	if !architecture.TieEmbeddings {
		output = embedding
	}
	q, k, v, o := h*h, h*kvWidth, h*kvWidth, h*h
	attention := architecture.Layers * (q + k + v + o)
	gate, up, down := h*architecture.IntermediateSize, h*architecture.IntermediateSize, architecture.IntermediateSize*h
	mlp := architecture.Layers * (gate + up + down)
	norms := (2*architecture.Layers + 1) * h
	tokenIO := embedding + output
	calculated := tokenIO + attention + mlp + norms
	if calculated != total {
		return ArchitectureFitness{}, fmt.Errorf("architecture decomposition %d does not match WALDO forecast %d", calculated, total)
	}
	return ArchitectureFitness{
		VocabularySize: architecture.VocabularySize, TiedEmbeddings: architecture.TieEmbeddings,
		TokenEmbeddingParameters: embedding, OutputProjectionParameters: output, TokenIOParameters: tokenIO,
		PositionalEmbeddingParameters: 0, QueryParametersPerLayer: q, KeyParametersPerLayer: k, ValueParametersPerLayer: v,
		AttentionOutputParametersPerLayer: o, AttentionParameters: attention,
		MLPGateParametersPerLayer: gate, MLPUpParametersPerLayer: up, MLPDownParametersPerLayer: down, MLPParameters: mlp,
		NormalizationParameters: norms, BiasParameters: 0, OtherParameters: norms,
		TotalParameters: total, NonEmbeddingParameters: total - tokenIO,
		TokenIOPercent: percent(tokenIO, total), AttentionPercent: percent(attention, total), MLPPercent: percent(mlp, total),
		EmbeddingPercent: percent(tokenIO, total), OtherPercent: percent(norms, total),
		GQAQueryHeads: architecture.AttentionHeads, GQAKeyValueHeads: architecture.KeyValueHeads, GQAKVWidth: kvWidth,
		Formula: "token embedding + optional untied output + layers * (Q + K + V + O + SwiGLU gate + up + down) + two RMSNorm vectors per layer + final RMSNorm; RoPE, QK normalization, and all projections are parameter-free/bias-free where applicable",
	}, nil
}

// AnalyzeArchitecture exposes the exact structural decomposition used by the
// compose forecast to text renderers and other read-only clients.
func AnalyzeArchitecture(architecture Architecture) (ArchitectureFitness, error) {
	forecast, err := architecture.Forecast()
	if err != nil {
		return ArchitectureFitness{}, err
	}
	return architectureFitness(architecture, forecast.ApproximateParameters)
}

func architectureTrainingFLOPsPerToken(a Architecture, sequenceLength uint64) float64 {
	h, i, s, v := float64(a.HiddenSize), float64(a.IntermediateSize), float64(sequenceLength), float64(a.VocabularySize)
	kv := h / float64(a.AttentionHeads) * float64(a.KeyValueHeads)
	forwardLayer := 4*h*h + 4*h*kv + 6*h*i + 4*h*s
	return 3 * (float64(a.Layers)*forwardLayer + 2*h*v)
}

func architectureWarnings(a ArchitectureFitness) []FitnessWarning {
	if a.TokenIOPercent > 50 {
		return []FitnessWarning{{Code: "token-io-majority", Severity: "warning", Message: fmt.Sprintf("token input/output weights consume %.1f%% of parameters", a.TokenIOPercent), Threshold: "warn above 50%", Evidence: "structural capacity-allocation diagnostic; not a capability threshold"}}
	}
	if a.TokenIOPercent > 20 {
		return []FitnessWarning{{Code: "token-io-high", Severity: "advisory", Message: fmt.Sprintf("token input/output weights consume %.1f%% of parameters", a.TokenIOPercent), Threshold: "advise above 20%", Evidence: "configurable design-review heuristic; not a capability threshold"}}
	}
	return nil
}

func percent(part, total uint64) float64 {
	if total == 0 {
		return 0
	}
	return 100 * float64(part) / float64(total)
}
