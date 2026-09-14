// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"fmt"
	"math"
)

const (
	ShuffledProfile           = "causal-pretrain-shuffled"
	BalancedProfile           = "causal-pretrain-balanced"
	WeightedProfile           = "causal-pretrain-weighted"
	DefaultProfile            = ShuffledProfile
	ProfileSchema             = 1
	defaultShuffleBufferBytes = int64(64 * 1024 * 1024)
)

const (
	legacyShuffledProfile = "causal-pretrain-v1"
	legacyBalancedProfile = "causal-pretrain-v2"
	legacyWeightedProfile = "causal-pretrain-v3"
)

func ResolveParameters(parameters Parameters) (ResolvedParameters, error) {
	steps, requestedTokens, err := resolveTrainingBudget(parameters)
	if err != nil {
		return ResolvedParameters{}, err
	}
	if steps == 0 {
		return ResolvedParameters{}, fmt.Errorf("epoch-derived training steps have not been resolved")
	}
	return resolveParameters(parameters, steps, requestedTokens)
}

// ResolvePlanningParameters validates parameters needed to select and scan a
// training stream before an epoch-derived step count is known.
func ResolvePlanningParameters(parameters Parameters) (ResolvedParameters, error) {
	steps, requestedTokens, err := resolveTrainingBudget(parameters)
	if err != nil {
		return ResolvedParameters{}, err
	}
	if steps == 0 {
		steps = max(int64(1), optionalInt64(parameters.WarmupSteps), optionalInt64(parameters.CheckpointEvery), optionalInt64(parameters.EvaluateEvery))
	}
	return resolveParameters(parameters, steps, requestedTokens)
}

// ResolveParametersForSteps pins the step count derived from a finite epoch
// stream. The declarative parameters remain epoch-driven.
func ResolveParametersForSteps(parameters Parameters, steps int64) (ResolvedParameters, error) {
	declaredSteps, requestedTokens, err := resolveTrainingBudget(parameters)
	if err != nil {
		return ResolvedParameters{}, err
	}
	if declaredSteps != 0 || requestedTokens != 0 {
		return ResolvedParameters{}, fmt.Errorf("derived steps require an epoch-only training budget")
	}
	if steps <= 0 {
		return ResolvedParameters{}, fmt.Errorf("derived steps must be positive")
	}
	return resolveParameters(parameters, steps, 0)
}

func resolveTrainingBudget(parameters Parameters) (int64, int64, error) {
	if parameters.Steps < 0 || parameters.Tokens < 0 {
		return 0, 0, fmt.Errorf("steps and tokens must not be negative")
	}
	if parameters.Tokens > 0 {
		if parameters.Steps > 0 || parameters.Epochs > 0 {
			return 0, 0, fmt.Errorf("tokens cannot be combined with steps or epochs")
		}
		capacity, overflow := multiplyInt64(parameters.BatchSize, parameters.SequenceLength)
		if overflow {
			return 0, 0, fmt.Errorf("training step token capacity overflows int64")
		}
		steps := parameters.Tokens / capacity
		if parameters.Tokens%capacity != 0 {
			steps++
		}
		return steps, parameters.Tokens, nil
	}
	if parameters.Steps > 0 {
		return parameters.Steps, 0, nil
	}
	if parameters.Epochs > 0 {
		return 0, 0, nil
	}
	return 0, 0, fmt.Errorf("one of tokens, epochs, or steps is required")
}

func optionalInt64(value *int64) int64 {
	if value == nil {
		return 0
	}
	return *value
}

func resolveParameters(parameters Parameters, steps, requestedTokens int64) (ResolvedParameters, error) {
	if err := ValidateParallelismRequest(parameters.Parallelism); err != nil {
		return ResolvedParameters{}, err
	}
	profile := parameters.Profile
	if profile == "" {
		profile = DefaultProfile
	}
	profile = CanonicalProfile(profile)
	if profile != DefaultProfile && profile != BalancedProfile && profile != WeightedProfile {
		return ResolvedParameters{}, fmt.Errorf("unsupported training profile %q", profile)
	}
	if steps <= 0 || parameters.BatchSize <= 0 || parameters.SequenceLength <= 0 || parameters.LearningRate <= 0 || math.IsNaN(parameters.LearningRate) || math.IsInf(parameters.LearningRate, 0) {
		return ResolvedParameters{}, fmt.Errorf("steps, batch_size, sequence_length, and learning_rate must be finite and positive")
	}
	optimizerName := parameters.Optimizer
	if optimizerName == "" {
		optimizerName = "adamw"
	}
	if optimizerName != "adamw" && optimizerName != "muon-adamw" {
		return ResolvedParameters{}, fmt.Errorf("unsupported optimizer %q", optimizerName)
	}
	scheduleName := parameters.Schedule
	if scheduleName == "" {
		scheduleName = "cosine"
	}
	if scheduleName != "cosine" && scheduleName != "warmup-stable-warmdown" {
		return ResolvedParameters{}, fmt.Errorf("unsupported learning-rate schedule %q", scheduleName)
	}
	gradientAccumulation := parameters.GradientAccumulation
	if gradientAccumulation == 0 {
		gradientAccumulation = 1
	}
	if gradientAccumulation < 1 || gradientAccumulation > parameters.BatchSize || parameters.BatchSize%gradientAccumulation != 0 {
		return ResolvedParameters{}, fmt.Errorf("gradient_accumulation_steps must be positive, no greater than batch_size, and divide batch_size exactly")
	}
	computePrecision := parameters.ComputePrecision
	if computePrecision == "" {
		computePrecision = "auto"
	}
	if computePrecision != "auto" && computePrecision != "float32" && computePrecision != "float16" && computePrecision != "bfloat16" {
		return ResolvedParameters{}, fmt.Errorf("compute_precision must be auto, float32, float16, or bfloat16")
	}
	if parameters.DistributionPolicy != "" && parameters.DistributionPolicy != "distributable" {
		return ResolvedParameters{}, fmt.Errorf("distribution_policy must be distributable when set")
	}
	epochs := parameters.Epochs
	if epochs == 0 {
		epochs = 1
	}
	if epochs < 1 || epochs > 1_000_000 {
		return ResolvedParameters{}, fmt.Errorf("epochs must be in 1..1000000")
	}
	capacity, overflow := multiplyInt64(steps, parameters.BatchSize, parameters.SequenceLength)
	if overflow {
		return ResolvedParameters{}, fmt.Errorf("planned token capacity overflows int64")
	}
	weightDecay := 0.1
	if parameters.WeightDecay != nil {
		weightDecay = *parameters.WeightDecay
	}
	if weightDecay < 0 || weightDecay > 1 || math.IsNaN(weightDecay) || math.IsInf(weightDecay, 0) {
		return ResolvedParameters{}, fmt.Errorf("weight_decay must be finite and in 0..1")
	}
	warmup := min(int64(100), steps/10)
	if steps > 1 && warmup == 0 {
		warmup = 1
	}
	if parameters.WarmupSteps != nil {
		warmup = *parameters.WarmupSteps
	}
	checkpointEvery := min(int64(500), steps)
	if parameters.CheckpointEvery != nil {
		checkpointEvery = *parameters.CheckpointEvery
	}
	evaluateEvery := min(int64(500), steps)
	if parameters.EvaluateEvery != nil {
		evaluateEvery = *parameters.EvaluateEvery
	}
	if warmup < 0 || warmup > steps {
		return ResolvedParameters{}, fmt.Errorf("warmup_steps must be in 0..steps")
	}
	warmdown := int64(0)
	if parameters.WarmdownSteps != nil {
		warmdown = *parameters.WarmdownSteps
	} else if scheduleName == "warmup-stable-warmdown" {
		warmdown = (steps - warmup) / 2
	}
	if warmdown < 0 || warmdown > steps || warmup+warmdown > steps {
		return ResolvedParameters{}, fmt.Errorf("warmup_steps plus warmdown_steps must fit within training steps")
	}
	minimumRateRatio := 0.1
	if parameters.MinimumRateRatio != nil {
		minimumRateRatio = *parameters.MinimumRateRatio
	} else if scheduleName == "warmup-stable-warmdown" {
		minimumRateRatio = 0
	}
	if minimumRateRatio < 0 || minimumRateRatio > 1 || math.IsNaN(minimumRateRatio) || math.IsInf(minimumRateRatio, 0) {
		return ResolvedParameters{}, fmt.Errorf("minimum_learning_rate_ratio must be finite and in 0..1")
	}
	if checkpointEvery < 0 || checkpointEvery > steps {
		return ResolvedParameters{}, fmt.Errorf("checkpoint_every must be in 0..steps")
	}
	if evaluateEvery < 0 || evaluateEvery > steps {
		return ResolvedParameters{}, fmt.Errorf("evaluate_every must be in 0..steps")
	}
	shuffleBuffer := 1024
	if parameters.ShuffleBufferRecords != nil {
		shuffleBuffer = *parameters.ShuffleBufferRecords
	}
	if shuffleBuffer < 1 || shuffleBuffer > 1_000_000 {
		return ResolvedParameters{}, fmt.Errorf("shuffle_buffer_records must be in 1..1000000")
	}
	shuffleBufferBytes := defaultShuffleBufferBytes
	if parameters.ShuffleBufferBytes != nil {
		shuffleBufferBytes = *parameters.ShuffleBufferBytes
	}
	if shuffleBufferBytes < 1 || shuffleBufferBytes > 16*1024*1024*1024 {
		return ResolvedParameters{}, fmt.Errorf("shuffle_buffer_bytes must be in 1..17179869184")
	}
	evaluationFraction := 0.01
	if parameters.EvaluationFraction != nil {
		evaluationFraction = *parameters.EvaluationFraction
	}
	if evaluationFraction < 0 || evaluationFraction >= 1 || math.IsNaN(evaluationFraction) || math.IsInf(evaluationFraction, 0) {
		return ResolvedParameters{}, fmt.Errorf("evaluation_fraction must be finite and in 0..<1")
	}
	evaluationMaxRecords := 256
	if parameters.EvaluationMaxRecords != nil {
		evaluationMaxRecords = *parameters.EvaluationMaxRecords
	}
	if evaluationMaxRecords < 0 || evaluationMaxRecords > 1_000_000 {
		return ResolvedParameters{}, fmt.Errorf("evaluation_max_records must be in 0..1000000")
	}
	evaluationMaxBytes := int64(1 * 1024 * 1024)
	if parameters.EvaluationMaxBytes != nil {
		evaluationMaxBytes = *parameters.EvaluationMaxBytes
	}
	if evaluationMaxBytes < 0 || evaluationMaxBytes > 16*1024*1024*1024 {
		return ResolvedParameters{}, fmt.Errorf("evaluation_max_bytes must be in 0..17179869184")
	}
	if evaluationFraction == 0 || evaluationMaxRecords == 0 || evaluationMaxBytes == 0 {
		evaluationFraction, evaluationMaxRecords, evaluationMaxBytes = 0, 0, 0
	}
	order := "bounded-shuffle-v1"
	selection := "lowest-sha256-v1"
	profileSchema := ProfileSchema
	if profile == BalancedProfile {
		order = "corpus-balanced-shuffle-v1"
		selection = "stratified-lowest-sha256-v1"
	}
	var weights map[string]uint64
	if len(parameters.CorpusWeights) != 0 {
		weights = make(map[string]uint64, len(parameters.CorpusWeights))
	}
	for name, weight := range parameters.CorpusWeights {
		if name == "" || weight == 0 || weight > 1_000_000 {
			return ResolvedParameters{}, fmt.Errorf("corpus_weights must use non-empty corpus paths and weights in 1..1000000")
		}
		weights[name] = weight
	}
	if profile == WeightedProfile {
		if len(weights) == 0 {
			return ResolvedParameters{}, fmt.Errorf("training profile %q requires corpus_weights", WeightedProfile)
		}
		order = "corpus-weighted-shuffle-v1"
		selection = "stratified-lowest-sha256-v1"
	} else if len(weights) != 0 {
		return ResolvedParameters{}, fmt.Errorf("corpus_weights require training profile %q", WeightedProfile)
	}
	return ResolvedParameters{
		Profile: profile, ProfileSchema: profileSchema,
		Epochs: epochs, RequestedTokens: requestedTokens, Steps: steps, BatchSize: parameters.BatchSize, GradientAccumulation: gradientAccumulation,
		ComputePrecision: computePrecision, ActivationCheckpointing: parameters.ActivationCheckpointing, Compile: parameters.Compile,
		DistributionPolicy: parameters.DistributionPolicy,
		SequenceLength:     parameters.SequenceLength, LearningRate: parameters.LearningRate,
		Seed: parameters.Seed, PlannedTokenCapacity: capacity,
		Optimizer:       Optimizer{Name: optimizerName, WeightDecay: weightDecay, Beta1: 0.9, Beta2: 0.95, Epsilon: 1e-8},
		Schedule:        Schedule{Name: scheduleName, WarmupSteps: warmup, WarmdownSteps: warmdown, MinimumRateRatio: minimumRateRatio},
		Data:            DataPlan{Order: order, ShuffleBufferRecords: shuffleBuffer, ShuffleBufferBytes: shuffleBufferBytes, Packing: "continuous-eos-v1", CorpusWeights: weights},
		Evaluation:      &EvaluationPolicy{Selection: selection, Fraction: evaluationFraction, MaxRecords: evaluationMaxRecords, MaxBytes: evaluationMaxBytes},
		CheckpointEvery: checkpointEvery, EvaluateEvery: evaluateEvery,
	}, nil
}

// ValidateBatchTopology ensures every rank receives the same number of unique
// sequences in each physical micro-batch.
func ValidateBatchTopology(parameters ResolvedParameters, worldSize int) error {
	if worldSize < 1 {
		return fmt.Errorf("training world size must be positive")
	}
	if parameters.GradientAccumulation < 1 || parameters.BatchSize%parameters.GradientAccumulation != 0 {
		return fmt.Errorf("resolved gradient accumulation does not divide the global batch")
	}
	globalMicroBatch := parameters.BatchSize / parameters.GradientAccumulation
	if globalMicroBatch < int64(worldSize) || globalMicroBatch%int64(worldSize) != 0 {
		return fmt.Errorf("global micro-batch %d (batch_size %d / gradient_accumulation_steps %d) must be divisible by world size %d", globalMicroBatch, parameters.BatchSize, parameters.GradientAccumulation, worldSize)
	}
	return nil
}

// CanonicalProfile maps deprecated numbered profile names to their
// behavior-named identities. Unknown names are returned unchanged so callers
// can still report the original invalid value.
func CanonicalProfile(profile string) string {
	switch profile {
	case legacyShuffledProfile:
		return ShuffledProfile
	case legacyBalancedProfile:
		return BalancedProfile
	case legacyWeightedProfile:
		return WeightedProfile
	default:
		return profile
	}
}

// NormalizeResolvedParameters makes persisted numbered profiles comparable
// with their behavior-named replacements during checkpoint resume.
func NormalizeResolvedParameters(parameters ResolvedParameters) ResolvedParameters {
	original := parameters.Profile
	parameters.Profile = CanonicalProfile(original)
	if (original == legacyShuffledProfile && parameters.ProfileSchema == 1) ||
		(original == legacyBalancedProfile && parameters.ProfileSchema == 2) ||
		(original == legacyWeightedProfile && parameters.ProfileSchema == 3) {
		parameters.ProfileSchema = ProfileSchema
	}
	return parameters
}

func multiplyInt64(values ...int64) (int64, bool) {
	result := int64(1)
	for _, value := range values {
		if value <= 0 || result > math.MaxInt64/value {
			return 0, true
		}
		result *= value
	}
	return result, false
}
