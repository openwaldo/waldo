// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"math"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

func TestHistoricalArchitectureDecomposition(t *testing.T) {
	tests := []struct {
		name, tokenizer                  string
		vocabulary, hidden, intermediate uint64
		layers, heads, kvHeads, context  uint64
		total, tokenIO, core             uint64
	}{
		{"r50k-tiny", "tiktoken/r50k_base", 50259, 256, 768, 4, 4, 2, 512, 16_014_336, 12_866_304, 3_148_032},
		{"r50k-small", "tiktoken/r50k_base", 50259, 640, 1792, 10, 10, 2, 1024, 76_416_000, 32_165_760, 44_250_240},
		{"bpe-canary", "byte-bpe", 16000, 256, 768, 4, 4, 2, 512, 7_244_032, 4_096_000, 3_148_032},
		{"bpe-small", "byte-bpe", 16000, 640, 1792, 15, 10, 2, 1024, 76_615_040, 10_240_000, 66_375_040},
		{"tinystories-8m", "byte-bpe", 10000, 256, 768, 8, 8, 2, 512, 8_593_664, 2_560_000, 6_033_664},
		{"cosmopedia-general-32m", "byte-bpe", 10000, 512, 1536, 9, 8, 2, 512, 32_261_632, 5_120_000, 27_141_632},
		{"byte-reference", "byte", 259, 384, 1024, 6, 6, 6, 256, 10_721_280, 99_456, 10_621_824},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			revision := "trained"
			if test.tokenizer == "byte" {
				revision = training.ByteTokenizerRevision
			} else if test.tokenizer == "tiktoken/r50k_base" {
				revision = training.TiktokenR50KRevision
			}
			architecture := Architecture{
				Family: "decoder-transformer", ContextTokens: test.context, VocabularySize: test.vocabulary,
				HiddenSize: test.hidden, IntermediateSize: test.intermediate, Layers: test.layers,
				AttentionHeads: test.heads, KeyValueHeads: test.kvHeads, TieEmbeddings: true,
				ParameterDType: "float32", Tokenizer: Tokenizer{Name: test.tokenizer, Revision: revision},
			}
			forecast, err := architecture.Forecast()
			if err != nil {
				t.Fatal(err)
			}
			analysis, err := architectureFitness(architecture, forecast.ApproximateParameters)
			if err != nil {
				t.Fatal(err)
			}
			if analysis.TotalParameters != test.total || analysis.TokenIOParameters != test.tokenIO || analysis.NonEmbeddingParameters != test.core {
				t.Fatalf("decomposition = total %d token I/O %d core %d", analysis.TotalParameters, analysis.TokenIOParameters, analysis.NonEmbeddingParameters)
			}
			if analysis.AttentionParameters+analysis.MLPParameters+analysis.OtherParameters != analysis.NonEmbeddingParameters {
				t.Fatalf("core components do not sum: %+v", analysis)
			}
		})
	}
}

func TestFitnessWarningsAreAdvisoryAndThresholded(t *testing.T) {
	majority := ArchitectureFitness{TokenIOPercent: 80.3}
	warnings := architectureWarnings(majority)
	if len(warnings) != 1 || warnings[0].Code != "token-io-majority" || warnings[0].Severity != "warning" {
		t.Fatalf("majority warnings = %+v", warnings)
	}
	high := ArchitectureFitness{TokenIOPercent: 42.1}
	warnings = architectureWarnings(high)
	if len(warnings) != 1 || warnings[0].Code != "token-io-high" || warnings[0].Severity != "advisory" {
		t.Fatalf("high warnings = %+v", warnings)
	}
	if warnings := architectureWarnings(ArchitectureFitness{TokenIOPercent: 13.4}); len(warnings) != 0 {
		t.Fatalf("low-share warnings = %+v", warnings)
	}
}

func TestByteContextAndInitialLossSanity(t *testing.T) {
	compose := validCompose()
	compose.Architecture = Architecture{
		Family: "decoder-transformer", ContextTokens: 256, VocabularySize: 259, HiddenSize: 384,
		IntermediateSize: 1024, Layers: 6, AttentionHeads: 6, KeyValueHeads: 6, TieEmbeddings: true,
		ParameterDType: "float32", Tokenizer: Tokenizer{Name: "byte", Revision: training.ByteTokenizerRevision},
	}
	report, err := ForecastCompose(compose)
	if err != nil {
		t.Fatal(err)
	}
	context := report.Fitness.Context
	if context.EffectiveContextBytes == nil || *context.EffectiveContextBytes != 256 || math.Abs(context.ExpectedUntrainedTokenLoss-math.Log(259)) > 1e-12 {
		t.Fatalf("context fitness = %+v", context)
	}
	if report.Fitness.Prediction.Status != "refused-insufficient-evidence" || report.Fitness.Prediction.MinimumSamples < 12 {
		t.Fatalf("prediction readiness = %+v", report.Fitness.Prediction)
	}
}

func TestArchitectureAwareComputeIncludesContextAndVocabulary(t *testing.T) {
	architecture := Architecture{ContextTokens: 256, VocabularySize: 259, HiddenSize: 384, IntermediateSize: 1024, Layers: 6, AttentionHeads: 6, KeyValueHeads: 6}
	baseline := architectureTrainingFLOPsPerToken(architecture, 256)
	longer := architectureTrainingFLOPsPerToken(architecture, 512)
	architecture.VocabularySize = 16_000
	largerVocabulary := architectureTrainingFLOPsPerToken(architecture, 256)
	if longer <= baseline || largerVocabulary <= baseline {
		t.Fatalf("architecture-aware FLOPs baseline=%f longer=%f vocabulary=%f", baseline, longer, largerVocabulary)
	}
}
