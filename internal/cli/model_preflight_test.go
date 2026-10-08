// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

type preflightFixtureSource []training.Record

func (source preflightFixtureSource) Stream(ctx context.Context, consume func(training.Record) error) error {
	for _, record := range source {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := consume(record); err != nil {
			return err
		}
	}
	return nil
}

func TestMeasureComposeRecordSourceReportsFertilityLengthsAndFit(t *testing.T) {
	_, codec, err := training.ResolveTokenizer("byte", training.ByteTokenizerRevision, 259)
	if err != nil {
		t.Fatal(err)
	}
	measurement, err := measureComposeRecordSource(context.Background(), preflightFixtureSource{
		{Text: "abc", Corpus: "alpha"},
		{Text: "é", Corpus: "alpha"},
		{Text: "12345", Corpus: "beta"},
	}, codec, 4)
	if err != nil {
		t.Fatal(err)
	}
	if measurement.Records != 3 || measurement.UTF8Bytes != 10 || measurement.Tokens != 10 || measurement.Characters != 9 {
		t.Fatalf("measurement = %+v", measurement)
	}
	if measurement.BytesPerToken != 1 || measurement.RecordP50Tokens != 3 || measurement.RecordP90Tokens != 5 || measurement.RecordP95Tokens != 5 {
		t.Fatalf("measurement = %+v", measurement)
	}
	if measurement.FitOneSequencePercent < 66.6 || measurement.FitOneSequencePercent > 66.7 || len(measurement.Corpora) != 2 {
		t.Fatalf("measurement = %+v", measurement)
	}
}

func TestApplyCorpusExposureUsesDeclaredWeights(t *testing.T) {
	corpora := []composeCorpusFertility{
		{Corpus: "alpha", Records: 10, Tokens: 90},
		{Corpus: "beta", Records: 20, Tokens: 180},
	}
	applyCorpusExposure(corpora, training.ResolvedParameters{Data: training.DataPlan{
		Order: "corpus-weighted-shuffle-v1", CorpusWeights: map[string]uint64{"alpha": 1, "beta": 3},
	}}, 400)
	if corpora[0].PlannedTargets != 100 || corpora[0].EffectivePasses != 1 || corpora[1].PlannedTargets != 300 || corpora[1].EffectivePasses != 1.5 {
		t.Fatalf("corpora = %+v", corpora)
	}
}

func TestComposeEpochTargetsPreservesContinuousPackingBoundary(t *testing.T) {
	targets, err := composeEpochTargets(99, 3)
	if err != nil || targets != 299 {
		t.Fatalf("targets = %d, error = %v", targets, err)
	}
}

func TestWriteComposePreflightMakesScopeExplicit(t *testing.T) {
	report := composePreflightReport{Stages: []composeStagePreflight{{
		Name: "example", SequenceTokens: 512, EffectiveContextBytes: 2048,
		Training:              composeCorpusMeasurement{Records: 10, UTF8Bytes: 1000, Tokens: 250, BytesPerToken: 4, FitOneSequencePercent: 100},
		Evaluation:            composeCorpusMeasurement{Records: 2, BytesPerToken: 4.1},
		UniqueTrainingTargets: 259, PlannedTrainingTokens: 518, EffectiveCorpusPasses: 2,
	}}}
	var output bytes.Buffer
	writeComposePreflight(&output, report)
	for _, want := range []string{"PREFLIGHT example", "4.000 bytes/token", "2.00 effective corpus passes", "not disjoint from tokenizer training"} {
		if !strings.Contains(output.String(), want) {
			t.Fatalf("output missing %q: %s", want, output.String())
		}
	}
}

func TestModelForecastPreflightMeasuresComposeWithoutTraining(t *testing.T) {
	seedMultiNodeCorpus(t)
	composePath := filepath.Join(t.TempDir(), "compose.yaml")
	compose := `kind: waldo-model-compose
schema: 1
architecture:
  family: decoder-transformer
  context_tokens: 64
  vocabulary_size: 259
  hidden_size: 64
  intermediate_size: 192
  layers: 2
  attention_heads: 4
  key_value_heads: 2
  tie_embeddings: true
  parameter_dtype: float32
  tokenizer:
    name: byte
    revision: builtin-byte-schema-1
stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    corpora:
      - books
    parameters:
      steps: 2
      batch_size: 1
      sequence_length: 64
      learning_rate: 0.001
      seed: 7
      evaluation_fraction: 0
`
	if err := os.WriteFile(composePath, []byte(compose), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"--json", "model", "forecast", composePath, "--preflight"}, &stdout, &stderr); code != 0 {
		t.Fatalf("code = %d, stderr = %q", code, stderr.String())
	}
	var output struct {
		Preflight composePreflightReport `json:"preflight"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatal(err)
	}
	if len(output.Preflight.Stages) != 1 {
		t.Fatalf("preflight = %+v", output.Preflight)
	}
	stage := output.Preflight.Stages[0]
	if stage.Training.Records != 1 || stage.Training.Tokens == 0 || stage.PlannedTrainingTokens != 128 || stage.EffectiveCorpusPasses <= 0 {
		t.Fatalf("stage = %+v", stage)
	}
}
