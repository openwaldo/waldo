// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	stdcontext "context"
	"fmt"
	"io"
	"math"
	"sort"
	"unicode/utf8"

	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

// composePreflightReport contains corpus measurements that cannot be derived
// from portable compose metadata. Running it reads and tokenizes records, but
// does not initialize a model or execute an optimizer step.
type composePreflightReport struct {
	Stages []composeStagePreflight `json:"stages"`
}

type composeStagePreflight struct {
	Name                        string                   `json:"name"`
	SequenceTokens              int64                    `json:"sequence_tokens"`
	EffectiveContextBytes       float64                  `json:"effective_context_bytes"`
	Training                    composeCorpusMeasurement `json:"training"`
	Evaluation                  composeCorpusMeasurement `json:"evaluation"`
	UniqueTrainingTargets       int64                    `json:"unique_training_targets"`
	PlannedTrainingTokens       int64                    `json:"planned_training_tokens"`
	EffectiveCorpusPasses       float64                  `json:"effective_corpus_passes"`
	RecordsEncountered          float64                  `json:"records_encountered"`
	EOSBoundariesEncountered    float64                  `json:"eos_boundaries_encountered"`
	PackedBoundariesPerSequence float64                  `json:"packed_boundaries_per_sequence"`
	MeasurementScope            string                   `json:"measurement_scope"`
}

type composeCorpusMeasurement struct {
	Records               int64                    `json:"records"`
	UTF8Bytes             int64                    `json:"utf8_bytes"`
	Characters            int64                    `json:"characters"`
	Tokens                int64                    `json:"tokens"`
	BytesPerToken         float64                  `json:"bytes_per_token"`
	TokensPerCharacter    float64                  `json:"tokens_per_character"`
	RecordP50Tokens       int64                    `json:"record_p50_tokens"`
	RecordP90Tokens       int64                    `json:"record_p90_tokens"`
	RecordP95Tokens       int64                    `json:"record_p95_tokens"`
	RecordP50Bytes        int64                    `json:"record_p50_bytes"`
	RecordP90Bytes        int64                    `json:"record_p90_bytes"`
	RecordP95Bytes        int64                    `json:"record_p95_bytes"`
	FitOneSequencePercent float64                  `json:"fit_one_sequence_percent"`
	Corpora               []composeCorpusFertility `json:"corpora"`
	tokenLengths          map[int]int64
	byteLengths           map[int]int64
	fitRecords            int64
}

type composeCorpusFertility struct {
	Corpus             string  `json:"corpus"`
	Records            int64   `json:"records"`
	UTF8Bytes          int64   `json:"utf8_bytes"`
	Characters         int64   `json:"characters"`
	Tokens             int64   `json:"tokens"`
	BytesPerToken      float64 `json:"bytes_per_token"`
	TokensPerCharacter float64 `json:"tokens_per_character"`
	PlannedTargets     float64 `json:"planned_targets"`
	EffectivePasses    float64 `json:"effective_passes"`
}

func preflightCompose(context Context, compose model.Compose, cache *lookaside.Cache, progress io.Writer) (composePreflightReport, error) {
	targets, err := sanityCheckComposeCorpora(context.Execution, compose, progress)
	if err != nil {
		return composePreflightReport{}, err
	}
	_, codec, err := compose.Architecture.ResolveTokenizer()
	if err != nil {
		return composePreflightReport{}, fmt.Errorf("compose tokenizer: %w", err)
	}
	report := composePreflightReport{Stages: make([]composeStagePreflight, 0, len(compose.Stages))}
	for _, stage := range compose.Stages {
		planned, err := planModelStage(context, stage, targets[stage.Name], cache, progress)
		if err != nil {
			return composePreflightReport{}, err
		}
		prepared, err := materializeModelStage(context, planned.Stage, planned.BOM, cache, progress, false)
		if err != nil {
			return composePreflightReport{}, err
		}
		measured, err := measureComposeStage(context.Execution, prepared, codec, progress)
		if err != nil {
			return composePreflightReport{}, err
		}
		report.Stages = append(report.Stages, measured)
		if _, err := cache.PurgeUsed(); err != nil {
			return composePreflightReport{}, fmt.Errorf("purge successful preflight cache: %w", err)
		}
	}
	return report, nil
}

func measureComposeStage(ctx stdcontext.Context, prepared model.PreparedStage, codec training.TokenCodec, progress io.Writer) (composeStagePreflight, error) {
	stage := prepared.Stage
	if stage.Objective != "causal-language-modeling" || stage.Conversation != nil {
		return composeStagePreflight{}, fmt.Errorf("stage %s preflight measurement currently requires text causal-language-modeling records", stage.Name)
	}
	parameters, err := stage.ResolvePlanningParameters()
	if err != nil {
		return composeStagePreflight{}, fmt.Errorf("stage %s training profile: %w", stage.Name, err)
	}
	partition, err := training.NewRecordPartitionContextWithTokenizerAndObjective(ctx, prepared.Inputs, parameters, codec, stage.Objective, partitionProgressPrinter(progress, stage.Name))
	if err != nil {
		return composeStagePreflight{}, fmt.Errorf("stage %s record partition: %w", stage.Name, err)
	}
	fmt.Fprintf(progress, "preflight/%s          measuring tokenizer fertility and record lengths\n", stage.Name)
	trainingMeasurement, err := measureComposeRecordSource(ctx, partition.TrainingMeasurementRecords(), codec, parameters.SequenceLength)
	if err != nil {
		return composeStagePreflight{}, fmt.Errorf("stage %s measure training records: %w", stage.Name, err)
	}
	evaluationMeasurement, err := measureComposeRecordSource(ctx, partition.EvaluationRecords(), codec, parameters.SequenceLength)
	if err != nil {
		return composeStagePreflight{}, fmt.Errorf("stage %s measure evaluation records: %w", stage.Name, err)
	}
	uniqueTargets := trainingMeasurement.Tokens + trainingMeasurement.Records
	if uniqueTargets > 0 {
		uniqueTargets-- // Causal shifting removes the first target in the packed stream.
	}
	plannedTokens := parameters.PlannedTokenCapacity
	if stage.Parameters.Epochs > 0 {
		plannedTokens, err = composeEpochTargets(uniqueTargets, stage.Parameters.Epochs)
		if err != nil {
			return composeStagePreflight{}, fmt.Errorf("stage %s epoch target count: %w", stage.Name, err)
		}
	}
	passes := float64(0)
	if uniqueTargets > 0 {
		passes = float64(plannedTokens) / float64(uniqueTargets)
	}
	bytesPerToken := trainingMeasurement.BytesPerToken
	result := composeStagePreflight{
		Name: stage.Name, SequenceTokens: parameters.SequenceLength,
		EffectiveContextBytes: float64(parameters.SequenceLength) * bytesPerToken,
		Training:              trainingMeasurement, Evaluation: evaluationMeasurement,
		UniqueTrainingTargets: uniqueTargets, PlannedTrainingTokens: plannedTokens,
		EffectiveCorpusPasses:    passes,
		RecordsEncountered:       passes * float64(trainingMeasurement.Records),
		EOSBoundariesEncountered: passes * float64(trainingMeasurement.Records),
		MeasurementScope:         "training records exclude the model evaluation partition; evaluation fertility is model-held-out, not tokenizer-training-disjoint",
	}
	framedTokens := trainingMeasurement.Tokens + trainingMeasurement.Records
	if framedTokens > 0 {
		result.PackedBoundariesPerSequence = float64(parameters.SequenceLength) * float64(trainingMeasurement.Records) / float64(framedTokens)
	}
	applyCorpusExposure(result.Training.Corpora, parameters, plannedTokens)
	return result, nil
}

func composeEpochTargets(oneEpochTargets, epochs int64) (int64, error) {
	if oneEpochTargets < 1 || epochs < 1 || oneEpochTargets == math.MaxInt64 || oneEpochTargets+1 > math.MaxInt64/epochs {
		return 0, fmt.Errorf("target count overflows int64")
	}
	return (oneEpochTargets+1)*epochs - 1, nil
}

func applyCorpusExposure(corpora []composeCorpusFertility, parameters training.ResolvedParameters, plannedTokens int64) {
	if len(corpora) == 0 || plannedTokens <= 0 {
		return
	}
	shares := make([]float64, len(corpora))
	var denominator float64
	for index := range corpora {
		switch parameters.Data.Order {
		case "corpus-balanced-shuffle-v1":
			shares[index] = 1
		case "corpus-weighted-shuffle-v1":
			shares[index] = float64(parameters.Data.CorpusWeights[corpora[index].Corpus])
		default:
			shares[index] = float64(corpora[index].Tokens + corpora[index].Records)
		}
		denominator += shares[index]
	}
	if denominator == 0 {
		return
	}
	for index := range corpora {
		corpora[index].PlannedTargets = float64(plannedTokens) * shares[index] / denominator
		unique := corpora[index].Tokens + corpora[index].Records
		if unique > 0 {
			corpora[index].EffectivePasses = corpora[index].PlannedTargets / float64(unique)
		}
	}
}

func partitionProgressPrinter(output io.Writer, stage string) func(training.PartitionProgress) {
	return func(progress training.PartitionProgress) {
		if progress.TotalShards == 0 {
			return
		}
		fmt.Fprintf(output, "preflight/%s          evaluation selection %d/%d shards, %d records indexed\n", stage, progress.CurrentShard, progress.TotalShards, progress.Records)
	}
}

func measureComposeRecordSource(ctx stdcontext.Context, source training.RecordSource, codec training.TokenCodec, sequenceLength int64) (composeCorpusMeasurement, error) {
	measurement := composeCorpusMeasurement{tokenLengths: map[int]int64{}, byteLengths: map[int]int64{}}
	byCorpus := map[string]*composeCorpusFertility{}
	err := source.Stream(ctx, func(record training.Record) error {
		bytes := len([]byte(record.Text))
		characters := utf8.RuneCountInString(record.Text)
		tokens := codec.Count(record.Text)
		measurement.Records++
		measurement.UTF8Bytes += int64(bytes)
		measurement.Characters += int64(characters)
		measurement.Tokens += int64(tokens)
		measurement.tokenLengths[tokens]++
		measurement.byteLengths[bytes]++
		if int64(tokens+1) <= sequenceLength { // Include the per-record EOS boundary.
			measurement.fitRecords++
		}
		corpus := record.Corpus
		if corpus == "" {
			corpus = "unattributed"
		}
		value := byCorpus[corpus]
		if value == nil {
			value = &composeCorpusFertility{Corpus: corpus}
			byCorpus[corpus] = value
		}
		value.Records++
		value.UTF8Bytes += int64(bytes)
		value.Characters += int64(characters)
		value.Tokens += int64(tokens)
		return nil
	})
	if err != nil {
		return composeCorpusMeasurement{}, err
	}
	measurement.finish(byCorpus)
	return measurement, nil
}

func (measurement *composeCorpusMeasurement) finish(byCorpus map[string]*composeCorpusFertility) {
	if measurement.Tokens > 0 {
		measurement.BytesPerToken = float64(measurement.UTF8Bytes) / float64(measurement.Tokens)
	}
	if measurement.Characters > 0 {
		measurement.TokensPerCharacter = float64(measurement.Tokens) / float64(measurement.Characters)
	}
	if measurement.Records > 0 {
		measurement.FitOneSequencePercent = 100 * float64(measurement.fitRecords) / float64(measurement.Records)
		measurement.RecordP50Tokens = histogramQuantile(measurement.tokenLengths, measurement.Records, 0.50)
		measurement.RecordP90Tokens = histogramQuantile(measurement.tokenLengths, measurement.Records, 0.90)
		measurement.RecordP95Tokens = histogramQuantile(measurement.tokenLengths, measurement.Records, 0.95)
		measurement.RecordP50Bytes = histogramQuantile(measurement.byteLengths, measurement.Records, 0.50)
		measurement.RecordP90Bytes = histogramQuantile(measurement.byteLengths, measurement.Records, 0.90)
		measurement.RecordP95Bytes = histogramQuantile(measurement.byteLengths, measurement.Records, 0.95)
	}
	for _, value := range byCorpus {
		if value.Tokens > 0 {
			value.BytesPerToken = float64(value.UTF8Bytes) / float64(value.Tokens)
		}
		if value.Characters > 0 {
			value.TokensPerCharacter = float64(value.Tokens) / float64(value.Characters)
		}
		measurement.Corpora = append(measurement.Corpora, *value)
	}
	sort.Slice(measurement.Corpora, func(i, j int) bool { return measurement.Corpora[i].Corpus < measurement.Corpora[j].Corpus })
	measurement.tokenLengths = nil
	measurement.byteLengths = nil
}

func histogramQuantile(histogram map[int]int64, total int64, quantile float64) int64 {
	if total <= 0 {
		return 0
	}
	keys := make([]int, 0, len(histogram))
	for value := range histogram {
		keys = append(keys, value)
	}
	sort.Ints(keys)
	target := int64(math.Ceil(float64(total) * quantile))
	var seen int64
	for _, value := range keys {
		seen += histogram[value]
		if seen >= target {
			return int64(value)
		}
	}
	return int64(keys[len(keys)-1])
}

func writeComposePreflight(output io.Writer, report composePreflightReport) {
	for _, stage := range report.Stages {
		fmt.Fprintf(output, "\nPREFLIGHT %s\n", stage.Name)
		fmt.Fprintf(output, "TRAINING:    %s records, %s UTF-8 bytes, %s tokenizer tokens\n", humanCount(stage.Training.Records), humanBytes(stage.Training.UTF8Bytes), humanCount(stage.Training.Tokens))
		fmt.Fprintf(output, "FERTILITY:   %.3f bytes/token; %d tokens = %.0f effective context bytes\n", stage.Training.BytesPerToken, stage.SequenceTokens, stage.EffectiveContextBytes)
		fmt.Fprintf(output, "RECORDS:     tokens p50/p90/p95 %d/%d/%d; bytes %d/%d/%d; %.1f%% fit one sequence\n", stage.Training.RecordP50Tokens, stage.Training.RecordP90Tokens, stage.Training.RecordP95Tokens, stage.Training.RecordP50Bytes, stage.Training.RecordP90Bytes, stage.Training.RecordP95Bytes, stage.Training.FitOneSequencePercent)
		fmt.Fprintf(output, "PACKING:     %.3f expected EOS/document boundaries per sequence\n", stage.PackedBoundariesPerSequence)
		fmt.Fprintf(output, "EXPOSURE:    %s unique targets, %s planned targets, %.2f effective corpus passes\n", humanCount(stage.UniqueTrainingTargets), humanCount(stage.PlannedTrainingTokens), stage.EffectiveCorpusPasses)
		fmt.Fprintf(output, "HELD OUT:    %s records, %.3f bytes/token (model evaluation partition)\n", humanCount(stage.Evaluation.Records), stage.Evaluation.BytesPerToken)
		for _, corpus := range stage.Training.Corpora {
			fmt.Fprintf(output, "CORPUS:      %s — %s records, %.3f bytes/token, %.2f effective passes\n", corpus.Corpus, humanCount(corpus.Records), corpus.BytesPerToken, corpus.EffectivePasses)
		}
		fmt.Fprintln(output, "NOTE:        held-out fertility uses the model evaluation partition; it is not disjoint from tokenizer training")
	}
}
