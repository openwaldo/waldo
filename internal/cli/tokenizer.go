// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	stdcontext "context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"unicode/utf8"

	"github.com/openwaldo/waldo/internal/corpus"
	waldoindex "github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
	waldotokenizer "github.com/openwaldo/waldo/internal/tokenizer"
	"github.com/openwaldo/waldo/internal/training"
)

var errTokenizerSampleComplete = errors.New("tokenizer sample complete")

type tokenizerTrainingResult struct {
	Artifact    waldotokenizer.Artifact
	Compression []waldotokenizer.Comparison
}

func runModelTrainTokenizer(context Context, args []string, stdout, stderr io.Writer) error {
	vocabularySize := intOption(context, "vocabulary-size")
	sampleBytes := int64Option(context, "sample-bytes")
	output, err := filepath.Abs(stringOption(context, "output"))
	if err != nil {
		return err
	}
	targets, err := resolveIndexArgumentsWithWarning(context.Execution, args, stderr)
	if err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	bom, err := buildTokenizerBOM(context.Execution, targets, cache, nil)
	if err != nil {
		return err
	}
	trained, err := trainTokenizerArtifact(context.Execution, bom, cache, vocabularySize, sampleBytes, 42, stderr)
	if err != nil {
		return err
	}
	flags := os.O_CREATE | os.O_EXCL | os.O_WRONLY
	if boolOption(context, "force") {
		flags = os.O_CREATE | os.O_TRUNC | os.O_WRONLY
	}
	file, err := os.OpenFile(output, flags, 0o644)
	if err != nil {
		return err
	}
	if err := writeJSON(file, trained.Artifact); err != nil {
		_ = file.Close()
		_ = os.Remove(output)
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if _, err := cache.PurgeUsed(); err != nil {
		return fmt.Errorf("purge successful tokenizer training cache: %w", err)
	}
	result := struct {
		Output      string                      `json:"output"`
		Artifact    waldotokenizer.Artifact     `json:"artifact"`
		Compression []waldotokenizer.Comparison `json:"compression"`
	}{output, trained.Artifact, trained.Compression}
	if context.JSON {
		return writeJSON(stdout, result)
	}
	fmt.Fprintf(stdout, "wrote tokenizer %s to %s\n", trained.Artifact.Revision, output)
	for _, comparison := range result.Compression {
		fmt.Fprintf(stdout, "  %-32s %.3f bytes/token (%s tokens)\n", comparison.Tokenizer, comparison.BytesPerToken, humanCount(comparison.Tokens))
	}
	return nil
}

func buildTokenizerBOM(execution stdcontext.Context, targets []waldoindex.Target, cache *lookaside.Cache, recordFilter *corpus.RecordFilterPolicy) (corpus.BOM, error) {
	policy, err := corpus.NewLicensePolicy(nil, nil)
	if err != nil {
		return corpus.BOM{}, err
	}
	bom, err := corpus.BuildBOM(execution, targets, policy, cache)
	if err != nil {
		return corpus.BOM{}, err
	}
	return configureTokenizerBOM(bom, recordFilter)
}

func configureTokenizerBOM(bom corpus.BOM, recordFilter *corpus.RecordFilterPolicy) (corpus.BOM, error) {
	if recordFilter == nil {
		recordFilter = &corpus.RecordFilterPolicy{Schema: corpus.RecordFilterSchema}
	}
	recordFilter.Distributable = true
	bom.RecordFilter = recordFilter
	if err := bom.Validate(); err != nil {
		return corpus.BOM{}, err
	}
	if _, err := corpus.ReviewDistributable(bom); err != nil {
		return corpus.BOM{}, fmt.Errorf("tokenizer training corpus is not distributable: %w", err)
	}
	return bom, nil
}

func trainTokenizerArtifact(execution stdcontext.Context, bom corpus.BOM, cache *lookaside.Cache, vocabularySize int, sampleBytes int64, seed uint64, progress io.Writer) (tokenizerTrainingResult, error) {
	encodedBOM, err := json.Marshal(bom)
	if err != nil {
		return tokenizerTrainingResult{}, err
	}
	bomDigest := sha256.Sum256(encodedBOM)
	bomSHA256 := hex.EncodeToString(bomDigest[:])
	fmt.Fprintf(progress, "tokenizer               resolving %s shards, %s records, %s reference tokens\n", humanInteger(bom.Totals.Shards), humanCount(bom.Totals.Docs), humanCount(bom.Totals.Tokens))
	materialized, err := corpus.Materialize(execution, bom, cache, modelMaterializeProgressPrinter(progress))
	if err != nil {
		return tokenizerTrainingResult{}, err
	}
	parameters, err := training.ResolveParameters(training.Parameters{Profile: training.BalancedProfile, Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: seed})
	if err != nil {
		return tokenizerTrainingResult{}, err
	}
	records, err := training.NewCanonicalRecordSource(verifiedTrainingInputs(materialized, bom), parameters)
	if err != nil {
		return tokenizerTrainingResult{}, err
	}
	var samples []waldotokenizer.Sample
	var bytesSelected int64
	err = records.Stream(execution, func(record training.Record) error {
		if bytesSelected >= sampleBytes {
			return errTokenizerSampleComplete
		}
		text := record.Text
		remaining := sampleBytes - bytesSelected
		if int64(len([]byte(text))) > remaining {
			text = string([]byte(text)[:remaining])
			for !utf8.ValidString(text) && len(text) > 0 {
				text = text[:len(text)-1]
			}
		}
		samples = append(samples, waldotokenizer.Sample{ID: record.ID, Text: text})
		bytesSelected += int64(len([]byte(text)))
		return nil
	})
	if err != nil && !errors.Is(err, errTokenizerSampleComplete) {
		return tokenizerTrainingResult{}, err
	}
	artifact, err := waldotokenizer.TrainByteBPE(samples, vocabularySize, sampleBytes, bomSHA256)
	if err != nil {
		return tokenizerTrainingResult{}, err
	}
	if artifact.VocabularySize != vocabularySize {
		return tokenizerTrainingResult{}, fmt.Errorf("tokenizer training produced %d of %d requested vocabulary entries", artifact.VocabularySize, vocabularySize)
	}
	trainedCodec, err := artifact.Codec()
	if err != nil {
		return tokenizerTrainingResult{}, err
	}
	r50k, err := waldotokenizer.NewCodec("tiktoken/r50k_base")
	if err != nil {
		return tokenizerTrainingResult{}, err
	}
	return tokenizerTrainingResult{Artifact: artifact, Compression: []waldotokenizer.Comparison{
		waldotokenizer.Compare(r50k, samples), waldotokenizer.Compare(trainedCodec, samples),
	}}, nil
}

func resolveComposeTokenizerTraining(context Context, compose model.Compose, cache *lookaside.Cache, progress io.Writer) (model.Compose, error) {
	declaration := compose.Architecture.Tokenizer.Training
	if declaration == nil || compose.Architecture.Tokenizer.Artifact != nil {
		return compose, nil
	}
	paths := model.CorpusPaths(declaration.Corpora)
	targets, err := resolveIndexArgumentsWithWarningPolicy(context.Execution, paths, progress, true)
	if err != nil {
		return model.Compose{}, fmt.Errorf("tokenizer training: %w", err)
	}
	policy, err := corpus.NewLicensePolicy(nil, nil)
	if err != nil {
		return model.Compose{}, err
	}
	bom, err := corpus.BuildBOM(context.Execution, targets, policy, cache)
	if err != nil {
		return model.Compose{}, fmt.Errorf("tokenizer training: %w", err)
	}
	recordFilter, err := declaration.RecordFilterPolicy(bom.Paths)
	if err != nil {
		return model.Compose{}, fmt.Errorf("tokenizer training: %w", err)
	}
	bom, err = configureTokenizerBOM(bom, recordFilter)
	if err != nil {
		return model.Compose{}, err
	}
	emitUnassessedFilterWarning(progress, "tokenizer", bom)
	result, err := trainTokenizerArtifact(context.Execution, bom, cache, int(compose.Architecture.VocabularySize), declaration.SampleBytes, declaration.Seed, progress)
	if err != nil {
		return model.Compose{}, err
	}
	for _, comparison := range result.Compression {
		fmt.Fprintf(progress, "tokenizer               %-32s %.3f bytes/token (%s tokens)\n", comparison.Tokenizer, comparison.BytesPerToken, humanCount(comparison.Tokens))
	}
	if err := validateTokenizerCompression(result.Compression, declaration.MaxTokenInflation); err != nil {
		return model.Compose{}, err
	}
	inflation := float64(result.Compression[1].Tokens)/float64(result.Compression[0].Tokens) - 1
	fmt.Fprintf(progress, "tokenizer               token inflation vs r50k %.1f%% (limit %.1f%%)\n", inflation*100, declaration.MaxTokenInflation*100)
	compose.Architecture.Tokenizer.Name = result.Artifact.Name
	compose.Architecture.Tokenizer.Revision = result.Artifact.Revision
	compose.Architecture.Tokenizer.Artifact = &result.Artifact
	compose.Architecture.Tokenizer.Training.CorpusBOM = &bom
	if err := compose.Validate(); err != nil {
		return model.Compose{}, err
	}
	fmt.Fprintf(progress, "tokenizer               selected %s with %s vocabulary entries from %s sampled bytes\n", result.Artifact.Revision, humanInteger(int64(result.Artifact.VocabularySize)), humanInteger(result.Artifact.TrainingBytes))
	return compose, nil
}

func validateTokenizerCompression(comparisons []waldotokenizer.Comparison, maximumInflation float64) error {
	if len(comparisons) != 2 {
		return fmt.Errorf("tokenizer compression gate requires baseline and candidate measurements")
	}
	baseline, candidate := comparisons[0], comparisons[1]
	if baseline.Tokens < 1 || candidate.Tokens < 1 {
		return fmt.Errorf("tokenizer compression gate requires non-empty measurements")
	}
	inflation := float64(candidate.Tokens)/float64(baseline.Tokens) - 1
	maximumTokens := float64(baseline.Tokens) * (1 + maximumInflation)
	if float64(candidate.Tokens) > maximumTokens+1e-9 {
		return fmt.Errorf("trained tokenizer uses %.1f%% more tokens than r50k; compose permits at most %.1f%% (%.3f vs %.3f bytes/token)", inflation*100, maximumInflation*100, candidate.BytesPerToken, baseline.BytesPerToken)
	}
	return nil
}
