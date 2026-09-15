// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
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
	"github.com/openwaldo/waldo/internal/lookaside"
	waldotokenizer "github.com/openwaldo/waldo/internal/tokenizer"
	"github.com/openwaldo/waldo/internal/training"
)

var errTokenizerSampleComplete = errors.New("tokenizer sample complete")

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
	policy, err := corpus.NewLicensePolicy(nil, nil)
	if err != nil {
		return err
	}
	bom, err := corpus.BuildBOM(context.Execution, targets, policy, cache)
	if err != nil {
		return err
	}
	if _, err := corpus.ReviewDistributable(bom); err != nil {
		return fmt.Errorf("tokenizer training corpus is not distributable: %w", err)
	}
	encodedBOM, err := json.Marshal(bom)
	if err != nil {
		return err
	}
	bomDigest := sha256.Sum256(encodedBOM)
	bomSHA256 := hex.EncodeToString(bomDigest[:])
	materialized, err := corpus.Materialize(context.Execution, bom, cache, modelMaterializeProgressPrinter(stderr))
	if err != nil {
		return err
	}
	parameters, err := training.ResolveParameters(training.Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 42})
	if err != nil {
		return err
	}
	records, err := training.NewCanonicalRecordSource(verifiedTrainingInputs(materialized, bom), parameters)
	if err != nil {
		return err
	}
	var samples []waldotokenizer.Sample
	var bytesSelected int64
	err = records.Stream(context.Execution, func(record training.Record) error {
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
		return err
	}
	artifact, err := waldotokenizer.TrainBytepiece(samples, vocabularySize, sampleBytes, bomSHA256)
	if err != nil {
		return err
	}
	trainedCodec, err := artifact.Codec()
	if err != nil {
		return err
	}
	r50k, err := waldotokenizer.NewCodec("tiktoken/r50k_base")
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
	if err := writeJSON(file, artifact); err != nil {
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
	}{output, artifact, []waldotokenizer.Comparison{waldotokenizer.Compare(r50k, samples), waldotokenizer.Compare(trainedCodec, samples)}}
	if context.JSON {
		return writeJSON(stdout, result)
	}
	fmt.Fprintf(stdout, "wrote tokenizer %s to %s\n", artifact.Revision, output)
	for _, comparison := range result.Compression {
		fmt.Fprintf(stdout, "  %-32s %.3f bytes/token (%s tokens)\n", comparison.Tokenizer, comparison.BytesPerToken, humanCount(comparison.Tokens))
	}
	return nil
}
