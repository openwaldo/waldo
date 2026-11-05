// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"strings"
	"testing"
)

// sharedBrokenCodec is shared by balanced corpus workers like the external
// tokenizer application client, which reports ConcurrentSafe.
type sharedBrokenCodec struct{ brokenCodec }

func (sharedBrokenCodec) ConcurrentSafe() bool { return true }

// Every held-out selection path must surface a fallible codec error instead of
// pinning a zero token-target count for the evaluation set.
func TestFallibleTokenizerFailsEvaluationPartition(t *testing.T) {
	many := []Input{writeTrainingShard(t, []string{"alpha", "bravo", "charlie", "delta"})}
	single := []Input{writeTrainingShard(t, []string{"a single record long enough to split"})}
	for selection, inputs := range map[string][]Input{"indexed-sample-v1": many, "lowest-sha256-v1": many, "contiguous-tail-v1": single} {
		fraction, records, bytes := 0.5, 4, int64(1<<20)
		parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 7, EvaluationSelection: selection, EvaluationFraction: &fraction, EvaluationMaxRecords: &records, EvaluationMaxBytes: &bytes})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := NewRecordPartitionContextWithTransform(context.Background(), inputs, parameters, brokenCodec{}, "causal-language-modeling", ConversationTransform{}, nil); err == nil || !strings.Contains(err.Error(), "tokenizer disconnected") {
			t.Fatalf("%s: partition error = %v", selection, err)
		}
	}
}

// Corpus-balanced streaming counts tokens either serially or in per-corpus
// workers; both must stop on a codec failure rather than emit zero-token records.
func TestFallibleTokenizerFailsBalancedStream(t *testing.T) {
	a := writeTrainingShard(t, []string{"alpha", "bravo"})
	a.Corpus = "corpus-a"
	b := writeTrainingShard(t, []string{"charlie", "delta"})
	b.Corpus = "corpus-b"
	zeroFraction, zeroRecords, zeroBytes := 0.0, 0, int64(0)
	parameters, err := ResolveParameters(Parameters{Profile: BalancedProfile, Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 7, EvaluationFraction: &zeroFraction, EvaluationMaxRecords: &zeroRecords, EvaluationMaxBytes: &zeroBytes})
	if err != nil {
		t.Fatal(err)
	}
	for name, codec := range map[string]TokenCodec{"serial": brokenCodec{}, "shared": sharedBrokenCodec{}} {
		partition, err := NewRecordPartitionContextWithTransform(context.Background(), []Input{a, b}, parameters, codec, "causal-language-modeling", ConversationTransform{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		source, err := partition.TrainingRecords()
		if err != nil {
			t.Fatal(err)
		}
		accepted := 0
		err = source.Stream(context.Background(), func(Record) error { accepted++; return nil })
		if err == nil || !strings.Contains(err.Error(), "tokenizer disconnected") || accepted != 0 {
			t.Fatalf("%s: stream error = %v after %d records", name, err, accepted)
		}
	}
}
