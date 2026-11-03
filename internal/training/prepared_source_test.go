// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestExternalPreparedSourcePackingAndResume(t *testing.T) {
	records := staticRecordSource{
		{ID: "a", Corpus: "books", Tokens: []int{10, 11, 12}, LossMask: []bool{false, true, false, true}},
		{ID: "b", Corpus: "code", Tokens: []int{20, 21, 22, 23}, LossMask: []bool{true, true, false, true, true}},
	}
	data := ExternalDataParameters{Steps: 2, GlobalBatchSequences: 2, SequenceLength: 2}
	spec := TokenizerSpec{VocabularySize: 30, EOSID: 2, PadID: 0}
	want := []PreparedSequence{
		{Ordinal: 0, Tokens: []int{10, 11, 12}, LossMask: []bool{true, false}, Consumption: map[string]int64{"books": 1}},
		{Ordinal: 1, Tokens: []int{12, 2, 20}, LossMask: []bool{true, true}, Consumption: map[string]int64{"books": 1, "code": 1}},
		{Ordinal: 2, Tokens: []int{20, 21, 22}, LossMask: []bool{true, false}, Consumption: map[string]int64{"code": 1}},
		{Ordinal: 3, Tokens: []int{22, 23, 2}, LossMask: []bool{true, true}, Consumption: map[string]int64{"code": 2}},
	}
	for step := int64(0); step <= 2; step++ {
		source, err := NewPreparedSequenceSource(records, data, spec, step)
		if err != nil {
			t.Fatal(err)
		}
		var got []PreparedSequence
		if err := source(t.Context(), func(s PreparedSequence) error { got = append(got, s); return nil }); err != nil {
			t.Fatal(err)
		}
		if len(got) != len(want[step*2:]) || len(got) > 0 && !reflect.DeepEqual(got, want[step*2:]) {
			t.Fatalf("step %d: %#v", step, got)
		}
	}
	source, err := NewPreparedSequenceSource(refusingRecordSource{}, data, spec, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err = source(t.Context(), func(PreparedSequence) error { return errors.New("unexpected callback") }); err != nil {
		t.Fatal(err)
	}
}

func TestExternalPreparedSourceSkipsUnsupervisedAndPadsTail(t *testing.T) {
	// A fully unsupervised window must not consume a global batch slot.
	source, err := NewPreparedSequenceSource(staticRecordSource{{Corpus: "a", Tokens: []int{10, 11, 12, 13}, LossMask: []bool{false, false, false, false, true}}}, ExternalDataParameters{Steps: 1, GlobalBatchSequences: 1, SequenceLength: 3}, TokenizerSpec{VocabularySize: 20, EOSID: 2}, 0)
	if err != nil {
		t.Fatal(err)
	}
	var got []PreparedSequence
	if err = source(t.Context(), func(s PreparedSequence) error { got = append(got, s); return nil }); err != nil {
		t.Fatal(err)
	}
	want := []PreparedSequence{{Ordinal: 0, Tokens: []int{13, 2, 0, 0}, LossMask: []bool{true, false, false}, Consumption: map[string]int64{"a": 1}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %#v", got)
	}
}

func TestExternalPreparedSourceFailures(t *testing.T) {
	data := ExternalDataParameters{Steps: 1, GlobalBatchSequences: 2, SequenceLength: 2}
	spec := TokenizerSpec{VocabularySize: 20, EOSID: 2}
	valid := Record{Corpus: "a", Tokens: []int{10, 11}, LossMask: []bool{true, true, true}}
	for _, tc := range []struct {
		name    string
		record  Record
		message string
	}{
		{"short", valid, "exact budget"},
		{"mask", Record{Corpus: "a", Tokens: []int{10}}, "EOS target"},
		{"vocabulary", Record{Corpus: "a", Tokens: []int{20}, LossMask: []bool{true, true}}, "outside vocabulary"},
		{"corpus", Record{Tokens: []int{10}, LossMask: []bool{true, true}}, "lacks corpus"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source, err := NewPreparedSequenceSource(staticRecordSource{tc.record}, data, spec, 0)
			if err != nil {
				t.Fatal(err)
			}
			err = source(t.Context(), func(PreparedSequence) error { return nil })
			if err == nil || !strings.Contains(err.Error(), tc.message) {
				t.Fatalf("got %v", err)
			}
		})
	}
	sentinel := errors.New("consumer stopped")
	source, err := NewPreparedSequenceSource(staticRecordSource{valid}, data, spec, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err = source(t.Context(), func(PreparedSequence) error { return sentinel }); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err = source(ctx, func(PreparedSequence) error { return nil }); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	for _, step := range []int64{-1, 2} {
		if _, err = NewPreparedSequenceSource(staticRecordSource{valid}, data, spec, step); err == nil {
			t.Fatal("invalid resume accepted")
		}
	}
	data.Steps = math.MaxInt64
	if _, err = NewPreparedSequenceSource(staticRecordSource{valid}, data, spec, 0); err == nil {
		t.Fatal("overflow accepted")
	}
}

func TestExternalPreparedSourceUsesPartitionedParquet(t *testing.T) {
	input := writeTrainingShard(t, []string{"aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd"})
	input.Corpus = "books"
	fraction := 0.25
	parameters, err := ResolveParameters(Parameters{Tokens: 12, LearningRate: 0.001, BatchSize: 1, SequenceLength: 3, Seed: 42, EvaluationFraction: &fraction})
	if err != nil {
		t.Fatal(err)
	}
	spec, codec, err := ResolveTokenizer("byte", ByteTokenizerRevision, 259)
	if err != nil {
		t.Fatal(err)
	}
	partition, err := NewRecordPartitionContextWithTokenizer(t.Context(), []Input{input}, parameters, codec, nil)
	if err != nil {
		t.Fatal(err)
	}
	heldoutToken := -1
	if err = partition.EvaluationRecords().Stream(t.Context(), func(r Record) error { heldoutToken = codec.Encode(r.Text)[0]; return nil }); err != nil {
		t.Fatal(err)
	}
	if heldoutToken < 0 {
		t.Fatal("fixture did not select holdout")
	}
	raw, err := partition.TrainingRecords()
	if err != nil {
		t.Fatal(err)
	}
	tokenized, err := NewTokenizedRecordSource(raw, codec, "causal-language-modeling", ConversationTransform{})
	if err != nil {
		t.Fatal(err)
	}
	data := ExternalDataParameters{Steps: parameters.Steps, GlobalBatchSequences: 1, SequenceLength: 3}
	collect := func(step int64) []PreparedSequence {
		t.Helper()
		source, err := NewPreparedSequenceSource(tokenized, data, spec, step)
		if err != nil {
			t.Fatal(err)
		}
		var got []PreparedSequence
		if err = source(t.Context(), func(s PreparedSequence) error { got = append(got, s); return nil }); err != nil {
			t.Fatal(err)
		}
		return got
	}
	full := collect(0)
	for _, s := range full {
		for _, token := range s.Tokens {
			if token == heldoutToken {
				t.Fatal("held-out record leaked into prepared stream")
			}
		}
	}
	if got := collect(1); !reflect.DeepEqual(got, full[1:]) {
		t.Fatalf("resumed Parquet stream differs: %#v", got)
	}
}

func TestExternalPreparedCacheResumeAndCorruption(t *testing.T) {
	records := staticRecordSource{{Corpus: "books", Tokens: []int{10, 11, 12, 13, 14, 15, 16, 17}, LossMask: []bool{true, true, true, true, true, true, true, true, true}}}
	data := ExternalDataParameters{Steps: 2, GlobalBatchSequences: 2, SequenceLength: 2}
	spec := TokenizerSpec{VocabularySize: 30, EOSID: 2}
	dir := t.TempDir()
	source, err := CachedPreparedSequenceSource(records, data, spec, 0, dir, "identity", strings.Repeat("a", 64), 0)
	if err != nil {
		t.Fatal(err)
	}
	var full []PreparedSequence
	if err = source(t.Context(), func(s PreparedSequence) error { full = append(full, s); return nil }); err != nil {
		t.Fatal(err)
	}
	source, err = CachedPreparedSequenceSource(refusingRecordSource{}, data, spec, 1, dir, "identity", strings.Repeat("a", 64), 0)
	if err != nil {
		t.Fatal(err)
	}
	var suffix []PreparedSequence
	if err = source(t.Context(), func(s PreparedSequence) error { suffix = append(suffix, s); return nil }); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(suffix, full[2:]) {
		t.Fatalf("cached suffix differs: %#v", suffix)
	}
	wrong, err := CachedPreparedSequenceSource(records, data, spec, 1, dir, "identity", strings.Repeat("b", 64), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err = wrong(t.Context(), func(PreparedSequence) error { return nil }); err == nil {
		t.Fatal("cache accepted changed run BOM")
	}
	var manifest preparedCacheManifest
	path := filepath.Join(dir, "identity", "node-0")
	raw, err := os.ReadFile(filepath.Join(path, "MANIFEST.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(path, manifest.Chunks[0].Path), []byte("corrupt"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = source(t.Context(), func(PreparedSequence) error { return nil }); err == nil {
		t.Fatal("corrupt cache accepted")
	}
}
