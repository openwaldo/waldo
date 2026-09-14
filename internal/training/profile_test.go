// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/record"
	"github.com/openwaldo/waldo/internal/shard"
	"github.com/parquet-go/parquet-go"
)

func TestResolveParametersPinsVersionedDefaultsAndOverrides(t *testing.T) {
	parameters := Parameters{Steps: 1000, BatchSize: 8, SequenceLength: 512, LearningRate: 0.0003, Seed: 42}
	resolved, err := ResolveParameters(parameters)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Profile != DefaultProfile || resolved.ProfileSchema != 1 || resolved.Epochs != 1 || resolved.Optimizer.Name != "adamw" || resolved.Optimizer.WeightDecay != 0.1 || resolved.Schedule.Name != "cosine" || resolved.Schedule.WarmupSteps != 100 || resolved.Data.Order != "bounded-shuffle-v1" || resolved.Data.Packing != "continuous-eos-v1" || resolved.Evaluation == nil || resolved.Evaluation.Selection != "lowest-sha256-v1" || resolved.Evaluation.Fraction != 0.01 || resolved.Evaluation.MaxRecords != 256 || resolved.Evaluation.MaxBytes != 1024*1024 || resolved.CheckpointEvery != 500 || resolved.EvaluateEvery != 500 || resolved.PlannedTokenCapacity != 4_096_000 {
		t.Fatalf("resolved = %+v", resolved)
	}
	encoded, err := json.Marshal(resolved)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(encoded, []byte(`"epochs":1`)) {
		t.Fatalf("resolved parameters do not persist the default epoch: %s", encoded)
	}
	zeroFloat := 0.0
	zeroInt := int64(0)
	buffer := 7
	bufferBytes := int64(4096)
	parameters.WeightDecay = &zeroFloat
	parameters.WarmupSteps = &zeroInt
	parameters.CheckpointEvery = &zeroInt
	parameters.EvaluateEvery = &zeroInt
	parameters.ShuffleBufferRecords = &buffer
	parameters.ShuffleBufferBytes = &bufferBytes
	resolved, err = ResolveParameters(parameters)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Optimizer.WeightDecay != 0 || resolved.Schedule.WarmupSteps != 0 || resolved.CheckpointEvery != 0 || resolved.EvaluateEvery != 0 || resolved.Data.ShuffleBufferRecords != 7 || resolved.Data.ShuffleBufferBytes != 4096 {
		t.Fatalf("overrides = %+v", resolved)
	}
	bad := parameters
	bad.Profile = "unknown"
	if _, err := ResolveParameters(bad); err == nil {
		t.Fatal("unknown profile accepted")
	}
	bad = parameters
	bad.Epochs = -1
	if _, err := ResolveParameters(bad); err == nil {
		t.Fatal("negative epochs accepted")
	}
}

func TestResolveParametersValidatesParallelismRequest(t *testing.T) {
	base := Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001}
	for _, value := range []string{"", ParallelismAuto, ParallelismData, ParallelismHybridSharded, ParallelismFullySharded} {
		parameters := base
		parameters.Parallelism = value
		if _, err := ResolveParameters(parameters); err != nil {
			t.Fatalf("parallelism %q: %v", value, err)
		}
	}
	base.Parallelism = "magic"
	if _, err := ResolveParameters(base); err == nil || !strings.Contains(err.Error(), "unsupported parallelism") {
		t.Fatalf("invalid parallelism error = %v", err)
	}
}

func TestResolveParametersSupportsTokenAndEpochBudgets(t *testing.T) {
	tokenBudget := Parameters{Tokens: 101, BatchSize: 2, SequenceLength: 8, LearningRate: 0.001}
	resolved, err := ResolveParameters(tokenBudget)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.RequestedTokens != 101 || resolved.Steps != 7 || resolved.PlannedTokenCapacity != 112 {
		t.Fatalf("token budget = %+v", resolved)
	}
	for _, invalid := range []Parameters{
		{Tokens: 100, Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001},
		{Tokens: 100, Epochs: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001},
		{BatchSize: 1, SequenceLength: 8, LearningRate: 0.001},
	} {
		if _, err := ResolvePlanningParameters(invalid); err == nil {
			t.Fatalf("invalid training budget accepted: %+v", invalid)
		}
	}
	epochBudget := Parameters{Epochs: 2, BatchSize: 2, SequenceLength: 8, LearningRate: 0.001}
	if _, err := ResolveParameters(epochBudget); err == nil {
		t.Fatal("unresolved epoch budget accepted for execution")
	}
	planning, err := ResolvePlanningParameters(epochBudget)
	if err != nil || planning.Epochs != 2 {
		t.Fatalf("epoch planning = %+v, err = %v", planning, err)
	}
	resolved, err = ResolveParametersForSteps(epochBudget, 9)
	if err != nil || resolved.Steps != 9 || resolved.PlannedTokenCapacity != 144 {
		t.Fatalf("derived epoch budget = %+v, err = %v", resolved, err)
	}
}

func TestBalancedProfilePinsCorpusBalancedDataAndEvaluation(t *testing.T) {
	resolved, err := ResolveParameters(Parameters{Profile: BalancedProfile, Steps: 10, BatchSize: 2, SequenceLength: 8, LearningRate: 0.001, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.ProfileSchema != 1 || resolved.Data.Order != "corpus-balanced-shuffle-v1" || resolved.Evaluation == nil || resolved.Evaluation.Selection != "stratified-lowest-sha256-v1" {
		t.Fatalf("balanced profile = %+v", resolved)
	}
}

func TestWeightedProfilePinsDeclaredCorpusWeights(t *testing.T) {
	parameters := Parameters{Profile: WeightedProfile, Steps: 10, BatchSize: 2, SequenceLength: 8, LearningRate: 0.001, Seed: 42, CorpusWeights: map[string]uint64{"corpus-a": 3, "corpus-b": 1}}
	resolved, err := ResolveParameters(parameters)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.ProfileSchema != 1 || resolved.Data.Order != "corpus-weighted-shuffle-v1" || !reflect.DeepEqual(resolved.Data.CorpusWeights, parameters.CorpusWeights) {
		t.Fatalf("weighted profile = %+v", resolved)
	}
	parameters.Profile = BalancedProfile
	if _, err := ResolveParameters(parameters); err == nil {
		t.Fatal("balanced profile accepted corpus_weights")
	}
}

func TestNumberedProfileAliasesResolveToCanonicalNames(t *testing.T) {
	for legacy, canonical := range map[string]string{
		"causal-pretrain-v1": ShuffledProfile,
		"causal-pretrain-v2": BalancedProfile,
		"causal-pretrain-v3": WeightedProfile,
	} {
		parameters := Parameters{Profile: legacy, Steps: 10, BatchSize: 2, SequenceLength: 8, LearningRate: 0.001, Seed: 42}
		if canonical == WeightedProfile {
			parameters.CorpusWeights = map[string]uint64{"corpus": 1}
		}
		resolved, err := ResolveParameters(parameters)
		if err != nil {
			t.Fatalf("alias %s: %v", legacy, err)
		}
		if resolved.Profile != canonical || resolved.ProfileSchema != ProfileSchema {
			t.Fatalf("alias %s resolved as %s schema %d", legacy, resolved.Profile, resolved.ProfileSchema)
		}
	}
	legacy := ResolvedParameters{Profile: "causal-pretrain-v3", ProfileSchema: 3}
	canonical := NormalizeResolvedParameters(legacy)
	if canonical.Profile != WeightedProfile || canonical.ProfileSchema != ProfileSchema {
		t.Fatalf("persisted profile normalization = %+v", canonical)
	}
}

func TestRecordPartitionPinsAndExcludesHeldOutRecords(t *testing.T) {
	var texts []string
	for index := 0; index < 100; index++ {
		texts = append(texts, fmt.Sprintf("record-%03d", index))
	}
	inputs := []Input{writeTrainingShard(t, texts)}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 16, LearningRate: 0.001, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	first, err := NewRecordPartition(inputs, parameters)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewRecordPartition(inputs, parameters)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first.Evaluation, second.Evaluation) || first.Evaluation.Records != 1 || first.Evaluation.TokenTargets == 0 || len(first.Evaluation.SHA256) != 64 {
		t.Fatalf("evaluation evidence = %+v / %+v", first.Evaluation, second.Evaluation)
	}
	trainingSource, err := first.TrainingRecords()
	if err != nil {
		t.Fatal(err)
	}
	trainingIDs := map[string]bool{}
	if err := trainingSource.Stream(context.Background(), func(value Record) error { trainingIDs[value.SelectionID] = true; return nil }); err != nil {
		t.Fatal(err)
	}
	evaluationIDs := map[string]bool{}
	if err := first.EvaluationRecords().Stream(context.Background(), func(value Record) error { evaluationIDs[value.SelectionID] = true; return nil }); err != nil {
		t.Fatal(err)
	}
	if len(trainingIDs) != 99 || len(evaluationIDs) != 1 {
		t.Fatalf("partition sizes = training %d, evaluation %d", len(trainingIDs), len(evaluationIDs))
	}
	for key := range evaluationIDs {
		if trainingIDs[key] {
			t.Fatalf("held-out record %s appears in training", key)
		}
	}
	if targets, err := first.TrainingByteTargets(context.Background()); err != nil || targets <= 0 {
		t.Fatalf("training targets = %d, err = %v", targets, err)
	}
}

func TestRecordFiltersApplyToPartitionTargetsAndTrainingStream(t *testing.T) {
	input := writeTrainingRows(t, []shard.Row{
		{SHA256: record.TextHash("keep"), Kind: record.KindPretrain, Text: "keep", Source: "source-a", SourceName: "project-a", License: "CC-BY-4.0", Lang: "en", Date: "2024", Tokens: 1},
		{SHA256: record.TextHash("keep unset"), Kind: record.KindPretrain, Text: "keep unset", Source: "source-a", SourceName: "project-a", License: "CC-BY-4.0", Date: "2024", Tokens: 1},
		{SHA256: record.TextHash("wrong language"), Kind: record.KindPretrain, Text: "wrong language", Source: "source-a", SourceName: "project-a", License: "CC-BY-4.0", Lang: "fr", Date: "2024", Tokens: 1},
		{SHA256: record.TextHash("wrong license"), Kind: record.KindPretrain, Text: "wrong license", Source: "source-a", SourceName: "project-a", License: "GPL-2.0-only", Lang: "en", Date: "2024", Tokens: 1},
	})
	input.Corpus = "example"
	input.RecordFilter = &corpus.RecordFilterPolicy{
		Schema:  corpus.RecordFilterSchema,
		Global:  &corpus.RecordFilter{Languages: &corpus.ValueFilter{Include: []string{"en"}, IncludeUnset: true}},
		Corpora: map[string]corpus.RecordFilter{"example": {Licenses: &corpus.ValueFilter{Include: []string{"CC-BY-*"}}, Date: &corpus.DateFilter{From: "2020"}}},
	}
	zeroFraction := 0.0
	zeroRecords := 0
	zeroBytes := int64(0)
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 42, EvaluationFraction: &zeroFraction, EvaluationMaxRecords: &zeroRecords, EvaluationMaxBytes: &zeroBytes})
	if err != nil {
		t.Fatal(err)
	}
	partition, err := NewRecordPartition([]Input{input}, parameters)
	if err != nil {
		t.Fatal(err)
	}
	source, err := partition.TrainingRecords()
	if err != nil {
		t.Fatal(err)
	}
	var texts []string
	if err := source.Stream(context.Background(), func(value Record) error {
		texts = append(texts, value.Text)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	sort.Strings(texts)
	if !reflect.DeepEqual(texts, []string{"keep", "keep unset"}) {
		t.Fatalf("filtered training texts = %v", texts)
	}
	if targets, err := CountByteTargets(context.Background(), []Input{input}); err != nil || targets != int64(len("keep")+len("keep unset")+1) {
		t.Fatalf("filtered byte targets = %d, err = %v", targets, err)
	}
}

type countingTokenCodec struct {
	counts int
}

func (codec *countingTokenCodec) Count(text string) int {
	codec.counts++
	return len([]byte(text))
}

func (*countingTokenCodec) Encode(text string) []int   { return byteCodec{}.Encode(text) }
func (*countingTokenCodec) Decode(tokens []int) string { return byteCodec{}.Decode(tokens) }

func TestRecordPartitionTokenizesAndCachesOnlySelectedRecords(t *testing.T) {
	texts := make([]string, 1000)
	for index := range texts {
		texts[index] = fmt.Sprintf("record-%04d", index)
	}
	input := writeTrainingShard(t, texts)
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 16, LearningRate: 0.001, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	codec := &countingTokenCodec{}
	partition, err := NewRecordPartitionContextWithTokenizer(context.Background(), []Input{input}, parameters, codec, nil)
	if err != nil {
		t.Fatal(err)
	}
	if codec.counts != int(partition.Evaluation.Records) || codec.counts != 10 {
		t.Fatalf("tokenizer calls = %d, evaluation records = %d", codec.counts, partition.Evaluation.Records)
	}
	if err := os.Remove(input.Path); err != nil {
		t.Fatal(err)
	}
	var cached int
	if err := partition.EvaluationRecords().Stream(context.Background(), func(Record) error {
		cached++
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if cached != codec.counts {
		t.Fatalf("cached evaluation records = %d, want %d", cached, codec.counts)
	}
}

func TestRecordPartitionHonorsCanceledContext(t *testing.T) {
	inputs := []Input{writeTrainingShard(t, []string{"one", "two"})}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewRecordPartitionContext(ctx, inputs, parameters, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("partition cancellation error = %v", err)
	}
}

func TestRecordPartitionReportsScanProgress(t *testing.T) {
	inputs := []Input{
		writeTrainingShard(t, []string{"one", "two"}),
		writeTrainingShard(t, []string{"three"}),
	}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	var events []PartitionProgress
	if _, err := NewRecordPartitionWithProgress(inputs, parameters, func(event PartitionProgress) {
		events = append(events, event)
	}); err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 {
		t.Fatalf("progress events = %+v", events)
	}
	last := events[len(events)-1]
	if last.CurrentShard != 2 || last.TotalShards != 2 || last.Records != 3 || last.Bytes != last.TotalBytes || last.TotalBytes <= 0 {
		t.Fatalf("final progress = %+v", last)
	}
}

func TestCanonicalRecordSourceIsDeterministicAndComplete(t *testing.T) {
	inputs := []Input{
		writeTrainingShard(t, []string{"one", "two", "three"}),
		writeTrainingShard(t, []string{"four", "five", "six"}),
	}
	buffer := 2
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 42, ShuffleBufferRecords: &buffer})
	if err != nil {
		t.Fatal(err)
	}
	first := collectRecords(t, inputs, parameters)
	second := collectRecords(t, inputs, parameters)
	if !reflect.DeepEqual(first, second) || len(first) != 6 {
		t.Fatalf("first = %v, second = %v", first, second)
	}
	parameters.Seed = 43
	third := collectRecords(t, inputs, parameters)
	if reflect.DeepEqual(first, third) {
		t.Fatalf("different seed produced the same order: %v", first)
	}
	sorted := append([]string(nil), first...)
	other := append([]string(nil), third...)
	sort.Strings(sorted)
	sort.Strings(other)
	if !reflect.DeepEqual(sorted, other) {
		t.Fatalf("record sets differ: %v, %v", sorted, other)
	}
}

func TestBalancedRecordSourceInterleavesDeclaredCorpora(t *testing.T) {
	first := writeTrainingShard(t, []string{"a1", "a2", "a3", "a4"})
	first.Corpus = "corpus-a"
	second := writeTrainingShard(t, []string{"b1", "b2", "b3", "b4"})
	second.Corpus = "corpus-b"
	parameters, err := ResolveParameters(Parameters{Profile: BalancedProfile, Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	source, err := NewCanonicalRecordSource([]Input{first, second}, parameters)
	if err != nil {
		t.Fatal(err)
	}
	var corpora []string
	if err := source.Stream(context.Background(), func(value Record) error {
		corpora = append(corpora, value.Corpus)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	want := []string{"corpus-a", "corpus-b", "corpus-a", "corpus-b", "corpus-a", "corpus-b", "corpus-a", "corpus-b"}
	if !reflect.DeepEqual(corpora, want) {
		t.Fatalf("corpus order = %v, want %v", corpora, want)
	}
}

func TestBalancedRecordSourceAccountsForTokenLength(t *testing.T) {
	first := writeTrainingShard(t, []string{strings.Repeat("a", 20), strings.Repeat("a", 20)})
	first.Corpus = "corpus-a"
	second := writeTrainingShard(t, []string{"b", "b", "b", "b", "b", "b"})
	second.Corpus = "corpus-b"
	parameters, err := ResolveParameters(Parameters{Profile: BalancedProfile, Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	source, err := NewCanonicalRecordSource([]Input{first, second}, parameters)
	if err != nil {
		t.Fatal(err)
	}
	var corpora []string
	if err := source.Stream(context.Background(), func(value Record) error {
		corpora = append(corpora, value.Corpus)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(corpora) < 4 || !reflect.DeepEqual(corpora[:4], []string{"corpus-a", "corpus-b", "corpus-b", "corpus-b"}) {
		t.Fatalf("token-balanced prefix = %v", corpora)
	}
}

func TestWeightedRecordSourceHonorsTokenRatios(t *testing.T) {
	first := writeTrainingShard(t, []string{"a1", "a2", "a3", "a4"})
	first.Corpus = "corpus-a"
	second := writeTrainingShard(t, []string{"b1", "b2", "b3", "b4"})
	second.Corpus = "corpus-b"
	parameters, err := ResolveParameters(Parameters{Profile: WeightedProfile, Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 42, CorpusWeights: map[string]uint64{"corpus-a": 3, "corpus-b": 1}})
	if err != nil {
		t.Fatal(err)
	}
	source, err := NewCanonicalRecordSource([]Input{first, second}, parameters)
	if err != nil {
		t.Fatal(err)
	}
	var corpora []string
	if err := source.Stream(context.Background(), func(value Record) error {
		corpora = append(corpora, value.Corpus)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(corpora) < 5 || !reflect.DeepEqual(corpora[:5], []string{"corpus-a", "corpus-b", "corpus-a", "corpus-a", "corpus-a"}) {
		t.Fatalf("weighted corpus prefix = %v", corpora)
	}
}

func TestBalancedEvaluationIncludesEveryCorpus(t *testing.T) {
	first := writeTrainingShard(t, []string{"a1", "a2", "a3", "a4"})
	first.Corpus = "corpus-a"
	second := writeTrainingShard(t, []string{"b1", "b2", "b3", "b4"})
	second.Corpus = "corpus-b"
	fraction := 0.5
	maximum := 4
	parameters, err := ResolveParameters(Parameters{Profile: BalancedProfile, Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 42, EvaluationFraction: &fraction, EvaluationMaxRecords: &maximum})
	if err != nil {
		t.Fatal(err)
	}
	partition, err := NewRecordPartition([]Input{first, second}, parameters)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]int{}
	if err := partition.EvaluationRecords().Stream(context.Background(), func(value Record) error {
		seen[value.Corpus]++
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if seen["corpus-a"] != 2 || seen["corpus-b"] != 2 {
		t.Fatalf("stratified evaluation = %v", seen)
	}
}

func TestCanonicalRecordSourceHonorsByteBound(t *testing.T) {
	inputs := []Input{writeTrainingShard(t, []string{"one", "two", "three"})}
	bufferRecords := 100
	bufferBytes := int64(1)
	parameters, err := ResolveParameters(Parameters{
		Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001,
		Seed: 42, ShuffleBufferRecords: &bufferRecords, ShuffleBufferBytes: &bufferBytes,
	})
	if err != nil {
		t.Fatal(err)
	}
	got := collectRecords(t, inputs, parameters)
	want := []string{record.TextHash("one"), record.TextHash("two"), record.TextHash("three")}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("byte-bounded order = %v, want %v", got, want)
	}
}

func TestCountByteTargetsUsesUTF8BytesAndEOS(t *testing.T) {
	inputs := []Input{writeTrainingShard(t, []string{"A", "é"})}
	// [A, EOS, 0xc3, 0xa9, EOS] has four next-token targets.
	targets, err := CountByteTargets(context.Background(), inputs)
	if err != nil {
		t.Fatal(err)
	}
	if targets != 4 {
		t.Fatalf("byte targets = %d, want 4", targets)
	}
}

func TestByteTargetsAndRecordSourceRepeatExactEpochs(t *testing.T) {
	inputs := []Input{writeTrainingShard(t, []string{"A", "é"})}
	oneEpoch, err := CountByteTargets(context.Background(), inputs)
	if err != nil {
		t.Fatal(err)
	}
	targets, err := ByteTargetsForEpochs(oneEpoch, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Each epoch contains five tokens including EOS. Across two continuous
	// epochs only the first token lacks a prediction target: 5*2-1 = 9.
	if targets != 9 {
		t.Fatalf("two-epoch targets = %d, want 9", targets)
	}
	parameters, err := ResolveParameters(Parameters{Epochs: 2, Steps: 1, BatchSize: 1, SequenceLength: 16, LearningRate: 0.001, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	got := collectRecords(t, inputs, parameters)
	if len(got) != 4 {
		t.Fatalf("two-epoch record count = %d, want 4: %v", len(got), got)
	}
}

func TestTrainingStepCapacityAccountsForHeldOutRecords(t *testing.T) {
	inputs := []Input{writeTrainingShard(t, []string{strings.Repeat("a", 20), strings.Repeat("b", 20)})}
	parameters, err := ResolveParameters(Parameters{Steps: 3, BatchSize: 2, SequenceLength: 8, LearningRate: 0.001, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	partition, err := NewRecordPartition(inputs, parameters)
	if err != nil {
		t.Fatal(err)
	}
	steps, sufficient, err := partition.TrainingStepCapacity(context.Background(), 3)
	if err != nil {
		t.Fatal(err)
	}
	if sufficient || steps != 2 {
		t.Fatalf("capacity = %d, sufficient = %t; want 2, false", steps, sufficient)
	}
	if steps, sufficient, err = partition.TrainingStepCapacity(context.Background(), 2); err != nil || !sufficient || steps != 2 {
		t.Fatalf("bounded capacity = %d, sufficient = %t, err = %v", steps, sufficient, err)
	}
	if steps, err = partition.TrainingSteps(context.Background()); err != nil || steps != 2 {
		t.Fatalf("derived steps = %d, err = %v", steps, err)
	}
}

func TestStagePreflightReconstructsTheSamePartition(t *testing.T) {
	inputs := []Input{writeTrainingShard(t, []string{"alpha record", "beta record", "gamma record", "delta record"})}
	fraction := 0.5
	maxRecords := 2
	maxBytes := int64(1024)
	parameters, err := ResolveParameters(Parameters{
		Steps: 2, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 7,
		EvaluationFraction: &fraction, EvaluationMaxRecords: &maxRecords, EvaluationMaxBytes: &maxBytes,
	})
	if err != nil {
		t.Fatal(err)
	}
	partition, err := NewRecordPartition(inputs, parameters)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := partition.Preflight(strings.Repeat("a", 64), parameters, false)
	restored, err := NewRecordPartitionFromPreflight(context.Background(), inputs, parameters, byteCodec{}, "causal-language-modeling", ConversationTransform{}, snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if restored.Evaluation != partition.Evaluation {
		t.Fatalf("restored evaluation = %+v, want %+v", restored.Evaluation, partition.Evaluation)
	}
	originalSource, err := partition.TrainingRecords()
	originalRecords := collectRecordSource(t, originalSource, err)
	restoredSource, err := restored.TrainingRecords()
	restoredRecords := collectRecordSource(t, restoredSource, err)
	if !reflect.DeepEqual(originalRecords, restoredRecords) {
		t.Fatalf("restored training records differ: %+v / %+v", originalRecords, restoredRecords)
	}
	snapshot.SelectedRecords = append(snapshot.SelectedRecords, snapshot.SelectedRecords[0])
	if _, err := NewRecordPartitionFromPreflight(context.Background(), inputs, parameters, byteCodec{}, "causal-language-modeling", ConversationTransform{}, snapshot); err == nil {
		t.Fatal("duplicate preflight selection was accepted")
	}
}

func TestStagePreflightRecordsCorporaEliminatedByFilters(t *testing.T) {
	filtered := writeTrainingRows(t, []shard.Row{{
		SHA256: record.TextHash("filtered"), Kind: record.KindPretrain, Text: "filtered", Source: "fixture", License: "CC0-1.0", Lang: "fr", Tokens: 1,
	}})
	filtered.Corpus = "filtered-corpus"
	filtered.RecordFilter = &corpus.RecordFilterPolicy{Schema: corpus.RecordFilterSchema, Global: &corpus.RecordFilter{Languages: &corpus.ValueFilter{Include: []string{"en"}}}}
	retained := writeTrainingShard(t, []string{"retained one", "retained two"})
	retained.Corpus = "retained-corpus"
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	partition, err := NewRecordPartition([]Input{filtered, retained}, parameters)
	if err != nil {
		t.Fatal(err)
	}
	if got := partition.ZeroEligibleCorpora(); !reflect.DeepEqual(got, []string{"filtered-corpus"}) {
		t.Fatalf("zero eligible corpora = %v", got)
	}
	snapshot := partition.Preflight(strings.Repeat("a", 64), parameters, false)
	if snapshot.EligibleRecords["filtered-corpus"] != 0 || snapshot.EligibleRecords["retained-corpus"] != 1 {
		t.Fatalf("eligible records = %v", snapshot.EligibleRecords)
	}
	restored, err := NewRecordPartitionFromPreflight(context.Background(), []Input{filtered, retained}, parameters, byteCodec{}, "causal-language-modeling", ConversationTransform{}, snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if got := restored.ZeroEligibleCorpora(); !reflect.DeepEqual(got, []string{"filtered-corpus"}) {
		t.Fatalf("restored zero eligible corpora = %v", got)
	}
	snapshot.EligibleRecords["retained-corpus"] = -1
	if err := snapshot.Validate(); err == nil {
		t.Fatal("negative eligible record count was accepted")
	}
}

func collectRecordSource(t *testing.T, source RecordSource, err error) []string {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	var records []string
	if err := source.Stream(context.Background(), func(value Record) error {
		records = append(records, value.SelectionID)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return records
}

func TestTrainingStepCapacityAccountsForRecordFilters(t *testing.T) {
	input := writeTrainingRows(t, []shard.Row{
		{SHA256: record.TextHash(strings.Repeat("k", 20)), Kind: record.KindPretrain, Text: strings.Repeat("k", 20), Source: "fixture", License: "CC0-1.0", Lang: "en", Tokens: 1},
		{SHA256: record.TextHash(strings.Repeat("x", 200)), Kind: record.KindPretrain, Text: strings.Repeat("x", 200), Source: "fixture", License: "CC0-1.0", Lang: "fr", Tokens: 1},
	})
	input.Corpus = "example"
	input.RecordFilter = &corpus.RecordFilterPolicy{Schema: corpus.RecordFilterSchema, Global: &corpus.RecordFilter{Languages: &corpus.ValueFilter{Include: []string{"en"}}}}
	zeroFraction, zeroRecords, zeroBytes := 0.0, 0, int64(0)
	parameters, err := ResolveParameters(Parameters{
		Steps: 4, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 7,
		EvaluationFraction: &zeroFraction, EvaluationMaxRecords: &zeroRecords, EvaluationMaxBytes: &zeroBytes,
	})
	if err != nil {
		t.Fatal(err)
	}
	partition, err := NewRecordPartition([]Input{input}, parameters)
	if err != nil {
		t.Fatal(err)
	}
	steps, sufficient, err := partition.TrainingStepCapacity(context.Background(), 4)
	if err != nil {
		t.Fatal(err)
	}
	if sufficient || steps != 3 {
		t.Fatalf("filtered capacity = %d, sufficient = %t; want 3, false", steps, sufficient)
	}
}

func TestTrainingStepCapacityAccountsForAssistantLossMasks(t *testing.T) {
	payload, err := record.EncodeConversation(record.Conversation{Messages: []record.Message{{Role: "user", Content: strings.Repeat("u", 20)}, {Role: "assistant", Content: strings.Repeat("a", 20)}}})
	if err != nil {
		t.Fatal(err)
	}
	input := writeTrainingRows(t, []shard.Row{{SHA256: record.TextHash(payload), Kind: record.KindConversation, Text: payload, Source: "fixture", License: "CC0-1.0", Tokens: 1}})
	zeroFraction, zeroRecords, zeroBytes := 0.0, 0, int64(0)
	parameters, err := ResolveParameters(Parameters{
		Steps: 3, BatchSize: 1, SequenceLength: 16, LearningRate: 0.001, Seed: 7,
		EvaluationFraction: &zeroFraction, EvaluationMaxRecords: &zeroRecords, EvaluationMaxBytes: &zeroBytes,
	})
	if err != nil {
		t.Fatal(err)
	}
	partition, err := NewRecordPartitionContextWithTransform(context.Background(), []Input{input}, parameters, byteCodec{}, "assistant-response-modeling", ConversationTransform{Template: ConversationTemplateUserAssistantV1, SupervisedRoles: []string{"assistant"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	steps, sufficient, err := partition.TrainingStepCapacity(context.Background(), 3)
	if err != nil {
		t.Fatal(err)
	}
	if sufficient || steps != 2 {
		t.Fatalf("assistant capacity = %d, sufficient = %t; want 2, false", steps, sufficient)
	}
}

func TestWorkerProtocolStreamsBeginRecordsEndAndValidatesOutput(t *testing.T) {
	inputs := []Input{writeTrainingShard(t, []string{"one", "two"})}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	begin := WorkerBegin{RunID: "run", Stage: "pretrain", Objective: "causal-language-modeling", ArchitectureSHA256: strings.Repeat("a", 64), Architecture: json.RawMessage(`{"family":"decoder-transformer"}`), Parameters: parameters}
	partition, err := NewRecordPartition(inputs, parameters)
	if err != nil {
		t.Fatal(err)
	}
	trainingSource, err := partition.TrainingRecords()
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteWorkerInput(context.Background(), &encoded, begin, trainingSource, partition.EvaluationRecords()); err != nil {
		t.Fatal(err)
	}
	var kinds []string
	var decodedBegin WorkerBegin
	scanner := bufio.NewScanner(bytes.NewReader(encoded.Bytes()))
	for scanner.Scan() {
		var frame WorkerInputFrame
		if err := json.Unmarshal(scanner.Bytes(), &frame); err != nil {
			t.Fatal(err)
		}
		kinds = append(kinds, frame.Kind)
		if frame.Begin != nil {
			decodedBegin = *frame.Begin
		}
	}
	if !reflect.DeepEqual(kinds, []string{"begin", "evaluation_record", "record", "end"}) {
		t.Fatalf("frame kinds = %v", kinds)
	}
	if !reflect.DeepEqual(decodedBegin, begin) {
		t.Fatalf("worker begin lost compose settings:\n got  %+v\n want %+v", decodedBegin, begin)
	}

	output := "{\"kind\":\"event\",\"schema\":1,\"event\":{\"kind\":\"progress\",\"step\":1}}\n" +
		"{\"kind\":\"complete\",\"schema\":1,\"observation\":{\"simulated\":false,\"steps\":1,\"consumed_tokens\":8,\"artifacts\":[]}}\n"
	count := 0
	if err := ReadWorkerOutput(strings.NewReader(output), func(WorkerOutputFrame) error { count++; return nil }); err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("output frames = %d", count)
	}
	if err := ReadWorkerOutput(strings.NewReader(`{"kind":"complete","schema":2,"observation":{}}`+"\n"), func(WorkerOutputFrame) error { return nil }); err == nil {
		t.Fatal("unsupported protocol schema accepted")
	}
	if err := ReadWorkerOutput(strings.NewReader(`{"kind":"event","schema":1,"event":{"kind":"mystery"}}`+"\n"), func(WorkerOutputFrame) error { return nil }); err == nil {
		t.Fatal("unsupported event kind accepted")
	}
}

func TestWorkerInputStopsTrainingRecordsAfterTargetSignal(t *testing.T) {
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	begin := WorkerBegin{RunID: "run", Stage: "pretrain", Objective: "causal-language-modeling", Parameters: parameters}
	stop := make(chan struct{})
	streamed := 0
	records := recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
		for position := 0; position < 100; position++ {
			if err := consume(Record{ID: fmt.Sprintf("record-%d", position), Text: "training text"}); err != nil {
				return err
			}
			streamed++
			if position == 0 {
				close(stop)
			}
		}
		return nil
	})
	var encoded bytes.Buffer
	if err := writeWorkerInputUntil(context.Background(), &encoded, begin, records, nil, stop); err != nil {
		t.Fatal(err)
	}
	if streamed != 1 {
		t.Fatalf("streamed %d records after target signal, want 1", streamed)
	}
	if !strings.Contains(encoded.String(), `"kind":"end"`) {
		t.Fatalf("worker stream did not terminate cleanly: %s", encoded.String())
	}
}

func TestResolveParametersPinsGradientAccumulation(t *testing.T) {
	resolved, err := ResolveParameters(Parameters{Steps: 2, BatchSize: 8, GradientAccumulation: 4, SequenceLength: 16, LearningRate: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.GradientAccumulation != 4 || resolved.PlannedTokenCapacity != 256 {
		t.Fatalf("resolved accumulation = %+v", resolved)
	}
	defaults, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 2, SequenceLength: 8, LearningRate: 0.001})
	if err != nil || defaults.GradientAccumulation != 1 {
		t.Fatalf("default accumulation = %+v, err=%v", defaults, err)
	}
	for _, accumulation := range []int64{-1, 3, 9} {
		_, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 8, GradientAccumulation: accumulation, SequenceLength: 8, LearningRate: 0.001})
		if err == nil {
			t.Fatalf("accepted accumulation %d for batch 8", accumulation)
		}
	}
}

func TestResolveParametersPinsExecutionControls(t *testing.T) {
	resolved, err := ResolveParameters(Parameters{
		Steps: 2, BatchSize: 2, SequenceLength: 16, LearningRate: 0.001,
		ComputePrecision: "bfloat16", ActivationCheckpointing: true, Compile: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.ComputePrecision != "bfloat16" || !resolved.ActivationCheckpointing || !resolved.Compile {
		t.Fatalf("execution controls = %+v", resolved)
	}
	defaults, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001})
	if err != nil || defaults.ComputePrecision != "auto" {
		t.Fatalf("default execution controls = %+v, err=%v", defaults, err)
	}
	if _, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, ComputePrecision: "fp8"}); err == nil {
		t.Fatal("accepted unsupported compute precision")
	}
}

func TestResolveParametersPinsDistributionPolicy(t *testing.T) {
	resolved, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, DistributionPolicy: "distributable"})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.DistributionPolicy != "distributable" {
		t.Fatalf("distribution policy = %q", resolved.DistributionPolicy)
	}
	if _, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, DistributionPolicy: "trust-me"}); err == nil {
		t.Fatal("accepted unsupported distribution policy")
	}
}

func TestResolveParametersPinsOptimizerAndWarmdownSchedule(t *testing.T) {
	warmdown := int64(40)
	minimum := 0.0
	resolved, err := ResolveParameters(Parameters{
		Steps: 100, BatchSize: 8, SequenceLength: 128, LearningRate: 0.001,
		Optimizer: "muon-adamw", Schedule: "warmup-stable-warmdown",
		WarmdownSteps: &warmdown, MinimumRateRatio: &minimum,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Optimizer.Name != "muon-adamw" || resolved.Schedule.Name != "warmup-stable-warmdown" || resolved.Schedule.WarmdownSteps != 40 || resolved.Schedule.MinimumRateRatio != 0 {
		t.Fatalf("resolved recipe = %+v / %+v", resolved.Optimizer, resolved.Schedule)
	}
	for _, parameters := range []Parameters{
		{Steps: 10, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Optimizer: "unknown"},
		{Steps: 10, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Schedule: "unknown"},
		{Steps: 10, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Schedule: "warmup-stable-warmdown", WarmupSteps: testInt64Pointer(6), WarmdownSteps: testInt64Pointer(5)},
	} {
		if _, err := ResolveParameters(parameters); err == nil {
			t.Fatalf("invalid recipe accepted: %+v", parameters)
		}
	}
}

func testInt64Pointer(value int64) *int64 { return &value }

func TestValidateBatchTopologyUsesPhysicalMicroBatch(t *testing.T) {
	parameters := ResolvedParameters{BatchSize: 64, GradientAccumulation: 4}
	if err := ValidateBatchTopology(parameters, 8); err != nil {
		t.Fatal(err)
	}
	if err := ValidateBatchTopology(parameters, 32); err == nil {
		t.Fatal("accepted a world size larger than the global micro-batch")
	}
	parameters.GradientAccumulation = 8
	if err := ValidateBatchTopology(parameters, 8); err != nil {
		t.Fatal(err)
	}
	if err := ValidateBatchTopology(parameters, 4); err != nil {
		t.Fatal(err)
	}
}

func TestPreparedSequencesAreDeterministicallyPartitionedByNode(t *testing.T) {
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 4, SequenceLength: 2, LearningRate: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	record := Record{ID: "one", Tokens: []int{10, 11, 12, 13, 14, 15, 16, 17}, LossMask: make([]bool, 9), Corpus: "corpus"}
	for index := range record.LossMask {
		record.LossMask[index] = true
	}
	seen := map[int64]PreparedSequence{}
	for node := 0; node < 2; node++ {
		var encoded bytes.Buffer
		begin := WorkerBegin{
			Parameters: parameters, Tokenizer: TokenizerSpec{EOSID: 2}, DataNodeRank: node,
			Parallelism: Parallelism{WorldSize: 4, GPUsPerNode: 2, DataPlane: DataPlaneNodeLocal},
		}
		if err := WriteWorkerInput(context.Background(), &encoded, begin, staticRecordSource{record}, nil); err != nil {
			t.Fatal(err)
		}
		decoder := json.NewDecoder(&encoded)
		boundaries := 0
		for {
			var frame WorkerInputFrame
			if err := decoder.Decode(&frame); err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				t.Fatal(err)
			}
			if frame.Kind == "sequence" {
				if _, duplicate := seen[frame.Sequence.Ordinal]; duplicate {
					t.Fatalf("sequence %d assigned to more than one node", frame.Sequence.Ordinal)
				}
				seen[frame.Sequence.Ordinal] = *frame.Sequence
			}
			if frame.Kind == "micro_batch_end" {
				boundaries++
			}
		}
		if boundaries != 1 {
			t.Fatalf("node %d received %d optimizer boundaries, want 1", node, boundaries)
		}
	}
	if len(seen) != 4 {
		t.Fatalf("prepared partitions cover %d sequences, want 4", len(seen))
	}
	if !reflect.DeepEqual(seen[0].Tokens, []int{10, 11, 12}) || !reflect.DeepEqual(seen[3].Tokens, []int{16, 17, 2}) {
		t.Fatalf("prepared packing changed canonical sequence order: %+v", seen)
	}
	for ordinal, sequence := range seen {
		if sequence.Consumption["corpus"] != 2 {
			t.Fatalf("sequence %d consumption = %+v", ordinal, sequence.Consumption)
		}
	}
}

func collectRecords(t *testing.T, inputs []Input, parameters ResolvedParameters) []string {
	t.Helper()
	source, err := NewCanonicalRecordSource(inputs, parameters)
	if err != nil {
		t.Fatal(err)
	}
	var result []string
	if err := source.Stream(context.Background(), func(record Record) error {
		result = append(result, record.ID)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return result
}

func writeTrainingShard(t *testing.T, texts []string) Input {
	t.Helper()
	rows := make([]shard.Row, 0, len(texts))
	for _, text := range texts {
		rows = append(rows, shard.Row{SHA256: record.TextHash(text), Kind: record.KindPretrain, Text: text, Source: "fixture", License: "CC0-1.0", Tokens: 1})
	}
	return writeTrainingRows(t, rows)
}

func writeTrainingRows(t *testing.T, rows []shard.Row) Input {
	t.Helper()
	var encoded bytes.Buffer
	writer := parquet.NewGenericWriter[shard.Row](&encoded)
	if _, err := writer.Write(rows); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	data := encoded.Bytes()
	digestArray := sha256.Sum256(data)
	digest := hex.EncodeToString(digestArray[:])
	path := filepath.Join(t.TempDir(), digest+".parquet")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return Input{Path: path, SHA256: digest, Bytes: int64(len(data)), Records: int64(len(rows))}
}
