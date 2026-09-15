// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"bytes"
	"container/heap"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/openwaldo/waldo/internal/record"
	"github.com/openwaldo/waldo/internal/shard"
)

type Record struct {
	SelectionID  string               `json:"selection_id"`
	ID           string               `json:"id"`
	Text         string               `json:"text"`
	Conversation *record.Conversation `json:"conversation,omitempty"`
	Source       string               `json:"source"`
	License      string               `json:"license"`
	Language     string               `json:"language,omitempty"`
	Corpus       string               `json:"corpus,omitempty"`
	Tokens       []int                `json:"tokens,omitempty"`
	LossMask     []bool               `json:"loss_mask,omitempty"`
	decodeErr    error
}

type RecordPartition struct {
	Evaluation        EvaluationSet
	selected          map[string]bool
	eligibleRecords   map[string]int64
	selectionSummary  RecordSelectionSummary
	evaluationRecords []Record
	inputs            []Input
	parameters        ResolvedParameters
	codec             TokenCodec
	objective         string
	conversation      ConversationTransform
}

// RecordSelectionSummary records the result of applying stage record filters.
// IncludedRecords includes held-out evaluation records; the records available
// to the trainer are therefore IncludedRecords minus HeldOutRecords.
type RecordSelectionSummary struct {
	InputRecords     int64            `json:"input_records"`
	IncludedRecords  int64            `json:"included_records"`
	SkippedRecords   int64            `json:"skipped_records"`
	HeldOutRecords   int64            `json:"held_out_records,omitempty"`
	IncludedLicenses map[string]int64 `json:"included_licenses,omitempty"`
	SkippedLicenses  map[string]int64 `json:"skipped_licenses,omitempty"`
}

const (
	StagePreflightKind   = "openwaldo-stage-preflight"
	StagePreflightSchema = 1
)

// StagePreflight is the immutable result of the expensive, deterministic
// record-partition scan. The model run BOM pins the serialized artifact.
type StagePreflight struct {
	Kind             string                 `json:"kind"`
	Schema           int                    `json:"schema"`
	IdentitySHA256   string                 `json:"identity_sha256"`
	Evaluation       EvaluationSet          `json:"evaluation"`
	SelectedRecords  []string               `json:"selected_records"`
	EligibleRecords  map[string]int64       `json:"eligible_records,omitempty"`
	SelectionSummary RecordSelectionSummary `json:"selection_summary,omitempty"`
	Parameters       ResolvedParameters     `json:"parameters"`
	CapacityVerified bool                   `json:"capacity_verified,omitempty"`
}

func (snapshot StagePreflight) Validate() error {
	if snapshot.Kind != StagePreflightKind || snapshot.Schema != StagePreflightSchema {
		return fmt.Errorf("unsupported stage preflight artifact %q schema %d", snapshot.Kind, snapshot.Schema)
	}
	identity, err := hex.DecodeString(snapshot.IdentitySHA256)
	if err != nil || len(identity) != sha256.Size || snapshot.IdentitySHA256 != strings.ToLower(snapshot.IdentitySHA256) {
		return fmt.Errorf("stage preflight identity SHA-256 is invalid")
	}
	if int64(len(snapshot.SelectedRecords)) != snapshot.Evaluation.Records {
		return fmt.Errorf("stage preflight contains %d selected record IDs but declares %d", len(snapshot.SelectedRecords), snapshot.Evaluation.Records)
	}
	for index, key := range snapshot.SelectedRecords {
		if index > 0 && key <= snapshot.SelectedRecords[index-1] {
			return fmt.Errorf("stage preflight selected record IDs are not unique and sorted")
		}
	}
	for corpus, records := range snapshot.EligibleRecords {
		if strings.TrimSpace(corpus) == "" || records < 0 {
			return fmt.Errorf("stage preflight eligible record count is invalid")
		}
	}
	if err := snapshot.SelectionSummary.validate(); err != nil {
		return fmt.Errorf("stage preflight selection summary: %w", err)
	}
	return nil
}

// Preflight returns a portable snapshot of this partition. Selection IDs are
// sorted so the artifact has a stable identity independent of map iteration.
func (partition RecordPartition) Preflight(identity string, parameters ResolvedParameters, capacityVerified bool) StagePreflight {
	selected := make([]string, 0, len(partition.selected))
	for key := range partition.selected {
		selected = append(selected, key)
	}
	sort.Strings(selected)
	return StagePreflight{
		Kind: StagePreflightKind, Schema: StagePreflightSchema, IdentitySHA256: identity,
		Evaluation: partition.Evaluation, SelectedRecords: selected, EligibleRecords: cloneRecordCounts(partition.eligibleRecords), SelectionSummary: partition.SelectionSummary(), Parameters: parameters,
		CapacityVerified: capacityVerified,
	}
}

// SelectionSummary returns a detached copy of the stage's filter accounting.
func (partition RecordPartition) SelectionSummary() RecordSelectionSummary {
	result := partition.selectionSummary
	result.IncludedLicenses = cloneRecordCounts(result.IncludedLicenses)
	result.SkippedLicenses = cloneRecordCounts(result.SkippedLicenses)
	return result
}

func (summary RecordSelectionSummary) validate() error {
	if summary.InputRecords == 0 && summary.IncludedRecords == 0 && summary.SkippedRecords == 0 && summary.HeldOutRecords == 0 && len(summary.IncludedLicenses) == 0 && len(summary.SkippedLicenses) == 0 {
		return nil // Legacy schema-1 preflight without selection accounting.
	}
	if summary.InputRecords < 0 || summary.IncludedRecords < 0 || summary.SkippedRecords < 0 || summary.HeldOutRecords < 0 || summary.IncludedRecords+summary.SkippedRecords != summary.InputRecords || summary.HeldOutRecords > summary.IncludedRecords {
		return fmt.Errorf("record counts are inconsistent")
	}
	for label, values := range map[string]map[string]int64{"included": summary.IncludedLicenses, "skipped": summary.SkippedLicenses} {
		for license, records := range values {
			if strings.TrimSpace(license) == "" || records <= 0 {
				return fmt.Errorf("%s license count is invalid", label)
			}
		}
	}
	return nil
}

// ZeroEligibleCorpora returns selected corpora for which the stage filters and
// held-out partition leave no training records. A nil result means the pinned
// preflight predates this evidence and cannot determine that without a scan.
func (partition RecordPartition) ZeroEligibleCorpora() []string {
	if partition.eligibleRecords == nil {
		return nil
	}
	var corpora []string
	for corpus, records := range partition.eligibleRecords {
		if records == 0 {
			corpora = append(corpora, corpus)
		}
	}
	sort.Strings(corpora)
	return corpora
}

// EligibleRecords returns the post-filter, post-held-out training record count
// for each selected corpus, when that evidence is available.
func (partition RecordPartition) EligibleRecords() map[string]int64 {
	return cloneRecordCounts(partition.eligibleRecords)
}

func cloneRecordCounts(values map[string]int64) map[string]int64 {
	if values == nil {
		return nil
	}
	cloned := make(map[string]int64, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

// NewRecordPartitionFromPreflight reconstructs a partition by reading only
// the pinned held-out rows, avoiding a full metadata selection scan.
func NewRecordPartitionFromPreflight(ctx context.Context, inputs []Input, parameters ResolvedParameters, codec TokenCodec, objective string, conversation ConversationTransform, snapshot StagePreflight) (RecordPartition, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if codec == nil {
		return RecordPartition{}, fmt.Errorf("record partition requires a tokenizer")
	}
	if err := snapshot.Validate(); err != nil {
		return RecordPartition{}, err
	}
	ordered := orderedInputs(inputs)
	positions := make(map[string]int, len(ordered))
	for index, input := range ordered {
		if _, exists := positions[input.SHA256]; exists {
			return RecordPartition{}, fmt.Errorf("stage preflight has duplicate input shard %s", input.SHA256)
		}
		positions[input.SHA256] = index
	}
	selected := make([]evaluationCandidate, 0, len(snapshot.SelectedRecords))
	for _, key := range snapshot.SelectedRecords {
		separator := strings.LastIndexByte(key, ':')
		if separator != 64 {
			return RecordPartition{}, fmt.Errorf("stage preflight selected record ID is invalid")
		}
		inputPosition, ok := positions[key[:separator]]
		if !ok {
			return RecordPartition{}, fmt.Errorf("stage preflight selected record references an unknown shard")
		}
		row, err := strconv.ParseInt(key[separator+1:], 10, 64)
		if err != nil || row < 0 || row >= ordered[inputPosition].Records {
			return RecordPartition{}, fmt.Errorf("stage preflight selected record has an invalid row")
		}
		selected = append(selected, evaluationCandidate{key: key, corpus: ordered[inputPosition].Corpus, input: inputPosition, row: row})
	}
	sizes, err := evaluationCandidateSizes(ctx, ordered, selected)
	if err != nil {
		return RecordPartition{}, err
	}
	var textBytes int64
	for index := range selected {
		selected[index].textBytes = sizes[selected[index].key]
		textBytes += selected[index].textBytes
	}
	records, tokenTargets, err := readEvaluationRecords(ctx, ordered, selected, codec, objective, conversation)
	if err != nil {
		return RecordPartition{}, err
	}
	hasher := sha256.New()
	selectedMap := make(map[string]bool, len(selected))
	for _, candidate := range selected {
		selectedMap[candidate.key] = true
		_, _ = fmt.Fprintln(hasher, candidate.key)
	}
	evaluation := EvaluationSet{
		Selection: snapshot.Evaluation.Selection, Seed: snapshot.Evaluation.Seed,
		Records: int64(len(selected)), TokenTargets: tokenTargets, TextBytes: textBytes,
		SHA256: hex.EncodeToString(hasher.Sum(nil)),
	}
	if evaluation != snapshot.Evaluation {
		return RecordPartition{}, fmt.Errorf("stage preflight held-out evidence does not match the selected records")
	}
	return RecordPartition{
		Evaluation: evaluation, selected: selectedMap, eligibleRecords: cloneRecordCounts(snapshot.EligibleRecords), selectionSummary: snapshot.SelectionSummary, evaluationRecords: records,
		inputs: ordered, parameters: parameters, codec: codec, objective: objective, conversation: conversation,
	}, nil
}

type PartitionProgress struct {
	CurrentShard int
	TotalShards  int
	Records      int64
	Bytes        int64
	TotalBytes   int64
}

type evaluationCandidate struct {
	key       string
	corpus    string
	score     [32]byte
	input     int
	row       int64
	textBytes int64
}

type evaluationHeap []evaluationCandidate

func (values evaluationHeap) Len() int { return len(values) }
func (values evaluationHeap) Less(i, j int) bool {
	return bytes.Compare(values[i].score[:], values[j].score[:]) > 0
}
func (values evaluationHeap) Swap(i, j int)   { values[i], values[j] = values[j], values[i] }
func (values *evaluationHeap) Push(value any) { *values = append(*values, value.(evaluationCandidate)) }
func (values *evaluationHeap) Pop() any {
	old := *values
	last := old[len(old)-1]
	*values = old[:len(old)-1]
	return last
}

func NewRecordPartition(inputs []Input, parameters ResolvedParameters) (RecordPartition, error) {
	return NewRecordPartitionContext(context.Background(), inputs, parameters, nil)
}

// NewRecordPartitionWithProgress deterministically selects held-out records
// while reporting metadata enumeration progress.
func NewRecordPartitionWithProgress(inputs []Input, parameters ResolvedParameters, progress func(PartitionProgress)) (RecordPartition, error) {
	return NewRecordPartitionContext(context.Background(), inputs, parameters, progress)
}

// NewRecordPartitionContext makes held-out metadata selection interruptible.
func NewRecordPartitionContext(ctx context.Context, inputs []Input, parameters ResolvedParameters, progress func(PartitionProgress)) (RecordPartition, error) {
	return NewRecordPartitionContextWithTokenizer(ctx, inputs, parameters, byteCodec{}, progress)
}

func NewRecordPartitionContextWithTokenizer(ctx context.Context, inputs []Input, parameters ResolvedParameters, codec TokenCodec, progress func(PartitionProgress)) (RecordPartition, error) {
	return NewRecordPartitionContextWithTokenizerAndObjective(ctx, inputs, parameters, codec, "causal-language-modeling", progress)
}

func NewRecordPartitionContextWithTokenizerAndObjective(ctx context.Context, inputs []Input, parameters ResolvedParameters, codec TokenCodec, objective string, progress func(PartitionProgress)) (RecordPartition, error) {
	return NewRecordPartitionContextWithTransform(ctx, inputs, parameters, codec, objective, ConversationTransform{}, progress)
}

func NewRecordPartitionContextWithTransform(ctx context.Context, inputs []Input, parameters ResolvedParameters, codec TokenCodec, objective string, conversation ConversationTransform, progress func(PartitionProgress)) (RecordPartition, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if codec == nil {
		return RecordPartition{}, fmt.Errorf("record partition requires a tokenizer")
	}
	ordered := orderedInputs(inputs)
	partition := RecordPartition{selected: make(map[string]bool), eligibleRecords: make(map[string]int64), inputs: ordered, parameters: parameters, codec: codec, objective: objective, conversation: conversation}
	for _, input := range ordered {
		partition.selectionSummary.InputRecords += input.Records
		if input.Corpus != "" {
			partition.eligibleRecords[input.Corpus] += 0
		}
	}
	policy := parameters.Evaluation
	if policy == nil {
		policy = &EvaluationPolicy{Selection: "none-v1"}
	}
	evaluationDisabled := policy.Fraction == 0 || policy.MaxRecords == 0 || policy.MaxBytes == 0
	if evaluationDisabled && !inputsHaveRecordFilters(ordered) {
		for _, input := range ordered {
			if input.Corpus != "" {
				partition.eligibleRecords[input.Corpus] += input.Records
			}
		}
		partition.selectionSummary.IncludedRecords = partition.selectionSummary.InputRecords
		partition.Evaluation = EvaluationSet{Selection: policy.Selection, Seed: parameters.Seed, SHA256: emptyEvaluationDigest()}
		return partition, nil
	}
	var candidates evaluationHeap
	heap.Init(&candidates)
	groupCandidates := map[string]*evaluationHeap{}
	var records, completedBytes, totalBytes int64
	for _, input := range ordered {
		totalBytes += input.Bytes
	}
	for inputPosition, input := range ordered {
		if input.Records <= 0 {
			return RecordPartition{}, fmt.Errorf("select evaluation records from shard %s: declared record count must be positive", input.SHA256)
		}
		physicalRecords, err := shard.RecordCount(input.Path)
		if err != nil {
			return RecordPartition{}, fmt.Errorf("read evaluation row count from shard %s: %w", input.SHA256, err)
		}
		if physicalRecords != input.Records {
			return RecordPartition{}, fmt.Errorf("shard %s contains %d records, corpus BOM declares %d", input.SHA256, physicalRecords, input.Records)
		}
		addCandidate := func(row int64, license string) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			records++
			partition.selectionSummary.IncludedRecords++
			if license != "" {
				if partition.selectionSummary.IncludedLicenses == nil {
					partition.selectionSummary.IncludedLicenses = map[string]int64{}
				}
				partition.selectionSummary.IncludedLicenses[license]++
			}
			if input.Corpus != "" {
				partition.eligibleRecords[input.Corpus]++
			}
			if evaluationDisabled {
				return nil
			}
			key := selectionID(input.SHA256, row)
			score := evaluationScore(parameters.Seed, key)
			candidate := evaluationCandidate{key: key, corpus: input.Corpus, score: score, input: inputPosition, row: row}
			candidateHeap := &candidates
			if policy.Selection == "stratified-lowest-sha256-v1" {
				candidateHeap = groupCandidates[input.Corpus]
				if candidateHeap == nil {
					candidateHeap = &evaluationHeap{}
					heap.Init(candidateHeap)
					groupCandidates[input.Corpus] = candidateHeap
				}
			}
			if candidateHeap.Len() < policy.MaxRecords {
				heap.Push(candidateHeap, candidate)
			} else if bytes.Compare(score[:], (*candidateHeap)[0].score[:]) < 0 {
				heap.Pop(candidateHeap)
				heap.Push(candidateHeap, candidate)
			}
			return nil
		}
		if input.RecordFilter == nil {
			for row := int64(0); row < input.Records; row++ {
				if err := addCandidate(row, ""); err != nil {
					return RecordPartition{}, err
				}
			}
		} else if err := shard.WalkRecords(input.Path, func(row int64, view shard.RecordView) error {
			if !inputAllows(input, view) {
				partition.selectionSummary.SkippedRecords++
				if partition.selectionSummary.SkippedLicenses == nil {
					partition.selectionSummary.SkippedLicenses = map[string]int64{}
				}
				partition.selectionSummary.SkippedLicenses[selectionLicense(view.License)]++
				return nil
			}
			return addCandidate(row, selectionLicense(view.License))
		}); err != nil {
			return RecordPartition{}, fmt.Errorf("apply record filters to shard %s: %w", input.SHA256, err)
		}
		completedBytes += input.Bytes
		if progress != nil {
			progress(PartitionProgress{CurrentShard: inputPosition + 1, TotalShards: len(ordered), Records: records, Bytes: completedBytes, TotalBytes: totalBytes})
		}
	}
	if records == 0 {
		return RecordPartition{}, fmt.Errorf("record filters select no training records")
	}
	if evaluationDisabled {
		partition.Evaluation = EvaluationSet{Selection: policy.Selection, Seed: parameters.Seed, SHA256: emptyEvaluationDigest()}
		return partition, nil
	}
	desired := int(math.Ceil(float64(records) * policy.Fraction))
	if records < 2 {
		desired = 0
	} else {
		desired = min(desired, policy.MaxRecords, int(records-1))
		if desired < 1 {
			desired = 1
		}
	}
	var selected []evaluationCandidate
	var selectedBytes int64
	if policy.Selection == "stratified-lowest-sha256-v1" {
		groups := make([]string, 0, len(groupCandidates))
		values := make(map[string][]evaluationCandidate, len(groupCandidates))
		maxRounds := 0
		for group, candidates := range groupCandidates {
			groups = append(groups, group)
			values[group] = append([]evaluationCandidate(nil), (*candidates)...)
			sort.Slice(values[group], func(i, j int) bool { return bytes.Compare(values[group][i].score[:], values[group][j].score[:]) < 0 })
			maxRounds = max(maxRounds, len(values[group]))
		}
		sizes, err := evaluationCandidateSizes(ctx, ordered, flattenEvaluationCandidates(values))
		if err != nil {
			return RecordPartition{}, err
		}
		for group := range values {
			for index := range values[group] {
				values[group][index].textBytes = sizes[values[group][index].key]
			}
		}
		sort.Strings(groups)
		for round := 0; round < maxRounds && len(selected) < desired; round++ {
			for _, group := range groups {
				if round >= len(values[group]) {
					continue
				}
				candidate := values[group][round]
				if candidate.textBytes > policy.MaxBytes-selectedBytes {
					continue
				}
				selected = append(selected, candidate)
				selectedBytes += candidate.textBytes
				if len(selected) == desired {
					break
				}
			}
		}
	} else {
		values := append([]evaluationCandidate(nil), candidates...)
		sort.Slice(values, func(i, j int) bool { return bytes.Compare(values[i].score[:], values[j].score[:]) < 0 })
		sizes, err := evaluationCandidateSizes(ctx, ordered, values)
		if err != nil {
			return RecordPartition{}, err
		}
		for index := range values {
			values[index].textBytes = sizes[values[index].key]
		}
		for _, candidate := range values {
			if len(selected) == desired {
				break
			}
			if candidate.textBytes > policy.MaxBytes-selectedBytes {
				continue
			}
			selected = append(selected, candidate)
			selectedBytes += candidate.textBytes
		}
	}
	if desired > 0 && len(selected) == 0 {
		return RecordPartition{}, fmt.Errorf("no held-out record fits evaluation_max_bytes=%d; increase the limit or explicitly disable evaluation", policy.MaxBytes)
	}
	evaluationRecords, tokenTargets, err := readEvaluationRecords(ctx, ordered, selected, codec, objective, conversation)
	if err != nil {
		return RecordPartition{}, err
	}
	partition.evaluationRecords = evaluationRecords
	sort.Slice(selected, func(i, j int) bool { return selected[i].key < selected[j].key })
	hasher := sha256.New()
	for _, candidate := range selected {
		partition.selected[candidate.key] = true
		if _, ok := partition.eligibleRecords[candidate.corpus]; ok {
			partition.eligibleRecords[candidate.corpus]--
		}
		_, _ = fmt.Fprintln(hasher, candidate.key)
	}
	partition.Evaluation = EvaluationSet{
		Selection: policy.Selection, Seed: parameters.Seed, Records: int64(len(selected)),
		TokenTargets: tokenTargets, TextBytes: selectedBytes, SHA256: hex.EncodeToString(hasher.Sum(nil)),
	}
	partition.selectionSummary.HeldOutRecords = int64(len(selected))
	return partition, nil
}

func selectionLicense(value string) string {
	if strings.TrimSpace(value) == "" {
		return "(unset)"
	}
	return value
}

func flattenEvaluationCandidates(groups map[string][]evaluationCandidate) []evaluationCandidate {
	var flattened []evaluationCandidate
	for _, values := range groups {
		flattened = append(flattened, values...)
	}
	return flattened
}

func evaluationCandidateSizes(ctx context.Context, inputs []Input, candidates []evaluationCandidate) (map[string]int64, error) {
	grouped := make(map[int][]evaluationCandidate)
	for _, candidate := range candidates {
		grouped[candidate.input] = append(grouped[candidate.input], candidate)
	}
	sizes := make(map[string]int64, len(candidates))
	for inputPosition := range inputs {
		values := grouped[inputPosition]
		if len(values) == 0 {
			continue
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		sort.Slice(values, func(i, j int) bool { return values[i].row < values[j].row })
		positions := make([]int64, len(values))
		for index, candidate := range values {
			positions[index] = candidate.row
		}
		valuesByPosition, err := shard.ReadRecordTextSizes(inputs[inputPosition].Path, positions)
		if err != nil {
			return nil, fmt.Errorf("read evaluation candidate sizes from shard %s: %w", inputs[inputPosition].SHA256, err)
		}
		for index, candidate := range values {
			sizes[candidate.key] = valuesByPosition[index]
		}
	}
	return sizes, nil
}

func readEvaluationRecords(ctx context.Context, inputs []Input, selected []evaluationCandidate, codec TokenCodec, objective string, conversation ConversationTransform) ([]Record, int64, error) {
	grouped := make(map[int][]evaluationCandidate)
	for _, candidate := range selected {
		grouped[candidate.input] = append(grouped[candidate.input], candidate)
	}
	records := make([]Record, 0, len(selected))
	var tokenTargets int64
	for inputPosition, input := range inputs {
		values := grouped[inputPosition]
		if len(values) == 0 {
			continue
		}
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		sort.Slice(values, func(i, j int) bool { return values[i].row < values[j].row })
		positions := make([]int64, len(values))
		for index, candidate := range values {
			positions[index] = candidate.row
		}
		err := shard.ReadRecordsAt(input.Path, positions, func(row int64, view shard.RecordView) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			record := recordFromView(input, row, view)
			if record.Conversation == nil && objective == "causal-language-modeling" {
				tokenTargets += int64(codec.Count(view.Text))
				records = append(records, record)
				return nil
			}
			_, mask, err := tokenizeRecord(record, codec, objective, conversation)
			if err != nil {
				return err
			}
			for _, supervised := range mask[1:] {
				if supervised {
					tokenTargets++
				}
			}
			records = append(records, record)
			return nil
		})
		if err != nil {
			return nil, 0, fmt.Errorf("read selected evaluation records from shard %s: %w", input.SHA256, err)
		}
	}
	return records, tokenTargets, nil
}

func (partition RecordPartition) TrainingRecords() (RecordSource, error) {
	source, err := NewCanonicalRecordSourceWithTokenizer(partition.inputs, partition.parameters, partition.codec)
	if err != nil {
		return nil, err
	}
	return filteredRecordSource{source: source, include: func(record Record) bool { return !partition.selected[record.SelectionID] }}, nil
}

func (partition RecordPartition) EvaluationRecords() RecordSource {
	return sliceRecordSource(partition.evaluationRecords)
}

func (partition RecordPartition) TrainingByteTargets(ctx context.Context) (int64, error) {
	var perEpoch int64
	err := rawRecordSource{inputs: partition.inputs, include: func(record Record) bool { return !partition.selected[record.SelectionID] }}.Stream(ctx, func(record Record) error {
		_, mask, err := tokenizeRecord(record, partition.codec, partition.objective, partition.conversation)
		if err != nil {
			return err
		}
		value := int64(0)
		for _, supervised := range mask[1:] {
			if supervised {
				value++
			}
		}
		if perEpoch > math.MaxInt64-value {
			return fmt.Errorf("training byte-token target count overflows int64")
		}
		perEpoch += value
		return nil
	})
	if err != nil {
		return 0, err
	}
	if perEpoch < 2 || perEpoch > math.MaxInt64/partition.parameters.Epochs {
		return 0, fmt.Errorf("held-out partition leaves no usable training targets")
	}
	total := perEpoch * partition.parameters.Epochs
	if partition.objective != "assistant-response-modeling" {
		total--
	}
	return total, nil
}

// TrainingStepCapacity verifies that the finite epoch stream can supply the
// requested optimizer steps. It stops as soon as the request is satisfiable;
// an exhausted stream returns its exact smaller capacity.
func (partition RecordPartition) TrainingStepCapacity(ctx context.Context, requested int64) (int64, bool, error) {
	if requested <= 0 || partition.parameters.BatchSize <= 0 || partition.parameters.SequenceLength <= 0 {
		return 0, false, fmt.Errorf("requested steps, batch size, and sequence length must be positive")
	}
	requiredSequences, overflow := multiplyInt64(requested, partition.parameters.BatchSize)
	if overflow {
		return 0, false, fmt.Errorf("requested training sequence count overflows int64")
	}
	sequences, sufficient, err := partition.trainingSequenceCapacity(ctx, requiredSequences)
	if err != nil {
		return 0, false, err
	}
	if sufficient {
		return requested, true, nil
	}
	steps := sequences / partition.parameters.BatchSize
	if sequences%partition.parameters.BatchSize != 0 {
		steps++
	}
	return steps, steps >= requested, nil
}

// WithMinimumEpochsForSteps returns the smallest deterministic pass count that
// can supply the requested optimizer steps. Token-budget stages use this during
// preflight so a finite corpus cannot run out after accelerators have started.
func (partition RecordPartition) WithMinimumEpochsForSteps(ctx context.Context, requested int64) (RecordPartition, int64, error) {
	if requested <= 0 {
		return RecordPartition{}, 0, fmt.Errorf("requested steps must be positive")
	}
	const maximumEpochs = int64(1_000_000)
	capacityAt := func(epochs int64) (int64, bool, error) {
		partition.parameters.Epochs = epochs
		return partition.TrainingStepCapacity(ctx, requested)
	}
	low, high := int64(0), max(int64(1), partition.parameters.Epochs)
	for {
		available, sufficient, err := capacityAt(high)
		if err != nil {
			return RecordPartition{}, 0, err
		}
		if sufficient {
			break
		}
		if available == 0 {
			return RecordPartition{}, 0, fmt.Errorf("training stream contains no usable optimizer steps")
		}
		low = high
		if high == maximumEpochs {
			return RecordPartition{}, 0, fmt.Errorf("training stream cannot supply %d optimizer steps within %d epochs", requested, maximumEpochs)
		}
		high = min(maximumEpochs, high*2)
	}
	for low+1 < high {
		middle := low + (high-low)/2
		_, sufficient, err := capacityAt(middle)
		if err != nil {
			return RecordPartition{}, 0, err
		}
		if sufficient {
			high = middle
		} else {
			low = middle
		}
	}
	partition.parameters.Epochs = high
	return partition, high, nil
}

// TrainingSteps scans the finite epoch stream and returns its exact optimizer
// step count after continuous packing and partial-batch flushing.
func (partition RecordPartition) TrainingSteps(ctx context.Context) (int64, error) {
	sequences, _, err := partition.trainingSequenceCapacity(ctx, 0)
	if err != nil {
		return 0, err
	}
	steps := sequences / partition.parameters.BatchSize
	if sequences%partition.parameters.BatchSize != 0 {
		steps++
	}
	if steps <= 0 {
		return 0, fmt.Errorf("held-out partition leaves no usable training steps")
	}
	return steps, nil
}

func (partition RecordPartition) trainingSequenceCapacity(ctx context.Context, requiredSequences int64) (int64, bool, error) {
	source, err := partition.TrainingRecords()
	if err != nil {
		return 0, false, err
	}
	var buffered int
	var masks []bool
	var sequences int64
	reached := errors.New("requested training step capacity reached")
	addSequence := func(targets []bool) error {
		if slices.Contains(targets, true) {
			sequences++
			if requiredSequences > 0 && sequences >= requiredSequences {
				return reached
			}
		}
		return nil
	}
	err = source.Stream(ctx, func(record Record) error {
		tokens, recordMask, err := tokenizeRecord(record, partition.codec, partition.objective, partition.conversation)
		if err != nil {
			return err
		}
		recordTokens := len(tokens) + 1 // Worker framing appends EOS.
		buffered += recordTokens
		masks = append(masks, recordMask...)
		window := int(partition.parameters.SequenceLength) + 1
		for buffered >= window {
			if err := addSequence(masks[1:window]); err != nil {
				return err
			}
			buffered -= int(partition.parameters.SequenceLength)
			masks = masks[int(partition.parameters.SequenceLength):]
		}
		return nil
	})
	if errors.Is(err, reached) {
		return sequences, true, nil
	}
	if err != nil {
		return 0, false, err
	}
	if buffered > 1 {
		if err := addSequence(masks[1:]); errors.Is(err, reached) {
			return sequences, true, nil
		}
	}
	return sequences, false, nil
}

type filteredRecordSource struct {
	source  RecordSource
	include func(Record) bool
}

type sliceRecordSource []Record

func (source sliceRecordSource) Stream(ctx context.Context, consume func(Record) error) error {
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

func (source filteredRecordSource) Stream(ctx context.Context, consume func(Record) error) error {
	return source.source.Stream(ctx, func(record Record) error {
		if !source.include(record) {
			return nil
		}
		return consume(record)
	})
}

type rawRecordSource struct {
	inputs  []Input
	include func(Record) bool
}

func (source rawRecordSource) Stream(ctx context.Context, consume func(Record) error) error {
	for _, input := range source.inputs {
		err := shard.WalkRecords(input.Path, func(row int64, view shard.RecordView) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if !inputAllows(input, view) {
				return nil
			}
			record := recordFromView(input, row, view)
			if source.include != nil && !source.include(record) {
				return nil
			}
			return consume(record)
		})
		if err != nil {
			return fmt.Errorf("stream canonical shard %s: %w", input.SHA256, err)
		}
	}
	return nil
}

func selectionID(shardHash string, row int64) string { return fmt.Sprintf("%s:%d", shardHash, row) }

func evaluationScore(seed uint64, key string) [32]byte {
	value := make([]byte, 8+len(key))
	binary.BigEndian.PutUint64(value, seed)
	copy(value[8:], key)
	return sha256.Sum256(value)
}

func emptyEvaluationDigest() string {
	digest := sha256.Sum256(nil)
	return hex.EncodeToString(digest[:])
}

type RecordSource interface {
	Stream(context.Context, func(Record) error) error
}

// CountByteTargets returns the exact number of next-token targets produced by
// continuous-EOS packing with the built-in byte tokenizer. The first token in
// the continuous stream is context rather than a prediction target.
func CountByteTargets(ctx context.Context, inputs []Input) (int64, error) {
	var tokens int64
	for _, input := range inputs {
		err := shard.WalkRecords(input.Path, func(_ int64, view shard.RecordView) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if !inputAllows(input, view) {
				return nil
			}
			recordTokens := int64(len(view.Text)) + 1 // UTF-8 bytes plus EOS.
			if tokens > math.MaxInt64-recordTokens {
				return fmt.Errorf("byte-token target count overflows int64")
			}
			tokens += recordTokens
			return nil
		})
		if err != nil {
			return 0, fmt.Errorf("count byte tokens in shard %s: %w", input.SHA256, err)
		}
	}
	if tokens < 2 {
		return 0, fmt.Errorf("canonical stream contains no byte-token prediction targets")
	}
	return tokens - 1, nil
}

// ByteTargetsForEpochs expands a one-epoch next-token target count while
// preserving continuous packing across epoch boundaries. CountByteTargets
// excludes the stream's first token once, not once per epoch.
func ByteTargetsForEpochs(oneEpochTargets, epochs int64) (int64, error) {
	if oneEpochTargets < 1 || epochs < 1 {
		return 0, fmt.Errorf("byte targets and epochs must be positive")
	}
	tokensPerEpoch := oneEpochTargets + 1
	if tokensPerEpoch <= 0 || tokensPerEpoch > math.MaxInt64/epochs {
		return 0, fmt.Errorf("epoch byte-token target count overflows int64")
	}
	return tokensPerEpoch*epochs - 1, nil
}

type canonicalRecordSource struct {
	inputs      []Input
	seed        uint64
	epochs      int64
	buffer      int
	bufferBytes int64
	balanced    bool
	weights     map[string]uint64
	codec       TokenCodec
}

func NewCanonicalRecordSource(inputs []Input, parameters ResolvedParameters) (RecordSource, error) {
	return NewCanonicalRecordSourceWithTokenizer(inputs, parameters, byteCodec{})
}

func NewCanonicalRecordSourceWithTokenizer(inputs []Input, parameters ResolvedParameters, codec TokenCodec) (RecordSource, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("canonical record source requires at least one shard input")
	}
	if parameters.Epochs < 1 {
		return nil, fmt.Errorf("canonical record source requires at least one epoch")
	}
	if codec == nil {
		return nil, fmt.Errorf("canonical record source requires a tokenizer")
	}
	if (parameters.Data.Order != "bounded-shuffle-v1" && parameters.Data.Order != "corpus-balanced-shuffle-v1" && parameters.Data.Order != "corpus-weighted-shuffle-v1") || parameters.Data.ShuffleBufferRecords < 1 || parameters.Data.ShuffleBufferBytes < 1 {
		return nil, fmt.Errorf("unsupported canonical record order %q", parameters.Data.Order)
	}
	ordered := orderedInputs(inputs)
	return &canonicalRecordSource{inputs: ordered, seed: parameters.Seed, epochs: parameters.Epochs, buffer: parameters.Data.ShuffleBufferRecords, bufferBytes: parameters.Data.ShuffleBufferBytes, balanced: parameters.Data.Order == "corpus-balanced-shuffle-v1" || parameters.Data.Order == "corpus-weighted-shuffle-v1", weights: parameters.Data.CorpusWeights, codec: codec}, nil
}

func orderedInputs(inputs []Input) []Input {
	ordered := append([]Input(nil), inputs...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].SHA256 != ordered[j].SHA256 {
			return ordered[i].SHA256 < ordered[j].SHA256
		}
		return ordered[i].Path < ordered[j].Path
	})
	return ordered
}

func (source *canonicalRecordSource) Stream(ctx context.Context, consume func(Record) error) error {
	if consume == nil {
		return fmt.Errorf("record consumer is required")
	}
	for epoch := int64(0); epoch < source.epochs; epoch++ {
		if err := source.streamEpoch(ctx, epoch, consume); err != nil {
			return err
		}
	}
	return nil
}

func (source *canonicalRecordSource) streamEpoch(ctx context.Context, epoch int64, consume func(Record) error) error {
	if source.balanced {
		return source.streamBalancedEpoch(ctx, epoch, consume)
	}
	return source.streamInputGroup(ctx, epoch, source.inputs, consume)
}

func (source *canonicalRecordSource) streamInputGroup(ctx context.Context, epoch int64, group []Input, consume func(Record) error) error {
	inputs := append([]Input(nil), group...)
	inputRandom := newSplitMix64(source.seed + uint64(epoch)*0x9e3779b97f4a7c15)
	for index := len(inputs) - 1; index > 0; index-- {
		other := inputRandom.intn(index + 1)
		inputs[index], inputs[other] = inputs[other], inputs[index]
	}
	random := newSplitMix64(source.seed ^ 0x6a09e667f3bcc909 ^ uint64(epoch)*0xbf58476d1ce4e5b9)
	buffer := make([]Record, 0, source.buffer)
	var bufferedBytes int64
	emit := func(record Record) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		return consume(record)
	}
	for _, input := range inputs {
		err := shard.WalkRecords(input.Path, func(row int64, view shard.RecordView) error {
			if !inputAllows(input, view) {
				return nil
			}
			record := recordFromView(input, row, view)
			recordBytes := recordMemoryBytes(record)
			if recordBytes > source.bufferBytes {
				return emit(record)
			}
			for len(buffer) > 0 && (len(buffer) >= source.buffer || bufferedBytes+recordBytes > source.bufferBytes) {
				position := random.intn(len(buffer))
				selected := buffer[position]
				if err := emit(selected); err != nil {
					return err
				}
				bufferedBytes -= recordMemoryBytes(selected)
				last := len(buffer) - 1
				buffer[position] = buffer[last]
				buffer = buffer[:last]
			}
			buffer = append(buffer, record)
			bufferedBytes += recordBytes
			return nil
		})
		if err != nil {
			return fmt.Errorf("stream canonical shard %s in epoch %d: %w", input.SHA256, epoch+1, err)
		}
	}
	for len(buffer) > 0 {
		position := random.intn(len(buffer))
		if err := emit(buffer[position]); err != nil {
			return err
		}
		last := len(buffer) - 1
		bufferedBytes -= recordMemoryBytes(buffer[position])
		buffer[position] = buffer[last]
		buffer = buffer[:last]
	}
	return nil
}

type balancedRecord struct {
	record Record
	err    error
}

// streamBalancedEpoch chooses the corpus with the fewest emitted tokenizer
// targets. This keeps cumulative training exposure balanced even when record
// lengths differ, while each corpus retains the bounded shuffle contract.
func (source *canonicalRecordSource) streamBalancedEpoch(ctx context.Context, epoch int64, consume func(Record) error) error {
	groups := map[string][]Input{}
	for _, input := range source.inputs {
		if input.Corpus == "" {
			return fmt.Errorf("corpus-balanced record order requires corpus identity for shard %s", input.SHA256)
		}
		groups[input.Corpus] = append(groups[input.Corpus], input)
	}
	names := make([]string, 0, len(groups))
	for name := range groups {
		names = append(names, name)
	}
	sort.Strings(names)
	weights := make([]uint64, len(names))
	for position, name := range names {
		weights[position] = 1
		if len(source.weights) != 0 {
			weights[position] = source.weights[name]
			if weights[position] == 0 {
				return fmt.Errorf("corpus-weighted record order requires a weight for corpus %q", name)
			}
		}
	}
	for name := range source.weights {
		if _, exists := groups[name]; !exists {
			return fmt.Errorf("corpus-weighted record order declares unknown corpus %q", name)
		}
	}
	streamContext, cancel := context.WithCancel(ctx)
	defer cancel()
	streams := make([]<-chan balancedRecord, 0, len(names))
	for position, name := range names {
		output := make(chan balancedRecord)
		streams = append(streams, output)
		group := groups[name]
		go func(groupPosition int, values []Input, destination chan<- balancedRecord) {
			defer close(destination)
			groupSource := *source
			groupSource.seed ^= uint64(groupPosition+1) * 0x9e3779b97f4a7c15
			err := groupSource.streamInputGroup(streamContext, epoch, values, func(record Record) error {
				select {
				case destination <- balancedRecord{record: record}:
					return nil
				case <-streamContext.Done():
					return streamContext.Err()
				}
			})
			if err != nil && streamContext.Err() == nil {
				select {
				case destination <- balancedRecord{err: err}:
				case <-streamContext.Done():
				}
			}
		}(position, group, output)
	}
	heads := make([]balancedRecord, len(streams))
	active := make([]bool, len(streams))
	emitted := make([]int64, len(streams))
	activeCount := 0
	for index, stream := range streams {
		item, ok := <-stream
		if !ok {
			continue
		}
		if item.err != nil {
			return item.err
		}
		heads[index], active[index] = item, true
		activeCount++
	}
	for activeCount > 0 {
		selected := -1
		for index := range streams {
			if active[index] && (selected < 0 || weightedBefore(emitted[index], weights[index], emitted[selected], weights[selected])) {
				selected = index
			}
		}
		record := heads[selected].record
		if err := consume(record); err != nil {
			return err
		}
		emitted[selected] += int64(source.codec.Count(record.Text)) + 1
		item, ok := <-streams[selected]
		if !ok {
			active[selected] = false
			activeCount--
			continue
		}
		if item.err != nil {
			return item.err
		}
		heads[selected] = item
	}
	return nil
}

func weightedBefore(leftTokens int64, leftWeight uint64, rightTokens int64, rightWeight uint64) bool {
	leftHigh, leftLow := bits.Mul64(uint64(leftTokens), rightWeight)
	rightHigh, rightLow := bits.Mul64(uint64(rightTokens), leftWeight)
	if leftHigh != rightHigh {
		return leftHigh < rightHigh
	}
	return leftLow < rightLow
}

func recordFromView(input Input, row int64, view shard.RecordView) Record {
	result := Record{SelectionID: selectionID(input.SHA256, row), ID: view.ID, Text: view.Text, Source: view.Source, License: view.License, Language: view.Language, Corpus: input.Corpus}
	if view.Kind == record.KindConversation {
		conversation, err := record.DecodeConversation(view.Text)
		if err != nil {
			result.decodeErr = fmt.Errorf("decode canonical conversation: %w", err)
		} else {
			result.Conversation = &conversation
		}
	}
	return result
}

func inputsHaveRecordFilters(inputs []Input) bool {
	for _, input := range inputs {
		if input.RecordFilter != nil {
			return true
		}
	}
	return false
}

func inputAllows(input Input, view shard.RecordView) bool {
	return input.RecordFilter == nil || input.RecordFilter.Allows(input.Corpus, view)
}

// recordMemoryBytes accounts for retained string data and the five string
// headers in Record. It is deliberately conservative and keeps the shuffle
// window bounded by bytes as well as record count.
func recordMemoryBytes(record Record) int64 {
	return int64(7*16 + len(record.SelectionID) + len(record.ID) + len(record.Text) + len(record.Source) + len(record.License) + len(record.Language) + len(record.Corpus))
}

// splitMix64 is specified here rather than delegated to math/rand so record
// ordering remains stable across Go runtime revisions.
type splitMix64 struct{ state uint64 }

func newSplitMix64(seed uint64) *splitMix64 { return &splitMix64{state: seed} }

func (random *splitMix64) next() uint64 {
	random.state += 0x9e3779b97f4a7c15
	value := random.state
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}

func (random *splitMix64) intn(limit int) int {
	if limit <= 0 {
		panic("splitMix64.intn called with a non-positive limit")
	}
	return int(random.next() % uint64(limit))
}
