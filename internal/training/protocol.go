// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
)

const WorkerProtocolSchema = 1

type WorkerBegin struct {
	RunID                  string                `json:"run_id"`
	Stage                  string                `json:"stage"`
	Objective              string                `json:"objective"`
	ArchitectureSHA256     string                `json:"architecture_sha256"`
	Architecture           json.RawMessage       `json:"architecture"`
	Parameters             ResolvedParameters    `json:"parameters"`
	Parallelism            Parallelism           `json:"parallelism,omitzero"`
	Tokenizer              TokenizerSpec         `json:"tokenizer"`
	EvaluationSet          EvaluationSet         `json:"evaluation_set"`
	Initialization         *WorkerInitialization `json:"initialization,omitempty"`
	Resume                 *WorkerResume         `json:"resume,omitempty"`
	DataNodeRank           int                   `json:"data_node_rank,omitempty"`
	PreparedCacheDirectory string                `json:"-"`
	PreparedCacheMaxBytes  int64                 `json:"-"`
	PreparedIdentity       string                `json:"-"`
}

type tokenizedRecordSource struct {
	source       RecordSource
	codec        TokenCodec
	objective    string
	conversation ConversationTransform
}

func (source tokenizedRecordSource) Stream(ctx context.Context, consume func(Record) error) error {
	return source.source.Stream(ctx, func(record Record) error {
		var err error
		record.Tokens, record.LossMask, err = tokenizeRecord(record, source.codec, source.objective, source.conversation)
		if err != nil {
			return fmt.Errorf("tokenize record %s: %w", record.ID, err)
		}
		record.Text = ""
		record.Conversation = nil
		return consume(record)
	})
}

type WorkerInitialization struct {
	SourceType  string   `json:"source_type,omitempty"`
	SourceID    string   `json:"source_id,omitempty"`
	SourceRunID string   `json:"source_run_id,omitempty"`
	Artifact    Artifact `json:"artifact"`
	Path        string   `json:"path"`
}

type WorkerResume struct {
	Step        int64        `json:"step"`
	Tokens      int64        `json:"tokens"`
	Checkpoint  Checkpoint   `json:"checkpoint"`
	Checkpoints []Checkpoint `json:"checkpoints,omitempty"`
	Evaluations []Evaluation `json:"evaluations,omitempty"`
	Paths       []string     `json:"paths"`
}

type WorkerInputFrame struct {
	Kind     string            `json:"kind"`
	Schema   int               `json:"schema"`
	Begin    *WorkerBegin      `json:"begin,omitempty"`
	Record   *Record           `json:"record,omitempty"`
	Sequence *PreparedSequence `json:"sequence,omitempty"`
}

// PreparedSequence is the deterministic packed unit consumed by one rank.
// It is produced once by each node coordinator after tokenization, before
// Python or accelerator work begins.
type PreparedSequence struct {
	Ordinal     int64            `json:"ordinal"`
	Tokens      []int            `json:"tokens"`
	LossMask    []bool           `json:"loss_mask"`
	Consumption map[string]int64 `json:"consumption,omitempty"`
}

type WorkerOutputFrame struct {
	Kind        string       `json:"kind"`
	Schema      int          `json:"schema"`
	Event       *Event       `json:"event,omitempty"`
	Observation *Observation `json:"observation,omitempty"`
	Error       string       `json:"error,omitempty"`
	ErrorClass  string       `json:"error_class,omitempty"`
}

const (
	WorkerErrorArtifactIntegrity  = "artifact-integrity"
	WorkerErrorNumericalIntegrity = "numerical-integrity"
)

type WorkerError struct {
	Message string
	Class   string
}

func (err *WorkerError) Error() string { return err.Message }

func IsNonRetryableWorkerError(err error) bool {
	class := WorkerErrorClass(err)
	return class == WorkerErrorArtifactIntegrity || class == WorkerErrorNumericalIntegrity
}

func WorkerErrorClass(err error) string {
	var worker *WorkerError
	if errors.As(err, &worker) {
		return worker.Class
	}
	return ""
}

func WriteWorkerInput(ctx context.Context, output io.Writer, begin WorkerBegin, records, evaluationRecords RecordSource) error {
	return writeWorkerInputUntil(ctx, output, begin, records, evaluationRecords, nil)
}

var errWorkerReachedTarget = errors.New("worker reached target steps")

func writeWorkerInputUntil(ctx context.Context, output io.Writer, begin WorkerBegin, records, evaluationRecords RecordSource, stopRecords <-chan struct{}) error {
	if records == nil {
		return fmt.Errorf("worker input requires a record source")
	}
	encoder := json.NewEncoder(output)
	if err := encoder.Encode(WorkerInputFrame{Kind: "begin", Schema: WorkerProtocolSchema, Begin: &begin}); err != nil {
		return err
	}
	if evaluationRecords != nil {
		if err := evaluationRecords.Stream(ctx, func(record Record) error {
			return encoder.Encode(WorkerInputFrame{Kind: "evaluation_record", Schema: WorkerProtocolSchema, Record: &record})
		}); err != nil {
			return err
		}
	}
	// A fully trained checkpoint may need only final artifact verification and
	// bookkeeping. Do not replay the complete corpus merely to finalize it.
	if begin.Resume != nil && begin.Resume.Step == begin.Parameters.Steps {
		return encoder.Encode(WorkerInputFrame{Kind: "end", Schema: WorkerProtocolSchema})
	}
	if begin.Parallelism.DataPlane == DataPlaneNodeLocal {
		if err := writePreparedSequences(ctx, encoder, begin, records, stopRecords); err != nil {
			return err
		}
		return encoder.Encode(WorkerInputFrame{Kind: "end", Schema: WorkerProtocolSchema})
	}
	if err := records.Stream(ctx, func(record Record) error {
		if stopRecords != nil {
			select {
			case <-stopRecords:
				return errWorkerReachedTarget
			default:
			}
		}
		return encoder.Encode(WorkerInputFrame{Kind: "record", Schema: WorkerProtocolSchema, Record: &record})
	}); err != nil && !errors.Is(err, errWorkerReachedTarget) {
		return err
	}
	return encoder.Encode(WorkerInputFrame{Kind: "end", Schema: WorkerProtocolSchema})
}

func writePreparedSequences(ctx context.Context, encoder *json.Encoder, begin WorkerBegin, records RecordSource, stopRecords <-chan struct{}) error {
	worldSize := begin.Parallelism.WorldSize
	GPUsPerNode := begin.Parallelism.GPUsPerNode
	if worldSize < 2 || GPUsPerNode < 1 || worldSize%GPUsPerNode != 0 || begin.DataNodeRank < 0 || begin.DataNodeRank >= worldSize/GPUsPerNode {
		return fmt.Errorf("invalid node-local prepared-data topology")
	}
	resumeSequences := int64(0)
	if begin.Resume != nil {
		var overflow bool
		resumeSequences, overflow = multiplyInt64(begin.Resume.Step, begin.Parameters.BatchSize)
		if overflow {
			return fmt.Errorf("prepared resume position overflows int64")
		}
	}
	if begin.PreparedCacheDirectory != "" && begin.PreparedIdentity != "" {
		replayed, err := replayPreparedSequences(begin.PreparedCacheDirectory, begin.PreparedIdentity, begin.DataNodeRank, worldSize, GPUsPerNode, begin.Parameters.BatchSize/begin.Parameters.GradientAccumulation, begin.PreparedCacheMaxBytes, resumeSequences, encoder)
		if err != nil {
			return err
		}
		if replayed {
			return nil
		}
	}
	sequenceLength := int(begin.Parameters.SequenceLength)
	globalMicroBatch := begin.Parameters.BatchSize / begin.Parameters.GradientAccumulation
	targetSequences, overflow := multiplyInt64(begin.Parameters.Steps, begin.Parameters.BatchSize)
	if overflow {
		return fmt.Errorf("prepared sequence target overflows int64")
	}
	var tokens []int
	var masks []bool
	var corpora []string
	ordinal := int64(0)
	cacheWriter, err := newPreparedCacheWriter(begin.PreparedCacheDirectory, begin.PreparedIdentity, begin.DataNodeRank, worldSize, GPUsPerNode, globalMicroBatch, begin.PreparedCacheMaxBytes)
	if err != nil {
		return err
	}
	defer cacheWriter.Abort()
	emitSequence := func(piece []int, targetMask []bool, targetCorpora []string) error {
		if !anyMask(targetMask) {
			return nil
		}
		ownerRank := int(ordinal % int64(worldSize))
		ownerNode := ownerRank / GPUsPerNode
		if ownerNode == begin.DataNodeRank {
			consumption := map[string]int64{}
			for index, supervised := range targetMask {
				if supervised {
					consumption[targetCorpora[index]]++
				}
			}
			sequence := PreparedSequence{Ordinal: ordinal, Tokens: append([]int(nil), piece...), LossMask: append([]bool(nil), targetMask...), Consumption: consumption}
			if err := cacheWriter.Append(sequence); err != nil {
				return err
			}
			if ordinal >= resumeSequences {
				if err := encoder.Encode(WorkerInputFrame{Kind: "sequence", Schema: WorkerProtocolSchema, Sequence: &sequence}); err != nil {
					return err
				}
			}
		}
		ordinal++
		if ordinal > resumeSequences && ordinal%globalMicroBatch == 0 {
			if err := encoder.Encode(WorkerInputFrame{Kind: "micro_batch_end", Schema: WorkerProtocolSchema}); err != nil {
				return err
			}
		}
		if ordinal >= targetSequences {
			return errWorkerReachedTarget
		}
		return nil
	}
	err = records.Stream(ctx, func(record Record) error {
		if stopRecords != nil {
			select {
			case <-stopRecords:
				return errWorkerReachedTarget
			default:
			}
		}
		if len(record.LossMask) != len(record.Tokens)+1 {
			return fmt.Errorf("prepared record %s loss mask does not include its EOS target", record.ID)
		}
		tokens = append(tokens, record.Tokens...)
		tokens = append(tokens, begin.Tokenizer.EOSID)
		masks = append(masks, record.LossMask...)
		for range record.LossMask {
			corpora = append(corpora, record.Corpus)
		}
		window := sequenceLength + 1
		for len(tokens) >= window {
			if err := emitSequence(tokens[:window], masks[1:window], corpora[1:window]); err != nil {
				return err
			}
			tokens = tokens[sequenceLength:]
			masks = masks[sequenceLength:]
			corpora = corpora[sequenceLength:]
		}
		return nil
	})
	if err != nil && !errors.Is(err, errWorkerReachedTarget) {
		return err
	}
	if ordinal >= targetSequences {
		return cacheWriter.Commit(ordinal)
	}
	if errors.Is(err, errWorkerReachedTarget) {
		return nil
	}
	if len(tokens) > 1 {
		if err := emitSequence(tokens, masks[1:], corpora[1:]); err != nil && !errors.Is(err, errWorkerReachedTarget) {
			return err
		}
	}
	// Epoch-derived plans round a partial final global batch up to one
	// optimizer step. The distributed worker pads the unused slots with
	// zero-loss sequences, so the prepared stream only needs enough real
	// sequences to enter that final step.
	minimumSequences := targetSequences - begin.Parameters.BatchSize + 1
	if ordinal < minimumSequences {
		return fmt.Errorf("prepared stream produced %d sequences; at least %d are required to complete %d steps (%d full batch slots)", ordinal, minimumSequences, begin.Parameters.Steps, targetSequences)
	}
	return cacheWriter.Commit(ordinal)
}

func anyMask(mask []bool) bool {
	for _, value := range mask {
		if value {
			return true
		}
	}
	return false
}

func ReadWorkerOutput(input io.Reader, consume func(WorkerOutputFrame) error) error {
	return ReadWorkerOutputWithSkipped(input, io.Discard, consume)
}

func ReadWorkerOutputWithSkipped(input io.Reader, skipped io.Writer, consume func(WorkerOutputFrame) error) error {
	if consume == nil {
		return fmt.Errorf("worker output consumer is required")
	}
	scanner := bufio.NewScanner(input)
	buffer := make([]byte, 64*1024)
	scanner.Buffer(buffer, 16*1024*1024)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if start := bytes.IndexByte(line, '{'); start < 0 {
			if len(line) > 0 {
				fmt.Fprintf(skipped, "%s\n", line)
			}
			continue
		} else if start > 0 {
			fmt.Fprintf(skipped, "%s\n", line[:start])
			line = line[start:]
		}
		var frame WorkerOutputFrame
		if err := json.Unmarshal(line, &frame); err != nil {
			return fmt.Errorf("decode worker output: %w", err)
		}
		if err := frame.Validate(); err != nil {
			return err
		}
		if err := consume(frame); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func (frame WorkerOutputFrame) Validate() error {
	if frame.Schema != WorkerProtocolSchema {
		return fmt.Errorf("unsupported worker protocol schema %d", frame.Schema)
	}
	payloads := 0
	if frame.Event != nil {
		payloads++
	}
	if frame.Observation != nil {
		payloads++
	}
	if frame.Error != "" {
		payloads++
	}
	if payloads != 1 {
		return fmt.Errorf("worker output %q must contain exactly one payload", frame.Kind)
	}
	switch frame.Kind {
	case "event":
		if frame.Event == nil {
			return fmt.Errorf("worker event frame is missing event")
		}
		if err := frame.Event.Validate(); err != nil {
			return err
		}
	case "complete":
		if frame.Observation == nil {
			return fmt.Errorf("worker complete frame is missing observation")
		}
	case "error":
		if frame.Error == "" {
			return fmt.Errorf("worker error frame is missing error")
		}
		if frame.ErrorClass != "" && frame.ErrorClass != WorkerErrorArtifactIntegrity && frame.ErrorClass != WorkerErrorNumericalIntegrity {
			return fmt.Errorf("worker error frame has unsupported error_class %q", frame.ErrorClass)
		}
	default:
		return fmt.Errorf("unsupported worker output kind %q", frame.Kind)
	}
	return nil
}

func (event Event) Validate() error {
	if event.Step < 0 || event.Tokens < 0 || event.LearningRate < 0 || event.TokensPerSecond < 0 || event.DurationSeconds < 0 || event.DataWaitSeconds < 0 || event.TrainingFLOPs < 0 || event.AchievedTFLOPS < 0 || event.ModelFLOPUtilization < 0 || event.SkippedSteps < 0 || event.ETASeconds < 0 {
		return fmt.Errorf("worker event %q contains negative progress", event.Kind)
	}
	if event.Loss != nil && (*event.Loss < 0 || math.IsNaN(*event.Loss) || math.IsInf(*event.Loss, 0)) {
		return fmt.Errorf("worker event %q contains invalid loss", event.Kind)
	}
	if event.GradientNorm != nil && (*event.GradientNorm < 0 || math.IsNaN(*event.GradientNorm) || math.IsInf(*event.GradientNorm, 0)) {
		return fmt.Errorf("worker event %q contains invalid gradient norm", event.Kind)
	}
	for _, value := range []float64{event.LearningRate, event.TokensPerSecond, event.DurationSeconds, event.DataWaitSeconds, event.TrainingFLOPs, event.AchievedTFLOPS, event.ModelFLOPUtilization} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("worker event %q contains invalid telemetry", event.Kind)
		}
	}
	if event.TokensPerSecond < 0 {
		return fmt.Errorf("worker event %q contains invalid throughput", event.Kind)
	}
	switch event.Kind {
	case "progress", "log":
		if event.Checkpoint != nil || event.Evaluation != nil {
			return fmt.Errorf("worker event %q contains a typed payload", event.Kind)
		}
	case "checkpoint":
		if event.Checkpoint == nil || event.Evaluation != nil {
			return fmt.Errorf("worker checkpoint event has an invalid payload")
		}
	case "evaluation":
		if event.Evaluation == nil || event.Checkpoint != nil {
			return fmt.Errorf("worker evaluation event has an invalid payload")
		}
	default:
		return fmt.Errorf("unsupported worker event kind %q", event.Kind)
	}
	return nil
}
