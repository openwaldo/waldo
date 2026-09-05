// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package training is the narrow adapter boundary between WALDO's durable
// model lifecycle and an execution framework.
package training

import (
	"context"
	"encoding/json"

	"github.com/openwaldo/waldo/internal/corpus"
)

type Identity struct {
	Name     string `json:"name" yaml:"name"`
	Revision string `json:"revision" yaml:"revision"`
}

type Capabilities struct {
	Objectives       []string `json:"objectives"`
	CheckpointResume bool     `json:"checkpoint_resume"`
	Distributed      bool     `json:"distributed"`
	Safetensors      bool     `json:"safetensors"`
}

type Descriptor struct {
	Identity     Identity     `json:"identity"`
	Framework    string       `json:"framework"`
	Capabilities Capabilities `json:"capabilities"`
}

type Host struct {
	OS           string `json:"os"`
	Architecture string `json:"architecture"`
}

type Accelerator struct {
	Manufacturer string `json:"manufacturer"`
	Model        string `json:"model"`
	MemoryBytes  uint64 `json:"memory_bytes"`
}

// Execution is the immutable environment selected for a build. It is
// persisted by the model domain; adapters never write lifecycle records.
type Execution struct {
	Backend      Identity      `json:"backend"`
	Framework    string        `json:"framework"`
	Runtime      string        `json:"runtime"`
	Host         Host          `json:"host"`
	Accelerators []Accelerator `json:"accelerators,omitempty"`
	Nodes        int           `json:"nodes"`
	WorldSize    int           `json:"world_size"`
}

type ResolveRequest struct {
	ArchitectureSHA256 string
	Architecture       json.RawMessage
	Objectives         []string
}

type Selection struct {
	Backend   Backend
	Execution Execution
}

type Resolver interface {
	Resolve(context.Context, ResolveRequest) (Selection, error)
}

type ResolverFunc func(context.Context, ResolveRequest) (Selection, error)

func (function ResolverFunc) Resolve(ctx context.Context, request ResolveRequest) (Selection, error) {
	return function(ctx, request)
}

type Parameters struct {
	Profile              string            `json:"profile,omitempty" yaml:"profile,omitempty"`
	Epochs               int64             `json:"epochs,omitempty" yaml:"epochs,omitempty"`
	Tokens               int64             `json:"tokens,omitempty" yaml:"tokens,omitempty"`
	Steps                int64             `json:"steps,omitempty" yaml:"steps,omitempty"`
	BatchSize            int64             `json:"batch_size" yaml:"batch_size"`
	SequenceLength       int64             `json:"sequence_length" yaml:"sequence_length"`
	LearningRate         float64           `json:"learning_rate" yaml:"learning_rate"`
	Seed                 uint64            `json:"seed" yaml:"seed"`
	WeightDecay          *float64          `json:"weight_decay,omitempty" yaml:"weight_decay,omitempty"`
	WarmupSteps          *int64            `json:"warmup_steps,omitempty" yaml:"warmup_steps,omitempty"`
	CheckpointEvery      *int64            `json:"checkpoint_every,omitempty" yaml:"checkpoint_every,omitempty"`
	EvaluateEvery        *int64            `json:"evaluate_every,omitempty" yaml:"evaluate_every,omitempty"`
	ShuffleBufferRecords *int              `json:"shuffle_buffer_records,omitempty" yaml:"shuffle_buffer_records,omitempty"`
	ShuffleBufferBytes   *int64            `json:"shuffle_buffer_bytes,omitempty" yaml:"shuffle_buffer_bytes,omitempty"`
	CorpusWeights        map[string]uint64 `json:"corpus_weights,omitempty" yaml:"corpus_weights,omitempty"`
	EvaluationFraction   *float64          `json:"evaluation_fraction,omitempty" yaml:"evaluation_fraction,omitempty"`
	EvaluationMaxRecords *int              `json:"evaluation_max_records,omitempty" yaml:"evaluation_max_records,omitempty"`
	EvaluationMaxBytes   *int64            `json:"evaluation_max_bytes,omitempty" yaml:"evaluation_max_bytes,omitempty"`
}

type ResolvedParameters struct {
	Profile              string            `json:"profile"`
	ProfileSchema        int               `json:"profile_schema"`
	Epochs               int64             `json:"epochs,omitempty"`
	RequestedTokens      int64             `json:"requested_tokens,omitempty"`
	Steps                int64             `json:"steps"`
	BatchSize            int64             `json:"batch_size"`
	SequenceLength       int64             `json:"sequence_length"`
	LearningRate         float64           `json:"learning_rate"`
	Seed                 uint64            `json:"seed"`
	Optimizer            Optimizer         `json:"optimizer"`
	Schedule             Schedule          `json:"schedule"`
	Data                 DataPlan          `json:"data"`
	Evaluation           *EvaluationPolicy `json:"evaluation,omitempty"`
	CheckpointEvery      int64             `json:"checkpoint_every"`
	EvaluateEvery        int64             `json:"evaluate_every"`
	PlannedTokenCapacity int64             `json:"planned_token_capacity"`
}

type Optimizer struct {
	Name        string  `json:"name"`
	WeightDecay float64 `json:"weight_decay"`
	Beta1       float64 `json:"beta1"`
	Beta2       float64 `json:"beta2"`
	Epsilon     float64 `json:"epsilon"`
}

type Schedule struct {
	Name             string  `json:"name"`
	WarmupSteps      int64   `json:"warmup_steps"`
	MinimumRateRatio float64 `json:"minimum_rate_ratio"`
}

type DataPlan struct {
	Order                string            `json:"order"`
	ShuffleBufferRecords int               `json:"shuffle_buffer_records"`
	ShuffleBufferBytes   int64             `json:"shuffle_buffer_bytes"`
	Packing              string            `json:"packing"`
	CorpusWeights        map[string]uint64 `json:"corpus_weights,omitempty"`
}

type EvaluationPolicy struct {
	Selection  string  `json:"selection"`
	Fraction   float64 `json:"fraction"`
	MaxRecords int     `json:"max_records"`
	MaxBytes   int64   `json:"max_bytes"`
}

type EvaluationSet struct {
	Selection    string `json:"selection"`
	Seed         uint64 `json:"seed"`
	Records      int64  `json:"records"`
	TokenTargets int64  `json:"token_targets"`
	TextBytes    int64  `json:"text_bytes"`
	SHA256       string `json:"sha256"`
}

type Input struct {
	Path         string
	SHA256       string
	Bytes        int64
	Records      int64
	Corpus       string
	RecordFilter *corpus.RecordFilterPolicy
}

type Initialization struct {
	SourceType  string   `json:"source_type,omitempty"`
	SourceID    string   `json:"source_id,omitempty"`
	SourceRunID string   `json:"source_run_id,omitempty"`
	Artifact    Artifact `json:"artifact"`
	Path        string   `json:"-"`
}

type Request struct {
	RunID              string
	Stage              string
	Objective          string
	Conversation       ConversationTransform
	ArchitectureSHA256 string
	Architecture       json.RawMessage
	Tokenizer          TokenizerSpec
	BOM                corpus.BOM
	Inputs             []Input
	Parameters         ResolvedParameters
	Records            RecordSource
	EvaluationRecords  RecordSource
	EvaluationSet      EvaluationSet
	PreTokenize        bool
	Initialization     *Initialization
	Resume             *ResumePoint
	ArtifactDirectory  string
	ArtifactPrefix     string
	Report             func(Event)
}

// ResumePoint is the newest verified, fully committed checkpoint from an
// interrupted run. Every artifact is content-addressed and remains relative
// to the run directory; Path is populated only for the backend handoff.
type ResumePoint struct {
	Step       int64      `json:"step"`
	Tokens     int64      `json:"tokens"`
	Checkpoint Checkpoint `json:"checkpoint"`
	Paths      []string   `json:"-"`
}

// Progress is durable, non-terminal evidence emitted while a backend runs.
// It allows interruption to retain verified checkpoints and evaluations
// without misrepresenting them as a complete observation.
type Progress struct {
	Steps          int64        `json:"steps"`
	ConsumedTokens int64        `json:"consumed_tokens"`
	LastLoss       *float64     `json:"last_loss,omitempty"`
	Checkpoints    []Checkpoint `json:"checkpoints,omitempty"`
	Evaluations    []Evaluation `json:"evaluations,omitempty"`
}

type Event struct {
	Kind            string      `json:"kind"`
	Message         string      `json:"message,omitempty"`
	Step            int64       `json:"step,omitempty"`
	Tokens          int64       `json:"tokens,omitempty"`
	Loss            *float64    `json:"loss,omitempty"`
	LearningRate    float64     `json:"learning_rate,omitempty"`
	TokensPerSecond float64     `json:"tokens_per_second,omitempty"`
	ETASeconds      int64       `json:"eta_seconds,omitempty"`
	Checkpoint      *Checkpoint `json:"checkpoint,omitempty"`
	Evaluation      *Evaluation `json:"evaluation,omitempty"`
}

type Artifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type Observation struct {
	Simulated      bool                `json:"simulated"`
	Steps          int64               `json:"steps"`
	ConsumedTokens int64               `json:"consumed_tokens"`
	FinalLoss      *float64            `json:"final_loss,omitempty"`
	Checkpoints    []Checkpoint        `json:"checkpoints,omitempty"`
	Evaluations    []Evaluation        `json:"evaluations,omitempty"`
	Artifacts      []Artifact          `json:"artifacts"`
	Consumption    []CorpusConsumption `json:"consumption,omitempty"`
}

// CorpusConsumption is exact next-token target usage attributed by the
// trainer after packing, not an estimate based on selected shard sizes.
type CorpusConsumption struct {
	Corpus       string `json:"corpus"`
	TokenTargets int64  `json:"token_targets"`
}

type Checkpoint struct {
	Step      int64      `json:"step"`
	Tokens    int64      `json:"tokens"`
	Artifacts []Artifact `json:"artifacts"`
}

type Evaluation struct {
	Step    int64              `json:"step"`
	Tokens  int64              `json:"tokens"`
	Metrics map[string]float64 `json:"metrics"`
}

type Backend interface {
	Descriptor() Descriptor
	Run(context.Context, Request) (Observation, error)
}
