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
	"fmt"
	"strings"

	"github.com/openwaldo/waldo/internal/corpus"
)

type Identity struct {
	Name     string `json:"name" yaml:"name"`
	Revision string `json:"revision" yaml:"revision"`
}

type Capabilities struct {
	Objectives              []string `json:"objectives"`
	CheckpointResume        bool     `json:"checkpoint_resume"`
	Distributed             bool     `json:"distributed"`
	Safetensors             bool     `json:"safetensors"`
	ActivationCheckpointing bool     `json:"activation_checkpointing"`
	Compile                 bool     `json:"compile"`
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
	Parallelism  Parallelism   `json:"parallelism,omitzero"`
}

const (
	ParallelismAuto          = "auto"
	ParallelismData          = "data-parallel"
	ParallelismHybridSharded = "hybrid-sharded-data-parallel"
	ParallelismFullySharded  = "fully-sharded-data-parallel"
	DataPlaneLocal           = "local"
	DataPlaneNodeLocal       = "node-local-cache"
	DataPlaneRankZero        = "rank-zero-broadcast"
)

// Parallelism is the resolved physical placement of one training stage. It
// belongs to execution provenance rather than the portable model architecture.
type Parallelism struct {
	Requested                string `json:"requested"`
	Strategy                 string `json:"strategy"`
	WorldSize                int    `json:"world_size"`
	Nodes                    int    `json:"nodes"`
	GPUsPerNode              int    `json:"gpus_per_node"`
	CompleteModelCopies      int    `json:"complete_model_copies"`
	GPUsSharingEachModelCopy int    `json:"gpus_sharing_each_model_copy"`
	LocalInterconnect        string `json:"local_interconnect,omitempty"`
	InterNodeInterconnect    string `json:"inter_node_interconnect,omitempty"`
	DataPlane                string `json:"data_plane,omitempty"`
	EstimatedModelStateBytes uint64 `json:"estimated_model_state_bytes"`
	MemoryPerGPUBytes        uint64 `json:"memory_per_gpu_bytes"`
}

func ValidateParallelismRequest(value string) error {
	switch value {
	case "", ParallelismAuto, ParallelismData, ParallelismHybridSharded, ParallelismFullySharded:
		return nil
	default:
		return fmt.Errorf("unsupported parallelism %q; use auto, data-parallel, hybrid-sharded-data-parallel, or fully-sharded-data-parallel", value)
	}
}

func (plan Parallelism) Validate(execution Execution) error {
	if plan.Strategy == "" {
		return nil
	}
	if err := ValidateParallelismRequest(plan.Requested); err != nil {
		return err
	}
	if plan.Strategy == ParallelismAuto {
		return fmt.Errorf("resolved parallelism strategy cannot remain auto")
	}
	if err := ValidateParallelismRequest(plan.Strategy); err != nil {
		return err
	}
	if plan.WorldSize != execution.WorldSize || plan.Nodes != execution.Nodes || plan.GPUsPerNode < 1 || plan.WorldSize != plan.Nodes*plan.GPUsPerNode {
		return fmt.Errorf("resolved parallelism topology does not match execution topology")
	}
	if plan.CompleteModelCopies < 1 || plan.GPUsSharingEachModelCopy < 1 || plan.CompleteModelCopies*plan.GPUsSharingEachModelCopy != plan.WorldSize {
		return fmt.Errorf("resolved parallelism model placement does not account for every GPU")
	}
	if plan.DataPlane != "" && plan.DataPlane != DataPlaneLocal && plan.DataPlane != DataPlaneNodeLocal && plan.DataPlane != DataPlaneRankZero {
		return fmt.Errorf("unsupported training data plane %q", plan.DataPlane)
	}
	return nil
}

// DescribeParallelism returns deliberately literal user-facing descriptions;
// distributed-training jargon is not required to understand model placement.
func DescribeParallelism(plan Parallelism, globalBatch int64) []string {
	if plan.Strategy == "" || plan.WorldSize < 1 {
		return nil
	}
	selection := "selected " + parallelismDisplayName(plan.Strategy) + " as requested by the compose"
	if plan.Requested == ParallelismAuto {
		selection = "automatically selected " + automaticParallelismDescription(plan.Strategy)
	}
	messages := []string{selection}
	sequences := int64(0)
	if globalBatch > 0 && globalBatch%int64(plan.WorldSize) == 0 {
		sequences = globalBatch / int64(plan.WorldSize)
	}
	switch plan.Strategy {
	case ParallelismData:
		messages = append(messages, dataShardingDescription(plan.WorldSize, sequences))
		messages = append(messages, fmt.Sprintf("model placement: each of the %d GPUs holds a synchronized complete copy; the run still produces one model", plan.WorldSize))
	case ParallelismHybridSharded:
		messages = append(messages, dataShardingDescription(plan.WorldSize, sequences))
		messages = append(messages, fmt.Sprintf("model placement: the %d GPUs within each of %d hosts jointly hold one synchronized complete copy; the run still produces one model", plan.GPUsSharingEachModelCopy, plan.CompleteModelCopies))
	case ParallelismFullySharded:
		messages = append(messages, dataShardingDescription(plan.WorldSize, sequences))
		messages = append(messages, fmt.Sprintf("model placement: one model is divided across all %d GPUs; no GPU or host holds a complete copy", plan.WorldSize))
	}
	var paths []string
	if plan.GPUsPerNode > 1 && plan.LocalInterconnect != "" {
		paths = append(paths, fmt.Sprintf("%s between the %d GPUs within each host", interconnectDisplayName(plan.LocalInterconnect), plan.GPUsPerNode))
	}
	if plan.Nodes > 1 && plan.InterNodeInterconnect != "" {
		paths = append(paths, fmt.Sprintf("%s between %d hosts", interconnectDisplayName(plan.InterNodeInterconnect), plan.Nodes))
	}
	if len(paths) > 0 {
		messages = append(messages, "model updates synchronize over "+strings.Join(paths, " and "))
	}
	switch plan.DataPlane {
	case DataPlaneNodeLocal:
		messages = append(messages, "data plane: each node verifies and prepares its own cached corpus stream; only rank-local sequence traffic stays within the node")
	case DataPlaneRankZero:
		messages = append(messages, "data plane: launcher compatibility mode broadcasts prepared records from global rank zero across nodes")
	}
	return messages
}

func parallelismDisplayName(strategy string) string {
	switch strategy {
	case ParallelismData:
		return "data parallelism"
	case ParallelismHybridSharded:
		return "hybrid sharding"
	case ParallelismFullySharded:
		return "full model sharding"
	default:
		return strategy
	}
}

func automaticParallelismDescription(strategy string) string {
	description := parallelismDisplayName(strategy)
	switch strategy {
	case ParallelismData:
		return description + " because the complete model state fits within WALDO's per-GPU memory allowance"
	case ParallelismHybridSharded:
		return description + " because the complete model state does not fit within one GPU's allowance but does fit across one host"
	case ParallelismFullySharded:
		return description + " because the model state must be divided across every GPU to fit within the memory allowance"
	default:
		return description
	}
}

func dataShardingDescription(GPUs int, sequences int64) string {
	if sequences < 1 {
		return fmt.Sprintf("training data: each global batch is sharded across %d GPUs without duplicating sequences between GPUs", GPUs)
	}
	if sequences == 1 {
		return fmt.Sprintf("training data: each global batch is sharded across %d GPUs; each GPU processes 1 unique sequence and no sequence is duplicated between GPUs", GPUs)
	}
	return fmt.Sprintf("training data: each global batch is sharded across %d GPUs; each GPU processes %d unique sequences and no sequence is duplicated between GPUs", GPUs, sequences)
}

func interconnectDisplayName(value string) string {
	switch value {
	case "nvlink":
		return "NVLink"
	case "rdma":
		return "RDMA"
	case "tcp":
		return "TCP networking"
	case "gpu-peer-to-peer":
		return "direct GPU peer-to-peer links"
	case "pcie":
		return "PCIe"
	default:
		return value
	}
}

type ResolveRequest struct {
	ArchitectureSHA256    string
	Architecture          json.RawMessage
	Objectives            []string
	ApproximateParameters uint64
	Parallelism           string
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
	Profile                 string            `json:"profile,omitempty" yaml:"profile,omitempty"`
	Parallelism             string            `json:"parallelism,omitempty" yaml:"parallelism,omitempty"`
	Epochs                  int64             `json:"epochs,omitempty" yaml:"epochs,omitempty"`
	Tokens                  int64             `json:"tokens,omitempty" yaml:"tokens,omitempty"`
	Steps                   int64             `json:"steps,omitempty" yaml:"steps,omitempty"`
	BatchSize               int64             `json:"batch_size" yaml:"batch_size"`
	GradientAccumulation    int64             `json:"gradient_accumulation_steps,omitempty" yaml:"gradient_accumulation_steps,omitempty"`
	ComputePrecision        string            `json:"compute_precision,omitempty" yaml:"compute_precision,omitempty"`
	ActivationCheckpointing bool              `json:"activation_checkpointing,omitempty" yaml:"activation_checkpointing,omitempty"`
	Compile                 bool              `json:"compile,omitempty" yaml:"compile,omitempty"`
	DistributionPolicy      string            `json:"distribution_policy,omitempty" yaml:"distribution_policy,omitempty"`
	SequenceLength          int64             `json:"sequence_length" yaml:"sequence_length"`
	LearningRate            float64           `json:"learning_rate" yaml:"learning_rate"`
	Optimizer               string            `json:"optimizer,omitempty" yaml:"optimizer,omitempty"`
	Schedule                string            `json:"schedule,omitempty" yaml:"schedule,omitempty"`
	Seed                    uint64            `json:"seed" yaml:"seed"`
	WeightDecay             *float64          `json:"weight_decay,omitempty" yaml:"weight_decay,omitempty"`
	WarmupSteps             *int64            `json:"warmup_steps,omitempty" yaml:"warmup_steps,omitempty"`
	WarmdownSteps           *int64            `json:"warmdown_steps,omitempty" yaml:"warmdown_steps,omitempty"`
	MinimumRateRatio        *float64          `json:"minimum_learning_rate_ratio,omitempty" yaml:"minimum_learning_rate_ratio,omitempty"`
	CheckpointEvery         *int64            `json:"checkpoint_every,omitempty" yaml:"checkpoint_every,omitempty"`
	EvaluateEvery           *int64            `json:"evaluate_every,omitempty" yaml:"evaluate_every,omitempty"`
	ShuffleBufferRecords    *int              `json:"shuffle_buffer_records,omitempty" yaml:"shuffle_buffer_records,omitempty"`
	ShuffleBufferBytes      *int64            `json:"shuffle_buffer_bytes,omitempty" yaml:"shuffle_buffer_bytes,omitempty"`
	CorpusWeights           map[string]uint64 `json:"corpus_weights,omitempty" yaml:"corpus_weights,omitempty"`
	EvaluationFraction      *float64          `json:"evaluation_fraction,omitempty" yaml:"evaluation_fraction,omitempty"`
	EvaluationMaxRecords    *int              `json:"evaluation_max_records,omitempty" yaml:"evaluation_max_records,omitempty"`
	EvaluationMaxBytes      *int64            `json:"evaluation_max_bytes,omitempty" yaml:"evaluation_max_bytes,omitempty"`
}

type ResolvedParameters struct {
	Profile                 string            `json:"profile"`
	ProfileSchema           int               `json:"profile_schema"`
	Epochs                  int64             `json:"epochs,omitempty"`
	RequestedTokens         int64             `json:"requested_tokens,omitempty"`
	Steps                   int64             `json:"steps"`
	BatchSize               int64             `json:"batch_size"`
	GradientAccumulation    int64             `json:"gradient_accumulation_steps,omitempty"`
	ComputePrecision        string            `json:"compute_precision,omitempty"`
	ActivationCheckpointing bool              `json:"activation_checkpointing,omitempty"`
	Compile                 bool              `json:"compile,omitempty"`
	DistributionPolicy      string            `json:"distribution_policy,omitempty"`
	SequenceLength          int64             `json:"sequence_length"`
	LearningRate            float64           `json:"learning_rate"`
	Seed                    uint64            `json:"seed"`
	Optimizer               Optimizer         `json:"optimizer"`
	Schedule                Schedule          `json:"schedule"`
	Data                    DataPlan          `json:"data"`
	Evaluation              *EvaluationPolicy `json:"evaluation,omitempty"`
	CheckpointEvery         int64             `json:"checkpoint_every"`
	EvaluateEvery           int64             `json:"evaluate_every"`
	PlannedTokenCapacity    int64             `json:"planned_token_capacity"`
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
	WarmdownSteps    int64   `json:"warmdown_steps,omitempty"`
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
	RunID                  string
	Stage                  string
	Objective              string
	Conversation           ConversationTransform
	ArchitectureSHA256     string
	Architecture           json.RawMessage
	Tokenizer              TokenizerSpec
	BOM                    corpus.BOM
	Inputs                 []Input
	Parameters             ResolvedParameters
	Parallelism            Parallelism
	Records                RecordSource
	EvaluationRecords      RecordSource
	EvaluationSet          EvaluationSet
	PreTokenize            bool
	DataNodeRank           int
	Initialization         *Initialization
	Resume                 *ResumePoint
	ArtifactDirectory      string
	ArtifactPrefix         string
	PreparedCacheDirectory string
	Report                 func(Event)
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
	Kind                 string      `json:"kind"`
	Message              string      `json:"message,omitempty"`
	Step                 int64       `json:"step,omitempty"`
	Tokens               int64       `json:"tokens,omitempty"`
	Loss                 *float64    `json:"loss,omitempty"`
	GradientNorm         *float64    `json:"gradient_norm,omitempty"`
	LearningRate         float64     `json:"learning_rate,omitempty"`
	TokensPerSecond      float64     `json:"tokens_per_second,omitempty"`
	DurationSeconds      float64     `json:"duration_seconds,omitempty"`
	DataWaitSeconds      float64     `json:"data_wait_seconds,omitempty"`
	PeakMemoryBytes      uint64      `json:"peak_memory_bytes,omitempty"`
	TrainingFLOPs        float64     `json:"training_flops,omitempty"`
	AchievedTFLOPS       float64     `json:"achieved_tflops,omitempty"`
	ModelFLOPUtilization float64     `json:"model_flop_utilization,omitempty"`
	SkippedSteps         int64       `json:"skipped_steps,omitempty"`
	ETASeconds           int64       `json:"eta_seconds,omitempty"`
	Checkpoint           *Checkpoint `json:"checkpoint,omitempty"`
	Evaluation           *Evaluation `json:"evaluation,omitempty"`
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
