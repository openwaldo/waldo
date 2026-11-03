// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/training"
	"gopkg.in/yaml.v3"
)

// ComposeDocument separates the opaque external contract from the native
// architecture. Exactly one branch is populated. Loading never acquires sources
// or executes a probe; those are explicit subsequent steps.
type ComposeDocument struct {
	Builtin  *Compose
	External *ExternalCompose
}
type TrainerApplication struct {
	Source      application.Source `json:"source" yaml:"source"`
	Environment string             `json:"environment" yaml:"environment"`
	Determinism string             `json:"determinism" yaml:"determinism"`
}
type TokenizerApplication struct {
	Source        application.Source `json:"source" yaml:"source"`
	Environment   string             `json:"environment" yaml:"environment"`
	Configuration map[string]any     `json:"configuration" yaml:"configuration"`
}
type ExternalCompose struct {
	Kind         string               `json:"kind" yaml:"kind"`
	Schema       int                  `json:"schema" yaml:"schema"`
	Base         *ComposeBase         `json:"base,omitempty" yaml:"base,omitempty"`
	Trainer      TrainerApplication   `json:"trainer" yaml:"trainer"`
	Tokenizer    TokenizerApplication `json:"tokenizer" yaml:"tokenizer"`
	Architecture map[string]any       `json:"architecture" yaml:"architecture"`
	Interaction  Interaction          `json:"interaction,omitzero" yaml:"interaction,omitempty"`
	Stages       []ExternalStage      `json:"stages" yaml:"stages"`
}
type ExternalStage struct {
	Name         string                          `json:"name" yaml:"name"`
	Type         string                          `json:"type" yaml:"type"`
	Objective    string                          `json:"objective" yaml:"objective"`
	Conversation *training.ConversationTransform `json:"conversation,omitempty" yaml:"conversation,omitempty"`
	Filter       *corpus.RecordFilter            `json:"filter,omitempty" yaml:"filter,omitempty"`
	Corpora      []CorpusSelection               `json:"corpora" yaml:"corpora"`
	Data         ExternalData                    `json:"data" yaml:"data"`
	Trainer      map[string]any                  `json:"trainer" yaml:"trainer"`
}

// ExternalData contains only WALDO-owned parameters. The budget, ordering,
// evaluation and cadence rules are the same as the native data contract.
type ExternalData struct {
	Profile              string   `json:"profile,omitempty" yaml:"profile,omitempty"`
	Tokens               int64    `json:"tokens,omitempty" yaml:"tokens,omitempty"`
	Epochs               int64    `json:"epochs,omitempty" yaml:"epochs,omitempty"`
	Steps                int64    `json:"steps,omitempty" yaml:"steps,omitempty"`
	SequenceLength       int64    `json:"sequence_length" yaml:"sequence_length"`
	GlobalBatchSequences int64    `json:"global_batch_sequences" yaml:"global_batch_sequences"`
	Seed                 uint64   `json:"seed" yaml:"seed"`
	CheckpointEvery      *int64   `json:"checkpoint_every,omitempty" yaml:"checkpoint_every,omitempty"`
	EvaluateEvery        *int64   `json:"evaluate_every,omitempty" yaml:"evaluate_every,omitempty"`
	ShuffleBufferRecords *int     `json:"shuffle_buffer_records,omitempty" yaml:"shuffle_buffer_records,omitempty"`
	ShuffleBufferBytes   *int64   `json:"shuffle_buffer_bytes,omitempty" yaml:"shuffle_buffer_bytes,omitempty"`
	EvaluationFraction   *float64 `json:"evaluation_fraction,omitempty" yaml:"evaluation_fraction,omitempty"`
	EvaluationSelection  string   `json:"evaluation_selection,omitempty" yaml:"evaluation_selection,omitempty"`
	EvaluationMaxRecords *int     `json:"evaluation_max_records,omitempty" yaml:"evaluation_max_records,omitempty"`
	EvaluationMaxBytes   *int64   `json:"evaluation_max_bytes,omitempty" yaml:"evaluation_max_bytes,omitempty"`
}

func LoadComposeDocument(path string) (ComposeDocument, string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return ComposeDocument{}, "", err
	}
	data, err := os.ReadFile(absolute)
	if err != nil {
		return ComposeDocument{}, "", err
	}
	// This header is only a dispatch hint. Both complete decoders remain strict.
	var fields map[string]yaml.Node
	if err := yaml.Unmarshal(data, &fields); err != nil {
		return ComposeDocument{}, "", err
	}
	if _, external := fields["trainer"]; !external {
		c, p, err := LoadCompose(path)
		if err != nil {
			return ComposeDocument{}, "", err
		}
		return ComposeDocument{Builtin: &c}, p, nil
	}
	var c ExternalCompose
	if err := application.DecodeStrict(data, &c); err != nil {
		return ComposeDocument{}, "", fmt.Errorf("%s: %w", absolute, err)
	}
	if err := c.Validate(); err != nil {
		return ComposeDocument{}, "", fmt.Errorf("%s: %w", absolute, err)
	}
	return ComposeDocument{External: &c}, absolute, nil
}

func (c ExternalCompose) Validate() error {
	if c.Kind != "waldo-model-compose" || c.Schema != 2 {
		return fmt.Errorf("external trainers require waldo-model-compose schema 2")
	}
	if err := c.Trainer.Source.Validate(); err != nil {
		return fmt.Errorf("trainer: %w", err)
	}
	if err := c.Tokenizer.Source.Validate(); err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	if c.Trainer.Environment == "" || c.Tokenizer.Environment == "" {
		return fmt.Errorf("trainer and tokenizer must select named environments")
	}
	switch c.Trainer.Determinism {
	case "required", "best-effort", "off":
	default:
		return fmt.Errorf("trainer.determinism must be required, best-effort or off")
	}
	if c.Architecture == nil {
		return fmt.Errorf("external architecture must be an object")
	}
	if c.Tokenizer.Configuration == nil {
		return fmt.Errorf("tokenizer.configuration must be an object")
	}
	if err := c.Interaction.Validate(); err != nil {
		return err
	}
	if c.Base != nil {
		return fmt.Errorf("external base initialization is not implemented yet")
	}
	if len(c.Stages) == 0 {
		return fmt.Errorf("at least one training stage is required")
	}
	seen := map[string]bool{}
	tools := false
	for _, s := range c.Stages {
		if !validName.MatchString(s.Name) || seen[s.Name] {
			return fmt.Errorf("invalid or duplicate stage name %q", s.Name)
		}
		seen[s.Name] = true
		switch s.Type {
		case "pre-training", "fine-tuning", "alignment", "other":
		default:
			return fmt.Errorf("stage %s has unsupported type %q", s.Name, s.Type)
		}
		if s.Objective != "causal-language-modeling" && s.Objective != "assistant-response-modeling" {
			return fmt.Errorf("stage %s has unsupported objective", s.Name)
		}
		if s.Trainer == nil {
			return fmt.Errorf("stage %s trainer must be an object (may be empty)", s.Name)
		}
		if s.Conversation != nil {
			if err := s.Conversation.Validate(); err != nil {
				return err
			}
			if s.Conversation.Tools || s.Conversation.Template != c.Interaction.Template {
				return fmt.Errorf("stage %s conversation must match model interaction", s.Name)
			}
			for _, role := range s.Conversation.SupervisedRoles {
				tools = tools || (role == "assistant" && s.Objective == "assistant-response-modeling")
			}
		} else if s.Objective == "assistant-response-modeling" {
			return fmt.Errorf("stage %s requires conversation transformation", s.Name)
		}
		if len(s.Corpora) == 0 {
			return fmt.Errorf("stage %s requires corpora", s.Name)
		}
		paths := map[string]bool{}
		for _, selection := range s.Corpora {
			if selection.Path == "" || paths[selection.Path] {
				return fmt.Errorf("stage %s has empty or duplicate corpus", s.Name)
			}
			paths[selection.Path] = true
			if selection.Filter != nil {
				if err := selection.Filter.Validate(); err != nil {
					return err
				}
			}
		}
		if s.Filter != nil {
			if err := s.Filter.Validate(); err != nil {
				return err
			}
		}
		if _, err := s.ResolveData(); err != nil {
			return fmt.Errorf("stage %s data: %w", s.Name, err)
		}
	}
	if c.Interaction.Tools && !tools {
		return fmt.Errorf("interaction tools require supervised assistant responses")
	}
	return nil
}

// ResolvedExternalData contains only resolved WALDO-owned data facts.
type ResolvedExternalData = training.ExternalDataParameters

// ResolveData reuses WALDO's existing data rules without exporting native
// optimizer defaults as external trainer parameters.
func (s ExternalStage) ResolveData() (ResolvedExternalData, error) {
	resolved, err := s.DataStage().ResolvePlanningParameters()
	if err != nil {
		return ResolvedExternalData{}, err
	}
	if resolved.Data.Order == "corpus-weighted-shuffle-v1" {
		for _, selection := range s.Corpora {
			if resolved.Data.CorpusWeights[selection.Path] == 0 {
				return ResolvedExternalData{}, fmt.Errorf("corpus %s requires a positive weight", selection.Path)
			}
		}
	}
	return ResolvedExternalData{
		Epochs: resolved.Epochs, RequestedTokens: resolved.RequestedTokens, Steps: resolved.Steps,
		GlobalBatchSequences: resolved.BatchSize, SequenceLength: resolved.SequenceLength, Seed: resolved.Seed,
		Order: resolved.Data, Evaluation: resolved.Evaluation, CheckpointEvery: resolved.CheckpointEvery,
		EvaluateEvery: resolved.EvaluateEvery, PlannedTokenCapacity: resolved.PlannedTokenCapacity,
	}, nil
}

type ExternalSources struct {
	Trainer   application.ResolvedSource
	Tokenizer application.ResolvedSource
}

func (c ExternalCompose) ResolveSources(ctx context.Context, cache string) (ExternalSources, error) {
	if err := c.Validate(); err != nil {
		return ExternalSources{}, err
	}
	trainer, err := application.ResolveSource(ctx, cache, c.Trainer.Source, "waldo-trainer", c.Trainer.Environment)
	if err != nil {
		return ExternalSources{}, fmt.Errorf("trainer: %w", err)
	}
	for _, s := range c.Stages {
		supported := false
		for _, o := range trainer.Manifest.Objectives {
			supported = supported || s.Objective == o
		}
		if !supported {
			return ExternalSources{}, fmt.Errorf("trainer does not support stage %s objective %s", s.Name, s.Objective)
		}
	}
	tokenizer, err := application.ResolveSource(ctx, cache, c.Tokenizer.Source, "waldo-tokenizer", c.Tokenizer.Environment)
	if err != nil {
		return ExternalSources{}, fmt.Errorf("tokenizer: %w", err)
	}
	return ExternalSources{Trainer: trainer, Tokenizer: tokenizer}, nil
}

// ValidateSchemas is called after probe, not during manifest parsing. Stage
// blocks may omit properties only when the received schema allows it; resolve
// fills defaults later. Complete resolve results use Validate and Split.
func (c ExternalCompose) ValidateSchemas(architecture, parameters, tokenizer *application.ParameterSchema) error {
	if architecture == nil || parameters == nil || tokenizer == nil {
		return fmt.Errorf("probe must supply architecture, parameter and tokenizer schemas")
	}
	if err := architecture.ValidateArchitecture(c.Architecture); err != nil {
		return fmt.Errorf("architecture: %w", err)
	}
	if err := tokenizer.Validate(c.Tokenizer.Configuration); err != nil {
		return fmt.Errorf("tokenizer.configuration: %w", err)
	}
	for _, stage := range c.Stages {
		if err := parameters.Validate(stage.Trainer); err != nil {
			return fmt.Errorf("stage %s trainer: %w", stage.Name, err)
		}
	}
	return nil
}
