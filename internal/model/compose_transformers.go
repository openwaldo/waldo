// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"

	"github.com/openwaldo/waldo/internal/training"
	"gopkg.in/yaml.v3"
)

func normalizeTransformersCompose(data []byte) ([]byte, error) {
	var input struct {
		Kind     string `yaml:"kind"`
		Schema   int    `yaml:"schema"`
		Training struct {
			Engine  string              `yaml:"engine"`
			Package training.PackagePin `yaml:"package"`
		} `yaml:"training"`
		Base         *ComposeBase `yaml:"base,omitempty"`
		Interaction  Interaction  `yaml:"interaction"`
		Architecture struct {
			Provider    string         `yaml:"provider"`
			ModelClass  string         `yaml:"model_class"`
			ConfigClass string         `yaml:"config_class"`
			Config      map[string]any `yaml:"config"`
			Tokenizer   Tokenizer      `yaml:"tokenizer"`
		} `yaml:"architecture"`
		Stages []struct {
			Stage   `yaml:",inline"`
			Trainer training.TransformersTrainer `yaml:"trainer"`
		} `yaml:"stages"`
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&input); err != nil {
		return nil, err
	}
	if input.Schema != 2 || input.Training.Engine != training.BackendTransformers || input.Architecture.Provider != training.BackendTransformers {
		return nil, fmt.Errorf("Transformers requires schema 2 and matching training.engine/architecture.provider")
	}
	if input.Base != nil && input.Base.Source != "" {
		return nil, fmt.Errorf("Transformers base.source is not supported; omit base for random initialization or use a compatible managed model")
	}
	spec := &training.TransformersModel{Package: input.Training.Package, ModelClass: input.Architecture.ModelClass, ConfigClass: input.Architecture.ConfigClass, Config: input.Architecture.Config}
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	architecture, err := transformersArchitecture(spec, input.Architecture.Tokenizer)
	if err != nil {
		return nil, err
	}
	if err := architecture.Validate(); err != nil {
		return nil, err
	}
	if err := architecture.validateTokenizer(); err != nil {
		return nil, err
	}
	compose := Compose{Kind: input.Kind, Schema: ComposeSchema, Base: input.Base, Architecture: architecture, Interaction: input.Interaction}
	for _, item := range input.Stages {
		stage := item.Stage
		if stage.Parameters.Trainer != nil {
			return nil, fmt.Errorf("use stage.trainer, not parameters.trainer")
		}
		if err := item.Trainer.Validate(); err != nil {
			return nil, err
		}
		if stage.Parameters.BatchSize != 0 || stage.Parameters.LearningRate != 0 || stage.Parameters.WeightDecay != nil || stage.Parameters.WarmupSteps != nil || stage.Parameters.CheckpointEvery != nil || stage.Parameters.EvaluateEvery != nil {
			return nil, fmt.Errorf("Transformers optimizer/batch arguments belong in stage.trainer.arguments; checkpoint/evaluation scheduling is not supported yet")
		}
		arguments, err := json.Marshal(item.Trainer.Arguments)
		if err != nil {
			return nil, err
		}
		var projected struct {
			Batch        int64   `json:"per_device_train_batch_size"`
			Accumulation int64   `json:"gradient_accumulation_steps"`
			Rate         float64 `json:"learning_rate"`
		}
		if err := json.Unmarshal(arguments, &projected); err != nil {
			return nil, err
		}
		if _, exists := item.Trainer.Arguments["gradient_accumulation_steps"]; !exists {
			projected.Accumulation = 1
		}
		if projected.Batch <= 0 || projected.Accumulation <= 0 || projected.Batch > math.MaxInt64/projected.Accumulation || projected.Rate <= 0 {
			return nil, fmt.Errorf("trainer requires positive per_device_train_batch_size, gradient_accumulation_steps, and learning_rate")
		}
		stage.Parameters.BatchSize = projected.Batch * projected.Accumulation
		stage.Parameters.LearningRate = projected.Rate
		zero := int64(0)
		stage.Parameters.CheckpointEvery, stage.Parameters.EvaluateEvery = &zero, &zero
		stage.Parameters.Trainer = &item.Trainer
		compose.Stages = append(compose.Stages, stage)
	}
	return json.Marshal(compose)
}

func (architecture Architecture) validateTokenizer() error {
	tokenizer := architecture.Tokenizer
	if tokenizer.HuggingFace != nil {
		if tokenizer.Name != "huggingface" || architecture.Transformers == nil {
			return fmt.Errorf("pinned Hugging Face assets require name: huggingface and the Transformers backend")
		}
		return tokenizer.HuggingFace.Validate(tokenizer.Revision, architecture.VocabularySize)
	}
	_, _, err := training.ResolveTokenizer(tokenizer.Name, tokenizer.Revision, architecture.VocabularySize)
	return err
}

// Derive lifecycle resource/tokenizer projections, never model construction.
func transformersArchitecture(spec *training.TransformersModel, tokenizer Tokenizer) (Architecture, error) {
	encoded, err := json.Marshal(spec.Config)
	if err != nil {
		return Architecture{}, err
	}
	var dimensions struct {
		Context      uint64 `json:"max_position_embeddings"`
		Vocabulary   uint64 `json:"vocab_size"`
		Hidden       uint64 `json:"hidden_size"`
		Intermediate uint64 `json:"intermediate_size"`
		Layers       uint64 `json:"num_hidden_layers"`
		Heads        uint64 `json:"num_attention_heads"`
		KVHeads      uint64 `json:"num_key_value_heads"`
		Tied         bool   `json:"tie_word_embeddings"`
		DType        string `json:"dtype"`
	}
	if err := json.Unmarshal(encoded, &dimensions); err != nil {
		return Architecture{}, fmt.Errorf("architecture.config dimensions: %w", err)
	}
	if dimensions.DType == "" {
		dimensions.DType = "float32"
	}
	return Architecture{Transformers: spec, Family: training.BackendTransformers, ContextTokens: dimensions.Context, VocabularySize: dimensions.Vocabulary, HiddenSize: dimensions.Hidden, IntermediateSize: dimensions.Intermediate, Layers: dimensions.Layers, AttentionHeads: dimensions.Heads, KeyValueHeads: dimensions.KVHeads, TieEmbeddings: dimensions.Tied, ParameterDType: dimensions.DType, Tokenizer: tokenizer}, nil
}
