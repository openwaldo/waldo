// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"fmt"
)

// Fitness evidence statuses. Native WALDO construction is exact. A provider
// constructs its own modules, so a WALDO decomposition of a provider config is
// at most an estimate; the measured count in runtime.json is authoritative.
const (
	FitnessExact       = "exact"
	FitnessEstimated   = "estimated"
	FitnessUnavailable = "unavailable"
)

const transformersHybridModelClass = "Qwen3_5ForCausalLM"

// transformersStructure is a family-aware projection of a provider config. It
// models the parameter-bearing modules of the pinned Transformers classes in
// the registry and their count-affecting config fields. Unknown provider
// behavior keeps it an estimate rather than an exact count.
type transformersStructure struct {
	tied                                    bool
	headDim, queryWidth, kvWidth            uint64
	embedding, output                       uint64
	query, key, value, attentionOutput      uint64 // per layer, weights only
	gate, up, down                          uint64 // per layer and per expert
	experts, expertsPerToken, router        uint64 // router is per layer
	normsPerLayer, qkNormsPerLayer, final   uint64
	attentionBiasPerLayer, mlpBiasPerExpert uint64
}

// transformersFamilies records the pinned defaults that change shapes. A zero
// head dimension means the provider derives hidden_size/num_attention_heads.
var transformersFamilies = map[string]struct {
	headDim                   uint64
	tied                      bool
	layerNorms                uint64
	attentionBias, mlpBias    bool // whether the config flag is honored
	alwaysQKVBias, qkNorm     bool
	fullWidthQKNorm, mixtures bool
}{
	"LlamaForCausalLM":   {layerNorms: 2, attentionBias: true, mlpBias: true},
	"Qwen2ForCausalLM":   {layerNorms: 2, alwaysQKVBias: true},
	"Qwen3ForCausalLM":   {headDim: 128, layerNorms: 2, attentionBias: true, qkNorm: true},
	"MistralForCausalLM": {layerNorms: 2},
	"Gemma2ForCausalLM":  {headDim: 256, tied: true, layerNorms: 4, attentionBias: true},
	"Phi3ForCausalLM":    {layerNorms: 2},
	"Olmo2ForCausalLM":   {layerNorms: 2, attentionBias: true, qkNorm: true, fullWidthQKNorm: true},
	"MixtralForCausalLM": {layerNorms: 2, mixtures: true},
}

// transformersStructure returns false for provider classes without a maintained
// decomposition, currently the Qwen3.5 linear/full-attention hybrid.
func (architecture Architecture) transformersStructure() (transformersStructure, bool, error) {
	spec := architecture.Transformers
	family, ok := transformersFamilies[spec.ModelClass]
	if !ok {
		return transformersStructure{}, false, nil
	}
	encoded, err := json.Marshal(spec.Config)
	if err != nil {
		return transformersStructure{}, false, err
	}
	var config struct {
		HeadDim       *uint64 `json:"head_dim"`
		Tied          *bool   `json:"tie_word_embeddings"`
		AttentionBias bool    `json:"attention_bias"`
		MLPBias       bool    `json:"mlp_bias"`
	}
	if err := json.Unmarshal(encoded, &config); err != nil {
		return transformersStructure{}, false, fmt.Errorf("architecture.config structure: %w", err)
	}
	h, i := architecture.HiddenSize, architecture.IntermediateSize
	s := transformersStructure{tied: family.tied, headDim: family.headDim, experts: 1, expertsPerToken: 1}
	if config.Tied != nil {
		s.tied = *config.Tied
	}
	if config.HeadDim != nil {
		s.headDim = *config.HeadDim
	}
	if s.headDim == 0 {
		s.headDim = h / architecture.AttentionHeads
	}
	if family.mixtures {
		if s.experts, s.expertsPerToken, err = spec.ExpertCounts(); err != nil {
			return transformersStructure{}, false, err
		}
	}
	var overflow error
	product := func(left, right uint64) uint64 {
		value, err := multiply(left, right)
		if overflow == nil {
			overflow = err
		}
		return value
	}
	s.queryWidth = product(architecture.AttentionHeads, s.headDim)
	s.kvWidth = product(architecture.KeyValueHeads, s.headDim)
	s.embedding = product(architecture.VocabularySize, h)
	s.query, s.key, s.attentionOutput = product(h, s.queryWidth), product(h, s.kvWidth), product(s.queryWidth, h)
	s.gate = product(h, i)
	s.normsPerLayer = product(family.layerNorms, h)
	if family.mixtures {
		s.router = product(h, s.experts)
	}
	if overflow != nil {
		return transformersStructure{}, false, overflow
	}
	s.value, s.up, s.down, s.final = s.key, s.gate, s.gate, h
	if !s.tied {
		s.output = s.embedding
	}
	if family.qkNorm {
		s.qkNormsPerLayer = 2 * s.headDim
		if family.fullWidthQKNorm {
			s.qkNormsPerLayer = s.queryWidth + s.kvWidth
		}
	}
	switch {
	case family.alwaysQKVBias:
		s.attentionBiasPerLayer = s.queryWidth + 2*s.kvWidth
	case family.attentionBias && config.AttentionBias:
		s.attentionBiasPerLayer = s.queryWidth + 2*s.kvWidth + h
	}
	if family.mlpBias && config.MLPBias {
		s.mlpBiasPerExpert = 2*i + h
	}
	if _, err := s.total(architecture.Layers); err != nil {
		return transformersStructure{}, false, err
	}
	return s, true, nil
}

func (s transformersStructure) attentionPerLayer() uint64 {
	return s.query + s.key + s.value + s.attentionOutput
}

func (s transformersStructure) expertParameters() uint64 {
	return s.gate + s.up + s.down + s.mlpBiasPerExpert
}

func (s transformersStructure) total(layers uint64) (uint64, error) {
	experts, err := multiply(s.experts, s.expertParameters())
	if err != nil {
		return 0, err
	}
	block, err := add(s.attentionPerLayer(), s.attentionBiasPerLayer, experts, s.router, s.normsPerLayer, s.qkNormsPerLayer)
	if err != nil {
		return 0, err
	}
	blocks, err := multiply(layers, block)
	if err != nil {
		return 0, err
	}
	return add(s.embedding, s.output, blocks, s.final)
}

// active counts parameters used for one token: all non-expert parameters
// plus the selected expert fraction, matching runtime.json's definition.
func (s transformersStructure) active(layers uint64) uint64 {
	return layers*(s.attentionPerLayer()+s.attentionBiasPerLayer+s.expertsPerToken*s.expertParameters()+s.router+s.normsPerLayer+s.qkNormsPerLayer) + s.embedding + s.output + s.final
}

func transformersArchitectureFitness(architecture Architecture, total uint64) (ArchitectureFitness, error) {
	s, ok, err := architecture.transformersStructure()
	if err != nil {
		return ArchitectureFitness{}, err
	}
	if !ok {
		return ArchitectureFitness{
			ParameterCountStatus: FitnessEstimated, DecompositionStatus: FitnessUnavailable, ActiveParameterStatus: FitnessUnavailable,
			VocabularySize: architecture.VocabularySize, TotalParameters: total,
			GQAQueryHeads: architecture.AttentionHeads, GQAKeyValueHeads: architecture.KeyValueHeads,
			Basis:   "dense-decoder proxy for a provider-constructed hybrid; WALDO does not decompose its linear-attention and full-attention layers, so component, active-parameter, and architecture-aware compute values are unavailable; runtime.json records the measured count",
			Formula: "unavailable",
		}, nil
	}
	layers := architecture.Layers
	calculated, err := s.total(layers)
	if err != nil {
		return ArchitectureFitness{}, err
	}
	if calculated != total {
		return ArchitectureFitness{}, fmt.Errorf("provider structure estimate %d does not match WALDO forecast %d", calculated, total)
	}
	tokenIO := s.embedding + s.output
	attention := layers * s.attentionPerLayer()
	mlp := layers * (s.experts*(s.gate+s.up+s.down) + s.router)
	norms := layers*(s.normsPerLayer+s.qkNormsPerLayer) + s.final
	biases := layers * (s.attentionBiasPerLayer + s.experts*s.mlpBiasPerExpert)
	active := s.active(layers)
	a := ArchitectureFitness{
		ParameterCountStatus: FitnessEstimated, DecompositionStatus: FitnessEstimated, ActiveParameterStatus: FitnessEstimated,
		VocabularySize: architecture.VocabularySize, TiedEmbeddings: s.tied,
		TokenEmbeddingParameters: s.embedding, OutputProjectionParameters: s.output, TokenIOParameters: tokenIO,
		QueryParametersPerLayer: s.query, KeyParametersPerLayer: s.key, ValueParametersPerLayer: s.value,
		AttentionOutputParametersPerLayer: s.attentionOutput, AttentionParameters: attention,
		MLPGateParametersPerLayer: s.experts * s.gate, MLPUpParametersPerLayer: s.experts * s.up, MLPDownParametersPerLayer: s.experts * s.down, MLPParameters: mlp,
		NormalizationParameters: norms, BiasParameters: biases, OtherParameters: norms + biases,
		TotalParameters: total, NonEmbeddingParameters: total - tokenIO, ActiveParametersPerToken: &active,
		TokenIOPercent: percent(tokenIO, total), AttentionPercent: percent(attention, total), MLPPercent: percent(mlp, total),
		EmbeddingPercent: percent(tokenIO, total), OtherPercent: percent(norms+biases, total),
		GQAQueryHeads: architecture.AttentionHeads, GQAKeyValueHeads: architecture.KeyValueHeads, GQAKVWidth: s.kvWidth, HeadDimension: s.headDim,
		Basis:   fmt.Sprintf("WALDO structural estimate of the pinned provider class %s from architecture.config; the provider constructs the modules, and runtime.json records the measured count", architecture.Transformers.ModelClass),
		Formula: "token embedding + optional untied output + layers * (Q + K + V + O + gated MLP + norms + config-enabled biases + QK norms) + final norm, using config head_dim or the pinned class default",
	}
	if s.experts > 1 {
		a.ExpertsPerLayer, a.ExpertsPerToken = s.experts, s.expertsPerToken
		a.ExpertParameters = layers * s.experts * s.expertParameters()
		a.RouterParameters = layers * s.router
		a.Formula = "token embedding + optional untied output + layers * (Q + K + V + O + experts * (gate + up + down) + router + norms) + final norm; total is resident parameters including every expert, active is non-expert parameters plus experts_per_token experts"
	}
	return a, nil
}

// transformersTrainingFLOPsPerToken mirrors the native estimate using the
// provider widths and only the routed experts each token executes.
func transformersTrainingFLOPsPerToken(architecture Architecture, sequenceLength uint64) (float64, bool, error) {
	s, ok, err := architecture.transformersStructure()
	if err != nil || !ok {
		return 0, false, err
	}
	matrices := float64(s.attentionPerLayer() + s.expertsPerToken*(s.gate+s.up+s.down) + s.router)
	forwardLayer := 2*matrices + 4*float64(s.queryWidth)*float64(sequenceLength)
	return 3 * (float64(architecture.Layers)*forwardLayer + 2*float64(architecture.HiddenSize)*float64(architecture.VocabularySize)), true, nil
}
