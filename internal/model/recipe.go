// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package model owns model compose files, immutable architecture identity, build
// plans, model/run BOMs, and durable lifecycle state.
package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/training"
	"gopkg.in/yaml.v3"
)

const ComposeSchema = 1

var validName = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)

type Compose struct {
	Kind         string       `json:"kind" yaml:"kind"`
	Schema       int          `json:"schema" yaml:"schema"`
	Base         *ComposeBase `json:"base,omitempty" yaml:"base,omitempty"`
	Architecture Architecture `json:"architecture" yaml:"architecture"`
	Interaction  Interaction  `json:"interaction,omitzero" yaml:"interaction,omitempty"`
	Stages       []Stage      `json:"stages" yaml:"stages"`
}

const InteractionUserAssistantV1 = "user-assistant-v1"
const InteractionChatMLV1 = "chatml-v1"

// Interaction is the portable inference-time prompt contract learned by a
// model. The zero value deliberately means raw causal continuation.
type Interaction struct {
	Template string `json:"template,omitempty" yaml:"template,omitempty"`
	Tools    bool   `json:"tools,omitempty" yaml:"tools,omitempty"`
}

func (interaction Interaction) IsZero() bool { return interaction.Template == "" && !interaction.Tools }

func (interaction Interaction) Validate() error {
	if interaction.Tools && !interaction.Conversational() {
		return fmt.Errorf("interaction tools require a conversational template")
	}
	if interaction.Template == "" || interaction.Template == InteractionUserAssistantV1 || interaction.Template == InteractionChatMLV1 {
		return nil
	}
	return fmt.Errorf("unsupported interaction template %q", interaction.Template)
}

func (interaction Interaction) Conversational() bool {
	return interaction.Template == InteractionUserAssistantV1 || interaction.Template == InteractionChatMLV1
}

func (interaction Interaction) Prompt(history, user string) string {
	if !interaction.Conversational() {
		if history == "" {
			return user
		}
		return history + "\n" + user
	}
	if interaction.Template == InteractionChatMLV1 {
		prefix := history
		if prefix != "" && !strings.HasSuffix(prefix, "\n") {
			prefix += "\n"
		}
		return prefix + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
	}
	prefix := ""
	if history != "" {
		prefix = strings.TrimRight(history, "\n") + "\n\n"
	}
	return prefix + "User: " + user + "\n\nAssistant:"
}

func (interaction Interaction) Stops() []string {
	if interaction.Template == InteractionChatMLV1 {
		return []string{"<|im_end|>"}
	}
	if interaction.Conversational() {
		return []string{"\n\nUser:"}
	}
	return nil
}

func (interaction Interaction) TrimResponse(value string) string {
	for _, stop := range interaction.Stops() {
		if index := strings.Index(value, stop); index >= 0 {
			value = value[:index]
		}
	}
	return strings.TrimRight(value, "\r\n")
}

func (interaction Interaction) CompleteTurn(prompt, response string) string {
	if interaction.Template == InteractionChatMLV1 {
		return prompt + response + "<|im_end|>\n"
	}
	return prompt + response
}

type ComposeBase struct {
	Model          string `json:"model,omitempty" yaml:"model,omitempty"`
	Source         string `json:"source,omitempty" yaml:"source,omitempty"`
	OriginSHA256   string `json:"origin_sha256,omitempty" yaml:"origin_sha256,omitempty"`
	ModelID        string `json:"model_id,omitempty" yaml:"model_id,omitempty"`
	RunID          string `json:"run_id,omitempty" yaml:"run_id,omitempty"`
	RunBOMSHA256   string `json:"run_bom_sha256,omitempty" yaml:"run_bom_sha256,omitempty"`
	ArtifactSHA256 string `json:"artifact_sha256,omitempty" yaml:"artifact_sha256,omitempty"`
	ArtifactBytes  int64  `json:"artifact_bytes,omitempty" yaml:"artifact_bytes,omitempty"`
}

type Architecture struct {
	Family           string    `json:"family" yaml:"family"`
	ContextTokens    uint64    `json:"context_tokens" yaml:"context_tokens"`
	VocabularySize   uint64    `json:"vocabulary_size" yaml:"vocabulary_size"`
	HiddenSize       uint64    `json:"hidden_size" yaml:"hidden_size"`
	IntermediateSize uint64    `json:"intermediate_size" yaml:"intermediate_size"`
	Layers           uint64    `json:"layers" yaml:"layers"`
	AttentionHeads   uint64    `json:"attention_heads" yaml:"attention_heads"`
	KeyValueHeads    uint64    `json:"key_value_heads" yaml:"key_value_heads"`
	Dropout          float64   `json:"dropout,omitempty" yaml:"dropout,omitempty"`
	TieEmbeddings    bool      `json:"tie_embeddings" yaml:"tie_embeddings"`
	ParameterDType   string    `json:"parameter_dtype" yaml:"parameter_dtype"`
	Tokenizer        Tokenizer `json:"tokenizer" yaml:"tokenizer"`
}

type Tokenizer struct {
	Name     string `json:"name" yaml:"name"`
	Revision string `json:"revision" yaml:"revision"`
}

type Stage struct {
	Name         string                          `json:"name" yaml:"name"`
	Type         string                          `json:"type" yaml:"type"`
	Objective    string                          `json:"objective" yaml:"objective"`
	Conversation *training.ConversationTransform `json:"conversation,omitempty" yaml:"conversation,omitempty"`
	Filter       *corpus.RecordFilter            `json:"filter,omitempty" yaml:"filter,omitempty"`
	Corpora      []CorpusSelection               `json:"corpora" yaml:"corpora"`
	Parameters   training.Parameters             `json:"parameters" yaml:"parameters"`
}

// CorpusSelection preserves the compact scalar form while allowing a corpus
// to carry its own portable training configuration.
type CorpusSelection struct {
	Path   string               `json:"path" yaml:"path"`
	Weight *uint64              `json:"weight,omitempty" yaml:"weight,omitempty"`
	Filter *corpus.RecordFilter `json:"filter,omitempty" yaml:"filter,omitempty"`
}

func (selection *CorpusSelection) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) > 0 && trimmed[0] == '"' {
		return json.Unmarshal(trimmed, &selection.Path)
	}
	type plain CorpusSelection
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode((*plain)(selection)); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("corpus selection contains trailing JSON")
		}
		return err
	}
	return nil
}

func (selection CorpusSelection) MarshalJSON() ([]byte, error) {
	if selection.Weight == nil && selection.Filter == nil {
		return json.Marshal(selection.Path)
	}
	type plain CorpusSelection
	return json.Marshal(plain(selection))
}

func (selection *CorpusSelection) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind == yaml.ScalarNode {
		if node.Tag != "!!str" {
			return fmt.Errorf("corpus selection must be a path string or mapping")
		}
		selection.Path = node.Value
		return nil
	}
	if node.Kind != yaml.MappingNode {
		return fmt.Errorf("corpus selection must be a path string or mapping")
	}
	if err := validateCorpusSelectionYAML(node); err != nil {
		return err
	}
	type plain CorpusSelection
	return node.Decode((*plain)(selection))
}

func (selection CorpusSelection) MarshalYAML() (any, error) {
	if selection.Weight == nil && selection.Filter == nil {
		return selection.Path, nil
	}
	type plain CorpusSelection
	return plain(selection), nil
}

func validateCorpusSelectionYAML(node *yaml.Node) error {
	if err := knownYAMLFields(node, map[string]bool{"path": true, "weight": true, "filter": true}); err != nil {
		return err
	}
	for index := 0; index < len(node.Content); index += 2 {
		if node.Content[index].Value == "filter" {
			return validateRecordFilterYAML(node.Content[index+1])
		}
	}
	return nil
}

func validateRecordFilterYAML(node *yaml.Node) error {
	if err := knownYAMLFields(node, map[string]bool{"main_content": true, "exclude": true, "licenses": true, "languages": true, "sources": true, "date": true}); err != nil {
		return err
	}
	for index := 0; index < len(node.Content); index += 2 {
		key, value := node.Content[index].Value, node.Content[index+1]
		if key == "main_content" {
			continue
		} else if key == "date" {
			if err := knownYAMLFields(value, map[string]bool{"from": true, "to": true}); err != nil {
				return err
			}
		} else if key == "exclude" {
			if err := knownYAMLFields(value, map[string]bool{"repetitive_content": true, "boilerplate_content": true, "licenses": true}); err != nil {
				return err
			}
		} else if err := knownYAMLFields(value, map[string]bool{"include": true, "exclude": true}); err != nil {
			return err
		}
	}
	return nil
}

func knownYAMLFields(node *yaml.Node, allowed map[string]bool) error {
	if node.Kind != yaml.MappingNode {
		return fmt.Errorf("expected a mapping")
	}
	seen := map[string]bool{}
	for index := 0; index < len(node.Content); index += 2 {
		name := node.Content[index].Value
		if !allowed[name] {
			return fmt.Errorf("field %s not found", name)
		}
		if seen[name] {
			return fmt.Errorf("field %s appears more than once", name)
		}
		seen[name] = true
	}
	return nil
}

func CorpusPaths(selections []CorpusSelection) []string {
	paths := make([]string, len(selections))
	for index, selection := range selections {
		paths[index] = selection.Path
	}
	return paths
}

func NewCorpusSelections(paths []string) []CorpusSelection {
	selections := make([]CorpusSelection, len(paths))
	for index, corpusPath := range paths {
		selections[index] = CorpusSelection{Path: corpusPath}
	}
	return selections
}

func (stage Stage) ResolveParameters() (training.ResolvedParameters, error) {
	parameters, err := stage.trainingParameters()
	if err != nil {
		return training.ResolvedParameters{}, err
	}
	return training.ResolveParameters(parameters)
}

func (stage Stage) ResolvePlanningParameters() (training.ResolvedParameters, error) {
	parameters, err := stage.trainingParameters()
	if err != nil {
		return training.ResolvedParameters{}, err
	}
	return training.ResolvePlanningParameters(parameters)
}

func (stage Stage) ResolveParametersForSteps(steps int64) (training.ResolvedParameters, error) {
	parameters, err := stage.trainingParameters()
	if err != nil {
		return training.ResolvedParameters{}, err
	}
	return training.ResolveParametersForSteps(parameters, steps)
}

func (stage Stage) trainingParameters() (training.Parameters, error) {
	parameters := stage.Parameters
	inline := false
	for _, selection := range stage.Corpora {
		inline = inline || selection.Weight != nil
	}
	if inline && len(parameters.CorpusWeights) != 0 {
		return training.Parameters{}, fmt.Errorf("inline corpus weights cannot be combined with parameters.corpus_weights")
	}
	if inline {
		parameters.CorpusWeights = make(map[string]uint64, len(stage.Corpora))
		for _, selection := range stage.Corpora {
			if selection.Weight != nil {
				parameters.CorpusWeights[selection.Path] = *selection.Weight
			}
		}
	}
	return parameters, nil
}

func (stage Stage) RecordFilterPolicy(paths []string) (*corpus.RecordFilterPolicy, error) {
	configured := stage.Filter != nil
	policy := corpus.RecordFilterPolicy{Schema: corpus.RecordFilterSchema, Global: stage.Filter}
	for _, selection := range stage.Corpora {
		if selection.Filter == nil {
			continue
		}
		configured = true
		resolved, err := resolveSelectedPath(selection.Path, paths)
		if err != nil {
			return nil, err
		}
		if policy.Corpora == nil {
			policy.Corpora = map[string]corpus.RecordFilter{}
		}
		policy.Corpora[resolved] = *selection.Filter
	}
	if !configured {
		return nil, nil
	}
	if err := policy.Validate(paths); err != nil {
		return nil, err
	}
	return &policy, nil
}

func resolveSelectedPath(declared string, paths []string) (string, error) {
	var resolved string
	for _, actual := range paths {
		logical := strings.TrimSuffix(strings.TrimSuffix(actual, ".yaml"), ".json")
		if declared != actual && declared != logical {
			continue
		}
		if resolved != "" {
			return "", fmt.Errorf("corpus path %q resolves ambiguously", declared)
		}
		resolved = actual
	}
	if resolved == "" {
		return "", fmt.Errorf("corpus path %q is not in the resolved selection", declared)
	}
	return resolved, nil
}

type ArchitectureForecast struct {
	ApproximateParameters uint64 `json:"approximate_parameters"`
	ParameterBytes        uint64 `json:"parameter_bytes"`
	Formula               string `json:"formula"`
}

func LoadCompose(path string) (Compose, string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return Compose{}, "", err
	}
	data, err := os.ReadFile(absolute)
	if err != nil {
		if os.IsNotExist(err) {
			return Compose{}, "", fmt.Errorf("model compose %s does not exist", absolute)
		}
		return Compose{}, "", err
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return Compose{}, "", fmt.Errorf("model compose %s is empty", absolute)
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	var compose Compose
	if err := decoder.Decode(&compose); err != nil {
		return Compose{}, "", fmt.Errorf("%s: %w", absolute, err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple YAML documents are not allowed")
		}
		return Compose{}, "", fmt.Errorf("%s: %w", absolute, err)
	}
	compose.normalizeLegacyInteraction()
	if err := compose.Validate(); err != nil {
		return Compose{}, "", fmt.Errorf("%s: %w", absolute, err)
	}
	return compose, absolute, nil
}

// normalizeLegacyInteraction preserves schema-1 composes written before tool
// capability moved from an individual training stage to the model contract.
func (compose *Compose) normalizeLegacyInteraction() {
	for index := range compose.Stages {
		conversation := compose.Stages[index].Conversation
		if conversation == nil || !conversation.Tools {
			continue
		}
		compose.Interaction.Tools = true
		conversation.Tools = false
	}
}

func IsComposeFile(path string) (bool, error) {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if info.IsDir() {
		return false, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return false, err
	}
	var header struct {
		Kind string `yaml:"kind"`
	}
	if err := yaml.Unmarshal(data, &header); err != nil {
		return false, nil
	}
	return header.Kind == "waldo-model-compose", nil
}

func (compose Compose) Validate() error {
	if compose.Kind != "waldo-model-compose" || compose.Schema != ComposeSchema {
		return fmt.Errorf("unsupported model compose identity %q schema %d", compose.Kind, compose.Schema)
	}
	if err := compose.Interaction.Validate(); err != nil {
		return err
	}
	inheritedArchitecture := compose.Base != nil && compose.Base.Source != "" && compose.Architecture == (Architecture{})
	if !inheritedArchitecture {
		if err := compose.Architecture.Validate(); err != nil {
			return err
		}
	}
	if compose.Base != nil {
		hasModel, hasSource := compose.Base.Model != "", compose.Base.Source != ""
		if hasModel == hasSource {
			return fmt.Errorf("base must declare exactly one of model or source")
		}
		if hasModel && !validName.MatchString(compose.Base.Model) {
			return fmt.Errorf("base.model must name a locally managed model")
		}
		if hasSource {
			if compose.Base.ModelID != "" || compose.Base.RunID != "" || compose.Base.RunBOMSHA256 != "" || compose.Base.ArtifactSHA256 != "" || compose.Base.ArtifactBytes != 0 {
				return fmt.Errorf("base.source cannot declare managed-model run pins")
			}
			_, revision, err := parseHuggingFaceSource(compose.Base.Source)
			if err != nil {
				return fmt.Errorf("base.source: %w", err)
			}
			if !huggingFaceCommit.MatchString(revision) {
				return fmt.Errorf("base.source must pin an immutable Hugging Face commit")
			}
		}
		if compose.Base.ArtifactBytes < 0 {
			return fmt.Errorf("base artifact_bytes cannot be negative")
		}
	}
	if len(compose.Stages) == 0 {
		return fmt.Errorf("at least one training stage is required")
	}
	seen := map[string]bool{}
	hasToolTraining := false
	for i, stage := range compose.Stages {
		if !validName.MatchString(stage.Name) || seen[stage.Name] {
			return fmt.Errorf("stage %d has invalid or duplicate name %q", i+1, stage.Name)
		}
		seen[stage.Name] = true
		if stage.Type != "pre-training" && stage.Type != "fine-tuning" && stage.Type != "alignment" && stage.Type != "other" {
			return fmt.Errorf("stage %s has unsupported type %q; use pre-training, fine-tuning, alignment, or other", stage.Name, stage.Type)
		}
		if stage.Objective != "causal-language-modeling" && stage.Objective != "assistant-response-modeling" {
			return fmt.Errorf("stage %s has unsupported objective %q", stage.Name, stage.Objective)
		}
		if stage.Conversation != nil {
			if stage.Conversation.Tools {
				return fmt.Errorf("stage %s conversation tools must be declared once as interaction.tools", stage.Name)
			}
			if err := stage.Conversation.Validate(); err != nil {
				return fmt.Errorf("stage %s conversation: %w", stage.Name, err)
			}
			if compose.Interaction.Template != stage.Conversation.Template {
				return fmt.Errorf("stage %s conversation template %q does not match interaction template %q", stage.Name, stage.Conversation.Template, compose.Interaction.Template)
			}
			if stage.Objective == "assistant-response-modeling" {
				for _, role := range stage.Conversation.SupervisedRoles {
					hasToolTraining = hasToolTraining || role == "assistant"
				}
			}
		} else if stage.Objective == "assistant-response-modeling" {
			return fmt.Errorf("stage %s assistant-response-modeling requires conversation transformation", stage.Name)
		}
		if len(stage.Corpora) == 0 {
			return fmt.Errorf("stage %s requires at least one index path in corpora", stage.Name)
		}
		corpora := make(map[string]bool, len(stage.Corpora))
		for _, selection := range stage.Corpora {
			corpusPath := selection.Path
			if corpusPath == "" {
				return fmt.Errorf("stage %s contains an empty corpus index path", stage.Name)
			}
			if corpora[corpusPath] {
				return fmt.Errorf("stage %s contains duplicate corpus path %q", stage.Name, corpusPath)
			}
			corpora[corpusPath] = true
			if selection.Filter != nil {
				if err := selection.Filter.Validate(); err != nil {
					return fmt.Errorf("stage %s corpus %s filter: %w", stage.Name, corpusPath, err)
				}
			}
		}
		if stage.Filter != nil {
			if err := stage.Filter.Validate(); err != nil {
				return fmt.Errorf("stage %s filter: %w", stage.Name, err)
			}
		}
		resolved, err := stage.ResolvePlanningParameters()
		if err != nil {
			return fmt.Errorf("stage %s training parameters: %w", stage.Name, err)
		}
		if resolved.Data.Order == "corpus-weighted-shuffle-v1" {
			for corpusPath := range corpora {
				if resolved.Data.CorpusWeights[corpusPath] == 0 {
					return fmt.Errorf("stage %s corpus_weights does not declare corpus %q", stage.Name, corpusPath)
				}
			}
			for corpusPath := range resolved.Data.CorpusWeights {
				if !corpora[corpusPath] {
					return fmt.Errorf("stage %s corpus_weights declares unselected corpus %q", stage.Name, corpusPath)
				}
			}
		}
		if !inheritedArchitecture && uint64(stage.Parameters.SequenceLength) > compose.Architecture.ContextTokens {
			return fmt.Errorf("stage %s sequence_length exceeds architecture context_tokens", stage.Name)
		}
	}
	if compose.Interaction.Tools && !hasToolTraining {
		return fmt.Errorf("interaction tools require an assistant-response-modeling stage that supervises assistant responses")
	}
	return nil
}

func stageWithInteraction(stage Stage, interaction Interaction) Stage {
	if stage.Conversation == nil {
		return stage
	}
	conversation := *stage.Conversation
	conversation.Tools = interaction.Tools
	stage.Conversation = &conversation
	return stage
}

func (architecture Architecture) Validate() error {
	if architecture.Family != "decoder-transformer" {
		return fmt.Errorf("unsupported architecture family %q", architecture.Family)
	}
	if architecture.ContextTokens == 0 || architecture.VocabularySize == 0 || architecture.HiddenSize == 0 || architecture.IntermediateSize == 0 || architecture.Layers == 0 || architecture.AttentionHeads == 0 || architecture.KeyValueHeads == 0 {
		return fmt.Errorf("architecture dimensions must be positive")
	}
	if architecture.HiddenSize%architecture.AttentionHeads != 0 || architecture.AttentionHeads%architecture.KeyValueHeads != 0 {
		return fmt.Errorf("architecture heads must divide hidden_size and key_value_heads must divide attention_heads")
	}
	if architecture.Dropout < 0 || architecture.Dropout >= 1 || math.IsNaN(architecture.Dropout) || math.IsInf(architecture.Dropout, 0) {
		return fmt.Errorf("architecture dropout must be finite and in 0..<1")
	}
	if architecture.ParameterDType != "float32" && architecture.ParameterDType != "float16" && architecture.ParameterDType != "bfloat16" {
		return fmt.Errorf("unsupported parameter_dtype %q", architecture.ParameterDType)
	}
	if architecture.Tokenizer.Name == "" || architecture.Tokenizer.Revision == "" {
		return fmt.Errorf("tokenizer name and immutable revision are required")
	}
	_, err := architecture.Forecast()
	return err
}

func (architecture Architecture) Forecast() (ArchitectureForecast, error) {
	embedding, err := multiply(architecture.VocabularySize, architecture.HiddenSize)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	kvWidth, err := multiply(architecture.HiddenSize/architecture.AttentionHeads, architecture.KeyValueHeads)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	hiddenSquared, err := multiply(architecture.HiddenSize, architecture.HiddenSize)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	qAndOutput, err := multiply(2, hiddenSquared)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	kvProjection, err := multiply(architecture.HiddenSize, kvWidth)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	kvProjection, err = multiply(2, kvProjection)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	attention, err := add(qAndOutput, kvProjection)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	mlp, err := multiply(architecture.HiddenSize, architecture.IntermediateSize)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	mlp, err = multiply(3, mlp)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	block, err := add(attention, mlp)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	blocks, err := multiply(architecture.Layers, block)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	normCount, err := multiply(2, architecture.Layers)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	normCount, err = add(normCount, 1)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	norms, err := multiply(normCount, architecture.HiddenSize)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	parameters, err := add(embedding, blocks, norms)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	if !architecture.TieEmbeddings {
		parameters, err = add(parameters, embedding)
		if err != nil {
			return ArchitectureForecast{}, err
		}
	}
	bytesPerParameter := uint64(4)
	if architecture.ParameterDType != "float32" {
		bytesPerParameter = 2
	}
	parameterBytes, err := multiply(parameters, bytesPerParameter)
	if err != nil {
		return ArchitectureForecast{}, err
	}
	return ArchitectureForecast{ApproximateParameters: parameters, ParameterBytes: parameterBytes, Formula: "embedding + decoder projections + gated MLP + norms; biases excluded"}, nil
}

func multiply(left, right uint64) (uint64, error) {
	high, low := bits.Mul64(left, right)
	if high != 0 {
		return 0, fmt.Errorf("architecture resource estimate overflows uint64")
	}
	return low, nil
}

func add(values ...uint64) (uint64, error) {
	var total uint64
	for _, value := range values {
		var carry uint64
		total, carry = bits.Add64(total, value, 0)
		if carry != 0 {
			return 0, fmt.Errorf("architecture resource estimate overflows uint64")
		}
	}
	return total, nil
}

func canonicalHash(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return sha256Hex(data), nil
}
