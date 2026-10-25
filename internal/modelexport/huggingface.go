// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package modelexport converts one verified WALDO model run into separately
// executable release formats without changing the managed model.
package modelexport

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/openwaldo/waldo/internal/inference"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/modelquant"
	"github.com/openwaldo/waldo/internal/training"
)

type Options struct {
	EUBOM        []byte
	Attribution  []byte
	Finalize     func(string) error
	Quantization *Quantization
	Report       func(string)
	result       *modelquant.Result
}

type Quantization struct {
	Requested   string
	Resolved    string
	Quantizer   modelquant.Quantizer
	Calibration *Calibration
}

type Calibration struct {
	TextPath        string          `json:"-"`
	Profile         string          `json:"profile"`
	ReferenceTokens int64           `json:"available_reference_tokens"`
	SampledTokens   int64           `json:"sampled_tokens"`
	Records         int64           `json:"records"`
	Shards          int             `json:"shards"`
	SelectionSHA256 string          `json:"selection_sha256"`
	Seed            uint64          `json:"seed"`
	Evidence        json.RawMessage `json:"evidence"`
}

type releaseBOM struct {
	Kind         string               `json:"kind"`
	Schema       int                  `json:"schema"`
	Subject      string               `json:"subject"`
	Format       string               `json:"format"`
	ModelID      string               `json:"model_id"`
	Name         string               `json:"name"`
	Interaction  model.Interaction    `json:"interaction,omitzero"`
	SourceType   string               `json:"source_type"`
	SourceID     string               `json:"source_id"`
	RunID        string               `json:"run_id,omitempty"`
	SourceBOM    string               `json:"source_bom_sha256"`
	Artifacts    []releaseArtifact    `json:"artifacts"`
	Quantization *releaseQuantization `json:"quantization,omitempty"`
	Generated    string               `json:"generated"`
}

type releaseQuantization struct {
	Requested   string           `json:"requested"`
	Resolved    string           `json:"resolved"`
	Profile     string           `json:"profile"`
	Quantizer   modelquant.Tool  `json:"quantizer"`
	Calibrator  *modelquant.Tool `json:"calibrator,omitempty"`
	Calibration *Calibration     `json:"calibration,omitempty"`
}

type releaseArtifact struct {
	Role   string `json:"role"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

func ExportHuggingFace(ctx context.Context, inspection model.Inspection, destination string, options Options) (string, error) {
	if inspection.Model.Architecture.Transformers != nil {
		return exportTransformers(ctx, inspection, destination, options)
	}
	return exportLlamaPackage(ctx, inspection, destination, options, "huggingface")
}

func ExportMLX(ctx context.Context, inspection model.Inspection, destination string, options Options) (string, error) {
	return exportLlamaPackage(ctx, inspection, destination, options, "mlx")
}

func exportLlamaPackage(ctx context.Context, inspection model.Inspection, destination string, options Options, format string) (string, error) {
	_ = ctx
	record := inspection.Model
	record.Interaction = inspection.EffectiveInteraction()
	if err := validateStandardLlamaArchitecture(record.Architecture, format); err != nil {
		return "", err
	}
	if err := validateByteTokenizerExport(record.Architecture, format); err != nil {
		return "", err
	}
	artifacts, err := inference.ResolveArtifacts(inspection)
	if err != nil {
		return "", err
	}
	destinationAbsolute, err := filepath.Abs(destination)
	if err != nil {
		return "", err
	}
	if withinPath(inspection.Path, destinationAbsolute) {
		return "", fmt.Errorf("model export destination must not be inside the source model")
	}
	absolute, temporary, cleanup, err := beginExport(destination)
	if err != nil {
		return "", err
	}
	committed := false
	defer func() { cleanup(committed) }()
	rewrite := rewriteHuggingFaceWeights
	architectureSource := huggingFaceArchitectureSource
	readme := huggingFaceReadme
	if format == "mlx" {
		rewrite = rewriteMLXWeights
		architectureSource = mlxArchitectureSource
		readme = mlxReadme
	}
	if err := rewrite(artifacts.Weights, filepath.Join(temporary, "model.safetensors")); err != nil {
		return "", fmt.Errorf("convert %s weights: %w", format, err)
	}
	configuration := huggingFaceConfig(inspection.Model.Architecture)
	if err := writeJSON(filepath.Join(temporary, "config.json"), configuration); err != nil {
		return "", err
	}
	files := map[string][]byte{
		"ATTRIBUTION.md":            options.Attribution,
		"EU-BOM.json":               options.EUBOM,
		"generation_config.json":    []byte(huggingFaceGenerationConfig),
		"special_tokens_map.json":   []byte(huggingFaceSpecialTokens),
		"tokenization_openwaldo.py": []byte(huggingFaceTokenizerPython()),
		"architecture.py":           []byte(architectureSource),
		"README.md":                 []byte(readme),
	}
	tokenizerArtifact, _, err := exportTokenizerArtifact(artifacts.Tokenizer, record.Architecture.VocabularySize)
	if err != nil {
		return "", err
	}
	files["openwaldo_tokenizer.json"] = tokenizerArtifact
	tokenizerConfiguration, interactionTemplate, err := huggingFaceTokenizerConfiguration(record)
	if err != nil {
		return "", err
	}
	if interactionTemplate != "" {
		files["chat_template.jinja"] = []byte(interactionTemplate + "\n")
	}
	tokenizerConfigurationJSON, err := json.MarshalIndent(tokenizerConfiguration, "", "  ")
	if err != nil {
		return "", err
	}
	files["tokenizer_config.json"] = append(tokenizerConfigurationJSON, '\n')
	for name, data := range files {
		if len(data) == 0 {
			return "", fmt.Errorf("model export file %s is empty", name)
		}
		if err := os.WriteFile(filepath.Join(temporary, name), data, 0o644); err != nil {
			return "", err
		}
	}
	sourceBOM, err := hashJSON(inspection.BOM)
	if err != nil {
		return "", err
	}
	roles := map[string]string{
		"model.safetensors": "weights", "config.json": "configuration",
		"generation_config.json": "generation-configuration",
		"tokenizer_config.json":  "tokenizer", "special_tokens_map.json": "tokenizer",
		"openwaldo_tokenizer.json":  "tokenizer-artifact",
		"tokenization_openwaldo.py": "tokenizer-code", "architecture.py": "architecture-code",
		"README.md": "documentation", "EU-BOM.json": "regulatory-disclosure", "ATTRIBUTION.md": "training-data-attribution",
	}
	if interactionTemplate != "" {
		roles["chat_template.jinja"] = "interaction-template"
	}
	inventory, err := inventoryFiles(temporary, roles)
	if err != nil {
		return "", err
	}
	bom := releaseBOM{Kind: "openwaldo-bom", Schema: 1, Subject: "model-release", Format: format, ModelID: inspection.Model.ID, Name: inspection.Model.Name, Interaction: record.Interaction, SourceType: artifacts.SourceType, SourceID: artifacts.SourceID, RunID: artifacts.RunID, SourceBOM: sourceBOM, Artifacts: inventory, Generated: inspection.Model.Updated}
	if err := writeJSON(filepath.Join(temporary, "BOM.json"), bom); err != nil {
		return "", err
	}
	if options.Finalize != nil {
		if err := options.Finalize(temporary); err != nil {
			return "", err
		}
	}
	if err := os.Rename(temporary, absolute); err != nil {
		return "", err
	}
	committed = true
	return absolute, nil
}

func exportTokenizerArtifact(path string, architectureVocabularySize uint64) ([]byte, training.TokenizerSpec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, training.TokenizerSpec{}, fmt.Errorf("read tokenizer artifact: %w", err)
	}
	var artifact struct {
		Kind   string `json:"kind"`
		Schema int    `json:"schema"`
		training.TokenizerSpec
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, training.TokenizerSpec{}, fmt.Errorf("decode tokenizer artifact: %w", err)
	}
	if artifact.Schema != 1 || (artifact.Kind != "waldo-tokenizer" && artifact.Kind != "waldo-byte-tokenizer") {
		return nil, training.TokenizerSpec{}, fmt.Errorf("unsupported WALDO tokenizer artifact")
	}
	spec, _, err := training.ResolveTokenizerSpec(artifact.TokenizerSpec)
	if err != nil {
		return nil, training.TokenizerSpec{}, fmt.Errorf("resolve tokenizer artifact: %w", err)
	}
	if uint64(spec.VocabularySize) != architectureVocabularySize {
		return nil, training.TokenizerSpec{}, fmt.Errorf("tokenizer vocabulary_size %d does not match architecture %d", spec.VocabularySize, architectureVocabularySize)
	}
	if spec.Name != "byte" && spec.Name != "waldo/byte-bpe" && spec.Name != "waldo/bytepiece" {
		return nil, training.TokenizerSpec{}, fmt.Errorf("%s export does not yet support tokenizer %s@%s", "huggingface/MLX", spec.Name, spec.Revision)
	}
	return append(data, '\n'), spec, nil
}

func validateStandardLlamaArchitecture(architecture model.Architecture, format string) error {
	if architecture.QKNormalization {
		return fmt.Errorf("%s export cannot preserve WALDO qk_normalization; use the native waldo export until this runtime has an exact architecture implementation", format)
	}
	return nil
}

func validateByteTokenizerExport(architecture model.Architecture, format string) error {
	if architecture.Tokenizer.Name != "byte" || architecture.Tokenizer.Revision != "builtin-byte-schema-1" || architecture.VocabularySize != 259 {
		return fmt.Errorf("%s export currently requires byte@builtin-byte-schema-1 with vocabulary_size 259", format)
	}
	return nil
}

func huggingFaceTokenizerConfiguration(record model.ModelRecord) (map[string]any, string, error) {
	interactionTemplate, err := jinjaInteractionTemplate(record.Interaction)
	if err != nil {
		return nil, "", err
	}
	configuration := map[string]any{
		"auto_map":  map[string]any{"AutoTokenizer": []any{"tokenization_openwaldo." + model.HuggingFaceTokenizerClass, nil}},
		"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
		"model_max_length": record.Architecture.ContextTokens,
		"tokenizer_class":  model.HuggingFaceTokenizerClass,
	}
	if interactionTemplate != "" {
		configuration["chat_template"] = interactionTemplate
	}
	return configuration, interactionTemplate, nil
}

func withinPath(parent, child string) bool {
	parent, parentErr := filepath.Abs(parent)
	child, childErr := filepath.Abs(child)
	if parentErr != nil || childErr != nil {
		return false
	}
	relative, err := filepath.Rel(parent, child)
	return err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator))
}

func beginExport(destination string) (string, string, func(bool), error) {
	absolute, err := filepath.Abs(destination)
	if err != nil {
		return "", "", nil, err
	}
	if _, err := os.Lstat(absolute); err == nil {
		return "", "", nil, fmt.Errorf("model export destination already exists: %s", absolute)
	} else if !os.IsNotExist(err) {
		return "", "", nil, err
	}
	parent := filepath.Dir(absolute)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return "", "", nil, err
	}
	temporary, err := os.MkdirTemp(parent, ".waldo-model-export-*")
	if err != nil {
		return "", "", nil, err
	}
	return absolute, temporary, func(committed bool) {
		if !committed {
			_ = os.RemoveAll(temporary)
		}
	}, nil
}

func huggingFaceConfig(architecture model.Architecture) map[string]any {
	dtype := map[string]string{"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}[architecture.ParameterDType]
	return map[string]any{
		"architectures": []string{"LlamaForCausalLM"}, "model_type": "llama",
		"hidden_size": architecture.HiddenSize, "intermediate_size": architecture.IntermediateSize,
		"num_hidden_layers": architecture.Layers, "num_attention_heads": architecture.AttentionHeads,
		"num_key_value_heads": architecture.KeyValueHeads, "vocab_size": architecture.VocabularySize,
		"max_position_embeddings": architecture.ContextTokens, "rms_norm_eps": 1e-5,
		"rope_theta": 10000.0, "tie_word_embeddings": architecture.TieEmbeddings,
		"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
		"torch_dtype": dtype, "use_cache": true,
	}
}

func inventoryFiles(directory string, roles map[string]string) ([]releaseArtifact, error) {
	names := make([]string, 0, len(roles))
	for name := range roles {
		names = append(names, name)
	}
	sort.Strings(names)
	result := make([]releaseArtifact, 0, len(names))
	for _, name := range names {
		file, err := os.Open(filepath.Join(directory, name))
		if err != nil {
			return nil, err
		}
		hasher := sha256.New()
		bytes, copyErr := io.Copy(hasher, file)
		closeErr := file.Close()
		if copyErr != nil {
			return nil, copyErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		result = append(result, releaseArtifact{Role: roles[name], Path: name, SHA256: hex.EncodeToString(hasher.Sum(nil)), Bytes: bytes})
	}
	return result, nil
}

func hashJSON(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func writeJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

const huggingFaceSpecialTokens = `{"bos_token":"<bos>","eos_token":"<eos>","pad_token":"<pad>"}
`

const huggingFaceGenerationConfig = `{"_from_model_config":true,"bos_token_id":1,"eos_token_id":2,"pad_token_id":0}
`

const huggingFaceTokenizerSource = `import base64
import json
import os
import shutil

from transformers import PreTrainedTokenizer


class OpenWALDOTokenizer(PreTrainedTokenizer):
	vocab_files_names = {"vocab_file": "openwaldo_tokenizer.json"}
	model_input_names = ["input_ids", "attention_mask"]

	def __init__(self, vocab_file=None, **kwargs):
		if vocab_file is None:
			vocab_file = kwargs.pop("vocab_file", None)
		if not vocab_file:
			raise ValueError("OpenWALDO tokenizer artifact is required")
		with open(vocab_file, "r", encoding="utf-8") as handle:
			payload = json.load(handle)
		self.vocab_file = vocab_file
		self._name = payload["name"]
		self._vocab_size = int(payload["vocabulary_size"])
		self._ranks = {}
		self._legacy_pieces = []
		self._token_bytes = [b""] * self._vocab_size
		for value in range(256):
			self._token_bytes[value + 3] = bytes([value])
		artifact = payload.get("artifact")
		if artifact:
			if artifact.get("schema") == 2 and artifact.get("name") == "waldo/byte-bpe":
				for rank, merge in enumerate(artifact.get("merges", [])):
					left, right = int(merge["left"]), int(merge["right"])
					self._ranks[(left, right)] = rank
					self._token_bytes[259 + rank] = self._token_bytes[left] + self._token_bytes[right]
			elif artifact.get("schema") == 1 and artifact.get("name") == "waldo/bytepiece":
				self._legacy_pieces = [base64.b64decode(value) for value in artifact.get("pieces", [])]
				for index, value in enumerate(self._legacy_pieces):
					self._token_bytes[259 + index] = value
			else:
				raise ValueError("unsupported OpenWALDO trained tokenizer artifact")
		elif self._name != "byte" or self._vocab_size != 259:
			raise ValueError("unsupported OpenWALDO tokenizer artifact")
		kwargs.setdefault("pad_token", "<pad>")
		kwargs.setdefault("bos_token", "<bos>")
		kwargs.setdefault("eos_token", "<eos>")
		super().__init__(**kwargs)

	@property
	def vocab_size(self):
		return self._vocab_size

	def get_vocab(self):
		return {self._token_name(index): index for index in range(self._vocab_size)}

	@staticmethod
	def _kind(character):
		if character == "_" or character.isalpha() or character.isnumeric():
			return 0
		if character.isspace():
			return 1
		return 2

	@classmethod
	def _chunks(cls, text):
		runs = []
		start = 0
		current = None
		for index, character in enumerate(text):
			kind = cls._kind(character)
			if current is not None and kind != current:
				runs.append((text[start:index].encode("utf-8"), current == 1))
				start = index
			current = kind
		if start < len(text):
			runs.append((text[start:].encode("utf-8"), current == 1))
		whitespace = b""
		result = []
		for value, space in runs:
			if space:
				whitespace += value
				continue
			value = whitespace + value
			whitespace = b""
			result.extend(value[offset:offset + 64] for offset in range(0, len(value), 64))
		if whitespace:
			result.extend(whitespace[offset:offset + 64] for offset in range(0, len(whitespace), 64))
		return result

	@staticmethod
	def _merge_pair(symbols, selected, replacement):
		result = []
		index = 0
		while index < len(symbols):
			if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == selected:
				result.append(replacement)
				index += 2
			else:
				result.append(symbols[index])
				index += 1
		return result

	def _encode_text(self, text):
		if self._legacy_pieces:
			pieces = sorted(enumerate(self._legacy_pieces), key=lambda item: -len(item[1]))
			data = text.encode("utf-8")
			result = []
			while data:
				match = next(((index, piece) for index, piece in pieces if data.startswith(piece)), None)
				if match:
					result.append(259 + match[0])
					data = data[len(match[1]):]
				else:
					result.append(data[0] + 3)
					data = data[1:]
			return result
		result = []
		for chunk in self._chunks(text):
			symbols = [value + 3 for value in chunk]
			while True:
				candidates = [(self._ranks[pair], pair) for pair in zip(symbols, symbols[1:]) if pair in self._ranks]
				if not candidates:
					break
				rank, selected = min(candidates, key=lambda item: item[0])
				symbols = self._merge_pair(symbols, selected, 259 + rank)
			result.extend(symbols)
		return result

	@staticmethod
	def _token_name(index):
		if index == 0: return "<pad>"
		if index == 1: return "<bos>"
		if index == 2: return "<eos>"
		return f"<|waldo:{index}|>"

	def _tokenize(self, text):
		return [self._token_name(index) for index in self._encode_text(text)]

	def _convert_token_to_id(self, token):
		if token == "<pad>": return 0
		if token == "<bos>": return 1
		if token == "<eos>": return 2
		if token.startswith("<|waldo:") and token.endswith("|>"):
			try:
				index = int(token[8:-2])
				return index if 0 <= index < self._vocab_size else 0
			except ValueError:
				pass
		return 0

	def _convert_id_to_token(self, index):
		return self._token_name(index) if 0 <= index < self._vocab_size else "<pad>"

	def convert_tokens_to_string(self, tokens):
		data = bytearray()
		text = []
        def flush():
            if data:
				text.append(bytes(data).decode("utf-8", errors="replace"))
				data.clear()
		for token in tokens:
			index = self._convert_token_to_id(token)
			if index >= 3:
				data.extend(self._token_bytes[index])
			else:
				flush()
				text.append(token)
        flush()
        return "".join(text)

	def save_vocabulary(self, save_directory, filename_prefix=None):
		name = ((filename_prefix + "-") if filename_prefix else "") + "openwaldo_tokenizer.json"
		destination = os.path.join(save_directory, name)
		if os.path.abspath(destination) != os.path.abspath(self.vocab_file):
			shutil.copyfile(self.vocab_file, destination)
		return (destination,)


# Compatibility for previously exported class references.
OpenWALDOByteTokenizer = OpenWALDOTokenizer
`

func huggingFaceTokenizerPython() string {
	return strings.ReplaceAll(huggingFaceTokenizerSource, "\t", "    ")
}

const huggingFaceArchitectureSource = `"""Executable architecture declaration for this OpenWALDO release."""
from transformers import LlamaConfig, LlamaForCausalLM

class OpenWALDOForCausalLM(LlamaForCausalLM):
    config_class = LlamaConfig
`

const huggingFaceReadme = `---
library_name: transformers
pipeline_tag: text-generation
---

# OpenWALDO model export

This package uses the standard Transformers Llama causal-language-model
architecture with OpenWALDO's exact byte or trained byte-BPE tokenizer. Load the tokenizer with
` + "`trust_remote_code=True`" + `. ` + "`BOM.json`" + ` inventories every release file and
` + "`EU-BOM.json`" + ` contains the EU GPAI training-content disclosure mapping.
`

const mlxArchitectureSource = `"""Executable architecture binding for this OpenWALDO MLX release."""
from mlx_lm.models.llama import Model, ModelArgs

OpenWALDOForCausalLM = Model
OpenWALDOConfig = ModelArgs
`

const mlxReadme = `---
library_name: mlx
pipeline_tag: text-generation
---

# OpenWALDO MLX model export

This package uses MLX-LM's standard Llama implementation and OpenWALDO's
exact byte or trained byte-BPE tokenizer. The Safetensors payload is copied without numerical
conversion. ` + "`BOM.json`" + ` inventories every release file and ` + "`EU-BOM.json`" + `
contains the EU GPAI training-content disclosure mapping.
`
