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
		"tokenization_openwaldo.py": []byte(huggingFaceTokenizerSource),
		"architecture.py":           []byte(architectureSource),
		"README.md":                 []byte(readme),
	}
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

func validateStandardLlamaArchitecture(architecture model.Architecture, format string) error {
	if architecture.QKNormalization {
		return fmt.Errorf("%s export cannot preserve WALDO qk_normalization; use the native waldo export until this runtime has an exact architecture implementation", format)
	}
	return nil
}

func huggingFaceTokenizerConfiguration(record model.ModelRecord) (map[string]any, string, error) {
	interactionTemplate, err := jinjaInteractionTemplate(record.Interaction)
	if err != nil {
		return nil, "", err
	}
	configuration := map[string]any{
		"auto_map":  map[string]any{"AutoTokenizer": []any{"tokenization_openwaldo.OpenWALDOByteTokenizer", nil}},
		"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
		"model_max_length": record.Architecture.ContextTokens,
		"tokenizer_class":  "OpenWALDOByteTokenizer",
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

const huggingFaceTokenizerSource = `from transformers import PreTrainedTokenizer

class OpenWALDOByteTokenizer(PreTrainedTokenizer):
    vocab_files_names = {}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs):
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return 259

    def get_vocab(self):
        result = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        result.update({f"<0x{value:02X}>": value + 3 for value in range(256)})
        return result

    def _tokenize(self, text):
        return [f"<0x{value:02X}>" for value in text.encode("utf-8")]

    def _convert_token_to_id(self, token):
        if token == "<pad>": return 0
        if token == "<bos>": return 1
        if token == "<eos>": return 2
        if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
            return int(token[3:5], 16) + 3
        return 0

    def _convert_id_to_token(self, index):
        if index == 0: return "<pad>"
        if index == 1: return "<bos>"
        if index == 2: return "<eos>"
        if 3 <= index < 259: return f"<0x{index - 3:02X}>"
        return "<pad>"

    def convert_tokens_to_string(self, tokens):
        data = bytearray()
        text = []
        def flush():
            if data:
                text.append(bytes(data).decode("utf-8", errors="replace"))
                data.clear()
        for token in tokens:
            if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
                data.append(int(token[3:5], 16))
            else:
                flush()
                text.append(token)
        flush()
        return "".join(text)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()
`

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
architecture with OpenWALDO's schema-1 byte tokenizer. Load the tokenizer with
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
schema-1 byte tokenizer. The Safetensors payload is copied without numerical
conversion. ` + "`BOM.json`" + ` inventories every release file and ` + "`EU-BOM.json`" + `
contains the EU GPAI training-content disclosure mapping.
`
