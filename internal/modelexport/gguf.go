// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package modelexport

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/openwaldo/waldo/internal/inference"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/modelquant"
	"github.com/openwaldo/waldo/internal/modelweights"
)

const ggufAlignment = 32

type safeTensorDescriptor struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

type ggufTensor struct {
	SourceName  string
	Name        string
	SourceType  string
	DType       uint32
	Shape       []uint64
	Start       uint64
	SourceBytes uint64
	Bytes       uint64
	Offset      uint64
	Permute     uint64
	ConvertF32  bool
}

type ggufMetadata struct {
	key    string
	typeID uint32
	value  any
}

const (
	ggufUint32  = 4
	ggufFloat32 = 6
	ggufBool    = 7
	ggufString  = 8
	ggufArray   = 9
)

const ollamaUserAssistantToolTemplate = `TEMPLATE """{{- if .System }}System: {{ .System }}{{ end }}{{- if .Tools }}{{ if .System }}

{{ end }}System: Available tools:
{{ json .Tools }}{{ end }}{{- range .Messages }}

{{- if eq .Role "assistant" }}Assistant:{{ if .Content }}{{ .Content }}{{ end }}{{- if .ToolCalls }}[{{ range $index, $_ := .ToolCalls }}{{ if $index }},{{ end }}{"name":{{ json .Function.Name }},"arguments":{{ json .Function.Arguments }}}{{ end }}]{{ end }}
{{- else if eq .Role "tool" }}Tool: {{ .Content }}
{{- else if eq .Role "user" }}User: {{ .Content }}
{{- else }}System: {{ .Content }}
{{- end }}{{- end }}

Assistant:"""
`

const ollamaUserAssistantTemplate = `TEMPLATE """{{- if .System }}System: {{ .System }}{{ end }}{{- range .Messages }}

{{- if eq .Role "assistant" }}Assistant:{{ .Content }}
{{- else if eq .Role "tool" }}Tool: {{ .Content }}
{{- else if eq .Role "user" }}User: {{ .Content }}
{{- else }}System: {{ .Content }}
{{- end }}{{- end }}

Assistant:"""
`

const ollamaChatMLToolTemplate = `TEMPLATE """{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}{{ .System }}{{ end }}{{- if .Tools }}{{ if .System }}

{{ end }}Available tools:
{{ json .Tools }}{{ end }}<|im_end|>
{{- end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{- if eq .Role "assistant" }}{{ if .Content }}{{ .Content }}{{ end }}{{- if .ToolCalls }}[{{ range $index, $_ := .ToolCalls }}{{ if $index }},{{ end }}{"name":{{ json .Function.Name }},"arguments":{{ json .Function.Arguments }}}{{ end }}]{{ end }}
{{- else if eq .Role "tool" }}{{ .Content }}
{{- else }}{{ .Content }}
{{- end }}<|im_end|>
{{- end }}<|im_start|>assistant
"""
`

const ollamaChatMLTemplate = `TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}<|im_start|>assistant
"""
`

func ExportGGUF(ctx context.Context, inspection model.Inspection, destination string, options Options) (string, error) {
	return exportGGUFPackage(ctx, inspection, destination, options, false)
}

func ExportOllama(ctx context.Context, inspection model.Inspection, destination string, options Options) (string, error) {
	return exportGGUFPackage(ctx, inspection, destination, options, true)
}

func exportGGUFPackage(ctx context.Context, inspection model.Inspection, destination string, options Options, ollama bool) (string, error) {
	record := inspection.Model
	record.Interaction = inspection.EffectiveInteraction()
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
	weightsPath := filepath.Join(temporary, "model.gguf")
	if options.Quantization == nil {
		if err := writeGGUF(ctx, artifacts.Weights, weightsPath, record); err != nil {
			return "", err
		}
	} else {
		if options.Quantization.Quantizer == nil {
			return "", fmt.Errorf("quantization runtime is required")
		}
		highPrecision := filepath.Join(temporary, ".waldo-high-precision.gguf")
		if options.Report != nil {
			options.Report("writing high-precision GGUF")
		}
		if err := writeGGUF(ctx, artifacts.Weights, highPrecision, record); err != nil {
			return "", err
		}
		calibrationText := ""
		if options.Quantization.Calibration != nil {
			calibrationText = options.Quantization.Calibration.TextPath
		}
		result, err := options.Quantization.Quantizer.Quantize(ctx, modelquant.Request{Input: highPrecision, Output: weightsPath, Resolved: options.Quantization.Resolved, CalibrationText: calibrationText, Report: options.Report})
		if err != nil {
			return "", err
		}
		if err := os.Remove(highPrecision); err != nil {
			return "", err
		}
		options.result = &result
	}
	if len(options.EUBOM) == 0 {
		return "", fmt.Errorf("EU-BOM.json is empty")
	}
	if err := os.WriteFile(filepath.Join(temporary, "EU-BOM.json"), options.EUBOM, 0o644); err != nil {
		return "", err
	}
	if len(options.Attribution) == 0 {
		return "", fmt.Errorf("ATTRIBUTION.md is empty")
	}
	if err := os.WriteFile(filepath.Join(temporary, "ATTRIBUTION.md"), options.Attribution, 0o644); err != nil {
		return "", err
	}
	format := "gguf"
	roles := map[string]string{"model.gguf": "weights", "EU-BOM.json": "regulatory-disclosure", "ATTRIBUTION.md": "training-data-attribution"}
	if ollama {
		format = "ollama"
		modelfile, err := ollamaModelfile(inspection)
		if err != nil {
			return "", err
		}
		if err := os.WriteFile(filepath.Join(temporary, "Modelfile"), []byte(modelfile), 0o644); err != nil {
			return "", err
		}
		roles["Modelfile"] = "ollama-configuration"
	}
	sourceBOM, err := hashJSON(inspection.BOM)
	if err != nil {
		return "", err
	}
	inventory, err := inventoryFiles(temporary, roles)
	if err != nil {
		return "", err
	}
	bom := releaseBOM{Kind: "openwaldo-bom", Schema: 1, Subject: "model-release", Format: format, ModelID: inspection.Model.ID, Name: inspection.Model.Name, Interaction: record.Interaction, SourceType: artifacts.SourceType, SourceID: artifacts.SourceID, RunID: artifacts.RunID, SourceBOM: sourceBOM, Artifacts: inventory, Generated: inspection.Model.Updated}
	if options.Quantization != nil {
		if options.result == nil {
			return "", fmt.Errorf("quantization result is missing")
		}
		bom.Quantization = &releaseQuantization{Requested: options.Quantization.Requested, Resolved: options.Quantization.Resolved, Profile: modelquant.Profile, Quantizer: options.result.Quantizer, Calibrator: options.result.Calibrator, Calibration: options.Quantization.Calibration}
		if bom.Quantization.Calibration != nil {
			copy := *bom.Quantization.Calibration
			copy.TextPath = ""
			if len(copy.Evidence) == 0 || !json.Valid(copy.Evidence) {
				return "", fmt.Errorf("calibration evidence is empty or invalid")
			}
			bom.Quantization.Calibration = &copy
		}
	}
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

func ollamaModelfile(inspection model.Inspection) (string, error) {
	result := fmt.Sprintf("FROM ./model.gguf\nPARAMETER num_ctx %d\n", inspection.Model.Architecture.ContextTokens)
	interaction := inspection.EffectiveInteraction()
	if err := interaction.Validate(); err != nil {
		return "", err
	}
	if !interaction.Conversational() {
		return result, nil
	}
	switch interaction.Template {
	case model.InteractionUserAssistantV1:
		if interaction.Tools {
			return result + ollamaUserAssistantToolTemplate, nil
		}
		return result + ollamaUserAssistantTemplate, nil
	case model.InteractionChatMLV1:
		if interaction.Tools {
			return result + ollamaChatMLToolTemplate, nil
		}
		return result + ollamaChatMLTemplate, nil
	default:
		return "", fmt.Errorf("unsupported Ollama interaction template %q", interaction.Template)
	}
}

func writeGGUF(ctx context.Context, source, destination string, record model.ModelRecord) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	tensors, dataStart, err := readSafeTensors(input, record.Architecture)
	if err != nil {
		return err
	}
	metadata, err := modelGGUFMetadata(record)
	if err != nil {
		return err
	}
	var header bytes.Buffer
	header.WriteString("GGUF")
	writeBinary(&header, uint32(3))
	writeBinary(&header, uint64(len(tensors)))
	writeBinary(&header, uint64(len(metadata)))
	for _, item := range metadata {
		writeGGUFString(&header, item.key)
		writeBinary(&header, item.typeID)
		if err := writeGGUFValue(&header, item.typeID, item.value); err != nil {
			return err
		}
	}
	for index := range tensors {
		tensor := &tensors[index]
		writeGGUFString(&header, tensor.Name)
		writeBinary(&header, uint32(len(tensor.Shape)))
		for dimension := len(tensor.Shape) - 1; dimension >= 0; dimension-- {
			writeBinary(&header, tensor.Shape[dimension])
		}
		writeBinary(&header, tensor.DType)
		writeBinary(&header, tensor.Offset)
	}
	output, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	keep := false
	defer func() {
		_ = output.Close()
		if !keep {
			_ = os.Remove(destination)
		}
	}()
	if _, err := output.Write(header.Bytes()); err != nil {
		return err
	}
	if err := writePadding(output, uint64(header.Len()), ggufAlignment); err != nil {
		return err
	}
	position := uint64(0)
	for _, tensor := range tensors {
		if err := ctx.Err(); err != nil {
			return err
		}
		if tensor.Offset < position {
			return fmt.Errorf("invalid GGUF tensor offset for %s", tensor.Name)
		}
		if err := writeZeroes(output, tensor.Offset-position); err != nil {
			return err
		}
		if tensor.ConvertF32 {
			if err := writeTensorAsF32(output, input, dataStart+tensor.Start, tensor); err != nil {
				return err
			}
		} else if tensor.Permute != 0 {
			if err := writePermutedTensor(output, input, dataStart+tensor.Start, tensor); err != nil {
				return err
			}
		} else if _, err := io.CopyN(output, io.NewSectionReader(input, int64(dataStart+tensor.Start), int64(tensor.SourceBytes)), int64(tensor.SourceBytes)); err != nil {
			return err
		}
		position = tensor.Offset + tensor.Bytes
	}
	if err := output.Sync(); err != nil {
		return err
	}
	if err := output.Close(); err != nil {
		return err
	}
	keep = true
	return nil
}

func readSafeTensors(input *os.File, architecture model.Architecture) ([]ggufTensor, uint64, error) {
	var length uint64
	if err := binary.Read(input, binary.LittleEndian, &length); err != nil {
		return nil, 0, err
	}
	if length == 0 || length > 1<<30 {
		return nil, 0, fmt.Errorf("invalid Safetensors header length %d", length)
	}
	header := make([]byte, length)
	if _, err := io.ReadFull(input, header); err != nil {
		return nil, 0, err
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(header, &entries); err != nil {
		return nil, 0, err
	}
	dataStart := uint64(8) + length
	stat, err := input.Stat()
	if err != nil {
		return nil, 0, err
	}
	if stat.Size() < 0 || uint64(stat.Size()) < dataStart {
		return nil, 0, fmt.Errorf("Safetensors file ends before its tensor data")
	}
	dataBytes := uint64(stat.Size()) - dataStart
	result := make([]ggufTensor, 0, len(entries)-1)
	targets := make(map[string]string)
	for name, raw := range entries {
		if name == "__metadata__" {
			continue
		}
		var descriptor safeTensorDescriptor
		if err := json.Unmarshal(raw, &descriptor); err != nil {
			return nil, 0, fmt.Errorf("tensor %s: %w", name, err)
		}
		if len(descriptor.Shape) == 0 || len(descriptor.DataOffsets) != 2 || descriptor.DataOffsets[1] <= descriptor.DataOffsets[0] {
			return nil, 0, fmt.Errorf("tensor %s has invalid shape or offsets", name)
		}
		typeID, elementBytes, err := ggmlType(descriptor.DType)
		if err != nil {
			return nil, 0, fmt.Errorf("tensor %s: %w", name, err)
		}
		elements := uint64(1)
		for _, dimension := range descriptor.Shape {
			if dimension == 0 || elements > math.MaxUint64/dimension {
				return nil, 0, fmt.Errorf("tensor %s has invalid dimensions", name)
			}
			elements *= dimension
		}
		if elements > math.MaxUint64/elementBytes {
			return nil, 0, fmt.Errorf("tensor %s byte count overflows", name)
		}
		bytes := descriptor.DataOffsets[1] - descriptor.DataOffsets[0]
		if elements*elementBytes != bytes {
			return nil, 0, fmt.Errorf("tensor %s byte count does not match shape", name)
		}
		if descriptor.DataOffsets[1] > dataBytes {
			return nil, 0, fmt.Errorf("tensor %s extends beyond the Safetensors file", name)
		}
		target, heads, err := ggufTensorName(name, architecture)
		if err != nil {
			return nil, 0, err
		}
		if prior, exists := targets[target]; exists {
			return nil, 0, fmt.Errorf("WALDO tensors %s and %s both map to GGUF tensor %s", prior, name, target)
		}
		targets[target] = name
		convertF32 := len(descriptor.Shape) == 1 && descriptor.DType != "F32"
		outputBytes := bytes
		if convertF32 {
			typeID = 0
			outputBytes = elements * 4
		}
		result = append(result, ggufTensor{SourceName: name, Name: target, SourceType: descriptor.DType, DType: typeID, Shape: descriptor.Shape, Start: descriptor.DataOffsets[0], SourceBytes: bytes, Bytes: outputBytes, Permute: heads, ConvertF32: convertF32})
	}
	if err := validateGGUFTensors(targets, architecture); err != nil {
		return nil, 0, err
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	position := uint64(0)
	for index := range result {
		position = align(position, ggufAlignment)
		result[index].Offset = position
		position += result[index].Bytes
	}
	return result, dataStart, nil
}

func validateGGUFTensors(tensors map[string]string, architecture model.Architecture) error {
	required := []string{"token_embd.weight", "output_norm.weight"}
	for layer := uint64(0); layer < architecture.Layers; layer++ {
		prefix := fmt.Sprintf("blk.%d.", layer)
		for _, tail := range []string{"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ffn_norm.weight"} {
			required = append(required, prefix+tail)
		}
	}
	for _, name := range required {
		if _, ok := tensors[name]; !ok {
			return fmt.Errorf("WALDO weights are missing tensor required for GGUF export: %s", name)
		}
	}
	expected := len(required)
	if !architecture.TieEmbeddings {
		expected++
		if _, ok := tensors["output.weight"]; !ok {
			return fmt.Errorf("untied WALDO weights are missing output.weight required for GGUF export")
		}
	} else if _, ok := tensors["output.weight"]; ok {
		return fmt.Errorf("tied WALDO weights unexpectedly contain output.weight")
	}
	if len(tensors) != expected {
		return fmt.Errorf("WALDO weights contain %d GGUF tensors; architecture requires %d", len(tensors), expected)
	}
	return nil
}

func ggmlType(dtype string) (uint32, uint64, error) {
	switch dtype {
	case "F32":
		return 0, 4, nil
	case "F16":
		return 1, 2, nil
	case "BF16":
		return 30, 2, nil
	default:
		return 0, 0, fmt.Errorf("unsupported GGUF source dtype %q", dtype)
	}
}

func ggufTensorName(name string, architecture model.Architecture) (string, uint64, error) {
	switch name {
	case "embedding.weight":
		return "token_embd.weight", 0, nil
	case "norm.weight":
		return "output_norm.weight", 0, nil
	case "output.weight":
		return "output.weight", 0, nil
	}
	layer, tensor, found := modelweights.WALDOLayer(name)
	if !found {
		return "", 0, fmt.Errorf("unsupported WALDO tensor %q", name)
	}
	mapping := map[string]string{
		"attention.q_proj.weight": "attn_q.weight", "attention.k_proj.weight": "attn_k.weight",
		"attention.v_proj.weight": "attn_v.weight", "attention.o_proj.weight": "attn_output.weight",
		"attention_norm.weight": "attn_norm.weight", "feed_forward.gate.weight": "ffn_gate.weight",
		"feed_forward.up.weight": "ffn_up.weight", "feed_forward.down.weight": "ffn_down.weight",
		"ffn_norm.weight": "ffn_norm.weight",
	}
	tail, ok := mapping[tensor]
	if !ok {
		return "", 0, fmt.Errorf("unsupported WALDO tensor %q", name)
	}
	heads := uint64(0)
	if tensor == "attention.q_proj.weight" {
		heads = architecture.AttentionHeads
	} else if tensor == "attention.k_proj.weight" {
		heads = architecture.KeyValueHeads
	}
	return "blk." + layer + "." + tail, heads, nil
}

func writePermutedTensor(output io.Writer, input io.ReaderAt, start uint64, tensor ggufTensor) error {
	if len(tensor.Shape) != 2 || tensor.Shape[0]%tensor.Permute != 0 || (tensor.Shape[0]/tensor.Permute)%2 != 0 {
		return fmt.Errorf("tensor %s cannot be permuted for %d heads", tensor.SourceName, tensor.Permute)
	}
	rows := tensor.Shape[0]
	rowBytes := tensor.Bytes / rows
	half := rows / tensor.Permute / 2
	buffer := make([]byte, rowBytes)
	for row := uint64(0); row < rows; row++ {
		within := row % (2 * half)
		sourceRow := row - within + (within%2)*half + within/2
		if _, err := input.ReadAt(buffer, int64(start+sourceRow*rowBytes)); err != nil {
			return err
		}
		if _, err := output.Write(buffer); err != nil {
			return err
		}
	}
	return nil
}

func writeTensorAsF32(output io.Writer, input io.ReaderAt, start uint64, tensor ggufTensor) error {
	elements := tensor.Bytes / 4
	inputBytes := tensor.SourceBytes / elements
	buffer := make([]byte, inputBytes)
	var encoded [4]byte
	for index := uint64(0); index < elements; index++ {
		if _, err := input.ReadAt(buffer, int64(start+index*inputBytes)); err != nil {
			return err
		}
		var bits uint32
		switch tensor.SourceType {
		case "BF16":
			bits = uint32(binary.LittleEndian.Uint16(buffer)) << 16
		case "F16":
			bits = math.Float32bits(float16ToFloat32(binary.LittleEndian.Uint16(buffer)))
		default:
			return fmt.Errorf("tensor %s cannot convert source type %s to F32", tensor.SourceName, tensor.SourceType)
		}
		binary.LittleEndian.PutUint32(encoded[:], bits)
		if _, err := output.Write(encoded[:]); err != nil {
			return err
		}
	}
	return nil
}

func float16ToFloat32(value uint16) float32 {
	sign := uint32(value&0x8000) << 16
	exponent := uint32(value>>10) & 0x1f
	fraction := uint32(value & 0x03ff)
	var bits uint32
	switch exponent {
	case 0:
		if fraction == 0 {
			bits = sign
		} else {
			exponent = 113
			for fraction&0x0400 == 0 {
				fraction <<= 1
				exponent--
			}
			bits = sign | exponent<<23 | (fraction&0x03ff)<<13
		}
	case 0x1f:
		bits = sign | 0x7f800000 | fraction<<13
	default:
		bits = sign | (exponent+112)<<23 | fraction<<13
	}
	return math.Float32frombits(bits)
}

func modelGGUFMetadata(record model.ModelRecord) ([]ggufMetadata, error) {
	architecture := record.Architecture
	if err := architecture.Validate(); err != nil {
		return nil, fmt.Errorf("GGUF architecture: %w", err)
	}
	if err := validateStandardLlamaArchitecture(architecture, "GGUF/Ollama"); err != nil {
		return nil, err
	}
	if architecture.Tokenizer.Name != "byte" || architecture.Tokenizer.Revision != "builtin-byte-schema-1" || architecture.VocabularySize != 259 {
		return nil, fmt.Errorf("GGUF export currently requires byte@builtin-byte-schema-1 with vocabulary_size 259")
	}
	for name, value := range map[string]uint64{
		"context_tokens": architecture.ContextTokens, "hidden_size": architecture.HiddenSize,
		"layers": architecture.Layers, "intermediate_size": architecture.IntermediateSize,
		"attention_heads": architecture.AttentionHeads, "key_value_heads": architecture.KeyValueHeads,
	} {
		if value > math.MaxUint32 {
			return nil, fmt.Errorf("GGUF %s exceeds the uint32 metadata limit", name)
		}
	}
	fileType := uint32(0)
	if architecture.ParameterDType == "float16" {
		fileType = 1
	} else if architecture.ParameterDType == "bfloat16" {
		fileType = 32
	}
	tokens := byteTokenizerVocabulary()
	scores := make([]float32, len(tokens))
	types := make([]int32, len(tokens))
	for index := range types {
		types[index] = 1
	}
	types[0], types[1], types[2] = 3, 3, 3
	metadata := []ggufMetadata{
		{"general.architecture", ggufString, "llama"}, {"general.name", ggufString, record.Name},
		{"general.alignment", ggufUint32, uint32(ggufAlignment)}, {"general.file_type", ggufUint32, fileType},
		{"llama.context_length", ggufUint32, uint32(architecture.ContextTokens)}, {"llama.embedding_length", ggufUint32, uint32(architecture.HiddenSize)},
		{"llama.block_count", ggufUint32, uint32(architecture.Layers)}, {"llama.feed_forward_length", ggufUint32, uint32(architecture.IntermediateSize)},
		{"llama.attention.head_count", ggufUint32, uint32(architecture.AttentionHeads)}, {"llama.attention.head_count_kv", ggufUint32, uint32(architecture.KeyValueHeads)},
		{"llama.rope.dimension_count", ggufUint32, uint32(architecture.HiddenSize / architecture.AttentionHeads)}, {"llama.rope.freq_base", ggufFloat32, float32(10000)},
		{"llama.attention.layer_norm_rms_epsilon", ggufFloat32, float32(1e-5)},
		{"tokenizer.ggml.model", ggufString, "gpt2"},
		{"tokenizer.ggml.tokens", ggufArray, tokens}, {"tokenizer.ggml.scores", ggufArray, scores}, {"tokenizer.ggml.token_type", ggufArray, types}, {"tokenizer.ggml.merges", ggufArray, []string{}},
		{"tokenizer.ggml.bos_token_id", ggufUint32, uint32(1)}, {"tokenizer.ggml.eos_token_id", ggufUint32, uint32(2)}, {"tokenizer.ggml.padding_token_id", ggufUint32, uint32(0)},
		{"tokenizer.ggml.add_bos_token", ggufBool, false}, {"tokenizer.ggml.add_eos_token", ggufBool, false},
	}
	chatTemplate, err := jinjaInteractionTemplate(record.Interaction)
	if err != nil {
		return nil, fmt.Errorf("GGUF interaction: %w", err)
	}
	if chatTemplate != "" {
		metadata = append(metadata, ggufMetadata{"tokenizer.chat_template", ggufString, chatTemplate})
	}
	return metadata, nil
}

func byteTokenizerVocabulary() []string {
	result := []string{"<pad>", "<bos>", "<eos>"}
	visible := map[int]rune{}
	for value := 33; value <= 126; value++ {
		visible[value] = rune(value)
	}
	for value := 161; value <= 172; value++ {
		visible[value] = rune(value)
	}
	for value := 174; value <= 255; value++ {
		visible[value] = rune(value)
	}
	next := rune(256)
	for value := 0; value < 256; value++ {
		if _, ok := visible[value]; !ok {
			visible[value] = next
			next++
		}
	}
	for value := 0; value < 256; value++ {
		result = append(result, string(visible[value]))
	}
	return result
}

func writeGGUFValue(writer io.Writer, typeID uint32, value any) error {
	switch typeID {
	case ggufUint32:
		writeBinary(writer, value.(uint32))
	case ggufFloat32:
		writeBinary(writer, value.(float32))
	case ggufBool:
		if value.(bool) {
			writeBinary(writer, uint8(1))
		} else {
			writeBinary(writer, uint8(0))
		}
	case ggufString:
		writeGGUFString(writer, value.(string))
	case ggufArray:
		switch values := value.(type) {
		case []string:
			writeBinary(writer, uint32(ggufString))
			writeBinary(writer, uint64(len(values)))
			for _, item := range values {
				writeGGUFString(writer, item)
			}
		case []float32:
			writeBinary(writer, uint32(ggufFloat32))
			writeBinary(writer, uint64(len(values)))
			for _, item := range values {
				writeBinary(writer, item)
			}
		case []int32:
			writeBinary(writer, uint32(5))
			writeBinary(writer, uint64(len(values)))
			for _, item := range values {
				writeBinary(writer, item)
			}
		default:
			return fmt.Errorf("unsupported GGUF array %T", value)
		}
	default:
		return fmt.Errorf("unsupported GGUF metadata type %d", typeID)
	}
	return nil
}

func writeGGUFString(writer io.Writer, value string) {
	writeBinary(writer, uint64(len(value)))
	_, _ = io.WriteString(writer, value)
}

func writeBinary(writer io.Writer, value any) { _ = binary.Write(writer, binary.LittleEndian, value) }

func align(value, alignment uint64) uint64 { return (value + alignment - 1) / alignment * alignment }

func writePadding(writer io.Writer, position, alignment uint64) error {
	return writeZeroes(writer, align(position, alignment)-position)
}

func writeZeroes(writer io.Writer, count uint64) error {
	zeroes := make([]byte, 4096)
	for count > 0 {
		chunk := uint64(len(zeroes))
		if count < chunk {
			chunk = count
		}
		if _, err := writer.Write(zeroes[:chunk]); err != nil {
			return err
		}
		count -= chunk
	}
	return nil
}
