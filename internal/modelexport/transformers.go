// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package modelexport

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

// Preserve provider-native weights/configuration; never run native tensor-name
// rewrites over a Transformers model. Finalize uses the normal release signer.
func exportTransformers(ctx context.Context, inspection model.Inspection, destination string, options Options) (string, error) {
	if options.Quantization != nil {
		return "", fmt.Errorf("Transformers release quantization is not supported yet")
	}
	tokenizerPin := inspection.Model.Architecture.Tokenizer.HuggingFace
	if inspection.Model.Architecture.Tokenizer.Name != "byte" && tokenizerPin == nil {
		return "", fmt.Errorf("Transformers releases currently require the WALDO byte tokenizer")
	}
	if withinPath(inspection.Path, destination) {
		return "", fmt.Errorf("model export destination must not be inside the source model")
	}
	var selected *model.ModelBOMRun
	for i := range inspection.BOM.Runs {
		if inspection.BOM.Runs[i].ID == inspection.BOM.CurrentRunID {
			selected = &inspection.BOM.Runs[i]
			break
		}
	}
	if selected == nil || selected.State != model.RunComplete || selected.Simulated || selected.Backend.Name != training.BackendTransformers {
		return "", fmt.Errorf("Transformers export requires a complete real Transformers run")
	}
	var runBOM *model.RunBOM
	for i := range inspection.RunBOMs {
		if inspection.RunBOMs[i].ID == selected.ID {
			runBOM = &inspection.RunBOMs[i]
			break
		}
	}
	if runBOM == nil || runBOM.Execution.Package == nil {
		return "", fmt.Errorf("Transformers export lacks package provenance")
	}
	absolute, temporary, cleanup, err := beginExport(destination)
	if err != nil {
		return "", err
	}
	committed := false
	defer func() { cleanup(committed) }()
	roles := map[string]string{}
	required := map[string]bool{"model.safetensors": true, "config.json": true, "tokenizer.json": true, "training_args.json": true, "runtime.json": true}
	if tokenizerPin != nil {
		required["waldo_tokenizer.json"] = true
		for name := range tokenizerPin.Files {
			required[name] = true
		}
	}
	for _, artifact := range selected.Artifacts {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		name := filepath.Base(artifact.Path)
		if !required[name] {
			continue
		}
		if tokenizerPin != nil && tokenizerPin.Files[name] != "" && tokenizerPin.Files[name] != artifact.SHA256 {
			return "", fmt.Errorf("tokenizer release artifact differs from compose pin: %s", name)
		}
		if _, exists := roles[name]; exists {
			return "", fmt.Errorf("duplicate Transformers release artifact %s", name)
		}
		source := filepath.Join(inspection.Path, filepath.FromSlash(artifact.Path))
		if !withinPath(inspection.Path, source) {
			return "", fmt.Errorf("invalid Transformers artifact path")
		}
		pin := training.Artifact{Path: artifact.Path, SHA256: artifact.SHA256, Bytes: artifact.Bytes}
		if err := model.VerifyArtifactFile(source, pin); err != nil {
			return "", err
		}
		if err := copyTransformersArtifact(source, filepath.Join(temporary, name)); err != nil {
			return "", err
		}
		if err := model.VerifyArtifactFile(filepath.Join(temporary, name), pin); err != nil {
			return "", err
		}
		roles[name] = artifact.Role
	}
	if len(roles) != len(required) {
		return "", fmt.Errorf("Transformers release requires weights, config, tokenizer, training arguments, and runtime evidence")
	}
	files := map[string][]byte{"EU-BOM.json": options.EUBOM}
	if tokenizerPin == nil {
		configuration, template, err := huggingFaceTokenizerConfiguration(inspection.Model)
		if err != nil {
			return "", err
		}
		if err := writeJSON(filepath.Join(temporary, "tokenizer_config.json"), configuration); err != nil {
			return "", err
		}
		roles["tokenizer_config.json"] = "tokenizer"
		files["tokenization_openwaldo.py"] = []byte(huggingFaceTokenizerSource)
		files["README.md"] = []byte("# WALDO Transformers release\n\nProvider-native random-initialized training output. Load the model with Transformers AutoModelForCausalLM.from_pretrained. The WALDO byte tokenizer uses the included reviewed tokenization_openwaldo.py; AutoTokenizer requires explicit trust_remote_code=True. This exports local code, not a remote model implementation. training-run.json contains the exact Transformers wheel pin and requested Trainer arguments; runtime.json inventories the tested runtime. Other dependencies are not wheel-locked.\n")
		if template != "" {
			files["chat_template.jinja"] = []byte(template + "\n")
		}
	} else {
		files["README.md"] = []byte("# WALDO Transformers release\n\nProvider-native weights and hash-pinned fast tokenizer assets. Load locally with AutoModelForCausalLM and AutoTokenizer, local_files_only=True, trust_remote_code=False. Training used WALDO record framing (no automatic BOS, one appended EOS) and WALDO conversation formatting, not the tokenizer's chat template. Any upstream chat template is preserved as source evidence only. training-run.json and waldo_tokenizer.json record provenance; runtime.json inventories dependencies, which are not all wheel-locked.\n")
	}
	for name, data := range files {
		if len(data) == 0 {
			return "", fmt.Errorf("model export file %s is empty", name)
		}
		if err := os.WriteFile(filepath.Join(temporary, name), data, 0644); err != nil {
			return "", err
		}
		roles[name] = "supporting-evidence"
	}
	if err := writeJSON(filepath.Join(temporary, "training-run.json"), runBOM); err != nil {
		return "", err
	}
	roles["training-run.json"] = "training-provenance"
	inventory, err := inventoryFiles(temporary, roles)
	if err != nil {
		return "", err
	}
	sourceHash, err := hashJSON(inspection.BOM)
	if err != nil {
		return "", err
	}
	bom := releaseBOM{Kind: "openwaldo-bom", Schema: 1, Subject: "model-release", Format: "huggingface", ModelID: inspection.Model.ID, Name: inspection.Model.Name, Interaction: inspection.Model.Interaction, SourceType: "run", SourceID: selected.ID, RunID: selected.ID, SourceBOM: sourceHash, Artifacts: inventory, Generated: inspection.Model.Updated}
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

func copyTransformersArtifact(source, destination string) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(output, input)
	if copyErr == nil {
		copyErr = output.Sync()
	}
	closeErr := output.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}
