// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/disclosure"
	"github.com/openwaldo/waldo/internal/modelexport"
	"github.com/openwaldo/waldo/internal/signing"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

func runExternalComposeTraining(command Context, name string, c model.ExternalCompose, stdout, stderr io.Writer) error {
	builder, err := configuredModelBuilder(command, stderr)
	if err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	builder.PreparedCacheMaxBytes = cache.MaxBytes()
	projection := model.Compose{}
	for _, s := range c.Stages {
		projection.Stages = append(projection.Stages, s.DataStage())
	}
	targets, err := sanityCheckComposeCorpora(command.Execution, projection, stderr)
	if err != nil {
		return err
	}
	// These are local execution settings; neither enters portable plan identity.
	applicationCache := os.Getenv("WALDO_APPLICATION_CACHE")
	if applicationCache == "" {
		applicationCache = filepath.Join(builder.Root, ".applications")
	}
	result, err := builder.ComposeExternal(command.Execution, name, c, model.ExternalOptions{Cache: applicationCache, Python: os.Getenv("WALDO_APPLICATION_PYTHON"), Prepare: func(ctx context.Context, s model.ExternalStage) (corpus.BOM, []training.Input, error) {
		stage := s.DataStage()
		planned, err := planModelStage(command, stage, targets[s.Name], cache, stderr)
		if err != nil {
			return corpus.BOM{}, nil, err
		}
		materialized, err := corpus.Materialize(ctx, planned.BOM, cache, modelMaterializeProgressPrinter(stderr))
		if err != nil {
			return corpus.BOM{}, nil, err
		}
		return planned.BOM, verifiedTrainingInputs(materialized, planned.BOM), nil
	}})
	if err != nil {
		return err
	}
	if command.JSON {
		return writeJSON(stdout, result)
	}
	fmt.Fprintf(stdout, "trained external model %s (%s): %d stages complete\n", name, result.Model.ID, len(result.Runs))
	return nil
}

func runExternalModelExport(command Context, root string, parsed modelExportOptions, stdout, stderr io.Writer) error {
	if parsed.Format != "waldo" && parsed.Format != "huggingface" {
		return fmt.Errorf("external hf-causal-lm export supports waldo or huggingface format")
	}
	if parsed.Quant != "" || parsed.Calibration != "" {
		return fmt.Errorf("external artifact conversion and quantization are not implemented")
	}
	inspection, err := model.InspectExternal(root, parsed.Name)
	if err != nil {
		return err
	}
	configuration, err := config.Load()
	if err != nil {
		return err
	}
	if configuration.Disclosure.Provider == "" {
		return fmt.Errorf("model export requires provider information; run waldo config set disclosure.provider <provider.json>")
	}
	provider, err := disclosure.LoadProvider(configuration.Disclosure.Provider)
	if err != nil {
		return err
	}
	report, err := disclosure.BuildExternalEUGPAIReport(inspection, &provider, time.Now())
	if err != nil {
		return err
	}
	if err = requireCompleteDisclosure(report, parsed.AllowIncomplete, stderr); err != nil {
		return err
	}
	euBOM, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	var boms []corpus.BOM
	for _, bom := range inspection.RunBOMs {
		boms = append(boms, bom.CorpusBOM)
	}
	attribution, err := corpus.AttributionNotice(boms)
	if err != nil {
		return err
	}
	options := modelexport.Options{EUBOM: append(euBOM, '\n'), Attribution: attribution}
	signed := signing.Configured(configuration.Signing)
	if signed {
		options.Finalize = func(directory string) error {
			return signing.SignExport(command.Execution, configuration.Signing, directory, stderr)
		}
	}
	output, err := modelexport.ExportExternal(command.Execution, inspection, parsed.Destination, options)
	if err != nil {
		return err
	}
	if !signed {
		fmt.Fprintln(stderr, "warning: model export is unsigned; configure signing.* to sign exports automatically")
	}
	if command.JSON {
		return writeJSON(stdout, struct {
			Name   string `json:"name"`
			Format string `json:"format"`
			Output string `json:"output"`
			Signed bool   `json:"signed"`
		}{parsed.Name, "hf-causal-lm", output, signed})
	}
	fmt.Fprintf(stdout, "exported hf-causal-lm model %s to %s\n", parsed.Name, output)
	return nil
}
