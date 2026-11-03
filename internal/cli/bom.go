// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/disclosure"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/provenance"
)

func isCorpusExportLocation(location string) bool {
	info, err := os.Stat(location)
	if err != nil {
		return false
	}
	if info.IsDir() {
		_, err = os.Stat(filepath.Join(location, "EXPORT.json"))
		return err == nil
	}
	return filepath.Base(location) == "EXPORT.json"
}

func runIndexBOM(context Context, args []string, stdout, progress io.Writer) error {
	if len(args) == 1 && isCorpusExportLocation(args[0]) {
		return runCorpusExportBOM(context, args, stdout)
	}
	targets, err := resolveIndexArgumentsWithWarning(context.Execution, args, progress)
	if err != nil {
		return err
	}
	policy, err := corpus.NewLicensePolicy(nil, nil)
	if err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	document, err := corpus.BuildBOM(context.Execution, targets, policy, cache)
	if err != nil {
		return err
	}
	if _, err := cache.PurgeUsed(); err != nil {
		return fmt.Errorf("purge successful BOM cache: %w", err)
	}
	return writeJSON(stdout, document)
}

func runCorpusExportBOM(context Context, args []string, stdout io.Writer) error {
	document, path, err := provenance.LoadCorpusExport(args[0])
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, document)
	}
	fmt.Fprintln(stdout, "OpenWALDO corpus export")
	fmt.Fprintf(stdout, "  document      %s\n", path)
	fmt.Fprintf(stdout, "  generated     %s\n", document.Generated)
	fmt.Fprintf(stdout, "  format        %s\n", document.Format)
	if document.BOM.Index.Remote != "" {
		fmt.Fprintf(stdout, "  index         %s\n", document.BOM.Index.Remote)
	}
	if document.BOM.Index.Commit != "" {
		commit := document.BOM.Index.Commit
		if len(commit) > 12 {
			commit = commit[:12]
		}
		dirty := ""
		if document.BOM.Index.Dirty {
			dirty = " (working tree was dirty)"
		}
		fmt.Fprintf(stdout, "  revision      %s%s\n", commit, dirty)
	}
	fmt.Fprintf(stdout, "  paths         %s\n", strings.Join(document.BOM.Paths, ", "))
	fmt.Fprintf(stdout, "  contents      %s shards, %s docs, %s tokens, %s source bytes\n",
		humanInteger(document.BOM.Totals.Shards), humanCount(document.BOM.Totals.Docs),
		humanCount(document.BOM.Totals.Tokens), humanBytes(document.BOM.Totals.Bytes))
	var exportedBytes int64
	for _, file := range document.Files {
		exportedBytes += file.Bytes
	}
	fmt.Fprintf(stdout, "  export        %s files, %s\n", humanInteger(int64(len(document.Files))), humanBytes(exportedBytes))
	return nil
}

func runCorpusExportVerify(context Context, args []string, stdout io.Writer) error {
	document, report, err := provenance.VerifyCorpusExport(args[0])
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Verification provenance.ExportVerification `json:"verification"`
			BOM          corpus.BOM                    `json:"bom"`
		}{Verification: report, BOM: document.BOM})
	}
	fmt.Fprintf(stdout, "verified OpenWALDO BOM and %s exported files (%s)\n", humanInteger(report.Files), humanBytes(report.Bytes))
	return nil
}

type modelBOMOptions struct {
	Model           string
	Output          string
	Format          string
	Provider        string
	AllowIncomplete bool
	Force           bool
}

func runModelEUGPAIBOM(context Context, options modelBOMOptions, stdout, stderr io.Writer) error {
	if options.Output != "" && !options.Force {
		if _, err := os.Stat(options.Output); err == nil {
			return fmt.Errorf("%s already exists; use --force to replace it", options.Output)
		} else if !os.IsNotExist(err) {
			return err
		}
	}
	configuration, err := config.Load()
	if err != nil {
		return err
	}
	root, err := config.EffectiveModelRoot(configuration)
	if err != nil {
		return err
	}
	var provider *disclosure.ProviderProfile
	providerPath := options.Provider
	if providerPath == "" {
		providerPath = configuration.Disclosure.Provider
	}
	if providerPath != "" {
		profile, err := disclosure.LoadProvider(providerPath)
		if err != nil {
			return err
		}
		provider = &profile
	}
	var report disclosure.EUGPAIReport
	if external, err := model.IsExternalModel(root, options.Model); err != nil {
		return err
	} else if external {
		i, err := model.InspectExternal(root, options.Model)
		if err != nil {
			return err
		}
		report, err = disclosure.BuildExternalEUGPAIReport(i, provider, time.Now())
		if err != nil {
			return err
		}
	} else {
		i, err := model.Inspect(root, options.Model)
		if err != nil {
			return err
		}
		report, err = disclosure.BuildEUGPAIReport(i, provider, disclosure.ReleaseFromModel(i), time.Now())
		if err != nil {
			return err
		}
	}

	blocking := report.BlockingGaps()
	if err := requireCompleteDisclosure(report, options.AllowIncomplete, stderr); err != nil {
		return err
	}
	if options.Output == "" {
		return writeJSON(stdout, report)
	}
	if err := disclosure.WriteEUGPAIReport(options.Output, report); err != nil {
		return err
	}
	absolute, _ := filepath.Abs(options.Output)
	if context.JSON {
		return writeJSON(stdout, struct {
			Output string                  `json:"output"`
			Report disclosure.EUGPAIReport `json:"report"`
		}{Output: absolute, Report: report})
	}
	fmt.Fprintf(stdout, "wrote %s EU GPAI disclosure mapping to %s\n", report.Status, absolute)
	fmt.Fprintf(stdout, "  template      %s (%s)\n", report.Template.Document, report.Template.Version)
	fmt.Fprintf(stdout, "  model         %s (%s)\n", report.Model.Name, shortModelHash(report.Model.ID))
	fmt.Fprintf(stdout, "  corpora       %s unique across %s stages\n", humanInteger(int64(len(report.Training.UniqueCorpora))), humanInteger(int64(len(report.Training.Stages))))
	fmt.Fprintf(stdout, "  gaps          %s\n", humanInteger(int64(len(report.Gaps))))
	if blocking > 0 {
		fmt.Fprintln(stdout, "  warning       incomplete draft; this is not a compliance finding")
	}
	return nil
}

func requireCompleteDisclosure(report disclosure.EUGPAIReport, allowIncomplete bool, stderr io.Writer) error {
	blocking := report.BlockingGaps()
	if blocking > 0 && !allowIncomplete {
		fmt.Fprintf(stderr, "EU GPAI export blocked by %s required disclosure gaps:\n", humanInteger(int64(blocking)))
		var required []disclosure.Gap
		for _, gap := range report.Gaps {
			if gap.Severity == "required" {
				required = append(required, gap)
			}
		}
		limit := len(required)
		if limit > 12 {
			limit = 12
		}
		for _, gap := range required[:limit] {
			context := ""
			if gap.Context != "" {
				context = " (" + gap.Context + ")"
			}
			fmt.Fprintf(stderr, "  %s %-34s %s%s\n", gap.Section, gap.Field, gap.Message, context)
		}
		if len(required) > limit {
			fmt.Fprintf(stderr, "  ... and %s more\n", humanInteger(int64(len(required)-limit)))
		}
		return fmt.Errorf("no output written; supply the missing facts or use --allow-incomplete for a marked draft")
	}
	return nil
}

func cobraModelBOMOptions(context Context, args []string) (modelBOMOptions, error) {
	options := modelBOMOptions{
		Format: stringOption(context, "format"), Provider: stringOption(context, "provider"),
		AllowIncomplete: boolOption(context, "allow-incomplete"), Force: boolOption(context, "force"),
	}
	if options.Format != "openwaldo" && options.Format != "eu-gpai" {
		return modelBOMOptions{}, fmt.Errorf("unsupported model BOM format %q; use openwaldo or eu-gpai", options.Format)
	}
	if len(args) == 1 && options.Force {
		return modelBOMOptions{}, fmt.Errorf("--force requires an output file")
	}
	if options.Format == "openwaldo" && (options.Provider != "" || options.AllowIncomplete) {
		return modelBOMOptions{}, fmt.Errorf("--provider and --allow-incomplete apply only to --format eu-gpai")
	}
	if options.Format == "eu-gpai" && len(args) == 2 && strings.ToLower(filepath.Ext(args[1])) != ".json" {
		return modelBOMOptions{}, fmt.Errorf("eu-gpai output must use a .json filename; editable document rendering is not implemented yet")
	}
	options.Model = args[0]
	if len(args) == 2 {
		options.Output = args[1]
	}
	return options, nil
}
