// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path"
	"sort"
	"strings"

	"github.com/openwaldo/waldo/internal/config"
	waldoindex "github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/ingest"
	"github.com/openwaldo/waldo/internal/lookaside"
)

var newIngestPublisher = func(ctx context.Context, publish config.Publish) (lookaside.Publisher, error) {
	return lookaside.NewPublisher(ctx, publish)
}

func runIndexIngest(context Context, args []string, stdout, stderr io.Writer) error {
	if boolOption(context, "update") {
		return runIndexIngestUpdate(context, args, stdout, stderr)
	}
	options, err := cobraIndexIngestOptions(context, args)
	if err != nil {
		return err
	}
	loadedCorpus, isCorpusDirectory, err := ingest.LoadCorpusDirectory(options.Inputs[0])
	if err != nil {
		return err
	}
	loadedSource, isSourceDirectory, err := ingest.LoadSourceDirectory(options.Inputs[0])
	if err != nil {
		return err
	}
	requestedDestination := options.Request.Destination
	if isCorpusDirectory {
		if requestedDestination == "" {
			return fmt.Errorf("corpus directory ingestion requires an explicit destination")
		}
		if len(options.MetadataOptions) > 0 {
			return fmt.Errorf("corpus directory manifest owns corpus metadata; remove %s", strings.Join(options.MetadataOptions, ", "))
		}
		loadedCorpus.Apply(&options.Request)
		options.Inputs = loadedCorpus.InputPaths()
	} else if isSourceDirectory {
		if requestedDestination == "" {
			return fmt.Errorf("source directory ingestion requires an explicit destination")
		}
		if len(options.MetadataOptions) > 0 {
			return fmt.Errorf("source directory manifest owns corpus metadata; remove %s", strings.Join(options.MetadataOptions, ", "))
		}
		loadedSource.Apply(&options.Request)
		options.Inputs = loadedSource.InputPaths()
	} else if options.Request.Destination == "" {
		return fmt.Errorf("direct index ingest requires a destination")
	} else if options.Request.Title == "" || options.Request.License == "" || options.Request.Source.URL == "" || options.Request.Source.Category == "" || options.Request.Source.Content == nil || len(options.Request.Source.Content.Languages) == 0 {
		return fmt.Errorf("direct index ingest requires --title, --license, --source, --source-category, and --language (repeat for each human language; use und if unknown)")
	}
	if options.InputProfile != "" {
		options.Request.Profile, err = ingest.LoadInputProfile(options.InputProfile)
		if err != nil {
			return fmt.Errorf("load input profile: %w", err)
		}
	}
	configuration, err := config.Load()
	if err != nil {
		return err
	}
	workers := options.Workers
	if workers == 0 && configuration.Lookaside.Publish != nil {
		workers = configuration.Lookaside.Publish.Workers
	}
	configuredRoot, managedDefault, err := config.EffectiveIndexRoot(configuration)
	if err != nil {
		return err
	}
	if managedDefault && !explicitIndexPath(options.Request.Destination) {
		return managedIndexMutationError("ingest into")
	}
	target, err := waldoindex.ResolveDestinationConfigured(configuredRoot, options.Request.Destination)
	if err != nil {
		return err
	}
	managed, err := config.IsManagedIndexPath(target.Root)
	if err != nil {
		return err
	}
	if managed {
		return managedIndexMutationError("ingest into")
	}
	options.Request.Destination = target.Rel
	if options.Request.Source.Name == "" {
		options.Request.Source.Name = path.Base(strings.TrimSuffix(target.Rel, "/"))
	}
	if !options.DryRun {
		if configuration.Lookaside.Publish == nil {
			return fmt.Errorf("index ingest needs a writable lookaside; run `waldo config set lookaside <s3-or-file-URL>`")
		}
	}
	execution := ingest.WithProgress(context.Execution, ingestProgressReporter(stderr, context.JSON))
	probe, err := ingest.ProbePathsWithWorkers(execution, options.Inputs, workers)
	if err != nil {
		return err
	}
	if isCorpusDirectory {
		if err := loadedCorpus.VerifyProbe(probe); err != nil {
			return err
		}
	} else if isSourceDirectory {
		if err := loadedSource.VerifyProbe(probe); err != nil {
			return err
		}
	}
	plan, err := ingest.NewPlan(probe, options.Request)
	if err != nil {
		return err
	}
	emitIngestForceFormatWarning(stderr, plan, context.JSON)
	emitIngestFallbackWarning(stderr, plan, context.JSON)
	identity, err := plan.Identity()
	if err != nil {
		return err
	}
	if options.DryRun && context.JSON {
		return writeJSON(stdout, struct {
			Identity string      `json:"identity"`
			Plan     ingest.Plan `json:"plan"`
		}{Identity: identity, Plan: plan})
	}
	if !options.DryRun {
		if err := ingest.CheckContributionDestination(target.Root, plan); err != nil {
			return err
		}
		staging, err := config.EffectiveStagingRoot(configuration, identity)
		if err != nil {
			return err
		}
		scratchRoot, err := config.EffectiveScratchRoot(configuration)
		if err != nil {
			return err
		}
		if err := ingest.ValidateWorkLocations(target.Root, staging, scratchRoot); err != nil {
			return err
		}
		publish := configuration.Lookaside.Publish
		publisher, err := newIngestPublisher(execution, *publish)
		if err != nil {
			return err
		}
		assembly, publication, err := ingest.ExecutePublication(execution, plan, staging, publisher, workers)
		if err != nil {
			return err
		}
		manifest, err := ingest.BuildManifest(plan, assembly, publication.BaseURL)
		if err != nil {
			return err
		}
		contribution, err := ingest.StageContribution(target.Root, staging, plan, manifest)
		if err != nil {
			return err
		}
		contribution, err = ingest.ApplyContribution(target.Root, contribution)
		if err != nil {
			return fmt.Errorf("apply verified contribution %s: %w", contribution.Root, err)
		}
		emitIngestExclusionWarning(stderr, assembly, plan)
		if context.JSON {
			return writeJSON(stdout, struct {
				Identity     string                    `json:"identity"`
				Plan         ingest.Plan               `json:"plan"`
				Assembly     ingest.AssemblyResult     `json:"assembly"`
				Publication  ingest.PublicationResult  `json:"publication"`
				Contribution ingest.ContributionResult `json:"contribution"`
			}{identity, plan, assembly, publication, contribution})
		}
		fmt.Fprintf(stdout, "ingestion %s complete\n", identity[:12])
		fmt.Fprintf(stdout, "  records      %s input, %s retained, %s duplicate", humanInteger(assembly.InputDocs), humanInteger(assembly.RetainedDocs), humanInteger(assembly.DuplicateDocs))
		if assembly.RejectedDocs > 0 {
			fmt.Fprintf(stdout, ", %s rejected %s", humanInteger(assembly.RejectedDocs), rejectionLabel(plan))
		}
		fmt.Fprintln(stdout)
		var tokens int64
		for _, object := range assembly.Objects {
			tokens += object.Tokens
		}
		fmt.Fprintf(stdout, "  tokens       %s (%s)\n", humanCount(tokens), manifest.ConvertedBy.Tokenizer)
		fmt.Fprintf(stdout, "  objects      %s published to %s\n", humanInteger(int64(len(publication.Objects))), publication.BaseURL)
		fmt.Fprintf(stdout, "  index        applied %s writes, %s removals to %s\n", humanInteger(int64(len(contribution.Files))), humanInteger(int64(len(contribution.Removed))), contribution.IndexRoot)
		fmt.Fprintf(stdout, "  contribution %s (retained)\n", contribution.Root)
		for _, file := range contribution.Files {
			fmt.Fprintf(stdout, "    %s\n", file)
		}
		for _, file := range contribution.Removed {
			fmt.Fprintf(stdout, "    remove %s\n", file)
		}
		if strings.HasPrefix(publication.BaseURL, "file://") {
			fmt.Fprintln(stdout, "local publication is for end-to-end testing only; do not commit this overlay to a shared index")
			return nil
		}
		fmt.Fprintln(stdout, "next steps (after reviewing the applied index changes):")
		fmt.Fprintf(stdout, "  waldo index verify %s\n", shellQuote(target.Root))
		fmt.Fprintf(stdout, "  git -C %s add --", shellQuote(target.Root))
		for _, file := range contribution.Files {
			fmt.Fprintf(stdout, " %s", shellQuote(file))
		}
		for _, file := range contribution.Removed {
			fmt.Fprintf(stdout, " %s", shellQuote(file))
		}
		fmt.Fprintln(stdout)
		fmt.Fprintf(stdout, "  git -C %s diff --cached --check\n", shellQuote(target.Root))
		fmt.Fprintf(stdout, "  git -C %s commit -s\n", shellQuote(target.Root))
		return nil
	}
	fmt.Fprintf(stdout, "ingestion plan %s\n", identity[:12])
	fmt.Fprintf(stdout, "  destination  %s\n", plan.Destination)
	fmt.Fprintf(stdout, "  title        %s\n", plan.Title)
	fmt.Fprintf(stdout, "  description  %s\n", plan.Description)
	if len(plan.Sources) == 0 {
		fmt.Fprintf(stdout, "  license      %s\n", plan.License)
		fmt.Fprintf(stdout, "  source       %s (%s)\n", plan.Source.Name, plan.Source.Category)
	} else {
		fmt.Fprintf(stdout, "  sources      %d\n", len(plan.Sources))
		for _, source := range plan.Sources {
			fmt.Fprintf(stdout, "    %-16s %s  %s\n", source.ID, source.License, source.URL)
		}
	}
	fmt.Fprintf(stdout, "  mode         %s\n", plan.Mode)
	fmt.Fprintf(stdout, "  memory       %s\n", humanBytes(plan.MemoryBytes))
	fmt.Fprintf(stdout, "  input        %s files, %s\n", humanInteger(int64(len(plan.Inputs))), humanBytes(probe.Totals.Bytes))
	for _, input := range plan.Inputs {
		mapping := input.Adapter
		if input.TextColumn != "" {
			mapping += ":" + input.TextColumn
		}
		fmt.Fprintf(stdout, "    %-18s %s (%s)\n", mapping, input.Artifact.Path, humanBytes(input.Artifact.Bytes))
	}
	fmt.Fprintf(stdout, "  writer       Parquet schema %d, %s target, %s row groups, %s\n",
		plan.Writer.RecordSchema, humanBytes(plan.Writer.CompressedTarget),
		humanBytes(plan.Writer.RowGroupLogicalBytes), plan.Writer.Compression)
	fmt.Fprintln(stdout, "dry run complete; no files were written")
	return nil
}

func rejectionLabel(plan ingest.Plan) string {
	for _, input := range plan.Inputs {
		if input.Profile.Type == ingest.ProfileXMLRecord && input.Profile.XML.OnMalformed == "skip" {
			return "malformed XML"
		}
	}
	return "empty"
}

func emitIngestExclusionWarning(output io.Writer, assembly ingest.AssemblyResult, plan ingest.Plan) {
	if assembly.RejectedDocs == 0 {
		return
	}
	detail := strings.ToUpper(rejectionLabel(plan))
	if len(assembly.Rejections) > 0 {
		reasons := make([]string, 0, len(assembly.Rejections))
		for reason := range assembly.Rejections {
			reasons = append(reasons, reason)
		}
		sort.Strings(reasons)
		parts := make([]string, 0, len(reasons))
		for _, reason := range reasons {
			parts = append(parts, fmt.Sprintf("%s %s", humanInteger(assembly.Rejections[reason]), strings.ToUpper(reason)))
		}
		detail = strings.Join(parts, ", ")
	}
	fmt.Fprintf(output, "\nWARNING: WALDO EXCLUDED %s RECORDS DURING INGESTION (%s).\n", humanInteger(assembly.RejectedDocs), detail)
	fmt.Fprintln(output, "WARNING: EXCLUDED RECORDS ARE NOT PRESENT IN THE PUBLISHED SHARDS; REVIEW THE SOURCE POLICY AND COUNTS BEFORE COMMITTING.")
	fmt.Fprintln(output)
}

func emitIngestFallbackWarning(output io.Writer, plan ingest.Plan, jsonOutput bool) {
	for _, fallback := range plan.TextFallbacks {
		representation := "RAW TEXT"
		if fallback.Adapter == "opaque-base64" {
			representation = "LOSSLESS BASE64 TEXT"
		}
		message := fmt.Sprintf("WALDO INGESTING %s %s ARTIFACTS (%s) AS %s; CONTENT IS RETAINED", humanInteger(fallback.Artifacts), strings.ToUpper(fallback.DetectedFormat), humanBytes(fallback.Bytes), representation)
		if jsonOutput {
			_ = json.NewEncoder(output).Encode(ingest.ProgressEvent{Phase: "plan", Status: "warning", Message: message})
			continue
		}
		fmt.Fprintf(output, "WARNING: %s.\n", message)
	}
}

func emitIngestForceFormatWarning(output io.Writer, plan ingest.Plan, jsonOutput bool) {
	counts := map[string]int64{}
	for _, input := range plan.Inputs {
		if input.DetectedFormat != "" {
			counts[input.DetectedFormat+"->"+input.Artifact.Format]++
		}
	}
	if len(counts) == 0 {
		return
	}
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		message := fmt.Sprintf("WALDO FORCE-FORMAT OVERRIDE %s FOR %s INPUT FILES; AUTOMATIC FORMAT SELECTION WAS OVERRIDDEN, BUT THE SELECTED ADAPTER WILL STILL PARSE AND VALIDATE CONTENT", strings.ToUpper(key), humanInteger(counts[key]))
		if jsonOutput {
			_ = json.NewEncoder(output).Encode(ingest.ProgressEvent{Phase: "plan", Status: "warning", Message: message})
			continue
		}
		fmt.Fprintf(output, "WARNING: %s.\n", message)
	}
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

func ingestProgressReporter(output io.Writer, jsonOutput bool) ingest.ProgressSink {
	last := map[string]int64{}
	var ingestBytes, ingestTotalBytes, ingestFiles, ingestTotalFiles, ingestDocs, ingestTokens, ingestShards int64
	return func(event ingest.ProgressEvent) {
		if jsonOutput {
			_ = json.NewEncoder(output).Encode(event)
			return
		}
		short := event.Shard
		if len(short) > 12 {
			short = short[:12]
		}
		switch {
		case event.Phase == "fetch" && event.Status == "started":
			fmt.Fprintf(output, "fetch %d  %s started\n", event.Sequence, event.Input)
		case event.Phase == "fetch" && event.Status == "completed":
			fmt.Fprintf(output, "fetch %d  %s completed\n", event.Sequence, event.Input)
		case event.Phase == "input" && event.Status == "probing":
			fmt.Fprintf(output, "probing  %s\n", event.Input)
		case event.Phase == "input" && event.Status == "detected":
			fmt.Fprintf(output, "detected %s as %s (%s)\n", event.Input, event.Adapter, humanBytes(event.Bytes))
		case event.Phase == "convert" && event.Status == "started":
			fmt.Fprintf(output, "convert  %s using %s\n", event.Input, event.Adapter)
		case event.Phase == "convert" && event.Status == "progress" && event.Message != "":
			fmt.Fprintf(output, "convert  %s\n", event.Message)
		case event.Phase == "convert" && event.Status == "completed":
			fmt.Fprintf(output, "converted %s (%s)\n", event.Input, humanBytes(event.Bytes))
		case event.Phase == "ingest":
			if event.TotalBytes > 0 {
				ingestTotalBytes = event.TotalBytes
			}
			if event.TotalFiles > 0 {
				ingestTotalFiles = event.TotalFiles
			}
			if event.Bytes > 0 {
				ingestBytes = event.Bytes
			}
			if event.Files > 0 {
				ingestFiles = event.Files
			}
			if event.Docs > 0 || event.Status == "completed" {
				ingestDocs = event.Docs
			}
			if event.Tokens > 0 || event.Status == "completed" {
				ingestTokens = event.Tokens
			}
			label := "ingest  "
			if event.Status == "started" {
				label = "ingest started "
			} else if event.Status == "completed" {
				label = "ingest complete"
			}
			fmt.Fprintf(output, "%s  %s/%s files  %s/%s  %s docs  %s tokens  %s output shards\n",
				label, humanInteger(ingestFiles), humanInteger(ingestTotalFiles),
				humanBytes(ingestBytes), humanBytes(ingestTotalBytes),
				humanInteger(ingestDocs), humanInteger(ingestTokens), humanInteger(ingestShards))
		case event.Phase == "audit" && event.Status == "started":
			fmt.Fprintf(output, "audit %d  %s started on worker %d\n", event.Sequence, short, event.Worker)
		case event.Phase == "audit" && event.Status == "completed":
			fmt.Fprintf(output, "audit %d  %s completed\n", event.Sequence, short)
		case event.Phase == "shard" && event.Status == "creating":
			fmt.Fprintf(output, "creating OpenWALDO Parquet file %d\n", event.Sequence)
		case event.Phase == "shard" && event.Status == "ready":
			ingestShards = max(ingestShards, int64(event.Sequence))
			fmt.Fprintf(output, "created  OpenWALDO Parquet file %d  %s  %s  %s docs  %s tokens\n",
				event.Sequence, short, humanBytes(event.Bytes), humanInteger(event.Docs), humanInteger(event.Tokens))
		case event.Phase == "upload" && event.Status == "started":
			fmt.Fprintf(output, "upload %d  %s started on worker %d\n", event.Sequence, short, event.Worker)
		case event.Phase == "upload" && event.Status == "progress" && (event.Bytes == event.TotalBytes || event.Bytes-last[event.Shard] >= 64<<20):
			last[event.Shard] = event.Bytes
			fmt.Fprintf(output, "upload %d  %s %s/%s\n", event.Sequence, short, humanBytes(event.Bytes), humanBytes(event.TotalBytes))
		case event.Phase == "upload" && event.Status == "verified":
			fmt.Fprintf(output, "upload %d  %s verified at %s\n", event.Sequence, short, event.Remote)
		case event.Phase == "staging" && event.Status == "purged":
			fmt.Fprintf(output, "purged %d  %s reclaimed %s\n", event.Sequence, short, humanBytes(event.ReclaimedBytes))
		}
	}
}

type indexIngestOptions struct {
	Request         ingest.PlanRequest
	Inputs          []string
	DryRun          bool
	InputProfile    string
	Workers         int
	MetadataOptions []string
}

func cobraIndexIngestOptions(context Context, args []string) (indexIngestOptions, error) {
	destination := ""
	if len(args) > 1 {
		destination = args[1]
	}
	languages := repeatedCommaOptions(context, "language")
	programmingLanguages := repeatedCommaOptions(context, "programming-language")
	var content *waldoindex.Content
	if len(languages) > 0 || len(programmingLanguages) > 0 {
		content = &waldoindex.Content{Languages: languages, ProgrammingLanguages: programmingLanguages}
	}
	options := indexIngestOptions{
		Inputs:       []string{args[0]},
		DryRun:       boolOption(context, "dry-run"),
		InputProfile: stringOption(context, "input-profile"),
		Workers:      intOption(context, "workers"),
		Request: ingest.PlanRequest{
			Title:       stringOption(context, "title"),
			Description: stringOption(context, "description"),
			License:     stringOption(context, "license"),
			TextColumn:  stringOption(context, "text-column"),
			ForceFormat: stringOption(context, "force-format"),
			Destination: destination,
			Source: ingest.PlanSource{
				URL:      stringOption(context, "source"),
				Name:     stringOption(context, "source-name"),
				Category: stringOption(context, "source-category"),
				Content:  content,
			},
		},
	}
	if options.Workers < 0 || options.Workers > 32 {
		return indexIngestOptions{}, fmt.Errorf("--workers must be an integer from 1 to 32, or 0 to use lookaside.workers")
	}
	for _, name := range []string{"title", "description", "license", "source", "source-name", "source-category", "language", "programming-language", "text-column", "input-profile"} {
		if optionChanged(context, name) {
			options.MetadataOptions = append(options.MetadataOptions, "--"+name)
		}
	}
	return options, nil
}

func repeatedCommaOptions(context Context, name string) []string {
	seen := map[string]bool{}
	var values []string
	for _, raw := range stringArrayOption(context, name) {
		for _, value := range splitComma(raw) {
			if !seen[value] {
				seen[value] = true
				values = append(values, value)
			}
		}
	}
	sort.Strings(values)
	return values
}
