// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/corpus"
	managedgit "github.com/openwaldo/waldo/internal/git"
	waldoindex "github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/shard"
)

func runIndexInit(context Context, args []string, stdout, _ io.Writer) error {
	managed, err := config.IsManagedIndexPath(args[0])
	if err != nil {
		return err
	}
	if managed {
		return managedIndexMutationError("initialize")
	}
	root, err := waldoindex.Initialize(args[0])
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Path   string `json:"path"`
			Schema int    `json:"schema"`
		}{Path: root, Schema: waldoindex.DirectorySchema})
	}
	fmt.Fprintf(stdout, "initialized WALDO index %s\n", root)
	fmt.Fprintf(stdout, "  index schema  %d\n", waldoindex.DirectorySchema)
	fmt.Fprintln(stdout, "  corpora       0")
	fmt.Fprintln(stdout, "Git initialization and the first signed-off commit remain explicit user actions.")
	return nil
}

func runIndexList(context Context, args []string, stdout, stderr io.Writer) error {
	target, err := resolveIndexArgument(context.Execution, args, stderr)
	if err != nil {
		return err
	}
	corpora, err := waldoindex.ListCorpora(target)
	if err != nil {
		return err
	}
	languages := repeatedCommaOptions(context, "language")
	programmingLanguages := repeatedCommaOptions(context, "programming-language")
	corpora = filterCorporaByLanguage(corpora, languages, programmingLanguages)
	if context.JSON {
		return writeJSON(stdout, struct {
			Path    string                  `json:"path"`
			Corpora []waldoindex.CorpusInfo `json:"corpora"`
		}{Path: target.Rel, Corpora: corpora})
	}
	if len(corpora) == 0 {
		fmt.Fprintf(stdout, "no corpora indexed beneath %s\n", displayPath(target.Rel))
		return nil
	}
	table := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(table, "PATH\tTITLE\tLANGUAGES\tPROGRAMMING\tSHARDS\tDOCS\tTOKENS\tSIZE\tLICENSE")
	for _, corpus := range corpora {
		fmt.Fprintf(table, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			corpus.Path, corpus.Title, declaredList(corpus.Languages), declaredList(corpus.ProgrammingLanguages), humanInteger(corpus.Shards), humanCount(corpus.Docs),
			humanCount(corpus.Tokens), humanBytes(corpus.Bytes), licenseSummary(corpus.Licenses))
	}
	return table.Flush()
}

func filterCorporaByLanguage(corpora []waldoindex.CorpusInfo, languages, programmingLanguages []string) []waldoindex.CorpusInfo {
	if len(languages) == 0 && len(programmingLanguages) == 0 {
		return corpora
	}
	filtered := make([]waldoindex.CorpusInfo, 0, len(corpora))
	for _, corpus := range corpora {
		if matchesAnyFold(corpus.Languages, languages) && matchesAnyFold(corpus.ProgrammingLanguages, programmingLanguages) {
			filtered = append(filtered, corpus)
		}
	}
	return filtered
}

func matchesAnyFold(declared, requested []string) bool {
	if len(requested) == 0 {
		return true
	}
	for _, want := range requested {
		for _, have := range declared {
			if strings.EqualFold(have, want) {
				return true
			}
		}
	}
	return false
}

func declaredList(values []string) string {
	if len(values) == 0 {
		return "(unknown)"
	}
	return truncateDisplay(strings.Join(values, ","), 40)
}

func runIndexShow(context Context, args []string, stdout, stderr io.Writer) error {
	target, err := resolveIndexArgument(context.Execution, args, stderr)
	if err != nil {
		return err
	}
	info, err := os.Stat(target.Abs)
	if err != nil {
		return err
	}
	if info.IsDir() {
		directory, err := waldoindex.LoadDirectory(target.Abs)
		if err != nil {
			return err
		}
		if len(directory.Entries) == 1 && directory.Entries[0].Type == "manifest" {
			manifestPath, err := waldoindex.EntryPath(target.Abs, directory.Entries[0])
			if err != nil {
				return err
			}
			target.Abs = manifestPath
			target.Rel = filepath.ToSlash(filepath.Join(target.Rel, directory.Entries[0].Name))
		} else {
			if context.JSON {
				return writeJSON(stdout, directory)
			}
			fmt.Fprintf(stdout, "index %s\n", displayPath(directory.Path))
			for _, entry := range waldoindex.SortedEntries(directory.Entries) {
				fmt.Fprintf(stdout, "  %-10s %s\n", entry.Type, entry.Name)
			}
			return nil
		}
	}

	manifest, err := waldoindex.LoadManifest(target.Abs)
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, manifest)
	}
	printManifest(stdout, target.Rel, manifest)
	return nil
}

func runIndexSummary(context Context, args []string, stdout, stderr io.Writer) error {
	target, err := resolveIndexArgument(context.Execution, args, stderr)
	if err != nil {
		return err
	}
	totals, err := waldoindex.Summarize(target)
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Path   string            `json:"path"`
			Totals waldoindex.Totals `json:"totals"`
		}{Path: target.Rel, Totals: totals})
	}
	fmt.Fprintf(stdout, "index %s\n", displayPath(target.Rel))
	fmt.Fprintf(stdout, "  corpora  %s\n", humanInteger(totals.Corpora))
	fmt.Fprintf(stdout, "  shards   %s\n", humanInteger(totals.Shards))
	fmt.Fprintf(stdout, "  docs     %s\n", humanCount(totals.Docs))
	fmt.Fprintf(stdout, "  tokens   %s\n", humanCount(totals.Tokens))
	fmt.Fprintf(stdout, "  bytes    %s\n", humanBytes(totals.Bytes))
	if len(totals.Licenses) > 0 {
		fmt.Fprintln(stdout, "\nlicenses")
		licenses := make([]string, 0, len(totals.Licenses))
		for license := range totals.Licenses {
			licenses = append(licenses, license)
		}
		sort.Strings(licenses)
		for _, license := range licenses {
			licenseTotals := totals.Licenses[license]
			shardWord := "shards"
			if licenseTotals.Shards == 1 {
				shardWord = "shard"
			}
			fmt.Fprintf(stdout, "  %-48s %12s tokens  %4s %s\n", truncateDisplay(license, 48), humanCount(licenseTotals.Tokens), humanInteger(licenseTotals.Shards), shardWord)
		}
	}
	return nil
}

func runIndexVerify(context Context, args []string, stdout, stderr io.Writer) error {
	if len(args) == 1 && isCorpusExportLocation(args[0]) {
		if boolOption(context, "objects") || boolOption(context, "offline") {
			return fmt.Errorf("--objects and --offline apply to index checkouts, not corpus exports")
		}
		return runCorpusExportVerify(context, args, stdout)
	}
	return runIndexVerifyWithProgress(context, args, stdout, stderr)
}

func runIndexAudit(context Context, args []string, stdout, progress io.Writer) error {
	auditOptions, err := cobraAuditOptions(context)
	if err != nil {
		return err
	}
	// Audit the exact checkout selected by the caller. Validation roots are
	// immutable review snapshots and must not require network Git credentials
	// or move commits before their shard claims are checked.
	target, err := resolveIndexArgumentPolicy(context.Execution, args, progress, false)
	if err != nil {
		return err
	}
	verification, err := waldoindex.Verify(target)
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
	bom, err := corpus.BuildBOM(context.Execution, []waldoindex.Target{target}, policy, cache)
	if err != nil {
		return err
	}
	fmt.Fprintf(progress, "auditing %s indexed shard references (%s)\n", humanInteger(int64(len(bom.Shards))), humanBytes(bom.Totals.Bytes))
	var audited shard.Summary
	if boolOption(context, "deep") {
		if err := validateAuditCacheCapacity(cache, bom.Totals.Bytes); err != nil {
			return err
		}
		materialized, err := corpus.Materialize(context.Execution, bom, cache, func(event corpus.MaterializeProgress) {
			if event.Phase != "complete" {
				return
			}
			writeIndexProgressRange(progress, event.Current, event.Total, "objects fetched")
		})
		if err != nil {
			return err
		}
		unique := make([]string, 0, len(materialized.Objects))
		seen := map[string]bool{}
		for _, object := range materialized.Objects {
			if !seen[object.Shard.SHA256] {
				seen[object.Shard.SHA256] = true
				unique = append(unique, object.Path)
			}
		}
		auditOptions.Progress = auditProgressPrinter(progress)
		audited, err = shard.AuditWithOptions(context.Execution, unique, auditOptions)
		if err == nil {
			err = corpus.AttachShardAttestations(&bom, materialized.Objects)
		}
	} else {
		audited, err = verifyAuditStream(context.Execution, &bom, cache, auditOptions, progress)
	}
	if err != nil {
		return err
	}
	if audited.Records != bom.Totals.Docs || audited.Tokens != bom.Totals.Tokens || audited.EncodedBytes != bom.Totals.Bytes {
		return fmt.Errorf("audited totals differ from manifests: records %d/%d, tokens %d/%d, bytes %d/%d", audited.Records, bom.Totals.Docs, audited.Tokens, bom.Totals.Tokens, audited.EncodedBytes, bom.Totals.Bytes)
	}
	purged, err := cache.PurgeUsed()
	if err != nil {
		return fmt.Errorf("purge successful audit scratch: %w", err)
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Status        string                  `json:"status"`
			Path          string                  `json:"path"`
			Verification  waldoindex.Verification `json:"verification"`
			Summary       shard.Summary           `json:"summary"`
			BOM           corpus.BOM              `json:"bom"`
			ScratchPurged lookaside.Stats         `json:"scratch_purged"`
		}{"verified", target.Rel, verification, audited, bom, purged})
	}
	fmt.Fprintln(stdout, "STATUS:         VERIFIED")
	printShardSummary(stdout, audited)
	printShardBOMEvidence(stdout, bom, boolOption(context, "show-boms"))
	return nil
}

// verifyAuditStream bounds disk use to the retained cache policy by verifying
// each content-addressed object immediately after Fetch. Unlike a deep audit,
// the attestation path never requires the entire corpus to coexist locally.
func verifyAuditStream(ctx context.Context, bom *corpus.BOM, cache *lookaside.Cache, options shard.AuditOptions, progress io.Writer) (shard.Summary, error) {
	seen := make(map[string]int64)
	unique := make([]corpus.ShardPin, 0, len(bom.Shards))
	for _, pin := range bom.Shards {
		if size, ok := seen[pin.SHA256]; ok {
			if size != pin.Bytes {
				return shard.Summary{}, fmt.Errorf("object %s has conflicting declared sizes %d and %d", pin.SHA256, size, pin.Bytes)
			}
			continue
		}
		seen[pin.SHA256] = pin.Bytes
		unique = append(unique, pin)
	}
	workers := options.Workers
	if workers <= 0 {
		workers = min(runtime.GOMAXPROCS(0), 4)
	}
	workers = min(workers, len(unique))
	if workers == 0 {
		return shard.Summary{}, nil
	}
	type auditOutcome struct {
		pin     corpus.ShardPin
		path    string
		summary shard.Summary
		err     error
	}
	auditContext, cancel := context.WithCancel(ctx)
	defer cancel()
	jobs := make(chan corpus.ShardPin)
	outcomes := make(chan auditOutcome, workers)
	var group sync.WaitGroup
	for range workers {
		group.Add(1)
		go func() {
			defer group.Done()
			for pin := range jobs {
				path, err := cache.Fetch(auditContext, pin.URL, pin.SHA256, pin.Bytes)
				var one shard.Summary
				if err == nil {
					one, err = shard.VerifyWithOptions(auditContext, []string{path}, shard.AuditOptions{Workers: 1})
				}
				outcomes <- auditOutcome{pin: pin, path: path, summary: one, err: err}
				if err != nil {
					cancel()
					return
				}
			}
		}()
	}
	go func() {
		defer close(jobs)
		for _, pin := range unique {
			select {
			case jobs <- pin:
			case <-auditContext.Done():
				return
			}
		}
	}()
	go func() {
		group.Wait()
		close(outcomes)
	}()
	licenses, recipes := map[string]bool{}, map[string]bool{}
	var total shard.Summary
	current := 0
	var auditErr error
	for outcome := range outcomes {
		if outcome.err != nil {
			if auditErr == nil || errors.Is(auditErr, context.Canceled) {
				auditErr = fmt.Errorf("%s shard %s: %w", outcome.pin.Manifest, outcome.pin.SHA256[:12], outcome.err)
			}
			continue
		}
		if err := corpus.AttachShardAttestation(bom, corpus.MaterializedObject{Shard: outcome.pin, Path: outcome.path}); err != nil {
			cancel()
			if auditErr == nil {
				auditErr = err
			}
			continue
		}
		addAuditSummary(&total, outcome.summary, licenses, recipes)
		current++
		if current == 1 || current == len(unique) || current%25 == 0 {
			fmt.Fprintf(progress, "  verified %s/%s  %s\n", humanInteger(int64(current)), humanInteger(int64(len(unique))), outcome.pin.SHA256[:12])
		}
	}
	if auditErr != nil {
		return shard.Summary{}, auditErr
	}
	if err := ctx.Err(); err != nil {
		return shard.Summary{}, err
	}
	for value := range licenses {
		total.Licenses = append(total.Licenses, value)
	}
	for value := range recipes {
		total.Recipes = append(total.Recipes, value)
	}
	sort.Strings(total.Licenses)
	sort.Strings(total.Recipes)
	if err := bom.Validate(); err != nil {
		return shard.Summary{}, err
	}
	return total, nil
}

func addAuditSummary(total *shard.Summary, one shard.Summary, licenses, recipes map[string]bool) {
	total.Shards += one.Shards
	total.Attested += one.Attested
	total.DeepScanned += one.DeepScanned
	total.Records += one.Records
	total.Tokens += one.Tokens
	total.ContentBytes += one.ContentBytes
	total.EncodedBytes += one.EncodedBytes
	total.RowGroups += one.RowGroups
	total.EmailAddressRecords += one.EmailAddressRecords
	total.RepetitiveContentRecords += one.RepetitiveContentRecords
	total.BoilerplateContentRecords += one.BoilerplateContentRecords
	total.Redaction.EmailAddresses += one.Redaction.EmailAddresses
	total.Redaction.IPAddresses += one.Redaction.IPAddresses
	total.Redaction.PhoneNumbers += one.Redaction.PhoneNumbers
	total.Redaction.MailRoutingHeaders += one.Redaction.MailRoutingHeaders
	total.Redaction.Credentials += one.Redaction.Credentials
	if one.Redaction.Policy != "" {
		total.Redaction.Policy = one.Redaction.Policy
		total.Redaction.NamesRetained = one.Redaction.NamesRetained
	}
	for _, value := range one.Licenses {
		licenses[value] = true
	}
	for _, value := range one.Recipes {
		recipes[value] = true
	}
}

func printShardBOMEvidence(output io.Writer, bom corpus.BOM, details bool) {
	embedded, implicit, deep := int64(0), int64(0), int64(0)
	for _, pin := range bom.Shards {
		if pin.Attestation == nil {
			continue
		}
		switch pin.Attestation.Status {
		case "embedded":
			embedded++
		case "implicit-v4":
			implicit++
		case "deep-validated":
			deep++
		}
	}
	fmt.Fprintf(output, "SHARD BOMS:     %s embedded, %s implicit-v4, %s deep-validated\n", humanInteger(embedded), humanInteger(implicit), humanInteger(deep))
	if !details {
		return
	}
	table := tabwriter.NewWriter(output, 0, 4, 2, ' ', 0)
	fmt.Fprintln(table, "SHARD\tSTATUS\tBOM SHA256\tPLAN SHA256")
	seen := map[string]bool{}
	for _, pin := range bom.Shards {
		if seen[pin.SHA256] || pin.Attestation == nil {
			continue
		}
		seen[pin.SHA256] = true
		bomHash, plan := "--", "--"
		if pin.Attestation.BOMSHA256 != "" {
			bomHash = pin.Attestation.BOMSHA256[:12]
		}
		if pin.Attestation.BOM != nil {
			plan = pin.Attestation.BOM.PlanSHA256[:12]
		}
		fmt.Fprintf(table, "%s\t%s\t%s\t%s\n", pin.SHA256[:12], pin.Attestation.Status, bomHash, plan)
	}
	_ = table.Flush()
}

func validateAuditCacheCapacity(cache *lookaside.Cache, required int64) error {
	if !cache.Retained() || cache.MaxBytes() <= 0 || cache.MaxBytes() >= required {
		return nil
	}
	return fmt.Errorf("full audit requires %s of simultaneously materialized shards, but lookaside.cache.max-size is %s; use a dedicated WALDO_CONFIG with a cache bound at least as large as the audited selection", humanBytes(required), humanBytes(cache.MaxBytes()))
}

func runIndexVerifyWithProgress(context Context, args []string, stdout, progress io.Writer) error {
	objects := boolOption(context, "objects")
	offline := boolOption(context, "offline")
	if objects && offline {
		return fmt.Errorf("--objects and --offline are different verification levels; choose one")
	}
	target, err := resolveIndexArgumentPolicy(context.Execution, args, progress, !offline)
	if err != nil {
		return err
	}
	verification, err := waldoindex.Verify(target)
	if err != nil {
		return err
	}
	if offline && context.JSON {
		return writeJSON(stdout, struct {
			Path         string                  `json:"path"`
			Verification waldoindex.Verification `json:"verification"`
		}{Path: target.Rel, Verification: verification})
	}
	if offline {
		fmt.Fprintf(stdout, "verified %s: %s directories, %s corpora, %s shards\n",
			displayPath(target.Rel), humanInteger(verification.Directories), humanInteger(verification.Corpora), humanInteger(verification.Shards))
		return nil
	}

	policy, err := corpus.NewLicensePolicy(nil, nil)
	if err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	bom, err := corpus.BuildBOM(context.Execution, []waldoindex.Target{target}, policy, cache)
	if err != nil {
		return err
	}
	if !objects {
		fmt.Fprintf(progress, "checking %s canonical object URLs (%s declared; headers only)\n",
			humanInteger(int64(len(bom.Shards))), humanBytes(bom.Totals.Bytes))
		availability, err := corpus.CheckAvailability(context.Execution, bom, cache, 8, func(event corpus.AvailabilityProgress) {
			writeIndexProgressRange(progress, event.Current, event.Total, "objects checked")
		})
		if err != nil {
			return err
		}
		purged, err := cache.PurgeUsed()
		if err != nil {
			return fmt.Errorf("purge successful availability-check scratch: %w", err)
		}
		if context.JSON {
			return writeJSON(stdout, struct {
				Path          string                  `json:"path"`
				Verification  waldoindex.Verification `json:"verification"`
				Availability  corpus.Availability     `json:"availability"`
				ScratchPurged lookaside.Stats         `json:"scratch_purged"`
			}{Path: target.Rel, Verification: verification, Availability: availability, ScratchPurged: purged})
		}
		fmt.Fprintf(stdout, "verified %s: %s directories, %s corpora, %s reachable objects (%s; bodies not downloaded)\n",
			displayPath(target.Rel), humanInteger(verification.Directories), humanInteger(verification.Corpora),
			humanInteger(int64(availability.Objects)), humanBytes(availability.Bytes))
		return nil
	}
	fmt.Fprintf(progress, "verifying %s objects (%s) through lookaside cache %s\n",
		humanInteger(int64(len(bom.Shards))), humanBytes(bom.Totals.Bytes), cache.Root())
	materialized, err := corpus.Materialize(context.Execution, bom, cache, func(event corpus.MaterializeProgress) {
		if event.Phase != "complete" {
			return
		}
		writeIndexProgressRange(progress, event.Current, event.Total, "objects verified")
	})
	if err != nil {
		return err
	}
	purged, err := cache.PurgeUsed()
	if err != nil {
		return fmt.Errorf("purge successful verification cache: %w", err)
	}
	if purged.Objects > 0 {
		fmt.Fprintf(progress, "purged %s cached objects (%s)\n", humanInteger(purged.Objects), humanBytes(purged.Bytes))
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Path         string                  `json:"path"`
			Verification waldoindex.Verification `json:"verification"`
			Objects      int                     `json:"objects_verified"`
			Bytes        int64                   `json:"bytes_verified"`
			Scratch      string                  `json:"scratch"`
			Purged       lookaside.Stats         `json:"scratch_purged"`
		}{Path: target.Rel, Verification: verification, Objects: len(materialized.Objects), Bytes: bom.Totals.Bytes, Scratch: cache.Root(), Purged: purged})
	}
	fmt.Fprintf(stdout, "verified %s: %s directories, %s corpora, %s objects (%s)\n",
		displayPath(target.Rel), humanInteger(verification.Directories), humanInteger(verification.Corpora),
		humanInteger(int64(len(materialized.Objects))), humanBytes(bom.Totals.Bytes))
	return nil
}

func indexProgressRange(current, total int) (int, int, bool) {
	if current <= 0 || total <= 0 || current > total || current != total && current%25 != 0 {
		return 0, 0, false
	}
	return (current-1)/25*25 + 1, current, true
}

func writeIndexProgressRange(output io.Writer, current, total int, label string) {
	start, end, report := indexProgressRange(current, total)
	if !report {
		return
	}
	fmt.Fprintf(output, "  %s-%s/%s  %s %s\n",
		humanInteger(int64(start)), humanInteger(int64(end)), humanInteger(int64(total)),
		humanInteger(int64(end-start+1)), label)
}

func resolveIndexArgument(execution context.Context, args []string, warnings io.Writer) (waldoindex.Target, error) {
	return resolveIndexArgumentPolicy(execution, args, warnings, true)
}

func resolveIndexArgumentPolicy(execution context.Context, args []string, warnings io.Writer, refresh bool) (waldoindex.Target, error) {
	targets, err := resolveIndexArgumentsWithWarningPolicy(execution, args, warnings, refresh)
	if err != nil {
		return waldoindex.Target{}, err
	}
	return targets[0], nil
}

func resolveIndexArguments(execution context.Context, args []string, progress io.Writer) ([]waldoindex.Target, error) {
	selection, err := resolveIndexSelection(execution, args, progress, true)
	return selection.Targets, err
}

type resolvedIndexSelection struct {
	Targets []waldoindex.Target
}

var indexGitManager = managedgit.DefaultManager()

func resolveIndexArgumentsWithWarning(execution context.Context, args []string, warnings io.Writer) ([]waldoindex.Target, error) {
	return resolveIndexArgumentsWithWarningPolicy(execution, args, warnings, true)
}

func resolveIndexArgumentsWithWarningPolicy(execution context.Context, args []string, warnings io.Writer, refresh bool) ([]waldoindex.Target, error) {
	selection, err := resolveIndexSelection(execution, args, warnings, refresh)
	if err != nil {
		return nil, err
	}
	return selection.Targets, nil
}

func resolveIndexSelection(execution context.Context, args []string, progress io.Writer, refresh bool) (resolvedIndexSelection, error) {
	configuration, err := config.Load()
	if err != nil {
		return resolvedIndexSelection{}, err
	}
	values := args
	if len(values) == 0 {
		values = []string{""}
	}
	targets := make([]waldoindex.Target, 0, len(values))
	knownRoot, managed, err := config.EffectiveIndexRoot(configuration)
	if err != nil {
		return resolvedIndexSelection{}, err
	}
	explicit := explicitIndexPath(values[0])
	if managed && !explicit && refresh {
		ensured, err := indexGitManager.Ensure(execution, knownRoot, progress)
		if err != nil {
			return resolvedIndexSelection{}, err
		}
		if ensured.Action != "cloned" {
			if err := refreshManagedIndexCheckout(execution, knownRoot, progress); err != nil {
				return resolvedIndexSelection{}, err
			}
		}
	} else if !explicit && refresh {
		if err := refreshIndexCheckout(execution, knownRoot, progress); err != nil {
			return resolvedIndexSelection{}, err
		}
	}
	for _, value := range values {
		var target waldoindex.Target
		if len(targets) == 0 {
			target, err = waldoindex.ResolveConfigured(knownRoot, value)
		} else {
			target, err = waldoindex.Resolve(knownRoot, value)
		}
		if err != nil {
			return resolvedIndexSelection{}, err
		}
		if len(targets) == 0 {
			knownRoot = target.Root
		} else if target.Root != targets[0].Root {
			return resolvedIndexSelection{}, fmt.Errorf("index targets span different checkouts: %s and %s", targets[0].Root, target.Root)
		}
		targets = append(targets, target)
	}
	return resolvedIndexSelection{Targets: targets}, nil
}

func refreshManagedIndexCheckout(execution context.Context, root string, progress io.Writer) error {
	result, err := indexGitManager.PullManaged(execution, root, progress)
	if err != nil {
		return fmt.Errorf("refresh managed index checkout %s: %w", root, err)
	}
	if (result.Action == "updated" || result.Action == "recovered") && progress != nil {
		commit := result.State.Commit
		if len(commit) > 12 {
			commit = commit[:12]
		}
		verb := "updated"
		if result.Action == "recovered" {
			verb = "recovered"
		}
		fmt.Fprintf(progress, "%s managed index checkout %s at %s\n", verb, root, commit)
	}
	return nil
}

func refreshIndexCheckout(execution context.Context, root string, progress io.Writer) error {
	result, err := managedgit.CheckoutPull(execution, root, progress)
	if errors.Is(err, managedgit.ErrNotRepository) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("refresh index checkout %s: %w", root, err)
	}
	if result.Action == "updated" && progress != nil {
		commit := result.State.Commit
		if len(commit) > 12 {
			commit = commit[:12]
		}
		fmt.Fprintf(progress, "updated index checkout %s to %s\n", root, commit)
	}
	return nil
}

func explicitIndexPath(value string) bool {
	return waldoindex.IsFilesystemPath(value)
}

func managedIndexMutationError(action string) error {
	root, err := config.ManagedIndexRoot()
	if err != nil {
		return err
	}
	return fmt.Errorf("cannot %s the managed read-only index %s; create a separate Git checkout and select it with `waldo config set index <directory>` for corpus contributions", action, root)
}

func printManifest(w io.Writer, path string, manifest waldoindex.Manifest) {
	fmt.Fprintf(w, "%s\n", manifest.Title)
	fmt.Fprintf(w, "  path         %s\n", path)
	fmt.Fprintf(w, "  name         %s\n", manifest.Name)
	fmt.Fprintf(w, "  license      %s\n", manifest.License)
	fmt.Fprintf(w, "  description  %s\n", manifest.Description)
	languages, programmingLanguages := waldoindex.DeclaredLanguages(manifest)
	fmt.Fprintf(w, "  languages    %s\n", declaredList(languages))
	fmt.Fprintf(w, "  programming  %s\n", declaredList(programmingLanguages))
	var shards, docs, tokens, bytes int64
	if manifest.Rollup != nil {
		shards, docs, tokens, bytes = manifest.Rollup.Count, manifest.Rollup.Docs, manifest.Rollup.Tokens, manifest.Rollup.Bytes
	} else {
		shards = int64(len(manifest.Shards))
		for _, shard := range manifest.Shards {
			docs += shard.Docs
			tokens += shard.Tokens
			bytes += shard.Bytes
		}
	}
	fmt.Fprintf(w, "  contents     %s shards, %s docs, %s tokens, %s\n", humanInteger(shards), humanCount(docs), humanCount(tokens), humanBytes(bytes))
	fmt.Fprintln(w, "  sources")
	for _, source := range manifest.Sources {
		fmt.Fprintf(w, "    %s — %s\n", source.Name, source.URL)
	}
}

func writeJSON(w io.Writer, value any) error {
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	encoder.SetEscapeHTML(false)
	return encoder.Encode(value)
}

func displayPath(path string) string {
	if path == "" {
		return "."
	}
	return path
}

func licenseSummary(licenses []string) string {
	switch len(licenses) {
	case 0:
		return "(none declared)"
	case 1:
		return truncateDisplay(licenses[0], 40)
	default:
		return fmt.Sprintf("%d licenses", len(licenses))
	}
}

func truncateDisplay(value string, max int) string {
	runes := []rune(value)
	if len(runes) <= max {
		return value
	}
	return string(runes[:max-1]) + "…"
}
