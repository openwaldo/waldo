// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
)

func runLookasideStatus(context Context, _ []string, stdout, _ io.Writer) error {
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	stats, err := cache.Stats()
	if err != nil {
		return err
	}
	configuration, err := config.Load()
	if err != nil {
		return err
	}
	protected, owners, err := protectedCacheObjects(configuration)
	if err != nil {
		return err
	}
	protectedStats, err := cache.StatsFor(protected)
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Cache       string                     `json:"cache"`
			Scratch     string                     `json:"scratch"`
			MaxBytes    int64                      `json:"cache_max_bytes"`
			Mirrors     []string                   `json:"mirrors"`
			Publish     *config.Publish            `json:"publish,omitempty"`
			Credentials *lookasideCredentialStatus `json:"credentials,omitempty"`
			Stats       lookaside.Stats            `json:"stats"`
			Protected   lookaside.Stats            `json:"protected"`
			Owners      []string                   `json:"protected_models,omitempty"`
		}{Cache: cache.Root(), Scratch: cache.Scratch(), MaxBytes: cache.MaxBytes(), Mirrors: cache.Mirrors(), Publish: configuration.Lookaside.Publish, Credentials: credentialStatus(configuration.Lookaside.Publish), Stats: stats, Protected: protectedStats, Owners: owners})
	}
	fmt.Fprintf(stdout, "lookaside cache    %s\n", cache.Root())
	fmt.Fprintf(stdout, "  limit          %s\n", humanBytes(cache.MaxBytes()))
	fmt.Fprintf(stdout, "lookaside scratch  %s\n", cache.Scratch())
	fmt.Fprintf(stdout, "  objects        %s\n", humanInteger(stats.Objects))
	fmt.Fprintf(stdout, "  bytes          %s\n", humanBytes(stats.Bytes))
	fmt.Fprintf(stdout, "  protected      %s objects, %s", humanInteger(protectedStats.Objects), humanBytes(protectedStats.Bytes))
	if len(owners) > 0 {
		fmt.Fprintf(stdout, " (%s)\n", joinHuman(owners))
	} else {
		fmt.Fprintln(stdout)
	}
	if stats.Other > 0 {
		fmt.Fprintf(stdout, "  other files    %s\n", humanInteger(stats.Other))
	}
	if len(cache.Mirrors()) == 0 {
		fmt.Fprintln(stdout, "  mirrors        (none)")
	} else {
		for i, mirror := range cache.Mirrors() {
			label := ""
			if i == 0 {
				label = "mirrors"
			}
			fmt.Fprintf(stdout, "  %-13s  %s\n", label, mirror)
		}
	}
	if configuration.Lookaside.Publish == nil {
		fmt.Fprintln(stdout, "  publish        (none)")
	} else {
		publish := configuration.Lookaside.Publish
		fmt.Fprintf(stdout, "  publish        %s (%d workers)\n", publish.URL, publish.Workers)
		if status := credentialStatus(publish); status != nil {
			switch {
			case status.Error != "":
				fmt.Fprintf(stdout, "  credentials    unavailable: %s\n", status.Error)
			case status.Present:
				credentialPath, err := lookaside.CredentialPath()
				if err != nil {
					return err
				}
				fmt.Fprintf(stdout, "  credentials    %s %s (%s)\n", credentialPath, status.Scope, status.AccessKey)
			default:
				fmt.Fprintln(stdout, "  credentials    no WALDO login; AWS default chain fallback")
			}
		}
	}
	return nil
}

func runLookasideCacheClean(context Context, _ []string, stdout, _ io.Writer) error {
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	configuration, err := config.Load()
	if err != nil {
		return err
	}
	protected, owners, err := protectedCacheObjects(configuration)
	if err != nil {
		return err
	}
	all := boolOption(context, "all")
	if all {
		protected = nil
	}
	result, err := cache.Clean(protected)
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Cache           string                `json:"cache"`
			All             bool                  `json:"all"`
			ProtectedModels []string              `json:"protected_models,omitempty"`
			Result          lookaside.CleanResult `json:"result"`
		}{Cache: cache.Root(), All: all, ProtectedModels: owners, Result: result})
	}
	fmt.Fprintf(stdout, "cleaned lookaside cache %s: removed %s objects (%s)\n", cache.Root(), humanInteger(result.Removed.Objects), humanBytes(result.Removed.Bytes))
	if result.Protected.Objects > 0 {
		fmt.Fprintf(stdout, "protected %s objects (%s) needed by %s; use --all to remove them\n", humanInteger(result.Protected.Objects), humanBytes(result.Protected.Bytes), joinHuman(owners))
	} else if all && len(owners) > 0 && result.Removed.Objects > 0 {
		fmt.Fprintf(stdout, "warning: removed cached objects without protecting running or resumable models: %s\n", joinHuman(owners))
	}
	return nil
}

func protectedCacheObjects(configuration config.Config) (map[string]bool, []string, error) {
	root, err := config.EffectiveModelRoot(configuration)
	if err != nil {
		return nil, nil, err
	}
	listings, err := model.List(root, nil)
	if err != nil {
		return nil, nil, err
	}
	protected := map[string]bool{}
	ownerSet := map[string]bool{}
	for _, listing := range listings {
		inspection, err := model.Inspect(root, listing.Name)
		if err != nil {
			return nil, nil, err
		}
		for index, run := range inspection.Runs {
			resumableFailure := run.State == model.RunFailed && index == len(inspection.Runs)-1 && model.HasRecoverableCheckpointFailure(inspection)
			if run.State != model.RunRunning && run.State != model.RunInterrupted && !resumableFailure {
				continue
			}
			if index >= len(inspection.RunBOMs) {
				continue
			}
			for _, shard := range inspection.RunBOMs[index].CorpusBOM.Shards {
				protected[shard.SHA256] = true
			}
			ownerSet[listing.Name] = true
		}
	}
	owners := make([]string, 0, len(ownerSet))
	for name := range ownerSet {
		owners = append(owners, name)
	}
	sort.Strings(owners)
	return protected, owners, nil
}

func joinHuman(values []string) string {
	if len(values) == 0 {
		return "no models"
	}
	return strings.Join(values, ", ")
}

func runLookasideVerify(context Context, _ []string, stdout, _ io.Writer) error {
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	result, err := cache.Scrub()
	if err != nil {
		return err
	}
	if context.JSON {
		if err := writeJSON(stdout, struct {
			Root   string                `json:"root"`
			Result lookaside.ScrubResult `json:"result"`
		}{Root: cache.Root(), Result: result}); err != nil {
			return err
		}
	} else {
		fmt.Fprintf(stdout, "scrubbed %s: %s verified, %s corrupt, %s total\n",
			cache.Root(), humanInteger(result.Verified), humanInteger(int64(len(result.Corrupt))), humanBytes(result.Bytes))
		for _, issue := range result.Corrupt {
			fmt.Fprintf(stdout, "  CORRUPT %s: %s\n", issue.Path, issue.Error)
		}
	}
	if len(result.Corrupt) > 0 {
		return fmt.Errorf("lookaside cache contains %d corrupt object(s)", len(result.Corrupt))
	}
	return nil
}
