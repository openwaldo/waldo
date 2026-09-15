// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package index

import (
	"fmt"
	"net/url"
	"os"
	pathpkg "path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

var sha256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
var contentHashPathPattern = regexp.MustCompile(`^[0-9a-f]{32,}$`)

type Verification struct {
	Directories int64 `json:"directories"`
	Corpora     int64 `json:"corpora"`
	Shards      int64 `json:"shards"`
}

// Verify checks the local metadata tree without network access. Object
// availability and content hashes are a separate, explicit operation.
func Verify(target Target) (Verification, error) {
	info, err := os.Stat(target.Abs)
	if err != nil {
		return Verification{}, err
	}
	var result Verification
	if !info.IsDir() {
		manifest, err := LoadManifest(target.Abs)
		if err != nil {
			return result, err
		}
		if err := verifyManifest(target.Abs, manifest); err != nil {
			return result, err
		}
		result.Corpora = 1
		result.Shards = manifestShardCount(manifest)
		return result, nil
	}
	if err := verifyDirectory(target.Root, target.Abs, &result); err != nil {
		return result, err
	}
	return result, nil
}

func verifyDirectory(root, dir string, result *Verification) error {
	index, err := LoadDirectory(dir)
	if err != nil {
		return err
	}
	path, err := DirectoryPath(dir)
	if err != nil {
		return err
	}
	if index.Kind != "index" {
		return fmt.Errorf("%s: kind is %q, want %q", path, index.Kind, "index")
	}
	if !SupportedDirectorySchema(index.Schema) {
		return fmt.Errorf("%s: unsupported index schema %d", path, index.Schema)
	}
	rel, err := filepath.Rel(root, dir)
	if err != nil {
		return err
	}
	if rel == "." {
		rel = ""
	}
	wantPath := filepath.ToSlash(rel)
	if index.Path != wantPath {
		return fmt.Errorf("%s: path is %q, want %q", path, index.Path, wantPath)
	}

	seen := map[string]bool{}
	names := make([]string, 0, len(index.Entries))
	for _, entry := range index.Entries {
		if entry.Name == "" || entry.Name == "." || entry.Name == ".." || filepath.Base(entry.Name) != entry.Name {
			return fmt.Errorf("%s: invalid entry name %q", path, entry.Name)
		}
		if seen[entry.Name] {
			return fmt.Errorf("%s: duplicate entry %q", path, entry.Name)
		}
		seen[entry.Name] = true
		names = append(names, entry.Name)
	}
	if !sort.StringsAreSorted(names) {
		return fmt.Errorf("%s: entries are not sorted by name", path)
	}

	result.Directories++
	for _, entry := range index.Entries {
		entryPath := filepath.Join(dir, entry.Name)
		info, err := os.Stat(entryPath)
		if err != nil {
			return fmt.Errorf("%s: indexed entry %q: %w", path, entry.Name, err)
		}
		switch entry.Type {
		case "dir":
			if !info.IsDir() {
				return fmt.Errorf("%s: entry %q is declared as a directory but is a file", path, entry.Name)
			}
			if err := verifyDirectory(root, entryPath, result); err != nil {
				return err
			}
		case "manifest":
			if info.IsDir() {
				return fmt.Errorf("%s: entry %q is declared as a manifest but is a directory", path, entry.Name)
			}
			manifest, err := LoadManifest(entryPath)
			if err != nil {
				return err
			}
			if err := verifyManifest(entryPath, manifest); err != nil {
				return err
			}
			result.Corpora++
			result.Shards += manifestShardCount(manifest)
		default:
			return fmt.Errorf("%s: entry %q has unsupported type %q", path, entry.Name, entry.Type)
		}
	}
	return nil
}

func verifyManifest(path string, manifest Manifest) error {
	if manifest.Kind != "manifest" {
		return fmt.Errorf("%s: kind is %q, want %q", path, manifest.Kind, "manifest")
	}
	if manifest.Schema != LegacyManifestSchema && manifest.Schema != ManifestSchema {
		return fmt.Errorf("%s: unsupported manifest schema %d", path, manifest.Schema)
	}
	wantName := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	if manifest.Name != wantName {
		return fmt.Errorf("%s: name is %q, want filename-derived %q", path, manifest.Name, wantName)
	}
	if manifest.Title == "" {
		return fmt.Errorf("%s: title is required", path)
	}
	if manifest.Description == "" {
		return fmt.Errorf("%s: description is required", path)
	}
	if manifest.Schema == LegacyManifestSchema && manifest.License == "" {
		return fmt.Errorf("%s: default license is required", path)
	}
	if manifest.Schema == ManifestSchema && ((manifest.License == "") == (len(manifest.Licenses) == 0)) {
		return fmt.Errorf("%s: exactly one of license or licenses is required", path)
	}
	if len(manifest.Licenses) > 0 && !sortedUniqueStrings(manifest.Licenses) {
		return fmt.Errorf("%s: licenses must be sorted, unique, and non-empty", path)
	}
	if len(manifest.Sources) == 0 {
		return fmt.Errorf("%s: at least one source is required", path)
	}
	if !validConversion(manifest.ConvertedBy) {
		return fmt.Errorf("%s: converted_by requires tool, version, profile, and recipe", path)
	}
	if manifest.RecordKind != "" && manifest.RecordKind != "pretrain" && manifest.RecordKind != "conversation" {
		return fmt.Errorf("%s: unsupported record_kind %q", path, manifest.RecordKind)
	}
	if len(manifest.Shards) == 0 && manifest.Rollup == nil {
		return fmt.Errorf("%s: shards or rollup is required", path)
	}
	if len(manifest.Shards) > 0 && manifest.Rollup != nil {
		return fmt.Errorf("%s: shards and rollup are mutually exclusive", path)
	}
	conversation := manifest.RecordKind == "conversation"
	if conversation || manifest.RecordSchema >= 2 {
		if err := validateContentAssessment(manifest.Assessment, -1); err != nil {
			return fmt.Errorf("%s: manifest assessment: %w", path, err)
		}
	}
	privacyRedacted := manifest.ConvertedBy.Recipe == "parquet-go/0.30.1/zstd-6/page-1m/rg-64m/v9-privacy-redaction" || manifest.ConvertedBy.Recipe == "parquet-go/0.30.1/zstd-6/page-1m/rg-64m/conversation-v1"
	if privacyRedacted {
		if err := validateContentRedaction(manifest.Redaction); err != nil {
			return fmt.Errorf("%s: manifest redaction: %w", path, err)
		}
	}

	sources := map[string]bool{}
	for i, source := range manifest.Sources {
		if source.Name == "" || source.Source == "" || source.URL == "" || source.SHA256 == "" {
			return fmt.Errorf("%s: source %d requires name, source, url, and sha256", path, i+1)
		}
		if sources[source.Name] {
			return fmt.Errorf("%s: duplicate source name %q", path, source.Name)
		}
		if !sha256Pattern.MatchString(source.SHA256) {
			return fmt.Errorf("%s: source %q has invalid sha256 %q", path, source.Name, source.SHA256)
		}
		sources[source.Name] = true
		for j, file := range source.Files {
			if file.Name == "" || file.URL == "" || !sha256Pattern.MatchString(file.SHA256) || file.Bytes < 0 {
				return fmt.Errorf("%s: source %q file %d requires name, URL, and lowercase 64-character sha256", path, source.Name, j+1)
			}
			if file.TextColumn != "" && file.Adapter != "parquet" {
				return fmt.Errorf("%s: source %q file %d has text_column without parquet adapter", path, source.Name, j+1)
			}
		}
	}
	if manifest.ComposedBy != nil {
		if err := ValidateIngestRecipeEvidence(*manifest.ComposedBy); err != nil {
			return fmt.Errorf("%s: composed_by: %w", path, err)
		}
	}
	var assessedEmailRecords int64
	var assessedRepetitiveRecords int64
	var assessedBoilerplateRecords int64
	redaction := &ContentRedaction{Policy: "waldo/privacy-redaction-v1", NamesRetained: true}
	anyPrivacyRedacted := privacyRedacted
	for i, shard := range manifest.Shards {
		if shard.URL == "" || !sha256Pattern.MatchString(shard.SHA256) {
			return fmt.Errorf("%s: shard %d requires a URL and lowercase 64-character sha256", path, i+1)
		}
		if shard.Docs < 0 || shard.Tokens < 0 || shard.Bytes < 0 {
			return fmt.Errorf("%s: shard %s has negative totals", path, shard.SHA256[:12])
		}
		if object, ok := contentHashPath(shard.URL); ok && object != shard.SHA256 {
			return fmt.Errorf("%s: shard %s URL object %q does not match its sha256", path, shard.SHA256[:12], object)
		}
		for _, name := range shard.Sources {
			if !sources[name] {
				return fmt.Errorf("%s: shard %s refers to unknown source %q", path, shard.SHA256[:12], name)
			}
		}
		if shard.License != "" && len(shard.Licenses) > 0 {
			return fmt.Errorf("%s: shard %s has both license and licenses", path, shard.SHA256[:12])
		}
		if len(shard.Licenses) > 0 && !sortedUniqueStrings(shard.Licenses) {
			return fmt.Errorf("%s: shard %s licenses must be sorted, unique, and non-empty", path, shard.SHA256[:12])
		}
		licenses := manifest.EffectiveLicenses(shard)
		if len(licenses) == 0 {
			return fmt.Errorf("%s: shard %s has no effective license", path, shard.SHA256[:12])
		}
		if len(licenses) > 1 {
			if len(shard.LicenseUsage) != len(licenses) {
				return fmt.Errorf("%s: shard %s license_usage must cover every represented license", path, shard.SHA256[:12])
			}
			var docs, tokens int64
			for _, license := range licenses {
				usage, ok := shard.LicenseUsage[license]
				if !ok || usage.Docs <= 0 || usage.Tokens < 0 || usage.Shards != 0 || usage.Bytes != 0 {
					return fmt.Errorf("%s: shard %s has invalid license_usage for %q", path, shard.SHA256[:12], license)
				}
				docs += usage.Docs
				tokens += usage.Tokens
			}
			if docs != shard.Docs || tokens != shard.Tokens {
				return fmt.Errorf("%s: shard %s license_usage does not match shard totals", path, shard.SHA256[:12])
			}
		}
		if shard.ConvertedBy != nil && !validConversion(*shard.ConvertedBy) {
			return fmt.Errorf("%s: shard %s has an incomplete converted_by override", path, shard.SHA256[:12])
		}
		if conversation || manifest.RecordSchema >= 2 {
			if err := validateContentAssessment(shard.Assessment, shard.Docs); err != nil {
				return fmt.Errorf("%s: shard %s assessment: %w", path, shard.SHA256[:12], err)
			}
			assessedEmailRecords += shard.Assessment.EmailAddresses.Records
			assessedRepetitiveRecords += shard.Assessment.RepetitiveContent.Records
			assessedBoilerplateRecords += shard.Assessment.BoilerplateContent.Records
		}
		shardPrivacyRedacted := privacyRedacted || shard.ConvertedBy != nil && (shard.ConvertedBy.Recipe == "parquet-go/0.30.1/zstd-6/page-1m/rg-64m/v9-privacy-redaction" || shard.ConvertedBy.Recipe == "parquet-go/0.30.1/zstd-6/page-1m/rg-64m/conversation-v1")
		if shardPrivacyRedacted {
			anyPrivacyRedacted = true
			if err := validateContentRedaction(shard.Redaction); err != nil {
				return fmt.Errorf("%s: shard %s redaction: %w", path, shard.SHA256[:12], err)
			}
			redaction.EmailAddresses += shard.Redaction.EmailAddresses
			redaction.IPAddresses += shard.Redaction.IPAddresses
			redaction.PhoneNumbers += shard.Redaction.PhoneNumbers
			redaction.MailRoutingHeaders += shard.Redaction.MailRoutingHeaders
			redaction.Credentials += shard.Redaction.Credentials
		}
	}
	if (conversation || manifest.RecordSchema >= 2) && manifest.Rollup == nil && (manifest.Assessment.EmailAddresses.Records != assessedEmailRecords || manifest.Assessment.RepetitiveContent.Records != assessedRepetitiveRecords || manifest.Assessment.BoilerplateContent.Records != assessedBoilerplateRecords) {
		return fmt.Errorf("%s: manifest content assessment does not equal shard counts", path)
	}
	if anyPrivacyRedacted && manifest.Rollup == nil {
		if err := validateContentRedaction(manifest.Redaction); err != nil {
			return fmt.Errorf("%s: manifest redaction: %w", path, err)
		}
		if *manifest.Redaction != *redaction {
			return fmt.Errorf("%s: manifest redaction does not equal shard counts", path)
		}
	}
	if manifest.Rollup != nil {
		if manifest.Rollup.URL == "" || !sha256Pattern.MatchString(manifest.Rollup.SHA256) {
			return fmt.Errorf("%s: rollup requires a URL and lowercase 64-character sha256", path)
		}
		if manifest.Rollup.Count <= 0 || manifest.Rollup.Docs <= 0 || manifest.Rollup.Tokens < 0 || manifest.Rollup.Bytes <= 0 {
			return fmt.Errorf("%s: rollup count, docs, and bytes must be positive and tokens non-negative", path)
		}
	}
	if err := validateManifestProvenance(manifest); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func validateContentRedaction(redaction *ContentRedaction) error {
	if redaction == nil || redaction.Policy != "waldo/privacy-redaction-v1" || !redaction.NamesRetained {
		return fmt.Errorf("privacy policy and names_retained are required")
	}
	if redaction.EmailAddresses < 0 || redaction.IPAddresses < 0 || redaction.PhoneNumbers < 0 || redaction.MailRoutingHeaders < 0 || redaction.Credentials < 0 {
		return fmt.Errorf("redaction counts must be non-negative")
	}
	return nil
}

func validateContentAssessment(assessment *ContentAssessment, documents int64) error {
	if assessment == nil {
		return fmt.Errorf("content assessment is required")
	}
	for _, field := range []struct {
		name    string
		measure *DetectionMeasure
	}{
		{name: "email_addresses", measure: assessment.EmailAddresses},
		{name: "repetitive_content", measure: assessment.RepetitiveContent},
		{name: "boilerplate_content", measure: assessment.BoilerplateContent},
	} {
		if field.measure == nil || field.measure.Detector == "" || field.measure.Records < 0 || documents >= 0 && field.measure.Records > documents {
			return fmt.Errorf("%s requires a detector and a valid record count", field.name)
		}
	}
	return nil
}

func sortedUniqueStrings(values []string) bool {
	if len(values) == 0 || !sort.StringsAreSorted(values) {
		return false
	}
	for position, value := range values {
		if value == "" || position > 0 && value == values[position-1] {
			return false
		}
	}
	return true
}

func contentHashPath(raw string) (string, bool) {
	parsed, err := url.Parse(raw)
	if err != nil {
		return "", false
	}
	object := pathpkg.Base(parsed.Path)
	return object, contentHashPathPattern.MatchString(object)
}

// ValidateIngestRecipeEvidence validates portable historical recipe evidence
// carried by an index manifest and copied into an OpenWALDO BOM.
func ValidateIngestRecipeEvidence(recipe IngestRecipeEvidence) error {
	if recipe.Path == "" || !sha256Pattern.MatchString(recipe.SHA256) || len(recipe.Steps) == 0 {
		return fmt.Errorf("path, sha256, and steps are required")
	}
	seenSteps := map[string]bool{}
	for i, step := range recipe.Steps {
		if step.Name == "" || seenSteps[step.Name] || step.Executable == "" || !sha256Pattern.MatchString(step.SHA256) {
			return fmt.Errorf("step %d requires unique name, script, and lowercase 64-character sha256", i+1)
		}
		seenSteps[step.Name] = true
	}
	return nil
}

func validConversion(conversion Conversion) bool {
	return conversion.Tool != "" && conversion.Version != "" && conversion.Profile != "" && conversion.Recipe != ""
}

// ValidateManifest validates a manifest using the filename-derived identity
// rules that apply inside an index checkout.
func ValidateManifest(path string, manifest Manifest) error {
	return verifyManifest(path, manifest)
}

func manifestShardCount(manifest Manifest) int64 {
	if manifest.Rollup != nil {
		return manifest.Rollup.Count
	}
	return int64(len(manifest.Shards))
}
