// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package corpus

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/record"
	waldoshard "github.com/openwaldo/waldo/internal/shard"
)

// Validate checks the internally provable claims in an OpenWALDO BOM. It does
// not contact Git or the network; hashes establish identity, not availability.
func (bom BOM) Validate() error {
	if bom.Kind != "openwaldo-bom" || bom.Schema != BOMSchema || bom.Subject != "corpus" {
		return fmt.Errorf("unsupported OpenWALDO BOM identity %q schema %d subject %q", bom.Kind, bom.Schema, bom.Subject)
	}
	if bom.Index.Commit != "" && !validGitHash(bom.Index.Commit) {
		return fmt.Errorf("index commit %q is not a lowercase Git object hash", bom.Index.Commit)
	}
	if !slices.IsSorted(bom.Paths) || hasDuplicate(bom.Paths) {
		return fmt.Errorf("BOM paths must be sorted and unique")
	}
	policy, err := NewLicensePolicy(bom.Policy.Include, bom.Policy.Exclude)
	if err != nil {
		return err
	}
	if !slices.Equal(policy.Include, bom.Policy.Include) || !slices.Equal(policy.Exclude, bom.Policy.Exclude) {
		return fmt.Errorf("license policy patterns must be non-empty and unique")
	}
	if bom.RecordFilter != nil {
		if err := bom.RecordFilter.Validate(bom.Paths); err != nil {
			return err
		}
	}

	manifests := make(map[string]ManifestPin, len(bom.Manifests))
	sourceNames := make(map[string]map[string]bool, len(bom.Manifests))
	manifestTotals := make(map[string]index.Measures, len(bom.Manifests))
	manifestModalities := make(map[string]index.Modalities, len(bom.Manifests))
	manifestLicenses := make(map[string]map[string]index.Measures, len(bom.Manifests))
	manifestEmailRecords := make(map[string]int64, len(bom.Manifests))
	manifestRepetitiveRecords := make(map[string]int64, len(bom.Manifests))
	manifestBoilerplateRecords := make(map[string]int64, len(bom.Manifests))
	for _, manifest := range bom.Manifests {
		manifest.RecordKind = effectiveRecordKind(manifest.RecordKind)
		if manifest.Path == "" || manifests[manifest.Path].Path != "" {
			return fmt.Errorf("manifest paths must be non-empty and unique: %q", manifest.Path)
		}
		if !validSHA256(manifest.SHA256) {
			return fmt.Errorf("manifest %s has invalid sha256 %q", manifest.Path, manifest.SHA256)
		}
		if manifest.Name == "" || manifest.Title == "" || manifest.Description == "" || (manifest.License == "") == (len(manifest.LicenseSet) == 0) || manifest.Format == "" || manifest.RecordSchema <= 0 || (manifest.RecordKind != record.KindPretrain && manifest.RecordKind != record.KindConversation) {
			return fmt.Errorf("manifest %s is missing resolved identity or format fields", manifest.Path)
		}
		if len(manifest.LicenseSet) > 0 && (!slices.IsSorted(manifest.LicenseSet) || hasDuplicate(manifest.LicenseSet)) {
			return fmt.Errorf("manifest %s license set must be sorted and unique", manifest.Path)
		}
		if !completeConversion(manifest.ConvertedBy, manifest.RecordSchema) {
			return fmt.Errorf("manifest %s has incomplete conversion provenance", manifest.Path)
		}
		names := map[string]bool{}
		for _, source := range manifest.Sources {
			if source.Name == "" || source.Source == "" || source.URL == "" || !validSHA256(source.SHA256) || names[source.Name] {
				return fmt.Errorf("manifest %s has an invalid or duplicate source %q", manifest.Path, source.Name)
			}
			names[source.Name] = true
			for _, file := range source.Files {
				if file.Name == "" || file.URL == "" || !validSHA256(file.SHA256) {
					return fmt.Errorf("manifest %s source %s has an invalid source file", manifest.Path, source.Name)
				}
			}
			if err := index.ValidateSourceProvenance(source); err != nil {
				return fmt.Errorf("manifest %s source %s: %w", manifest.Path, source.Name, err)
			}
		}
		if manifest.Processing != nil {
			if err := index.ValidateProcessing(*manifest.Processing); err != nil {
				return fmt.Errorf("manifest %s processing: %w", manifest.Path, err)
			}
		}
		if manifest.ComposedBy != nil {
			if err := index.ValidateIngestRecipeEvidence(*manifest.ComposedBy); err != nil {
				return fmt.Errorf("manifest %s composed_by: %w", manifest.Path, err)
			}
		}
		if manifest.RecordKind == record.KindConversation || manifest.RecordSchema >= waldoshard.TextRecordSchema {
			if err := validateAssessment(manifest.Assessment, manifest.Totals.Docs); err != nil {
				return fmt.Errorf("manifest %s assessment: %w", manifest.Path, err)
			}
		}
		if manifest.ConvertedBy.Recipe == waldoshard.TextWriterRecipe || manifest.ConvertedBy.Recipe == waldoshard.ConversationWriterRecipe {
			if err := validateRedaction(manifest.Redaction); err != nil {
				return fmt.Errorf("manifest %s redaction: %w", manifest.Path, err)
			}
		}
		if err := index.ValidateModalities("manifest "+manifest.Path, manifest.Modalities); err != nil {
			return err
		}
		manifests[manifest.Path] = manifest
		sourceNames[manifest.Path] = names
		manifestLicenses[manifest.Path] = map[string]index.Measures{}
		manifestModalities[manifest.Path] = index.Modalities{}
	}
	subManifests, err := validateSubManifestPins(bom.SubManifests, manifests)
	if err != nil {
		return err
	}

	calculated := index.Measures{}
	calculatedModalities := index.Modalities{}
	licenses := map[string]index.Measures{}
	for position, shard := range bom.Shards {
		shard.RecordKind = effectiveRecordKind(shard.RecordKind)
		if manifests[shard.Manifest].Path == "" {
			return fmt.Errorf("shard %d refers to unknown manifest %q", position+1, shard.Manifest)
		}
		if shard.URL == "" || !validSHA256(shard.SHA256) || shard.Format == "" || shard.RecordSchema <= 0 || (shard.RecordKind != record.KindPretrain && shard.RecordKind != record.KindConversation) || (shard.License == "") == (len(shard.Licenses) == 0) {
			return fmt.Errorf("shard %d has incomplete resolved identity", position+1)
		}
		if len(shard.Licenses) > 0 && (!slices.IsSorted(shard.Licenses) || hasDuplicate(shard.Licenses)) {
			return fmt.Errorf("shard %s licenses must be sorted and unique", shard.SHA256[:12])
		}
		if !completeConversion(shard.ConvertedBy, shard.RecordSchema) {
			return fmt.Errorf("shard %s has incomplete conversion provenance", shard.SHA256[:12])
		}
		if shard.RecordsRoot != "" && !validSHA256(shard.RecordsRoot) {
			return fmt.Errorf("shard %s has invalid records_root", shard.SHA256[:12])
		}
		if shard.SubManifestSHA256 != "" && !subManifests[shard.Manifest+"\x00"+shard.SubManifestSHA256] {
			return fmt.Errorf("shard %s refers to an unpinned submanifest", shard.SHA256[:12])
		}
		if shard.Docs <= 0 || shard.Tokens < 0 || shard.Bytes <= 0 {
			return fmt.Errorf("shard %s has non-positive totals", shard.SHA256[:12])
		}
		if err := index.ValidateModalities("shard "+shard.SHA256[:12], shard.Modalities); err != nil {
			return err
		}
		if len(shard.Modalities) > 0 && modalityTokens(shard.Modalities) != shard.Tokens {
			return fmt.Errorf("shard %s modality tokens do not match its token total", shard.SHA256[:12])
		}
		if shard.RecordKind == record.KindConversation || shard.RecordSchema >= waldoshard.TextRecordSchema {
			if err := validateAssessment(shard.Assessment, shard.Docs); err != nil {
				return fmt.Errorf("shard %s assessment: %w", shard.SHA256[:12], err)
			}
			manifestEmailRecords[shard.Manifest] += shard.Assessment.EmailAddresses.Records
			manifestRepetitiveRecords[shard.Manifest] += shard.Assessment.RepetitiveContent.Records
			manifestBoilerplateRecords[shard.Manifest] += shard.Assessment.BoilerplateContent.Records
		}
		if shard.ConvertedBy.Recipe == waldoshard.TextWriterRecipe || shard.ConvertedBy.Recipe == waldoshard.ConversationWriterRecipe {
			if err := validateRedaction(shard.Redaction); err != nil {
				return fmt.Errorf("shard %s redaction: %w", shard.SHA256[:12], err)
			}
		}
		seenSources := map[string]bool{}
		for _, source := range shard.Sources {
			if !sourceNames[shard.Manifest][source] || seenSources[source] {
				return fmt.Errorf("shard %s has unknown or duplicate source %q", shard.SHA256[:12], source)
			}
			seenSources[source] = true
		}
		licensesForShard := shard.Licenses
		if len(licensesForShard) == 0 {
			licensesForShard = []string{shard.License}
		}
		for _, license := range licensesForShard {
			if !bom.Policy.Allows(license) {
				return fmt.Errorf("shard %s license %q violates the BOM policy", shard.SHA256[:12], license)
			}
		}
		if len(licensesForShard) > 1 {
			var docs, tokens int64
			if len(shard.LicenseUsage) != len(licensesForShard) {
				return fmt.Errorf("shard %s license usage must cover every represented license", shard.SHA256[:12])
			}
			for _, license := range licensesForShard {
				usage, ok := shard.LicenseUsage[license]
				if !ok || usage.Docs <= 0 || usage.Tokens < 0 || usage.Shards != 0 || usage.Bytes != 0 {
					return fmt.Errorf("shard %s has invalid usage for license %q", shard.SHA256[:12], license)
				}
				docs += usage.Docs
				tokens += usage.Tokens
			}
			if docs != shard.Docs || tokens != shard.Tokens {
				return fmt.Errorf("shard %s license usage does not match shard totals", shard.SHA256[:12])
			}
		} else if len(shard.LicenseUsage) > 0 {
			usage, ok := shard.LicenseUsage[licensesForShard[0]]
			if len(shard.LicenseUsage) != 1 || !ok || usage.Docs != shard.Docs || usage.Tokens != shard.Tokens || usage.Shards != 0 || usage.Bytes != 0 {
				return fmt.Errorf("shard %s has invalid usage for license %q", shard.SHA256[:12], licensesForShard[0])
			}
		}
		if err := validateShardAttestation(shard); err != nil {
			return err
		}
		measure := index.Measures{Shards: 1, Docs: shard.Docs, Tokens: shard.Tokens, Bytes: shard.Bytes}
		addMeasure(&calculated, measure)
		if len(licensesForShard) == 1 || len(shard.LicenseUsage) == 0 {
			addMeasureMap(licenses, licensesForShard[0], measure)
		} else {
			for _, license := range licensesForShard {
				usage := shard.LicenseUsage[license]
				usage.Shards = 1
				addMeasureMap(licenses, license, usage)
			}
		}
		manifestMeasure := manifestTotals[shard.Manifest]
		addMeasure(&manifestMeasure, measure)
		manifestTotals[shard.Manifest] = manifestMeasure
		addModalities(manifestModalities[shard.Manifest], shard.Modalities)
		addModalities(calculatedModalities, shard.Modalities)
		if len(licensesForShard) == 1 || len(shard.LicenseUsage) == 0 {
			addMeasureMap(manifestLicenses[shard.Manifest], licensesForShard[0], measure)
		} else {
			for _, license := range licensesForShard {
				usage := shard.LicenseUsage[license]
				usage.Shards = 1
				addMeasureMap(manifestLicenses[shard.Manifest], license, usage)
			}
		}
	}
	if calculated != bom.Totals || !maps.Equal(licenses, bom.Licenses) || !maps.Equal(calculatedModalities, bom.Modalities) {
		return fmt.Errorf("BOM totals or license totals do not match its shards")
	}
	for path, manifest := range manifests {
		if manifest.Totals != manifestTotals[path] || !maps.Equal(manifest.Licenses, manifestLicenses[path]) || !maps.Equal(manifest.Modalities, manifestModalities[path]) {
			return fmt.Errorf("manifest %s totals do not match its selected shards", path)
		}
		if manifest.RecordSchema >= waldoshard.TextRecordSchema && (manifest.Assessment.EmailAddresses.Records != manifestEmailRecords[path] || manifest.Assessment.RepetitiveContent.Records != manifestRepetitiveRecords[path] || manifest.Assessment.BoilerplateContent.Records != manifestBoilerplateRecords[path]) {
			return fmt.Errorf("manifest %s assessment does not match its selected shards", path)
		}
	}
	return nil
}

func validateAssessment(assessment *index.ContentAssessment, documents int64) error {
	if assessment == nil {
		return fmt.Errorf("content assessment is required")
	}
	for _, field := range []struct {
		name    string
		measure *index.DetectionMeasure
	}{
		{name: "email_addresses", measure: assessment.EmailAddresses},
		{name: "repetitive_content", measure: assessment.RepetitiveContent},
		{name: "boilerplate_content", measure: assessment.BoilerplateContent},
	} {
		if field.measure == nil || field.measure.Detector == "" {
			return fmt.Errorf("%s detector is required", field.name)
		}
		if field.measure.Records < 0 || field.measure.Records > documents {
			return fmt.Errorf("%s record count is invalid", field.name)
		}
	}
	return nil
}

func validateRedaction(redaction *index.ContentRedaction) error {
	if redaction == nil || redaction.Policy != waldoshard.PrivacyRedactionPolicy || !redaction.NamesRetained {
		return fmt.Errorf("privacy policy and names_retained are required")
	}
	if redaction.EmailAddresses < 0 || redaction.IPAddresses < 0 || redaction.PhoneNumbers < 0 || redaction.MailRoutingHeaders < 0 || redaction.Credentials < 0 {
		return fmt.Errorf("redaction counts must be non-negative")
	}
	return nil
}

func validateShardAttestation(pin ShardPin) error {
	if pin.Attestation == nil {
		return nil
	}
	attestation := pin.Attestation
	switch attestation.Status {
	case "embedded":
		if attestation.BOM == nil || !validSHA256(attestation.BOMSHA256) {
			return fmt.Errorf("shard %s has incomplete embedded BOM evidence", pin.SHA256[:12])
		}
		if err := attestation.BOM.Validate(); err != nil {
			return fmt.Errorf("shard %s embedded BOM: %w", pin.SHA256[:12], err)
		}
		encoded, err := json.Marshal(attestation.BOM)
		if err != nil {
			return err
		}
		digest := sha256.Sum256(encoded)
		if hex.EncodeToString(digest[:]) != attestation.BOMSHA256 {
			return fmt.Errorf("shard %s embedded BOM digest differs", pin.SHA256[:12])
		}
		licenses := pin.Licenses
		if len(licenses) == 0 {
			licenses = []string{pin.License}
		}
		emailRecords := int64(0)
		repetitiveRecords := int64(0)
		boilerplateRecords := int64(0)
		if pin.Assessment != nil && pin.Assessment.EmailAddresses != nil {
			emailRecords = pin.Assessment.EmailAddresses.Records
		}
		if pin.Assessment != nil && pin.Assessment.RepetitiveContent != nil {
			repetitiveRecords = pin.Assessment.RepetitiveContent.Records
		}
		if pin.Assessment != nil && pin.Assessment.BoilerplateContent != nil {
			boilerplateRecords = pin.Assessment.BoilerplateContent.Records
		}
		redaction := index.ContentRedaction{}
		if pin.Redaction != nil {
			redaction = *pin.Redaction
		}
		if attestation.WriterRecipe != attestation.BOM.WriterRecipe || effectiveRecordKind(attestation.BOM.RecordKind) != effectiveRecordKind(pin.RecordKind) || attestation.BOM.RecordSchema != pin.RecordSchema || attestation.BOM.Records != pin.Docs || attestation.BOM.Tokens != pin.Tokens || attestation.BOM.EmailAddressRecords != emailRecords || attestation.BOM.RepetitiveContentRecords != repetitiveRecords || attestation.BOM.BoilerplateContentRecords != boilerplateRecords || attestation.BOM.Redaction != redaction || !slices.Equal(attestation.BOM.Licenses, licenses) {
			return fmt.Errorf("shard %s embedded BOM differs from its corpus pin", pin.SHA256[:12])
		}
	case "implicit-v4":
		if attestation.WriterRecipe != waldoshard.FormerTextRecipe || attestation.BOM != nil || attestation.BOMSHA256 != "" {
			return fmt.Errorf("shard %s has invalid implicit-v4 evidence", pin.SHA256[:12])
		}
	case "deep-validated":
		if attestation.BOM != nil || attestation.BOMSHA256 != "" {
			return fmt.Errorf("shard %s has invalid deep-validation evidence", pin.SHA256[:12])
		}
	default:
		return fmt.Errorf("shard %s has unsupported attestation status %q", pin.SHA256[:12], attestation.Status)
	}
	return nil
}

func validateSubManifestPins(pins []SubManifestPin, manifests map[string]ManifestPin) (map[string]bool, error) {
	seen := map[string]bool{}
	parents := map[string]string{}
	roots := map[string]int{}
	for _, pin := range pins {
		key := pin.Manifest + "\x00" + pin.SHA256
		if manifests[pin.Manifest].Path == "" || seen[key] || pin.URL == "" || !validSHA256(pin.SHA256) {
			return nil, fmt.Errorf("invalid or duplicate submanifest pin %s", shortHash(pin.SHA256))
		}
		if pin.ParentSHA256 != "" && !validSHA256(pin.ParentSHA256) {
			return nil, fmt.Errorf("submanifest %s has invalid parent hash", pin.SHA256[:12])
		}
		if pin.Count <= 0 || pin.Docs <= 0 || pin.Tokens < 0 || pin.Bytes <= 0 || pin.EncodedBytes <= 0 {
			return nil, fmt.Errorf("submanifest %s has non-positive totals", pin.SHA256[:12])
		}
		if err := index.ValidateModalities("submanifest "+pin.SHA256[:12], pin.Modalities); err != nil {
			return nil, err
		}
		if len(pin.Modalities) > 0 && modalityTokens(pin.Modalities) != pin.Tokens {
			return nil, fmt.Errorf("submanifest %s modality tokens do not match its token total", pin.SHA256[:12])
		}
		seen[key] = true
		if pin.ParentSHA256 == "" {
			roots[pin.Manifest]++
		} else {
			parents[key] = pin.Manifest + "\x00" + pin.ParentSHA256
		}
	}
	for _, pin := range pins {
		key := pin.Manifest + "\x00" + pin.SHA256
		if parent := parents[key]; parent != "" && !seen[parent] {
			return nil, fmt.Errorf("submanifest %s refers to an unpinned parent", pin.SHA256[:12])
		}
		chain := map[string]bool{}
		for current := key; current != ""; current = parents[current] {
			if chain[current] {
				return nil, fmt.Errorf("submanifest %s belongs to a parent cycle", pin.SHA256[:12])
			}
			chain[current] = true
		}
	}
	for manifest, count := range roots {
		if count != 1 {
			return nil, fmt.Errorf("manifest %s has %d submanifest roots, want one", manifest, count)
		}
	}
	return seen, nil
}

func addMeasure(target *index.Measures, value index.Measures) {
	target.Shards += value.Shards
	target.Docs += value.Docs
	target.Tokens += value.Tokens
	target.Bytes += value.Bytes
}

func addMeasureMap(target map[string]index.Measures, key string, value index.Measures) {
	measure := target[key]
	addMeasure(&measure, value)
	target[key] = measure
}

func validGitHash(value string) bool {
	return (len(value) == 40 || len(value) == 64) && strings.ToLower(value) == value && lowerHex(value)
}

func completeConversion(conversion index.Conversion, recordSchema int) bool {
	if conversion.Tool == "" || conversion.Version == "" || conversion.Profile == "" || conversion.Recipe == "" {
		return false
	}
	// Canonical text record schema 1 deliberately persists tokenizer-neutral
	// text. Legacy records must continue to identify the tokenizer whose counts
	// and representation they encode.
	return recordSchema == 1 || conversion.Tokenizer != ""
}

func lowerHex(value string) bool {
	for i := range len(value) {
		if c := value[i]; (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

func hasDuplicate(values []string) bool {
	for i := 1; i < len(values); i++ {
		if values[i] == values[i-1] {
			return true
		}
	}
	return false
}

func shortHash(value string) string {
	if len(value) >= 12 {
		return value[:12]
	}
	return value
}
