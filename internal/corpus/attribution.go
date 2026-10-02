// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package corpus

import (
	"fmt"
	"sort"
	"strings"
)

// AttributionNotice renders the source and license evidence pinned by a set
// of distributable training BOMs. It is release metadata, not legal advice.
func AttributionNotice(boms []BOM) ([]byte, error) {
	if len(boms) == 0 {
		return []byte("# Training Data Attribution\n\nNo WALDO training corpus is recorded for this imported model. See the origin BOM for upstream model provenance.\n"), nil
	}
	licenses := map[string]bool{}
	obligations := map[string]bool{}
	manifests := map[string]ManifestPin{}
	for _, bom := range boms {
		review, err := ReviewDistributable(bom)
		if err != nil {
			return nil, err
		}
		for _, license := range review.Licenses {
			licenses[license] = true
		}
		for _, obligation := range review.Obligations {
			obligations[obligation] = true
		}
		for _, manifest := range bom.Manifests {
			manifests[manifest.SHA256] = manifest
		}
	}
	licenseNames := sortedKeys(licenses)
	obligationNames := sortedKeys(obligations)
	manifestValues := make([]ManifestPin, 0, len(manifests))
	for _, manifest := range manifests {
		manifestValues = append(manifestValues, manifest)
	}
	sort.Slice(manifestValues, func(i, j int) bool {
		if manifestValues[i].Path != manifestValues[j].Path {
			return manifestValues[i].Path < manifestValues[j].Path
		}
		return manifestValues[i].SHA256 < manifestValues[j].SHA256
	})
	var output strings.Builder
	output.WriteString("# Training Data Attribution\n\n")
	output.WriteString("This file is generated from the immutable corpus BOMs included with the model. It preserves declared source and license evidence and is not legal advice.\n\n")
	output.WriteString("## Licenses\n\n")
	for _, license := range licenseNames {
		fmt.Fprintf(&output, "- %s\n", license)
	}
	if len(obligationNames) > 0 {
		output.WriteString("\n## Recorded obligations\n\n")
		for _, obligation := range obligationNames {
			fmt.Fprintf(&output, "- %s\n", obligation)
		}
	}
	output.WriteString("\n## Sources\n")
	for _, manifest := range manifestValues {
		fmt.Fprintf(&output, "\n### %s\n\n", manifest.Title)
		fmt.Fprintf(&output, "- Index manifest: `%s`\n- Manifest SHA-256: `%s`\n- License: %s\n", manifest.Path, manifest.SHA256, manifest.License)
		for _, source := range manifest.Sources {
			fmt.Fprintf(&output, "- Source: %s — %s\n", source.Name, source.URL)
			if source.Version != "" {
				fmt.Fprintf(&output, "  - Version: `%s`\n", source.Version)
			}
			if source.SHA256 != "" {
				fmt.Fprintf(&output, "  - Source SHA-256: `%s`\n", source.SHA256)
			}
			if source.LicenseEvidence.Declaration != "" {
				fmt.Fprintf(&output, "  - Upstream license declaration: %s\n", source.LicenseEvidence.Declaration)
			}
			if source.LicenseEvidence.URL != "" {
				fmt.Fprintf(&output, "  - License evidence: %s\n", source.LicenseEvidence.URL)
			}
		}
	}
	return []byte(output.String()), nil
}

func sortedKeys(values map[string]bool) []string {
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}
