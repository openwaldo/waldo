// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package corpus

import (
	"fmt"
	"strings"
)

const DistributionPolicyDistributable = "distributable"

// DistributionReview is the pinned result of the conservative corpus license
// gate. It proves what WALDO checked; it is not legal advice.
type DistributionReview struct {
	Policy      string   `json:"policy"`
	Licenses    []string `json:"licenses"`
	Obligations []string `json:"obligations,omitempty"`
}

func ReviewDistributable(bom BOM) (DistributionReview, error) {
	review := DistributionReview{Policy: DistributionPolicyDistributable}
	licenses := map[string]bool{}
	obligations := map[string]bool{}
	for _, manifest := range bom.Manifests {
		selected := false
		manifestLicenses := manifest.Licenses
		if len(manifestLicenses) == 0 {
			manifestLicenses = bom.Licenses
		}
		for license := range manifestLicenses {
			if !DistributableLicense(license) || bom.RecordFilter != nil && !bom.RecordFilter.AllowsLicense(manifest.Path, license) {
				continue
			}
			selected = true
			licenses[license] = true
			if obligation, _ := distributableLicense(license); obligation != "" {
				obligations[obligation] = true
			}
		}
		if !selected {
			continue
		}
		for _, source := range manifest.Sources {
			if strings.TrimSpace(source.Version) == "" && !lowerSHA256(source.SHA256) {
				return DistributionReview{}, fmt.Errorf("manifest %s source %q has no pinned upstream version or source digest", manifest.Path, source.Name)
			}
			if source.LicenseEvidence == nil || strings.TrimSpace(source.LicenseEvidence.Declaration) == "" && strings.TrimSpace(source.LicenseEvidence.URL) == "" {
				return DistributionReview{}, fmt.Errorf("manifest %s source %q has no upstream license evidence", manifest.Path, source.Name)
			}
		}
	}
	review.Licenses = sortedKeys(licenses)
	review.Obligations = sortedKeys(obligations)
	if len(review.Licenses) == 0 {
		return DistributionReview{}, fmt.Errorf("distribution policy selects no records with an approved distributable license")
	}
	return review, nil
}

// DistributableLicense reports whether a row's normalized effective license is
// accepted by WALDO's conservative distributable policy.
func DistributableLicense(value string) bool {
	_, ok := distributableLicense(value)
	return ok
}

func lowerSHA256(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, character := range value {
		if character < '0' || character > '9' {
			if character < 'a' || character > 'f' {
				return false
			}
		}
	}
	return true
}

func distributableLicense(value string) (string, bool) {
	switch value {
	case "Apache-2.0":
		return "preserve Apache-2.0 license and NOTICE material", true
	case "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC":
		return "preserve copyright and license notices", true
	case "CC0-1.0", "PDDL-1.0", "Unlicense":
		return "", true
	case "CC-BY-2.0", "CC-BY-2.5", "CC-BY-3.0", "CC-BY-4.0", "ODC-BY-1.0":
		return "preserve source attribution", true
	case "CC-BY-SA-2.0", "CC-BY-SA-2.5", "CC-BY-SA-3.0", "CC-BY-SA-4.0", "ODbL-1.0":
		return "preserve source attribution and share-alike terms for redistributed corpus artifacts", true
	default:
		return "", false
	}
}
