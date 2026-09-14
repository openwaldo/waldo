// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package corpus

import (
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/index"
)

func TestDistributableReviewRequiresApprovedPinnedEvidence(t *testing.T) {
	bom := BOM{
		Licenses: map[string]index.Measures{"Apache-2.0": {Docs: 1}},
		Manifests: []ManifestPin{{Path: "sft", Sources: []index.Source{{
			Name: "upstream", Version: "commit-123", LicenseEvidence: &index.LicenseEvidence{URL: "https://example.test/license"},
		}}}},
	}
	review, err := ReviewDistributable(bom)
	if err != nil {
		t.Fatal(err)
	}
	if review.Policy != DistributionPolicyDistributable || len(review.Obligations) != 1 {
		t.Fatalf("review = %+v", review)
	}
	for name, mutate := range map[string]func(*BOM){
		"noncommercial": func(value *BOM) { value.Licenses = map[string]index.Measures{"CC-BY-NC-4.0": {Docs: 1}} },
		"unknown":       func(value *BOM) { value.Licenses = map[string]index.Measures{"LicenseRef-Unknown": {Docs: 1}} },
		"unversioned":   func(value *BOM) { value.Manifests[0].Sources[0].Version = "" },
		"no evidence":   func(value *BOM) { value.Manifests[0].Sources[0].LicenseEvidence = nil },
	} {
		t.Run(name, func(t *testing.T) {
			copy := bom
			copy.Manifests = append([]ManifestPin(nil), bom.Manifests...)
			copy.Manifests[0].Sources = append([]index.Source(nil), bom.Manifests[0].Sources...)
			mutate(&copy)
			if _, err := ReviewDistributable(copy); err == nil || !strings.Contains(err.Error(), map[string]string{"noncommercial": "not approved", "unknown": "not approved", "unversioned": "no pinned", "no evidence": "no upstream"}[name]) {
				t.Fatalf("review error = %v", err)
			}
		})
	}
}
