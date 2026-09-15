// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package corpus

import (
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/shard"
)

func TestDistributableReviewRequiresApprovedPinnedEvidence(t *testing.T) {
	mixedLicenses := map[string]index.Measures{"Apache-2.0": {Docs: 1}, "CC-BY-NC-4.0": {Docs: 1}, "LicenseRef-Unknown": {Docs: 1}}
	bom := BOM{
		Licenses: mixedLicenses,
		Manifests: []ManifestPin{{Path: "sft", Licenses: mixedLicenses, Sources: []index.Source{{
			Name: "upstream", Version: "commit-123", LicenseEvidence: &index.LicenseEvidence{URL: "https://example.test/license"},
		}}}},
	}
	review, err := ReviewDistributable(bom)
	if err != nil {
		t.Fatal(err)
	}
	if review.Policy != DistributionPolicyDistributable || len(review.Licenses) != 1 || review.Licenses[0] != "Apache-2.0" || len(review.Obligations) != 1 {
		t.Fatalf("review = %+v", review)
	}
	for name, mutate := range map[string]func(*BOM){
		"no distributable rows": func(value *BOM) {
			value.Licenses = map[string]index.Measures{"CC-BY-NC-4.0": {Docs: 1}, "LicenseRef-Unknown": {Docs: 1}}
			value.Manifests[0].Licenses = value.Licenses
		},
		"unversioned": func(value *BOM) { value.Manifests[0].Sources[0].Version = "" },
		"no evidence": func(value *BOM) { value.Manifests[0].Sources[0].LicenseEvidence = nil },
	} {
		t.Run(name, func(t *testing.T) {
			copy := bom
			copy.Manifests = append([]ManifestPin(nil), bom.Manifests...)
			copy.Manifests[0].Sources = append([]index.Source(nil), bom.Manifests[0].Sources...)
			mutate(&copy)
			if _, err := ReviewDistributable(copy); err == nil || !strings.Contains(err.Error(), map[string]string{"no distributable rows": "selects no records", "unversioned": "no pinned", "no evidence": "no upstream"}[name]) {
				t.Fatalf("review error = %v", err)
			}
		})
	}
}

func TestDistributableRecordPolicySkipsUnapprovedRows(t *testing.T) {
	policy := RecordFilterPolicy{Schema: RecordFilterSchema, Distributable: true}
	if !policy.Allows("mixed", shard.RecordView{License: "Apache-2.0"}) {
		t.Fatal("approved row was excluded")
	}
	for _, license := range []string{"CC-BY-NC-4.0", "LicenseRef-Public-Domain", "LicenseRef-Unknown"} {
		if policy.Allows("mixed", shard.RecordView{License: license}) {
			t.Fatalf("unapproved row %q was selected", license)
		}
	}
}
