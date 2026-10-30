// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package corpus

import (
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/index"
)

func TestAttributionNoticeIsDeterministicAndRequiresDistributableEvidence(t *testing.T) {
	manifest := ManifestPin{Path: "core/example", SHA256: strings.Repeat("a", 64), Title: "Example", License: "CC-BY-4.0", Sources: []index.Source{{Name: "Upstream", URL: "https://example.test/data", Version: "v1", LicenseEvidence: &index.LicenseEvidence{Declaration: "CC-BY-4.0"}}}}
	bom := BOM{Licenses: map[string]index.Measures{"CC-BY-4.0": {}}, Manifests: []ManifestPin{manifest}}
	first, err := AttributionNotice([]BOM{bom, bom})
	if err != nil {
		t.Fatal(err)
	}
	text := string(first)
	if strings.Count(text, "### Example") != 1 || !strings.Contains(text, "preserve source attribution") || !strings.Contains(text, "https://example.test/data") {
		t.Fatalf("notice = %s", text)
	}
	bom.Licenses = map[string]index.Measures{"CC-BY-NC-4.0": {}}
	if _, err := AttributionNotice([]BOM{bom}); err == nil {
		t.Fatal("non-distributable attribution input accepted")
	}
}

func TestAttributionNoticeToleratesUnselectedManifestsWithoutEvidence(t *testing.T) {
	trained := ManifestPin{Path: "core/example", SHA256: strings.Repeat("a", 64), Title: "Example", License: "CC-BY-4.0",
		Licenses: map[string]index.Measures{"CC-BY-4.0": {}},
		Sources:  []index.Source{{Name: "Upstream", URL: "https://example.test/data", Version: "v1", LicenseEvidence: &index.LicenseEvidence{Declaration: "CC-BY-4.0"}}}}
	// A public-domain manifest isn't on the distributable list, so none of its records train and it needs no evidence.
	unselected := ManifestPin{Path: "core/peps", SHA256: strings.Repeat("b", 64), Title: "PEPs", License: "LicenseRef-Public-Domain",
		Licenses: map[string]index.Measures{"LicenseRef-Public-Domain": {}},
		Sources:  []index.Source{{Name: "peps", URL: "https://example.test/peps", Version: "v1"}}}
	notice, err := AttributionNotice([]BOM{{Manifests: []ManifestPin{trained, unselected}}})
	if err != nil || !strings.Contains(string(notice), "https://example.test/peps") {
		t.Fatalf("notice = %s, %v", notice, err)
	}
}
