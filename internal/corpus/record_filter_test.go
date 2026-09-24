// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package corpus

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/shard"
)

func TestRecordFilterCombinesGlobalAndCorpusConditions(t *testing.T) {
	policy := RecordFilterPolicy{
		Schema: RecordFilterSchema,
		Global: &RecordFilter{Licenses: &ValueFilter{Include: []string{"CC-*"}, Exclude: []string{"CC-BY-NC-*"}}},
		Corpora: map[string]RecordFilter{"books.yaml": {
			Languages: &ValueFilter{Include: []string{"en"}},
			Sources:   &ValueFilter{Include: []string{"project-*"}},
			Date:      &DateFilter{From: "2000", To: "2025-06"},
		}},
	}
	if err := policy.Validate([]string{"books.yaml"}); err != nil {
		t.Fatal(err)
	}
	allowed := shard.RecordView{License: "CC-BY-4.0", Language: "en", SourceName: "project-books", Date: "2024-02-03"}
	if !policy.Allows("books.yaml", allowed) {
		t.Fatal("matching record was rejected")
	}
	for name, record := range map[string]shard.RecordView{
		"license":  {License: "CC-BY-NC-4.0", Language: "en", SourceName: "project-books", Date: "2024"},
		"language": {License: "CC-BY-4.0", Language: "fr", SourceName: "project-books", Date: "2024"},
		"source":   {License: "CC-BY-4.0", Language: "en", SourceName: "other", Date: "2024"},
		"date":     {License: "CC-BY-4.0", Language: "en", SourceName: "project-books", Date: "1999"},
	} {
		t.Run(name, func(t *testing.T) {
			if policy.Allows("books.yaml", record) {
				t.Fatal("nonmatching record was allowed")
			}
		})
	}
}

func TestRecordFilterValidationAndBOMRoundTrip(t *testing.T) {
	policy := &RecordFilterPolicy{Schema: RecordFilterSchema, Global: &RecordFilter{Languages: &ValueFilter{Include: []string{"en"}, IncludeUnset: true}}}
	bom := BOM{Kind: "openwaldo-bom", Schema: 1, Subject: "corpus", Paths: []string{"books"}, RecordFilter: policy}
	data, err := json.Marshal(bom)
	if err != nil {
		t.Fatal(err)
	}
	var decoded BOM
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded.RecordFilter, policy) {
		t.Fatalf("record filter round trip = %+v", decoded.RecordFilter)
	}
	if err := (&RecordFilterPolicy{Schema: RecordFilterSchema}).Validate([]string{"books"}); err == nil || !strings.Contains(err.Error(), "must declare") {
		t.Fatalf("empty policy error = %v", err)
	}
	if err := (&RecordFilterPolicy{Schema: RecordFilterSchema, Corpora: map[string]RecordFilter{"other": {Languages: &ValueFilter{Include: []string{"en"}}}}}).Validate([]string{"books"}); err == nil || !strings.Contains(err.Error(), "unselected corpus") {
		t.Fatalf("unknown corpus error = %v", err)
	}
}

func TestLicenseValueFilterAppliesToEveryExpressionTerm(t *testing.T) {
	filter := RecordFilter{Licenses: &ValueFilter{Include: []string{"CC-*", "MIT"}, Exclude: []string{"CC-BY-NC-*"}}}
	if err := filter.Validate(); err != nil {
		t.Fatal(err)
	}
	for license, want := range map[string]bool{
		"CC-BY-4.0 AND MIT":          true,
		"CC-BY-NC-4.0 AND MIT":       false,
		"CC-BY-4.0 AND GPL-3.0-only": false,
		"MIT OR CC-BY-NC-SA-4.0":     false,
		"":                           false,
	} {
		if got := filter.Allows(shard.RecordView{License: license}); got != want {
			t.Errorf("Allows(%q) = %v, want %v", license, got, want)
		}
	}
}

func TestLanguageFilterCanExplicitlyIncludeUnsetRows(t *testing.T) {
	filter := RecordFilter{Languages: &ValueFilter{Include: []string{"en"}, IncludeUnset: true}}
	if err := filter.Validate(); err != nil {
		t.Fatal(err)
	}
	if !filter.Allows(shard.RecordView{Language: "en"}) || !filter.Allows(shard.RecordView{}) {
		t.Fatal("English or unset language was rejected")
	}
	if filter.Allows(shard.RecordView{Language: "fr"}) || filter.Allows(shard.RecordView{Language: "mul"}) {
		t.Fatal("known non-English language was allowed")
	}
	if err := (RecordFilter{Licenses: &ValueFilter{Include: []string{"CC-*"}, IncludeUnset: true}}).Validate(); err == nil || !strings.Contains(err.Error(), "only for languages") {
		t.Fatalf("non-language include_unset error = %v", err)
	}
	if err := (RecordFilter{Languages: &ValueFilter{IncludeUnset: true}}).Validate(); err == nil || !strings.Contains(err.Error(), "include or exclude") {
		t.Fatalf("unbounded include_unset error = %v", err)
	}
}

func TestUnifiedExclusionFilterMatchesContentFlagsOrLicenses(t *testing.T) {
	present, absent := true, false
	filter := RecordFilter{Exclude: &ExclusionFilter{
		RepetitiveContent: &present, BoilerplateContent: &present,
		Licenses: []string{"CC-BY-NC-*", "LicenseRef-Restricted-*"},
	}}
	if err := filter.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, record := range map[string]shard.RecordView{
		"repetitive":  {License: "CC0-1.0", EmailAddresses: &absent, RepetitiveContent: &present, BoilerplateContent: &absent},
		"boilerplate": {License: "CC0-1.0", EmailAddresses: &absent, RepetitiveContent: &absent, BoilerplateContent: &present},
		"license":     {License: "CC-BY-NC-4.0", EmailAddresses: &absent, RepetitiveContent: &absent, BoilerplateContent: &absent},
		"restricted":  {License: "LicenseRef-Restricted-Example", EmailAddresses: &absent, RepetitiveContent: &absent, BoilerplateContent: &absent},
	} {
		t.Run(name, func(t *testing.T) {
			if filter.Allows(record) {
				t.Fatal("excluded record was allowed")
			}
		})
	}
	if !filter.Allows(shard.RecordView{License: "CC0-1.0", EmailAddresses: &absent, RepetitiveContent: &absent, BoilerplateContent: &absent}) {
		t.Fatal("clean record was excluded")
	}
	if !filter.Allows(shard.RecordView{License: "CC0-1.0"}) {
		t.Fatal("unassessed record should not match a boolean exclusion")
	}
	if filter.Allows(shard.RecordView{License: "Apache-2.0 AND CC-BY-NC-4.0", EmailAddresses: &absent, RepetitiveContent: &absent, BoilerplateContent: &absent}) {
		t.Fatal("compound license with an excluded term was allowed")
	}
	legacy := RecordFilter{Exclude: &ExclusionFilter{Licenses: []string{"CC-*"}}, Licenses: &ValueFilter{Exclude: []string{"CC0-*"}}}
	if err := legacy.Validate(); err == nil || !strings.Contains(err.Error(), "cannot be combined") {
		t.Fatalf("mixed syntax error = %v", err)
	}
}

func TestMainContentFilterIsDirectAndDefaultsOlderRowsToMain(t *testing.T) {
	want := true
	filter := RecordFilter{MainContent: &want}
	if err := filter.Validate(); err != nil {
		t.Fatal(err)
	}
	if !filter.Allows(shard.RecordView{MainContent: true}) || filter.Allows(shard.RecordView{MainContent: false}) {
		t.Fatal("main-content filter did not require the declared boolean")
	}
}

func TestContentAssessmentExclusionsCombineGlobalAndCorpusPolicy(t *testing.T) {
	present := true
	policy := RecordFilterPolicy{
		Global: &RecordFilter{Exclude: &ExclusionFilter{BoilerplateContent: &present}},
		Corpora: map[string]RecordFilter{
			"books": {Exclude: &ExclusionFilter{RepetitiveContent: &present}},
		},
	}
	if got := policy.ContentAssessmentExclusions("books"); !reflect.DeepEqual(got, []string{"boilerplate_content", "repetitive_content"}) {
		t.Fatalf("books assessment exclusions = %v", got)
	}
	if got := policy.ContentAssessmentExclusions("science"); !reflect.DeepEqual(got, []string{"boilerplate_content"}) {
		t.Fatalf("science assessment exclusions = %v", got)
	}
}
