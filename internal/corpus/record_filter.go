// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package corpus

import (
	"fmt"
	"path"
	"regexp"
	"slices"
	"time"

	"github.com/openwaldo/waldo/internal/shard"
)

const RecordFilterSchema = 1

var (
	yearPattern      = regexp.MustCompile(`^\d{4}$`)
	yearMonthPattern = regexp.MustCompile(`^\d{4}-\d{2}$`)
	datePattern      = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
)

// RecordFilterPolicy is the immutable record-level selection contract pinned
// in a corpus BOM. A record must satisfy the global filter and its selected
// logical corpus filter.
type RecordFilterPolicy struct {
	Schema  int                     `json:"schema" yaml:"schema"`
	Global  *RecordFilter           `json:"global,omitempty" yaml:"global,omitempty"`
	Corpora map[string]RecordFilter `json:"corpora,omitempty" yaml:"corpora,omitempty"`
}

type RecordFilter struct {
	MainContent *bool            `json:"main_content,omitempty" yaml:"main_content,omitempty"`
	Exclude     *ExclusionFilter `json:"exclude,omitempty" yaml:"exclude,omitempty"`
	Licenses    *ValueFilter     `json:"licenses,omitempty" yaml:"licenses,omitempty"`
	Languages   *ValueFilter     `json:"languages,omitempty" yaml:"languages,omitempty"`
	Sources     *ValueFilter     `json:"sources,omitempty" yaml:"sources,omitempty"`
	Date        *DateFilter      `json:"date,omitempty" yaml:"date,omitempty"`
}

// ExclusionFilter is the canonical deny-list form. A record is excluded when
// any declared condition matches.
type ExclusionFilter struct {
	RepetitiveContent  *bool    `json:"repetitive_content,omitempty" yaml:"repetitive_content,omitempty"`
	BoilerplateContent *bool    `json:"boilerplate_content,omitempty" yaml:"boilerplate_content,omitempty"`
	Licenses           []string `json:"licenses,omitempty" yaml:"licenses,omitempty"`
}

type ValueFilter struct {
	Include      []string `json:"include,omitempty" yaml:"include,omitempty"`
	Exclude      []string `json:"exclude,omitempty" yaml:"exclude,omitempty"`
	IncludeUnset bool     `json:"include_unset,omitempty" yaml:"include_unset,omitempty"`
}

type DateFilter struct {
	From string `json:"from,omitempty" yaml:"from,omitempty"`
	To   string `json:"to,omitempty" yaml:"to,omitempty"`
}

func (policy RecordFilterPolicy) Validate(paths []string) error {
	if policy.Schema != RecordFilterSchema {
		return fmt.Errorf("unsupported record filter policy schema %d", policy.Schema)
	}
	if policy.Global != nil {
		if err := policy.Global.Validate(); err != nil {
			return fmt.Errorf("global record filter: %w", err)
		}
	}
	if policy.Global == nil && len(policy.Corpora) == 0 {
		return fmt.Errorf("record filter policy must declare a global or corpus filter")
	}
	selected := make(map[string]bool, len(paths))
	for _, corpusPath := range paths {
		selected[corpusPath] = true
	}
	for corpusPath, filter := range policy.Corpora {
		if !selected[corpusPath] {
			return fmt.Errorf("record filter declares unselected corpus %q", corpusPath)
		}
		if err := filter.Validate(); err != nil {
			return fmt.Errorf("corpus %s record filter: %w", corpusPath, err)
		}
	}
	return nil
}

func (filter RecordFilter) Validate() error {
	if filter.Exclude != nil {
		if err := filter.Exclude.Validate(); err != nil {
			return fmt.Errorf("exclude: %w", err)
		}
		if len(filter.Exclude.Licenses) > 0 && filter.Licenses != nil {
			return fmt.Errorf("exclude.licenses cannot be combined with legacy licenses filtering")
		}
	}
	for _, field := range []struct {
		name   string
		values *ValueFilter
	}{{"licenses", filter.Licenses}, {"languages", filter.Languages}, {"sources", filter.Sources}} {
		if field.values != nil {
			if err := field.values.Validate(); err != nil {
				return fmt.Errorf("%s: %w", field.name, err)
			}
			if field.name != "languages" && field.values.IncludeUnset {
				return fmt.Errorf("%s: include_unset is supported only for languages", field.name)
			}
		}
	}
	if filter.Date != nil {
		if err := filter.Date.Validate(); err != nil {
			return err
		}
	}
	if filter.empty() {
		return fmt.Errorf("record filter must declare at least one condition")
	}
	return nil
}

func (filter RecordFilter) empty() bool {
	return filter.MainContent == nil && filter.Exclude == nil && filter.Licenses == nil && filter.Languages == nil && filter.Sources == nil && filter.Date == nil
}

func (filter ExclusionFilter) Validate() error {
	if filter.RepetitiveContent == nil && filter.BoilerplateContent == nil && len(filter.Licenses) == 0 {
		return fmt.Errorf("at least one exclusion is required")
	}
	return validatePatterns(filter.Licenses)
}

func (filter ValueFilter) Validate() error {
	if len(filter.Include) == 0 && len(filter.Exclude) == 0 {
		return fmt.Errorf("include or exclude is required")
	}
	if filter.IncludeUnset && len(filter.Include) == 0 {
		return fmt.Errorf("include_unset requires include")
	}
	return validatePatterns(append(append([]string(nil), filter.Include...), filter.Exclude...))
}

func validatePatterns(patterns []string) error {
	seen := map[string]bool{}
	for _, pattern := range patterns {
		if pattern == "" {
			return fmt.Errorf("patterns must not be empty")
		}
		if _, err := path.Match(pattern, "probe"); err != nil {
			return fmt.Errorf("invalid pattern %q: %w", pattern, err)
		}
		if seen[pattern] {
			return fmt.Errorf("pattern %q appears more than once", pattern)
		}
		seen[pattern] = true
	}
	return nil
}

func (filter DateFilter) Validate() error {
	if filter.From == "" && filter.To == "" {
		return fmt.Errorf("date filter requires from or to")
	}
	var from, to time.Time
	if filter.From != "" {
		value, _, err := dateInterval(filter.From)
		if err != nil {
			return fmt.Errorf("date.from: %w", err)
		}
		from = value
	}
	if filter.To != "" {
		_, value, err := dateInterval(filter.To)
		if err != nil {
			return fmt.Errorf("date.to: %w", err)
		}
		to = value
	}
	if !from.IsZero() && !to.IsZero() && from.After(to) {
		return fmt.Errorf("date.from must not be after date.to")
	}
	return nil
}

func (policy RecordFilterPolicy) Allows(corpusPath string, record shard.RecordView) bool {
	if policy.Global != nil && !policy.Global.Allows(record) {
		return false
	}
	filter, exists := policy.Corpora[corpusPath]
	return !exists || filter.Allows(record)
}

func (filter RecordFilter) Allows(record shard.RecordView) bool {
	if filter.MainContent != nil && record.MainContent != *filter.MainContent {
		return false
	}
	if filter.Exclude != nil && filter.Exclude.Matches(record) {
		return false
	}
	if filter.Licenses != nil && !filter.Licenses.AllowsLicense(record.License) {
		return false
	}
	if filter.Languages != nil && !filter.Languages.Allows(record.Language) {
		return false
	}
	if filter.Sources != nil && !filter.Sources.AllowsAny(record.Source, record.SourceName) {
		return false
	}
	return filter.Date == nil || filter.Date.Allows(record.Date)
}

func (filter ExclusionFilter) Matches(record shard.RecordView) bool {
	if filter.RepetitiveContent != nil && record.RepetitiveContent != nil && *filter.RepetitiveContent == *record.RepetitiveContent {
		return true
	}
	if filter.BoilerplateContent != nil && record.BoilerplateContent != nil && *filter.BoilerplateContent == *record.BoilerplateContent {
		return true
	}
	return licenseExcluded(filter.Licenses, record.License)
}

// ContentAssessmentExclusions returns the assessment fields whose true or
// false value can exclude a record from the selected logical corpus. An empty
// result means the policy can be applied completely to an unassessed row.
func (policy RecordFilterPolicy) ContentAssessmentExclusions(corpusPath string) []string {
	fields := map[string]bool{}
	if policy.Global != nil {
		policy.Global.addContentAssessmentExclusions(fields)
	}
	if filter, exists := policy.Corpora[corpusPath]; exists {
		filter.addContentAssessmentExclusions(fields)
	}
	result := make([]string, 0, len(fields))
	for field := range fields {
		result = append(result, field)
	}
	slices.Sort(result)
	return result
}

func (filter RecordFilter) addContentAssessmentExclusions(fields map[string]bool) {
	if filter.Exclude == nil {
		return
	}
	if filter.Exclude.RepetitiveContent != nil {
		fields["repetitive_content"] = true
	}
	if filter.Exclude.BoilerplateContent != nil {
		fields["boilerplate_content"] = true
	}
}

func (filter ValueFilter) Allows(value string) bool {
	return filter.AllowsAny(value)
}

// AllowsLicense applies the filter to a license expression with the same
// term semantics as LicensePolicy: any excluded term rejects the record and
// includes must cover every term.
func (filter ValueFilter) AllowsLicense(license string) bool {
	if licenseExcluded(filter.Exclude, license) {
		return false
	}
	return len(filter.Include) == 0 || licenseIncluded(filter.Include, license)
}

func (filter ValueFilter) AllowsAny(values ...string) bool {
	for _, pattern := range filter.Exclude {
		for _, value := range values {
			if matched, _ := path.Match(pattern, value); matched {
				return false
			}
		}
	}
	if filter.IncludeUnset {
		for _, value := range values {
			if value == "" {
				return true
			}
		}
	}
	if len(filter.Include) == 0 {
		return true
	}
	for _, pattern := range filter.Include {
		for _, value := range values {
			if matched, _ := path.Match(pattern, value); matched {
				return true
			}
		}
	}
	return false
}

func (filter DateFilter) Allows(value string) bool {
	start, end, err := dateInterval(value)
	if err != nil {
		return false
	}
	if filter.From != "" {
		from, _, _ := dateInterval(filter.From)
		if end.Before(from) {
			return false
		}
	}
	if filter.To != "" {
		_, to, _ := dateInterval(filter.To)
		if start.After(to) {
			return false
		}
	}
	return true
}

func dateInterval(value string) (time.Time, time.Time, error) {
	var start time.Time
	var err error
	switch {
	case yearPattern.MatchString(value):
		start, err = time.Parse("2006", value)
		if err == nil {
			return start, start.AddDate(1, 0, 0).Add(-time.Nanosecond), nil
		}
	case yearMonthPattern.MatchString(value):
		start, err = time.Parse("2006-01", value)
		if err == nil {
			return start, start.AddDate(0, 1, 0).Add(-time.Nanosecond), nil
		}
	case datePattern.MatchString(value):
		start, err = time.Parse("2006-01-02", value)
		if err == nil {
			return start, start.AddDate(0, 0, 1).Add(-time.Nanosecond), nil
		}
	default:
		start, err = time.Parse(time.RFC3339Nano, value)
		if err == nil {
			return start, start, nil
		}
	}
	return time.Time{}, time.Time{}, fmt.Errorf("%q must be YYYY, YYYY-MM, YYYY-MM-DD, or RFC 3339", value)
}
