// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package corpus resolves indexed manifests into immutable, verified selections
// consumed by export and model workflows.
package corpus

import (
	"fmt"
	"path"
	"regexp"
	"strings"
)

// licenseOperator separates the terms of an SPDX-style license expression,
// such as the "A AND B" form produced for multi-license records.
var licenseOperator = regexp.MustCompile(`(?i)\s+(?:AND|OR)\s+`)

type LicensePolicy struct {
	Include []string `json:"include,omitempty" yaml:"include,omitempty"`
	Exclude []string `json:"exclude,omitempty" yaml:"exclude,omitempty"`
}

func NewLicensePolicy(include, exclude []string) (LicensePolicy, error) {
	policy := LicensePolicy{Include: compact(include), Exclude: compact(exclude)}
	for _, pattern := range append(append([]string{}, policy.Include...), policy.Exclude...) {
		if _, err := path.Match(pattern, "probe"); err != nil {
			return LicensePolicy{}, fmt.Errorf("invalid license pattern %q: %w", pattern, err)
		}
	}
	return policy, nil
}

// Allows reports whether a license expression satisfies the policy. An
// exclude pattern rejects the expression when it matches the whole expression
// or any of its terms. Include patterns must cover every term, unless an
// include pattern is itself a compound expression matching the whole value.
func (p LicensePolicy) Allows(license string) bool {
	if licenseExcluded(p.Exclude, license) {
		return false
	}
	return len(p.Include) == 0 || licenseIncluded(p.Include, license)
}

// licenseExcluded reports whether any pattern matches the whole license
// expression or one of its terms, so a restricted term cannot hide behind a
// more permissive one.
func licenseExcluded(patterns []string, license string) bool {
	terms := licenseTerms(license)
	for _, pattern := range patterns {
		if matchLicense(pattern, license) {
			return true
		}
		for _, term := range terms {
			if matchLicense(pattern, term) {
				return true
			}
		}
	}
	return false
}

// licenseIncluded reports whether every term of the license expression is
// matched by some pattern. A pattern matches the whole expression only when
// the expression is a single term or the pattern is itself a compound
// expression; otherwise a wildcard such as "CC-*" would span an operator and
// admit unlisted terms. OR terms are treated like AND terms: an alternative
// that is not included keeps the expression out.
func licenseIncluded(patterns []string, license string) bool {
	terms := licenseTerms(license)
	for _, pattern := range patterns {
		if (len(terms) <= 1 || licenseOperator.MatchString(pattern)) && matchLicense(pattern, license) {
			return true
		}
	}
	if len(terms) == 0 {
		return false
	}
	for _, term := range terms {
		included := false
		for _, pattern := range patterns {
			if matchLicense(pattern, term) {
				included = true
				break
			}
		}
		if !included {
			return false
		}
	}
	return true
}

// licenseTerms splits a license expression on AND/OR operators. Grouping
// parentheses are dropped and WITH exceptions stay attached to their license.
func licenseTerms(license string) []string {
	parts := licenseOperator.Split(license, -1)
	terms := make([]string, 0, len(parts))
	for _, part := range parts {
		term := strings.TrimSpace(strings.Trim(strings.TrimSpace(part), "()"))
		if term != "" {
			terms = append(terms, term)
		}
	}
	return terms
}

// matchLicense applies a shell-style pattern in which '*' and '?' also match
// '/', because license declarations preserved verbatim may contain URLs.
func matchLicense(pattern, value string) bool {
	matched, _ := path.Match(strings.ReplaceAll(pattern, "/", "\x00"), strings.ReplaceAll(value, "/", "\x00"))
	return matched
}

func compact(values []string) []string {
	result := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		result = append(result, value)
	}
	return result
}
