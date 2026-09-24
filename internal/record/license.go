// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package record

import (
	"regexp"
	"sort"
	"strings"
)

// creativeCommonsURL matches a whole declaration that is only a Creative
// Commons URL, optionally preceded by a descriptive Creative Commons label.
// It is anchored so text that merely mentions a CC URL next to other terms is
// preserved verbatim instead of collapsing into a single CC identifier.
var creativeCommonsURL = regexp.MustCompile(`(?i)^(?:Creative\s+Commons(?:(?:\s*-\s*|\s+)(?:Attribution|Share-?Alike|Non-?Commercial|No-?Deriv(?:ative)?s|Zero|Public|Domain|Universal|Dedication|International|Licen[cs]e|[0-9]+(?:\.[0-9]+)*))*\s*-\s*)?https?://creativecommons\.org/(?:licenses/([a-z-]+)/([0-9.]+)|publicdomain/zero/([0-9.]+))/?(?:legalcode/?)?(?:[?#][^[:space:]]*)?$`)
var creativeCommonsName = regexp.MustCompile(`(?i)^CC[ _-]*(BY(?:[ _-]*(?:NC|ND|SA)){0,2}|ZERO|0)[ _-]*([0-9]+(?:\.[0-9]+)?)$`)

// NormalizeLicense canonicalizes recognized Creative Commons identifiers.
// Unknown expressions are preserved verbatim after surrounding whitespace is
// removed; WALDO does not invent an identity for upstream license evidence.
func NormalizeLicense(raw string) string {
	value := strings.TrimSpace(raw)
	for name, normalized := range map[string]string{
		"Apache 2 License - https://www.apache.org/licenses/LICENSE-2.0": "Apache-2.0",
		"BSD 2-Clause": "BSD-2-Clause",
		"BSD 3-Clause": "BSD-3-Clause",
		"Community Data License Agreement - Permissive 1.0 - https://cdla.dev/": "CDLA-Permissive-1.0",
		"ISC License": "ISC",
		"MIT License": "MIT",
	} {
		if strings.EqualFold(value, name) {
			return normalized
		}
	}
	if match := creativeCommonsURL.FindStringSubmatch(value); match != nil {
		if match[3] != "" {
			return "CC0-" + match[3]
		}
		return "CC-" + strings.ToUpper(match[1]) + "-" + match[2]
	}
	if strings.EqualFold(value, "Public Domain") {
		return "LicenseRef-Public-Domain"
	}
	if match := creativeCommonsName.FindStringSubmatch(value); match != nil {
		name := strings.ToUpper(strings.NewReplacer(" ", "-", "_", "-").Replace(match[1]))
		name = strings.ReplaceAll(name, "ZERO", "0")
		if name == "0" {
			return "CC0-" + match[2]
		}
		return "CC-" + name + "-" + match[2]
	}
	return value
}

// NormalizeLicenseSet conservatively represents simultaneous upstream license
// declarations. Each term is normalized independently, duplicates are removed,
// and multiple terms are joined with deterministic AND semantics. An array is
// never collapsed to a more permissive single term.
func NormalizeLicenseSet(raw []string) string {
	terms := make([]string, 0, len(raw))
	seen := map[string]bool{}
	for _, value := range raw {
		term := NormalizeLicense(value)
		if term == "" || seen[term] {
			continue
		}
		seen[term] = true
		terms = append(terms, term)
	}
	sort.Strings(terms)
	return strings.Join(terms, " AND ")
}
