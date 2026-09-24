// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package corpus

import "testing"

func TestLicensePolicy(t *testing.T) {
	policy, err := NewLicensePolicy([]string{"CC-*", "Apache-2.0"}, []string{"CC-BY-NC-*"})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		license string
		want    bool
	}{
		{"CC-BY-4.0", true},
		{"Apache-2.0", true},
		{"CC-BY-NC-4.0", false},
		{"MIT", false},
	}
	for _, test := range tests {
		if got := policy.Allows(test.license); got != test.want {
			t.Errorf("Allows(%q) = %v, want %v", test.license, got, test.want)
		}
	}
}

func TestLicensePolicyAppliesToEveryExpressionTerm(t *testing.T) {
	policy, err := NewLicensePolicy([]string{"CC-*", "Apache-2.0", "MIT"}, []string{"CC-BY-NC-*", "*NC*"})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		license string
		want    bool
	}{
		{"Apache-2.0 AND CC-BY-NC-4.0", false},
		{"CC-BY-NC-4.0 AND MIT", false},
		{"(Apache-2.0 OR CC-BY-NC-SA-4.0)", false},
		{"Apache-2.0 and CC-BY-NC-4.0", false},
		{"https://example.org/licenses/NC/1.0", false},
		{"Apache-2.0 AND CC-BY-4.0", true},
		{"CC-BY-SA-4.0 AND MIT", true},
		{"Apache-2.0 AND GPL-3.0-only", false},
		{"CC-BY-4.0 AND GPL-3.0-only", false},
		{"MIT OR GPL-3.0-only", false},
		{"", false},
	}
	for _, test := range tests {
		if got := policy.Allows(test.license); got != test.want {
			t.Errorf("Allows(%q) = %v, want %v", test.license, got, test.want)
		}
	}
	exact, err := NewLicensePolicy([]string{"Apache-2.0 AND GPL-3.0-only"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !exact.Allows("Apache-2.0 AND GPL-3.0-only") {
		t.Error("include pattern matching the whole expression was rejected")
	}
}

func TestLicensePolicyRejectsInvalidGlob(t *testing.T) {
	if _, err := NewLicensePolicy([]string{"["}, nil); err == nil {
		t.Fatal("NewLicensePolicy() accepted invalid glob")
	}
}
