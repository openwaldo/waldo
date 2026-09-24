// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package record

import "testing"

func TestNormalizeLicense(t *testing.T) {
	for input, wanted := range map[string]string{
		"https://creativecommons.org/licenses/by/4.0/":                                                 "CC-BY-4.0",
		"http://creativecommons.org/licenses/by-nc-sa/3.0":                                             "CC-BY-NC-SA-3.0",
		"https://creativecommons.org/publicdomain/zero/1.0/":                                           "CC0-1.0",
		"Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/":                "CC-BY-4.0",
		"Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/": "CC-BY-SA-4.0",
		"Creative Commons Zero - Public Domain - https://creativecommons.org/publicdomain/zero/1.0/":   "CC0-1.0",
		"Public Domain":       "LicenseRef-Public-Domain",
		"CC BY 4.0":           "CC-BY-4.0",
		"LicenseRef-Upstream": "LicenseRef-Upstream",
		"Apache 2 License - https://www.apache.org/licenses/LICENSE-2.0": "Apache-2.0",
		"BSD 2-Clause": "BSD-2-Clause",
		"BSD 3-Clause": "BSD-3-Clause",
		"Community Data License Agreement - Permissive 1.0 - https://cdla.dev/": "CDLA-Permissive-1.0",
		"ISC License": "ISC",
		"MIT License": "MIT",
		"GPL-3.0-only; docs: https://creativecommons.org/licenses/by/4.0/":                         "GPL-3.0-only; docs: https://creativecommons.org/licenses/by/4.0/",
		"https://creativecommons.org/licenses/by/4.0/ AND GPL-3.0-only":                            "https://creativecommons.org/licenses/by/4.0/ AND GPL-3.0-only",
		"Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/ or GPL-2.0": "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/ or GPL-2.0",
		"GPL-2.0 - https://creativecommons.org/licenses/by/4.0/":                                   "GPL-2.0 - https://creativecommons.org/licenses/by/4.0/",
	} {
		if got := NormalizeLicense(input); got != wanted {
			t.Errorf("NormalizeLicense(%q) = %q, want %q", input, got, wanted)
		}
	}
}

func TestNormalizeLicenseSetIsConservativeAndDeterministic(t *testing.T) {
	got := NormalizeLicenseSet([]string{
		"MIT License",
		"Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/",
		"MIT License",
	})
	want := "CC-BY-SA-4.0 AND MIT"
	if got != want {
		t.Fatalf("NormalizeLicenseSet() = %q, want %q", got, want)
	}
}
