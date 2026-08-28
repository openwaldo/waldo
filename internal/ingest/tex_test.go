// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package ingest

import "testing"

func TestIsLatexRoot(t *testing.T) {
	tests := []struct {
		name string
		data string
		want bool
	}{
		{"root with preamble", "\\documentclass{book}\n\\begin{document}\nhello\n\\end{document}", true},
		{"root minimal", "\\begin{document}hello\\end{document}", true},
		{"include chapter", "\\section{Intro}\nsome text", false},
		{"empty", "", false},
		{"near miss", "\\begindocument but not really", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isLatexRoot([]byte(tt.data)); got != tt.want {
				t.Errorf("isLatexRoot(%q) = %v, want %v", tt.data, got, tt.want)
			}
		})
	}
}
