// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"strings"
	"testing"

	waldotokenizer "github.com/openwaldo/waldo/internal/tokenizer"
)

func TestTokenizerCompressionGateBoundsTokenInflation(t *testing.T) {
	baseline := waldotokenizer.Comparison{Tokenizer: "r50k", Bytes: 400, Tokens: 100, BytesPerToken: 4}
	if err := validateTokenizerCompression([]waldotokenizer.Comparison{baseline, {Tokenizer: "candidate", Bytes: 400, Tokens: 110, BytesPerToken: 400.0 / 110}}, 0.10); err != nil {
		t.Fatal(err)
	}
	if err := validateTokenizerCompression([]waldotokenizer.Comparison{baseline, {Tokenizer: "candidate", Bytes: 400, Tokens: 90, BytesPerToken: 400.0 / 90}}, 0.10); err != nil {
		t.Fatal(err)
	}
	err := validateTokenizerCompression([]waldotokenizer.Comparison{baseline, {Tokenizer: "candidate", Bytes: 400, Tokens: 111, BytesPerToken: 400.0 / 111}}, 0.10)
	if err == nil || !strings.Contains(err.Error(), "more tokens than r50k") {
		t.Fatalf("token inflation error = %v", err)
	}
}
