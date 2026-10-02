// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"errors"
	"strings"
	"testing"
)

func TestHuggingFaceTokenizerPinValidation(t *testing.T) {
	base := HuggingFaceTokenizer{Source: "Qwen/Qwen3-0.6B", Files: map[string]string{"tokenizer.json": strings.Repeat("a", 64), "tokenizer_config.json": strings.Repeat("b", 64)}, PadID: 0, EOSID: 2}
	if err := base.Validate(strings.Repeat("c", 40), 259); err != nil {
		t.Fatal(err)
	}
	if base.Validate("main", 259) == nil {
		t.Fatal("mutable revision accepted")
	}
	if base.Validate(strings.Repeat("c", 40), 2) == nil {
		t.Fatal("out of range special ID accepted")
	}
	base.Files["../tokenizer.json"] = strings.Repeat("a", 64)
	if base.Validate(strings.Repeat("c", 40), 259) == nil {
		t.Fatal("path traversal accepted")
	}
}

type failedCodec struct{ byteCodec }

func (failedCodec) EncodeChecked(string) ([]int, error) {
	return nil, errors.New("tokenizer process failed")
}
func (failedCodec) CountChecked(string) (int, error) {
	return 0, errors.New("tokenizer process failed")
}

func TestTokenizerErrorsFailClosed(t *testing.T) {
	if _, _, err := tokenizeRecord(Record{Text: "hello"}, failedCodec{}, "causal-language-modeling", ConversationTransform{}); err == nil {
		t.Fatal("encoding failure became an empty record")
	}
	if _, err := countTokens(failedCodec{}, "hello"); err == nil {
		t.Fatal("encoding failure became a zero planning count")
	}
}
