// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"fmt"

	waldoTokenizer "github.com/openwaldo/waldo/internal/tokenizer"
)

const (
	ByteTokenizerRevision    = "builtin-byte-schema-1"
	TiktokenCL100KRevision   = "tiktoken-cl100k-base"
	TiktokenCL100KVocabulary = 100259
	TiktokenCL100KPadID      = 100256
	TiktokenCL100KBOSID      = 100257
	TiktokenCL100KEOSID      = 100258
	TiktokenR50KRevision     = "tiktoken-r50k-base"
	TiktokenR50KVocabulary   = 50259
	TiktokenR50KPadID        = 50256
	TiktokenR50KBOSID        = 50257
	TiktokenR50KEOSID        = 50258
)

type TokenizerSpec struct {
	Name           string                `json:"name"`
	Revision       string                `json:"revision"`
	VocabularySize int                   `json:"vocabulary_size"`
	PadID          int                   `json:"pad_id"`
	BOSID          int                   `json:"bos_id"`
	EOSID          int                   `json:"eos_id"`
	HuggingFace    *HuggingFaceTokenizer `json:"huggingface,omitempty"`
}

type TokenCodec interface {
	DecodeChecked([]int) (string, error)
	CountChecked(string) (int, error)
	EncodeChecked(string) ([]int, error)
}

type localTokenCodec interface {
	Count(string) int
	Encode(string) []int
	Decode([]int) string
}

// Local codecs adapt to the fallible training interface without changing their
// native API. Interpreter failures must never become empty training records.
type localCodecAdapter struct{ localTokenCodec }

func (codec localCodecAdapter) DecodeChecked(tokens []int) (string, error) {
	return codec.Decode(tokens), nil
}

func (codec localCodecAdapter) EncodeChecked(text string) ([]int, error) {
	return codec.Encode(text), nil
}
func (codec localCodecAdapter) CountChecked(text string) (int, error) { return codec.Count(text), nil }

func encodeTokens(codec TokenCodec, text string) ([]int, error) {
	return codec.EncodeChecked(text)
}

func countTokens(codec TokenCodec, text string) (int, error) {
	return codec.CountChecked(text)
}

type byteCodec struct{}

func (codec byteCodec) DecodeChecked(tokens []int) (string, error) { return codec.Decode(tokens), nil }

func (codec byteCodec) EncodeChecked(text string) ([]int, error) { return codec.Encode(text), nil }
func (codec byteCodec) CountChecked(text string) (int, error)    { return codec.Count(text), nil }

func (byteCodec) Count(text string) int { return len([]byte(text)) }
func (byteCodec) Encode(text string) []int {
	encoded := make([]int, len([]byte(text)))
	for index, value := range []byte(text) {
		encoded[index] = int(value) + 3
	}
	return encoded
}
func (byteCodec) Decode(tokens []int) string {
	decoded := make([]byte, 0, len(tokens))
	for _, token := range tokens {
		if token >= 3 && token <= 258 {
			decoded = append(decoded, byte(token-3))
		}
	}
	return string(decoded)
}

func ResolveTokenizer(name, revision string, vocabularySize uint64) (TokenizerSpec, TokenCodec, error) {
	switch {
	case name == "byte" && revision == ByteTokenizerRevision && vocabularySize == 259:
		return TokenizerSpec{Name: name, Revision: revision, VocabularySize: 259, PadID: 0, BOSID: 1, EOSID: 2}, byteCodec{}, nil
	case name == waldoTokenizer.Default && revision == TiktokenCL100KRevision && vocabularySize == TiktokenCL100KVocabulary:
		codec, err := waldoTokenizer.NewCodec(name)
		if err != nil {
			return TokenizerSpec{}, nil, err
		}
		return TokenizerSpec{Name: name, Revision: revision, VocabularySize: TiktokenCL100KVocabulary, PadID: TiktokenCL100KPadID, BOSID: TiktokenCL100KBOSID, EOSID: TiktokenCL100KEOSID}, localCodecAdapter{codec}, nil
	case name == "tiktoken/r50k_base" && revision == TiktokenR50KRevision && vocabularySize == TiktokenR50KVocabulary:
		codec, err := waldoTokenizer.NewCodec(name)
		if err != nil {
			return TokenizerSpec{}, nil, err
		}
		return TokenizerSpec{Name: name, Revision: revision, VocabularySize: TiktokenR50KVocabulary, PadID: TiktokenR50KPadID, BOSID: TiktokenR50KBOSID, EOSID: TiktokenR50KEOSID}, localCodecAdapter{codec}, nil
	default:
		return TokenizerSpec{}, nil, fmt.Errorf("unsupported tokenizer %s@%s with vocabulary_size %d", name, revision, vocabularySize)
	}
}

// ResolveTokenizerCodec resolves tokenization semantics before backend
// selection. Legacy test and imported architectures may carry opaque byte
// tokenizer revisions; real backends still use ResolveTokenizer to fail closed
// on the exact executable artifact contract.
func ResolveTokenizerCodec(name string) (TokenCodec, error) {
	if name == "byte" {
		return byteCodec{}, nil
	}
	if name == waldoTokenizer.Default || name == "tiktoken/r50k_base" {
		codec, err := waldoTokenizer.NewCodec(name)
		if err != nil {
			return nil, err
		}
		return localCodecAdapter{codec}, nil
	}
	return nil, fmt.Errorf("unsupported tokenizer %q", name)
}
