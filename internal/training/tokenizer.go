// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"encoding/json"
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
	Name           string                   `json:"name"`
	Revision       string                   `json:"revision"`
	VocabularySize int                      `json:"vocabulary_size"`
	PadID          int                      `json:"pad_id"`
	BOSID          int                      `json:"bos_id"`
	EOSID          int                      `json:"eos_id"`
	Artifact       *waldoTokenizer.Artifact `json:"artifact,omitempty"`
}

func ResolveArchitectureTokenizer(raw json.RawMessage) (TokenizerSpec, TokenCodec, error) {
	var architecture struct {
		VocabularySize uint64 `json:"vocabulary_size"`
		Tokenizer      struct {
			Name     string                   `json:"name"`
			Revision string                   `json:"revision"`
			Artifact *waldoTokenizer.Artifact `json:"artifact,omitempty"`
		} `json:"tokenizer"`
	}
	if err := json.Unmarshal(raw, &architecture); err != nil {
		return TokenizerSpec{}, nil, err
	}
	spec := TokenizerSpec{Name: architecture.Tokenizer.Name, Revision: architecture.Tokenizer.Revision, VocabularySize: int(architecture.VocabularySize), Artifact: architecture.Tokenizer.Artifact}
	if spec.Artifact != nil {
		spec.PadID, spec.BOSID, spec.EOSID = spec.Artifact.PadID, spec.Artifact.BOSID, spec.Artifact.EOSID
	}
	return ResolveTokenizerSpec(spec)
}

// ValidateArchitectureTokenizer accepts a declared compose-time tokenizer
// training phase during forecasting. Real training requests must carry the
// resolved embedded artifact and continue through ResolveArchitectureTokenizer.
func ValidateArchitectureTokenizer(raw json.RawMessage) error {
	var architecture struct {
		Tokenizer struct {
			Artifact *waldoTokenizer.Artifact `json:"artifact,omitempty"`
			Training *struct {
				Algorithm string `json:"algorithm"`
			} `json:"training,omitempty"`
		} `json:"tokenizer"`
	}
	if err := json.Unmarshal(raw, &architecture); err != nil {
		return err
	}
	if architecture.Tokenizer.Artifact == nil && architecture.Tokenizer.Training != nil {
		if architecture.Tokenizer.Training.Algorithm != "byte-bpe-v1" {
			return fmt.Errorf("unsupported tokenizer training algorithm %q", architecture.Tokenizer.Training.Algorithm)
		}
		return nil
	}
	_, _, err := ResolveArchitectureTokenizer(raw)
	return err
}

// ResolveTokenizerSpec resolves both built-in tokenizers and a content-pinned
// trained tokenizer embedded in the portable model contract.
func ResolveTokenizerSpec(spec TokenizerSpec) (TokenizerSpec, TokenCodec, error) {
	if !waldoTokenizer.IsTrainedName(spec.Name) {
		if spec.Artifact != nil {
			return TokenizerSpec{}, nil, fmt.Errorf("built-in tokenizer %s cannot embed a trained artifact", spec.Name)
		}
		return ResolveTokenizer(spec.Name, spec.Revision, uint64(spec.VocabularySize))
	}
	if spec.Artifact == nil {
		return TokenizerSpec{}, nil, fmt.Errorf("trained tokenizer %s@%s requires its embedded artifact", spec.Name, spec.Revision)
	}
	artifact := *spec.Artifact
	if err := artifact.Validate(); err != nil {
		return TokenizerSpec{}, nil, err
	}
	if artifact.Name != spec.Name || artifact.Revision != spec.Revision || artifact.VocabularySize != spec.VocabularySize || artifact.PadID != spec.PadID || artifact.BOSID != spec.BOSID || artifact.EOSID != spec.EOSID {
		return TokenizerSpec{}, nil, fmt.Errorf("trained tokenizer specification does not match its artifact")
	}
	codec, err := artifact.Codec()
	if err != nil {
		return TokenizerSpec{}, nil, err
	}
	return spec, codec, nil
}

type TokenCodec interface {
	Count(string) int
	Encode(string) []int
	Decode([]int) string
}

type byteCodec struct{}

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
		return TokenizerSpec{Name: name, Revision: revision, VocabularySize: TiktokenCL100KVocabulary, PadID: TiktokenCL100KPadID, BOSID: TiktokenCL100KBOSID, EOSID: TiktokenCL100KEOSID}, codec, nil
	case name == "tiktoken/r50k_base" && revision == TiktokenR50KRevision && vocabularySize == TiktokenR50KVocabulary:
		codec, err := waldoTokenizer.NewCodec(name)
		if err != nil {
			return TokenizerSpec{}, nil, err
		}
		return TokenizerSpec{Name: name, Revision: revision, VocabularySize: TiktokenR50KVocabulary, PadID: TiktokenR50KPadID, BOSID: TiktokenR50KBOSID, EOSID: TiktokenR50KEOSID}, codec, nil
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
		return waldoTokenizer.NewCodec(name)
	}
	return nil, fmt.Errorf("unsupported tokenizer %q", name)
}
