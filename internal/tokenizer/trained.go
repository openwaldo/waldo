// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"unicode"
	"unicode/utf8"
)

const (
	TrainedKind = "waldo-trained-tokenizer"
	TrainedName = "waldo/bytepiece"
)

type Sample struct {
	ID   string
	Text string
}

type Artifact struct {
	Kind                string   `json:"kind"`
	Schema              int      `json:"schema"`
	Name                string   `json:"name"`
	Revision            string   `json:"revision"`
	VocabularySize      int      `json:"vocabulary_size"`
	PadID               int      `json:"pad_id"`
	BOSID               int      `json:"bos_id"`
	EOSID               int      `json:"eos_id"`
	TrainingInputSHA256 string   `json:"training_input_sha256"`
	CorpusBOMSHA256     string   `json:"corpus_bom_sha256"`
	TrainingBytes       int64    `json:"training_bytes"`
	Pieces              []string `json:"pieces"`
}

type Comparison struct {
	Tokenizer     string  `json:"tokenizer"`
	Bytes         int64   `json:"bytes"`
	Tokens        int64   `json:"tokens"`
	BytesPerToken float64 `json:"bytes_per_token"`
}

// TrainBytepiece builds a deterministic byte-fallback vocabulary from common
// UTF-8 word, punctuation, and whitespace runs. It makes one bounded pass over
// the admitted sample rather than performing vocabulary-size full-corpus BPE
// rescans.
func TrainBytepiece(samples []Sample, vocabularySize int, maxBytes int64, corpusBOMSHA256 string) (Artifact, error) {
	if vocabularySize < 259 || vocabularySize > 100_000 {
		return Artifact{}, fmt.Errorf("tokenizer vocabulary_size must be in 259..100000")
	}
	if maxBytes < 1 {
		return Artifact{}, fmt.Errorf("tokenizer sample byte limit must be positive")
	}
	if len(corpusBOMSHA256) != 64 {
		return Artifact{}, fmt.Errorf("tokenizer training requires a corpus BOM SHA-256")
	}
	samples = append([]Sample(nil), samples...)
	sort.Slice(samples, func(i, j int) bool { return samples[i].ID < samples[j].ID })
	seen := map[string]bool{}
	counts := map[string]int64{}
	inputHash := sha256.New()
	var used int64
	for _, sample := range samples {
		if sample.ID == "" || seen[sample.ID] {
			return Artifact{}, fmt.Errorf("tokenizer samples require unique non-empty IDs")
		}
		seen[sample.ID] = true
		remaining := maxBytes - used
		if remaining <= 0 {
			break
		}
		text := []byte(sample.Text)
		if int64(len(text)) > remaining {
			text = text[:remaining]
			for !utf8.Valid(text) && len(text) > 0 {
				text = text[:len(text)-1]
			}
		}
		inputHash.Write([]byte(sample.ID))
		inputHash.Write([]byte{0})
		inputHash.Write(text)
		inputHash.Write([]byte{0})
		used += int64(len(text))
		for _, piece := range lexicalPieces(string(text)) {
			if len(piece) > 1 && len(piece) <= 64 {
				counts[piece]++
			}
		}
	}
	type candidate struct {
		piece string
		count int64
	}
	values := make([]candidate, 0, len(counts))
	for piece, count := range counts {
		values = append(values, candidate{piece, count})
	}
	sort.Slice(values, func(i, j int) bool {
		if values[i].count == values[j].count {
			if len(values[i].piece) == len(values[j].piece) {
				return values[i].piece < values[j].piece
			}
			return len(values[i].piece) > len(values[j].piece)
		}
		return values[i].count > values[j].count
	})
	limit := vocabularySize - 259
	if limit > len(values) {
		limit = len(values)
	}
	pieces := make([]string, limit)
	for index := 0; index < limit; index++ {
		pieces[index] = base64.StdEncoding.EncodeToString([]byte(values[index].piece))
	}
	artifact := Artifact{Kind: TrainedKind, Schema: 1, Name: TrainedName, VocabularySize: 259 + len(pieces), PadID: 0, BOSID: 1, EOSID: 2, TrainingInputSHA256: hex.EncodeToString(inputHash.Sum(nil)), CorpusBOMSHA256: corpusBOMSHA256, TrainingBytes: used, Pieces: pieces}
	revision, err := artifactRevision(artifact)
	if err != nil {
		return Artifact{}, err
	}
	artifact.Revision = revision
	return artifact, artifact.Validate()
}

func (artifact Artifact) Validate() error {
	if artifact.Kind != TrainedKind || artifact.Schema != 1 || artifact.Name != TrainedName || artifact.PadID != 0 || artifact.BOSID != 1 || artifact.EOSID != 2 || artifact.VocabularySize != 259+len(artifact.Pieces) || len(artifact.TrainingInputSHA256) != 64 || len(artifact.CorpusBOMSHA256) != 64 || artifact.TrainingBytes < 1 {
		return fmt.Errorf("invalid trained tokenizer identity")
	}
	want, err := artifactRevision(artifact)
	if err != nil || artifact.Revision != want {
		return fmt.Errorf("trained tokenizer revision differs")
	}
	seen := map[string]bool{}
	for _, encoded := range artifact.Pieces {
		piece, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || len(piece) < 2 || len(piece) > 64 || seen[string(piece)] {
			return fmt.Errorf("invalid trained tokenizer piece")
		}
		seen[string(piece)] = true
	}
	return nil
}

func (artifact Artifact) Codec() (Codec, error) {
	if err := artifact.Validate(); err != nil {
		return nil, err
	}
	codec := &trainedCodec{name: artifact.Name + "@" + artifact.Revision, byFirst: map[byte][]trainedPiece{}}
	for index, encoded := range artifact.Pieces {
		piece, _ := base64.StdEncoding.DecodeString(encoded)
		codec.byFirst[piece[0]] = append(codec.byFirst[piece[0]], trainedPiece{bytes: piece, id: 259 + index})
	}
	for first := range codec.byFirst {
		sort.Slice(codec.byFirst[first], func(i, j int) bool { return len(codec.byFirst[first][i].bytes) > len(codec.byFirst[first][j].bytes) })
	}
	codec.pieces = artifact.Pieces
	return codec, nil
}

func Compare(codec Codec, samples []Sample) Comparison {
	result := Comparison{Tokenizer: codec.Name()}
	for _, sample := range samples {
		result.Bytes += int64(len([]byte(sample.Text)))
		result.Tokens += int64(codec.Count(sample.Text))
	}
	if result.Tokens > 0 {
		result.BytesPerToken = float64(result.Bytes) / float64(result.Tokens)
	}
	return result
}

type trainedPiece struct {
	bytes []byte
	id    int
}

type trainedCodec struct {
	name    string
	pieces  []string
	byFirst map[byte][]trainedPiece
}

func (codec *trainedCodec) Name() string          { return codec.name }
func (codec *trainedCodec) Count(text string) int { return len(codec.Encode(text)) }
func (codec *trainedCodec) Encode(text string) []int {
	input := []byte(text)
	result := make([]int, 0, len(input))
	for len(input) > 0 {
		matched := false
		for _, piece := range codec.byFirst[input[0]] {
			if bytes.HasPrefix(input, piece.bytes) {
				result = append(result, piece.id)
				input = input[len(piece.bytes):]
				matched = true
				break
			}
		}
		if !matched {
			result = append(result, int(input[0])+3)
			input = input[1:]
		}
	}
	return result
}
func (codec *trainedCodec) Decode(tokens []int) string {
	var output []byte
	for _, token := range tokens {
		if token >= 3 && token <= 258 {
			output = append(output, byte(token-3))
		} else if token >= 259 && token-259 < len(codec.pieces) {
			piece, _ := base64.StdEncoding.DecodeString(codec.pieces[token-259])
			output = append(output, piece...)
		}
	}
	return string(output)
}

func lexicalPieces(text string) []string {
	var pieces []string
	start := 0
	kind := -1
	for index, value := range text {
		next := 2
		if unicode.IsLetter(value) || unicode.IsNumber(value) || value == '_' {
			next = 0
		} else if unicode.IsSpace(value) {
			next = 1
		}
		if kind != -1 && next != kind {
			pieces = append(pieces, text[start:index])
			start = index
		}
		kind = next
	}
	if start < len(text) {
		pieces = append(pieces, text[start:])
	}
	return pieces
}

func artifactRevision(artifact Artifact) (string, error) {
	artifact.Revision = ""
	encoded, err := json.Marshal(artifact)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}
