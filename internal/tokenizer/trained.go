// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"container/heap"
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
	TrainedKind       = "waldo-trained-tokenizer"
	TrainedName       = "waldo/byte-bpe"
	LegacyTrainedName = "waldo/bytepiece"
	maxBPEChunkBytes  = 64
	maxBPESequences   = 100_000
)

type Sample struct {
	ID   string
	Text string
}

type Merge struct {
	Left  int `json:"left" yaml:"left"`
	Right int `json:"right" yaml:"right"`
}

type Artifact struct {
	Kind                string   `json:"kind" yaml:"kind"`
	Schema              int      `json:"schema" yaml:"schema"`
	Name                string   `json:"name" yaml:"name"`
	Revision            string   `json:"revision" yaml:"revision"`
	VocabularySize      int      `json:"vocabulary_size" yaml:"vocabulary_size"`
	PadID               int      `json:"pad_id" yaml:"pad_id"`
	BOSID               int      `json:"bos_id" yaml:"bos_id"`
	EOSID               int      `json:"eos_id" yaml:"eos_id"`
	TrainingInputSHA256 string   `json:"training_input_sha256" yaml:"training_input_sha256"`
	CorpusBOMSHA256     string   `json:"corpus_bom_sha256" yaml:"corpus_bom_sha256"`
	TrainingBytes       int64    `json:"training_bytes" yaml:"training_bytes"`
	Merges              []Merge  `json:"merges,omitempty" yaml:"merges,omitempty"`
	Pieces              []string `json:"pieces,omitempty" yaml:"pieces,omitempty"`
}

type Comparison struct {
	Tokenizer     string  `json:"tokenizer"`
	Bytes         int64   `json:"bytes"`
	Tokens        int64   `json:"tokens"`
	BytesPerToken float64 `json:"bytes_per_token"`
}

type bpeSequence struct {
	symbols []int
	weight  int64
}

type pair struct {
	left  int
	right int
}

type pairState struct {
	count      int64
	sequences  map[int]struct{}
	generation uint64
}

type pairCandidate struct {
	pair       pair
	count      int64
	generation uint64
}

type pairHeap []pairCandidate

func (values pairHeap) Len() int { return len(values) }
func (values pairHeap) Less(i, j int) bool {
	if values[i].count != values[j].count {
		return values[i].count > values[j].count
	}
	if values[i].pair.left != values[j].pair.left {
		return values[i].pair.left < values[j].pair.left
	}
	return values[i].pair.right < values[j].pair.right
}
func (values pairHeap) Swap(i, j int)   { values[i], values[j] = values[j], values[i] }
func (values *pairHeap) Push(value any) { *values = append(*values, value.(pairCandidate)) }
func (values *pairHeap) Pop() any {
	old := *values
	last := old[len(old)-1]
	*values = old[:len(old)-1]
	return last
}

// TrainByteBPE trains deterministic byte-level BPE merge rules over a bounded,
// content-pinned sample. Every byte remains directly representable. Training
// operates on weighted unique chunks and incrementally updates only sequences
// affected by a merge, avoiding a full sample rescan for every vocabulary item.
func TrainByteBPE(samples []Sample, vocabularySize int, maxBytes int64, corpusBOMSHA256 string) (Artifact, error) {
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
		for _, chunk := range byteBPEChunks(text) {
			if len(chunk) > 1 {
				counts[string(chunk)]++
			}
		}
	}
	if used == 0 {
		return Artifact{}, fmt.Errorf("tokenizer training sample is empty")
	}

	type weightedChunk struct {
		bytes  string
		weight int64
	}
	chunks := make([]weightedChunk, 0, len(counts))
	for value, count := range counts {
		chunks = append(chunks, weightedChunk{bytes: value, weight: count})
	}
	sort.Slice(chunks, func(i, j int) bool {
		if chunks[i].weight != chunks[j].weight {
			return chunks[i].weight > chunks[j].weight
		}
		return chunks[i].bytes < chunks[j].bytes
	})
	if len(chunks) > maxBPESequences {
		chunks = chunks[:maxBPESequences]
	}
	sequences := make([]bpeSequence, len(chunks))
	for index, chunk := range chunks {
		symbols := make([]int, len(chunk.bytes))
		for offset, value := range []byte(chunk.bytes) {
			symbols[offset] = int(value) + 3
		}
		sequences[index] = bpeSequence{symbols: symbols, weight: chunk.weight}
	}

	states := map[pair]*pairState{}
	for sequenceID, sequence := range sequences {
		for value, occurrences := range sequencePairCounts(sequence.symbols) {
			state := states[value]
			if state == nil {
				state = &pairState{sequences: map[int]struct{}{}}
				states[value] = state
			}
			state.count += int64(occurrences) * sequence.weight
			state.sequences[sequenceID] = struct{}{}
		}
	}
	candidates := make(pairHeap, 0, len(states))
	for value, state := range states {
		candidates = append(candidates, pairCandidate{pair: value, count: state.count, generation: state.generation})
	}
	heap.Init(&candidates)

	merges := make([]Merge, 0, vocabularySize-259)
	for len(merges) < vocabularySize-259 {
		var selected pairCandidate
		found := false
		for candidates.Len() > 0 {
			candidate := heap.Pop(&candidates).(pairCandidate)
			state := states[candidate.pair]
			if state != nil && state.count > 0 && state.count == candidate.count && state.generation == candidate.generation {
				selected, found = candidate, true
				break
			}
		}
		if !found {
			break
		}
		newSymbol := 259 + len(merges)
		merges = append(merges, Merge{Left: selected.pair.left, Right: selected.pair.right})
		affected := make([]int, 0, len(states[selected.pair].sequences))
		for sequenceID := range states[selected.pair].sequences {
			affected = append(affected, sequenceID)
		}
		sort.Ints(affected)
		touched := map[pair]struct{}{}
		for _, sequenceID := range affected {
			sequence := &sequences[sequenceID]
			before := sequencePairCounts(sequence.symbols)
			if before[selected.pair] == 0 {
				continue
			}
			for value, occurrences := range before {
				state := states[value]
				state.count -= int64(occurrences) * sequence.weight
				delete(state.sequences, sequenceID)
				state.generation++
				touched[value] = struct{}{}
			}
			sequence.symbols = mergePair(sequence.symbols, selected.pair, newSymbol)
			for value, occurrences := range sequencePairCounts(sequence.symbols) {
				state := states[value]
				if state == nil {
					state = &pairState{sequences: map[int]struct{}{}}
					states[value] = state
				}
				state.count += int64(occurrences) * sequence.weight
				state.sequences[sequenceID] = struct{}{}
				state.generation++
				touched[value] = struct{}{}
			}
		}
		for value := range touched {
			state := states[value]
			if state.count > 0 {
				heap.Push(&candidates, pairCandidate{pair: value, count: state.count, generation: state.generation})
			}
		}
	}

	artifact := Artifact{Kind: TrainedKind, Schema: 2, Name: TrainedName, VocabularySize: 259 + len(merges), PadID: 0, BOSID: 1, EOSID: 2, TrainingInputSHA256: hex.EncodeToString(inputHash.Sum(nil)), CorpusBOMSHA256: corpusBOMSHA256, TrainingBytes: used, Merges: merges}
	revision, err := artifactRevision(artifact)
	if err != nil {
		return Artifact{}, err
	}
	artifact.Revision = revision
	return artifact, artifact.Validate()
}

func sequencePairCounts(symbols []int) map[pair]int {
	counts := make(map[pair]int)
	for index := 0; index+1 < len(symbols); index++ {
		counts[pair{left: symbols[index], right: symbols[index+1]}]++
	}
	return counts
}

func mergePair(symbols []int, selected pair, replacement int) []int {
	merged := make([]int, 0, len(symbols))
	for index := 0; index < len(symbols); {
		if index+1 < len(symbols) && symbols[index] == selected.left && symbols[index+1] == selected.right {
			merged = append(merged, replacement)
			index += 2
		} else {
			merged = append(merged, symbols[index])
			index++
		}
	}
	return merged
}

func IsTrainedName(name string) bool {
	return name == TrainedName || name == LegacyTrainedName
}

func (artifact Artifact) Validate() error {
	if artifact.Kind != TrainedKind || artifact.PadID != 0 || artifact.BOSID != 1 || artifact.EOSID != 2 || len(artifact.TrainingInputSHA256) != 64 || len(artifact.CorpusBOMSHA256) != 64 || artifact.TrainingBytes < 1 {
		return fmt.Errorf("invalid trained tokenizer identity")
	}
	switch {
	case artifact.Schema == 2 && artifact.Name == TrainedName:
		if len(artifact.Pieces) != 0 || artifact.VocabularySize != 259+len(artifact.Merges) {
			return fmt.Errorf("invalid trained tokenizer identity")
		}
		seen := map[Merge]bool{}
		for index, merge := range artifact.Merges {
			newSymbol := 259 + index
			if merge.Left < 3 || merge.Right < 3 || merge.Left >= newSymbol || merge.Right >= newSymbol || seen[merge] {
				return fmt.Errorf("invalid trained tokenizer merge %d", index)
			}
			seen[merge] = true
		}
	case artifact.Schema == 1 && artifact.Name == LegacyTrainedName:
		if len(artifact.Merges) != 0 || artifact.VocabularySize != 259+len(artifact.Pieces) {
			return fmt.Errorf("invalid trained tokenizer identity")
		}
		seen := map[string]bool{}
		for _, encoded := range artifact.Pieces {
			piece, err := base64.StdEncoding.DecodeString(encoded)
			if err != nil || len(piece) < 2 || len(piece) > maxBPEChunkBytes || seen[string(piece)] {
				return fmt.Errorf("invalid trained tokenizer piece")
			}
			seen[string(piece)] = true
		}
	default:
		return fmt.Errorf("invalid trained tokenizer identity")
	}
	want, err := artifactRevision(artifact)
	if err != nil || artifact.Revision != want {
		return fmt.Errorf("trained tokenizer revision differs")
	}
	return nil
}

func (artifact Artifact) Codec() (Codec, error) {
	if err := artifact.Validate(); err != nil {
		return nil, err
	}
	if artifact.Schema == 1 {
		return newLegacyCodec(artifact), nil
	}
	codec := &bpeCodec{name: artifact.Name + "@" + artifact.Revision, ranks: make(map[pair]int, len(artifact.Merges)), tokenBytes: make([][]byte, artifact.VocabularySize)}
	for value := 0; value < 256; value++ {
		codec.tokenBytes[value+3] = []byte{byte(value)}
	}
	for index, merge := range artifact.Merges {
		token := 259 + index
		codec.ranks[pair{left: merge.Left, right: merge.Right}] = index
		codec.tokenBytes[token] = append(append([]byte(nil), codec.tokenBytes[merge.Left]...), codec.tokenBytes[merge.Right]...)
	}
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

type bpeCodec struct {
	name       string
	ranks      map[pair]int
	tokenBytes [][]byte
}

func (codec *bpeCodec) Name() string          { return codec.name }
func (codec *bpeCodec) Count(text string) int { return len(codec.Encode(text)) }
func (codec *bpeCodec) Encode(text string) []int {
	var result []int
	for _, chunk := range byteBPEChunks([]byte(text)) {
		symbols := make([]int, len(chunk))
		for index, value := range chunk {
			symbols[index] = int(value) + 3
		}
		for {
			bestRank := len(codec.ranks) + 1
			best := pair{}
			for index := 0; index+1 < len(symbols); index++ {
				value := pair{left: symbols[index], right: symbols[index+1]}
				if rank, ok := codec.ranks[value]; ok && rank < bestRank {
					bestRank, best = rank, value
				}
			}
			if bestRank > len(codec.ranks) {
				break
			}
			symbols = mergePair(symbols, best, 259+bestRank)
		}
		result = append(result, symbols...)
	}
	return result
}
func (codec *bpeCodec) Decode(tokens []int) string {
	var output []byte
	for _, token := range tokens {
		if token >= 3 && token < len(codec.tokenBytes) {
			output = append(output, codec.tokenBytes[token]...)
		}
	}
	return string(output)
}

func byteBPEChunks(text []byte) [][]byte {
	runs := lexicalRuns(text)
	chunks := make([][]byte, 0, len(runs))
	var whitespace []byte
	for _, run := range runs {
		if run.space {
			whitespace = append(whitespace, run.bytes...)
			continue
		}
		value := append(append([]byte(nil), whitespace...), run.bytes...)
		whitespace = nil
		chunks = appendSplitChunks(chunks, value)
	}
	if len(whitespace) > 0 {
		chunks = appendSplitChunks(chunks, whitespace)
	}
	return chunks
}

type lexicalRun struct {
	bytes []byte
	space bool
}

func lexicalRuns(text []byte) []lexicalRun {
	var result []lexicalRun
	start, current := 0, -1
	for index := 0; index < len(text); {
		value, size := utf8.DecodeRune(text[index:])
		if value == utf8.RuneError && size == 1 {
			value = rune(text[index])
		}
		kind := 2
		if unicode.IsLetter(value) || unicode.IsNumber(value) || value == '_' {
			kind = 0
		} else if unicode.IsSpace(value) {
			kind = 1
		}
		if current != -1 && kind != current {
			result = append(result, lexicalRun{bytes: text[start:index], space: current == 1})
			start = index
		}
		current = kind
		index += size
	}
	if start < len(text) {
		result = append(result, lexicalRun{bytes: text[start:], space: current == 1})
	}
	return result
}

func appendSplitChunks(chunks [][]byte, value []byte) [][]byte {
	for len(value) > maxBPEChunkBytes {
		chunks = append(chunks, value[:maxBPEChunkBytes])
		value = value[maxBPEChunkBytes:]
	}
	if len(value) > 0 {
		chunks = append(chunks, value)
	}
	return chunks
}

type legacyPiece struct {
	bytes []byte
	id    int
}

type legacyCodec struct {
	name    string
	pieces  []string
	byFirst map[byte][]legacyPiece
}

func newLegacyCodec(artifact Artifact) *legacyCodec {
	codec := &legacyCodec{name: artifact.Name + "@" + artifact.Revision, pieces: artifact.Pieces, byFirst: map[byte][]legacyPiece{}}
	for index, encoded := range artifact.Pieces {
		piece, _ := base64.StdEncoding.DecodeString(encoded)
		codec.byFirst[piece[0]] = append(codec.byFirst[piece[0]], legacyPiece{bytes: piece, id: 259 + index})
	}
	for first := range codec.byFirst {
		sort.Slice(codec.byFirst[first], func(i, j int) bool { return len(codec.byFirst[first][i].bytes) > len(codec.byFirst[first][j].bytes) })
	}
	return codec
}

func (codec *legacyCodec) Name() string          { return codec.name }
func (codec *legacyCodec) Count(text string) int { return len(codec.Encode(text)) }
func (codec *legacyCodec) Encode(text string) []int {
	input := []byte(text)
	result := make([]int, 0, len(input))
	for len(input) > 0 {
		matched := false
		for _, piece := range codec.byFirst[input[0]] {
			if len(input) >= len(piece.bytes) && string(input[:len(piece.bytes)]) == string(piece.bytes) {
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
func (codec *legacyCodec) Decode(tokens []int) string {
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

func artifactRevision(artifact Artifact) (string, error) {
	artifact.Revision = ""
	encoded, err := json.Marshal(artifact)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}
