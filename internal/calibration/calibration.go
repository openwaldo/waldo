// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package calibration builds bounded, reproducible quantization-calibration
// inputs from verified WALDO corpus selections. It never trains a model.
package calibration

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"unicode/utf8"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/shard"
)

const (
	Profile       = "deterministic-byte-sample-schema-1"
	DefaultTokens = int64(100_000)
	DefaultSeed   = uint64(42)
)

type CorpusPin struct {
	BOMSHA256 string         `json:"bom_sha256"`
	Index     index.Identity `json:"index"`
	Paths     []string       `json:"paths"`
	Tokens    int64          `json:"reference_tokens"`
}

type BOM struct {
	Kind            string            `json:"kind"`
	Schema          int               `json:"schema"`
	Subject         string            `json:"subject"`
	Profile         string            `json:"profile"`
	Corpus          CorpusPin         `json:"corpus"`
	BudgetTokens    int64             `json:"budget_byte_tokens"`
	Seed            uint64            `json:"seed"`
	Shards          []corpus.ShardPin `json:"shards"`
	Records         int64             `json:"records"`
	SampledTokens   int64             `json:"sampled_byte_tokens"`
	SelectionSHA256 string            `json:"selection_sha256"`
}

type Prepared struct {
	TextPath string
	BOM      BOM
	JSON     []byte
	SHA256   string
	Cleanup  func()
}

type Progress struct {
	Phase   string
	Current int
	Total   int
	Shard   string
}

func Prepare(ctx context.Context, source corpus.BOM, cache *lookaside.Cache, budget int64, seed uint64, progress func(Progress)) (Prepared, error) {
	if cache == nil {
		return Prepared{}, fmt.Errorf("calibration requires a lookaside cache")
	}
	if err := source.Validate(); err != nil {
		return Prepared{}, fmt.Errorf("calibration corpus OpenWALDO BOM: %w", err)
	}
	if budget < 1 {
		return Prepared{}, fmt.Errorf("calibration token budget must be positive")
	}
	corpusHash, err := hashJSON(source)
	if err != nil {
		return Prepared{}, err
	}
	// MkdirTemp requires its parent to exist.
	if err := cache.EnsureScratch(); err != nil {
		return Prepared{}, fmt.Errorf("prepare calibration scratch %s: %w; set a writable location with `waldo config set lookaside.scratch <directory>`", cache.Scratch(), err)
	}
	directory, err := os.MkdirTemp(cache.Scratch(), ".waldo-calibration-*")
	if err != nil {
		return Prepared{}, err
	}
	cleanup := func() { _ = os.RemoveAll(directory) }
	committed := false
	defer func() {
		if !committed {
			cleanup()
		}
	}()
	textPath := filepath.Join(directory, "calibration.txt")
	file, err := os.OpenFile(textPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return Prepared{}, err
	}
	writer := bufio.NewWriter(file)
	selection := sha256.New()
	result := BOM{
		Kind: "openwaldo-bom", Schema: 1, Subject: "quantization-calibration", Profile: Profile,
		Corpus:       CorpusPin{BOMSHA256: corpusHash, Index: source.Index, Paths: append([]string(nil), source.Paths...), Tokens: source.Totals.Tokens},
		BudgetTokens: budget, Seed: seed,
	}
	ordered := uniqueShards(source.Shards, seed)
	for position, pin := range ordered {
		if result.SampledTokens >= budget {
			break
		}
		if err := ctx.Err(); err != nil {
			file.Close()
			return Prepared{}, err
		}
		if progress != nil {
			progress(Progress{Phase: "fetch", Current: position + 1, Total: len(ordered), Shard: pin.SHA256})
		}
		path, err := cache.Fetch(ctx, pin.URL, pin.SHA256, pin.Bytes)
		if err != nil {
			file.Close()
			return Prepared{}, fmt.Errorf("calibration shard %s: %w", pin.SHA256[:12], err)
		}
		if _, err := shard.Audit(ctx, []string{path}); err != nil {
			file.Close()
			return Prepared{}, fmt.Errorf("audit calibration shard %s: %w", pin.SHA256[:12], err)
		}
		used := false
		err = shard.WalkRecords(path, func(_ int64, record shard.RecordView) error {
			if result.SampledTokens >= budget {
				return nil
			}
			remaining := budget - result.SampledTokens
			text := truncateUTF8(record.Text, remaining)
			if text == "" {
				return nil
			}
			if result.Records > 0 {
				if _, err := writer.WriteString("\n\n"); err != nil {
					return err
				}
			}
			if _, err := writer.WriteString(text); err != nil {
				return err
			}
			writeSelection(selection, record.ID, len(text))
			result.Records++
			result.SampledTokens += int64(len([]byte(text)))
			used = true
			return nil
		})
		if err != nil {
			file.Close()
			return Prepared{}, err
		}
		if used {
			result.Shards = append(result.Shards, pin)
		}
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return Prepared{}, err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return Prepared{}, err
	}
	if err := file.Close(); err != nil {
		return Prepared{}, err
	}
	if result.SampledTokens == 0 {
		return Prepared{}, fmt.Errorf("calibration corpus contains no usable text")
	}
	result.SelectionSHA256 = hex.EncodeToString(selection.Sum(nil))
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return Prepared{}, err
	}
	encoded = append(encoded, '\n')
	digest := sha256.Sum256(encoded)
	committed = true
	return Prepared{TextPath: textPath, BOM: result, JSON: encoded, SHA256: hex.EncodeToString(digest[:]), Cleanup: cleanup}, nil
}

func uniqueShards(shards []corpus.ShardPin, seed uint64) []corpus.ShardPin {
	seen := map[string]bool{}
	var result []corpus.ShardPin
	for _, pin := range shards {
		if !seen[pin.SHA256] {
			seen[pin.SHA256] = true
			result = append(result, pin)
		}
	}
	sort.Slice(result, func(i, j int) bool {
		return shardOrder(result[i].SHA256, seed) < shardOrder(result[j].SHA256, seed)
	})
	return result
}

func shardOrder(value string, seed uint64) string {
	var bytes [8]byte
	binary.LittleEndian.PutUint64(bytes[:], seed)
	digest := sha256.Sum256(append(bytes[:], value...))
	return hex.EncodeToString(digest[:])
}

func truncateUTF8(value string, limit int64) string {
	if int64(len([]byte(value))) <= limit {
		return value
	}
	data := []byte(value)
	if limit < int64(len(data)) {
		data = data[:limit]
	}
	for len(data) > 0 && !utf8.Valid(data) {
		data = data[:len(data)-1]
	}
	return string(data)
}

func writeSelection(target hash.Hash, id string, bytes int) {
	_, _ = target.Write([]byte(id))
	_, _ = target.Write([]byte{0})
	_, _ = target.Write([]byte(strconv.Itoa(bytes)))
	_, _ = target.Write([]byte{'\n'})
}

func hashJSON(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}
