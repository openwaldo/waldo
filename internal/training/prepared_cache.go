// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const preparedChunkLimit = 16 << 20

type preparedCacheManifest struct {
	Kind             string               `json:"kind"`
	Schema           int                  `json:"schema"`
	Identity         string               `json:"identity"`
	NodeRank         int                  `json:"node_rank"`
	WorldSize        int                  `json:"world_size"`
	GPUsPerNode      int                  `json:"gpus_per_node"`
	GlobalMicroBatch int64                `json:"global_micro_batch"`
	Sequences        int64                `json:"sequences"`
	GlobalSequences  int64                `json:"global_sequences,omitempty"`
	Chunks           []preparedCacheChunk `json:"chunks"`
}

type preparedCacheChunk struct {
	Path      string `json:"path"`
	SHA256    string `json:"sha256"`
	Bytes     int64  `json:"bytes"`
	Sequences int64  `json:"sequences"`
}

type preparedCacheWriter struct {
	enabled    bool
	base       string
	target     string
	temporary  string
	manifest   preparedCacheManifest
	file       *os.File
	hash       hashWriter
	bytes      int64
	rows       int64
	totalBytes int64
	maxBytes   int64
}

type hashWriter struct {
	hash io.Writer
	sum  interface{ Sum([]byte) []byte }
}

func newPreparedCacheWriter(base, identity string, nodeRank, worldSize, GPUsPerNode int, globalMicroBatch, maxBytes int64) (*preparedCacheWriter, error) {
	writer := &preparedCacheWriter{}
	if base == "" || identity == "" {
		return writer, nil
	}
	parent := filepath.Join(base, identity)
	if err := os.MkdirAll(parent, 0o700); err != nil {
		return nil, err
	}
	temporary, err := os.MkdirTemp(parent, fmt.Sprintf(".node-%d-*", nodeRank))
	if err != nil {
		return nil, err
	}
	writer.enabled = true
	writer.base = base
	writer.target = filepath.Join(parent, fmt.Sprintf("node-%d", nodeRank))
	writer.temporary = temporary
	writer.maxBytes = maxBytes
	writer.manifest = preparedCacheManifest{Kind: "openwaldo-prepared-sequences", Schema: 2, Identity: identity, NodeRank: nodeRank, WorldSize: worldSize, GPUsPerNode: GPUsPerNode, GlobalMicroBatch: globalMicroBatch}
	return writer, nil
}

func (writer *preparedCacheWriter) Append(sequence PreparedSequence) error {
	if !writer.enabled {
		return nil
	}
	encoded, err := json.Marshal(sequence)
	if err != nil {
		return err
	}
	encoded = append(encoded, '\n')
	if len(encoded) > preparedChunkLimit {
		return fmt.Errorf("prepared sequence %d exceeds %d-byte chunk limit", sequence.Ordinal, preparedChunkLimit)
	}
	if writer.maxBytes > 0 && writer.totalBytes+int64(len(encoded)) > writer.maxBytes {
		writer.Abort()
		writer.enabled = false
		return nil
	}
	if writer.file == nil || writer.bytes > 0 && writer.bytes+int64(len(encoded)) > preparedChunkLimit {
		if err := writer.closeChunk(); err != nil {
			return err
		}
		path := fmt.Sprintf("chunk-%06d.ndjson", len(writer.manifest.Chunks))
		file, err := os.OpenFile(filepath.Join(writer.temporary, path), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err != nil {
			return err
		}
		digest := sha256.New()
		writer.file = file
		writer.hash = hashWriter{hash: io.MultiWriter(file, digest), sum: digest}
		writer.bytes, writer.rows = 0, 0
	}
	written, err := writer.hash.hash.Write(encoded)
	writer.bytes += int64(written)
	if err != nil {
		return err
	}
	if written != len(encoded) {
		return io.ErrShortWrite
	}
	writer.rows++
	writer.totalBytes += int64(written)
	writer.manifest.Sequences++
	return nil
}

func (writer *preparedCacheWriter) closeChunk() error {
	if writer.file == nil {
		return nil
	}
	if err := writer.file.Sync(); err != nil {
		_ = writer.file.Close()
		return err
	}
	if err := writer.file.Close(); err != nil {
		return err
	}
	path := fmt.Sprintf("chunk-%06d.ndjson", len(writer.manifest.Chunks))
	writer.manifest.Chunks = append(writer.manifest.Chunks, preparedCacheChunk{Path: path, SHA256: hex.EncodeToString(writer.hash.sum.Sum(nil)), Bytes: writer.bytes, Sequences: writer.rows})
	writer.file = nil
	return nil
}

func (writer *preparedCacheWriter) Commit(globalSequences int64) error {
	if !writer.enabled {
		return nil
	}
	if globalSequences < 1 {
		return fmt.Errorf("prepared cache global sequence count must be positive")
	}
	writer.manifest.GlobalSequences = globalSequences
	if err := writer.closeChunk(); err != nil {
		return err
	}
	manifest, err := json.MarshalIndent(writer.manifest, "", "  ")
	if err != nil {
		return err
	}
	manifest = append(manifest, '\n')
	if err := os.WriteFile(filepath.Join(writer.temporary, "MANIFEST.json"), manifest, 0o600); err != nil {
		return err
	}
	if err := os.Rename(writer.temporary, writer.target); err != nil {
		if !os.IsExist(err) {
			return err
		}
		if _, validateErr := loadPreparedCache(writer.target, writer.manifest.Identity, writer.manifest.NodeRank, writer.manifest.WorldSize, writer.manifest.GPUsPerNode, writer.manifest.GlobalMicroBatch); validateErr != nil {
			return fmt.Errorf("concurrent prepared cache differs: %w", validateErr)
		}
	}
	writer.temporary = ""
	return nil
}

func (writer *preparedCacheWriter) Abort() {
	if writer == nil {
		return
	}
	if writer.file != nil {
		_ = writer.file.Close()
	}
	if writer.temporary != "" {
		_ = os.RemoveAll(writer.temporary)
	}
}

func replayPreparedSequences(base, identity string, nodeRank, worldSize, GPUsPerNode int, globalMicroBatch, maxBytes int64, encoder *json.Encoder) (bool, error) {
	directory := filepath.Join(base, identity, fmt.Sprintf("node-%d", nodeRank))
	manifest, err := loadPreparedCache(directory, identity, nodeRank, worldSize, GPUsPerNode, globalMicroBatch)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("prepared sequence cache: %w", err)
	}
	var totalBytes int64
	for _, chunk := range manifest.Chunks {
		totalBytes += chunk.Bytes
	}
	if maxBytes > 0 && totalBytes > maxBytes {
		if err := os.RemoveAll(directory); err != nil {
			return false, fmt.Errorf("remove oversized prepared sequence cache: %w", err)
		}
		return false, nil
	}
	sequences := int64(0)
	boundaries := int64(0)
	previousOrdinal := int64(-1)
	localMicroBatch := globalMicroBatch / int64(worldSize/GPUsPerNode)
	if localMicroBatch < 1 {
		return false, fmt.Errorf("prepared cache has invalid local micro-batch")
	}
	for _, chunk := range manifest.Chunks {
		file, err := os.Open(filepath.Join(directory, chunk.Path))
		if err != nil {
			return false, err
		}
		decoder := json.NewDecoder(bufio.NewReader(file))
		chunkSequences := int64(0)
		for {
			var sequence PreparedSequence
			if err := decoder.Decode(&sequence); err != nil {
				if err == io.EOF {
					break
				}
				_ = file.Close()
				return false, err
			}
			ownerRank := int(sequence.Ordinal % int64(worldSize))
			if sequence.Ordinal <= previousOrdinal || ownerRank/GPUsPerNode != nodeRank || len(sequence.Tokens) < 2 || len(sequence.LossMask) != len(sequence.Tokens)-1 {
				_ = file.Close()
				return false, fmt.Errorf("prepared cache contains invalid sequence %d", sequence.Ordinal)
			}
			for _, count := range sequence.Consumption {
				if count < 0 {
					_ = file.Close()
					return false, fmt.Errorf("prepared cache contains invalid consumption")
				}
			}
			previousOrdinal = sequence.Ordinal
			if encoder != nil {
				if err := encoder.Encode(WorkerInputFrame{Kind: "sequence", Schema: WorkerProtocolSchema, Sequence: &sequence}); err != nil {
					_ = file.Close()
					return false, err
				}
			}
			sequences++
			chunkSequences++
			if encoder != nil && sequences%localMicroBatch == 0 && boundaries < manifest.GlobalSequences/globalMicroBatch {
				if err := encoder.Encode(WorkerInputFrame{Kind: "micro_batch_end", Schema: WorkerProtocolSchema}); err != nil {
					_ = file.Close()
					return false, err
				}
				boundaries++
			}
		}
		if err := file.Close(); err != nil {
			return false, err
		}
		if chunkSequences != chunk.Sequences {
			return false, fmt.Errorf("prepared cache chunk sequence count differs")
		}
	}
	if sequences != manifest.Sequences {
		return false, fmt.Errorf("prepared cache sequence count differs")
	}
	return true, nil
}

func loadPreparedCache(directory, identity string, nodeRank, worldSize, GPUsPerNode int, globalMicroBatch int64) (preparedCacheManifest, error) {
	data, err := os.ReadFile(filepath.Join(directory, "MANIFEST.json"))
	if err != nil {
		return preparedCacheManifest{}, err
	}
	var manifest preparedCacheManifest
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&manifest); err != nil {
		return preparedCacheManifest{}, err
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return preparedCacheManifest{}, fmt.Errorf("unexpected content after prepared cache manifest")
	}
	if manifest.Kind != "openwaldo-prepared-sequences" || manifest.Schema < 1 || manifest.Schema > 2 || manifest.Identity != identity || manifest.NodeRank != nodeRank || manifest.WorldSize != worldSize || manifest.GPUsPerNode != GPUsPerNode || manifest.GlobalMicroBatch != globalMicroBatch || manifest.Sequences < 1 || len(manifest.Chunks) == 0 {
		return preparedCacheManifest{}, fmt.Errorf("prepared cache manifest identity differs")
	}
	if manifest.Schema == 1 {
		// Schema 1 caches could only be committed for complete global batches.
		manifest.GlobalSequences = manifest.Sequences * int64(worldSize/GPUsPerNode)
	} else if manifest.GlobalSequences < manifest.Sequences {
		return preparedCacheManifest{}, fmt.Errorf("prepared cache manifest has invalid global sequence count")
	}
	var sequences int64
	for position, chunk := range manifest.Chunks {
		wantPath := fmt.Sprintf("chunk-%06d.ndjson", position)
		if chunk.Path != wantPath || chunk.Bytes < 1 || chunk.Bytes > preparedChunkLimit || chunk.Sequences < 1 {
			return preparedCacheManifest{}, fmt.Errorf("invalid prepared cache chunk %d", position)
		}
		file, err := os.Open(filepath.Join(directory, chunk.Path))
		if err != nil {
			return preparedCacheManifest{}, err
		}
		digest := sha256.New()
		bytes, copyErr := io.Copy(digest, file)
		closeErr := file.Close()
		if copyErr != nil || closeErr != nil || bytes != chunk.Bytes || hex.EncodeToString(digest.Sum(nil)) != chunk.SHA256 {
			return preparedCacheManifest{}, fmt.Errorf("prepared cache chunk %d digest differs", position)
		}
		sequences += chunk.Sequences
	}
	if sequences != manifest.Sequences {
		return preparedCacheManifest{}, fmt.Errorf("prepared cache manifest sequence count differs")
	}
	return manifest, nil
}
