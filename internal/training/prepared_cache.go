// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
)

var preparedChunkLimit int64 = 16 << 20

const preparedChunkMagic = "OWPS3\x00\x00\x01"

type preparedCacheManifest struct {
	Kind             string               `json:"kind"`
	Schema           int                  `json:"schema"`
	Identity         string               `json:"identity"`
	BOMSHA256        string               `json:"run_bom_sha256,omitempty"`
	Format           string               `json:"format,omitempty"`
	NodeRank         int                  `json:"node_rank"`
	WorldSize        int                  `json:"world_size"`
	GPUsPerNode      int                  `json:"gpus_per_node"`
	GlobalMicroBatch int64                `json:"global_micro_batch"`
	Sequences        int64                `json:"sequences"`
	GlobalSequences  int64                `json:"global_sequences,omitempty"`
	Chunks           []preparedCacheChunk `json:"chunks"`
}

type preparedCacheChunk struct {
	Path         string `json:"path"`
	SHA256       string `json:"sha256"`
	Bytes        int64  `json:"bytes"`
	Sequences    int64  `json:"sequences"`
	FirstOrdinal int64  `json:"first_ordinal"`
	LastOrdinal  int64  `json:"last_ordinal"`
}

type preparedCacheWriter struct {
	enabled      bool
	base         string
	target       string
	temporary    string
	manifest     preparedCacheManifest
	file         *os.File
	hash         hashWriter
	bytes        int64
	rows         int64
	firstOrdinal int64
	lastOrdinal  int64
	totalBytes   int64
	maxBytes     int64
}

type hashWriter struct {
	hash io.Writer
	sum  interface{ Sum([]byte) []byte }
}

func encodePreparedSequence(sequence PreparedSequence) ([]byte, error) {
	if sequence.Ordinal < 0 || len(sequence.Tokens) < 2 || len(sequence.LossMask) != len(sequence.Tokens)-1 || len(sequence.Tokens) > math.MaxUint32 || len(sequence.Consumption) > math.MaxUint32 {
		return nil, fmt.Errorf("prepared sequence %d has invalid dimensions", sequence.Ordinal)
	}
	var encoded bytes.Buffer
	write := func(value any) error { return binary.Write(&encoded, binary.LittleEndian, value) }
	if err := write(uint64(sequence.Ordinal)); err != nil {
		return nil, err
	}
	if err := write(uint32(len(sequence.Tokens))); err != nil {
		return nil, err
	}
	if err := write(uint32(len(sequence.Consumption))); err != nil {
		return nil, err
	}
	for _, token := range sequence.Tokens {
		if token < 0 || uint64(token) > math.MaxUint32 {
			return nil, fmt.Errorf("prepared sequence %d contains invalid token %d", sequence.Ordinal, token)
		}
		if err := write(uint32(token)); err != nil {
			return nil, err
		}
	}
	mask := make([]byte, (len(sequence.LossMask)+7)/8)
	for index, enabled := range sequence.LossMask {
		if enabled {
			mask[index/8] |= 1 << uint(index%8)
		}
	}
	if _, err := encoded.Write(mask); err != nil {
		return nil, err
	}
	corpora := make([]string, 0, len(sequence.Consumption))
	for corpus := range sequence.Consumption {
		corpora = append(corpora, corpus)
	}
	sort.Strings(corpora)
	for _, corpus := range corpora {
		count := sequence.Consumption[corpus]
		if corpus == "" || len(corpus) > math.MaxUint16 || count < 0 {
			return nil, fmt.Errorf("prepared sequence %d has invalid corpus consumption", sequence.Ordinal)
		}
		if err := write(uint16(len(corpus))); err != nil {
			return nil, err
		}
		if _, err := encoded.WriteString(corpus); err != nil {
			return nil, err
		}
		if err := write(uint64(count)); err != nil {
			return nil, err
		}
	}
	return encoded.Bytes(), nil
}

func decodePreparedChunk(input io.Reader, consume func(PreparedSequence) error) (int64, int64, int64, error) {
	magic := make([]byte, len(preparedChunkMagic))
	if _, err := io.ReadFull(input, magic); err != nil {
		return 0, 0, 0, err
	}
	if string(magic) != preparedChunkMagic {
		return 0, 0, 0, fmt.Errorf("prepared cache chunk has invalid format")
	}
	rows, first, last := int64(0), int64(-1), int64(-1)
	for {
		var ordinal uint64
		if err := binary.Read(input, binary.LittleEndian, &ordinal); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return 0, 0, 0, err
		}
		var tokenCount, consumptionCount uint32
		if err := binary.Read(input, binary.LittleEndian, &tokenCount); err != nil {
			return 0, 0, 0, err
		}
		if err := binary.Read(input, binary.LittleEndian, &consumptionCount); err != nil {
			return 0, 0, 0, err
		}
		if ordinal > math.MaxInt64 || tokenCount < 2 || int64(tokenCount) > preparedChunkLimit/4 || consumptionCount > tokenCount {
			return 0, 0, 0, fmt.Errorf("prepared cache contains invalid sequence header")
		}
		tokens := make([]int, int(tokenCount))
		for index := range tokens {
			var token uint32
			if err := binary.Read(input, binary.LittleEndian, &token); err != nil {
				return 0, 0, 0, err
			}
			tokens[index] = int(token)
		}
		maskCount := int(tokenCount) - 1
		packedMask := make([]byte, (maskCount+7)/8)
		if _, err := io.ReadFull(input, packedMask); err != nil {
			return 0, 0, 0, err
		}
		if remainder := maskCount % 8; remainder != 0 && packedMask[len(packedMask)-1]&^byte((1<<uint(remainder))-1) != 0 {
			return 0, 0, 0, fmt.Errorf("prepared cache sequence has non-canonical loss mask")
		}
		mask := make([]bool, maskCount)
		masked := int64(0)
		for index := range mask {
			mask[index] = packedMask[index/8]&(1<<uint(index%8)) != 0
			if mask[index] {
				masked++
			}
		}
		consumption := make(map[string]int64, int(consumptionCount))
		previousCorpus := ""
		consumed := int64(0)
		for index := uint32(0); index < consumptionCount; index++ {
			var nameBytes uint16
			if err := binary.Read(input, binary.LittleEndian, &nameBytes); err != nil {
				return 0, 0, 0, err
			}
			if nameBytes == 0 {
				return 0, 0, 0, fmt.Errorf("prepared cache sequence has empty corpus")
			}
			name := make([]byte, int(nameBytes))
			if _, err := io.ReadFull(input, name); err != nil {
				return 0, 0, 0, err
			}
			corpus := string(name)
			if corpus <= previousCorpus {
				return 0, 0, 0, fmt.Errorf("prepared cache sequence corpus order is not canonical")
			}
			var value uint64
			if err := binary.Read(input, binary.LittleEndian, &value); err != nil || value > math.MaxInt64 {
				return 0, 0, 0, fmt.Errorf("prepared cache sequence has invalid consumption")
			}
			if int64(value) > math.MaxInt64-consumed {
				return 0, 0, 0, fmt.Errorf("prepared cache sequence consumption overflows int64")
			}
			consumption[corpus] = int64(value)
			consumed += int64(value)
			previousCorpus = corpus
		}
		if consumed != masked {
			return 0, 0, 0, fmt.Errorf("prepared cache sequence consumption does not match its loss mask")
		}
		sequence := PreparedSequence{Ordinal: int64(ordinal), Tokens: tokens, LossMask: mask, Consumption: consumption}
		if consume != nil {
			if err := consume(sequence); err != nil {
				return 0, 0, 0, err
			}
		}
		if rows == 0 {
			first = sequence.Ordinal
		}
		last = sequence.Ordinal
		rows++
	}
	if rows == 0 {
		return 0, 0, 0, fmt.Errorf("prepared cache chunk is empty")
	}
	return rows, first, last, nil
}

func newPreparedCacheWriter(base, identity, bomSHA256 string, nodeRank, worldSize, GPUsPerNode int, globalMicroBatch, maxBytes int64) (*preparedCacheWriter, error) {
	writer := &preparedCacheWriter{}
	if base == "" || identity == "" {
		return writer, nil
	}
	if !validPreparedSHA256(bomSHA256) {
		return nil, fmt.Errorf("prepared cache requires a valid Run BOM SHA-256")
	}
	parent := filepath.Join(base, identity)
	if err := os.MkdirAll(parent, 0o700); err != nil {
		return nil, err
	}
	stale, err := filepath.Glob(filepath.Join(parent, fmt.Sprintf(".node-%d-*", nodeRank)))
	if err != nil {
		return nil, err
	}
	for _, path := range stale {
		if err := os.RemoveAll(path); err != nil {
			return nil, fmt.Errorf("remove interrupted prepared cache %s: %w", path, err)
		}
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
	writer.manifest = preparedCacheManifest{Kind: "openwaldo-prepared-sequences", Schema: 3, Identity: identity, BOMSHA256: bomSHA256, Format: "compact-binary-v1", NodeRank: nodeRank, WorldSize: worldSize, GPUsPerNode: GPUsPerNode, GlobalMicroBatch: globalMicroBatch}
	return writer, nil
}

func (writer *preparedCacheWriter) Append(sequence PreparedSequence) error {
	if !writer.enabled {
		return nil
	}
	encoded, err := encodePreparedSequence(sequence)
	if err != nil {
		return err
	}
	if int64(len(encoded)+len(preparedChunkMagic)) > preparedChunkLimit {
		return fmt.Errorf("prepared sequence %d exceeds %d-byte chunk limit", sequence.Ordinal, preparedChunkLimit)
	}
	newChunk := writer.file == nil || writer.bytes > int64(len(preparedChunkMagic)) && writer.bytes+int64(len(encoded)) > preparedChunkLimit
	additional := int64(len(encoded))
	if newChunk {
		additional += int64(len(preparedChunkMagic))
	}
	if writer.maxBytes > 0 && writer.totalBytes+additional > writer.maxBytes {
		writer.Abort()
		writer.enabled = false
		return nil
	}
	if newChunk {
		if err := writer.closeChunk(); err != nil {
			return err
		}
		path := fmt.Sprintf("chunk-%06d.bin", len(writer.manifest.Chunks))
		file, err := os.OpenFile(filepath.Join(writer.temporary, path), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err != nil {
			return err
		}
		digest := sha256.New()
		writer.file = file
		writer.hash = hashWriter{hash: io.MultiWriter(file, digest), sum: digest}
		writer.bytes, writer.rows = 0, 0
		writer.firstOrdinal, writer.lastOrdinal = sequence.Ordinal, sequence.Ordinal
		written, err := writer.hash.hash.Write([]byte(preparedChunkMagic))
		writer.bytes += int64(written)
		writer.totalBytes += int64(written)
		if err != nil {
			return err
		}
		if written != len(preparedChunkMagic) {
			return io.ErrShortWrite
		}
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
	writer.lastOrdinal = sequence.Ordinal
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
	path := fmt.Sprintf("chunk-%06d.bin", len(writer.manifest.Chunks))
	writer.manifest.Chunks = append(writer.manifest.Chunks, preparedCacheChunk{Path: path, SHA256: hex.EncodeToString(writer.hash.sum.Sum(nil)), Bytes: writer.bytes, Sequences: writer.rows, FirstOrdinal: writer.firstOrdinal, LastOrdinal: writer.lastOrdinal})
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
		if _, validateErr := loadPreparedCache(writer.target, writer.manifest.Identity, writer.manifest.BOMSHA256, writer.manifest.NodeRank, writer.manifest.WorldSize, writer.manifest.GPUsPerNode, writer.manifest.GlobalMicroBatch, true); validateErr != nil {
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

func replayPreparedSequences(base, identity, bomSHA256 string, nodeRank, worldSize, GPUsPerNode int, globalMicroBatch, maxBytes, resumeSequences int64, encoder *json.Encoder) (bool, error) {
	directory := filepath.Join(base, identity, fmt.Sprintf("node-%d", nodeRank))
	manifest, err := loadPreparedCache(directory, identity, bomSHA256, nodeRank, worldSize, GPUsPerNode, globalMicroBatch, false)
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
	if manifest.Schema == 3 {
		return replayIndexedPreparedSequences(directory, manifest, nodeRank, worldSize, GPUsPerNode, globalMicroBatch, resumeSequences, encoder)
	}
	return replayLegacyPreparedSequences(directory, manifest, nodeRank, worldSize, GPUsPerNode, globalMicroBatch, resumeSequences, encoder)
}

func preparedCacheReady(begin WorkerBegin) (bool, error) {
	if begin.PreparedCacheDirectory == "" || begin.PreparedIdentity == "" {
		return false, nil
	}
	directory := filepath.Join(begin.PreparedCacheDirectory, begin.PreparedIdentity, fmt.Sprintf("node-%d", begin.DataNodeRank))
	_, err := loadPreparedCache(directory, begin.PreparedIdentity, begin.PreparedBOMSHA256, begin.DataNodeRank, begin.Parallelism.WorldSize, begin.Parallelism.GPUsPerNode, begin.Parameters.BatchSize/begin.Parameters.GradientAccumulation, false)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("prepared sequence cache: %w", err)
	}
	return true, nil
}

func replayIndexedPreparedSequences(directory string, manifest preparedCacheManifest, nodeRank, worldSize, GPUsPerNode int, globalMicroBatch, resumeSequences int64, encoder *json.Encoder) (bool, error) {
	boundaryLimit := manifest.GlobalSequences/globalMicroBatch - resumeSequences/globalMicroBatch
	if boundaryLimit < 0 {
		return false, fmt.Errorf("prepared cache resume position exceeds stream")
	}
	localMicroBatch := globalMicroBatch / int64(worldSize/GPUsPerNode)
	if localMicroBatch < 1 {
		return false, fmt.Errorf("prepared cache has invalid local micro-batch")
	}
	expectedFirst := resumeSequences + int64(nodeRank*GPUsPerNode)
	emitted, boundaries := int64(0), int64(0)
	previousOrdinal := int64(-1)
	for _, chunk := range manifest.Chunks {
		if chunk.LastOrdinal < expectedFirst {
			continue
		}
		path := filepath.Join(directory, chunk.Path)
		file, err := os.Open(path)
		if err != nil {
			return false, err
		}
		digest := sha256.New()
		rows, first, last, decodeErr := decodePreparedChunk(io.TeeReader(file, digest), func(sequence PreparedSequence) error {
			ownerRank := int(sequence.Ordinal % int64(worldSize))
			if sequence.Ordinal <= previousOrdinal || ownerRank/GPUsPerNode != nodeRank {
				return fmt.Errorf("prepared cache contains invalid sequence %d", sequence.Ordinal)
			}
			previousOrdinal = sequence.Ordinal
			if sequence.Ordinal < resumeSequences {
				return nil
			}
			if emitted == 0 && sequence.Ordinal != expectedFirst {
				return fmt.Errorf("prepared cache does not begin at resume ordinal %d", expectedFirst)
			}
			if encoder != nil {
				if err := encoder.Encode(WorkerInputFrame{Kind: "sequence", Schema: WorkerProtocolSchema, Sequence: &sequence}); err != nil {
					return err
				}
			}
			emitted++
			if emitted%localMicroBatch == 0 && boundaries < boundaryLimit {
				if encoder != nil {
					if err := encoder.Encode(WorkerInputFrame{Kind: "micro_batch_end", Schema: WorkerProtocolSchema}); err != nil {
						return err
					}
				}
				boundaries++
			}
			return nil
		})
		closeErr := file.Close()
		if decodeErr != nil {
			return false, decodeErr
		}
		if closeErr != nil {
			return false, closeErr
		}
		if rows != chunk.Sequences || first != chunk.FirstOrdinal || last != chunk.LastOrdinal || hex.EncodeToString(digest.Sum(nil)) != chunk.SHA256 {
			return false, fmt.Errorf("prepared cache chunk %s differs from its manifest", chunk.Path)
		}
	}
	if emitted == 0 && expectedFirst < manifest.GlobalSequences {
		return false, fmt.Errorf("prepared cache has no sequence at resume ordinal %d", expectedFirst)
	}
	return true, nil
}

func replayLegacyPreparedSequences(directory string, manifest preparedCacheManifest, nodeRank, worldSize, GPUsPerNode int, globalMicroBatch, resumeSequences int64, encoder *json.Encoder) (bool, error) {
	sequences := int64(0)
	emitted := int64(0)
	boundaries := int64(0)
	boundaryLimit := manifest.GlobalSequences/globalMicroBatch - resumeSequences/globalMicroBatch
	if boundaryLimit < 0 {
		return false, fmt.Errorf("prepared cache resume position exceeds stream")
	}
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
			if encoder != nil && sequence.Ordinal >= resumeSequences {
				if err := encoder.Encode(WorkerInputFrame{Kind: "sequence", Schema: WorkerProtocolSchema, Sequence: &sequence}); err != nil {
					_ = file.Close()
					return false, err
				}
				emitted++
				if emitted%localMicroBatch == 0 && boundaries < boundaryLimit {
					if err := encoder.Encode(WorkerInputFrame{Kind: "micro_batch_end", Schema: WorkerProtocolSchema}); err != nil {
						_ = file.Close()
						return false, err
					}
					boundaries++
				}
			}
			sequences++
			chunkSequences++
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

func loadPreparedCache(directory, identity, bomSHA256 string, nodeRank, worldSize, GPUsPerNode int, globalMicroBatch int64, verify bool) (preparedCacheManifest, error) {
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
	if manifest.Kind != "openwaldo-prepared-sequences" || manifest.Schema < 1 || manifest.Schema > 3 || manifest.Identity != identity || manifest.NodeRank != nodeRank || manifest.WorldSize != worldSize || manifest.GPUsPerNode != GPUsPerNode || manifest.GlobalMicroBatch != globalMicroBatch || manifest.Sequences < 1 || len(manifest.Chunks) == 0 {
		return preparedCacheManifest{}, fmt.Errorf("prepared cache manifest identity differs")
	}
	if manifest.Schema == 3 && (!validPreparedSHA256(bomSHA256) || manifest.BOMSHA256 != bomSHA256 || manifest.Format != "compact-binary-v1") {
		return preparedCacheManifest{}, fmt.Errorf("prepared cache manifest Run BOM binding differs")
	}
	if manifest.Schema == 1 {
		// Schema 1 caches could only be committed for complete global batches.
		manifest.GlobalSequences = manifest.Sequences * int64(worldSize/GPUsPerNode)
	} else if manifest.GlobalSequences < manifest.Sequences {
		return preparedCacheManifest{}, fmt.Errorf("prepared cache manifest has invalid global sequence count")
	}
	var sequences int64
	previousOrdinal := int64(-1)
	for position, chunk := range manifest.Chunks {
		wantPath := fmt.Sprintf("chunk-%06d.ndjson", position)
		if manifest.Schema == 3 {
			wantPath = fmt.Sprintf("chunk-%06d.bin", position)
		}
		if chunk.Path != wantPath || chunk.Bytes < 1 || chunk.Bytes > preparedChunkLimit || chunk.Sequences < 1 || manifest.Schema == 3 && (chunk.FirstOrdinal < 0 || chunk.LastOrdinal < chunk.FirstOrdinal || chunk.FirstOrdinal <= previousOrdinal) {
			return preparedCacheManifest{}, fmt.Errorf("invalid prepared cache chunk %d", position)
		}
		path := filepath.Join(directory, chunk.Path)
		info, err := os.Stat(path)
		if err != nil {
			return preparedCacheManifest{}, err
		}
		if info.Size() != chunk.Bytes {
			return preparedCacheManifest{}, fmt.Errorf("prepared cache chunk %d digest differs", position)
		}
		if verify {
			file, err := os.Open(path)
			if err != nil {
				return preparedCacheManifest{}, err
			}
			digest := sha256.New()
			copied, copyErr := io.Copy(digest, file)
			closeErr := file.Close()
			if copyErr != nil || closeErr != nil || copied != chunk.Bytes || hex.EncodeToString(digest.Sum(nil)) != chunk.SHA256 {
				return preparedCacheManifest{}, fmt.Errorf("prepared cache chunk %d digest differs", position)
			}
		}
		sequences += chunk.Sequences
		previousOrdinal = chunk.LastOrdinal
	}
	if sequences != manifest.Sequences {
		return preparedCacheManifest{}, fmt.Errorf("prepared cache manifest sequence count differs")
	}
	return manifest, nil
}

func validPreparedSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
