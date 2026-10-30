// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package modelweights

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
)

type Descriptor struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

type sourceTensor struct {
	name       string
	descriptor Descriptor
	file       *os.File
	dataStart  int64
}

// Inspect validates a Safetensors file and returns its tensor descriptors and
// string metadata without materializing tensor payloads.
func Inspect(path string) (map[string]Descriptor, map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()
	header, dataStart, err := readHeader(file)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	info, err := file.Stat()
	if err != nil {
		return nil, nil, err
	}
	if info.Size() < dataStart {
		return nil, nil, fmt.Errorf("%s ends before its Safetensors payload", path)
	}
	dataBytes := uint64(info.Size() - dataStart)
	tensors := make(map[string]Descriptor, len(header))
	metadata := map[string]string{}
	var ranges [][2]uint64
	for name, raw := range header {
		if name == "__metadata__" {
			if err := json.Unmarshal(raw, &metadata); err != nil {
				return nil, nil, fmt.Errorf("%s metadata: %w", path, err)
			}
			continue
		}
		var descriptor Descriptor
		if err := json.Unmarshal(raw, &descriptor); err != nil {
			return nil, nil, fmt.Errorf("%s tensor %s: %w", path, name, err)
		}
		if len(descriptor.DataOffsets) != 2 || descriptor.DataOffsets[1] < descriptor.DataOffsets[0] || descriptor.DataOffsets[1] > dataBytes {
			return nil, nil, fmt.Errorf("%s tensor %s has invalid data offsets", path, name)
		}
		expected, err := tensorBytes(descriptor)
		if err != nil {
			return nil, nil, fmt.Errorf("%s tensor %s: %w", path, name, err)
		}
		if descriptor.DataOffsets[1]-descriptor.DataOffsets[0] != expected {
			return nil, nil, fmt.Errorf("%s tensor %s payload is %d bytes; shape and dtype require %d", path, name, descriptor.DataOffsets[1]-descriptor.DataOffsets[0], expected)
		}
		ranges = append(ranges, [2]uint64{descriptor.DataOffsets[0], descriptor.DataOffsets[1]})
		tensors[name] = descriptor
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i][0] < ranges[j][0] })
	if len(ranges) > 0 && ranges[0][0] != 0 {
		return nil, nil, fmt.Errorf("%s Safetensors payload starts at byte %d, want 0", path, ranges[0][0])
	}
	for index := 1; index < len(ranges); index++ {
		if ranges[index][0] < ranges[index-1][1] {
			return nil, nil, fmt.Errorf("%s contains overlapping tensor payloads", path)
		}
		if ranges[index][0] != ranges[index-1][1] {
			return nil, nil, fmt.Errorf("%s contains a gap between tensor payloads", path)
		}
	}
	if len(ranges) > 0 && ranges[len(ranges)-1][1] != dataBytes {
		return nil, nil, fmt.Errorf("%s contains %d unclaimed payload bytes", path, dataBytes-ranges[len(ranges)-1][1])
	}
	return tensors, metadata, nil
}

func NormalizeHuggingFace(sources []string, destination string) (map[string]Descriptor, error) {
	if len(sources) == 0 {
		return nil, fmt.Errorf("at least one Safetensors source is required")
	}
	var opened []*os.File
	defer func() {
		for _, file := range opened {
			_ = file.Close()
		}
	}()
	tensors := map[string]sourceTensor{}
	for _, path := range sources {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		opened = append(opened, file)
		header, dataStart, err := readHeader(file)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		info, err := file.Stat()
		if err != nil {
			return nil, err
		}
		if info.Size() < dataStart {
			return nil, fmt.Errorf("%s ends before its Safetensors payload", path)
		}
		dataBytes := uint64(info.Size() - dataStart)
		var ranges [][2]uint64
		for sourceName, raw := range header {
			if sourceName == "__metadata__" {
				continue
			}
			var descriptor Descriptor
			if err := json.Unmarshal(raw, &descriptor); err != nil {
				return nil, fmt.Errorf("%s tensor %s: %w", path, sourceName, err)
			}
			if len(descriptor.DataOffsets) != 2 || descriptor.DataOffsets[1] < descriptor.DataOffsets[0] {
				return nil, fmt.Errorf("%s tensor %s has invalid data offsets", path, sourceName)
			}
			if descriptor.DataOffsets[1] > dataBytes {
				return nil, fmt.Errorf("%s tensor %s extends beyond the Safetensors payload", path, sourceName)
			}
			expected, err := tensorBytes(descriptor)
			if err != nil {
				return nil, fmt.Errorf("%s tensor %s: %w", path, sourceName, err)
			}
			if descriptor.DataOffsets[1]-descriptor.DataOffsets[0] != expected {
				return nil, fmt.Errorf("%s tensor %s payload is %d bytes; shape and dtype require %d", path, sourceName, descriptor.DataOffsets[1]-descriptor.DataOffsets[0], expected)
			}
			ranges = append(ranges, [2]uint64{descriptor.DataOffsets[0], descriptor.DataOffsets[1]})
			target, err := WALDOName(sourceName)
			if err != nil {
				return nil, err
			}
			if _, exists := tensors[target]; exists {
				return nil, fmt.Errorf("Safetensors tensor mapping collides at %q", target)
			}
			tensors[target] = sourceTensor{target, descriptor, file, dataStart}
		}
		sort.Slice(ranges, func(i, j int) bool { return ranges[i][0] < ranges[j][0] })
		for index := 1; index < len(ranges); index++ {
			if ranges[index][0] < ranges[index-1][1] {
				return nil, fmt.Errorf("%s contains overlapping tensor payloads", path)
			}
		}
	}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	result := make(map[string]Descriptor, len(names))
	var offset uint64
	for _, name := range names {
		value := tensors[name]
		length := value.descriptor.DataOffsets[1] - value.descriptor.DataOffsets[0]
		descriptor := value.descriptor
		descriptor.DataOffsets = []uint64{offset, offset + length}
		result[name] = descriptor
		offset += length
	}
	header := map[string]any{"__metadata__": map[string]string{"format": "pt", "source_format": "huggingface", "normalized_by": "openwaldo"}}
	for name, descriptor := range result {
		header[name] = descriptor
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		return nil, err
	}
	for len(encoded)%8 != 0 {
		encoded = append(encoded, ' ')
	}
	output, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	committed := false
	defer func() {
		_ = output.Close()
		if !committed {
			_ = os.Remove(destination)
		}
	}()
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(encoded)))
	if _, err := output.Write(length[:]); err != nil {
		return nil, err
	}
	if _, err := output.Write(encoded); err != nil {
		return nil, err
	}
	for _, name := range names {
		value := tensors[name]
		start, end := value.descriptor.DataOffsets[0], value.descriptor.DataOffsets[1]
		if _, err := io.CopyN(output, io.NewSectionReader(value.file, value.dataStart+int64(start), int64(end-start)), int64(end-start)); err != nil {
			return nil, fmt.Errorf("copy tensor %s: %w", name, err)
		}
	}
	if err := output.Sync(); err != nil {
		return nil, err
	}
	if err := output.Close(); err != nil {
		return nil, err
	}
	committed = true
	return result, nil
}

func tensorBytes(descriptor Descriptor) (uint64, error) {
	width, ok := map[string]uint64{
		"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
		"U16": 2, "I16": 2, "F16": 2, "BF16": 2,
		"U32": 4, "I32": 4, "F32": 4,
		"U64": 8, "I64": 8, "F64": 8,
	}[descriptor.DType]
	if !ok {
		return 0, fmt.Errorf("unsupported dtype %q", descriptor.DType)
	}
	count := uint64(1)
	for _, dimension := range descriptor.Shape {
		if dimension != 0 && count > ^uint64(0)/dimension {
			return 0, fmt.Errorf("tensor shape overflows its byte length")
		}
		count *= dimension
	}
	if width != 0 && count > ^uint64(0)/width {
		return 0, fmt.Errorf("tensor byte length overflows")
	}
	return count * width, nil
}

func readHeader(file *os.File) (map[string]json.RawMessage, int64, error) {
	var length [8]byte
	if _, err := io.ReadFull(file, length[:]); err != nil {
		return nil, 0, fmt.Errorf("read Safetensors header length: %w", err)
	}
	headerLength := binary.LittleEndian.Uint64(length[:])
	if headerLength == 0 || headerLength > 1<<30 {
		return nil, 0, fmt.Errorf("invalid Safetensors header length %d", headerLength)
	}
	header := make([]byte, headerLength)
	if _, err := io.ReadFull(file, header); err != nil {
		return nil, 0, fmt.Errorf("read Safetensors header: %w", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(header, &decoded); err != nil {
		return nil, 0, fmt.Errorf("decode Safetensors header: %w", err)
	}
	return decoded, int64(8 + headerLength), nil
}
