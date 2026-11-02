// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package modelweights

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"sort"
)

type mergeSource struct {
	file      *os.File
	tensors   map[string]Descriptor
	dataStart int64
}

// Merge writes the weighted mean of every source's tensors to destination.
// Sources must hold the same tensors with the same dtypes and shapes. Values
// accumulate in float32 in source order, rounding each product before it is
// added, so the same inputs always produce the same bytes.
func Merge(sources []string, weights []float64, destination string) (map[string]Descriptor, error) {
	if len(sources) < 2 {
		return nil, fmt.Errorf("a merge needs at least two sources")
	}
	if weights == nil {
		weights = make([]float64, len(sources))
		for index := range weights {
			weights[index] = 1
		}
	}
	if len(weights) != len(sources) {
		return nil, fmt.Errorf("a merge needs one weight per source; got %d for %d sources", len(weights), len(sources))
	}
	var total float64
	for _, weight := range weights {
		if !(weight > 0) || math.IsInf(weight, 0) {
			return nil, fmt.Errorf("merge weights must be positive and finite")
		}
		total += weight
	}
	opened := make([]mergeSource, 0, len(sources))
	defer func() {
		for _, source := range opened {
			_ = source.file.Close()
		}
	}()
	for _, path := range sources {
		source, err := openMergeSource(path)
		if err != nil {
			return nil, err
		}
		opened = append(opened, source)
	}
	names := make([]string, 0, len(opened[0].tensors))
	for name, descriptor := range opened[0].tensors {
		if descriptor.DType != "BF16" && descriptor.DType != "F32" {
			return nil, fmt.Errorf("tensor %s is %s; merges support BF16 and F32", name, descriptor.DType)
		}
		names = append(names, name)
	}
	sort.Strings(names)
	for index, source := range opened[1:] {
		if len(source.tensors) != len(names) {
			return nil, fmt.Errorf("%s has %d tensors; %s has %d", sources[index+1], len(source.tensors), sources[0], len(names))
		}
		for _, name := range names {
			want, got := opened[0].tensors[name], source.tensors[name]
			if got.DType != want.DType || !reflect.DeepEqual(got.Shape, want.Shape) {
				return nil, fmt.Errorf("%s tensor %s does not match %s in dtype and shape", sources[index+1], name, sources[0])
			}
		}
	}
	result := make(map[string]Descriptor, len(names))
	header := map[string]any{"__metadata__": map[string]string{"format": "pt", "merged_by": "openwaldo"}}
	var offset uint64
	for _, name := range names {
		descriptor := opened[0].tensors[name]
		length := descriptor.DataOffsets[1] - descriptor.DataOffsets[0]
		result[name] = Descriptor{descriptor.DType, descriptor.Shape, []uint64{offset, offset + length}}
		header[name] = result[name]
		offset += length
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
	// Python and MLX evaluate the same mean as float32(sum of weights) in float64.
	divisor := float32(total)
	for _, name := range names {
		descriptor := opened[0].tensors[name]
		width := uint64(2)
		if descriptor.DType == "F32" {
			width = 4
		}
		size := descriptor.DataOffsets[1] - descriptor.DataOffsets[0]
		sums := make([]float32, size/width)
		raw := make([]byte, size)
		for index, source := range opened {
			start := source.tensors[name].DataOffsets[0]
			if _, err := source.file.ReadAt(raw, source.dataStart+int64(start)); err != nil {
				return nil, fmt.Errorf("read %s tensor %s: %w", sources[index], name, err)
			}
			share := float32(weights[index])
			for element := range sums {
				// The explicit conversion stops Go from fusing the multiply into the add.
				sums[element] += float32(decodeFloat(raw, element, width) * share)
			}
		}
		for element, sum := range sums {
			encodeFloat(raw, element, width, sum/divisor)
		}
		if _, err := output.Write(raw); err != nil {
			return nil, fmt.Errorf("write tensor %s: %w", name, err)
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

func openMergeSource(path string) (mergeSource, error) {
	file, err := os.Open(path)
	if err != nil {
		return mergeSource{}, err
	}
	header, dataStart, err := readHeader(file)
	if err != nil {
		_ = file.Close()
		return mergeSource{}, fmt.Errorf("%s: %w", path, err)
	}
	info, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return mergeSource{}, err
	}
	dataBytes := uint64(info.Size() - dataStart)
	tensors := make(map[string]Descriptor, len(header))
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var descriptor Descriptor
		if err := json.Unmarshal(raw, &descriptor); err != nil {
			_ = file.Close()
			return mergeSource{}, fmt.Errorf("%s tensor %s: %w", path, name, err)
		}
		expected, err := tensorBytes(descriptor)
		if err != nil || len(descriptor.DataOffsets) != 2 || descriptor.DataOffsets[1] > dataBytes || descriptor.DataOffsets[1] < descriptor.DataOffsets[0] || descriptor.DataOffsets[1]-descriptor.DataOffsets[0] != expected {
			_ = file.Close()
			return mergeSource{}, fmt.Errorf("%s tensor %s has an invalid payload", path, name)
		}
		tensors[name] = descriptor
	}
	return mergeSource{file, tensors, dataStart}, nil
}

func decodeFloat(raw []byte, index int, width uint64) float32 {
	if width == 4 {
		return math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4:]))
	}
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[index*2:])) << 16)
}

func encodeFloat(raw []byte, index int, width uint64, value float32) {
	if width == 4 {
		binary.LittleEndian.PutUint32(raw[index*4:], math.Float32bits(value))
		return
	}
	binary.LittleEndian.PutUint16(raw[index*2:], bfloat16(value))
}

// bfloat16 rounds to nearest, ties to even, as MLX and PyTorch do.
func bfloat16(value float32) uint16 {
	if value != value {
		return 0x7fc0
	}
	bits := math.Float32bits(value)
	return uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
}
