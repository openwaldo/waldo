// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package modelweights

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fixtureTensor struct {
	dtype  string
	shape  []uint64
	values []float32
}

func writeFixture(t *testing.T, tensors map[string]fixtureTensor) string {
	t.Helper()
	header := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var payload []byte
	for _, name := range []string{"a.weight", "b.weight"} {
		tensor, ok := tensors[name]
		if !ok {
			continue
		}
		width := uint64(2)
		if tensor.dtype == "F32" {
			width = 4
		}
		raw := make([]byte, uint64(len(tensor.values))*width)
		for index, value := range tensor.values {
			if width == 4 {
				binary.LittleEndian.PutUint32(raw[index*4:], math.Float32bits(value))
			} else {
				binary.LittleEndian.PutUint16(raw[index*2:], bfloat16(value))
			}
		}
		header[name] = Descriptor{tensor.dtype, tensor.shape, []uint64{uint64(len(payload)), uint64(len(payload) + len(raw))}}
		payload = append(payload, raw...)
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(encoded)))
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, append(append(length[:], encoded...), payload...), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func readFixture(t *testing.T, path, name string) []float32 {
	t.Helper()
	source, err := openMergeSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer source.file.Close()
	descriptor := source.tensors[name]
	width := uint64(2)
	if descriptor.DType == "F32" {
		width = 4
	}
	raw := make([]byte, descriptor.DataOffsets[1]-descriptor.DataOffsets[0])
	if _, err := source.file.ReadAt(raw, source.dataStart+int64(descriptor.DataOffsets[0])); err != nil {
		t.Fatal(err)
	}
	values := make([]float32, uint64(len(raw))/width)
	for index := range values {
		values[index] = decodeFloat(raw, index, width)
	}
	return values
}

func TestMergeAveragesEveryTensor(t *testing.T) {
	first := writeFixture(t, map[string]fixtureTensor{"a.weight": {"BF16", []uint64{2}, []float32{1, 2}}, "b.weight": {"F32", []uint64{1}, []float32{0.5}}})
	second := writeFixture(t, map[string]fixtureTensor{"a.weight": {"BF16", []uint64{2}, []float32{3, 4}}, "b.weight": {"F32", []uint64{1}, []float32{1.5}}})
	destination := filepath.Join(t.TempDir(), "merged.safetensors")
	descriptors, err := Merge([]string{first, second}, nil, destination)
	if err != nil {
		t.Fatal(err)
	}
	if len(descriptors) != 2 || descriptors["a.weight"].DType != "BF16" || descriptors["b.weight"].DType != "F32" {
		t.Fatalf("descriptors = %+v", descriptors)
	}
	if got := readFixture(t, destination, "a.weight"); got[0] != 2 || got[1] != 3 {
		t.Fatalf("a.weight = %v", got)
	}
	if got := readFixture(t, destination, "b.weight"); got[0] != 1 {
		t.Fatalf("b.weight = %v", got)
	}
}

func TestMergeWeightsSources(t *testing.T) {
	first := writeFixture(t, map[string]fixtureTensor{"a.weight": {"F32", []uint64{1}, []float32{1}}})
	second := writeFixture(t, map[string]fixtureTensor{"a.weight": {"F32", []uint64{1}, []float32{4}}})
	destination := filepath.Join(t.TempDir(), "merged.safetensors")
	if _, err := Merge([]string{first, second}, []float64{1, 3}, destination); err != nil {
		t.Fatal(err)
	}
	if got := readFixture(t, destination, "a.weight"); got[0] != 3.25 {
		t.Fatalf("a.weight = %v", got)
	}
}

func TestBFloat16RoundsToNearestEven(t *testing.T) {
	for value, want := range map[float32]uint16{1: 0x3f80, 1 + 1.0/256: 0x3f80, 1 + 3.0/256: 0x3f82, 1 + 1.5/256: 0x3f81, -2: 0xc000, float32(math.Inf(1)): 0x7f80} {
		if got := bfloat16(value); got != want {
			t.Fatalf("bfloat16(%v) = %#x, want %#x", value, got, want)
		}
	}
	if got := bfloat16(float32(math.NaN())); got != 0x7fc0 {
		t.Fatalf("bfloat16(NaN) = %#x", got)
	}
}

func TestMergeRejectsIncompatibleSources(t *testing.T) {
	base := writeFixture(t, map[string]fixtureTensor{"a.weight": {"BF16", []uint64{2}, []float32{1, 2}}})
	for name, test := range map[string]struct {
		other   string
		weights []float64
		want    string
	}{
		"shape":   {writeFixture(t, map[string]fixtureTensor{"a.weight": {"BF16", []uint64{1, 2}, []float32{1, 2}}}), nil, "dtype and shape"},
		"dtype":   {writeFixture(t, map[string]fixtureTensor{"a.weight": {"F32", []uint64{2}, []float32{1, 2}}}), nil, "dtype and shape"},
		"tensors": {writeFixture(t, map[string]fixtureTensor{"a.weight": {"BF16", []uint64{2}, []float32{1, 2}}, "b.weight": {"BF16", []uint64{1}, []float32{1}}}), nil, "tensors"},
		"weights": {base, []float64{1, 0}, "positive"},
	} {
		destination := filepath.Join(t.TempDir(), "merged.safetensors")
		if _, err := Merge([]string{base, test.other}, test.weights, destination); err == nil || !strings.Contains(err.Error(), test.want) {
			t.Fatalf("%s: error = %v", name, err)
		}
		if _, err := os.Stat(destination); !os.IsNotExist(err) {
			t.Fatalf("%s: merge left %s behind", name, destination)
		}
	}
	if _, err := Merge([]string{base}, nil, filepath.Join(t.TempDir(), "merged.safetensors")); err == nil {
		t.Fatal("a single source merged")
	}
}
