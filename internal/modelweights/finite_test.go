// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package modelweights

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func tensorFixture(t *testing.T, header string, payload []byte) *os.File {
	t.Helper()
	raw := make([]byte, 8)
	binary.LittleEndian.PutUint64(raw, uint64(len(header)))
	raw = append(raw, header...)
	raw = append(raw, payload...)
	path := filepath.Join(t.TempDir(), "tensor.safetensors")
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}
func TestVerifyFiniteDtypes(t *testing.T) {
	for _, tt := range []struct {
		dtype string
		width int
		bad   uint64
	}{
		{"F8_E4M3", 1, 0x7f}, {"F8_E5M2", 1, 0xfc}, {"F16", 2, 0x7e00}, {"BF16", 2, 0xff80}, {"F32", 4, 0x7f800000}, {"F64", 8, 0xfff0000000000000},
	} {
		t.Run(tt.dtype, func(t *testing.T) {
			for _, bad := range []bool{false, true} {
				payload := make([]byte, 8)
				if bad {
					binary.LittleEndian.PutUint64(payload, tt.bad)
				}
				header := fmt.Sprintf(`{"x":{"dtype":%q,"shape":[1],"data_offsets":[0,%d]}}`, tt.dtype, tt.width)
				err := VerifyFinite(tensorFixture(t, header, payload[:tt.width]))
				if bad && !errors.Is(err, ErrNonFinite) || !bad && err != nil {
					t.Fatalf("bad=%v: %v", bad, err)
				}
			}
		})
	}
}
func TestVerifyFiniteMalformed(t *testing.T) {
	for _, header := range []string{
		`{}`, `null`, `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}`,
		`{"x":{"dtype":"F64","shape":[1],"data_offsets":[1,9]}}`,
		`{"x":{"dtype":"F64","shape":[1],"data_offsets":[0,8]},"x":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}`,
		`{"x":{"dtype":"F64","shape":[18446744073709551615,2],"data_offsets":[0,8]}}`,
		`{"x":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}} {}`,
	} {
		if err := VerifyFinite(tensorFixture(t, header, make([]byte, 8))); err == nil {
			t.Fatalf("accepted %s", header)
		}
	}
}

// Infinity (not only NaN) must be rejected, and the largest finite value of
// every floating dtype must be accepted, so exponent masks cannot drift.
func TestVerifyFiniteInfinityAndLargestFinite(t *testing.T) {
	for _, tt := range []struct {
		dtype     string
		width     int
		value     uint64
		nonfinite bool
	}{
		{"F16", 2, 0x7c00, true}, {"F16", 2, 0xfc00, true}, {"BF16", 2, 0x7f80, true}, {"BF16", 2, 0x7fc0, true}, {"F32", 4, 0x7fc00000, true}, {"F64", 8, 0x7ff8000000000000, true}, {"F8_E5M2", 1, 0x7c, true},
		{"F8_E4M3", 1, 0x7e, false}, {"F8_E5M2", 1, 0x7b, false}, {"F16", 2, 0x7bff, false}, {"F16", 2, 0xfbff, false}, {"BF16", 2, 0x7f7f, false}, {"F32", 4, 0x7f7fffff, false}, {"F64", 8, 0x7fefffffffffffff, false},
	} {
		payload := make([]byte, 8)
		binary.LittleEndian.PutUint64(payload, tt.value)
		header := fmt.Sprintf(`{"x":{"dtype":%q,"shape":[1],"data_offsets":[0,%d]}}`, tt.dtype, tt.width)
		err := VerifyFinite(tensorFixture(t, header, payload[:tt.width]))
		if tt.nonfinite && !errors.Is(err, ErrNonFinite) || !tt.nonfinite && err != nil {
			t.Fatalf("%s %#x: %v", tt.dtype, tt.value, err)
		}
	}
}

// A non-finite value after the first element and beyond the 64 KiB scan buffer
// must still be found.
func TestVerifyFiniteScansWholeTensor(t *testing.T) {
	for _, elements := range []int{2, 20000} {
		payload := make([]byte, 4*elements)
		binary.LittleEndian.PutUint32(payload[len(payload)-4:], 0x7f800000)
		header := fmt.Sprintf(`{"x":{"dtype":"F32","shape":[%d],"data_offsets":[0,%d]}}`, elements, len(payload))
		if err := VerifyFinite(tensorFixture(t, header, payload)); !errors.Is(err, ErrNonFinite) {
			t.Fatalf("%d elements: %v", elements, err)
		}
	}
}

// Payload layout must be dense: no leading gap, inter-tensor gap or unclaimed tail.
func TestVerifyFiniteRejectsPayloadGaps(t *testing.T) {
	for name, tc := range map[string]struct {
		header  string
		payload int
	}{
		"leading":   {`{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`, 8},
		"between":   {`{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}`, 12},
		"unclaimed": {`{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 8},
		"dtype":     {`{"x":{"dtype":"F12","shape":[1],"data_offsets":[0,4]}}`, 4},
	} {
		if err := VerifyFinite(tensorFixture(t, tc.header, make([]byte, tc.payload))); err == nil {
			t.Fatalf("%s: accepted %s", name, tc.header)
		}
	}
	ok := `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`
	if err := VerifyFinite(tensorFixture(t, ok, make([]byte, 8))); err != nil {
		t.Fatal(err)
	}
}
