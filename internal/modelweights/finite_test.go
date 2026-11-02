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
