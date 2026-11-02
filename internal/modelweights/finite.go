// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package modelweights

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

var ErrNonFinite = errors.New("non-finite tensor")

// VerifyFinite checks the complete file structure and scans floating tensors
// with bounded memory. It never executes code or materializes whole tensors.
func VerifyFinite(file *os.File) error {
	tensors, _, err := inspectFile(file)
	if err != nil {
		return err
	}
	if len(tensors) == 0 {
		return fmt.Errorf("Safetensors file has no tensors")
	}
	if _, err = file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	_, start, err := readHeader(file)
	if err != nil {
		return err
	}
	buffer := make([]byte, 64<<10)
	for name, d := range tensors {
		width := map[string]int{"F8_E4M3": 1, "F8_E5M2": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}[d.DType]
		if width == 0 {
			continue
		}
		r := io.NewSectionReader(file, start+int64(d.DataOffsets[0]), int64(d.DataOffsets[1]-d.DataOffsets[0]))
		remaining := d.DataOffsets[1] - d.DataOffsets[0]
		for remaining > 0 {
			n := min(uint64(len(buffer)), remaining)
			if _, err := io.ReadFull(r, buffer[:n]); err != nil {
				return err
			}
			for i := 0; i < int(n); i += width {
				var bad bool
				switch d.DType {
				case "F8_E4M3":
					bad = buffer[i]&0x7f == 0x7f
				case "F8_E5M2":
					bad = buffer[i]&0x7c == 0x7c
				case "F16":
					bad = binary.LittleEndian.Uint16(buffer[i:])&0x7c00 == 0x7c00
				case "BF16":
					bad = binary.LittleEndian.Uint16(buffer[i:])&0x7f80 == 0x7f80
				case "F32":
					bad = binary.LittleEndian.Uint32(buffer[i:])&0x7f800000 == 0x7f800000
				case "F64":
					bad = binary.LittleEndian.Uint64(buffer[i:])&0x7ff0000000000000 == 0x7ff0000000000000
				}
				if bad {
					return fmt.Errorf("%w: %s", ErrNonFinite, name)
				}
			}
			remaining -= n
		}
	}
	return nil
}
