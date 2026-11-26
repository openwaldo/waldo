// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package modelweights

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Native and external weights share this parser: scalars ("shape": []) stay
// valid, while a tensor without a shape, a non-object header and a header
// larger than the file are refused.
func TestInspectSafetensorsHeaderStrictness(t *testing.T) {
	scalar := tensorFixture(t, `{"__metadata__":{"format":"pt"},"model.norm.weight":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}  `, make([]byte, 4))
	descriptors, metadata, err := Inspect(scalar.Name())
	if err != nil || len(descriptors) != 1 || metadata["format"] != "pt" {
		t.Fatalf("scalar tensor: %v %v %v", descriptors, metadata, err)
	}
	valid := tensorFixture(t, `{"model.norm.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, make([]byte, 4))
	if _, err := NormalizeHuggingFace([]string{valid.Name()}, filepath.Join(t.TempDir(), "model.safetensors")); err != nil {
		t.Fatalf("valid normalization: %v", err)
	}
	for name, header := range map[string]string{
		"missing shape":     `{"model.norm.weight":{"dtype":"F32","data_offsets":[0,4]}}`,
		"null shape":        `{"model.norm.weight":{"dtype":"F32","shape":null,"data_offsets":[0,4]}}`,
		"leading space":     ` {"model.norm.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`,
		"duplicate tensor":  `{"model.norm.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"model.norm.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`,
		"trailing document": `{"model.norm.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}{}`,
	} {
		file := tensorFixture(t, header, make([]byte, 4))
		if _, _, err := Inspect(file.Name()); err == nil {
			t.Fatalf("%s accepted", name)
		}
		if _, err := NormalizeHuggingFace([]string{file.Name()}, filepath.Join(t.TempDir(), "model.safetensors")); err == nil {
			t.Fatalf("%s accepted by normalization", name)
		}
	}
	truncated := tensorFixture(t, `{"model.norm.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, make([]byte, 4))
	if err := os.Truncate(truncated.Name(), 20); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Inspect(truncated.Name()); err == nil || !strings.Contains(err.Error(), "exceeds file size") {
		t.Fatalf("truncated header: %v", err)
	}
}
