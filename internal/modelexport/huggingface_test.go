// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package modelexport

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/model"
)

func TestLlamaPackageExportsRejectUnsupportedTokenizersBeforeWriting(t *testing.T) {
	formats := []struct {
		name   string
		export func(context.Context, model.Inspection, string, Options) (string, error)
	}{
		{name: "huggingface", export: ExportHuggingFace},
		{name: "mlx", export: ExportMLX},
	}
	tokenizers := []struct {
		name       string
		tokenizer  model.Tokenizer
		vocabulary uint64
	}{
		{name: "r50k", tokenizer: model.Tokenizer{Name: "tiktoken/r50k_base", Revision: "tiktoken-r50k-base"}, vocabulary: 50259},
		{name: "cl100k", tokenizer: model.Tokenizer{Name: "tiktoken/cl100k_base", Revision: "tiktoken-cl100k-base"}, vocabulary: 100259},
		{name: "wrong byte revision", tokenizer: model.Tokenizer{Name: "byte", Revision: "other"}, vocabulary: 259},
		{name: "wrong byte vocabulary", tokenizer: model.Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"}, vocabulary: 50259},
	}
	for _, format := range formats {
		for _, tokenizer := range tokenizers {
			t.Run(format.name+"/"+tokenizer.name, func(t *testing.T) {
				inspection := model.Inspection{Model: model.ModelRecord{
					Name: "fixture",
					Architecture: model.Architecture{
						Tokenizer: tokenizer.tokenizer, VocabularySize: tokenizer.vocabulary,
					},
				}}
				destination := filepath.Join(t.TempDir(), "export")
				_, err := format.export(context.Background(), inspection, destination, Options{})
				if err == nil || !strings.Contains(err.Error(), format.name+" export currently requires byte@builtin-byte-schema-1 with vocabulary_size 259") {
					t.Fatalf("export error = %v", err)
				}
				if _, err := os.Stat(destination); !os.IsNotExist(err) {
					t.Fatalf("export destination exists or cannot be checked: %v", err)
				}
			})
		}
	}
}

func TestLlamaPackageExportsContinuePastSupportedByteTokenizer(t *testing.T) {
	inspection := model.Inspection{Model: model.ModelRecord{
		Name: "fixture",
		Architecture: model.Architecture{
			Tokenizer:      model.Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"},
			VocabularySize: 259,
		},
	}}
	for _, export := range []func(context.Context, model.Inspection, string, Options) (string, error){ExportHuggingFace, ExportMLX} {
		_, err := export(context.Background(), inspection, filepath.Join(t.TempDir(), "export"), Options{})
		if err == nil || !strings.Contains(err.Error(), "no complete non-simulated run with usable weights") {
			t.Fatalf("supported byte tokenizer did not pass the compatibility check: %v", err)
		}
	}
}
