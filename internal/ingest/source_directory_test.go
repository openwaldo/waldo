// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadSourceDirectoryUsesOnlyRecursiveRawInputs(t *testing.T) {
	root := t.TempDir()
	raw := filepath.Join(root, "raw")
	if err := os.MkdirAll(filepath.Join(raw, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	content := []byte("raw record\n")
	writeProbeFile(t, filepath.Join(raw, "nested", "record.txt"), string(content))
	fileHash := sha256.Sum256(content)
	inventory := fmt.Sprintf("%x\t%d\tnested/record.txt\n", fileHash, len(content))
	treeHash := sha256.Sum256([]byte(inventory))
	manifest := fmt.Sprintf(`{
  "kind":"waldo-source-directory",
  "schema":1,
  "retrieved_at":"2026-08-19T00:00:00Z",
  "corpus":{"id":"example","title":"Example","description":"Recursive fixture."},
  "sources":[{
    "id":"example","path":"","license":"CC0-1.0",
    "source":{"name":"Example","url":"https://example.test/data","category":"public-dataset","license_evidence":{"declaration":"CC0-1.0"}},
    "input":{"format":"text"},"artifacts":[{"url":"https://example.test/data/record.txt","path":"nested/record.txt"}]
  }],
  "fetcher":{"script":"corpora/example.sh","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","dirty":false},
  "raw":{"path":"raw","file_count":1,"byte_count":%d,"tree_sha256":"%x"}
}`, len(content), treeHash)
	if err := os.WriteFile(filepath.Join(root, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}

	loaded, ok, err := LoadSourceDirectory(root)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("source directory was not recognized")
	}
	inputs := loaded.InputPaths()
	if len(inputs) != 1 || inputs[0] != raw {
		t.Fatalf("InputPaths() = %v, want [%s]", inputs, raw)
	}
	probe, err := ProbePaths(context.Background(), inputs)
	if err != nil {
		t.Fatal(err)
	}
	if probe.Totals.Artifacts != 1 || probe.Artifacts[0].Path != filepath.Join(raw, "nested", "record.txt") {
		t.Fatalf("probe = %+v", probe)
	}
	if err := loaded.VerifyProbe(probe); err != nil {
		t.Fatal(err)
	}
	request := PlanRequest{Destination: "tests/explicit"}
	loaded.Apply(&request)
	if request.Destination != "tests/explicit" || len(request.Sources) != 1 || request.Sources[0].InputRoot != raw {
		t.Fatalf("request = %+v", request)
	}
}

func TestLoadSourceDirectoryRejectsUndeclaredRootFile(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "raw"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeProbeFile(t, filepath.Join(root, "manifest.json"), `{"kind":"waldo-source-directory","schema":1}`)
	_, ok, err := LoadSourceDirectory(root)
	if !ok || err == nil {
		t.Fatalf("LoadSourceDirectory() = ok %v, err %v", ok, err)
	}
}

func TestSourceDirectoryAcceptsPDFAndEPUBFormats(t *testing.T) {
	for _, format := range []string{"pdf", "epub"} {
		t.Run(format, func(t *testing.T) {
			source := SourceDirectorySource{
				ID: "documents", License: "CC0-1.0",
				Source: SourceMetadata{URL: "https://example.test/documents", Category: "public-dataset"},
				Input:  InputProfile{Format: format},
			}
			if err := validateSourceDirectorySource(source); err != nil {
				t.Fatal(err)
			}
		})
	}
}
