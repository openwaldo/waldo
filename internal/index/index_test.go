// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package index

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveSummarizeAndVerify(t *testing.T) {
	root := fixtureIndex(t)

	target, err := Resolve(filepath.Join(root, "alpha"), "")
	if err != nil {
		t.Fatal(err)
	}
	if target.Root != root {
		t.Fatalf("Resolve() root = %q, want %q", target.Root, root)
	}
	if target.Rel != "" {
		t.Fatalf("Resolve() rel = %q, want root", target.Rel)
	}

	alpha, err := Resolve(root, "alpha")
	if err != nil {
		t.Fatal(err)
	}
	totals, err := Summarize(alpha)
	if err != nil {
		t.Fatal(err)
	}
	if totals.Corpora != 1 || totals.Shards != 2 || totals.Docs != 5 || totals.Tokens != 50 || totals.Bytes != 500 {
		t.Fatalf("Summarize() = %+v", totals)
	}
	if totals.Licenses["CC0-1.0"].Tokens != 20 || totals.Licenses["CC-BY-4.0"].Tokens != 30 {
		t.Fatalf("Summarize() licenses = %+v", totals.Licenses)
	}
	corpora, err := ListCorpora(target)
	if err != nil {
		t.Fatal(err)
	}
	if len(corpora) != 1 || corpora[0].Path != "alpha/books" || corpora[0].Tokens != 50 || len(corpora[0].Licenses) != 2 {
		t.Fatalf("ListCorpora() = %+v", corpora)
	}

	verified, err := Verify(target)
	if err != nil {
		t.Fatal(err)
	}
	if verified.Directories != 2 || verified.Corpora != 1 || verified.Shards != 2 {
		t.Fatalf("Verify() = %+v", verified)
	}
}

func TestResolveRejectsPathOutsideCheckout(t *testing.T) {
	root := fixtureIndex(t)
	if _, err := Resolve(root, "../elsewhere"); err == nil || !strings.Contains(err.Error(), "outside index checkout") {
		t.Fatalf("Resolve() error = %v, want outside-checkout error", err)
	}
}

func TestResolveDiscoversCheckoutFromAbsoluteTargets(t *testing.T) {
	root := fixtureIndex(t)
	for _, test := range []struct {
		path string
		rel  string
	}{
		{path: root, rel: ""},
		{path: filepath.Join(root, "alpha"), rel: "alpha"},
		{path: filepath.Join(root, "alpha", "books.json"), rel: "alpha/books.json"},
	} {
		target, err := Resolve("", test.path)
		if err != nil {
			t.Fatal(err)
		}
		if target.Root != root || target.Rel != test.rel {
			t.Errorf("Resolve(%q) = root %q rel %q, want root %q rel %q", test.path, target.Root, target.Rel, root, test.rel)
		}
	}
}

func TestResolveAcceptsLogicalCorpusPathPrintedByList(t *testing.T) {
	root := fixtureIndex(t)
	for _, targetPath := range []string{"alpha/books", filepath.Join(root, "alpha", "books")} {
		target, err := Resolve(root, targetPath)
		if filepath.IsAbs(targetPath) {
			target, err = Resolve("", targetPath)
		}
		if err != nil {
			t.Fatal(err)
		}
		if target.Abs != filepath.Join(root, "alpha", "books.json") || target.Rel != "alpha/books.json" {
			t.Fatalf("Resolve(%q) = %+v", targetPath, target)
		}
	}
}

func TestResolveConfiguredDistinguishesLogicalAndFilesystemPaths(t *testing.T) {
	root := fixtureIndex(t)
	working := t.TempDir()
	t.Chdir(working)
	target, err := ResolveConfigured(root, "alpha")
	if err != nil {
		t.Fatal(err)
	}
	if target.Root != root || target.Rel != "alpha" {
		t.Fatalf("logical target = %+v", target)
	}

	local := fixtureIndex(t)
	t.Chdir(local)
	target, err = ResolveConfigured(root, "./alpha")
	if err != nil {
		t.Fatal(err)
	}
	if target.Root != local || target.Rel != "alpha" {
		t.Fatalf("explicit local target = %+v", target)
	}

	other := fixtureIndex(t)
	target, err = ResolveConfigured(root, filepath.Join(other, "alpha"))
	if err != nil {
		t.Fatal(err)
	}
	if target.Root != other || target.Rel != "alpha" {
		t.Fatalf("absolute override = %+v", target)
	}
}

func TestResolveDestinationNormalizesProspectiveAbsolutePath(t *testing.T) {
	root := fixtureIndex(t)
	target, err := ResolveDestination(filepath.Join(root, "alpha", "new", "corpus"))
	if err != nil {
		t.Fatal(err)
	}
	if target.Root != root || target.Rel != "alpha/new/corpus" || target.Abs != filepath.Join(root, "alpha", "new", "corpus") {
		t.Fatalf("ResolveDestination() = %+v", target)
	}
}

func TestResolveDestinationConfiguredUsesConfiguredCheckout(t *testing.T) {
	root := fixtureIndex(t)
	t.Chdir(t.TempDir())
	target, err := ResolveDestinationConfigured(root, "alpha/new/corpus")
	if err != nil {
		t.Fatal(err)
	}
	if target.Root != root || target.Rel != "alpha/new/corpus" || target.Abs != filepath.Join(root, "alpha", "new", "corpus") {
		t.Fatalf("ResolveDestinationConfigured() = %+v", target)
	}
	local := fixtureIndex(t)
	t.Chdir(local)
	target, err = ResolveDestinationConfigured(root, "./alpha/new/corpus")
	if err != nil {
		t.Fatal(err)
	}
	if target.Root != local || target.Rel != "alpha/new/corpus" || target.Abs != filepath.Join(local, "alpha", "new", "corpus") {
		t.Fatalf("local destination = %+v", target)
	}
}

func TestIsFilesystemPathRecognizesExistingRelativeParent(t *testing.T) {
	working := t.TempDir()
	t.Chdir(working)
	if err := os.Mkdir("checkout", 0o755); err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{"./missing", "../missing", "/missing", "checkout", "checkout/new"} {
		if !IsFilesystemPath(value) {
			t.Errorf("IsFilesystemPath(%q) = false", value)
		}
	}
	if IsFilesystemPath("logical/new") {
		t.Fatal("nonexistent unprefixed logical path treated as filesystem path")
	}
}

func TestVerifyRejectsUnsortedDirectory(t *testing.T) {
	root := fixtureIndex(t)
	writeFile(t, filepath.Join(root, "index.json"), `{
  "kind": "index", "schema": 1, "path": "",
  "entries": [
    {"name": "z", "type": "dir"},
    {"name": "alpha", "type": "dir"}
  ]
}`)
	target := Target{Root: root, Abs: root}
	if _, err := Verify(target); err == nil || !strings.Contains(err.Error(), "not sorted") {
		t.Fatalf("Verify() error = %v, want sorting error", err)
	}
}

func TestVerifyAcceptsPublicJSONDirectorySchemaTwo(t *testing.T) {
	root := fixtureIndex(t)
	writeFile(t, filepath.Join(root, "index.json"), `{
  "kind": "index", "schema": 2, "path": "",
  "entries": [{"name": "alpha", "type": "dir"}]
}`)
	target := Target{Root: root, Abs: root}
	directory, err := LoadDirectory(root)
	if err != nil {
		t.Fatal(err)
	}
	if directory.Schema != DirectorySchema {
		t.Fatalf("LoadDirectory() schema = %d, want normalized schema %d", directory.Schema, DirectorySchema)
	}
	verified, err := Verify(target)
	if err != nil {
		t.Fatal(err)
	}
	if verified.Directories != 2 || verified.Corpora != 1 {
		t.Fatalf("Verify() = %+v", verified)
	}
}

func TestVerifyRejectsUnknownDirectorySchema(t *testing.T) {
	root := fixtureIndex(t)
	writeFile(t, filepath.Join(root, "index.json"), `{
  "kind": "index", "schema": 3, "path": "",
  "entries": [{"name": "alpha", "type": "dir"}]
}`)
	target := Target{Root: root, Abs: root}
	if _, err := Verify(target); err == nil || !strings.Contains(err.Error(), "unsupported index schema 3") {
		t.Fatalf("Verify() error = %v, want unsupported schema error", err)
	}
}

func TestRollupManifestUsesPolymorphicShardsField(t *testing.T) {
	root := fixtureIndex(t)
	hash := strings.Repeat("d", 64)
	manifest := fmt.Sprintf(`{
  "kind": "manifest", "schema": 1, "name": "books",
  "title": "Books", "description": "Rolled-up books.", "license": "CC0-1.0",
  "sources": [{"name": "upstream", "source": "Example", "url": "https://example.test", "sha256": %q}],
  "converted_by": {"tool": "test", "version": "1", "profile": "text", "recipe": "test/v1", "tokenizer": "byte"},
  "shards": {"url": "https://objects.example/sub", "sha256": %q, "count": 2, "docs": 5, "tokens": 50, "bytes": 500}
}`, strings.Repeat("a", 64), hash)
	writeFile(t, filepath.Join(root, "alpha", "books.json"), manifest)
	target, err := Resolve(root, "alpha/books.json")
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadManifest(target.Abs)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Rollup == nil || loaded.Rollup.Count != 2 || len(loaded.Shards) != 0 {
		t.Fatalf("rollup manifest = %+v", loaded)
	}
	verified, err := Verify(target)
	if err != nil {
		t.Fatal(err)
	}
	if verified.Shards != 2 {
		t.Fatalf("verified rollup = %+v", verified)
	}
	totals, err := Summarize(target)
	if err != nil {
		t.Fatal(err)
	}
	if totals.Shards != 2 || totals.Docs != 5 || totals.Tokens != 50 || totals.Bytes != 500 {
		t.Fatalf("rollup totals = %+v", totals)
	}
}

func TestVerifyAcceptsAdditiveMultimodalProvenance(t *testing.T) {
	root := fixtureIndex(t)
	manifest := fmt.Sprintf(`{
  "kind": "manifest", "schema": 1, "name": "books",
  "title": "Images", "description": "Example image records.", "license": "CC0-1.0",
  "sources": [{
    "name": "upstream", "source": "Example", "url": "https://example.test", "sha256": %q,
    "category": "public-dataset",
    "usage": {"image": {"samples": 2, "items": 3, "content_bytes": 100}},
    "content": {"types": ["photography"], "copyrighted": "unknown"}
  }],
  "converted_by": {"tool": "test", "version": "1", "profile": "image", "recipe": "test/v2", "tokenizer": "none"},
  "processing": {
    "steps": [{"name": "validate", "description": "Validated media payloads."}],
    "rights_reservation_measures": ["Honoured recorded upstream exclusions."],
    "illegal_content_measures": ["Rejected payloads matching the configured blocklist."]
  },
  "record_schema": 1,
  "format": "parquet",
  "shards": [{
    "url": "https://example.test/a", "sha256": %q, "sources": ["upstream"],
    "docs": 2, "tokens": 0, "bytes": 200,
    "modalities": {"image": {"samples": 2, "items": 3, "content_bytes": 100}}
  }]
}`, strings.Repeat("a", 64), strings.Repeat("b", 64))
	path := filepath.Join(root, "alpha", "books.json")
	writeFile(t, path, manifest)
	target, err := Resolve(root, "alpha/books.json")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Verify(target); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.RecordSchema != 1 || loaded.Shards[0].Tokens != 0 || loaded.Shards[0].Modalities["image"].Items != 3 {
		t.Fatalf("multimodal manifest = %+v", loaded)
	}
}

func TestVerifyRejectsSourceUsageMismatch(t *testing.T) {
	root := fixtureIndex(t)
	manifest := fmt.Sprintf(`{
  "kind": "manifest", "schema": 1, "name": "books",
  "title": "Images", "description": "Example image records.", "license": "CC0-1.0",
  "sources": [{
    "name": "upstream", "source": "Example", "url": "https://example.test", "sha256": %q,
    "category": "public-dataset", "usage": {"image": {"samples": 1, "items": 1}}
  }],
  "converted_by": {"tool": "test", "version": "1", "profile": "image", "recipe": "test/v2", "tokenizer": "none"},
  "shards": [{
    "url": "https://example.test/a", "sha256": %q, "sources": ["upstream"],
    "docs": 2, "tokens": 0, "bytes": 200,
    "modalities": {"image": {"samples": 2, "items": 2}}
  }]
}`, strings.Repeat("a", 64), strings.Repeat("b", 64))
	path := filepath.Join(root, "alpha", "books.json")
	writeFile(t, path, manifest)
	target, err := Resolve(root, "alpha/books.json")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Verify(target); err == nil || !strings.Contains(err.Error(), "does not reconcile") {
		t.Fatalf("Verify() error = %v, want source-usage reconciliation error", err)
	}
}

func TestWebCrawlProvenanceRequiresCrawlerEvidence(t *testing.T) {
	source := Source{
		Name: "crawl", Category: SourceWebCrawl,
		Usage:       Modalities{"text": {Samples: 1, Tokens: 2}},
		Acquisition: &Acquisition{Domains: []DomainMeasure{{Domain: "example.com", AcquiredBytes: 20, RetainedBytes: 10}}},
	}
	if err := validateSourceProvenance(source); err == nil || !strings.Contains(err.Error(), "crawler details") {
		t.Fatalf("validateSourceProvenance() error = %v, want crawler error", err)
	}
	source.Acquisition.Crawler = &Crawler{Name: "waldo", Purpose: "Acquire public pages.", Behaviour: "Honours robots.txt.", Protocols: []string{"robots.txt"}}
	if err := validateSourceProvenance(source); err != nil {
		t.Fatal(err)
	}
}

func TestSourceProvenanceSeparatesLicenseAndPeriodEvidence(t *testing.T) {
	source := Source{
		Name: "dataset", Category: SourcePublicDataset,
		License: "Apache-2.0",
		LicenseEvidence: &LicenseEvidence{
			Declaration: "Apache License, Version 2.0",
			URL:         "https://example.test/LICENSE",
		},
		CollectedFrom: "2026-08-08T10:00:00Z",
		CollectedTo:   "2026-08-08T10:05:00Z",
		Content: &Content{
			Types: []string{"text"}, Languages: []string{"en"},
			From: "2020", To: "2025-12", Selection: "Pinned train split.",
		},
	}
	if err := ValidateSourceProvenance(source); err != nil {
		t.Fatal(err)
	}
	source.LicenseEvidence.URL = "relative/LICENSE"
	if err := ValidateSourceProvenance(source); err == nil || !strings.Contains(err.Error(), "absolute URL") {
		t.Fatalf("license evidence error = %v", err)
	}
}

func TestPrivateSourceRequiresAcquisitionBasis(t *testing.T) {
	source := Source{Name: "private", Category: SourcePrivateThirdParty}
	if err := ValidateSourceProvenance(source); err == nil || !strings.Contains(err.Error(), "acquisition details") {
		t.Fatalf("missing acquisition error = %v", err)
	}
	source.Acquisition = &Acquisition{Basis: "Private data-sharing agreement; terms withheld."}
	if err := ValidateSourceProvenance(source); err != nil {
		t.Fatal(err)
	}
}

func TestPublicIndexAcceptance(t *testing.T) {
	path := os.Getenv("WALDO_ACCEPTANCE_INDEX")
	if path == "" {
		t.Skip("set WALDO_ACCEPTANCE_INDEX to an absolute public-checkout path to run acceptance tests")
	}
	if !filepath.IsAbs(path) {
		t.Fatalf("WALDO_ACCEPTANCE_INDEX must be absolute, got %q", path)
	}
	target, err := Resolve(path, "")
	if err != nil {
		t.Fatal(err)
	}
	verified, err := Verify(target)
	if err != nil {
		t.Fatal(err)
	}
	if verified.Corpora != 20 || verified.Shards != 1087 {
		t.Fatalf("public index shape changed: %+v", verified)
	}
	totals, err := Summarize(target)
	if err != nil {
		t.Fatal(err)
	}
	if totals.Docs != 75_122_304 || totals.Tokens != 124_010_554_159 || totals.Bytes != 169_363_482_410 {
		t.Fatalf("public index totals changed: %+v", totals)
	}
}

func fixtureIndex(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "index.json"), `{
  "kind": "index", "schema": 1, "path": "",
  "entries": [{"name": "alpha", "type": "dir"}]
}`)
	writeFile(t, filepath.Join(root, "alpha", "index.json"), `{
  "kind": "index", "schema": 1, "path": "alpha",
  "entries": [{"name": "books.json", "type": "manifest"}]
}`)
	hashA := strings.Repeat("a", 64)
	hashB := strings.Repeat("b", 64)
	hashC := strings.Repeat("c", 64)
	manifest := fmt.Sprintf(`{
  "kind": "manifest", "schema": 1, "name": "books",
  "title": "Books", "description": "Example books.", "license": "CC0-1.0",
  "sources": [{"name": "upstream", "source": "Example", "url": "https://example.test", "sha256": %q}],
  "converted_by": {"tool": "test", "version": "1", "profile": "text", "recipe": "test/v1", "tokenizer": "byte"},
  "shards": [
    {"url": "https://example.test/a", "sha256": %q, "sources": ["upstream"], "docs": 2, "tokens": 20, "bytes": 200},
    {"url": "https://example.test/b", "sha256": %q, "license": "CC-BY-4.0", "sources": ["upstream"], "docs": 3, "tokens": 30, "bytes": 300}
  ]
}`, hashA, hashB, hashC)
	writeFile(t, filepath.Join(root, "alpha", "books.json"), manifest)
	return root
}

func TestContentHashPathRecognizesCanonicalObjectNames(t *testing.T) {
	digest := strings.Repeat("a", 64)
	if object, ok := contentHashPath("s3://bucket/lookaside/aa/aa/" + digest); !ok || object != digest {
		t.Fatalf("contentHashPath() = %q, %t", object, ok)
	}
	if _, ok := contentHashPath("https://objects.example/shard.parquet"); ok {
		t.Fatal("ordinary object name was treated as a content hash")
	}
	if object, ok := contentHashPath("s3://bucket/lookaside/aa/aa/" + digest[:62]); !ok || object == digest {
		t.Fatalf("truncated contentHashPath() = %q, %t", object, ok)
	}
}

func writeFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestIndexWalkersRejectUnsafeEntries(t *testing.T) {
	tests := []struct {
		name  string
		setup func(t *testing.T, root string)
		want  string
	}{
		{
			name: "current directory entry",
			setup: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, "alpha", "index.json"), `{
  "kind": "index", "schema": 1, "path": "alpha",
  "entries": [{"name": ".", "type": "dir"}, {"name": "books.json", "type": "manifest"}]
}`)
			},
			want: `invalid entry name "."`,
		},
		{
			name: "parent traversal entry",
			setup: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(filepath.Dir(root), "outside.json"), `{"kind": "manifest", "schema": 1, "name": "outside"}`)
				writeFile(t, filepath.Join(root, "alpha", "index.json"), `{
  "kind": "index", "schema": 1, "path": "alpha",
  "entries": [{"name": "../../outside.json", "type": "manifest"}]
}`)
			},
			want: `invalid entry name "../../outside.json"`,
		},
		{
			name: "symbolic link directory",
			setup: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, "index.json"), `{
  "kind": "index", "schema": 1, "path": "",
  "entries": [{"name": "alpha", "type": "dir"}, {"name": "loop", "type": "dir"}]
}`)
				if err := os.Symlink(".", filepath.Join(root, "loop")); err != nil {
					t.Fatal(err)
				}
			},
			want: `indexed entry "loop" is a symbolic link`,
		},
		{
			name: "symbolic link manifest",
			setup: func(t *testing.T, root string) {
				outside := filepath.Join(t.TempDir(), "books.json")
				data, err := os.ReadFile(filepath.Join(root, "alpha", "books.json"))
				if err != nil {
					t.Fatal(err)
				}
				writeFile(t, outside, string(data))
				if err := os.Remove(filepath.Join(root, "alpha", "books.json")); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(outside, filepath.Join(root, "alpha", "books.json")); err != nil {
					t.Fatal(err)
				}
			},
			want: `indexed entry "books.json" is a symbolic link`,
		},
		{
			name: "symbolic link index metadata",
			setup: func(t *testing.T, root string) {
				outside := filepath.Join(t.TempDir(), "index.json")
				writeFile(t, outside, `{"kind": "index", "schema": 1, "path": "alpha", "entries": []}`)
				if err := os.Remove(filepath.Join(root, "alpha", "index.json")); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(outside, filepath.Join(root, "alpha", "index.json")); err != nil {
					t.Fatal(err)
				}
			},
			want: "index metadata must be a regular file",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := fixtureIndex(t)
			test.setup(t, root)
			target := Target{Root: root, Abs: root}
			err := WalkCorpora(target, func(Corpus) error { return nil })
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("WalkCorpora() error = %v, want %q", err, test.want)
			}
			if _, err := Verify(target); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Verify() error = %v, want %q", err, test.want)
			}
		})
	}
}
