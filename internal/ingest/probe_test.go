// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/parquet-go/parquet-go"
)

type probeRow struct {
	Text   string `parquet:"text"`
	Source string `parquet:"source"`
}

func TestProbePathsDetectsAndOrdersArtifacts(t *testing.T) {
	root := t.TempDir()
	writeProbeFile(t, filepath.Join(root, "b.md"), "# Heading\n\nMarkdown body.\n")
	writeProbeFile(t, filepath.Join(root, "a.txt"), "plain text\n")
	parquetPath := filepath.Join(root, "c.parquet")
	if err := parquet.WriteFile(parquetPath, []probeRow{{Text: "hello", Source: "fixture"}}); err != nil {
		t.Fatal(err)
	}

	probe, err := ProbePaths(context.Background(), []string{root})
	if err != nil {
		t.Fatal(err)
	}
	if probe.Kind != "waldo-ingest-probe" || probe.Schema != 1 || probe.Totals.Artifacts != 3 {
		t.Fatalf("ProbePaths() = %+v", probe)
	}
	formats := []string{probe.Artifacts[0].Format, probe.Artifacts[1].Format, probe.Artifacts[2].Format}
	if strings.Join(formats, ",") != "text,markdown,parquet" {
		t.Fatalf("formats = %v", formats)
	}
	parquetArtifact := probe.Artifacts[2]
	if parquetArtifact.Parquet == nil || parquetArtifact.Parquet.Rows != 1 || parquetArtifact.Parquet.RowGroups != 1 {
		t.Fatalf("Parquet probe = %+v", parquetArtifact.Parquet)
	}
	if strings.Join(parquetArtifact.Parquet.Columns, ",") != "text,source" {
		t.Fatalf("Parquet columns = %v", parquetArtifact.Parquet.Columns)
	}
	for i := 1; i < len(probe.Artifacts); i++ {
		if probe.Artifacts[i-1].Path >= probe.Artifacts[i].Path {
			t.Fatalf("artifacts are not sorted: %+v", probe.Artifacts)
		}
	}
}

func TestProbePathsUsesSignaturesBeforeExtensions(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "not-really-markdown.md")
	writeProbeFile(t, path, string([]byte{0x1f, 0x8b, 0x08, 0x00}))
	probe, err := ProbePaths(context.Background(), []string{path})
	if err != nil {
		t.Fatal(err)
	}
	artifact := probe.Artifacts[0]
	if artifact.Format != "compressed" || artifact.Compression != "gzip" {
		t.Fatalf("artifact = %+v", artifact)
	}
}

func TestProbePathsAllowsUTF8RuneSplitAtSampleBoundary(t *testing.T) {
	path := filepath.Join(t.TempDir(), "large.txt")
	contents := strings.Repeat("a", probeBytes-1) + "é" + "tail"
	writeProbeFile(t, path, contents)
	probe, err := ProbePaths(context.Background(), []string{path})
	if err != nil {
		t.Fatal(err)
	}
	if probe.Artifacts[0].Format != "text" {
		t.Fatalf("format = %q, want text", probe.Artifacts[0].Format)
	}
}

func TestProbePathsDetectsJSONObjectLargerThanSample(t *testing.T) {
	path := filepath.Join(t.TempDir(), "large.json")
	contents := `{"text":"` + strings.Repeat("a", probeBytes+1) + `"}`
	writeProbeFile(t, path, contents)

	probe, err := ProbePaths(context.Background(), []string{path})
	if err != nil {
		t.Fatal(err)
	}
	artifact := probe.Artifacts[0]
	if artifact.Format != "json" || !slices.Contains(artifact.Evidence, "extension-hint:.json") {
		t.Fatalf("artifact = %+v", artifact)
	}
}

func TestNewPlanPinsMappingsAndIdentity(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "input.parquet")
	if err := parquet.WriteFile(path, []probeRow{{Text: "hello", Source: "fixture"}}); err != nil {
		t.Fatal(err)
	}
	probe, err := ProbePaths(context.Background(), []string{path})
	if err != nil {
		t.Fatal(err)
	}
	request := PlanRequest{
		Destination: "core/example", Title: "Example", License: "CC0-1.0", TextColumn: "text",
		Source: PlanSource{Name: "example", URL: "https://example.test", Category: "public-dataset"},
	}
	plan, err := NewPlan(probe, request)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Inputs[0].Adapter != "parquet" || plan.Inputs[0].TextColumn != "text" || plan.MemoryBytes != 2<<30 {
		t.Fatalf("NewPlan() = %+v", plan)
	}
	first, err := plan.Identity()
	if err != nil {
		t.Fatal(err)
	}
	second, err := plan.Identity()
	if err != nil {
		t.Fatal(err)
	}
	if first != second || len(first) != 64 {
		t.Fatalf("identities = %q, %q", first, second)
	}
	plan.Mode = "canonical"
	third, err := plan.Identity()
	if err != nil {
		t.Fatal(err)
	}
	if third == first {
		t.Fatal("plan identity did not change with mode")
	}
}

func TestNewPlanRejectsParquetWithoutMapping(t *testing.T) {
	type ambiguousRow struct {
		Text    string `parquet:"text"`
		Content string `parquet:"content"`
	}
	path := filepath.Join(t.TempDir(), "ambiguous.parquet")
	if err := parquet.WriteFile(path, []ambiguousRow{{Text: "a", Content: "b"}}); err != nil {
		t.Fatal(err)
	}
	probe, err := ProbePaths(context.Background(), []string{path})
	if err != nil {
		t.Fatal(err)
	}
	request := PlanRequest{
		Destination: "core/example", Title: "Example", License: "CC0-1.0",
		Source: PlanSource{Name: "example", URL: "https://example.test", Category: "public-dataset"},
	}
	if _, err := NewPlan(probe, request); err == nil || !strings.Contains(err.Error(), "Parquet input requires a record input profile or an explicit text column") {
		t.Fatalf("error = %v", err)
	}
	request.TextColumn = "content"
	if _, err := NewPlan(probe, request); err != nil {
		t.Fatal(err)
	}
}

func TestNewPlanForceFormatPinsOverride(t *testing.T) {
	path := filepath.Join(t.TempDir(), "unknown.bin")
	if err := os.WriteFile(path, []byte{0, 1, 2}, 0o644); err != nil {
		t.Fatal(err)
	}
	probe, err := ProbePaths(context.Background(), []string{path})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewPlan(probe, PlanRequest{
		Destination: "core/example", Title: "Example", License: "CC0-1.0", ForceFormat: "text",
		Source: PlanSource{Name: "example", URL: "https://example.test", Category: "public-dataset"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Inputs) != 1 || plan.Inputs[0].DetectedFormat != "unknown" || plan.Inputs[0].Artifact.Format != "text" || plan.Inputs[0].Adapter != "text" {
		t.Fatalf("forced plan = %+v", plan)
	}
	if err := StreamCanonicalTextBatches(context.Background(), plan, func(TextBatch) error { return nil }); err == nil {
		t.Fatal("forced text adapter did not validate invalid UTF-8/NUL content")
	}
}

func TestNewPlanRejectsUnknownForceFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input.txt")
	writeProbeFile(t, path, "text")
	probe, err := ProbePaths(context.Background(), []string{path})
	if err != nil {
		t.Fatal(err)
	}
	_, err = NewPlan(probe, PlanRequest{
		Destination: "core/example", Title: "Example", License: "CC0-1.0", ForceFormat: "imaginary",
		Source: PlanSource{Name: "example", URL: "https://example.test", Category: "public-dataset"},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported --force-format") {
		t.Fatalf("error = %v", err)
	}
}

func TestNewPlanRejectsManifestFormatMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input.txt")
	writeProbeFile(t, path, "plain text\n")
	probe, err := ProbePaths(context.Background(), []string{path})
	if err != nil {
		t.Fatal(err)
	}
	_, err = NewPlan(probe, PlanRequest{
		Destination: "core/example", Title: "Example", License: "CC0-1.0",
		Source:  PlanSource{Name: "example", URL: "https://example.test", Category: "public-dataset"},
		Profile: InputProfile{Format: "jsonl", Type: ProfileRecordMap, Fields: ProfileFields{Text: []string{"text"}}},
	})
	if err == nil || !strings.Contains(err.Error(), `ingestion manifest declares input format "jsonl" but WALDO detected "text"`) || !strings.Contains(err.Error(), "correct the acquisition manifest") {
		t.Fatalf("error = %v", err)
	}
}

func TestNewPlanRejectsCategoryWithoutRequiredAcquisitionEvidence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input.txt")
	writeProbeFile(t, path, "text")
	probe, err := ProbePaths(context.Background(), []string{path})
	if err != nil {
		t.Fatal(err)
	}
	_, err = NewPlan(probe, PlanRequest{
		Destination: "core/example", Title: "Example", License: "CC0-1.0",
		Source: PlanSource{Name: "crawl", URL: "https://example.test", Category: "web-crawl"},
	})
	if err == nil {
		t.Fatal("expected missing acquisition evidence rejection")
	}
}

func TestProbePathsRejectsSymlinks(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "target.txt")
	writeProbeFile(t, target, "text")
	link := filepath.Join(root, "link.txt")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	if _, err := ProbePaths(context.Background(), []string{link}); err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("ProbePaths() error = %v, want symlink error", err)
	}
}

func writeProbeFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}
