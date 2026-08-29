// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestProbeAndExtractPDF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "document.pdf")
	writeTestPDF(t, path, "Hello from a PDF document.")
	plan := documentFormatPlan(t, path, "pdf")
	if got := plan.Inputs[0].Adapter; got != PDFTextAdapter {
		t.Fatalf("adapter = %q", got)
	}
	var batches []TextBatch
	if err := StreamCanonicalTextBatches(context.Background(), plan, func(batch TextBatch) error {
		batches = append(batches, batch)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(batches) != 1 || len(batches[0].Rows) != 1 || !strings.Contains(batches[0].Rows[0].Text, "Hello from a PDF document.") {
		t.Fatalf("batches = %+v", batches)
	}
	if batches[0].Rows[0].Meta == nil || !strings.Contains(*batches[0].Rows[0].Meta, `"pdf.title":"Fixture"`) || !strings.Contains(*batches[0].Rows[0].Meta, `"pdf.pages":"1"`) {
		t.Fatalf("metadata = %v", batches[0].Rows[0].Meta)
	}
	if profile := conversionProfile(plan); !strings.HasPrefix(profile, "canonical-document-text-v1@sha256:") {
		t.Fatalf("conversion profile = %q", profile)
	}
	assertDocumentAssembly(t, plan, "pdf")
}

func TestProbeAndExtractEPUBInSpineOrder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "book.epub")
	writeTestEPUB(t, path, map[string]string{
		"mimetype":               "application/epub+zip",
		"META-INF/container.xml": `<?xml version="1.0"?><container><rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles></container>`,
		"OPS/package.opf":        `<?xml version="1.0"?><package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>Fixture Book</dc:title><dc:creator>Example Author</dc:creator><dc:language>en</dc:language><dc:date>2026-08-29</dc:date><dc:rights>Copyright fixture</dc:rights></metadata><manifest><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/><item id="two" href="two.xhtml" media-type="application/xhtml+xml"/><item id="one" href="one.xhtml" media-type="application/xhtml+xml"/><item id="aux" href="aux.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="nav"/><itemref idref="one"/><itemref idref="aux" linear="no"/><itemref idref="two"/></spine></package>`,
		"OPS/nav.xhtml":          `<html><body><nav>Table of contents</nav></body></html>`,
		"OPS/one.xhtml":          `<html><head><style>hidden</style></head><body><h1>First</h1><p>Hello <b>EPUB</b>.</p></body></html>`,
		"OPS/two.xhtml":          `<html><body><h1>Second</h1><script>hidden()</script><p>In reading order.</p></body></html>`,
		"OPS/aux.xhtml":          `<html><body>Auxiliary text</body></html>`,
	})
	plan := documentFormatPlan(t, path, "epub")
	if got := plan.Inputs[0].Adapter; got != EPUBTextAdapter {
		t.Fatalf("adapter = %q", got)
	}
	var rowText, metadata string
	if err := StreamCanonicalTextBatches(context.Background(), plan, func(batch TextBatch) error {
		rowText = batch.Rows[0].Text
		metadata = *batch.Rows[0].Meta
		if batch.Rows[0].Language == nil || *batch.Rows[0].Language != "en" || batch.Rows[0].Date == nil || *batch.Rows[0].Date != "2026-08-29" {
			t.Fatalf("language/date = %v/%v", batch.Rows[0].Language, batch.Rows[0].Date)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(rowText, "First\nHello EPUB.\n\nSecond\nIn reading order.") || strings.Contains(rowText, "Table of contents") || strings.Contains(rowText, "Auxiliary") || strings.Contains(rowText, "hidden") {
		t.Fatalf("text = %q", rowText)
	}
	if !strings.Contains(metadata, `"epub.title":"Fixture Book"`) || !strings.Contains(metadata, `"epub.rights":"Copyright fixture"`) {
		t.Fatalf("metadata = %s", metadata)
	}
	assertDocumentAssembly(t, plan, "epub")
}

func TestPDFWithoutTextLayerFailsClosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "image-only.pdf")
	writeTestPDF(t, path, "")
	plan := documentFormatPlan(t, path, "pdf")
	if err := StreamCanonicalTextBatches(context.Background(), plan, func(TextBatch) error { return nil }); err == nil || !strings.Contains(err.Error(), "no extractable text layer") {
		t.Fatalf("error = %v", err)
	}
}

func TestPDF2FailsWithVersionGuidance(t *testing.T) {
	path := filepath.Join(t.TempDir(), "version-two.pdf")
	writeTestPDF(t, path, "PDF two fixture")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data = bytes.Replace(data, []byte("%PDF-1.4"), []byte("%PDF-2.0"), 1)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	plan := documentFormatPlan(t, path, "pdf")
	if err := StreamCanonicalTextBatches(context.Background(), plan, func(TextBatch) error { return nil }); err == nil || !strings.Contains(err.Error(), "PDF 2.x is not supported") {
		t.Fatalf("error = %v", err)
	}
}

func TestEPUBRejectsEscapingSpinePath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "unsafe.epub")
	writeTestEPUB(t, path, map[string]string{
		"mimetype":               "application/epub+zip",
		"META-INF/container.xml": `<container><rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles></container>`,
		"OPS/package.opf":        `<package><metadata/><manifest><item id="bad" href="../../outside.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="bad"/></spine></package>`,
	})
	plan := documentFormatPlan(t, path, "epub")
	if err := StreamCanonicalTextBatches(context.Background(), plan, func(TextBatch) error { return nil }); err == nil || !strings.Contains(err.Error(), "escapes the EPUB container") {
		t.Fatalf("error = %v", err)
	}
}

func TestProbeRejectsEPUBExtensionWithoutEPUBStructure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "not-an-epub.epub")
	writeTestEPUB(t, path, map[string]string{"hello.txt": "hello"})
	if _, err := ProbePaths(context.Background(), []string{path}); err == nil || !strings.Contains(err.Error(), "invalid EPUB") {
		t.Fatalf("error = %v", err)
	}
}

func documentFormatPlan(t *testing.T, path, format string) Plan {
	t.Helper()
	probe, err := ProbePaths(context.Background(), []string{path})
	if err != nil {
		t.Fatal(err)
	}
	if probe.Artifacts[0].Format != format {
		t.Fatalf("format = %q, want %q", probe.Artifacts[0].Format, format)
	}
	plan, err := NewPlan(probe, PlanRequest{
		Destination: "core/documents", Title: "Documents", License: "CC0-1.0",
		Source:  PlanSource{Name: "documents", URL: "https://example.test/documents", Category: "public-dataset"},
		Profile: InputProfile{Format: format},
	})
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func assertDocumentAssembly(t *testing.T, plan Plan, format string) {
	t.Helper()
	assembly, err := AssembleTextObjects(context.Background(), plan, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if assembly.InputDocs != 1 || assembly.RetainedDocs != 1 || len(assembly.Objects) != 1 || assembly.Objects[0].Docs != 1 {
		t.Fatalf("assembly = %+v", assembly)
	}
	manifest, err := BuildManifest(plan, assembly, "https://objects.example.test")
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Sources) != 1 || len(manifest.Sources[0].InputFormats) != 1 || manifest.Sources[0].InputFormats[0] != format || !strings.HasPrefix(manifest.ConvertedBy.Profile, "canonical-document-text-v1@sha256:") {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func writeTestEPUB(t *testing.T, destination string, entries map[string]string) {
	t.Helper()
	file, err := os.Create(destination)
	if err != nil {
		t.Fatal(err)
	}
	writer := zip.NewWriter(file)
	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		header := &zip.FileHeader{Name: name, Method: zip.Deflate}
		if name == "mimetype" {
			header.Method = zip.Store
		}
		entry, err := writer.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write([]byte(entries[name])); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeTestPDF(t *testing.T, destination, text string) {
	t.Helper()
	stream := fmt.Sprintf("BT /F1 12 Tf 72 720 Td (%s) Tj ET", strings.NewReplacer("\\", "\\\\", "(", "\\(", ")", "\\)").Replace(text))
	objects := []string{
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
		fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(stream), stream),
		"<< /Title (Fixture) /Author (Example Author) /CreationDate (D:20260829000000Z) >>",
	}
	var output bytes.Buffer
	output.WriteString("%PDF-1.4\n")
	offsets := make([]int, len(objects)+1)
	for position, object := range objects {
		offsets[position+1] = output.Len()
		fmt.Fprintf(&output, "%d 0 obj\n%s\nendobj\n", position+1, object)
	}
	xref := output.Len()
	fmt.Fprintf(&output, "xref\n0 %d\n0000000000 65535 f \n", len(objects)+1)
	for position := 1; position <= len(objects); position++ {
		fmt.Fprintf(&output, "%010d 00000 n \n", offsets[position])
	}
	fmt.Fprintf(&output, "trailer\n<< /Size %d /Root 1 0 R /Info 6 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objects)+1, xref)
	if err := os.WriteFile(destination, output.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
