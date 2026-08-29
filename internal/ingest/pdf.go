// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	pdf "github.com/ledongthuc/pdf"
	"github.com/openwaldo/waldo/internal/shard"
)

const PDFTextAdapter = "pdf-text-ledongthuc-5959a4027728-v1"

// StreamPDFTextBatches extracts the embedded text layer from each PDF as one
// logical document. It never renders pages, performs OCR, or invokes an
// external program.
func StreamPDFTextBatches(ctx context.Context, plan Plan, consume func(TextBatch) error) error {
	if err := plan.Validate(); err != nil {
		return err
	}
	if consume == nil {
		return fmt.Errorf("PDF batch consumer is required")
	}
	for _, input := range plan.Inputs {
		if input.Adapter != PDFTextAdapter {
			return fmt.Errorf("input %s requires the %s adapter, not the PDF adapter", input.Artifact.Path, input.Adapter)
		}
		row, err := extractPDFText(ctx, plan, input)
		if err != nil {
			return fmt.Errorf("adapt %s: %w", input.Artifact.Path, err)
		}
		if err := consume(TextBatch{Rows: []shard.TextRow{row}, LogicalBytes: int64(len(row.Text)), InputBytes: input.Artifact.Bytes}); err != nil {
			return err
		}
	}
	return nil
}

func extractPDFText(ctx context.Context, plan Plan, input PlanInput) (row shard.TextRow, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("PDF parser failed safely: %v", recovered)
		}
	}()
	file, verified, err := openVerifiedInput(ctx, input.Artifact)
	if err != nil {
		return shard.TextRow{}, err
	}
	defer file.Close()
	header := make([]byte, 8)
	if _, err := file.ReadAt(header, 0); err != nil {
		return shard.TextRow{}, fmt.Errorf("read PDF header: %w", err)
	}
	if string(header[:7]) == "%PDF-2." {
		return shard.TextRow{}, fmt.Errorf("PDF 2.x is not supported by %s; supported text-layer versions are PDF 1.0 through 1.7", PDFTextAdapter)
	}
	reader, err := pdf.NewReader(file, input.Artifact.Bytes)
	if err != nil {
		message := strings.ToLower(err.Error())
		if strings.Contains(message, "encrypt") || strings.Contains(message, "password") {
			return shard.TextRow{}, fmt.Errorf("encrypted PDFs are not supported")
		}
		return shard.TextRow{}, fmt.Errorf("parse PDF: %w", err)
	}
	if !reader.Trailer().Key("Encrypt").IsNull() {
		return shard.TextRow{}, fmt.Errorf("encrypted PDFs are not supported")
	}
	pages := reader.NumPage()
	if pages <= 0 || pages > 100_000 {
		return shard.TextRow{}, fmt.Errorf("PDF page count %d is outside the supported range", pages)
	}
	var document strings.Builder
	for pageNumber := 1; pageNumber <= pages; pageNumber++ {
		if err := ctx.Err(); err != nil {
			return shard.TextRow{}, err
		}
		page := reader.Page(pageNumber)
		if page.V.IsNull() {
			return shard.TextRow{}, fmt.Errorf("PDF page %d is missing", pageNumber)
		}
		fonts := make(map[string]*pdf.Font)
		for _, name := range page.Fonts() {
			font := page.Font(name)
			fonts[name] = &font
		}
		text, err := page.GetPlainText(fonts)
		if err != nil {
			return shard.TextRow{}, fmt.Errorf("extract page %d: %w", pageNumber, err)
		}
		text = normalizeExtractedDocumentText(text)
		if text != "" {
			if document.Len() > 0 {
				document.WriteString("\n\n")
			}
			document.WriteString(text)
			if int64(document.Len()) > plan.Writer.RecordMaximumBytes {
				return shard.TextRow{}, fmt.Errorf("extracted PDF text exceeds the %d-byte maximum record size", plan.Writer.RecordMaximumBytes)
			}
		}
		if pageNumber == pages || pageNumber == 1 || pageNumber%max(1, pages/100) == 0 {
			emitProgress(ctx, ProgressEvent{Phase: "convert", Status: "progress", Input: input.Artifact.Path, Adapter: PDFTextAdapter, Sequence: pageNumber, Files: int64(pageNumber), TotalFiles: int64(pages), Message: fmt.Sprintf("%s page %d/%d", filepath.Base(input.Artifact.Path), pageNumber, pages)})
		}
	}
	if document.Len() == 0 {
		return shard.TextRow{}, fmt.Errorf("PDF contains no extractable text layer; scanned documents require OCR before WALDO can ingest them")
	}
	metadata := map[string]string{"pdf.pages": fmt.Sprint(pages)}
	info := reader.Trailer().Key("Info")
	for key, name := range map[string]string{"Title": "pdf.title", "Author": "pdf.author", "Subject": "pdf.subject", "Keywords": "pdf.keywords", "Creator": "pdf.creator", "Producer": "pdf.producer", "CreationDate": "pdf.creation_date", "ModDate": "pdf.modified_date"} {
		if value := strings.TrimSpace(info.Key(key).Text()); value != "" {
			metadata[name] = value
		}
	}
	if input.SourcePath != "" {
		metadata["source_path"] = input.SourcePath
	}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		return shard.TextRow{}, err
	}
	meta := string(encoded)
	row, err = profiledFileRow(plan, input, document.String()+"\n", "", metadata["pdf.creation_date"], "", "", &meta)
	if err != nil {
		return shard.TextRow{}, err
	}
	if err := unchangedInput(file, verified); err != nil {
		return shard.TextRow{}, err
	}
	return row, nil
}

func normalizeExtractedDocumentText(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	return strings.TrimSpace(text)
}
