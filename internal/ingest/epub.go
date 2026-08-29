// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"archive/zip"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/url"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/openwaldo/waldo/internal/shard"
	"golang.org/x/net/html"
)

const (
	EPUBTextAdapter          = "epub-spine-text-v1"
	epubMaximumEntries       = 100_000
	epubMaximumExpandedBytes = 2 << 30
	epubMaximumEntryBytes    = 256 << 20
	epubMaximumXMLBytes      = 8 << 20
	epubMaximumRatio         = 1_000
)

type epubContainer struct {
	Rootfiles []struct {
		FullPath string `xml:"full-path,attr"`
	} `xml:"rootfiles>rootfile"`
}

type epubPackage struct {
	Metadata struct {
		Titles      []string `xml:"title"`
		Creators    []string `xml:"creator"`
		Languages   []string `xml:"language"`
		Identifiers []string `xml:"identifier"`
		Dates       []string `xml:"date"`
		Publishers  []string `xml:"publisher"`
		Rights      []string `xml:"rights"`
	} `xml:"metadata"`
	Manifest struct {
		Items []epubManifestItem `xml:"item"`
	} `xml:"manifest"`
	Spine struct {
		Items []struct {
			IDRef  string `xml:"idref,attr"`
			Linear string `xml:"linear,attr"`
		} `xml:"itemref"`
	} `xml:"spine"`
}

type epubManifestItem struct {
	ID         string `xml:"id,attr"`
	Href       string `xml:"href,attr"`
	MediaType  string `xml:"media-type,attr"`
	Properties string `xml:"properties,attr"`
	Fallback   string `xml:"fallback,attr"`
}

func StreamEPUBTextBatches(ctx context.Context, plan Plan, consume func(TextBatch) error) error {
	if err := plan.Validate(); err != nil {
		return err
	}
	if consume == nil {
		return fmt.Errorf("EPUB batch consumer is required")
	}
	for _, input := range plan.Inputs {
		if input.Adapter != EPUBTextAdapter {
			return fmt.Errorf("input %s requires the %s adapter, not the EPUB adapter", input.Artifact.Path, input.Adapter)
		}
		row, err := extractEPUBText(ctx, plan, input)
		if err != nil {
			return fmt.Errorf("adapt %s: %w", input.Artifact.Path, err)
		}
		if err := consume(TextBatch{Rows: []shard.TextRow{row}, LogicalBytes: int64(len(row.Text)), InputBytes: input.Artifact.Bytes}); err != nil {
			return err
		}
	}
	return nil
}

func extractEPUBText(ctx context.Context, plan Plan, input PlanInput) (shard.TextRow, error) {
	file, verified, err := openVerifiedInput(ctx, input.Artifact)
	if err != nil {
		return shard.TextRow{}, err
	}
	defer file.Close()
	archive, err := zip.NewReader(file, input.Artifact.Bytes)
	if err != nil {
		return shard.TextRow{}, fmt.Errorf("parse EPUB container: %w", err)
	}
	entries, err := validateEPUBArchive(archive)
	if err != nil {
		return shard.TextRow{}, err
	}
	mimetype, err := readEPUBEntry(ctx, entries["mimetype"], 64)
	if err != nil || string(mimetype) != "application/epub+zip" {
		return shard.TextRow{}, fmt.Errorf("EPUB has an invalid mimetype entry")
	}
	containerData, err := readEPUBEntry(ctx, entries["META-INF/container.xml"], epubMaximumXMLBytes)
	if err != nil {
		return shard.TextRow{}, fmt.Errorf("read EPUB container.xml: %w", err)
	}
	var container epubContainer
	if err := xml.Unmarshal(containerData, &container); err != nil || len(container.Rootfiles) == 0 {
		return shard.TextRow{}, fmt.Errorf("EPUB container.xml has no valid package document")
	}
	opfPath, err := safeEPUBPath("", container.Rootfiles[0].FullPath)
	if err != nil {
		return shard.TextRow{}, fmt.Errorf("invalid EPUB package path: %w", err)
	}
	opfData, err := readEPUBEntry(ctx, entries[opfPath], epubMaximumXMLBytes)
	if err != nil {
		return shard.TextRow{}, fmt.Errorf("read EPUB package %s: %w", opfPath, err)
	}
	var pkg epubPackage
	if err := xml.Unmarshal(opfData, &pkg); err != nil {
		return shard.TextRow{}, fmt.Errorf("parse EPUB package %s: %w", opfPath, err)
	}
	items := make(map[string]epubManifestItem, len(pkg.Manifest.Items))
	for _, item := range pkg.Manifest.Items {
		if item.ID == "" || item.Href == "" || items[item.ID].ID != "" {
			return shard.TextRow{}, fmt.Errorf("EPUB package has an invalid or duplicate manifest item id %q", item.ID)
		}
		items[item.ID] = item
	}
	var document strings.Builder
	retained := 0
	for position, itemref := range pkg.Spine.Items {
		if err := ctx.Err(); err != nil {
			return shard.TextRow{}, err
		}
		if strings.EqualFold(strings.TrimSpace(itemref.Linear), "no") {
			continue
		}
		item, ok := items[itemref.IDRef]
		if !ok {
			return shard.TextRow{}, fmt.Errorf("EPUB spine references missing item %q", itemref.IDRef)
		}
		item, err = supportedEPUBItem(item, items)
		if err != nil {
			return shard.TextRow{}, err
		}
		if slices.Contains(strings.Fields(item.Properties), "nav") {
			continue
		}
		contentPath, err := safeEPUBPath(path.Dir(opfPath), item.Href)
		if err != nil {
			return shard.TextRow{}, fmt.Errorf("invalid EPUB content path %q: %w", item.Href, err)
		}
		data, err := readEPUBEntry(ctx, entries[contentPath], epubMaximumEntryBytes)
		if err != nil {
			return shard.TextRow{}, fmt.Errorf("read EPUB spine item %s: %w", contentPath, err)
		}
		text, err := extractEPUBHTML(data)
		if err != nil {
			return shard.TextRow{}, fmt.Errorf("parse EPUB spine item %s: %w", contentPath, err)
		}
		if text != "" {
			if document.Len() > 0 {
				document.WriteString("\n\n")
			}
			document.WriteString(text)
			retained++
			if int64(document.Len()) > plan.Writer.RecordMaximumBytes {
				return shard.TextRow{}, fmt.Errorf("extracted EPUB text exceeds the %d-byte maximum record size", plan.Writer.RecordMaximumBytes)
			}
		}
		emitProgress(ctx, ProgressEvent{Phase: "convert", Status: "progress", Input: input.Artifact.Path, Adapter: EPUBTextAdapter, Sequence: position + 1, Files: int64(position + 1), TotalFiles: int64(len(pkg.Spine.Items)), Message: fmt.Sprintf("%s spine item %d/%d", filepath.Base(input.Artifact.Path), position+1, len(pkg.Spine.Items))})
	}
	if document.Len() == 0 {
		return shard.TextRow{}, fmt.Errorf("EPUB contains no extractable linear spine text")
	}
	metadata := map[string]string{"epub.spine_items": fmt.Sprint(retained)}
	addEPUBMetadata(metadata, "epub.title", pkg.Metadata.Titles)
	addEPUBMetadata(metadata, "epub.creator", pkg.Metadata.Creators)
	addEPUBMetadata(metadata, "epub.identifier", pkg.Metadata.Identifiers)
	addEPUBMetadata(metadata, "epub.publisher", pkg.Metadata.Publishers)
	addEPUBMetadata(metadata, "epub.rights", pkg.Metadata.Rights)
	if input.SourcePath != "" {
		metadata["source_path"] = input.SourcePath
	}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		return shard.TextRow{}, err
	}
	meta := string(encoded)
	row, err := profiledFileRow(plan, input, document.String()+"\n", "", firstEPUBMetadata(pkg.Metadata.Dates), firstEPUBMetadata(pkg.Metadata.Languages), "", &meta)
	if err != nil {
		return shard.TextRow{}, err
	}
	if err := unchangedInput(file, verified); err != nil {
		return shard.TextRow{}, err
	}
	return row, nil
}

func validateEPUBArchive(archive *zip.Reader) (map[string]*zip.File, error) {
	if len(archive.File) == 0 || len(archive.File) > epubMaximumEntries {
		return nil, fmt.Errorf("EPUB entry count %d is outside the supported range", len(archive.File))
	}
	entries := make(map[string]*zip.File, len(archive.File))
	var expanded uint64
	for _, entry := range archive.File {
		name := entry.Name
		cleanName := strings.TrimSuffix(name, "/")
		if strings.Contains(name, "\\") || path.IsAbs(name) || path.Clean(cleanName) != cleanName || cleanName == "." || strings.HasPrefix(cleanName, "../") {
			return nil, fmt.Errorf("EPUB contains unsafe entry path %q", name)
		}
		if entry.FileInfo().IsDir() {
			continue
		}
		if entries[name] != nil {
			return nil, fmt.Errorf("EPUB contains duplicate entry path %q", name)
		}
		if entry.UncompressedSize64 > epubMaximumEntryBytes {
			return nil, fmt.Errorf("EPUB entry %q exceeds the expanded-size limit", name)
		}
		expanded += entry.UncompressedSize64
		if expanded > epubMaximumExpandedBytes {
			return nil, fmt.Errorf("EPUB exceeds the total expanded-size limit")
		}
		if entry.UncompressedSize64 > 1<<20 && (entry.CompressedSize64 == 0 || entry.UncompressedSize64/entry.CompressedSize64 > epubMaximumRatio) {
			return nil, fmt.Errorf("EPUB entry %q exceeds the compression-ratio limit", name)
		}
		entries[name] = entry
	}
	if entries["mimetype"] == nil || entries["META-INF/container.xml"] == nil {
		return nil, fmt.Errorf("EPUB is missing mimetype or META-INF/container.xml")
	}
	return entries, nil
}

func readEPUBEntry(ctx context.Context, entry *zip.File, maximum uint64) ([]byte, error) {
	if entry == nil {
		return nil, fmt.Errorf("referenced entry is missing")
	}
	if entry.Flags&1 != 0 {
		return nil, fmt.Errorf("entry %q is encrypted", entry.Name)
	}
	if entry.UncompressedSize64 > maximum {
		return nil, fmt.Errorf("entry %q exceeds the %d-byte limit", entry.Name, maximum)
	}
	stream, err := entry.Open()
	if err != nil {
		return nil, err
	}
	data, readErr := io.ReadAll(io.LimitReader(&contextReader{ctx: ctx, reader: stream}, int64(maximum)+1))
	closeErr := stream.Close()
	if readErr != nil {
		return nil, readErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	if uint64(len(data)) > maximum {
		return nil, fmt.Errorf("entry %q exceeds the %d-byte limit", entry.Name, maximum)
	}
	return data, nil
}

func safeEPUBPath(base, reference string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(reference))
	if err != nil || parsed.Scheme != "" || parsed.Host != "" || parsed.Path == "" {
		return "", fmt.Errorf("path must be a non-empty local reference")
	}
	if strings.Contains(parsed.Path, "\\") || strings.ContainsRune(parsed.Path, 0) {
		return "", fmt.Errorf("path contains unsafe characters")
	}
	joined := path.Clean(path.Join(base, parsed.Path))
	if path.IsAbs(joined) || joined == "." || strings.HasPrefix(joined, "../") {
		return "", fmt.Errorf("path escapes the EPUB container")
	}
	return joined, nil
}

func supportedEPUBItem(item epubManifestItem, items map[string]epubManifestItem) (epubManifestItem, error) {
	seen := map[string]bool{}
	for {
		if item.MediaType == "application/xhtml+xml" || item.MediaType == "text/html" || item.MediaType == "image/svg+xml" {
			return item, nil
		}
		if item.Fallback == "" || seen[item.ID] {
			return epubManifestItem{}, fmt.Errorf("EPUB spine item %q has unsupported media type %q and no usable fallback", item.ID, item.MediaType)
		}
		seen[item.ID] = true
		next, ok := items[item.Fallback]
		if !ok {
			return epubManifestItem{}, fmt.Errorf("EPUB manifest item %q references missing fallback %q", item.ID, item.Fallback)
		}
		item = next
	}
}

func extractEPUBHTML(data []byte) (string, error) {
	document, err := html.Parse(strings.NewReader(string(data)))
	if err != nil {
		return "", err
	}
	var output epubTextOutput
	var walk func(*html.Node, bool)
	walk = func(node *html.Node, skipped bool) {
		if node.Type == html.ElementNode {
			switch strings.ToLower(node.Data) {
			case "head", "script", "style", "template":
				skipped = true
			case "br", "hr":
				output.breakLine()
			}
			if epubBlockElement(node.Data) {
				output.breakLine()
			}
		}
		if !skipped && node.Type == html.TextNode {
			output.writeWords(node.Data)
		}
		if !skipped {
			for child := node.FirstChild; child != nil; child = child.NextSibling {
				walk(child, false)
			}
		}
		if node.Type == html.ElementNode && epubBlockElement(node.Data) {
			output.breakLine()
		}
	}
	walk(document, false)
	lines := strings.Split(strings.ReplaceAll(output.String(), "\r", ""), "\n")
	clean := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.Join(strings.Fields(line), " ")
		if line != "" {
			clean = append(clean, line)
		}
	}
	return strings.Join(clean, "\n"), nil
}

func epubBlockElement(name string) bool {
	switch strings.ToLower(name) {
	case "address", "article", "aside", "blockquote", "div", "dl", "dt", "dd", "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "main", "nav", "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul", "text":
		return true
	default:
		return false
	}
}

type epubTextOutput struct {
	builder strings.Builder
	last    byte
}

func (output *epubTextOutput) writeWords(value string) {
	value = strings.Join(strings.Fields(value), " ")
	if value == "" {
		return
	}
	if output.last != 0 && output.last != '\n' && output.last != ' ' && !strings.ContainsRune(".,;:!?)]}", rune(value[0])) {
		output.builder.WriteByte(' ')
	}
	output.builder.WriteString(value)
	output.last = value[len(value)-1]
}

func (output *epubTextOutput) breakLine() {
	if output.last != 0 && output.last != '\n' {
		output.builder.WriteByte('\n')
		output.last = '\n'
	}
}

func (output *epubTextOutput) String() string { return output.builder.String() }

func addEPUBMetadata(metadata map[string]string, name string, values []string) {
	clean := make([]string, 0, len(values))
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			clean = append(clean, value)
		}
	}
	if len(clean) > 0 {
		metadata[name] = strings.Join(clean, "; ")
	}
}

func firstEPUBMetadata(values []string) string {
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return ""
}
