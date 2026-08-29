// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package ingest probes acquired artifacts and builds immutable execution plans
// for the corpus ingestion workflow. Format adapters and writers consume the
// plan; they do not repeat detection or silently choose a different mapping.
package ingest

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/parquet-go/parquet-go"
)

const probeBytes = 64 << 10

type Probe struct {
	Kind      string      `json:"kind"`
	Schema    int         `json:"schema"`
	Artifacts []Artifact  `json:"artifacts"`
	Totals    ProbeTotals `json:"totals"`
}

type ProbeTotals struct {
	Artifacts int64 `json:"artifacts"`
	Bytes     int64 `json:"bytes"`
}

type Artifact struct {
	Path        string       `json:"path"`
	SHA256      string       `json:"sha256"`
	Bytes       int64        `json:"bytes"`
	Format      string       `json:"format"`
	Compression string       `json:"compression,omitempty"`
	MediaType   string       `json:"media_type,omitempty"`
	Evidence    []string     `json:"evidence"`
	Parquet     *ParquetInfo `json:"parquet,omitempty"`
}

type ParquetInfo struct {
	Rows      int64    `json:"rows"`
	RowGroups int      `json:"row_groups"`
	Columns   []string `json:"columns"`
	Schema    string   `json:"schema"`
}

// ProbePaths recursively inspects regular files beneath roots. It hashes every
// artifact in a bounded buffer and returns artifacts in absolute path order.
// Symlinks are rejected so a recorded plan cannot silently change its target.
func ProbePaths(ctx context.Context, roots []string) (Probe, error) {
	return ProbePathsWithWorkers(ctx, roots, 0)
}

func ProbePathsWithWorkers(ctx context.Context, roots []string, workers int) (Probe, error) {
	if len(roots) == 0 {
		return Probe{}, fmt.Errorf("at least one input path is required")
	}
	paths, err := regularFiles(roots)
	if err != nil {
		return Probe{}, err
	}
	if workers <= 0 {
		workers = min(runtime.GOMAXPROCS(0), 32)
	}
	workers = min(workers, len(paths))
	artifacts := make([]Artifact, len(paths))
	jobs := make(chan int)
	probeContext, cancel := context.WithCancel(ctx)
	defer cancel()
	var group sync.WaitGroup
	var firstErr error
	var errorMutex sync.Mutex
	for worker := 0; worker < workers; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			for position := range jobs {
				path := paths[position]
				emitProgress(ctx, ProgressEvent{Phase: "input", Status: "probing", Input: path})
				artifact, err := probeFile(probeContext, path)
				if err != nil {
					errorMutex.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("probe %s: %w", path, err)
						cancel()
					}
					errorMutex.Unlock()
					continue
				}
				artifacts[position] = artifact
				emitProgress(ctx, ProgressEvent{Phase: "input", Status: "detected", Input: path, Adapter: artifact.Format, Bytes: artifact.Bytes})
			}
		}()
	}
	for position := range paths {
		select {
		case jobs <- position:
		case <-probeContext.Done():
			break
		}
		if probeContext.Err() != nil {
			break
		}
	}
	close(jobs)
	group.Wait()
	if firstErr != nil {
		return Probe{}, firstErr
	}
	if err := ctx.Err(); err != nil {
		return Probe{}, err
	}
	result := Probe{Kind: "waldo-ingest-probe", Schema: 1, Artifacts: artifacts}
	for _, artifact := range artifacts {
		result.Totals.Artifacts++
		result.Totals.Bytes += artifact.Bytes
	}
	if len(result.Artifacts) == 0 {
		return Probe{}, fmt.Errorf("input paths contain no regular files")
	}
	return result, nil
}

func regularFiles(roots []string) ([]string, error) {
	seen := map[string]bool{}
	var paths []string
	for _, root := range roots {
		abs, err := filepath.Abs(root)
		if err != nil {
			return nil, err
		}
		if seen[abs] {
			continue
		}
		seen[abs] = true
		info, err := os.Lstat(abs)
		if err != nil {
			return nil, err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("input path %s is a symlink", abs)
		}
		if info.Mode().IsRegular() {
			paths = append(paths, abs)
			continue
		}
		if !info.IsDir() {
			return nil, fmt.Errorf("input path %s is not a regular file or directory", abs)
		}
		err = filepath.WalkDir(abs, func(path string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if path == abs {
				return nil
			}
			if entry.Type()&os.ModeSymlink != 0 {
				return fmt.Errorf("input path %s is a symlink", path)
			}
			if entry.IsDir() {
				return nil
			}
			info, err := entry.Info()
			if err != nil {
				return err
			}
			if !info.Mode().IsRegular() {
				return fmt.Errorf("input path %s is not a regular file", path)
			}
			paths = append(paths, path)
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	slices.Sort(paths)
	paths = slices.Compact(paths)
	return paths, nil
}

func probeFile(ctx context.Context, path string) (Artifact, error) {
	file, err := os.Open(path)
	if err != nil {
		return Artifact{}, err
	}
	defer file.Close()
	before, err := file.Stat()
	if err != nil {
		return Artifact{}, err
	}
	hash := sha256.New()
	sample := make([]byte, 0, probeBytes)
	buffer := make([]byte, 1<<20)
	var size int64
	for {
		if err := ctx.Err(); err != nil {
			return Artifact{}, err
		}
		count, readErr := file.Read(buffer)
		if count > 0 {
			if _, err := hash.Write(buffer[:count]); err != nil {
				return Artifact{}, err
			}
			if len(sample) < probeBytes {
				keep := min(count, probeBytes-len(sample))
				sample = append(sample, buffer[:keep]...)
			}
			size += int64(count)
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return Artifact{}, readErr
		}
	}
	after, err := file.Stat()
	if err != nil {
		return Artifact{}, err
	}
	if before.Size() != size || after.Size() != size || !before.ModTime().Equal(after.ModTime()) {
		return Artifact{}, fmt.Errorf("file changed while it was being probed")
	}
	artifact := Artifact{
		Path: path, SHA256: hex.EncodeToString(hash.Sum(nil)), Bytes: size,
		MediaType: http.DetectContentType(sample),
	}
	if err := detect(file, sample, &artifact); err != nil {
		return Artifact{}, err
	}
	return artifact, nil
}

func detect(file *os.File, sample []byte, artifact *Artifact) error {
	if len(sample) == 0 {
		artifact.Format = "empty"
		artifact.Evidence = []string{"zero-byte-file"}
		return nil
	}
	if len(sample) >= 4 && (bytes.Equal(sample[:4], []byte("PAR1")) || bytes.Equal(sample[:4], []byte("PARE"))) {
		artifact.Format = "parquet"
		artifact.Evidence = []string{"parquet-magic"}
		if bytes.Equal(sample[:4], []byte("PARE")) {
			artifact.Evidence = append(artifact.Evidence, "encrypted-footer")
			return nil
		}
		info, err := readParquetInfo(file, artifact.Bytes)
		if err != nil {
			return fmt.Errorf("invalid parquet: %w", err)
		}
		artifact.Parquet = &info
		return nil
	}
	if bytes.HasPrefix(sample, []byte("%PDF-")) {
		artifact.Format = "pdf"
		artifact.Evidence = []string{"pdf-header"}
		return nil
	}
	if len(sample) >= 4 && bytes.Equal(sample[:4], []byte{'P', 'K', 0x03, 0x04}) {
		epub, err := detectEPUB(file, artifact.Bytes)
		if err != nil {
			if strings.EqualFold(filepath.Ext(artifact.Path), ".epub") {
				return fmt.Errorf("invalid EPUB: %w", err)
			}
		} else if epub {
			artifact.Format = "epub"
			artifact.MediaType = "application/epub+zip"
			artifact.Evidence = []string{"zip-magic", "epub-mimetype", "epub-container"}
			return nil
		}
		if strings.EqualFold(filepath.Ext(artifact.Path), ".epub") {
			return fmt.Errorf("invalid EPUB: ZIP container is missing the EPUB mimetype or META-INF/container.xml")
		}
	}
	if compression := detectCompression(sample); compression != "" {
		artifact.Format = "compressed"
		artifact.Compression = compression
		artifact.Evidence = []string{compression + "-magic"}
		format, inspectErr := detectCompressedContent(file, artifact.Path, compression)
		if inspectErr != nil && compressedJSONLExtension(artifact.Path) {
			return fmt.Errorf("invalid compressed JSONL: %w", inspectErr)
		}
		if format == "jsonl" || format == "mbox" {
			artifact.Format = format
			artifact.Evidence = append(artifact.Evidence, format+"-structure")
		}
		return nil
	}
	if format := detectMedia(sample, artifact.MediaType); format != "" {
		artifact.Format = format
		artifact.Evidence = []string{"content-signature"}
		return nil
	}
	trimmed := bytes.TrimSpace(sample)
	if looksMbox(sample) {
		artifact.Format = "mbox"
		artifact.Evidence = []string{"mbox-envelope"}
		return nil
	}
	if bytes.HasPrefix(trimmed, []byte("WARC/")) {
		artifact.Format = "warc"
		artifact.Evidence = []string{"warc-header"}
		return nil
	}
	if looksHTML(trimmed) {
		artifact.Format = "html"
		artifact.Evidence = []string{"html-root"}
		return nil
	}
	if looksXML(trimmed) || (strings.EqualFold(filepath.Ext(artifact.Path), ".xml") && bytes.HasPrefix(trimmed, []byte("<"))) {
		artifact.Format = "xml"
		artifact.Evidence = []string{"xml-root"}
		return nil
	}
	if format := detectJSON(trimmed, artifact.Bytes <= probeBytes); format != "" {
		if format == "json" && strings.EqualFold(filepath.Ext(artifact.Path), ".jsonl") {
			format = "jsonl"
		}
		artifact.Format = format
		artifact.Evidence = []string{"json-structure"}
		return nil
	}
	// A single JSON object may be larger than the bounded probe sample. In that
	// case the truncated prefix cannot pass json.Valid, but the .json extension
	// plus a JSON container opener is sufficient to choose the JSON adapter;
	// the streaming reader still validates the complete record during ingest.
	if artifact.Bytes > probeBytes && strings.EqualFold(filepath.Ext(artifact.Path), ".json") &&
		(bytes.HasPrefix(trimmed, []byte("{")) || bytes.HasPrefix(trimmed, []byte("["))) {
		artifact.Format = "json"
		artifact.Evidence = []string{"json-container-prefix", "extension-hint:.json"}
		return nil
	}
	if validUTF8Sample(sample, artifact.Bytes <= int64(len(sample))) && !bytes.ContainsRune(sample, '\x00') {
		extension := strings.ToLower(filepath.Ext(artifact.Path))
		if extension == ".md" || extension == ".markdown" {
			artifact.Format = "markdown"
			artifact.Evidence = []string{"utf8-sample", "extension-hint:" + extension}
		} else {
			artifact.Format = "text"
			artifact.Evidence = []string{"utf8-sample"}
		}
		return nil
	}
	artifact.Format = "unknown"
	artifact.Evidence = []string{"no-recognized-signature"}
	return nil
}

func detectEPUB(file *os.File, size int64) (bool, error) {
	reader, err := zip.NewReader(file, size)
	if err != nil {
		return false, err
	}
	var mimetype *zip.File
	hasContainer := false
	for _, entry := range reader.File {
		switch entry.Name {
		case "mimetype":
			mimetype = entry
		case "META-INF/container.xml":
			hasContainer = true
		}
	}
	if mimetype == nil || !hasContainer {
		return false, nil
	}
	if mimetype.UncompressedSize64 > 64 {
		return false, fmt.Errorf("mimetype entry is too large")
	}
	stream, err := mimetype.Open()
	if err != nil {
		return false, err
	}
	data, readErr := io.ReadAll(io.LimitReader(stream, 65))
	closeErr := stream.Close()
	if readErr != nil {
		return false, readErr
	}
	if closeErr != nil {
		return false, closeErr
	}
	return string(data) == "application/epub+zip", nil
}

func detectCompressedContent(file *os.File, artifactPath, compression string) (string, error) {
	if compression != "gzip" && compression != "zstd" {
		return "", nil
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return "", err
	}
	reader, err := openDecompressed(file, compression)
	if err != nil {
		return "", err
	}
	sample, readErr := io.ReadAll(io.LimitReader(reader, probeBytes+1))
	closeErr := reader.Close()
	if readErr != nil {
		return "", readErr
	}
	if closeErr != nil {
		return "", closeErr
	}
	complete := len(sample) <= probeBytes
	if !complete {
		sample = sample[:probeBytes]
	}
	trimmed := bytes.TrimSpace(sample)
	if looksMbox(sample) {
		return "mbox", nil
	}
	format := detectJSON(trimmed, complete)
	compoundJSONL := compressedJSONLExtension(artifactPath)
	if format == "jsonl" || (format == "json" && compoundJSONL) ||
		(format == "" && compoundJSONL && bytes.HasPrefix(trimmed, []byte("{")) && validUTF8Sample(sample, complete)) {
		return "jsonl", nil
	}
	return "", nil
}

func looksMbox(sample []byte) bool {
	if !bytes.HasPrefix(sample, []byte("From ")) {
		return false
	}
	firstLineEnd := bytes.IndexByte(sample, '\n')
	if firstLineEnd < 0 {
		return false
	}
	headers := sample[firstLineEnd+1:]
	for _, name := range [][]byte{[]byte("From:"), []byte("Date:"), []byte("Subject:"), []byte("Message-ID:")} {
		if bytes.HasPrefix(headers, name) || bytes.Contains(headers, append([]byte("\n"), name...)) {
			return true
		}
	}
	return false
}

func compressedJSONLExtension(path string) bool {
	lower := strings.ToLower(path)
	for _, suffix := range []string{".jsonl.gz", ".jsonl.gzip", ".jsonl.zst", ".jsonl.zstd"} {
		if strings.HasSuffix(lower, suffix) {
			return true
		}
	}
	return false
}

func validUTF8Sample(sample []byte, complete bool) bool {
	if complete {
		return utf8.Valid(sample)
	}
	for trim := 0; trim <= 3 && trim <= len(sample); trim++ {
		if utf8.Valid(sample[:len(sample)-trim]) {
			return true
		}
	}
	return false
}

func readParquetInfo(file *os.File, size int64) (ParquetInfo, error) {
	parquetFile, err := parquet.OpenFile(file, size)
	if err != nil {
		return ParquetInfo{}, err
	}
	columns := parquetFile.Schema().Columns()
	columnNames := make([]string, len(columns))
	for i, column := range columns {
		columnNames[i] = strings.Join(column, ".")
	}
	return ParquetInfo{
		Rows: parquetFile.NumRows(), RowGroups: len(parquetFile.RowGroups()),
		Columns: columnNames, Schema: parquetFile.Schema().String(),
	}, nil
}

func detectCompression(sample []byte) string {
	switch {
	case len(sample) >= 2 && sample[0] == 0x1f && sample[1] == 0x8b:
		return "gzip"
	case len(sample) >= 4 && bytes.Equal(sample[:4], []byte{0x28, 0xb5, 0x2f, 0xfd}):
		return "zstd"
	case len(sample) >= 4 && bytes.Equal(sample[:4], []byte{'P', 'K', 0x03, 0x04}):
		return "zip"
	case len(sample) >= 3 && bytes.Equal(sample[:3], []byte{'B', 'Z', 'h'}):
		return "bzip2"
	case len(sample) >= 6 && bytes.Equal(sample[:6], []byte{0xfd, '7', 'z', 'X', 'Z', 0x00}):
		return "xz"
	default:
		return ""
	}
}

func detectMedia(sample []byte, mediaType string) string {
	switch {
	case strings.HasPrefix(mediaType, "image/"):
		return "image"
	case strings.HasPrefix(mediaType, "audio/"):
		return "audio"
	case strings.HasPrefix(mediaType, "video/"):
		return "video"
	default:
		return ""
	}
}

func looksHTML(trimmed []byte) bool {
	lower := bytes.ToLower(trimmed)
	return bytes.HasPrefix(lower, []byte("<!doctype html")) || bytes.HasPrefix(lower, []byte("<html"))
}

func looksXML(trimmed []byte) bool {
	return bytes.HasPrefix(trimmed, []byte("<?xml")) || bytes.HasPrefix(trimmed, []byte("<article"))
}

func detectJSON(trimmed []byte, complete bool) string {
	if len(trimmed) == 0 {
		return ""
	}
	if complete && json.Valid(trimmed) {
		return "json"
	}
	lines := bytes.Split(trimmed, []byte{'\n'})
	if !complete && len(lines) > 0 {
		lines = lines[:len(lines)-1]
	}
	valid := 0
	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		if !json.Valid(line) {
			return ""
		}
		valid++
		if valid == 3 {
			return "jsonl"
		}
	}
	if valid >= 2 {
		return "jsonl"
	}
	return ""
}
