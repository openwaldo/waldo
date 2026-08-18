// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package lookaside

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestFetchVerifiesAndCachesHTTPObject(t *testing.T) {
	content := "verified object"
	digest := digestOf(content)
	transport := &fakeTransport{content: content}
	cache, err := NewCache(t.TempDir(), &http.Client{Transport: transport})
	if err != nil {
		t.Fatal(err)
	}

	path, err := cache.Fetch(context.Background(), "https://objects.example/item", digest, int64(len(content)))
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != content || transport.requests != 1 {
		t.Fatalf("first fetch = %q, requests = %d", data, transport.requests)
	}

	if _, err := cache.Fetch(context.Background(), "https://objects.example/item", digest, int64(len(content))); err != nil {
		t.Fatal(err)
	}
	if transport.requests != 1 {
		t.Fatalf("cache hit made %d HTTP requests, want 1", transport.requests)
	}
}

func TestFetchReportsDownloadAndCacheVerificationBytes(t *testing.T) {
	content := "progress-bearing object"
	digest := digestOf(content)
	cache, err := NewCache(t.TempDir(), &http.Client{Transport: &fakeTransport{content: content}})
	if err != nil {
		t.Fatal(err)
	}
	var download, cached FetchProgress
	if _, err := cache.FetchWithProgress(context.Background(), "https://objects.example/item", digest, int64(len(content)), func(event FetchProgress) {
		if event.Phase == "download" {
			download = event
		}
	}); err != nil {
		t.Fatal(err)
	}
	if download.Written != int64(len(content)) || download.Total != int64(len(content)) {
		t.Fatalf("download progress = %+v", download)
	}
	if _, err := cache.FetchWithProgress(context.Background(), "https://objects.example/item", digest, int64(len(content)), func(event FetchProgress) {
		if event.Phase == "cache" {
			cached = event
		}
	}); err != nil {
		t.Fatal(err)
	}
	if cached.Written != int64(len(content)) || cached.Total != int64(len(content)) {
		t.Fatalf("cache progress = %+v", cached)
	}
}

func TestFetchRejectsWrongDigestWithoutCacheEntry(t *testing.T) {
	cache, err := NewCache(t.TempDir(), &http.Client{Transport: &fakeTransport{content: "wrong"}})
	if err != nil {
		t.Fatal(err)
	}
	digest := digestOf("right")
	if _, err := cache.Fetch(context.Background(), "https://objects.example/item", digest, 0); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("Fetch() error = %v", err)
	}
	path, _ := cache.Path(digest)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("invalid object was left at cache path: %v", err)
	}
}

func TestFetchRepairsCorruptCacheEntry(t *testing.T) {
	content := "correct"
	digest := digestOf(content)
	transport := &fakeTransport{content: content}
	cache, err := NewCache(t.TempDir(), &http.Client{Transport: transport})
	if err != nil {
		t.Fatal(err)
	}
	path, _ := cache.Path(digest)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := cache.Fetch(context.Background(), "https://objects.example/item", digest, int64(len(content))); err != nil {
		t.Fatal(err)
	}
	if transport.requests != 1 {
		t.Fatalf("repair made %d requests, want 1", transport.requests)
	}
}

func TestPurgeUsedRemovesSuccessfulFetches(t *testing.T) {
	root := t.TempDir()
	content := "temporary object"
	digest := digestOf(content)
	cache, err := NewCache(root, &http.Client{Transport: &fakeTransport{content: content}})
	if err != nil {
		t.Fatal(err)
	}
	path, err := cache.Fetch(context.Background(), "https://objects.example/item", digest, int64(len(content)))
	if err != nil {
		t.Fatal(err)
	}
	purged, err := cache.PurgeUsed()
	if err != nil {
		t.Fatal(err)
	}
	if purged.Objects != 1 || purged.Bytes != int64(len(content)) {
		t.Fatalf("PurgeUsed() = %+v", purged)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("fetched cache object remains: %v", err)
	}
}

func TestPersistentCacheRetainsVerifiedObjectAndCleansScratch(t *testing.T) {
	root, scratch := t.TempDir(), t.TempDir()
	content := "retained verified object"
	digest := digestOf(content)
	transport := &fakeTransport{content: content}
	cache, err := NewCache(root, &http.Client{Transport: transport}, WithPersistentStorage(scratch, 1<<20))
	if err != nil {
		t.Fatal(err)
	}
	path, err := cache.Fetch(context.Background(), "https://objects.example/item", digest, int64(len(content)))
	if err != nil {
		t.Fatal(err)
	}
	if purged, err := cache.PurgeUsed(); err != nil || purged.Objects != 0 {
		t.Fatalf("PurgeUsed() = %+v, %v", purged, err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("retained object: %v", err)
	}
	entries, err := os.ReadDir(scratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("scratch entries = %v", entries)
	}
	second, err := NewCache(root, &http.Client{Transport: transport}, WithPersistentStorage(scratch, 1<<20))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := second.Fetch(context.Background(), "https://objects.example/item", digest, int64(len(content))); err != nil {
		t.Fatal(err)
	}
	if transport.requests != 1 {
		t.Fatalf("persistent cache made %d requests, want 1", transport.requests)
	}
}

func TestPersistentCachePrunesLeastRecentlyUsedObjects(t *testing.T) {
	root, scratch := t.TempDir(), t.TempDir()
	cache, err := NewCache(root, nil, WithPersistentStorage(scratch, 3))
	if err != nil {
		t.Fatal(err)
	}
	oldPath, _ := cache.Path(digestOf("old"))
	newPath, _ := cache.Path(digestOf("new"))
	for path, value := range map[string]string{oldPath: "old", newPath: "new"} {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(value), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Now()
	if err := os.Chtimes(oldPath, now.Add(-time.Hour), now.Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(newPath, now, now); err != nil {
		t.Fatal(err)
	}
	if _, err := NewCache(root, nil, WithPersistentStorage(scratch, 3)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(oldPath); !os.IsNotExist(err) {
		t.Fatalf("least-recently-used object remains: %v", err)
	}
	if data, err := os.ReadFile(newPath); err != nil || string(data) != "new" {
		t.Fatalf("newest object = %q, %v", data, err)
	}
}

func TestFailedDownloadLeavesNeitherCacheNorScratchObject(t *testing.T) {
	root, scratch := t.TempDir(), t.TempDir()
	cache, err := NewCache(root, &http.Client{Transport: failingTransport{}}, WithPersistentStorage(scratch, 1<<20))
	if err != nil {
		t.Fatal(err)
	}
	digest := digestOf("complete content")
	if _, err := cache.Fetch(context.Background(), "https://objects.example/item", digest, 0); err == nil || !strings.Contains(err.Error(), "unexpected EOF") {
		t.Fatalf("Fetch() error = %v", err)
	}
	destination, _ := cache.Path(digest)
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatalf("partial cache object remains: %v", err)
	}
	entries, err := os.ReadDir(scratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("scratch entries = %v", entries)
	}
}

func TestFetchFallsBackToConfiguredMirror(t *testing.T) {
	content := "from mirror"
	digest := digestOf(content)
	transport := &fallbackTransport{content: content}
	cache, err := NewCache(t.TempDir(), &http.Client{Transport: transport}, WithMirrors([]string{"https://mirror.example/lookaside/v1"}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cache.Fetch(context.Background(), "https://primary.example/object", digest, int64(len(content))); err != nil {
		t.Fatal(err)
	}
	wantMirror := "https://mirror.example/lookaside/v1/" + digest[:2] + "/" + digest[2:4] + "/" + digest
	if len(transport.urls) != 2 || transport.urls[0] != "https://primary.example/object" || transport.urls[1] != wantMirror {
		t.Fatalf("request order = %v, want primary then %s", transport.urls, wantMirror)
	}
}

func TestS3URLTranslation(t *testing.T) {
	tests := map[string]string{
		"s3://bucket/key": "https://bucket.s3.amazonaws.com/key",
		"s3://s3.us-east-2.amazonaws.com/bucket/key":            "https://s3.us-east-2.amazonaws.com/bucket/key",
		"s3://s3.amazonaws.com/bucket/key?versionId=identified": "https://s3.amazonaws.com/bucket/key?versionId=identified",
	}
	for input, want := range tests {
		parsed, err := url.Parse(input)
		if err != nil {
			t.Fatal(err)
		}
		if got := s3HTTPS(parsed); got != want {
			t.Errorf("s3HTTPS(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestProbeHTTPUsesHEADWithoutReadingBody(t *testing.T) {
	transport := &probeTransport{size: 1234}
	cache, err := NewCache(t.TempDir(), &http.Client{Transport: transport})
	if err != nil {
		t.Fatal(err)
	}
	result, err := cache.Probe(context.Background(), "https://objects.example/item", 1234)
	if err != nil {
		t.Fatal(err)
	}
	if result.Method != "HEAD" || result.Bytes != 1234 || len(transport.methods) != 1 || transport.methods[0] != http.MethodHead {
		t.Fatalf("result = %+v, methods = %v", result, transport.methods)
	}
}

func TestProbeHTTPFallsBackToOneByteRange(t *testing.T) {
	transport := &probeTransport{size: 9876, rejectHead: true}
	cache, err := NewCache(t.TempDir(), &http.Client{Transport: transport})
	if err != nil {
		t.Fatal(err)
	}
	result, err := cache.Probe(context.Background(), "https://objects.example/item", 9876)
	if err != nil {
		t.Fatal(err)
	}
	if result.Method != "GET range" || len(transport.methods) != 2 || transport.rangeHeader != "bytes=0-0" {
		t.Fatalf("result = %+v, methods = %v, range = %q", result, transport.methods, transport.rangeHeader)
	}
}

func TestProbeRejectsDeclaredSizeMismatch(t *testing.T) {
	cache, err := NewCache(t.TempDir(), &http.Client{Transport: &probeTransport{size: 10}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cache.Probe(context.Background(), "https://objects.example/item", 11); err == nil || !strings.Contains(err.Error(), "size mismatch") {
		t.Fatalf("Probe() error = %v", err)
	}
}

func TestProbeLocalFileUsesStat(t *testing.T) {
	file := filepath.Join(t.TempDir(), "object.parquet")
	if err := os.WriteFile(file, []byte("12345"), 0o644); err != nil {
		t.Fatal(err)
	}
	cache, err := NewCache(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	result, err := cache.Probe(context.Background(), file, 5)
	if err != nil {
		t.Fatal(err)
	}
	if result.Method != "stat" || result.Bytes != 5 {
		t.Fatalf("Probe() = %+v", result)
	}
}

type fakeTransport struct {
	content  string
	requests int
}

type failingTransport struct{}

func (failingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Body: io.NopCloser(&failingReader{}), Header: make(http.Header)}, nil
}

type failingReader struct{ sent bool }

func (reader *failingReader) Read(buffer []byte) (int, error) {
	if reader.sent {
		return 0, io.ErrUnexpectedEOF
	}
	reader.sent = true
	return copy(buffer, "partial"), nil
}

type fallbackTransport struct {
	content string
	urls    []string
}

type probeTransport struct {
	size        int64
	rejectHead  bool
	methods     []string
	rangeHeader string
}

func (transport *probeTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	transport.methods = append(transport.methods, request.Method)
	if request.Method == http.MethodHead && transport.rejectHead {
		return &http.Response{StatusCode: http.StatusMethodNotAllowed, Status: "405 Method Not Allowed", Body: io.NopCloser(strings.NewReader("")), Header: make(http.Header)}, nil
	}
	response := &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Body: io.NopCloser(strings.NewReader("")), Header: make(http.Header), ContentLength: transport.size}
	if request.Method == http.MethodGet {
		transport.rangeHeader = request.Header.Get("Range")
		response.StatusCode = http.StatusPartialContent
		response.Status = "206 Partial Content"
		response.ContentLength = 1
		response.Header.Set("Content-Range", fmt.Sprintf("bytes 0-0/%d", transport.size))
	}
	return response, nil
}

func (transport *fallbackTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	transport.urls = append(transport.urls, request.URL.String())
	status := http.StatusNotFound
	body := "missing"
	if request.URL.Host == "mirror.example" {
		status = http.StatusOK
		body = transport.content
	}
	return &http.Response{
		StatusCode: status,
		Status:     http.StatusText(status),
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}, nil
}

func (transport *fakeTransport) RoundTrip(*http.Request) (*http.Response, error) {
	transport.requests++
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Body:       io.NopCloser(strings.NewReader(transport.content)),
		Header:     make(http.Header),
	}, nil
}

func digestOf(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}

func TestEnsureScratchTightensAnExistingRoot(t *testing.T) {
	root := filepath.Join(t.TempDir(), "scratch")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	cache, err := NewCache(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.EnsureScratch(); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(cache.Scratch())
	if err != nil {
		t.Fatal(err)
	}
	if mode := info.Mode().Perm(); mode != 0o700 {
		t.Fatalf("mode = %04o; a root left wider by an older command must be corrected", mode)
	}
}
