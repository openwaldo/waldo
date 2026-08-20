// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package lookaside provides verified access to content-addressed objects. It
// knows where bytes live and whether they match a hash; it never assigns those
// bytes corpus meaning.
package lookaside

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/openwaldo/waldo/internal/config"
)

type Cache struct {
	root     string
	scratch  string
	retain   bool
	maxBytes int64
	client   *http.Client
	mirrors  []string
	mu       sync.Mutex
	used     map[string]bool
}

type ProbeResult struct {
	URL    string `json:"url"`
	Bytes  int64  `json:"bytes"`
	Method string `json:"method"`
}

type FetchProgress struct {
	Phase   string
	Written int64
	Total   int64
}

type Option func(*Cache)

func WithMirrors(mirrors []string) Option {
	return func(cache *Cache) {
		cache.mirrors = append([]string(nil), mirrors...)
	}
}

func WithPersistentStorage(scratch string, maxBytes int64) Option {
	return func(cache *Cache) { cache.scratch = scratch; cache.retain = true; cache.maxBytes = maxBytes }
}

func NewCache(root string, client *http.Client, options ...Option) (*Cache, error) {
	if root == "" {
		return nil, fmt.Errorf("lookaside scratch root is required")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	if client == nil {
		client = http.DefaultClient
	}
	cache := &Cache{root: abs, scratch: abs, client: client, used: map[string]bool{}}
	for _, option := range options {
		option(cache)
	}
	if cache.retain {
		if err := cache.prune(); err != nil {
			return nil, fmt.Errorf("prune lookaside cache: %w", err)
		}
	}
	return cache, nil
}

func DefaultCache() (*Cache, error) {
	configuration, err := config.Load()
	if err != nil {
		return nil, err
	}
	// A pre-cache configuration that names only lookaside.scratch retains its
	// historical purge-on-success behavior. New/default configurations use the
	// distinct retained cache and disposable download scratch.
	if configuration.Lookaside.Cache == "" && configuration.Lookaside.Scratch != "" {
		return NewCache(configuration.Lookaside.Scratch, nil, WithMirrors(configuration.Lookaside.Mirrors))
	}
	root, err := config.EffectiveCacheRoot(configuration)
	if err != nil {
		return nil, err
	}
	scratch, err := config.EffectiveScratchRoot(configuration)
	if err != nil {
		return nil, err
	}
	return NewCache(root, nil, WithMirrors(configuration.Lookaside.Mirrors), WithPersistentStorage(scratch, config.EffectiveCacheMaxBytes(configuration)))
}

func (cache *Cache) Root() string    { return cache.root }
func (cache *Cache) Scratch() string { return cache.scratch }

// EnsureScratch creates the scratch root and holds it at 0700. MkdirAll applies
// its mode only on creation, so a root left wider by an older command is
// corrected rather than inherited.
func (cache *Cache) EnsureScratch() error {
	if err := os.MkdirAll(cache.scratch, 0o700); err != nil {
		return err
	}
	return os.Chmod(cache.scratch, 0o700)
}
func (cache *Cache) MaxBytes() int64 { return cache.maxBytes }
func (cache *Cache) Retained() bool  { return cache.retain }

func (cache *Cache) Mirrors() []string { return append([]string(nil), cache.mirrors...) }

// Probe checks the canonical object location without transferring its body.
// HTTP and S3 URLs use HEAD, with a one-byte range request only when HEAD is
// unsupported or does not report a size. Local paths use stat.
func (cache *Cache) Probe(ctx context.Context, objectURL string, expectedBytes int64) (ProbeResult, error) {
	parsed, err := url.Parse(objectURL)
	if err != nil {
		return ProbeResult{}, fmt.Errorf("parse lookaside URL %q: %w", objectURL, err)
	}
	switch parsed.Scheme {
	case "http", "https":
		return cache.probeHTTP(ctx, parsed.String(), expectedBytes)
	case "s3":
		return cache.probeHTTP(ctx, s3HTTPS(parsed), expectedBytes)
	case "file":
		localPath, err := url.PathUnescape(parsed.Path)
		if err != nil {
			return ProbeResult{}, err
		}
		return probeFile(objectURL, filepath.FromSlash(localPath), expectedBytes)
	case "":
		return probeFile(objectURL, objectURL, expectedBytes)
	default:
		return ProbeResult{}, fmt.Errorf("unsupported lookaside URL scheme %q", parsed.Scheme)
	}
}

func probeFile(objectURL, localPath string, expectedBytes int64) (ProbeResult, error) {
	info, err := os.Stat(localPath)
	if err != nil {
		return ProbeResult{}, fmt.Errorf("probe %s: %w", objectURL, err)
	}
	if info.IsDir() {
		return ProbeResult{}, fmt.Errorf("probe %s: object is a directory", objectURL)
	}
	if err := checkProbeSize(objectURL, info.Size(), expectedBytes); err != nil {
		return ProbeResult{}, err
	}
	return ProbeResult{URL: objectURL, Bytes: info.Size(), Method: "stat"}, nil
}

func (cache *Cache) probeHTTP(ctx context.Context, objectURL string, expectedBytes int64) (ProbeResult, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodHead, objectURL, nil)
	if err != nil {
		return ProbeResult{}, err
	}
	request.Header.Set("Accept-Encoding", "identity")
	response, err := cache.client.Do(request)
	if err != nil {
		return ProbeResult{}, fmt.Errorf("probe %s: %w", objectURL, err)
	}
	_ = response.Body.Close()
	if response.StatusCode >= 200 && response.StatusCode < 300 && response.ContentLength >= 0 {
		if err := checkProbeSize(objectURL, response.ContentLength, expectedBytes); err != nil {
			return ProbeResult{}, err
		}
		return ProbeResult{URL: objectURL, Bytes: response.ContentLength, Method: "HEAD"}, nil
	}
	if response.StatusCode >= 200 && response.StatusCode < 300 || response.StatusCode == http.StatusMethodNotAllowed || response.StatusCode == http.StatusNotImplemented {
		return cache.probeRange(ctx, objectURL, expectedBytes)
	}
	return ProbeResult{}, fmt.Errorf("probe %s: HTTP %s", objectURL, response.Status)
}

func (cache *Cache) probeRange(ctx context.Context, objectURL string, expectedBytes int64) (ProbeResult, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, objectURL, nil)
	if err != nil {
		return ProbeResult{}, err
	}
	request.Header.Set("Range", "bytes=0-0")
	request.Header.Set("Accept-Encoding", "identity")
	response, err := cache.client.Do(request)
	if err != nil {
		return ProbeResult{}, fmt.Errorf("probe %s: %w", objectURL, err)
	}
	defer response.Body.Close()
	var size int64
	switch response.StatusCode {
	case http.StatusPartialContent:
		size, err = contentRangeSize(response.Header.Get("Content-Range"))
		if err != nil {
			return ProbeResult{}, fmt.Errorf("probe %s: %w", objectURL, err)
		}
	case http.StatusOK:
		size = response.ContentLength
		if size < 0 {
			return ProbeResult{}, fmt.Errorf("probe %s: server reported neither object size nor range support", objectURL)
		}
	default:
		return ProbeResult{}, fmt.Errorf("probe %s: HTTP %s", objectURL, response.Status)
	}
	if err := checkProbeSize(objectURL, size, expectedBytes); err != nil {
		return ProbeResult{}, err
	}
	return ProbeResult{URL: objectURL, Bytes: size, Method: "GET range"}, nil
}

func contentRangeSize(value string) (int64, error) {
	const prefix = "bytes 0-0/"
	if !strings.HasPrefix(value, prefix) {
		return 0, fmt.Errorf("invalid Content-Range %q", value)
	}
	size, err := strconv.ParseInt(strings.TrimPrefix(value, prefix), 10, 64)
	if err != nil || size <= 0 {
		return 0, fmt.Errorf("invalid Content-Range %q", value)
	}
	return size, nil
}

func checkProbeSize(objectURL string, got, want int64) error {
	if want > 0 && got != want {
		return fmt.Errorf("probe %s: size mismatch: got %d bytes, want %d", objectURL, got, want)
	}
	return nil
}

func (cache *Cache) Path(digest string) (string, error) {
	if err := validateDigest(digest); err != nil {
		return "", err
	}
	return filepath.Join(cache.root, digest[:2], digest[2:4], digest), nil
}

// Fetch returns a local verified object path. An existing cache entry is
// re-hashed before use. Downloads are streamed to a sibling temporary file and
// become visible only after their digest and optional expected size match.
func (cache *Cache) Fetch(ctx context.Context, objectURL, digest string, expectedBytes int64) (string, error) {
	return cache.FetchWithProgress(ctx, objectURL, digest, expectedBytes, nil)
}

// FetchWithProgress is Fetch with byte-level progress for cache verification
// and remote transfer. Progress is observational and never changes identity
// verification or installation semantics.
func (cache *Cache) FetchWithProgress(ctx context.Context, objectURL, digest string, expectedBytes int64, progress func(FetchProgress)) (string, error) {
	destination, err := cache.Path(digest)
	if err != nil {
		return "", err
	}
	if info, err := os.Stat(destination); err == nil && !info.IsDir() {
		if err := verifyFileWithProgress(destination, digest, expectedBytes, progress); err == nil {
			_ = os.Chtimes(destination, time.Now(), time.Now())
			cache.markUsed(destination)
			return destination, nil
		}
		// A cache entry is derived and addressed by its expected content. Once
		// it fails that identity it has no valid use, so remove it before the
		// repair fetch rather than allowing any consumer to observe it.
		if err := os.Remove(destination); err != nil {
			return "", fmt.Errorf("remove corrupt cached object %s: %w", digest, err)
		}
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return "", err
	}

	candidates := []string{objectURL}
	for _, mirror := range cache.mirrors {
		candidates = append(candidates, mirrorObjectURL(mirror, digest))
	}
	seen := map[string]bool{}
	var failures []error
	for _, candidate := range candidates {
		if seen[candidate] {
			continue
		}
		seen[candidate] = true
		if err := cache.fetchCandidate(ctx, candidate, destination, digest, expectedBytes, progress); err == nil {
			cache.markUsed(destination)
			if cache.retain {
				_ = cache.prune()
			}
			return destination, nil
		} else {
			failures = append(failures, err)
		}
	}
	return "", fmt.Errorf("object %s was unavailable from its manifest URL and %d configured mirror(s): %w", digest, len(cache.mirrors), errors.Join(failures...))
}

// PurgeUsed removes only objects successfully returned by Fetch on this Cache
// instance. Callers invoke it after the consuming operation commits; failures
// deliberately leave objects available for diagnosis and retry.
func (cache *Cache) PurgeUsed() (Stats, error) {
	if cache.retain {
		cache.mu.Lock()
		cache.used = map[string]bool{}
		cache.mu.Unlock()
		return Stats{}, nil
	}
	cache.mu.Lock()
	paths := make([]string, 0, len(cache.used))
	for path := range cache.used {
		paths = append(paths, path)
	}
	cache.mu.Unlock()
	var purged Stats
	for _, objectPath := range paths {
		digest := filepath.Base(objectPath)
		expected, err := cache.Path(digest)
		if err != nil || filepath.Clean(objectPath) != expected {
			return purged, fmt.Errorf("refuse to purge invalid cache object path %q", objectPath)
		}
		info, err := os.Stat(objectPath)
		if err == nil {
			if err := os.Remove(objectPath); err != nil {
				return purged, err
			}
			purged.Objects++
			purged.Bytes += info.Size()
		} else if !os.IsNotExist(err) {
			return purged, err
		}
		cache.mu.Lock()
		delete(cache.used, objectPath)
		cache.mu.Unlock()
		second := filepath.Dir(objectPath)
		first := filepath.Dir(second)
		_ = os.Remove(second)
		_ = os.Remove(first)
	}
	if purged.Objects > 0 {
		if err := syncDirectory(cache.root); err != nil && !os.IsNotExist(err) {
			return purged, err
		}
	}
	return purged, nil
}

func (cache *Cache) markUsed(path string) {
	cache.mu.Lock()
	cache.used[path] = true
	cache.mu.Unlock()
}

func (cache *Cache) fetchCandidate(ctx context.Context, objectURL, destination, digest string, expectedBytes int64, progress func(FetchProgress)) error {
	reader, err := cache.open(ctx, objectURL)
	if err != nil {
		return err
	}
	defer reader.Close()
	if err := cache.EnsureScratch(); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(cache.scratch, ".waldo-download-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() {
		_ = temporary.Close()
		_ = os.Remove(temporaryPath)
	}()

	hasher := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(temporary, hasher), &fetchProgressReader{reader: reader, total: expectedBytes, phase: "download", progress: progress})
	if copyErr != nil {
		return fmt.Errorf("fetch %s: %w", objectURL, copyErr)
	}
	if expectedBytes > 0 && written != expectedBytes {
		return fmt.Errorf("fetch %s: size mismatch: got %d bytes, want %d", objectURL, written, expectedBytes)
	}
	if got := hex.EncodeToString(hasher.Sum(nil)); got != digest {
		return fmt.Errorf("fetch %s: sha256 mismatch: got %s, want %s", objectURL, got, digest)
	}
	if err := temporary.Sync(); err != nil {
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	installed, err := os.CreateTemp(filepath.Dir(destination), ".waldo-object-*")
	if err != nil {
		return err
	}
	installedPath := installed.Name()
	installedOK := false
	defer func() {
		_ = installed.Close()
		if !installedOK {
			_ = os.Remove(installedPath)
		}
	}()
	input, err := os.Open(temporaryPath)
	if err != nil {
		return err
	}
	_, copyErr = io.Copy(installed, input)
	closeErr := input.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	if err := installed.Sync(); err != nil {
		return err
	}
	if err := installed.Close(); err != nil {
		return err
	}
	if err := os.Rename(installedPath, destination); err != nil {
		return err
	}
	installedOK = true
	return nil
}

func (cache *Cache) prune() error {
	if cache.maxBytes <= 0 {
		return nil
	}
	type entry struct {
		path string
		size int64
		used time.Time
	}
	var entries []entry
	var total int64
	err := cache.walk(func(path, digest string, info os.FileInfo) error {
		if digest != "" {
			entries = append(entries, entry{path, info.Size(), info.ModTime()})
			total += info.Size()
		}
		return nil
	})
	if err != nil || total <= cache.maxBytes {
		return err
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].used.Before(entries[j].used) })
	cache.mu.Lock()
	inUse := make(map[string]bool, len(cache.used))
	for path := range cache.used {
		inUse[path] = true
	}
	cache.mu.Unlock()
	for _, candidate := range entries {
		if total <= cache.maxBytes {
			break
		}
		if inUse[candidate.path] {
			continue
		}
		if err := os.Remove(candidate.path); err != nil && !os.IsNotExist(err) {
			return err
		}
		total -= candidate.size
	}
	return nil
}

func (cache *Cache) open(ctx context.Context, objectURL string) (io.ReadCloser, error) {
	parsed, err := url.Parse(objectURL)
	if err != nil {
		return nil, fmt.Errorf("parse lookaside URL %q: %w", objectURL, err)
	}
	switch parsed.Scheme {
	case "http", "https":
		return cache.openHTTP(ctx, parsed.String())
	case "s3":
		return cache.openHTTP(ctx, s3HTTPS(parsed))
	case "file":
		path, err := url.PathUnescape(parsed.Path)
		if err != nil {
			return nil, err
		}
		return os.Open(filepath.FromSlash(path))
	case "":
		return os.Open(objectURL)
	default:
		return nil, fmt.Errorf("unsupported lookaside URL scheme %q", parsed.Scheme)
	}
}

func (cache *Cache) openHTTP(ctx context.Context, objectURL string) (io.ReadCloser, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, objectURL, nil)
	if err != nil {
		return nil, err
	}
	response, err := cache.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("fetch %s: %w", objectURL, err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		_ = response.Body.Close()
		return nil, fmt.Errorf("fetch %s: HTTP %s", objectURL, response.Status)
	}
	return response.Body, nil
}

func s3HTTPS(parsed *url.URL) string {
	host := parsed.Host
	if strings.HasPrefix(host, "s3.") || strings.HasPrefix(host, "s3-") || host == "s3.amazonaws.com" {
		return (&url.URL{Scheme: "https", Host: host, Path: parsed.Path, RawQuery: parsed.RawQuery}).String()
	}
	return (&url.URL{Scheme: "https", Host: host + ".s3.amazonaws.com", Path: parsed.Path, RawQuery: parsed.RawQuery}).String()
}

func mirrorObjectURL(base, digest string) string {
	objectPath := path.Join(digest[:2], digest[2:4], digest)
	parsed, err := url.Parse(base)
	if err == nil && parsed.Scheme != "" {
		parsed.Path = path.Join(parsed.Path, objectPath)
		return parsed.String()
	}
	return filepath.Join(base, filepath.FromSlash(objectPath))
}

func VerifyFile(path, digest string, expectedBytes int64) error {
	return verifyFileWithProgress(path, digest, expectedBytes, nil)
}

func verifyFileWithProgress(path, digest string, expectedBytes int64, progress func(FetchProgress)) error {
	if err := validateDigest(digest); err != nil {
		return err
	}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	if expectedBytes > 0 && info.Size() != expectedBytes {
		return fmt.Errorf("%s: size mismatch: got %d bytes, want %d", path, info.Size(), expectedBytes)
	}
	hasher := sha256.New()
	total := expectedBytes
	if total <= 0 {
		total = info.Size()
	}
	if _, err := io.Copy(hasher, &fetchProgressReader{reader: file, total: total, phase: "cache", progress: progress}); err != nil {
		return err
	}
	return compareDigest(path, hasher, digest)
}

type fetchProgressReader struct {
	reader   io.Reader
	written  int64
	total    int64
	phase    string
	progress func(FetchProgress)
}

func (reader *fetchProgressReader) Read(buffer []byte) (int, error) {
	read, err := reader.reader.Read(buffer)
	reader.written += int64(read)
	if reader.progress != nil && (read > 0 || err != nil) {
		reader.progress(FetchProgress{Phase: reader.phase, Written: reader.written, Total: reader.total})
	}
	return read, err
}

func compareDigest(path string, hasher hash.Hash, want string) error {
	if got := hex.EncodeToString(hasher.Sum(nil)); got != want {
		return fmt.Errorf("%s: sha256 mismatch: got %s, want %s", path, got, want)
	}
	return nil
}

func validateDigest(digest string) error {
	if len(digest) != sha256.Size*2 {
		return fmt.Errorf("invalid sha256 %q: want 64 lowercase hexadecimal characters", digest)
	}
	decoded, err := hex.DecodeString(digest)
	if err != nil || hex.EncodeToString(decoded) != digest {
		return fmt.Errorf("invalid sha256 %q: want 64 lowercase hexadecimal characters", digest)
	}
	return nil
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (reader *contextReader) Read(buffer []byte) (int, error) {
	if err := reader.ctx.Err(); err != nil {
		return 0, err
	}
	return reader.reader.Read(buffer)
}

func syncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	defer directory.Close()
	return directory.Sync()
}
