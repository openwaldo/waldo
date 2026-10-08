// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package config owns machine-local preferences. Configuration may influence
// transport and execution choices, but it never supplies corpus meaning.
package config

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"os/user"
	"path/filepath"
	"strings"
)

const Schema = 1

type Config struct {
	Schema     int        `json:"schema"`
	Index      string     `json:"index,omitempty"`
	Lookaside  Lookaside  `json:"lookaside,omitempty"`
	Ingest     Ingest     `json:"ingest,omitempty"`
	Model      Model      `json:"model,omitempty"`
	AI         AI         `json:"ai,omitempty"`
	Disclosure Disclosure `json:"disclosure,omitempty"`
	Signing    Signing    `json:"signing,omitempty"`
}

type Disclosure struct {
	Provider string `json:"provider,omitempty"`
}

type Signing struct {
	Method string `json:"method,omitempty"`
	Key    string `json:"key,omitempty"`
}

type Ingest struct {
	Staging string `json:"staging,omitempty"`
}

type Model struct {
	Root          string `json:"root,omitempty"`
	Backend       string `json:"backend,omitempty"`
	NCCLInterface string `json:"nccl_interface,omitempty"`
	NCCLHCA       string `json:"nccl_hca,omitempty"`
}

type AI struct {
	Provider string `json:"provider,omitempty"`
	Model    string `json:"model,omitempty"`
	APIKey   string `json:"api_key,omitempty"`
}

type Lookaside struct {
	Cache           string   `json:"cache,omitempty"`
	CacheMaxBytes   int64    `json:"cache_max_bytes,omitempty"`
	RetainCompleted bool     `json:"retain_completed,omitempty"`
	Scratch         string   `json:"scratch,omitempty"`
	Mirrors         []string `json:"mirrors,omitempty"`
	Publish         *Publish `json:"publish,omitempty"`
}

type Publish struct {
	URL     string `json:"url"`
	Region  string `json:"region,omitempty"`
	Workers int    `json:"workers,omitempty"`
}

func Default() Config { return Config{Schema: Schema} }

func Path() (string, error) {
	if configured := os.Getenv("WALDO_CONFIG"); configured != "" {
		return filepath.Abs(configured)
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("find user configuration directory: %w", err)
	}
	return filepath.Join(dir, "waldo", "config.json"), nil
}

func Load() (Config, error) {
	path, err := Path()
	if err != nil {
		return Config{}, err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return Default(), nil
	}
	if err != nil {
		return Config{}, err
	}
	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	if config.Schema != Schema {
		return Config{}, fmt.Errorf("%s: unsupported configuration schema %d", path, config.Schema)
	}
	config.Lookaside.Mirrors = normalizeMirrors(config.Lookaside.Mirrors)
	if err := validateMirrors(config.Lookaside.Mirrors); err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	if err := validatePublish(config.Lookaside.Publish); err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	if config.Lookaside.CacheMaxBytes < 0 {
		return Config{}, fmt.Errorf("%s: lookaside cache maximum must not be negative", path)
	}
	if err := validateModelBackend(config.Model.Backend); err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	if err := validateAI(config.AI); err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	if err := validateSigning(config.Signing); err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	return config, nil
}

func Save(config Config) error {
	config.Schema = Schema
	config.Lookaside.Mirrors = normalizeMirrors(config.Lookaside.Mirrors)
	if err := validateMirrors(config.Lookaside.Mirrors); err != nil {
		return err
	}
	if err := validatePublish(config.Lookaside.Publish); err != nil {
		return err
	}
	if config.Lookaside.CacheMaxBytes < 0 {
		return fmt.Errorf("lookaside cache maximum must not be negative")
	}
	if err := validateModelBackend(config.Model.Backend); err != nil {
		return err
	}
	if err := validateAI(config.AI); err != nil {
		return err
	}
	if err := validateSigning(config.Signing); err != nil {
		return err
	}
	path, err := Path()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".waldo-config-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	committed := false
	defer func() {
		_ = temporary.Close()
		if !committed {
			_ = os.Remove(temporaryPath)
		}
	}()
	if err := temporary.Chmod(0o600); err != nil {
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		return err
	}
	if err := temporary.Sync(); err != nil {
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return err
	}
	committed = true
	return nil
}

func validateAI(value AI) error {
	if value.Provider != "" && value.Provider != "auto" && value.Provider != "openai" && value.Provider != "anthropic" && value.Provider != "deterministic" && value.Provider != "local" {
		return fmt.Errorf("AI provider must be auto, openai, anthropic, deterministic, or local")
	}
	return nil
}

func validatePublish(publish *Publish) error {
	if publish == nil {
		return nil
	}
	parsed, err := url.Parse(strings.TrimRight(strings.TrimSpace(publish.URL), "/"))
	if err != nil {
		return fmt.Errorf("invalid lookaside publish URL: %w", err)
	}
	switch parsed.Scheme {
	case "s3":
		if parsed.Host == "" {
			return fmt.Errorf("S3 lookaside publish URL requires a bucket")
		}
	case "file":
		if (parsed.Host != "" && parsed.Host != "localhost") || parsed.Path == "" || !filepath.IsAbs(filepath.FromSlash(parsed.Path)) || parsed.RawQuery != "" || parsed.Fragment != "" {
			return fmt.Errorf("local lookaside publish URL must be an absolute file:/// URL")
		}
		if publish.Region != "" {
			return fmt.Errorf("S3 region cannot be used with a local publisher")
		}
	default:
		return fmt.Errorf("lookaside publish URL must use s3:// or file://")
	}
	publish.URL = parsed.String()
	if publish.Workers == 0 {
		publish.Workers = 4
	}
	if publish.Workers < 1 || publish.Workers > 32 {
		return fmt.Errorf("lookaside publish workers must be in 1..32")
	}
	return nil
}

func EffectiveScratchRoot(config Config) (string, error) {
	if config.Lookaside.Scratch != "" {
		return filepath.Abs(config.Lookaside.Scratch)
	}
	return filepath.Join(temporaryRoot(), "scratch"), nil
}

func EffectiveCacheRoot(config Config) (string, error) {
	if config.Lookaside.Cache != "" {
		return filepath.Abs(config.Lookaside.Cache)
	}
	return filepath.Join(temporaryRoot(), "cache"), nil
}

func EffectiveCacheMaxBytes(config Config) int64 {
	if config.Lookaside.CacheMaxBytes > 0 {
		return config.Lookaside.CacheMaxBytes
	}
	return 20 << 30
}

func EffectiveStagingBase(config Config) (string, error) {
	if config.Ingest.Staging != "" {
		return filepath.Abs(config.Ingest.Staging)
	}
	return filepath.Join(temporaryRoot(), "ingest"), nil
}

func EffectiveModelRoot(config Config) (string, error) {
	if config.Model.Root != "" {
		return filepath.Abs(config.Model.Root)
	}
	base, err := durableRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "models"), nil
}

// EffectiveIndexRoot returns the selected index checkout and whether it is
// WALDO's managed read-only checkout.
func EffectiveIndexRoot(config Config) (string, bool, error) {
	if config.Index != "" {
		absolute, err := filepath.Abs(config.Index)
		return absolute, false, err
	}
	root, err := ManagedIndexRoot()
	return root, true, err
}

// ManagedIndexRoot is WALDO's automatically synchronized index checkout.
func ManagedIndexRoot() (string, error) {
	base, err := durableRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "index"), nil
}

// IsManagedIndexPath reports whether path is the reserved managed index or is
// contained beneath it. It is intentionally lexical as well as symlink-aware
// when both paths exist.
func IsManagedIndexPath(path string) (bool, error) {
	managed, err := ManagedIndexRoot()
	if err != nil {
		return false, err
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return false, err
	}
	if pathWithin(managed, absolute) {
		return true, nil
	}
	managedResolved, managedErr := filepath.EvalSymlinks(managed)
	pathResolved, pathErr := filepath.EvalSymlinks(absolute)
	return managedErr == nil && pathErr == nil && pathWithin(managedResolved, pathResolved), nil
}

func pathWithin(parent, child string) bool {
	relative, err := filepath.Rel(filepath.Clean(parent), filepath.Clean(child))
	return err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator))
}

func EffectiveModelBackend(config Config) string {
	if config.Model.Backend != "" {
		return config.Model.Backend
	}
	return "auto"
}

func validateModelBackend(backend string) error {
	if backend != "" && backend != "auto" && backend != "mlx" && backend != "torchtitan" && backend != "pytorch" && backend != "fake" {
		return fmt.Errorf("model backend must be auto, mlx, torchtitan, pytorch, or fake")
	}
	return nil
}

func validateSigning(signing Signing) error {
	if signing.Method != "" && signing.Method != "sigstore-keyless" && signing.Method != "sigstore-key" {
		return fmt.Errorf("signing method must be sigstore-keyless or sigstore-key")
	}
	if signing.Method == "sigstore-keyless" && signing.Key != "" {
		return fmt.Errorf("signing.key cannot be used with sigstore-keyless")
	}
	return nil
}

func durableRoot() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("find home directory: %w", err)
	}
	return filepath.Join(home, ".waldo"), nil
}

func temporaryRoot() string {
	name := "waldo"
	if current, err := user.Current(); err == nil && current.Uid != "" {
		uid := strings.Map(func(character rune) rune {
			if character >= '0' && character <= '9' || character >= 'A' && character <= 'Z' || character >= 'a' && character <= 'z' || character == '-' || character == '_' {
				return character
			}
			return '-'
		}, current.Uid)
		if uid != "" {
			name += "-" + uid
		}
	}
	return filepath.Join(os.TempDir(), name)
}

// EffectiveStagingRoot returns a private, plan-specific staging directory.
func EffectiveStagingRoot(config Config, identity string) (string, error) {
	base, err := EffectiveStagingBase(config)
	if err != nil {
		return "", err
	}
	return filepath.Join(base, identity), nil
}

func normalizeMirrors(values []string) []string {
	result := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		value = strings.TrimRight(strings.TrimSpace(value), "/")
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		result = append(result, value)
	}
	return result
}

func validateMirrors(values []string) error {
	for _, value := range values {
		parsed, err := url.Parse(value)
		if err != nil {
			return fmt.Errorf("invalid lookaside mirror %q: %w", value, err)
		}
		switch parsed.Scheme {
		case "", "file", "http", "https", "s3":
		default:
			return fmt.Errorf("lookaside mirror %q uses unsupported scheme %q", value, parsed.Scheme)
		}
	}
	return nil
}
