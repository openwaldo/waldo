// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

const ComposeHistoryDirectory = "composes"

var composeHistoryName = regexp.MustCompile(`^(\d{4})-(.+)\.yaml$`)
var sourceComposeOrdinal = regexp.MustCompile(`^[0-9]{4}-(.+)$`)
var composeSlugInvalid = regexp.MustCompile(`[^a-z0-9._-]+`)

// ArchiveCompose preserves each distinct compose in ordered, human-readable
// YAML while COMPOSE.json remains the canonical compatibility record.
func ArchiveCompose(modelPath string, compose Compose, sourceName string) (string, error) {
	directory := filepath.Join(modelPath, ComposeHistoryDirectory)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return "", err
	}
	entries, err := os.ReadDir(directory)
	if err != nil {
		return "", err
	}
	next := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		match := composeHistoryName.FindStringSubmatch(entry.Name())
		if match == nil {
			continue
		}
		ordinal, _ := strconv.Atoi(match[1])
		if ordinal >= next {
			next = ordinal + 1
		}
		existing, _, loadErr := LoadCompose(filepath.Join(directory, entry.Name()))
		if loadErr != nil {
			return "", fmt.Errorf("load compose history %s: %w", entry.Name(), loadErr)
		}
		if reflect.DeepEqual(existing, compose) {
			return filepath.Join(directory, entry.Name()), nil
		}
	}
	if next > 9999 {
		return "", fmt.Errorf("compose history exceeds 10000 entries")
	}
	slug := composeSlug(sourceName)
	path := filepath.Join(directory, fmt.Sprintf("%04d-%s.yaml", next, slug))
	data, err := yaml.Marshal(compose)
	if err != nil {
		return "", err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return "", err
	}
	written := false
	defer func() {
		_ = file.Close()
		if !written {
			_ = os.Remove(path)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return "", err
	}
	if err := file.Sync(); err != nil {
		return "", err
	}
	if err := file.Close(); err != nil {
		return "", err
	}
	written = true
	return path, nil
}

// PersistCompletedCompose restores the complete requested compose as the
// model's canonical current compose after its completed runs have been
// verified. This is used when an older interrupted retry persisted only the
// remaining stages before WALDO's durable compose normalization was fixed.
func PersistCompletedCompose(modelPath string, compose Compose, sourceName string) error {
	if _, err := ArchiveCompose(modelPath, compose, sourceName); err != nil {
		return err
	}
	return writeJSONAtomic(filepath.Join(modelPath, "COMPOSE.json"), compose)
}

func composeSlug(sourceName string) string {
	name := strings.ToLower(strings.TrimSpace(filepath.Base(sourceName)))
	for _, extension := range []string{".yaml", ".yml", ".json"} {
		name = strings.TrimSuffix(name, extension)
	}
	if match := sourceComposeOrdinal.FindStringSubmatch(name); match != nil {
		name = match[1]
	}
	name = strings.Trim(composeSlugInvalid.ReplaceAllString(name, "-"), "-._")
	if name == "" {
		return "compose"
	}
	return name
}

// LatestComposePath returns the newest numbered compose, with COMPOSE.json as
// a compatibility fallback for models created before compose history existed.
func LatestComposePath(modelPath string) (string, error) {
	directory := filepath.Join(modelPath, ComposeHistoryDirectory)
	entries, err := os.ReadDir(directory)
	if err == nil {
		var names []string
		for _, entry := range entries {
			if !entry.IsDir() && composeHistoryName.MatchString(entry.Name()) {
				names = append(names, entry.Name())
			}
		}
		if len(names) > 0 {
			sort.Strings(names)
			return filepath.Join(directory, names[len(names)-1]), nil
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	fallback := filepath.Join(modelPath, "COMPOSE.json")
	if _, err := os.Stat(fallback); err != nil {
		return "", err
	}
	return fallback, nil
}

// HasPendingCompose reports whether a durable compose transaction exists for
// the model. A transaction is the authority for whether continue is legal.
func HasPendingCompose(root, name string) (bool, error) {
	if err := ValidateName(name); err != nil {
		return false, err
	}
	transaction, err := pendingComposeTransaction(root, name)
	return transaction != nil, err
}
