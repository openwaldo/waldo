// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package index

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const indexFile = "index.yaml"

var indexFiles = []string{"index.yaml", "index.yml", "index.json"}

// Target is a resolved path inside one explicit index checkout.
type Target struct {
	Root string
	Abs  string
	Rel  string
}

// Resolve locates a target inside an index checkout. A knownRoot confines the
// target to an already-discovered checkout. With no known root, an explicit
// filesystem path discovers its checkout and a logical path uses the current
// checkout.
func Resolve(knownRoot, target string) (Target, error) {
	if knownRoot != "" {
		root, err := findRoot(knownRoot)
		if err != nil {
			return Target{}, fmt.Errorf("index checkout %s: %w", knownRoot, err)
		}
		if target == "" {
			return Target{Root: root, Abs: root}, nil
		}
		return within(root, target)
	}

	if target != "" && isFilesystemPath(target) {
		abs, err := filepath.Abs(target)
		if err != nil {
			return Target{}, err
		}
		root, err := findRoot(abs)
		if err != nil && os.IsNotExist(err) {
			root, err = findRoot(filepath.Dir(abs))
		}
		if err != nil {
			return Target{}, err
		}
		return within(root, abs)
	}

	cwd, err := os.Getwd()
	if err != nil {
		return Target{}, err
	}
	root, err := findRoot(cwd)
	if err != nil {
		return Target{}, fmt.Errorf("current directory: %w (pass an index path positionally)", err)
	}
	if target == "" {
		return Target{Root: root, Abs: root}, nil
	}
	return within(root, target)
}

// ResolveConfigured gives explicit filesystem paths precedence over a machine
// default. Unadorned logical paths are confined to the configured checkout.
func ResolveConfigured(configuredRoot, target string) (Target, error) {
	expanded, err := expandUser(target)
	if err != nil {
		return Target{}, err
	}
	if target != "" && IsFilesystemPath(expanded) {
		return Resolve("", expanded)
	}
	return Resolve(configuredRoot, target)
}

// ResolveDestination discovers the checkout for a prospective positional
// destination. Unlike Resolve, the leaf need not exist yet; its nearest
// existing ancestor must be inside an index checkout.
func ResolveDestination(target string) (Target, error) {
	return ResolveDestinationConfigured("", target)
}

// ResolveDestinationConfigured resolves relative prospective destinations
// beneath a configured checkout while preserving absolute-path discovery.
func ResolveDestinationConfigured(configuredRoot, target string) (Target, error) {
	if strings.TrimSpace(target) == "" {
		return Target{}, fmt.Errorf("index destination is required")
	}
	expanded, err := expandUser(target)
	if err != nil {
		return Target{}, err
	}
	filesystemPath := IsFilesystemPath(expanded)
	var abs, root string
	if configuredRoot != "" && !filesystemPath {
		root, err = findRoot(configuredRoot)
		if err != nil {
			return Target{}, fmt.Errorf("index checkout %s: %w", configuredRoot, err)
		}
		abs = filepath.Join(root, filepath.FromSlash(expanded))
	} else if filesystemPath {
		abs, err = filepath.Abs(expanded)
		if err != nil {
			return Target{}, err
		}
	} else {
		cwd, err := os.Getwd()
		if err != nil {
			return Target{}, err
		}
		root, err := findRoot(cwd)
		if err != nil {
			return Target{}, fmt.Errorf("current directory: %w (pass an absolute destination inside an index checkout)", err)
		}
		abs = filepath.Join(root, filepath.FromSlash(expanded))
	}
	abs = filepath.Clean(abs)
	ancestor := abs
	for {
		if _, err := os.Stat(ancestor); err == nil {
			break
		} else if !os.IsNotExist(err) {
			return Target{}, err
		}
		parent := filepath.Dir(ancestor)
		if parent == ancestor {
			return Target{}, fmt.Errorf("index destination %s has no existing ancestor", target)
		}
		ancestor = parent
	}
	if root == "" {
		root, err = findRoot(ancestor)
		if err != nil {
			return Target{}, fmt.Errorf("index destination %s: %w", target, err)
		}
	}
	rel, err := filepath.Rel(root, abs)
	if err != nil {
		return Target{}, err
	}
	if rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return Target{}, fmt.Errorf("index destination must name a new path beneath checkout %s", root)
	}
	return Target{Root: root, Abs: abs, Rel: filepath.ToSlash(rel)}, nil
}

func findRoot(location string) (string, error) {
	location, err := expandUser(location)
	if err != nil {
		return "", err
	}
	abs, err := filepath.Abs(location)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		abs = filepath.Dir(abs)
	}

	var nearest string
	for dir := abs; ; dir = filepath.Dir(dir) {
		present, err := isIndexDirectory(dir)
		if err != nil {
			return "", err
		}
		if present {
			nearest = dir
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}
	if nearest == "" {
		return "", fmt.Errorf("no index.yaml, index.yml, or index.json found in this path or its parents")
	}

	root := nearest
	for {
		parent := filepath.Dir(root)
		if parent == root {
			return root, nil
		}
		present, err := isIndexDirectory(parent)
		if err != nil {
			return "", err
		}
		if !present {
			return root, nil
		}
		root = parent
	}
}

func within(root, target string) (Target, error) {
	abs, err := expandUser(target)
	if err != nil {
		return Target{}, err
	}
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(root, filepath.FromSlash(abs))
	}
	abs = filepath.Clean(abs)
	rel, err := filepath.Rel(root, abs)
	if err != nil {
		return Target{}, err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return Target{}, fmt.Errorf("target %s is outside index checkout %s", target, root)
	}
	if _, err := os.Stat(abs); err != nil {
		if !os.IsNotExist(err) {
			return Target{}, fmt.Errorf("index path %s: %w", target, err)
		}
		manifest, found, resolveErr := indexedManifestPath(abs)
		if resolveErr != nil {
			return Target{}, fmt.Errorf("index path %s: %w", target, resolveErr)
		}
		if !found {
			return Target{}, fmt.Errorf("index path %s: %w", target, err)
		}
		abs = manifest
		rel, err = filepath.Rel(root, abs)
		if err != nil {
			return Target{}, err
		}
	}
	if rel == "." {
		rel = ""
	}
	return Target{Root: root, Abs: abs, Rel: filepath.ToSlash(rel)}, nil
}

func indexedManifestPath(logicalPath string) (string, bool, error) {
	dir := filepath.Dir(logicalPath)
	directory, err := LoadDirectory(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", false, nil
		}
		return "", false, err
	}
	name := filepath.Base(logicalPath)
	var matches []string
	var entries []Entry
	for _, entry := range directory.Entries {
		if entry.Type != "manifest" || !ValidEntryName(entry.Name) {
			continue
		}
		extension := strings.ToLower(filepath.Ext(entry.Name))
		if extension != ".json" && extension != ".yaml" && extension != ".yml" {
			continue
		}
		if strings.TrimSuffix(entry.Name, filepath.Ext(entry.Name)) == name {
			matches = append(matches, filepath.Join(dir, entry.Name))
			entries = append(entries, entry)
		}
	}
	if len(matches) == 0 {
		return "", false, nil
	}
	if len(matches) > 1 {
		return "", false, fmt.Errorf("logical corpus path is ambiguous; matches %s", strings.Join(matches, ", "))
	}
	path, err := EntryPath(dir, entries[0])
	if err != nil {
		return "", false, err
	}
	return path, true, nil
}

func isFilesystemPath(value string) bool {
	return filepath.IsAbs(value) || strings.HasPrefix(value, ".") || strings.HasPrefix(value, "~")
}

// IsFilesystemPath reports whether a positional index argument clearly names
// the local filesystem rather than a logical path in the configured index.
// Explicit path syntax always wins. An unprefixed relative path also wins when
// it or its non-current-directory parent already exists locally.
func IsFilesystemPath(value string) bool {
	if isFilesystemPath(value) {
		return true
	}
	expanded, err := expandUser(value)
	if err != nil {
		return false
	}
	if _, err := os.Stat(expanded); err == nil {
		return true
	}
	parent := filepath.Dir(expanded)
	if parent == "." || parent == "" {
		return false
	}
	info, err := os.Stat(parent)
	return err == nil && info.IsDir()
}

func expandUser(value string) (string, error) {
	if value != "~" && !strings.HasPrefix(value, "~/") {
		return value, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("expand %s: %w", value, err)
	}
	if value == "~" {
		return home, nil
	}
	return filepath.Join(home, filepath.FromSlash(strings.TrimPrefix(value, "~/"))), nil
}

func isIndexDirectory(dir string) (bool, error) {
	_, err := DirectoryPath(dir)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

// DirectoryPath resolves the single directory-navigation document. JSON and
// both normal YAML suffixes are readable; competing files are rejected.
func DirectoryPath(dir string) (string, error) {
	var matches []string
	for _, name := range indexFiles {
		path := filepath.Join(dir, name)
		info, err := os.Stat(path)
		if err == nil && !info.IsDir() {
			matches = append(matches, path)
		} else if err != nil && !os.IsNotExist(err) {
			return "", err
		}
	}
	if len(matches) == 0 {
		return "", &os.PathError{Op: "open", Path: filepath.Join(dir, indexFile), Err: os.ErrNotExist}
	}
	if len(matches) > 1 {
		return "", fmt.Errorf("index directory %s contains competing metadata files %s; keep exactly one", dir, strings.Join(matches, ", "))
	}
	return matches[0], nil
}

func LoadDirectory(dir string) (Directory, error) {
	path, err := DirectoryPath(dir)
	if err != nil {
		return Directory{}, err
	}
	if info, err := os.Lstat(path); err != nil {
		return Directory{}, err
	} else if !info.Mode().IsRegular() {
		return Directory{}, fmt.Errorf("%s: index metadata must be a regular file, not a symbolic link or special file", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Directory{}, err
	}
	var index Directory
	if err := decodeMetadata(path, data, &index); err != nil {
		return Directory{}, fmt.Errorf("%s: %w", path, err)
	}
	if !SupportedDirectorySchema(index.Schema) {
		return Directory{}, fmt.Errorf("%s: unsupported index schema %d", path, index.Schema)
	}
	// Schema 2 used the same directory fields as schema 1. Normalize it at the
	// read boundary so any subsequently written metadata uses the current schema.
	index.Schema = DirectorySchema
	return index, nil
}

func LoadManifest(path string) (Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Manifest{}, err
	}
	var manifest Manifest
	if err := decodeMetadata(path, data, &manifest); err != nil {
		return Manifest{}, fmt.Errorf("%s: %w", path, err)
	}
	if manifest.Schema != LegacyManifestSchema && manifest.Schema != ManifestSchema {
		return Manifest{}, fmt.Errorf("%s: unsupported manifest schema %d", path, manifest.Schema)
	}
	return manifest, nil
}

// ValidEntryName reports whether name can appear in index directory metadata:
// a single path component other than "." or "..".
func ValidEntryName(name string) bool {
	return name != "" && name != "." && name != ".." && filepath.Base(name) == name
}

// EntryPath resolves an indexed entry to its path beneath dir. The name must
// be a single path component, and the filesystem object must match the
// declared type without following symbolic links, so walking the index can
// neither leave the checkout nor revisit a directory.
func EntryPath(dir string, entry Entry) (string, error) {
	if !ValidEntryName(entry.Name) {
		return "", fmt.Errorf("invalid entry name %q", entry.Name)
	}
	path := filepath.Join(dir, entry.Name)
	info, err := os.Lstat(path)
	if err != nil {
		return "", fmt.Errorf("indexed entry %q: %w", entry.Name, err)
	}
	switch {
	case info.Mode()&os.ModeSymlink != 0:
		return "", fmt.Errorf("indexed entry %q is a symbolic link", entry.Name)
	case entry.Type == "dir" && !info.IsDir():
		return "", fmt.Errorf("entry %q is declared as a directory but is not a directory", entry.Name)
	case entry.Type == "manifest" && !info.Mode().IsRegular():
		return "", fmt.Errorf("entry %q is declared as a manifest but is not a regular file", entry.Name)
	}
	return path, nil
}

// WalkCorpora visits every manifest indexed beneath target. Filesystem content
// absent from the directory's index metadata is deliberately ignored.
func WalkCorpora(target Target, visit func(Corpus) error) error {
	info, err := os.Stat(target.Abs)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		manifest, err := LoadManifest(target.Abs)
		if err != nil {
			return err
		}
		return visit(Corpus{Path: target.Rel, Manifest: manifest})
	}
	return walkDirectory(target.Root, target.Abs, visit)
}

func walkDirectory(root, dir string, visit func(Corpus) error) error {
	index, err := LoadDirectory(dir)
	if err != nil {
		return err
	}
	for _, entry := range index.Entries {
		path, err := EntryPath(dir, entry)
		if err != nil {
			directoryPath, _ := DirectoryPath(dir)
			return fmt.Errorf("%s: %w", directoryPath, err)
		}
		switch entry.Type {
		case "dir":
			if err := walkDirectory(root, path, visit); err != nil {
				return err
			}
		case "manifest":
			manifest, err := LoadManifest(path)
			if err != nil {
				return err
			}
			rel, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			if err := visit(Corpus{Path: filepath.ToSlash(rel), Manifest: manifest}); err != nil {
				return err
			}
		default:
			directoryPath, _ := DirectoryPath(dir)
			return fmt.Errorf("%s: entry %q has unsupported type %q", directoryPath, entry.Name, entry.Type)
		}
	}
	return nil
}

func SortedEntries(entries []Entry) []Entry {
	result := append([]Entry(nil), entries...)
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result
}
