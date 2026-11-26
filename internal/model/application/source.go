// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package application

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	git "github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/filemode"
	"github.com/go-git/go-git/v5/plumbing/object"
)

type ResolvedSource struct {
	Source         Source   `json:"source"`
	Tree           string   `json:"tree"`
	ManifestSHA256 string   `json:"manifest_sha256"`
	Directory      string   `json:"-"`
	Manifest       Manifest `json:"-"`
}

// ResolveSource acquires a pinned commit without running hooks, submodules,
// filters or application code. Cache entries are verified on every reuse.
func ResolveSource(ctx context.Context, cache string, source Source, kind, environment string) (ResolvedSource, error) {
	var result ResolvedSource
	if err := source.Validate(); err != nil {
		return result, err
	}
	if kind != "waldo-trainer" && kind != "waldo-tokenizer" {
		return result, fmt.Errorf("unknown application kind %q", kind)
	}
	source.Commit = strings.ToLower(source.Commit)
	key, _ := json.Marshal([]string{source.Repository, source.Commit})
	entry := filepath.Join(cache, Digest(key))
	if err := os.MkdirAll(cache, 0755); err != nil {
		return result, err
	}
	if _, err := os.Lstat(entry); os.IsNotExist(err) {
		temp, err := os.MkdirTemp(cache, ".source-")
		if err != nil {
			return result, err
		}
		defer removeTemporarySource(temp)
		repo, err := git.PlainCloneContext(ctx, filepath.Join(temp, "git"), true, &git.CloneOptions{URL: source.Repository, NoCheckout: true})
		if err != nil {
			return result, fmt.Errorf("fetch pinned source: %w", err)
		}
		if err := verifyGitObject(repo, plumbing.NewHash(source.Commit)); err != nil {
			return result, err
		}
		commit, err := repo.CommitObject(plumbing.NewHash(source.Commit))
		if err != nil {
			return result, fmt.Errorf("pinned commit %s unavailable: %w", source.Commit, err)
		}
		tree, err := commit.Tree()
		if err != nil {
			return result, err
		}
		if err := exportTree(tree, filepath.Join(temp, "checkout")); err != nil {
			return result, err
		}
		if err := makeReadOnly(temp); err != nil {
			return result, err
		}
		if err := os.Rename(temp, entry); err != nil {
			if _, other := os.Stat(entry); other != nil {
				return result, err
			}
		}
	} else if err != nil {
		return result, err
	}
	if err := rejectLinks(entry); err != nil {
		return result, err
	}
	repo, err := git.PlainOpen(filepath.Join(entry, "git"))
	if err != nil {
		return result, fmt.Errorf("open source cache: %w", err)
	}
	if err := verifyGitObject(repo, plumbing.NewHash(source.Commit)); err != nil {
		return result, err
	}
	commit, err := repo.CommitObject(plumbing.NewHash(source.Commit))
	if err != nil {
		return result, err
	}
	tree, err := commit.Tree()
	if err != nil {
		return result, err
	}
	if err := verifyGitTree(repo, tree); err != nil {
		return result, err
	}
	checkout := filepath.Join(entry, "checkout")
	if err := verifyTree(tree, checkout); err != nil {
		return result, fmt.Errorf("source cache integrity: %w", err)
	}
	selected := tree
	if source.Path != "." {
		selected, err = tree.Tree(source.Path)
		if err != nil {
			return result, fmt.Errorf("source.path %s: %w", source.Path, err)
		}
	}
	directory := filepath.Join(checkout, filepath.FromSlash(source.Path))
	data, err := os.ReadFile(filepath.Join(directory, kind+".yaml"))
	if err != nil {
		return result, err
	}
	manifest, digest, err := ParseManifest(data, kind)
	if err != nil {
		return result, fmt.Errorf("%s.yaml: %w", kind, err)
	}
	env, ok := manifest.Environments[environment]
	if !ok {
		return result, fmt.Errorf("manifest has no environment %q", environment)
	}
	for _, name := range []string{env.Lock, manifest.Entrypoint.Script} {
		if name == "" {
			continue
		}
		info, err := os.Stat(filepath.Join(directory, filepath.FromSlash(name)))
		if err != nil {
			return result, err
		}
		if !info.Mode().IsRegular() {
			return result, fmt.Errorf("application file %s is not regular", name)
		}
	}
	return ResolvedSource{Source: source, Tree: selected.Hash.String(), ManifestSHA256: digest, Directory: directory, Manifest: manifest}, nil
}

func exportTree(tree *object.Tree, root string) error {
	if err := validateTreeModes(tree); err != nil {
		return err
	}
	if err := os.Mkdir(root, 0755); err != nil {
		return err
	}
	return tree.Files().ForEach(func(f *object.File) error {
		if f.Mode != filemode.Regular && f.Mode != filemode.Executable {
			return fmt.Errorf("source file %s has unsupported mode %s (symlinks and submodules are not supported)", f.Name, f.Mode)
		}
		if !relativePath(f.Name) || f.Name == "." {
			return fmt.Errorf("unsafe source path %q", f.Name)
		}
		dest := filepath.Join(root, filepath.FromSlash(f.Name))
		if err := os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
			return err
		}
		r, err := f.Reader()
		if err != nil {
			return err
		}
		defer r.Close()
		mode := os.FileMode(0644)
		if f.Mode == filemode.Executable {
			mode = 0755
		}
		w, err := os.OpenFile(dest, os.O_CREATE|os.O_EXCL|os.O_WRONLY, mode)
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(w, r)
		closeErr := w.Close()
		if copyErr != nil {
			return copyErr
		}
		return closeErr
	})
}
func rejectLinks(root string) error {
	return filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("symlink in source cache: %s", path)
		}
		return nil
	})
}
func verifyTree(tree *object.Tree, root string) error {
	if err := validateTreeModes(tree); err != nil {
		return err
	}
	expected := map[string]bool{}
	err := tree.Files().ForEach(func(f *object.File) error {
		if f.Mode != filemode.Regular && f.Mode != filemode.Executable {
			return fmt.Errorf("unsupported source mode at %s", f.Name)
		}
		if !relativePath(f.Name) {
			return fmt.Errorf("unsafe source path %q", f.Name)
		}
		expected[filepath.FromSlash(f.Name)] = true
		p := filepath.Join(root, filepath.FromSlash(f.Name))
		info, err := os.Lstat(p)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() || info.Mode().Perm()&0222 != 0 {
			return fmt.Errorf("source file %s is not read-only and regular", f.Name)
		}
		if (info.Mode().Perm()&0111 != 0) != (f.Mode == filemode.Executable) {
			return fmt.Errorf("source file mode mismatch: %s", f.Name)
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		if plumbing.ComputeHash(plumbing.BlobObject, data) != f.Hash {
			return fmt.Errorf("source file hash mismatch: %s", f.Name)
		}
		return nil
	})
	if err != nil {
		return err
	}
	return filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if info.Mode().Perm()&0222 != 0 {
			return fmt.Errorf("writable source cache path %s", p)
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		if !expected[rel] {
			return fmt.Errorf("unexpected source cache file %s", rel)
		}
		return nil
	})
}
func makeReadOnly(root string) error {
	return filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		return os.Chmod(p, info.Mode().Perm()&0555)
	})
}
func removeTemporarySource(root string) {
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err == nil && d.IsDir() {
			_ = os.Chmod(p, 0755)
		}
		return nil
	})
	_ = os.RemoveAll(root)
}

func validateTreeModes(tree *object.Tree) error {
	walker := object.NewTreeWalker(tree, true, nil)
	defer walker.Close()
	for {
		name, entry, err := walker.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if !relativePath(name) {
			return fmt.Errorf("unsafe source path %q", name)
		}
		if entry.Mode != filemode.Dir && entry.Mode != filemode.Regular && entry.Mode != filemode.Executable {
			return fmt.Errorf("source path %s has unsupported mode %s (symlinks and submodules are not supported)", name, entry.Mode)
		}
	}
}

func verifyGitObject(repo *git.Repository, hash plumbing.Hash) error {
	encoded, err := repo.Storer.EncodedObject(plumbing.AnyObject, hash)
	if err != nil {
		return err
	}
	reader, err := encoded.Reader()
	if err != nil {
		return err
	}
	defer reader.Close()
	hasher := plumbing.NewHasher(encoded.Type(), encoded.Size())
	if _, err := io.Copy(hasher, reader); err != nil {
		return err
	}
	if hasher.Sum() != hash {
		return fmt.Errorf("cached Git object hash mismatch: %s", hash)
	}
	return nil
}
func verifyGitTree(repo *git.Repository, tree *object.Tree) error {
	if err := verifyGitObject(repo, tree.Hash); err != nil {
		return err
	}
	walker := object.NewTreeWalker(tree, true, nil)
	defer walker.Close()
	for {
		_, entry, err := walker.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if entry.Mode == filemode.Dir {
			if err := verifyGitObject(repo, entry.Hash); err != nil {
				return err
			}
		}
	}
}
