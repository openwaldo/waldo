// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/openwaldo/waldo/internal/training"
)

type Listing struct {
	Name       string `json:"name"`
	ID         string `json:"id"`
	Path       string `json:"path"`
	Parameters uint64 `json:"approximate_parameters"`
	Runs       int    `json:"runs"`
	State      string `json:"state,omitempty"`
	Updated    string `json:"updated"`
}

func List(root string, patterns []string) ([]Listing, error) {
	entries, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	for _, pattern := range patterns {
		if _, err := filepath.Match(pattern, ""); err != nil {
			return nil, fmt.Errorf("invalid model pattern %q: %w", pattern, err)
		}
	}
	result := make([]Listing, 0)
	for _, entry := range entries {
		if !entry.IsDir() || !validName.MatchString(entry.Name()) || !matchesAny(entry.Name(), patterns) {
			continue
		}
		inspection, err := Inspect(root, entry.Name())
		if err != nil {
			return nil, err
		}
		listing := listingFromInspection(inspection)
		result = append(result, listing)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result, nil
}

func listingFromInspection(inspection Inspection) Listing {
	listing := Listing{
		Name: inspection.Model.Name, ID: inspection.Model.ID, Path: inspection.Path,
		Parameters: inspection.Model.Forecast.ApproximateParameters,
		Runs:       len(inspection.Model.Runs), Updated: inspection.Model.Updated,
	}
	if inspection.Origin != nil {
		listing.State = "downloaded"
	}
	if len(inspection.Model.Runs) > 0 {
		listing.State = string(inspection.Model.Runs[len(inspection.Model.Runs)-1].State)
	}
	return listing
}

func matchesAny(name string, patterns []string) bool {
	if len(patterns) == 0 {
		return true
	}
	for _, pattern := range patterns {
		if matched, _ := filepath.Match(pattern, name); matched {
			return true
		}
	}
	return false
}

func Remove(root string, names []string) ([]string, error) {
	if len(names) == 0 {
		return nil, fmt.Errorf("at least one model name is required")
	}
	paths := make([]string, len(names))
	seen := map[string]bool{}
	for index, name := range names {
		if err := ValidateName(name); err != nil {
			return nil, err
		}
		if seen[name] {
			return nil, fmt.Errorf("model %q was named more than once", name)
		}
		seen[name] = true
		inspection, err := Inspect(root, name)
		if err != nil {
			return nil, err
		}
		paths[index] = inspection.Path
	}
	for _, path := range paths {
		if err := os.RemoveAll(path); err != nil {
			return nil, err
		}
	}
	return append([]string(nil), names...), nil
}

func Exists(root, name string) (bool, error) {
	if err := ValidateName(name); err != nil {
		return false, err
	}
	info, err := os.Stat(filepath.Join(root, name))
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if !info.IsDir() {
		return false, fmt.Errorf("model path for %q is not a directory", name)
	}
	if _, err := Inspect(root, name); err != nil {
		return false, err
	}
	return true, nil
}

type ExportOptions struct {
	Files    map[string][]byte
	Finalize func(string) error
}

func Export(root, name, destination string) (string, error) {
	return ExportPackage(root, name, destination, ExportOptions{})
}

func ExportPackage(root, name, destination string, options ExportOptions) (string, error) {
	inspection, err := Inspect(root, name)
	if err != nil {
		return "", err
	}
	absolute, err := filepath.Abs(destination)
	if err != nil {
		return "", err
	}
	inside, err := filepath.Rel(inspection.Path, absolute)
	if err != nil {
		return "", err
	}
	if inside == "." || inside != ".." && !strings.HasPrefix(inside, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("model export destination must not be inside the source model")
	}
	if _, err := os.Stat(absolute); err == nil {
		return "", fmt.Errorf("%s already exists", absolute)
	} else if !os.IsNotExist(err) {
		return "", err
	}
	parent := filepath.Dir(absolute)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return "", err
	}
	temporary, err := os.MkdirTemp(parent, ".waldo-model-export-*")
	if err != nil {
		return "", err
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(temporary)
		}
	}()
	if err := copyTree(inspection.Path, temporary); err != nil {
		return "", err
	}
	if err := os.Remove(filepath.Join(temporary, "MODEL-BOM.json")); err != nil && !os.IsNotExist(err) {
		return "", err
	}
	// Export always carries the normalized schema-1 BOM, including when the
	// managed model was created before model-root-relative paths were added.
	if err := writeJSONAtomic(filepath.Join(temporary, "BOM.json"), inspection.BOM); err != nil {
		return "", err
	}
	extraNames := make([]string, 0, len(options.Files))
	for name := range options.Files {
		extraNames = append(extraNames, name)
	}
	sort.Strings(extraNames)
	for _, name := range extraNames {
		data := options.Files[name]
		if filepath.Base(name) != name || name == "." || name == "" || name == "BOM.json" || name == "MODEL-BOM.json" {
			return "", fmt.Errorf("invalid additional model export file %q", name)
		}
		if err := os.WriteFile(filepath.Join(temporary, name), data, 0o644); err != nil {
			return "", err
		}
	}
	exported, err := Inspect("", temporary)
	if err != nil {
		return "", fmt.Errorf("verify exported model metadata: %w", err)
	}
	if err := verifyModelArtifacts(exported); err != nil {
		return "", fmt.Errorf("verify exported model artifacts: %w", err)
	}
	if options.Finalize != nil {
		if err := options.Finalize(temporary); err != nil {
			return "", err
		}
	}
	if err := os.Rename(temporary, absolute); err != nil {
		return "", err
	}
	committed = true
	return absolute, nil
}

func verifyModelArtifacts(inspection Inspection) error {
	for index, run := range inspection.Runs {
		if run.Observation == nil {
			continue
		}
		pin := inspection.Model.Runs[index]
		runDirectory := filepath.Join(inspection.Path, "runs", runDirectoryName(pin))
		artifacts := append([]training.Artifact(nil), run.Observation.Artifacts...)
		for _, checkpoint := range run.Observation.Checkpoints {
			artifacts = append(artifacts, checkpoint.Artifacts...)
		}
		for _, artifact := range artifacts {
			if err := VerifyArtifactFile(filepath.Join(runDirectory, filepath.FromSlash(artifact.Path)), artifact); err != nil {
				return fmt.Errorf("run %s: %w", run.ID, err)
			}
		}
	}
	return nil
}

// VerifyCurrentModelArtifacts verifies the published artifacts from the most
// recent completed run. Checkpoints are intentionally excluded: completion is
// about the model artifact that WALDO will load, not obsolete recovery state.
func VerifyCurrentModelArtifacts(inspection Inspection) error {
	for index := len(inspection.Runs) - 1; index >= 0; index-- {
		run := inspection.Runs[index]
		if run.State != RunComplete {
			continue
		}
		if run.Observation == nil || len(run.Observation.Artifacts) == 0 || index >= len(inspection.Model.Runs) {
			return fmt.Errorf("run %s has no published model artifacts", run.ID)
		}
		runDirectory := filepath.Join(inspection.Path, "runs", runDirectoryName(inspection.Model.Runs[index]))
		for _, artifact := range run.Observation.Artifacts {
			if err := VerifyArtifactFile(filepath.Join(runDirectory, filepath.FromSlash(artifact.Path)), artifact); err != nil {
				return fmt.Errorf("run %s: %w", run.ID, err)
			}
		}
		return nil
	}
	return fmt.Errorf("model has no completed run artifacts")
}

func copyTree(source, destination string) error {
	return filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		target := filepath.Join(destination, relative)
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("model export refuses symbolic link %s", path)
		}
		if entry.IsDir() {
			return os.MkdirAll(target, info.Mode().Perm())
		}
		sourceFile, err := os.Open(path)
		if err != nil {
			return err
		}
		targetFile, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, info.Mode().Perm())
		if err != nil {
			_ = sourceFile.Close()
			return err
		}
		_, copyErr := io.Copy(targetFile, sourceFile)
		sourceCloseErr := sourceFile.Close()
		closeErr := targetFile.Close()
		if copyErr != nil {
			return copyErr
		}
		if sourceCloseErr != nil {
			return sourceCloseErr
		}
		return closeErr
	})
}
