// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package modelexport

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/model/trainerapp"
)

// ExportExternal publishes the verified hf-causal-lm artifacts without converting
// opaque weights. Private checkpoints and trainer executable code are excluded.
func ExportExternal(ctx context.Context, inspection model.ExternalInspection, destination string, options Options) (string, error) {
	if inspection.Model.State != model.RunComplete || len(inspection.Runs) == 0 {
		return "", fmt.Errorf("external export requires a complete model")
	}
	absolute, err := filepath.Abs(destination)
	if err != nil {
		return "", err
	}
	relative, err := filepath.Rel(inspection.Path, absolute)
	if err != nil {
		return "", err
	}
	if relative == "." || relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("export destination must be outside model")
	}
	if _, err = os.Lstat(absolute); !os.IsNotExist(err) {
		return "", fmt.Errorf("export destination already exists or cannot be inspected")
	}
	if err = os.MkdirAll(filepath.Dir(absolute), 0755); err != nil {
		return "", err
	}
	temp, err := os.MkdirTemp(filepath.Dir(absolute), ".external-export-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(temp)
	root, err := os.OpenRoot(inspection.Path)
	if err != nil {
		return "", err
	}
	defer root.Close()
	copyFile := func(source, target string) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		input, err := root.Open(source)
		if err != nil {
			return err
		}
		defer input.Close()
		info, err := input.Stat()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("export source is not regular")
		}
		path := filepath.Join(temp, target)
		if err = os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return err
		}
		output, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(output, input)
		closeErr := output.Close()
		if copyErr != nil {
			return copyErr
		}
		return closeErr
	}
	for _, file := range []string{"PLAN.json", "TRAINER-SCHEMAS.json"} {
		if err = copyFile(file, file); err != nil {
			return "", err
		}
	}
	refs := make([]model.ExternalBOMRun, len(inspection.BOM.Runs))
	for i, run := range inspection.BOM.Runs {
		if i >= len(inspection.Runs) || inspection.Runs[i].Complete == nil || inspection.Runs[i].Complete.ArtifactProfile != "hf-causal-lm" {
			return "", fmt.Errorf("external export requires hf-causal-lm artifacts")
		}
		refs[i] = run
		refs[i].Artifacts = nil
		if err = copyFile(run.BOM.Path, run.BOM.Path); err != nil {
			return "", err
		}
		for _, a := range run.Artifacts {
			target := a.Path
			// The last stage is directly loadable from the package root; earlier
			// stages retain their relative paths for provenance.
			if i == len(inspection.BOM.Runs)-1 {
				prefix := filepath.Join("runs", run.ID, "artifacts") + string(filepath.Separator)
				if !strings.HasPrefix(a.Path, prefix) {
					return "", fmt.Errorf("unexpected external artifact path")
				}
				target = strings.TrimPrefix(a.Path, prefix)
				for _, reserved := range []string{"PLAN.json", "TRAINER-SCHEMAS.json", "BOM.json", "BOM.sigstore.json", "EU-BOM.json", "EU-BOM.sigstore.json", "ATTRIBUTION.md", "runs"} {
					if target == reserved || strings.HasPrefix(target, reserved+string(filepath.Separator)) {
						return "", fmt.Errorf("artifact collides with reserved export path %s", target)
					}
				}
			}
			if err = copyFile(a.Path, target); err != nil {
				return "", err
			}
			a.Path = target
			refs[i].Artifacts = append(refs[i].Artifacts, a)
		}
		if err = trainerapp.VerifyPublished(temp, refs[i].Artifacts); err != nil {
			return "", err
		}
	}
	if _, err = model.WriteExternalModelBOM(temp, true, refs); err != nil {
		return "", err
	}
	for name, data := range map[string][]byte{"EU-BOM.json": options.EUBOM, "ATTRIBUTION.md": options.Attribution} {
		if len(data) > 0 {
			if err = os.WriteFile(filepath.Join(temp, name), data, 0644); err != nil {
				return "", err
			}
		}
	}
	if options.Finalize != nil {
		if err = options.Finalize(temp); err != nil {
			return "", err
		}
	}
	if err = os.Rename(temp, absolute); err != nil {
		return "", err
	}
	return absolute, nil
}
