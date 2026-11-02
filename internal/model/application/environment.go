// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package application

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/openwaldo/waldo/internal/model/pythonenv"
)

// EnsureEnvironment explicitly materializes a selected manifest environment.
// Use a freshly resolved/verified source; this does not launch application code.
func (source ResolvedSource) EnsureEnvironment(ctx context.Context, name, python, cache string) (pythonenv.Result, error) {
	environment, ok := source.Manifest.Environments[name]
	if !ok || environment.Kind != "python-lock" || !relativePath(environment.Lock) || environment.Lock == "." || source.Directory == "" {
		return pythonenv.Result{}, fmt.Errorf("invalid or missing python-lock environment %q", name)
	}
	if err := rejectLinks(source.Directory); err != nil {
		return pythonenv.Result{}, err
	}
	return pythonenv.Ensure(ctx, pythonenv.Request{Name: name, Python: python, PythonVersion: environment.Python, Cache: cache, Lock: filepath.Join(source.Directory, filepath.FromSlash(environment.Lock))})
}
