// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

// Package pythonenv creates and verifies external application environments.
// It never imports or executes installed application packages.
package pythonenv

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/openwaldo/waldo/internal/training"
)

// Identity is portable environment evidence for the external BOM builder.
// Cache and installer observations deliberately live in RunEvidence instead.
type Identity struct {
	Name               string `json:"name"`
	Kind               string `json:"kind"`
	LockSHA256         string `json:"lock_sha256"`
	InterpreterVersion string `json:"interpreter_version"`
	InterpreterSHA256  string `json:"interpreter_sha256"`
	PlatformTag        string `json:"platform_tag"`
}

type RunEvidence struct {
	CachePath        string                        `json:"cache_path"`
	Installer        string                        `json:"installer"`
	InstallerVersion string                        `json:"installer_version"`
	Host             training.EnvironmentHostFacts `json:"host"`
}

type Result struct {
	Identity Identity    `json:"identity"`
	Run      RunEvidence `json:"run"`
}

// Request uses an explicitly selected trusted base Python (3.11+ with pip
// 22.3+). Lock is the resolved manifest lock path; PythonVersion is its exact
// interpreter pin. The helper and installer never run inside the application
// checkout. The caller must resolve/verify that checkout before this call.
type Request struct {
	Name          string `json:"name"`
	Lock          string `json:"lock"`
	PythonVersion string `json:"python_version"`
	Cache         string `json:"cache"`
	Python        string `json:"-"`
}

//go:embed environment.py
var helper string

// Ensure creates on a miss and independently verifies on every use. Entries
// are published atomically and never repaired or modified on cache hits.
func Ensure(ctx context.Context, request Request) (Result, error) {
	return invoke(ctx, request, "ensure")
}

// Verify checks an existing managed cache entry without installing anything.
func Verify(ctx context.Context, request Request) (Result, error) {
	return invoke(ctx, request, "verify")
}

func invoke(ctx context.Context, request Request, action string) (Result, error) {
	var result Result
	if request.Name == "" || request.Python == "" || request.PythonVersion == "" || request.Cache == "" || request.Lock == "" {
		return result, fmt.Errorf("python-lock requires name, interpreter, exact version, lock and cache")
	}
	var err error
	request.Lock, err = filepath.Abs(request.Lock)
	if err != nil {
		return result, err
	}
	request.Cache, err = filepath.Abs(request.Cache)
	if err != nil {
		return result, err
	}
	python, err := exec.LookPath(request.Python)
	if err != nil {
		return result, err
	}
	python, err = filepath.Abs(python)
	if err != nil {
		return result, err
	}
	dir, err := os.MkdirTemp("", "waldo-python-env-")
	if err != nil {
		return result, err
	}
	defer os.RemoveAll(dir)
	script := filepath.Join(dir, "environment.py")
	if err := os.WriteFile(script, []byte(helper), 0600); err != nil {
		return result, err
	}
	data, err := json.Marshal(request)
	if err != nil {
		return result, err
	}
	command := exec.CommandContext(ctx, python, "-I", "-B", script, action)
	command.Cancel = func() error { return command.Process.Signal(os.Interrupt) }
	command.WaitDelay = 5 * time.Second
	command.Dir = dir
	// No Python, pip, proxy or user configuration leaks into installation.
	command.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + dir, "LANG=C.UTF-8"}
	command.Stdin = bytes.NewReader(data)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		return result, fmt.Errorf("python-lock %s: %w: %s", action, err, stderr.String())
	}
	if err := json.Unmarshal(output, &result); err != nil {
		return result, fmt.Errorf("decode environment evidence: %w", err)
	}
	result.Run.Host = training.InspectEnvironmentHostFacts(ctx)
	return result, nil
}
