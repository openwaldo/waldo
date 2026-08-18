// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package modelquant adapts the upstream llama.cpp quantization tools without
// giving them ownership of WALDO model, index, or provenance state.
package modelquant

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

const Profile = "openwaldo-gguf-quant-schema-1"

const (
	toolProbeTimeout   = 10 * time.Second
	toolProbeWaitDelay = 2 * time.Second
	toolVersionMaximum = 4096
	// llama-quantize answers an unknown flag with usage text and this status.
	toolUsageExit = 1
)

var profiles = map[string]string{
	"2": "Q2_K",
	"3": "Q3_K_M",
	"4": "Q4_K_M",
	"5": "Q5_K_M",
	"6": "Q6_K",
	"8": "Q8_0",
}

type Tool struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
	path    string
}

type Result struct {
	Quantizer  Tool  `json:"quantizer"`
	Calibrator *Tool `json:"calibrator,omitempty"`
}

type Request struct {
	Input           string
	Output          string
	Resolved        string
	CalibrationText string
	Report          func(string)
}

type Quantizer interface {
	Quantize(context.Context, Request) (Result, error)
}

type Runtime struct {
	Quantizer  Tool
	Calibrator *Tool
}

func ResolveProfile(requested string) (string, error) {
	value := strings.TrimSpace(requested)
	if resolved, ok := profiles[value]; ok {
		return resolved, nil
	}
	return "", fmt.Errorf("unsupported quantization %q; use 2, 3, 4, 5, 6, or 8", requested)
}

func ResolveRuntime(ctx context.Context, calibration bool) (Runtime, error) {
	quantizer, err := resolveTool(ctx, "llama-quantize")
	if err != nil {
		return Runtime{}, fmt.Errorf("quantized export requires llama-quantize from llama.cpp; install llama.cpp and ensure llama-quantize is in PATH: %w", err)
	}
	runtime := Runtime{Quantizer: quantizer}
	if calibration {
		tool, err := resolveTool(ctx, "llama-imatrix")
		if err != nil {
			return Runtime{}, fmt.Errorf("calibrated export requires llama-imatrix from llama.cpp; install llama.cpp and ensure llama-imatrix is in PATH: %w", err)
		}
		runtime.Calibrator = &tool
	}
	return runtime, nil
}

func (runtime Runtime) Quantize(ctx context.Context, request Request) (Result, error) {
	if request.Input == "" || request.Output == "" || request.Resolved == "" {
		return Result{}, fmt.Errorf("quantization input, output, and resolved profile are required")
	}
	result := Result{Quantizer: publicTool(runtime.Quantizer)}
	arguments := []string{}
	var matrix string
	if request.CalibrationText != "" {
		if runtime.Calibrator == nil {
			return Result{}, fmt.Errorf("calibration was requested without llama-imatrix")
		}
		matrix = filepath.Join(filepath.Dir(request.Output), ".waldo-imatrix.gguf")
		if request.Report != nil {
			request.Report("measuring calibration importance")
		}
		if err := run(ctx, *runtime.Calibrator, []string{"-m", request.Input, "-f", request.CalibrationText, "-o", matrix}); err != nil {
			return Result{}, fmt.Errorf("generate importance matrix: %w", err)
		}
		defer os.Remove(matrix)
		arguments = append(arguments, "--imatrix", matrix)
		tool := publicTool(*runtime.Calibrator)
		result.Calibrator = &tool
	}
	if request.Report != nil {
		request.Report("quantizing weights as " + request.Resolved)
	}
	arguments = append(arguments, request.Input, request.Output, request.Resolved)
	if err := run(ctx, runtime.Quantizer, arguments); err != nil {
		_ = os.Remove(request.Output)
		return Result{}, fmt.Errorf("quantize GGUF as %s: %w", request.Resolved, err)
	}
	info, err := os.Stat(request.Output)
	if err != nil {
		return Result{}, fmt.Errorf("quantizer did not create output: %w", err)
	}
	if !info.Mode().IsRegular() || info.Size() == 0 {
		_ = os.Remove(request.Output)
		return Result{}, fmt.Errorf("quantizer created an empty or non-regular output")
	}
	return result, nil
}

func resolveTool(ctx context.Context, name string) (Tool, error) {
	path, err := exec.LookPath(name)
	if err != nil {
		return Tool{}, err
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return Tool{}, err
	}
	digest, err := hashFile(absolute)
	if err != nil {
		return Tool{}, err
	}
	version, err := probeVersion(ctx, absolute, toolProbeTimeout, declinesVersion(name))
	if err != nil {
		return Tool{}, err
	}
	return Tool{Name: name, Version: version, SHA256: digest, path: absolute}, nil
}

// declinesVersion reports the tools known not to implement --version.
func declinesVersion(name string) bool { return name == "llama-quantize" }

// probeVersion asks a tool to name itself within budget. The version is best
// effort; only a tool that failed in a way it should not is an error.
func probeVersion(ctx context.Context, absolute string, budget time.Duration, declines bool) (string, error) {
	probeCtx, cancelProbe := context.WithTimeout(ctx, budget)
	defer cancelProbe()
	command := exec.CommandContext(probeCtx, absolute, "--version")
	// A wrapper can leave a grandchild holding the pipe after the child is killed.
	command.WaitDelay = toolProbeWaitDelay
	output, probeErr := command.CombinedOutput()
	var exitErr *exec.ExitError
	switch {
	// First: a late cancel reports ErrWaitDelay, which would read as success.
	case ctx.Err() != nil:
		return "", fmt.Errorf("%s --version: %w", absolute, ctx.Err())
	case probeErr == nil:
		return toolVersion(output), nil
	case errors.Is(probeErr, exec.ErrWaitDelay):
		return toolVersion(output), nil
	// A slow host must not fail an export.
	case probeCtx.Err() != nil:
		return "", nil
	// ExitCode is -1 for a signalled process, so a crash never lands here.
	case declines && errors.As(probeErr, &exitErr) && exitErr.ExitCode() == toolUsageExit:
		return "", nil
	default:
		return "", fmt.Errorf("%s --version: %w: %s", absolute, probeErr, bound(strings.TrimSpace(string(output))))
	}
}

// toolVersion reduces a tool's output to the identity the release BOM keeps.
// Diagnostics can precede the version, so prefer the line that names one; a
// lone line is taken at its word and anything else reports nothing.
func toolVersion(output []byte) string {
	var seen []string
	for _, line := range strings.Split(string(output), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToLower(line), "version:") {
			return bound(line)
		}
		if line != "" {
			seen = append(seen, line)
		}
	}
	if len(seen) == 1 {
		return bound(seen[0])
	}
	return ""
}

// bound keeps the value to what a provenance field can carry: valid UTF-8 that
// reads back from the BOM as written, and short enough to stay a version.
func bound(value string) string {
	value = strings.ToValidUTF8(value, "")
	if len(value) <= toolVersionMaximum {
		return value
	}
	data := []byte(value)[:toolVersionMaximum]
	for len(data) > 0 && !utf8.Valid(data) {
		data = data[:len(data)-1]
	}
	return string(data)
}

func run(ctx context.Context, tool Tool, arguments []string) error {
	command := exec.CommandContext(ctx, tool.path, arguments...)
	output, err := command.CombinedOutput()
	if err != nil {
		message := strings.TrimSpace(string(output))
		if message != "" {
			return fmt.Errorf("%w: %s", err, message)
		}
		return err
	}
	return nil
}

func hashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func publicTool(tool Tool) Tool {
	tool.path = ""
	return tool
}
