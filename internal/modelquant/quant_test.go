// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package modelquant

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func TestResolveProfile(t *testing.T) {
	for requested, expected := range map[string]string{"2": "Q2_K", "4": "Q4_K_M", "8": "Q8_0"} {
		actual, err := ResolveProfile(requested)
		if err != nil || actual != expected {
			t.Fatalf("ResolveProfile(%q) = %q, %v", requested, actual, err)
		}
	}
	if _, err := ResolveProfile("7"); err == nil {
		t.Fatal("unsupported quantization was accepted")
	}
}

func TestRuntimeQuantizesWithCalibration(t *testing.T) {
	directory := t.TempDir()
	input := filepath.Join(directory, "input.gguf")
	output := filepath.Join(directory, "output.gguf")
	calibration := filepath.Join(directory, "calibration.txt")
	if err := os.WriteFile(input, []byte("high precision"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(calibration, []byte("sample"), 0o600); err != nil {
		t.Fatal(err)
	}
	quantizerPath := filepath.Join(directory, "llama-quantize")
	calibratorPath := filepath.Join(directory, "llama-imatrix")
	writeExecutable(t, calibratorPath, "#!/bin/sh\nout=\"\"\nwhile [ $# -gt 0 ]; do if [ \"$1\" = \"-o\" ]; then out=$2; shift 2; else shift; fi; done\nprintf matrix > \"$out\"\n")
	writeExecutable(t, quantizerPath, "#!/bin/sh\nif [ \"$1\" = \"--imatrix\" ]; then test -f \"$2\" || exit 9; shift 2; fi\ncp \"$1\" \"$2\"\n")
	calibratorTool := Tool{Name: "llama-imatrix", Version: "test", SHA256: "calibrator", path: calibratorPath}
	runtime := Runtime{Quantizer: Tool{Name: "llama-quantize", Version: "test", SHA256: "quantizer", path: quantizerPath}, Calibrator: &calibratorTool}
	result, err := runtime.Quantize(context.Background(), Request{Input: input, Output: output, Resolved: "Q4_K_M", CalibrationText: calibration})
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "high precision" || result.Quantizer.Name != "llama-quantize" || result.Calibrator == nil {
		t.Fatalf("output/result = %q %+v", data, result)
	}
	if _, err := os.Stat(filepath.Join(directory, ".waldo-imatrix.gguf")); !os.IsNotExist(err) {
		t.Fatal("temporary importance matrix was not removed")
	}
}

func writeExecutable(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o700); err != nil {
		t.Fatal(err)
	}
}

func TestResolveToolAcceptsUnversionedTool(t *testing.T) {
	directory := t.TempDir()
	silent := filepath.Join(directory, "llama-quantize")
	writeExecutable(t, silent, "#!/bin/sh\necho 'usage: llama-quantize [--help]' >&2\nexit 1\n")
	speaking := filepath.Join(directory, "llama-imatrix")
	writeExecutable(t, speaking, "#!/bin/sh\necho 'version: 9843 (4b72bb9d7)'\n")
	t.Setenv("PATH", directory)

	quantizer, err := resolveTool(context.Background(), "llama-quantize")
	if err != nil {
		t.Fatalf("a present, working quantizer was rejected: %v", err)
	}
	if quantizer.SHA256 == "" {
		t.Fatal("the tool must still be pinned by digest when it reports no version")
	}
	if quantizer.Version != "" {
		t.Fatalf("version = %q, want empty for a tool that does not report one", quantizer.Version)
	}

	calibrator, err := resolveTool(context.Background(), "llama-imatrix")
	if err != nil {
		t.Fatal(err)
	}
	if calibrator.Version != "version: 9843 (4b72bb9d7)" {
		t.Fatalf("a tool that reports a version must still have it recorded: %q", calibrator.Version)
	}
}

func TestResolveToolFailsWhenContextEnds(t *testing.T) {
	directory := t.TempDir()
	slow := filepath.Join(directory, "llama-quantize")
	writeExecutable(t, slow, "#!/bin/sh\n/bin/sleep 2\n")
	t.Setenv("PATH", directory)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := resolveTool(ctx, "llama-quantize"); err == nil {
		t.Fatal("a tool that outlived the context was accepted")
	} else if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error should identify the expired context: %v", err)
	}
}

func TestResolveToolRejectsUnrunnableBinary(t *testing.T) {
	directory := t.TempDir()
	broken := filepath.Join(directory, "llama-quantize")
	writeExecutable(t, broken, "\x00\x01\x02not an executable image")
	t.Setenv("PATH", directory)

	// The advice naming llama-quantize is static wrapper text. Assert on the
	// path, which reaches the message only when the probe reports the failure.
	if _, err := ResolveRuntime(context.Background(), false); err == nil {
		t.Fatal("a binary that cannot be executed was accepted at resolution")
	} else if !strings.Contains(err.Error(), broken) {
		t.Fatalf("error should name the file the operator must fix: %v", err)
	}
}

func TestResolveToolRecordsVersionLineNotDeviceNoise(t *testing.T) {
	directory := t.TempDir()
	noisy := filepath.Join(directory, "llama-quantize")
	// llama.cpp reports its version on standard error, after device diagnostics.
	writeExecutable(t, noisy, "#!/bin/sh\n{ echo 'ggml_cuda_init: found 1 ROCm devices'; echo '  Device 0: AMD Radeon Graphics'; echo 'version: 8298 (f90bd1dd8)'; echo 'built with GNU 11.5.0'; } >&2\n")
	t.Setenv("PATH", directory)

	tool, err := resolveTool(context.Background(), "llama-quantize")
	if err != nil {
		t.Fatal(err)
	}
	if tool.Version != "version: 8298 (f90bd1dd8)" {
		t.Fatalf("version = %q, want the reported version rather than a diagnostic", tool.Version)
	}
}

func TestResolveToolKeepsVersionWhenGrandchildHoldsPipe(t *testing.T) {
	directory := t.TempDir()
	lingering := filepath.Join(directory, "llama-imatrix")
	// The tool answers and exits; a background child it spawned keeps the
	// inherited pipe open, which is what makes CombinedOutput report
	// exec.ErrWaitDelay instead of success.
	writeExecutable(t, lingering, "#!/bin/sh\n/bin/sleep 30 &\necho 'version: 8298 (f90bd1dd8)'\n")
	t.Setenv("PATH", directory)

	tool, err := resolveTool(context.Background(), "llama-imatrix")
	if err != nil {
		t.Fatalf("a tool that answered and exited must resolve: %v", err)
	}
	if tool.Version != "version: 8298 (f90bd1dd8)" {
		t.Fatalf("version = %q; the answer was read before the pipe was abandoned", tool.Version)
	}
}

func TestProbeVersionFailsWhenCallerCancelsAsPipeLingers(t *testing.T) {
	directory := t.TempDir()
	lingering := filepath.Join(directory, "llama-quantize")
	// The tool exits successfully but a grandchild holds the pipe, so the probe
	// reports ErrWaitDelay. A caller that gave up inside that window must still
	// fail rather than have the cancellation read as a successful probe.
	writeExecutable(t, lingering, "#!/bin/sh\n/bin/sleep 10 &\necho 'version: 8298 (f90bd1dd8)'\n")

	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()
	if _, err := probeVersion(ctx, lingering, 10*time.Second, true); err == nil {
		t.Fatal("cancellation was swallowed by the lingering pipe")
	} else if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error should identify the expired context: %v", err)
	}
}

func TestProbeVersionAcceptsToolThatOutlivesProbe(t *testing.T) {
	directory := t.TempDir()
	slow := filepath.Join(directory, "llama-quantize")
	writeExecutable(t, slow, "#!/bin/sh\n/bin/sleep 60\n")

	// The budget is a parameter so this pins the behaviour without paying the
	// production timeout on every run.
	budget := 200 * time.Millisecond
	started := time.Now()
	// declines=false: a slow host is excused even for a tool that should answer.
	version, err := probeVersion(context.Background(), slow, budget, false)
	if err != nil {
		t.Fatalf("a tool too slow to answer must not fail the export: %v", err)
	}
	if version != "" {
		t.Fatalf("version = %q; an unanswered probe records none", version)
	}
	if elapsed := time.Since(started); elapsed > budget+toolProbeWaitDelay+2*time.Second {
		t.Fatalf("the probe took %s; it did not bound its own wait", elapsed)
	}
}

func TestResolveToolBoundsReportedVersion(t *testing.T) {
	directory := t.TempDir()
	chatty := filepath.Join(directory, "llama-quantize")
	// PATH is restricted to the temp directory below, so build the long line
	// with shell builtins rather than a command that will not be found.
	writeExecutable(t, chatty, "#!/bin/sh\ns=xxxxxxxxxxxxxxxx\nwhile [ ${#s} -lt 200000 ]; do s=\"$s$s\"; done\nprintf '%s' \"$s\"\n")
	t.Setenv("PATH", directory)

	tool, err := resolveTool(context.Background(), "llama-quantize")
	if err != nil {
		t.Fatal(err)
	}
	if len(tool.Version) > toolVersionMaximum {
		t.Fatalf("version is %d bytes; the release BOM must not carry an unbounded value", len(tool.Version))
	}
}

func TestToolVersionReportsOneLineOrNothing(t *testing.T) {
	// A field the release BOM carries as the tool's identity must never hold a
	// multi-line banner, and must not hold the first line of a diagnostic just
	// because the tool printed one.
	for name, expectation := range map[string]struct{ output, want string }{
		"named":       {"ggml_cuda_init: found 1 ROCm devices\nversion: 9843 (4b72bb9d7)\nbuilt with GNU\n", "version: 9843 (4b72bb9d7)"},
		"single":      {"llama-imatrix v1.2.3\n", "llama-imatrix v1.2.3"},
		"padded":      {"\n\n  quantize 0.9  \n\n", "quantize 0.9"},
		"banner":      {"llama-imatrix v1.2.3\nbuilt with clang 19\ntarget x86_64-linux\n", ""},
		"diagnostics": {"ggml_vulkan: Found 1 Vulkan devices:\nllama-quantize (llama.cpp) b9843\n", ""},
		"usage":       {"usage: tool [--help]\n  --version:  print version\n  --foo\n", ""},
		"blank":       {"   \n\n", ""},
	} {
		if got := toolVersion([]byte(expectation.output)); got != expectation.want {
			t.Errorf("%s: version = %q, want %q", name, got, expectation.want)
		}
	}
}

func TestBoundKeepsValueJSONSafe(t *testing.T) {
	// The value is written to the release BOM, so it has to read back as it was
	// written whether or not the tool emitted valid UTF-8.
	for name, value := range map[string]string{
		"invalid":   "version: \xff\xfe broken",
		"oversized": "version: " + strings.Repeat("æ—¥", 3000),
		"clean":     "version: 9843 (4b72bb9d7)",
	} {
		got := bound(value)
		if !utf8.ValidString(got) {
			t.Errorf("%s: bound produced invalid UTF-8", name)
		}
		encoded, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var decoded string
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if decoded != got {
			t.Errorf("%s: value did not survive a JSON round trip", name)
		}
		if len(got) > toolVersionMaximum {
			t.Errorf("%s: %d bytes exceeds the bound", name, len(got))
		}
	}
}

func TestProbeVersionRejectsFailuresThatAreNotAUsageRefusal(t *testing.T) {
	directory := t.TempDir()
	for name, script := range map[string]string{
		"crash":       "#!/bin/sh\nkill -SEGV $$\n",
		"exit127":     "#!/bin/sh\necho 'error while loading shared libraries: libfoo.so' >&2\nexit 127\n",
		"exit2":       "#!/bin/sh\nexit 2\n",
		"usageButNot": "#!/bin/sh\nexit 1\n",
	} {
		tool := filepath.Join(directory, name)
		writeExecutable(t, tool, script)
		// declines=false: a tool expected to answer --version must not have any
		// of these excused, including the usage status itself.
		if _, err := probeVersion(context.Background(), tool, 10*time.Second, false); err == nil {
			t.Errorf("%s: a tool expected to report a version was accepted after failing", name)
		}
	}
}

func TestProbeVersionExcusesOnlyTheUsageRefusal(t *testing.T) {
	directory := t.TempDir()
	usage := filepath.Join(directory, "llama-quantize")
	writeExecutable(t, usage, "#!/bin/sh\necho 'usage: llama-quantize [--help]' >&2\nexit 1\n")
	if version, err := probeVersion(context.Background(), usage, 10*time.Second, true); err != nil || version != "" {
		t.Fatalf("the known usage refusal must be excused: %q %v", version, err)
	}

	crashing := filepath.Join(directory, "crashing")
	writeExecutable(t, crashing, "#!/bin/sh\nkill -SEGV $$\n")
	// Even for the tool allowed to decline, a signal is not a refusal.
	if _, err := probeVersion(context.Background(), crashing, 10*time.Second, true); err == nil {
		t.Fatal("a crash was excused as a usage refusal")
	}
}
