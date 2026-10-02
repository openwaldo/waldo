// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

func TestApproximateDurationUsesHoursUntilOneHundred(t *testing.T) {
	for _, test := range []struct {
		seconds int64
		want    string
	}{
		{seconds: 30 * 60, want: "under 1 hour"},
		{seconds: 60 * 60, want: "1 hour"},
		{seconds: 99 * 60 * 60, want: "99 hours"},
		{seconds: 100 * 60 * 60, want: "4 days"},
		{seconds: 12 * 24 * 60 * 60, want: "12 days"},
	} {
		if got := approximateDuration(test.seconds); got != test.want {
			t.Errorf("approximateDuration(%d) = %q, want %q", test.seconds, got, test.want)
		}
	}
}

func TestModelForecastAcceptsConfiguredMultipleIndexPaths(t *testing.T) {
	root := fixtureCLIIndex(t)
	configPath := filepath.Join(t.TempDir(), "config.json")
	t.Setenv("WALDO_CONFIG", configPath)
	t.Chdir(t.TempDir())
	if err := config.Save(config.Config{
		Index: root,
		Model: config.Model{Backend: "fake"},
		Lookaside: config.Lookaside{
			Cache: filepath.Join(t.TempDir(), "cache"), Scratch: filepath.Join(t.TempDir(), "scratch"),
		},
	}); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run([]string{"--json", "model", "forecast", "books", "books/books.json", "--compare-hosts"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code = %d, stderr = %q", code, stderr.String())
	}
	var output struct {
		Paths      []string `json:"paths"`
		Preset     string   `json:"preset"`
		Tokens     int64    `json:"tokens"`
		Parameters uint64   `json:"approximate_parameters"`
		Forecast   struct {
			Configurations []model.HardwareConfiguration `json:"configurations"`
		} `json:"forecast"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatal(err)
	}
	if output.Tokens != 2 || output.Preset != "10m" || output.Parameters == 0 || len(output.Paths) != 2 || len(output.Forecast.Configurations) == 0 {
		t.Fatalf("output = %+v", output)
	}
	stdout.Reset()
	stderr.Reset()
	code = Run([]string{"--json", "model", "forecast"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("whole-index forecast code = %d, stderr = %q", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("whole-index forecast stderr = %q", stderr.String())
	}
	if strings.Contains(stdout.String(), `"configurations"`) || !strings.Contains(stdout.String(), `"host"`) {
		t.Fatalf("default JSON forecast exposes comparison or omits host: %s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run([]string{"model", "forecast", filepath.Join(root, "books")}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("directory forecast code = %d, stderr = %q", code, stderr.String())
	}
	for _, want := range []string{"MODEL:", "10m", "PARAMETERS:", "TOKENS:", "BUDGET:", "one pass", "HOST:", "ACCELERATOR:", "READY:", "no", "REASON:"} {
		if !strings.Contains(stdout.String(), want) {
			t.Errorf("forecast missing %q:\n%s", want, stdout.String())
		}
	}
	if strings.Contains(stdout.String(), "HOST COMPARISON") || strings.Contains(stdout.String(), "MFR") {
		t.Fatalf("default forecast includes host comparison:\n%s", stdout.String())
	}
	stdout.Reset()
	stderr.Reset()
	code = Run([]string{"model", "forecast", filepath.Join(root, "books"), "--compare-hosts"}, &stdout, &stderr)
	if code != 0 || !strings.Contains(stdout.String(), "HOST COMPARISON") || !strings.Contains(stdout.String(), "MFR") {
		t.Fatalf("comparison forecast code = %d, stderr = %q:\n%s", code, stderr.String(), stdout.String())
	}
}

func TestModelForecastDistinguishesComposeInputErrorsFromIndexSelections(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing.yaml")
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"model", "forecast", missing}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "model compose") || !strings.Contains(stderr.String(), "does not exist") || strings.Contains(stderr.String(), "index path") {
		t.Fatalf("missing compose code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}

	empty := filepath.Join(t.TempDir(), "empty.yaml")
	if err := os.WriteFile(empty, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"model", "forecast", empty}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "model compose") || !strings.Contains(stderr.String(), "is empty") || strings.Contains(stderr.String(), "index path") {
		t.Fatalf("empty compose code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}

	root := fixtureCLIIndex(t)
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Index: root, Model: config.Model{Backend: "fake"}}); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"model", "forecast", "missing/corpus"}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "index path missing/corpus") || strings.Contains(stderr.String(), "model compose") {
		t.Fatalf("missing index code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
}

func TestWriteModelForecastUsesApprovedCompactColumns(t *testing.T) {
	report := model.ResourceForecast{ApproximateParameters: 9_543_210, PlannedTokens: 1_048_576_000, Configurations: []model.HardwareConfiguration{
		{Manufacturer: "Apple", Accelerator: "M4 Max 40-core GPU", GPUs: 1, Nodes: 1, MemoryPerGPUBytes: 128 << 30, ApproximateSeconds: 48 * 24 * 60 * 60},
		{Manufacturer: "NVIDIA", Accelerator: "H100 SXM", GPUs: 8, Nodes: 1, MemoryPerGPUBytes: 80 << 30, ApproximateSeconds: 44 * 60 * 60},
	}}
	var output bytes.Buffer
	writeHostComparison(&output, report)
	lines := strings.Split(strings.TrimSpace(output.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("output = %q", output.String())
	}
	for lineNumber, want := range []string{"GPUS", "1", "8"} {
		fields := strings.Fields(lines[lineNumber])
		if len(fields) == 0 || fields[0] != want {
			t.Errorf("line %d does not lead with %q:\n%s", lineNumber+1, want, lines[lineNumber])
		}
	}
	for _, want := range []string{"MFR", "ACCELERATOR", "GPUS", "NODES", "MEMORY/GPU", "APPROX. TIME", "Apple", "128 GB", "48 days", "NVIDIA", "80 GB", "44 hours"} {
		if !strings.Contains(output.String(), want) {
			t.Errorf("output missing %q:\n%s", want, output.String())
		}
	}
	for _, unwanted := range []string{"BACKEND", "READY", "FIT", "~", "unified"} {
		if strings.Contains(output.String(), unwanted) {
			t.Errorf("output unexpectedly contains %q:\n%s", unwanted, output.String())
		}
	}
}

func TestWriteModelForecastIdentifiesEpochDerivedWork(t *testing.T) {
	report := model.ResourceForecast{ApproximateParameters: 10, PlannedTokens: 1000, EpochDerivedStages: []string{"midtrain", "post-train"}}
	var output bytes.Buffer
	writeModelForecast(&output, report, model.HostForecast{Ready: true, Execution: training.Execution{Host: training.Host{OS: "linux", Architecture: "amd64"}}}, false, nil)
	for _, want := range []string{"at least 1.0K plus 2 epoch-derived stage(s)", "midtrain, post-train resolve during training preflight"} {
		if !strings.Contains(output.String(), want) {
			t.Fatalf("forecast output missing %q: %q", want, output.String())
		}
	}
}

func TestWriteModelForecastReportsEmbeddingDominatedArchitecture(t *testing.T) {
	report := model.ResourceForecast{ApproximateParameters: 16_014_336, PlannedTokens: 5_000_000}
	architecture := model.Architecture{VocabularySize: 50_259, HiddenSize: 256, TieEmbeddings: true}
	var output bytes.Buffer
	writeModelForecast(&output, report, model.HostForecast{Ready: true}, false, &architecture)
	for _, want := range []string{"TOKEN I/O:", "12.9M", "80.3% of parameters", "exceed 50% of model capacity"} {
		if !strings.Contains(output.String(), want) {
			t.Errorf("forecast missing %q:\n%s", want, output.String())
		}
	}
}

func TestWriteModelForecastRecommendsRemoteComputeWithoutCatalogFit(t *testing.T) {
	report := model.ResourceForecast{
		ApproximateParameters: 1_000_000_000_000,
		PlannedTokens:         1,
		CatalogNote:           "no configuration in the forecast catalog has sufficient memory for this workload",
	}
	host := model.HostForecast{
		Reason:         "training requires 1.0 TiB per device, but this host has 128.0 GiB",
		Recommendation: "use remote compute with at least 1.0 TiB of usable memory per device",
		Execution:      training.Execution{Host: training.Host{OS: "linux", Architecture: "amd64"}},
	}
	var output bytes.Buffer
	writeModelForecast(&output, report, host, true, nil)
	for _, want := range []string{"READY:       no", "REASON:", "RECOMMEND:   use remote compute", "HOST COMPARISON", "NOTE:        no configuration"} {
		if !strings.Contains(output.String(), want) {
			t.Errorf("forecast output missing %q:\n%s", want, output.String())
		}
	}
}
