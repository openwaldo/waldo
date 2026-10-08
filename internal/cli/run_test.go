// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/shard"
)

type cliPublisher struct{ objects map[string]int64 }

type cliCredentialStore struct {
	credentials lookaside.Credentials
	found       bool
	err         error
}

func (store *cliCredentialStore) Get(string) (lookaside.Credentials, bool, error) {
	return store.credentials, store.found, store.err
}

func (store *cliCredentialStore) Set(_ string, credentials lookaside.Credentials) error {
	store.credentials, store.found = credentials, true
	return store.err
}

func (store *cliCredentialStore) Delete(string) error {
	store.credentials, store.found = lookaside.Credentials{}, false
	return store.err
}

func useCLICredentialStore(t *testing.T, store lookaside.CredentialStore) {
	t.Helper()
	previous := lookasideCredentialStore
	lookasideCredentialStore = store
	t.Cleanup(func() { lookasideCredentialStore = previous })
}

func (publisher *cliPublisher) BaseURL() string { return "s3://openwaldo/lookaside/v1" }
func (publisher *cliPublisher) Publish(_ context.Context, source, digest string, size int64, progress func(lookaside.PublishProgress)) (lookaside.PublishedObject, error) {
	if err := lookaside.VerifyFile(source, digest, size); err != nil {
		return lookaside.PublishedObject{}, err
	}
	if publisher.objects == nil {
		publisher.objects = map[string]int64{}
	}
	publisher.objects[digest] = size
	if progress != nil {
		progress(lookaside.PublishProgress{Written: size, Total: size})
	}
	return lookaside.PublishedObject{URL: publisher.BaseURL() + "/" + digest[:2] + "/" + digest[2:4] + "/" + digest, SHA256: digest, Bytes: size}, nil
}
func (publisher *cliPublisher) Verify(_ context.Context, digest string, size int64) (lookaside.PublishedObject, error) {
	if publisher.objects[digest] != size {
		return lookaside.PublishedObject{}, fmt.Errorf("missing object")
	}
	return lookaside.PublishedObject{SHA256: digest, Bytes: size}, nil
}

func TestIndexAddDryRunProducesImmutablePlan(t *testing.T) {
	input := filepath.Join(t.TempDir(), "document.md")
	if err := os.WriteFile(input, []byte("# Example\n\nTraining text.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	root := emptyCLIIndex(t)
	var stdout, stderr bytes.Buffer
	code := Run([]string{
		"index", "ingest", input, filepath.Join(root, "core", "example"),
		"--title", "Example", "--license", "CC0-1.0",
		"--source", "https://example.test/data", "--source-category", "public-dataset", "--language", "en",
		"--dry-run", "--json",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	var output struct {
		Identity string `json:"identity"`
		Plan     struct {
			Kind        string `json:"kind"`
			Destination string `json:"destination"`
			Writer      struct {
				RecordSchema int `json:"record_schema"`
			} `json:"writer"`
			Inputs []struct {
				Adapter string `json:"adapter"`
			} `json:"inputs"`
		} `json:"plan"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatal(err)
	}
	if len(output.Identity) != 64 || output.Plan.Kind != "waldo-ingest-plan" || output.Plan.Destination != "core/example" || output.Plan.Writer.RecordSchema != shard.TextRecordSchema || len(output.Plan.Inputs) != 1 || output.Plan.Inputs[0].Adapter != "markdown" {
		t.Fatalf("index ingest output = %+v", output)
	}
}

func TestIndexIngestSourceDirectoryRecursesRawAndUsesManifest(t *testing.T) {
	handoff := t.TempDir()
	raw := filepath.Join(handoff, "raw", "nested")
	if err := os.MkdirAll(raw, 0o755); err != nil {
		t.Fatal(err)
	}
	content := []byte("Training text.\n")
	if err := os.WriteFile(filepath.Join(raw, "document.txt"), content, 0o644); err != nil {
		t.Fatal(err)
	}
	fileHash := sha256.Sum256(content)
	inventory := fmt.Sprintf("%x\t%d\tnested/document.txt\n", fileHash, len(content))
	treeHash := sha256.Sum256([]byte(inventory))
	manifest := fmt.Sprintf(`{
  "kind":"waldo-source-directory","schema":1,
  "corpus":{"id":"source-example","title":"Source Example","description":"Fetcher handoff."},
  "sources":[{"id":"source-example","path":"","license":"CC0-1.0","source":{"name":"Example","url":"https://example.test/data","category":"public-dataset","license_evidence":{"declaration":"CC0-1.0"}},"input":{"format":"text"},"artifacts":[{"url":"https://example.test/data/document.txt","path":"nested/document.txt"}]}],
  "fetcher":{},"raw":{"path":"raw","file_count":1,"byte_count":%d,"tree_sha256":"%x"}
}`, len(content), treeHash)
	if err := os.WriteFile(filepath.Join(handoff, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	indexRoot := emptyCLIIndex(t)
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Index: indexRoot}); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run([]string{"--json", "index", "ingest", handoff, "core/source-example", "--dry-run"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	var output struct {
		Plan struct {
			Destination string `json:"destination"`
			Inputs      []struct {
				Artifact struct {
					Path string `json:"path"`
				} `json:"artifact"`
			} `json:"inputs"`
		} `json:"plan"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatal(err)
	}
	if output.Plan.Destination != "core/source-example" || len(output.Plan.Inputs) != 1 ||
		output.Plan.Inputs[0].Artifact.Path != filepath.Join(raw, "document.txt") {
		t.Fatalf("source-directory plan = %+v", output.Plan)
	}
}

func TestIndexIngestResolvesRelativeDestinationAgainstConfiguredIndex(t *testing.T) {
	input := filepath.Join(t.TempDir(), "document.txt")
	if err := os.WriteFile(input, []byte("Training text.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	root := emptyCLIIndex(t)
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Index: root}); err != nil {
		t.Fatal(err)
	}
	t.Chdir(t.TempDir())
	var stdout, stderr bytes.Buffer
	code := Run([]string{
		"--json", "index", "ingest", input, "core/example",
		"--title", "Example", "--license", "CC0-1.0",
		"--source", "https://example.test/data", "--source-category", "public-dataset", "--language", "en", "--dry-run",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code = %d, stderr = %q", code, stderr.String())
	}
	var output struct {
		Plan struct {
			Destination string `json:"destination"`
		} `json:"plan"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatal(err)
	}
	if output.Plan.Destination != "core/example" {
		t.Fatalf("destination = %q", output.Plan.Destination)
	}
}

func TestIndexIngestExplicitRelativeDestinationOverridesMissingConfiguredIndex(t *testing.T) {
	input := filepath.Join(t.TempDir(), "document.txt")
	if err := os.WriteFile(input, []byte("Training text.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	root := fixtureGitIndex(t)
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Index: filepath.Join(t.TempDir(), "missing-index")}); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)
	var stdout, stderr bytes.Buffer
	code := Run([]string{
		"--json", "index", "ingest", input, "./post-train/sft/example",
		"--title", "Example", "--license", "CC0-1.0",
		"--source", "https://example.test/data", "--source-category", "public-dataset", "--language", "en", "--dry-run",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code = %d, stderr = %q", code, stderr.String())
	}
	var output struct {
		Plan struct {
			Destination string `json:"destination"`
		} `json:"plan"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatal(err)
	}
	if output.Plan.Destination != "post-train/sft/example" {
		t.Fatalf("destination = %q", output.Plan.Destination)
	}
}

func TestIndexInitCreatesInspectableEmptyIndex(t *testing.T) {
	root := filepath.Join(t.TempDir(), "index")
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"index", "init", root}, &stdout, &stderr); code != 0 {
		t.Fatalf("index init code = %d, stderr = %q", code, stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"index", "verify", root, "--offline"}, &stdout, &stderr); code != 0 {
		t.Fatalf("index verify code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "1 directories, 0 corpora, 0 shards") {
		t.Fatalf("index verify output = %q", stdout.String())
	}
}

func TestIndexIngestRejectsFormerToOption(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run([]string{
		"index", "ingest", "input", "destination", "--to", "other",
		"--title", "Example", "--license", "CC0-1.0",
		"--source", "https://example.test/data", "--source-category", "public-dataset", "--language", "en", "--dry-run",
	}, &stdout, &stderr)
	if code != 1 || !strings.Contains(stderr.String(), "unknown flag: --to") {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
}

func TestIndexIngestRejectsRemovedModeFlags(t *testing.T) {
	for _, removed := range []string{"--local-only", "--object-base", "--mode", "--memory", "--staging"} {
		var stdout, stderr bytes.Buffer
		args := []string{
			"index", "ingest", "input", "destination",
			"--title", "Example", "--license", "CC0-1.0",
			"--source", "https://example.test/data", "--source-category", "public-dataset", "--language", "en",
			"--dry-run", removed,
		}
		if removed != "--local-only" {
			args = append(args, "value")
		}
		code := Run(args, &stdout, &stderr)
		if code != 1 || !strings.Contains(stderr.String(), "unknown flag: "+removed) {
			t.Fatalf("%s: code = %d, stderr = %q", removed, code, stderr.String())
		}
	}
}

func TestIndexIngestExecutionRequiresWritableLookaside(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "missing.json"))
	root := emptyCLIIndex(t)
	var stdout, stderr bytes.Buffer
	code := Run([]string{
		"index", "ingest", "/does/not/need/to/exist", filepath.Join(root, "core", "example"),
		"--title", "Example", "--license", "CC0-1.0",
		"--source", "https://example.test/data", "--source-category", "public-dataset", "--language", "en",
	}, &stdout, &stderr)
	if code != 1 || !strings.Contains(stderr.String(), "needs a writable lookaside") {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
}

func TestIndexIngestPublishesAndBuildsContributionOverlay(t *testing.T) {
	input := filepath.Join(t.TempDir(), "document.txt")
	if err := os.WriteFile(input, []byte("training document"), 0o644); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "index.json"), []byte("{\n  \"kind\": \"index\",\n  \"schema\": 1,\n  \"path\": \"\",\n  \"entries\": [{\"name\": \"core\", \"type\": \"dir\"}]\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "core"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "core", "index.json"), []byte("{\n  \"kind\": \"index\",\n  \"schema\": 1,\n  \"path\": \"core\",\n  \"entries\": []\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	staging := t.TempDir()
	cache := t.TempDir()
	configurationPath := filepath.Join(t.TempDir(), "config.json")
	t.Setenv("WALDO_CONFIG", configurationPath)
	if err := config.Save(config.Config{
		Lookaside: config.Lookaside{Scratch: cache, Publish: &config.Publish{URL: "s3://openwaldo/lookaside/v1", Workers: 4}},
		Ingest:    config.Ingest{Staging: staging},
	}); err != nil {
		t.Fatal(err)
	}
	originalPublisher := newIngestPublisher
	remote := &cliPublisher{}
	newIngestPublisher = func(context.Context, config.Publish) (lookaside.Publisher, error) { return remote, nil }
	t.Cleanup(func() { newIngestPublisher = originalPublisher })
	var stdout, stderr bytes.Buffer
	code := Run([]string{
		"--json", "index", "ingest", input, filepath.Join(root, "core", "example"),
		"--title", "Example", "--description", "Example corpus.",
		"--license", "CC0-1.0", "--source", "https://example.test/data",
		"--source-category", "public-dataset", "--language", "en",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	var output struct {
		Assembly struct {
			RetainedDocs int64 `json:"retained_docs"`
		} `json:"assembly"`
		Publication struct {
			Objects []struct {
				SHA256 string `json:"sha256"`
			} `json:"objects"`
		} `json:"publication"`
		Contribution struct {
			Root  string   `json:"root"`
			Files []string `json:"files"`
		} `json:"contribution"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatal(err)
	}
	if output.Assembly.RetainedDocs != 1 || len(output.Publication.Objects) != 1 || len(output.Contribution.Files) != 3 {
		t.Fatalf("output = %+v", output)
	}
	if remote.objects[output.Publication.Objects[0].SHA256] == 0 {
		t.Fatal("published object is absent from fake lookaside")
	}
	if _, err := os.Stat(filepath.Join(output.Contribution.Root, "core", "example", "example.yaml")); err != nil {
		t.Fatal(err)
	}
}

func TestIndexIngestPublishesToConfiguredLocalLookaside(t *testing.T) {
	input := filepath.Join(t.TempDir(), "document.txt")
	if err := os.WriteFile(input, []byte("local publication document"), 0o644); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "index.json"), []byte("{\n  \"kind\": \"index\",\n  \"schema\": 1,\n  \"path\": \"\",\n  \"entries\": [{\"name\": \"core\", \"type\": \"dir\"}]\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "core"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "core", "index.json"), []byte("{\n  \"kind\": \"index\",\n  \"schema\": 1,\n  \"path\": \"core\",\n  \"entries\": []\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	configurationPath := filepath.Join(t.TempDir(), "config.json")
	publishedRoot, staging := t.TempDir(), t.TempDir()
	t.Setenv("WALDO_CONFIG", configurationPath)
	lookasideURL := (&url.URL{Scheme: "file", Path: filepath.ToSlash(publishedRoot)}).String()
	var stdout, stderr bytes.Buffer
	for _, command := range [][]string{
		{"config", "set", "lookaside", lookasideURL},
		{"config", "set", "lookaside.workers", "2"},
		{"config", "set", "lookaside.scratch", t.TempDir()},
		{"config", "set", "ingest.staging", staging},
	} {
		stdout.Reset()
		stderr.Reset()
		if code := Run(command, &stdout, &stderr); code != 0 {
			t.Fatalf("%v: code = %d, stderr = %q", command, code, stderr.String())
		}
	}
	stdout.Reset()
	stderr.Reset()
	code := Run([]string{
		"--json", "index", "ingest", input, filepath.Join(root, "core", "local-published"),
		"--title", "Locally Published", "--license", "CC0-1.0",
		"--source", "https://example.test/local", "--source-category", "public-dataset", "--language", "en",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	var output struct {
		Publication struct {
			Objects []struct {
				SHA256 string `json:"sha256"`
				URL    string `json:"url"`
			} `json:"objects"`
		} `json:"publication"`
		Contribution struct {
			Root string `json:"root"`
		} `json:"contribution"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatal(err)
	}
	if len(output.Publication.Objects) != 1 {
		t.Fatalf("publication = %+v", output.Publication)
	}
	object := output.Publication.Objects[0]
	wantPath := filepath.Join(publishedRoot, object.SHA256[:2], object.SHA256[2:4], object.SHA256)
	if _, err := os.Stat(wantPath); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(object.URL, "file://") {
		t.Fatalf("object URL = %q", object.URL)
	}
	manifestPath := filepath.Join(output.Contribution.Root, "core", "local-published", "local-published.yaml")
	manifest, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(manifest, []byte(object.URL)) {
		t.Fatalf("manifest does not reference local published object %q", object.URL)
	}
}

func emptyCLIIndex(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeCLIFile(t, filepath.Join(root, "index.json"), `{
  "kind": "index", "schema": 1, "path": "", "entries": []
}`)
	return root
}

func TestShellQuoteHandlesSingleQuote(t *testing.T) {
	if got, want := shellQuote("a'b"), "'a'\\''b'"; got != want {
		t.Fatalf("shellQuote() = %q, want %q", got, want)
	}
}

func TestRootHelpLocksCommandVocabulary(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"--help"}, &stdout, &stderr); code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	help := stdout.String()
	for _, want := range []string{"status", "index", "lookaside", "model", "config"} {
		if !strings.Contains(help, want) {
			t.Errorf("root help does not contain %q:\n%s", want, help)
		}
	}
	for _, unwanted := range []string{"store", "corpus", "\n  bom"} {
		if strings.Contains(help, unwanted) {
			t.Errorf("root help unexpectedly contains %q:\n%s", unwanted, help)
		}
	}
}

func TestRootRejectsRemovedBOMCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"bom", "show", "anything"}, &stdout, &stderr); code == 0 {
		t.Fatalf("removed bom command unexpectedly succeeded: stdout = %q, stderr = %q", stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "unknown command") {
		t.Fatalf("removed bom command error = %q", stderr.String())
	}
}

func TestModelRejectsRemovedDiscloseCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"model", "disclose", "anything"}, &stdout, &stderr); code == 0 {
		t.Fatalf("removed model disclose command unexpectedly succeeded: stdout = %q, stderr = %q", stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "unknown command") {
		t.Fatalf("removed model disclose command error = %q", stderr.String())
	}
}

func TestIndexOwnsCorpusWorkflows(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"index", "--help"}, &stdout, &stderr); code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	for _, want := range []string{"pull", "list", "show", "summary", "bom", "verify", "ingest", "export"} {
		if !strings.Contains(stdout.String(), want) {
			t.Errorf("index help does not contain %q:\n%s", want, stdout.String())
		}
	}
	for _, unwanted := range []string{"status", "fetch", "clone", "remove", "update"} {
		if strings.Contains(stdout.String(), unwanted) {
			t.Errorf("index help unexpectedly contains %q:\n%s", unwanted, stdout.String())
		}
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"index", "ingest", "--help"}, &stdout, &stderr); code != 0 || !strings.Contains(stdout.String(), "--update") {
		t.Fatalf("ingest help code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
}

func TestIndexSummaryTruncatesLongLicenseOnlyInHumanOutput(t *testing.T) {
	root := fixtureCLIIndex(t)
	manifestPath := filepath.Join(root, "books", "books.json")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	longLicense := "LicenseRef-Open-Parliament-Licence---https-www.parliament.uk-site-information-copyright-parliament-open-parliament-licence"
	data = bytes.Replace(data, []byte("CC0-1.0"), []byte(longLicense), 1)
	if err := os.WriteFile(manifestPath, data, 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if code := Run([]string{"index", "summary", root}, &stdout, &stderr); code != 0 {
		t.Fatalf("human summary code = %d, stderr = %q", code, stderr.String())
	}
	if strings.Contains(stdout.String(), longLicense) || !strings.Contains(stdout.String(), "LicenseRef-Open-Parliament-Licence---https-www.…") {
		t.Fatalf("human summary did not truncate license:\n%s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"--json", "index", "summary", root}, &stdout, &stderr); code != 0 {
		t.Fatalf("JSON summary code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), longLicense) {
		t.Fatalf("JSON summary lost full license: %s", stdout.String())
	}
}

func TestModelOwnsLifecycleCommands(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"model", "--help"}, &stdout, &stderr); code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	for _, want := range []string{"init", "pull", "list", "summary", "bom", "forecast", "train", "compose", "export", "chat", "rm"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("model help does not contain %s:\n%s", want, stdout.String())
		}
	}
	if strings.Contains(stdout.String(), "download") {
		t.Fatalf("model help retains removed download command:\n%s", stdout.String())
	}
}

func TestLeafHelpDoesNotExecuteCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"index", "summary", "--help"}, &stdout, &stderr); code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "waldo index summary") {
		t.Fatalf("leaf help does not name command:\n%s", stdout.String())
	}
}

func TestFlagRichHelpExplainsRetainedOptions(t *testing.T) {
	for _, test := range []struct {
		args []string
		want []string
	}{
		{[]string{"index", "ingest", "--help"}, []string{"Required:", "--text-column", "waldo config set", "no transport or scratch flags"}},
		{[]string{"config", "set", "--help"}, []string{"lookaside.scratch", "file:///tmp", "waldo lookaside login", "~/.waldo/credentials"}},
		{[]string{"lookaside", "login", "--help"}, []string{"S3 access key", "hidden secret key", "~/.waldo/credentials", "0600"}},
		{[]string{"index", "export", "--help"}, []string{"--force", "purged only after", "OpenWALDO BOM"}},
	} {
		var stdout, stderr bytes.Buffer
		if code := Run(test.args, &stdout, &stderr); code != 0 {
			t.Fatalf("%v: code = %d, stderr = %q", test.args, code, stderr.String())
		}
		for _, want := range test.want {
			if !strings.Contains(stdout.String(), want) {
				t.Errorf("%v help missing %q:\n%s", test.args, want, stdout.String())
			}
		}
	}
}

func TestModelTrainRequiresName(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"model", "train"}, &stdout, &stderr); code != 1 {
		t.Fatalf("Run() code = %d, want 1", code)
	}
	if !strings.Contains(stderr.String(), "Error: requires at least 1 arg") || !strings.Contains(stdout.String(), "waldo model train <name> [index-path") {
		t.Fatalf("Cobra error/usage = stderr %q, stdout %q", stderr.String(), stdout.String())
	}
}

func TestModelTrainAllowsOmittedIndexSelection(t *testing.T) {
	context, args, err := parseCobraCommand(t, []string{"model", "train"}, []string{"example"})
	if err != nil {
		t.Fatal(err)
	}
	if len(args) != 1 || args[0] != "example" || int64Option(context, "epochs") != 1 {
		t.Fatalf("args = %v, epochs = %d", args, int64Option(context, "epochs"))
	}
}

func TestUnknownCommandSuggestsScopedHelp(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"lookaside", "explode"}, &stdout, &stderr); code != 1 {
		t.Fatalf("Run() code = %d, want 1", code)
	}
	if !strings.Contains(stderr.String(), "unknown command \"explode\" for \"waldo lookaside\"") || !strings.Contains(stdout.String(), "waldo lookaside [flags]") {
		t.Fatalf("Cobra error/usage = stderr %q, stdout %q", stderr.String(), stdout.String())
	}
}

func TestIndexVerifyRejectsConflictingVerificationLevels(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run([]string{"index", "verify", "--offline", "--objects"}, &stdout, &stderr)
	if code != 1 || !strings.Contains(stderr.String(), "different verification levels") {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
}

func TestIndexVerifyAcceptsAbsoluteCorpusDirectory(t *testing.T) {
	root := fixtureCLIIndex(t)
	var stdout, stderr bytes.Buffer
	code := Run([]string{"index", "verify", filepath.Join(root, "books"), "--offline"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "1 directories, 1 corpora, 1 shards") {
		t.Fatalf("stdout = %q", stdout.String())
	}
}

func TestRemovedIndexOptionIsRejected(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run([]string{"--index", "/tmp/index", "index", "list"}, &stdout, &stderr)
	if code != 1 || !strings.Contains(stderr.String(), "unknown flag: --index") {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
}

func fixtureCLIIndex(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeCLIFile(t, filepath.Join(root, "index.json"), `{
  "kind": "index", "schema": 1, "path": "",
  "entries": [{"name": "books", "type": "dir"}]
}`)
	writeCLIFile(t, filepath.Join(root, "books", "index.json"), `{
  "kind": "index", "schema": 1, "path": "books",
  "entries": [{"name": "books.json", "type": "manifest"}]
}`)
	writeCLIFile(t, filepath.Join(root, "books", "books.json"), `{
  "kind": "manifest", "schema": 1, "name": "books", "title": "Books",
  "description": "Fixture books.", "license": "CC0-1.0",
  "sources": [{"name": "fixture", "source": "Fixture", "url": "https://example.test", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],
  "converted_by": {"tool": "test", "version": "1", "profile": "text", "recipe": "test/v1", "tokenizer": "byte"},
  "shards": [{"url": "https://objects.example/item", "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "sources": ["fixture"], "docs": 1, "tokens": 2, "bytes": 3}]
}`)
	return root
}

func TestLookasideStatusUsesNamedBackend(t *testing.T) {
	scratchRoot := filepath.Join(t.TempDir(), "objects")
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Lookaside: config.Lookaside{Scratch: scratchRoot}}); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"lookaside", "status"}, &stdout, &stderr); code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "lookaside scratch") || !strings.Contains(stdout.String(), scratchRoot) {
		t.Fatalf("lookaside status = %q", stdout.String())
	}
}

func TestConfigSetPersistsLookasideSettings(t *testing.T) {
	configurationPath := filepath.Join(t.TempDir(), "config.json")
	scratchRoot := filepath.Join(t.TempDir(), "objects")
	t.Setenv("WALDO_CONFIG", configurationPath)
	useCLICredentialStore(t, &cliCredentialStore{})
	var stdout, stderr bytes.Buffer
	commands := [][]string{
		{"config", "set", "lookaside", "s3://bucket/lookaside/v1/"},
		{"config", "set", "lookaside.region", "us-west-2"},
		{"config", "set", "lookaside.workers", "6"},
		{"config", "set", "lookaside.scratch", scratchRoot},
		{"config", "set", "lookaside.cache.retain-completed", "true"},
		{"config", "set", "lookaside.mirrors", "https://mirror.example/lookaside/v1/"},
	}
	for _, command := range commands {
		stdout.Reset()
		stderr.Reset()
		if code := Run(command, &stdout, &stderr); code != 0 {
			t.Fatalf("%v: code = %d, stderr = %q", command, code, stderr.String())
		}
	}
	configuration, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	publish := configuration.Lookaside.Publish
	if publish == nil || publish.URL != "s3://bucket/lookaside/v1" || publish.Region != "us-west-2" || publish.Workers != 6 || configuration.Lookaside.Scratch != scratchRoot || !configuration.Lookaside.RetainCompleted || len(configuration.Lookaside.Mirrors) != 1 {
		t.Fatalf("configuration = %+v", configuration)
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"lookaside", "status"}, &stdout, &stderr); code != 0 {
		t.Fatalf("status code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), scratchRoot) || !strings.Contains(stdout.String(), "https://mirror.example/lookaside/v1") {
		t.Fatalf("lookaside status = %q", stdout.String())
	}
}

func TestConfigSetLookasidePreservesRegionAndWorkers(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Lookaside: config.Lookaside{Publish: &config.Publish{
		URL: "s3://bucket/old-prefix", Region: "us-east-2", Workers: 7,
	}}}); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"config", "set", "lookaside", "s3://bucket/new-prefix"}, &stdout, &stderr); code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	configuration, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	publish := configuration.Lookaside.Publish
	if publish == nil || publish.URL != "s3://bucket/new-prefix" || publish.Region != "us-east-2" || publish.Workers != 7 {
		t.Fatalf("publish = %+v", publish)
	}

	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"config", "set", "lookaside", "file:///tmp/waldo-test-lookaside"}, &stdout, &stderr); code != 0 {
		t.Fatalf("file code=%d stderr=%q", code, stderr.String())
	}
	configuration, err = config.Load()
	if err != nil {
		t.Fatal(err)
	}
	if configuration.Lookaside.Publish == nil || configuration.Lookaside.Publish.Region != "" || configuration.Lookaside.Publish.Workers != 7 {
		t.Fatalf("file publish = %+v", configuration.Lookaside.Publish)
	}
}

func TestLookasideLoginStatusAndLogoutUseCredentialStore(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Lookaside: config.Lookaside{Publish: &config.Publish{URL: "s3://openwaldo/waldo-index", Region: "us-east-2", Workers: 2}}}); err != nil {
		t.Fatal(err)
	}
	store := &cliCredentialStore{}
	useCLICredentialStore(t, store)
	previousPrompt := promptS3Credentials
	promptS3Credentials = func(io.Writer) (lookaside.Credentials, error) {
		return lookaside.Credentials{AccessKey: "AKIAEXAMPLE1234", SecretKey: "never-print-this-secret"}, nil
	}
	t.Cleanup(func() { promptS3Credentials = previousPrompt })
	previousValidator := validateS3Credentials
	validated := false
	validateS3Credentials = func(_ context.Context, publish config.Publish, credentials lookaside.Credentials) error {
		if store.found {
			return fmt.Errorf("credentials were stored before validation")
		}
		if publish.URL != "s3://openwaldo/waldo-index" || credentials.AccessKey != "AKIAEXAMPLE1234" {
			return fmt.Errorf("unexpected validation inputs")
		}
		validated = true
		return nil
	}
	t.Cleanup(func() { validateS3Credentials = previousValidator })

	var stdout, stderr bytes.Buffer
	if code := Run([]string{"lookaside", "login"}, &stdout, &stderr); code != 0 {
		t.Fatalf("login code = %d, stderr = %q", code, stderr.String())
	}
	if !validated || !store.found || store.credentials.AccessKey != "AKIAEXAMPLE1234" || store.credentials.SecretKey != "never-print-this-secret" {
		t.Fatalf("stored credentials = %+v, found=%v", store.credentials, store.found)
	}
	if strings.Contains(stdout.String(), "AKIAEXAMPLE1234") || strings.Contains(stdout.String(), "never-print-this-secret") || !strings.Contains(stdout.String(), "…1234") {
		t.Fatalf("login output leaked or omitted credential identity: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"lookaside", "status"}, &stdout, &stderr); code != 0 || !strings.Contains(stdout.String(), filepath.Join(home, ".waldo", "credentials")+" s3://openwaldo (…1234)") {
		t.Fatalf("status code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if strings.Contains(stdout.String(), "never-print-this-secret") || strings.Contains(stdout.String(), "AKIAEXAMPLE1234") {
		t.Fatalf("status leaked credentials: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"lookaside", "logout"}, &stdout, &stderr); code != 0 || store.found {
		t.Fatalf("logout code = %d, found=%v, stderr = %q", code, store.found, stderr.String())
	}
}

func TestLookasideLoginValidationFailurePreservesStoredCredentials(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Lookaside: config.Lookaside{Publish: &config.Publish{URL: "s3://openwaldo/waldo-index", Region: "us-east-2", Workers: 2}}}); err != nil {
		t.Fatal(err)
	}
	old := lookaside.Credentials{AccessKey: "old-access", SecretKey: "old-secret"}
	store := &cliCredentialStore{credentials: old, found: true}
	useCLICredentialStore(t, store)
	previousPrompt := promptS3Credentials
	promptS3Credentials = func(io.Writer) (lookaside.Credentials, error) {
		return lookaside.Credentials{AccessKey: "new-access", SecretKey: "new-secret"}, nil
	}
	t.Cleanup(func() { promptS3Credentials = previousPrompt })
	previousValidator := validateS3Credentials
	validateS3Credentials = func(context.Context, config.Publish, lookaside.Credentials) error {
		return fmt.Errorf("write probe object: access denied")
	}
	t.Cleanup(func() { validateS3Credentials = previousValidator })

	var stdout, stderr bytes.Buffer
	if code := Run([]string{"lookaside", "login"}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "access denied") {
		t.Fatalf("login code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !store.found || store.credentials != old {
		t.Fatalf("stored credentials changed to %+v", store.credentials)
	}
}

func TestLookasideLoginRequiresConfiguredS3(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Lookaside: config.Lookaside{Publish: &config.Publish{URL: "file:///tmp/waldo-published", Workers: 2}}}); err != nil {
		t.Fatal(err)
	}
	useCLICredentialStore(t, &cliCredentialStore{})
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"lookaside", "login"}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "requires a configured s3:// lookaside") {
		t.Fatalf("login code = %d, stderr = %q", code, stderr.String())
	}
}

func TestConfigSetAcceptsFileLookasideAndUnset(t *testing.T) {
	configurationPath := filepath.Join(t.TempDir(), "config.json")
	root := filepath.Join(t.TempDir(), "published objects")
	t.Setenv("WALDO_CONFIG", configurationPath)
	var stdout, stderr bytes.Buffer
	want := (&url.URL{Scheme: "file", Path: filepath.ToSlash(root)}).String()
	code := Run([]string{"config", "set", "lookaside", want}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("Run() code = %d, stderr = %q", code, stderr.String())
	}
	configuration, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	publish := configuration.Lookaside.Publish
	if publish == nil || publish.URL != want || publish.Workers != 4 {
		t.Fatalf("local publisher = %+v, want %s", publish, want)
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"config", "unset", "lookaside"}, &stdout, &stderr); code != 0 {
		t.Fatalf("unset code = %d, stderr = %q", code, stderr.String())
	}
	configuration, err = config.Load()
	if err != nil || configuration.Lookaside.Publish != nil {
		t.Fatalf("configuration after unset = %+v, err = %v", configuration, err)
	}
}

func TestConfigShowAndGetUseCanonicalKeys(t *testing.T) {
	configurationPath := filepath.Join(t.TempDir(), "config.json")
	t.Setenv("WALDO_CONFIG", configurationPath)
	if err := config.Save(config.Config{Lookaside: config.Lookaside{Publish: &config.Publish{URL: "s3://bucket/root", Workers: 3}}}); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"config", "get", "lookaside"}, &stdout, &stderr); code != 0 || !strings.Contains(stdout.String(), "lookaside                   s3://bucket/root") || !strings.Contains(stdout.String(), "lookaside.region            (unset)") || !strings.Contains(stdout.String(), "lookaside.workers           3") {
		t.Fatalf("get code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"config", "show"}, &stdout, &stderr); code != 0 {
		t.Fatalf("show code = %d, stderr = %q", code, stderr.String())
	}
	for _, want := range []string{"index", "lookaside", "lookaside.workers", "lookaside.scratch", "ingest.staging", "model.root"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("config show missing %q: %s", want, stdout.String())
		}
	}
}

func TestConfigSetIndexEnablesLogicalIndexPaths(t *testing.T) {
	root := fixtureCLIIndex(t)
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	t.Chdir(t.TempDir())
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"config", "set", "index", root}, &stdout, &stderr); code != 0 {
		t.Fatalf("set code = %d, stderr = %q", code, stderr.String())
	}
	configuration, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	if configuration.Index != root {
		t.Fatalf("configured index = %q, want %q", configuration.Index, root)
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"index", "summary", "books"}, &stdout, &stderr); code != 0 {
		t.Fatalf("summary code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "tokens   2") {
		t.Fatalf("summary = %q", stdout.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"--json", "index", "summary"}, &stdout, &stderr); code != 0 {
		t.Fatalf("whole summary code = %d, stderr = %q", code, stderr.String())
	}
	if stderr.Len() != 0 || !json.Valid(stdout.Bytes()) {
		t.Fatalf("whole summary stdout = %q, stderr = %q", stdout.String(), stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"index", "summary", root}, &stdout, &stderr); code != 0 || strings.Contains(stderr.String(), "warning:") {
		t.Fatalf("explicit root code = %d, stderr = %q", code, stderr.String())
	}
}

func TestConfigGetWithoutKeyListsAllSupportedValues(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"config", "get"}, &stdout, &stderr); code != 0 {
		t.Fatalf("get code = %d, stderr = %q", code, stderr.String())
	}
	for _, key := range configKeys {
		if !strings.Contains(stdout.String(), key) {
			t.Fatalf("config get missing %q: %s", key, stdout.String())
		}
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"config", "get", "lookaside.region"}, &stdout, &stderr); code != 0 || strings.TrimSpace(stdout.String()) != "(unset)" {
		t.Fatalf("unset leaf code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"config", "get", "lookaside.r"}, &stdout, &stderr); code != 0 || !strings.Contains(stdout.String(), "lookaside.region") {
		t.Fatalf("partial prefix code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
}

func TestConfigGetJSONPreservesOrderedMatchesAndUnsetState(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"--json", "config", "get", "lookaside"}, &stdout, &stderr); code != 0 {
		t.Fatalf("get code = %d, stderr = %q", code, stderr.String())
	}
	var output struct {
		Matches []configMatch `json:"matches"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatal(err)
	}
	if len(output.Matches) != 8 || output.Matches[0].Key != "lookaside" || output.Matches[1].Key != "lookaside.region" || output.Matches[1].Set {
		t.Fatalf("matches = %+v", output.Matches)
	}
}

func TestConfigRejectsInvalidCompletedCacheRetention(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"config", "set", "lookaside.cache.retain-completed", "yes"}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "must be true or false") {
		t.Fatalf("code = %d, stderr = %q", code, stderr.String())
	}
}

func TestConfigRejectsUnknownKey(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"config", "set", "mystery", "value"}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "unknown configuration key") {
		t.Fatalf("code = %d, stderr = %q", code, stderr.String())
	}
}

func TestRuntimeErrorsDoNotPrintCommandUsage(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"config", "set", "ai.provider", "invalid"}, &stdout, &stderr); code != 1 {
		t.Fatalf("code = %d, stderr = %q", code, stderr.String())
	}
	if strings.Contains(stdout.String(), "Usage:") || strings.Contains(stderr.String(), "Usage:") {
		t.Fatalf("runtime error printed usage: stdout = %q, stderr = %q", stdout.String(), stderr.String())
	}
}

func TestConfigMasksStoredAIKeys(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	var stdout, stderr bytes.Buffer
	secret := "never-print-this-secret"
	if code := Run([]string{"config", "set", "ai.api-key", secret}, &stdout, &stderr); code != 0 {
		t.Fatalf("set code = %d, stderr = %q", code, stderr.String())
	}
	if strings.Contains(stdout.String(), secret) || !strings.Contains(stdout.String(), "(set)") {
		t.Fatalf("set output exposed or omitted key state: %q", stdout.String())
	}
	stdout.Reset()
	if code := Run([]string{"config", "get", "ai.api-key"}, &stdout, &stderr); code != 0 || strings.Contains(stdout.String(), secret) || strings.TrimSpace(stdout.String()) != "(set)" {
		t.Fatalf("get code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
}
