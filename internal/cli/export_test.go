// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	waldoai "github.com/openwaldo/waldo/internal/ai"
	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/record"
	"github.com/openwaldo/waldo/internal/shard"
	"github.com/parquet-go/parquet-go"
)

func TestIndexExportEndToEnd(t *testing.T) {
	root := t.TempDir()
	text := "small native shard"
	var parquetData bytes.Buffer
	writer := parquet.NewGenericWriter[shard.Row](&parquetData)
	if _, err := writer.Write([]shard.Row{{
		SHA256: record.TextHash(text), Kind: record.KindPretrain, Text: text,
		Source: "fixture", License: "CC0-1.0", Tokens: 3,
	}}); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	content := parquetData.Bytes()
	digestArray := sha256.Sum256(content)
	digest := hex.EncodeToString(digestArray[:])
	source := filepath.Join(root, "source.parquet")
	if err := os.WriteFile(source, content, 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLIFile(t, filepath.Join(root, "index.json"), `{
  "kind": "index", "schema": 1, "path": "",
  "entries": [{"name": "books", "type": "dir"}]
}`)
	writeCLIFile(t, filepath.Join(root, "books", "index.json"), `{
  "kind": "index", "schema": 1, "path": "books",
  "entries": [{"name": "books.json", "type": "manifest"}]
}`)
	manifest := fmt.Sprintf(`{
  "kind": "manifest", "schema": 1, "name": "books", "title": "Books",
  "description": "Small books.", "license": "CC0-1.0",
  "sources": [{"name": "source", "source": "Fixture", "url": "https://example.test", "sha256": %q}],
  "converted_by": {"tool": "test", "version": "1", "profile": "text", "recipe": "test/v1", "tokenizer": "byte"},
  "shards": [{"url": %q, "sha256": %q, "sources": ["source"], "docs": 1, "tokens": 3, "bytes": %d}]
}`, strings.Repeat("a", 64), source, digest, len(content))
	writeCLIFile(t, filepath.Join(root, "books", "books.json"), manifest)

	cache := filepath.Join(t.TempDir(), "cache")
	models := filepath.Join(t.TempDir(), "models")
	destination := filepath.Join(t.TempDir(), "export")
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Index: root, Lookaside: config.Lookaside{Scratch: cache}, Model: config.Model{Root: models, Backend: "fake"}}); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := Run([]string{"index", "bom", filepath.Join(root, "books")}, &stdout, &stderr); code != 0 {
		t.Fatalf("index bom code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), `"subject": "corpus"`) || !strings.Contains(stdout.String(), `"paths": [`+"\n"+`    "books"`) {
		t.Fatalf("index bom output = %q", stdout.String())
	}
	stdout.Reset()
	stderr.Reset()
	code := Run([]string{"index", "export", filepath.Join(root, "books"), destination}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("Run() code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "EXPORT.json") {
		t.Fatalf("stdout = %q", stdout.String())
	}
	if _, err := os.Stat(filepath.Join(destination, "EXPORT.json")); err != nil {
		t.Fatal(err)
	}
	matches, err := filepath.Glob(filepath.Join(destination, "data", "books", "*.parquet"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 {
		t.Fatalf("exported parquet files = %v", matches)
	}
	assertNoCacheFiles(t, cache)
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"index", "verify", destination}, &stdout, &stderr); code != 0 {
		t.Fatalf("index verify export code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "verified OpenWALDO BOM") {
		t.Fatalf("bom verify output = %q", stdout.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"index", "bom", destination}, &stdout, &stderr); code != 0 {
		t.Fatalf("index bom export code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "OpenWALDO corpus export") || !strings.Contains(stdout.String(), "native") {
		t.Fatalf("bom show output = %q", stdout.String())
	}

	compose := filepath.Join(t.TempDir(), "smoke.yaml")
	composeData := fmt.Sprintf(`kind: waldo-model-compose
schema: 1
architecture:
  family: decoder-transformer
  context_tokens: 128
  vocabulary_size: 256
  hidden_size: 64
  intermediate_size: 192
  layers: 2
  attention_heads: 4
  key_value_heads: 2
  tie_embeddings: true
  parameter_dtype: float32
  tokenizer:
    name: byte
    revision: sha256:fixture
stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    corpora:
      - %q
    parameters:
      steps: 2
      batch_size: 1
      sequence_length: 64
      learning_rate: 0.001
      seed: 7
`, filepath.Join(root, "books"))
	if err := os.WriteFile(compose, []byte(composeData), 0o644); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"model", "train", "smoke", compose}, &stdout, &stderr); code != 0 {
		t.Fatalf("compose-driven model train code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "trained model smoke") || !strings.Contains(stderr.String(), "preflight/pretrain") {
		t.Fatalf("compose-driven model train stdout = %q, stderr = %q", stdout.String(), stderr.String())
	}
	runBOMs, err := filepath.Glob(filepath.Join(models, "smoke", "runs", "*", "RUN-BOM.json"))
	if err != nil || len(runBOMs) != 1 {
		t.Fatalf("run BOM paths = %v, error = %v", runBOMs, err)
	}
	runBOMData, err := os.ReadFile(runBOMs[0])
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(runBOMData, []byte(`"attestation"`)) {
		t.Fatal("default model compose unexpectedly audited shard attestations")
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"model", "summary", "smoke"}, &stdout, &stderr); code != 0 {
		t.Fatalf("model summary code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "complete") || !strings.Contains(stdout.String(), "simulated") || !strings.Contains(stdout.String(), "ADVICE:        complete") {
		t.Fatalf("model summary stdout = %q", stdout.String())
	}
	advisorCompose, _, err := model.LoadCompose(filepath.Join(models, "smoke", "COMPOSE.json"))
	if err != nil {
		t.Fatal(err)
	}
	advisorCompose.Stages[0].Parameters.Steps = 3
	proposal, err := json.Marshal(advisorReply{Reply: "A slightly longer run tests continued improvement.", Changes: []string{"increase steps from 2 to 3"}, Compose: &advisorCompose})
	if err != nil {
		t.Fatal(err)
	}
	previousAdvisorAsk := modelAdvisorAsk
	previousAdvisorInput := modelAdvisorInput
	previousAdvisorTerminal := modelAdvisorTerminal
	modelAdvisorAsk = func(context.Context, waldoai.Selection, string) (string, error) { return string(proposal), nil }
	modelAdvisorInput = strings.NewReader("How should I improve it?\nyes\nquit\n")
	modelAdvisorTerminal = func() bool { return true }
	t.Cleanup(func() {
		modelAdvisorAsk = previousAdvisorAsk
		modelAdvisorInput = previousAdvisorInput
		modelAdvisorTerminal = previousAdvisorTerminal
	})
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	advisorDirectory := t.TempDir()
	workingDirectory, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(advisorDirectory); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(workingDirectory) })
	advisorPath := filepath.Join(advisorDirectory, "0000-smoke-advisor.yaml")
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"model", "advisor", "smoke", "--provider", "anthropic", "--model", "claude-sonnet-5"}, &stdout, &stderr); code != 0 {
		t.Fatalf("model advisor code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	generated, _, err := model.LoadCompose(advisorPath)
	if err != nil || generated.Stages[0].Parameters.Steps != 3 || !strings.Contains(stdout.String(), "updated ") || !strings.Contains(stderr.String(), "thinking with anthropic/claude-sonnet-5") {
		t.Fatalf("generated advisor compose = %+v, err = %v, stdout = %q, stderr = %q", generated, err, stdout.String(), stderr.String())
	}
	newCompose := advisorTestCompose()
	newCompose.Stages[0].Corpora = model.NewCorpusSelections([]string{"books"})
	newProposal, err := json.Marshal(advisorReply{
		Reply:   "I have enough information to propose a small test model.",
		Changes: []string{"train a small byte-level model on the indexed books corpus"}, Compose: &newCompose,
	})
	if err != nil {
		t.Fatal(err)
	}
	modelAdvisorAsk = func(_ context.Context, _ waldoai.Selection, prompt string) (string, error) {
		if strings.Contains(prompt, "monitoring a WALDO training build") {
			return `{"reply":"Checkpoint is healthy; let the run continue."}`, nil
		}
		return string(newProposal), nil
	}
	modelAdvisorInput = strings.NewReader("Make a tiny test model for this machine.\nyes\nyes\n")
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"advisor", "fresh", "--provider", "anthropic", "--model", "claude-sonnet-5"}, &stdout, &stderr); code != 0 {
		t.Fatalf("new-model advisor code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if exists, err := model.Exists(models, "fresh"); err != nil || !exists || !strings.Contains(stdout.String(), "trained model fresh") {
		t.Fatalf("new advisor model exists = %v, err = %v, stdout = %q", exists, err, stdout.String())
	}
	for _, path := range []string{
		filepath.Join(models, "smoke", "composes", "0000-smoke.yaml"),
		filepath.Join(models, "fresh", "composes", "0000-fresh-advisor.yaml"),
		filepath.Join(models, "smoke", "advisor", "CHAT.jsonl"),
		filepath.Join(models, "fresh", "advisor", "CHAT.jsonl"),
	} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("durable advisor/build history %s: %v", path, err)
		}
	}
	chat, err := os.ReadFile(filepath.Join(models, "fresh", "advisor", "CHAT.jsonl"))
	if err != nil || !bytes.Contains(chat, []byte(`"category":"build"`)) || !bytes.Contains(chat, []byte(`"category":"checkpoint-monitor"`)) {
		t.Fatalf("fresh advisor chat = %s, err = %v", chat, err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"model", "continue", "smoke"}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "no interrupted compose") {
		t.Fatalf("completed continue code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	for _, name := range []string{"PLAN.json", "MODEL.json", "MODEL-BOM.json"} {
		if _, err := os.Stat(filepath.Join(models, "smoke", name)); err != nil {
			t.Fatal(err)
		}
	}

	disclosureOutput := filepath.Join(t.TempDir(), "eu-gpai.json")
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"model", "bom", "smoke", disclosureOutput, "--format", "eu-gpai"}, &stdout, &stderr); code != 1 {
		t.Fatalf("complete disclosure code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "export blocked") {
		t.Fatalf("blocked disclosure stderr = %q", stderr.String())
	}
	if _, err := os.Stat(disclosureOutput); !os.IsNotExist(err) {
		t.Fatalf("blocked disclosure wrote output: %v", err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"model", "bom", "smoke", disclosureOutput, "--format=eu-gpai", "--allow-incomplete"}, &stdout, &stderr); code != 0 {
		t.Fatalf("draft disclosure code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	disclosureData, err := os.ReadFile(disclosureOutput)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(disclosureData), `"status": "incomplete-draft"`) || !strings.Contains(string(disclosureData), `"field": "provider.profile"`) || !strings.Contains(string(disclosureData), `"field": "training.observed-consumption"`) {
		t.Fatalf("draft disclosure = %s", disclosureData)
	}
	stdout.Reset()
	stderr.Reset()
	if code := Run([]string{"model", "bom", "smoke", "--format", "eu-gpai", "--allow-incomplete"}, &stdout, &stderr); code != 0 {
		t.Fatalf("stdout disclosure code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.HasPrefix(stdout.String(), "{\n") || !strings.Contains(stdout.String(), `"kind": "waldo-eu-gpai-training-content"`) || !strings.Contains(stdout.String(), `"status": "incomplete-draft"`) {
		t.Fatalf("stdout disclosure = %q", stdout.String())
	}

	jsonlDestination := filepath.Join(t.TempDir(), "jsonl-export")
	stdout.Reset()
	stderr.Reset()
	code = Run([]string{"index", "export", filepath.Join(root, "books"), "--format=jsonl", jsonlDestination}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("JSONL Run() code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	jsonlMatches, err := filepath.Glob(filepath.Join(jsonlDestination, "data", "books", "*.jsonl"))
	if err != nil || len(jsonlMatches) != 1 {
		t.Fatalf("exported JSONL files = %v, error = %v", jsonlMatches, err)
	}
	jsonl, err := os.ReadFile(jsonlMatches[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(jsonl), `"text":"small native shard"`) {
		t.Fatalf("JSONL = %q", jsonl)
	}
	assertNoCacheFiles(t, cache)
}

func TestModelExportRequiresDisclosureAndPublishesBothBOMs(t *testing.T) {
	models := filepath.Join(t.TempDir(), "models")
	if _, err := (&model.Builder{Root: models}).Initialize("release", model.Architecture{
		Family: "decoder-transformer", ContextTokens: 128, VocabularySize: 256,
		HiddenSize: 64, IntermediateSize: 192, Layers: 2, AttentionHeads: 4,
		KeyValueHeads: 2, TieEmbeddings: true, ParameterDType: "float32",
		Tokenizer: model.Tokenizer{Name: "byte", Revision: "sha256:test"},
	}); err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(t.TempDir(), "config.json")
	t.Setenv("WALDO_CONFIG", configPath)
	if err := config.Save(config.Config{Model: config.Model{Root: models}}); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	missingProviderOutput := filepath.Join(t.TempDir(), "missing-provider")
	if code := Run([]string{"model", "export", "release", missingProviderOutput, "--allow-incomplete"}, &stdout, &stderr); code != 1 {
		t.Fatalf("missing provider code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "config set disclosure.provider") {
		t.Fatalf("missing provider stderr = %q", stderr.String())
	}
	if _, err := os.Stat(missingProviderOutput); !os.IsNotExist(err) {
		t.Fatalf("failed export created destination: %v", err)
	}

	provider := filepath.Join(t.TempDir(), "provider.json")
	writeCLIFile(t, provider, `{
  "kind": "waldo-disclosure-provider", "schema": 1,
  "provider": {"name": "Example", "address": "1 Test Way", "contact": "test@example.invalid"},
  "code_of_practice_status": "not-assessed",
  "copyright_policy_url": "https://example.invalid/copyright"
}`)
	if err := config.Save(config.Config{Model: config.Model{Root: models}, Disclosure: config.Disclosure{Provider: provider}}); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	blockedOutput := filepath.Join(t.TempDir(), "blocked")
	if code := Run([]string{"model", "export", "release", blockedOutput}, &stdout, &stderr); code != 1 {
		t.Fatalf("incomplete disclosure code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "export blocked") {
		t.Fatalf("incomplete disclosure stderr = %q", stderr.String())
	}
	if _, err := os.Stat(blockedOutput); !os.IsNotExist(err) {
		t.Fatalf("blocked export created destination: %v", err)
	}

	stdout.Reset()
	stderr.Reset()
	destination := filepath.Join(t.TempDir(), "release-export")
	if code := Run([]string{"model", "export", "release", destination, "--allow-incomplete"}, &stdout, &stderr); code != 0 {
		t.Fatalf("model export code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	for _, name := range []string{"BOM.json", "EU-BOM.json", "ATTRIBUTION.md"} {
		if _, err := os.Stat(filepath.Join(destination, name)); err != nil {
			t.Fatalf("missing %s: %v", name, err)
		}
	}
	if _, err := os.Stat(filepath.Join(destination, "MODEL-BOM.json")); !os.IsNotExist(err) {
		t.Fatalf("export retained internal MODEL-BOM name: %v", err)
	}
	if !strings.Contains(stderr.String(), "model export is unsigned") {
		t.Fatalf("unsigned export stderr = %q", stderr.String())
	}
	if _, err := model.Inspect(t.TempDir(), destination); err != nil {
		t.Fatalf("inspect relocated export: %v", err)
	}

	configured, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	configured.Signing.Method = "sigstore-keyless"
	if err := config.Save(configured); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", t.TempDir())
	stdout.Reset()
	stderr.Reset()
	unsignedFallback := filepath.Join(t.TempDir(), "must-not-fall-back")
	if code := Run([]string{"model", "export", "release", unsignedFallback, "--allow-incomplete"}, &stdout, &stderr); code != 1 {
		t.Fatalf("configured signing failure code = %d, stdout = %q, stderr = %q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "cosign is not installed") {
		t.Fatalf("configured signing failure stderr = %q", stderr.String())
	}
	if _, err := os.Stat(unsignedFallback); !os.IsNotExist(err) {
		t.Fatalf("signing failure published an unsigned directory: %v", err)
	}
}

func assertNoCacheFiles(t *testing.T, root string) {
	t.Helper()
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			return err
		}
		if !entry.IsDir() {
			t.Fatalf("cache file remains after successful command: %s", path)
		}
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
}

func TestParseExportOptions(t *testing.T) {
	context, args, err := parseCobraCommand(t, []string{"index", "export"}, []string{"core", "science", "--format=jsonl", "--license", "CC0-*, CC-BY-*", "--exclude-license=CC-BY-NC-*", "--force", "dest"})
	if err == nil {
		var options exportOptions
		options, err = cobraExportOptions(context, args)
		if err == nil && (len(options.Paths) != 2 || len(options.Include) != 2 || len(options.Exclude) != 1 || options.Output != "dest" || options.Format != "jsonl" || !options.Force) {
			t.Fatalf("cobraExportOptions() = %+v", options)
		}
	}
	if err != nil {
		t.Fatal(err)
	}
}

func TestParseExportOptionsAllowsWholeIndexSelection(t *testing.T) {
	context, args, err := parseCobraCommand(t, []string{"index", "export"}, []string{"destination"})
	if err != nil {
		t.Fatal(err)
	}
	options, err := cobraExportOptions(context, args)
	if err != nil {
		t.Fatal(err)
	}
	if options.Output != "destination" || len(options.Paths) != 0 {
		t.Fatalf("cobraExportOptions() = %+v", options)
	}
}

func TestParseModelBOMOptions(t *testing.T) {
	defaultContext, defaultArgs, err := parseCobraCommand(t, []string{"model", "bom"}, []string{"smoke"})
	if err != nil {
		t.Fatal(err)
	}
	defaultOptions, err := cobraModelBOMOptions(defaultContext, defaultArgs)
	if err != nil || defaultOptions.Format != "openwaldo" || defaultOptions.Model != "smoke" {
		t.Fatalf("default model BOM options = %+v, %v", defaultOptions, err)
	}
	context, args, err := parseCobraCommand(t, []string{"model", "bom"}, []string{"smoke", "report.json", "--format=eu-gpai", "--provider", "provider.json", "--allow-incomplete", "--force"})
	if err != nil {
		t.Fatal(err)
	}
	options, err := cobraModelBOMOptions(context, args)
	if err != nil {
		t.Fatal(err)
	}
	if options.Model != "smoke" || options.Output != "report.json" || options.Provider != "provider.json" || !options.AllowIncomplete || !options.Force {
		t.Fatalf("cobraModelBOMOptions() = %+v", options)
	}
	context, args, err = parseCobraCommand(t, []string{"model", "bom"}, []string{"smoke", "report.docx", "--format", "eu-gpai"})
	if _, err := cobraModelBOMOptions(context, args); err == nil || !strings.Contains(err.Error(), ".json") {
		t.Fatalf("document output error = %v", err)
	}
	context, args, err = parseCobraCommand(t, []string{"model", "bom"}, []string{"smoke", "--format", "eu-gpai", "--allow-incomplete"})
	if err != nil {
		t.Fatal(err)
	}
	stdoutOptions, err := cobraModelBOMOptions(context, args)
	if err != nil || stdoutOptions.Model != "smoke" || stdoutOptions.Output != "" || !stdoutOptions.AllowIncomplete {
		t.Fatalf("stdout parse = %+v, %v", stdoutOptions, err)
	}
	context, args, err = parseCobraCommand(t, []string{"model", "bom"}, []string{"smoke", "--format", "eu-gpai", "--force"})
	if _, err := cobraModelBOMOptions(context, args); err == nil || !strings.Contains(err.Error(), "requires an output file") {
		t.Fatalf("stdout force error = %v", err)
	}
	context, args, err = parseCobraCommand(t, []string{"model", "bom"}, []string{"smoke", "--provider", "provider.json"})
	if _, err := cobraModelBOMOptions(context, args); err == nil || !strings.Contains(err.Error(), "apply only") {
		t.Fatalf("OpenWALDO provider error = %v", err)
	}
}

func TestIndexExportRejectsRemovedOutputOption(t *testing.T) {
	_, _, err := parseCobraCommand(t, []string{"index", "export"}, []string{"core", "--output", "dest"})
	if err == nil || !strings.Contains(err.Error(), "unknown flag") {
		t.Fatalf("Cobra parse error = %v", err)
	}
}

func writeCLIFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}
