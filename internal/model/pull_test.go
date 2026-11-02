// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/openwaldo/waldo/internal/modelweights"
	"github.com/openwaldo/waldo/internal/training"
)

func isolateHuggingFaceToken(t *testing.T) {
	t.Helper()
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HF_TOKEN_PATH", filepath.Join(t.TempDir(), "absent"))
}

func TestDownloadHuggingFaceModelCreatesVerifiedOrigin(t *testing.T) {
	isolateHuggingFaceToken(t)
	architecture := Architecture{Family: "decoder-transformer", ContextTokens: 32, VocabularySize: 259, HiddenSize: 4, IntermediateSize: 8, Layers: 1, AttentionHeads: 2, KeyValueHeads: 1, TieEmbeddings: true, ParameterDType: "bfloat16", Tokenizer: Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"}}
	files := huggingFaceFixture(t, architecture, HuggingFaceTokenizerClass)
	revision := strings.Repeat("a", 40)
	client := &http.Client{Transport: huggingFaceTransport(t, files, revision)}
	root := t.TempDir()
	var phases []string
	inspection, err := (Puller{Root: root, Endpoint: "https://hub.test", Client: client, Token: "secret", Now: func() time.Time { return time.Date(2026, 8, 5, 1, 2, 3, 0, time.UTC) }, Progress: func(progress PullProgress) { phases = append(phases, progress.Phase) }}).Pull(t.Context(), "tiny", "huggingface://org/tiny")
	if err != nil {
		t.Fatal(err)
	}
	if inspection.Origin == nil || inspection.Origin.Source.Revision != revision || inspection.Origin.Source.RequestedRevision != "main" || inspection.BOM.CurrentOriginSHA256 == "" || inspection.BOM.CurrentRunID != "" || inspection.Plan.OriginBOMSHA256 != inspection.Model.OriginBOMSHA256 {
		t.Fatalf("inspection = %+v", inspection)
	}
	if got := strings.Join(phases, ","); !strings.Contains(got, "resolve") || !strings.Contains(got, "normalize") || !strings.HasSuffix(got, "complete") {
		t.Fatalf("phases = %s", got)
	}
	initialization, err := resolveInitialization(inspection)
	if err != nil {
		t.Fatal(err)
	}
	if initialization == nil || initialization.SourceType != "origin" || initialization.SourceID != inspection.Model.OriginBOMSHA256 || initialization.SourceRunID != "" {
		t.Fatalf("initialization = %+v", initialization)
	}
	if _, err := os.Stat(filepath.Join(inspection.Path, "origin", "artifacts", "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(inspection.Path, ".source")); !os.IsNotExist(err) {
		t.Fatalf("source staging remains: %v", err)
	}
	stage := testStage("continue")
	stage.Parameters.SequenceLength = 32
	compose := Compose{Kind: "waldo-model-compose", Schema: 1, Base: &ComposeBase{Model: "tiny", OriginSHA256: inspection.Model.OriginBOMSHA256}, Architecture: architecture, Stages: []Stage{stage}}
	continued, err := (Builder{Root: root, NewID: func() (string, error) { return "continue0001", nil }, Resolver: training.FakeResolver()}).Compose(t.Context(), "continued", compose, []PreparedStage{preparedFixture(t, stage)})
	if err != nil {
		t.Fatal(err)
	}
	if continued.Model.OriginBOMSHA256 != inspection.Model.OriginBOMSHA256 || continued.Origin == nil || continued.Plan.OriginBOMSHA256 != inspection.Model.OriginBOMSHA256 {
		t.Fatalf("continued origin = %+v", continued)
	}
}

func TestDownloadRejectsUnsupportedTokenizerWithoutPublishing(t *testing.T) {
	isolateHuggingFaceToken(t)
	architecture := Architecture{Family: "decoder-transformer", ContextTokens: 32, VocabularySize: 259, HiddenSize: 4, IntermediateSize: 8, Layers: 1, AttentionHeads: 2, KeyValueHeads: 1, TieEmbeddings: true, ParameterDType: "bfloat16", Tokenizer: Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"}}
	files := huggingFaceFixture(t, architecture, "LlamaTokenizerFast")
	client := &http.Client{Transport: huggingFaceTransport(t, files, strings.Repeat("b", 40))}
	root := t.TempDir()
	_, err := (Puller{Root: root, Endpoint: "https://hub.test", Client: client}).Pull(t.Context(), "bad", "huggingface://org/bad@main")
	if err == nil || !strings.Contains(err.Error(), "currently supports OpenWALDOTokenizer") {
		t.Fatalf("error = %v", err)
	}
	if _, statErr := os.Stat(filepath.Join(root, "bad")); !os.IsNotExist(statErr) {
		t.Fatalf("model was published: %v", statErr)
	}
}

func TestValidateDownloadedTokenizerAcceptsLegacyClassAlias(t *testing.T) {
	directory := t.TempDir()
	configuration, err := json.Marshal(map[string]any{"tokenizer_class": LegacyHuggingFaceTokenizerClass, "pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>"})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "tokenizer_config.json"), configuration, 0o644); err != nil {
		t.Fatal(err)
	}
	architecture := Architecture{VocabularySize: 259}
	if err := validateDownloadedTokenizer(directory, architecture); err != nil {
		t.Fatalf("legacy exported tokenizer rejected: %v", err)
	}
}

func TestComposeAcquiresPinnedHuggingFaceBaseWithoutVisibleModel(t *testing.T) {
	isolateHuggingFaceToken(t)
	architecture := Architecture{Family: "decoder-transformer", ContextTokens: 32, VocabularySize: 259, HiddenSize: 4, IntermediateSize: 8, Layers: 1, AttentionHeads: 2, KeyValueHeads: 1, TieEmbeddings: true, ParameterDType: "bfloat16", Tokenizer: Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"}}
	files := huggingFaceFixture(t, architecture, HuggingFaceTokenizerClass)
	revision := strings.Repeat("c", 40)
	client := &http.Client{Transport: huggingFaceTransport(t, files, revision)}
	root := t.TempDir()
	builder := Builder{Root: root, OriginPuller: &Puller{Endpoint: "https://hub.test", Client: client, Token: "secret"}, NewID: func() (string, error) { return "source0001", nil }, Resolver: training.FakeResolver()}
	stage := testStage("adapt")
	stage.Parameters.SequenceLength = 32
	compose := Compose{Kind: "waldo-model-compose", Schema: 1, Base: &ComposeBase{Source: "huggingface://org/tiny@" + revision}, Stages: []Stage{stage}}
	probed, err := builder.ResolveCompose(t.Context(), "", compose, false)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(probed.Architecture, architecture) || probed.Base.OriginSHA256 != "" {
		t.Fatalf("probed compose = %+v", probed)
	}
	prepared := preparedFixture(t, stage)
	trained, err := builder.Compose(t.Context(), "adapted", compose, []PreparedStage{prepared})
	if err != nil {
		t.Fatal(err)
	}
	if trained.Origin == nil || trained.Origin.Source.Repository != "org/tiny" || trained.Origin.Source.Revision != revision || trained.Model.Architecture != architecture {
		t.Fatalf("trained origin = %+v, architecture = %+v", trained.Origin, trained.Model.Architecture)
	}
	listed, err := List(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(listed) != 1 || listed[0].Name != "adapted" {
		t.Fatalf("visible models = %+v", listed)
	}
	if _, err := os.Stat(filepath.Join(root, ".origins")); err != nil {
		t.Fatalf("hidden origin cache: %v", err)
	}
}

func TestComposeBaseSourceRequiresImmutableCommit(t *testing.T) {
	compose := validCompose()
	compose.Base = &ComposeBase{Source: "huggingface://org/model@main"}
	if err := compose.Validate(); err == nil || !strings.Contains(err.Error(), "immutable Hugging Face commit") {
		t.Fatalf("mutable base source error = %v", err)
	}
}

func TestValidateHuggingFaceShardIndex(t *testing.T) {
	directory := t.TempDir()
	first := "model-00001-of-00002.safetensors"
	second := "model-00002-of-00002.safetensors"
	index := []byte(`{"metadata":{"total_size":42},"weight_map":{"model.embed_tokens.weight":"model-00001-of-00002.safetensors","model.norm.weight":"model-00002-of-00002.safetensors"}}`)
	if err := os.WriteFile(filepath.Join(directory, "model.safetensors.index.json"), index, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := validateShardIndex(directory, []string{first, second, "model.safetensors.index.json"}); err != nil {
		t.Fatal(err)
	}
	if err := validateShardIndex(directory, []string{first, "model.safetensors.index.json"}); err == nil || !strings.Contains(err.Error(), "unavailable shard") {
		t.Fatalf("missing shard error = %v", err)
	}
}

func huggingFaceFixture(t *testing.T, architecture Architecture, tokenizerClass string) map[string][]byte {
	t.Helper()
	config, err := json.Marshal(map[string]any{"model_type": "llama", "hidden_size": architecture.HiddenSize, "intermediate_size": architecture.IntermediateSize, "num_hidden_layers": architecture.Layers, "num_attention_heads": architecture.AttentionHeads, "num_key_value_heads": architecture.KeyValueHeads, "vocab_size": architecture.VocabularySize, "max_position_embeddings": architecture.ContextTokens, "tie_word_embeddings": architecture.TieEmbeddings, "torch_dtype": architecture.ParameterDType, "rms_norm_eps": 1e-5, "rope_theta": 10000, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2})
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, _ := json.Marshal(map[string]any{"tokenizer_class": tokenizerClass, "pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>"})
	return map[string][]byte{"config.json": config, "tokenizer_config.json": tokenizer, "README.md": []byte("fixture"), "model.safetensors": huggingFaceWeights(t, architecture)}
}

func huggingFaceWeights(t *testing.T, architecture Architecture) []byte {
	t.Helper()
	header := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var offset uint64
	shapes := expectedTensorShapes(architecture)
	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	sortStrings(names)
	for _, name := range names {
		hf, err := modelweights.HuggingFaceName(name)
		if err != nil {
			t.Fatal(err)
		}
		elements := uint64(1)
		for _, dimension := range shapes[name] {
			elements *= dimension
		}
		bytes := elements * 2
		header[hf] = map[string]any{"dtype": "BF16", "shape": shapes[name], "data_offsets": []uint64{offset, offset + bytes}}
		offset += bytes
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(encoded)%8 != 0 {
		encoded = append(encoded, ' ')
	}
	result := make([]byte, 8+len(encoded)+int(offset))
	binary.LittleEndian.PutUint64(result[:8], uint64(len(encoded)))
	copy(result[8:], encoded)
	for index := 8 + len(encoded); index < len(result); index++ {
		result[index] = byte(index % 251)
	}
	return result
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func huggingFaceTransport(t *testing.T, files map[string][]byte, revision string) http.RoundTripper {
	t.Helper()
	return roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if authorization := request.Header.Get("Authorization"); authorization != "" && authorization != "Bearer secret" {
			t.Errorf("request carried an unexpected authorization header (%d bytes, redacted)", len(authorization))
		}
		if strings.HasPrefix(request.URL.Path, "/api/models/org/tiny/revision/") || strings.HasPrefix(request.URL.Path, "/api/models/org/bad/revision/") {
			siblings := make([]map[string]any, 0, len(files))
			for name, data := range files {
				siblings = append(siblings, map[string]any{"rfilename": name, "size": len(data)})
			}
			data, _ := json.Marshal(map[string]any{"id": strings.TrimPrefix(strings.Split(request.URL.Path, "/revision/")[0], "/api/models/"), "sha": revision, "siblings": siblings, "cardData": map[string]any{"license": "apache-2.0"}})
			return testResponse(http.StatusOK, data), nil
		}
		prefixes := []string{"/org/tiny/resolve/" + revision + "/", "/org/bad/resolve/" + revision + "/"}
		for _, prefix := range prefixes {
			if strings.HasPrefix(request.URL.Path, prefix) {
				name := strings.TrimPrefix(request.URL.Path, prefix)
				if data, ok := files[name]; ok {
					return testResponse(http.StatusOK, data), nil
				}
			}
		}
		return testResponse(http.StatusNotFound, []byte(fmt.Sprintf("unexpected %s", request.URL.Path))), nil
	})
}

func testResponse(status int, data []byte) *http.Response {
	return &http.Response{StatusCode: status, Status: fmt.Sprintf("%d %s", status, http.StatusText(status)), Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(data))}
}

func sortStrings(values []string) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j] < values[j-1]; j-- {
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}
