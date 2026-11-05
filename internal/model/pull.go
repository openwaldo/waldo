// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/openwaldo/waldo/internal/modelweights"
	waldotokenizer "github.com/openwaldo/waldo/internal/tokenizer"
	"github.com/openwaldo/waldo/internal/training"
)

const (
	HuggingFaceTokenizerClass       = "OpenWALDOTokenizer"
	LegacyHuggingFaceTokenizerClass = "OpenWALDOByteTokenizer"
	// TokenizerArtifactFile is the exported tokenizer artifact, a trained tokenizer's vocabulary.
	TokenizerArtifactFile = "openwaldo_tokenizer.json"
)

type PullProgress struct {
	Phase   string `json:"phase"`
	File    string `json:"file,omitempty"`
	Message string `json:"message"`
}

type Puller struct {
	Root     string
	Client   *http.Client
	Endpoint string
	Token    string
	Now      func() time.Time
	Progress func(PullProgress)
}

type hubModel struct {
	ID       string `json:"id"`
	SHA      string `json:"sha"`
	Siblings []struct {
		Name string `json:"rfilename"`
		Size int64  `json:"size"`
	} `json:"siblings"`
	CardData struct {
		License string `json:"license"`
	} `json:"cardData"`
}

var huggingFaceRepository = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)
var huggingFaceCommit = regexp.MustCompile(`^[a-fA-F0-9]{40,64}$`)
var provenanceFile = regexp.MustCompile(`^provenance-([a-f0-9]{64})\.json$`)
var huggingFaceShard = regexp.MustCompile(`^model-[0-9]{5}-of-[0-9]{5}\.safetensors$`)

// Probe resolves only the portable architecture and tokenizer contract. It is
// used for forecasting an inherited compose without downloading model weights.
func (puller Puller) Probe(ctx context.Context, source string) (Architecture, error) {
	metadata, repository, requested, endpoint, token, client, err := puller.resolveHub(ctx, source)
	if err != nil {
		return Architecture{}, err
	}
	names, err := selectModelFiles(metadata)
	if err != nil {
		return Architecture{}, err
	}
	if err := os.MkdirAll(puller.Root, 0o755); err != nil {
		return Architecture{}, err
	}
	directory, err := os.MkdirTemp(puller.Root, ".waldo-probe-*")
	if err != nil {
		return Architecture{}, err
	}
	defer os.RemoveAll(directory)
	sizes := map[string]int64{}
	for _, sibling := range metadata.Siblings {
		sizes[sibling.Name] = sibling.Size
	}
	for _, filename := range []string{"config.json", "tokenizer_config.json", TokenizerArtifactFile} {
		if !slicesContains(names, filename) {
			if filename == TokenizerArtifactFile {
				continue
			}
			return Architecture{}, fmt.Errorf("Hugging Face model must contain %s", filename)
		}
		fileURL := endpoint + "/" + repository + "/resolve/" + metadata.SHA + "/" + escapePath(filename)
		if _, _, err := downloadFile(ctx, client, fileURL, token, filepath.Join(directory, filename), sizes[filename]); err != nil {
			return Architecture{}, fmt.Errorf("probe %s@%s %s: %w", repository, requested, filename, err)
		}
	}
	return loadHuggingFaceArchitecture(directory)
}

func (puller Puller) Pull(ctx context.Context, name, source string) (Inspection, error) {
	if err := ValidateName(name); err != nil {
		return Inspection{}, err
	}
	if puller.Root == "" {
		return Inspection{}, fmt.Errorf("model root is required")
	}
	destination := filepath.Join(puller.Root, name)
	if _, err := os.Stat(destination); err == nil {
		return Inspection{}, fmt.Errorf("model %q already exists", name)
	} else if !os.IsNotExist(err) {
		return Inspection{}, err
	}
	if err := os.MkdirAll(puller.Root, 0o755); err != nil {
		return Inspection{}, err
	}
	temporary, err := os.MkdirTemp(puller.Root, ".waldo-pull-*")
	if err != nil {
		return Inspection{}, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(temporary)
		}
	}()
	metadata, repository, requested, endpoint, token, client, err := puller.resolveHub(ctx, source)
	if err != nil {
		return Inspection{}, err
	}
	names, err := selectModelFiles(metadata)
	if err != nil {
		return Inspection{}, err
	}
	sourceDirectory := filepath.Join(temporary, ".source")
	if err := os.MkdirAll(sourceDirectory, 0o755); err != nil {
		return Inspection{}, err
	}
	var sourceArtifacts []OriginArtifact
	sizes := map[string]int64{}
	for _, sibling := range metadata.Siblings {
		sizes[sibling.Name] = sibling.Size
	}
	for _, filename := range names {
		puller.report(PullProgress{Phase: "download", File: filename, Message: "pulling " + filename})
		target := filepath.Join(sourceDirectory, filepath.FromSlash(filename))
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return Inspection{}, err
		}
		fileURL := endpoint + "/" + repository + "/resolve/" + metadata.SHA + "/" + escapePath(filename)
		digest, bytes, err := downloadFile(ctx, client, fileURL, token, target, sizes[filename])
		if err != nil {
			return Inspection{}, fmt.Errorf("download %s: %w", filename, err)
		}
		sourceArtifacts = append(sourceArtifacts, OriginArtifact{Role: sourceRole(filename), Path: filename, SHA256: digest, Bytes: bytes})
	}
	origin := OriginSource{Provider: "huggingface", Repository: metadata.ID, RequestedRevision: requested, Revision: metadata.SHA, URL: endpoint + "/" + metadata.ID, License: metadata.CardData.License}
	if err := puller.importSource(name, temporary, sourceDirectory, names, sourceArtifacts, origin); err != nil {
		return Inspection{}, err
	}
	committed = true
	puller.report(PullProgress{Phase: "complete", Message: fmt.Sprintf("pulled %s at %s", metadata.ID, metadata.SHA)})
	return Inspect(puller.Root, name)
}

// PullDirectory imports a Hugging Face package exported to a local directory.
// Its origin names the directory and pins every imported file by hash.
func (puller Puller) PullDirectory(name, directory string) (Inspection, error) {
	if err := ValidateName(name); err != nil {
		return Inspection{}, err
	}
	directory, err := filepath.Abs(directory)
	if err != nil {
		return Inspection{}, err
	}
	if _, err := os.Stat(filepath.Join(puller.Root, name)); err == nil {
		return Inspection{}, fmt.Errorf("model %q already exists", name)
	}
	entries, err := os.ReadDir(directory)
	if err != nil {
		return Inspection{}, err
	}
	var listing hubModel
	for _, entry := range entries {
		if entry.Type().IsRegular() {
			listing.Siblings = append(listing.Siblings, struct {
				Name string `json:"rfilename"`
				Size int64  `json:"size"`
			}{Name: entry.Name()})
		}
	}
	names, err := selectModelFiles(listing)
	if err != nil {
		return Inspection{}, err
	}
	if err := os.MkdirAll(puller.Root, 0o755); err != nil {
		return Inspection{}, err
	}
	temporary, err := os.MkdirTemp(puller.Root, ".waldo-pull-*")
	if err != nil {
		return Inspection{}, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(temporary)
		}
	}()
	sourceDirectory := filepath.Join(temporary, ".source")
	var sourceArtifacts []OriginArtifact
	revision := ""
	for _, filename := range names {
		target := filepath.Join(sourceDirectory, filename)
		if err := copyArtifactFile(filepath.Join(directory, filename), target); err != nil {
			return Inspection{}, err
		}
		digest, bytes, err := hashFile(target)
		if err != nil {
			return Inspection{}, err
		}
		if filename == "model.safetensors" {
			revision = digest
		}
		sourceArtifacts = append(sourceArtifacts, OriginArtifact{Role: sourceRole(filename), Path: filename, SHA256: digest, Bytes: bytes})
	}
	origin := OriginSource{Provider: "file", Repository: filepath.Base(directory), RequestedRevision: revision, Revision: revision, URL: "file://" + filepath.ToSlash(directory)}
	if err := puller.importSource(name, temporary, sourceDirectory, names, sourceArtifacts, origin); err != nil {
		return Inspection{}, err
	}
	committed = true
	puller.report(PullProgress{Phase: "complete", Message: fmt.Sprintf("pulled %s", origin.URL)})
	return Inspect(puller.Root, name)
}

// importSource validates a Hugging Face layout in sourceDirectory and
// publishes it as model name with an origin BOM naming source.
func (puller Puller) importSource(name, temporary, sourceDirectory string, names []string, sourceArtifacts []OriginArtifact, source OriginSource) error {
	if err := validateShardIndex(sourceDirectory, names); err != nil {
		return err
	}
	architecture, err := loadHuggingFaceArchitecture(sourceDirectory)
	if err != nil {
		return err
	}
	architectureHash, err := hashJSON(architecture)
	if err != nil {
		return err
	}
	artifactDirectory := filepath.Join(temporary, "origin", "artifacts")
	if err := os.MkdirAll(artifactDirectory, 0o755); err != nil {
		return err
	}
	var weightFiles []string
	for _, filename := range names {
		if strings.HasSuffix(filename, ".safetensors") {
			weightFiles = append(weightFiles, filepath.Join(sourceDirectory, filepath.FromSlash(filename)))
		}
	}
	sort.Strings(weightFiles)
	puller.report(PullProgress{Phase: "normalize", Message: fmt.Sprintf("normalizing %d Safetensors file(s)", len(weightFiles))})
	descriptors, err := modelweights.NormalizeHuggingFace(weightFiles, filepath.Join(artifactDirectory, "model.safetensors"))
	if err != nil {
		return fmt.Errorf("normalize Hugging Face weights: %w", err)
	}
	if err := validateDownloadedTensors(architecture, descriptors); err != nil {
		return err
	}
	artifacts, err := writeOriginRuntime(temporary, architecture, architectureHash)
	if err != nil {
		return err
	}
	// Keep the release's run BOMs with the origin, each verified against the hash it is named by.
	records := map[string]string{}
	var corpora []string
	for _, name := range names {
		match := provenanceFile.FindStringSubmatch(name)
		if match == nil {
			continue
		}
		// Run BOMs are pinned by the hash of their canonical JSON, as models pin their runs.
		path := filepath.Join(sourceDirectory, name)
		var runBOM RunBOM
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if json.Unmarshal(data, &runBOM) != nil {
			return fmt.Errorf("%s is not a run BOM", name)
		}
		if digest, err := hashJSON(runBOM); err != nil || digest != match[1] {
			return fmt.Errorf("%s does not hash to the run BOM it names", name)
		}
		if digest, err := hashJSON(runBOM.CorpusBOM); err != nil || digest != runBOM.CorpusBOMSHA256 {
			return fmt.Errorf("%s does not hash to the corpus BOM it records", name)
		}
		records[match[1]] = path
		corpora = append(corpora, runBOM.CorpusBOMSHA256)
	}
	provenance, err := copyProvenance(temporary, records)
	if err != nil {
		return err
	}
	artifacts = append(artifacts, provenance...)
	// Keep the release's training closure, so models built from it inherit its whole history.
	if data, err := os.ReadFile(filepath.Join(sourceDirectory, ClosureFile)); err == nil {
		var closure TrainingClosure
		if err := json.Unmarshal(data, &closure); err != nil || closure.Kind != "openwaldo-bom" || closure.Subject != "training-closure" || closure.Schema != 1 {
			return fmt.Errorf("%s is not a schema-1 training closure", ClosureFile)
		}
		// A closure that leaves out the release's own runs misstates its history.
		for _, digest := range corpora {
			if !slices.ContainsFunc(closure.Corpora, func(entry ClosureCorpus) bool { return entry.CorpusBOMSHA256 == digest }) {
				return fmt.Errorf("%s leaves out corpus %s that the release's own run BOMs trained on", ClosureFile, digest[:12])
			}
		}
		kept, err := writeClosure(temporary, closure)
		if err != nil {
			return err
		}
		artifacts = append(artifacts, kept...)
	} else if !os.IsNotExist(err) {
		return err
	}
	// Keep a distributed run's round record, the weights' link back to the records that produced them.
	if data, err := os.ReadFile(filepath.Join(sourceDirectory, RoundFile)); err == nil {
		var round struct {
			Kind    string `json:"kind"`
			Subject string `json:"subject"`
			Schema  int    `json:"schema"`
		}
		if err := json.Unmarshal(data, &round); err != nil || round.Kind != "openwaldo-bom" || round.Subject != "join-round" || round.Schema != 1 {
			return fmt.Errorf("%s is not a schema-1 join round record", RoundFile)
		}
		logical := "origin/" + RoundFile
		if err := copyArtifactFile(filepath.Join(sourceDirectory, RoundFile), filepath.Join(temporary, filepath.FromSlash(logical))); err != nil {
			return err
		}
		kept, err := inventoryOriginArtifacts(temporary, []string{logical})
		if err != nil {
			return err
		}
		for index := range kept {
			kept[index].Role = RoundRole
		}
		artifacts = append(artifacts, kept...)
	} else if !os.IsNotExist(err) {
		return err
	}
	// The release's BOM.json is kept; lineage it cannot keep is noted, never fatal.
	release, err := os.ReadFile(filepath.Join(sourceDirectory, "BOM.json"))
	if err == nil {
		if _, _, note := releaseLineage(release, sourceArtifacts); note != "" {
			puller.report(PullProgress{Phase: "lineage", Message: note})
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	origin := OriginBOM{
		Kind: "openwaldo-bom", Schema: 1, Subject: "model-origin",
		Source:             source,
		ArchitectureSHA256: architectureHash, SourceArtifacts: sourceArtifacts, Artifacts: artifacts,
	}
	if err := writeReleaseBOM(temporary, release); err != nil {
		return err
	}
	if err := os.RemoveAll(sourceDirectory); err != nil {
		return err
	}
	now := time.Now
	if puller.Now != nil {
		now = puller.Now
	}
	return publishOrigin(puller.Root, name, temporary, origin, architecture, now())
}

func (puller Puller) resolveHub(ctx context.Context, source string) (hubModel, string, string, string, string, *http.Client, error) {
	repository, requested, err := parseHuggingFaceSource(source)
	if err != nil {
		return hubModel{}, "", "", "", "", nil, err
	}
	client, endpoint, token := puller.hubClient()
	puller.report(PullProgress{Phase: "resolve", Message: fmt.Sprintf("resolving %s@%s", repository, requested)})
	metadataURL := endpoint + "/api/models/" + repository + "/revision/" + url.PathEscape(requested)
	var metadata hubModel
	if err := getJSON(ctx, client, metadataURL, token, &metadata); err != nil {
		return hubModel{}, "", "", "", "", nil, fmt.Errorf("resolve Hugging Face model %s@%s: %w", repository, requested, err)
	}
	if metadata.ID == "" || !huggingFaceCommit.MatchString(metadata.SHA) {
		return hubModel{}, "", "", "", "", nil, fmt.Errorf("Hugging Face returned incomplete model identity for %s@%s", repository, requested)
	}
	return metadata, repository, requested, endpoint, token, client, nil
}

func slicesContains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func parseHuggingFaceSource(source string) (string, string, error) {
	const prefix = "huggingface://"
	if !strings.HasPrefix(source, prefix) {
		return "", "", fmt.Errorf("model source must use huggingface://organization/repository[@revision]")
	}
	value := strings.TrimPrefix(source, prefix)
	revision := "main"
	if position := strings.LastIndex(value, "@"); position >= 0 {
		revision, value = value[position+1:], value[:position]
	}
	if !huggingFaceRepository.MatchString(value) || revision == "" || strings.ContainsAny(revision, "?#") {
		return "", "", fmt.Errorf("invalid Hugging Face model source %q", source)
	}
	return value, revision, nil
}

func selectModelFiles(metadata hubModel) ([]string, error) {
	available := map[string]bool{}
	for _, file := range metadata.Siblings {
		if file.Name != "" && !strings.HasPrefix(file.Name, ".") && filepath.ToSlash(filepath.Clean(filepath.FromSlash(file.Name))) == file.Name {
			available[file.Name] = true
		}
	}
	if !available["config.json"] || !available["tokenizer_config.json"] {
		return nil, fmt.Errorf("Hugging Face model must contain config.json and tokenizer_config.json")
	}
	var selected []string
	for _, name := range []string{"config.json", "tokenizer_config.json", "special_tokens_map.json", "tokenization_openwaldo.py", TokenizerArtifactFile, "README.md", "LICENSE", "LICENSE.md", "BOM.json"} {
		if available[name] {
			selected = append(selected, name)
		}
	}
	for _, name := range []string{ClosureFile, RoundFile} {
		if available[name] {
			selected = append(selected, name)
		}
	}
	// A WALDO release publishes its training run BOMs beside its weights.
	for name := range available {
		if provenanceFile.MatchString(name) {
			selected = append(selected, name)
		}
	}
	if available["model.safetensors"] {
		selected = append(selected, "model.safetensors")
	} else {
		weightCount := 0
		for name := range available {
			if huggingFaceShard.MatchString(name) {
				selected = append(selected, name)
				weightCount++
			}
		}
		if weightCount == 0 || !available["model.safetensors.index.json"] {
			return nil, fmt.Errorf("Hugging Face model must contain Safetensors weights")
		}
		selected = append(selected, "model.safetensors.index.json")
	}
	sort.Strings(selected)
	return selected, nil
}

// loadHuggingFaceArchitecture reads a package's config.json and tokenizer into a WALDO architecture.
func loadHuggingFaceArchitecture(directory string) (Architecture, error) {
	var config struct {
		ModelType        string  `json:"model_type"`
		HiddenSize       uint64  `json:"hidden_size"`
		IntermediateSize uint64  `json:"intermediate_size"`
		Layers           uint64  `json:"num_hidden_layers"`
		Heads            uint64  `json:"num_attention_heads"`
		KVHeads          uint64  `json:"num_key_value_heads"`
		Vocabulary       uint64  `json:"vocab_size"`
		Context          uint64  `json:"max_position_embeddings"`
		Tie              bool    `json:"tie_word_embeddings"`
		DType            string  `json:"torch_dtype"`
		DTypeNew         string  `json:"dtype"`
		RMS              float64 `json:"rms_norm_eps"`
		Rope             float64 `json:"rope_theta"`
		HiddenAct        string  `json:"hidden_act"`
		AttentionBias    bool    `json:"attention_bias"`
		MLPBias          bool    `json:"mlp_bias"`
		RopeScaling      any     `json:"rope_scaling"`
		BOS              int     `json:"bos_token_id"`
		EOS              int     `json:"eos_token_id"`
		PAD              int     `json:"pad_token_id"`
	}
	if err := readJSON(filepath.Join(directory, "config.json"), &config); err != nil {
		return Architecture{}, err
	}
	if config.ModelType != "llama" || config.RMS != 1e-5 || config.Rope != 0 && config.Rope != 10000 || config.HiddenAct != "" && config.HiddenAct != "silu" || config.AttentionBias || config.MLPBias || config.RopeScaling != nil || config.BOS != 1 || config.EOS != 2 || config.PAD != 0 {
		return Architecture{}, fmt.Errorf("download currently supports standard Llama configuration with SiLU, bias-free projections, rms_norm_eps 1e-5, rope_theta 10000, no rope_scaling, and tokenizer IDs pad=0, bos=1, eos=2")
	}
	if config.DType == "" {
		config.DType = config.DTypeNew
	}
	dtype := map[string]string{"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}[config.DType]
	if dtype == "" {
		return Architecture{}, fmt.Errorf("unsupported Hugging Face torch_dtype %q", config.DType)
	}
	tokenizer, err := downloadedTokenizer(directory, config.Vocabulary)
	if err != nil {
		return Architecture{}, err
	}
	architecture := Architecture{Family: "decoder-transformer", ContextTokens: config.Context, VocabularySize: config.Vocabulary, HiddenSize: config.HiddenSize, IntermediateSize: config.IntermediateSize, Layers: config.Layers, AttentionHeads: config.Heads, KeyValueHeads: config.KVHeads, TieEmbeddings: config.Tie, ParameterDType: dtype, Tokenizer: tokenizer}
	if err := architecture.Validate(); err != nil {
		return Architecture{}, err
	}
	return architecture, nil
}

// downloadedTokenizer is the package's tokenizer: the trained artifact WALDO exports beside
// a trained-tokenizer model, or the byte tokenizer.
func downloadedTokenizer(directory string, vocabulary uint64) (Tokenizer, error) {
	var config struct {
		TokenizerClass string `json:"tokenizer_class"`
		BOS            string `json:"bos_token"`
		EOS            string `json:"eos_token"`
		PAD            string `json:"pad_token"`
	}
	if err := readJSON(filepath.Join(directory, "tokenizer_config.json"), &config); err != nil {
		return Tokenizer{}, err
	}
	byteTokenizer := Tokenizer{Name: "byte", Revision: "builtin-byte-schema-1"}
	var artifact struct {
		Kind   string `json:"kind"`
		Schema int    `json:"schema"`
		training.TokenizerSpec
	}
	if err := readJSON(filepath.Join(directory, TokenizerArtifactFile), &artifact); err == nil && artifact.Kind == "waldo-tokenizer" && waldotokenizer.IsTrainedName(artifact.Name) {
		if config.TokenizerClass != HuggingFaceTokenizerClass || config.BOS != "<bos>" || config.EOS != "<eos>" || config.PAD != "<pad>" {
			return Tokenizer{}, fmt.Errorf("download of trained tokenizer %s requires %s with <bos>, <eos>, and <pad>", artifact.Name, HuggingFaceTokenizerClass)
		}
		if artifact.Schema != 1 || artifact.VocabularySize != int(vocabulary) {
			return Tokenizer{}, fmt.Errorf("%s does not match the model's vocabulary_size %d", TokenizerArtifactFile, vocabulary)
		}
		spec, _, err := training.ResolveTokenizerSpec(artifact.TokenizerSpec)
		if err != nil {
			return Tokenizer{}, fmt.Errorf("%s: %w", TokenizerArtifactFile, err)
		}
		return Tokenizer{Name: spec.Name, Revision: spec.Revision, Artifact: spec.Artifact}, nil
	} else if err != nil && !os.IsNotExist(err) {
		return Tokenizer{}, fmt.Errorf("read %s: %w", TokenizerArtifactFile, err)
	}
	// Earlier exports used LegacyHuggingFaceTokenizerClass, which remains an
	// alias of the same byte-tokenizer implementation in exported modules.
	if (config.TokenizerClass != HuggingFaceTokenizerClass && config.TokenizerClass != LegacyHuggingFaceTokenizerClass) || config.BOS != "<bos>" || config.EOS != "<eos>" || config.PAD != "<pad>" || vocabulary != 259 {
		return Tokenizer{}, fmt.Errorf("download currently supports %s (or legacy %s) with vocabulary_size 259 or a trained tokenizer in %s; this model uses %q with vocabulary_size %d", HuggingFaceTokenizerClass, LegacyHuggingFaceTokenizerClass, TokenizerArtifactFile, config.TokenizerClass, vocabulary)
	}
	return byteTokenizer, nil
}

func validateDownloadedTensors(architecture Architecture, actual map[string]modelweights.Descriptor) error {
	expected := expectedTensorShapes(architecture)
	if len(actual) != len(expected) {
		return fmt.Errorf("pulled model has %d tensors; WALDO architecture requires %d", len(actual), len(expected))
	}
	dtype := map[string]string{"float32": "F32", "float16": "F16", "bfloat16": "BF16"}[architecture.ParameterDType]
	for name, shape := range expected {
		descriptor, ok := actual[name]
		if !ok {
			return fmt.Errorf("pulled model is missing tensor %s", name)
		}
		if descriptor.DType != dtype || !equalShape(descriptor.Shape, shape) {
			return fmt.Errorf("downloaded tensor %s is %s %v; expected %s %v", name, descriptor.DType, descriptor.Shape, dtype, shape)
		}
	}
	return nil
}

// ValidateCheckpointWeights verifies that checkpoint tensors can be loaded by
// the immutable model architecture. Checkpoints preserve FP32 master weights,
// while MLX checkpoints may retain another supported floating-point dtype.
func ValidateCheckpointWeights(path string, architecture Architecture) error {
	actual, _, err := modelweights.Inspect(path)
	if err != nil {
		return err
	}
	expected := expectedTensorShapes(architecture)
	if len(actual) != len(expected) {
		return fmt.Errorf("checkpoint has %d tensors; model architecture requires %d", len(actual), len(expected))
	}
	for name, shape := range expected {
		descriptor, ok := actual[name]
		if !ok {
			return fmt.Errorf("checkpoint is missing tensor %s", name)
		}
		if descriptor.DType != "F32" && descriptor.DType != "F16" && descriptor.DType != "BF16" {
			return fmt.Errorf("checkpoint tensor %s has unsupported dtype %s", name, descriptor.DType)
		}
		if !equalShape(descriptor.Shape, shape) {
			return fmt.Errorf("checkpoint tensor %s has shape %v; model architecture requires %v", name, descriptor.Shape, shape)
		}
	}
	return nil
}

func expectedTensorShapes(a Architecture) map[string][]uint64 {
	kv := a.HiddenSize / a.AttentionHeads * a.KeyValueHeads
	result := map[string][]uint64{"embedding.weight": {a.VocabularySize, a.HiddenSize}, "norm.weight": {a.HiddenSize}}
	if !a.TieEmbeddings {
		result["output.weight"] = []uint64{a.VocabularySize, a.HiddenSize}
	}
	for layer := uint64(0); layer < a.Layers; layer++ {
		prefix := fmt.Sprintf("layers.%d.", layer)
		result[prefix+"attention.q_proj.weight"] = []uint64{a.HiddenSize, a.HiddenSize}
		result[prefix+"attention.k_proj.weight"] = []uint64{kv, a.HiddenSize}
		result[prefix+"attention.v_proj.weight"] = []uint64{kv, a.HiddenSize}
		result[prefix+"attention.o_proj.weight"] = []uint64{a.HiddenSize, a.HiddenSize}
		result[prefix+"attention_norm.weight"] = []uint64{a.HiddenSize}
		result[prefix+"feed_forward.gate.weight"] = []uint64{a.IntermediateSize, a.HiddenSize}
		result[prefix+"feed_forward.up.weight"] = []uint64{a.IntermediateSize, a.HiddenSize}
		result[prefix+"feed_forward.down.weight"] = []uint64{a.HiddenSize, a.IntermediateSize}
		result[prefix+"ffn_norm.weight"] = []uint64{a.HiddenSize}
	}
	return result
}

func equalShape(left, right []uint64) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func inventoryOriginArtifacts(root string, paths []string) ([]OriginArtifact, error) {
	var result []OriginArtifact
	for _, logical := range paths {
		digest, bytes, err := hashFile(filepath.Join(root, filepath.FromSlash(logical)))
		if err != nil {
			return nil, err
		}
		result = append(result, OriginArtifact{Role: artifactRole(logical), Path: logical, SHA256: digest, Bytes: bytes})
	}
	return result, nil
}

func getJSON(ctx context.Context, client *http.Client, target, token string, value any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return err
	}
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return fmt.Errorf("HTTP %s: %s", response.Status, strings.TrimSpace(string(data)))
	}
	return json.NewDecoder(response.Body).Decode(value)
}

func downloadFile(ctx context.Context, client *http.Client, target, token, path string, expected int64) (string, int64, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return "", 0, err
	}
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response, err := client.Do(request)
	if err != nil {
		return "", 0, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return "", 0, fmt.Errorf("HTTP %s: %s", response.Status, strings.TrimSpace(string(data)))
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return "", 0, err
	}
	hasher := sha256.New()
	bytes, copyErr := io.Copy(io.MultiWriter(file, hasher), response.Body)
	closeErr := file.Close()
	if copyErr != nil {
		_ = os.Remove(path)
		return "", 0, copyErr
	}
	if closeErr != nil {
		return "", 0, closeErr
	}
	if expected > 0 && bytes != expected {
		_ = os.Remove(path)
		return "", 0, fmt.Errorf("received %d bytes; repository metadata declares %d", bytes, expected)
	}
	return hex.EncodeToString(hasher.Sum(nil)), bytes, nil
}

func validateShardIndex(directory string, selected []string) error {
	indexPath := filepath.Join(directory, "model.safetensors.index.json")
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return err
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := readJSON(indexPath, &index); err != nil {
		return fmt.Errorf("read Hugging Face Safetensors index: %w", err)
	}
	if len(index.WeightMap) == 0 {
		return fmt.Errorf("Hugging Face Safetensors index has an empty weight_map")
	}
	available := map[string]bool{}
	for _, name := range selected {
		if strings.HasSuffix(name, ".safetensors") {
			available[name] = true
		}
	}
	used := map[string]bool{}
	for tensor, shard := range index.WeightMap {
		if tensor == "" || !available[shard] {
			return fmt.Errorf("Hugging Face Safetensors index maps tensor %q to unavailable shard %q", tensor, shard)
		}
		used[shard] = true
	}
	if len(used) != len(available) {
		return fmt.Errorf("Hugging Face Safetensors index does not reference every selected shard")
	}
	return nil
}

func hashFile(path string) (string, int64, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer file.Close()
	hasher := sha256.New()
	bytes, err := io.Copy(hasher, file)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(hasher.Sum(nil)), bytes, nil
}
func escapePath(value string) string {
	parts := strings.Split(value, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}
func sourceRole(name string) string {
	switch {
	case strings.HasSuffix(name, ".safetensors"):
		return "source-weights"
	case name == "config.json":
		return "source-configuration"
	case strings.Contains(name, "tokenizer") || name == "special_tokens_map.json":
		return "source-tokenizer"
	case name == "BOM.json":
		return "source-bom"
	case provenanceFile.MatchString(name):
		return "source-" + ProvenanceRole
	case name == ClosureFile:
		return "source-" + ClosureRole
	case strings.HasPrefix(name, "LICENSE"):
		return "source-license"
	default:
		return "source-documentation"
	}
}
func (puller Puller) report(progress PullProgress) {
	if puller.Progress != nil {
		puller.Progress(progress)
	}
}

func huggingFaceToken() string {
	if token := strings.TrimSpace(os.Getenv("HF_TOKEN")); token != "" {
		return token
	}
	path := os.Getenv("HF_TOKEN_PATH")
	if path == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		base := os.Getenv("HF_HOME")
		if base == "" {
			base = filepath.Join(home, ".cache", "huggingface")
		}
		path = filepath.Join(base, "token")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// writeOriginRuntime writes the runtime configuration and tokenizer
// beside origin/artifacts/model.safetensors and inventories all three.
func writeOriginRuntime(temporary string, architecture Architecture, architectureHash string) ([]OriginArtifact, error) {
	artifactDirectory := filepath.Join(temporary, "origin", "artifacts")
	if err := writeJSONAtomic(filepath.Join(artifactDirectory, "config.json"), map[string]any{
		"kind": "waldo-downloaded-model-config", "schema": 1,
		"architecture_sha256": architectureHash, "architecture": architecture,
		"backend": map[string]any{"name": "portable-safetensors", "revision": "huggingface-schema-1"},
	}); err != nil {
		return nil, err
	}
	var tokenizer any = map[string]any{
		"kind": "waldo-byte-tokenizer", "schema": 1, "name": "byte", "revision": "builtin-byte-schema-1",
		"pad_id": 0, "bos_id": 1, "eos_id": 2, "byte_offset": 3, "vocabulary_size": 259,
	}
	if trained := architecture.Tokenizer.Artifact; trained != nil {
		tokenizer = struct {
			Kind   string `json:"kind"`
			Schema int    `json:"schema"`
			training.TokenizerSpec
		}{"waldo-tokenizer", 1, training.TokenizerSpec{Name: trained.Name, Revision: trained.Revision, VocabularySize: trained.VocabularySize, PadID: trained.PadID, BOSID: trained.BOSID, EOSID: trained.EOSID, Artifact: trained}}
	}
	if err := writeJSONAtomic(filepath.Join(artifactDirectory, "tokenizer.json"), tokenizer); err != nil {
		return nil, err
	}
	return inventoryOriginArtifacts(temporary, []string{"origin/artifacts/model.safetensors", "origin/artifacts/config.json", "origin/artifacts/tokenizer.json"})
}

// publishOrigin writes the origin BOM, plan, record, and model BOM into
// temporary and renames it into place as model name under root.
func publishOrigin(root, name, temporary string, origin OriginBOM, architecture Architecture, now time.Time) error {
	originHash, err := hashJSON(origin)
	if err != nil {
		return err
	}
	plan, err := composePlan(name, Compose{Architecture: architecture})
	if err != nil {
		return err
	}
	plan.OriginBOMSHA256 = originHash
	planHash, err := hashJSON(plan)
	if err != nil {
		return err
	}
	created := formatTime(now)
	record := ModelRecord{
		Kind: "waldo-model", Schema: 1, ID: planHash, Name: name, PlanSHA256: planHash,
		ArchitectureSHA256: origin.ArchitectureSHA256, Architecture: architecture, Forecast: plan.Forecast,
		Created: created, Updated: created, Runs: []RunPin{}, OriginBOMSHA256: originHash, OriginArtifacts: origin.Artifacts,
	}
	if err := writeJSONAtomic(filepath.Join(temporary, "ORIGIN-BOM.json"), origin); err != nil {
		return err
	}
	if err := writeJSONAtomic(filepath.Join(temporary, "PLAN.json"), plan); err != nil {
		return err
	}
	if err := writeJSONAtomic(filepath.Join(temporary, "MODEL.json"), record); err != nil {
		return err
	}
	if err := writeJSONAtomic(filepath.Join(temporary, "MODEL-BOM.json"), modelBOM(record)); err != nil {
		return err
	}
	if err := os.Rename(temporary, filepath.Join(root, name)); err != nil {
		return fmt.Errorf("publish model %q: %w", name, err)
	}
	return nil
}

// writeReleaseBOM keeps a release's BOM.json in a model directory; nothing when data is nil.
func writeReleaseBOM(directory string, data []byte) error {
	if data == nil {
		return nil
	}
	path := filepath.Join(directory, filepath.FromSlash(releaseBOMPath))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

var pinnedFileName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

// FetchPinned copies exactly the named files of a file:// release into
// directory, and refuses the release if any file's SHA-256 differs from its
// pin: the result is the release the pins name, or nothing.
func (puller Puller) FetchPinned(ctx context.Context, source string, files map[string]string, directory string) error {
	names := make([]string, 0, len(files))
	for name, digest := range files {
		if !pinnedFileName.MatchString(name) || len(digest) != 64 {
			return fmt.Errorf("pinned file %q must be a plain name with a SHA-256", name)
		}
		names = append(names, name)
	}
	sort.Strings(names)
	partial := directory + ".partial"
	_ = os.RemoveAll(partial)
	if err := os.MkdirAll(partial, 0o755); err != nil {
		return err
	}
	defer os.RemoveAll(partial)
	if !strings.HasPrefix(source, "file://") {
		return fmt.Errorf("%s is not a file:// release", source)
	}
	for _, name := range names {
		digest, _, err := copyHashed(filepath.Join(strings.TrimPrefix(source, "file://"), name), filepath.Join(partial, name))
		if err != nil {
			return fmt.Errorf("fetch %s: %w", name, err)
		}
		if digest != files[name] {
			return fmt.Errorf("%s in %s does not match its pinned SHA-256", name, source)
		}
	}
	if err := os.RemoveAll(directory); err != nil {
		return err
	}
	return os.Rename(partial, directory)
}

// hubClient is the HTTP client, endpoint, and token Hugging Face requests use.
func (puller Puller) hubClient() (*http.Client, string, string) {
	client := puller.Client
	if client == nil {
		client = &http.Client{}
	}
	endpoint := strings.TrimRight(puller.Endpoint, "/")
	if endpoint == "" {
		endpoint = strings.TrimRight(os.Getenv("HF_ENDPOINT"), "/")
		if endpoint == "" {
			endpoint = "https://huggingface.co"
		}
	}
	token := puller.Token
	if token == "" {
		token = huggingFaceToken()
	}
	return client, endpoint, token
}

// copyHashed copies source to a new file at path and returns its SHA-256.
func copyHashed(source, path string) (string, int64, error) {
	input, err := os.Open(source)
	if err != nil {
		return "", 0, err
	}
	defer input.Close()
	output, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return "", 0, err
	}
	hasher := sha256.New()
	bytes, err := io.Copy(io.MultiWriter(output, hasher), input)
	if closeErr := output.Close(); err == nil {
		err = closeErr
	}
	return hex.EncodeToString(hasher.Sum(nil)), bytes, err
}
