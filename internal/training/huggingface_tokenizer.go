// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"sync"
	"time"
)

// HuggingFaceTokenizer pins a local, data-only fast tokenizer snapshot. Source
// is provenance, never a command or an instruction to download anything.
type HuggingFaceTokenizer struct {
	Source string            `json:"source" yaml:"source"`
	Files  map[string]string `json:"files" yaml:"files"`
	PadID  int               `json:"pad_id" yaml:"pad_id"`
	BOSID  *int              `json:"bos_id,omitempty" yaml:"bos_id,omitempty"`
	EOSID  int               `json:"eos_id" yaml:"eos_id"`
}

func (pin HuggingFaceTokenizer) Validate(revision string, vocabulary uint64) error {
	if !regexp.MustCompile(`^[a-f0-9]{40}$`).MatchString(revision) || !regexp.MustCompile(`^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$`).MatchString(pin.Source) {
		return fmt.Errorf("Hugging Face tokenizer requires owner/repository source and an immutable 40-character commit revision")
	}
	allowed := map[string]bool{"tokenizer.json": true, "tokenizer_config.json": true, "special_tokens_map.json": true, "chat_template.jinja": true}
	for name, hash := range pin.Files {
		if !allowed[name] || !regexp.MustCompile(`^[a-f0-9]{64}$`).MatchString(hash) {
			return fmt.Errorf("invalid tokenizer file pin %q", name)
		}
	}
	if pin.Files["tokenizer.json"] == "" || pin.Files["tokenizer_config.json"] == "" {
		return fmt.Errorf("pin tokenizer.json and tokenizer_config.json")
	}
	for _, id := range []int{pin.PadID, pin.EOSID} {
		if id < 0 || uint64(id) >= vocabulary {
			return fmt.Errorf("tokenizer special ID is outside model vocabulary")
		}
	}
	if pin.BOSID != nil && (*pin.BOSID < 0 || uint64(*pin.BOSID) >= vocabulary) {
		return fmt.Errorf("tokenizer BOS ID is outside model vocabulary")
	}
	return nil
}

func (pin *HuggingFaceTokenizer) Spec(revision string, vocabulary uint64) TokenizerSpec {
	bos := -1
	if pin.BOSID != nil {
		bos = *pin.BOSID
	}
	return TokenizerSpec{Name: "huggingface", Revision: revision, VocabularySize: int(vocabulary), PadID: pin.PadID, BOSID: bos, EOSID: pin.EOSID, HuggingFace: pin}
}

type huggingFaceCodec struct {
	mu      sync.Mutex
	command *exec.Cmd
	input   io.WriteCloser
	encoder *json.Encoder
	decoder *json.Decoder
	cancel  context.CancelFunc
	stderr  cappedBuffer
}

// OpenHuggingFaceTokenizer verifies the exact Transformers wheel and tokenizer
// files before exposing encoding to preflight or the training stream.
func OpenHuggingFaceTokenizer(ctx context.Context, spec TokenizerSpec, model *TransformersModel) (TokenCodec, func(), error) {
	if spec.HuggingFace == nil || model == nil {
		return nil, nil, fmt.Errorf("missing Hugging Face tokenizer/model specification")
	}
	if err := spec.HuggingFace.Validate(spec.Revision, uint64(spec.VocabularySize)); err != nil {
		return nil, nil, err
	}
	python, wheel, directory := os.Getenv("WALDO_TRANSFORMERS_PYTHON"), os.Getenv("WALDO_TRANSFORMERS_WHEEL"), os.Getenv("WALDO_HF_TOKENIZER_DIR")
	if python == "" || wheel == "" || directory == "" {
		return nil, nil, fmt.Errorf("set WALDO_TRANSFORMERS_PYTHON, WALDO_TRANSFORMERS_WHEEL, and WALDO_HF_TOKENIZER_DIR for the pinned local tokenizer")
	}
	payload, err := json.Marshal(map[string]any{"tokenizer": spec, "package": model.Package})
	if err != nil {
		return nil, nil, err
	}
	processCtx, cancel := context.WithCancel(ctx)
	startup := time.AfterFunc(60*time.Second, cancel)
	defer startup.Stop()
	codec := &huggingFaceCodec{cancel: cancel}
	codec.command = exec.CommandContext(processCtx, python, "-c", transformersWorker, "--tokenizer", wheel, directory, string(payload))
	codec.command.Stderr = &codec.stderr
	codec.input, err = codec.command.StdinPipe()
	if err != nil {
		cancel()
		return nil, nil, err
	}
	output, err := codec.command.StdoutPipe()
	if err != nil {
		codec.input.Close()
		cancel()
		return nil, nil, err
	}
	codec.encoder, codec.decoder = json.NewEncoder(codec.input), json.NewDecoder(output)
	if err := codec.command.Start(); err != nil {
		codec.input.Close()
		output.Close()
		cancel()
		return nil, nil, err
	}
	close := func() { cancel(); codec.input.Close(); _ = codec.command.Wait() }
	var ready struct {
		Ready bool   `json:"ready"`
		Error string `json:"error"`
	}
	if err := codec.decoder.Decode(&ready); err != nil || !ready.Ready {
		close()
		return nil, nil, fmt.Errorf("initialize pinned tokenizer: %v %s%s", err, ready.Error, workerStderr(codec.stderr.String()))
	}
	return codec, close, nil
}

func (codec *huggingFaceCodec) EncodeChecked(text string) ([]int, error) {
	codec.mu.Lock()
	defer codec.mu.Unlock()
	if err := codec.encoder.Encode(map[string]string{"text": text}); err != nil {
		return nil, fmt.Errorf("tokenizer request: %w", err)
	}
	var response struct {
		Tokens []int  `json:"tokens"`
		Error  string `json:"error"`
	}
	if err := codec.decoder.Decode(&response); err != nil {
		return nil, fmt.Errorf("tokenizer response: %w", err)
	}
	if response.Error != "" {
		return nil, fmt.Errorf("tokenizer: %s", response.Error)
	}
	if response.Tokens == nil {
		return nil, fmt.Errorf("tokenizer response lacks tokens")
	}
	return response.Tokens, nil
}

func (codec *huggingFaceCodec) CountChecked(text string) (int, error) {
	tokens, err := codec.EncodeChecked(text)
	return len(tokens), err
}

func (*huggingFaceCodec) DecodeChecked([]int) (string, error) {
	return "", fmt.Errorf("Hugging Face decoding is not supported by WALDO inference yet")
}
