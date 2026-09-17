// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package inference

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/openwaldo/waldo/internal/mlxruntime"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

//go:embed workers/mlx.py
var mlxChatWorker []byte

type workerFrame struct {
	Kind         string `json:"kind"`
	Schema       int    `json:"schema"`
	Data         string `json:"data,omitempty"`
	TokenID      *int   `json:"token_id,omitempty"`
	Tokens       int    `json:"tokens,omitempty"`
	FinishReason string `json:"finish_reason,omitempty"`
	DurationMS   int64  `json:"duration_ms,omitempty"`
	Context      int    `json:"context_tokens,omitempty"`
	Error        string `json:"error,omitempty"`
}

type workerRequest struct {
	Kind         string  `json:"kind"`
	Schema       int     `json:"schema"`
	Prompt       string  `json:"prompt"`
	TokenIDs     []int   `json:"token_ids,omitempty"`
	MaxTokens    int     `json:"max_tokens"`
	Temperature  float64 `json:"temperature"`
	TopP         float64 `json:"top_p"`
	Seed         *uint64 `json:"seed,omitempty"`
	StopTokenIDs [][]int `json:"stop_token_ids,omitempty"`
}

type MLXSession struct {
	command *exec.Cmd
	stdin   io.WriteCloser
	encoder *json.Encoder
	scanner *bufio.Scanner
	stderr  bytes.Buffer
	mu      sync.Mutex
	closed  bool
	codec   training.TokenCodec
	spec    training.TokenizerSpec
}

func Open(ctx context.Context, inspection model.Inspection) (Opened, error) {
	if inspection.Model.Architecture.Transformers != nil {
		return openTransformers(ctx, inspection)
	}
	artifacts, err := ResolveArtifacts(inspection)
	if err != nil {
		return Opened{}, err
	}
	switch artifacts.Backend.Name {
	case training.BackendMLX, "portable-safetensors":
		return openMLX(ctx, inspection, artifacts)
	case training.BackendPyTorch, training.BackendTorchTitan:
		return openPyTorch(ctx, inspection, artifacts)
	default:
		return Opened{}, fmt.Errorf("model %q current weights require unsupported chat backend %s@%s", artifacts.Model, artifacts.Backend.Name, artifacts.Backend.Revision)
	}
}

func openMLX(ctx context.Context, inspection model.Inspection, artifacts Artifacts) (Opened, error) {
	architecture, err := json.Marshal(inspection.Model.Architecture)
	if err != nil {
		return Opened{}, err
	}
	selection, err := training.NewMLXResolver().Resolve(ctx, training.ResolveRequest{
		ArchitectureSHA256: inspection.Model.ArchitectureSHA256,
		Architecture:       architecture,
		Objectives:         []string{"causal-language-modeling"},
	})
	if err != nil {
		return Opened{}, fmt.Errorf("open MLX chat runtime: %w", err)
	}
	backend, ok := selection.Backend.(training.MLX)
	if !ok || backend.Python == "" {
		return Opened{}, fmt.Errorf("resolved MLX chat runtime is invalid")
	}
	session, contextTokens, err := startMLXSession(ctx, backend.Python, artifacts)
	if err != nil {
		return Opened{}, err
	}
	return Opened{Description: Description{
		Model: artifacts.Model, SourceType: artifacts.SourceType, SourceID: artifacts.SourceID, RunID: artifacts.RunID, Backend: training.BackendMLX,
		ContextTokens: contextTokens,
	}, Session: session}, nil
}

func startMLXSession(ctx context.Context, python string, artifacts Artifacts) (*MLXSession, int, error) {
	spec, codec, err := loadTokenizer(artifacts.Tokenizer)
	if err != nil {
		return nil, 0, err
	}
	command := exec.CommandContext(ctx, python, "-c", mlxruntime.WithModel(mlxChatWorker), artifacts.Weights, artifacts.Configuration, artifacts.Tokenizer)
	stdin, err := command.StdinPipe()
	if err != nil {
		return nil, 0, err
	}
	stdout, err := command.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, 0, err
	}
	session := &MLXSession{command: command, stdin: stdin, encoder: json.NewEncoder(stdin), scanner: bufio.NewScanner(stdout), codec: codec, spec: spec}
	session.scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	command.Stderr = &session.stderr
	if err := command.Start(); err != nil {
		_ = stdin.Close()
		return nil, 0, fmt.Errorf("start MLX chat worker with %s: %w", python, err)
	}
	frame, err := session.nextFrame()
	if err != nil {
		_ = session.Close()
		return nil, 0, fmt.Errorf("initialize MLX chat worker: %w", err)
	}
	if frame.Kind != "ready" || frame.Context < 1 {
		_ = session.Close()
		return nil, 0, fmt.Errorf("MLX chat worker returned invalid readiness frame")
	}
	if artifacts.ContextTokens != frame.Context {
		_ = session.Close()
		return nil, 0, fmt.Errorf("MLX chat worker context is %d tokens, model requires %d", frame.Context, artifacts.ContextTokens)
	}
	return session, frame.Context, nil
}

func (session *MLXSession) Generate(ctx context.Context, prompt string, options Options, emit func(Token) error) (Result, error) {
	if err := options.Validate(); err != nil {
		return Result{}, err
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return Result{}, fmt.Errorf("MLX chat session is closed")
	}
	request := workerRequest{Kind: "generate", Schema: 1, Prompt: prompt, MaxTokens: options.MaxTokens, Temperature: options.Temperature, TopP: options.TopP, Seed: options.Seed}
	for _, stop := range options.Stop {
		tokens, err := session.codec.EncodeChecked(stop)
		if err != nil {
			return Result{}, err
		}
		request.StopTokenIDs = append(request.StopTokenIDs, tokens)
	}
	if session.spec.Name != "byte" {
		tokens, err := session.codec.EncodeChecked(prompt)
		if err != nil {
			return Result{}, err
		}
		request.TokenIDs = tokens
		request.Prompt = ""
	}
	if err := session.encoder.Encode(request); err != nil {
		return Result{}, fmt.Errorf("send MLX chat request: %w", err)
	}
	var generated []byte
	var emitErr error
	for {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		frame, err := session.nextFrame()
		if err != nil {
			return Result{}, err
		}
		switch frame.Kind {
		case "token":
			var data []byte
			if frame.TokenID != nil {
				decoded, err := session.codec.DecodeChecked([]int{*frame.TokenID})
				if err != nil {
					return Result{}, err
				}
				data = []byte(decoded)
			} else {
				data, err = base64.StdEncoding.DecodeString(frame.Data)
				if err != nil {
					return Result{}, fmt.Errorf("decode MLX chat token: %w", err)
				}
			}
			generated = append(generated, data...)
			if emit != nil && emitErr == nil {
				emitErr = emit(Token{Bytes: append([]byte(nil), data...)})
			}
		case "complete":
			if emitErr != nil {
				return Result{}, emitErr
			}
			return Result{
				Text: strings.ToValidUTF8(string(generated), "�"), Tokens: frame.Tokens,
				FinishReason: frame.FinishReason, Duration: time.Duration(frame.DurationMS) * time.Millisecond,
				DurationMS: frame.DurationMS,
			}, nil
		case "error":
			return Result{}, errors.New(frame.Error)
		default:
			return Result{}, fmt.Errorf("MLX chat worker returned unexpected frame %q", frame.Kind)
		}
	}
}

func (session *MLXSession) nextFrame() (workerFrame, error) {
	if !session.scanner.Scan() {
		if err := session.scanner.Err(); err != nil {
			return workerFrame{}, err
		}
		detail := strings.TrimSpace(session.stderr.String())
		if detail != "" {
			return workerFrame{}, fmt.Errorf("MLX chat worker exited: %s", detail)
		}
		return workerFrame{}, fmt.Errorf("MLX chat worker exited unexpectedly")
	}
	var frame workerFrame
	if err := json.Unmarshal(session.scanner.Bytes(), &frame); err != nil {
		return workerFrame{}, fmt.Errorf("decode MLX chat frame: %w", err)
	}
	if frame.Schema != 1 {
		return workerFrame{}, fmt.Errorf("unsupported MLX chat protocol schema %d", frame.Schema)
	}
	return frame, nil
}

func (session *MLXSession) Close() error {
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return nil
	}
	session.closed = true
	closeErr := session.stdin.Close()
	waitErr := session.command.Wait()
	if waitErr != nil {
		detail := strings.TrimSpace(session.stderr.String())
		if detail != "" {
			waitErr = fmt.Errorf("%w: %s", waitErr, detail)
		}
	}
	return errors.Join(closeErr, waitErr)
}
