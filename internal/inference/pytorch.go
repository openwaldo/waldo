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

	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

//go:embed workers/pytorch.py
var pyTorchChatWorker []byte

type PyTorchSession struct {
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

func openPyTorch(ctx context.Context, inspection model.Inspection, artifacts Artifacts) (Opened, error) {
	architecture, err := json.Marshal(inspection.Model.Architecture)
	if err != nil {
		return Opened{}, err
	}
	selection, err := training.NewPyTorchResolver().Resolve(ctx, training.ResolveRequest{
		ArchitectureSHA256: inspection.Model.ArchitectureSHA256,
		Architecture:       architecture,
		Objectives:         []string{"causal-language-modeling"},
	})
	if err != nil {
		return Opened{}, fmt.Errorf("open PyTorch chat runtime: %w", err)
	}
	backend, ok := selection.Backend.(training.PyTorch)
	if !ok || backend.Python == "" || backend.Device == "" {
		return Opened{}, fmt.Errorf("resolved PyTorch chat runtime is invalid")
	}
	session, contextTokens, err := startPyTorchSession(ctx, backend.Python, backend.Device, artifacts)
	if err != nil {
		return Opened{}, err
	}
	return Opened{Description: Description{
		Model: artifacts.Model, SourceType: artifacts.SourceType, SourceID: artifacts.SourceID, RunID: artifacts.RunID,
		Backend: training.BackendPyTorch, ContextTokens: contextTokens,
	}, Session: session}, nil
}

func startPyTorchSession(ctx context.Context, python, device string, artifacts Artifacts) (*PyTorchSession, int, error) {
	spec, codec, err := loadTokenizer(artifacts.Tokenizer)
	if err != nil {
		return nil, 0, err
	}
	command := exec.CommandContext(ctx, python, "-c", string(pyTorchChatWorker), artifacts.Weights, artifacts.Configuration, artifacts.Tokenizer, device)
	stdin, err := command.StdinPipe()
	if err != nil {
		return nil, 0, err
	}
	stdout, err := command.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, 0, err
	}
	session := &PyTorchSession{command: command, stdin: stdin, encoder: json.NewEncoder(stdin), scanner: bufio.NewScanner(stdout), codec: codec, spec: spec}
	session.scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	command.Stderr = &session.stderr
	if err := command.Start(); err != nil {
		_ = stdin.Close()
		return nil, 0, fmt.Errorf("start PyTorch chat worker with %s: %w", python, err)
	}
	frame, err := session.nextFrame()
	if err != nil {
		_ = session.Close()
		return nil, 0, fmt.Errorf("initialize PyTorch chat worker: %w", err)
	}
	if frame.Kind != "ready" || frame.Context < 1 {
		_ = session.Close()
		return nil, 0, fmt.Errorf("PyTorch chat worker returned invalid readiness frame")
	}
	if artifacts.ContextTokens != frame.Context {
		_ = session.Close()
		return nil, 0, fmt.Errorf("PyTorch chat worker context is %d tokens, model requires %d", frame.Context, artifacts.ContextTokens)
	}
	return session, frame.Context, nil
}

func (session *PyTorchSession) Generate(ctx context.Context, prompt string, options Options, emit func(Token) error) (Result, error) {
	if err := options.Validate(); err != nil {
		return Result{}, err
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return Result{}, fmt.Errorf("PyTorch chat session is closed")
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
		return Result{}, fmt.Errorf("send PyTorch chat request: %w", err)
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
					return Result{}, fmt.Errorf("decode PyTorch chat token: %w", err)
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
			return Result{Text: strings.ToValidUTF8(string(generated), "�"), Tokens: frame.Tokens, FinishReason: frame.FinishReason, Duration: time.Duration(frame.DurationMS) * time.Millisecond, DurationMS: frame.DurationMS}, nil
		case "error":
			return Result{}, errors.New(frame.Error)
		default:
			return Result{}, fmt.Errorf("PyTorch chat worker returned unexpected frame %q", frame.Kind)
		}
	}
}

func (session *PyTorchSession) nextFrame() (workerFrame, error) {
	if !session.scanner.Scan() {
		if err := session.scanner.Err(); err != nil {
			return workerFrame{}, err
		}
		detail := strings.TrimSpace(session.stderr.String())
		if detail != "" {
			return workerFrame{}, fmt.Errorf("PyTorch chat worker exited: %s", detail)
		}
		return workerFrame{}, fmt.Errorf("PyTorch chat worker exited unexpectedly")
	}
	var frame workerFrame
	if err := json.Unmarshal(session.scanner.Bytes(), &frame); err != nil {
		return workerFrame{}, fmt.Errorf("decode PyTorch chat frame: %w", err)
	}
	if frame.Schema != 1 {
		return workerFrame{}, fmt.Errorf("unsupported PyTorch chat protocol schema %d", frame.Schema)
	}
	return frame, nil
}

func (session *PyTorchSession) Close() error {
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return nil
	}
	session.closed = true
	return errors.Join(session.stdin.Close(), session.command.Wait())
}
