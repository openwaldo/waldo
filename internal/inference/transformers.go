// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package inference

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

//go:embed workers/transformers.py
var transformersChatWorker string

type transformersFrame struct {
	frame workerFrame
	err   error
}
type transformersSession struct {
	command *exec.Cmd
	input   io.WriteCloser
	encoder *json.Encoder
	frames  chan transformersFrame
	cancel  context.CancelFunc
	done    chan struct{}
	mu      sync.Mutex
	closed  bool
}

func transformersArtifacts(inspection model.Inspection) (map[string]string, error) {
	for _, run := range inspection.BOM.Runs {
		if run.ID != inspection.BOM.CurrentRunID {
			continue
		}
		if run.State != model.RunComplete || run.Simulated || run.Backend.Name != training.BackendTransformers {
			break
		}
		paths := map[string]string{}
		for _, artifact := range run.Artifacts {
			path, err := resolveModelPath(inspection.Path, artifact.Path)
			if err != nil {
				return nil, err
			}
			if err := verifyArtifact(path, artifact); err != nil {
				return nil, err
			}
			name := filepath.Base(path)
			if paths[name] != "" {
				return nil, fmt.Errorf("duplicate Transformers artifact %s", name)
			}
			paths[name] = path
		}
		for _, name := range []string{"model.safetensors", "config.json", "tokenizer.json"} {
			if paths[name] == "" {
				return nil, fmt.Errorf("missing Transformers artifact %s", name)
			}
		}
		return paths, nil
	}
	return nil, fmt.Errorf("model %q has no complete real Transformers run", inspection.Model.Name)
}

func openTransformers(ctx context.Context, inspection model.Inspection) (Opened, error) {
	paths, err := transformersArtifacts(inspection)
	if err != nil {
		return Opened{}, err
	}
	spec := inspection.Model.Architecture.Transformers
	if err := spec.Validate(); err != nil {
		return Opened{}, err
	}
	python, wheel := os.Getenv("WALDO_TRANSFORMERS_PYTHON"), os.Getenv("WALDO_TRANSFORMERS_WHEEL")
	if python == "" || wheel == "" {
		return Opened{}, fmt.Errorf("Transformers chat requires WALDO_TRANSFORMERS_PYTHON and WALDO_TRANSFORMERS_WHEEL matching the training package pin")
	}
	device, err := training.ProbeTransformersDevice(ctx, python)
	if err != nil {
		return Opened{}, err
	}
	payload, err := json.Marshal(map[string]any{"architecture": inspection.Model.Architecture, "paths": paths, "wheel": wheel, "device": device})
	if err != nil {
		return Opened{}, err
	}
	support, _ := json.Marshal(training.TransformersPythonSupport())
	program := "support = {'__name__': 'waldo_transformers_support'}\nexec(" + string(support) + ", support)\n" + transformersChatWorker
	processCtx, cancel := context.WithCancel(ctx)
	session := &transformersSession{cancel: cancel, frames: make(chan transformersFrame), done: make(chan struct{})}
	session.command = exec.CommandContext(processCtx, python, "-c", program, string(payload))
	session.command.Stderr = io.Discard // Worker errors travel over the bounded JSON protocol.
	session.input, err = session.command.StdinPipe()
	if err != nil {
		cancel()
		return Opened{}, err
	}
	output, err := session.command.StdoutPipe()
	if err != nil {
		session.input.Close()
		cancel()
		return Opened{}, err
	}
	session.encoder = json.NewEncoder(session.input)
	if err := session.command.Start(); err != nil {
		session.input.Close()
		output.Close()
		cancel()
		return Opened{}, err
	}
	go func() {
		defer close(session.done)
		defer close(session.frames)
		scanner := bufio.NewScanner(output)
		scanner.Buffer(make([]byte, 65536), 1024*1024)
		for scanner.Scan() {
			var frame workerFrame
			err := json.Unmarshal(scanner.Bytes(), &frame)
			if err == nil && frame.Schema != 1 {
				err = fmt.Errorf("invalid Transformers chat protocol")
			}
			select {
			case session.frames <- transformersFrame{frame, err}:
			case <-processCtx.Done():
				return
			}
		}
	}()
	readyCtx, stop := context.WithTimeout(ctx, 60*time.Second)
	defer stop()
	frame, err := session.next(readyCtx)
	if err != nil || frame.Kind != "ready" || frame.Context != int(inspection.Model.Architecture.ContextTokens) {
		session.Close()
		return Opened{}, fmt.Errorf("initialize Transformers chat: %v (%s)", err, frame.Error)
	}
	return Opened{Session: session, Description: Description{Model: inspection.Model.Name, SourceType: "run", SourceID: inspection.BOM.CurrentRunID, RunID: inspection.BOM.CurrentRunID, Backend: training.BackendTransformers, ContextTokens: frame.Context}}, nil
}

func (s *transformersSession) next(ctx context.Context) (workerFrame, error) {
	select {
	case <-ctx.Done():
		s.cancel()
		return workerFrame{}, ctx.Err()
	case item, ok := <-s.frames:
		if !ok {
			return workerFrame{}, fmt.Errorf("Transformers chat worker exited")
		}
		if item.frame.Kind == "error" {
			return item.frame, fmt.Errorf("Transformers chat: %s", item.frame.Error)
		}
		return item.frame, item.err
	}
}

func (s *transformersSession) Generate(ctx context.Context, prompt string, options Options, emit func(Token) error) (Result, error) {
	if err := options.Validate(); err != nil {
		return Result{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return Result{}, fmt.Errorf("Transformers chat session closed")
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	stop := context.AfterFunc(ctx, s.cancel)
	defer stop()
	if err := s.encoder.Encode(map[string]any{"kind": "generate", "schema": 1, "prompt": prompt, "options": options}); err != nil {
		s.cancel()
		return Result{}, err
	}
	var text strings.Builder
	for {
		frame, err := s.next(ctx)
		if err != nil {
			s.cancel()
			return Result{}, err
		}
		switch frame.Kind {
		case "token":
			data, err := base64.StdEncoding.DecodeString(frame.Data)
			if err != nil {
				s.cancel()
				return Result{}, err
			}
			text.Write(data)
			if emit != nil {
				if err := emit(Token{Bytes: data}); err != nil {
					s.cancel()
					return Result{}, err
				}
			}
		case "complete":
			return Result{Text: strings.ToValidUTF8(text.String(), "�"), Tokens: frame.Tokens, FinishReason: frame.FinishReason, DurationMS: frame.DurationMS, Duration: time.Duration(frame.DurationMS) * time.Millisecond}, nil
		default:
			s.cancel()
			return Result{}, fmt.Errorf("unexpected Transformers frame %q", frame.Kind)
		}
	}
}

func (s *transformersSession) Close() error {
	s.cancel()
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	s.input.Close()
	<-s.done
	_ = s.command.Wait()
	return nil
}
