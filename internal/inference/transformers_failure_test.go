// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package inference

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestTransformersFailureHelper(t *testing.T) {
	mode := ""
	for _, argument := range os.Args {
		if strings.HasPrefix(argument, "--hf-failure-helper=") {
			mode = strings.TrimPrefix(argument, "--hf-failure-helper=")
		}
	}
	if mode == "" {
		return
	}
	if mode == "startup-wait" {
		time.Sleep(30 * time.Second)
		os.Exit(0)
	}
	if mode == "startup-malformed" {
		fmt.Println("not json")
		time.Sleep(30 * time.Second)
		os.Exit(0)
	}
	if mode == "startup-exit" {
		fmt.Fprintln(os.Stderr, "startup diagnostic")
		os.Exit(7)
	}
	fmt.Println(`{"schema":1,"kind":"ready","context_tokens":128}`)
	if mode == "no-read" {
		time.Sleep(30 * time.Second)
		os.Exit(0)
	}
	bufio.NewScanner(os.Stdin).Scan()
	switch mode {
	case "malformed":
		fmt.Println("not json")
	case "oversized":
		fmt.Println(strings.Repeat("x", 1024*1024+1))
	case "exit":
		fmt.Fprintln(os.Stderr, "worker diagnostic")
		os.Exit(7)
	case "error":
		fmt.Println(`{"schema":1,"kind":"error","error":"intentional failure"}`)
	default:
		fmt.Println(`{"schema":1,"kind":"token","data":"eA=="}`)
	}
	time.Sleep(30 * time.Second)
	os.Exit(0)
}

func startFailureSession(t *testing.T, ctx context.Context, mode string) (*transformersSession, error) {
	t.Helper()
	return startTransformersSession(ctx, os.Args[0], []string{"-test.run=^TestTransformersFailureHelper$", "--", "--hf-failure-helper=" + mode}, 128)
}

func assertTransformersReaped(t *testing.T, session *transformersSession) {
	t.Helper()
	done := make(chan struct{})
	go func() { session.Close(); session.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(8 * time.Second):
		t.Fatal("worker cleanup timed out")
	}
	if session.command.ProcessState == nil {
		t.Fatal("worker was not reaped")
	}
}

func TestTransformersStartupFailures(t *testing.T) {
	for _, mode := range []string{"startup-wait", "startup-malformed", "startup-exit"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
			defer cancel()
			_, err := startFailureSession(t, ctx, mode)
			if err == nil {
				t.Fatal("accepted failed worker")
			}
			if mode == "startup-wait" && !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("lost deadline: %v", err)
			}
			if mode == "startup-exit" && !strings.Contains(err.Error(), "startup diagnostic") {
				t.Fatalf("lost diagnostics: %v", err)
			}
			if mode == "startup-malformed" && !strings.Contains(err.Error(), "invalid character") {
				t.Fatalf("lost protocol error: %v", err)
			}
		})
	}
}

func TestTransformersGenerationFailures(t *testing.T) {
	for _, mode := range []string{"cancel", "emit", "exit", "error", "malformed", "oversized"} {
		t.Run(mode, func(t *testing.T) {
			lifetime, stop := context.WithTimeout(t.Context(), 10*time.Second)
			defer stop()
			session, err := startFailureSession(t, lifetime, mode)
			if err != nil {
				t.Fatal(err)
			}
			defer assertTransformersReaped(t, session)
			ctx, cancel := context.WithCancel(lifetime)
			defer cancel()
			sentinel := errors.New("consumer stopped")
			_, err = session.Generate(ctx, "hello", Options{MaxTokens: 8, TopP: 1}, func(Token) error {
				if mode == "cancel" {
					cancel()
				}
				if mode == "emit" {
					return sentinel
				}
				return nil
			})
			if err == nil {
				t.Fatal("accepted failed generation")
			}
			if mode == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatalf("lost cancellation: %v", err)
			}
			if mode == "emit" && !errors.Is(err, sentinel) {
				t.Fatalf("lost callback error: %v", err)
			}
			for scenario, message := range map[string]string{"exit": "worker diagnostic", "error": "intentional failure", "malformed": "invalid character", "oversized": "token too long"} {
				if mode == scenario && !strings.Contains(err.Error(), message) {
					t.Fatalf("lost %s diagnostics: %v", scenario, err)
				}
			}
		})
	}
}

func TestTransformersCancellationWinsOverClosedFrames(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	frames := make(chan transformersFrame)
	close(frames)
	session := transformersSession{frames: frames, cancel: func() {}}
	for i := 0; i < 100; i++ {
		if _, err := session.next(ctx); !errors.Is(err, context.Canceled) {
			t.Fatalf("lost cancellation: %v", err)
		}
	}
}

func TestTransformersCancellationDuringInputWrite(t *testing.T) {
	lifetime, stop := context.WithTimeout(t.Context(), 10*time.Second)
	defer stop()
	session, err := startFailureSession(t, lifetime, "no-read")
	if err != nil {
		t.Fatal(err)
	}
	defer assertTransformersReaped(t, session)
	ctx, cancel := context.WithTimeout(lifetime, 200*time.Millisecond)
	defer cancel()
	_, err = session.Generate(ctx, strings.Repeat("x", 2*1024*1024), Options{MaxTokens: 1, TopP: 1}, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("lost write cancellation: %v", err)
	}
}

func TestTransformersDiagnosticsBound(t *testing.T) {
	var diagnostics transformersDiagnostics
	input := []byte(strings.Repeat("x", 128*1024))
	if n, err := diagnostics.Write(input); n != len(input) || err != nil {
		t.Fatalf("write: %d %v", n, err)
	}
	if len(diagnostics.String()) != 65536 {
		t.Fatal("unbounded diagnostics")
	}
}

func TestTransformersLargeValidUnicodeFrames(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("Python required for actual Output framing test")
	}
	// Use the real Python Output class through the real Go protocol reader,
	// without requiring model dependencies or downloading a tokenizer.
	program := `
import ast, json, pathlib, sys
def emit(kind, **data):
    print(json.dumps(dict(schema=1, kind=kind, **data)), flush=True)
path = pathlib.Path("workers/transformers.py")
module = ast.parse(path.read_text())
module.body = [node for node in module.body if not isinstance(node, ast.Try)]
scope = {"support": {"emit": emit}}
exec(compile(module, str(path), "exec"), scope)
emit("ready", context_tokens=128)
for line in sys.stdin:
    scope["Output"]([]).write("é世界😀" * 100000, final=True)
    emit("complete", tokens=1, finish_reason="max_tokens")
`
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	session, err := startTransformersSession(ctx, python, []string{"-c", program}, 128)
	if err != nil {
		t.Fatal(err)
	}
	defer assertTransformersReaped(t, session)
	var streamed strings.Builder
	result, err := session.Generate(ctx, "test", Options{MaxTokens: 1, TopP: 1}, func(token Token) error { streamed.Write(token.Bytes); return nil })
	if err != nil {
		t.Fatal(err)
	}
	expected := strings.Repeat("é世界😀", 100000)
	if result.Text != expected || streamed.String() != expected {
		t.Fatal("chunking changed Unicode output")
	}
}
