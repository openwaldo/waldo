// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package applaunch

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/model/pythonenv"
)

func fixture(t *testing.T, code string) Request {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	source := filepath.Join(root, "source")
	env := filepath.Join(root, "env")
	scratch := filepath.Join(root, "scratch")
	for _, dir := range []string{source, filepath.Join(env, "bin"), scratch} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(python, filepath.Join(env, "bin", "python")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "app.py"), []byte(code), 0444); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "pylock.toml"), []byte("fixture lock"), 0444); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(source, 0555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(source, 0755) })
	return Request{Source: application.ResolvedSource{Directory: source, Manifest: application.Manifest{Kind: "waldo-trainer", Schema: 1, Protocol: 1, Name: "test", Entrypoint: application.Entrypoint{Module: "app"}, Commands: application.Commands{Probe: "inspect", Resolve: "resolve", Train: "train"}, LaunchProfiles: []string{"single-process", "distributed"}, Objectives: []string{"causal-language-modeling"}, ArtifactProfile: "test", Environments: map[string]application.Environment{"cpu": {Kind: "python-lock", Lock: "pylock.toml", Python: "3.11.1"}}}}, Environment: pythonenv.Result{Identity: pythonenv.Identity{Name: "cpu", Kind: "python-lock", InterpreterVersion: "3.11.1", LockSHA256: application.Digest([]byte("fixture lock"))}, Run: pythonenv.RunEvidence{CachePath: env}}, Command: "probe", Scratch: scratch}
}

func TestCommandsAndIsolation(t *testing.T) {
	t.Setenv("WALDO_LAUNCH_SECRET", "must-not-leak")
	t.Setenv("PYTHONPATH", "/must/not/leak")
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	for _, kind := range []string{"module", "script"} {
		t.Run(kind, func(t *testing.T) {
			r := fixture(t, "import json, os, sys\nprint(json.dumps({'env':dict(os.environ),'cwd':os.getcwd(),'argv':sys.argv,'flags':[sys.flags.ignore_environment,sys.flags.no_user_site]}))\n")
			if kind == "script" {
				r.Source.Manifest.Entrypoint = application.Entrypoint{Script: "app.py"}
			}
			r.Args = []string{"argument with spaces", "; no shell"}
			e, err := Prepare(r)
			if err != nil {
				t.Fatal(err)
			}
			expected := []string{filepath.Join(r.Environment.Run.CachePath, "bin", "python"), "-E", "-s"}
			if kind == "module" {
				expected = append(expected, "-m", "app")
			} else {
				expected = append(expected, "app.py")
			}
			expected = append(expected, "inspect")
			expected = append(expected, r.Args...)
			if !reflect.DeepEqual(e.Argv, expected) {
				t.Fatalf("argv: %v", e.Argv)
			}
			p, err := Start(context.Background(), r)
			if err != nil {
				t.Fatal(err)
			}
			defer p.Stdout.Close()
			p.Stdin.Close()
			data, err := io.ReadAll(p.Stdout)
			if err != nil {
				t.Fatal(err)
			}
			if err = p.Wait(); err != nil {
				t.Fatal(err)
			}
			var got struct {
				Env   map[string]string
				Cwd   string
				Argv  []string
				Flags []int
			}
			if err = json.Unmarshal(data, &got); err != nil {
				t.Fatalf("%s: %v", data, err)
			}
			wantEnv := map[string]string{}
			for _, v := range e.Environment {
				k, v, _ := strings.Cut(v, "=")
				wantEnv[k] = v
			}
			if !reflect.DeepEqual(got.Env, wantEnv) || got.Cwd != r.Source.Directory || !reflect.DeepEqual(got.Flags, []int{1, 1}) || !reflect.DeepEqual(got.Argv[1:], expected[len(expected)-3:]) {
				t.Fatalf("unexpected child: %+v", got)
			}
			snapshot := p.Evidence()
			snapshot.Environment[0] = "changed"
			if p.Evidence().Environment[0] == "changed" {
				t.Fatal("mutable evidence")
			}
		})
	}
}

func TestRefusalsAndCommandRoles(t *testing.T) {
	r := fixture(t, "")
	for _, role := range []string{"probe", "resolve", "train"} {
		r.Command = role
		if _, err := Prepare(r); err != nil {
			t.Fatal(err)
		}
	}
	r.Profile = "distributed"
	if _, err := Prepare(r); err == nil || !strings.Contains(err.Error(), "not implemented") {
		t.Fatal(err)
	}
	r.Profile = "single-process"
	r.World = World{Nodes: 2, ProcessesPerNode: 2, WorldSize: 4}
	if _, err := Prepare(r); err == nil {
		t.Fatal("accepted world")
	}
	r.World = World{}
	r.Source.Manifest.LaunchProfiles = []string{"distributed"}
	if _, err := Prepare(r); err == nil {
		t.Fatal("undeclared profile")
	}
	r.Source.Manifest.Kind = "waldo-tokenizer"
	r.Source.Manifest.Commands = application.Commands{Probe: "probe", Serve: "serve"}
	r.Source.Manifest.LaunchProfiles = nil
	r.Source.Manifest.Objectives = nil
	r.Source.Manifest.ArtifactProfile = ""
	for _, role := range []string{"probe", "serve"} {
		r.Command = role
		if _, err := Prepare(r); err != nil {
			t.Fatal(err)
		}
	}
	r.Command = "train"
	if _, err := Prepare(r); err == nil {
		t.Fatal("tokenizer train")
	}
	r.Command = "probe"
	r.Environment.Identity.LockSHA256 = application.Digest([]byte("other lock"))
	if _, err := Prepare(r); err == nil {
		t.Fatal("wrong lock accepted")
	}
	r.Environment.Identity.InterpreterVersion = "3.12.0"
	if _, err := Prepare(r); err == nil {
		t.Fatal("wrong environment")
	}
}

func TestStderrAndExit(t *testing.T) {
	r := fixture(t, "import sys\nsys.stderr.write('x'*100000)\nprint('{}')\nsys.exit(7)\n")
	p, err := Start(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stdout.Close()
	p.Stdin.Close()
	data, _ := io.ReadAll(p.Stdout)
	var exit *exec.ExitError
	if err = p.Wait(); !errors.As(err, &exit) || exit.ExitCode() != 7 {
		t.Fatal(err)
	}
	if string(data) != "{}\n" || len(p.Stderr()) != StderrLimit+len("\n[stderr truncated]") {
		t.Fatal("streams mixed or uncapped")
	}
}

func TestCancellation(t *testing.T) {
	for _, ignore := range []bool{false, true} {
		t.Run(map[bool]string{false: "drain", true: "kill"}[ignore], func(t *testing.T) {
			handler := "lambda *args: (print('drained', flush=True), sys.exit(0))"
			if ignore {
				handler = "signal.SIG_IGN"
			}
			r := fixture(t, "import signal, sys, time\nsignal.signal(signal.SIGTERM, "+handler+")\nprint('ready',flush=True)\nwhile True: time.sleep(1)\n")
			e, err := Prepare(r)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			p, err := start(ctx, e, 100*time.Millisecond)
			if err != nil {
				t.Fatal(err)
			}
			defer p.Stdout.Close()
			scanner := bufio.NewScanner(p.Stdout)
			if !scanner.Scan() || scanner.Text() != "ready" {
				t.Fatal("not ready")
			}
			cancel()
			done := make(chan error, 1)
			go func() { done <- p.Wait() }()
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("missing cancellation")
				}
			case <-time.After(3 * time.Second):
				t.Fatal("cancellation hung")
			}
			if !ignore && (!scanner.Scan() || scanner.Text() != "drained") {
				t.Fatal("did not drain")
			}
		})
	}
}

func TestDescendantPipeCleanup(t *testing.T) {
	r := fixture(t, "import os, signal, time\nif os.fork()==0:\n signal.signal(signal.SIGTERM,signal.SIG_IGN)\n time.sleep(60)\nelse:\n print('done',flush=True)\n")
	e, err := Prepare(r)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	p, err := start(ctx, e, 100*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stdout.Close()
	data, err := io.ReadAll(p.Stdout)
	if err != nil || string(data) != "done\n" {
		t.Fatalf("%s %v", data, err)
	}
	if err := p.Wait(); !errors.Is(err, exec.ErrWaitDelay) {
		t.Fatalf("expected inherited stderr drain timeout: %v", err)
	}
}

func TestProtocolPipesAndStartFailures(t *testing.T) {
	r := fixture(t, "import sys\nfor line in sys.stdin:\n print(line.strip(),flush=True)\n")
	p, err := Start(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stdout.Close()
	if _, err := io.WriteString(p.Stdin, "{\"type\":\"begin\"}\n"); err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(p.Stdout)
	if !scanner.Scan() || scanner.Text() != "{\"type\":\"begin\"}" {
		t.Fatal("protocol round trip failed")
	}
	p.Stdin.Close()
	if err := p.Wait(); err != nil {
		t.Fatal(err)
	}
	if err := p.Wait(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Start(ctx, r); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(r.Environment.Run.CachePath, "bin", "python")); err != nil {
		t.Fatal(err)
	}
	if _, err := Start(context.Background(), r); err == nil {
		t.Fatal("started missing interpreter")
	}
}
