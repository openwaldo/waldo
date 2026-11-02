// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

// Package applaunch launches verified external applications, independently of
// their protocol codecs and the built-in training workers.
package applaunch

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/model/pythonenv"
)

const StderrLimit = 64 * 1024
const exitDrain = 5 * time.Second

// World describes one node's launch. Distributed execution is reserved for
// torchrun: local rank zero receives this coordinator's begin/data stream;
// only global rank zero reports protocol results (ADR 0072).
type World struct {
	Nodes            int    `json:"nodes"`
	NodeRank         int    `json:"node_rank"`
	ProcessesPerNode int    `json:"processes_per_node"`
	WorldSize        int    `json:"world_size"`
	MasterAddress    string `json:"master_address,omitempty"`
	MasterPort       int    `json:"master_port,omitempty"`
}

type Request struct {
	Source      application.ResolvedSource
	Environment pythonenv.Result
	Command     string // Manifest command role: probe, resolve, train, or serve.
	Args        []string
	Profile     string
	World       World
	Scratch     string // Existing run scratch directory, used as HOME.
}

// Evidence is run-only provenance. The lifecycle caller must persist it in the
// run BOM; none of these host paths, world facts or values enter plan identity.
type Evidence struct {
	Profile     string   `json:"profile"`
	Launcher    string   `json:"launcher"`
	World       World    `json:"world"`
	Directory   string   `json:"directory"`
	Argv        []string `json:"argv"`
	Environment []string `json:"environment"`
}

// Prepare requires freshly resolved/verified source and environment results.
// It performs no acquisition or installation and never accepts arbitrary env.
func Prepare(r Request) (Evidence, error) {
	var e Evidence
	m := r.Source.Manifest
	if err := m.Validate(m.Kind); err != nil {
		return e, err
	}
	if r.Profile == "" {
		r.Profile = "single-process"
	}
	if r.Profile == "distributed" {
		return e, fmt.Errorf("distributed launch profile (torchrun, node-local-cache) is not implemented yet")
	}
	if r.Profile != "single-process" {
		return e, fmt.Errorf("unsupported launch profile %q", r.Profile)
	}
	if m.Kind == "waldo-trainer" && !slices.Contains(m.LaunchProfiles, r.Profile) {
		return e, fmt.Errorf("trainer does not declare launch profile %q", r.Profile)
	}
	if r.World == (World{}) {
		r.World = World{Nodes: 1, ProcessesPerNode: 1, WorldSize: 1}
	}
	if r.World != (World{Nodes: 1, ProcessesPerNode: 1, WorldSize: 1}) {
		return e, fmt.Errorf("single-process requires one node and one process")
	}
	command := map[string]string{"probe": m.Commands.Probe, "resolve": m.Commands.Resolve, "train": m.Commands.Train, "serve": m.Commands.Serve}[r.Command]
	if command == "" {
		return e, fmt.Errorf("manifest does not declare command %q", r.Command)
	}
	env, ok := m.Environments[r.Environment.Identity.Name]
	if !ok || env.Kind != r.Environment.Identity.Kind || env.Python != r.Environment.Identity.InterpreterVersion || r.Environment.Identity.LockSHA256 == "" {
		return e, fmt.Errorf("verified environment does not match selected manifest environment")
	}
	for _, dir := range []string{r.Source.Directory, r.Scratch, r.Environment.Run.CachePath} {
		if !filepath.IsAbs(dir) {
			return e, fmt.Errorf("launch paths must be absolute: %q", dir)
		}
		info, err := os.Stat(dir)
		if err != nil {
			return e, err
		}
		if !info.IsDir() {
			return e, fmt.Errorf("not a directory: %s", dir)
		}
	}
	lock, err := os.ReadFile(filepath.Join(r.Source.Directory, filepath.FromSlash(env.Lock)))
	if err != nil {
		return e, fmt.Errorf("read selected environment lock: %w", err)
	}
	if application.Digest(lock) != r.Environment.Identity.LockSHA256 {
		return e, fmt.Errorf("verified environment lock does not match selected manifest environment")
	}
	argv := []string{filepath.Join(r.Environment.Run.CachePath, "bin", "python"), "-E", "-s"}
	if m.Entrypoint.Module != "" {
		argv = append(argv, "-m", m.Entrypoint.Module)
	} else {
		argv = append(argv, m.Entrypoint.Script)
	}
	argv = append(argv, command)
	argv = append(argv, r.Args...)
	environment := []string{"WALDO_PROTOCOL=1", "PATH=/usr/bin:/bin", "LANG=C.UTF-8", "LC_ALL=C.UTF-8", "HOME=" + r.Scratch}
	for _, name := range []string{"CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG"} {
		if value, ok := os.LookupEnv(name); ok {
			environment = append(environment, name+"="+value)
		}
	}
	return Evidence{Profile: r.Profile, Launcher: "python", World: r.World, Directory: r.Source.Directory, Argv: argv, Environment: environment}, nil
}

// Process owns a single node's process group. Consume Stdout concurrently with
// writing Stdin; close Stdin at protocol EOF. Always call Wait and close Stdout.
// Cancellation of ctx (including caller-handled SIGINT/SIGTERM) drains for five
// seconds after SIGTERM, then kills the group. Stdout is never mixed with stderr.
type Process struct {
	Stdin    *os.File
	Stdout   *os.File
	evidence Evidence
	done     chan struct{}
	err      error
	stderr   cappedBuffer
}

func Start(ctx context.Context, r Request) (*Process, error) {
	evidence, err := Prepare(r)
	if err != nil {
		return nil, err
	}
	return start(ctx, evidence, exitDrain)
}

func start(ctx context.Context, evidence Evidence, drain time.Duration) (*Process, error) {
	p := &Process{evidence: evidence, done: make(chan struct{})}
	input, writer, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	reader, output, err := os.Pipe()
	if err != nil {
		input.Close()
		writer.Close()
		return nil, err
	}
	cmd := exec.CommandContext(ctx, evidence.Argv[0], evidence.Argv[1:]...)
	cmd.Dir, cmd.Env = evidence.Directory, slices.Clone(evidence.Environment)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = input, output, &p.stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.WaitDelay = drain
	cmd.Cancel = func() error {
		err := syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
		if errors.Is(err, syscall.ESRCH) {
			return os.ErrProcessDone
		}
		return err
	}
	if err := cmd.Start(); err != nil {
		input.Close()
		writer.Close()
		reader.Close()
		output.Close()
		return nil, err
	}
	input.Close()
	output.Close()
	p.Stdin, p.Stdout = writer, reader
	go func() {
		p.err = cmd.Wait()
		if p.err != nil && ctx.Err() != nil {
			p.err = errors.Join(p.err, ctx.Err())
		}
		// Also terminate descendants which kept protocol pipes open after the leader
		// exited, or ignored the graceful termination signal.
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		writer.Close()
		close(p.done)
	}()
	return p, nil
}

// Evidence returns an independent snapshot suitable for run BOM persistence.
func (p *Process) Evidence() Evidence {
	e := p.evidence
	e.Argv = slices.Clone(e.Argv)
	e.Environment = slices.Clone(e.Environment)
	return e
}

// Wait is repeatable and preserves exec.ExitError and context cancellation.
func (p *Process) Wait() error {
	<-p.done
	if p.err != nil {
		return fmt.Errorf("external application exited: %w; stderr: %s", p.err, p.Stderr())
	}
	return nil
}
func (p *Process) Stderr() string { return p.stderr.String() }

type cappedBuffer struct {
	mu        sync.Mutex
	data      []byte
	truncated bool
}

func (b *cappedBuffer) Write(data []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := len(data)
	remaining := StderrLimit - len(b.data)
	if len(data) > remaining {
		data = data[:remaining]
		b.truncated = true
	}
	b.data = append(b.data, data...)
	return n, nil
}
func (b *cappedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	suffix := ""
	if b.truncated {
		suffix = "\n[stderr truncated]"
	}
	return strings.ToValidUTF8(string(b.data), "�") + suffix
}
