// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestWorkerCancellationRemainsContextCancellation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	command := exec.CommandContext(ctx, os.Args[0], "-test.run=TestWorkerCancellationHelper")
	command.Env = append(os.Environ(), "WALDO_WORKER_CANCELLATION_HELPER=1")
	_, err := runWorkerCommand(ctx, "test", command, Request{
		ArtifactDirectory: t.TempDir(),
		Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
			return consume(Record{ID: "one", Text: "hello"})
		}),
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("worker cancellation error = %v", err)
	}
}

func TestWorkerCancellationHelper(t *testing.T) {
	if os.Getenv("WALDO_WORKER_CANCELLATION_HELPER") != "1" {
		return
	}
	_, _ = io.Copy(io.Discard, os.Stdin)
	time.Sleep(time.Hour)
}

func TestGracefulCancellationLetsLauncherReapWorker(t *testing.T) {
	directory := t.TempDir()
	launcher := filepath.Join(directory, "launcher")
	pidPath := filepath.Join(directory, "worker.pid")
	markerPath := filepath.Join(directory, "stopped")
	script := `#!/bin/sh
sleep 30 &
worker=$!
printf '%s' "$worker" > "$1"
trap 'kill "$worker" 2>/dev/null; wait "$worker" 2>/dev/null; printf stopped > "$2"; exit 0' TERM
wait "$worker"
`
	if err := os.WriteFile(launcher, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	command := exec.CommandContext(ctx, launcher, pidPath, markerPath)
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	configureGracefulCancellation(command)
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	var workerPID int
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		contents, err := os.ReadFile(pidPath)
		if err == nil {
			if _, err := fmt.Sscanf(string(contents), "%d", &workerPID); err == nil {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	if workerPID == 0 {
		cancel()
		_ = command.Wait()
		t.Fatal("launcher did not record its worker PID")
	}
	cancel()
	_ = command.Wait()
	if _, err := os.Stat(markerPath); err != nil {
		t.Fatalf("launcher did not handle SIGTERM: %v", err)
	}
	if err := syscall.Kill(workerPID, 0); !errors.Is(err, syscall.ESRCH) {
		t.Fatalf("launcher worker %d remains after cancellation: %v", workerPID, err)
	}
}

func TestGracefulCancellationForcesUnresponsiveLauncherDown(t *testing.T) {
	previous := workerExitDrain
	workerExitDrain = 100 * time.Millisecond
	defer func() { workerExitDrain = previous }()
	directory := t.TempDir()
	launcher := filepath.Join(directory, "launcher")
	ready := filepath.Join(directory, "ready")
	if err := os.WriteFile(launcher, []byte("#!/bin/sh\ntrap '' TERM\nprintf ready > \"$1\"\nsleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	command := exec.CommandContext(ctx, launcher, ready)
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	configureGracefulCancellation(command)
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := os.Stat(ready); err != nil {
		cancel()
		_ = command.Wait()
		t.Fatalf("launcher did not become ready: %v", err)
	}
	started := time.Now()
	cancel()
	_ = command.Wait()
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("unresponsive launcher cancellation took %v", elapsed)
	}
}

func TestWorkerErrorGracefullyStopsLauncherRanks(t *testing.T) {
	directory := t.TempDir()
	launcher := filepath.Join(directory, "launcher")
	pidPath := filepath.Join(directory, "rank.pid")
	markerPath := filepath.Join(directory, "stopped")
	script := `#!/bin/sh
sleep 30 &
rank=$!
printf '%s' "$rank" > "$1"
trap 'kill "$rank" 2>/dev/null; wait "$rank" 2>/dev/null; printf stopped > "$2"; exit 0' TERM
printf '%s\n' '{"kind":"error","schema":1,"error":"synthetic backend failure"}'
wait "$rank"
`
	if err := os.WriteFile(launcher, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(context.Background(), launcher, pidPath, markerPath)
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	configureGracefulCancellation(command)
	_, err := runWorkerCommand(context.Background(), "test", command, Request{
		ArtifactDirectory: directory,
		Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
			return consume(Record{ID: "one", Text: "hello"})
		}),
	})
	if err == nil || !strings.Contains(err.Error(), "synthetic backend failure") {
		t.Fatalf("worker error = %v", err)
	}
	if _, err := os.Stat(markerPath); err != nil {
		t.Fatalf("launcher did not reap its rank after backend failure: %v", err)
	}
	contents, err := os.ReadFile(pidPath)
	if err != nil {
		t.Fatal(err)
	}
	var rankPID int
	if _, err := fmt.Sscanf(string(contents), "%d", &rankPID); err != nil {
		t.Fatal(err)
	}
	if err := syscall.Kill(rankPID, 0); !errors.Is(err, syscall.ESRCH) {
		t.Fatalf("launcher rank %d remains after backend failure: %v", rankPID, err)
	}
}

func TestWorkerReportsRecordProducerFailureBeforeGenericEOF(t *testing.T) {
	command := exec.Command("/bin/sh", "-c", `cat >/dev/null; printf '%s\n' '{"kind":"error","schema":1,"error":"worker input ended without begin/end framing"}'`)
	_, err := runWorkerCommand(context.Background(), "test", command, Request{
		ArtifactDirectory: t.TempDir(),
		Records: recordSourceFunc(func(context.Context, func(Record) error) error {
			return errors.New("prepared stream produced 10 of 20 required sequences")
		}),
	})
	if err == nil || !strings.Contains(err.Error(), "prepared stream produced 10 of 20 required sequences") {
		t.Fatalf("worker error = %v", err)
	}
}

func TestWorkerTargetStopsUpstreamRecordStream(t *testing.T) {
	command := exec.Command(os.Args[0], "-test.run=TestWorkerTargetHelper")
	command.Env = append(os.Environ(), "WALDO_WORKER_TARGET_HELPER=1")
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	reported := make(chan struct{})
	streamed := 0
	observation, err := runWorkerCommand(context.Background(), "test", command, Request{
		RunID: "run", Stage: "pretrain", Objective: "causal-language-modeling",
		Parameters: parameters, ArtifactDirectory: t.TempDir(),
		Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
			if err := consume(Record{ID: "first", Text: "hello"}); err != nil {
				return err
			}
			streamed++
			<-reported
			for position := 1; position < 100; position++ {
				if err := consume(Record{ID: fmt.Sprintf("record-%d", position), Text: "unused"}); err != nil {
					return err
				}
				streamed++
			}
			return nil
		}),
		Report: func(event Event) {
			if event.Step == 1 {
				close(reported)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if observation.Steps != 1 || streamed != 1 {
		t.Fatalf("observation steps/streamed records = %d/%d, want 1/1", observation.Steps, streamed)
	}
}

func TestWorkerTargetHelper(t *testing.T) {
	if os.Getenv("WALDO_WORKER_TARGET_HELPER") != "1" {
		return
	}
	scanner := bufio.NewScanner(os.Stdin)
	reported := false
	for scanner.Scan() {
		var frame WorkerInputFrame
		if err := json.Unmarshal(scanner.Bytes(), &frame); err != nil {
			os.Exit(2)
		}
		if frame.Kind == "record" && !reported {
			reported = true
			fmt.Println(`{"kind":"event","schema":1,"event":{"kind":"progress","message":"target reached","step":1,"tokens":8}}`)
		}
		if frame.Kind == "end" {
			fmt.Println(`{"kind":"complete","schema":1,"observation":{"simulated":false,"steps":1,"consumed_tokens":8,"artifacts":[]}}`)
			os.Exit(0)
		}
	}
	os.Exit(3)
}

func TestWorkerCommandFailsWhenOrphanHoldsOutputStream(t *testing.T) {
	previous := workerExitDrain
	workerExitDrain = 300 * time.Millisecond
	defer func() { workerExitDrain = previous }()
	worker := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
sleep 30 &
while IFS= read -r line; do :; done
exit 0
`
	if err := os.WriteFile(worker, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	command := exec.Command(worker)
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	done := make(chan error, 1)
	go func() {
		_, runErr := runWorkerCommand(context.Background(), "TorchTitan", command, Request{
			ArtifactDirectory: t.TempDir(), ArtifactPrefix: "artifacts", Parameters: parameters,
			Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
				return consume(Record{ID: "one", Text: "hello"})
			}),
		})
		done <- runErr
	}()
	select {
	case runErr := <-done:
		if runErr == nil || !strings.Contains(runErr.Error(), "held its output stream open") {
			t.Fatalf("orphan error = %v", runErr)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("runWorkerCommand hung after the worker exited with an orphan holding stdout")
	}
}

func TestWorkerCommandFailsWhenOrphanHoldsPipesAndWorkerNeverReadsInput(t *testing.T) {
	previous := workerExitDrain
	workerExitDrain = 300 * time.Millisecond
	defer func() { workerExitDrain = previous }()
	worker := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
sleep 30 &
exit 0
`
	if err := os.WriteFile(worker, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	parameters, err := ResolveParameters(Parameters{Steps: 100, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	command := exec.Command(worker)
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	done := make(chan error, 1)
	go func() {
		_, runErr := runWorkerCommand(context.Background(), "TorchTitan", command, Request{
			ArtifactDirectory: t.TempDir(), ArtifactPrefix: "artifacts", Parameters: parameters,
			Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
				for index := 0; index < 20000; index++ {
					if err := consume(Record{ID: fmt.Sprintf("record-%d", index), Text: strings.Repeat("harbor lanterns ", 8)}); err != nil {
						return err
					}
				}
				return nil
			}),
		})
		done <- runErr
	}()
	select {
	case runErr := <-done:
		if runErr == nil {
			t.Fatal("expected a failure when the worker exits without draining input")
		}
	case <-time.After(20 * time.Second):
		t.Fatal("runWorkerCommand hung writing records to a worker that exited")
	}
}

func TestWorkerCommandToleratesForeignStdoutLines(t *testing.T) {
	worker := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
while IFS= read -r line; do :; done
printf '%s\n' 'NOTE: Redirects are currently not supported in Windows or MacOs.'
printf '%s\n' 'spark-1:42:99 [0] NCCL WARN Connect to 10.10.10.13<45191> failed : Connection refused'
printf '%s\n' '{"kind":"event","schema":1,"event":{"kind":"progress","message":"step 1","step":1,"tokens":2}}'
printf '%s\n' 'W0813 12:00:00.000000 42 torch/distributed/elastic/agent.py:1 some warning'
printf '%s\n' '{"kind":"complete","schema":1,"observation":{"simulated":false,"steps":1,"consumed_tokens":2,"final_loss":1.0,"artifacts":[]}}'
`
	if err := os.WriteFile(worker, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	reported := false
	observation, err := runWorkerCommand(context.Background(), "TorchTitan", exec.Command(worker), Request{
		ArtifactDirectory: t.TempDir(), ArtifactPrefix: "artifacts", Parameters: parameters,
		Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
			return consume(Record{ID: "one", Text: "hello"})
		}),
		Report: func(event Event) { reported = true },
	})
	if err != nil {
		t.Fatalf("foreign stdout lines must not fail the run: %v", err)
	}
	if observation.Simulated || observation.Steps != 1 || !reported {
		t.Fatalf("observation = %+v, reported = %v", observation, reported)
	}
}

func TestWorkerCommandKeepsCompletionWhenWorkerExitsImmediately(t *testing.T) {
	worker := filepath.Join(t.TempDir(), "fake-python")
	script := `#!/bin/sh
while IFS= read -r line; do :; done
printf '%s\n' '{"kind":"event","schema":1,"event":{"kind":"progress","message":"step 1","step":1,"tokens":2}}'
printf '%s\n' '{"kind":"complete","schema":1,"observation":{"simulated":false,"steps":1,"consumed_tokens":2,"final_loss":1.0,"artifacts":[]}}'
exit 0
`
	if err := os.WriteFile(worker, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	parameters, err := ResolveParameters(Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	for attempt := 0; attempt < 40; attempt++ {
		observation, err := runWorkerCommand(context.Background(), "MLX", exec.Command(worker), Request{
			ArtifactDirectory: t.TempDir(), ArtifactPrefix: "artifacts", Parameters: parameters,
			Records: recordSourceFunc(func(_ context.Context, consume func(Record) error) error {
				return consume(Record{ID: "one", Text: "hello"})
			}),
		})
		if err != nil {
			t.Fatalf("attempt %d: worker that exits right after completing must not fail: %v", attempt, err)
		}
		if observation.Steps != 1 {
			t.Fatalf("attempt %d: observation = %+v", attempt, observation)
		}
	}
}
