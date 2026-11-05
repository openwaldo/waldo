// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package applaunch

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The child environment must equal the documented allowlist, independently of
// what Evidence records, so a leaked host variable cannot hide in both.
func TestEnvironmentIsExactAllowlist(t *testing.T) {
	for _, name := range []string{"WALDO_LAUNCH_SECRET", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "LD_PRELOAD", "PYTHONHOME", "VIRTUAL_ENV"} {
		t.Setenv(name, "sentinel-must-not-leak")
	}
	r := fixture(t, "import json, os\nprint(json.dumps(dict(os.environ)))\n")
	p, err := Start(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stdout.Close()
	p.Stdin.Close()
	data, _ := io.ReadAll(p.Stdout)
	if err = p.Wait(); err != nil {
		t.Fatal(err)
	}
	var env map[string]string
	if err = json.Unmarshal(data, &env); err != nil {
		t.Fatalf("%s: %v", data, err)
	}
	want := map[string]string{"WALDO_PROTOCOL": "1", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": r.Scratch}
	for _, name := range []string{"CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG"} {
		if value, ok := os.LookupEnv(name); ok {
			want[name] = value
		}
	}
	for k, v := range env {
		if want[k] != v || strings.Contains(v, "sentinel") {
			t.Errorf("unexpected child environment %s=%s", k, v)
		}
	}
	if len(env) != len(want) {
		t.Fatalf("child environment %v, want %v", env, want)
	}
}

// A descendant that detached from the protocol pipes and ignores SIGTERM must
// still be gone once Wait returns after a clean leader exit.
func TestNoOrphanAfterLeaderExit(t *testing.T) {
	r := fixture(t, `import os, signal, sys, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    time.sleep(60)
    os._exit(0)
print(pid, flush=True)
`)
	p, err := Start(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stdout.Close()
	p.Stdin.Close()
	scanner := bufio.NewScanner(p.Stdout)
	if !scanner.Scan() {
		t.Fatal("missing descendant pid")
	}
	pid := strings.TrimSpace(scanner.Text())
	if err = p.Wait(); err != nil {
		t.Fatal(err)
	}
	assertGone(t, pid)
}

// Cancellation signals the whole group, so a descendant (e.g. a torchrun
// worker) receives SIGTERM and may drain before the group is killed.
func TestCancellationSignalsDescendants(t *testing.T) {
	r := fixture(t, `import os, signal, sys, time
pid = os.fork()
if pid == 0:
    def drain(*args):
        open(os.path.join(os.environ["HOME"], "drained"), "w").close()
        os._exit(0)
    signal.signal(signal.SIGTERM, drain)
    print("ready", flush=True)
    while True: time.sleep(1)
signal.signal(signal.SIGTERM, lambda *args: (os.waitpid(pid, 0), sys.exit(0)))
while True: time.sleep(1)
`)
	e, err := Prepare(r)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p, err := start(ctx, e, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stdout.Close()
	scanner := bufio.NewScanner(p.Stdout)
	if !scanner.Scan() || scanner.Text() != "ready" {
		t.Fatal("not ready")
	}
	cancel()
	p.Wait()
	if _, err := os.Stat(filepath.Join(r.Scratch, "drained")); err != nil {
		t.Fatalf("descendant did not receive SIGTERM: %v", err)
	}
}

func assertGone(t *testing.T, pid string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		stat, err := os.ReadFile(filepath.Join("/proc", pid, "stat"))
		// A zombie awaiting reaping by an unrelated init is no longer running.
		if err != nil || strings.Contains(string(stat), ") Z ") {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("descendant %s still running after Wait", pid)
}
