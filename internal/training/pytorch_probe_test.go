// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// fakeTorch implements only what pyTorchProbeProgram touches. FAKE_TORCH_NUMPY=missing makes
// Tensor.numpy fail the way torch does when numpy is not installed.
const fakeTorch = `
import os

__version__ = "fake-1"
uint8 = "uint8"


class _Cuda:
    def is_available(self):
        return False


class _Tensor:
    def view(self, dtype):
        return self

    def numpy(self):
        if os.environ.get("FAKE_TORCH_NUMPY") == "missing":
            raise RuntimeError("Numpy is not available")
        return self

    def tobytes(self):
        return b"\x00"

    def item(self):
        return 1.0


cuda = _Cuda()


def tensor(values, device=None):
    return _Tensor()


def zeros(*shape):
    return _Tensor()


def sum(value):
    return value
`

func TestProbePyTorchChecksTheCheckpointNumpyCall(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is not available")
	}
	modules := t.TempDir()
	if err := os.MkdirAll(filepath.Join(modules, "torch"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modules, "torch", "__init__.py"), []byte(fakeTorch), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PYTHONPATH", modules)

	t.Setenv("FAKE_TORCH_NUMPY", "ok")
	facts, err := probePyTorch(context.Background(), python)
	if err != nil {
		t.Fatalf("probe with numpy available: %v", err)
	}
	if facts.TorchVersion != "fake-1" || facts.Device != "cpu" {
		t.Fatalf("unexpected probe facts: %+v", facts)
	}

	t.Setenv("FAKE_TORCH_NUMPY", "missing")
	_, err = probePyTorch(context.Background(), python)
	if err == nil {
		t.Fatal("probe without numpy succeeded, so the first checkpoint save would fail hours into a run")
	}
	for _, want := range []string{"needs a working numpy next to torch", "Numpy is not available"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("probe error %q does not contain %q", err, want)
		}
	}
}
