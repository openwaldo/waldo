// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"fmt"
	"testing"
)

func TestEnvironmentHostFacts(t *testing.T) {
	facts := inspectEnvironmentHostFacts(func(name string, args ...string) (string, error) {
		switch name {
		case "uname":
			return "Linux 6.1", nil
		case "nvcc":
			return "Cuda compilation tools, release 12.8, V12.8.93", nil
		case "nvidia-smi":
			if len(args) == 0 {
				return "CUDA Version: 13.0", nil
			}
			return "GPU A, 580.1\nGPU B, 580.1", nil
		}
		return "", fmt.Errorf("missing")
	})
	if facts.Kernel != "Linux 6.1" || len(facts.GPUs) != 2 || facts.GPUs[1].DriverVersion != "580.1" || facts.CUDADriverCapability != "13.0" || facts.CUDAToolkitVersion != "12.8" || len(facts.Unavailable) != 0 {
		t.Fatalf("unexpected facts: %+v", facts)
	}
}

func TestEnvironmentHostFactsUnavailable(t *testing.T) {
	facts := inspectEnvironmentHostFacts(func(string, ...string) (string, error) { return "", fmt.Errorf("not found") })
	if len(facts.Unavailable) != 4 || len(facts.GPUs) != 0 {
		t.Fatalf("unavailable facts: %+v", facts)
	}
}
