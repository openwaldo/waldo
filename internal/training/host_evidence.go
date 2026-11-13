// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"encoding/csv"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

// EnvironmentHostFacts is run-only evidence. CUDA driver capability and the
// installed toolkit version are distinct; neither claims a framework runtime.
// Unavailable observations carry an explicit reason, never an invented value.
type EnvironmentHostFacts struct {
	Kernel               string            `json:"kernel"`
	GPUs                 []GPUDriverFact   `json:"gpus"`
	CUDADriverCapability string            `json:"cuda_driver_capability,omitempty"`
	CUDAToolkitVersion   string            `json:"cuda_toolkit_version,omitempty"`
	Unavailable          map[string]string `json:"unavailable,omitempty"`
}

type GPUDriverFact struct {
	Model         string `json:"model"`
	DriverVersion string `json:"driver_version"`
}

func InspectEnvironmentHostFacts(ctx context.Context) EnvironmentHostFacts {
	return inspectEnvironmentHostFacts(func(name string, args ...string) (string, error) {
		probe, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		output, err := exec.CommandContext(probe, name, args...).Output()
		return strings.TrimSpace(string(output)), err
	})
}

func inspectEnvironmentHostFacts(probe func(string, ...string) (string, error)) EnvironmentHostFacts {
	facts := EnvironmentHostFacts{GPUs: []GPUDriverFact{}, Unavailable: map[string]string{}}
	read := func(field, command string, args ...string) string {
		output, err := probe(command, args...)
		if err != nil || output == "" {
			facts.Unavailable[field] = command + " unavailable or query failed"
			return ""
		}
		return output
	}
	facts.Kernel = read("kernel", "uname", "-sr")
	gpus := read("gpus", "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
	if gpus != "" {
		rows, err := csv.NewReader(strings.NewReader(gpus)).ReadAll()
		if err == nil {
			for _, row := range rows {
				if len(row) != 2 || strings.TrimSpace(row[0]) == "" || strings.TrimSpace(row[1]) == "" {
					err = csv.ErrFieldCount
					break
				}
				facts.GPUs = append(facts.GPUs, GPUDriverFact{Model: strings.TrimSpace(row[0]), DriverVersion: strings.TrimSpace(row[1])})
			}
		}
		if err != nil {
			facts.GPUs = []GPUDriverFact{}
			facts.Unavailable["gpus"] = "invalid nvidia-smi GPU/driver output"
		}
	}
	for _, query := range []struct {
		field, command, pattern string
		args                    []string
	}{
		{"cuda_driver_capability", "nvidia-smi", `CUDA Version:\s*([0-9]+\.[0-9]+)`, nil},
		{"cuda_toolkit_version", "nvcc", `release\s+([0-9]+\.[0-9]+)`, []string{"--version"}},
	} {
		output := read(query.field, query.command, query.args...)
		match := regexp.MustCompile(query.pattern).FindStringSubmatch(output)
		if len(match) != 2 {
			facts.Unavailable[query.field] = query.command + " did not report a CUDA version"
			continue
		}
		if query.field == "cuda_driver_capability" {
			facts.CUDADriverCapability = match[1]
		} else {
			facts.CUDAToolkitVersion = match[1]
		}
	}
	return facts
}
