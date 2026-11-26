// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/pythonenv"
)

// ExecutionConfig is the exact resume boundary. Host observations (GPU,
// driver, CUDA and kernel) are evidence, deliberately absent here.
type ExecutionConfig struct {
	Profile              string                      `json:"profile"`
	Launcher             string                      `json:"launcher"`
	World                applaunch.World             `json:"world"`
	Parameters           map[string]any              `json:"parameters"`
	Determinism          string                      `json:"determinism"`
	TrainerEnvironment   pythonenv.ExecutionEvidence `json:"trainer_environment"`
	TokenizerEnvironment pythonenv.ExecutionEvidence `json:"tokenizer_environment"`
}

// CheckExecution reports every differing field rather than a digest-only error.
func CheckExecution(saved, current ExecutionConfig) error {
	normalize := func(v ExecutionConfig) any {
		v.World.NodeRank = 0
		raw, _ := json.Marshal(v)
		var out any
		_ = json.Unmarshal(raw, &out)
		return out
	}
	var fields []string
	var compare func(string, any, any)
	compare = func(path string, a, b any) {
		am, aok := a.(map[string]any)
		bm, bok := b.(map[string]any)
		if aok && bok {
			keys := map[string]bool{}
			for k := range am {
				keys[k] = true
			}
			for k := range bm {
				keys[k] = true
			}
			for k := range keys {
				compare(strings.TrimPrefix(path+"."+k, "."), am[k], bm[k])
			}
		} else if !reflect.DeepEqual(a, b) {
			fields = append(fields, path)
		}
	}
	compare("", normalize(saved), normalize(current))
	if len(fields) == 0 {
		return nil
	}
	sort.Strings(fields)
	return fmt.Errorf("external resume execution config changed: %s", strings.Join(fields, ", "))
}

func (b Begin) executionConfig(parameters map[string]any) ExecutionConfig {
	config := b.ExecutionConfig
	if config.TrainerEnvironment.Wheels == nil {
		config.TrainerEnvironment.Wheels = []string{}
	}
	if config.TokenizerEnvironment.Wheels == nil {
		config.TokenizerEnvironment.Wheels = []string{}
	}
	config.World = b.World
	config.Parameters = parameters
	config.Determinism = b.Determinism
	return config
}
