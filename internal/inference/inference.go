// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package inference owns ephemeral model generation sessions. It consumes
// verified model artifacts but never mutates model lifecycle state.
package inference

import (
	"context"
	"fmt"
	"math"
	"time"
)

type Options struct {
	MaxTokens   int      `json:"max_tokens"`
	Temperature float64  `json:"temperature"`
	TopP        float64  `json:"top_p"`
	Seed        *uint64  `json:"seed,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

func (options Options) Validate() error {
	if options.MaxTokens < 1 || options.MaxTokens > 65536 {
		return fmt.Errorf("max tokens %d must be between 1 and 65536", options.MaxTokens)
	}
	if math.IsNaN(options.Temperature) || math.IsInf(options.Temperature, 0) || options.Temperature < 0 || options.Temperature > 10 {
		return fmt.Errorf("temperature %v must be between 0 and 10", options.Temperature)
	}
	if math.IsNaN(options.TopP) || math.IsInf(options.TopP, 0) || options.TopP <= 0 || options.TopP > 1 {
		return fmt.Errorf("top-p %v must be greater than 0 and at most 1", options.TopP)
	}
	if len(options.Stop) > 8 {
		return fmt.Errorf("at most 8 stop strings are supported")
	}
	for _, stop := range options.Stop {
		if stop == "" {
			return fmt.Errorf("stop strings must not be empty")
		}
	}
	return nil
}

type Token struct {
	Bytes []byte
}

type Result struct {
	Text         string        `json:"text"`
	Tokens       int           `json:"tokens"`
	FinishReason string        `json:"finish_reason"`
	Duration     time.Duration `json:"-"`
	DurationMS   int64         `json:"duration_ms"`
}

type Session interface {
	Generate(context.Context, string, Options, func(Token) error) (Result, error)
	Close() error
}

type Description struct {
	Model                 string `json:"model"`
	SourceType            string `json:"source_type"`
	SourceID              string `json:"source_id"`
	RunID                 string `json:"run_id,omitempty"`
	CheckpointStep        int64  `json:"checkpoint_step,omitempty"`
	InterruptedCheckpoint bool   `json:"interrupted_checkpoint,omitempty"`
	Backend               string `json:"backend"`
	ContextTokens         int    `json:"context_tokens"`
}

type Opened struct {
	Description Description
	Session     Session
}
