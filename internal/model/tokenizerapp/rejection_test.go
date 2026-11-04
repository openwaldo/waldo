// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package tokenizerapp

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

// Each codec failure must be reported for its own reason and promptly, not via
// a later timeout or an unrelated check.
func TestCodecRejectionReasons(t *testing.T) {
	code, err := os.ReadFile("testdata/tokenizer.py")
	if err != nil {
		t.Fatal(err)
	}
	for mode, want := range map[string]string{
		"error":          "request 1: fixture refused",
		"unknown-field":  `unknown field "surprise"`,
		"missing-tokens": "request 1 missing tokens",
		"negative":       "request 1 token outside vocabulary",
		"wrong-id":       "unexpected response id 2",
		"oversize":       "token too long",
		"crash":          "fixture crashed",
	} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			c, err := Open(ctx, fixture(t, string(code)), map[string]any{"mode": mode})
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			start := time.Now()
			_, err = c.EncodeChecked("test")
			if err == nil || !strings.Contains(err.Error(), want) || time.Since(start) > 5*time.Second {
				t.Fatalf("want %q promptly, got %v after %v", want, err, time.Since(start))
			}
		})
	}
}

func TestRequestFrameLimit(t *testing.T) {
	code, err := os.ReadFile("testdata/tokenizer.py")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	c, err := Open(ctx, fixture(t, string(code)), map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err = c.EncodeChecked(strings.Repeat("x", frameLimit)); err == nil || !strings.Contains(err.Error(), "request exceeds frame limit") {
		t.Fatalf("oversize request: %v", err)
	}
}

func TestProbeTrailingJSON(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	quoted, _ := json.Marshal(`{"name":"fixture","vocabulary_size":259,"special_tokens":{"pad":0,"bos":1,"eos":2},"deterministic":true,"schemas":{"configuration":{"type":"object"}}} {}`)
	code := "import sys\nsys.stdin.readline()\nprint(" + string(quoted) + ")\n"
	if c, err := Open(ctx, fixture(t, code), map[string]any{}); err == nil || !strings.Contains(err.Error(), "trailing JSON") {
		if c != nil {
			c.Close()
		}
		t.Fatalf("trailing probe JSON: %v", err)
	}
}
