// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package inference

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTransformersRealChat(t *testing.T) {
	root, name := os.Getenv("WALDO_HF_CHAT_TEST_ROOT"), os.Getenv("WALDO_HF_CHAT_TEST_MODEL")
	if root == "" || name == "" {
		t.Skip("set WALDO_HF_CHAT_TEST_ROOT and WALDO_HF_CHAT_TEST_MODEL to a completed smoke model")
	}
	inspection, err := model.Inspect(root, name)
	if err != nil {
		t.Fatal(err)
	}
	opened, err := Open(t.Context(), inspection)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Session.Close()
	seed := uint64(42)
	options := Options{MaxTokens: 8, Temperature: 0, TopP: 1, Seed: &seed}
	var stream strings.Builder
	first, err := opened.Session.Generate(t.Context(), "Once upon a time", options, func(token Token) error { stream.Write(token.Bytes); return nil })
	if err != nil {
		t.Fatal(err)
	}
	second, err := opened.Session.Generate(t.Context(), "Once upon a time", options, nil)
	if err != nil {
		t.Fatal(err)
	}
	if first.Text != stream.String() || first.Text != second.Text || first.Tokens != second.Tokens {
		t.Fatalf("stream/determinism mismatch: %+v %+v", first, second)
	}
	if first.Tokens > 8 || first.FinishReason == "" {
		t.Fatalf("invalid result %+v", first)
	}
	t.Logf("%s: %q", name, first.Text)
}

func TestTransformersArtifacts(t *testing.T) {
	root := t.TempDir()
	run := model.ModelBOMRun{ID: "run", State: model.RunComplete, Backend: training.Identity{Name: training.BackendTransformers}}
	for _, name := range []string{"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "waldo_tokenizer.json"} {
		data := []byte(name)
		if err := os.WriteFile(filepath.Join(root, name), data, 0600); err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(data)
		run.Artifacts = append(run.Artifacts, model.ModelBOMArtifact{Path: name, Bytes: int64(len(data)), SHA256: hex.EncodeToString(hash[:])})
	}
	inspection := model.Inspection{Path: root, BOM: model.ModelBOM{CurrentRunID: "run", Runs: []model.ModelBOMRun{run}}}
	if paths, err := transformersArtifacts(inspection); err != nil || len(paths) != 5 {
		t.Fatalf("resolve: %v %v", paths, err)
	}
	inspection.BOM.Runs[0].Simulated = true
	if _, err := transformersArtifacts(inspection); err == nil {
		t.Fatal("accepted fake weights")
	}
	inspection.BOM.Runs[0].Simulated = false
	if err := os.WriteFile(filepath.Join(root, "tokenizer_config.json"), []byte("tampered"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := transformersArtifacts(inspection); err == nil {
		t.Fatal("accepted corrupt tokenizer")
	}
}

func TestTransformersReadCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	killed := false
	session := transformersSession{frames: make(chan transformersFrame), cancel: func() { killed = true }}
	if _, err := session.next(ctx); !errors.Is(err, context.Canceled) || !killed {
		t.Fatalf("cancel: %v, killed %v", err, killed)
	}
}
