// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package tokenizerapp

import (
	"context"
	"encoding/json"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/model/pythonenv"
	"github.com/openwaldo/waldo/internal/record"
	"github.com/openwaldo/waldo/internal/training"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func fixture(t *testing.T, code string) applaunch.Request {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	source := filepath.Join(root, "source")
	env := filepath.Join(root, "env")
	scratch := filepath.Join(root, "scratch")
	for _, dir := range []string{source, filepath.Join(env, "bin"), scratch} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(python, filepath.Join(env, "bin", "python")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "app.py"), []byte(code), 0444); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "pylock.toml"), []byte("fixture lock"), 0444); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(source, 0555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(source, 0755) })
	return applaunch.Request{Source: application.ResolvedSource{Directory: source, Manifest: application.Manifest{Kind: "waldo-tokenizer", Schema: 1, Protocol: 1, Name: "test", Entrypoint: application.Entrypoint{Module: "app"}, Commands: application.Commands{Probe: "probe", Serve: "serve"}, Environments: map[string]application.Environment{"cpu": {Kind: "python-lock", Lock: "pylock.toml", Python: "3.11.1"}}}}, Environment: pythonenv.Result{Identity: pythonenv.Identity{Name: "cpu", Kind: "python-lock", InterpreterVersion: "3.11.1", LockSHA256: application.Digest([]byte("fixture lock"))}, Run: pythonenv.RunEvidence{CachePath: env}}, Command: "probe", Scratch: scratch}
}

func TestCodec(t *testing.T) {
	code, err := os.ReadFile("testdata/tokenizer.py")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req := fixture(t, string(code))
	c, err := Open(ctx, req, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	texts := []string{"hello", "", "世界", "a\nb"}
	got, err := c.EncodeBatch(texts)
	if err != nil {
		t.Fatal(err)
	}
	for i, text := range texts {
		want := make([]int, 0)
		for _, b := range []byte(text) {
			want = append(want, int(b)+3)
		}
		if !reflect.DeepEqual(got[i], want) {
			t.Fatalf("%v != %v", got[i], want)
		}
	}
	if c.Spec().Revision != c.Identity().SHA256() || c.Spec().EOSID != 2 {
		t.Fatal("identity/spec mismatch")
	}
	original := c.Identity()
	changed := c.Identity()
	changed.ConfigurationSHA256 = "different"
	if original.SHA256() == changed.SHA256() {
		t.Fatal("configuration not in identity")
	}
	if _, err := Open(ctx, req, map[string]any{"unknown": true}); err == nil {
		t.Fatal("invalid configuration accepted")
	}
	if err := c.Close(); err != nil {
		t.Fatalf("clean shutdown: %v", err)
	}
}
func TestFailures(t *testing.T) {
	code, err := os.ReadFile("testdata/tokenizer.py")
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"crash", "malformed", "range", "wrong-id", "null-token"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			c, err := Open(ctx, fixture(t, string(code)), map[string]any{"mode": mode})
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			_, err = c.EncodeChecked("test")
			if err == nil || !strings.Contains(err.Error(), "request 1") {
				t.Fatalf("missing request failure: %v", err)
			}
			if _, err = c.EncodeChecked("later"); err == nil {
				t.Fatal("failure not sticky")
			}
		})
	}
}

func TestPipelinedOutOfOrder(t *testing.T) {
	code, err := os.ReadFile("testdata/tokenizer.py")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	c, err := Open(ctx, fixture(t, string(code)), map[string]any{"mode": "reverse"})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	texts := make([]string, 130)
	for i := range texts {
		texts[i] = strings.Repeat("x", i)
	}
	tokens, err := c.EncodeBatch(texts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range texts {
		if len(tokens[i]) != i {
			t.Fatalf("response order %d: %v", i, tokens[i])
		}
	}
}
func TestCancellation(t *testing.T) {
	code, err := os.ReadFile("testdata/tokenizer.py")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	c, err := Open(ctx, fixture(t, string(code)), map[string]any{"mode": "hang"})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err = c.EncodeChecked("hello"); err == nil {
		t.Fatal("cancellation ignored")
	}
}

type fixtureRecords []training.Record

func (r fixtureRecords) Stream(ctx context.Context, consume func(training.Record) error) error {
	for _, v := range r {
		if err := consume(v); err != nil {
			return err
		}
	}
	return nil
}
func TestConversationMasks(t *testing.T) {
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
	records := fixtureRecords{{ID: "conversation-1", Conversation: &record.Conversation{Messages: []record.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "yes"}}}}}
	source, err := training.NewTokenizedRecordSource(records, c, "assistant-response-modeling", training.ConversationTransform{Template: training.ConversationTemplateUserAssistantV1, SupervisedRoles: []string{"assistant"}})
	if err != nil {
		t.Fatal(err)
	}
	err = source.Stream(ctx, func(r training.Record) error {
		if r.Text != "" || r.Conversation != nil {
			t.Fatal("raw text reached trainer")
		}
		prefix := "User: hi\n\nAssistant:"
		if len(r.Tokens) != len(prefix)+3 || len(r.LossMask) != len(r.Tokens)+1 {
			t.Fatalf("bad dimensions: %+v", r)
		}
		for i, enabled := range r.LossMask {
			if enabled != (i >= len(prefix)) {
				t.Fatalf("mask %d: %v", i, enabled)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestInvalidProbe(t *testing.T) {
	valid := `{"name":"fixture","vocabulary_size":259,"special_tokens":{"pad":0,"bos":1,"eos":2},"deterministic":true,"schemas":{"configuration":{"type":"object"}}}`
	cases := map[string]string{
		"missing-determinism": strings.Replace(valid, `"deterministic":true,`, "", 1),
		"null-special":        strings.Replace(valid, `"eos":2`, `"eos":null`, 1),
		"missing-eos":         strings.Replace(valid, `,"eos":2`, "", 1),
		"special-range":       strings.Replace(valid, `"eos":2`, `"eos":259`, 1),
		"schema":              strings.Replace(valid, `{"type":"object"}`, `{"type":"invalid"}`, 1),
		"extra-frame":         valid + "\n" + valid,
	}
	for name, response := range cases {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			// repr through JSON is also a valid Python string for this ASCII fixture.
			quoted, _ := json.Marshal(response)
			code := "import sys\nsys.stdin.readline()\nprint(" + string(quoted) + ")\n"
			if c, err := Open(ctx, fixture(t, code), map[string]any{}); err == nil {
				c.Close()
				t.Fatal("invalid probe accepted")
			}
		})
	}
}
