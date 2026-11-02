// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package application

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	git "github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/object"
)

const trainerManifest = `kind: waldo-trainer
schema: 1
name: test
protocol: 1
entrypoint: {module: trainer.main}
commands: {probe: probe, resolve: resolve, train: train}
launch_profiles: [single-process]
objectives: [causal-language-modeling]
artifact_profile: hf-causal-lm
environments:
  cpu: {kind: python-lock, lock: pylock.toml, python: 3.12.3}
`
const tokenizerManifest = `kind: waldo-tokenizer
schema: 1
name: test
protocol: 1
entrypoint: {script: tokenizer.py}
commands: {probe: probe, serve: serve}
environments:
  cpu: {kind: python-lock, lock: pylock.toml, python: 3.12.3}
`

func TestManifest(t *testing.T) {
	for _, fixture := range []struct{ kind, data string }{{"waldo-trainer", trainerManifest}, {"waldo-tokenizer", tokenizerManifest}} {
		m, hash, err := ParseManifest([]byte(fixture.data), fixture.kind)
		if err != nil {
			t.Fatal(err)
		}
		if hash != Digest([]byte(fixture.data)) {
			t.Fatal("manifest bytes not hashed")
		}
		b, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := ParseManifest(b, fixture.kind); err != nil {
			t.Fatal(err)
		}
	}
	cases := map[string]string{
		"schemas":          trainerManifest + "schemas: {architecture: arch.json}\n",
		"kind":             strings.Replace(trainerManifest, "waldo-trainer", "trainer", 1),
		"schema":           strings.Replace(trainerManifest, "schema: 1", "schema: 2", 1),
		"protocol":         strings.Replace(trainerManifest, "protocol: 1", "protocol: 2", 1),
		"executable":       strings.Replace(trainerManifest, "module: trainer.main", "executable: trainer", 1),
		"both entrypoints": strings.Replace(trainerManifest, "module: trainer.main", "module: trainer.main, script: train.py", 1),
		"escape script":    strings.Replace(trainerManifest, "module: trainer.main", "script: ../train.py", 1),
		"environment":      strings.Replace(trainerManifest, "python-lock", "venv", 1),
		"lock escape":      strings.Replace(trainerManifest, "pylock.toml", "../pylock.toml", 1),
		"unknown nested":   strings.Replace(trainerManifest, "python: 3.12.3", "python: 3.12.3, extra: true", 1),
		"extra doc":        trainerManifest + "---\n{}\n",
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if _, _, err := ParseManifest([]byte(data), "waldo-trainer"); err == nil {
				t.Fatal("accepted invalid manifest")
			}
		})
	}
}
func TestSourcePins(t *testing.T) {
	for _, commit := range []string{"main", "v1", "abcd", strings.Repeat("g", 40), strings.Repeat("a", 64)} {
		if (Source{Repository: "repo", Commit: commit, Path: "."}).Validate() == nil {
			t.Fatalf("accepted %q", commit)
		}
	}
	for _, path := range []string{"", "../x", "/x", "a/../x", "a\\b", "C:/x"} {
		if (Source{Repository: "repo", Commit: strings.Repeat("a", 40), Path: path}).Validate() == nil {
			t.Fatalf("accepted path %q", path)
		}
	}
}
func fixtureRepo(t *testing.T) (string, string) {
	t.Helper()
	root := t.TempDir()
	repo, err := git.PlainInit(root, false)
	if err != nil {
		t.Fatal(err)
	}
	files := map[string]string{"app/waldo-trainer.yaml": trainerManifest, "app/waldo-tokenizer.yaml": tokenizerManifest, "app/tokenizer.py": "# not executed\n", "app/pylock.toml": "lock-version = '1.0'\n"}
	for name, data := range files {
		p := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
	}
	wt, err := repo.Worktree()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := wt.Add("."); err != nil {
		t.Fatal(err)
	}
	hash, err := wt.Commit("fixture", &git.CommitOptions{Author: &object.Signature{Name: "Test", Email: "test@example.invalid", When: time.Unix(1, 0)}})
	if err != nil {
		t.Fatal(err)
	}
	return root, hash.String()
}
func TestSourceCache(t *testing.T) {
	repo, commit := fixtureRepo(t)
	cache := t.TempDir()
	t.Cleanup(func() { removeTemporarySource(cache) })
	source := Source{Repository: repo, Commit: commit, Path: "app"}
	first, err := ResolveSource(context.Background(), cache, source, "waldo-trainer", "cpu")
	if err != nil {
		t.Fatal(err)
	}
	second, err := ResolveSource(context.Background(), cache, source, "waldo-tokenizer", "cpu")
	if err != nil {
		t.Fatal(err)
	}
	if first.Directory != second.Directory || first.Tree != second.Tree || first.ManifestSHA256 != Digest([]byte(trainerManifest)) {
		t.Fatal("source identity/cache mismatch")
	}
	r, err := git.PlainOpen(repo)
	if err != nil {
		t.Fatal(err)
	}
	commitObject, err := r.CommitObject(plumbing.NewHash(commit))
	if err != nil {
		t.Fatal(err)
	}
	tree, err := commitObject.Tree()
	if err != nil {
		t.Fatal(err)
	}
	subtree, err := tree.Tree("app")
	if err != nil {
		t.Fatal(err)
	}
	if first.Tree != subtree.Hash.String() {
		t.Fatal("selected path tree hash not recorded")
	}
	// Prove reuse without a remote, and verify every file rather than just HEAD.
	if err := os.Rename(repo, repo+"-offline"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Rename(repo+"-offline", repo) })
	if _, err := ResolveSource(context.Background(), cache, source, "waldo-trainer", "cpu"); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(first.Directory, "pylock.toml")
	if err := os.Chmod(p, 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("tampered"), 0444); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(p, 0444); err != nil {
		t.Fatal(err)
	}
	if _, err := ResolveSource(context.Background(), cache, source, "waldo-trainer", "cpu"); err == nil || !strings.Contains(err.Error(), "hash mismatch") {
		t.Fatalf("tampering: %v", err)
	}
}
func TestSourceRefusals(t *testing.T) {
	repo, commit := fixtureRepo(t)
	for _, tc := range []struct{ name, path, environment, commit string }{{"missing path", "missing", "cpu", commit}, {"missing env", "app", "gpu", commit}, {"missing commit", "app", "cpu", strings.Repeat("a", 40)}} {
		t.Run(tc.name, func(t *testing.T) {
			cache := t.TempDir()
			t.Cleanup(func() { removeTemporarySource(cache) })
			_, err := ResolveSource(context.Background(), cache, Source{Repository: repo, Commit: tc.commit, Path: tc.path}, "waldo-trainer", tc.environment)
			if err == nil {
				t.Fatal("accepted invalid source")
			}
		})
	}
	if err := os.Symlink("/etc/passwd", filepath.Join(repo, "escape")); err != nil {
		t.Fatal(err)
	}
	r, _ := git.PlainOpen(repo)
	wt, _ := r.Worktree()
	_, _ = wt.Add("escape")
	hash, err := wt.Commit("link", &git.CommitOptions{Author: &object.Signature{Name: "Test", Email: "test@example.invalid", When: time.Unix(2, 0)}})
	if err != nil {
		t.Fatal(err)
	}
	cache := t.TempDir()
	t.Cleanup(func() { removeTemporarySource(cache) })
	_, err = ResolveSource(context.Background(), cache, Source{Repository: repo, Commit: hash.String(), Path: "app"}, "waldo-trainer", "cpu")
	if err == nil {
		t.Fatal("accepted symlink")
	}
}
func TestSchemaRoles(t *testing.T) {
	cases := []struct {
		name, schema        string
		architecture, valid bool
	}{
		{"default", `{"type":"object","properties":{"lr":{"type":"number"}}}`, false, true},
		{"recipe", `{"x-waldo-role":"recipe","properties":{"x":{"x-waldo-role":"recipe"}}}`, true, true},
		{"unknown", `{"properties":{"x":{"x-waldo-role":"host"}}}`, false, false},
		{"numeric", `{"x-waldo-role":1}`, false, false},
		{"nested", `{"x-waldo-role":"execution","properties":{"x":{"x-waldo-role":"recipe"}}}`, false, false},
		{"explicit recipe subtree", `{"x-waldo-role":"recipe","properties":{"x":{"x-waldo-role":"execution"}}}`, false, false},
		{"architecture", `{"properties":{"x":{"x-waldo-role":"execution"}}}`, true, false},
		{"ref", `{"$defs":{"host":{"x-waldo-role":"execution"}},"properties":{"host":{"$ref":"#/$defs/host"}}}`, false, true},
		{"ref architecture", `{"$defs":{"host":{"x-waldo-role":"execution"}},"properties":{"host":{"$ref":"#/$defs/host"}}}`, true, false},
		{"ref contradiction", `{"$defs":{"host":{"x-waldo-role":"execution"}},"properties":{"host":{"$ref":"#/$defs/host","x-waldo-role":"recipe"}}}`, false, false},
		{"ref sibling subtree", `{"$defs":{"host":{"x-waldo-role":"execution"}},"$ref":"#/$defs/host","properties":{"x":{"x-waldo-role":"recipe"}}}`, false, false},
		{"nested unused definition", `{"properties":{"x":{"$defs":{"unused":{"x-waldo-role":"invalid"}}}}}`, false, false},
		{"unused definition", `{"$defs":{"unused":{"x-waldo-role":"invalid"}}}`, false, false},
		{"remote ref", `{"$ref":"https://example.invalid/schema"}`, false, false},
		{"file ref", `{"$ref":"file:///etc/passwd"}`, false, false},
		{"recursive", `{"type":"object","properties":{"next":{"$ref":"#"}}}`, false, true},
		{"anchor", `{"$defs":{"host":{"$anchor":"host","x-waldo-role":"execution"}},"properties":{"host":{"$ref":"#host"}}}`, false, true},
		{"invalid schema", `{"type":7}`, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := CompileSchema([]byte(tc.schema), tc.architecture)
			if (err == nil) != tc.valid {
				t.Fatalf("valid=%v err=%v", tc.valid, err)
			}
		})
	}
}
func TestSchemaValidationProjectionAndPreservation(t *testing.T) {
	raw := []byte(`{
 "$schema":"https://json-schema.org/draft/2020-12/schema",
 "x-vendor-note":{"nested":[1,2]}, "type":"object", "additionalProperties":false,
 "$defs":{"host":{"type":"object","x-waldo-role":"execution","properties":{"micro_batch":{"type":"integer","minimum":1}},"required":["micro_batch"]}},
 "properties":{"lr":{"type":"number","exclusiveMinimum":0},"host":{"$ref":"#/$defs/host"},"nested":{"type":"object","properties":{"compile":{"type":"boolean","x-waldo-role":"execution"},"seed":{"type":"integer"}}}},"required":["lr"]
}`)
	s, err := CompileSchema(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	if string(s.Document()) != string(raw) || s.SHA256() != Digest(raw) {
		t.Fatal("schema bytes changed")
	}
	if len(s.schema.Extensions) == 0 || s.schema.Extensions[0].(annotations)["x-vendor-note"] == nil {
		t.Fatal("vendor annotation discarded")
	}
	v := map[string]any{"lr": 0.1, "host": map[string]any{"micro_batch": 2}, "nested": map[string]any{"compile": true, "seed": 7}}
	r, e, err := s.Split(v)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r["host"]; ok {
		t.Fatal("host entered identity")
	}
	if _, ok := e["host"]; !ok {
		t.Fatal("host missing from execution")
	}
	rn := r["nested"].(map[string]any)
	if len(rn) != 1 || rn["seed"] == nil {
		t.Fatalf("bad recipe projection: %#v", r)
	}
	before, _ := json.Marshal(r)
	v["host"] = map[string]any{"micro_batch": 8}
	after, _, err := s.Split(v)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(after)
	if string(before) != string(b) {
		t.Fatal("execution changed recipe")
	}
	v["lr"] = -1
	if err := s.Validate(v); err == nil || !strings.Contains(err.Error(), "lr") {
		t.Fatalf("missing useful error: %v", err)
	}
}

func TestConcurrentSourceCache(t *testing.T) {
	repo, commit := fixtureRepo(t)
	cache := t.TempDir()
	t.Cleanup(func() { removeTemporarySource(cache) })
	source := Source{Repository: repo, Commit: commit, Path: "app"}
	results := make(chan error, 4)
	for i := 0; i < 4; i++ {
		go func() {
			_, err := ResolveSource(context.Background(), cache, source, "waldo-trainer", "cpu")
			results <- err
		}()
	}
	for i := 0; i < 4; i++ {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
	entries, err := os.ReadDir(cache)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("cache has %d entries after concurrent fetch", len(entries))
	}
}

func TestSchemaCombinatorProjection(t *testing.T) {
	s, err := CompileSchema([]byte(`{"type":"object","properties":{"mode":{"enum":["a","b"]}},"allOf":[{"properties":{"seed":{"type":"integer"}}}],"if":{"properties":{"mode":{"const":"a"}}},"then":{"properties":{"workers":{"type":"integer","x-waldo-role":"execution"}}},"else":{"properties":{"workers":{"type":"integer"}}}}`), false)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"a", "b"} {
		recipe, execution, err := s.Split(map[string]any{"mode": mode, "workers": 2, "seed": 1})
		if err != nil {
			t.Fatal(err)
		}
		_, inRecipe := recipe["workers"]
		_, inExecution := execution["workers"]
		if inRecipe != (mode == "b") || inExecution != (mode == "a") {
			t.Fatalf("wrong conditional projection: %v %v", recipe, execution)
		}
	}
}
