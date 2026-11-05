// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package model

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	git "github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing/object"
	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/training"
)

func externalLifecycleFixture(t *testing.T, transforms ...func(string) string) (Builder, ExternalCompose, ExternalOptions) {
	t.Helper()
	dir := t.TempDir()
	t.Cleanup(func() {
		_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err == nil && info.IsDir() {
				return os.Chmod(path, 0755)
			}
			return nil
		})
	})
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal(err)
	}
	version, err := exec.Command(python, "-I", "-c", "import sys; print('.'.join(map(str,sys.version_info[:3])))").Output()
	if err != nil {
		t.Fatal(err)
	}
	repoDir := filepath.Join(dir, "repo")
	repo, err := git.PlainInit(repoDir, false)
	if err != nil {
		t.Fatal(err)
	}
	base, err := os.ReadFile("trainerapp/testdata/trainer.py")
	if err != nil {
		t.Fatal(err)
	}
	branch, err := os.ReadFile("trainerapp/testdata/resumable.py")
	if err != nil {
		t.Fatal(err)
	}
	code := strings.Split(string(base), "\nelse:\n")[0] + "\nelse:\n    " + strings.ReplaceAll(strings.TrimSpace(string(branch)), "\n", "\n    ") + "\n"
	code = strings.Replace(code, "batch = []\n    # One real", "batch = []\n        __import__('time').sleep(0.1)\n    # One real", 1)
	for _, transform := range transforms {
		code = transform(code)
	}
	tokenizer, err := os.ReadFile("tokenizerapp/testdata/tokenizer.py")
	if err != nil {
		t.Fatal(err)
	}
	environment := "environments:\n  cpu: {kind: python-lock, lock: pylock.toml, python: '" + strings.TrimSpace(string(version)) + "'}\n"
	files := map[string]string{
		"trainer.py": code, "tokenizer.py": string(tokenizer), "pylock.toml": "lock-version = \"1.0\"\ncreated-by = \"waldo-test\"\npackages = []\n",
		"waldo-trainer.yaml":   "kind: waldo-trainer\nschema: 1\nname: fixture\nprotocol: 1\nentrypoint: {module: trainer}\ncommands: {probe: probe, resolve: resolve, train: train}\nlaunch_profiles: [single-process]\nobjectives: [causal-language-modeling]\nartifact_profile: hf-causal-lm\n" + environment,
		"waldo-tokenizer.yaml": "kind: waldo-tokenizer\nschema: 1\nname: fixture\nprotocol: 1\nentrypoint: {module: tokenizer}\ncommands: {probe: probe, serve: serve}\n" + environment,
	}
	worktree, err := repo.Worktree()
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range files {
		if err = os.WriteFile(filepath.Join(repoDir, name), []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
		if _, err = worktree.Add(name); err != nil {
			t.Fatal(err)
		}
	}
	commit, err := worktree.Commit("test fixture", &git.CommitOptions{Author: &object.Signature{Name: "Fixture", Email: "fixture@example.invalid", When: time.Unix(1, 0)}})
	if err != nil {
		t.Fatal(err)
	}
	source := application.Source{Repository: repoDir, Commit: commit.String(), Path: "."}
	zero := 0.0
	c := ExternalCompose{Kind: "waldo-model-compose", Schema: 2, Trainer: TrainerApplication{Source: source, Environment: "cpu", Determinism: "required"}, Tokenizer: TokenizerApplication{Source: source, Environment: "cpu", Configuration: map[string]any{}}, Architecture: map[string]any{}, Stages: []ExternalStage{{Name: "train", Type: "pre-training", Objective: "causal-language-modeling", Corpora: NewCorpusSelections([]string{"example"}), Data: ExternalData{Steps: 3, GlobalBatchSequences: 2, SequenceLength: 4, Seed: 42, EvaluationFraction: &zero}, Trainer: map[string]any{}}}}
	prepared := preparedFixture(t, c.Stages[0].DataStage())
	for n := range prepared.Inputs {
		prepared.Inputs[n].Corpus = "example"
	}
	options := ExternalOptions{Cache: filepath.Join(dir, "cache"), Python: python, Prepare: func(context.Context, ExternalStage) (corpus.BOM, []training.Input, error) {
		return prepared.BOM, prepared.Inputs, nil
	}}
	return Builder{Root: filepath.Join(dir, "models")}, c, options
}

func TestExternalLifecycleResumeAndNoop(t *testing.T) {
	b, c, o := externalLifecycleFixture(t)
	control, err := b.ComposeExternal(t.Context(), "control", c, o)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	b.Progress = func(p Progress) {
		if p.Training != nil && p.Training.Kind == "checkpoint" {
			cancel()
		}
	}
	if _, err = b.ComposeExternal(ctx, "resumed", c, o); err == nil {
		t.Fatal("interruption succeeded")
	}
	interrupted, err := InspectExternal(b.Root, "resumed")
	if err != nil {
		t.Fatal(err)
	}
	if interrupted.Model.State != RunInterrupted || len(interrupted.Runs[0].Checkpoints) == 0 {
		t.Fatalf("not resumable: %+v", interrupted)
	}
	runPath := filepath.Join(interrupted.Path, "runs/train/RUN-BOM.json")
	before, err := os.ReadFile(runPath)
	if err != nil {
		t.Fatal(err)
	}
	b.Progress = nil
	resumed, err := b.ComposeExternal(t.Context(), "resumed", c, o)
	if err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(runPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) || len(resumed.Runs[0].Attempts) != 2 || resumed.Model.State != RunComplete {
		t.Fatal("resume rewrote identity or lost attempts")
	}
	a, err := os.ReadFile(filepath.Join(control.Path, "runs/train/artifacts/model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	z, err := os.ReadFile(filepath.Join(resumed.Path, "runs/train/artifacts/model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if string(a) != string(z) {
		t.Fatal("resumed weights differ")
	}
	noop, err := b.ComposeExternal(t.Context(), "resumed", c, o)
	if err != nil {
		t.Fatal(err)
	}
	if len(noop.Runs[0].Attempts) != 2 {
		t.Fatal("completed compose trained again")
	}
	c.Architecture["different"] = true
	if _, err = b.ComposeExternal(t.Context(), "resumed", c, o); err == nil {
		t.Fatal("changed plan accepted")
	}
}

func TestExternalLifecycleRejectsCorruptCheckpoint(t *testing.T) {
	b, c, o := externalLifecycleFixture(t)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	b.Progress = func(p Progress) {
		if p.Training != nil && p.Training.Kind == "checkpoint" {
			cancel()
		}
	}
	if _, err := b.ComposeExternal(ctx, "corrupt", c, o); err == nil {
		t.Fatal("interruption succeeded")
	}
	inspection, err := InspectExternal(b.Root, "corrupt")
	if err != nil {
		t.Fatal(err)
	}
	ref := inspection.Runs[0].Checkpoints[0]
	path := filepath.Join(inspection.Path, "runs/train/bundles", ref.Path)
	if err = os.WriteFile(path, []byte("{}"), 0600); err != nil {
		t.Fatal(err)
	}
	b.Progress = nil
	if _, err = b.ComposeExternal(t.Context(), "corrupt", c, o); err == nil {
		t.Fatal("corrupt checkpoint accepted")
	}
}

func TestExternalLifecycleIntegrityFailureIsTerminal(t *testing.T) {
	b, c, o := externalLifecycleFixture(t, func(code string) string {
		return strings.Replace(code, "struct.pack('<d', state['weight'])", "struct.pack('<d', float('nan'))", 1)
	})
	if _, err := b.ComposeExternal(t.Context(), "failed", c, o); !training.IsNonRetryableWorkerError(err) {
		t.Fatalf("expected integrity failure: %v", err)
	}
	i, err := InspectExternal(b.Root, "failed")
	if err != nil {
		t.Fatal(err)
	}
	if i.Model.State != RunFailed || i.Runs[0].State != RunFailed || len(i.Runs[0].Checkpoints) == 0 {
		t.Fatal("integrity failure is not terminal")
	}
	if _, err = b.ComposeExternal(t.Context(), "failed", c, o); err == nil {
		t.Fatal("terminal run resumed")
	}
}

func TestExternalLifecycleMultiStageInitializationAndListing(t *testing.T) {
	b, c, o := externalLifecycleFixture(t)
	next := c.Stages[0]
	next.Name = "second"
	c.Stages = append(c.Stages, next)
	i, err := b.ComposeExternal(t.Context(), "staged", c, o)
	if err != nil {
		t.Fatal(err)
	}
	if len(i.Runs) != 2 || i.Runs[1].Initialization == nil || !sameExternalJSON(i.Runs[1].Initialization, i.RunBOMs[1].Initialization) {
		t.Fatal("stage initialization is not pinned")
	}
	if i.Runs[1].Initialization.SourceRunID != "train" {
		t.Fatal("wrong preceding stage")
	}
	listings, err := List(b.Root, nil)
	if err != nil || len(listings) != 1 || listings[0].State != "complete" {
		t.Fatalf("list: %+v %v", listings, err)
	}
	telemetry, err := os.ReadFile(filepath.Join(i.Path, TelemetryFilename))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(telemetry), "checkpoint") || !strings.Contains(string(telemetry), "second") {
		t.Fatal("external telemetry missing")
	}
	if _, err = Remove(b.Root, []string{"staged"}); err != nil {
		t.Fatal(err)
	}
}

func TestExternalLifecycleFiniteEpochBudget(t *testing.T) {
	b, c, o := externalLifecycleFixture(t)
	c.Stages[0].Data.Steps = 0
	c.Stages[0].Data.Epochs = 1
	c.Stages[0].Data.SequenceLength = 1024
	c.Stages[0].Data.GlobalBatchSequences = 1
	i, err := b.ComposeExternal(t.Context(), "epoch", c, o)
	if err != nil {
		t.Fatal(err)
	}
	if i.Plan.Identity.Stages[0].Data.Steps != 1 || i.Runs[0].Complete.Steps != 1 {
		t.Fatal("epoch step budget not resolved")
	}
	c.Stages[0].Data.GlobalBatchSequences = 2
	if _, err = b.ComposeExternal(t.Context(), "partial", c, o); err == nil || !strings.Contains(err.Error(), "finite epoch budget") {
		t.Fatalf("missing batch slots accepted: %v", err)
	}
	if _, err = os.Stat(filepath.Join(b.Root, "partial")); !os.IsNotExist(err) {
		t.Fatal("invalid epoch budget created a model")
	}
}

func TestExternalLifecycleRejectsChangedResumeProbeSchemas(t *testing.T) {
	b, c, o := externalLifecycleFixture(t, func(code string) string {
		code = strings.Replace(code, "    if MODE == 'version':", "    if os.path.exists(os.path.join(os.environ['HOME'], 'changed-schema')): result['schemas']['architecture']['title'] = 'changed'\n    if MODE == 'version':", 1)
		return strings.Replace(code, "    begin = request['begin']", "    begin = request['begin']\n    open(os.path.join(begin['paths']['scratch'], 'changed-schema'), 'w').close()", 1)
	})
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	b.Progress = func(p Progress) {
		if p.Training != nil && p.Training.Kind == "checkpoint" {
			cancel()
		}
	}
	if _, err := b.ComposeExternal(ctx, "schema", c, o); err == nil {
		t.Fatal("interruption succeeded")
	}
	b.Progress = nil
	if _, err := b.ComposeExternal(t.Context(), "schema", c, o); err == nil || !strings.Contains(err.Error(), "probe schemas changed") {
		t.Fatalf("changed schema accepted: %v", err)
	}
}

func TestExternalSecondaryRejectsPinsBeforeReadiness(t *testing.T) {
	for _, mismatch := range []string{"source", "environment"} {
		t.Run(mismatch, func(t *testing.T) {
			_, c, o := externalLifecycleFixture(t)
			sources, err := c.ResolveSources(context.Background(), filepath.Join(o.Cache, "sources"))
			if err != nil {
				t.Fatal(err)
			}
			env, err := sources.Trainer.EnsureEnvironment(context.Background(), c.Trainer.Environment, o.Python, filepath.Join(o.Cache, "environments"))
			if err != nil {
				t.Fatal(err)
			}
			p := ExternalNodePlan{Compose: c, BOM: ExternalRunBOM{Identity: ExternalIdentity{Trainer: sources.Trainer, Environment: env.Identity}}}
			if mismatch == "source" {
				p.BOM.Identity.Trainer.Tree = strings.Repeat("a", 40)
			} else {
				p.BOM.Identity.Environment.LockSHA256 = strings.Repeat("a", 64)
			}
			ready := false
			err = RunExternalSecondary(context.Background(), p, o, t.TempDir(), func() error { ready = true; return nil })
			if err == nil || !strings.Contains(err.Error(), mismatch+" mismatch") || ready {
				t.Fatalf("ready=%v err=%v", ready, err)
			}
		})
	}
}

// ComposeExternal builds the plan identity from the compose; WALDO-owned data
// and determinism changes must fork it, so reusing a model name is refused
// before any run state changes.
func TestExternalComposeIdentityTracksDeterminismAndData(t *testing.T) {
	b, c, o := externalLifecycleFixture(t)
	if _, err := b.ComposeExternal(t.Context(), "model", c, o); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*ExternalCompose){
		"determinism": func(c *ExternalCompose) { c.Trainer.Determinism = "best-effort" },
		"data seed":   func(c *ExternalCompose) { c.Stages[0].Data.Seed = 43 },
	} {
		t.Run(name, func(t *testing.T) {
			changed := c
			changed.Stages = append([]ExternalStage(nil), c.Stages...)
			mutate(&changed)
			if _, err := b.ComposeExternal(t.Context(), "model", changed, o); err == nil || !strings.Contains(err.Error(), "differs from immutable plan") {
				t.Fatalf("changed identity input accepted: %v", err)
			}
		})
	}
}

// RUN-BOM.json is the pinned run identity; a modified copy must not be resumed
// and republished.
func TestExternalLifecycleRejectsTamperedRunBOMOnResume(t *testing.T) {
	b, c, o := externalLifecycleFixture(t)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	b.Progress = func(p Progress) {
		if p.Training != nil && p.Training.Kind == "checkpoint" {
			cancel()
		}
	}
	if _, err := b.ComposeExternal(ctx, "tampered", c, o); err == nil {
		t.Fatal("interruption succeeded")
	}
	inspection, err := InspectExternal(b.Root, "tampered")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(inspection.Path, "runs/train/RUN-BOM.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Drop launch evidence without re-encoding the pinned fields.
	tampered := strings.Replace(string(raw), `"launches": [`, `"launches": [], "x": [`, 1)
	if tampered == string(raw) {
		t.Fatal("fixture has no launches")
	}
	if err = os.WriteFile(path, []byte(tampered), 0600); err != nil {
		t.Fatal(err)
	}
	b.Progress = nil
	if _, err = b.ComposeExternal(t.Context(), "tampered", c, o); err == nil || !strings.Contains(err.Error(), "identity digest mismatch") {
		t.Fatalf("tampered run BOM resumed: %v", err)
	}
}
