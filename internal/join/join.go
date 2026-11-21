// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package join

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/openwaldo/waldo/internal/config"
)

// Joiner is one identity's view of a run.
type Joiner struct {
	// Home holds the identity's signing key, runs-repo clone, models, and exports.
	Home     string
	Identity string
	Key      string // ssh signing key; Home/signing-key when empty
	Email    string // commit author email
	Waldo    string // this waldo executable, run with the identity's own config
	Log      io.Writer
	Now      func() time.Time
	// Rounds, when positive, is how many rounds this identity takes part in.
	Rounds int

	lastRound int // the last round to work in when Rounds is set, once known

	clone      string
	planSHA256 string // the run file this pass's records pin
	runDir     string
	run        string // the run's directory in the runs repository, as given
	plan       Plan
}

func (joiner *Joiner) Open(ctx context.Context, url, runDir string) error {
	if joiner.Now == nil {
		joiner.Now = time.Now
	}
	if joiner.Log == nil {
		joiner.Log = io.Discard
	}
	if absolute, err := filepath.Abs(url); err == nil {
		if _, statErr := os.Stat(absolute); statErr == nil {
			url = absolute
		}
	}
	name := strings.TrimSuffix(filepath.Base(strings.TrimRight(url, "/")), ".git")
	joiner.clone = filepath.Join(joiner.Home, name)
	joiner.runDir = filepath.Join(joiner.clone, filepath.FromSlash(runDir))
	joiner.run = filepath.ToSlash(filepath.Clean(runDir))
	if err := os.MkdirAll(joiner.Home, 0o700); err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(joiner.clone, ".git")); os.IsNotExist(err) {
		if _, err := joiner.git(ctx, joiner.Home, "clone", "-q", url, joiner.clone); err != nil {
			return err
		}
	}
	if err := joiner.configure(ctx); err != nil {
		return err
	}
	if err := joiner.ready(ctx); err != nil {
		return err
	}
	if joiner.Key == "" {
		joiner.Key = filepath.Join(joiner.Home, "signing-key")
	}
	if _, err := os.Stat(joiner.Key); os.IsNotExist(err) {
		comment := "waldo " + joiner.Identity
		if output, err := exec.CommandContext(ctx, "ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", comment, "-f", joiner.Key).CombinedOutput(); err != nil {
			return fmt.Errorf("create signing key: %w: %s", err, bytes.TrimSpace(output))
		}
	}
	return nil
}

// configure gives the identity its own waldo config and model store.
func (joiner *Joiner) configure(ctx context.Context) error {
	if _, err := os.Stat(joiner.config()); os.IsNotExist(err) {
		if source, err := config.Path(); err == nil {
			if data, err := os.ReadFile(source); err == nil {
				if err := os.WriteFile(joiner.config(), data, 0o600); err != nil {
					return err
				}
			}
		}
	}
	if err := joiner.waldo(ctx, "config", "set", "model.root", joiner.models()); err != nil {
		return err
	}
	// Every round reads the same shards; keep them rather than pull them from under another identity.
	return joiner.waldo(ctx, "config", "set", "lookaside.cache.retain-completed", "true")
}

// ready checks the machine can train before the identity claims any work.
func (joiner *Joiner) ready(ctx context.Context) error {
	output, err := joiner.waldoOutput(ctx, "--json", "status")
	if err != nil {
		return err
	}
	var status struct {
		Ready    bool `json:"ready"`
		Training struct {
			Ready     bool `json:"ready"`
			Execution struct {
				Backend struct {
					Name string `json:"name"`
				} `json:"backend"`
			} `json:"execution"`
		} `json:"training"`
	}
	if err := json.Unmarshal(output, &status); err != nil {
		return fmt.Errorf("read waldo status: %w", err)
	}
	if !status.Ready || !status.Training.Ready {
		return fmt.Errorf("this machine is not ready to train; see waldo status: %s", bytes.TrimSpace(output))
	}
	fmt.Fprintf(joiner.Log, "%s trains with %s\n", joiner.Identity, status.Training.Execution.Backend.Name)
	return nil
}

// Step does one unit of work and reports "done", "stopped", "idle", or "worked".
func (joiner *Joiner) Step(ctx context.Context) (string, error) {
	for {
		if err := joiner.sync(ctx); err != nil {
			return "", err
		}
		state, err := joiner.load()
		if err != nil {
			return "", err
		}
		action := state.Next(joiner.Identity, joiner.Now())
		if joiner.stops(action) {
			fmt.Fprintf(joiner.Log, "round %d is merged; %s stops after its %d round(s), as asked\n", action.Round-1, joiner.Identity, joiner.Rounds)
			return "stopped", nil
		}
		switch action.Kind {
		case "done":
			final := state.Rounds[action.Round].Merge
			for number, round := range state.Rounds {
				agreeing, dissenting := round.Agreeing()
				for _, record := range dissenting {
					fmt.Fprintf(joiner.Log, "round %d: %s's merge %s dissents from the %d closers who agreed on %s\n",
						number, record.Identity, record.ModelSHA256[:12], len(agreeing), round.Merge.ModelSHA256[:12])
				}
			}
			if state.Stopped {
				fmt.Fprintf(joiner.Log, "run %s stopped: round %d's gate after %s did not pass in its extra rounds; the last merge is %s (%s), %s\n",
					joiner.plan.Name, action.Round, final.Gate.After, final.Model, final.ModelSHA256[:12], final.URL)
				return "done", nil
			}
			fmt.Fprintf(joiner.Log, "run %s is complete: %s (%s), %s\n", joiner.plan.Name, final.Model, final.ModelSHA256[:12], final.URL)
			return "done", nil
		case "idle":
			fmt.Fprintf(joiner.Log, "round %d: nothing for %s until others finish\n", action.Round, joiner.Identity)
			return "idle", nil
		case "claim":
			won, err := joiner.claim(ctx, action)
			if err != nil || !won {
				return "worked", err
			}
		case "contribute":
			return "worked", joiner.contribute(ctx, state, action)
		default:
			return "worked", joiner.close(ctx, state, action)
		}
	}
}

func (joiner *Joiner) stops(action Action) bool {
	if joiner.Rounds <= 0 || action.Kind == "done" {
		return false
	}
	if joiner.lastRound == 0 {
		joiner.lastRound = action.Round + joiner.Rounds
	}
	return action.Round >= joiner.lastRound
}

func (joiner *Joiner) claim(ctx context.Context, action Action) (bool, error) {
	record := Claim{
		Kind: ClaimKind, Schema: RecordSchema, Run: joiner.plan.Name, Round: action.Round, Stage: action.Stage,
		Slot: action.Slot, Attempt: action.Attempt, Identity: joiner.Identity, CreatedAt: joiner.timestamp(), Plan: joiner.planSHA256,
	}
	path := filepath.Join(roundDir(joiner.runDir, action.Round), "claims", claimFile(action.Slot, action.Attempt))
	won, err := joiner.commit(ctx, path, record, fmt.Sprintf("%s round %d: %s claims %s", joiner.plan.Name, action.Round, joiner.Identity, action.Slot))
	if err == nil && !won {
		fmt.Fprintf(joiner.Log, "round %d: another identity claimed %s first\n", action.Round, action.Slot)
	}
	return won, err
}

// load rereads the run file, since amendments land through git.
func (joiner *Joiner) load() (State, error) {
	plan, err := LoadRun(filepath.Join(joiner.runDir, RunFile))
	if err != nil {
		return State{}, err
	}
	public, err := os.ReadFile(joiner.Key + ".pub")
	if err != nil {
		return State{}, err
	}
	key := strings.Join(strings.Fields(string(public))[:2], " ")
	if plan.Identities[joiner.Identity].Key != key {
		return State{}, fmt.Errorf("%s is not in %s with this key; add under join.identities:\n    %s:\n      key: %s", joiner.Identity, filepath.Join(joiner.runDir, RunFile), joiner.Identity, key)
	}
	joiner.plan = plan
	if joiner.planSHA256, err = Pin(joiner.runDir, plan); err != nil {
		return State{}, err
	}
	allowed := filepath.Join(joiner.workDir(), "allowed_signers")
	if err := os.MkdirAll(joiner.workDir(), 0o755); err != nil {
		return State{}, err
	}
	if err := writeAllowedSigners(allowed, plan.Identities); err != nil {
		return State{}, err
	}
	return ReadState(joiner.runDir, plan, allowed)
}

func (joiner *Joiner) workDir() string {
	return filepath.Join(joiner.Home, "work", filepath.Base(joiner.runDir))
}

func (joiner *Joiner) commit(ctx context.Context, path string, record any, message string) (bool, error) {
	if err := writeSigned(path, record, joiner.Key); err != nil {
		return false, err
	}
	if _, err := joiner.git(ctx, joiner.clone, "add", path, path+".sig"); err != nil {
		return false, err
	}
	if _, err := joiner.git(ctx, joiner.clone, "-c", "user.name="+joiner.Identity, "-c", "user.email="+joiner.Email, "commit", "-q", "-s", "-m", message); err != nil {
		return false, err
	}
	return joiner.push(ctx)
}

func (joiner *Joiner) push(ctx context.Context) (bool, error) {
	for range 5 {
		if _, err := joiner.git(ctx, joiner.clone, "push", "-q"); err == nil {
			return true, nil
		}
		if lost, err := joiner.rebase(ctx); lost || err != nil {
			return false, err
		}
	}
	return false, fmt.Errorf("push kept being rejected; try again later")
}

// rebase replays unpushed records; lost is true when another identity's claim won a slot.
func (joiner *Joiner) rebase(ctx context.Context) (bool, error) {
	for attempt := 0; ; attempt++ {
		_, err := joiner.git(ctx, joiner.clone, "pull", "-q", "--rebase")
		if err == nil {
			// A record that landed meanwhile may have filled a slot an unpushed claim retakes.
			if displaced, err := joiner.displaced(ctx); displaced || err != nil {
				if err == nil {
					err = joiner.standDown(ctx)
				}
				return displaced, err
			}
			return false, nil
		}
		if _, abortErr := joiner.git(ctx, joiner.clone, "rebase", "--abort"); abortErr != nil {
			return false, err // no rebase stopped on a conflict; the pull itself failed
		}
		// The first stop may be a rebase a killed pass left under way.
		if attempt > 0 {
			return true, joiner.standDown(ctx)
		}
	}
}

func (joiner *Joiner) standDown(ctx context.Context) error {
	_, err := joiner.git(ctx, joiner.clone, "reset", "-q", "--hard", "@{u}")
	return err
}

func (joiner *Joiner) displaced(ctx context.Context) (bool, error) {
	output, err := joiner.git(ctx, joiner.clone, "diff", "--name-only", "@{u}", "HEAD", "--", joiner.run)
	if err != nil {
		return false, err
	}
	var claims []string
	for _, path := range strings.Fields(output) {
		if filepath.Base(filepath.Dir(path)) == "claims" {
			claims = append(claims, path)
		}
	}
	if len(claims) == 0 {
		return false, nil
	}
	state, err := joiner.load()
	if err != nil {
		return false, err
	}
	for _, path := range claims {
		var number int
		if _, err := fmt.Sscanf(filepath.Base(filepath.Dir(filepath.Dir(path))), "round-%d", &number); err == nil && state.displaces(number, filepath.Base(path)) {
			return true, nil
		}
	}
	return false, nil
}

// sync brings the clone up to date and pushes what a cut-short pass committed.
func (joiner *Joiner) sync(ctx context.Context) error {
	if _, err := joiner.rebase(ctx); err != nil {
		return err
	}
	_, err := joiner.push(ctx)
	return err
}

func (joiner *Joiner) git(ctx context.Context, directory string, args ...string) (string, error) {
	command := exec.CommandContext(ctx, "git", args...)
	command.Dir = directory
	output, err := command.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, bytes.TrimSpace(output))
	}
	return string(output), nil
}
