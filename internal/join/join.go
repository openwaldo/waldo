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

// Joiner is one identity's view of a run: its own clone of the runs repo,
// signing key, and working directory under Home.
type Joiner struct {
	// Home is everything of one identity: its signing key, its clone of the
	// runs repo, models, and exports.
	Home     string
	Identity string
	Key      string // ssh signing key; Home/signing-key when empty
	Email    string // commit author email
	Waldo    string // this waldo executable, run with the identity's own config
	Log      io.Writer
	Now      func() time.Time
	// Rounds, when positive, is how many rounds this identity takes part in,
	// counting from the first round still open when it starts; then it stops.
	Rounds int

	lastRound int // the last round to work in when Rounds is set, once known

	clone      string
	url        string // the runs repository, as round records name it
	planSHA256 string // the plan.yaml this pass's records pin
	runDir     string
	plan       Plan
}

// Open clones the runs repo when needed and makes sure the signing key exists.
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
	joiner.url = url
	name := strings.TrimSuffix(filepath.Base(strings.TrimRight(url, "/")), ".git")
	joiner.clone = filepath.Join(joiner.Home, name)
	joiner.runDir = filepath.Join(joiner.clone, filepath.FromSlash(runDir))
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

// configure gives the identity its own copy of the waldo config with its own
// model store, so identities on one machine never share a model name.
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
	// Every round reads the same corpus shards, so keep them between commands rather than
	// purging and re-downloading them, which would also pull them from under another
	// identity sharing this machine's cache.
	return joiner.waldo(ctx, "config", "set", "lookaside.cache.retain-completed", "true")
}

// ready checks, with waldo status, that this identity's machine can train
// before it claims any work, so a missing backend fails now, not an hour in.
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

// Step does one unit of work and reports "done", "stopped", "idle", or "worked". Work
// is claimed first, so a pass that wins its claim goes straight on to the work.
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

// stops is whether action is past the rounds this identity was asked to take part in.
// Rounds are the run's pause points: a merged round leaves nothing in flight.
func (joiner *Joiner) stops(action Action) bool {
	if joiner.Rounds <= 0 || action.Kind == "done" {
		return false
	}
	if joiner.lastRound == 0 {
		joiner.lastRound = action.Round + joiner.Rounds
	}
	return action.Round >= joiner.lastRound
}

// claim commits this identity's claim on a slot; false if another identity's claim landed first.
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

// load reads the run file as of this pass, since amendments land through git, and checks this identity is in it.
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

// commit signs record into path and pushes it; false if another identity's claim on the same slot won the race.
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
		if _, err := joiner.git(ctx, joiner.clone, "pull", "-q", "--rebase"); err != nil {
			// Only another claim on the same slot can conflict; theirs stands.
			_, _ = joiner.git(ctx, joiner.clone, "rebase", "--abort")
			_, resetErr := joiner.git(ctx, joiner.clone, "reset", "-q", "--hard", "@{u}")
			return false, resetErr
		}
	}
	return false, fmt.Errorf("push kept being rejected; try again later")
}

// sync brings the clone up to date and pushes any record a cut-short pass committed but never pushed.
func (joiner *Joiner) sync(ctx context.Context) error {
	if _, err := joiner.git(ctx, joiner.clone, "pull", "-q", "--rebase"); err != nil {
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
