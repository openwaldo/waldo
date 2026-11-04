// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package join contributes to a training run kept in a git repository: each
// pass reads the run's plan and signed records, does the next unit of work,
// and commits its own signed record. No process coordinates the run.
package join

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/openwaldo/waldo/internal/model"
	"gopkg.in/yaml.v3"
)

// RunFile is a run's one file in its directory of the runs repository: a WALDO
// compose with a join block.
const RunFile = "run.yaml"

// Plan is how a run is divided and checked, from its run file's join block:
// units, rounds, identities, store, and verification. Compose is the run file
// itself, the training each unit's work is cut from.
type Plan struct {
	Name             string              `yaml:"name"`
	UnitData         map[string][]string `yaml:"units"` // unit to the sources its pre-training stage reads
	Replicas         int                 `yaml:"replicas"`
	Closers          int                 `yaml:"closers"`
	ClaimMinutes     float64             `yaml:"claim_minutes"`
	Store            string              `yaml:"store"`
	Identities       map[string]Identity `yaml:"identities"`
	Base             *PlanBase           `yaml:"base"`
	Rounds           RoundsPlan          `yaml:"rounds"`
	MinStepsPerRound int                 `yaml:"min_steps_per_round"`
	ReplicaTolerance *float64            `yaml:"replica_tolerance"`
	ReplicaFloor     *float64            `yaml:"replica_floor"`
	Gates            []Gate              `yaml:"gates"`

	Units   []string   `yaml:"-"` // unit names in order; a fine-tuning unit takes the slice at its position
	Compose *yaml.Node `yaml:"-"` // the run file's compose mapping
}

// Gate is what a stage's merged model must meet before the run moves on: a
// held-out loss, or an evaluation's metrics checked by waldo model gate. The
// stage's last round checks it; failing, the run adds up to ExtraRounds more of
// the stage, each checked again, and stops if none passes.
type Gate struct {
	After             string   `yaml:"after"` // pretrain or posttrain
	HeldoutLossAtMost *float64 `yaml:"heldout_loss_at_most"`
	Evaluation        string   `yaml:"evaluation"` // evaluation BOM in the run directory
	Evaluator         []string `yaml:"evaluator"`  // run in the run directory as: <command...> <model> <evaluation BOM> <results.json>
	ExtraRounds       int      `yaml:"extra_rounds"`
}

// RoundsPlan sizes the run: the bootstrap, then pretraining rounds of the
// pre-training stage, then post-training rounds of the fine-tuning stage.
type RoundsPlan struct {
	BootstrapTokens int64 `yaml:"bootstrap_tokens"`
	Pretrain        int   `yaml:"pretrain"`
	TokensPerRound  int64 `yaml:"tokens_per_round"` // each pretraining unit's budget, as fixed steps
	Posttrain       int   `yaml:"posttrain"`
	EpochsPerRound  int   `yaml:"epochs_per_round"` // each post-training unit's budget over its slice
}

// PlanBase is an existing model a run starts from instead of random weights: a
// published package pinned by its weights and every file's SHA-256. It may live
// in any Hugging Face account, at a pinned commit, or in a file store.
type PlanBase struct {
	Model       string            `yaml:"model"`
	URL         string            `yaml:"url"`
	ModelSHA256 string            `yaml:"model_sha256"`
	Files       map[string]string `yaml:"files"`
}

// Identity is a plan contributor: an ssh signing key and, for Hugging Face
// stores, the account its uploads come from. A bare string is just the key.
type Identity struct {
	Key         string `yaml:"key"`
	HuggingFace string `yaml:"hf"` // accepted for older run files; unused
}

func (identity *Identity) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind == yaml.ScalarNode {
		identity.Key = node.Value
		return nil
	}
	if node.Kind != yaml.MappingNode {
		return fmt.Errorf("identity must be a signing key or a {key, hf} mapping")
	}
	for index := 0; index < len(node.Content); index += 2 {
		if key := node.Content[index].Value; key != "key" && key != "hf" {
			return fmt.Errorf("identity has unknown field %q", key)
		}
	}
	type plain Identity
	return node.Decode((*plain)(identity))
}

// LoadRun reads a run file: a compose, valid as one, whose join block is
// decoded strictly into the run's plan.
func LoadRun(path string) (Plan, error) {
	if _, _, err := model.LoadCompose(path); err != nil {
		return Plan{}, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Plan{}, err
	}
	var document yaml.Node
	if err := yaml.Unmarshal(data, &document); err != nil || len(document.Content) != 1 || document.Content[0].Kind != yaml.MappingNode {
		return Plan{}, fmt.Errorf("%s: a run file is a compose mapping", path)
	}
	compose := document.Content[0]
	block := mapGet(compose, "join")
	if block == nil {
		return Plan{}, fmt.Errorf("%s: a run file needs a join block", path)
	}
	encoded, err := yaml.Marshal(block)
	if err != nil {
		return Plan{}, err
	}
	var plan Plan
	decoder := yaml.NewDecoder(bytes.NewReader(encoded))
	decoder.KnownFields(true)
	if err := decoder.Decode(&plan); err != nil {
		return Plan{}, fmt.Errorf("%s: join: %w", path, err)
	}
	plan.Compose = compose
	for unit := range plan.UnitData {
		plan.Units = append(plan.Units, unit)
	}
	sort.Strings(plan.Units)
	if err := plan.check(); err != nil {
		return Plan{}, fmt.Errorf("%s: %w", path, err)
	}
	return plan, nil
}

// check fills the plan's defaults and refuses one join cannot run.
func (plan *Plan) check() error {
	if plan.Replicas == 0 {
		plan.Replicas = 1
	}
	if plan.Closers == 0 {
		plan.Closers = 3
	}
	if plan.ClaimMinutes == 0 {
		plan.ClaimMinutes = 60
	}
	if plan.Name == "" || len(plan.Units) == 0 || plan.Replicas < 1 || plan.Closers < 1 || plan.ClaimMinutes < 0 {
		return fmt.Errorf("join needs a name, units, and replicas")
	}
	if !strings.HasPrefix(plan.Store, "file://") {
		return fmt.Errorf("join store %q: expected file:///path", plan.Store)
	}
	// Run, unit, and identity names become directory, model, and branch names.
	names := append([]string{plan.Name}, plan.Units...)
	for name, identity := range plan.Identities {
		if identity.Key == "" {
			return fmt.Errorf("identity %s has no signing key", name)
		}
		names = append(names, name)
	}
	for _, name := range names {
		if model.ValidateName(name) != nil {
			return fmt.Errorf("%q must use lowercase letters, digits, '.', '_', and '-'", name)
		}
	}
	if slices.Contains(plan.Units, "all") || slices.Contains(plan.Units, "close") {
		return fmt.Errorf("units must not be named all, the bootstrap's unit, or close, the closing slots' name")
	}
	for _, unit := range plan.Units {
		if len(plan.UnitData[unit]) == 0 {
			return fmt.Errorf("unit %s needs sources", unit)
		}
	}
	if base := plan.Base; base != nil {
		if model.ValidateName(base.Model) != nil || !strings.HasPrefix(base.URL, "file://") || !validFiles(base.Files, base.ModelSHA256) {
			return fmt.Errorf("base needs a model name, a file:// url, its model_sha256, and every file's sha256, model.safetensors included")
		}
	}
	rounds := plan.Rounds
	if rounds.Pretrain < 0 || rounds.Posttrain < 0 || (plan.Base == nil && rounds.BootstrapTokens <= 0) ||
		(rounds.Pretrain > 0 && rounds.TokensPerRound <= 0) || (rounds.Posttrain > 0 && rounds.EpochsPerRound <= 0) {
		return fmt.Errorf("rounds need bootstrap_tokens (without a base), tokens_per_round for pretraining, and epochs_per_round for post-training")
	}
	seen := map[string]bool{}
	for _, gate := range plan.Gates {
		loss, evaluation := gate.HeldoutLossAtMost != nil, gate.Evaluation != "" || len(gate.Evaluator) > 0
		if (gate.After != "pretrain" && gate.After != "posttrain") || seen[gate.After] || gate.ExtraRounds < 0 || loss == evaluation ||
			(evaluation && (gate.Evaluation == "" || len(gate.Evaluator) == 0 || filepath.IsAbs(gate.Evaluation) || strings.Contains(gate.Evaluation, ".."))) {
			return fmt.Errorf("each gate names after (pretrain or posttrain, once each), and either heldout_loss_at_most or an evaluation BOM in the run directory with its evaluator")
		}
		seen[gate.After] = true
	}
	if plan.Compose != nil {
		if (plan.Base == nil || rounds.Pretrain > 0) && stageNode(plan.Compose, "pre-training") == nil {
			return fmt.Errorf("a bootstrap or pretraining rounds need a pre-training stage")
		}
		if rounds.Posttrain > 0 && stageNode(plan.Compose, "fine-tuning") == nil {
			return fmt.Errorf("post-training rounds need a fine-tuning stage")
		}
		if rounds.Pretrain > 0 && plan.MinStepsPerRound > 0 {
			if steps, err := plan.pretrainSteps(); err != nil || steps < int64(plan.MinStepsPerRound) {
				return fmt.Errorf("pretraining rounds of %d steps are under min_steps_per_round %d (%v)", steps, plan.MinStepsPerRound, err)
			}
		}
	}
	return nil
}

// Slot is one round of the schedule: its stage, and the gate its merge checks.
type Slot struct {
	Stage string
	Gate  *Gate
}

func (plan Plan) gate(stage string) *Gate {
	for index := range plan.Gates {
		if plan.Gates[index].After == stage {
			return &plan.Gates[index]
		}
	}
	return nil
}

// Stages lists the planned rounds in order: the bootstrap, unless the run
// starts from a base, then pretraining, then post-training. Gates may add
// rounds; Schedule says which.
func (plan Plan) Stages() []string {
	slots, _ := plan.Schedule(nil)
	var stages []string
	for _, slot := range slots {
		stages = append(stages, slot.Stage)
	}
	return stages
}

// Schedule is every round as far as the decided merges determine it: a stage's
// last round checks its gate, and while the gate fails another round of the
// stage follows, up to the gate's extra rounds. The schedule ends at a gate
// round not yet decided, and stops the run at a gate that never passed.
func (plan Plan) Schedule(decided []*MergeRecord) ([]Slot, bool) {
	var slots []Slot
	if plan.Base == nil {
		slots = append(slots, Slot{Stage: "bootstrap"})
	}
	for _, segment := range []struct {
		stage  string
		rounds int
	}{{"pretrain", plan.Rounds.Pretrain}, {"posttrain", plan.Rounds.Posttrain}} {
		gate := plan.gate(segment.stage)
		for index := range segment.rounds {
			slot := Slot{Stage: segment.stage}
			if index == segment.rounds-1 {
				slot.Gate = gate
			}
			slots = append(slots, slot)
		}
		if gate == nil || segment.rounds == 0 {
			continue
		}
		for extra := 0; ; extra++ {
			last := len(slots) - 1
			if last >= len(decided) || decided[last] == nil {
				return slots, false // the gate's round is not decided yet; nothing after it is known
			}
			if result := decided[last].Gate; result != nil && result.Passed {
				break
			}
			if extra == gate.ExtraRounds {
				return slots, true // the gate never passed: the run stops here
			}
			slots = append(slots, Slot{Stage: segment.stage, Gate: gate})
		}
	}
	return slots, false
}

// RoundUnits is the bootstrap's single shared unit, or the plan's units.
func (plan Plan) RoundUnits(stage string) []string {
	if stage == "bootstrap" {
		return []string{"all"}
	}
	return plan.Units
}

func (plan Plan) Tolerance() float64 {
	if plan.ReplicaTolerance == nil {
		return 0.25
	}
	return *plan.ReplicaTolerance
}

// Floor is the loss gap below which honest replicas always agree, since a
// converged round moves the loss less than replica noise.
func (plan Plan) Floor() float64 {
	if plan.ReplicaFloor == nil {
		return 0.001
	}
	return *plan.ReplicaFloor
}

// Quorum is how many closers must sign the same merge for a round to close:
// a majority of the plan's closers, or of its identities if it has fewer.
func (plan Plan) Quorum() int {
	closers := plan.Closers
	if count := len(plan.Identities); count > 0 && count < closers {
		closers = count
	}
	return closers/2 + 1
}

// Closing is how many closing slots are open before any disagreement: the
// plan's closers, or every identity if it has fewer.
func (plan Plan) Closing() int {
	if count := len(plan.Identities); count > 0 && count < plan.Closers {
		return count
	}
	return plan.Closers
}

// Pin is what every record pins: the SHA-256 of a manifest of the run file
// and the files its gates use (each evaluation BOM, and each evaluator argument
// that is a file in the run directory), one "<path> <sha256>" line per file,
// so the run's training, rules, and bar are all fixed.
func Pin(runDir string, plan Plan) (string, error) {
	files := []string{RunFile}
	for _, gate := range plan.Gates {
		if gate.Evaluation != "" {
			files = append(files, gate.Evaluation)
		}
		for _, argument := range gate.Evaluator {
			if info, err := os.Stat(filepath.Join(runDir, argument)); err == nil && info.Mode().IsRegular() && !filepath.IsAbs(argument) && !strings.Contains(argument, "..") {
				files = append(files, argument)
			}
		}
	}
	var manifest strings.Builder
	for _, file := range files {
		data, err := os.ReadFile(filepath.Join(runDir, file))
		if err != nil {
			return "", err
		}
		digest := sha256.Sum256(data)
		fmt.Fprintf(&manifest, "%s %s\n", file, hex.EncodeToString(digest[:]))
	}
	digest := sha256.Sum256([]byte(manifest.String()))
	return hex.EncodeToString(digest[:]), nil
}

// start is the plan's base as the record round 0 starts from, or nil for a bootstrap.
func (plan Plan) start() *MergeRecord {
	if plan.Base == nil {
		return nil
	}
	return &MergeRecord{Model: plan.Base.Model, ModelSHA256: plan.Base.ModelSHA256, Files: plan.Base.Files, URL: plan.Base.URL}
}
