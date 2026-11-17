// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package join

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/openwaldo/waldo/internal/model"
)

// Round is what the repo records for one round: its claims by slot, the
// submissions and merges of identities holding claims, and, once a majority
// of closers sign the same merge, that merge.
type Round struct {
	Claims      map[string][]Claim // valid claims by slot, attempts in order
	Latest      map[string]Attempt // each slot's latest claim file, valid or not
	Submissions map[string][]Submission
	Merges      []MergeRecord   // every valid merge from a closing slot's holder, agreeing or not
	Votes       []MergeRecord   // one merge per closing slot, in slot order
	Merged      map[string]bool // identities with any merge file, valid or not
	Merge       *MergeRecord    // the majority's merge, once the round is decided
}

// Attempt is a slot's latest claim file as written, so a malformed claim still
// holds its attempt number and expires like any other.
type Attempt struct {
	Number    int
	CreatedAt string `json:"created_at"`
}

var retakenClaim = regexp.MustCompile(`^(.+-[1-9][0-9]*)\.([2-9]|[1-9][0-9]+)\.yaml$`)

// claimAttempt is the slot and attempt a claim file's name holds.
func claimAttempt(file string) (string, int) {
	if match := retakenClaim.FindStringSubmatch(file); match != nil {
		number, _ := strconv.Atoi(match[2])
		return match[1], number
	}
	return strings.TrimSuffix(file, ".yaml"), 1
}

// State is a run as its signed records describe it; unsigned or misplaced records are ignored.
type State struct {
	Slots   []Slot // the rounds as far as the decided ones determine them
	Stopped bool   // a gate never passed, and the run ended there
	Plan    Plan
	Rounds  []Round
}

func roundDir(runDir string, number int) string {
	return filepath.Join(runDir, "rounds", fmt.Sprintf("round-%04d", number))
}

// slotUnit is the unit a slot replicates, or "close" for a closing slot.
func slotUnit(slot string) string {
	unit, _, _ := cutLast(slot, "-")
	return unit
}

func cutLast(text, separator string) (string, string, bool) {
	if index := strings.LastIndex(text, separator); index >= 0 {
		return text[:index], text[index+len(separator):], true
	}
	return text, "", false
}

// ReadState reads every round's records under runDir, verifying each signature against the plan's keys.
func ReadState(runDir string, plan Plan, allowedSigners string) (State, error) {
	pin, err := Pin(runDir, plan)
	if err != nil {
		return State{}, err
	}
	state := State{Plan: plan}
	var decided []*MergeRecord
	for number := 0; ; number++ {
		slots, _ := plan.Schedule(decided)
		if number >= len(slots) {
			break
		}
		stage := slots[number].Stage
		directory := roundDir(runDir, number)
		round := Round{Claims: map[string][]Claim{}, Latest: map[string]Attempt{}, Submissions: map[string][]Submission{}, Merged: map[string]bool{}}
		paths, err := filepath.Glob(filepath.Join(directory, "claims", "*.yaml"))
		if err != nil {
			return State{}, err
		}
		present := map[string]bool{} // slot and attempt of every claim file
		var claims []Claim
		for _, path := range paths {
			slot, attempt := claimAttempt(filepath.Base(path))
			present[fmt.Sprintf("%s %d", slot, attempt)] = true
			if attempt > round.Latest[slot].Number {
				latest := Attempt{Number: attempt}
				if data, err := os.ReadFile(path); err == nil {
					_ = json.Unmarshal(data, &latest)
				}
				latest.Number = attempt
				round.Latest[slot] = latest
			}
			var record Claim
			ok, err := readSigned(path, allowedSigners, plan.Identities, &record, func() string { return record.Identity })
			if err != nil {
				return State{}, err
			}
			if ok && record.Round == number && record.Attempt >= 1 && record.Plan == pin && filepath.Base(path) == claimFile(record.Slot, record.Attempt) && validClaim(record, plan, number, stage) {
				claims = append(claims, record)
			}
		}
		// Attempts in order, each only after the one before it was claimed, valid or not,
		// and one slot per identity per unit.
		sort.Slice(claims, func(i, j int) bool {
			if claims[i].Slot != claims[j].Slot {
				return claims[i].Slot < claims[j].Slot
			}
			return claims[i].Attempt < claims[j].Attempt
		})
		holds := map[string]string{} // identity and unit to the slot it holds
		for _, record := range claims {
			key := record.Identity + " " + slotUnit(record.Slot)
			chained := true
			for earlier := 1; earlier < record.Attempt; earlier++ {
				chained = chained && present[fmt.Sprintf("%s %d", record.Slot, earlier)]
			}
			if held, ok := holds[key]; (ok && held != record.Slot) || !chained {
				continue
			}
			holds[key] = record.Slot
			round.Claims[record.Slot] = append(round.Claims[record.Slot], record)
		}
		paths, err = filepath.Glob(filepath.Join(directory, "submissions", "*.yaml"))
		if err != nil {
			return State{}, err
		}
		sort.Strings(paths)
		for _, path := range paths {
			var record Submission
			ok, err := readSigned(path, allowedSigners, plan.Identities, &record, func() string { return record.Identity })
			if err != nil {
				return State{}, err
			}
			// A record must match the file and run it is in, so none can be replayed elsewhere,
			// and its identity must have claimed the unit, so none trains a unit unasked.
			if _, claimed := holds[record.Identity+" "+record.Unit]; ok && claimed && record.Round == number && record.Plan == pin &&
				filepath.Base(path) == record.Unit+"-"+record.Identity+".yaml" && validSubmission(record, plan, number, stage) {
				round.Submissions[record.Unit] = append(round.Submissions[record.Unit], record)
			}
		}
		paths, err = filepath.Glob(filepath.Join(directory, "merges", "*.yaml"))
		if err != nil {
			return State{}, err
		}
		sort.Strings(paths)
		previous := plan.start()
		if number > 0 {
			previous = state.Rounds[number-1].Merge
		}
		for _, path := range paths {
			round.Merged[strings.TrimSuffix(filepath.Base(path), ".yaml")] = true
			var record MergeRecord
			ok, err := readSigned(path, allowedSigners, plan.Identities, &record, func() string { return record.Identity })
			if err != nil {
				return State{}, err
			}
			if _, claimed := holds[record.Identity+" close"]; ok && claimed && record.Round == number && record.Plan == pin &&
				filepath.Base(path) == record.Identity+".yaml" && validMerge(record, plan, number, stage, previous) && gated(record, slots[number].Gate) {
				round.Merges = append(round.Merges, record)
			}
		}
		round.Votes = votes(round, plan)
		round.Merge = decide(round.Votes, plan.Quorum())
		state.Rounds = append(state.Rounds, round)
		decided = append(decided, round.Merge)
		if round.Merge == nil {
			break
		}
	}
	state.Slots, state.Stopped = plan.Schedule(decided)
	return state, nil
}

// votes are the round's merges counted toward its decision, one per closing
// slot from the earliest of its holders to land a merge, in slot order. A slot
// past the plan's closers votes only once every slot before it has voted, since
// it opens only when those closers have not agreed.
func votes(round Round, plan Plan) []MergeRecord {
	byIdentity := map[string]MergeRecord{}
	for _, record := range round.Merges {
		byIdentity[record.Identity] = record
	}
	var result []MergeRecord
	everyVoted := true
	for position := 1; ; position++ {
		slot := fmt.Sprintf("close-%d", position)
		if round.Latest[slot].Number == 0 || (position > plan.Closing() && !everyVoted) {
			return result
		}
		voted := false
		for _, claim := range round.Claims[slot] {
			if record, ok := byIdentity[claim.Identity]; ok {
				result, voted = append(result, record), true
				break
			}
		}
		everyVoted = everyVoted && voted
	}
}

// decision is what closers must agree on: the round's starting weights, the
// units accepted, the method, the merged weights, and the training provenance
// and round record published with them, so no closer in a majority can publish the agreed
// weights with a different history. Measured losses may differ slightly across
// hardware, and release metadata carries each closer's own timestamps; both
// are evidence, not the decision.
func (record MergeRecord) decision() string {
	base := ""
	if record.BaseSHA256 != nil {
		base = *record.BaseSHA256
	}
	parts := []string{base, record.Method, record.ModelSHA256}
	for _, unit := range record.Accepted {
		parts = append(parts, unit.Unit+" "+unit.Identity+" "+unit.ModelSHA256)
	}
	var provenance []string
	for name, digest := range record.Files {
		if name == model.ClosureFile || name == model.RoundFile || strings.HasPrefix(name, "provenance-") {
			provenance = append(provenance, name+" "+digest)
		}
	}
	slices.Sort(provenance)
	parts = append(parts, provenance...)
	// A gate round's merges must also agree whether the gate passed.
	if record.Gate != nil {
		parts = append(parts, fmt.Sprintf("gate %s %t", record.Gate.After, record.Gate.Passed))
	}
	return strings.Join(parts, "\n")
}

// decide returns the first decision, in slot order, that a quorum of votes
// signs. Votes are only ever added, so once a round is decided no later record
// changes its decision.
func decide(votes []MergeRecord, quorum int) *MergeRecord {
	byDecision := map[string][]MergeRecord{}
	for _, record := range votes {
		agreeing := append(byDecision[record.decision()], record)
		byDecision[record.decision()] = agreeing
		if len(agreeing) == quorum {
			chosen := agreeing[0]
			return &chosen
		}
	}
	return nil
}

// Agreeing is every merge that signed the round's decision, so its weights can
// be fetched from any of their stores; Dissenting is every other one.
func (round Round) Agreeing() (agreeing, dissenting []MergeRecord) {
	for _, record := range round.Merges {
		if round.Merge != nil && record.decision() == round.Merge.decision() {
			agreeing = append(agreeing, record)
		} else {
			dissenting = append(dissenting, record)
		}
	}
	return agreeing, dissenting
}

// Names and hashes in records become paths, model names, and URLs, so a record
// counts only if every one of them is well formed.
var (
	fileName    = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)
	sha256Value = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

// validFiles requires plain file names, SHA-256 values, and model.safetensors pinned to the record's model hash.
func validFiles(files map[string]string, modelSHA256 string) bool {
	if !sha256Value.MatchString(modelSHA256) || files["model.safetensors"] != modelSHA256 {
		return false
	}
	for name, digest := range files {
		if !fileName.MatchString(name) || !sha256Value.MatchString(digest) {
			return false
		}
	}
	return true
}

func validSubmission(record Submission, plan Plan, number int, stage string) bool {
	return record.Kind == SubmissionKind && record.Schema == RecordSchema && record.Run == plan.Name && record.Stage == stage &&
		slices.Contains(plan.RoundUnits(stage), record.Unit) && model.ValidateName(record.Model) == nil &&
		validFiles(record.Files, record.ModelSHA256)
}

// gated is whether a merge carries a gate result exactly when its round checks a gate.
func gated(record MergeRecord, gate *Gate) bool {
	if gate == nil {
		return record.Gate == nil
	}
	return record.Gate != nil && record.Gate.After == gate.After
}

func validClaim(record Claim, plan Plan, number int, stage string) bool {
	unit, index, _ := cutLast(record.Slot, "-")
	position, err := strconv.Atoi(index)
	if err != nil || position < 1 || strconv.Itoa(position) != index {
		return false
	}
	if unit != "close" && (!slices.Contains(plan.RoundUnits(stage), unit) || position > plan.Replicas) {
		return false
	}
	_, err = time.Parse(time.RFC3339, record.CreatedAt)
	return err == nil && record.Kind == ClaimKind && record.Schema == RecordSchema && record.Run == plan.Name && record.Stage == stage
}

func validMerge(record MergeRecord, plan Plan, number int, stage string, previous *MergeRecord) bool {
	// Every round starts from the one before's decision, so the chain between rounds holds.
	if (previous == nil) != (record.BaseSHA256 == nil) || (previous != nil && *record.BaseSHA256 != previous.ModelSHA256) {
		return false
	}
	if record.Kind != MergeKind || record.Schema != RecordSchema || record.Run != plan.Name || record.Stage != stage ||
		model.ValidateName(record.Model) != nil || !validFiles(record.Files, record.ModelSHA256) {
		return false
	}
	for _, unit := range record.Accepted {
		if !slices.Contains(plan.RoundUnits(stage), unit.Unit) || model.ValidateName(unit.Model) != nil {
			return false
		}
	}
	return true
}

// Action is the next thing one identity should do in a run.
type Action struct {
	Kind    string // "done", "idle", "claim", "contribute", or "close"
	Round   int
	Stage   string
	Unit    string // the unit to train, or "close"
	Slot    string // the slot to claim
	Attempt int
}

// expired is whether a claim's holder has had its claim_minutes to land its record.
func (state State) expired(claim Claim, now time.Time) bool {
	created, err := time.Parse(time.RFC3339, claim.CreatedAt)
	return err != nil || now.After(created.Add(time.Duration(state.Plan.ClaimMinutes*float64(time.Minute))))
}

// filled is whether any holder of slot has landed its record.
func (round Round) filled(slot string) bool {
	unit := slotUnit(slot)
	for _, claim := range round.Claims[slot] {
		if unit == "close" && slices.ContainsFunc(round.Merges, func(record MergeRecord) bool { return record.Identity == claim.Identity }) {
			return true
		}
		if slices.ContainsFunc(round.Submissions[unit], func(record Submission) bool { return record.Identity == claim.Identity }) {
			return true
		}
	}
	return false
}

// replicas are a unit's submissions in slot order, one per slot from its
// earliest holder who submitted, so every closer judges the same set even when
// a slot's expired holder submits after the attempt that replaced it.
func (round Round) replicas(unit string, count int) []Submission {
	var result []Submission
	for position := 1; position <= count; position++ {
		for _, claim := range round.Claims[fmt.Sprintf("%s-%d", unit, position)] {
			index := slices.IndexFunc(round.Submissions[unit], func(record Submission) bool { return record.Identity == claim.Identity })
			if index >= 0 && !slices.ContainsFunc(result, func(record Submission) bool { return record.Identity == claim.Identity }) {
				result = append(result, round.Submissions[unit][index])
				break
			}
		}
	}
	return result
}

// allFilled is whether closing slots 1..count all have their merges.
func (round Round) allFilled(count int) bool {
	for position := 1; position <= count; position++ {
		if !round.filled(fmt.Sprintf("close-%d", position)) {
			return false
		}
	}
	return true
}

// holds is whether identity holds a claim on a slot of unit.
func (round Round) holds(identity, unit string) bool {
	for slot, claims := range round.Claims {
		if slotUnit(slot) == unit && slices.ContainsFunc(claims, func(claim Claim) bool { return claim.Identity == identity }) {
			return true
		}
	}
	return false
}

// open is the slot's next attempt if it can be claimed now: never claimed,
// or its latest holder let the claim expire without landing a record.
func (state State) open(round Round, slot string, now time.Time) (int, bool) {
	latest, ok := round.Latest[slot]
	if !ok {
		return 1, true
	}
	if round.filled(slot) || !state.expired(Claim{CreatedAt: latest.CreatedAt}, now) {
		return 0, false
	}
	return latest.Number + 1, true
}

// positions are slots 1..count starting at an offset of identity's own, so
// identities looking for a free slot at the same moment mostly try different ones.
func positions(identity string, round, count int) []int {
	result := make([]int, count)
	offset := pieceSeed(identity, round)
	for index := range result {
		result[index] = (offset+index)%count + 1
	}
	return result
}

// Next picks this pass's work. A full round is closed by its claimed
// closers: the plan's closers, then one more for each that disagrees until a
// majority signs one merge. Otherwise an identity trains the unit it claimed,
// else claims a replica slot of the lowest unit it has not taken part in.
func (state State) Next(identity string, now time.Time) Action {
	number := len(state.Rounds) - 1
	if number < 0 {
		number = 0
	}
	if number < len(state.Rounds) && state.Rounds[number].Merge != nil {
		number++
	}
	slots := state.Slots
	if number >= len(slots) {
		return Action{Kind: "done", Round: len(slots) - 1}
	}
	stage := slots[number].Stage
	units := state.Plan.RoundUnits(stage)
	round := Round{Claims: map[string][]Claim{}, Submissions: map[string][]Submission{}}
	if number < len(state.Rounds) {
		round = state.Rounds[number]
	}
	full := 0
	for _, unit := range units {
		if len(round.replicas(unit, state.Plan.Replicas)) >= state.Plan.Replicas {
			full++
		}
	}
	if full == len(units) {
		if round.Merged[identity] {
			return Action{Kind: "idle", Round: number, Stage: stage}
		}
		if round.holds(identity, "close") {
			return Action{Kind: "close", Round: number, Stage: stage, Unit: "close"}
		}
		// Past the plan's closers, a slot opens only once every one before it has a merge and none has a majority.
		order := positions(identity, number, state.Plan.Closing())
		for position := state.Plan.Closing() + 1; position <= state.Plan.Closing()+len(state.Plan.Identities); position++ {
			order = append(order, position)
		}
		for _, position := range order {
			slot := fmt.Sprintf("close-%d", position)
			if position > state.Plan.Closing() && !round.allFilled(position-1) {
				break
			}
			if attempt, ok := state.open(round, slot, now); ok {
				return Action{Kind: "claim", Round: number, Stage: stage, Unit: "close", Slot: slot, Attempt: attempt}
			}
		}
		return Action{Kind: "idle", Round: number, Stage: stage}
	}
	submitted := func(unit string) bool {
		return slices.ContainsFunc(round.Submissions[unit], func(record Submission) bool { return record.Identity == identity })
	}
	for _, unit := range units {
		if !submitted(unit) && round.holds(identity, unit) {
			return Action{Kind: "contribute", Round: number, Stage: stage, Unit: unit}
		}
	}
	for _, unit := range units {
		if submitted(unit) || round.holds(identity, unit) || len(round.replicas(unit, state.Plan.Replicas)) >= state.Plan.Replicas {
			continue
		}
		for _, position := range positions(identity, number, state.Plan.Replicas) {
			slot := fmt.Sprintf("%s-%d", unit, position)
			if attempt, ok := state.open(round, slot, now); ok {
				return Action{Kind: "claim", Round: number, Stage: stage, Unit: unit, Slot: slot, Attempt: attempt}
			}
		}
	}
	return Action{Kind: "idle", Round: number, Stage: stage}
}
