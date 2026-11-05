// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package join

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/openwaldo/waldo/internal/model"
)

// pieceSeed is the same for every replica of one piece of work and different across pieces.
func pieceSeed(parts ...any) int {
	words := make([]string, len(parts))
	for index, part := range parts {
		words[index] = fmt.Sprint(part)
	}
	digest := sha256.Sum256([]byte(strings.Join(words, "-")))
	return int(digest[0])<<16 | int(digest[1])<<8 | int(digest[2])
}

func (joiner *Joiner) modelName(number int, suffix string) string {
	return fmt.Sprintf("waldo-%s-r%04d-%s", joiner.plan.Name, number, suffix)
}

func (joiner *Joiner) waldo(ctx context.Context, args ...string) error {
	command := exec.CommandContext(ctx, joiner.Waldo, args...)
	command.Env = append(os.Environ(), "WALDO_CONFIG="+joiner.config())
	command.Stdout, command.Stderr = joiner.Log, joiner.Log
	fmt.Fprintf(joiner.Log, "$ waldo %s\n", strings.Join(args, " "))
	if err := command.Run(); err != nil {
		return fmt.Errorf("waldo %s: %w", args[0]+" "+args[1], err)
	}
	return nil
}

// waldoOutput runs waldo like waldo() but returns its standard output.
func (joiner *Joiner) waldoOutput(ctx context.Context, args ...string) ([]byte, error) {
	command := exec.CommandContext(ctx, joiner.Waldo, args...)
	command.Env = append(os.Environ(), "WALDO_CONFIG="+joiner.config())
	command.Stderr = joiner.Log
	fmt.Fprintf(joiner.Log, "$ waldo %s\n", strings.Join(args, " "))
	output, err := command.Output()
	if err != nil {
		return nil, fmt.Errorf("waldo %s: %w", strings.Join(args, " "), err)
	}
	return output, nil
}

func (joiner *Joiner) config() string {
	return filepath.Join(joiner.Home, "waldo-config.json")
}

func (joiner *Joiner) models() string {
	return filepath.Join(joiner.Home, "models")
}

// built is whether an earlier pass already built name: a completed training run,
// or a merged origin. A run that failed leaves a model holding only its starting
// weights, which must be trained again, not taken as the result.
func (joiner *Joiner) built(name string) bool {
	inspection, err := model.Inspect(joiner.models(), name)
	return err == nil && (inspection.BOM.CurrentRunID != "" || (inspection.Origin != nil && inspection.Origin.Merge != nil))
}

// train trains name from the compose text unless an earlier pass already did.
func (joiner *Joiner) train(ctx context.Context, name, text string) error {
	if joiner.built(name) {
		fmt.Fprintf(joiner.Log, "   %s already trained\n", name)
		return nil
	}
	// A pass cut short (a closed laptop, a killed process) leaves training to resume from its newest
	// checkpoint; waldo refuses a run that failed outright, and that one starts over.
	if _, err := model.Inspect(joiner.models(), name); err == nil {
		fmt.Fprintf(joiner.Log, "   %s did not finish training; resuming it\n", name)
		if err := joiner.waldo(ctx, "model", "continue", name); err == nil && joiner.built(name) {
			return nil
		}
		fmt.Fprintf(joiner.Log, "   %s cannot resume; starting it over\n", name)
		if err := joiner.waldo(ctx, "model", "rm", name); err != nil {
			return err
		}
	}
	path := filepath.Join(joiner.workDir(), "composes", name+".yaml")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		return err
	}
	return joiner.waldo(ctx, "model", "train", name, path)
}

// maxTelemetry is the largest value of a column in the last run's TELEMETRY.csv.
func (joiner *Joiner) maxTelemetry(name, column string) (float64, error) {
	runs, err := filepath.Glob(filepath.Join(joiner.models(), name, "runs", "*", "TELEMETRY.csv"))
	if err != nil || len(runs) == 0 {
		return 0, fmt.Errorf("model %s has no telemetry", name)
	}
	sort.Strings(runs)
	file, err := os.Open(runs[len(runs)-1])
	if err != nil {
		return 0, err
	}
	defer file.Close()
	rows, err := csv.NewReader(file).ReadAll()
	if err != nil || len(rows) == 0 {
		return 0, fmt.Errorf("read %s telemetry: %v", name, err)
	}
	index := slices.Index(rows[0], column)
	best := 0.0
	for _, row := range rows[1:] {
		if index >= 0 && index < len(row) {
			if value, err := strconv.ParseFloat(row[index], 64); err == nil && value > best {
				best = value
			}
		}
	}
	return best, nil
}

func (joiner *Joiner) export(ctx context.Context, name string) (string, error) {
	target := filepath.Join(joiner.workDir(), "exports", name)
	if _, err := os.Stat(filepath.Join(target, "model.safetensors")); err == nil {
		return target, nil
	}
	_ = os.RemoveAll(target)
	return target, joiner.waldo(ctx, "model", "export", name, target, "--format", "huggingface", "--allow-incomplete")
}

// fileHashes pins every file a record publishes, so none can be swapped afterwards.
func fileHashes(directory string) (map[string]string, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	result := map[string]string{}
	for _, entry := range entries {
		if !entry.Type().IsRegular() || strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		digest, err := hashFile(filepath.Join(directory, entry.Name()))
		if err != nil {
			return nil, err
		}
		result[entry.Name()] = digest
	}
	return result, nil
}

func hashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func (joiner *Joiner) store() string {
	return filepath.Join(strings.TrimPrefix(joiner.plan.Store, "file://"), joiner.plan.Name)
}

// publish copies an export into the run's store and returns its URL.
func (joiner *Joiner) publish(ctx context.Context, directory string) (string, error) {
	// Each identity's packages live apart, so closers of one round never share a directory.
	target := filepath.Join(joiner.store(), joiner.Identity, filepath.Base(directory))
	if _, err := os.Stat(target); os.IsNotExist(err) {
		fmt.Fprintf(joiner.Log, "   publishing %s to %s\n", filepath.Base(directory), target)
		if err := copyDirectory(directory, target); err != nil {
			return "", err
		}
	}
	return "file://" + target, nil
}

// fetched is where a record's package is kept once fetched: by publisher, since
// closers of one round publish packages of one name that differ in their metadata.
func (joiner *Joiner) fetched(identity, name string) string {
	if identity == "" {
		identity = "_plan" // the plan's own base; no identity name starts with _
	}
	return filepath.Join(joiner.workDir(), "fetched", identity, name)
}

// fetch brings a record's package from the store, exactly the files it pins,
// each checked against its hash, and keeps it for later passes.
func (joiner *Joiner) fetch(ctx context.Context, name, identity, sha string, files map[string]string, url string) (string, error) {
	if files["model.safetensors"] != sha {
		return "", fmt.Errorf("%s: its files and model_sha256 disagree", name)
	}
	target := joiner.fetched(identity, name)
	if actual, err := fileHashes(target); err == nil && sameFiles(actual, files) {
		return target, nil
	}
	fmt.Fprintf(joiner.Log, "   fetching %s\n", url)
	if err := (model.Puller{}).FetchPinned(ctx, url, files, target); err != nil {
		return "", fmt.Errorf("%s: %w", name, err)
	}
	return target, nil
}

// sameFiles is whether a kept package still holds every pinned file unchanged.
func sameFiles(actual, pinned map[string]string) bool {
	for file, digest := range pinned {
		if actual[file] != digest {
			return false
		}
	}
	return true
}

// published is the local name of a model taken from the network: its record's
// name and weights. It never resolves to a model this identity trained or merged
// itself, so every identity imports the same published package and records the
// same lineage for it.
func published(name, sha256 string) string {
	return name + "-" + sha256[:12]
}

// ensure imports a fetched package as a local model so composes can name it as a base.
func (joiner *Joiner) ensure(name, directory string) error {
	if inspection, err := model.Inspect(joiner.models(), name); err == nil && (inspection.BOM.CurrentRunID != "" || inspection.BOM.CurrentOriginSHA256 != "") {
		return nil
	}
	_, err := (model.Puller{Root: joiner.models()}).PullDirectory(name, directory)
	return err
}

// base fetches and imports the round's starting weights: the previous round's
// majority merge, from the store of any closer who signed it.
func (joiner *Joiner) base(ctx context.Context, state State, number int) (*MergeRecord, error) {
	if number == 0 {
		return joiner.planBase(ctx)
	}
	agreeing, _ := state.Rounds[number-1].Agreeing()
	var err error
	for _, record := range agreeing {
		var directory string
		if directory, err = joiner.fetch(ctx, record.Model, record.Identity, record.ModelSHA256, record.Files, record.URL); err != nil {
			fmt.Fprintf(joiner.Log, "   %v\n", err)
			continue
		}
		return &record, joiner.ensure(published(record.Model, record.ModelSHA256), directory)
	}
	return nil, fmt.Errorf("round %d's merged weights could not be fetched from any closer who signed them: %w", number-1, err)
}

// planBase fetches and imports the model a run starts from, checking its files
// against the plan's pins; nil when the run bootstraps from random weights. Its
// training history must be completely recorded, since every round inherits it.
func (joiner *Joiner) planBase(ctx context.Context) (*MergeRecord, error) {
	start := joiner.plan.start()
	if start == nil {
		return nil, nil
	}
	directory, err := joiner.fetch(ctx, start.Model, "", start.ModelSHA256, start.Files, start.URL)
	if err != nil {
		return nil, fmt.Errorf("the plan's base: %w", err)
	}
	name := published(start.Model, start.ModelSHA256)
	if err := joiner.ensure(name, directory); err != nil {
		return nil, err
	}
	inspection, err := model.Inspect(joiner.models(), name)
	if err != nil {
		return nil, err
	}
	closure, err := model.ModelClosure(inspection)
	if err != nil {
		return nil, err
	}
	if !closure.Complete {
		return nil, fmt.Errorf("the plan's base %s has an incompletely recorded training history; a run inherits it, so it must be complete", start.Model)
	}
	return start, nil
}

func (joiner *Joiner) contribute(ctx context.Context, state State, action Action) error {
	name := joiner.modelName(action.Round, action.Unit+"-"+joiner.Identity)
	fmt.Fprintf(joiner.Log, "\n== Round %d (%s), unit %s: %s trains %s ==\n", action.Round, action.Stage, action.Unit, joiner.Identity, name)
	base, err := joiner.base(ctx, state, action.Round)
	if err != nil {
		return err
	}
	baseName := ""
	if base != nil {
		baseName = published(base.Model, base.ModelSHA256)
	}
	budget, err := joiner.plan.budget(action.Stage)
	if err != nil {
		return err
	}
	text, err := joiner.plan.unitCompose(action.Stage, action.Unit, baseName, pieceSeed(joiner.plan.Name, action.Round, action.Unit), budget)
	if err != nil {
		return err
	}
	if err := joiner.train(ctx, name, text); err != nil {
		return err
	}
	export, err := joiner.export(ctx, name)
	if err != nil {
		return err
	}
	tokens, err := joiner.maxTelemetry(name, "tokens")
	if err != nil {
		return err
	}
	files, err := fileHashes(export)
	if err != nil {
		return err
	}
	url, err := joiner.publish(ctx, export)
	if err != nil {
		return err
	}
	record := Submission{
		Kind: SubmissionKind, Schema: RecordSchema, Run: joiner.plan.Name, Round: action.Round, Stage: action.Stage, Unit: action.Unit,
		Identity: joiner.Identity, Model: name, ModelSHA256: files["model.safetensors"], Files: files, Tokens: int64(tokens),
		URL: url, CreatedAt: joiner.timestamp(), Plan: joiner.planSHA256,
	}
	if base != nil {
		record.Base, record.BaseSHA256 = &base.Model, &base.ModelSHA256
	}
	path := filepath.Join(roundDir(joiner.runDir, action.Round), "submissions", action.Unit+"-"+joiner.Identity+".yaml")
	won, err := joiner.commit(ctx, path, record, fmt.Sprintf("%s round %d: %s by %s", joiner.plan.Name, action.Round, action.Unit, joiner.Identity))
	if err != nil || won {
		return err
	}
	return fmt.Errorf("round %d unit %s: the runs repo did not accept this submission", action.Round, action.Unit)
}

func (joiner *Joiner) timestamp() string {
	return joiner.Now().UTC().Format("2006-01-02T15:04:05+00:00")
}

// probe is the held-out loss the verifier itself measures on an export's
// exact weights, never the contributor's own report. seed must be the unit's
// training seed, so the probe scores the records every replica held out.
func (joiner *Joiner) probe(ctx context.Context, stage, unit, export string, seed int) (model.Evaluation, error) {
	name := filepath.Base(export)
	if err := joiner.ensure(name, export); err != nil {
		return model.Evaluation{}, err
	}
	text, err := joiner.plan.unitCompose(stage, unit, "", seed, probeBudget)
	if err != nil {
		return model.Evaluation{}, err
	}
	path := filepath.Join(joiner.workDir(), "composes", fmt.Sprintf("%s-probe-%d.yaml", name, seed))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return model.Evaluation{}, err
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		return model.Evaluation{}, err
	}
	output, err := joiner.waldoOutput(ctx, "--json", "model", "train", name, path, "--evaluate-only")
	if err != nil {
		return model.Evaluation{}, err
	}
	var evaluation model.Evaluation
	if err := json.Unmarshal(output, &evaluation); err != nil {
		return model.Evaluation{}, fmt.Errorf("read evaluation of %s: %w", name, err)
	}
	return evaluation, nil
}

// verify returns the index of the replica most others agree with, among
// those agreeing with at least half of the others, or -1 without a majority. Honest replicas improve alike from the round's start; a
// free-rider does not, so gaps are judged against each round's progress.
func (joiner *Joiner) verify(ctx context.Context, stage, label string, exports []string, records []Submission, corpora []string, base *MergeRecord, round int, unit string) (int, []int, []int, *UnitEvidence, error) {
	seed := pieceSeed(joiner.plan.Name, round, unit)
	evidence := &UnitEvidence{Unit: unit}
	var start float64
	if base != nil {
		measured, err := joiner.probe(ctx, stage, unit, joiner.fetched(base.Identity, base.Model), seed)
		if err != nil {
			return -1, nil, nil, nil, err
		}
		start = measured.Loss
	} else {
		// A bootstrap starts from random weights, whose loss is about uniform over the vocabulary.
		vocabulary := joiner.plan.vocabulary()
		if vocabulary < 2 {
			return -1, nil, nil, nil, fmt.Errorf("the run file's architecture needs vocabulary_size to judge bootstrap replicas")
		}
		start = math.Log(float64(vocabulary))
	}
	evidence.StartLoss = start
	losses := make([]float64, len(exports))
	for index, export := range exports {
		measured, err := joiner.probe(ctx, stage, unit, export, seed)
		if err != nil {
			return -1, nil, nil, nil, err
		}
		losses[index] = measured.Loss
		evidence.EvaluationSetSHA256, evidence.CorpusBOMSHA256 = measured.Set.SHA256, measured.CorpusBOMSHA256
		evidence.Replicas = append(evidence.Replicas, ReplicaEvidence{Identity: records[index].Identity, ModelSHA256: records[index].ModelSHA256, HeldoutLoss: measured.Loss})
	}
	fmt.Fprintf(joiner.Log, "   round start probe loss %.4f on this unit's data\n", start)
	// A replica whose run BOM records another corpus selection did not train this unit.
	var elsewhere []int
	eligible := len(exports)
	for index := range exports {
		if corpora[index] != evidence.CorpusBOMSHA256 {
			elsewhere = append(elsewhere, index)
			eligible--
			fmt.Fprintf(joiner.Log, "   replica %d trained on corpus %.12s, not this unit's %.12s\n", index+1, corpora[index], evidence.CorpusBOMSHA256)
		}
	}
	agree := make([]int, len(exports))
	for first := range exports {
		for second := first + 1; second < len(exports); second++ {
			if slices.Contains(elsewhere, first) || slices.Contains(elsewhere, second) {
				continue
			}
			progress := math.Max(math.Max(start-losses[first], start-losses[second]), 1e-9)
			gap := math.Abs(losses[first]-losses[second]) / progress
			close := gap <= joiner.plan.Tolerance() || math.Abs(losses[first]-losses[second]) <= joiner.plan.Floor()
			verdict := "DISAGREE"
			if close {
				verdict = "agree"
				agree[first]++
				agree[second]++
			}
			measure := fmt.Sprintf("gap %.0f%% of the round's progress", gap*100)
			if progress <= 1e-9 {
				measure = fmt.Sprintf("no progress; gap %.4f", math.Abs(losses[first]-losses[second]))
			}
			fmt.Fprintf(joiner.Log, "   replicas %d vs %d: probe loss %.4f vs %.4f (%s)  %s\n",
				first+1, second+1, losses[first], losses[second], measure, verdict)
		}
	}
	var accepted, rejected []string
	var disagreeing []int
	winner := -1
	for index, count := range agree {
		if slices.Contains(elsewhere, index) {
			continue
		}
		if count >= eligible/2 {
			// The most widely agreed replica wins, so one that some peers dispute never beats one none dispute.
			if winner < 0 || count > agree[winner] {
				winner = index
			}
			accepted = append(accepted, strconv.Itoa(index+1))
		} else {
			rejected = append(rejected, strconv.Itoa(index+1))
			disagreeing = append(disagreeing, index)
		}
	}
	if len(accepted) <= eligible/2 || eligible < joiner.plan.Replicas/2+1 {
		fmt.Fprintf(joiner.Log, "%s: no majority of replicas agree within %.0f%% of the round's progress; not merging\n", label, joiner.plan.Tolerance()*100)
		return -1, nil, elsewhere, evidence, nil
	}
	message := fmt.Sprintf("   %s: accepted replica %d", label, winner+1)
	if len(rejected) > 0 {
		message += ", rejected replica " + strings.Join(rejected, ", ")
	}
	fmt.Fprintln(joiner.Log, message)
	return winner, disagreeing, elsewhere, evidence, nil
}

// checkProvenance imports a submission's published package and verifies its
// training closure is exactly the round base's plus its own run BOMs, so no
// replica can drop or invent a corpus from the history every later round inherits.
// It returns the submission's one training run.
func (joiner *Joiner) checkProvenance(record Submission, directory string, base *MergeRecord) (model.RunBOM, error) {
	name := published(record.Model, record.ModelSHA256)
	if err := joiner.ensure(name, directory); err != nil {
		return model.RunBOM{}, err
	}
	inspection, err := model.Inspect(joiner.models(), name)
	if err != nil {
		return model.RunBOM{}, err
	}
	var parent *model.Inspection
	if base != nil {
		inspected, err := model.Inspect(joiner.models(), published(base.Model, base.ModelSHA256))
		if err != nil {
			return model.RunBOM{}, err
		}
		parent = &inspected
	}
	if err := model.CheckClosure(inspection, parent); err != nil {
		return model.RunBOM{}, err
	}
	runs, err := model.PublishedRunBOMs(inspection)
	if err != nil {
		return model.RunBOM{}, err
	}
	if len(runs) != 1 {
		return model.RunBOM{}, fmt.Errorf("a unit trains in one run, and %s carries %d run BOMs", record.Model, len(runs))
	}
	return runs[0], nil
}

// budgets sets aside replicas whose run trained with other parameters than a
// majority of replicas, or another step count than the plan gives the unit: a
// replica that trained a fraction of the budget and claims all of it.
func (joiner *Joiner) budgets(stage, unit string, exports []string, present []Submission, runs []model.RunBOM, refused []DroppedUnit) ([]string, []Submission, []model.RunBOM, []DroppedUnit) {
	steps := int64(0)
	if stage == "pretrain" {
		steps, _ = joiner.plan.pretrainSteps()
	}
	counts := map[string]int{}
	keys := make([]string, len(runs))
	for index, run := range runs {
		data, _ := json.Marshal(run.Parameters)
		keys[index] = string(data)
		counts[keys[index]]++
	}
	var keptExports []string
	var keptPresent []Submission
	var keptRuns []model.RunBOM
	for index, run := range runs {
		reason := ""
		if counts[keys[index]]*2 <= len(runs) {
			reason = "its run's training parameters differ from a majority of replicas'"
		} else if steps > 0 && run.Parameters.Steps != steps {
			reason = fmt.Sprintf("its run trained %d steps; the plan gives this unit %d", run.Parameters.Steps, steps)
		}
		if reason != "" {
			refused = append(refused, DroppedUnit{Unit: unit, Reason: reason, Identities: []string{present[index].Identity}})
			continue
		}
		keptExports, keptPresent, keptRuns = append(keptExports, exports[index]), append(keptPresent, present[index]), append(keptRuns, run)
	}
	return keptExports, keptPresent, keptRuns, refused
}

// screen sets aside the replicas that cannot be judged on their loss: one that
// did not start from the round's merged weights, and one whose tokens trained
// differ from a majority's. Identical weights are agreement, not copying:
// training on the same hardware and software can reproduce weights bit for bit.
func screen(records []Submission, base *MergeRecord) ([]Submission, []DroppedUnit) {
	var kept []Submission
	var refused []DroppedUnit
	refuse := func(record Submission, reason string) {
		refused = append(refused, DroppedUnit{Unit: record.Unit, Reason: reason, Identities: []string{record.Identity}})
	}
	tokens := map[int64]int{}
	for _, record := range records {
		tokens[record.Tokens]++
	}
	for _, record := range records {
		switch {
		case (base == nil) != (record.BaseSHA256 == nil) || (base != nil && *record.BaseSHA256 != base.ModelSHA256):
			refuse(record, "it did not start from the round's merged weights")
		case tokens[record.Tokens]*2 <= len(records):
			refuse(record, "its tokens trained differ from a majority of replicas'")
		default:
			kept = append(kept, record)
		}
	}
	return kept, refused
}

func (joiner *Joiner) close(ctx context.Context, state State, action Action) error {
	fmt.Fprintf(joiner.Log, "\n== Round %d (%s): %s closes it, one of %d closers ==\n", action.Round, action.Stage, joiner.Identity, joiner.plan.Closing())
	base, err := joiner.base(ctx, state, action.Round)
	if err != nil {
		return err
	}
	round := state.Rounds[action.Round]
	var accepted []Submission
	var evidence []UnitEvidence
	var rejected []DroppedUnit
	dropped := []DroppedUnit{}
	for _, unit := range joiner.plan.RoundUnits(action.Stage) {
		records := round.replicas(unit, joiner.plan.Replicas)
		var identities []string
		for _, record := range records {
			identities = append(identities, record.Identity)
		}
		if len(records) < joiner.plan.Replicas {
			dropped = append(dropped, DroppedUnit{Unit: unit, Reason: fmt.Sprintf("%d of %d replicas by the deadline", len(records), joiner.plan.Replicas)})
			continue
		}
		// Each bad replica costs only itself; the rest still decide the unit.
		kept, refused := screen(records, base)
		var exports []string
		var present []Submission
		var runs []model.RunBOM
		for _, record := range kept {
			directory, err := joiner.fetch(ctx, record.Model, record.Identity, record.ModelSHA256, record.Files, record.URL)
			if err != nil {
				fmt.Fprintf(joiner.Log, "   %v\n", err)
				refused = append(refused, DroppedUnit{Unit: unit, Reason: "its files could not be fetched and checked", Identities: []string{record.Identity}})
				continue
			}
			run, err := joiner.checkProvenance(record, directory, base)
			if err != nil {
				fmt.Fprintf(joiner.Log, "   %s's %s: %v\n", record.Identity, unit, err)
				refused = append(refused, DroppedUnit{Unit: unit, Reason: "its training provenance does not check: " + err.Error(), Identities: []string{record.Identity}})
				continue
			}
			exports = append(exports, directory)
			present = append(present, record)
			runs = append(runs, run)
		}
		exports, present, runs, refused = joiner.budgets(action.Stage, unit, exports, present, runs, refused)
		var corpora []string
		for _, run := range runs {
			corpora = append(corpora, run.CorpusBOMSHA256)
		}
		rejected = append(rejected, refused...)
		// A unit needs a majority of its planned replicas left to compare.
		if len(present) < joiner.plan.Replicas/2+1 {
			dropped = append(dropped, DroppedUnit{Unit: unit, Reason: fmt.Sprintf("%d of %d replicas left after rejections", len(present), joiner.plan.Replicas), Identities: identities})
			continue
		}
		winner, disagreeing, elsewhere, unitEvidence, err := joiner.verify(ctx, action.Stage, "unit "+unit, exports, present, corpora, base, action.Round, unit)
		if unitEvidence != nil {
			evidence = append(evidence, *unitEvidence)
		}
		if err != nil {
			return err
		}
		for _, index := range elsewhere {
			rejected = append(rejected, DroppedUnit{Unit: unit, Reason: "its run BOM records a corpus selection other than its unit's", Identities: []string{present[index].Identity}})
		}
		if winner < 0 {
			dropped = append(dropped, DroppedUnit{Unit: unit, Reason: "replicas disagree", Identities: identities})
			continue
		}
		for _, index := range disagreeing {
			rejected = append(rejected, DroppedUnit{Unit: unit, Reason: "its held-out loss disagrees with the other replicas", Identities: []string{present[index].Identity}})
		}
		accepted = append(accepted, present[winner])
	}
	if len(accepted) == 0 {
		return fmt.Errorf("round %d: no unit was accepted; the run cannot continue without new submissions", action.Round)
	}
	// Named by the units it merges, so a retried close with a different accepted set never reuses another set's merge.
	digest := sha256.New()
	for _, submission := range accepted {
		fmt.Fprintf(digest, "%s %s\n", submission.Unit, submission.ModelSHA256)
	}
	name := joiner.modelName(action.Round, "merge-"+hex.EncodeToString(digest.Sum(nil))[:12])
	target := filepath.Join(joiner.workDir(), "exports", name)
	method := "single"
	if _, err := os.Stat(filepath.Join(target, "model.safetensors")); os.IsNotExist(err) {
		_ = os.RemoveAll(target)
		if len(accepted) == 1 {
			// One accepted unit (always so for the bootstrap) is the merge.
			if err := copyDirectory(joiner.fetched(accepted[0].Identity, accepted[0].Model), target); err != nil {
				return err
			}
		} else if err := joiner.merge(ctx, name, accepted, target); err != nil {
			return err
		}
	}
	if len(accepted) > 1 {
		method = "weighted-mean-f32-v1"
	}
	if err := joiner.ensure(name, target); err != nil {
		return err
	}
	fmt.Fprintf(joiner.Log, "-- Merge round %d: %d units accepted, %d dropped\n", action.Round, len(accepted), len(dropped))
	if err := joiner.writeRound(target, action, base, method, accepted); err != nil {
		return err
	}
	var gate *GateResult
	if action.Round < len(state.Slots) && state.Slots[action.Round].Gate != nil {
		if gate, err = joiner.checkGate(ctx, *state.Slots[action.Round].Gate, action, name, target); err != nil {
			return err
		}
	}
	files, err := fileHashes(target)
	if err != nil {
		return err
	}
	url, err := joiner.publish(ctx, target)
	if err != nil {
		return err
	}
	record := MergeRecord{
		Kind: MergeKind, Schema: RecordSchema, Run: joiner.plan.Name, Round: action.Round, Stage: action.Stage, Identity: joiner.Identity,
		Method: method, Dropped: dropped, Rejected: rejected, Evidence: evidence, Gate: gate, Model: name, ModelSHA256: files["model.safetensors"], Files: files, URL: url, ClosedAt: joiner.timestamp(), Plan: joiner.planSHA256,
	}
	if base != nil {
		record.Base, record.BaseSHA256 = &base.Model, &base.ModelSHA256
	}
	for _, submission := range accepted {
		record.Accepted = append(record.Accepted, AcceptedUnit{Unit: submission.Unit, Identity: submission.Identity, Model: submission.Model, ModelSHA256: submission.ModelSHA256, Tokens: submission.Tokens})
	}
	path := filepath.Join(roundDir(joiner.runDir, action.Round), "merges", joiner.Identity+".yaml")
	won, err := joiner.commit(ctx, path, record, fmt.Sprintf("%s round %d: merge by %s", joiner.plan.Name, action.Round, joiner.Identity))
	if err != nil {
		return err
	}
	if !won {
		return fmt.Errorf("round %d: the runs repo did not accept this merge", action.Round)
	}
	return nil
}

// GateReportFile is an evaluation gate's report in a merge package.
const GateReportFile = "GATE-REPORT.json"

// checkGate measures the merged weights against the round's gate. A held-out
// loss gate averages the merge's loss on every unit's held-out records; an
// evaluation gate runs the plan's evaluator on the merge, then waldo model gate,
// which checks the evaluation against everything the lineage trained on and
// writes its report into the package, so the merge publishes the bar it met.
func (joiner *Joiner) checkGate(ctx context.Context, gate Gate, action Action, name, target string) (*GateResult, error) {
	result := &GateResult{After: gate.After}
	if gate.HeldoutLossAtMost != nil {
		units := joiner.plan.RoundUnits(action.Stage)
		total := 0.0
		for _, unit := range units {
			measured, err := joiner.probe(ctx, action.Stage, unit, target, pieceSeed(joiner.plan.Name, action.Round, unit))
			if err != nil {
				return nil, err
			}
			total += measured.Loss
		}
		mean := total / float64(len(units))
		result.Value, result.Passed = &mean, mean <= *gate.HeldoutLossAtMost
		fmt.Fprintf(joiner.Log, "   gate after %s: held-out loss %.4f, at most %.4f: passed %t\n", gate.After, mean, *gate.HeldoutLossAtMost, result.Passed)
		return result, nil
	}
	scratch, err := os.MkdirTemp("", "waldo-join-gate-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(scratch)
	evaluation := filepath.Join(joiner.runDir, gate.Evaluation)
	results := filepath.Join(scratch, "results.json")
	command := exec.CommandContext(ctx, gate.Evaluator[0], append(append([]string{}, gate.Evaluator[1:]...), name, evaluation, results)...)
	command.Dir = joiner.runDir
	command.Env = append(os.Environ(), "WALDO="+joiner.Waldo, "WALDO_CONFIG="+joiner.config())
	command.Stdout, command.Stderr = joiner.Log, joiner.Log
	fmt.Fprintf(joiner.Log, "$ %s %s %s %s\n", strings.Join(gate.Evaluator, " "), name, gate.Evaluation, results)
	if err := command.Run(); err != nil {
		return nil, fmt.Errorf("gate after %s: evaluator: %w", gate.After, err)
	}
	report := filepath.Join(target, GateReportFile)
	_ = os.Remove(report)
	// waldo model gate exits non-zero when the model does not pass; the report says so either way,
	// so its output is kept back unless it wrote no report.
	command = exec.CommandContext(ctx, joiner.Waldo, "--json", "model", "gate", name, evaluation, results, report)
	command.Env = append(os.Environ(), "WALDO_CONFIG="+joiner.config())
	var gateOutput bytes.Buffer
	command.Stdout, command.Stderr = io.Discard, &gateOutput
	fmt.Fprintf(joiner.Log, "$ waldo model gate %s %s %s %s\n", name, gate.Evaluation, filepath.Base(results), GateReportFile)
	_ = command.Run()
	data, err := os.ReadFile(report)
	if err != nil {
		return nil, fmt.Errorf("gate after %s: waldo model gate wrote no report: %s", gate.After, bytes.TrimSpace(gateOutput.Bytes()))
	}
	var written struct {
		Promotion struct {
			Passed   bool     `json:"passed"`
			Failures []string `json:"failures"`
		} `json:"promotion"`
	}
	if err := json.Unmarshal(data, &written); err != nil {
		return nil, fmt.Errorf("gate after %s: read report: %w", gate.After, err)
	}
	result.Passed, result.Report = written.Promotion.Passed, GateReportFile
	fmt.Fprintf(joiner.Log, "   gate after %s: evaluation passed %t %v\n", gate.After, result.Passed, written.Promotion.Failures)
	return result, nil
}

// writeRound writes the merge package's RoundRecord.
func (joiner *Joiner) writeRound(target string, action Action, base *MergeRecord, method string, accepted []Submission) error {
	round := RoundRecord{Kind: "openwaldo-bom", Schema: 1, Subject: "join-round", RunsRepository: joiner.url,
		Run: joiner.plan.Name, Round: action.Round, Stage: action.Stage, PlanSHA256: joiner.planSHA256, Method: method}
	if base != nil {
		round.BaseSHA256 = &base.ModelSHA256
	}
	for _, submission := range accepted {
		path := filepath.Join(roundDir(joiner.runDir, action.Round), "submissions", submission.Unit+"-"+submission.Identity+".yaml")
		digest, err := hashFile(path)
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(joiner.clone, path)
		if err != nil {
			return err
		}
		round.Accepted = append(round.Accepted, RoundSubmission{Unit: submission.Unit, Identity: submission.Identity,
			ModelSHA256: submission.ModelSHA256, Record: filepath.ToSlash(relative), RecordSHA256: digest})
	}
	data, err := json.MarshalIndent(round, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(target, model.RoundFile), append(data, '\n'), 0o644)
}

// merge imports the accepted exports and composes their token-weighted
// merge in waldo, whose records pin every parent and their shared ancestry;
// the exported merge carries that lineage to the next round.
func (joiner *Joiner) merge(ctx context.Context, name string, accepted []Submission, target string) error {
	interaction, err := joiner.plan.interaction()
	if err != nil {
		return err
	}
	text := "kind: waldo-model-compose\nschema: 1\n" + interaction + "base:\n  merge:\n"
	for _, submission := range accepted {
		name := published(submission.Model, submission.ModelSHA256)
		if err := joiner.ensure(name, joiner.fetched(submission.Identity, submission.Model)); err != nil {
			return err
		}
		text += fmt.Sprintf("    - model: %s\n      weight: %d\n", name, submission.Tokens)
	}
	if err := joiner.train(ctx, name, text); err != nil {
		return err
	}
	_, err = joiner.export(ctx, name)
	return err
}

func copyDirectory(source, target string) error {
	entries, err := os.ReadDir(source)
	if err != nil {
		return err
	}
	partial := target + ".partial"
	_ = os.RemoveAll(partial)
	if err := os.MkdirAll(partial, 0o755); err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.Type().IsRegular() && !strings.HasPrefix(entry.Name(), ".") {
			if err := copyFile(filepath.Join(source, entry.Name()), filepath.Join(partial, entry.Name())); err != nil {
				return err
			}
		}
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	return os.Rename(partial, target)
}

func copyFile(source, target string) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.Create(target)
	if err != nil {
		return err
	}
	if _, err := io.Copy(output, input); err != nil {
		_ = output.Close()
		return err
	}
	return output.Close()
}
