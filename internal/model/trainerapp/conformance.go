// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
)

// ConformanceAttempt supplies an isolated run and a seekable WALDO stream.
// SourceAt receives the first sequence ordinal, not a token or record offset.
type ConformanceAttempt struct {
	Client      *Client
	Begin       Begin
	Evaluations []EvaluationRecord
	SourceAt    func(int64) SequenceSource
}
type ConformanceFactory func(context.Context, string, bool) (ConformanceAttempt, error)
type ConformanceReport struct {
	ResumeBitIdentical bool    `json:"resume_bit_identical"`
	CheckpointStep     int64   `json:"checkpoint_step"`
	ComparedLosses     int     `json:"compared_losses"`
	MaximumLossDelta   float64 `json:"maximum_loss_delta"`
	AbsoluteTolerance  float64 `json:"absolute_tolerance"`
	RelativeTolerance  float64 `json:"relative_tolerance"`
}

// CheckConformance runs a control, interrupts a second run at its first durable
// checkpoint and resumes it, then runs another batch decomposition. The factory
// must return independent directories, the same recipe/data/seed, and required
// determinism. The alternative flag changes execution parameters only. Trainers
// should use uneven supervised-token counts across micro-batches in this fixture.
// Exact weight-file equality is stricter than tensor equality (metadata included).
func CheckConformance(ctx context.Context, factory ConformanceFactory, absolute, relative float64) (ConformanceReport, error) {
	report := ConformanceReport{AbsoluteTolerance: absolute, RelativeTolerance: relative}
	if factory == nil || !finite(absolute) || !finite(relative) || absolute < 0 || relative < 0 {
		return report, fmt.Errorf("invalid conformance configuration")
	}
	makeAttempt := func(name string, alternate bool) (ConformanceAttempt, error) {
		a, err := factory(ctx, name, alternate)
		if err != nil {
			return a, err
		}
		if a.Client == nil || a.SourceAt == nil || a.Begin.Resume != nil || a.Begin.Determinism != "required" || a.Begin.Data.Steps < 2 {
			return a, fmt.Errorf("conformance needs fresh deterministic runs of at least two steps")
		}
		return a, nil
	}
	control, err := makeAttempt("control", false)
	if err != nil {
		return report, err
	}
	interrupted, err := makeAttempt("interrupted", false)
	if err != nil {
		return report, err
	}
	alternative, err := makeAttempt("decomposition", true)
	if err != nil {
		return report, err
	}
	if control.Begin.CheckpointDirectory == interrupted.Begin.CheckpointDirectory || control.Begin.Paths.Artifacts == interrupted.Begin.Paths.Artifacts || control.Begin.CheckpointDirectory == alternative.Begin.CheckpointDirectory || control.Begin.Paths.Artifacts == alternative.Begin.Paths.Artifacts || interrupted.Begin.CheckpointDirectory == alternative.Begin.CheckpointDirectory || interrupted.Begin.Paths.Artifacts == alternative.Begin.Paths.Artifacts {
		return report, fmt.Errorf("conformance runs must be isolated")
	}
	comparable := func(a ConformanceAttempt) error {
		x, y := control.Begin, a.Begin
		xr, _, err := control.Client.parameters.Split(x.Parameters)
		if err != nil {
			return err
		}
		yr, _, err := a.Client.parameters.Split(y.Parameters)
		if err != nil {
			return err
		}
		if x.CheckpointIdentity.TrainerSHA256 != y.CheckpointIdentity.TrainerSHA256 || x.CheckpointIdentity.TokenizerSHA256 != y.CheckpointIdentity.TokenizerSHA256 || x.CheckpointIdentity.EnvironmentSHA256 != y.CheckpointIdentity.EnvironmentSHA256 || digest(xr) != digest(yr) || digest(x.Data) != digest(y.Data) || x.ArchitectureSHA256 != y.ArchitectureSHA256 || digest(x.Tokenizer) != digest(y.Tokenizer) || digest(control.Evaluations) != digest(a.Evaluations) || x.Objective != y.Objective || digest(x.Initialization) != digest(y.Initialization) {
			return fmt.Errorf("conformance recipe or data differs")
		}
		return nil
	}
	if err = comparable(interrupted); err != nil {
		return report, err
	}
	if err = comparable(alternative); err != nil {
		return report, err
	}
	if control.Begin.ParametersSHA256 != interrupted.Begin.ParametersSHA256 || control.Begin.Batch != interrupted.Begin.Batch || control.Begin.World != interrupted.Begin.World {
		return report, fmt.Errorf("interrupted control execution differs")
	}
	if control.Begin.Batch == alternative.Begin.Batch {
		return report, fmt.Errorf("conformance needs two distinct batch decompositions")
	}
	run := func(a ConformanceAttempt, sink func(Event) error) (Complete, error) {
		ordinal := int64(0)
		if a.Begin.Resume != nil {
			ordinal = a.Begin.Resume.Step * a.Begin.Data.GlobalBatchSequences
		}
		return a.Client.Train(ctx, a.Begin, a.Evaluations, a.SourceAt(ordinal), sink)
	}
	losses := map[int64]float64{}
	full, err := run(control, func(e Event) error {
		if e.Kind == "progress" {
			losses[e.Step] = *e.Loss
		}
		return nil
	})
	if err != nil {
		return report, err
	}
	stop := errors.New("conformance interruption after durable checkpoint")
	var ref *CheckpointReference
	_, err = run(interrupted, func(e Event) error {
		if e.DurableCheckpoint != nil {
			copy := *e.DurableCheckpoint
			ref = &copy
			report.CheckpointStep = e.Step
			return stop
		}
		return nil
	})
	if !errors.Is(err, stop) || ref == nil || report.CheckpointStep <= 0 || report.CheckpointStep >= interrupted.Begin.Data.Steps {
		return report, fmt.Errorf("conformance requires an intermediate durable checkpoint: %w", err)
	}
	interrupted.Begin.Resume, err = LoadCheckpoint(interrupted.Begin, *ref)
	if err != nil {
		return report, err
	}
	resumed, err := run(interrupted, nil)
	if err != nil {
		return report, err
	}
	weightHashes := func(a ConformanceAttempt, result Complete) (map[string]string, error) {
		root, err := os.OpenRoot(a.Begin.Paths.Artifacts)
		if err != nil {
			return nil, err
		}
		defer root.Close()
		hashes := map[string]string{}
		for _, file := range result.Artifacts {
			if file.Role == "weights" {
				if err := copyVerified(root, file, ""); err != nil {
					return nil, err
				}
				hashes[file.Path] = file.SHA256
			}
		}
		if len(hashes) == 0 {
			return nil, fmt.Errorf("conformance has no terminal weights")
		}
		return hashes, nil
	}
	expected, err := weightHashes(control, full)
	if err != nil {
		return report, err
	}
	actual, err := weightHashes(interrupted, resumed)
	if err != nil {
		return report, err
	}
	if digest(expected) != digest(actual) {
		return report, fmt.Errorf("resumed terminal weight files are not bit-identical")
	}
	report.ResumeBitIdentical = true
	alternateLosses := map[int64]float64{}
	_, err = run(alternative, func(e Event) error {
		if e.Kind == "progress" {
			alternateLosses[e.Step] = *e.Loss
		}
		return nil
	})
	if err != nil {
		return report, err
	}
	if len(losses) == 0 || len(losses) != len(alternateLosses) {
		return report, fmt.Errorf("decomposition loss reports differ or are absent")
	}
	for step, want := range losses {
		got, ok := alternateLosses[step]
		if !ok {
			return report, fmt.Errorf("missing decomposition loss at step %d", step)
		}
		delta := math.Abs(want - got)
		report.MaximumLossDelta = math.Max(report.MaximumLossDelta, delta)
		report.ComparedLosses++
		if delta > absolute+relative*math.Max(math.Abs(want), math.Abs(got)) {
			return report, fmt.Errorf("decomposition loss differs at step %d: %.17g vs %.17g", step, want, got)
		}
	}
	return report, nil
}
