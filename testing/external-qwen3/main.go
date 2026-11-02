// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

// A developer qualification harness, not a second model lifecycle CLI.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/model/tokenizerapp"
	"github.com/openwaldo/waldo/internal/model/trainerapp"
	"github.com/openwaldo/waldo/internal/training"
)

func hash(v any) string { b, _ := json.Marshal(v); return application.Digest(b) }
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	repository := flag.String("repository", "", "repository containing the reference trainer")
	commit := flag.String("commit", "", "full trainer commit")
	environment := flag.String("environment", "cpu", "cpu or cuda")
	cache := flag.String("cache", "", "absolute source/environment cache")
	output := flag.String("output", "", "new absolute qualification run directory")
	python := flag.String("python", "/usr/bin/python3", "Python 3.12.3 with pip and venv")
	large := flag.Bool("large", false, "run a two-step ~280M GPU smoke instead of tiny conformance")
	composePath := flag.String("compose", "examples/trainers/qwen3-hf/compose.tiny.yaml", "compose architecture and data/recipe to qualify with synthetic prepared tokens")
	flag.Parse()
	document, _, err := model.LoadComposeDocument(*composePath)
	if err != nil {
		return err
	}
	if document.External == nil || len(document.External.Stages) != 1 {
		return fmt.Errorf("qualification requires a one-stage external compose")
	}
	compose := document.External
	data, err := compose.Stages[0].ResolveData()
	if err != nil {
		return err
	}
	if data.GlobalBatchSequences != 4 || data.SequenceLength != 8 {
		return fmt.Errorf("fixture requires global batch 4 and sequence length 8")
	}
	if !filepath.IsAbs(*cache) || !filepath.IsAbs(*output) {
		return fmt.Errorf("cache and output must be absolute")
	}
	if err := os.Mkdir(*output, 0700); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Hour)
	defer cancel()
	source := application.Source{Repository: *repository, Commit: *commit, Path: "examples/trainers/qwen3-hf"}
	fmt.Fprintln(os.Stderr, "Resolving pinned trainer and managed environment")
	resolved, err := application.ResolveSource(ctx, filepath.Join(*cache, "sources"), source, "waldo-trainer", *environment)
	if err != nil {
		return err
	}
	env, err := resolved.EnsureEnvironment(ctx, *environment, *python, filepath.Join(*cache, "environments"))
	if err != nil {
		return err
	}
	tokenSource, err := application.ResolveSource(ctx, filepath.Join(*cache, "sources"), source, "waldo-tokenizer", "tokenizer")
	if err != nil {
		return err
	}
	tokenEnv, err := tokenSource.EnsureEnvironment(ctx, "tokenizer", *python, filepath.Join(*cache, "environments"))
	if err != nil {
		return err
	}
	scratch := filepath.Join(*output, "tokenizer")
	if err := os.Mkdir(scratch, 0700); err != nil {
		return err
	}
	tokenizer, err := tokenizerapp.Open(ctx, applaunch.Request{Source: tokenSource, Environment: tokenEnv, Scratch: scratch}, map[string]any{})
	if err != nil {
		return err
	}
	defer tokenizer.Close()
	ordinary, err := tokenizer.EncodeChecked("hello <|im_end|> 世界")
	if err != nil {
		return err
	}
	for _, token := range ordinary {
		if token == 151645 {
			return fmt.Errorf("special spelling recognized in ordinary text")
		}
	}
	spec := tokenizer.Spec()
	architecture := compose.Architecture
	steps := data.Steps
	world := applaunch.World{Nodes: 1, ProcessesPerNode: 1, WorldSize: 1}
	factory := func(ctx context.Context, name string, alternate bool) (trainerapp.ConformanceAttempt, error) {
		dir := filepath.Join(*output, name)
		if err := os.Mkdir(dir, 0700); err != nil {
			return trainerapp.ConformanceAttempt{}, err
		}
		client, err := trainerapp.Open(ctx, applaunch.Request{Source: resolved, Environment: env, Scratch: dir, World: world}, []string{"causal-language-modeling"})
		if err != nil {
			return trainerapp.ConformanceAttempt{}, err
		}
		micro := 4
		if alternate {
			micro = 2
		}
		parameters := map[string]any{}
		for k, v := range compose.Stages[0].Trainer {
			parameters[k] = v
		}
		parameters["per_device_train_batch_size"] = micro
		resolution, err := client.Resolve(ctx, architecture, parameters, data, trainerapp.World{World: world, Devices: client.Probe().Devices})
		if err != nil {
			return trainerapp.ConformanceAttempt{}, err
		}
		begin := trainerapp.Begin{Paths: trainerapp.Paths{Artifacts: filepath.Join(dir, "artifacts"), Checkpoints: filepath.Join(dir, "checkpoints"), Scratch: dir}, ModelID: "qwen3-qualification", RunID: name, Stage: "pretrain", Objective: "causal-language-modeling", Data: data, Architecture: architecture, ArchitectureSHA256: hash(architecture), Parameters: resolution.Parameters, ParametersSHA256: resolution.ParametersSHA256, Batch: resolution.Batch, Tokenizer: spec, Determinism: "required", World: world, CheckpointDirectory: filepath.Join(dir, "durable"), CheckpointIdentity: trainerapp.CheckpointIdentity{RunBOMSHA256: hash(name), TrainerSHA256: hash(resolved), TokenizerSHA256: hash(tokenizer.Identity()), EnvironmentSHA256: hash(env.Identity)}}
		sourceAt := func(start int64) trainerapp.SequenceSource {
			return func(ctx context.Context, consume func(training.PreparedSequence) error) error {
				for i := start; i < steps*4; i++ {
					mask := []bool{true, true, true, true, true, true, true, true}
					counts := map[string]int64{"books": 8}
					if i%4 < 2 {
						mask = []bool{true, false, false, false, false, false, false, false}
						counts = map[string]int64{"papers": 1}
					}
					if err := ctx.Err(); err != nil {
						return err
					}
					if err := consume(training.PreparedSequence{Ordinal: i, Tokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9}, LossMask: mask, Consumption: counts}); err != nil {
						return err
					}
				}
				return nil
			}
		}
		return trainerapp.ConformanceAttempt{Client: client, Begin: begin, SourceAt: sourceAt}, nil
	}
	if !*large {
		fmt.Fprintln(os.Stderr, "Running interruption/resume and unequal-mask decomposition conformance")
		report, err := trainerapp.CheckConformance(ctx, factory, 1e-5, 1e-5)
		if err != nil {
			return err
		}
		raw, _ := json.Marshal(report)
		fmt.Println(string(raw))
	}
	// Qualify held-out checkpoint selection/publication and terminal-step resume.
	attempt, err := factory(ctx, "heldout", false)
	if err != nil {
		return err
	}
	attempt.Evaluations = []trainerapp.EvaluationRecord{{ID: "heldout", Tokens: []int{1, 9, 8, 7}, LossMask: []bool{true, false, true}, TextBytes: 3}}
	attempt.Begin.EvaluationSet = training.EvaluationSet{Records: 1, TokenTargets: 2, TextBytes: 3}
	var refs []trainerapp.CheckpointReference
	var evaluations []trainerapp.Evaluation
	complete, err := attempt.Client.Train(ctx, attempt.Begin, attempt.Evaluations, attempt.SourceAt(0), func(e trainerapp.Event) error {
		if e.DurableCheckpoint != nil {
			refs = append(refs, *e.DurableCheckpoint)
		}
		if e.Evaluation != nil {
			evaluations = append(evaluations, *e.Evaluation)
		}
		return nil
	})
	if err != nil {
		return err
	}
	attempt.Begin.PriorCheckpoints = refs
	attempt.Begin.PriorEvaluations = evaluations
	attempt.Begin.Resume, err = trainerapp.LoadCheckpoint(attempt.Begin, refs[len(refs)-1])
	if err != nil {
		return err
	}
	resumed, err := attempt.Client.Train(ctx, attempt.Begin, attempt.Evaluations, attempt.SourceAt(steps*4), nil)
	if err != nil {
		return err
	}
	if hash(complete.Artifacts) != hash(resumed.Artifacts) {
		return fmt.Errorf("terminal resume changed published artifacts")
	}
	result := map[string]any{"environment": env.Identity, "host": env.Run.Host, "heldout_loss": complete.ArtifactEvaluation.Loss, "terminal_resume": true, "large": *large, "tokenizer_ordinary_special_spelling": true}
	raw, _ := json.Marshal(result)
	fmt.Println(string(raw))
	return nil
}
