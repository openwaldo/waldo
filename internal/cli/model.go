// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/openwaldo/waldo/internal/calibration"
	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/disclosure"
	"github.com/openwaldo/waldo/internal/inference"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/modelexport"
	"github.com/openwaldo/waldo/internal/modelquant"
	"github.com/openwaldo/waldo/internal/shard"
	"github.com/openwaldo/waldo/internal/signing"
	"github.com/openwaldo/waldo/internal/training"
	"golang.org/x/term"
)

func runModelForecast(context Context, args []string, stdout, stderr io.Writer) error {
	if len(args) == 1 {
		isCompose, err := model.IsComposeFile(args[0])
		if err != nil {
			return err
		}
		if isCompose {
			return runModelComposeForecast(context, args[0], stdout, stderr)
		}
	}
	return runModelIndexForecast(context, args, stdout, stderr)
}

func runModelComposeForecast(context Context, path string, stdout, progress io.Writer) error {
	compose, composePath, err := model.LoadCompose(path)
	if err != nil {
		return err
	}
	builder, err := configuredModelBuilder(context, progress)
	if err != nil {
		return err
	}
	compose, err = builder.ResolveCompose(context.Execution, compose, false)
	if err != nil {
		return err
	}
	calibrations, err := configuredForecastCalibration()
	if err != nil {
		return err
	}
	report, err := model.ForecastComposeWithCalibration(compose, calibrations)
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Compose  string                 `json:"compose"`
			Forecast model.ResourceForecast `json:"forecast"`
		}{Compose: composePath, Forecast: report})
	}
	fmt.Fprintf(stdout, "COMPOSE:     %s\n", composePath)
	writeModelForecast(stdout, report)
	return nil
}

func runModelIndexForecast(context Context, paths []string, stdout, warnings io.Writer) error {
	targets, err := resolveIndexArgumentsWithWarning(context.Execution, paths, warnings)
	if err != nil {
		return err
	}
	policy, err := corpus.NewLicensePolicy(nil, nil)
	if err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	bom, err := corpus.BuildBOM(context.Execution, targets, policy, cache)
	if err != nil {
		return err
	}
	calibrations, err := configuredForecastCalibration()
	if err != nil {
		return err
	}
	preset, report, err := model.ForecastIndexSelectionWithCalibration(bom.Totals.Tokens, calibrations)
	if err != nil {
		return err
	}
	parameters, err := preset.Architecture.Forecast()
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Index      any                    `json:"index"`
			Paths      []string               `json:"paths"`
			Preset     string                 `json:"preset"`
			Parameters uint64                 `json:"approximate_parameters"`
			Tokens     int64                  `json:"tokens"`
			Budget     string                 `json:"budget"`
			Forecast   model.ResourceForecast `json:"forecast"`
		}{Index: bom.Index, Paths: bom.Paths, Preset: preset.Name, Parameters: parameters.ApproximateParameters, Tokens: bom.Totals.Tokens, Budget: "one-pass", Forecast: report})
	}
	fmt.Fprintf(stdout, "MODEL:       %s\n", preset.Name)
	fmt.Fprintln(stdout, "BUDGET:      one pass")
	writeModelForecast(stdout, report)
	return nil
}

func writeModelForecast(stdout io.Writer, report model.ResourceForecast) {
	fmt.Fprintf(stdout, "PARAMETERS:  %s\n", humanModelParameters(report.ApproximateParameters))
	fmt.Fprintf(stdout, "TOKENS:      %s\n", humanCount(report.PlannedTokens))
	fmt.Fprintln(stdout)

	type row struct {
		manufacturer string
		accelerator  string
		GPUs         string
		nodes        string
		memory       string
		duration     string
	}
	rows := make([]row, 0, len(report.Configurations))
	observedRuns := 0
	manufacturerWidth, acceleratorWidth := len("MFR"), len("ACCELERATOR")
	GPUsWidth, nodesWidth := len("GPUS"), len("NODES")
	memoryWidth, durationWidth := len("MEMORY/GPU"), len("APPROX. TIME")
	for _, configuration := range report.Configurations {
		if configuration.EstimateSource == "observed-runs" {
			observedRuns += configuration.ObservedRuns
		}
		candidate := row{
			manufacturer: configuration.Manufacturer,
			accelerator:  configuration.Accelerator,
			GPUs:         fmt.Sprintf("%d", configuration.GPUs),
			nodes:        fmt.Sprintf("%d", configuration.Nodes),
			memory:       hardwareMemory(configuration.MemoryPerGPUBytes),
			duration:     approximateDuration(configuration.ApproximateSeconds),
		}
		rows = append(rows, candidate)
		manufacturerWidth = max(manufacturerWidth, len(candidate.manufacturer))
		acceleratorWidth = max(acceleratorWidth, len(candidate.accelerator))
		GPUsWidth = max(GPUsWidth, len(candidate.GPUs))
		nodesWidth = max(nodesWidth, len(candidate.nodes))
		memoryWidth = max(memoryWidth, len(candidate.memory))
		durationWidth = max(durationWidth, len(candidate.duration))
	}
	if observedRuns > 0 {
		label := "runs"
		if observedRuns == 1 {
			label = "run"
		}
		fmt.Fprintf(stdout, "CALIBRATION: %d completed local %s applied\n\n", observedRuns, label)
	}
	fmt.Fprintf(stdout, "%*s  %*s  %-*s  %-*s  %*s  %*s\n", GPUsWidth, "GPUS", nodesWidth, "NODES", manufacturerWidth, "MFR", acceleratorWidth, "ACCELERATOR", memoryWidth, "MEMORY/GPU", durationWidth, "APPROX. TIME")
	for _, candidate := range rows {
		fmt.Fprintf(stdout, "%*s  %*s  %-*s  %-*s  %*s  %*s\n", GPUsWidth, candidate.GPUs, nodesWidth, candidate.nodes, manufacturerWidth, candidate.manufacturer, acceleratorWidth, candidate.accelerator, memoryWidth, candidate.memory, durationWidth, candidate.duration)
	}
}

func configuredForecastCalibration() ([]model.ForecastCalibration, error) {
	root, err := configuredModelRoot()
	if err != nil {
		return nil, err
	}
	return model.LoadForecastCalibration(root)
}

func hardwareMemory(bytes uint64) string {
	const gibibyte = uint64(1 << 30)
	if bytes%gibibyte == 0 {
		return fmt.Sprintf("%d GB", bytes/gibibyte)
	}
	return humanBytesUint(bytes)
}

func approximateDuration(seconds int64) string {
	hours := float64(seconds) / float64(time.Hour/time.Second)
	if hours < 1 {
		return "under 1 hour"
	}
	if hours < 100 {
		value := int64(math.Round(hours))
		if value < 1 {
			value = 1
		}
		if value == 1 {
			return "1 hour"
		}
		return fmt.Sprintf("%d hours", value)
	}
	days := int64(math.Round(hours / 24))
	if days == 1 {
		return "1 day"
	}
	return fmt.Sprintf("%d days", days)
}

func runModelInit(context Context, args []string, stdout, stderr io.Writer) error {
	name, presetName := args[0], stringOption(context, "preset")
	preset, err := model.PresetByName(presetName)
	if err != nil {
		return err
	}
	builder, err := configuredModelBuilder(context, stderr)
	if err != nil {
		return err
	}
	inspection, err := builder.Initialize(name, preset.Architecture)
	if err != nil {
		return err
	}
	if context.JSON {
		advice, err := model.BuildAdvice(inspection, time.Now())
		if err != nil {
			return err
		}
		return writeJSON(stdout, struct {
			model.Inspection
			Advice model.Advice `json:"advice"`
		}{Inspection: inspection, Advice: advice})
	}
	fmt.Fprintf(stdout, "initialized model %s\n", name)
	fmt.Fprintf(stdout, "  preset        %s\n", preset.Name)
	fmt.Fprintf(stdout, "  location      %s\n", inspection.Path)
	fmt.Fprintf(stdout, "  model id      %s\n", shortModelHash(inspection.Model.ID))
	fmt.Fprintf(stdout, "  estimate      %s parameters, %s weights\n", humanIntegerUint(inspection.Model.Forecast.ApproximateParameters), humanBytesUint(inspection.Model.Forecast.ParameterBytes))
	return nil
}

func runModelPull(context Context, args []string, stdout, stderr io.Writer) error {
	root, err := configuredModelRoot()
	if err != nil {
		return err
	}
	puller := model.Puller{Root: root, Client: &http.Client{}, Progress: func(progress model.PullProgress) {
		fmt.Fprintln(stderr, progress.Message)
	}}
	inspection, err := puller.Pull(context.Execution, args[0], args[1])
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, inspection)
	}
	fmt.Fprintf(stdout, "pulled model %s\n", inspection.Model.Name)
	fmt.Fprintf(stdout, "  location      %s\n", inspection.Path)
	fmt.Fprintf(stdout, "  model id      %s\n", shortModelHash(inspection.Model.ID))
	fmt.Fprintf(stdout, "  origin        %s@%s\n", inspection.Origin.Source.Repository, shortModelHash(inspection.Origin.Source.Revision))
	fmt.Fprintf(stdout, "  weights       %s\n", humanBytesUint(inspection.Model.Forecast.ParameterBytes))
	return nil
}

func runModelList(context Context, args []string, stdout, _ io.Writer) error {
	root, err := configuredModelRoot()
	if err != nil {
		return err
	}
	models, err := model.List(root, args)
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, models)
	}
	if len(models) == 0 {
		return nil
	}
	table := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(table, "NAME\tSTATE\tPARAMETERS\tRUNS\tUPDATED (UTC)")
	for _, item := range models {
		state := "untrained"
		if item.State != "" {
			state = string(item.State)
		}
		fmt.Fprintf(table, "%s\t%s\t%s\t%s\t%s\n", item.Name, state, humanCount(int64(item.Parameters)), humanInteger(int64(item.Runs)), item.Updated)
	}
	return table.Flush()
}

func runModelSummary(context Context, args []string, stdout, _ io.Writer) error {
	root, err := configuredModelRoot()
	if err != nil {
		return err
	}
	inspection, err := model.Inspect(root, args[0])
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, inspection)
	}
	state := "untrained"
	if inspection.Origin != nil {
		state = "downloaded"
	}
	if len(inspection.Model.Runs) > 0 {
		state = string(inspection.Model.Runs[len(inspection.Model.Runs)-1].State)
	}
	var consumed int64
	for _, run := range inspection.Runs {
		if run.Observation != nil {
			consumed += run.Observation.ConsumedTokens
		} else if run.Progress != nil {
			consumed += run.Progress.ConsumedTokens
		}
	}
	fmt.Fprintf(stdout, "NAME:          %s\n", inspection.Model.Name)
	fmt.Fprintf(stdout, "STATE:         %s\n", state)
	fmt.Fprintf(stdout, "MODEL ID:      %s\n", shortModelHash(inspection.Model.ID))
	fmt.Fprintf(stdout, "CREATED:       %s\n", inspection.Model.Created)
	fmt.Fprintf(stdout, "PARAMETERS:    %s\n", humanModelParameters(inspection.Model.Forecast.ApproximateParameters))
	fmt.Fprintf(stdout, "WEIGHTS:       %s\n", humanBytesUint(inspection.Model.Forecast.ParameterBytes))
	fmt.Fprintf(stdout, "RUNS:          %s\n", humanInteger(int64(len(inspection.Model.Runs))))
	fmt.Fprintf(stdout, "TOKENS:        %s\n", humanCount(consumed))
	fmt.Fprintf(stdout, "ARCHITECTURE:  %s %s, %s layers, width %s, %s/%s heads\n",
		shortModelHash(inspection.Model.ArchitectureSHA256), inspection.Model.Architecture.Family,
		humanIntegerUint(inspection.Model.Architecture.Layers), humanIntegerUint(inspection.Model.Architecture.HiddenSize),
		humanIntegerUint(inspection.Model.Architecture.AttentionHeads), humanIntegerUint(inspection.Model.Architecture.KeyValueHeads))
	fmt.Fprintf(stdout, "TOKENIZER:     %s@%s\n", inspection.Model.Architecture.Tokenizer.Name, inspection.Model.Architecture.Tokenizer.Revision)
	if inspection.Origin != nil {
		fmt.Fprintf(stdout, "ORIGIN:        %s@%s (%s)\n", inspection.Origin.Source.Repository, shortModelHash(inspection.Origin.Source.Revision), inspection.Origin.Source.Provider)
	}
	for position, pin := range inspection.Model.Runs {
		tokens := int64(0)
		simulated := ""
		detail := ""
		if position < len(inspection.Runs) && inspection.Runs[position].Observation != nil {
			tokens = inspection.Runs[position].Observation.ConsumedTokens
			if inspection.Runs[position].Observation.Simulated {
				simulated = ", simulated"
			}
		} else if position < len(inspection.Runs) && inspection.Runs[position].Progress != nil {
			run := inspection.Runs[position]
			tokens = run.Progress.ConsumedTokens
			if len(run.Progress.Checkpoints) > 0 {
				detail = fmt.Sprintf(", checkpoint step %s", humanInteger(run.Progress.Checkpoints[len(run.Progress.Checkpoints)-1].Step))
			}
			if len(run.Attempts) > 1 {
				detail += fmt.Sprintf(", %s attempts", humanInteger(int64(len(run.Attempts))))
			}
		}
		fmt.Fprintf(stdout, "RUN %04d:      %-16s %-11s %s tokens%s%s\n", pin.Ordinal, pin.Stage, pin.State, humanCount(tokens), simulated, detail)
		if position < len(inspection.Runs) && inspection.Runs[position].Observation != nil {
			observation := inspection.Runs[position].Observation
			if len(observation.Evaluations) > 0 {
				initial := observation.Evaluations[0].Metrics["heldout_loss"]
				finalMetrics := observation.Evaluations[len(observation.Evaluations)-1].Metrics
				best := initial
				for _, evaluation := range observation.Evaluations {
					if loss, ok := evaluation.Metrics["heldout_loss"]; ok && loss < best {
						best = loss
					}
				}
				fmt.Fprintf(stdout, "  EVALUATION:  held-out loss initial %.4f, best %.4f, final %.4f", initial, best, finalMetrics["heldout_loss"])
				if artifactLoss, ok := finalMetrics["artifact_heldout_loss"]; ok {
					fmt.Fprintf(stdout, "; reloaded artifact %.4f", artifactLoss)
				}
				fmt.Fprintln(stdout)
			}
			for _, item := range observation.Consumption {
				fmt.Fprintf(stdout, "  CORPUS:      %-40s %s token targets\n", item.Corpus, humanCount(item.TokenTargets))
			}
		}
		if position < len(inspection.BOM.Runs) {
			runDirectory := filepath.Dir(filepath.FromSlash(inspection.BOM.Runs[position].RunBOM))
			fmt.Fprintf(stdout, "  TELEMETRY:   %s\n", filepath.Join(inspection.Path, runDirectory, model.TelemetryFilename))
		}
	}
	advice, err := model.BuildAdvice(inspection, time.Now())
	if err != nil {
		return err
	}
	fmt.Fprintf(stdout, "ADVICE:        %s — %s\n", advice.Action, advice.Summary)
	for _, finding := range advice.Findings {
		fmt.Fprintf(stdout, "  - %s\n", finding)
	}
	return nil
}

func runModelBOM(context Context, args []string, stdout, stderr io.Writer) error {
	options, err := cobraModelBOMOptions(context, args)
	if err != nil {
		return err
	}
	if options.Format == "eu-gpai" {
		return runModelEUGPAIBOM(context, options, stdout, stderr)
	}
	root, err := configuredModelRoot()
	if err != nil {
		return err
	}
	inspection, err := model.Inspect(root, options.Model)
	if err != nil {
		return err
	}
	if options.Output == "" {
		return writeJSON(stdout, inspection.BOM)
	}
	output, err := filepath.Abs(options.Output)
	if err != nil {
		return err
	}
	flags := os.O_CREATE | os.O_EXCL | os.O_WRONLY
	if options.Force {
		flags = os.O_CREATE | os.O_TRUNC | os.O_WRONLY
	}
	file, err := os.OpenFile(output, flags, 0o644)
	if err != nil {
		return err
	}
	if err := writeJSON(file, inspection.BOM); err != nil {
		_ = file.Close()
		_ = os.Remove(output)
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Output string `json:"output"`
		}{output})
	}
	fmt.Fprintf(stdout, "wrote model OpenWALDO BOM to %s\n", output)
	return nil
}

func runModelTrain(context Context, args []string, stdout, stderr io.Writer) error {
	name, inputs := args[0], args[1:]
	composePath, err := trainingComposeInput(inputs)
	if err != nil {
		return err
	}
	if composePath != "" {
		if context.Command != nil && context.Command.Flags().Changed("epochs") {
			return fmt.Errorf("--epochs cannot be used with compose %q; set each stage budget in the compose file", composePath)
		}
		for _, flag := range []string{"batch-size", "learning-rate", "seed"} {
			if context.Command != nil && context.Command.Flags().Changed(flag) {
				return fmt.Errorf("--%s cannot be used with compose %q; compose stages define their own parameters", flag, composePath)
			}
		}
		cluster, err := trainingClusterFromFlags(context, 0)
		if err != nil {
			return err
		}
		return runModelComposeTraining(context, name, composePath, cluster, stdout, stderr)
	}
	epochs := int64Option(context, "epochs")
	if epochs < 1 || epochs > 1_000_000 {
		return fmt.Errorf("--epochs must be an integer in 1..1000000")
	}
	batch := int64Option(context, "batch-size")
	if batch < 1 || batch > 1_000_000 {
		return fmt.Errorf("--batch-size must be an integer in 1..1000000")
	}
	learningRate := float64Option(context, "learning-rate")
	if learningRate <= 0 || math.IsNaN(learningRate) || learningRate > 1 {
		return fmt.Errorf("--learning-rate must be a positive number no greater than 1")
	}
	seed := uint64Option(context, "seed")
	root, err := configuredModelRoot()
	if err != nil {
		return err
	}
	exists, err := model.Exists(root, name)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("model %q does not exist; train it from a compose file or initialize it with waldo model init", name)
	}
	inspection, err := model.Inspect(root, name)
	if err != nil {
		return err
	}
	cluster, err := trainingClusterFromFlags(context, 0)
	if err != nil {
		return err
	}
	builder, err := configuredModelBuilderForCluster(context, stderr, cluster)
	if err != nil {
		return err
	}
	if err := builder.CheckBackend(context.Execution, inspection.Model.Architecture, []string{"causal-language-modeling"}); err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	stage, err := prepareDefaultTrainingStage(context, inspection, inputs, epochs, batch, learningRate, seed, boolOption(context, "audit"), cache, stderr)
	if err != nil {
		return err
	}
	result, err := builder.Train(context.Execution, name, stage)
	if err != nil {
		return err
	}
	if _, err := cache.PurgeUsed(); err != nil {
		return fmt.Errorf("purge successful training scratch: %w", err)
	}
	return writeModelMutationResult(context, stdout, result, "trained")
}

func trainingComposeInput(inputs []string) (string, error) {
	var composePath string
	for _, input := range inputs {
		isCompose, err := model.IsComposeFile(input)
		if err != nil {
			return "", err
		}
		if !isCompose {
			continue
		}
		if len(inputs) != 1 {
			return "", fmt.Errorf("compose file %q must be the only training input", input)
		}
		composePath = input
	}
	return composePath, nil
}

var validRunLabel = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

func trainingClusterFromFlags(commandContext Context, nodeRank int) (training.Cluster, error) {
	nodes := intOption(commandContext, "nodes")
	if nodes < 1 {
		return training.Cluster{}, fmt.Errorf("--nodes must be an integer greater than or equal to 1")
	}
	cluster := training.Cluster{
		Nodes:        nodes,
		NodeRank:     nodeRank,
		Rendezvous:   strings.TrimSpace(stringOption(commandContext, "rendezvous")),
		RendezvousID: strings.TrimSpace(stringOption(commandContext, "rendezvous-id")),
	}
	if nodes == 1 && (cluster.Rendezvous != "" || cluster.RendezvousID != "") {
		return training.Cluster{}, fmt.Errorf("--rendezvous and --rendezvous-id require --nodes greater than 1; they would be silently ignored on a single-node run")
	}
	if nodes > 1 {
		if cluster.Rendezvous == "" || cluster.RendezvousID == "" {
			return training.Cluster{}, fmt.Errorf("multi-node training requires --rendezvous host:port and --rendezvous-id")
		}
		if _, _, err := net.SplitHostPort(cluster.Rendezvous); err != nil {
			return training.Cluster{}, fmt.Errorf("--rendezvous %q must be host:port: %w", cluster.Rendezvous, err)
		}
		if !validRunLabel.MatchString(cluster.RendezvousID) {
			return training.Cluster{}, fmt.Errorf("--rendezvous-id %q must start with a letter or digit and contain only letters, digits, '.', '_', and '-'; it names the shared plan path on every node", cluster.RendezvousID)
		}
	}
	configuration, err := config.Load()
	if err != nil {
		return training.Cluster{}, err
	}
	cluster.Interface = configuration.Model.NCCLInterface
	cluster.HCA = configuration.Model.NCCLHCA
	return cluster, nil
}

func runModelTrainWorker(commandContext Context, _ []string, stdout, stderr io.Writer) error {
	nodeRank := intOption(commandContext, "node-rank")
	cluster, err := trainingClusterFromFlags(commandContext, nodeRank)
	if err != nil {
		return err
	}
	if cluster.Nodes < 2 {
		return fmt.Errorf("model train-worker requires --nodes greater than 1")
	}
	if nodeRank < 1 || nodeRank >= cluster.Nodes {
		return fmt.Errorf("--node-rank must be in 1..%d for a %d-node run", cluster.Nodes-1, cluster.Nodes)
	}
	planWaitValue := strings.TrimSpace(stringOption(commandContext, "plan-wait"))
	planWait, err := time.ParseDuration(planWaitValue)
	if err != nil {
		return fmt.Errorf("--plan-wait %q must be a Go duration such as 30m: %w", planWaitValue, err)
	}
	if planWait <= 0 {
		return fmt.Errorf("--plan-wait %q must be a positive duration such as 30m", planWaitValue)
	}
	configuration, err := config.Load()
	if err != nil {
		return err
	}
	modelRoot, err := config.EffectiveModelRoot(configuration)
	if err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	// The cache owns the scratch root, including its mode. Creating a per-node
	// directory beneath it must not decide how the shared root is created.
	if err := cache.EnsureScratch(); err != nil {
		return err
	}
	scratch := filepath.Join(cache.Scratch(), "train-worker", cluster.RendezvousID, fmt.Sprintf("node-%d", cluster.NodeRank))
	if err := os.MkdirAll(scratch, 0o700); err != nil {
		return err
	}
	defer os.RemoveAll(scratch)
	defer func() {
		if _, purgeErr := cache.PurgeUsed(); purgeErr != nil {
			fmt.Fprintf(stderr, "warning: purge secondary training scratch: %v\n", purgeErr)
		}
	}()
	run := func(runCluster training.Cluster, request training.Request) error {
		return training.RunSecondaryTorchTitan(commandContext.Execution, runCluster, request)
	}
	return runSecondaryStages(commandContext, cluster, modelRoot, scratch, cache, planWait, run, stdout, stderr)
}

func runSecondaryStages(commandContext Context, cluster training.Cluster, modelRoot, scratch string, cache *lookaside.Cache, planWait time.Duration, run func(training.Cluster, training.Request) error, stdout, stderr io.Writer) error {
	lastRunID := ""
	for {
		fmt.Fprintf(stderr, "node %d/%d awaiting the primary node's training plan (rendezvous id %s)\n", cluster.NodeRank, cluster.Nodes, cluster.RendezvousID)
		plan, err := awaitMultiNodePlan(commandContext.Execution, modelRoot, cluster.RendezvousID, planWait, lastRunID, stderr)
		if err != nil {
			return err
		}
		if plan.Nodes != cluster.Nodes {
			return fmt.Errorf("primary published a %d-node run but this worker was started with --nodes %d; every node must agree on the topology", plan.Nodes, cluster.Nodes)
		}
		stageScratch := filepath.Join(scratch, plan.RunID)
		if err := os.MkdirAll(stageScratch, 0o700); err != nil {
			return err
		}
		request, err := secondaryTrainingRequest(commandContext, plan, modelRoot, cache, stageScratch, stderr)
		if err != nil {
			return err
		}
		fmt.Fprintf(stderr, "joining rendezvous %s as node %d of %d for stage %d/%d\n", cluster.Rendezvous, cluster.NodeRank, cluster.Nodes, plan.StageOrdinal, plan.StageCount)
		if err := run(cluster, request); err != nil {
			return err
		}
		_ = os.RemoveAll(stageScratch)
		lastRunID = plan.RunID
		if plan.StageOrdinal == plan.StageCount {
			fmt.Fprintln(stdout, "secondary node completed")
			return nil
		}
		fmt.Fprintf(stderr, "stage %d/%d complete; awaiting the next stage plan\n", plan.StageOrdinal, plan.StageCount)
	}
}

func awaitMultiNodePlan(ctx context.Context, modelRoot, rendezvousID string, wait time.Duration, skipRunID string, progress io.Writer) (model.MultiNodePlan, error) {
	path := model.MultiNodePlanPath(modelRoot, rendezvousID)
	const poll = time.Second
	deadline := time.Now().Add(wait)
	announced := false
	announcedStale := false
	for {
		data, readErr := os.ReadFile(path)
		if readErr == nil {
			var plan model.MultiNodePlan
			if err := json.Unmarshal(data, &plan); err != nil {
				return model.MultiNodePlan{}, fmt.Errorf("decode multi-node plan %s: %w", path, err)
			}
			if plan.Kind != model.MultiNodePlanKind {
				return model.MultiNodePlan{}, fmt.Errorf("multi-node plan %s has kind %q, expected %q", path, plan.Kind, model.MultiNodePlanKind)
			}
			if plan.Schema != model.MultiNodePlanSchema {
				return model.MultiNodePlan{}, fmt.Errorf("multi-node plan %s has schema %d; this build supports %d — run the same waldo build on every node", path, plan.Schema, model.MultiNodePlanSchema)
			}
			if skipRunID != "" && plan.RunID == skipRunID {
				if !announcedStale {
					fmt.Fprintf(progress, "  previous stage's plan is still on disk; awaiting the next stage\n")
					announcedStale = true
				}
			} else {
				if plan.StageOrdinal < 1 || plan.StageCount < plan.StageOrdinal {
					return model.MultiNodePlan{}, fmt.Errorf("multi-node plan %s has stage accounting %d/%d; run the same waldo build on every node", path, plan.StageOrdinal, plan.StageCount)
				}
				if !validRunLabel.MatchString(plan.RunID) {
					return model.MultiNodePlan{}, fmt.Errorf("multi-node plan %s has run id %q, which is not a safe path component", path, plan.RunID)
				}
				return plan, nil
			}
		} else if !errors.Is(readErr, os.ErrNotExist) {
			return model.MultiNodePlan{}, fmt.Errorf("read multi-node plan %s: %w", path, readErr)
		} else if !announced {
			fmt.Fprintf(progress, "  plan not published yet; polling %s\n", path)
			announced = true
		}
		if time.Now().After(deadline) {
			return model.MultiNodePlan{}, fmt.Errorf("primary node did not publish a multi-node plan for rendezvous id %s at %s within %s; check that every node uses the same --rendezvous-id and model root, and that the primary is still running — a primary that failed or finished removes its plan and publishes nothing further", rendezvousID, path, wait)
		}
		select {
		case <-ctx.Done():
			return model.MultiNodePlan{}, ctx.Err()
		case <-time.After(poll):
		}
	}
}

func secondaryTrainingRequest(commandContext Context, plan model.MultiNodePlan, modelRoot string, cache *lookaside.Cache, scratch string, progress io.Writer) (training.Request, error) {
	fmt.Fprintf(progress, "  materializing %s shards for run %s\n", humanInteger(plan.CorpusBOM.Totals.Shards), plan.RunID)
	materialized, err := corpus.Materialize(commandContext.Execution, plan.CorpusBOM, cache, modelMaterializeProgressPrinter(progress))
	if err != nil {
		return training.Request{}, err
	}
	inputs := verifiedTrainingInputs(materialized, plan.CorpusBOM)
	if len(inputs) == 0 {
		return training.Request{}, fmt.Errorf("primary plan resolved no verified shard inputs")
	}
	var architecture struct {
		VocabularySize uint64 `json:"vocabulary_size"`
		Tokenizer      struct {
			Name     string `json:"name"`
			Revision string `json:"revision"`
		} `json:"tokenizer"`
	}
	if err := json.Unmarshal(plan.Architecture, &architecture); err != nil {
		return training.Request{}, fmt.Errorf("decode primary plan architecture: %w", err)
	}
	tokenizerSpec, codec, err := training.ResolveTokenizer(architecture.Tokenizer.Name, architecture.Tokenizer.Revision, architecture.VocabularySize)
	if err != nil {
		return training.Request{}, fmt.Errorf("resolve primary plan tokenizer: %w", err)
	}
	partition, err := training.NewRecordPartitionContextWithTokenizer(commandContext.Execution, inputs, plan.Parameters, codec, nil)
	if err != nil {
		return training.Request{}, fmt.Errorf("reselect held-out evaluation split: %w", err)
	}
	if plan.EvaluationSet == nil {
		return training.Request{}, fmt.Errorf("multi-node plan carries no held-out split to verify against; nodes could train on divergent data")
	}
	if partition.Evaluation.SHA256 != plan.EvaluationSet.SHA256 {
		return training.Request{}, fmt.Errorf("secondary held-out split %s does not match the primary's %s; nodes would train on divergent data", partition.Evaluation.SHA256, plan.EvaluationSet.SHA256)
	}
	records, err := partition.TrainingRecords()
	if err != nil {
		return training.Request{}, err
	}
	initialization, err := secondaryInitialization(plan, modelRoot)
	if err != nil {
		return training.Request{}, err
	}
	return training.Request{
		RunID: plan.RunID, Stage: plan.Stage, Objective: plan.Objective,
		ArchitectureSHA256: plan.ArchitectureSHA256, Architecture: plan.Architecture,
		Parameters: plan.Parameters, Records: records, EvaluationRecords: partition.EvaluationRecords(),
		EvaluationSet: model.EvaluationSetValue(plan.EvaluationSet), Initialization: initialization,
		Tokenizer:         tokenizerSpec,
		ArtifactDirectory: scratch, ArtifactPrefix: "artifacts",
	}, nil
}

func secondaryInitialization(plan model.MultiNodePlan, modelRoot string) (*training.Initialization, error) {
	if plan.Initialization == nil {
		if plan.InitializationPath != "" {
			return nil, fmt.Errorf("plan carries an initialization path without initialization weights; run the same waldo build on every node")
		}
		return nil, nil
	}
	if plan.InitializationPath == "" {
		return nil, fmt.Errorf("plan carries initialization weights without a portable path; run the same waldo build on every node")
	}
	relative := filepath.FromSlash(plan.InitializationPath)
	if !filepath.IsLocal(relative) {
		return nil, fmt.Errorf("plan initialization path %q escapes the model root", plan.InitializationPath)
	}
	path := filepath.Join(modelRoot, relative)
	if err := model.VerifyArtifactFile(path, plan.Initialization.Artifact); err != nil {
		return nil, fmt.Errorf("verify initialization weights on this node's model root: %w", err)
	}
	initialization := *plan.Initialization
	initialization.Path = path
	return &initialization, nil
}

func looksLikeIndexPath(value string) bool {
	return value == "." || value == ".." || value == "~" || strings.ContainsAny(value, `/\\`)
}

func runModelComposeTraining(context Context, name, path string, cluster training.Cluster, stdout, stderr io.Writer) error {
	compose, composePath, err := model.LoadCompose(path)
	if err != nil {
		return err
	}
	builder, err := configuredModelBuilderForCluster(context, stderr, cluster)
	if err != nil {
		return err
	}
	compose, err = builder.ResolveCompose(context.Execution, compose, true)
	if err != nil {
		return err
	}
	if err := builder.CheckComposeTarget(name, compose); err != nil {
		return err
	}
	builder.ComposeName = filepath.Base(composePath)
	objectives := make([]string, 0, len(compose.Stages))
	for _, stage := range compose.Stages {
		if !slices.Contains(objectives, stage.Objective) {
			objectives = append(objectives, stage.Objective)
		}
	}
	if err := builder.CheckBackend(context.Execution, compose.Architecture, objectives); err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	prepared := make([]model.PreparedStage, 0, len(compose.Stages))
	for _, stage := range compose.Stages {
		resolved, err := prepareModelStage(context, stage, cache, stderr, boolOption(context, "audit"))
		if err != nil {
			return err
		}
		prepared = append(prepared, resolved)
	}
	result, err := builder.Compose(context.Execution, name, compose, prepared)
	if err != nil {
		return err
	}
	if _, err := cache.PurgeUsed(); err != nil {
		return fmt.Errorf("purge successful compose scratch: %w", err)
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Compose string           `json:"compose"`
			Result  model.Inspection `json:"result"`
		}{Compose: composePath, Result: result})
	}
	return writeModelMutationResult(context, stdout, result, "trained")
}

func runModelContinue(context Context, args []string, stdout, stderr io.Writer) error {
	root, err := configuredModelRoot()
	if err != nil {
		return err
	}
	name := args[0]
	inspection, err := model.Inspect(root, name)
	if err != nil {
		return err
	}
	pending, err := model.HasPendingCompose(root, name)
	if err != nil {
		return err
	}
	if !pending {
		state := "untrained"
		if len(inspection.Runs) > 0 {
			state = string(inspection.Runs[len(inspection.Runs)-1].State)
		}
		if !model.HasRecoverableFinalizationFailure(inspection) {
			return fmt.Errorf("model %q has no interrupted compose to continue (current state: %s)", name, state)
		}
		fmt.Fprintf(stderr, "continue               recovering checkpoint-backed finalization failure for %s\n", name)
	}
	if pending && len(inspection.RunBOMs) > 0 && inspection.RunBOMs[len(inspection.RunBOMs)-1].Execution.Nodes > 1 {
		return fmt.Errorf("model %q has an interrupted multi-node compose; continue runs single-node and would silently change the topology — re-run `waldo model train %s <compose>` with the original --nodes, --rendezvous, and --rendezvous-id", name, name)
	}
	composePath, err := model.LatestComposePath(inspection.Path)
	if err != nil {
		return fmt.Errorf("locate last compose for model %q: %w", name, err)
	}
	fmt.Fprintf(stderr, "continue               resuming %s from %s\n", name, composePath)
	return runModelComposeTraining(context, name, composePath, training.Cluster{}, stdout, stderr)
}

func runModelExport(context Context, args []string, stdout, stderr io.Writer) error {
	parsed, err := cobraModelExportOptions(context, args)
	if err != nil {
		return err
	}
	configuration, err := config.Load()
	if err != nil {
		return err
	}
	root, err := config.EffectiveModelRoot(configuration)
	if err != nil {
		return err
	}
	inspection, err := model.Inspect(root, parsed.Name)
	if err != nil {
		return err
	}
	if configuration.Disclosure.Provider == "" {
		return fmt.Errorf("model export requires provider information; run waldo config set disclosure.provider <provider.json>")
	}
	provider, err := disclosure.LoadProvider(configuration.Disclosure.Provider)
	if err != nil {
		return fmt.Errorf("load configured disclosure.provider: %w", err)
	}
	report, err := disclosure.BuildEUGPAIReport(inspection, &provider, disclosure.ReleaseFromModel(inspection), time.Now())
	if err != nil {
		return err
	}
	if err := requireCompleteDisclosure(report, parsed.AllowIncomplete, stderr); err != nil {
		return err
	}
	euBOM, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	euBOM = append(euBOM, '\n')
	signed := signing.Configured(configuration.Signing)
	finalize := func(string) error { return nil }
	if signed {
		finalize = func(directory string) error {
			return signing.SignExport(context.Execution, configuration.Signing, directory, stderr)
		}
	}
	var output string
	var quantization *modelexport.Quantization
	var preparedCalibration *calibration.Prepared
	var cache *lookaside.Cache
	if parsed.Quant != "" {
		resolved, err := modelquant.ResolveProfile(parsed.Quant)
		if err != nil {
			return err
		}
		runtime, err := modelquant.ResolveRuntime(context.Execution, parsed.Calibration != "")
		if err != nil {
			return err
		}
		quantization = &modelexport.Quantization{Requested: parsed.Quant, Resolved: resolved, Quantizer: runtime}
		if parsed.Calibration != "" {
			cache, err = lookaside.DefaultCache()
			if err != nil {
				return err
			}
			targets, err := resolveIndexArguments(context.Execution, []string{parsed.Calibration}, stderr)
			if err != nil {
				return fmt.Errorf("calibration: %w", err)
			}
			policy, err := corpus.NewLicensePolicy(nil, nil)
			if err != nil {
				return err
			}
			bom, err := corpus.BuildBOM(context.Execution, targets, policy, cache)
			if err != nil {
				return fmt.Errorf("calibration: %w", err)
			}
			fmt.Fprintf(stderr, "calibration        resolved %s: %s shards, %s reference tokens\n", strings.Join(bom.Paths, ", "), humanInteger(bom.Totals.Shards), humanCount(bom.Totals.Tokens))
			prepared, err := calibration.Prepare(context.Execution, bom, cache, calibration.DefaultTokens, calibration.DefaultSeed, func(event calibration.Progress) {
				if event.Current == 1 || event.Current%25 == 0 || event.Current == event.Total {
					fmt.Fprintf(stderr, "calibration        shard %s/%s  %s\n", humanInteger(int64(event.Current)), humanInteger(int64(event.Total)), event.Shard[:12])
				}
			})
			if err != nil {
				return err
			}
			preparedCalibration = &prepared
			defer prepared.Cleanup()
			quantization.Calibration = &modelexport.Calibration{TextPath: prepared.TextPath, Profile: prepared.BOM.Profile, ReferenceTokens: prepared.BOM.Corpus.Tokens, SampledTokens: prepared.BOM.SampledTokens, Records: prepared.BOM.Records, Shards: len(prepared.BOM.Shards), SelectionSHA256: prepared.BOM.SelectionSHA256, Seed: prepared.BOM.Seed, Evidence: json.RawMessage(prepared.JSON)}
			fmt.Fprintf(stderr, "calibration        selected %s byte tokens from %s records in %s shards\n", humanCount(prepared.BOM.SampledTokens), humanCount(prepared.BOM.Records), humanInteger(int64(len(prepared.BOM.Shards))))
		}
	}
	switch parsed.Format {
	case "waldo":
		options := model.ExportOptions{Files: map[string][]byte{signing.EUBOM: euBOM}}
		if signed {
			options.Finalize = finalize
		}
		output, err = model.ExportPackage(root, parsed.Name, parsed.Destination, options)
	case "huggingface":
		options := modelexport.Options{EUBOM: euBOM}
		if signed {
			options.Finalize = finalize
		}
		output, err = modelexport.ExportHuggingFace(context.Execution, inspection, parsed.Destination, options)
	case "mlx":
		options := modelexport.Options{EUBOM: euBOM}
		if signed {
			options.Finalize = finalize
		}
		output, err = modelexport.ExportMLX(context.Execution, inspection, parsed.Destination, options)
	case "gguf":
		options := modelexport.Options{EUBOM: euBOM, Quantization: quantization, Report: func(message string) { fmt.Fprintln(stderr, "quantization      "+message) }}
		if signed {
			options.Finalize = finalize
		}
		output, err = modelexport.ExportGGUF(context.Execution, inspection, parsed.Destination, options)
	case "ollama":
		options := modelexport.Options{EUBOM: euBOM, Quantization: quantization, Report: func(message string) { fmt.Fprintln(stderr, "quantization      "+message) }}
		if signed {
			options.Finalize = finalize
		}
		output, err = modelexport.ExportOllama(context.Execution, inspection, parsed.Destination, options)
	}
	if err != nil {
		return err
	}
	if preparedCalibration != nil {
		if _, err := cache.PurgeUsed(); err != nil {
			return fmt.Errorf("purge successful calibration scratch: %w", err)
		}
	}
	if !signed {
		fmt.Fprintln(stderr, "warning: model export is unsigned; configure signing.* to sign exports automatically")
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Name   string `json:"name"`
			Format string `json:"format"`
			Output string `json:"output"`
			Signed bool   `json:"signed"`
		}{parsed.Name, parsed.Format, output, signed})
	}
	fmt.Fprintf(stdout, "exported %s model %s to %s\n", parsed.Format, parsed.Name, output)
	return nil
}

type modelExportOptions struct {
	Name            string
	Destination     string
	Format          string
	Quant           string
	Calibration     string
	AllowIncomplete bool
}

func cobraModelExportOptions(context Context, args []string) (modelExportOptions, error) {
	result := modelExportOptions{
		Name: args[0], Destination: args[1],
		Format: stringOption(context, "format"), Quant: stringOption(context, "quant"),
		Calibration: stringOption(context, "calibration"), AllowIncomplete: boolOption(context, "allow-incomplete"),
	}
	if result.Format != "waldo" && result.Format != "huggingface" && result.Format != "mlx" && result.Format != "gguf" && result.Format != "ollama" {
		return modelExportOptions{}, fmt.Errorf("model export format %q is not implemented; use waldo, huggingface, mlx, gguf, or ollama", result.Format)
	}
	if result.Quant != "" && result.Format != "gguf" && result.Format != "ollama" {
		return modelExportOptions{}, fmt.Errorf("--quant is supported only with --format gguf or ollama")
	}
	if result.Quant != "" {
		if _, err := modelquant.ResolveProfile(result.Quant); err != nil {
			return modelExportOptions{}, err
		}
	}
	if result.Calibration != "" && result.Quant == "" {
		return modelExportOptions{}, fmt.Errorf("--calibration requires --quant")
	}
	return result, nil
}

func runModelRemove(context Context, args []string, stdout, _ io.Writer) error {
	root, err := configuredModelRoot()
	if err != nil {
		return err
	}
	removed, err := model.Remove(root, args)
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Removed []string `json:"removed"`
		}{removed})
	}
	for _, name := range removed {
		fmt.Fprintf(stdout, "removed model %s\n", name)
	}
	return nil
}

var openModelChat = inference.Open
var modelChatInput io.Reader = os.Stdin
var modelChatTerminal = func() bool { return term.IsTerminal(int(os.Stdin.Fd())) }

func runModelChat(context Context, args []string, stdout, stderr io.Writer) error {
	name, prompt, options, err := cobraModelChatOptions(context, args)
	if err != nil {
		return err
	}
	root, err := configuredModelRoot()
	if err != nil {
		return err
	}
	inspection, err := model.Inspect(root, name)
	if err != nil {
		return err
	}
	if len(inspection.Model.Runs) == 0 && inspection.Origin == nil {
		return fmt.Errorf("model %q is untrained", name)
	}
	interactive := prompt == nil && modelChatTerminal()
	if context.JSON && interactive {
		return fmt.Errorf("--json requires a positional prompt or piped standard input")
	}
	if !interactive && prompt == nil {
		data, err := io.ReadAll(modelChatInput)
		if err != nil {
			return fmt.Errorf("read chat prompt from standard input: %w", err)
		}
		value := string(data)
		prompt = &value
	}
	if interactive {
		fmt.Fprintf(stderr, "loading model %s...\n", name)
	}
	opened, err := openModelChat(context.Execution, inspection)
	if err != nil {
		fmt.Fprintf(stderr, "warning: model chat unavailable: %v\n", err)
		return err
	}
	var chatErr error
	if interactive {
		chatErr = runInteractiveChat(context.Execution, opened, options, stdout)
	} else {
		chatErr = runOneShotChat(context, opened, *prompt, options, stdout)
	}
	return errors.Join(chatErr, opened.Session.Close())
}

func cobraModelChatOptions(context Context, args []string) (string, *string, inference.Options, error) {
	options := inference.Options{MaxTokens: intOption(context, "max-tokens"), Temperature: float64Option(context, "temperature"), TopP: float64Option(context, "top-p")}
	if optionChanged(context, "seed") {
		seed := uint64Option(context, "seed")
		options.Seed = &seed
	}
	if err := options.Validate(); err != nil {
		return "", nil, options, err
	}
	if len(args) == 2 {
		return args[0], &args[1], options, nil
	}
	return args[0], nil, options, nil
}

func runOneShotChat(context Context, opened inference.Opened, prompt string, options inference.Options, stdout io.Writer) error {
	var renderer safeTokenWriter
	if !context.JSON {
		renderer.writer = stdout
	}
	result, err := opened.Session.Generate(context.Execution, prompt, options, func(token inference.Token) error {
		if context.JSON {
			return nil
		}
		return renderer.Write(token.Bytes)
	})
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Model      string           `json:"model"`
			SourceType string           `json:"source_type"`
			SourceID   string           `json:"source_id"`
			RunID      string           `json:"run_id,omitempty"`
			Prompt     string           `json:"prompt"`
			Result     inference.Result `json:"result"`
		}{opened.Description.Model, opened.Description.SourceType, opened.Description.SourceID, opened.Description.RunID, prompt, result})
	}
	if err := renderer.Flush(); err != nil {
		return err
	}
	if !strings.HasSuffix(result.Text, "\n") {
		_, err = fmt.Fprintln(stdout)
	}
	return err
}

func runInteractiveChat(ctx context.Context, opened inference.Opened, options inference.Options, stdout io.Writer) error {
	fmt.Fprintf(stdout, "OpenWALDO model %s\n", opened.Description.Model)
	fmt.Fprintf(stdout, "Backend: %s\n", strings.ToUpper(opened.Description.Backend))
	fmt.Fprintf(stdout, "Context: %d tokens\n", opened.Description.ContextTokens)
	fmt.Fprintln(stdout, "Mode: raw causal continuation (this model has no chat template)")
	fmt.Fprintln(stdout, "Commands: /clear, /help, /exit")
	reader := bufio.NewReader(modelChatInput)
	history := ""
	for {
		fmt.Fprint(stdout, "\nyou> ")
		line, err := reader.ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" && errors.Is(err, io.EOF) {
			fmt.Fprintln(stdout)
			return nil
		}
		switch line {
		case "/exit":
			return nil
		case "/clear":
			history = ""
			fmt.Fprintln(stdout, "context cleared")
			continue
		case "/help":
			fmt.Fprintln(stdout, "/clear resets context; /exit or Ctrl-D closes the session")
			continue
		}
		prompt := line
		if history != "" {
			prompt = history + "\n" + line
		}
		fmt.Fprintf(stdout, "%s> ", opened.Description.Model)
		renderer := safeTokenWriter{writer: stdout}
		result, generateErr := opened.Session.Generate(ctx, prompt, options, func(token inference.Token) error {
			return renderer.Write(token.Bytes)
		})
		if generateErr != nil {
			return generateErr
		}
		if err := renderer.Flush(); err != nil {
			return err
		}
		if !strings.HasSuffix(result.Text, "\n") {
			fmt.Fprintln(stdout)
		}
		history = boundChatHistory(prompt+result.Text, opened.Description.ContextTokens)
		if errors.Is(err, io.EOF) {
			return nil
		}
	}
}

func boundChatHistory(history string, contextTokens int) string {
	if contextTokens < 1 || len(history) <= contextTokens {
		return history
	}
	return strings.ToValidUTF8(history[len(history)-contextTokens:], "�")
}

func configuredModelRoot() (string, error) {
	configuration, err := config.Load()
	if err != nil {
		return "", err
	}
	return config.EffectiveModelRoot(configuration)
}

func configuredModelBuilderForCluster(commandContext Context, progress io.Writer, cluster training.Cluster) (model.Builder, error) {
	configuration, err := config.Load()
	if err != nil {
		return model.Builder{}, err
	}
	root, err := config.EffectiveModelRoot(configuration)
	if err != nil {
		return model.Builder{}, err
	}
	builder := model.Builder{Root: root, Progress: func(event model.Progress) {
		if commandContext.JSON {
			_ = json.NewEncoder(progress).Encode(event)
		} else {
			label := event.Phase
			if event.Stage != "" {
				label += "/" + event.Stage
			}
			message := modelProgressMessage(event)
			if event.State != "" {
				fmt.Fprintf(progress, "%-22s %-11s %s\n", label, event.State, message)
			} else {
				fmt.Fprintf(progress, "%-22s %s\n", label, message)
			}
		}
		if commandContext.Progress != nil {
			commandContext.Progress(event)
		}
	}}
	builder.OriginPuller = &model.Puller{Client: &http.Client{}, Progress: func(event model.PullProgress) {
		fmt.Fprintln(progress, event.Message)
	}}
	backend := config.EffectiveModelBackend(configuration)
	resolver := training.NewEnvironmentResolverForCluster(backend, cluster)
	builder.Resolver = training.ResolverFunc(func(execution context.Context, request training.ResolveRequest) (training.Selection, error) {
		selection, err := resolver.Resolve(execution, request)
		if err != nil {
			if commandContext.JSON {
				_ = json.NewEncoder(progress).Encode(model.Progress{Phase: "backend", State: "unavailable", Message: err.Error()})
			}
		}
		return selection, err
	})
	if cluster.Nodes > 1 && cluster.NodeRank == 0 {
		builder.MultiNode = model.MultiNodeHandoff{RendezvousID: cluster.RendezvousID, Nodes: cluster.Nodes, StageOrdinal: 1, StageCount: 1}
	}
	return builder, nil
}

func configuredModelBuilder(commandContext Context, progress io.Writer) (model.Builder, error) {
	return configuredModelBuilderForCluster(commandContext, progress, training.Cluster{})
}

func modelProgressMessage(event model.Progress) string {
	if event.Training == nil || event.Training.Kind != "progress" || event.Training.Step <= 1 || event.Training.ETASeconds <= 0 {
		return event.Message
	}
	return fmt.Sprintf("%s, ETA %s", event.Message, compactDuration(event.Training.ETASeconds))
}

func compactDuration(seconds int64) string {
	days := seconds / (24 * 60 * 60)
	seconds %= 24 * 60 * 60
	hours := seconds / (60 * 60)
	seconds %= 60 * 60
	minutes := seconds / 60
	seconds %= 60
	switch {
	case days > 0:
		return fmt.Sprintf("%dd %dh", days, hours)
	case hours > 0:
		return fmt.Sprintf("%dh %dm", hours, minutes)
	case minutes > 0:
		return fmt.Sprintf("%dm %ds", minutes, seconds)
	default:
		return fmt.Sprintf("%ds", seconds)
	}
}

func prepareDefaultTrainingStage(context Context, inspection model.Inspection, paths []string, epochs int64, batch int64, learningRate float64, seed uint64, audit bool, cache *lookaside.Cache, progress io.Writer) (model.PreparedStage, error) {
	architecture := inspection.Model.Architecture
	if architecture.Tokenizer.Name != "byte" || architecture.Tokenizer.Revision != training.ByteTokenizerRevision || architecture.VocabularySize != 259 {
		return model.PreparedStage{}, fmt.Errorf("automatic one-pass training currently requires byte@%s with vocabulary_size 259; subword models train from a compose file", training.ByteTokenizerRevision)
	}
	targets, err := resolveIndexArgumentsWithWarning(context.Execution, paths, progress)
	if err != nil {
		return model.PreparedStage{}, err
	}
	policy, err := corpus.NewLicensePolicy(nil, nil)
	if err != nil {
		return model.PreparedStage{}, err
	}
	bom, err := corpus.BuildBOM(context.Execution, targets, policy, cache)
	if err != nil {
		return model.PreparedStage{}, err
	}
	sequence := int64(inspection.Model.Architecture.ContextTokens)
	stageName := fmt.Sprintf("train-%04d", len(inspection.Model.Runs)+1)
	if len(inspection.Model.Runs) > 0 {
		last := inspection.Model.Runs[len(inspection.Model.Runs)-1]
		if last.State == model.RunInterrupted && strings.HasPrefix(last.Stage, "train-") {
			stageName = last.Stage
		}
	}
	stage := model.Stage{
		Name: stageName, Type: "pre-training",
		Objective: "causal-language-modeling", Corpora: model.NewCorpusSelections(paths),
		Parameters: training.Parameters{Epochs: epochs, Steps: 1, BatchSize: batch, SequenceLength: sequence, LearningRate: learningRate, Seed: seed},
	}
	prepared, err := materializeModelStage(context, stage, bom, cache, progress, audit)
	if err != nil {
		return model.PreparedStage{}, err
	}
	resolved, err := prepared.Stage.ResolveParameters()
	if err != nil {
		return model.PreparedStage{}, err
	}
	partition, err := training.NewRecordPartitionContext(context.Execution, prepared.Inputs, resolved, nil)
	if err != nil {
		return model.PreparedStage{}, err
	}
	tokenTargets, err := partition.TrainingByteTargets(context.Execution)
	if err != nil {
		return model.PreparedStage{}, err
	}
	capacity := batch * sequence
	steps := tokenTargets / capacity
	if tokenTargets%capacity != 0 {
		steps++
	}
	prepared.Stage.Parameters.Steps = steps
	epochLabel := "epochs"
	if epochs == 1 {
		epochLabel = "epoch"
	}
	fmt.Fprintf(progress, "preflight/%s          training %s %s, %s byte targets, %s optimizer steps (%s × %s tokens); held out %s records\n", stage.Name, humanInteger(epochs), epochLabel, humanCount(tokenTargets), humanInteger(steps), humanInteger(batch), humanInteger(sequence), humanInteger(partition.Evaluation.Records))
	return model.PrepareStage(prepared.Stage, prepared.BOM, prepared.Inputs)
}

func prepareModelStage(context Context, stage model.Stage, cache *lookaside.Cache, progress io.Writer, audit bool) (model.PreparedStage, error) {
	targets, err := resolveIndexArguments(context.Execution, model.CorpusPaths(stage.Corpora), progress)
	if err != nil {
		return model.PreparedStage{}, fmt.Errorf("stage %s: %w", stage.Name, err)
	}
	policy, err := corpus.NewLicensePolicy(nil, nil)
	if err != nil {
		return model.PreparedStage{}, err
	}
	bom, err := corpus.BuildBOM(context.Execution, targets, policy, cache)
	if err != nil {
		return model.PreparedStage{}, fmt.Errorf("stage %s: %w", stage.Name, err)
	}
	recordFilter, err := stage.RecordFilterPolicy(bom.Paths)
	if err != nil {
		return model.PreparedStage{}, fmt.Errorf("stage %s: %w", stage.Name, err)
	}
	bom.RecordFilter = recordFilter
	if err := bom.Validate(); err != nil {
		return model.PreparedStage{}, fmt.Errorf("stage %s filtered corpus BOM: %w", stage.Name, err)
	}
	emitUnassessedFilterWarning(progress, stage.Name, bom)
	return materializeModelStage(context, stage, bom, cache, progress, audit)
}

func emitUnassessedFilterWarning(output io.Writer, stageName string, bom corpus.BOM) {
	if bom.RecordFilter == nil {
		return
	}
	fields := map[string]bool{}
	affected := 0
	for _, selected := range bom.Shards {
		if selected.RecordSchema >= shard.TextRecordSchema {
			continue
		}
		corpusPath := selectedCorpusGroup(selected.Manifest, bom.Paths)
		shardFields := bom.RecordFilter.ContentAssessmentExclusions(corpusPath)
		if len(shardFields) == 0 {
			continue
		}
		affected++
		for _, field := range shardFields {
			fields[field] = true
		}
	}
	if affected == 0 {
		return
	}
	names := make([]string, 0, len(fields))
	for field := range fields {
		names = append(names, field)
	}
	slices.Sort(names)
	shardLabel, reference := "shards", "those shards"
	if affected == 1 {
		shardLabel, reference = "shard", "that shard"
	}
	fmt.Fprintf(output, "warning: stage %s: %s schema-1 %s have no content assessments; %s filters will be ignored for records from %s\n", stageName, humanInteger(int64(affected)), shardLabel, strings.Join(names, ", "), reference)
}

func materializeModelStage(context Context, stage model.Stage, bom corpus.BOM, cache *lookaside.Cache, progress io.Writer, audit bool) (model.PreparedStage, error) {
	fmt.Fprintf(progress, "preflight/%s          resolving %s shards, %s records, %s reference tokens\n", stage.Name, humanInteger(bom.Totals.Shards), humanCount(bom.Totals.Docs), humanCount(bom.Totals.Tokens))
	materialized, err := corpus.Materialize(context.Execution, bom, cache, modelMaterializeProgressPrinter(progress))
	if err != nil {
		return model.PreparedStage{}, err
	}
	inputs := verifiedTrainingInputs(materialized, bom)
	paths := make([]string, 0, len(inputs))
	for _, input := range inputs {
		paths = append(paths, input.Path)
	}
	if audit {
		fmt.Fprintf(progress, "preflight/%s          auditing %s materialized shards\n", stage.Name, humanInteger(int64(len(paths))))
		audited, err := shard.VerifyWithOptions(context.Execution, paths, shard.AuditOptions{Progress: auditProgressPrinter(progress)})
		if err != nil {
			return model.PreparedStage{}, fmt.Errorf("stage %s shard audit: %w", stage.Name, err)
		}
		if audited.Records != bom.Totals.Docs || audited.Tokens != bom.Totals.Tokens || audited.EncodedBytes != bom.Totals.Bytes {
			return model.PreparedStage{}, fmt.Errorf("stage %s audited totals differ from index manifests", stage.Name)
		}
		if err := corpus.AttachShardAttestations(&bom, materialized.Objects); err != nil {
			return model.PreparedStage{}, fmt.Errorf("stage %s shard BOM evidence: %w", stage.Name, err)
		}
	}
	return model.PrepareStage(stage, bom, inputs)
}

func verifiedTrainingInputs(materialized corpus.Materialized, bom corpus.BOM) []training.Input {
	seen := map[string]bool{}
	var inputs []training.Input
	for _, object := range materialized.Objects {
		if seen[object.Shard.SHA256] {
			continue
		}
		seen[object.Shard.SHA256] = true
		inputs = append(inputs, training.Input{Path: object.Path, SHA256: object.Shard.SHA256, Bytes: object.Shard.Bytes, Records: object.Shard.Docs, Corpus: selectedCorpusGroup(object.Shard.Manifest, bom.Paths), RecordFilter: bom.RecordFilter})
	}
	return inputs
}

func selectedCorpusGroup(manifest string, selections []string) string {
	best := ""
	bestLogicalLength := 0
	manifestLogical := strings.TrimSuffix(strings.TrimSuffix(strings.TrimSpace(manifest), ".yaml"), ".json")
	for _, selection := range selections {
		selection = strings.TrimSuffix(strings.TrimSpace(selection), "/")
		logical := strings.TrimSuffix(strings.TrimSuffix(selection, ".yaml"), ".json")
		if manifestLogical == logical || strings.HasPrefix(manifestLogical, logical+"/") {
			if len(logical) > bestLogicalLength {
				best = selection
				bestLogicalLength = len(logical)
			}
		}
	}
	return best
}

func modelMaterializeProgressPrinter(output io.Writer) func(corpus.MaterializeProgress) {
	type terminalWriter interface{ Fd() uintptr }
	terminal := false
	if writer, ok := output.(terminalWriter); ok {
		terminal = term.IsTerminal(int(writer.Fd()))
	}
	lastUpdate := time.Time{}
	return func(event corpus.MaterializeProgress) {
		if !terminal {
			if event.Phase == "complete" {
				fmt.Fprintf(output, "  materialized %s/%s  %s/%s  %s\n",
					humanInteger(int64(event.Current)), humanInteger(int64(event.Total)),
					humanBytes(event.Bytes), humanBytes(event.TotalBytes), event.Shard.SHA256[:12])
			}
			return
		}
		now := time.Now()
		if event.Phase != "complete" && !lastUpdate.IsZero() && now.Sub(lastUpdate) < 100*time.Millisecond {
			return
		}
		lastUpdate = now
		const width = 24
		filled := 0
		if event.TotalBytes > 0 {
			filled = int(event.Bytes * width / event.TotalBytes)
			if filled > width {
				filled = width
			}
		}
		phase := event.Phase
		if phase == "complete" {
			phase = "verified"
		}
		fmt.Fprintf(output, "\r\x1b[K  materialize [%-24s] %3d%%  %s/%s  %s/%s  %-8s %s",
			strings.Repeat("=", filled), percentage(event.Bytes, event.TotalBytes),
			humanInteger(int64(event.Current)), humanInteger(int64(event.Total)),
			humanBytes(event.Bytes), humanBytes(event.TotalBytes), phase, event.Shard.SHA256[:12])
		if event.Phase == "complete" && event.Current == event.Total {
			fmt.Fprintln(output)
		}
	}
}

func percentage(current, total int64) int64 {
	if total <= 0 {
		return 0
	}
	value := current * 100 / total
	if value > 100 {
		return 100
	}
	return value
}

func writeModelMutationResult(context Context, stdout io.Writer, inspection model.Inspection, verb string) error {
	if context.JSON {
		return writeJSON(stdout, inspection)
	}
	fmt.Fprintf(stdout, "%s model %s\n", verb, inspection.Model.Name)
	fmt.Fprintf(stdout, "  location      %s\n", inspection.Path)
	fmt.Fprintf(stdout, "  model id      %s\n", shortModelHash(inspection.Model.ID))
	fmt.Fprintf(stdout, "  runs          %s\n", humanInteger(int64(len(inspection.Model.Runs))))
	if len(inspection.RunBOMs) > 0 {
		backend := inspection.RunBOMs[len(inspection.RunBOMs)-1].Execution.Backend
		fmt.Fprintf(stdout, "  backend       %s@%s\n", backend.Name, backend.Revision)
		if backend.Name == "fake" {
			fmt.Fprintln(stdout, "  warning       explicitly simulated; artifacts are not trained model weights")
		}
	}
	return nil
}

func shortModelHash(value string) string {
	if len(value) > 12 {
		return value[:12]
	}
	return value
}

func humanIntegerUint(value uint64) string {
	if value <= uint64(^uint64(0)>>1) {
		return humanInteger(int64(value))
	}
	return fmt.Sprintf("%d", value)
}

func humanModelParameters(value uint64) string {
	exact := humanIntegerUint(value)
	if value > uint64(math.MaxInt64) {
		return exact
	}
	compact := humanCount(int64(value))
	if compact == exact {
		return exact
	}
	return fmt.Sprintf("%s (%s)", compact, exact)
}

func humanBytesUint(value uint64) string {
	if value <= uint64(^uint64(0)>>1) {
		return humanBytes(int64(value))
	}
	return fmt.Sprintf("%d B", value)
}
