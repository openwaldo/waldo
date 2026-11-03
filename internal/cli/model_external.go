// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/disclosure"
	"github.com/openwaldo/waldo/internal/modelexport"
	"github.com/openwaldo/waldo/internal/signing"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/training"
)

func runExternalComposeTraining(command Context, name string, c model.ExternalCompose, stdout, stderr io.Writer) error {
	return runExternalComposeTrainingCluster(command, name, c, training.Cluster{}, nil, stdout, stderr)
}

func runExternalComposeTrainingCluster(command Context, name string, c model.ExternalCompose, cluster training.Cluster, session *hostfileSession, stdout, stderr io.Writer) error {
	builder, err := configuredModelBuilder(command, stderr)
	if err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	builder.PreparedCacheMaxBytes = cache.MaxBytes()
	projection := model.Compose{}
	for _, s := range c.Stages {
		projection.Stages = append(projection.Stages, s.DataStage())
	}
	targets, err := sanityCheckComposeCorpora(command.Execution, projection, stderr)
	if err != nil {
		return err
	}
	// These are local execution settings; neither enters portable plan identity.
	applicationCache := os.Getenv("WALDO_APPLICATION_CACHE")
	if applicationCache == "" {
		applicationCache = filepath.Join(builder.Root, ".applications")
	}
	profile := stringOption(command, "launch-profile")
	processes := intOption(command, "processes-per-node")
	if processes == 0 {
		processes = 1
	}
	if profile == "" {
		profile = "single-process"
	}
	if profile == "single-process" && processes != 1 {
		return fmt.Errorf("multiple processes require --launch-profile distributed")
	}
	if processes < 1 {
		return fmt.Errorf("processes per node must be positive")
	}
	configuration, err := config.Load()
	if err != nil {
		return err
	}
	world := applaunch.World{Nodes: 1, ProcessesPerNode: processes, WorldSize: processes}
	var publish func(model.ExternalNodePlan) error
	var hosts []string
	if session != nil {
		hosts = append([]string(nil), session.hostfile.Hosts...)
		profile = "distributed"
		world.Nodes = cluster.Nodes
		world.WorldSize = cluster.WorldSize
		world.ProcessesPerNode = cluster.WorldSize / cluster.Nodes
		host, port, err := net.SplitHostPort(cluster.Rendezvous)
		if err != nil {
			return err
		}
		world.MasterAddress = host
		world.MasterPort, err = strconv.Atoi(port)
		if err != nil {
			return err
		}
		publish = func(p model.ExternalNodePlan) error {
			ordinal := 0
			for i, s := range c.Stages {
				if s.Name == p.Stage.Name {
					ordinal = i + 1
				}
			}
			return session.publish(model.MultiNodePlan{Kind: model.MultiNodePlanKind, Schema: model.MultiNodePlanSchema, RunID: p.Begin.RunID, Stage: p.Stage.Name, StageOrdinal: ordinal, StageCount: len(c.Stages), Nodes: world.Nodes, External: &p})
		}
	}
	if command.Command != nil && command.Command.Name() == "continue" {
		inspection, err := model.InspectExternal(builder.Root, name)
		if err != nil {
			return err
		}
		if len(inspection.Runs) > 0 && len(inspection.Runs[0].Attempts) > 0 {
			launch := inspection.Runs[0].Attempts[0].Launch
			if launch.World.Nodes > 1 {
				return fmt.Errorf("repeat the original hostfile command to continue a distributed multi-node run")
			}
			profile = launch.Profile
			world = launch.World
		}
	}
	result, err := builder.ComposeExternal(command.Execution, name, c, model.ExternalOptions{Hosts: hosts, Publish: publish, Profile: profile, World: world, NCCLInterface: configuration.Model.NCCLInterface, NCCLHCA: configuration.Model.NCCLHCA, Cache: applicationCache, Python: os.Getenv("WALDO_APPLICATION_PYTHON"), Prepare: func(ctx context.Context, s model.ExternalStage) (corpus.BOM, []training.Input, error) {
		stage := s.DataStage()
		planned, err := planModelStage(command, stage, targets[s.Name], cache, stderr)
		if err != nil {
			return corpus.BOM{}, nil, err
		}
		materialized, err := corpus.Materialize(ctx, planned.BOM, cache, modelMaterializeProgressPrinter(stderr))
		if err != nil {
			return corpus.BOM{}, nil, err
		}
		return planned.BOM, verifiedTrainingInputs(materialized, planned.BOM), nil
	}})
	if err != nil {
		return err
	}
	if command.JSON {
		return writeJSON(stdout, result)
	}
	fmt.Fprintf(stdout, "trained external model %s (%s): %d stages complete\n", name, result.Model.ID, len(result.Runs))
	return nil
}

func runExternalModelExport(command Context, root string, parsed modelExportOptions, stdout, stderr io.Writer) error {
	if parsed.Format != "waldo" && parsed.Format != "huggingface" {
		return fmt.Errorf("external hf-causal-lm export supports waldo or huggingface format")
	}
	if parsed.Quant != "" || parsed.Calibration != "" {
		return fmt.Errorf("external artifact conversion and quantization are not implemented")
	}
	inspection, err := model.InspectExternal(root, parsed.Name)
	if err != nil {
		return err
	}
	configuration, err := config.Load()
	if err != nil {
		return err
	}
	if configuration.Disclosure.Provider == "" {
		return fmt.Errorf("model export requires provider information; run waldo config set disclosure.provider <provider.json>")
	}
	provider, err := disclosure.LoadProvider(configuration.Disclosure.Provider)
	if err != nil {
		return err
	}
	report, err := disclosure.BuildExternalEUGPAIReport(inspection, &provider, time.Now())
	if err != nil {
		return err
	}
	if err = requireCompleteDisclosure(report, parsed.AllowIncomplete, stderr); err != nil {
		return err
	}
	euBOM, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	var boms []corpus.BOM
	for _, bom := range inspection.RunBOMs {
		boms = append(boms, bom.CorpusBOM)
	}
	attribution, err := corpus.AttributionNotice(boms)
	if err != nil {
		return err
	}
	options := modelexport.Options{EUBOM: append(euBOM, '\n'), Attribution: attribution}
	signed := signing.Configured(configuration.Signing)
	if signed {
		options.Finalize = func(directory string) error {
			return signing.SignExport(command.Execution, configuration.Signing, directory, stderr)
		}
	}
	output, err := modelexport.ExportExternal(command.Execution, inspection, parsed.Destination, options)
	if err != nil {
		return err
	}
	if !signed {
		fmt.Fprintln(stderr, "warning: model export is unsigned; configure signing.* to sign exports automatically")
	}
	if command.JSON {
		return writeJSON(stdout, struct {
			Name   string `json:"name"`
			Format string `json:"format"`
			Output string `json:"output"`
			Signed bool   `json:"signed"`
		}{parsed.Name, "hf-causal-lm", output, signed})
	}
	fmt.Fprintf(stdout, "exported hf-causal-lm model %s to %s\n", parsed.Name, output)
	return nil
}

// externalHostCapabilities verifies application pins before probing the shared
// NCCL/interface/memlock host contract in the managed trainer environment.
func externalHostCapabilities(ctx context.Context, raw string, bootstrap ...string) (training.TorchTitanHost, error) {
	var c model.ExternalCompose
	if err := json.Unmarshal([]byte(raw), &c); err != nil {
		return training.TorchTitanHost{}, err
	}
	cache := os.Getenv("WALDO_APPLICATION_CACHE")
	if cache == "" {
		root, err := os.UserCacheDir()
		if err != nil {
			return training.TorchTitanHost{}, err
		}
		cache = filepath.Join(root, "waldo", "applications")
	}
	python := os.Getenv("WALDO_APPLICATION_PYTHON")
	if len(bootstrap) > 0 && bootstrap[0] != "" {
		python = bootstrap[0]
	}
	if python == "" {
		python = "python3"
	}
	sources, err := c.ResolveSources(ctx, filepath.Join(cache, "sources"))
	if err != nil {
		return training.TorchTitanHost{}, err
	}
	environment, err := sources.Trainer.EnsureEnvironment(ctx, c.Trainer.Environment, python, filepath.Join(cache, "environments"))
	if err != nil {
		return training.TorchTitanHost{}, err
	}
	return training.InspectExternalDistributedHost(ctx, filepath.Join(environment.Run.CachePath, "bin", "python"))
}

func runExternalSecondaryPlan(command Context, cluster training.Cluster, plan model.MultiNodePlan, cache *lookaside.Cache, scratch string, stdout, stderr io.Writer) error {
	p := *plan.External
	world := p.Begin.World
	world.NodeRank = cluster.NodeRank
	materialized, err := corpus.Materialize(command.Execution, p.BOM.CorpusBOM, cache, modelMaterializeProgressPrinter(stderr))
	if err != nil {
		return err
	}
	applicationCache := os.Getenv("WALDO_APPLICATION_CACHE")
	if applicationCache == "" {
		applicationCache = filepath.Join(cache.Root(), "external-applications")
	}
	options := model.ExternalOptions{PreparedCacheMaxBytes: cache.MaxBytes(), Cache: applicationCache, Python: p.Python, World: world, NCCLInterface: cluster.Interface, NCCLHCA: cluster.HCA, Prepare: model.ExternalCorpusProvider(p.BOM.CorpusBOM, verifiedTrainingInputs(materialized, p.BOM.CorpusBOM))}
	return model.RunExternalSecondary(command.Execution, p, options, scratch, func() error {
		if err := checkTrainingRendezvous(command.Execution, cluster.Rendezvous); err != nil {
			return err
		}
		return json.NewEncoder(stdout).Encode(hostfileStageReady{Kind: hostfileStageReadyKind, Schema: 1, RunID: plan.RunID, Stage: plan.Stage, StageOrdinal: plan.StageOrdinal, StageCount: plan.StageCount})
	})
}
