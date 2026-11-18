// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"fmt"
	"io"

	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
)

// runModelEvaluate prepares the compose's first stage as training would and scores the model on it.
func runModelEvaluate(context Context, args []string, stdout, stderr io.Writer) error {
	for _, flag := range []string{"epochs", "batch-size", "learning-rate", "seed", "hostfile", "rendezvous-port", "nodes", "rendezvous", "rendezvous-id", "audit"} {
		if context.Command != nil && context.Command.Flags().Changed(flag) {
			return fmt.Errorf("--%s cannot be used with --evaluate-only, which scores the compose's first stage on this host", flag)
		}
	}
	name := args[0]
	path, err := trainingComposeInput(args[1:])
	if err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("--evaluate-only needs a compose file that selects the held-out records")
	}
	cluster, err := trainingClusterFromFlags(context, 0)
	if err != nil {
		return err
	}
	compose, _, err := model.LoadCompose(path)
	if err != nil {
		return err
	}
	builder, err := configuredModelBuilderForCluster(context, stderr, cluster)
	if err != nil {
		return err
	}
	if err := compose.Validate(); err != nil {
		return err
	}
	compose.Stages = compose.Stages[:1]
	stage := compose.Stages[0]
	if err := builder.CheckEvaluation(context.Execution, name, compose); err != nil {
		return err
	}
	targets, err := sanityCheckComposeCorpora(context.Execution, compose, stderr)
	if err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	planned, err := planModelStage(context, stage, targets[stage.Name], cache, stderr)
	if err != nil {
		return err
	}
	prepared, err := materializeModelStage(context, planned.Stage, planned.BOM, cache, stderr, false)
	if err != nil {
		return err
	}
	result, err := builder.Evaluate(context.Execution, name, prepared)
	if err != nil {
		return err
	}
	if _, err := cache.PurgeUsed(); err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, result)
	}
	fmt.Fprintf(stdout, "evaluated model %s\n", name)
	fmt.Fprintf(stdout, "  held-out loss  %.4f\n", result.Loss)
	fmt.Fprintf(stdout, "  held-out set   %d records, %d targets, %s\n", result.Set.Records, result.Set.TokenTargets, shortModelHash(result.Set.SHA256))
	fmt.Fprintf(stdout, "  weights        %s\n", shortModelHash(result.WeightsSHA256))
	return nil
}
