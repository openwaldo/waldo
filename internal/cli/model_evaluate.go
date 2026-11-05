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
	"github.com/openwaldo/waldo/internal/training"
)

// runModelEvaluate scores a model's current weights on the held-out records
// the compose's first stage selects, preparing the corpus exactly as training
// would, without training the model or recording a run.
func runModelEvaluate(context Context, name, path string, cluster training.Cluster, stdout, stderr io.Writer) error {
	compose, _, err := model.LoadCompose(path)
	if err != nil {
		return err
	}
	builder, err := configuredModelBuilderForCluster(context, stderr, cluster)
	if err != nil {
		return err
	}
	compose, err = builder.ResolveCompose(context.Execution, name, compose, true)
	if err != nil {
		return err
	}
	if len(compose.Stages) == 0 {
		return fmt.Errorf("--evaluate-only needs a compose with a stage that selects held-out records")
	}
	stage := compose.Stages[0]
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
		return fmt.Errorf("purge evaluation scratch: %w", err)
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
