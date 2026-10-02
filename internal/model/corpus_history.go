// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"

	"github.com/openwaldo/waldo/internal/training"
)

// SkippedCorpus identifies a compose selection omitted because this model has
// already completed a run containing the same logical corpus path.
type SkippedCorpus struct {
	Stage string `json:"stage"`
	Path  string `json:"path"`
}

// SkipCompletedCorpora removes previously completed corpus paths from a
// compose. It intentionally compares logical paths only; it does not attempt
// record- or shard-level delta training.
func SkipCompletedCorpora(compose Compose, inspection Inspection) (Compose, []SkippedCorpus) {
	completed := map[string]bool{}
	for position, pin := range inspection.Model.Runs {
		if pin.State != RunComplete || position >= len(inspection.RunBOMs) {
			continue
		}
		for _, path := range inspection.RunBOMs[position].CorpusBOM.Paths {
			completed[logicalCorpusPath(path)] = true
		}
	}
	if len(completed) == 0 {
		return compose, nil
	}

	filtered := compose
	filtered.Stages = make([]Stage, 0, len(compose.Stages))
	var skipped []SkippedCorpus
	for _, original := range compose.Stages {
		stage := original
		stage.Corpora = make([]CorpusSelection, 0, len(original.Corpora))
		stage.Parameters.CorpusWeights = maps.Clone(original.Parameters.CorpusWeights)
		for _, selection := range original.Corpora {
			if !completed[logicalCorpusPath(selection.Path)] {
				stage.Corpora = append(stage.Corpora, selection)
				continue
			}
			skipped = append(skipped, SkippedCorpus{Stage: stage.Name, Path: selection.Path})
			for path := range stage.Parameters.CorpusWeights {
				if logicalCorpusPath(path) == logicalCorpusPath(selection.Path) {
					delete(stage.Parameters.CorpusWeights, path)
				}
			}
		}
		if len(stage.Corpora) > 0 {
			filtered.Stages = append(filtered.Stages, stage)
		}
	}
	return filtered, skipped
}

func logicalCorpusPath(path string) string {
	path = strings.TrimSuffix(strings.TrimSpace(path), "/")
	return strings.TrimSuffix(strings.TrimSuffix(path, ".yaml"), ".json")
}

// ComposeCompleted reports whether the final successful run lineage proves
// every stage in the requested compose. Corpus-path history alone is
// deliberately insufficient: stage order, filters, budgets, conversation
// transforms, and all resolved training parameters must also match.
func ComposeCompleted(compose Compose, inspection Inspection) (bool, error) {
	if len(compose.Stages) == 0 || len(inspection.Runs) != len(inspection.RunBOMs) || len(inspection.Runs) < len(compose.Stages) {
		return false, nil
	}
	// Failed and interrupted attempts do not publish model weights, so they do
	// not change the completed lineage. The requested compose must match the
	// suffix of that lineage: an older matching sequence is not proof of the
	// model's current state after later successful training.
	var completed []int
	for position, run := range inspection.Runs {
		if run.State == RunComplete {
			completed = append(completed, position)
		}
	}
	if len(completed) < len(compose.Stages) {
		return false, nil
	}
	completed = completed[len(completed)-len(compose.Stages):]
	for offset, stage := range compose.Stages {
		position := completed[offset]
		stageMatches, err := completedRunMatchesStage(stage, inspection.RunBOMs[position])
		if err != nil {
			return false, fmt.Errorf("compare completed run %d with stage %s: %w", position+1, stage.Name, err)
		}
		if !stageMatches {
			return false, nil
		}
	}
	return true, nil
}

func completedRunMatchesStage(stage Stage, bom RunBOM) (bool, error) {
	conversation := training.ConversationTransform{}
	if stage.Conversation != nil {
		conversation = *stage.Conversation
	}
	if bom.Stage != stage.Name || bom.StageType != stage.Type || bom.Objective != stage.Objective || !reflect.DeepEqual(bom.Conversation, conversation) {
		return false, nil
	}
	declaredPaths := make([]string, len(stage.Corpora))
	for index, selection := range stage.Corpora {
		declaredPaths[index] = logicalCorpusPath(selection.Path)
	}
	completedPaths := make([]string, len(bom.CorpusBOM.Paths))
	for index, path := range bom.CorpusBOM.Paths {
		completedPaths[index] = logicalCorpusPath(path)
	}
	slices.Sort(declaredPaths)
	slices.Sort(completedPaths)
	if !reflect.DeepEqual(declaredPaths, completedPaths) {
		return false, nil
	}
	filter, err := stage.RecordFilterPolicy(bom.CorpusBOM.Paths)
	if err != nil {
		return false, err
	}
	if !reflect.DeepEqual(filter, bom.CorpusBOM.RecordFilter) {
		return false, nil
	}
	var parameters training.ResolvedParameters
	if stage.Parameters.Epochs > 0 && stage.Parameters.Steps == 0 && stage.Parameters.Tokens == 0 {
		parameters, err = stage.ResolveParametersForSteps(bom.Parameters.Steps)
	} else {
		parameters, err = stage.ResolveParameters()
	}
	if err != nil {
		return false, err
	}
	if parameters.Data.Order == "corpus-weighted-shuffle-v1" {
		parameters.Data.CorpusWeights, err = resolveCorpusWeights(parameters.Data.CorpusWeights, bom.CorpusBOM.Paths)
		if err != nil {
			return false, err
		}
	}
	return equivalentResumeParameters(parameters, bom.Parameters), nil
}
