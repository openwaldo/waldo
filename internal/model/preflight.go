// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"math"
	"reflect"
	"strings"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/record"
	"github.com/openwaldo/waldo/internal/shard"
	"github.com/openwaldo/waldo/internal/training"
)

// PreparedStage is the model-domain boundary for an already resolved and
// verified corpus selection. The CLI/corpus layers own index and lookaside
// access; the model lifecycle receives only an immutable BOM and local,
// content-addressed inputs.
type PreparedStage struct {
	Stage  Stage
	BOM    corpus.BOM
	Inputs []training.Input
}

func PrepareStage(stage Stage, bom corpus.BOM, inputs []training.Input) (PreparedStage, error) {
	if err := validateStage(stage, Architecture{}); err != nil {
		return PreparedStage{}, err
	}
	if err := bom.Validate(); err != nil {
		return PreparedStage{}, fmt.Errorf("corpus OpenWALDO BOM: %w", err)
	}
	if len(bom.Shards) == 0 || bom.Totals.Docs <= 0 || bom.Totals.Tokens <= 0 {
		return PreparedStage{}, fmt.Errorf("stage %s corpus selection contains no training records", stage.Name)
	}
	for _, selected := range bom.Shards {
		kind := selected.RecordKind
		if kind == "" {
			kind = record.KindPretrain
		}
		supportedText := kind == record.KindPretrain && selected.RecordSchema >= shard.FormerTextRecordSchema && selected.RecordSchema <= shard.TextRecordSchema
		supportedConversation := kind == record.KindConversation && selected.RecordSchema == shard.ConversationRecordSchema
		if selected.Format != "parquet" || (!supportedText && !supportedConversation) {
			return PreparedStage{}, fmt.Errorf("stage %s shard %s is %s record schema %d; causal-language-modeling requires supported Parquet record schema", stage.Name, selected.SHA256[:12], selected.Format, selected.RecordSchema)
		}
		if stage.Objective == "assistant-response-modeling" && !supportedConversation {
			return PreparedStage{}, fmt.Errorf("stage %s assistant-response-modeling requires structured conversation shards; %s is %s", stage.Name, selected.SHA256[:12], kind)
		}
		if supportedConversation && stage.Conversation == nil {
			return PreparedStage{}, fmt.Errorf("stage %s selects conversation shard %s without a conversation transformation", stage.Name, selected.SHA256[:12])
		}
	}
	if len(inputs) == 0 {
		return PreparedStage{}, fmt.Errorf("stage %s has no materialized shard inputs", stage.Name)
	}
	type expectedInput struct {
		bytes   int64
		records int64
	}
	expected := make(map[string]expectedInput, len(bom.Shards))
	for _, selected := range bom.Shards {
		value := expectedInput{bytes: selected.Bytes, records: selected.Docs}
		if previous, exists := expected[selected.SHA256]; exists && previous != value {
			return PreparedStage{}, fmt.Errorf("stage %s object %s has conflicting declared sizes", stage.Name, selected.SHA256[:12])
		}
		expected[selected.SHA256] = value
	}
	seen := make(map[string]bool, len(inputs))
	resolved, err := stage.ResolvePlanningParameters()
	if err != nil {
		return PreparedStage{}, fmt.Errorf("stage %s training parameters: %w", stage.Name, err)
	}
	selected := make(map[string]bool, len(bom.Paths))
	for _, path := range bom.Paths {
		selected[path] = true
	}
	if resolved.Data.Order == "corpus-weighted-shuffle-v1" {
		canonical, err := resolveCorpusWeights(resolved.Data.CorpusWeights, bom.Paths)
		if err != nil {
			return PreparedStage{}, fmt.Errorf("stage %s %w", stage.Name, err)
		}
		resolved.Data.CorpusWeights = canonical
	}
	for _, input := range inputs {
		expectedInput, exists := expected[input.SHA256]
		if input.Path == "" || !exists || input.Bytes != expectedInput.bytes || input.Records != expectedInput.records || seen[input.SHA256] || !reflect.DeepEqual(input.RecordFilter, bom.RecordFilter) {
			return PreparedStage{}, fmt.Errorf("stage %s has an invalid or duplicate materialized input %s", stage.Name, input.SHA256)
		}
		if (resolved.Data.Order == "corpus-balanced-shuffle-v1" || resolved.Data.Order == "corpus-weighted-shuffle-v1") && !selected[input.Corpus] {
			return PreparedStage{}, fmt.Errorf("stage %s input %s has invalid corpus identity %q", stage.Name, input.SHA256, input.Corpus)
		}
		seen[input.SHA256] = true
	}
	if resolved.Data.Order == "corpus-weighted-shuffle-v1" {
		for path := range selected {
			if resolved.Data.CorpusWeights[path] == 0 {
				return PreparedStage{}, fmt.Errorf("stage %s corpus_weights does not declare selected corpus %q", stage.Name, path)
			}
		}
		for path := range resolved.Data.CorpusWeights {
			if !selected[path] {
				return PreparedStage{}, fmt.Errorf("stage %s corpus_weights declares unselected corpus %q", stage.Name, path)
			}
		}
	}
	if len(seen) != len(expected) {
		return PreparedStage{}, fmt.Errorf("stage %s materialized %d of %d unique shard objects", stage.Name, len(seen), len(expected))
	}
	return PreparedStage{Stage: stage, BOM: bom, Inputs: append([]training.Input(nil), inputs...)}, nil
}

func resolveCorpusWeights(declared map[string]uint64, paths []string) (map[string]uint64, error) {
	resolved := make(map[string]uint64, len(paths))
	used := make(map[string]bool, len(declared))
	for _, path := range paths {
		logical := strings.TrimSuffix(strings.TrimSuffix(path, ".yaml"), ".json")
		for name, weight := range declared {
			if name != path && name != logical {
				continue
			}
			if _, exists := resolved[path]; exists || used[name] {
				return nil, fmt.Errorf("corpus_weights ambiguously resolves corpus %q", path)
			}
			resolved[path], used[name] = weight, true
		}
		if resolved[path] == 0 {
			return nil, fmt.Errorf("corpus_weights does not declare selected corpus %q", logical)
		}
	}
	for name := range declared {
		if !used[name] {
			return nil, fmt.Errorf("corpus_weights declares unselected corpus %q", name)
		}
	}
	return resolved, nil
}

func composePlan(name string, compose Compose) (Plan, error) {
	architectureHash, err := canonicalHash(compose.Architecture)
	if err != nil {
		return Plan{}, err
	}
	forecast, err := compose.Architecture.Forecast()
	if err != nil {
		return Plan{}, err
	}
	plan := Plan{
		Kind: "waldo-model-plan", Schema: PlanSchema, Name: name,
		ArchitectureSHA256: architectureHash, Architecture: compose.Architecture,
		Interaction: compose.Interaction, Forecast: forecast,
	}
	if compose.Base != nil {
		plan.OriginBOMSHA256 = compose.Base.OriginSHA256
		if compose.Base.RunID != "" {
			plan.Parent = &ModelParent{
				Model: compose.Base.Model, ModelID: compose.Base.ModelID,
				RunID: compose.Base.RunID, RunBOMSHA256: compose.Base.RunBOMSHA256,
				Artifact: training.Artifact{Path: "base/model.safetensors", SHA256: compose.Base.ArtifactSHA256, Bytes: compose.Base.ArtifactBytes},
			}
		}
	}
	return plan, nil
}

func forecastPlanForCompose(compose Compose) (Plan, error) {
	plan, err := composePlan("forecast", compose)
	if err != nil {
		return Plan{}, err
	}
	for _, stage := range compose.Stages {
		resolved, err := stage.ResolvePlanningParameters()
		if err != nil {
			return Plan{}, fmt.Errorf("stage %s training parameters: %w", stage.Name, err)
		}
		plannedTokens := resolved.PlannedTokenCapacity
		if stage.Parameters.Steps == 0 && stage.Parameters.Tokens == 0 {
			plannedTokens = 0
		}
		plan.Stages = append(plan.Stages, PlannedStage{Name: stage.Name, Type: stage.Type, Objective: stage.Objective, Parameters: stage.Parameters, PlannedTokens: plannedTokens})
	}
	return plan, nil
}

func validateStage(stage Stage, architecture Architecture) error {
	if !validName.MatchString(stage.Name) {
		return fmt.Errorf("invalid training stage name %q", stage.Name)
	}
	if stage.Type != "pre-training" && stage.Type != "fine-tuning" && stage.Type != "alignment" && stage.Type != "other" {
		return fmt.Errorf("stage %s has unsupported type %q", stage.Name, stage.Type)
	}
	if stage.Objective != "causal-language-modeling" && stage.Objective != "assistant-response-modeling" {
		return fmt.Errorf("stage %s has unsupported objective %q", stage.Name, stage.Objective)
	}
	if stage.Conversation != nil {
		if err := stage.Conversation.Validate(); err != nil {
			return fmt.Errorf("stage %s conversation: %w", stage.Name, err)
		}
	} else if stage.Objective == "assistant-response-modeling" {
		return fmt.Errorf("stage %s assistant-response-modeling requires conversation transformation", stage.Name)
	}
	parameters := stage.Parameters
	if _, err := stage.ResolvePlanningParameters(); err != nil {
		return fmt.Errorf("stage %s training parameters: %w", stage.Name, err)
	}
	if architecture.ContextTokens > 0 && uint64(parameters.SequenceLength) > architecture.ContextTokens {
		return fmt.Errorf("stage %s sequence_length exceeds architecture context_tokens", stage.Name)
	}
	return nil
}

func multiplyInt64(values ...int64) (int64, bool) {
	result := int64(1)
	for _, value := range values {
		if value <= 0 || result > math.MaxInt64/value {
			return 0, true
		}
		result *= value
	}
	return result, false
}
