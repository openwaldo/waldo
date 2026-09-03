// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package composes_test

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

func TestModelComposeGuideNamesEverySchemaField(t *testing.T) {
	guide, err := os.ReadFile(filepath.Join("..", "docs", "MODEL-COMPOSE.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []any{
		model.Compose{}, model.ComposeBase{}, model.Architecture{}, model.Tokenizer{}, model.Stage{}, model.CorpusSelection{}, corpus.RecordFilter{}, corpus.ValueFilter{}, corpus.DateFilter{}, training.ConversationTransform{}, training.Parameters{},
	} {
		typeOf := reflect.TypeOf(value)
		for index := 0; index < typeOf.NumField(); index++ {
			name := strings.Split(typeOf.Field(index).Tag.Get("yaml"), ",")[0]
			if name == "" || name == "-" {
				continue
			}
			if !strings.Contains(string(guide), "`"+name+"`") && !strings.Contains(string(guide), name+":") {
				t.Errorf("%s.%s YAML field %q is absent from MODEL-COMPOSE.md", typeOf.Name(), typeOf.Field(index).Name, name)
			}
		}
	}
}

func TestEveryReferenceComposeSettingResolvesIntoTrainingContract(t *testing.T) {
	files, err := filepath.Glob("*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		t.Run(file, func(t *testing.T) {
			compose, _, err := model.LoadCompose(file)
			if err != nil {
				t.Fatal(err)
			}
			for _, stage := range compose.Stages {
				raw := stage.Parameters
				resolved, err := stage.ResolvePlanningParameters()
				if err != nil {
					t.Fatalf("stage %s: %v", stage.Name, err)
				}
				profile := raw.Profile
				if profile == "" {
					profile = training.DefaultProfile
				}
				epochs := raw.Epochs
				if epochs == 0 {
					epochs = 1
				}
				if resolved.Profile != profile || resolved.Epochs != epochs || resolved.BatchSize != raw.BatchSize || resolved.SequenceLength != raw.SequenceLength || resolved.LearningRate != raw.LearningRate || resolved.Seed != raw.Seed {
					t.Fatalf("stage %s direct settings were not preserved: raw=%+v resolved=%+v", stage.Name, raw, resolved)
				}
				if raw.Tokens > 0 && (resolved.RequestedTokens != raw.Tokens || resolved.PlannedTokenCapacity < raw.Tokens || resolved.PlannedTokenCapacity-raw.Tokens >= raw.BatchSize*raw.SequenceLength) {
					t.Fatalf("stage %s token budget = %d requested/%d planned", stage.Name, resolved.RequestedTokens, resolved.PlannedTokenCapacity)
				}
				if raw.Steps > 0 && resolved.PlannedTokenCapacity != raw.Steps*raw.BatchSize*raw.SequenceLength {
					t.Fatalf("stage %s planned capacity = %d", stage.Name, resolved.PlannedTokenCapacity)
				}
				assertOptionalFloat(t, stage.Name+" weight_decay", raw.WeightDecay, resolved.Optimizer.WeightDecay)
				assertOptionalInt64(t, stage.Name+" warmup_steps", raw.WarmupSteps, resolved.Schedule.WarmupSteps)
				assertOptionalInt64(t, stage.Name+" checkpoint_every", raw.CheckpointEvery, resolved.CheckpointEvery)
				assertOptionalInt64(t, stage.Name+" evaluate_every", raw.EvaluateEvery, resolved.EvaluateEvery)
				if raw.ShuffleBufferRecords != nil && resolved.Data.ShuffleBufferRecords != *raw.ShuffleBufferRecords {
					t.Fatalf("stage %s shuffle_buffer_records = %d, want %d", stage.Name, resolved.Data.ShuffleBufferRecords, *raw.ShuffleBufferRecords)
				}
				assertOptionalInt64(t, stage.Name+" shuffle_buffer_bytes", raw.ShuffleBufferBytes, resolved.Data.ShuffleBufferBytes)
				expectedWeights := raw.CorpusWeights
				for _, selection := range stage.Corpora {
					if selection.Weight == nil {
						continue
					}
					if expectedWeights == nil {
						expectedWeights = map[string]uint64{}
					}
					expectedWeights[selection.Path] = *selection.Weight
				}
				if !reflect.DeepEqual(resolved.Data.CorpusWeights, expectedWeights) {
					t.Fatalf("stage %s corpus weights = %v, want %v", stage.Name, resolved.Data.CorpusWeights, expectedWeights)
				}
				if resolved.Evaluation == nil {
					t.Fatalf("stage %s has no resolved evaluation policy", stage.Name)
				}
				assertOptionalFloat(t, stage.Name+" evaluation_fraction", raw.EvaluationFraction, resolved.Evaluation.Fraction)
				if raw.EvaluationMaxRecords != nil && resolved.Evaluation.MaxRecords != *raw.EvaluationMaxRecords {
					t.Fatalf("stage %s evaluation_max_records = %d, want %d", stage.Name, resolved.Evaluation.MaxRecords, *raw.EvaluationMaxRecords)
				}
				assertOptionalInt64(t, stage.Name+" evaluation_max_bytes", raw.EvaluationMaxBytes, resolved.Evaluation.MaxBytes)
			}
		})
	}
}

func assertOptionalInt64(t *testing.T, name string, declared *int64, resolved int64) {
	t.Helper()
	if declared != nil && resolved != *declared {
		t.Fatalf("%s = %d, want %d", name, resolved, *declared)
	}
}

func assertOptionalFloat(t *testing.T, name string, declared *float64, resolved float64) {
	t.Helper()
	if declared != nil && resolved != *declared {
		t.Fatalf("%s = %g, want %g", name, resolved, *declared)
	}
}

func TestReferenceCanaryIsExecutableAndCompact(t *testing.T) {
	files, err := filepath.Glob("*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"0000-canary.yaml", "0001-babble.yaml", "0002-conversation1.yaml", "0002-conversation2.yaml", "0003-tool-use.yaml"}
	if !reflect.DeepEqual(files, want) {
		t.Fatalf("reference composes = %v, want %v", files, want)
	}
	compose, _, err := model.LoadCompose("0000-canary.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(compose.Stages) != 1 || len(compose.Stages[0].Corpora) != 4 {
		t.Fatalf("canary stages/corpora = %d/%d", len(compose.Stages), len(compose.Stages[0].Corpora))
	}
	if compose.Architecture.Tokenizer.Name != "tiktoken/cl100k_base" || compose.Architecture.Tokenizer.Revision != "tiktoken-cl100k-base" || compose.Architecture.VocabularySize != 100259 {
		t.Fatalf("canary does not use the portable subword tokenizer: %+v", compose.Architecture.Tokenizer)
	}
	forecast, err := model.ForecastCompose(compose)
	if err != nil {
		t.Fatal(err)
	}
	if forecast.ApproximateParameters != 13620736 || forecast.PlannedTokens != 4096000 {
		t.Fatalf("canary forecast = %d parameters/%d tokens", forecast.ApproximateParameters, forecast.PlannedTokens)
	}
}

func TestToolUseComposeHasSizedBaseAndStructuredToolStage(t *testing.T) {
	tooling, _, err := model.LoadCompose("0003-tool-use.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if tooling.Base == nil || tooling.Base.Model != "conversation" {
		t.Fatalf("tooling base = %+v", tooling.Base)
	}
	if len(tooling.Stages) != 1 || tooling.Stages[0].Name != "tool-use-sft" {
		t.Fatalf("tooling curriculum = %+v", tooling.Stages)
	}
	stage := tooling.Stages[0]
	if tooling.Interaction.Template != model.InteractionUserAssistantV1 || !tooling.Interaction.Tools || stage.Objective != "assistant-response-modeling" || stage.Conversation == nil || stage.Conversation.Tools || !reflect.DeepEqual(stage.Conversation.SupervisedRoles, []string{"assistant"}) {
		t.Fatalf("tool interaction contract = %+v / %+v", tooling.Interaction, stage)
	}
	wantTools := []string{"post-train/sft/hermes-function-calling", "post-train/sft/interaction-contract-v1", "post-train/sft/helpsteer2"}
	if got := corpusPaths(stage.Corpora); !reflect.DeepEqual(got, wantTools) {
		t.Fatalf("tool-use corpora = %v, want %v", got, wantTools)
	}
	wantWeights := []uint64{4, 4, 2}
	for index, selection := range stage.Corpora {
		if selection.Weight == nil || *selection.Weight != wantWeights[index] {
			t.Fatalf("tool-use corpus %s weight = %v, want %d", selection.Path, selection.Weight, wantWeights[index])
		}
	}
	if stage.Parameters.Tokens != 20000000 || stage.Parameters.Epochs != 0 {
		t.Fatalf("tool-use budget = %d tokens/%d epochs", stage.Parameters.Tokens, stage.Parameters.Epochs)
	}
	forecast, err := model.ForecastCompose(tooling)
	if err != nil {
		t.Fatal(err)
	}
	if forecast.ApproximateParameters != 336637440 || forecast.PlannedTokens != 20004864 {
		t.Fatalf("tool forecast = %d parameters/%d tokens", forecast.ApproximateParameters, forecast.PlannedTokens)
	}
}

func corpusPaths(selections []model.CorpusSelection) []string {
	paths := make([]string, len(selections))
	for index, selection := range selections {
		paths[index] = selection.Path
	}
	return paths
}

func TestBasicConversationPreservesValidatedTrainingSequence(t *testing.T) {
	compose, _, err := model.LoadCompose("0002-conversation1.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(compose.Stages) != 3 || compose.Stages[0].Name != "pretrain" || compose.Stages[1].Name != "conversational-midtrain" || compose.Stages[2].Name != "post-train" {
		t.Fatalf("basic conversation stages = %+v", compose.Stages)
	}
	for _, stage := range compose.Stages {
		if stage.Filter == nil || stage.Filter.MainContent == nil || !*stage.Filter.MainContent || stage.Filter.Exclude == nil || stage.Filter.Exclude.RepetitiveContent == nil || stage.Filter.Exclude.BoilerplateContent == nil {
			t.Fatalf("basic conversation stage %s quality filter = %+v", stage.Name, stage.Filter)
		}
	}
	if compose.Stages[1].Corpora[0].Path != "post-train/sft/oasst1" || compose.Stages[1].Corpora[1].Path != "post-train/sft/oasst2" {
		t.Fatalf("basic conversation conversational corpora = %+v", compose.Stages[1].Corpora)
	}
	if compose.Stages[2].Corpora[0].Path != "post-train/sft/interaction-contract-v1" || compose.Stages[2].Corpora[1].Path != "post-train/sft/helpsteer2" {
		t.Fatalf("basic conversation post-training corpora = %+v", compose.Stages[2].Corpora)
	}
	if compose.Stages[1].Objective != "causal-language-modeling" || compose.Stages[2].Objective != "causal-language-modeling" {
		t.Fatalf("basic conversation SFT objectives = %s/%s", compose.Stages[1].Objective, compose.Stages[2].Objective)
	}
	if compose.Stages[0].Parameters.LearningRate <= compose.Stages[1].Parameters.LearningRate || compose.Stages[1].Parameters.LearningRate <= compose.Stages[2].Parameters.LearningRate {
		t.Fatalf("basic conversation learning rates do not decay by phase: %+v", compose.Stages)
	}
	forecast, err := model.ForecastCompose(compose)
	if err != nil {
		t.Fatal(err)
	}
	if forecast.ApproximateParameters != 336637440 || forecast.PlannedTokens != 11999969280 || len(forecast.EpochDerivedStages) != 2 {
		t.Fatalf("basic conversation forecast = %d parameters/%d tokens", forecast.ApproximateParameters, forecast.PlannedTokens)
	}
}

func TestConversationTwoExtendsConversationOneWithNewDialogueData(t *testing.T) {
	baseline, _, err := model.LoadCompose("0002-conversation1.yaml")
	if err != nil {
		t.Fatal(err)
	}
	variant, _, err := model.LoadCompose("0002-conversation2.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if variant.Base != nil || variant.Architecture != baseline.Architecture || variant.Interaction != baseline.Interaction {
		t.Fatalf("conversation2 model contract = %+v", variant)
	}
	if len(variant.Stages) != len(baseline.Stages)+2 || !reflect.DeepEqual(variant.Stages[:len(baseline.Stages)], baseline.Stages) {
		t.Fatalf("conversation2 does not preserve the complete conversation1 curriculum")
	}
	technical := variant.Stages[len(baseline.Stages)]
	if technical.Name != "technical-midtrain" || technical.Type != "pre-training" || technical.Objective != "causal-language-modeling" {
		t.Fatalf("conversation2 technical stage = %+v", technical)
	}
	wantTechnical := []string{
		"community/linux-kernel-mailing-list", "community/git-mailing-list", "community/python-mailing-lists",
		"community/apache-mailing-lists", "community/gcc-mailing-lists", "community/glibc-mailing-lists",
		"community/gnu-mailing-lists", "community/qemu-devel-mailing-list", "community/alpine-linux-mailing-list",
		"community/opensource-mailing-lists", "community/github-archive", "code/stack-v2-html",
	}
	if got := corpusPaths(technical.Corpora); !reflect.DeepEqual(got, wantTechnical) {
		t.Fatalf("conversation2 technical corpora = %v, want %v", got, wantTechnical)
	}
	if technical.Parameters.Tokens != 400000000 || technical.Filter == nil || technical.Filter.Languages == nil || !reflect.DeepEqual(technical.Filter.Languages.Include, []string{"en"}) {
		t.Fatalf("conversation2 technical budget/filter = %+v / %+v", technical.Parameters, technical.Filter)
	}
	stage := variant.Stages[len(variant.Stages)-1]
	if stage.Name != "expanded-conversation-sft" || stage.Objective != "assistant-response-modeling" {
		t.Fatalf("conversation2 added stage = %+v", stage)
	}
	wantCorpora := []string{"post-train/sft/smol-smoltalk", "post-train/sft/ultrachat-200k"}
	if got := corpusPaths(stage.Corpora); !reflect.DeepEqual(got, wantCorpora) {
		t.Fatalf("conversation2 corpora = %v, want %v", got, wantCorpora)
	}
	forecast, err := model.ForecastCompose(variant)
	if err != nil {
		t.Fatal(err)
	}
	if forecast.ApproximateParameters != 336637440 || forecast.PlannedTokens != 12499992576 {
		t.Fatalf("conversation2 forecast = %d parameters/%d tokens", forecast.ApproximateParameters, forecast.PlannedTokens)
	}
}

func TestBabbleUsesCleanPretrainingAndLightConversationTuning(t *testing.T) {
	compose, _, err := model.LoadCompose("0001-babble.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(compose.Stages) != 3 || compose.Stages[0].Type != "pre-training" || compose.Stages[1].Objective != "assistant-response-modeling" || compose.Stages[2].Objective != "assistant-response-modeling" {
		t.Fatalf("babble stages = %+v", compose.Stages)
	}
	if compose.Interaction.Template != model.InteractionUserAssistantV1 {
		t.Fatalf("babble interaction = %+v", compose.Interaction)
	}
	for _, stage := range compose.Stages {
		if stage.Filter == nil || stage.Filter.MainContent == nil || !*stage.Filter.MainContent || stage.Filter.Exclude == nil || stage.Filter.Exclude.RepetitiveContent == nil || stage.Filter.Exclude.BoilerplateContent == nil {
			t.Fatalf("babble stage %s quality filter = %+v", stage.Name, stage.Filter)
		}
	}
	if compose.Architecture.ContextTokens != 1024 || compose.Architecture.Dropout != 0.1 || compose.Architecture.Tokenizer.Name != "tiktoken/r50k_base" || compose.Architecture.VocabularySize != 50259 {
		t.Fatalf("babble architecture = %+v", compose.Architecture)
	}
	forecast, err := model.ForecastCompose(compose)
	if err != nil {
		t.Fatal(err)
	}
	if forecast.ApproximateParameters != 76416000 || forecast.PlannedTokens != 1572864000 || len(forecast.EpochDerivedStages) != 2 {
		t.Fatalf("babble forecast = %d parameters/%d tokens", forecast.ApproximateParameters, forecast.PlannedTokens)
	}
	if compose.Stages[0].Parameters.LearningRate <= compose.Stages[1].Parameters.LearningRate || compose.Stages[1].Parameters.LearningRate <= compose.Stages[2].Parameters.LearningRate {
		t.Fatalf("babble learning rates do not decay by phase: %+v", compose.Stages)
	}
	if compose.Stages[2].Corpora[0].Path != "post-train/sft/interaction-contract-v1" || compose.Stages[2].Corpora[1].Path != "post-train/sft/helpsteer2" {
		t.Fatalf("babble post-training corpora = %+v", compose.Stages[2].Corpora)
	}
}

func TestReferenceComposesDoNotEncodeHardwareInNames(t *testing.T) {
	for _, legacy := range []string{"babble-mac.yaml", "h200-02h.yaml", "h200-06h.yaml", "h200-12h.yaml", "h200-24h.yaml", "h200-48h.yaml"} {
		if _, err := os.Stat(legacy); !os.IsNotExist(err) {
			t.Fatalf("hardware-specific compose %s still exists", legacy)
		}
	}
}
