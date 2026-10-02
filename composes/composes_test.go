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

var foundationFiles = []string{
	"0001-foundation-pipeline-canary.yaml",
	"0002-foundation-mixture-canary.yaml",
	"0003-foundation-small-language.yaml",
	"0003b-foundation-small-language-full.yaml",
	"0004-foundation-small-general.yaml",
	"0005-foundation-medium-pilot.yaml",
	"0006-foundation-medium.yaml",
}

func TestModelComposeGuideNamesEverySchemaField(t *testing.T) {
	guide, err := os.ReadFile(filepath.Join("..", "docs", "MODEL-COMPOSE.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []any{
		model.Compose{}, model.ComposeBase{}, model.Architecture{}, model.Tokenizer{}, model.TokenizerTraining{}, model.Stage{}, model.CorpusSelection{}, corpus.RecordFilter{}, corpus.ValueFilter{}, corpus.DateFilter{}, training.ConversationTransform{}, training.Parameters{},
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
			compose := loadCompose(t, file)
			for _, stage := range compose.Stages {
				raw := stage.Parameters
				resolved, err := stage.ResolvePlanningParameters()
				if err != nil {
					t.Fatalf("stage %s: %v", stage.Name, err)
				}
				if resolved.Profile != raw.Profile || resolved.BatchSize != raw.BatchSize || resolved.SequenceLength != raw.SequenceLength || resolved.LearningRate != raw.LearningRate || resolved.Seed != raw.Seed {
					t.Fatalf("stage %s direct settings were not preserved: raw=%+v resolved=%+v", stage.Name, raw, resolved)
				}
				if raw.Tokens > 0 && (resolved.RequestedTokens != raw.Tokens || resolved.PlannedTokenCapacity < raw.Tokens || resolved.PlannedTokenCapacity-raw.Tokens >= raw.BatchSize*raw.SequenceLength) {
					t.Fatalf("stage %s token budget = %d requested/%d planned", stage.Name, resolved.RequestedTokens, resolved.PlannedTokenCapacity)
				}
				assertOptionalFloat(t, stage.Name+" weight_decay", raw.WeightDecay, resolved.Optimizer.WeightDecay)
				assertOptionalInt64(t, stage.Name+" warmup_steps", raw.WarmupSteps, resolved.Schedule.WarmupSteps)
				assertOptionalInt64(t, stage.Name+" warmdown_steps", raw.WarmdownSteps, resolved.Schedule.WarmdownSteps)
				assertOptionalInt64(t, stage.Name+" checkpoint_every", raw.CheckpointEvery, resolved.CheckpointEvery)
				assertOptionalInt64(t, stage.Name+" evaluate_every", raw.EvaluateEvery, resolved.EvaluateEvery)
				if raw.ShuffleBufferRecords != nil && resolved.Data.ShuffleBufferRecords != *raw.ShuffleBufferRecords {
					t.Fatalf("stage %s shuffle_buffer_records = %d, want %d", stage.Name, resolved.Data.ShuffleBufferRecords, *raw.ShuffleBufferRecords)
				}
				assertOptionalInt64(t, stage.Name+" shuffle_buffer_bytes", raw.ShuffleBufferBytes, resolved.Data.ShuffleBufferBytes)
				if resolved.Evaluation == nil {
					t.Fatalf("stage %s has no resolved evaluation policy", stage.Name)
				}
			}
		})
	}
}

func TestFoundationLadderFilesAndForecasts(t *testing.T) {
	files, err := filepath.Glob("*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(files, foundationFiles) {
		t.Fatalf("reference composes = %v, want %v", files, foundationFiles)
	}
	want := []struct {
		parameters uint64
		tokens     int64
	}{
		{7244032, 10010624},
		{7244032, 50003968},
		{76615040, 760020992},
		{76615040, 1500053504},
		{76615040, 1500053504},
		{297171072, 3000107008},
		{297171072, 6000082944},
	}
	for index, file := range foundationFiles {
		forecast, err := model.ForecastCompose(loadCompose(t, file))
		if err != nil {
			t.Fatalf("%s: %v", file, err)
		}
		if forecast.ApproximateParameters != want[index].parameters || forecast.PlannedTokens != want[index].tokens {
			t.Fatalf("%s forecast = %d parameters/%d tokens, want %+v", file, forecast.ApproximateParameters, forecast.PlannedTokens, want[index])
		}
	}
}

func TestFoundationLadderKeepsOneControlledRecipe(t *testing.T) {
	generalCorpora := []string{
		"core/common-pile/wikimedia",
		"core/common-pile/pressbooks",
		"science/plos",
	}
	generalWeights := []uint64{5, 2, 1}
	for _, file := range foundationFiles {
		compose := loadCompose(t, file)
		if compose.Base != nil || compose.Interaction.Template != "" || len(compose.Stages) != 1 {
			t.Fatalf("%s is not a fresh, foundation-only compose", file)
		}
		architecture := compose.Architecture
		if architecture.Tokenizer.Name != "" || architecture.Tokenizer.Artifact != nil || architecture.Tokenizer.Training == nil || architecture.Tokenizer.Training.Algorithm != model.TokenizerAlgorithmByteBPEV1 || architecture.Tokenizer.Training.SampleBytes != 268435456 || architecture.Tokenizer.Training.MaxTokenInflation != 0.35 || architecture.VocabularySize != 16000 || architecture.Dropout != 0 || !architecture.QKNormalization || architecture.Initialization != "depth-scaled" || !architecture.TieEmbeddings || architecture.ParameterDType != "float32" {
			t.Fatalf("%s architecture controls = %+v", file, architecture)
		}
		stage := compose.Stages[0]
		if stage.Name != "foundation-pretrain" || stage.Type != "pre-training" || stage.Objective != "causal-language-modeling" || stage.Conversation != nil {
			t.Fatalf("%s stage contract = %+v", file, stage)
		}
		if stage.Filter == nil || stage.Filter.MainContent == nil || !*stage.Filter.MainContent || stage.Filter.Exclude == nil || stage.Filter.Exclude.RepetitiveContent == nil || !*stage.Filter.Exclude.RepetitiveContent || stage.Filter.Exclude.BoilerplateContent == nil || !*stage.Filter.Exclude.BoilerplateContent {
			t.Fatalf("%s quality filter = %+v", file, stage.Filter)
		}
		wantCorpora := generalCorpora
		wantWeights := generalWeights
		if file == foundationFiles[0] {
			wantCorpora = []string{"core/common-pile/pressbooks"}
			wantWeights = []uint64{1}
		} else if file == foundationFiles[2] || file == foundationFiles[3] {
			wantCorpora = []string{"core/common-pile/wikimedia", "core/common-pile/pressbooks"}
			wantWeights = []uint64{1, 1}
		}
		if got := corpusPaths(stage.Corpora); !reflect.DeepEqual(got, wantCorpora) {
			t.Fatalf("%s corpora = %v, want %v", file, got, wantCorpora)
		}
		for index, selection := range stage.Corpora {
			if selection.Weight == nil || *selection.Weight != wantWeights[index] {
				t.Fatalf("%s corpus %s weight = %v, want %d", file, selection.Path, selection.Weight, wantWeights[index])
			}
		}
		parameters := stage.Parameters
		if parameters.Profile != "causal-pretrain-weighted" || parameters.Parallelism != training.ParallelismAuto || parameters.ComputePrecision != "bfloat16" || parameters.Compile || parameters.Optimizer != "adamw" || parameters.Schedule != "warmup-stable-warmdown" || parameters.DistributionPolicy != "" || parameters.Seed != 42 {
			t.Fatalf("%s execution controls = %+v", file, parameters)
		}
		if index := slicesIndex(foundationFiles, file); index >= 2 {
			forecast, err := architecture.Forecast()
			if err != nil {
				t.Fatal(err)
			}
			embedding := architecture.VocabularySize * architecture.HiddenSize
			if embedding*2 > forecast.ApproximateParameters {
				t.Fatalf("%s embeds %d of %d parameters; capability rungs permit at most 50%%", file, embedding, forecast.ApproximateParameters)
			}
		}
	}
}

func TestPilotAndQualificationPairsKeepArchitecture(t *testing.T) {
	mediumPilot := loadCompose(t, foundationFiles[5])
	medium := loadCompose(t, foundationFiles[6])
	smallLanguage := loadCompose(t, foundationFiles[2])
	smallLanguageFull := loadCompose(t, foundationFiles[3])
	smallGeneral := loadCompose(t, foundationFiles[4])
	if !reflect.DeepEqual(smallLanguage.Architecture, smallLanguageFull.Architecture) {
		t.Fatal("small language diagnostic architectures differ")
	}
	if !reflect.DeepEqual(smallLanguage.Architecture, smallGeneral.Architecture) {
		t.Fatal("small pilot and qualification architectures differ")
	}
	if !reflect.DeepEqual(mediumPilot.Architecture, medium.Architecture) {
		t.Fatal("medium pilot and qualification architectures differ")
	}
	if !reflect.DeepEqual(mediumPilot.Stages[0].Corpora, medium.Stages[0].Corpora) {
		t.Fatal("medium pilot and qualification corpus recipes differ")
	}
}

func TestSmallLanguageFullChangesOnlyTrainingHorizon(t *testing.T) {
	short := loadCompose(t, foundationFiles[2])
	full := loadCompose(t, foundationFiles[3])
	if !reflect.DeepEqual(short.Architecture, full.Architecture) || !reflect.DeepEqual(short.Stages[0].Corpora, full.Stages[0].Corpora) || !reflect.DeepEqual(short.Stages[0].Filter, full.Stages[0].Filter) {
		t.Fatal("Gate 3B changed architecture or data recipe")
	}
	shortParameters := short.Stages[0].Parameters
	fullParameters := full.Stages[0].Parameters
	shortParameters.Tokens, fullParameters.Tokens = 0, 0
	shortParameters.WarmupSteps, fullParameters.WarmupSteps = nil, nil
	shortParameters.WarmdownSteps, fullParameters.WarmdownSteps = nil, nil
	shortParameters.CheckpointEvery, fullParameters.CheckpointEvery = nil, nil
	shortParameters.EvaluateEvery, fullParameters.EvaluateEvery = nil, nil
	if !reflect.DeepEqual(shortParameters, fullParameters) {
		t.Fatalf("Gate 3B changed controls beyond the training horizon: short=%+v full=%+v", shortParameters, fullParameters)
	}
}

func TestFoundationREADMEDefinesEvaluationContract(t *testing.T) {
	content, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatal(err)
	}
	text := string(content)
	for _, required := range append(foundationFiles,
		"The Linux kernel is",
		"The capital city of France is",
		"Two plus two equals",
		"Once upon a time",
		"Promotion gates",
		"seed 43",
		"conversation tuning",
	) {
		if !strings.Contains(text, required) {
			t.Errorf("README does not contain %q", required)
		}
	}
}

func TestToolUseComposeHasSizedBaseAndStructuredToolStage(t *testing.T) {
	tooling := loadCompose(t, "holding/tool-use.yaml")
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
	forecast, err := model.ForecastCompose(tooling)
	if err != nil {
		t.Fatal(err)
	}
	if forecast.ApproximateParameters != 336637440 || forecast.PlannedTokens != 20004864 {
		t.Fatalf("tool forecast = %d parameters/%d tokens", forecast.ApproximateParameters, forecast.PlannedTokens)
	}
}

func loadCompose(t *testing.T, path string) model.Compose {
	t.Helper()
	compose, _, err := model.LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	return compose
}

func slicesIndex(values []string, value string) int {
	for index, candidate := range values {
		if candidate == value {
			return index
		}
	}
	return -1
}

func corpusPaths(selections []model.CorpusSelection) []string {
	paths := make([]string, len(selections))
	for index, selection := range selections {
		paths[index] = selection.Path
	}
	return paths
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
