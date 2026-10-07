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
	"archive/2026-10-byte-bpe-ladder-retired/0001-foundation-pipeline-canary.yaml",
	"archive/2026-10-byte-bpe-ladder-retired/0002-foundation-mixture-canary.yaml",
	"archive/2026-10-byte-bpe-ladder-retired/0003-foundation-small-language.yaml",
	"archive/2026-10-byte-bpe-ladder-retired/0003b-foundation-small-language-full.yaml",
	"archive/2026-10-byte-bpe-ladder-retired/0004-foundation-small-general.yaml",
	"archive/2026-10-byte-bpe-ladder-retired/0005-foundation-medium-pilot.yaml",
	"archive/2026-10-byte-bpe-ladder-retired/0006-foundation-medium.yaml",
}

var activeLadderFiles = []string{
	"0001-tiny-shakespeare.yaml",
	"0002-tinystories-byte.yaml",
	"0003-tinystories-context-512.yaml",
	"0004-tinystories-capacity-pilot.yaml",
	"0005-tinystories-capacity-20tpp.yaml",
}

var tinyStoriesFiles = []string{
	"tinystories/0001-tinystories-canary.yaml",
	"tinystories/0002-tinystories-8m.yaml",
	"tinystories/0003-tinystories-32m.yaml",
}

var generalFoundationFiles = []string{
	"general-foundation/0001-general-mixture-32m.yaml",
	"general-foundation/0002-general-foundation-125m-pilot.yaml",
	"general-foundation/0003-general-foundation-125m.yaml",
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

func TestRetiredFoundationLadderFilesAndForecasts(t *testing.T) {
	files, err := filepath.Glob("archive/2026-10-byte-bpe-ladder-retired/*.yaml")
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

func TestActiveReferenceLadderForecastsAndControls(t *testing.T) {
	files, err := filepath.Glob("*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(files, activeLadderFiles) {
		t.Fatalf("active ladder composes = %v, want %v", files, activeLadderFiles)
	}
	wantParameters := []uint64{10721280, 10721280, 10721280, 30813184, 30813184}
	wantTokens := []int64{81920000, 245760000, 245760000, 245760000, 613613568}
	var reference model.Compose
	for index, file := range files {
		compose := loadCompose(t, file)
		forecast, err := model.ForecastCompose(compose)
		if err != nil {
			t.Fatalf("%s: %v", file, err)
		}
		if forecast.ApproximateParameters != wantParameters[index] || forecast.PlannedTokens != wantTokens[index] {
			t.Fatalf("%s forecast = %d parameters/%d tokens", file, forecast.ApproximateParameters, forecast.PlannedTokens)
		}
		if index == 0 {
			reference = compose
			continue
		}
		wantArchitecture := reference.Architecture
		if index >= 2 {
			wantArchitecture.ContextTokens = 512
		}
		if index >= 3 {
			wantArchitecture.HiddenSize = 512
			wantArchitecture.IntermediateSize = 1536
			wantArchitecture.Layers = 9
			wantArchitecture.AttentionHeads = 8
			wantArchitecture.KeyValueHeads = 8
		}
		if !reflect.DeepEqual(wantArchitecture, compose.Architecture) {
			t.Fatalf("%s changes more than the intended context control", file)
		}
	}
	shakespeare := reference.Stages[0]
	stories := loadCompose(t, activeLadderFiles[1]).Stages[0]
	storiesContext512 := loadCompose(t, activeLadderFiles[2]).Stages[0]
	storiesCapacity := loadCompose(t, activeLadderFiles[3]).Stages[0]
	storiesCapacity20TPP := loadCompose(t, activeLadderFiles[4]).Stages[0]
	if !reflect.DeepEqual(corpusPaths(shakespeare.Corpora), []string{"core/reference/tiny-shakespeare"}) || !reflect.DeepEqual(corpusPaths(stories.Corpora), []string{"core/synthetic/tinystories-reference"}) || !reflect.DeepEqual(corpusPaths(storiesContext512.Corpora), []string{"core/synthetic/tinystories-reference"}) || !reflect.DeepEqual(corpusPaths(storiesCapacity.Corpora), []string{"core/synthetic/tinystories-reference"}) || !reflect.DeepEqual(corpusPaths(storiesCapacity20TPP.Corpora), []string{"core/synthetic/tinystories-reference"}) {
		t.Fatalf("active ladder corpora = %v / %v / %v / %v / %v", corpusPaths(shakespeare.Corpora), corpusPaths(stories.Corpora), corpusPaths(storiesContext512.Corpora), corpusPaths(storiesCapacity.Corpora), corpusPaths(storiesCapacity20TPP.Corpora))
	}
	for _, stage := range []model.Stage{shakespeare, stories} {
		parameters := stage.Parameters
		if parameters.Profile != "causal-pretrain-shuffled" || parameters.BatchSize != 64 || parameters.GradientAccumulation != 8 || parameters.SequenceLength != 256 || parameters.LearningRate != 0.001 || parameters.Optimizer != "adamw" || parameters.Schedule != "cosine" || parameters.Seed != 42 {
			t.Fatalf("active ladder recipe changed = %+v", parameters)
		}
	}
	contextParameters := storiesContext512.Parameters
	wantContextParameters := stories.Parameters
	wantContextParameters.BatchSize = 32
	wantContextParameters.SequenceLength = 512
	if !reflect.DeepEqual(contextParameters, wantContextParameters) {
		t.Fatalf("context rung recipe changed beyond sequence/batch control: got=%+v want=%+v", contextParameters, wantContextParameters)
	}
	if stories.Parameters.BatchSize*stories.Parameters.SequenceLength != contextParameters.BatchSize*contextParameters.SequenceLength {
		t.Fatalf("context rung tokens/update = %d, want %d", contextParameters.BatchSize*contextParameters.SequenceLength, stories.Parameters.BatchSize*stories.Parameters.SequenceLength)
	}
	if stories.Parameters.Tokens != contextParameters.Tokens {
		t.Fatalf("context rung tokens = %d, want %d", contextParameters.Tokens, stories.Parameters.Tokens)
	}
	capacityParameters := storiesCapacity.Parameters
	wantCapacityParameters := contextParameters
	wantCapacityParameters.LearningRate = 0.0006
	if !reflect.DeepEqual(capacityParameters, wantCapacityParameters) {
		t.Fatalf("capacity pilot recipe changed beyond scale-adjusted learning rate: got=%+v want=%+v", capacityParameters, wantCapacityParameters)
	}
	capacity20TPPParameters := storiesCapacity20TPP.Parameters
	wantCapacity20TPPParameters := capacityParameters
	wantCapacity20TPPParameters.Tokens = 613611520
	if !reflect.DeepEqual(capacity20TPPParameters, wantCapacity20TPPParameters) {
		t.Fatalf("capacity qualification changed controls beyond token horizon: got=%+v want=%+v", capacity20TPPParameters, wantCapacity20TPPParameters)
	}
	if shakespeare.Parameters.EvaluationSelection != "contiguous-tail-v1" || stories.Parameters.EvaluationSelection != "lowest-sha256-v1" || storiesContext512.Parameters.EvaluationSelection != "lowest-sha256-v1" || storiesCapacity.Parameters.EvaluationSelection != "lowest-sha256-v1" || storiesCapacity20TPP.Parameters.EvaluationSelection != "lowest-sha256-v1" {
		t.Fatalf("active ladder evaluation policies = %q / %q / %q / %q / %q", shakespeare.Parameters.EvaluationSelection, stories.Parameters.EvaluationSelection, storiesContext512.Parameters.EvaluationSelection, storiesCapacity.Parameters.EvaluationSelection, storiesCapacity20TPP.Parameters.EvaluationSelection)
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

func TestRootREADMEDefinesActiveReferenceLadder(t *testing.T) {
	content, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatal(err)
	}
	text := string(content)
	for _, required := range []string{
		"Reference-model training ladder",
		"Rung 0001: Tiny Shakespeare reference — passed",
		"b4f8477a55ad",
		"step 1,250",
		"under 15 minutes",
		"Rung 0002: TinyStories byte control — diagnostic complete",
		"EOS is measured at a horizon calibrated to the corpus",
		"Rung 0003: TinyStories 512-byte context — passed",
		"Rung 0004: TinyStories capacity pilot — passed",
		"Rung 0005: TinyStories capacity qualification — behavioral near-miss",
		"Rung 0006 must be a compose-native compact byte-BPE source-exposure control",
	} {
		if !strings.Contains(text, required) {
			t.Errorf("README does not contain %q", required)
		}
	}
}

func TestGeneralFoundationLadderForecastsAndControls(t *testing.T) {
	want := []struct {
		parameters uint64
		tokens     int64
	}{
		{32261632, 1000013824},
		{125562624, 2500067328},
		{125562624, 5000003584},
	}
	wantCorpora := []string{
		"core/common-pile/wikimedia",
		"core/common-pile/stackexchange",
		"science/plos",
		"core/common-pile/pressbooks",
	}
	wantWeights := []uint64{11, 5, 3, 1}
	wantTokenizerCorpora := []string{
		"core/common-pile/wikimedia",
		"core/common-pile/pressbooks",
		"science/plos",
	}
	var pilot model.Compose
	for index, file := range generalFoundationFiles {
		compose := loadCompose(t, file)
		forecast, err := model.ForecastCompose(compose)
		if err != nil {
			t.Fatalf("%s: %v", file, err)
		}
		if forecast.ApproximateParameters != want[index].parameters || forecast.PlannedTokens != want[index].tokens {
			t.Fatalf("%s forecast = %d parameters/%d tokens, want %+v", file, forecast.ApproximateParameters, forecast.PlannedTokens, want[index])
		}
		if compose.Base != nil || compose.Interaction.Template != "" || len(compose.Stages) != 1 {
			t.Fatalf("%s is not a fresh one-stage foundation compose", file)
		}
		stage := compose.Stages[0]
		if stage.Name != "general-pretrain" || stage.Type != "pre-training" || stage.Objective != "causal-language-modeling" || !reflect.DeepEqual(corpusPaths(stage.Corpora), wantCorpora) {
			t.Fatalf("%s stage = %+v", file, stage)
		}
		for position, selection := range stage.Corpora {
			if selection.Weight == nil || *selection.Weight != wantWeights[position] {
				t.Fatalf("%s corpus %s weight = %v, want %d", file, selection.Path, selection.Weight, wantWeights[position])
			}
		}
		if stage.Filter == nil || stage.Filter.MainContent == nil || !*stage.Filter.MainContent || stage.Filter.Exclude == nil || stage.Filter.Exclude.RepetitiveContent == nil || !*stage.Filter.Exclude.RepetitiveContent || stage.Filter.Exclude.BoilerplateContent == nil || !*stage.Filter.Exclude.BoilerplateContent {
			t.Fatalf("%s quality filter = %+v", file, stage.Filter)
		}
		parameters := stage.Parameters
		if parameters.Profile != "causal-pretrain-weighted" || parameters.Parallelism != training.ParallelismAuto || parameters.DistributionPolicy != "" || parameters.ComputePrecision != "bfloat16" || parameters.Compile || parameters.Optimizer != "adamw" || parameters.Schedule != "warmup-stable-warmdown" || parameters.Seed != 42 {
			t.Fatalf("%s parameters = %+v", file, parameters)
		}
		tokenizer := compose.Architecture.Tokenizer.Training
		if tokenizer == nil || tokenizer.Algorithm != model.TokenizerAlgorithmByteBPEV1 || tokenizer.SampleBytes != 268435456 || tokenizer.Seed != 42 || tokenizer.DistributionPolicy != corpus.DistributionPolicyDistributable {
			t.Fatalf("%s tokenizer = %+v", file, tokenizer)
		}
		if index == 0 {
			if compose.Architecture.VocabularySize != 10000 || !reflect.DeepEqual(corpusPaths(tokenizer.Corpora), []string{"core/common-pile/pressbooks"}) {
				t.Fatalf("%s controlled-ablation tokenizer changed: %+v", file, compose.Architecture)
			}
		} else {
			if compose.Architecture.VocabularySize != 16000 || !reflect.DeepEqual(corpusPaths(tokenizer.Corpora), wantTokenizerCorpora) {
				t.Fatalf("%s 125M tokenizer = %+v", file, tokenizer)
			}
			if index == 1 {
				pilot = compose
			}
		}
	}
	qualified := loadCompose(t, generalFoundationFiles[2])
	if !reflect.DeepEqual(pilot.Architecture, qualified.Architecture) || !reflect.DeepEqual(pilot.Stages[0].Corpora, qualified.Stages[0].Corpora) || !reflect.DeepEqual(pilot.Stages[0].Filter, qualified.Stages[0].Filter) {
		t.Fatal("125M pilot and qualification changed architecture or data recipe")
	}
}

func TestGeneralMixtureGateChangesOnlyPretrainingCorpora(t *testing.T) {
	cosmopedia := loadCompose(t, "tinystories/0003-tinystories-32m.yaml")
	general := loadCompose(t, generalFoundationFiles[0])
	if !reflect.DeepEqual(cosmopedia.Architecture, general.Architecture) {
		t.Fatal("general mixture ablation changed the controlled 32M architecture")
	}
	left, right := cosmopedia.Stages[0], general.Stages[0]
	left.Name, right.Name = "", ""
	left.Corpora, right.Corpora = nil, nil
	if !reflect.DeepEqual(left, right) {
		t.Fatalf("general mixture ablation changed controls beyond corpora: cosmopedia=%+v general=%+v", left, right)
	}
}

func TestGeneralFoundationREADMEDefinesDataAndPostTrainingPlan(t *testing.T) {
	content, err := os.ReadFile("general-foundation/README.md")
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{
		"0001-general-mixture-32m.yaml",
		"0002-general-foundation-125m-pilot.yaml",
		"0003-general-foundation-125m.yaml",
		"55%",
		"controlled data ablation",
		"near deduplication",
		"evaluation prompt and benchmark",
		"Do not paraphrase",
		"assistant tokens",
		"interaction-contract-v1",
		"Promotion gates",
		"Questions recorded at every gate",
		"seed 43",
	} {
		if !strings.Contains(string(content), required) {
			t.Errorf("general-foundation README does not contain %q", required)
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

func TestAssistantEOSExperimentPinsBaseAndSupervisesAssistant(t *testing.T) {
	compose := loadCompose(t, "experiments/0001-assistant-eos-canary.yaml")
	if compose.Base == nil || compose.Base.Model != "foundation-small-language-full-bpe-01" || compose.Base.ModelID != "3587a82e93488cf709a14c5b9f0dfcca18839f12de2ba336121e5db25cc8aa07" || compose.Base.RunID != "11c4dd9bde7821c3" {
		t.Fatalf("experiment base = %+v", compose.Base)
	}
	if compose.Architecture != (model.Architecture{}) {
		t.Fatalf("experiment architecture should be inherited: %+v", compose.Architecture)
	}
	if compose.Interaction.Template != model.InteractionUserAssistantV1 || len(compose.Stages) != 1 {
		t.Fatalf("experiment contract = %+v / %d stages", compose.Interaction, len(compose.Stages))
	}
	stage := compose.Stages[0]
	if stage.Objective != "assistant-response-modeling" || stage.Conversation == nil || !reflect.DeepEqual(stage.Conversation.SupervisedRoles, []string{"assistant"}) || stage.Parameters.Tokens != 10000000 {
		t.Fatalf("experiment stage = %+v", stage)
	}
}

func TestBroadAssistantEOSExperimentChangesOnlyCorpora(t *testing.T) {
	contract := loadCompose(t, "experiments/0001-assistant-eos-canary.yaml")
	broad := loadCompose(t, "experiments/0002-assistant-eos-broad-canary.yaml")
	if !reflect.DeepEqual(contract.Base, broad.Base) || !reflect.DeepEqual(contract.Interaction, broad.Interaction) || len(contract.Stages) != 1 || len(broad.Stages) != 1 {
		t.Fatalf("broad experiment changed base or interaction contract")
	}
	contractStage, broadStage := contract.Stages[0], broad.Stages[0]
	contractStage.Corpora, broadStage.Corpora = nil, nil
	if !reflect.DeepEqual(contractStage, broadStage) {
		t.Fatalf("broad experiment changed more than corpora: contract=%+v broad=%+v", contractStage, broadStage)
	}
	want := []string{"post-train/sft/oasst2", "post-train/sft/helpsteer2", "post-train/sft/dolly"}
	if got := corpusPaths(broad.Stages[0].Corpora); !reflect.DeepEqual(got, want) {
		t.Fatalf("broad experiment corpora = %v, want %v", got, want)
	}
}

func TestTinyStoriesLadderForecastsAndControls(t *testing.T) {
	want := []struct {
		parameters uint64
		tokens     int64
	}{
		{8593664, 10027008},
		{8593664, 500006912},
		{32261632, 1000013824},
	}
	var first model.Compose
	for index, file := range tinyStoriesFiles {
		compose := loadCompose(t, file)
		forecast, err := model.ForecastCompose(compose)
		if err != nil {
			t.Fatalf("%s: %v", file, err)
		}
		if forecast.ApproximateParameters != want[index].parameters || forecast.PlannedTokens != want[index].tokens {
			t.Fatalf("%s forecast = %d parameters/%d tokens, want %+v", file, forecast.ApproximateParameters, forecast.PlannedTokens, want[index])
		}
		architecture := compose.Architecture
		if architecture.ContextTokens != 512 || architecture.VocabularySize != 10000 || architecture.Dropout != 0 || !architecture.QKNormalization || architecture.Initialization != "depth-scaled" || !architecture.TieEmbeddings || architecture.ParameterDType != "float32" || architecture.Tokenizer.Training == nil {
			t.Fatalf("%s architecture controls = %+v", file, architecture)
		}
		tokenizer := architecture.Tokenizer.Training
		if tokenizer.Algorithm != model.TokenizerAlgorithmByteBPEV1 || tokenizer.SampleBytes != 268435456 || tokenizer.Seed != 42 || tokenizer.DistributionPolicy != corpus.DistributionPolicyDistributable || !reflect.DeepEqual(corpusPaths(tokenizer.Corpora), []string{"core/common-pile/pressbooks"}) {
			t.Fatalf("%s tokenizer controls = %+v", file, tokenizer)
		}
		if tokenizer.Filter == nil || tokenizer.Filter.MainContent == nil || !*tokenizer.Filter.MainContent || tokenizer.Filter.Exclude == nil || tokenizer.Filter.Exclude.RepetitiveContent == nil || !*tokenizer.Filter.Exclude.RepetitiveContent || tokenizer.Filter.Exclude.BoilerplateContent == nil || !*tokenizer.Filter.Exclude.BoilerplateContent {
			t.Fatalf("%s tokenizer filter = %+v", file, tokenizer.Filter)
		}
		if len(compose.Stages) != 1 {
			t.Fatalf("%s stages = %d", file, len(compose.Stages))
		}
		stage := compose.Stages[0]
		if stage.Name != "tinystories-pretrain" || stage.Type != "pre-training" || stage.Objective != "causal-language-modeling" || !reflect.DeepEqual(corpusPaths(stage.Corpora), []string{"core/synthetic/cosmopedia-v2"}) || stage.Corpora[0].Weight == nil || *stage.Corpora[0].Weight != 1 {
			t.Fatalf("%s stage = %+v", file, stage)
		}
		if stage.Filter == nil || stage.Filter.MainContent == nil || !*stage.Filter.MainContent || stage.Filter.Exclude == nil || stage.Filter.Exclude.RepetitiveContent == nil || !*stage.Filter.Exclude.RepetitiveContent || stage.Filter.Exclude.BoilerplateContent == nil || !*stage.Filter.Exclude.BoilerplateContent {
			t.Fatalf("%s quality filter = %+v", file, stage.Filter)
		}
		parameters := stage.Parameters
		if parameters.Profile != "causal-pretrain-weighted" || parameters.DistributionPolicy != "" || parameters.BatchSize != 64 || parameters.SequenceLength != 512 || parameters.LearningRate != 0.0005 || parameters.Optimizer != "adamw" || parameters.Schedule != "warmup-stable-warmdown" || parameters.Seed != 42 {
			t.Fatalf("%s parameters = %+v", file, parameters)
		}
		if index == 0 {
			first = compose
		} else if index == 1 && !reflect.DeepEqual(first.Architecture, compose.Architecture) {
			t.Fatal("TinyStories canary and 8M qualification architectures differ")
		}
	}
}

func TestTinyStoriesREADMEDefinesExperimentContract(t *testing.T) {
	content, err := os.ReadFile("tinystories/README.md")
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{
		"0001-tinystories-canary.yaml",
		"0002-tinystories-8m.yaml",
		"0003-tinystories-32m.yaml",
		"core/synthetic/cosmopedia-v2",
		"core/common-pile/pressbooks",
		"No download or index ingestion step is required",
		"not a TinyStories dataset reproduction",
		"Promotion gates",
		"Did it learn EOS without post-training?",
		"evaluate-tinystories.sh",
		"cosmopedia-32m-01",
		"2.1102",
		"Do not scale this recipe further",
	} {
		if !strings.Contains(string(content), required) {
			t.Errorf("TinyStories README does not contain %q", required)
		}
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
