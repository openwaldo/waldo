// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/training"
)

func TestSkipCompletedCorporaFiltersCompletedPathsAndWeights(t *testing.T) {
	compose := validCompose()
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"core/books", "science/new", "post-train/dialogue"})
	compose.Stages[0].Parameters.CorpusWeights = map[string]uint64{"core/books": 3, "science/new": 2, "post-train/dialogue": 1}
	compose.Stages = append(compose.Stages, Stage{
		Name: "fine-tune", Type: "fine-tuning", Objective: "causal-language-modeling",
		Corpora: NewCorpusSelections([]string{"post-train/dialogue"}), Parameters: testStage("unused").Parameters,
	})
	inspection := Inspection{
		Model: ModelRecord{Runs: []RunPin{
			{State: RunComplete},
			{State: RunFailed},
		}},
		RunBOMs: []RunBOM{
			{CorpusBOM: corpus.BOM{Paths: []string{"core/books.yaml", "post-train/dialogue"}}},
			{CorpusBOM: corpus.BOM{Paths: []string{"science/new"}}},
		},
	}

	filtered, skipped := SkipCompletedCorpora(compose, inspection)
	if !reflect.DeepEqual(skipped, []SkippedCorpus{{Stage: "pretrain", Path: "core/books"}, {Stage: "pretrain", Path: "post-train/dialogue"}, {Stage: "fine-tune", Path: "post-train/dialogue"}}) {
		t.Fatalf("skipped = %+v", skipped)
	}
	if len(filtered.Stages) != 1 || !reflect.DeepEqual(CorpusPaths(filtered.Stages[0].Corpora), []string{"science/new"}) || !reflect.DeepEqual(filtered.Stages[0].Parameters.CorpusWeights, map[string]uint64{"science/new": 2}) {
		t.Fatalf("filtered compose = %+v", filtered)
	}
	if len(compose.Stages) != 2 || len(compose.Stages[0].Corpora) != 3 || len(compose.Stages[0].Parameters.CorpusWeights) != 3 {
		t.Fatal("input compose was mutated")
	}
}

func TestComposeCompletedRequiresExactOrderedStageEvidence(t *testing.T) {
	first := testStage("pretrain")
	second := testStage("post-train")
	second.Corpora = NewCorpusSelections([]string{"post-train/dialogue"})
	compose := validCompose()
	compose.Stages = []Stage{first, second}
	inspection := Inspection{
		Runs: []RunRecord{{State: RunComplete}, {State: RunComplete}},
		RunBOMs: []RunBOM{
			completedStageBOM(t, first, []string{"example.yaml"}),
			completedStageBOM(t, second, []string{"post-train/dialogue"}),
		},
	}
	completed, err := ComposeCompleted(compose, inspection)
	if err != nil || !completed {
		t.Fatalf("exact compose completion = %t, %v", completed, err)
	}

	changedBudget := compose
	changedBudget.Stages = append([]Stage(nil), compose.Stages...)
	changedBudget.Stages[0].Parameters.Steps++
	if completed, err := ComposeCompleted(changedBudget, inspection); err != nil || completed {
		t.Fatalf("changed budget completion = %t, %v", completed, err)
	}

	reordered := compose
	reordered.Stages = []Stage{second, first}
	if completed, err := ComposeCompleted(reordered, inspection); err != nil || completed {
		t.Fatalf("reordered completion = %t, %v", completed, err)
	}

	changedFilter := compose
	changedFilter.Stages = append([]Stage(nil), compose.Stages...)
	changedFilter.Stages[0].Filter = &corpus.RecordFilter{MainContent: boolPointer(true)}
	if completed, err := ComposeCompleted(changedFilter, inspection); err != nil || completed {
		t.Fatalf("changed filter completion = %t, %v", completed, err)
	}

	newer := testStage("newer")
	newer.Corpora = NewCorpusSelections([]string{"science/newer"})
	withNewerRun := inspection
	withNewerRun.Runs = append(append([]RunRecord(nil), inspection.Runs...), RunRecord{State: RunComplete})
	withNewerRun.RunBOMs = append(append([]RunBOM(nil), inspection.RunBOMs...), completedStageBOM(t, newer, []string{"science/newer"}))
	if completed, err := ComposeCompleted(compose, withNewerRun); err != nil || completed {
		t.Fatalf("obsolete completed lineage = %t, %v", completed, err)
	}

	withFailedAttempt := inspection
	withFailedAttempt.Runs = []RunRecord{{State: RunComplete}, {State: RunFailed}, {State: RunComplete}}
	withFailedAttempt.RunBOMs = []RunBOM{
		inspection.RunBOMs[0],
		completedStageBOM(t, second, []string{"post-train/dialogue"}),
		inspection.RunBOMs[1],
	}
	if completed, err := ComposeCompleted(compose, withFailedAttempt); err != nil || !completed {
		t.Fatalf("completed lineage with failed attempt = %t, %v", completed, err)
	}
}

func completedStageBOM(t *testing.T, stage Stage, paths []string) RunBOM {
	t.Helper()
	parameters, err := stage.ResolveParameters()
	if err != nil {
		t.Fatal(err)
	}
	if parameters.Data.Order == "corpus-weighted-shuffle-v1" {
		parameters.Data.CorpusWeights, err = resolveCorpusWeights(parameters.Data.CorpusWeights, paths)
		if err != nil {
			t.Fatal(err)
		}
	}
	filter, err := stage.RecordFilterPolicy(paths)
	if err != nil {
		t.Fatal(err)
	}
	conversation := training.ConversationTransform{}
	if stage.Conversation != nil {
		conversation = *stage.Conversation
	}
	return RunBOM{
		Stage: stage.Name, StageType: stage.Type, Objective: stage.Objective,
		Conversation: conversation, Parameters: parameters,
		CorpusBOM: corpus.BOM{Paths: paths, RecordFilter: filter},
	}
}

func boolPointer(value bool) *bool { return &value }

func TestPersistCompletedComposeRestoresCanonicalCompose(t *testing.T) {
	root := t.TempDir()
	compose := validCompose()
	if err := os.WriteFile(filepath.Join(root, "COMPOSE.json"), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := PersistCompletedCompose(root, compose, "0003-conversation.yaml"); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := LoadCompose(filepath.Join(root, "COMPOSE.json"))
	if err != nil || !reflect.DeepEqual(loaded, compose) {
		t.Fatalf("canonical compose = %+v, %v", loaded, err)
	}
	entries, err := os.ReadDir(filepath.Join(root, ComposeHistoryDirectory))
	if err != nil || len(entries) != 1 || entries[0].Name() != "0000-conversation.yaml" {
		t.Fatalf("compose history = %+v, %v", entries, err)
	}
}

func TestSkipCompletedCorporaCanProduceNoWork(t *testing.T) {
	compose := validCompose()
	inspection := Inspection{
		Model:   ModelRecord{Runs: []RunPin{{State: RunComplete}}},
		RunBOMs: []RunBOM{{CorpusBOM: corpus.BOM{Paths: []string{"example.json"}}}},
	}
	filtered, skipped := SkipCompletedCorpora(compose, inspection)
	if len(filtered.Stages) != 0 || len(skipped) != 1 || skipped[0].Path != "example" {
		t.Fatalf("filtered = %+v, skipped = %+v", filtered, skipped)
	}
}
