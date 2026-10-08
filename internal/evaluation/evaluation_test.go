// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package evaluation

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/training"
)

type records []training.Record

func (values records) Stream(ctx context.Context, consume func(training.Record) error) error {
	for _, value := range values {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := consume(value); err != nil {
			return err
		}
	}
	return nil
}

func TestLoadDefinitionRejectsUnknownAndIncompleteFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "evaluation.yaml")
	valid := "kind: waldo-evaluation-definition\nschema: 1\nname: core\ntask: multiple-choice\nsplit: test\ncorpora: [evaluation/core]\nmetrics:\n  - {name: accuracy, direction: max, threshold: 0.5}\ncontamination: {max_exact_records: 0, max_fuzzy_records: 0, fuzzy_ratio: 0.8, shingle_words: 13}\n"
	if err := os.WriteFile(path, []byte(valid), 0o600); err != nil {
		t.Fatal(err)
	}
	if definition, err := LoadDefinition(path); err != nil || definition.Name != "core" {
		t.Fatalf("definition = %+v, err=%v", definition, err)
	}
	if err := os.WriteFile(path, []byte(valid+"surprise: true\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadDefinition(path); err == nil {
		t.Fatal("unknown evaluation field accepted")
	}
}

func TestContaminationAndPromotionAreFailClosed(t *testing.T) {
	limit := OverlapLimit{ExactRecords: 0, FuzzyRecords: 0, FuzzyRatio: 0.5, ShingleWords: 3}
	trainingRecords := records{{ID: "train-exact", Text: "The quick brown fox jumps over the dog"}, {ID: "train-fuzzy", Text: "alpha beta gamma delta epsilon zeta"}}
	evaluationRecords := records{{ID: "eval-exact", Text: " the QUICK brown fox jumps over the dog "}, {ID: "eval-fuzzy", Text: "alpha beta gamma delta epsilon other"}}
	report, err := CheckContamination(context.Background(), strings.Repeat("a", 64), strings.Repeat("b", 64), trainingRecords, evaluationRecords, limit)
	if err != nil {
		t.Fatal(err)
	}
	if report.Passed || len(report.Exact) != 1 || len(report.Fuzzy) != 1 || len(report.SHA256) != 64 {
		t.Fatalf("contamination report = %+v", report)
	}
	bom := evaluationBOM(t, limit)
	promotion := DecidePromotion(bom, report, []Result{{Metric: "accuracy", Value: 0.9}})
	if promotion.Passed || len(promotion.Failures) != 1 || promotion.Failures[0] != "contamination policy failed" {
		t.Fatalf("promotion = %+v", promotion)
	}
}

func TestPromotionRequiresEveryThreshold(t *testing.T) {
	bom := evaluationBOM(t, OverlapLimit{ExactRecords: 0, FuzzyRecords: 0, FuzzyRatio: 0.8, ShingleWords: 3})
	report := ContaminationReport{Passed: true}
	if got := DecidePromotion(bom, report, []Result{{Metric: "accuracy", Value: 0.79}}); got.Passed || len(got.Failures) != 1 {
		t.Fatalf("low metric promotion = %+v", got)
	}
	if got := DecidePromotion(bom, report, []Result{{Metric: "accuracy", Value: 0.8}}); !got.Passed {
		t.Fatalf("passing promotion = %+v", got)
	}
	if got := DecidePromotion(bom, report, nil); got.Passed || got.Failures[0] != "missing metric accuracy" {
		t.Fatalf("missing metric promotion = %+v", got)
	}
}

func TestGatePinsResultsAndEveryContaminationReport(t *testing.T) {
	bom := evaluationBOM(t, OverlapLimit{ExactRecords: 0, FuzzyRecords: 0, FuzzyRatio: 0.8, ShingleWords: 3})
	bomSHA256, err := bom.SHA256()
	if err != nil {
		t.Fatal(err)
	}
	report, err := CheckContamination(context.Background(), strings.Repeat("a", 64), bomSHA256, records{{ID: "train", Text: "training record only"}}, records{{ID: "eval", Text: "evaluation record only"}}, bom.Contamination)
	if err != nil {
		t.Fatal(err)
	}
	results := ResultsDocument{Kind: "openwaldo-evaluation-results", Schema: 1, EvaluationBOMSHA256: bomSHA256, Results: []Result{{Metric: "accuracy", Value: 0.9}}}
	gate, err := BuildGateReport("model-id", bom, []ContaminationReport{report}, results)
	if err != nil || !gate.Promotion.Passed || len(gate.SHA256) != 64 {
		t.Fatalf("gate = %+v, err=%v", gate, err)
	}
	results.EvaluationBOMSHA256 = strings.Repeat("c", 64)
	if _, err := BuildGateReport("model-id", bom, []ContaminationReport{report}, results); err == nil {
		t.Fatal("mismatched evaluation results accepted")
	}
}

func evaluationBOM(t *testing.T, limit OverlapLimit) BOM {
	t.Helper()
	base := corpus.BOM{
		Kind: "openwaldo-bom", Schema: 1, Subject: "corpus",
		Index: index.Identity{Commit: strings.Repeat("c", 40)},
		Paths: []string{"evaluation/test"}, Manifests: []corpus.ManifestPin{}, Shards: []corpus.ShardPin{}, Licenses: map[string]index.Measures{},
	}
	// An empty corpus BOM is structurally useful for this isolated gate test.
	// Populate the totals and lists exactly as corpus validation expects.
	bom, err := NewBOM("core", "multiple-choice", "test", base, []Metric{{Name: "accuracy", Direction: "max", Threshold: 0.8}}, limit)
	if err != nil {
		t.Fatal(err)
	}
	return bom
}
