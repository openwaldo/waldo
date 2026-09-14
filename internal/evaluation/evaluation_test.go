// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package evaluation

import (
	"context"
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
