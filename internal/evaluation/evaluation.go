// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package evaluation owns immutable evaluation definitions, corpus BOM pins,
// contamination evidence, metric results, and deterministic promotion gates.
package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/training"
)

const (
	BOMKind   = "openwaldo-evaluation-bom"
	BOMSchema = 1
)

type Metric struct {
	Name      string  `json:"name" yaml:"name"`
	Direction string  `json:"direction" yaml:"direction"`
	Threshold float64 `json:"threshold" yaml:"threshold"`
}

type BOM struct {
	Kind             string       `json:"kind"`
	Schema           int          `json:"schema"`
	Name             string       `json:"name"`
	Task             string       `json:"task"`
	Split            string       `json:"split"`
	CorpusBOMSHA256  string       `json:"corpus_bom_sha256"`
	CorpusBOM        corpus.BOM   `json:"corpus_bom"`
	Metrics          []Metric     `json:"metrics"`
	Contamination    OverlapLimit `json:"contamination"`
	DefinitionSHA256 string       `json:"definition_sha256"`
}

type OverlapLimit struct {
	ExactRecords int     `json:"max_exact_records" yaml:"max_exact_records"`
	FuzzyRecords int     `json:"max_fuzzy_records" yaml:"max_fuzzy_records"`
	FuzzyRatio   float64 `json:"fuzzy_ratio" yaml:"fuzzy_ratio"`
	ShingleWords int     `json:"shingle_words" yaml:"shingle_words"`
}

type RecordOverlap struct {
	TrainingID   string  `json:"training_id"`
	EvaluationID string  `json:"evaluation_id"`
	Ratio        float64 `json:"ratio"`
}

type ContaminationReport struct {
	Kind                string          `json:"kind"`
	Schema              int             `json:"schema"`
	TrainingBOMSHA256   string          `json:"training_bom_sha256"`
	EvaluationBOMSHA256 string          `json:"evaluation_bom_sha256"`
	Exact               []RecordOverlap `json:"exact"`
	Fuzzy               []RecordOverlap `json:"fuzzy"`
	Passed              bool            `json:"passed"`
	SHA256              string          `json:"sha256"`
}

type Result struct {
	Metric string  `json:"metric"`
	Value  float64 `json:"value"`
}

type Promotion struct {
	Passed   bool     `json:"passed"`
	Failures []string `json:"failures,omitempty"`
}

func NewBOM(name, task, split string, source corpus.BOM, metrics []Metric, limit OverlapLimit) (BOM, error) {
	corpusHash, err := digest(source)
	if err != nil {
		return BOM{}, err
	}
	bom := BOM{Kind: BOMKind, Schema: BOMSchema, Name: name, Task: task, Split: split, CorpusBOMSHA256: corpusHash, CorpusBOM: source, Metrics: append([]Metric(nil), metrics...), Contamination: limit}
	definition := struct {
		Name          string       `json:"name"`
		Task          string       `json:"task"`
		Split         string       `json:"split"`
		Metrics       []Metric     `json:"metrics"`
		Contamination OverlapLimit `json:"contamination"`
	}{name, task, split, bom.Metrics, limit}
	bom.DefinitionSHA256, err = digest(definition)
	if err != nil {
		return BOM{}, err
	}
	if err := bom.Validate(); err != nil {
		return BOM{}, err
	}
	return bom, nil
}

func (bom BOM) Validate() error {
	if bom.Kind != BOMKind || bom.Schema != BOMSchema {
		return fmt.Errorf("unsupported evaluation BOM %q schema %d", bom.Kind, bom.Schema)
	}
	if strings.TrimSpace(bom.Name) == "" || strings.TrimSpace(bom.Task) == "" || strings.TrimSpace(bom.Split) == "" {
		return fmt.Errorf("evaluation name, task, and split are required")
	}
	if err := bom.CorpusBOM.Validate(); err != nil {
		return fmt.Errorf("evaluation corpus BOM: %w", err)
	}
	corpusHash, err := digest(bom.CorpusBOM)
	if err != nil || corpusHash != bom.CorpusBOMSHA256 {
		return fmt.Errorf("evaluation corpus BOM digest differs")
	}
	if len(bom.Metrics) == 0 {
		return fmt.Errorf("at least one evaluation metric is required")
	}
	seen := map[string]bool{}
	for _, metric := range bom.Metrics {
		if metric.Name == "" || seen[metric.Name] || (metric.Direction != "min" && metric.Direction != "max") || math.IsNaN(metric.Threshold) || math.IsInf(metric.Threshold, 0) {
			return fmt.Errorf("invalid evaluation metric %+v", metric)
		}
		seen[metric.Name] = true
	}
	limit := bom.Contamination
	if limit.ExactRecords < 0 || limit.FuzzyRecords < 0 || limit.ShingleWords < 1 || limit.FuzzyRatio <= 0 || limit.FuzzyRatio > 1 || math.IsNaN(limit.FuzzyRatio) {
		return fmt.Errorf("invalid contamination policy %+v", limit)
	}
	definitionHash, err := digest(struct {
		Name          string       `json:"name"`
		Task          string       `json:"task"`
		Split         string       `json:"split"`
		Metrics       []Metric     `json:"metrics"`
		Contamination OverlapLimit `json:"contamination"`
	}{bom.Name, bom.Task, bom.Split, bom.Metrics, bom.Contamination})
	if err != nil || definitionHash != bom.DefinitionSHA256 {
		return fmt.Errorf("evaluation definition digest differs")
	}
	return nil
}

func CheckContamination(ctx context.Context, trainingBOMSHA256, evaluationBOMSHA256 string, trainingRecords, evaluationRecords training.RecordSource, limit OverlapLimit) (ContaminationReport, error) {
	if trainingRecords == nil || evaluationRecords == nil {
		return ContaminationReport{}, fmt.Errorf("training and evaluation record streams are required")
	}
	evaluationSamples, err := fingerprints(ctx, evaluationRecords, limit.ShingleWords)
	if err != nil {
		return ContaminationReport{}, err
	}
	report := ContaminationReport{Kind: "openwaldo-contamination-report", Schema: 1, TrainingBOMSHA256: trainingBOMSHA256, EvaluationBOMSHA256: evaluationBOMSHA256}
	exact := map[string][]int{}
	inverted := map[string][]int{}
	for index, candidate := range evaluationSamples {
		exact[candidate.exact] = append(exact[candidate.exact], index)
		for shingle := range candidate.shingles {
			inverted[shingle] = append(inverted[shingle], index)
		}
	}
	err = trainingRecords.Stream(ctx, func(record training.Record) error {
		source := fingerprintRecord(record, limit.ShingleWords)
		exactMatches := map[int]bool{}
		for _, index := range exact[source.exact] {
			exactMatches[index] = true
			report.Exact = append(report.Exact, RecordOverlap{TrainingID: source.id, EvaluationID: evaluationSamples[index].id, Ratio: 1})
		}
		candidates := map[int]bool{}
		for shingle := range source.shingles {
			for _, index := range inverted[shingle] {
				candidates[index] = true
			}
		}
		for index := range candidates {
			if exactMatches[index] {
				continue
			}
			ratio := shingleOverlap(source.shingles, evaluationSamples[index].shingles)
			if ratio >= limit.FuzzyRatio {
				report.Fuzzy = append(report.Fuzzy, RecordOverlap{TrainingID: source.id, EvaluationID: evaluationSamples[index].id, Ratio: ratio})
			}
		}
		return nil
	})
	if err != nil {
		return ContaminationReport{}, err
	}
	sortOverlaps(report.Exact)
	sortOverlaps(report.Fuzzy)
	report.Passed = len(report.Exact) <= limit.ExactRecords && len(report.Fuzzy) <= limit.FuzzyRecords
	report.SHA256, err = digest(struct {
		Training   string          `json:"training"`
		Evaluation string          `json:"evaluation"`
		Exact      []RecordOverlap `json:"exact"`
		Fuzzy      []RecordOverlap `json:"fuzzy"`
		Passed     bool            `json:"passed"`
	}{trainingBOMSHA256, evaluationBOMSHA256, report.Exact, report.Fuzzy, report.Passed})
	return report, err
}

func DecidePromotion(bom BOM, report ContaminationReport, results []Result) Promotion {
	promotion := Promotion{Passed: true}
	if !report.Passed {
		promotion.Failures = append(promotion.Failures, "contamination policy failed")
	}
	values := map[string]float64{}
	for _, result := range results {
		values[result.Metric] = result.Value
	}
	for _, metric := range bom.Metrics {
		value, ok := values[metric.Name]
		if !ok || math.IsNaN(value) || math.IsInf(value, 0) {
			promotion.Failures = append(promotion.Failures, "missing metric "+metric.Name)
			continue
		}
		if metric.Direction == "min" && value > metric.Threshold || metric.Direction == "max" && value < metric.Threshold {
			promotion.Failures = append(promotion.Failures, fmt.Sprintf("metric %s=%g missed %s threshold %g", metric.Name, value, metric.Direction, metric.Threshold))
		}
	}
	sort.Strings(promotion.Failures)
	promotion.Passed = len(promotion.Failures) == 0
	return promotion
}

type fingerprint struct {
	id       string
	exact    string
	shingles map[string]bool
}

func fingerprints(ctx context.Context, source training.RecordSource, width int) ([]fingerprint, error) {
	var result []fingerprint
	err := source.Stream(ctx, func(record training.Record) error {
		result = append(result, fingerprintRecord(record, width))
		return nil
	})
	return result, err
}

func fingerprintRecord(record training.Record, width int) fingerprint {
	words := strings.Fields(strings.ToLower(record.Text))
	normalized := strings.Join(words, " ")
	sum := sha256.Sum256([]byte(normalized))
	shingles := map[string]bool{}
	if len(words) < width {
		if normalized != "" {
			shingles[normalized] = true
		}
	} else {
		for index := 0; index+width <= len(words); index++ {
			shingles[strings.Join(words[index:index+width], " ")] = true
		}
	}
	return fingerprint{id: record.ID, exact: hex.EncodeToString(sum[:]), shingles: shingles}
}

func shingleOverlap(left, right map[string]bool) float64 {
	if len(left) == 0 || len(right) == 0 {
		return 0
	}
	intersection := 0
	union := len(left)
	for value := range right {
		if left[value] {
			intersection++
		} else {
			union++
		}
	}
	return float64(intersection) / float64(union)
}

func sortOverlaps(values []RecordOverlap) {
	sort.Slice(values, func(i, j int) bool {
		if values[i].EvaluationID == values[j].EvaluationID {
			return values[i].TrainingID < values[j].TrainingID
		}
		return values[i].EvaluationID < values[j].EvaluationID
	})
}

func digest(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}
