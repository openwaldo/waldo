// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/evaluation"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

func runModelEvaluationBOM(context Context, args []string, stdout, stderr io.Writer) error {
	definition, err := evaluation.LoadDefinition(args[0])
	if err != nil {
		return fmt.Errorf("evaluation definition: %w", err)
	}
	targets, err := resolveIndexArgumentsWithWarning(context.Execution, definition.Corpora, stderr)
	if err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	policy, err := corpus.NewLicensePolicy(nil, nil)
	if err != nil {
		return err
	}
	corpusBOM, err := corpus.BuildBOM(context.Execution, targets, policy, cache)
	if err != nil {
		return err
	}
	if _, err := corpus.ReviewDistributable(corpusBOM); err != nil {
		return fmt.Errorf("evaluation corpus is not distributable: %w", err)
	}
	bom, err := evaluation.NewBOM(definition.Name, definition.Task, definition.Split, corpusBOM, definition.Metrics, definition.Contamination)
	if err != nil {
		return err
	}
	output, err := writeEvaluationJSON(args[1], bom, boolOption(context, "force"))
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Output string         `json:"output"`
			BOM    evaluation.BOM `json:"bom"`
		}{output, bom})
	}
	fmt.Fprintf(stdout, "wrote evaluation BOM %s to %s\n", bom.DefinitionSHA256[:12], output)
	return nil
}

func runModelGate(context Context, args []string, stdout, stderr io.Writer) error {
	var evaluationBOM evaluation.BOM
	if err := readStrictJSON(args[1], &evaluationBOM); err != nil {
		return fmt.Errorf("evaluation BOM: %w", err)
	}
	if err := evaluationBOM.Validate(); err != nil {
		return err
	}
	var results evaluation.ResultsDocument
	if err := readStrictJSON(args[2], &results); err != nil {
		return fmt.Errorf("evaluation results: %w", err)
	}
	root, err := configuredModelRoot()
	if err != nil {
		return err
	}
	inspection, err := model.Inspect(root, args[0])
	if err != nil {
		return err
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	evaluationMaterialized, err := corpus.Materialize(context.Execution, evaluationBOM.CorpusBOM, cache, modelMaterializeProgressPrinter(stderr))
	if err != nil {
		return err
	}
	evaluationParameters, err := training.ResolveParameters(training.Parameters{Steps: 1, Epochs: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 42})
	if err != nil {
		return err
	}
	evaluationRecords, err := training.NewCanonicalRecordSource(verifiedTrainingInputs(evaluationMaterialized, evaluationBOM.CorpusBOM), evaluationParameters)
	if err != nil {
		return err
	}
	evaluationBOMSHA256, err := evaluationBOM.SHA256()
	if err != nil {
		return err
	}
	for _, run := range inspection.Runs {
		if run.State == model.RunComplete && (run.Observation == nil || run.Observation.Simulated) {
			return fmt.Errorf("model %s run %s has simulated or absent training evidence", inspection.Model.Name, run.ID)
		}
	}
	// Every corpus the weights' whole lineage trained on, inherited or its own; a
	// history not completely recorded cannot show the evaluation is clean.
	closure, err := model.ModelClosure(inspection)
	if err != nil {
		return err
	}
	if !closure.Complete {
		return fmt.Errorf("model %s's training history is not completely recorded, so contamination cannot be ruled out", inspection.Model.Name)
	}
	var reports []evaluation.ContaminationReport
	for _, trained := range closure.Corpora {
		trainingMaterialized, err := corpus.Materialize(context.Execution, trained.CorpusBOM, cache, modelMaterializeProgressPrinter(stderr))
		if err != nil {
			return fmt.Errorf("materialize training corpus %s: %w", trained.CorpusBOMSHA256[:12], err)
		}
		trainingRecords, err := training.NewCanonicalRecordSource(verifiedTrainingInputs(trainingMaterialized, trained.CorpusBOM), evaluationParameters)
		if err != nil {
			return fmt.Errorf("training corpus %s records: %w", trained.CorpusBOMSHA256[:12], err)
		}
		report, err := evaluation.CheckContamination(context.Execution, trained.CorpusBOMSHA256, evaluationBOMSHA256, trainingRecords, evaluationRecords, evaluationBOM.Contamination)
		if err != nil {
			return fmt.Errorf("training corpus %s contamination: %w", trained.CorpusBOMSHA256[:12], err)
		}
		reports = append(reports, report)
	}
	gate, err := evaluation.BuildGateReport(inspection.Model.ID, evaluationBOM, reports, results)
	if err != nil {
		return err
	}
	output, err := writeEvaluationJSON(args[3], gate, boolOption(context, "force"))
	if err != nil {
		return err
	}
	if context.JSON {
		return writeJSON(stdout, struct {
			Output string                `json:"output"`
			Gate   evaluation.GateReport `json:"gate"`
		}{output, gate})
	}
	fmt.Fprintf(stdout, "wrote evaluation gate to %s (passed: %t)\n", output, gate.Promotion.Passed)
	if !gate.Promotion.Passed {
		return fmt.Errorf("model %s did not pass evaluation promotion gates", inspection.Model.Name)
	}
	return nil
}

func readStrictJSON(path string, target any) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return fmt.Errorf("unexpected content after JSON document")
	}
	return nil
}

func writeEvaluationJSON(path string, value any, force bool) (string, error) {
	output, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	flags := os.O_CREATE | os.O_EXCL | os.O_WRONLY
	if force {
		flags = os.O_CREATE | os.O_TRUNC | os.O_WRONLY
	}
	file, err := os.OpenFile(output, flags, 0o644)
	if err != nil {
		return "", err
	}
	if err := writeJSON(file, value); err != nil {
		_ = file.Close()
		_ = os.Remove(output)
		return "", err
	}
	if err := file.Close(); err != nil {
		return "", err
	}
	return output, nil
}
