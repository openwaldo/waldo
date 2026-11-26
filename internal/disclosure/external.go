// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package disclosure

import (
	"fmt"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
	"time"
)

// BuildExternalEUGPAIReport reuses the corpus-only disclosure projection. Opaque
// architectures and application identities remain authoritative in the model BOM.
func BuildExternalEUGPAIReport(external model.ExternalInspection, provider *ProviderProfile, now time.Time) (EUGPAIReport, error) {
	projection := model.Inspection{Model: model.ModelRecord{ID: external.Model.ID, Name: external.Model.Name, PlanSHA256: external.Model.ID, ArchitectureSHA256: external.Plan.Identity.ArchitectureSHA256}}
	if len(external.Runs) != len(external.RunBOMs) {
		return EUGPAIReport{}, fmt.Errorf("external run/BOM count differs")
	}
	for i, r := range external.Runs {
		bom := external.RunBOMs[i]
		var stage *model.ExternalStageIdentity
		for n := range external.Plan.Identity.Stages {
			if external.Plan.Identity.Stages[n].Name == r.Stage {
				stage = &external.Plan.Identity.Stages[n]
			}
		}
		if stage == nil {
			return EUGPAIReport{}, fmt.Errorf("external stage absent from plan")
		}
		run := model.RunRecord{State: r.State}
		if r.Complete != nil {
			run.Observation = &training.Observation{ConsumedTokens: r.Complete.ConsumedTokens}
		}
		projection.Runs = append(projection.Runs, run)
		projection.RunBOMs = append(projection.RunBOMs, model.RunBOM{Ordinal: i, Stage: r.Stage, StageType: stage.Type, Objective: stage.Objective, CorpusBOMSHA256: stage.CorpusBOMSHA256, CorpusBOM: bom.CorpusBOM, Parameters: training.ResolvedParameters{Steps: stage.Data.Steps, BatchSize: stage.Data.GlobalBatchSequences, SequenceLength: stage.Data.SequenceLength}})
	}
	report, err := BuildEUGPAIReport(projection, provider, ReleaseFromModel(projection), now)
	if err != nil {
		return report, err
	}
	report.Model.BOMSHA256, err = hashJSON(external.BOM)
	return report, err
}
