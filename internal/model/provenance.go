// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sort"

	"github.com/openwaldo/waldo/internal/corpus"
)

// ClosureRole marks the training closure a release publishes and an origin keeps.
const ClosureRole = "training-closure"

const ClosureFile = "TRAINING-CLOSURE.json"

// RoundRole marks a join merge's round record, linking its weights to the signed records behind them.
const RoundRole = "join-round"

const RoundFile = "JOIN-ROUND.json"

// TrainingClosure is every corpus behind a model's weights; Complete is false where history isn't recorded.
type TrainingClosure struct {
	Kind     string          `json:"kind"`
	Schema   int             `json:"schema"`
	Subject  string          `json:"subject"`
	Complete bool            `json:"complete"`
	Corpora  []ClosureCorpus `json:"corpora"`
}

type ClosureCorpus struct {
	CorpusBOMSHA256 string     `json:"corpus_bom_sha256"`
	Uses            int        `json:"uses"`
	CorpusBOM       corpus.BOM `json:"corpus_bom"`
}

func emptyClosure(complete bool) TrainingClosure {
	return TrainingClosure{Kind: "openwaldo-bom", Schema: 1, Subject: "training-closure", Complete: complete, Corpora: []ClosureCorpus{}}
}

func unionClosures(closures ...TrainingClosure) TrainingClosure {
	result := emptyClosure(true)
	byHash := map[string]*ClosureCorpus{}
	for _, closure := range closures {
		result.Complete = result.Complete && closure.Complete
		for _, entry := range closure.Corpora {
			if existing := byHash[entry.CorpusBOMSHA256]; existing != nil {
				existing.Uses += entry.Uses
				continue
			}
			copied := entry
			byHash[entry.CorpusBOMSHA256] = &copied
		}
	}
	keys := make([]string, 0, len(byHash))
	for key := range byHash {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		result.Corpora = append(result.Corpora, *byHash[key])
	}
	return result
}

func ownClosure(inspection Inspection) TrainingClosure {
	closure := emptyClosure(true)
	for position, runBOM := range inspection.RunBOMs {
		if position < len(inspection.Runs) && inspection.Runs[position].State == RunComplete && (inspection.Runs[position].Observation == nil || !inspection.Runs[position].Observation.Simulated) {
			closure.Corpora = append(closure.Corpora, ClosureCorpus{CorpusBOMSHA256: runBOM.CorpusBOMSHA256, Uses: 1, CorpusBOM: runBOM.CorpusBOM})
		}
	}
	return unionClosures(closure)
}

// InheritedClosure is the training behind a model's starting weights, from lineage where no closure was recorded.
func InheritedClosure(inspection Inspection) (TrainingClosure, error) {
	if parent := inspection.Model.Parent; parent != nil {
		ancestor, err := Inspect(filepath.Dir(inspection.Path), parent.Model)
		if err != nil || ancestor.Model.ID != parent.ModelID {
			return historyClosure(inspection), nil
		}
		// Only the parent's training up to the run its weights came from.
		kept := 0
		for position, run := range ancestor.Runs {
			if run.ID == parent.RunID {
				kept = position + 1
			}
		}
		ancestor.Runs, ancestor.RunBOMs = ancestor.Runs[:min(kept, len(ancestor.Runs))], ancestor.RunBOMs[:min(kept, len(ancestor.RunBOMs))]
		return ModelClosure(ancestor)
	}
	if inspection.Origin == nil {
		return emptyClosure(true), nil
	}
	for _, artifact := range inspection.Origin.Artifacts {
		if artifact.Role != ClosureRole {
			continue
		}
		var closure TrainingClosure
		data, err := os.ReadFile(filepath.Join(inspection.Path, filepath.FromSlash(artifact.Path)))
		if err != nil {
			return TrainingClosure{}, err
		}
		if err := json.Unmarshal(data, &closure); err != nil || closure.Kind != "openwaldo-bom" || closure.Subject != "training-closure" || closure.Schema != 1 {
			return TrainingClosure{}, fmt.Errorf("model origin training closure is not a schema-1 training closure")
		}
		return closure, nil
	}
	return historyClosure(inspection), nil
}

func historyClosure(inspection Inspection) TrainingClosure {
	history := TrainingHistory(filepath.Dir(inspection.Path), inspection)
	closure := emptyClosure(history.Complete)
	for digest, uses := range history.Corpora {
		for _, used := range uses {
			if used.ModelID != inspection.Model.ID {
				closure.Corpora = append(closure.Corpora, ClosureCorpus{CorpusBOMSHA256: digest, Uses: 1, CorpusBOM: used.BOM})
			}
		}
	}
	return unionClosures(closure)
}

// ModelClosure is what the model's starting weights inherited plus its own runs.
func ModelClosure(inspection Inspection) (TrainingClosure, error) {
	inherited, err := InheritedClosure(inspection)
	if err != nil {
		return TrainingClosure{}, err
	}
	return unionClosures(inherited, ownClosure(inspection)), nil
}

func writeClosure(temporary string, closure TrainingClosure) ([]OriginArtifact, error) {
	logical := "origin/" + ClosureFile
	if err := writeJSONAtomic(filepath.Join(temporary, filepath.FromSlash(logical)), closure); err != nil {
		return nil, err
	}
	artifacts, err := inventoryOriginArtifacts(temporary, []string{logical})
	for index := range artifacts {
		artifacts[index].Role = ClosureRole
	}
	return artifacts, err
}

// PublishedRunBOMs reads the run BOMs a model carries, checked against the hashes that name them.
func PublishedRunBOMs(inspection Inspection) ([]RunBOM, error) {
	paths, err := ProvenanceRecords(inspection)
	if err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(paths))
	for key := range paths {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var runs []RunBOM
	for _, key := range keys {
		var run RunBOM
		data, err := os.ReadFile(paths[key])
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(data, &run); err != nil {
			return nil, fmt.Errorf("run BOM %s: %w", key[:12], err)
		}
		runs = append(runs, run)
	}
	return runs, nil
}

// CheckClosure checks a model's recorded closure is its start's closure plus its own runs, complete.
func CheckClosure(inspection Inspection, parent *Inspection) error {
	runs, err := PublishedRunBOMs(inspection)
	if err != nil {
		return err
	}
	if len(runs) == 0 {
		return fmt.Errorf("model %s carries no run BOM", inspection.Model.Name)
	}
	expected := emptyClosure(true)
	if parent != nil {
		if expected, err = ModelClosure(*parent); err != nil {
			return err
		}
	}
	own := emptyClosure(true)
	for _, run := range runs {
		own.Corpora = append(own.Corpora, ClosureCorpus{CorpusBOMSHA256: run.CorpusBOMSHA256, Uses: 1, CorpusBOM: run.CorpusBOM})
	}
	expected = unionClosures(expected, unionClosures(own))
	claimed, err := ModelClosure(inspection)
	if err != nil {
		return err
	}
	if !expected.Complete || !claimed.Complete {
		return fmt.Errorf("model %s's training history is not completely recorded", inspection.Model.Name)
	}
	summary := func(closure TrainingClosure) string {
		var parts []string
		for _, entry := range closure.Corpora {
			parts = append(parts, fmt.Sprintf("%s×%d", entry.CorpusBOMSHA256[:12], entry.Uses))
		}
		return fmt.Sprint(parts)
	}
	if summary(claimed) != summary(expected) {
		return fmt.Errorf("model %s's training closure %s does not match its run BOMs and starting weights %s", inspection.Model.Name, summary(claimed), summary(expected))
	}
	return nil
}

// ProvenanceRole marks a run BOM published beside weights, named by its hash.
const ProvenanceRole = "training-provenance"

func provenanceName(digest string) string {
	return "provenance-" + digest + ".json"
}

// ProvenanceRecords are the run BOMs a release publishes: its own, a merge's parents', or a pull's release runs.
func ProvenanceRecords(inspection Inspection) (map[string]string, error) {
	records := map[string]string{}
	trained := false
	for _, pin := range inspection.Model.Runs {
		if pin.State != RunComplete || pin.Simulated {
			continue
		}
		trained = true
		records[pin.BOMSHA256] = filepath.Join(inspection.Path, "runs", runDirectoryName(pin), "RUN-BOM.json")
	}
	if trained || inspection.Origin == nil {
		return records, nil
	}
	if !IsMerge(inspection.Origin) && !releaseTrained(inspection) {
		return records, nil
	}
	for _, artifact := range inspection.Origin.Artifacts {
		match := provenanceFile.FindStringSubmatch(filepath.Base(artifact.Path))
		if artifact.Role == ProvenanceRole && match != nil {
			records[match[1]] = filepath.Join(inspection.Path, filepath.FromSlash(artifact.Path))
		}
	}
	return records, nil
}

func copyProvenance(temporary string, records map[string]string) ([]OriginArtifact, error) {
	var paths []string
	for digest, source := range records {
		logical := "origin/provenance/" + provenanceName(digest)
		if err := copyArtifactFile(source, filepath.Join(temporary, filepath.FromSlash(logical))); err != nil {
			return nil, err
		}
		paths = append(paths, logical)
	}
	sort.Strings(paths)
	artifacts, err := inventoryOriginArtifacts(temporary, paths)
	for index := range artifacts {
		artifacts[index].Role = ProvenanceRole
	}
	return artifacts, err
}

func releaseTrained(inspection Inspection) bool {
	for _, artifact := range inspection.Origin.SourceArtifacts {
		if artifact.Role != "source-bom" {
			continue
		}
		index := newLineageIndex("", nil)
		index.loadModel(inspection.Path, true)
		var release releaseBOM
		return json.Unmarshal(index.docs[artifact.SHA256], &release) == nil && release.RunID != ""
	}
	return false
}

// CheckHistory is whether a model's lineage and closure record the same training; lineage isn't pinned.
func CheckHistory(root string, inspection Inspection) error {
	closure, err := ModelClosure(inspection)
	if err != nil {
		return err
	}
	history := TrainingHistory(root, inspection)
	var pinned []string
	for _, entry := range closure.Corpora {
		pinned = append(pinned, entry.CorpusBOMSHA256)
	}
	if history.Complete != closure.Complete || !slices.Equal(slices.Sorted(maps.Keys(history.Corpora)), pinned) {
		return fmt.Errorf("model %s's lineage and training closure record different histories", inspection.Model.Name)
	}
	return nil
}
