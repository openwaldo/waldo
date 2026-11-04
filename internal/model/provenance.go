// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/openwaldo/waldo/internal/corpus"
)

// ClosureRole marks the training closure a release publishes and a pulled or merged origin keeps.
const ClosureRole = "training-closure"

// ClosureFile is the training closure's name in a release package.
const ClosureFile = "TRAINING-CLOSURE.json"

// RoundRole marks a distributed run's round record, which a release made by
// `waldo model join` publishes and a pulled origin keeps: the link from the
// weights back to the runs repository and the signed records that produced them.
const RoundRole = "join-round"

// RoundFile is the round record's name in a release package.
const RoundFile = "JOIN-ROUND.json"

// TrainingClosure is every distinct corpus a model's weights were trained on,
// across its whole lineage: its own runs, its parents', theirs, and so on.
// Keyed by corpus BOM, it grows with the corpora a history uses, not with its
// length. Complete is false when part of the history is not recorded.
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

// unionClosures sums uses per corpus; the union is complete only if every part is.
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

// ownClosure is the corpora of the model's own completed real runs.
func ownClosure(inspection Inspection) TrainingClosure {
	closure := emptyClosure(true)
	for position, runBOM := range inspection.RunBOMs {
		if position < len(inspection.Runs) && inspection.Runs[position].State == RunComplete && (inspection.Runs[position].Observation == nil || !inspection.Runs[position].Observation.Simulated) {
			closure.Corpora = append(closure.Corpora, ClosureCorpus{CorpusBOMSHA256: runBOM.CorpusBOMSHA256, Uses: 1, CorpusBOM: runBOM.CorpusBOM})
		}
	}
	return unionClosures(closure)
}

// InheritedClosure is the training history behind a model's starting
// weights: a local parent's whole closure, the closure a pulled release or a
// merge recorded, or nothing for weights initialized at random. Where that
// history is not recorded, the result is incomplete rather than silently empty.
func InheritedClosure(inspection Inspection) (TrainingClosure, error) {
	if parent := inspection.Model.Parent; parent != nil {
		ancestor, err := Inspect(filepath.Dir(inspection.Path), parent.Model)
		if err != nil || ancestor.Model.ID != parent.ModelID {
			return emptyClosure(false), nil
		}
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
	// A release that recorded no closure, or weights from outside WALDO, carry unrecorded history.
	return emptyClosure(false), nil
}

// ModelClosure is the model's whole training history: what its starting
// weights inherited and its own runs.
func ModelClosure(inspection Inspection) (TrainingClosure, error) {
	inherited, err := InheritedClosure(inspection)
	if err != nil {
		return TrainingClosure{}, err
	}
	return unionClosures(inherited, ownClosure(inspection)), nil
}

// writeClosure places a closure in origin/ under temporary and inventories it.
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

// PublishedRunBOMs reads the run BOMs a model carries: its own runs, or a pulled
// release's, which the pull checked against the canonical hashes they are named by.
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

// CheckClosure verifies a model's recorded training history against what it
// must be: the closure of the weights it started from (none for weights
// initialized at random), plus the corpora of its own run BOMs, complete. A
// release whose closure leaves out or adds a corpus, or whose history is not
// recorded, fails.
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

// ProvenanceRecords are the run BOMs a release of this model publishes,
// keyed by hash: a trained model's own runs; a merged model's parents' own
// runs, so a merge carries its round; a pulled model's release runs. Older
// history is published by the releases it came from, so a release never
// grows with its history.
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
	// A merge publishes every run BOM it carries. A pulled release whose node
	// names a run came from a trained model, which published its own runs;
	// any other release's run BOMs belong to the releases it came from.
	release, _, err := keptReleaseLineage(inspection)
	if err != nil {
		return nil, err
	}
	trainedRelease := release != nil && len(release.Runs) > 0
	if inspection.Origin.Merge == nil && !trainedRelease {
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

// copyProvenance places run BOMs in origin/provenance under temporary and inventories them.
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
