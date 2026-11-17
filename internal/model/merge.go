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
	"slices"
	"sort"
	"strings"

	"github.com/openwaldo/waldo/internal/modelweights"
)

const mergeKind = "waldo-model-merge"

// MergeRecord names no models or places, so everyone merging the same parents writes the same bytes.
type MergeRecord struct {
	Kind    string        `json:"kind"`
	Schema  int           `json:"schema"`
	Method  string        `json:"method"`
	Parents []MergeParent `json:"parents"`
}

type MergeParent struct {
	RunBOMSHA256    string  `json:"run_bom_sha256,omitempty"`
	OriginBOMSHA256 string  `json:"origin_bom_sha256,omitempty"`
	WeightsSHA256   string  `json:"weights_sha256"`
	Share           float64 `json:"share,omitempty"`
}

func IsMerge(origin *OriginBOM) bool {
	return origin != nil && origin.Source.Provider == "merge"
}

type mergeParent struct {
	inspection Inspection
	path       string
	entry      MergeParent
}

func (builder Builder) mergeParents(base ComposeBase) ([]mergeParent, error) {
	var parents []mergeParent
	for _, declared := range base.Merge {
		inspection, err := Inspect(builder.Root, declared.Model)
		if err != nil {
			return nil, fmt.Errorf("resolve base.merge parent %q: %w", declared.Model, err)
		}
		if len(parents) > 0 && inspection.Model.ArchitectureSHA256 != parents[0].inspection.Model.ArchitectureSHA256 {
			return nil, fmt.Errorf("base.merge parent %q does not have the architecture of %q", declared.Model, parents[0].inspection.Model.Name)
		}
		if dtype := inspection.Model.Architecture.ParameterDType; dtype != "bfloat16" && dtype != "float32" {
			return nil, fmt.Errorf("base.merge supports bfloat16 and float32 parameters; %q uses %s", declared.Model, dtype)
		}
		path, entry, err := currentWeights(inspection)
		if err != nil {
			return nil, fmt.Errorf("base.merge parent: %w", err)
		}
		entry.Share = declared.Weight
		for _, parent := range parents {
			if parent.entry.WeightsSHA256 == entry.WeightsSHA256 {
				return nil, fmt.Errorf("base.merge parents %q and %q have the same weights", parent.inspection.Model.Name, declared.Model)
			}
		}
		parents = append(parents, mergeParent{inspection, path, entry})
	}
	// Float addition depends on order, so every merger sums in weights order.
	sort.Slice(parents, func(i, j int) bool { return parents[i].entry.WeightsSHA256 < parents[j].entry.WeightsSHA256 })
	return parents, nil
}

func currentWeights(inspection Inspection) (string, MergeParent, error) {
	initialization, err := resolveRunInitialization(inspection, "")
	if err != nil {
		return "", MergeParent{}, err
	}
	if initialization != nil {
		for _, pin := range inspection.Model.Runs {
			if pin.ID == initialization.SourceRunID {
				return initialization.Path, MergeParent{RunBOMSHA256: pin.BOMSHA256, WeightsSHA256: initialization.Artifact.SHA256}, nil
			}
		}
	}
	if inspection.Origin != nil && inspection.BOM.CurrentOriginSHA256 != "" {
		for _, artifact := range inspection.Origin.Artifacts {
			if artifact.Role == "weights" {
				return filepath.Join(inspection.Path, filepath.FromSlash(artifact.Path)), MergeParent{OriginBOMSHA256: inspection.Model.OriginBOMSHA256, WeightsSHA256: artifact.SHA256}, nil
			}
		}
	}
	return "", MergeParent{}, fmt.Errorf("model %q has no complete real weights", inspection.Model.Name)
}

func checkMergeHistory(root string, parents []mergeParent, warn func(string)) error {
	walk := newLineageWalk(newLineageIndex(root, warn))
	tips := map[string]string{}
	for _, parent := range parents {
		runID := ""
		if parent.entry.RunBOMSHA256 != "" {
			runID = parent.inspection.BOM.CurrentRunID
		}
		walk.model(parent.inspection, runID)
		tips[parent.entry.WeightsSHA256] = parent.inspection.Model.Name
	}
	return walk.mergeBase(tips)
}

// mergeBase finds git's merge-bases and refuses gaps, ends, or outside merges above them.
func (walk *lineageWalk) mergeBase(tips map[string]string) error {
	var common map[string]bool
	for tip := range tips {
		common = intersect(common, walk.ancestors(tip))
	}
	if len(common) == 0 && len(walk.gaps) > 0 {
		return fmt.Errorf("base.merge parents' histories have gaps, so where they meet can't be shown: %s", gapNames(walk.sortedGaps()))
	}
	if len(common) == 0 {
		return fmt.Errorf("base.merge parents share no ancestor; only models trained from common weights can be merged")
	}
	base := map[string]bool{}
	for candidate := range common {
		base[candidate] = true
		for other := range common {
			if other != candidate && walk.ancestors(other)[candidate] {
				base[candidate] = false
				break
			}
		}
	}
	below := map[string]bool{}
	for candidate, lowest := range base {
		if lowest {
			for ancestor := range walk.ancestors(candidate) {
				below[ancestor] = true
			}
		}
	}
	for tip, name := range tips {
		for weights := range walk.ancestors(tip) {
			if below[weights] {
				continue
			}
			switch {
			case walk.broken[weights]:
				return fmt.Errorf("base.merge parent %q has a gap in its history above where the parents meet (weights %s): %s", name, short(weights), gapNames(walk.sortedGaps()))
			case walk.ends[weights]:
				return fmt.Errorf("base.merge parent %q history starts at weights %s, above where the parents meet", name, short(weights))
			case walk.merged[weights]:
				// An earlier merge's parents are family when they reach any common ancestor:
				// a merge of m1 = merge(left, right) with left meets at left, and right is still family.
				for _, from := range walk.parents[weights] {
					if !reachesAny(walk.ancestors(from), common) {
						return fmt.Errorf("base.merge parent %q descends from a merge with a parent outside the family (weights %s)", name, short(from))
					}
				}
			}
		}
	}
	return nil
}

func (walk *lineageWalk) ancestors(weights string) map[string]bool {
	reached := map[string]bool{weights: true}
	queue := []string{weights}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for _, from := range walk.parents[current] {
			if !reached[from] {
				reached[from] = true
				queue = append(queue, from)
			}
		}
	}
	return reached
}

func reachesAny(reached, base map[string]bool) bool {
	for weights, lowest := range base {
		if lowest && reached[weights] {
			return true
		}
	}
	return false
}

func intersect(shared, reached map[string]bool) map[string]bool {
	if shared == nil {
		return reached
	}
	for digest := range shared {
		if !reached[digest] {
			delete(shared, digest)
		}
	}
	return shared
}

func mergeRecord(parents []mergeParent) (MergeRecord, []byte, error) {
	record := MergeRecord{Kind: mergeKind, Schema: 1, Method: "mean-f32-v1"}
	for _, parent := range parents {
		record.Parents = append(record.Parents, parent.entry)
	}
	if parents[0].entry.Share > 0 {
		record.Method = "weighted-mean-f32-v1"
	}
	data, err := json.Marshal(record)
	return record, data, err
}

// recordedMerge keeps an existing model on the merge it was built from, as a run pin keeps a fine-tune's base.
func (builder Builder) recordedMerge(name string, base ComposeBase) (*Inspection, error) {
	if len(base.Merge) == 0 || ValidateName(name) != nil {
		return nil, nil
	}
	if _, err := os.Stat(filepath.Join(builder.Root, name)); os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	target, err := Inspect(builder.Root, name)
	if err != nil {
		return nil, err
	}
	mismatch := fmt.Errorf("compose base does not match existing model %q; use a new model name", name)
	if !IsMerge(target.Origin) {
		return nil, mismatch
	}
	key := target.Origin.Source.Revision
	merged, err := Inspect(filepath.Join(builder.Root, ".merges"), key)
	if err != nil || merged.Model.OriginBOMSHA256 != target.Model.OriginBOMSHA256 {
		return nil, fmt.Errorf("model %q was built from the merge in .merges/%s, which is missing or changed; restore it or use a new model name", name, short(key))
	}
	record, err := ReadMerge(merged)
	if err != nil || record == nil || len(record.Parents) != len(base.Merge) {
		return nil, mismatch
	}
	holds := make([][]bool, len(base.Merge))
	for index, declared := range base.Merge {
		current, err := Inspect(builder.Root, declared.Model)
		if err != nil {
			return nil, fmt.Errorf("resolve base.merge parent %q: %w", declared.Model, err)
		}
		for _, parent := range record.Parents {
			holds[index] = append(holds[index], parent.Share == declared.Weight && holdsWeights(current, parent.WeightsSHA256))
		}
	}
	// Each declared parent must answer for a different recorded parent.
	owner := slices.Repeat([]int{-1}, len(record.Parents))
	var assign func(int, []bool) bool
	assign = func(declared int, tried []bool) bool {
		for recorded, held := range holds[declared] {
			if held && !tried[recorded] {
				tried[recorded] = true
				if owner[recorded] < 0 || assign(owner[recorded], tried) {
					owner[recorded] = declared
					return true
				}
			}
		}
		return false
	}
	for index, declared := range base.Merge {
		if !assign(index, make([]bool, len(record.Parents))) {
			return nil, fmt.Errorf("base.merge parent %q no longer holds weights model %q merged that no other parent does; use a new model name", declared.Model, name)
		}
	}
	return &merged, nil
}

// ReadMerge is nil for models that aren't merges.
func ReadMerge(inspection Inspection) (*MergeRecord, error) {
	if !IsMerge(inspection.Origin) {
		return nil, nil
	}
	index := newLineageIndex("", nil)
	index.loadModel(inspection.Path, true)
	var record MergeRecord
	if data := index.docs[inspection.Origin.Source.Revision]; data == nil || json.Unmarshal(data, &record) != nil {
		return nil, fmt.Errorf("model %q MERGE.json isn't in its history", inspection.Model.Name)
	}
	return &record, nil
}

func holdsWeights(inspection Inspection, digest string) bool {
	if inspection.Origin != nil {
		if weights, ok := originWeights(inspection.Origin.Artifacts); ok && weights == digest {
			return true
		}
	}
	for _, run := range inspection.BOM.Runs {
		if weights, ok := bomRunWeights(run); ok && weights == digest {
			return true
		}
	}
	return false
}

// mergeOrigin names a merge by its MERGE.json hash under .merges, so it is built once.
func (builder Builder) mergeOrigin(base ComposeBase) (Inspection, error) {
	parents, err := builder.mergeParents(base)
	if err != nil {
		return Inspection{}, err
	}
	warn := func(message string) { builder.report(Progress{Phase: "lineage", Message: message}) }
	if err := checkMergeHistory(builder.Root, parents, warn); err != nil {
		return Inspection{}, err
	}
	record, data, err := mergeRecord(parents)
	if err != nil {
		return Inspection{}, err
	}
	key := sha256Hex(data)
	root := filepath.Join(builder.Root, ".merges")
	if _, err := os.Stat(filepath.Join(root, key)); err == nil {
		return Inspect(root, key)
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return Inspection{}, err
	}
	temporary, err := os.MkdirTemp(root, ".waldo-merge-*")
	if err != nil {
		return Inspection{}, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(temporary)
		}
	}()
	artifactDirectory := filepath.Join(temporary, "origin", "artifacts")
	if err := os.MkdirAll(artifactDirectory, 0o755); err != nil {
		return Inspection{}, err
	}
	var paths, names []string
	var shares []float64
	for _, parent := range parents {
		paths = append(paths, parent.path)
		names = append(names, parent.inspection.Model.Name)
		shares = append(shares, parent.entry.Share)
	}
	if record.Method == "mean-f32-v1" {
		shares = nil
	}
	builder.report(Progress{Phase: "merge", Message: fmt.Sprintf("merging %s (%s)", strings.Join(names, ", "), record.Method)})
	if _, err := modelweights.Merge(paths, shares, filepath.Join(artifactDirectory, "model.safetensors")); err != nil {
		return Inspection{}, err
	}
	architecture := parents[0].inspection.Model.Architecture
	artifacts, err := writeOriginRuntime(temporary, architecture, parents[0].inspection.Model.ArchitectureSHA256)
	if err != nil {
		return Inspection{}, err
	}
	carried := map[string]string{}
	for _, parent := range parents {
		records, err := ProvenanceRecords(parent.inspection)
		if err != nil {
			return Inspection{}, err
		}
		for digest, path := range records {
			carried[digest] = path
		}
	}
	provenance, err := copyProvenance(temporary, carried)
	if err != nil {
		return Inspection{}, err
	}
	artifacts = append(artifacts, provenance...)
	// The merge inherits every parent's whole training history.
	var closures []TrainingClosure
	for _, parent := range parents {
		closure, err := ModelClosure(parent.inspection)
		if err != nil {
			return Inspection{}, err
		}
		closures = append(closures, closure)
	}
	closure, err := writeClosure(temporary, unionClosures(closures...))
	if err != nil {
		return Inspection{}, err
	}
	artifacts = append(artifacts, closure...)
	origin := OriginBOM{
		Kind: "openwaldo-bom", Schema: 1, Subject: "model-origin",
		Source:             OriginSource{Provider: "merge", RequestedRevision: key, Revision: key},
		ArchitectureSHA256: parents[0].inspection.Model.ArchitectureSHA256,
		SourceArtifacts:    []OriginArtifact{{Role: "source-merge", Path: "MERGE.json", SHA256: key, Bytes: int64(len(data))}},
		Artifacts:          artifacts,
	}
	lineage, err := mergeLineage(builder.Root, parents, origin, data, warn)
	if err != nil {
		return Inspection{}, err
	}
	if err := writeJSONAtomic(filepath.Join(temporary, lineageFile), lineage); err != nil {
		return Inspection{}, err
	}
	if err := publishOrigin(root, key, temporary, origin, architecture, builder.clock()()); err != nil {
		return Inspection{}, err
	}
	committed = true
	return Inspect(root, key)
}

func mergeLineage(root string, parents []mergeParent, origin OriginBOM, record []byte, warn func(string)) (LineageFile, error) {
	index := newLineageIndex(root, warn)
	for _, parent := range parents {
		index.loadModel(parent.inspection.Path, true)
	}
	index.docs[sha256Hex(record)] = record
	digest, err := index.addRecord(origin)
	if err != nil {
		return LineageFile{}, err
	}
	walk := newLineageWalk(index)
	weights, _ := originWeights(origin.Artifacts)
	walk.origin(digest, weights)
	return walk.file(), nil
}
