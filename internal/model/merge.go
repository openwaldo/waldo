// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/openwaldo/waldo/internal/modelweights"
)

// ancestry is the weights a node's history reaches, like git merge-base: its own, every held
// node's, and every link's. With recorded, it stops at pulled origins, whose parents a release only declared.
func ancestry(node LineageNode, history []LineageNode, recorded bool) map[string]bool {
	held := map[string]LineageNode{}
	for _, ancestor := range history {
		if digest, err := hashJSON(ancestor); err == nil {
			held[digest] = ancestor
		}
	}
	reached := map[string]bool{}
	queue := []LineageNode{node}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		reached[current.WeightsSHA256] = true
		if recorded && current.SourceSHA256 != "" {
			continue
		}
		for _, link := range current.Parents {
			reached[link.WeightsSHA256] = true
			if next, ok := held[link.LineageSHA256]; ok {
				delete(held, link.LineageSHA256)
				queue = append(queue, next)
			}
		}
	}
	return reached
}

// mergeParent is a parent's inspection, current weights, link to its node, and the history behind it.
type mergeParent struct {
	inspection Inspection
	path       string
	link       LineageLink
	node       LineageNode
	history    []LineageNode
	weight     float64
}

// mergeParents resolves base.merge and fails closed unless every parent has
// the same architecture and all of them share an ancestor their lineage
// shows. Averaging weights is only meaningful for models trained from common
// weights. It also reports whether that ancestor rests on lineage a pulled
// release declared rather than on this store's own records.
func (builder Builder) mergeParents(base ComposeBase) ([]mergeParent, bool, error) {
	var parents []mergeParent
	var shared, recorded map[string]bool
	inherited := false
	for _, declared := range base.Merge {
		inspection, err := Inspect(builder.Root, declared.Model)
		if err != nil {
			return nil, false, fmt.Errorf("resolve base.merge parent %q: %w", declared.Model, err)
		}
		if len(parents) > 0 && inspection.Model.ArchitectureSHA256 != parents[0].inspection.Model.ArchitectureSHA256 {
			return nil, false, fmt.Errorf("base.merge parent %q does not have the architecture of %q", declared.Model, parents[0].inspection.Model.Name)
		}
		if dtype := inspection.Model.Architecture.ParameterDType; dtype != "bfloat16" && dtype != "float32" {
			return nil, false, fmt.Errorf("base.merge supports bfloat16 and float32 parameters; %q uses %s", declared.Model, dtype)
		}
		path, err := currentWeights(inspection)
		if err != nil {
			return nil, false, fmt.Errorf("base.merge parent: %w", err)
		}
		node, history, err := LineageHistory(inspection, "")
		if err != nil {
			return nil, false, fmt.Errorf("base.merge parent %q lineage: %w", declared.Model, err)
		}
		digest, err := hashJSON(node)
		if err != nil {
			return nil, false, err
		}
		shared = intersect(shared, ancestry(node, history, false))
		recorded = intersect(recorded, ancestry(node, history, true))
		// A parent built on a declared merge makes this one declared too.
		inherited = inherited || inspection.Origin != nil && inspection.Origin.Merge != nil && inspection.Origin.Merge.DeclaredAncestry
		parents = append(parents, mergeParent{inspection, path, LineageLink{WeightsSHA256: node.WeightsSHA256, LineageSHA256: digest}, node, history, declared.Weight})
	}
	if len(shared) == 0 {
		return nil, false, fmt.Errorf("base.merge parents share no recorded ancestor; only models trained from common weights can be merged")
	}
	return parents, len(recorded) == 0 || inherited, nil
}

// intersect keeps the weights both sets reached; a nil set is the first one seen.
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

// currentWeights is the path of the newest complete real run's weights, else the origin's.
func currentWeights(inspection Inspection) (string, error) {
	initialization, err := resolveRunInitialization(inspection, "")
	if err != nil {
		return "", err
	}
	if initialization != nil {
		return initialization.Path, nil
	}
	if inspection.Origin != nil && inspection.BOM.CurrentOriginSHA256 != "" {
		for _, artifact := range inspection.Origin.Artifacts {
			if artifact.Role == "weights" {
				return filepath.Join(inspection.Path, filepath.FromSlash(artifact.Path)), nil
			}
		}
	}
	return "", fmt.Errorf("model %q has no complete real weights", inspection.Model.Name)
}

// recordedMerge is the merge an existing model name was built from: its base
// stays that merge after the parents train on, as a run pin keeps a
// fine-tune's base. An existing model built otherwise, or from parents whose
// names no longer hold the weights that were averaged, is refused before
// anything is merged.
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
	if target.Origin == nil || target.Origin.Merge == nil || len(target.Origin.Merge.Parents) != len(base.Merge) {
		return nil, mismatch
	}
	for index, parent := range target.Origin.Merge.Parents {
		if parent.Model != base.Merge[index].Model || parent.Weight != base.Merge[index].Weight {
			return nil, mismatch
		}
		current, err := Inspect(builder.Root, parent.Model)
		if err != nil {
			return nil, fmt.Errorf("resolve base.merge parent %q: %w", parent.Model, err)
		}
		if held, err := holdsNode(current, parent.LineageSHA256); err != nil {
			return nil, err
		} else if !held {
			return nil, fmt.Errorf("base.merge parent %q no longer holds the weights model %q merged; use a new model name", parent.Model, name)
		}
	}
	key, err := hashJSON(*target.Origin.Merge)
	if err != nil {
		return nil, err
	}
	// The model's weights are this merge's; averaging the parents' newer
	// weights instead would build a different one.
	merged, err := Inspect(filepath.Join(builder.Root, ".merges"), key)
	if err != nil || merged.Model.OriginBOMSHA256 != target.Model.OriginBOMSHA256 {
		return nil, fmt.Errorf("model %q was built from the merge in .merges/%s, which is missing or changed; restore it or use a new model name", name, key)
	}
	return &merged, nil
}

// holdsNode reports whether a model's own history, its origin's node or one
// of its runs', is the lineage node digest.
func holdsNode(inspection Inspection, digest string) (bool, error) {
	start, held, err := startingLineage(inspection)
	if err != nil {
		return false, err
	}
	if start != nil && start.LineageSHA256 == digest {
		return true, nil
	}
	nodes, err := runLineage(inspection, start, held)
	if err != nil {
		return false, err
	}
	for _, run := range nodes {
		if hashed, err := hashJSON(run.node); err != nil {
			return false, err
		} else if hashed == digest {
			return true, nil
		}
	}
	return false, nil
}

// mergeOrigin averages the parents into a merged origin under .merges, named
// by the parents and weights so the same merge is built once. Its lineage
// node names every parent's node, like a git merge commit.
func (builder Builder) mergeOrigin(base ComposeBase) (Inspection, error) {
	parents, declared, err := builder.mergeParents(base)
	if err != nil {
		return Inspection{}, err
	}
	// Method names version the exact arithmetic: float32 accumulation in parent
	// order, rounded to nearest-even, so the same parents always merge to the same bytes.
	merge := &OriginMerge{Method: "mean-f32-v1", DeclaredAncestry: declared}
	var paths []string
	var weights []float64
	for _, parent := range parents {
		merge.Parents = append(merge.Parents, OriginMergeParent{Model: parent.inspection.Model.Name, ModelID: parent.inspection.Model.ID, WeightsSHA256: parent.link.WeightsSHA256, LineageSHA256: parent.link.LineageSHA256, Weight: parent.weight})
		paths = append(paths, parent.path)
		weights = append(weights, parent.weight)
	}
	if parents[0].weight > 0 {
		merge.Method = "weighted-mean-f32-v1"
	} else {
		weights = nil
	}
	key, err := hashJSON(merge)
	if err != nil {
		return Inspection{}, err
	}
	root := filepath.Join(builder.Root, ".merges")
	if _, err := os.Stat(filepath.Join(root, key)); err == nil {
		cached, err := Inspect(root, key)
		if err != nil {
			return Inspection{}, err
		}
		return cached, nil
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
	builder.report(Progress{Phase: "merge", Message: fmt.Sprintf("merging %d models (%s)", len(parents), merge.Method)})
	if _, err := modelweights.Merge(paths, weights, filepath.Join(artifactDirectory, "model.safetensors")); err != nil {
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
		Kind: "openwaldo-bom", Schema: 1, Subject: "model-origin", Source: OriginSource{Provider: "merge"},
		ArchitectureSHA256: parents[0].inspection.Model.ArchitectureSHA256, SourceArtifacts: []OriginArtifact{}, Artifacts: artifacts,
		Merge: merge,
	}
	// LINEAGE.json holds every parent's node and history, so the parents can be deleted.
	node, err := mergeNode(origin)
	if err != nil {
		return Inspection{}, err
	}
	record := lineageRecord{Kind: lineageKind, Schema: 1, Parent: node}
	seen := map[string]bool{}
	for _, parent := range parents {
		for _, ancestor := range append([]LineageNode{parent.node}, parent.history...) {
			if digest, err := hashJSON(ancestor); err != nil {
				return Inspection{}, err
			} else if !seen[digest] {
				seen[digest] = true
				record.Ancestors = append(record.Ancestors, ancestor)
			}
		}
	}
	if err := writeJSONAtomic(filepath.Join(temporary, lineageFile), record); err != nil {
		return Inspection{}, err
	}
	if err := publishOrigin(root, key, temporary, origin, architecture, builder.clock()()); err != nil {
		return Inspection{}, err
	}
	committed = true
	return Inspect(root, key)
}
