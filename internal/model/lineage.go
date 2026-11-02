// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"github.com/openwaldo/waldo/internal/modelweights"
)

var lineageDigest = regexp.MustCompile(`^[a-f0-9]{64}$`)

func (node LineageNode) validate() error {
	if !lineageDigest.MatchString(node.WeightsSHA256) {
		return fmt.Errorf("weights %q is not a SHA-256", node.WeightsSHA256)
	}
	for _, link := range node.Parents {
		if !lineageDigest.MatchString(link.WeightsSHA256) || (link.LineageSHA256 != "" && !lineageDigest.MatchString(link.LineageSHA256)) {
			return fmt.Errorf("parent %q is not a SHA-256", link.WeightsSHA256)
		}
	}
	for _, run := range node.Runs {
		if !lineageDigest.MatchString(run) {
			return fmt.Errorf("run BOM %q is not a SHA-256", run)
		}
	}
	return nil
}

// ModelLineage is a model's lineage node: its current weights and the weights
// they came from. Trained weights descend from the model's starting weights;
// a model with no runs is its origin, pulled or merged.
func ModelLineage(inspection Inspection) (LineageNode, error) {
	initialization, err := resolveRunInitialization(inspection, "")
	if err != nil {
		return LineageNode{}, err
	}
	if initialization == nil {
		if inspection.Origin == nil || inspection.BOM.CurrentOriginSHA256 == "" {
			return LineageNode{}, fmt.Errorf("model %q has no complete real weights", inspection.Model.Name)
		}
		return originNode(*inspection.Origin)
	}
	node := LineageNode{WeightsSHA256: initialization.Artifact.SHA256}
	for _, pin := range inspection.Model.Runs {
		if pin.State == RunComplete && !pin.Simulated {
			node.Runs = append(node.Runs, pin.BOMSHA256)
		}
	}
	if parent := inspection.Model.Parent; parent != nil {
		node.Parents = []LineageLink{{WeightsSHA256: parent.Artifact.SHA256, LineageSHA256: parent.LineageSHA256}}
	} else if inspection.Origin != nil {
		start, err := originNode(*inspection.Origin)
		if err != nil {
			return LineageNode{}, err
		}
		link, err := linkTo(start)
		if err != nil {
			return LineageNode{}, err
		}
		node.Parents = []LineageLink{link}
	}
	return node, nil
}

// originNode is an origin's node. A merge records its own; pulled weights
// descend from the release they were normalized from, whose node the release
// recorded, if it did.
func originNode(origin OriginBOM) (LineageNode, error) {
	weights := ""
	for _, artifact := range origin.Artifacts {
		if artifact.Role == "weights" {
			weights = artifact.SHA256
		}
	}
	if weights == "" {
		return LineageNode{}, fmt.Errorf("model origin has no weights artifact")
	}
	if origin.Merge != nil {
		if origin.Lineage == nil || origin.Lineage.WeightsSHA256 != weights {
			return LineageNode{}, fmt.Errorf("merged origin does not record its own lineage")
		}
		return *origin.Lineage, nil
	}
	node := LineageNode{WeightsSHA256: weights}
	if origin.Lineage != nil {
		link, err := linkTo(*origin.Lineage)
		if err != nil {
			return LineageNode{}, err
		}
		node.Parents = []LineageLink{link}
	}
	return node, nil
}

func linkTo(node LineageNode) (LineageLink, error) {
	digest, err := hashJSON(node)
	return LineageLink{WeightsSHA256: node.WeightsSHA256, LineageSHA256: digest}, err
}

// lineageIndex holds every lineage node this model store can show, keyed by
// node hash: each model's own node and each release node its origin records.
func (builder Builder) lineageIndex() map[string]LineageNode {
	index := map[string]LineageNode{}
	add := func(node LineageNode) {
		if digest, err := hashJSON(node); err == nil {
			index[digest] = node
		}
	}
	for _, root := range []string{builder.Root, filepath.Join(builder.Root, ".merges"), filepath.Join(builder.Root, ".origins")} {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if !entry.IsDir() || !validName.MatchString(entry.Name()) {
				continue
			}
			inspection, err := Inspect(root, entry.Name())
			if err != nil {
				continue
			}
			if node, err := ModelLineage(inspection); err == nil {
				add(node)
			}
			if inspection.Origin != nil && inspection.Origin.Lineage != nil {
				add(*inspection.Origin.Lineage)
			}
		}
	}
	return index
}

// ancestry walks a node's chain through the nodes the index holds, like git
// merge-base, collecting the weights it reaches; a branch ends where its node
// is not held.
func ancestry(node LineageNode, index map[string]LineageNode) map[string]bool {
	reached := map[string]bool{}
	visited := map[string]bool{}
	queue := []LineageNode{node}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		reached[current.WeightsSHA256] = true
		for _, link := range current.Parents {
			reached[link.WeightsSHA256] = true
			if next, ok := index[link.LineageSHA256]; ok && !visited[link.LineageSHA256] {
				visited[link.LineageSHA256] = true
				queue = append(queue, next)
			}
		}
	}
	return reached
}

type mergeParent struct {
	inspection Inspection
	path       string
	link       LineageLink
	weight     float64
}

// mergeParents resolves base.merge and fails closed unless every parent has
// the same architecture and all of them share an ancestor their lineage
// shows. Averaging weights is only meaningful for models trained from common
// weights.
func (builder Builder) mergeParents(base ComposeBase) ([]mergeParent, error) {
	var parents []mergeParent
	var shared map[string]bool
	index := builder.lineageIndex()
	for _, declared := range base.Merge {
		inspection, err := Inspect(builder.Root, declared.Model)
		if err != nil {
			return nil, fmt.Errorf("resolve base.merge parent %q: %w", declared.Model, err)
		}
		if len(parents) > 0 && inspection.Model.ArchitectureSHA256 != parents[0].inspection.Model.ArchitectureSHA256 {
			return nil, fmt.Errorf("base.merge parent %q does not have the architecture of %q", declared.Model, parents[0].inspection.Model.Name)
		}
		if inspection.Model.Architecture.Tokenizer.Name != "byte" {
			return nil, fmt.Errorf("base.merge supports byte-tokenizer models; %q uses %s", declared.Model, inspection.Model.Architecture.Tokenizer.Name)
		}
		path, _, err := currentWeights(inspection)
		if err != nil {
			return nil, fmt.Errorf("base.merge parent: %w", err)
		}
		node, err := ModelLineage(inspection)
		if err != nil {
			return nil, fmt.Errorf("base.merge parent %q lineage: %w", declared.Model, err)
		}
		link, err := linkTo(node)
		if err != nil {
			return nil, err
		}
		reached := ancestry(node, index)
		if shared == nil {
			shared = reached
		} else {
			for digest := range shared {
				if !reached[digest] {
					delete(shared, digest)
				}
			}
		}
		parents = append(parents, mergeParent{inspection, path, link, declared.Weight})
	}
	if len(shared) == 0 {
		return nil, fmt.Errorf("base.merge parents share no recorded ancestor; only models trained from common weights can be merged")
	}
	return parents, nil
}

// currentWeights is the newest complete real run's weights, else the origin's.
func currentWeights(inspection Inspection) (string, string, error) {
	initialization, err := resolveRunInitialization(inspection, "")
	if err != nil {
		return "", "", err
	}
	if initialization != nil {
		return initialization.Path, initialization.Artifact.SHA256, nil
	}
	if inspection.Origin != nil && inspection.BOM.CurrentOriginSHA256 != "" {
		for _, artifact := range inspection.Origin.Artifacts {
			if artifact.Role == "weights" {
				return filepath.Join(inspection.Path, filepath.FromSlash(artifact.Path)), artifact.SHA256, nil
			}
		}
	}
	return "", "", fmt.Errorf("model %q has no complete real weights", inspection.Model.Name)
}

// mergeOrigin averages the parents into a merged origin under .merges, named
// by the parents and weights so the same merge is built once. Its lineage
// node names every parent's node, like a git merge commit.
func (builder Builder) mergeOrigin(base ComposeBase) (Inspection, error) {
	parents, err := builder.mergeParents(base)
	if err != nil {
		return Inspection{}, err
	}
	// Method names version the exact arithmetic: float32 accumulation in parent
	// order, rounded to nearest-even, so the same parents always merge to the same bytes.
	merge := &OriginMerge{Method: "mean-f32-v1"}
	var paths []string
	var weights []float64
	var links []LineageLink
	for _, parent := range parents {
		merge.Parents = append(merge.Parents, OriginMergeParent{Model: parent.inspection.Model.Name, ModelID: parent.inspection.Model.ID, WeightsSHA256: parent.link.WeightsSHA256, LineageSHA256: parent.link.LineageSHA256, Weight: parent.weight})
		paths = append(paths, parent.path)
		weights = append(weights, parent.weight)
		links = append(links, parent.link)
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
	builder.report(Progress{Phase: "merge", Message: fmt.Sprintf("merging %d models (%s)", len(parents), merge.Method)})
	if _, err := modelweights.Merge(paths, weights, filepath.Join(artifactDirectory, "model.safetensors")); err != nil {
		return Inspection{}, err
	}
	architecture := parents[0].inspection.Model.Architecture
	artifacts, err := writeOriginRuntime(temporary, architecture, parents[0].inspection.Model.ArchitectureSHA256)
	if err != nil {
		return Inspection{}, err
	}
	origin := OriginBOM{
		Kind: "openwaldo-bom", Schema: 1, Subject: "model-origin", Source: OriginSource{Provider: "merge"},
		ArchitectureSHA256: parents[0].inspection.Model.ArchitectureSHA256, SourceArtifacts: []OriginArtifact{}, Artifacts: artifacts,
		Merge: merge,
	}
	for _, artifact := range artifacts {
		if artifact.Role == "weights" {
			origin.Lineage = &LineageNode{WeightsSHA256: artifact.SHA256, Parents: links}
		}
	}
	if err := publishOrigin(root, key, temporary, origin, architecture, builder.clock()()); err != nil {
		return Inspection{}, err
	}
	committed = true
	return Inspect(root, key)
}
