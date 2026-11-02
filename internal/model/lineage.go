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
		if !lineageDigest.MatchString(link.WeightsSHA256) {
			return fmt.Errorf("parent weights %q is not a SHA-256", link.WeightsSHA256)
		}
		if link.LineageSHA256 != "" && !lineageDigest.MatchString(link.LineageSHA256) {
			return fmt.Errorf("parent lineage %q is not a SHA-256", link.LineageSHA256)
		}
	}
	for _, run := range node.Runs {
		if !lineageDigest.MatchString(run) {
			return fmt.Errorf("run BOM %q is not a SHA-256", run)
		}
	}
	return nil
}

// ModelLineage is the lineage node of a model's weights after its run runID,
// or after its newest complete real run when runID is empty. A model with no
// such run is its origin, pulled or merged.
func ModelLineage(inspection Inspection, runID string) (LineageNode, error) {
	nodes, err := runLineage(inspection)
	if err != nil {
		return LineageNode{}, err
	}
	for index := len(nodes) - 1; index >= 0; index-- {
		if runID == "" || nodes[index].runID == runID {
			return nodes[index].node, nil
		}
	}
	if runID != "" {
		return LineageNode{}, fmt.Errorf("model %q has no complete real weights for run %s", inspection.Model.Name, runID)
	}
	if inspection.Origin == nil || inspection.BOM.CurrentOriginSHA256 == "" {
		return LineageNode{}, fmt.Errorf("model %q has no complete real weights", inspection.Model.Name)
	}
	return originNode(*inspection.Origin)
}

type runNode struct {
	runID string
	node  LineageNode
}

// runLineage is a node for each complete real run, in order. Like a git commit,
// each names the run that produced its weights and the node of the weights it
// started from: an earlier run's, the parent pin's, or the origin's. Weights
// with no node here, such as a simulated run's, are named by hash alone.
func runLineage(inspection Inspection) ([]runNode, error) {
	known := map[string]LineageLink{}
	if parent := inspection.Model.Parent; parent != nil {
		known[parent.Artifact.SHA256] = LineageLink{WeightsSHA256: parent.Artifact.SHA256, LineageSHA256: parent.LineageSHA256}
	} else if inspection.Origin != nil {
		start, err := originNode(*inspection.Origin)
		if err != nil {
			return nil, err
		}
		link, err := linkTo(start)
		if err != nil {
			return nil, err
		}
		known[start.WeightsSHA256] = link
	}
	var nodes []runNode
	for position, run := range inspection.Runs {
		if !realRun(inspection, position) {
			continue
		}
		weights := ""
		for _, artifact := range run.Observation.Artifacts {
			if artifact.Path == "artifacts/model.safetensors" {
				weights = artifact.SHA256
			}
		}
		if weights == "" {
			continue
		}
		node := LineageNode{WeightsSHA256: weights, Runs: []string{run.BOMSHA256}}
		if start := inspection.RunBOMs[position].Initialization; start != nil {
			link, ok := known[start.Artifact.SHA256]
			if !ok {
				link = LineageLink{WeightsSHA256: start.Artifact.SHA256}
			}
			node.Parents = []LineageLink{link}
		}
		link, err := linkTo(node)
		if err != nil {
			return nil, err
		}
		known[weights] = link
		nodes = append(nodes, runNode{run.ID, node})
	}
	return nodes, nil
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
// node hash: each run's and origin's node, and each release node an origin
// records. A pulled origin's link to its release node is declared: the release
// says where its weights came from, whatever node it names, and nothing here
// can check it. So it also returns the nodes whose parents are only claims:
// pulled origins' nodes, and release nodes no local record produced.
func (builder Builder) lineageIndex() (map[string]LineageNode, map[string]bool) {
	index := map[string]LineageNode{}
	local := map[string]bool{}
	claimed := map[string]bool{}
	stop := map[string]bool{}
	add := func(node LineageNode, into map[string]bool) string {
		digest, err := hashJSON(node)
		if err != nil {
			return ""
		}
		index[digest] = node
		into[digest] = true
		return digest
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
			// Every run's node, not only the newest, so a pin to an earlier run
			// still meets the rest of the chain after the model trains on.
			if nodes, err := runLineage(inspection); err == nil {
				for _, run := range nodes {
					add(run.node, local)
				}
			}
			if inspection.Origin == nil {
				continue
			}
			// The origin's node too, since its first run's node links to it.
			if node, err := originNode(*inspection.Origin); err == nil {
				if digest := add(node, local); inspection.Origin.Merge == nil {
					stop[digest] = true
				}
			}
			if inspection.Origin.Merge == nil && inspection.Origin.Lineage != nil {
				add(*inspection.Origin.Lineage, claimed)
			}
		}
	}
	// A release can republish a node whose model is gone from this store.
	for digest := range claimed {
		if !local[digest] {
			stop[digest] = true
		}
	}
	return index, stop
}

// ancestry walks a node's chain through the nodes the index holds, like git
// merge-base, collecting the weights it reaches; a branch ends where its node
// is not held, or at a node in stop, whose parents it does not follow.
func ancestry(node LineageNode, index map[string]LineageNode, stop map[string]bool) map[string]bool {
	reached := map[string]bool{}
	visited := map[string]bool{}
	queue := []LineageNode{node}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		reached[current.WeightsSHA256] = true
		if digest, err := hashJSON(current); err != nil || stop[digest] {
			continue
		}
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
// weights. It also reports whether that ancestor rests on lineage a pulled
// release declared rather than on this store's own records.
func (builder Builder) mergeParents(base ComposeBase) ([]mergeParent, bool, error) {
	var parents []mergeParent
	var shared, recorded map[string]bool
	index, stop := builder.lineageIndex()
	for _, declared := range base.Merge {
		inspection, err := Inspect(builder.Root, declared.Model)
		if err != nil {
			return nil, false, fmt.Errorf("resolve base.merge parent %q: %w", declared.Model, err)
		}
		if len(parents) > 0 && inspection.Model.ArchitectureSHA256 != parents[0].inspection.Model.ArchitectureSHA256 {
			return nil, false, fmt.Errorf("base.merge parent %q does not have the architecture of %q", declared.Model, parents[0].inspection.Model.Name)
		}
		if inspection.Model.Architecture.Tokenizer.Name != "byte" {
			return nil, false, fmt.Errorf("base.merge supports byte-tokenizer models; %q uses %s", declared.Model, inspection.Model.Architecture.Tokenizer.Name)
		}
		if dtype := inspection.Model.Architecture.ParameterDType; dtype != "bfloat16" && dtype != "float32" {
			return nil, false, fmt.Errorf("base.merge supports bfloat16 and float32 parameters; %q uses %s", declared.Model, dtype)
		}
		path, err := currentWeights(inspection)
		if err != nil {
			return nil, false, fmt.Errorf("base.merge parent: %w", err)
		}
		node, err := ModelLineage(inspection, "")
		if err != nil {
			return nil, false, fmt.Errorf("base.merge parent %q lineage: %w", declared.Model, err)
		}
		link, err := linkTo(node)
		if err != nil {
			return nil, false, err
		}
		shared = intersect(shared, ancestry(node, index, nil))
		recorded = intersect(recorded, ancestry(node, index, stop))
		parents = append(parents, mergeParent{inspection, path, link, declared.Weight})
	}
	if len(shared) == 0 {
		return nil, false, fmt.Errorf("base.merge parents share no recorded ancestor; only models trained from common weights can be merged")
	}
	return parents, len(recorded) == 0, nil
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
	merged, err := Inspect(filepath.Join(builder.Root, ".merges"), key)
	if err != nil || merged.Model.OriginBOMSHA256 != target.Model.OriginBOMSHA256 {
		return nil, nil
	}
	return &merged, nil
}

// holdsNode reports whether a model's own history, its origin's node or one
// of its runs', is the lineage node digest.
func holdsNode(inspection Inspection, digest string) (bool, error) {
	nodes, err := runLineage(inspection)
	if err != nil {
		return false, err
	}
	var own []LineageNode
	for _, run := range nodes {
		own = append(own, run.node)
	}
	if inspection.Origin != nil {
		node, err := originNode(*inspection.Origin)
		if err != nil {
			return false, err
		}
		own = append(own, node)
	}
	for _, node := range own {
		if hashed, err := hashJSON(node); err != nil {
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
