// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"regexp"
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
	if node.OriginSHA256 != "" && !lineageDigest.MatchString(node.OriginSHA256) {
		return fmt.Errorf("origin BOM %q is not a SHA-256", node.OriginSHA256)
	}
	return nil
}

// ModelLineage is the lineage node of a model's weights after its run runID,
// or after its newest complete real run when runID is empty. A model with no
// such run is its pulled origin.
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

// originNode is a pulled origin's node: its weights descend from the release
// they were normalized from, whose node the release recorded, if it did.
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
	digest, err := hashJSON(origin)
	if err != nil {
		return LineageNode{}, err
	}
	node := LineageNode{WeightsSHA256: weights, OriginSHA256: digest}
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
