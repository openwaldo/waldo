// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
)

var lineageDigest = regexp.MustCompile(`^[a-f0-9]{64}$`)

const (
	// LineageFile is the chain behind a model's parent run, kept beside its records.
	LineageFile = "LINEAGE.json"
	// ReleaseBOMPath is where a pull keeps the release's BOM.json, which its origin pins.
	ReleaseBOMPath = "origin/release/BOM.json"
	lineageKind    = "waldo-model-lineage"
)

// lineageRecord is LINEAGE.json. A later WALDO that changes it bumps Schema.
type lineageRecord struct {
	Kind      string        `json:"kind"`
	Schema    int           `json:"schema"`
	Parent    LineageNode   `json:"parent"`
	Ancestors []LineageNode `json:"ancestors,omitempty"`
}

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
	if node.SourceSHA256 != "" && !lineageDigest.MatchString(node.SourceSHA256) {
		return fmt.Errorf("source %q is not a SHA-256", node.SourceSHA256)
	}
	return nil
}

// ModelLineage is the node of a model's weights after run runID, or after its
// newest complete real run when runID is empty, else its pulled origin's.
func ModelLineage(inspection Inspection, runID string) (LineageNode, error) {
	node, _, err := modelLineage(inspection, runID)
	return node, err
}

// LineageHistory is ModelLineage's node and every node behind it the model
// holds, nearest first; the chain ends at the first node it does not hold.
func LineageHistory(inspection Inspection, runID string) (LineageNode, []LineageNode, error) {
	node, held, err := modelLineage(inspection, runID)
	if err != nil {
		return LineageNode{}, nil, err
	}
	return node, chain(node, held), nil
}

func modelLineage(inspection Inspection, runID string) (LineageNode, map[string]LineageNode, error) {
	start, held, err := startingLineage(inspection)
	if err != nil {
		return LineageNode{}, nil, err
	}
	nodes, err := runLineage(inspection, start, held)
	if err != nil {
		return LineageNode{}, nil, err
	}
	for index := len(nodes) - 1; index >= 0; index-- {
		if runID == "" || nodes[index].runID == runID {
			return nodes[index].node, held, nil
		}
	}
	if runID != "" {
		return LineageNode{}, nil, fmt.Errorf("model %q has no complete real weights for run %s", inspection.Model.Name, runID)
	}
	if inspection.Origin == nil || inspection.BOM.CurrentOriginSHA256 == "" {
		return LineageNode{}, nil, fmt.Errorf("model %q has no complete real weights", inspection.Model.Name)
	}
	return held[start.LineageSHA256], held, nil
}

type runNode struct {
	runID string
	node  LineageNode
}

// runLineage is a node for each complete real run, in order, each linked to
// the node of the weights it started from, and holds each in held.
func runLineage(inspection Inspection, start *LineageLink, held map[string]LineageNode) ([]runNode, error) {
	known := map[string]LineageLink{}
	if start != nil {
		known[start.WeightsSHA256] = *start
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
		if initialization := inspection.RunBOMs[position].Initialization; initialization != nil {
			link, ok := known[initialization.Artifact.SHA256]
			if !ok {
				link = LineageLink{WeightsSHA256: initialization.Artifact.SHA256}
			}
			node.Parents = []LineageLink{link}
		}
		link, err := hold(held, node)
		if err != nil {
			return nil, err
		}
		known[weights] = link
		nodes = append(nodes, runNode{run.ID, node})
	}
	return nodes, nil
}

// startingLineage is the link to the node a model's first run starts from,
// and the nodes the model holds behind it. A model without lineage starts
// from its parent's weights alone.
func startingLineage(inspection Inspection) (*LineageLink, map[string]LineageNode, error) {
	held := map[string]LineageNode{}
	if parent := inspection.Model.Parent; parent != nil {
		record, err := readLineageRecord(filepath.Join(inspection.Path, LineageFile))
		if err != nil {
			return nil, nil, err
		}
		if record == nil {
			return &LineageLink{WeightsSHA256: parent.Artifact.SHA256}, held, nil
		}
		// The parent node is the parent run's, which the pin names by weights and run BOM.
		if record.Parent.WeightsSHA256 != parent.Artifact.SHA256 || !reflect.DeepEqual(record.Parent.Runs, []string{parent.RunBOMSHA256}) {
			return nil, nil, fmt.Errorf("%s names weights %s from runs %v, not the parent pin's %s from run %s", LineageFile, record.Parent.WeightsSHA256, record.Parent.Runs, parent.Artifact.SHA256, parent.RunBOMSHA256)
		}
		link, err := holdChain(held, record.Parent, record.Ancestors)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", LineageFile, err)
		}
		return &link, held, nil
	}
	if inspection.Origin == nil {
		return nil, held, nil
	}
	release, ancestors, err := keptReleaseLineage(inspection)
	if err != nil {
		return nil, nil, err
	}
	node, err := originNode(*inspection.Origin, release, held, ancestors)
	if err != nil {
		return nil, nil, err
	}
	link, err := hold(held, node)
	return &link, held, err
}

// keptReleaseLineage is the lineage in a pulled origin's kept release BOM.json,
// read once and checked against the hash its origin pinned. A model pulled
// before releases were kept, or whose kept file is gone, is a root.
func keptReleaseLineage(inspection Inspection) (*LineageNode, []LineageNode, error) {
	for _, pinned := range inspection.Origin.SourceArtifacts {
		if pinned.Path != "BOM.json" {
			continue
		}
		path := filepath.Join(inspection.Path, filepath.FromSlash(ReleaseBOMPath))
		info, err := os.Lstat(path)
		if os.IsNotExist(err) {
			return nil, nil, nil
		} else if err != nil {
			return nil, nil, err
		}
		if !info.Mode().IsRegular() {
			return nil, nil, fmt.Errorf("kept release BOM.json is not a regular file")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, nil, err
		}
		digest := sha256.Sum256(data)
		if hex.EncodeToString(digest[:]) != pinned.SHA256 || int64(len(data)) != pinned.Bytes {
			return nil, nil, fmt.Errorf("kept release BOM.json does not match the hash its origin pinned")
		}
		node, ancestors, _, err := releaseLineage(data, inspection.Origin.SourceArtifacts)
		return node, ancestors, err
	}
	return nil, nil, nil
}

// originNode is a pulled origin's node: a root naming its release, whose
// weights descend from the node the release declared, if any.
func originNode(origin OriginBOM, release *LineageNode, held map[string]LineageNode, ancestors []LineageNode) (LineageNode, error) {
	weights := ""
	for _, artifact := range origin.Artifacts {
		if artifact.Role == "weights" {
			weights = artifact.SHA256
		}
	}
	if weights == "" {
		return LineageNode{}, fmt.Errorf("model origin has no weights artifact")
	}
	// The release is its repository, resolved revision, license, and files,
	// not the revision as the pull named it or the mirror that served it.
	digest, err := hashJSON(struct {
		Provider   string           `json:"provider"`
		Repository string           `json:"repository"`
		Revision   string           `json:"revision"`
		License    string           `json:"license,omitempty"`
		Artifacts  []OriginArtifact `json:"artifacts"`
	}{origin.Source.Provider, origin.Source.Repository, origin.Source.Revision, origin.Source.License, origin.SourceArtifacts})
	if err != nil {
		return LineageNode{}, err
	}
	node := LineageNode{WeightsSHA256: weights, SourceSHA256: digest}
	if release != nil {
		link, err := holdChain(held, *release, ancestors)
		if err != nil {
			return LineageNode{}, fmt.Errorf("release lineage: %w", err)
		}
		node.Parents = []LineageLink{link}
	}
	return node, nil
}

// holdChain holds node and its ancestors, which must be exactly the chain
// behind node, and returns the link to node.
func holdChain(held map[string]LineageNode, node LineageNode, ancestors []LineageNode) (LineageLink, error) {
	pool := map[string]LineageNode{}
	for _, ancestor := range ancestors {
		if _, err := hold(pool, ancestor); err != nil {
			return LineageLink{}, err
		}
	}
	if err := node.validate(); err != nil {
		return LineageLink{}, err
	}
	if reached := chain(node, pool); len(reached) != len(ancestors) {
		return LineageLink{}, fmt.Errorf("%d ancestors, of which the chain reaches %d", len(ancestors), len(reached))
	}
	for digest, ancestor := range pool {
		held[digest] = ancestor
	}
	return hold(held, node)
}

func hold(held map[string]LineageNode, node LineageNode) (LineageLink, error) {
	if err := node.validate(); err != nil {
		return LineageLink{}, err
	}
	digest, err := hashJSON(node)
	if err != nil {
		return LineageLink{}, err
	}
	held[digest] = node
	return LineageLink{WeightsSHA256: node.WeightsSHA256, LineageSHA256: digest}, nil
}

// chain is every node behind node that held reaches, nearest first, each once.
func chain(node LineageNode, held map[string]LineageNode) []LineageNode {
	var reached []LineageNode
	seen := map[string]bool{}
	queue := append([]LineageLink(nil), node.Parents...)
	for len(queue) > 0 {
		link := queue[0]
		queue = queue[1:]
		parent, ok := held[link.LineageSHA256]
		if link.LineageSHA256 == "" || !ok || seen[link.LineageSHA256] {
			continue
		}
		seen[link.LineageSHA256] = true
		reached = append(reached, parent)
		queue = append(queue, parent.Parents...)
	}
	return reached
}

// parentLineage is the LINEAGE.json of a model trained from base's run runID.
func parentLineage(base Inspection, runID, weights string) (*lineageRecord, error) {
	node, ancestors, err := LineageHistory(base, runID)
	if err != nil {
		return nil, err
	}
	if node.WeightsSHA256 != weights {
		return nil, fmt.Errorf("base model %q's lineage names weights %s, not its run's %s", base.Model.Name, node.WeightsSHA256, weights)
	}
	return &lineageRecord{Kind: lineageKind, Schema: 1, Parent: node, Ancestors: ancestors}, nil
}

// readLineageRecord reads LINEAGE.json; nil when it is absent or a later
// schema, either of which leaves a shorter history rather than a different one.
func readLineageRecord(path string) (*lineageRecord, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	var header struct {
		Kind   string `json:"kind"`
		Schema int    `json:"schema"`
	}
	if err := json.Unmarshal(data, &header); err != nil || header.Kind != lineageKind || header.Schema < 1 {
		return nil, fmt.Errorf("%s is not a %s record", path, lineageKind)
	}
	if header.Schema > 1 {
		return nil, nil
	}
	var record lineageRecord
	if err := readStrictJSON(path, &record); err != nil {
		return nil, err
	}
	return &record, nil
}

// releaseLineage is the lineage node, and the chain behind it, that a WALDO
// release's BOM.json declares for the weights files pulled with it. Lineage
// that is malformed, or read differently by other readers, is refused. Lineage
// a later schema wrote, or that names other weights files than these, is left
// out with a note, so the origin is a root.
func releaseLineage(data []byte, files []OriginArtifact) (*LineageNode, []LineageNode, string, error) {
	release, err := topLevelFields(data)
	if err != nil {
		if bytes.Contains(bytes.ToLower(data), []byte("lineage")) {
			return nil, nil, "", fmt.Errorf("release BOM.json lineage: %w", err)
		}
		return nil, nil, "", nil
	}
	declares := false
	for key := range release {
		for _, name := range []string{"lineage", "lineage_ancestors"} {
			declares = declares || strings.EqualFold(key, name)
		}
	}
	if !declares {
		return nil, nil, "", nil
	}
	for key := range release {
		for _, name := range []string{"kind", "schema", "subject", "artifacts", "lineage", "lineage_ancestors"} {
			if key != name && strings.EqualFold(key, name) {
				return nil, nil, "", fmt.Errorf("release BOM.json lineage: unexpected key %q", key)
			}
		}
	}
	var kind, subject string
	if json.Unmarshal(release["kind"], &kind) != nil || json.Unmarshal(release["subject"], &subject) != nil || kind != "openwaldo-bom" || subject != "model-release" {
		return nil, nil, "", fmt.Errorf("release BOM.json lineage: not in an OpenWALDO model-release BOM")
	}
	var schema int
	if json.Unmarshal(release["schema"], &schema) != nil || schema < 1 {
		return nil, nil, "", fmt.Errorf("release BOM.json lineage: unsupported schema %s", release["schema"])
	}
	if schema > 1 {
		return nil, nil, fmt.Sprintf("release BOM.json is schema %d; its lineage is left out", schema), nil
	}
	raw := release["lineage"]
	if raw == nil || string(raw) == "null" {
		return nil, nil, "", fmt.Errorf("release BOM.json lineage: ancestors without a node")
	}
	var lineage LineageNode
	if err := exactLineage(raw, &lineage); err != nil {
		return nil, nil, "", err
	}
	var ancestors []LineageNode
	if extra := release["lineage_ancestors"]; extra != nil {
		if err := exactLineage(extra, &ancestors); err != nil {
			return nil, nil, "", err
		}
	}
	if _, err := holdChain(map[string]LineageNode{}, lineage, ancestors); err != nil {
		return nil, nil, "", fmt.Errorf("release BOM.json lineage: %w", err)
	}
	if !inventoriesWeights(release["artifacts"], files) {
		return nil, nil, "release BOM.json does not inventory the weights pulled with it; its lineage is left out", nil
	}
	return &lineage, ancestors, "", nil
}

// inventoriesWeights reports whether a release's artifacts list every pulled
// weights file with its hash and size, so its lineage is these weights'.
func inventoriesWeights(raw json.RawMessage, files []OriginArtifact) bool {
	var listed []OriginArtifact
	if json.Unmarshal(raw, &listed) != nil {
		return false
	}
	inventory := map[string]OriginArtifact{}
	for _, artifact := range listed {
		inventory[artifact.Path] = artifact
	}
	weights := 0
	for _, file := range files {
		if file.Role != "source-weights" {
			continue
		}
		weights++
		if listed, ok := inventory[file.Path]; !ok || listed.SHA256 != file.SHA256 || listed.Bytes != file.Bytes {
			return false
		}
	}
	return weights > 0
}

// topLevelFields decodes a JSON object's members, refusing repeated keys and
// trailing data, which other readers could read differently.
func topLevelFields(data []byte) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil, fmt.Errorf("not a JSON object")
	}
	fields := map[string]json.RawMessage{}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		key := token.(string)
		if _, repeated := fields[key]; repeated {
			return nil, fmt.Errorf("repeated key %q", key)
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, err
		}
		fields[key] = value
	}
	if _, err := decoder.Token(); err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, fmt.Errorf("data after the JSON object")
	}
	return fields, nil
}

// exactLineage decodes raw into target, which must be exactly what WALDO
// writes for it, so the bytes a release publishes say what a pull reads.
func exactLineage(raw json.RawMessage, target any) error {
	if err := json.Unmarshal(raw, target); err != nil {
		return fmt.Errorf("release BOM.json lineage: %w", err)
	}
	canonical, err := json.Marshal(target)
	if err != nil {
		return err
	}
	var published bytes.Buffer
	if err := json.Compact(&published, raw); err != nil || !bytes.Equal(published.Bytes(), canonical) {
		return fmt.Errorf("release BOM.json lineage is not lineage in WALDO's form")
	}
	return nil
}
