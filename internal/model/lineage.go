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

// ModelLineage is the node of a model's weights after run runID, or its newest real run's, or its origin's.
func ModelLineage(inspection Inspection, runID string) (LineageNode, error) {
	node, _, err := modelLineage(inspection, runID)
	return node, err
}

// LineageHistory is ModelLineage's node and every node behind it the model holds, nearest first.
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
		return LineageNode{}, nil, fmt.Errorf("model %q lineage: %w", inspection.Model.Name, err)
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

// runLineage is a node for each complete real run, linked to the node it started from, held in held.
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

// startingLineage is the link a model's first run starts from and the nodes held behind it.
func startingLineage(inspection Inspection) (*LineageLink, map[string]LineageNode, error) {
	held := map[string]LineageNode{}
	if parent := inspection.Model.Parent; parent != nil {
		link := &LineageLink{WeightsSHA256: parent.Artifact.SHA256, LineageSHA256: parent.LineageSHA256}
		// A pin from before lineage names no history, so no LINEAGE.json can add one.
		if parent.LineageSHA256 == "" {
			return link, held, nil
		}
		record, err := readLineageRecord(filepath.Join(inspection.Path, LineageFile))
		if err != nil {
			return nil, nil, err
		}
		// Without the file, the pinned hash still names the parent's history; it just isn't held.
		if record == nil {
			return link, held, nil
		}
		digest, err := hashJSON(record.Parent)
		if err != nil {
			return nil, nil, err
		}
		if digest != parent.LineageSHA256 {
			return nil, nil, fmt.Errorf("%s parent node %s is not the one the parent pin names, %s", LineageFile, digest, parent.LineageSHA256)
		}
		start, err := holdChain(held, record.Parent, record.Ancestors)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", LineageFile, err)
		}
		return &start, held, nil
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

// keptReleaseLineage is the lineage in a pulled origin's kept release BOM.json.
func keptReleaseLineage(inspection Inspection) (*LineageNode, []LineageNode, error) {
	data, err := keptReleaseBOM(inspection.Path, *inspection.Origin)
	if err != nil || data == nil {
		return nil, nil, err
	}
	node, ancestors, _ := releaseLineage(data, inspection.Origin.SourceArtifacts)
	return node, ancestors, nil
}

// keptReleaseBOM reads the kept release BOM.json once and checks it against origin's pin; nil if none.
func keptReleaseBOM(directory string, origin OriginBOM) ([]byte, error) {
	for _, pinned := range origin.SourceArtifacts {
		if pinned.Path != "BOM.json" {
			continue
		}
		data, err := readRegularFile(filepath.Join(directory, filepath.FromSlash(ReleaseBOMPath)))
		if err != nil || data == nil {
			return nil, err
		}
		digest := sha256.Sum256(data)
		if hex.EncodeToString(digest[:]) != pinned.SHA256 {
			return nil, fmt.Errorf("kept release BOM.json does not match the hash its origin pinned")
		}
		return data, nil
	}
	return nil, nil
}

// readRegularFile reads path once, refusing anything but a regular file; nil when absent.
func readRegularFile(path string) ([]byte, error) {
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("%s is not a regular file", path)
	}
	return os.ReadFile(path)
}

// originNode is a pulled origin's node, naming its release, with the release's node as parent if kept.
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

// holdChain holds node and ancestors, which must be exactly the chain behind it, and links to node.
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

// chain is every held node behind node whose link names its weights, nearest first, each once.
func chain(node LineageNode, held map[string]LineageNode) []LineageNode {
	var reached []LineageNode
	seen := map[string]bool{}
	queue := append([]LineageLink(nil), node.Parents...)
	for len(queue) > 0 {
		link := queue[0]
		queue = queue[1:]
		parent, ok := held[link.LineageSHA256]
		if link.LineageSHA256 == "" || !ok || seen[link.LineageSHA256] || parent.WeightsSHA256 != link.WeightsSHA256 {
			continue
		}
		seen[link.LineageSHA256] = true
		reached = append(reached, parent)
		queue = append(queue, parent.Parents...)
	}
	return reached
}

// baseLineage is the LINEAGE.json of a model trained from base's run runID, whose
// parent node must hash to asserted when set.
func baseLineage(base Inspection, runID, asserted string) (*lineageRecord, error) {
	node, ancestors, err := LineageHistory(base, runID)
	if err != nil {
		return nil, err
	}
	if asserted != "" {
		digest, err := hashJSON(node)
		if err != nil {
			return nil, err
		}
		if digest != asserted {
			return nil, fmt.Errorf("base model %q's lineage is %s, not requested %s", base.Model.Name, digest, asserted)
		}
	}
	return &lineageRecord{Kind: lineageKind, Schema: 1, Parent: node, Ancestors: ancestors}, nil
}

// readLineageRecord reads LINEAGE.json once, exactly as any reader would; nil
// when absent or a later schema, which leaves a shorter history.
func readLineageRecord(path string) (*lineageRecord, error) {
	data, err := readRegularFile(path)
	if err != nil || data == nil {
		return nil, err
	}
	fields, err := topLevelFields(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	for key := range fields {
		if (key != "kind" && strings.EqualFold(key, "kind")) || (key != "schema" && strings.EqualFold(key, "schema")) {
			return nil, fmt.Errorf("%s: unexpected key %q", path, key)
		}
	}
	var record lineageRecord
	if json.Unmarshal(fields["kind"], &record.Kind) != nil || record.Kind != lineageKind || json.Unmarshal(fields["schema"], &record.Schema) != nil || record.Schema < 1 {
		return nil, fmt.Errorf("%s is not a %s record", path, lineageKind)
	}
	if record.Schema > 1 {
		return nil, nil
	}
	for key := range fields {
		if key != "kind" && key != "schema" && key != "parent" && key != "ancestors" {
			return nil, fmt.Errorf("%s: unexpected key %q", path, key)
		}
	}
	if err := exactJSON(fields["parent"], &record.Parent); err != nil {
		return nil, fmt.Errorf("%s: parent %w", path, err)
	}
	if ancestors := fields["ancestors"]; ancestors != nil {
		if err := exactJSON(ancestors, &record.Ancestors); err != nil {
			return nil, fmt.Errorf("%s: ancestors %w", path, err)
		}
	}
	return &record, nil
}

// releaseLineage is the lineage a release BOM.json declares for the pulled weights, if every reader
// reads it alike and it inventories them; otherwise nil with a note saying why, never an error.
func releaseLineage(data []byte, files []OriginArtifact) (*LineageNode, []LineageNode, string) {
	leftOut := func(reason string) (*LineageNode, []LineageNode, string) {
		return nil, nil, "release BOM.json " + reason + "; its lineage is left out"
	}
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
	release, err := topLevelFields(data)
	if err != nil {
		// Other readers may still find lineage in it, as its first value or in its text.
		var first map[string]json.RawMessage
		_ = json.NewDecoder(bytes.NewReader(data)).Decode(&first)
		// UTF-16 and UTF-32 hide the word from a byte search but not from every reader.
		if declaresLineage(first) || bytes.Contains(bytes.ToLower(data), []byte("lineage")) || bytes.IndexByte(data, 0) >= 0 {
			return leftOut(fmt.Sprintf("is not one JSON object (%v)", err))
		}
		return nil, nil, ""
	}
	if !declaresLineage(release) {
		return nil, nil, ""
	}
	for key := range release {
		for _, name := range []string{"kind", "schema", "subject", "artifacts", "lineage", "lineage_ancestors"} {
			if key != name && strings.EqualFold(key, name) {
				return leftOut(fmt.Sprintf("has a key %q other readers may read as %q", key, name))
			}
		}
	}
	var kind, subject string
	if json.Unmarshal(release["kind"], &kind) != nil || json.Unmarshal(release["subject"], &subject) != nil || kind != "openwaldo-bom" || subject != "model-release" {
		return leftOut("is not an OpenWALDO model-release BOM")
	}
	var schema int
	if json.Unmarshal(release["schema"], &schema) != nil || schema != 1 {
		return leftOut(fmt.Sprintf("is schema %s, not 1", release["schema"]))
	}
	var lineage LineageNode
	if err := exactJSON(release["lineage"], &lineage); err != nil {
		return leftOut("lineage is not a node in WALDO's form")
	}
	var ancestors []LineageNode
	if extra := release["lineage_ancestors"]; extra != nil {
		if err := exactJSON(extra, &ancestors); err != nil {
			return leftOut("lineage_ancestors are not nodes in WALDO's form")
		}
	}
	if _, err := holdChain(map[string]LineageNode{}, lineage, ancestors); err != nil {
		return leftOut(fmt.Sprintf("lineage is not one chain (%v)", err))
	}
	if !inventoriesWeights(release["artifacts"], files) {
		return leftOut("does not list the pulled weights files once each, in WALDO's form")
	}
	return &lineage, ancestors, ""
}

func declaresLineage(fields map[string]json.RawMessage) bool {
	for key := range fields {
		if strings.EqualFold(key, "lineage") || strings.EqualFold(key, "lineage_ancestors") {
			return true
		}
	}
	return false
}

// inventoriesWeights reports whether artifacts list every pulled weights file
// once, by hash and size, in entries every reader reads alike.
func inventoriesWeights(raw json.RawMessage, files []OriginArtifact) bool {
	var entries []json.RawMessage
	if json.Unmarshal(raw, &entries) != nil {
		return false
	}
	weights := map[string]OriginArtifact{}
	for _, file := range files {
		if file.Role == "source-weights" {
			weights[file.Path] = file
		}
	}
	listed := map[string]bool{}
	for _, entry := range entries {
		fields, err := topLevelFields(entry)
		if err != nil {
			return false
		}
		for key := range fields {
			if key != "path" && strings.EqualFold(key, "path") {
				return false
			}
		}
		var path string
		if json.Unmarshal(fields["path"], &path) != nil {
			continue
		}
		file, ok := weights[path]
		if !ok {
			continue
		}
		var artifact OriginArtifact
		if listed[path] || exactJSON(entry, &artifact) != nil || artifact.SHA256 != file.SHA256 || artifact.Bytes != file.Bytes {
			return false
		}
		listed[path] = true
	}
	return len(listed) == len(weights)
}

// topLevelFields decodes one JSON object's members, refusing repeated keys and trailing data.
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

// exactJSON decodes raw into target, which must re-encode to the same bytes, so all readers agree.
func exactJSON(raw json.RawMessage, target any) error {
	if raw == nil {
		return fmt.Errorf("missing")
	}
	if err := json.Unmarshal(raw, target); err != nil {
		return err
	}
	canonical, err := json.Marshal(target)
	if err != nil {
		return err
	}
	var published bytes.Buffer
	if err := json.Compact(&published, raw); err != nil || !bytes.Equal(published.Bytes(), canonical) {
		return fmt.Errorf("not in WALDO's form")
	}
	return nil
}
