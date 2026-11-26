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
	"strings"
)

// LINEAGE.json holds the documents behind a model by SHA-256 (ADR 0084).
const (
	lineageFile   = "LINEAGE.json"
	lineageKind   = "waldo-model-lineage"
	lineageSchema = 1
)

type LineageFile struct {
	Kind      string              `json:"kind"`
	Schema    int                 `json:"schema"`
	Documents map[string][]byte   `json:"documents"`
	Anchors   map[string][]string `json:"anchors,omitempty"`
	Gaps      []LineageGap        `json:"gaps,omitempty"`
}

type LineageGap struct {
	Gap  string `json:"gap"`
	Link string `json:"link"`
}

func (gap LineageGap) String() string {
	return gap.Link + " " + gap.Gap[:min(12, len(gap.Gap))]
}

type lineageIndex struct {
	root   string
	docs   map[string][]byte
	claims map[string][]string // MODEL-BOMs by the model ID they describe
	loaded map[string]bool
	all    bool
	warn   func(string)
}

func newLineageIndex(root string, warn func(string)) *lineageIndex {
	if warn == nil {
		warn = func(string) {}
	}
	return &lineageIndex{root: root, docs: map[string][]byte{}, claims: map[string][]string{}, loaded: map[string]bool{}, warn: warn}
}

func (index *lineageIndex) addRecord(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	digest := sha256Hex(data)
	index.docs[digest] = data
	return digest, nil
}

func (index *lineageIndex) addFile(file LineageFile) {
	for digest, data := range file.Documents {
		if sha256Hex(data) == digest {
			index.docs[digest] = data
		}
	}
	for _, id := range slices.Sorted(maps.Keys(file.Anchors)) {
		index.claims[id] = append(index.claims[id], file.Anchors[id]...)
	}
}

func (index *lineageIndex) loadModel(directory string, start bool) {
	if index.loaded[directory] {
		return
	}
	index.loaded[directory] = true
	var plan Plan
	if readJSON(filepath.Join(directory, "PLAN.json"), &plan) == nil {
		_, _ = index.addRecord(plan)
	}
	var origin OriginBOM
	if readJSON(filepath.Join(directory, "ORIGIN-BOM.json"), &origin) == nil {
		_, _ = index.addRecord(origin)
	}
	var record ModelRecord
	if readJSON(filepath.Join(directory, "MODEL.json"), &record) != nil || record.ID == "" {
		return
	}
	for _, pin := range record.Runs {
		var bom RunBOM
		if readJSON(filepath.Join(directory, "runs", runDirectoryName(pin), "RUN-BOM.json"), &bom) == nil {
			_, _ = index.addRecord(bom)
		}
	}
	file, err := readLineageFile(filepath.Join(directory, lineageFile))
	if err != nil && start {
		index.warn(err.Error())
	}
	if file == nil {
		file = &LineageFile{}
	}
	index.addFile(*file)
	if digest, err := index.addRecord(modelBOM(record)); err == nil {
		index.claims[record.ID] = append(index.claims[record.ID], digest)
	}
}

func (index *lineageIndex) loadAll() {
	if index.all || index.root == "" {
		return
	}
	index.all = true
	entries, err := os.ReadDir(index.root)
	if err != nil {
		return
	}
	for _, entry := range entries {
		if entry.IsDir() && !strings.HasPrefix(entry.Name(), ".") {
			index.loadModel(filepath.Join(index.root, entry.Name()), false)
		}
	}
}

func (index *lineageIndex) document(digest string) []byte {
	if data, ok := index.docs[digest]; ok {
		return data
	}
	index.loadAll()
	return index.docs[digest]
}

func readLineageFile(path string) (*LineageFile, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	return parseLineageFile(path, data)
}

func parseLineageFile(name string, data []byte) (*LineageFile, error) {
	var file LineageFile
	if err := json.Unmarshal(data, &file); err != nil || file.Kind != lineageKind {
		return nil, fmt.Errorf("%s can't be read; its history shows as gaps", name)
	}
	if file.Schema != lineageSchema {
		return nil, fmt.Errorf("%s is schema %d, which this WALDO can't read; its history shows as gaps", name, file.Schema)
	}
	return &file, nil
}

type lineageWalk struct {
	index   *lineageIndex
	held    map[string][]byte
	anchors map[string][]string
	seen    map[string]bool
	gaps    map[LineageGap]bool
}

func newLineageWalk(index *lineageIndex) *lineageWalk {
	return &lineageWalk{index: index, held: map[string][]byte{}, anchors: map[string][]string{}, seen: map[string]bool{}, gaps: map[LineageGap]bool{}}
}

func (walk *lineageWalk) gap(digest, link string) {
	walk.gaps[LineageGap{Gap: digest, Link: link}] = true
}

func (walk *lineageWalk) fetch(digest, link string, target any) bool {
	data := walk.index.document(digest)
	if data == nil || json.Unmarshal(data, target) != nil {
		walk.gap(digest, link)
		return false
	}
	walk.held[digest] = data
	return true
}

func (walk *lineageWalk) plan(modelID string) *Plan {
	var plan Plan
	if !walk.fetch(modelID, "plan", &plan) {
		return nil
	}
	return &plan
}

func (walk *lineageWalk) anchor(digest, modelID string) *ModelBOM {
	var bom ModelBOM
	if !walk.fetch(digest, "model-bom", &bom) {
		return nil
	}
	if bom.ModelID != modelID {
		walk.gap(digest, "model-bom")
		return nil
	}
	if !slices.Contains(walk.anchors[modelID], digest) {
		walk.anchors[modelID] = append(walk.anchors[modelID], digest)
	}
	return &bom
}

func (walk *lineageWalk) anchoredRun(modelID, runID string) (ModelBOMRun, string, bool) {
	for pass := 0; pass < 2; pass++ {
		digests := append(slices.Clone(walk.anchors[modelID]), walk.index.claims[modelID]...)
		for _, digest := range digests {
			var bom ModelBOM
			if data := walk.index.docs[digest]; data == nil || json.Unmarshal(data, &bom) != nil || bom.ModelID != modelID {
				continue
			}
			for _, run := range bom.Runs {
				if weights, ok := bomRunWeights(run); run.ID == runID && ok {
					walk.anchor(digest, modelID)
					return run, weights, true
				}
			}
		}
		walk.index.loadAll()
	}
	return ModelBOMRun{}, "", false
}

// model walks from one run's weights, or from where the model started.
func (walk *lineageWalk) model(inspection Inspection, runID string) {
	walk.index.loadModel(inspection.Path, true)
	plan := walk.plan(inspection.Model.ID)
	for _, run := range inspection.BOM.Runs {
		if runID != "" && run.ID == runID {
			walk.run(run.BOMSHA256, inspection.Model.ID, run.ID)
			return
		}
	}
	if plan == nil {
		return
	}
	if plan.Parent != nil {
		walk.parent(*plan.Parent)
	} else if plan.OriginBOMSHA256 != "" {
		weights := ""
		if inspection.Origin != nil {
			weights, _ = originWeights(inspection.Origin.Artifacts)
		}
		walk.origin(plan.OriginBOMSHA256, weights)
	}
}

func (walk *lineageWalk) parent(parent ModelParent) {
	walk.plan(parent.ModelID)
	if _, weights, ok := walk.anchoredRun(parent.ModelID, parent.RunID); ok && weights != parent.Artifact.SHA256 {
		walk.gap(parent.Artifact.SHA256, "initialization")
		return
	}
	walk.run(parent.RunBOMSHA256, parent.ModelID, parent.RunID)
}

// run walks from a run BOM that a pin names as modelID's runID.
func (walk *lineageWalk) run(digest, modelID, runID string) {
	var bom RunBOM
	if !walk.fetch(digest, "run-bom", &bom) {
		return
	}
	if bom.ModelID != modelID || bom.ID != runID {
		walk.gap(digest, "run-bom")
		return
	}
	if walk.seen[digest] || bom.Initialization == nil {
		return
	}
	walk.seen[digest] = true
	start := bom.Initialization
	from := start.Artifact.SHA256
	plan := walk.plan(bom.ModelID)
	switch {
	case plan == nil:
		walk.gap(from, "initialization")
	case start.SourceType == "origin" && start.SourceID == plan.OriginBOMSHA256:
		walk.origin(start.SourceID, from)
	case start.SourceType == "run" && start.SourceID == start.SourceRunID:
		earlier, weights, ok := walk.anchoredRun(bom.ModelID, start.SourceRunID)
		if !ok || weights != from {
			walk.gap(from, "initialization")
			return
		}
		walk.run(earlier.BOMSHA256, bom.ModelID, earlier.ID)
	case start.SourceType == "run" && plan.Parent != nil && start.SourceID == plan.Parent.ModelID && start.SourceRunID == plan.Parent.RunID && from == plan.Parent.Artifact.SHA256:
		walk.parent(*plan.Parent)
	default:
		walk.gap(from, "initialization")
	}
}

func (walk *lineageWalk) origin(digest, weights string) {
	if walk.seen[digest] {
		return
	}
	walk.seen[digest] = true
	var origin OriginBOM
	if !walk.fetch(digest, "origin-bom", &origin) {
		return
	}
	if pinned, _ := originWeights(origin.Artifacts); weights != "" && pinned != weights {
		walk.gap(digest, "origin-bom")
		return
	}
	for _, artifact := range origin.SourceArtifacts {
		if artifact.Role == "source-bom" {
			walk.release(artifact.SHA256, origin)
		}
	}
}

type releaseBOM struct {
	Kind      string `json:"kind"`
	Schema    int    `json:"schema"`
	Subject   string `json:"subject"`
	ModelID   string `json:"model_id"`
	RunID     string `json:"run_id"`
	SourceBOM string `json:"source_bom_sha256"`
	Artifacts []struct {
		Path   string `json:"path"`
		Role   string `json:"role"`
		SHA256 string `json:"sha256"`
	} `json:"artifacts"`
}

func (release releaseBOM) listed(role string) []string {
	var digests []string
	for _, artifact := range release.Artifacts {
		if artifact.Role == role {
			digests = append(digests, artifact.SHA256)
		}
	}
	return digests
}

// release links only when BOM.json lists the pulled weights.
func (walk *lineageWalk) release(digest string, origin OriginBOM) {
	var release releaseBOM
	if !walk.fetch(digest, "release-bom", &release) {
		return
	}
	var pulled []string
	for _, artifact := range origin.SourceArtifacts {
		if artifact.Role == "source-weights" {
			pulled = append(pulled, artifact.SHA256)
		}
	}
	if release.Kind != "openwaldo-bom" || release.Schema != 1 || release.Subject != "model-release" || len(pulled) != 1 || !slices.Equal(release.listed("weights"), pulled) {
		walk.gap(digest, "release-bom")
		return
	}
	plan := walk.plan(release.ModelID)
	anchor := walk.anchor(release.SourceBOM, release.ModelID)
	if anchor == nil {
		return
	}
	for _, run := range anchor.Runs {
		if release.RunID != "" && run.ID == release.RunID {
			walk.run(run.BOMSHA256, release.ModelID, run.ID)
			return
		}
	}
	if release.RunID == "" && plan != nil && anchor.Origin != nil && anchor.Origin.SHA256 == plan.OriginBOMSHA256 {
		walk.origin(anchor.Origin.SHA256, "")
		return
	}
	walk.gap(release.SourceBOM, "model-bom")
}

func bomRunWeights(run ModelBOMRun) (string, bool) {
	weights := ""
	for _, artifact := range run.Artifacts {
		if artifact.Role == "weights" {
			if weights != "" {
				return "", false
			}
			weights = artifact.SHA256
		}
	}
	return weights, weights != ""
}

func originWeights(artifacts []OriginArtifact) (string, bool) {
	weights := ""
	for _, artifact := range artifacts {
		if artifact.Role == "weights" {
			if weights != "" {
				return "", false
			}
			weights = artifact.SHA256
		}
	}
	return weights, weights != ""
}

func (walk *lineageWalk) sortedGaps() []LineageGap {
	gaps := make([]LineageGap, 0, len(walk.gaps))
	for gap := range walk.gaps {
		gaps = append(gaps, gap)
	}
	sort.Slice(gaps, func(i, j int) bool {
		if gaps[i].Link != gaps[j].Link {
			return gaps[i].Link < gaps[j].Link
		}
		return gaps[i].Gap < gaps[j].Gap
	})
	return gaps
}

func (walk *lineageWalk) file() LineageFile {
	return LineageFile{Kind: lineageKind, Schema: lineageSchema, Documents: walk.held, Anchors: walk.anchors, Gaps: walk.sortedGaps()}
}

func (walk *lineageWalk) warnGaps(name string) {
	gaps := walk.sortedGaps()
	if len(gaps) == 0 {
		return
	}
	names := make([]string, len(gaps))
	for index, gap := range gaps {
		names[index] = gap.String()
	}
	walk.index.warn(fmt.Sprintf("model %q history has %d gap(s): %s", name, len(gaps), strings.Join(names, ", ")))
}

func writeHistory(index *lineageIndex, inspection Inspection, path string) error {
	walk := newLineageWalk(index)
	walk.model(inspection, "")
	walk.warnGaps(inspection.Model.Name)
	return writeJSONAtomic(path, walk.file())
}

// fillLineage writes LINEAGE.json once, as a model is made or its compose resumes.
func fillLineage(root string, inspection Inspection, base *Inspection, warn func(string)) error {
	if inspection.Plan.Parent == nil && inspection.Plan.OriginBOMSHA256 == "" {
		return nil
	}
	path := filepath.Join(inspection.Path, lineageFile)
	if _, err := os.Lstat(path); !os.IsNotExist(err) {
		return err
	}
	index := newLineageIndex(root, warn)
	if base != nil {
		index.loadModel(base.Path, true)
	}
	return writeHistory(index, inspection, path)
}

// ReleaseLineage is the LINEAGE.json a release ships.
func ReleaseLineage(root string, inspection Inspection, warn func(string)) ([]byte, error) {
	index := newLineageIndex(root, warn)
	walk := newLineageWalk(index)
	walk.model(inspection, inspection.BOM.CurrentRunID)
	digest, err := index.addRecord(inspection.BOM)
	if err != nil {
		return nil, err
	}
	walk.anchor(digest, inspection.Model.ID)
	walk.warnGaps(inspection.Model.Name)
	return json.Marshal(walk.file())
}

func listedLineage(data []byte) string {
	var release releaseBOM
	if json.Unmarshal(data, &release) != nil {
		return ""
	}
	for _, artifact := range release.Artifacts {
		if artifact.Path == lineageFile {
			return artifact.SHA256
		}
	}
	return ""
}

// pullLineage never fails a pull; what doesn't check out is a gap.
func pullLineage(root, directory string, origin OriginBOM, plan Plan, warn func(string)) (LineageFile, error) {
	index := newLineageIndex(root, warn)
	originDigest, err := index.addRecord(origin)
	if err != nil {
		return LineageFile{}, err
	}
	modelID, err := index.addRecord(plan)
	if err != nil {
		return LineageFile{}, err
	}
	if release, err := os.ReadFile(filepath.Join(directory, "BOM.json")); err == nil {
		index.docs[sha256Hex(release)] = release
		if listed := listedLineage(release); listed != "" {
			data, err := os.ReadFile(filepath.Join(directory, lineageFile))
			if err != nil || sha256Hex(data) != listed {
				warn("release LINEAGE.json wasn't fetched or doesn't match BOM.json; its history shows as gaps")
			} else if file, err := parseLineageFile("release LINEAGE.json", data); err != nil {
				warn(err.Error())
			} else {
				index.addFile(*file)
			}
		}
	}
	walk := newLineageWalk(index)
	walk.plan(modelID)
	weights, _ := originWeights(origin.Artifacts)
	walk.origin(originDigest, weights)
	walk.warnGaps(plan.Name)
	return walk.file(), nil
}
