// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
)

// CheckpointIdentity is supplied from the immutable run BOM, never by a trainer.
// RunBOMSHA256 pins stage, corpus BOMs, evaluation selection and execution facts.
type CheckpointIdentity struct {
	RunBOMSHA256      string `json:"run_bom_sha256"`
	TrainerSHA256     string `json:"trainer_sha256"`
	TokenizerSHA256   string `json:"tokenizer_sha256"`
	EnvironmentSHA256 string `json:"environment_sha256"`
}

// CheckpointReference must be persisted by the lifecycle independently of the
// trainer-owned files. A manifest cannot authenticate its own digest.
type CheckpointReference struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}
type CheckpointManifest struct {
	Kind             string             `json:"kind"`
	Schema           int                `json:"schema"`
	Identity         CheckpointIdentity `json:"identity"`
	BeginSHA256      string             `json:"begin_sha256"`
	ParametersSHA256 string             `json:"parameters_sha256"`
	Execution        map[string]any     `json:"execution"`
	World            applaunch.World    `json:"world"`
	Step             int64              `json:"step"`
	Tokens           int64              `json:"tokens"`
	Consumption      map[string]int64   `json:"consumption"`
	Files            []Artifact         `json:"files"`
}

func validHash(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size && hex.EncodeToString(b) == s
}
func (i CheckpointIdentity) validate() error {
	for _, h := range []string{i.RunBOMSHA256, i.TrainerSHA256, i.TokenizerSHA256, i.EnvironmentSHA256} {
		if !validHash(h) {
			return fmt.Errorf("checkpoint identity requires run BOM, trainer, tokenizer and environment SHA-256 pins")
		}
	}
	return nil
}
func beginDigest(b Begin) string {
	b.Paths = Paths{}
	b.World.NodeRank = 0
	if b.World.WorldSize > 1 && b.Initialization != nil {
		initialization := *b.Initialization
		initialization.Path = ""
		b.Initialization = &initialization
	}
	b.Resume = nil
	return digest(b)
}
func syncDir(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}

// copyVerified uses a rooted open to prevent symlink traversal outside the
// checkpoint tree, and copies the bytes that were actually hashed.
func copyVerified(root *os.Root, a Artifact, destination string) error {
	f, err := root.Open(a.Path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() != a.Bytes {
		return fmt.Errorf("checkpoint file %s is not regular or size differs", a.Path)
	}
	var out *os.File
	var writer io.Writer = io.Discard
	if destination != "" {
		if err := os.MkdirAll(filepath.Dir(destination), 0700); err != nil {
			return err
		}
		out, err = os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			return err
		}
		defer out.Close()
		writer = out
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(h, writer), io.LimitReader(f, a.Bytes+1))
	if err != nil {
		return err
	}
	if n != a.Bytes || hex.EncodeToString(h.Sum(nil)) != a.SHA256 {
		return fmt.Errorf("checkpoint file %s hash/size mismatch", a.Path)
	}
	if out != nil {
		return out.Sync()
	}
	return nil
}
func (c *Client) commitCheckpoint(b Begin, checkpoint Checkpoint, consumption map[string]int64) (CheckpointReference, error) {
	var ref CheckpointReference
	if err := validateArtifacts(checkpoint.Artifacts); err != nil {
		return ref, err
	}
	total := int64(0)
	for _, n := range consumption {
		total += n
	}
	if total != checkpoint.Tokens {
		return ref, fmt.Errorf("checkpoint consumption differs from WALDO stream at step %d", checkpoint.Step)
	}
	_, execution, err := c.parameters.Split(b.Parameters)
	if err != nil {
		return ref, err
	}
	m := CheckpointManifest{Kind: "waldo-trainer-checkpoint", Schema: 1, Identity: b.CheckpointIdentity, BeginSHA256: beginDigest(b), ParametersSHA256: b.ParametersSHA256, Execution: execution, World: b.World, Step: checkpoint.Step, Tokens: total, Consumption: consumption, Files: checkpoint.Artifacts}
	root, err := os.OpenRoot(b.Paths.Checkpoints)
	if err != nil {
		return ref, err
	}
	defer root.Close()
	if err := os.MkdirAll(b.CheckpointDirectory, 0700); err != nil {
		return ref, err
	}
	temp, err := os.MkdirTemp(b.CheckpointDirectory, ".pending-")
	if err != nil {
		return ref, err
	}
	defer os.RemoveAll(temp)
	for _, a := range m.Files {
		if a.Path == "MANIFEST.json" {
			return ref, fmt.Errorf("reserved checkpoint filename")
		}
		if err := copyVerified(root, a, filepath.Join(temp, a.Path)); err != nil {
			return ref, err
		}
	}
	raw, err := json.Marshal(m)
	if err != nil {
		return ref, err
	}
	f, err := os.OpenFile(filepath.Join(temp, "MANIFEST.json"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return ref, err
	}
	_, err = f.Write(raw)
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return ref, err
	}
	if closeErr != nil {
		return ref, closeErr
	}
	// Sync every containing directory before publishing the whole bundle.
	err = filepath.WalkDir(temp, func(path string, d os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if d.IsDir() {
			return syncDir(path)
		}
		return nil
	})
	if err != nil {
		return ref, err
	}
	name := fmt.Sprintf("step-%020d-%s", m.Step, application.Digest(raw))
	dest := filepath.Join(b.CheckpointDirectory, name)
	if err := os.Rename(temp, dest); err != nil {
		ref = CheckpointReference{Path: filepath.Join(name, "MANIFEST.json"), SHA256: application.Digest(raw)}
		if _, verifyErr := LoadCheckpoint(b, ref); verifyErr != nil {
			return ref, err
		}
		return ref, nil
	}
	if err := syncDir(b.CheckpointDirectory); err != nil {
		return ref, err
	}
	return CheckpointReference{Path: filepath.Join(name, "MANIFEST.json"), SHA256: application.Digest(raw)}, nil
}

// LoadCheckpoint verifies the independently pinned manifest and every file,
// checks exact run compatibility, and returns the only accepted Resume value.
// Train repeats this verification immediately before launching the trainer.
func LoadCheckpoint(b Begin, ref CheckpointReference) (*Resume, error) {
	if !filepath.IsLocal(ref.Path) || filepath.Clean(ref.Path) != ref.Path || !validHash(ref.SHA256) {
		return nil, fmt.Errorf("invalid checkpoint reference")
	}
	root, err := os.OpenRoot(b.CheckpointDirectory)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	f, err := root.Open(ref.Path)
	if err != nil {
		return nil, err
	}
	raw, err := io.ReadAll(io.LimitReader(f, 16<<20))
	f.Close()
	if err != nil {
		return nil, err
	}
	if application.Digest(raw) != ref.SHA256 {
		return nil, fmt.Errorf("checkpoint manifest hash mismatch")
	}
	var m CheckpointManifest
	if err := strict(raw, &m); err != nil {
		return nil, err
	}
	if m.Kind != "waldo-trainer-checkpoint" || m.Schema != 1 || m.Identity != b.CheckpointIdentity || m.BeginSHA256 != beginDigest(b) || m.ParametersSHA256 != b.ParametersSHA256 || !sameCheckpointWorld(m.World, b.World) {
		return nil, fmt.Errorf("checkpoint run BOM, parameters or world mismatch")
	}
	if err := m.Identity.validate(); err != nil {
		return nil, err
	}
	if m.Step < 0 || m.Step > b.Data.Steps || m.Tokens < 0 || m.Consumption == nil || m.Execution == nil {
		return nil, fmt.Errorf("invalid checkpoint position/metadata")
	}
	total := int64(0)
	for corpus, n := range m.Consumption {
		if corpus == "" || n <= 0 || n > m.Tokens-total {
			return nil, fmt.Errorf("invalid checkpoint consumption")
		}
		total += n
	}
	if total != m.Tokens {
		return nil, fmt.Errorf("checkpoint consumption total mismatch")
	}
	if err := validateArtifacts(m.Files); err != nil {
		return nil, err
	}
	bundle, err := root.OpenRoot(filepath.Dir(ref.Path))
	if err != nil {
		return nil, err
	}
	defer bundle.Close()
	paths := map[string]string{}
	for _, a := range m.Files {
		if err := copyVerified(bundle, a, ""); err != nil {
			return nil, err
		}
		paths[a.Path] = filepath.Join(b.CheckpointDirectory, filepath.Dir(ref.Path), a.Path)
	}
	return &Resume{Step: m.Step, Tokens: m.Tokens, CheckpointManifest: raw, Paths: paths, reference: ref, consumption: m.Consumption}, nil
}

// The producer can run ahead of checkpoint events. A disk ledger retains exact
// per-step prefixes with memory bounded by the corpus count, not the run length.
type consumptionLedger struct {
	sync.Mutex
	file   *os.File
	counts map[string]int64
}

func newLedger(dir string, initial map[string]int64) (*consumptionLedger, error) {
	f, err := os.CreateTemp(dir, "checkpoint-ledger-")
	if err != nil {
		return nil, err
	}
	counts := map[string]int64{}
	for k, v := range initial {
		counts[k] = v
	}
	return &consumptionLedger{file: f, counts: counts}, nil
}
func (l *consumptionLedger) close() { l.file.Close(); os.Remove(l.file.Name()) }
func (l *consumptionLedger) add(step int64, counts map[string]int64) error {
	l.Lock()
	defer l.Unlock()
	for k, v := range counts {
		l.counts[k] += v
	}
	return json.NewEncoder(l.file).Encode(struct {
		Step   int64
		Counts map[string]int64
	}{step, l.counts})
}
func (l *consumptionLedger) at(step int64) (map[string]int64, error) {
	l.Lock()
	defer l.Unlock()
	n, err := l.file.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}
	d := json.NewDecoder(io.NewSectionReader(l.file, 0, n))
	for {
		var row struct {
			Step   int64
			Counts map[string]int64
		}
		if err := d.Decode(&row); err != nil {
			return nil, fmt.Errorf("checkpoint step %d not sent: %w", step, err)
		}
		if row.Step == step {
			return row.Counts, nil
		}
	}
}
func sameResume(a, b *Resume) bool {
	return a.Step == b.Step && a.Tokens == b.Tokens && bytes.Equal(a.CheckpointManifest, b.CheckpointManifest) && digest(a.Paths) == digest(b.Paths)
}

func sameCheckpointWorld(a, b applaunch.World) bool { a.NodeRank = 0; b.NodeRank = 0; return a == b }
