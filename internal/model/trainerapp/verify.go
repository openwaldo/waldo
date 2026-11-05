// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/openwaldo/waldo/internal/modelweights"
	"github.com/openwaldo/waldo/internal/training"
)

func integrity(err error) error {
	if err == nil {
		return nil
	}
	if training.IsNonRetryableWorkerError(err) {
		return err
	}
	class := training.WorkerErrorArtifactIntegrity
	if errors.Is(err, modelweights.ErrNonFinite) {
		class = training.WorkerErrorNumericalIntegrity
	}
	return &training.WorkerError{Class: class, Message: err.Error()}
}

// VerifyPublished checks all declarations under a rooted directory. The PoC
// publishes only Safetensors weights and JSON configuration/tokenizer assets;
// an allowlist avoids guessing whether arbitrary bytes might be a pickle.
// Checkpoint files are private and intentionally use a different verifier.
func VerifyPublished(directory string, artifacts []Artifact) error {
	if err := validateArtifacts(artifacts); err != nil {
		return integrity(err)
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return integrity(err)
	}
	defer root.Close()
	weights := false
	for _, a := range artifacts {
		if strings.Contains(a.Path, "\\") {
			return integrity(fmt.Errorf("nonportable artifact path %q", a.Path))
		}
		switch a.Role {
		case "weights":
			if a.Format != "safetensors" {
				return integrity(fmt.Errorf("published weights require safetensors"))
			}
			weights = true
		case "configuration", "tokenizer":
			if a.Format != "json" {
				return integrity(fmt.Errorf("published %s requires JSON in the PoC", a.Role))
			}
		default:
			return integrity(fmt.Errorf("unsupported published role %q", a.Role))
		}
		if err := verifyPublishedFile(root, a); err != nil {
			return integrity(fmt.Errorf("artifact %s: %w", a.Path, err))
		}
	}
	if !weights {
		return integrity(fmt.Errorf("published artifacts require weights"))
	}
	return nil
}
func verifyPublishedFile(root *os.Root, a Artifact) error {
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
		return fmt.Errorf("not regular or size mismatch")
	}
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return err
	}
	if n != a.Bytes || hex.EncodeToString(h.Sum(nil)) != a.SHA256 {
		return fmt.Errorf("hash/size mismatch")
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if a.Format == "safetensors" {
		return modelweights.VerifyFinite(f)
	}
	// JSON assets are bounded independently of potentially very large weights.
	if a.Bytes > 16<<20 {
		return fmt.Errorf("JSON artifact exceeds 16 MiB")
	}
	raw, err := io.ReadAll(io.LimitReader(f, 16<<20+1))
	if err != nil {
		return err
	}
	if !json.Valid(raw) {
		return fmt.Errorf("invalid JSON artifact (pickle is not publishable)")
	}
	return nil
}

func verifyCompletionFiles(c Complete, b Begin, candidates map[int64]CheckpointReference, observed []Evaluation) error {
	selection := c.SelectedCheckpoint
	if selection == nil {
		return integrity(fmt.Errorf("missing checkpoint selection"))
	}
	ref, ok := candidates[selection.Step]
	if !ok {
		return integrity(fmt.Errorf("selected checkpoint was not persisted"))
	}
	selected, err := LoadCheckpoint(b, ref)
	if err != nil {
		return integrity(err)
	}
	if selected.Tokens != selection.Tokens {
		return integrity(fmt.Errorf("selected checkpoint token mismatch"))
	}
	completed := map[int64]Evaluation{}
	for _, e := range c.Evaluations {
		if _, ok := completed[e.Step]; ok {
			return integrity(fmt.Errorf("duplicate completion evaluation"))
		}
		completed[e.Step] = e
	}
	for _, e := range observed {
		if reported, ok := completed[e.Step]; !ok || reported != e {
			return integrity(fmt.Errorf("completion changed or omitted observed evaluation"))
		}
	}
	for _, e := range append(append([]Evaluation(nil), observed...), c.Evaluations...) {
		ref, ok := candidates[e.Step]
		if !ok {
			return integrity(fmt.Errorf("evaluation has no persisted checkpoint at step %d", e.Step))
		}
		resume, err := LoadCheckpoint(b, ref)
		if err != nil {
			return integrity(err)
		}
		if resume.Tokens != e.Tokens {
			return integrity(fmt.Errorf("evaluation checkpoint token mismatch"))
		}
	}
	// A completion cannot omit a better candidate already observed by WALDO.
	if b.EvaluationSet.Records > 0 {
		s := c.SelectedCheckpoint
		for _, e := range observed {
			if e.Loss < s.Value || e.Loss == s.Value && e.Step < s.Step {
				return integrity(fmt.Errorf("completion omitted a better evaluated checkpoint"))
			}
		}
		if c.ArtifactEvaluation == nil {
			return integrity(fmt.Errorf("missing published artifact evaluation"))
		}
		e := *c.ArtifactEvaluation
		if err := validateEvaluation(e, b); err != nil {
			return integrity(err)
		}
		if e.Step != s.Step || e.Tokens != s.Tokens || math.Abs(e.Loss-s.Value) > math.Max(0.02, 0.01*math.Abs(s.Value)) {
			return integrity(fmt.Errorf("published artifact loss differs from selected checkpoint"))
		}
	} else if c.ArtifactEvaluation != nil {
		return integrity(fmt.Errorf("artifact evaluation without held-out pins"))
	}
	return VerifyPublished(b.Paths.Artifacts, c.Artifacts)
}
