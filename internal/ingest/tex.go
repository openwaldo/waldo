// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package ingest

import (
 "bytes"
 "context"
 "crypto/sha256"
 "encoding/hex"
 "encoding/json"
 "fmt"
 "os"
 "os/exec"
 "path/filepath"

 "github.com/openwaldo/waldo/internal/shard"
)

// StreamLatexTextBatches converts a single LaTeX root artifact into one
// canonical TextRow via the external tex2waldo.sh pipeline. Non-root .tex
// files (includes, chapters) are silently skipped — the root already embeds
// them — to avoid content duplication in the parquet shards.
func StreamLatexTextBatches(ctx context.Context, plan Plan, consume func(TextBatch) error) error {
	if err := plan.Validate(); err != nil {
		return err
	}
	if consume == nil {
		return fmt.Errorf("text batch consumer is required")
	}
	for _, input := range plan.Inputs {
		if input.Adapter != "latex" {
			return fmt.Errorf("input %s requires the latex adapter", input.Artifact.Path)
		}
	}
	for _, input := range plan.Inputs {
		data, err := os.ReadFile(input.Artifact.Path)
		if err != nil {
			return err
		}
		// Contract v1: only the root document (contains \begin{document})
		// is converted; includes are skipped.
		if !isLatexRoot(data) {
			if err := consume(TextBatch{InputBytes: input.Artifact.Bytes}); err != nil {
				return err
			}
			continue
		}
		row, err := runTex2Waldo(ctx, input, data)
		if err != nil {
			return fmt.Errorf("tex2waldo %s: %w", input.Artifact.Path, err)
		}
		batch := TextBatch{
			Rows:         []shard.TextRow{row},
			LogicalBytes: int64(len(row.Text)),
			InputBytes:   input.Artifact.Bytes,
		}
		if err := consume(batch); err != nil {
			return err
		}
	}
	return nil
}

func runTex2Waldo(ctx context.Context, input PlanInput, source []byte) (shard.TextRow, error) {
	bin, err := resolveTex2Waldo()
	if err != nil {
		return shard.TextRow{}, err
	}
	outdir, err := os.MkdirTemp("", "tex2waldo-")
	if err != nil {
		return shard.TextRow{}, err
	}
	defer os.RemoveAll(outdir)

	bookDir := filepath.Dir(input.Artifact.Path)
	mainTex := filepath.Base(input.Artifact.Path)
	cmd := exec.CommandContext(ctx, bin, "--waldo", bookDir, mainTex, outdir)
	cmd.Env = append(os.Environ(),
		"WALDO_FETCH_DIR="+bookDir,
		"TEX2WALDO_DOI="+resolveDOI(input),
		"TEX2WALDO_LICENSE="+resolveLicense(input),
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return shard.TextRow{}, fmt.Errorf("%v\nstderr: %s", err, stderr.String())
	}

	mdBytes, err := os.ReadFile(filepath.Join(outdir, "output.md"))
	if err != nil {
		return shard.TextRow{}, err
	}
	manifestBytes, err := os.ReadFile(filepath.Join(outdir, "manifest.json"))
	if err != nil {
		return shard.TextRow{}, err
	}
	var manifest struct {
		Schema string `json:"schema"`
		Metadata struct {
			DOI     string `json:"doi"`
			License string `json:"license"`
		} `json:"metadata"`
	}
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return shard.TextRow{}, fmt.Errorf("bad manifest: %w", err)
	}

	contentHash := sha256.Sum256(mdBytes)
	license := manifest.Metadata.License
	if license == "" {
		license = resolveLicense(input)
	}
	metaJSON, _ := json.Marshal(map[string]any{
		"manifest":  json.RawMessage(manifestBytes),
		"source_sha": sha256Of(source),
	})
	meta := string(metaJSON)

	row := shard.TextRow{
		Text:     string(mdBytes),
		Source:   resolveSource(input),
		License:  license,
		ContentSHA256: contentHash,
		Meta:     &meta,
	}
	if doi := manifest.Metadata.DOI; doi != "" || resolveDOI(input) != "" {
		name := doi
		if name == "" {
			name = resolveDOI(input)
		}
		row.SourceName = &name
	}
	return row, nil
}

func resolveTex2Waldo() (string, error) {
	if v := os.Getenv("WALDO_TEX2WALDO"); v != "" {
		return v, nil
	}
	return exec.LookPath("tex2waldo.sh")
}

func resolveSource(input PlanInput) string {
	if input.SourceID != "" {
		return input.SourceID
	}
	return "latex:" + filepath.Base(input.Artifact.Path)
}

func resolveDOI(input PlanInput) string {
	// Hook: recipe/manifest puede inyectarlo; por ahora vacío.
	return ""
}

func resolveLicense(input PlanInput) string {
	// El PlanRequest trae la licencia vía plan.Writer o source.License.
	// Dejamos vacío; el caller la propaga en el parquet a nivel de shard.
	return ""
}

func sha256Of(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
// isLatexRoot reports whether the bytes contain a LaTeX document root.
// Used by StreamLatexTextBatches to skip includes and avoid content duplication.
func isLatexRoot(data []byte) bool {
	return bytes.Contains(data, []byte("\\begin{document}"))
}
