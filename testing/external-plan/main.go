// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

// Developer identity qualification: resolve a compose against an immutable
// corpus BOM without creating model/run state or executing training.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/training"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	compose := flag.String("compose", "", "one-stage compose")
	bomPath := flag.String("corpus-bom", "", "immutable corpus BOM JSON")
	name := flag.String("name", "", "model name (part of plan envelope)")
	cachePath := flag.String("cache", "", "application cache")
	python := flag.String("python", "/usr/bin/python3", "trusted bootstrap interpreter")
	flag.Parse()
	document, _, err := model.LoadComposeDocument(*compose)
	if err != nil {
		return err
	}
	if document.External == nil || len(document.External.Stages) != 1 {
		return fmt.Errorf("requires one external stage")
	}
	raw, err := os.ReadFile(*bomPath)
	if err != nil {
		return err
	}
	var bom corpus.BOM
	if err = json.Unmarshal(raw, &bom); err != nil {
		return err
	}
	if len(bom.Paths) != 1 || len(document.External.Stages[0].Corpora) != 1 || document.External.Stages[0].Corpora[0].Path != bom.Paths[0] {
		return fmt.Errorf("qualification requires one matching corpus selection")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Hour)
	defer cancel()
	cache, err := lookaside.DefaultCache()
	if err != nil {
		return err
	}
	objects, err := corpus.Materialize(ctx, bom, cache, nil)
	if err != nil {
		return err
	}
	var inputs []training.Input
	seen := map[string]bool{}
	for _, object := range objects.Objects {
		if seen[object.Shard.SHA256] {
			continue
		}
		seen[object.Shard.SHA256] = true
		inputs = append(inputs, training.Input{Path: object.Path, SHA256: object.Shard.SHA256, Bytes: object.Shard.Bytes, Records: object.Shard.Docs, Corpus: bom.Paths[0], RecordFilter: bom.RecordFilter})
	}
	scratch, err := os.MkdirTemp("", "waldo-plan-qualification-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(scratch)
	builder := model.Builder{Root: filepath.Join(scratch, "models")}
	out, err := builder.ComposeExternal(ctx, *name, *document.External, model.ExternalOptions{PlanOnly: true, Cache: *cachePath, Python: *python, Prepare: model.ExternalCorpusProvider(bom, inputs)})
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(struct {
		Plan   model.ExternalPlan `json:"plan"`
		SHA256 string             `json:"sha256"`
	}{out.Plan, out.Model.ID})
}
