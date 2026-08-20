// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package calibration

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/index"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/shard"
	"github.com/openwaldo/waldo/internal/tokenizer"
)

func TestPrepareIsBoundedReproducibleAndAudited(t *testing.T) {
	object, objectSHA, objectBytes, referenceTokens := calibrationShard(t, []string{"alpha", "bravo", "charlie"})
	cache, err := lookaside.NewCache(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	bom := calibrationCorpus(object, objectSHA, objectBytes, referenceTokens, 3)
	first, err := Prepare(context.Background(), bom, cache, 8, 42, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Cleanup()
	second, err := Prepare(context.Background(), bom, cache, 8, 42, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Cleanup()
	if first.BOM.SampledTokens != 8 || first.BOM.Records != 2 || len(first.BOM.Shards) != 1 {
		t.Fatalf("calibration BOM = %+v", first.BOM)
	}
	if first.SHA256 != second.SHA256 || first.BOM.SelectionSHA256 != second.BOM.SelectionSHA256 {
		t.Fatalf("same selection was not reproducible: %s/%s versus %s/%s", first.SHA256, first.BOM.SelectionSHA256, second.SHA256, second.BOM.SelectionSHA256)
	}
	data, err := os.ReadFile(first.TextPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "alpha\n\nbra" {
		t.Fatalf("sample = %q", data)
	}
	digest := sha256.Sum256(first.JSON)
	if hex.EncodeToString(digest[:]) != first.SHA256 {
		t.Fatal("calibration BOM digest does not match its bytes")
	}
}

func TestPrepareRejectsCorruptShardObject(t *testing.T) {
	object, objectSHA, objectBytes, referenceTokens := calibrationShard(t, []string{"valid"})
	data, err := os.ReadFile(object)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)/2] ^= 0xff
	if err := os.WriteFile(object, data, 0o644); err != nil {
		t.Fatal(err)
	}
	cache, err := lookaside.NewCache(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Prepare(context.Background(), calibrationCorpus(object, objectSHA, objectBytes, referenceTokens, 1), cache, 8, 42, nil); err == nil {
		t.Fatal("corrupt shard was accepted")
	}
}

func calibrationShard(t *testing.T, texts []string) (string, string, int64, int64) {
	t.Helper()
	counter, err := tokenizer.Get(tokenizer.Default)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "calibration.parquet")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	writer := shard.NewTextParquetWriter(file)
	var rows []shard.TextRow
	var tokens, contentBytes int64
	for _, text := range texts {
		digest := sha256.Sum256([]byte(text))
		count := int64(counter.Count(text))
		tokens += count
		contentBytes += int64(len(text))
		rows = append(rows, shard.TextRow{ContentSHA256: digest, Text: text, Source: "fixture", License: "CC0-1.0", TokenCount: &count})
	}
	if _, err := writer.Write(rows); err != nil {
		t.Fatal(err)
	}
	writer.SetKeyValueMetadata("waldo.records", fmt.Sprint(len(rows)))
	writer.SetKeyValueMetadata("waldo.tokens", fmt.Sprint(tokens))
	writer.SetKeyValueMetadata("waldo.content_bytes", fmt.Sprint(contentBytes))
	writer.SetKeyValueMetadata("waldo.email_address_records", "0")
	writer.SetKeyValueMetadata("waldo.repetitive_content_records", "0")
	writer.SetKeyValueMetadata("waldo.boilerplate_content_records", "0")
	writer.SetKeyValueMetadata("waldo.privacy_redaction_policy", shard.PrivacyRedactionPolicy)
	writer.SetKeyValueMetadata("waldo.redacted_email_addresses", "0")
	writer.SetKeyValueMetadata("waldo.redacted_ip_addresses", "0")
	writer.SetKeyValueMetadata("waldo.redacted_phone_numbers", "0")
	writer.SetKeyValueMetadata("waldo.removed_mail_routing_headers", "0")
	writer.SetKeyValueMetadata("waldo.redacted_credentials", "0")
	writer.SetKeyValueMetadata("waldo.licenses", `["CC0-1.0"]`)
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	return path, hex.EncodeToString(digest[:]), int64(len(data)), tokens
}

func calibrationCorpus(object, objectSHA string, objectBytes, tokens, records int64) corpus.BOM {
	conversion := index.Conversion{Tool: "waldo", Version: "test", Profile: "text", Recipe: shard.TextWriterRecipe, Tokenizer: tokenizer.Default}
	measure := index.Measures{Shards: 1, Docs: records, Tokens: tokens, Bytes: objectBytes}
	licenses := map[string]index.Measures{"CC0-1.0": measure}
	assessment := &index.ContentAssessment{
		EmailAddresses:     &index.DetectionMeasure{Detector: shard.EmailDetector},
		RepetitiveContent:  &index.DetectionMeasure{Detector: shard.RepetitionDetector},
		BoilerplateContent: &index.DetectionMeasure{Detector: shard.BoilerplateDetector},
	}
	redaction := &index.ContentRedaction{Policy: shard.PrivacyRedactionPolicy, NamesRetained: true}
	manifest := corpus.ManifestPin{
		Path: "core/test.json", SHA256: fmt.Sprintf("%064x", 1), Name: "test", Title: "Test", Description: "Calibration fixture.",
		License: "CC0-1.0", Format: "parquet", RecordSchema: shard.TextRecordSchema, ConvertedBy: conversion, Assessment: assessment, Redaction: redaction,
		Sources: []index.Source{{Name: "fixture", Source: "Fixture", URL: "https://example.invalid/source", SHA256: fmt.Sprintf("%064x", 2)}},
		Totals:  measure, Licenses: licenses,
	}
	pin := corpus.ShardPin{Manifest: manifest.Path, URL: object, SHA256: objectSHA, Format: "parquet", RecordSchema: shard.TextRecordSchema, License: "CC0-1.0", ConvertedBy: conversion, Docs: records, Tokens: tokens, Bytes: objectBytes, Assessment: assessment, Redaction: redaction}
	return corpus.BOM{Kind: "openwaldo-bom", Schema: 1, Subject: "corpus", Paths: []string{"core/test"}, Manifests: []corpus.ManifestPin{manifest}, Shards: []corpus.ShardPin{pin}, Totals: measure, Licenses: licenses}
}

func TestPrepareCreatesMissingScratchRoot(t *testing.T) {
	object, objectSHA, objectBytes, referenceTokens := calibrationShard(t, []string{"alpha", "bravo", "charlie"})
	absent := filepath.Join(t.TempDir(), "never-created")
	cache, err := lookaside.NewCache(absent, nil)
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := Prepare(context.Background(), calibrationCorpus(object, objectSHA, objectBytes, referenceTokens, 3), cache, 8, 42, nil)
	if err != nil {
		t.Fatalf("calibration failed on a scratch root that did not exist yet: %v", err)
	}
	defer prepared.Cleanup()
	info, err := os.Stat(cache.Scratch())
	if err != nil {
		t.Fatal(err)
	}
	if mode := info.Mode().Perm(); mode != 0o700 {
		t.Fatalf("scratch root mode = %04o, want 0700 to match the lookaside cache", mode)
	}
}
