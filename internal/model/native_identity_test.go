// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

// TestNativePlanGoldenHash pins a hand-built Plan. This pins the identity the
// real builder writes for a native compose; the value was computed with the
// same fixture on upstream main cec5e69 before external trainers existed.
func TestNativeComposeBuilderModelIDMatchesUpstream(t *testing.T) {
	compose := validCompose()
	stage := preparedFixture(t, compose.Stages[0])
	next := 0
	builder := Builder{Root: t.TempDir(), NewID: func() (string, error) {
		next++
		return fmt.Sprintf("run%04d", next), nil
	}, Resolver: training.FakeResolver()}
	result, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{stage})
	if err != nil {
		t.Fatal(err)
	}
	const upstream = "bff4a9785b17741bc4d4d580a42aaadb3d4a530fdb5fa66879a6c14f7c0d5bfc"
	if result.Model.ID != upstream || result.Model.PlanSHA256 != upstream {
		t.Fatalf("native model identity = %s / %s, want %s", result.Model.ID, result.Model.PlanSHA256, upstream)
	}
	if external, err := IsExternalModel(builder.Root, "smoke"); err != nil || external {
		t.Fatalf("native model classified external: %v %v", external, err)
	}
}

// Dispatch keys on the top-level trainer block only. Partial external blocks
// must be refused, never silently trained with the built-in trainer.
func TestComposeDispatchRefusesPartialExternalBlocks(t *testing.T) {
	native := strings.Replace(composeYAML(""), "schema: 1", "schema: 2", 1)
	for name, data := range map[string]string{
		"tokenizer without trainer": native + "tokenizer:\n  source: {repository: https://example.invalid/t.git, commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, path: app}\n  environment: cpu\n  configuration: {}\n",
		"null trainer":              native + "trainer: null\n",
		"empty trainer":             native + "trainer: {}\n",
		"stage trainer only":        strings.Replace(native, "      seed: 7\n", "      seed: 7\n    trainer: {learning_rate: 0.001}\n", 1),
		"external schema 1":         strings.Replace(externalComposeYAML, "schema: 2", "schema: 1", 1),
	} {
		doc, err := loadExternalFixture(t, data)
		if err == nil || doc.Builtin != nil || doc.External != nil {
			t.Fatalf("%s: accepted (builtin=%v external=%v): %v", name, doc.Builtin != nil, doc.External != nil, err)
		}
	}
}
