// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"strings"
	"testing"
)

// A secondary re-derives tokenizer, corpus, data and held-out pins from the
// primary's published node plan and must refuse any difference before it
// acknowledges readiness. The fixture trainer declares only single-process, so
// an unmodified plan passes every pin and stops at the distributed launch.
func TestExternalSecondaryRejectsDataPinsBeforeReadiness(t *testing.T) {
	b, c, o := externalLifecycleFixture(t)
	var published []ExternalNodePlan
	o.Publish = func(p ExternalNodePlan) error { published = append(published, p); return nil }
	if _, err := b.ComposeExternal(t.Context(), "primary", c, o); err != nil {
		t.Fatal(err)
	}
	if len(published) != 1 {
		t.Fatalf("published %d node plans", len(published))
	}
	o.Publish = nil
	for name, test := range map[string]struct {
		mutate func(*ExternalNodePlan)
		want   string
	}{
		"unchanged":     {func(*ExternalNodePlan) {}, "launch profile"},
		"tokenizer":     {func(p *ExternalNodePlan) { p.BOM.Identity.Tokenizer.ConfigurationSHA256 = strings.Repeat("b", 64) }, "secondary tokenizer mismatch"},
		"corpus":        {func(p *ExternalNodePlan) { p.BOM.CorpusBOM.Paths = append([]string{"other"}, p.BOM.CorpusBOM.Paths...) }, "secondary corpus/data/evaluation mismatch"},
		"data seed":     {func(p *ExternalNodePlan) { p.Begin.Data.Seed++ }, "secondary corpus/data/evaluation mismatch"},
		"evaluation":    {func(p *ExternalNodePlan) { p.BOM.EvaluationSet.SHA256 = strings.Repeat("c", 64) }, "secondary corpus/data/evaluation mismatch"},
		"stage seed":    {func(p *ExternalNodePlan) { p.Stage.Data.Seed++ }, "secondary corpus/data/evaluation mismatch"},
		"data sequence": {func(p *ExternalNodePlan) { p.Begin.Data.SequenceLength++ }, "secondary corpus/data/evaluation mismatch"},
	} {
		t.Run(name, func(t *testing.T) {
			p := published[0]
			test.mutate(&p)
			ready := false
			err := RunExternalSecondary(context.Background(), p, o, t.TempDir(), func() error { ready = true; return nil })
			if err == nil || !strings.Contains(err.Error(), test.want) || ready {
				t.Fatalf("ready=%v err=%v, want %q", ready, err, test.want)
			}
		})
	}
}
