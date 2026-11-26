// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import "testing"

// Resolve output must satisfy micro-batch × accumulation × world size = global
// batch exactly, with the trainer's world size equal to WALDO's.
func TestResolvedBatchMustMatchGlobalBatchAndWorld(t *testing.T) {
	for _, ok := range []struct {
		global, world int64
		batch         Batch
	}{
		{8, 2, Batch{MicroBatchSequences: 2, AccumulationSteps: 2, WorldSize: 2}},
		{8, 4, Batch{MicroBatchSequences: 1, AccumulationSteps: 2, WorldSize: 4}},
		{8, 1, Batch{MicroBatchSequences: 8, AccumulationSteps: 1, WorldSize: 1}},
	} {
		if err := ok.batch.validate(ok.global, ok.world); err != nil {
			t.Fatalf("%+v refused: %v", ok, err)
		}
	}
	for name, bad := range map[string]struct {
		global, world int64
		batch         Batch
	}{
		"trainer world differs":     {8, 2, Batch{MicroBatchSequences: 2, AccumulationSteps: 2, WorldSize: 1}},
		"global not divisible":      {5, 2, Batch{MicroBatchSequences: 1, AccumulationSteps: 2, WorldSize: 2}},
		"micro does not divide":     {6, 2, Batch{MicroBatchSequences: 2, AccumulationSteps: 1, WorldSize: 2}},
		"accumulation wrong":        {8, 2, Batch{MicroBatchSequences: 2, AccumulationSteps: 1, WorldSize: 2}},
		"zero micro":                {8, 2, Batch{AccumulationSteps: 2, WorldSize: 2}},
		"zero world":                {8, 0, Batch{MicroBatchSequences: 2, AccumulationSteps: 2}},
		"non-positive global batch": {0, 1, Batch{MicroBatchSequences: 1, AccumulationSteps: 1, WorldSize: 1}},
	} {
		if err := bad.batch.validate(bad.global, bad.world); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}
