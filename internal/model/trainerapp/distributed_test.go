// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"context"
	"encoding/json"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/training"
	"io"
	"os"
	"reflect"
	"testing"
)

func TestDistributedStreamPartitionsAndResume(t *testing.T) {
	for _, start := range []int64{0, 1} {
		for node := 0; node < 2; node++ {
			reader, writer, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			ledger, err := newLedger(t.TempDir(), nil)
			if err != nil {
				t.Fatal(err)
			}
			b := Begin{World: applaunch.World{Nodes: 2, NodeRank: node, ProcessesPerNode: 2, WorldSize: 4}, Data: training.ExternalDataParameters{Steps: 2, GlobalBatchSequences: 8, SequenceLength: 2}, Batch: Batch{MicroBatchSequences: 1, AccumulationSteps: 2, WorldSize: 4}, Tokenizer: training.TokenizerSpec{VocabularySize: 8}}
			if start > 0 {
				b.Resume = &Resume{Step: start}
			}
			source := func(ctx context.Context, consume func(training.PreparedSequence) error) error {
				for ordinal := start * 8; ordinal < 16; ordinal++ {
					if err := consume(training.PreparedSequence{Ordinal: ordinal, Tokens: []int{1, 2, 3}, LossMask: []bool{true, true}, Consumption: map[string]int64{"books": 2}}); err != nil {
						return err
					}
				}
				return nil
			}
			done := make(chan error, 1)
			go func() {
				counts, err := send(context.Background(), &applaunch.Process{Stdin: writer}, b, nil, source, ledger)
				if counts["books"] != (2-start)*16 {
					t.Errorf("global ledger: %v", counts)
				}
				writer.Close()
				done <- err
			}()
			var ordinals, boundaries []int64
			decoder := json.NewDecoder(reader)
			for {
				var frame struct {
					Kind    string
					Record  training.PreparedSequence
					Ordinal int64
				}
				err := decoder.Decode(&frame)
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				if frame.Kind == "record" {
					ordinals = append(ordinals, frame.Record.Ordinal)
				}
				if frame.Kind == "micro_batch_end" {
					boundaries = append(boundaries, frame.Ordinal)
				}
			}
			reader.Close()
			if err = <-done; err != nil {
				t.Fatal(err)
			}
			ledger.close()
			var wantOrdinals, wantBoundaries []int64
			for i := start * 8; i < 16; i++ {
				if i%4/2 == int64(node) {
					wantOrdinals = append(wantOrdinals, i)
				}
				if (i+1)%4 == 0 {
					wantBoundaries = append(wantBoundaries, i+1)
				}
			}
			if !reflect.DeepEqual(ordinals, wantOrdinals) || !reflect.DeepEqual(boundaries, wantBoundaries) {
				t.Fatalf("node %d start %d: %v %v", node, start, ordinals, boundaries)
			}
		}
	}
}
