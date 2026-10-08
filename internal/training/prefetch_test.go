// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestPrefetchedRecordSourcePreservesOrder(t *testing.T) {
	source := prefetchedRecordSource{source: staticRecordSource{
		{ID: "one"}, {ID: "two"}, {ID: "three"},
	}, depth: 2}
	var got []string
	if err := source.Stream(context.Background(), func(record Record) error {
		got = append(got, record.ID)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if want := []string{"one", "two", "three"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("prefetched order = %v, want %v", got, want)
	}
}

func TestPrefetchedRecordSourceStopsProducerWhenConsumerStops(t *testing.T) {
	want := errors.New("stop")
	source := prefetchedRecordSource{source: staticRecordSource{{ID: "one"}, {ID: "two"}}, depth: 1}
	if err := source.Stream(context.Background(), func(Record) error { return want }); !errors.Is(err, want) {
		t.Fatalf("prefetch error = %v, want %v", err, want)
	}
}

func TestPrefetchedRecordSourceRejectsMissingInput(t *testing.T) {
	if err := (prefetchedRecordSource{}).Stream(context.Background(), func(Record) error { return nil }); err == nil {
		t.Fatal("missing prefetch source accepted")
	}
}
