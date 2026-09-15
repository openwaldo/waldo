// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"errors"
)

const defaultRecordPrefetch = 64

// prefetchedRecordSource overlaps canonical decoding and tokenization with
// worker-protocol encoding. The queue is bounded so backpressure cannot turn
// prefetch into unbounded host memory use.
type prefetchedRecordSource struct {
	source RecordSource
	depth  int
}

type prefetchedRecord struct {
	record Record
	err    error
}

func (source prefetchedRecordSource) Stream(ctx context.Context, consume func(Record) error) error {
	if source.source == nil {
		return errors.New("prefetch requires a record source")
	}
	depth := source.depth
	if depth < 1 {
		depth = defaultRecordPrefetch
	}
	prefetchContext, cancel := context.WithCancel(ctx)
	defer cancel()
	queue := make(chan prefetchedRecord, depth)
	go func() {
		defer close(queue)
		err := source.source.Stream(prefetchContext, func(record Record) error {
			select {
			case queue <- prefetchedRecord{record: record}:
				return nil
			case <-prefetchContext.Done():
				return prefetchContext.Err()
			}
		})
		if err != nil && !errors.Is(err, context.Canceled) {
			select {
			case queue <- prefetchedRecord{err: err}:
			case <-prefetchContext.Done():
			}
		}
	}()
	for item := range queue {
		if item.err != nil {
			return item.err
		}
		if err := consume(item.record); err != nil {
			return err
		}
	}
	return ctx.Err()
}

func prefetchRecords(source RecordSource) RecordSource {
	if source == nil {
		return nil
	}
	return prefetchedRecordSource{source: source, depth: defaultRecordPrefetch}
}
