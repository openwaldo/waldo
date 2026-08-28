// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"fmt"
)

// StreamCanonicalTextBatches routes each planned input through its accepted
// adapter in stable plan order. The adapter choice is never re-detected during
// execution.
func StreamCanonicalTextBatches(ctx context.Context, plan Plan, consume func(TextBatch) error) error {
	if err := plan.Validate(); err != nil {
		return err
	}
	if consume == nil {
		return fmt.Errorf("text batch consumer is required")
	}
	var totalBytes int64
	for _, input := range plan.Inputs {
		totalBytes += input.Artifact.Bytes
	}
	emitProgress(ctx, ProgressEvent{Phase: "ingest", Status: "started", TotalBytes: totalBytes, TotalFiles: int64(len(plan.Inputs))})
	var completedBytes, completedFiles int64
	for _, input := range plan.Inputs {
		emitProgress(ctx, ProgressEvent{Phase: "convert", Status: "started", Input: input.Artifact.Path, Adapter: input.Adapter, TotalBytes: input.Artifact.Bytes})
		inputPlan := plan
		inputPlan.Inputs = []PlanInput{input}
		consumeWithProgress := func(batch TextBatch) error {
			inputBytes := min(max(batch.InputBytes, 0), input.Artifact.Bytes)
			batch.ProgressBytes = completedBytes + inputBytes
			batch.ProgressTotalBytes = totalBytes
			batch.ProgressFiles = completedFiles
			batch.ProgressTotalFiles = int64(len(plan.Inputs))
			return consume(batch)
		}
		var err error
		if input.Profile.recordProfile() {
			err = StreamMappedRecordBatches(ctx, inputPlan, consumeWithProgress)
			if err != nil {
				return err
			}
			emitProgress(ctx, ProgressEvent{Phase: "convert", Status: "completed", Input: input.Artifact.Path, Adapter: input.Adapter, Bytes: input.Artifact.Bytes, TotalBytes: input.Artifact.Bytes})
			completedBytes += input.Artifact.Bytes
			completedFiles++
			emitProgress(ctx, ProgressEvent{Phase: "ingest", Status: "progress", Bytes: completedBytes, TotalBytes: totalBytes, Files: completedFiles, TotalFiles: int64(len(plan.Inputs))})
			continue
		}
		if input.Adapter == ProfileBoundedText || input.Adapter == ProfileXMLRecord {
			err = StreamProfiledFileBatches(ctx, inputPlan, consumeWithProgress)
			if err != nil {
				return err
			}
			emitProgress(ctx, ProgressEvent{Phase: "convert", Status: "completed", Input: input.Artifact.Path, Adapter: input.Adapter, Bytes: input.Artifact.Bytes, TotalBytes: input.Artifact.Bytes})
			completedBytes += input.Artifact.Bytes
			completedFiles++
			emitProgress(ctx, ProgressEvent{Phase: "ingest", Status: "progress", Bytes: completedBytes, TotalBytes: totalBytes, Files: completedFiles, TotalFiles: int64(len(plan.Inputs))})
			continue
		}
		switch input.Adapter {
		case "text", "markdown":
			err = StreamTextBatches(ctx, inputPlan, consumeWithProgress)
		case "parquet":
			err = StreamParquetTextBatches(ctx, inputPlan, consumeWithProgress)
		case "jsonl":
			err = StreamJSONLTextBatches(ctx, inputPlan, consumeWithProgress)
		case "mbox":
			err = StreamMboxTextBatches(ctx, inputPlan, consumeWithProgress)
		case "opaque-base64":
			err = StreamOpaqueTextBatches(ctx, inputPlan, consumeWithProgress)
		case "latex":
			err = StreamLatexTextBatches(ctx, inputPlan, consumeWithProgress)
		default:
			err = fmt.Errorf("unsupported accepted adapter %q", input.Adapter)
		}
		if err != nil {
			return err
		}
		emitProgress(ctx, ProgressEvent{Phase: "convert", Status: "completed", Input: input.Artifact.Path, Adapter: input.Adapter, Bytes: input.Artifact.Bytes, TotalBytes: input.Artifact.Bytes})
		completedBytes += input.Artifact.Bytes
		completedFiles++
		emitProgress(ctx, ProgressEvent{Phase: "ingest", Status: "progress", Bytes: completedBytes, TotalBytes: totalBytes, Files: completedFiles, TotalFiles: int64(len(plan.Inputs))})
	}
	return nil
}
