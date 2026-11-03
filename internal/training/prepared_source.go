// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"errors"
	"fmt"
)

// sequencePacker owns continuous EOS packing for both worker protocols. Masks
// describe tokens before shifting; the first token in each window is context.
// A consumer must copy slices it retains beyond the callback.
type sequencePacker struct {
	length  int
	eos     int
	tokens  []int
	masks   []bool
	corpora []string
}

func (p *sequencePacker) Add(record Record, emit func([]int, []bool, []string) error) error {
	if len(record.LossMask) != len(record.Tokens)+1 {
		return fmt.Errorf("prepared record %s loss mask does not include its EOS target", record.ID)
	}
	p.tokens = append(p.tokens, record.Tokens...)
	p.tokens = append(p.tokens, p.eos)
	p.masks = append(p.masks, record.LossMask...)
	for range record.LossMask {
		p.corpora = append(p.corpora, record.Corpus)
	}
	window := p.length + 1
	for len(p.tokens) >= window {
		if err := emit(p.tokens[:window], p.masks[1:window], p.corpora[1:window]); err != nil {
			return err
		}
		p.tokens = p.tokens[p.length:]
		p.masks = p.masks[p.length:]
		p.corpora = p.corpora[p.length:]
	}
	return nil
}

func (p *sequencePacker) Finish(emit func([]int, []bool, []string) error) error {
	if len(p.tokens) > 1 {
		return emit(p.tokens, p.masks[1:], p.corpora[1:])
	}
	return nil
}

// NewPreparedSequenceSource packs an already tokenized, ordered training view
// for external trainer protocol 1. The caller owns corpus selection, held-out
// exclusion and tokenization (NewTokenizedRecordSource accepts app codecs).
// The returned function is assignable to trainerapp.SequenceSource without a
// dependency from training to model packages.
//
// Resume replays the deterministic prefix but emits only the suffix. A terminal
// resume never opens the source. Short final sequences are padded with masked
// targets; missing batch slots fail rather than inventing supervised examples.
func NewPreparedSequenceSource(records RecordSource, data ExternalDataParameters, tokenizer TokenizerSpec, resumeStep int64) (func(context.Context, func(PreparedSequence) error) error, error) {
	if records == nil || data.Steps <= 0 || data.GlobalBatchSequences <= 0 || data.SequenceLength <= 0 || data.SequenceLength >= int64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("prepared source requires records and positive, representable data dimensions")
	}
	if resumeStep < 0 || resumeStep > data.Steps {
		return nil, fmt.Errorf("prepared resume step outside data budget")
	}
	limit, overflow := multiplyInt64(data.Steps, data.GlobalBatchSequences)
	if overflow {
		return nil, fmt.Errorf("prepared sequence target overflows int64")
	}
	if tokenizer.VocabularySize <= 0 || tokenizer.EOSID < 0 || tokenizer.EOSID >= tokenizer.VocabularySize || tokenizer.PadID < 0 || tokenizer.PadID >= tokenizer.VocabularySize {
		return nil, fmt.Errorf("prepared source requires valid tokenizer vocabulary and special IDs")
	}
	start := resumeStep * data.GlobalBatchSequences
	return func(ctx context.Context, consume func(PreparedSequence) error) error {
		if consume == nil {
			return fmt.Errorf("prepared sequence consumer is required")
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if start == limit {
			return nil
		}
		packer := sequencePacker{length: int(data.SequenceLength), eos: tokenizer.EOSID}
		ordinal := int64(0)
		emit := func(tokens []int, mask []bool, corpora []string) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if !anyMask(mask) {
				return nil
			}
			if ordinal >= start {
				sequence := PreparedSequence{Ordinal: ordinal, Tokens: append([]int(nil), tokens...), LossMask: append([]bool(nil), mask...), Consumption: map[string]int64{}}
				for i, supervised := range mask {
					if supervised {
						sequence.Consumption[corpora[i]]++
					}
				}
				for len(sequence.Tokens) < packer.length+1 {
					sequence.Tokens = append(sequence.Tokens, tokenizer.PadID)
					sequence.LossMask = append(sequence.LossMask, false)
				}
				if err := consume(sequence); err != nil {
					return err
				}
			}
			ordinal++
			if ordinal == limit {
				return errWorkerReachedTarget
			}
			return nil
		}
		err := records.Stream(ctx, func(record Record) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if record.Corpus == "" {
				return fmt.Errorf("prepared record %s lacks corpus", record.ID)
			}
			for _, token := range record.Tokens {
				if token < 0 || token >= tokenizer.VocabularySize {
					return fmt.Errorf("prepared record %s token outside vocabulary", record.ID)
				}
			}
			return packer.Add(record, emit)
		})
		if errors.Is(err, errWorkerReachedTarget) {
			return nil
		}
		if err != nil {
			return err
		}
		if err = packer.Finish(emit); err != nil && !errors.Is(err, errWorkerReachedTarget) {
			return err
		}
		if ordinal != limit {
			return fmt.Errorf("prepared stream produced %d sequences; exact budget requires %d", ordinal, limit)
		}
		return nil
	}, nil
}

// CachedPreparedSequenceSource shares the native compact, hash-verified cache.
// A fresh attempt writes the whole stream as it is consumed. Prefix replay is
// retained when an interrupted producer never completed the cache.
func CachedPreparedSequenceSource(records RecordSource, data ExternalDataParameters, tokenizer TokenizerSpec, resumeStep int64, directory, identity, bomSHA256 string, maxBytes int64) (func(context.Context, func(PreparedSequence) error) error, error) {
	suffix, err := NewPreparedSequenceSource(records, data, tokenizer, resumeStep)
	if err != nil {
		return nil, err
	}
	if directory == "" || identity == "" {
		return suffix, nil
	}
	full, err := NewPreparedSequenceSource(records, data, tokenizer, 0)
	if err != nil {
		return nil, err
	}
	return func(ctx context.Context, consume func(PreparedSequence) error) error {
		if consume == nil {
			return fmt.Errorf("prepared sequence consumer is required")
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if resumeStep == data.Steps {
			return nil
		}
		start := resumeStep * data.GlobalBatchSequences
		replayed, err := replayPreparedSequences(directory, identity, bomSHA256, 0, 1, 1, data.GlobalBatchSequences, maxBytes, start, externalPreparedEncoder{ctx, consume})
		if err != nil || replayed {
			return err
		}
		writer, err := newPreparedCacheWriter(directory, identity, bomSHA256, 0, 1, 1, data.GlobalBatchSequences, maxBytes)
		if err != nil {
			return err
		}
		defer writer.Abort()
		err = full(ctx, func(s PreparedSequence) error {
			if err := writer.Append(s); err != nil {
				return err
			}
			if s.Ordinal >= start {
				return consume(s)
			}
			return nil
		})
		if err != nil {
			return err
		}
		return writer.Commit(data.Steps * data.GlobalBatchSequences)
	}, nil
}

type externalPreparedEncoder struct {
	ctx     context.Context
	consume func(PreparedSequence) error
}

func (e externalPreparedEncoder) Encode(value any) error {
	if err := e.ctx.Err(); err != nil {
		return err
	}
	frame, ok := value.(WorkerInputFrame)
	if !ok {
		return fmt.Errorf("invalid prepared cache frame")
	}
	if frame.Sequence != nil {
		return e.consume(*frame.Sequence)
	}
	return nil
}
