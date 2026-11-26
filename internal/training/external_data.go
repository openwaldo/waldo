// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package training

// ExternalDataParameters contains only WALDO-owned resolved data facts.
type ExternalDataParameters struct {
	Epochs               int64             `json:"epochs,omitempty"`
	RequestedTokens      int64             `json:"requested_tokens,omitempty"`
	Steps                int64             `json:"steps"`
	GlobalBatchSequences int64             `json:"global_batch_sequences"`
	SequenceLength       int64             `json:"sequence_length"`
	Seed                 uint64            `json:"seed"`
	Order                DataPlan          `json:"order"`
	Evaluation           *EvaluationPolicy `json:"evaluation,omitempty"`
	CheckpointEvery      int64             `json:"checkpoint_every"`
	EvaluateEvery        int64             `json:"evaluate_every"`
	PlannedTokenCapacity int64             `json:"planned_token_capacity"`
}
