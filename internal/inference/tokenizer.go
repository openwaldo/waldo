// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package inference

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/openwaldo/waldo/internal/training"
)

func loadTokenizer(path string) (training.TokenizerSpec, training.TokenCodec, error) {
	if path == "" {
		return training.ResolveTokenizer("byte", training.ByteTokenizerRevision, 259)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return training.TokenizerSpec{}, nil, err
	}
	var artifact struct {
		Kind   string `json:"kind"`
		Schema int    `json:"schema"`
		training.TokenizerSpec
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		return training.TokenizerSpec{}, nil, err
	}
	if artifact.Schema != 1 || (artifact.Kind != "waldo-tokenizer" && artifact.Kind != "waldo-byte-tokenizer") {
		return training.TokenizerSpec{}, nil, fmt.Errorf("unsupported WALDO tokenizer artifact")
	}
	spec, codec, err := training.ResolveTokenizerSpec(artifact.TokenizerSpec)
	if err != nil {
		return training.TokenizerSpec{}, nil, err
	}
	if spec.PadID != artifact.PadID || spec.BOSID != artifact.BOSID || spec.EOSID != artifact.EOSID {
		return training.TokenizerSpec{}, nil, fmt.Errorf("WALDO tokenizer artifact has invalid special token IDs")
	}
	return spec, codec, nil
}
