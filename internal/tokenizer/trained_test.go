// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"reflect"
	"strings"
	"testing"
)

func TestTrainBytepieceIsDeterministicAndReversible(t *testing.T) {
	samples := []Sample{{ID: "b", Text: "hello world hello world"}, {ID: "a", Text: "hello WALDO"}}
	first, err := TrainBytepiece(samples, 32000, 1024, strings.Repeat("a", 64))
	if err != nil {
		t.Fatal(err)
	}
	second, err := TrainBytepiece([]Sample{samples[1], samples[0]}, 32000, 1024, strings.Repeat("a", 64))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) || first.VocabularySize <= 259 {
		t.Fatalf("trained artifacts differ: %+v / %+v", first, second)
	}
	codec, err := first.Codec()
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"hello world", "UTF-8: Grüße 世界", "unseen bytes"} {
		if decoded := codec.Decode(codec.Encode(text)); decoded != text {
			t.Fatalf("round trip %q = %q", text, decoded)
		}
	}
	comparison := Compare(codec, samples)
	if comparison.Tokens <= 0 || comparison.BytesPerToken <= 1 {
		t.Fatalf("comparison = %+v", comparison)
	}
}

func TestTrainBytepieceValidatesInputIdentity(t *testing.T) {
	if _, err := TrainBytepiece([]Sample{{ID: "same", Text: "one"}, {ID: "same", Text: "two"}}, 32000, 1024, strings.Repeat("a", 64)); err == nil {
		t.Fatal("duplicate sample IDs accepted")
	}
	if _, err := TrainBytepiece([]Sample{{ID: "one", Text: "text"}}, 258, 1024, strings.Repeat("a", 64)); err == nil {
		t.Fatal("undersized vocabulary accepted")
	}
}
