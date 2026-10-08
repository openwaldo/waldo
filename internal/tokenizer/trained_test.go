// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"encoding/base64"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestTrainByteBPEIsDeterministicAndReversible(t *testing.T) {
	samples := []Sample{{ID: "b", Text: "hello world hello world"}, {ID: "a", Text: "hello WALDO"}}
	first, err := TrainByteBPE(samples, 32000, 1024, strings.Repeat("a", 64))
	if err != nil {
		t.Fatal(err)
	}
	second, err := TrainByteBPE([]Sample{samples[1], samples[0]}, 32000, 1024, strings.Repeat("a", 64))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) || first.Schema != 2 || first.Name != TrainedName || first.VocabularySize <= 259 || len(first.Merges) == 0 || len(first.Pieces) != 0 {
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

func TestTrainByteBPEFillsVocabularyAndCompressesTrainingText(t *testing.T) {
	var text strings.Builder
	for index := 0; index < 2000; index++ {
		fmt.Fprintf(&text, " common tokenizer term%04d reusable%03d", index, index%137)
	}
	artifact, err := TrainByteBPE([]Sample{{ID: "sample", Text: text.String()}}, 1000, int64(text.Len()), strings.Repeat("b", 64))
	if err != nil {
		t.Fatal(err)
	}
	if artifact.VocabularySize != 1000 || len(artifact.Merges) != 741 {
		t.Fatalf("vocabulary = %d, merges = %d", artifact.VocabularySize, len(artifact.Merges))
	}
	codec, err := artifact.Codec()
	if err != nil {
		t.Fatal(err)
	}
	if encoded := codec.Encode(text.String()); len(encoded) >= text.Len()/2 || codec.Decode(encoded) != text.String() {
		t.Fatalf("encoded %d tokens from %d bytes", len(encoded), text.Len())
	}
}

func TestLegacyBytepieceArtifactRemainsReadable(t *testing.T) {
	artifact := Artifact{
		Kind: TrainedKind, Schema: 1, Name: LegacyTrainedName,
		VocabularySize: 260, PadID: 0, BOSID: 1, EOSID: 2,
		TrainingInputSHA256: strings.Repeat("a", 64), CorpusBOMSHA256: strings.Repeat("b", 64), TrainingBytes: 5,
		Pieces:   []string{base64.StdEncoding.EncodeToString([]byte("hello"))},
		Revision: "sha256:a19305a3c90e4db82e983b31aef91ed7c615c280511ec24c1db766106fa20665",
	}
	codec, err := artifact.Codec()
	if err != nil {
		t.Fatal(err)
	}
	if got := codec.Decode(codec.Encode("hello world")); got != "hello world" {
		t.Fatalf("legacy round trip = %q", got)
	}
}

func TestTrainByteBPEValidatesInputIdentity(t *testing.T) {
	if _, err := TrainByteBPE([]Sample{{ID: "same", Text: "one"}, {ID: "same", Text: "two"}}, 32000, 1024, strings.Repeat("a", 64)); err == nil {
		t.Fatal("duplicate sample IDs accepted")
	}
	if _, err := TrainByteBPE([]Sample{{ID: "one", Text: "text"}}, 258, 1024, strings.Repeat("a", 64)); err == nil {
		t.Fatal("undersized vocabulary accepted")
	}
}

func TestByteBPEArtifactSchemaFixture(t *testing.T) {
	artifact, err := TrainByteBPE([]Sample{{ID: "sample", Text: "banana banana bandana"}}, 265, 1024, strings.Repeat("c", 64))
	if err != nil {
		t.Fatal(err)
	}
	wantMerges := []Merge{{100, 113}, {101, 259}, {259, 100}, {35, 260}, {103, 261}, {260, 261}}
	if artifact.Revision != "sha256:74042e6ca7d2fa613fd42934efa18dbc71167ca9060bdc29b1e6fc988b05020f" || !reflect.DeepEqual(artifact.Merges, wantMerges) {
		t.Fatalf("artifact fixture changed: %+v", artifact)
	}
}
