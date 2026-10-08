// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/record"
	waldotokenizer "github.com/openwaldo/waldo/internal/tokenizer"
)

func TestResolveTokenizerSpecUsesEmbeddedTrainedArtifact(t *testing.T) {
	artifact, err := waldotokenizer.TrainByteBPE(
		[]waldotokenizer.Sample{{ID: "one", Text: "hello hello world"}},
		300,
		1024,
		strings.Repeat("a", 64),
	)
	if err != nil {
		t.Fatal(err)
	}
	spec := TokenizerSpec{
		Name: artifact.Name, Revision: artifact.Revision,
		VocabularySize: artifact.VocabularySize,
		PadID:          artifact.PadID, BOSID: artifact.BOSID, EOSID: artifact.EOSID,
		Artifact: &artifact,
	}
	resolved, codec, err := ResolveTokenizerSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	text := "hello world"
	if resolved.Revision != artifact.Revision || codec.Decode(codec.Encode(text)) != text {
		t.Fatalf("resolved = %+v, round trip = %q", resolved, codec.Decode(codec.Encode(text)))
	}

	missing := spec
	missing.Artifact = nil
	if _, _, err := ResolveTokenizerSpec(missing); err == nil || !strings.Contains(err.Error(), "embedded artifact") {
		t.Fatalf("missing artifact error = %v", err)
	}
	mismatch := spec
	mismatch.VocabularySize++
	if _, _, err := ResolveTokenizerSpec(mismatch); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("mismatched artifact error = %v", err)
	}
}

func TestValidateArchitectureTokenizerAllowsDeclaredTrainingOnlyForForecast(t *testing.T) {
	raw := json.RawMessage(`{"vocabulary_size":16000,"tokenizer":{"training":{"algorithm":"byte-bpe-v1"}}}`)
	if err := ValidateArchitectureTokenizer(raw); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ResolveArchitectureTokenizer(raw); err == nil {
		t.Fatalf("unresolved execution error = %v", err)
	}
	invalid := json.RawMessage(`{"vocabulary_size":16000,"tokenizer":{"training":{"algorithm":"unknown"}}}`)
	if err := ValidateArchitectureTokenizer(invalid); err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("invalid training algorithm error = %v", err)
	}
}

func TestCL100KTokenizerRoundTripAndSpecialFraming(t *testing.T) {
	spec, codec, err := ResolveTokenizer("tiktoken/cl100k_base", TiktokenCL100KRevision, TiktokenCL100KVocabulary)
	if err != nil {
		t.Fatal(err)
	}
	text := "France: Paris — 日本"
	tokens := codec.Encode(text)
	if len(tokens) == 0 || codec.Decode(tokens) != text {
		t.Fatalf("round trip = %q through %v", codec.Decode(tokens), tokens)
	}
	if spec.PadID != 100256 || spec.BOSID != 100257 || spec.EOSID != 100258 || spec.VocabularySize != 100259 {
		t.Fatalf("special framing = %+v", spec)
	}
	for _, token := range tokens {
		if token >= spec.PadID {
			t.Fatalf("ordinary token %d overlaps special token range", token)
		}
	}
}

func TestR50KTokenizerRoundTripAndSpecialFraming(t *testing.T) {
	spec, codec, err := ResolveTokenizer("tiktoken/r50k_base", TiktokenR50KRevision, TiktokenR50KVocabulary)
	if err != nil {
		t.Fatal(err)
	}
	text := "Once upon a time, there was a compact language model."
	tokens := codec.Encode(text)
	if len(tokens) == 0 || codec.Decode(tokens) != text {
		t.Fatalf("round trip = %q through %v", codec.Decode(tokens), tokens)
	}
	if spec.PadID != 50256 || spec.BOSID != 50257 || spec.EOSID != 50258 || spec.VocabularySize != 50259 {
		t.Fatalf("special framing = %+v", spec)
	}
	for _, token := range tokens {
		if token >= spec.PadID {
			t.Fatalf("ordinary token %d overlaps special token range", token)
		}
	}
}

func TestTokenizedRecordSourceRemovesRawText(t *testing.T) {
	_, codec, err := ResolveTokenizer("tiktoken/cl100k_base", TiktokenCL100KRevision, TiktokenCL100KVocabulary)
	if err != nil {
		t.Fatal(err)
	}
	source := tokenizedRecordSource{source: staticRecordSource{{Text: "hello", Corpus: "example"}}, codec: codec}
	if err := source.Stream(t.Context(), func(record Record) error {
		if record.Text != "" || len(record.Tokens) == 0 || record.Corpus != "example" {
			t.Fatalf("tokenized record = %+v", record)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestAssistantResponseModelingMasksNonAssistantTurns(t *testing.T) {
	conversation := record.Conversation{Messages: []record.Message{{Role: "system", Content: "Be concise."}, {Role: "user", Content: "Hello"}, {Role: "assistant", Content: "Hi"}, {Role: "tool", Content: "ignored"}, {Role: "assistant", Content: "Done"}}}
	tokens, mask, err := (ConversationTransform{Template: ConversationTemplateUserAssistantV1, SupervisedRoles: []string{"assistant"}}).render(conversation, byteCodec{})
	if err != nil {
		t.Fatal(err)
	}
	if len(mask) != len(tokens) {
		t.Fatalf("mask framing = %d tokens, %d mask values", len(tokens), len(mask))
	}
	for index, token := range tokens {
		character := byte(token - 3)
		if mask[index] && !strings.Contains("HiDone", string(character)) {
			t.Fatalf("supervised non-assistant byte %q at %d", character, index)
		}
	}
}

func TestUserAssistantTemplateUsesInferenceCompletionBoundary(t *testing.T) {
	conversation := record.Conversation{Messages: []record.Message{{Role: "user", Content: "Hello"}, {Role: "assistant", Content: "Hi"}}}
	tokens, _, err := (ConversationTransform{Template: ConversationTemplateUserAssistantV1, SupervisedRoles: []string{"assistant"}}).render(conversation, byteCodec{})
	if err != nil {
		t.Fatal(err)
	}
	if rendered := (byteCodec{}).Decode(tokens); rendered != "User: Hello\n\nAssistant:Hi" {
		t.Fatalf("rendered conversation = %q", rendered)
	}
}

func TestConversationTransformPreservesContextAndToolsUntilTraining(t *testing.T) {
	conversation := record.Conversation{
		Messages: []record.Message{
			{Role: "user", Content: "Use the lookup.", Context: "Account 42"},
			{Role: "assistant", Content: "Done."},
		},
		Tools: []byte(`[{"name":"lookup"}]`),
	}
	tokens, mask, err := (ConversationTransform{Template: ConversationTemplateChatMLV1, SupervisedRoles: []string{"assistant"}, Tools: true}).render(conversation, byteCodec{})
	if err != nil {
		t.Fatal(err)
	}
	rendered := byteCodec{}.Decode(tokens)
	for _, expected := range []string{"Available tools:\n[{\"name\":\"lookup\"}]", "Use the lookup.\n\nAccount 42", "<|im_start|>assistant\nDone.<|im_end|>"} {
		if !strings.Contains(rendered, expected) {
			t.Fatalf("rendered conversation %q does not contain %q", rendered, expected)
		}
	}
	var supervised []byte
	for index, token := range tokens {
		if mask[index] {
			supervised = append(supervised, byte(token-3))
		}
	}
	if string(supervised) != "Done.<|im_end|>\n" {
		t.Fatalf("supervised content = %q", supervised)
	}
}

func TestConversationTransformRequiresPinnedToolContract(t *testing.T) {
	conversation := record.Conversation{
		Messages: []record.Message{{Role: "user", Content: "Use a tool"}, {Role: "assistant", Content: `{\"name\":\"lookup\"}`}},
		Tools:    json.RawMessage(`[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]`),
	}
	_, _, err := (ConversationTransform{Template: ConversationTemplateChatMLV1, SupervisedRoles: []string{"assistant"}}).render(conversation, byteCodec{})
	if err == nil || !strings.Contains(err.Error(), "interaction") {
		t.Fatalf("tool contract error = %v", err)
	}
	if err := (ConversationTransform{Template: ConversationTemplateChatMLV1, SupervisedRoles: []string{"tool"}, Tools: true}).Validate(); err == nil || !strings.Contains(err.Error(), "assistant") {
		t.Fatalf("tool supervision error = %v", err)
	}
}

func TestAssistantResponseModelingRejectsFlatText(t *testing.T) {
	_, _, err := tokenizeRecord(Record{Text: "User: hello\n\nAssistant: hi"}, byteCodec{}, "assistant-response-modeling", ConversationTransform{})
	if err == nil || !strings.Contains(err.Error(), "structured conversation") {
		t.Fatalf("flat-text error = %v", err)
	}
}

func TestAssistantResponseModelingRequiresMatchingSupervisedRole(t *testing.T) {
	conversation := record.Conversation{Messages: []record.Message{{Role: "user", Content: "Hello"}, {Role: "assistant", Content: "Hi"}}}
	_, _, err := tokenizeRecord(Record{Conversation: &conversation}, byteCodec{}, "assistant-response-modeling", ConversationTransform{Template: ConversationTemplateUserAssistantV1, SupervisedRoles: []string{"tool"}})
	if err == nil || !strings.Contains(err.Error(), "no targets") {
		t.Fatalf("missing target error = %v", err)
	}
}

func TestAssistantResponseModelingSupervisesFinalAssistantEOS(t *testing.T) {
	conversation := record.Conversation{Messages: []record.Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
	}}
	tokens, mask, err := tokenizeRecord(Record{Conversation: &conversation}, byteCodec{}, "assistant-response-modeling", ConversationTransform{Template: ConversationTemplateUserAssistantV1, SupervisedRoles: []string{"assistant"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(mask) != len(tokens)+1 || !mask[len(mask)-1] {
		t.Fatalf("assistant EOS target is not supervised: %d tokens, mask %v", len(tokens), mask)
	}
}

type staticRecordSource []Record

func (source staticRecordSource) Stream(ctx context.Context, consume func(Record) error) error {
	for _, record := range source {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := consume(record); err != nil {
			return err
		}
	}
	return nil
}
