// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"context"
	"io"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/inference"
	"github.com/openwaldo/waldo/internal/model"
)

func TestParseModelChatSupportsOneShotGenerationOptions(t *testing.T) {
	context, args, err := parseCobraCommand(t, []string{"model", "chat"}, []string{"foo", "hello world", "--max-tokens", "12", "--temperature", "0", "--top-p", "1", "--seed", "9", "--run-id", "run1", "--raw"})
	if err != nil {
		t.Fatal(err)
	}
	name, prompt, options, runID, raw, err := cobraModelChatOptions(context, args)
	if err != nil {
		t.Fatal(err)
	}
	if name != "foo" || prompt == nil || *prompt != "hello world" || options.MaxTokens != 12 || options.Temperature != 0 || options.TopP != 1 || options.Seed == nil || *options.Seed != 9 || runID != "run1" || !raw {
		t.Fatalf("name = %q, prompt = %v, options = %+v, run = %q, raw = %t", name, prompt, options, runID, raw)
	}
	context, args, err = parseCobraCommand(t, []string{"model", "chat"}, []string{"foo", "--temperature", "-1"})
	if _, _, _, _, _, err := cobraModelChatOptions(context, args); err == nil {
		t.Fatal("negative temperature accepted")
	}
	context, args, err = parseCobraCommand(t, []string{"model", "chat"}, []string{"foo", "--top-p", "NaN"})
	if _, _, _, _, _, err := cobraModelChatOptions(context, args); err == nil {
		t.Fatal("NaN top-p accepted")
	}
}

func TestSelectModelChatRunRequiresCompletedRealRun(t *testing.T) {
	inspection := model.Inspection{Model: model.ModelRecord{Name: "example"}, BOM: model.ModelBOM{
		CurrentRunID: "current",
		Runs: []model.ModelBOMRun{
			{ID: "complete", State: model.RunComplete},
			{ID: "running", State: model.RunRunning},
			{ID: "simulated", State: model.RunComplete, Simulated: true},
		},
	}}
	selected, err := selectModelChatRun(inspection, "complete")
	if err != nil {
		t.Fatal(err)
	}
	if selected.BOM.CurrentRunID != "complete" || inspection.BOM.CurrentRunID != "current" {
		t.Fatalf("selected current run = %q, original = %q", selected.BOM.CurrentRunID, inspection.BOM.CurrentRunID)
	}
	for _, runID := range []string{"running", "simulated", "missing"} {
		if _, err := selectModelChatRun(inspection, runID); err == nil {
			t.Fatalf("run %q accepted", runID)
		}
	}
}

func TestOneShotChatRendersSafeMarkdownAndReturnsJSON(t *testing.T) {
	opened := inference.Opened{Description: inference.Description{Model: "foo", RunID: "run"}, Session: &chatSession{data: []byte{'A', 0x1b, 0xff}}}
	options := inference.Options{MaxTokens: 3, Temperature: 0, TopP: 1}
	var output bytes.Buffer
	if err := runOneShotChat(Context{Execution: context.Background()}, opened, model.Interaction{}, "hello", options, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "A\\x1b\\xff") {
		t.Fatalf("safe output = %q", output.String())
	}
	output.Reset()
	if err := runOneShotChat(Context{Execution: context.Background(), JSON: true}, opened, model.Interaction{}, "hello", options, &output); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"model": "foo"`, `"prompt": "hello"`, `"finish_reason": "max_tokens"`} {
		if !strings.Contains(output.String(), want) {
			t.Fatalf("JSON = %s", output.String())
		}
	}
}

func TestOneShotChatRendersMarkdownAtTerminalWidth(t *testing.T) {
	previousWidth := terminalOutputWidth
	terminalOutputWidth = func() int { return 60 }
	t.Cleanup(func() { terminalOutputWidth = previousWidth })
	markdown := "## Answer\n\n- first item\n- second item\n\n" + strings.Repeat("wrapped words ", 12)
	opened := inference.Opened{Description: inference.Description{Model: "foo"}, Session: &chatSession{data: []byte(markdown)}}
	var output bytes.Buffer
	if err := runOneShotChat(Context{Execution: context.Background()}, opened, model.Interaction{}, "hello", inference.Options{}, &output); err != nil {
		t.Fatal(err)
	}
	rendered := output.String()
	if !strings.Contains(rendered, "## Answer") || !strings.Contains(rendered, "• first item") || !strings.Contains(rendered, "second item") {
		t.Fatalf("Markdown was not rendered: %q", rendered)
	}
	for _, line := range strings.Split(rendered, "\n") {
		if len([]rune(line)) > 60 {
			t.Fatalf("rendered line exceeds width: %d: %q", len([]rune(line)), line)
		}
	}
}

func TestOneShotChatStreamsMarkdownOnTerminal(t *testing.T) {
	previousLive := terminalMarkdownLive
	terminalMarkdownLive = func(io.Writer) bool { return true }
	t.Cleanup(func() { terminalMarkdownLive = previousLive })
	markdown := "## Answer\n\n- first item\n- second item\n\n" + strings.Repeat("streamed words ", 12)
	opened := inference.Opened{Description: inference.Description{Model: "foo"}, Session: &chatSession{data: []byte(markdown)}}
	var output bytes.Buffer
	if err := runOneShotChat(Context{Execution: context.Background()}, opened, model.Interaction{}, "hello", inference.Options{}, &output); err != nil {
		t.Fatal(err)
	}
	rendered := output.String()
	if !strings.Contains(rendered, "\x1b[") || !strings.Contains(rendered, "• first item") || !strings.Contains(rendered, "streamed words") {
		t.Fatalf("live Markdown output = %q", rendered)
	}
}

func TestSafeTokenWriterNormalizesCRLFAndEscapesLoneCarriageReturn(t *testing.T) {
	var output bytes.Buffer
	renderer := safeTokenWriter{writer: &output}
	for _, piece := range [][]byte{[]byte("first\r"), []byte("\nsecond\r"), []byte("third")} {
		if err := renderer.Write(piece); err != nil {
			t.Fatal(err)
		}
	}
	if err := renderer.Flush(); err != nil {
		t.Fatal(err)
	}
	if output.String() != "first\nsecond\\x0dthird" {
		t.Fatalf("rendered output = %q", output.String())
	}
}

func TestInteractiveChatMaintainsAndClearsContext(t *testing.T) {
	session := &chatSession{data: []byte(" answer")}
	opened := inference.Opened{Description: inference.Description{Model: "foo", Backend: "mlx", ContextTokens: 512}, Session: session}
	previous := modelChatInput
	modelChatInput = strings.NewReader("first\n/clear\nsecond\n/exit\n")
	defer func() { modelChatInput = previous }()
	var output bytes.Buffer
	if err := runInteractiveChat(context.Background(), opened, model.Interaction{}, inference.Options{MaxTokens: 2, Temperature: 0, TopP: 1}, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "raw causal continuation") || len(session.prompts) != 2 || session.prompts[0] != "first" || session.prompts[1] != "second" {
		t.Fatalf("output = %q, prompts = %#v", output.String(), session.prompts)
	}
}

func TestInteractiveConversationFormatsTurnsAndStopsAtNextUser(t *testing.T) {
	session := &chatSession{data: []byte(" First answer\n\nUser: invented turn")}
	opened := inference.Opened{Description: inference.Description{Model: "foo", Backend: "pytorch", ContextTokens: 2048}, Session: session}
	previous := modelChatInput
	modelChatInput = strings.NewReader("first\nsecond\n/exit\n")
	defer func() { modelChatInput = previous }()
	var output bytes.Buffer
	interaction := model.Interaction{Template: model.InteractionUserAssistantV1}
	if err := runInteractiveChat(context.Background(), opened, interaction, inference.Options{MaxTokens: 64, Temperature: 0, TopP: 1}, &output); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(output.String(), "invented turn") || !strings.Contains(output.String(), "user/assistant conversation") {
		t.Fatalf("output = %q", output.String())
	}
	if len(session.prompts) != 2 || session.prompts[0] != "User: first\n\nAssistant:" {
		t.Fatalf("prompts = %#v", session.prompts)
	}
	wantSecond := "User: first\n\nAssistant: First answer\n\nUser: second\n\nAssistant:"
	if session.prompts[1] != wantSecond {
		t.Fatalf("second prompt = %q, want %q", session.prompts[1], wantSecond)
	}
}

func TestBoundChatHistoryUsesByteTokenizerContext(t *testing.T) {
	if got := boundChatHistory("123456789", 4); got != "6789" {
		t.Fatalf("bounded history = %q", got)
	}
}

type chatSession struct {
	data    []byte
	prompts []string
}

func (session *chatSession) Generate(_ context.Context, prompt string, _ inference.Options, emit func(inference.Token) error) (inference.Result, error) {
	session.prompts = append(session.prompts, prompt)
	for _, value := range session.data {
		if emit != nil {
			if err := emit(inference.Token{Bytes: []byte{value}}); err != nil {
				return inference.Result{}, err
			}
		}
	}
	return inference.Result{Text: strings.ToValidUTF8(string(session.data), "�"), Tokens: len(session.data), FinishReason: "max_tokens", DurationMS: 1}, nil
}

func (*chatSession) Close() error { return nil }
