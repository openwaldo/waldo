// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openwaldo/waldo/internal/record"
)

const (
	ConversationTemplateUserAssistantV1 = "user-assistant-v1"
	ConversationTemplateChatMLV1        = "chatml-v1"
)

// ConversationTransform is the versioned, model-side transformation from a
// canonical conversation to token targets. It belongs to a training stage,
// never an ingestion recipe or corpus manifest.
type ConversationTransform struct {
	Template        string   `json:"template" yaml:"template"`
	SupervisedRoles []string `json:"supervised_roles" yaml:"supervised_roles"`
	Tools           bool     `json:"tools,omitempty" yaml:"tools,omitempty"`
}

func (transform ConversationTransform) Validate() error {
	if transform.Template != ConversationTemplateUserAssistantV1 && transform.Template != ConversationTemplateChatMLV1 {
		return fmt.Errorf("unsupported conversation template %q", transform.Template)
	}
	if len(transform.SupervisedRoles) == 0 {
		return fmt.Errorf("conversation supervised_roles is required")
	}
	seen := map[string]bool{}
	for _, role := range transform.SupervisedRoles {
		if role != "system" && role != "user" && role != "assistant" && role != "tool" || seen[role] {
			return fmt.Errorf("invalid or duplicate supervised conversation role %q", role)
		}
		seen[role] = true
	}
	if transform.Tools && !seen["assistant"] {
		return fmt.Errorf("tool conversation requires assistant in supervised_roles")
	}
	return nil
}

func (transform ConversationTransform) supervised(role string) bool {
	for _, candidate := range transform.SupervisedRoles {
		if candidate == role {
			return true
		}
	}
	return false
}

func (transform ConversationTransform) render(conversation record.Conversation, codec TokenCodec) ([]int, []bool, error) {
	if err := transform.Validate(); err != nil {
		return nil, nil, err
	}
	if err := conversation.Validate(); err != nil {
		return nil, nil, err
	}
	if len(conversation.Tools) > 0 && !transform.Tools {
		return nil, nil, fmt.Errorf("conversation contains tools but the model interaction does not enable tools")
	}
	messages := append([]record.Message(nil), conversation.Messages...)
	if len(conversation.Tools) > 0 {
		tools := string(conversation.Tools)
		var encodedString string
		if json.Unmarshal(conversation.Tools, &encodedString) == nil {
			tools = encodedString
		}
		toolMessage := "Available tools:\n" + tools
		if len(messages) > 0 && messages[0].Role == "system" {
			messages[0].Content = strings.TrimRight(messages[0].Content, "\n") + "\n\n" + toolMessage
		} else {
			messages = append([]record.Message{{Role: "system", Content: toolMessage}}, messages...)
		}
	}
	var tokens []int
	var mask []bool
	var encodeErr error
	appendPart := func(value string, supervised bool) {
		if encodeErr != nil {
			return
		}
		var part []int
		part, encodeErr = encodeTokens(codec, value)
		tokens = append(tokens, part...)
		for range part {
			mask = append(mask, supervised)
		}
	}
	for position, message := range messages {
		supervised := transform.supervised(message.Role)
		content := message.Content
		if message.Context != "" {
			content += "\n\n" + message.Context
		}
		switch transform.Template {
		case ConversationTemplateUserAssistantV1:
			if position > 0 {
				appendPart("\n\n", false)
			}
			marker := strings.ToUpper(message.Role[:1]) + message.Role[1:] + ":"
			if message.Role != "assistant" {
				marker += " "
			}
			appendPart(marker, false)
			appendPart(content, supervised)
		case ConversationTemplateChatMLV1:
			appendPart("<|im_start|>"+message.Role+"\n", false)
			appendPart(content, supervised)
			appendPart("<|im_end|>\n", supervised)
		}
	}
	if encodeErr != nil {
		return nil, nil, encodeErr
	}
	if len(tokens) == 0 {
		return nil, nil, fmt.Errorf("conversation transformation produced no tokens")
	}
	return tokens, mask, nil
}

func tokenizeRecord(record Record, codec TokenCodec, objective string, transform ConversationTransform) ([]int, []bool, error) {
	if record.decodeErr != nil {
		return nil, nil, record.decodeErr
	}
	if record.Conversation != nil {
		tokens, mask, err := transform.render(*record.Conversation, codec)
		if err != nil {
			return nil, nil, err
		}
		if objective == "causal-language-modeling" {
			for index := range mask {
				mask[index] = true
			}
		}
		lastRole := record.Conversation.Messages[len(record.Conversation.Messages)-1].Role
		mask = append(mask, objective == "causal-language-modeling" || transform.supervised(lastRole))
		if objective == "assistant-response-modeling" {
			hasTarget := false
			for _, supervised := range mask {
				hasTarget = hasTarget || supervised
			}
			if !hasTarget {
				return nil, nil, fmt.Errorf("conversation contains no targets for supervised roles")
			}
		}
		return tokens, mask, nil
	}
	if objective == "assistant-response-modeling" {
		return nil, nil, fmt.Errorf("assistant-response-modeling requires structured conversation records")
	}
	tokens, err := encodeTokens(codec, record.Text)
	if err != nil {
		return nil, nil, err
	}
	mask := make([]bool, len(tokens)+1)
	for index := range mask {
		mask[index] = true
	}
	return tokens, mask, nil
}
