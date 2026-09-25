// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/openwaldo/waldo/internal/model"
)

func TestTorchTitanHostfileScriptRendersValidConversationCompose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compose.yaml")
	command := exec.Command("./model-torchtitan-hostfile.sh", "--render-compose", "post-train/sft/waldo-project-v1", "4", path)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("render hostfile compose: %v\n%s", err, output)
	}
	compose, _, err := model.LoadCompose(path)
	if err != nil {
		t.Fatalf("load rendered hostfile compose: %v", err)
	}
	if compose.Interaction.Template != model.InteractionUserAssistantV1 || len(compose.Stages) != 2 {
		t.Fatalf("rendered interaction/stages = %+v / %d", compose.Interaction, len(compose.Stages))
	}
	for _, stage := range compose.Stages {
		if stage.Objective != "assistant-response-modeling" || stage.Conversation == nil || stage.Conversation.Template != model.InteractionUserAssistantV1 {
			t.Fatalf("rendered stage %s conversation contract = %q / %+v", stage.Name, stage.Objective, stage.Conversation)
		}
		if stage.Parameters.BatchSize != 4 || stage.Parameters.Compile {
			t.Fatalf("rendered stage %s parameters = %+v", stage.Name, stage.Parameters)
		}
	}
}
