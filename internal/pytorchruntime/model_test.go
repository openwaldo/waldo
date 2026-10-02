// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package pytorchruntime

import (
	"strings"
	"testing"
)

func TestSharedModelOwnsInferenceVisibleArchitectureSemantics(t *testing.T) {
	for _, required := range []string{
		`architecture.get("dropout", 0.0)`,
		`architecture.get("qk_normalization", False)`,
		`if self.qk_normalization:`,
		`self.residual_dropout(self.attention`,
		`self.residual_dropout(self.feed_forward`,
		`if self.activation_checkpointing and self.training:`,
	} {
		if !strings.Contains(modelSource, required) {
			t.Fatalf("shared PyTorch model does not contain %q", required)
		}
	}
}
