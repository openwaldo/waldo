// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package modelexport

import (
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/model"
)

func TestStandardLlamaExportsRejectQKNormalization(t *testing.T) {
	architecture := model.Architecture{QKNormalization: true}
	for _, format := range []string{"huggingface", "mlx", "GGUF/Ollama"} {
		if err := validateStandardLlamaArchitecture(architecture, format); err == nil || !strings.Contains(err.Error(), "cannot preserve WALDO qk_normalization") {
			t.Fatalf("%s error = %v", format, err)
		}
	}
	architecture.QKNormalization = false
	if err := validateStandardLlamaArchitecture(architecture, "huggingface"); err != nil {
		t.Fatalf("standard architecture rejected: %v", err)
	}
}
