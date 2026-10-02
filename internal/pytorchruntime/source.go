// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

// Package pytorchruntime owns the Python model definition shared by WALDO's
// PyTorch/TorchTitan training and PyTorch inference workers.
package pytorchruntime

import _ "embed"

//go:embed model.py
var modelSource string

func WithModel(worker []byte) string {
	return modelSource + "\n" + string(worker)
}
