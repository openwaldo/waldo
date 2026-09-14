// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"regexp"
	"strings"
	"testing"
)

// The artifact check in ADR 0044 exists to detect serialization, precision, and
// loader faults in the persisted weights. It can only measure those if the
// reloaded model is evaluated exactly as the live model was. Evaluating the two
// at different compute precisions instead measures autocast rounding, which
// accumulates with depth until it exceeds the tolerance for every sufficiently
// large model while revealing nothing about the artifact.
func TestPyTorchWorkerEvaluatesArtifactAtLiveEvaluationPrecision(t *testing.T) {
	pattern := regexp.MustCompile(`evaluate_model\([^)]*mixed_precision=(True|False)\)`)
	matches := pattern.FindAllStringSubmatch(string(pyTorchWorker), -1)
	if len(matches) < 2 {
		t.Fatalf("expected the worker to evaluate both the live and reloaded model, found %d call(s)", len(matches))
	}
	for _, match := range matches[1:] {
		if match[1] != matches[0][1] {
			t.Fatalf("held-out evaluations disagree on compute precision: %q vs %q", matches[0][0], match[0])
		}
	}
}

func TestTorchTitanWorkerPartitionsGlobalBatchAcrossRanks(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`self.global_micro_batch_size = self.global_batch_size // self.gradient_accumulation_steps`,
		`self.batch_size = self.global_micro_batch_size // self.world_size`,
		`owner = self.sequence_number % self.world_size`,
		`if owner == self.rank:`,
		`if self.sequence_number % self.global_micro_batch_size == 0:`,
		`torch.distributed.all_reduce(global_valid_tokens`,
		`torch.distributed.all_reduce(global_loss_sum`,
		`torch.distributed.all_gather_object(states, local)`,
		`distinct sequences per micro-batch`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("TorchTitan worker omits distributed batch behavior %q", expected)
		}
	}
}

func TestTorchTitanWorkerKeepsSingleNodeBatchSemantics(t *testing.T) {
	source := string(pyTorchWorker)
	if !strings.Contains(source, `else:
            self.batch_size = self.global_micro_batch_size`) {
		t.Fatal("PyTorch worker no longer uses the global micro-batch on a single process")
	}
}

func TestPyTorchWorkerAccumulatesTokenNormalizedGradients(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`self.optimizer.zero_grad(set_to_none=True)`,
		`if self.accumulation_number < self.gradient_accumulation_steps and not final:`,
		`parameter.grad.div_(self.accumulated_tokens)`,
		`self.optimizer.step()`,
		`self.replay_micro_batches = self.resume["step"] * self.gradient_accumulation_steps`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("PyTorch worker omits gradient accumulation behavior %q", expected)
		}
	}
}

func TestTorchTitanWorkerCompletesPartialFinalGlobalBatch(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`if max(pending) > 0:`,
		`missing = self.batch_size - len(self.batch)`,
		`zero_mask = [0.0] * self.sequence_length`,
		`self.batch.extend((padding, zero_mask, {}) for _ in range(missing))`,
		`empty slots contribute zero loss`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("TorchTitan worker omits final partial-batch behavior %q", expected)
		}
	}
	if strings.Contains(source, `if min(pending) > 0:`) {
		t.Fatal("TorchTitan worker still discards a final batch when one rank has no real sequence")
	}
}

func TestTorchTitanWorkerKeepsNodeLocalDataOffTrainingNetwork(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`WALDO_TORCH_DATA_PLANE`,
		`torch.distributed.new_group(ranks=ranks)`,
		`source_rank = node_rank * local_world`,
		`group=stream_group`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("TorchTitan worker omits node-local stream behavior %q", expected)
		}
	}
}

func TestPyTorchWorkerPinsMemoryAndPrecisionControls(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`checkpoint(layer, value, use_reentrant=False)`,
		`torch.compile(self.model, dynamic=False)`,
		`self.compute_dtype == torch.float16`,
		`self.scaler.scale(loss).backward()`,
		`self.scaler.unscale_(self.optimizer)`,
		`self.scaler.step(self.optimizer)`,
		`tokens.pin_memory().to(self.device, non_blocking=True)`,
		`mask.pin_memory().to(self.device, non_blocking=True)`,
		`"scaler": self.scaler.state_dict()`,
		`self.scaler.load_state_dict(runtime.get("scaler", {}))`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("PyTorch worker omits execution control %q", expected)
		}
	}
}
