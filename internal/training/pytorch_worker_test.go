// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/pytorchruntime"
)

func TestPyTorchWorkerEvaluatesEveryExecutionBoundary(t *testing.T) {
	source := string(pyTorchWorker)
	if !strings.Contains(source, `WORKER_REVISION = "`+PyTorchRevision+`"`) || !strings.Contains(source, `TORCHTITAN_REVISION = "`+TorchTitanRevision+`"`) {
		t.Fatalf("embedded PyTorch worker revisions do not match the Go adapters")
	}
	for _, expected := range []string{
		`eager_compute_loss = self.evaluate_model(self.model, mixed_precision=True, compiled=False)`,
		`self.evaluate_model(self.model, mixed_precision=True, compiled=True)`,
		`if self.parameters.get("compile", False)`,
		`eager_float32_loss = self.evaluate_model(self.model, mixed_precision=False, compiled=False)`,
		`self.evaluate_model(artifact_model, mixed_precision=False, compiled=False)`,
		`"live_eager_compute_heldout_loss": eager_compute_loss`,
		`"live_eager_heldout_loss": eager_float32_loss`,
		`"compute_precision_loss_delta": eager_compute_loss - eager_float32_loss`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("PyTorch worker omits execution-boundary evaluation %q", expected)
		}
	}
}

func TestPyTorchTrainingUsesSharedInferenceModel(t *testing.T) {
	source := pytorchruntime.WithModel(pyTorchWorker)
	for _, expected := range []string{
		`WALDO_SHARED_DECODER_LM = DecoderLM`,
		`DecoderLM = WALDO_SHARED_DECODER_LM`,
		`architecture.get("qk_normalization", False)`,
		`if self.qk_normalization:`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("composed PyTorch worker omits shared model contract %q", expected)
		}
	}
}

func TestPyTorchWorkerFailsClosedOnArtifactDegradation(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`"live_compiled_heldout_loss": compiled_loss`,
		`"live_eager_heldout_loss": eager_float32_loss`,
		`"publishable_checkpoint_heldout_loss": artifact_loss`,
		`if abs(compiled_loss - eager_compute_loss) > compile_tolerance:`,
		`disable compile or correct compiled execution before spending more compute`,
		`if abs(eager_compute_loss - eager_float32_loss) > precision_tolerance:`,
		`correct compute-precision behavior before spending more compute`,
		`if abs(artifact_loss - eager_float32_loss) > tolerance:`,
		`"correct checkpoint serialization or target-dtype conversion before spending more compute"`,
		`if abs(artifact_loss - candidate_loss) > tolerance:`,
		`error_class = "artifact-integrity"`,
		`error_class = "numerical-integrity"`,
		`"value": candidate_loss`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("PyTorch worker omits fail-closed artifact behavior %q", expected)
		}
	}
}

func TestPyTorchWorkerFailsEarlyOnNumericalAndExecutionDrift(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`non-finite training loss at optimizer step`,
		`non-finite gradient norm at optimizer step`,
		`if self.parameters.get("compile", False):`,
		`safety_steps.update((min(100, self.target_steps), min(1000, self.target_steps)))`,
		`self.step_number in safety_steps`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("PyTorch worker omits early training guard %q", expected)
		}
	}
}

func TestPyTorchWorkerPublishesBestEvaluatedCheckpoint(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`selected_evaluation = min(candidates, key=lambda evaluation: evaluation["metrics"]["heldout_loss"])`,
		`load_safetensors(selected_path)`,
		`"selected_checkpoint": selection`,
		`selected checkpoint step {selected_evaluation['step']}`,
		`and not any(checkpoint["step"] == self.step_number for checkpoint in self.checkpoints):`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("PyTorch worker omits best-checkpoint publication behavior %q", expected)
		}
	}
}

func TestPyTorchWorkerSeparatesResumeAndPortableWeightPrecision(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`portable=False`,
		`self.parameter_dtype if portable else None`,
		`"storage_dtype": "float32" if storage_dtype is None else self.architecture["parameter_dtype"]`,
		`self.evaluate_weight_file(checkpoint_path, quantize=True)`,
		`self.evaluate_weight_file(weights_path, quantize=False)`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("PyTorch worker omits precision boundary %q", expected)
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

func TestTorchTitanWorkerSynchronizesCheckpointReplayAcrossNodes(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`restored checkpoint step {self.resume['step']}; waiting for all ranks`,
		`rank 0 restored checkpoint step {self.resume['step']}; waiting for {self.world_size - 1} other ranks`,
		`positioning each node's deterministic input stream at the checkpoint boundary`,
		`else "replaying the deterministic input stream"`,
		`torch.distributed.distributed_c10d._get_default_store()`,
		`store.wait(keys, datetime.timedelta(hours=6))`,
		`torch.distributed.barrier()`,
		`rendezvous_barrier(self.begin["run_id"], "checkpoint-replayed", self.rank, self.world_size)`,
		`checkpoint replay {replayed_steps}/{target_steps} optimizer steps complete and synchronized across all ranks`,
		`checkpoint replay {replayed_steps}/{target_steps} optimizer steps complete on rank 0`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("TorchTitan worker omits synchronized checkpoint replay behavior %q", expected)
		}
	}
	if strings.Contains(source, "replay_sync_micro_batches") {
		t.Fatal("TorchTitan worker still inserts periodic training-network barriers during checkpoint replay")
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

func TestTorchTitanWorkerUsesNodeLocalIPCForData(t *testing.T) {
	source := string(pyTorchWorker)
	for _, expected := range []string{
		`WALDO_TORCH_DATA_PLANE`,
		`source_rank = node_rank * local_world`,
		`socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)`,
		`connection.sendall(header)`,
		`receive_exact(connection, byte_count)`,
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
		`class MuonAdamW(torch.optim.Optimizer)`,
		`zeropower_via_newton_schulz5`,
		`schedule["name"] == "warmup-stable-warmdown"`,
		`WALDO_SHARED_DECODER_LM`,
		`"scaler": self.scaler.state_dict()`,
		`self.scaler.load_state_dict(runtime.get("scaler", {}))`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("PyTorch worker omits execution control %q", expected)
		}
	}
}
