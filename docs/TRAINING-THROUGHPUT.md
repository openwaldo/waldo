# Training throughput qualification

WALDO includes a fixed-input DDP benchmark for measuring training changes
without reading a corpus or modifying a model run:

```console
torchrun --standalone --nproc-per-node=4 \
  testing/python/pytorch_throughput_benchmark.py \
  --model-source internal/pytorchruntime/model.py
```

The default workload is the 1.034B-parameter, 2048-token rung architecture
with global batch 8. It runs the same synthetic global batches through these
variants:

1. the legacy accumulation-2 path;
2. accumulation 1, which gives each of four ranks two sequences per microbatch;
3. accumulation 2 with DDP `no_sync()` on the first microbatch;
4. one loss/token collective at the optimizer boundary;
5. foreach gradient processing and AdamW;
6. fused CUDA AdamW; and
7. compiled accumulation-1 execution.

Each JSON result reports optimizer-step duration, tokens/sec, achieved
TFLOP/sec, and forward, backward, scalar-communication, gradient-processing,
and optimizer time. DDP gradient communication remains nested in backward.
The benchmark intentionally does not install a custom DDP communication hook:
doing so changes the reducer path being qualified and can alter gradient
accumulation under `no_sync()`. Input creation and host-to-device transfer are
outside the timed region.

Every variant begins from identical weights and consumes identical sequences.
Its final weights are compared with the legacy result. The benchmark fails if
the FP32 master weights differ by more than `2e-5`, or `2e-4` for a different
microbatch shape or compiled execution. These are qualification tolerances,
not permission to change training semantics.

For a quick wiring check, add `--tiny --sequence-length 128`. Benchmark results
must be recorded from the intended GPU topology; GPU utilization alone is not
evidence of an improvement. `torch.compile` remains opt-in and must not be
enabled for a compose until its loss, gradients, updates, checkpoint reload,
and deterministic generation satisfy the repository conformance gates.
