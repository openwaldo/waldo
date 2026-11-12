# Qwen3 external trainer reference

This is the protocol-1 reference implementation of ADR 0084. It initializes
Qwen3 weights from scratch with Transformers 4.57.1 and PyTorch 2.9.0. It never
reads a corpus index, Parquet or raw text. WALDO supplies ordered, packed token
IDs, loss masks, evaluation records and exact optimizer-step budgets.

`waldo model train <name> compose.tiny.yaml` executes this example (see
`docs/TESTING.md`, "External trainer end-to-end"); `compose.yaml` selects
the `cuda` environment and the ~281M model. The `distributed` profile uses DDP
and Unix sockets for node-local stream fanout. Select it with
`--launch-profile distributed --processes-per-node 2`, or a hostfile. Restrict
CUDA visibility to one device when using the single-process profile.

## Recipe and execution

`qwen3_hf/schemas.py` is the source of all three probe schemas and supported
parameter defaults. There are no generated schema files or manifest schema
references. The deliberately bounded TrainingArguments surface supports AdamW,
constant/linear/cosine schedules, warmup, clipping, label smoothing, precision
and initialization seed. Unsupported options fail validation. Compilation is
reserved and currently accepts only `false`.

Micro-batch size, accumulation, checkpoint recomputation, attention implementation
and logging carry execution roles. Resolve chooses a divisible micro-batch from
the world shape; explicit micro-batch/accumulation overrides must satisfy the
same global batch. Precision never silently falls back. CPU supports FP32;
CUDA additionally supports FP16, BF16 when available, and TF32. Architecture
values complete the explicit Qwen3 defaults in Python; the full Qwen3Config is
written with the weights. Framework versions and source pins cover fixed
framework behavior outside the exposed parameter surface.

The labels have an ignored first position and preserve WALDO's target masks.
Trainer's `num_items_in_batch` counts supervised targets over the complete
accumulated batch; summed cross entropy uses that denominator. Input remains
bounded to the current optimizer batch and WALDO's bounded held-out set. No
framework shuffle, worker processes, tokenizer calls or data skipping occur.

Each attempt writes fresh checkpoint directories. Checkpoints contain FP32
weights, optimizer, scheduler, optional scaler, Python/NumPy/Torch/CUDA RNG,
consumption and complete evaluation/best-checkpoint history. RNG serialization
uses tensors and primitive values, and loading explicitly uses
`weights_only=True`; Transformers' optimizer/scheduler/scaler loaders also use
that mode. WALDO authenticates every member before resume. The framework never
loads its serialized `training_args.bin`; it rebuilds arguments from `begin`.

Every evaluated candidate and the terminal step are checkpointed, including
candidates between periodic saves. The best weights are retained inside each
resume bundle. Publication reloads the best weights, writes Safetensors and
JSON only, reloads the publication, and reports `artifact_evaluation`. Terminal
resume supports zero remaining sequences. The caller must retain WALDO's prior
checkpoint references and evaluations as required by the checkpoint contract.

## Assets and environments

The committed tokenizer assets and their upstream revision/license are recorded
in `tokenizer/README.md`. Vocabulary 151936 is Qwen3's padded embedding size;
pad/BOS are 151643 and EOS is 151645. Ordinary text never recognizes special
spellings or receives inserted BOS/EOS tokens.

All manifests require Python **3.12.3**. The host interpreter needs pip and venv
for WALDO's managed environment creation. The `cpu` lock includes cross-platform
wheel alternatives; the CPU qualification below was run on Linux aarch64.
`cuda` is one universal Linux aarch64/x86_64 CUDA 13.0 lock.
It is also available to the tokenizer when sharing
the trainer environment is useful. `tokenizer` contains the
codec without PyTorch. WALDO installs hash-verified wheels and revalidates the
read-only environment inventory. The locks are authoritative; `requirements.in`
files are only regeneration inputs. The exact uv commands are in lock headers.
Lock regeneration changes model identity and requires requalification.

## Developer qualification

`compose.tiny.yaml` has two 32-wide layers, 4,880,608 parameters and three
optimizer steps. `compose.yaml` has ten 1024-wide layers, tied embeddings,
281,434,368 parameters and a two-step smoke budget. Both use sequence length 8
and global batch 4 so the same small fixture can qualify them. These budgets
exercise the interfaces; they do not train a useful model. The compose source
pins the implementation commit in the WALDO fork while this PoC is under review.

The harness reads compose architecture, data and trainer values using WALDO's
compose reader and data resolver. It supplies synthetic prepared sequences
with two corpus labels and unequal target masks, rather than selecting a real
corpus. Source and environment verification, tokenizer serving, probe/resolve,
process isolation, checkpoint authentication and output validation all use the
real developer APIs. Fixture run identity hashes are used; this is not a
production model/run BOM writer.

Build `testing/external-qwen3` with Go 1.25 (or build the Linux binary in the
project's Go container). Run on the host with Python 3.12.3 and the GPU driver:

```sh
go build -o qwen-qualify ./testing/external-qwen3
CUBLAS_WORKSPACE_CONFIG=:4096:8 ./qwen-qualify \
  -repository "$PWD" -commit a1011336c464cc3d951f3b694cf3f96f639aee77 \
  -environment cpu -cache "$PWD/.qualification-cache" \
  -output "$PWD/.qualification-cpu" \
  -compose examples/trainers/qwen3-hf/compose.tiny.yaml
```

Repeat with `-environment cuda` and a fresh output directory. For the large
smoke, add `-large -compose examples/trainers/qwen3-hf/compose.yaml`. The harness
requires an absent output directory and leaves evidence/checkpoints there.
Repository/commit and environment flags explicitly select the application
under test; compose pins are the copyable example defaults. Network access is
needed for the initial source/environment acquisition only.

Tiny qualification runs the reusable `trainerapp.CheckConformance` harness:
three-step control, interruption after durable step 1, resumed completion, and
micro-batch 4 versus micro-batch 2/accumulation 2. It requires byte-identical
terminal weights and compares all three progress losses with absolute and
relative tolerances of 1e-5. Both tiny and large also test held-out selection,
reloaded artifact evaluation and terminal-step resume. Results are JSON on
stdout; setup/errors go to stderr. Framework/device-specific qualification
results are recorded on WALDO-49.

Dependency-free schema/wire tests run in `testing/python.sh`. The real-device
harness is explicit and fails on missing prerequisites; generic CI does not
claim a reference-device pass when PyTorch/GPU tests are skipped.

Qualification of implementation `a1011336c464cc3d951f3b694cf3f96f639aee77`
on Spark GB10, Linux aarch64, Python 3.12.3, driver 580.173.02:

| Environment | Tiny resume | Compared steps | Maximum loss delta |
| --- | --- | --- | --- |
| CPU, Torch 2.9.0+cpu | byte-identical | 3 | 4.76837158203125e-7 |
| CUDA, Torch 2.9.0+cu130 | byte-identical | 3 | 1.3113021850585938e-6 |

Both passed held-out publication and terminal resume. The 281M CUDA compose
also completed two steps and terminal resume, with verified published held-out
loss 9.024673461914062. These are FP32 fixture checks, not model-quality,
cross-device byte equality, other-precision or distributed qualifications.
Torch 2.9 emits a compute-capability range warning for GB10 (12.1); the actual
CUDA runs above completed successfully.


## Two-GPU qualification

The CLI lifecycle was exercised on two RTX 4060 Ti 16 GB GPUs on Linux x86_64,
with Python 3.12.3 and the former architecture-specific lock (now merged into `cuda`): uninterrupted training, interruption
after a verified checkpoint, same-run resume, byte-identical published weights,
completed no-ops, and verified export passed. Reproduce it with the opt-in
`testing/e2e/model-external-distributed.sh` described in `docs/TESTING.md`.
This does not claim cross-world-size determinism or physical multi-node
qualification. Hostfile preflight retains the native requirement for matching
runtime and accelerator topology on every node.

## Portable plan identity

The stage `trainer` block contains identity-bearing recipe inputs (`bf16`, `fp16`,
optimizer, learning rate, schedule, seeds and warmup). Put execution overrides in
stage `execution`: `per_device_train_batch_size`, `gradient_accumulation_steps`,
`attn_implementation`, `gradient_checkpointing`, `torch_compile`, `logging_steps`
and `report_to`. The probe schema declares these with `x-waldo-role: execution`;
WALDO rejects values in the wrong block. Compilation remains disabled until
numerically qualified. Global batch size stays in `data` and must equal
micro-batch × accumulation × world size.

`trainer.determinism` is retained but excluded from plan identity. Interpreter
bytes, platform, selected wheels and hardware observations are run evidence;
Python version and the universal lock digest are identity. Resume requires the
same plan and exact execution config, including determinism and environment
evidence. GPU/driver/CUDA/kernel changes are recorded per attempt, not enforced.
Equal plans describe equivalent specifications, not necessarily identical weights;
weights are identified by artifact digest. See ADR 0089 for the unverified
bitwise/statistical reproducibility claim levels.
