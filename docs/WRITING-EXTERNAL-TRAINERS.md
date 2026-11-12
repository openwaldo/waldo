# Writing an external trainer

An external trainer is a user-supplied training program that WALDO pins by git
commit, runs in a pinned Python environment, and records as part of the model's
identity. Start from the reference implementation,
[`examples/trainers/qwen3-hf`](../examples/trainers/qwen3-hf/README.md)
(Qwen3 from scratch with Hugging Face Transformers), and read
ADR [0084](adr/0084-external-trainers.md) for the design.

## What you provide

- `waldo-trainer.yaml`: `kind`, `schema`, `name`, `protocol: 1`, `entrypoint`
  (`module` or `script`), `commands` (`probe`, `resolve`, `train`),
  `launch_profiles` (`single-process` or `distributed`),
  `objectives`, `artifact_profile` (for example `hf-causal-lm`), `schemas`
  (JSON Schema for `architecture` and trainer `parameters`) and named
  `environments` (`kind: python-lock` with a hash-pinned lock file).
- `waldo-tokenizer.yaml` with `probe` and `serve` commands. The tokenizer is a
  separate application that WALDO drives for all tokenization; it may live in
  the same repository and commit as the trainer.
- A compose (schema 2) that pins both. See
  [Model compose](MODEL-COMPOSE.md#external-trainer-poc).

## What the trainer must do

WALDO launches each command with `cwd` set to the read-only checkout, a clean
environment plus a recorded allowlist, and stdin/stdout carrying NDJSON frames
only (stderr is captured). The full frame definitions are in
[External trainer protocol 1](EXTERNAL-TRAINER-PROTOCOL.md).

- `probe`: report capabilities, frameworks, devices, driver and CUDA, and the
  schemas. The probe is the only schema source.
- `resolve`: complete the stage parameters and choose execution values.
  micro-batch x accumulation x world size must equal `global_batch_sequences`.
- `train`: read `begin`, then consume WALDO's packed token IDs and loss masks.
  Never read the corpus index or Parquet, and never tokenize. Emit `event`
  frames (progress, log, checkpoint, evaluation with held-out NLL sums and
  counts) and finish with `complete` (artifacts with roles, consumption per
  corpus, evaluations, selected checkpoint, environment evidence) or `error`.
- Resume is required. Write checkpoints WALDO can authenticate
  ([External checkpoints](EXTERNAL-CHECKPOINTS.md)), load them with
  `weights_only=True`, and keep pickle out of published artifacts.
- Honor `determinism: required` (for example deterministic CUDA algorithms);
  WALDO's setting alone does not enforce it.

WALDO verifies the result before publishing it; see
[External trainer lifecycle](MODEL-LIFECYCLE.md#external-trainer-lifecycle).

## Testing your trainer

`trainerapp.CheckConformance` compares an interrupted-and-resumed run with an
uninterrupted one and checks decomposition across micro-batch sizes; the
reference harness is `testing/external-qwen3`. For a full CLI run use the
[external trainer end-to-end test](TESTING.md#external-trainer-end-to-end).

## Distributed trainers

Declare `distributed` in both the manifest and probe. Probe and resolve still
run as one Python process; train runs through the pinned environment's
`python -E -s -m torch.distributed.run`. Single-node launches use `--standalone`;
multi-node launches pin node count, rank, master address and port with restarts
disabled. Resolve must preserve the recipe across world shapes and return a
batch decomposition satisfying micro-batch × accumulation × world = global
batch.

Only local rank zero reads stdin. Its coordinator sends `begin`, evaluation
records, its node's sequence records, `micro_batch_end` frames, and `end`.
A boundary carries `ordinal`, the exclusive end of the global micro-batch.
Sequence `ordinal % world_size` determines the owning global rank. Consume the
whole local micro-batch before entering collectives; a leader entering a
collective before forwarding a sibling's records can deadlock the job. The
Qwen3 reference uses Unix sockets for local fanout. No training sequences cross
nodes. Each node tokenizes the same canonical input independently.

Only global rank zero emits protocol events/completion and publishes files.
Aggregate loss denominators and corpus consumption across all ranks, preserve
each rank's RNG state in private checkpoint state, and restore the verified
bundle on every node. Do not average per-rank mean losses when supervised-token
counts differ. The Qwen3 reference supports DDP with the universal `cuda` lock for
x86_64 and `cuda` for aarch64.
