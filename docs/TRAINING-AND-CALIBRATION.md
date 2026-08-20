# Training, tuning, and quantization calibration

These operations can all consume text, but they do fundamentally different
work. WALDO records them separately so that a release never describes
calibration as training or implies that a model learned from data it only used
to measure numerical sensitivity.

| Operation | Changes weights | Uses gradients and an optimizer | Main purpose |
| --- | --- | --- | --- |
| Pre-training | Yes | Yes | Learn a model from a broad corpus, usually from a blank architecture. |
| Continued pre-training | Yes | Yes | Continue the same causal-language objective from existing open weights. |
| Supervised fine-tuning (SFT) | Yes | Yes | Learn desired responses from prompt/response examples. |
| Preference tuning | Yes | Usually | Favor preferred responses over rejected responses. |
| Evaluation | No | No | Measure model behavior on held-out tasks. |
| Quantization calibration | No | No | Measure which activations and weights are most sensitive before reducing weight precision. |
| Quantization | Yes, numerically | No | Encode existing weights with fewer bits for smaller, faster inference. It does not teach new behavior. |

## What WALDO trains today

`waldo model train <name> <index-path...> --epochs <n>` performs causal
language-model training over canonical records selected from a WALDO index.
Starting from a blank architecture is pre-training. Starting from pulled or
previously trained weights is continued pre-training. Both update weights with
gradients and an optimizer, and both create an immutable run BOM followed by
observations and output artifact hashes.

SFT and preference objectives are deliberately not inferred from ordinary
text corpora. They require their own record contract, objective, evaluation,
and pinned chat-template behavior before WALDO can describe them honestly.

## What calibration does

Calibration is an optional part of a quantized GGUF or Ollama export:

```bash
waldo model export small ./small-q4 \
  --format gguf \
  --quant 4 \
  --calibration core/books
```

`core/books` is a WALDO index selection, not a raw file and not a second
training run. WALDO resolves it recursively, builds and validates its corpus
BOM, retrieves selected Parquet shards through the hash-verifying cache,
audits each selected shard, and writes a deterministic bounded text sample.
The default sample budget is 100,000 tokens under WALDO's current byte-token
model contract, with seed 42.

Selection is deterministic: WALDO orders unique shard pins from their SHA-256
and the seed, then streams records until the budget is full. It therefore does
not download or scan a very large corpus in full. The upstream llama.cpp
importance-matrix tool runs forward measurements over that sample; it does not
backpropagate, run an optimizer, or modify the source weights. The subsequent
quantizer uses those measurements while encoding the lower-precision release.

The release records:

- the requested simple quant level and exact llama.cpp recipe;
- SHA-256 and name of the quantizer and calibration executables, plus the
  version each reports; the version is best effort and is absent when a tool
  does not implement `--version` or does not answer promptly, so the SHA-256 is
  what identifies a tool;
- the source corpus BOM hash and selected index paths;
- selected shard hashes, budget, seed, record count, and sample hash; and
- the complete compact calibration evidence embedded in the release
  `BOM.json`.

This evidence makes the calibration reproducible without pretending the
calibration text trained the model. The source managed model is unchanged.

## Choosing a quant level

The public interface uses simple integers. WALDO records the exact recipe in
the release BOM:

| `--quant` | llama.cpp recipe | Practical tradeoff |
| --- | --- | --- |
| `2` | `Q2_K` | Smallest, with the largest quality risk. |
| `3` | `Q3_K_M` | Very compact. |
| `4` | `Q4_K_M` | Common balance of size, speed, and quality. |
| `5` | `Q5_K_M` | More quality at a larger size. |
| `6` | `Q6_K` | High-quality quantized inference. |
| `8` | `Q8_0` | Largest quantized form, closest to source precision. |

Calibration can improve a quantized representation, especially at lower bit
levels, but it cannot recover behavior the source model never learned. Keep
the training-quality Safetensors or native WALDO export as the source for
future training; treat quantized GGUF as a derived inference release.
