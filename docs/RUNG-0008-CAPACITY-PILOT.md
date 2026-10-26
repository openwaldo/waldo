# Rejected rung-0008 capacity pilot

The first rung-0008 candidate tested one falsifiable hypothesis: increasing
core capacity from 314.6M to 969.0M parameters at the same 1.573B-token
exposure would improve held-out BPB without regressing the frozen rung-0006
generation suite. Architecture capacity was the only optimization variable;
the tokenizer source, corpus views and weights, context, effective batch,
optimizer, learning rate, schedule, seed, held-out selection, and evaluator
were inherited from rung 0006.

## Candidate

- Model: `general-base-1b-capacity-pilot-01`
- Model ID: `e60e67e9aa51e8de3c5c48cd580e5130aacf38ce4a613c81aa837930e33036fb`
- Run ID: `84eb20f5de672846`
- Architecture: 1.035B total / 969.0M core parameters; 22 layers, width
  2,048, SwiGLU 5,632, 32 query heads, four key/value heads, tied 32K
  embeddings, and 2,048-token context
- Exposure: 1,573,011,456 packed targets, or 1.62 core TPP
- Selected checkpoint: step 96,000 of 96,009
- Held-out loss: 10.6982 initially, 2.3308 selected, 2.3315 final, and
  2.3308 after artifact reload
- Held-out BPB: 0.8487 selected

The matched rung-0006 baseline selected BPB 0.8442 at the same 1.573B-token
exposure. The candidate therefore regressed by 0.0045 BPB, or about 0.54%, and
failed the predeclared primary gate. The selected checkpoint was in the final
20%, metrics were finite, and reload was exact.

## Native generation

The unchanged rung-0006 suite produced:

- 20/20 sampled outputs with at least two complete sentences;
- 17/20 conservative manual passes for grammatical, on-topic continuation
  through three sentences;
- 18/20 sampled EOS within 512 tokens;
- zero sampled loops; and
- two greedy loops.

Those results pass the behavioral gates. They do not override the failed BPB
gate. Several continuations also showed the expected limitations at this low
exposure: entity substitutions, simple-story intrusion, factual errors, and
long but weakly grounded explanations.

For public context, the pinned official TinyLlama 1.1B 3T-token base produced
20/20 two-sentence completions on the same prompts but two sampled and four
greedy loops. It was not an exposure-matched promotion threshold.

## Decision

Reject the candidate and do not create a numbered rung directory. The result
shows that tripling core capacity did not improve distribution fit at fixed
tokens when the rung-0006 optimizer controls were carried over unchanged. It
does not show that the architecture fails at an appropriate exposure or tuned
learning rate.

The next authorized experiment is a low-cost learning-rate calibration for the
same 1B architecture. Compare `6e-4` with TinyLlama's
[documented `4e-4` rate](https://github.com/jzhang38/TinyLlama) at a small,
identical token budget and frozen seed/data controls. Only a lower-rate
candidate that wins the matched early BPB comparison may receive another full
matched-token qualification run. No 5-TPP extension is authorized from this
failed pilot.
