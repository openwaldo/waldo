# Retired r50k foundation ladder

This ladder is preserved for reproducibility and must not be resumed.

It correctly validated multi-host execution and the repaired 5:2:1 weighted
sampler. Its 76.4M-parameter capability rung nevertheless failed: 15/15
deterministic probes showed serious factual, relevance, or repetition failures,
and sampled decoding did not remove the collapse. Held-out loss reached 3.2127,
so execution success and falling loss did not predict usable generation.

The architecture spent 32.2M parameters (42.1%) on the 50,259-token r50k
embedding table, leaving only about 44.2M parameters for transformer blocks.
The active ladder therefore starts with a corpus-trained 16K tokenizer and
treats small models as diagnostics rather than general-capability promises.

Key evidence:

- mixture canary: `foundation-mixture-canary-01`, run `84c76087cb8bf05c`;
  exact 62.549% Wikimedia, 24.929% PressBooks, 12.522% PLOS;
- corrected small pilot: `foundation-small-pilot-02`, run
  `8b1c1cefe2109355`; 760,020,992 tokens; held-out loss 10.9603 to 3.2127;
- consumption: 475,461,315 Wikimedia, 189,531,297 PressBooks, and 95,028,380
  PLOS token targets.

