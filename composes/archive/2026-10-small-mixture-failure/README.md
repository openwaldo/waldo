# Failed small-model mixture experiment

These composes and evaluation outputs are preserved as negative evidence. Do
not rerun or promote them.

Both runs used a 76,416,000-parameter architecture and declared a 5:2:1
Wikimedia:PLOS:PressBooks mixture.

| Model | Run ID | Tokens | Held-out loss | Result |
| --- | --- | ---: | ---: | --- |
| `foundation-small-pilot-01` | `c83ec669ead7e866` | 760M | 3.1515 | Related English, but 15/15 deterministic responses collapsed into repetition |
| `foundation-small-01` | `8f81b0ad7b9eab9f` | 1.5B | 3.0751 | No material capability improvement; failed Gate 2 repetition and prompt-score thresholds |

The 760M run consumed the intended 62.5% Wikimedia, 25.0% PLOS, and 12.5%
PressBooks mixture. The 1.5B run consumed 65.8%, 26.3%, and 7.9%. The weighted
record stream maintained ratios only until PressBooks exhausted its current
pass, then drained Wikimedia and PLOS. The active implementation now ends a
weighted epoch at first-corpus exhaustion and lets capacity preflight add
deterministic passes.

The outputs also show excessive scientific and encyclopedic style for a
general foundation. The replacement recipe keeps Wikimedia at weight 5,
raises PressBooks from 1 to 2, and lowers PLOS from 2 to 1.

Files in this directory are the exact reference composes plus deterministic
and temperature-0.7 outputs supplied for both completed runs.
