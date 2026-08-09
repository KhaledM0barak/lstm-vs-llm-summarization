# Results

Test set: `data/processed/test_llm.jsonl` (500 examples)


## Overall (ROUGE F1 x100, 95% bootstrap CI)

| System | N | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Len |
|---|---|---|---|---|---|
| lead3_baseline | 500 | 39.75 [38.74, 40.79] | 17.5 [16.55, 18.49] | 36.2 [35.19, 37.19] | 86.296 |
| lstm_beam | 500 | 35.0 [34.03, 36.04] | 13.75 [12.85, 14.64] | 32.23 [31.33, 33.25] | 48.188 |
| short_context | 500 | 34.02 [33.1, 34.91] | 13.44 [12.59, 14.29] | 31.48 [30.59, 32.31] | 41.76 |
| unidirectional | 500 | 33.12 [32.21, 34.08] | 12.64 [11.84, 13.49] | 30.37 [29.48, 31.28] | 45.776 |
| lstm_greedy | 500 | 32.6 [31.66, 33.62] | 12.05 [11.25, 12.85] | 30.5 [29.57, 31.47] | 48.274 |
| lstm_beam_norepeat | 500 | 29.65 [28.69, 30.71] | 10.92 [10.14, 11.76] | 26.98 [26.06, 28.02] | 51.838 |
| no_attention | 500 | 20.97 [20.32, 21.67] | 3.47 [3.15, 3.81] | 19.36 [18.8, 19.97] | 38.222 |

## Diagnostics (means)

| System | Dup-trigram | Novel-bigram | Unsupported content | OOV rate | Empty |
|---|---|---|---|---|---|
| lstm_beam | 0.0 | 0.0796 | 0.0152 | 0.0 | 0.0 |
| lstm_greedy | 0.0 | 0.218 | 0.0648 | 0.0 | 0.0 |
| lstm_beam_norepeat | 0.2817 | 0.0614 | 0.0075 | 0.0 | 0.0 |
| no_attention | 0.0 | 0.704 | 0.5297 | 0.0 | 0.0 |
| unidirectional | 0.0 | 0.0876 | 0.0242 | 0.0 | 0.0 |
| short_context | 0.0 | 0.1501 | 0.04 | 0.0 | 0.0 |
| lead3_baseline | 0.0113 | 0.0 | 0.0 | 0.0225 | 0.0 |

## ROUGE-1 by length

| System | long (>834 tok) | medium (534-834 tok) | short (<=534 tok) |
|---|---|---|---|
| lstm_beam | 33.15 | 36.16 | 35.69 |
| lstm_greedy | 30.47 | 32.54 | 34.79 |
| lstm_beam_norepeat | 27.43 | 31.01 | 30.52 |
| no_attention | 20.81 | 20.4 | 21.71 |
| unidirectional | 30.75 | 33.96 | 34.67 |
| short_context | 31.14 | 35.37 | 35.57 |
| lead3_baseline | 36.86 | 40.39 | 41.99 |

## ROUGE-1 by abstractiveness

| System | abstractive (high novelty) | extractive (low novelty) | mixed |
|---|---|---|---|
| lstm_beam | 30.52 | 40.04 | 34.44 |
| lstm_greedy | 28.18 | 37.39 | 32.23 |
| lstm_beam_norepeat | 26.71 | 33.52 | 28.71 |
| no_attention | 19.32 | 22.15 | 21.46 |
| unidirectional | 28.85 | 37.38 | 33.15 |
| short_context | 30.71 | 38.19 | 33.16 |
| lead3_baseline | 34.57 | 45.04 | 39.63 |
