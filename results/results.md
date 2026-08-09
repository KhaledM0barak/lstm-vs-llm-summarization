# Results

Test set: `data/processed/test_llm.jsonl` (500 examples)


## Overall (ROUGE F1 x100, 95% bootstrap CI)

| System | N | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Len |
|---|---|---|---|---|---|
| llm_B_zeroshot | 500 | 41.35 [40.44, 42.2] | 18.07 [17.25, 18.84] | 38.14 [37.29, 39.02] | 82.086 |
| llm_B_fewshot | 500 | 40.48 [39.62, 41.34] | 16.58 [15.82, 17.36] | 37.58 [36.74, 38.41] | 79.1 |
| lead3_baseline | 500 | 39.89 [38.87, 40.93] | 17.6 [16.58, 18.6] | 36.35 [35.35, 37.38] | 86.224 |
| llm_A_fewshot | 500 | 39.4 [38.6, 40.21] | 14.9 [14.23, 15.58] | 36.15 [35.43, 36.91] | 81.106 |
| llm_A_zeroshot | 500 | 38.52 [37.63, 39.34] | 14.91 [14.28, 15.56] | 34.65 [33.83, 35.41] | 109.862 |
| lstm_beam | 500 | 35.0 [34.03, 36.04] | 13.75 [12.85, 14.64] | 32.25 [31.35, 33.27] | 48.188 |
| short_context | 500 | 34.02 [33.1, 34.91] | 13.44 [12.59, 14.29] | 31.49 [30.61, 32.32] | 41.76 |
| unidirectional | 500 | 33.12 [32.21, 34.08] | 12.64 [11.84, 13.49] | 30.37 [29.49, 31.26] | 45.776 |
| lstm_greedy | 500 | 32.6 [31.66, 33.62] | 12.05 [11.25, 12.85] | 30.57 [29.64, 31.53] | 48.274 |
| lstm_beam_norepeat | 500 | 29.65 [28.69, 30.71] | 10.92 [10.14, 11.76] | 26.99 [26.07, 28.03] | 51.838 |
| no_attention | 500 | 20.97 [20.32, 21.67] | 3.47 [3.15, 3.81] | 19.4 [18.82, 20.02] | 38.222 |

## Diagnostics (means)

| System | Dup-trigram | Novel-bigram | Unsupported content | OOV rate | Empty |
|---|---|---|---|---|---|
| lstm_beam | 0.0 | 0.0796 | 0.0152 | 0.0 | 0.0 |
| lstm_greedy | 0.0 | 0.218 | 0.0648 | 0.0 | 0.0 |
| lstm_beam_norepeat | 0.2817 | 0.0614 | 0.0075 | 0.0 | 0.0 |
| no_attention | 0.0 | 0.704 | 0.5297 | 0.0 | 0.0 |
| unidirectional | 0.0 | 0.0876 | 0.0242 | 0.0 | 0.0 |
| short_context | 0.0 | 0.1501 | 0.04 | 0.0 | 0.0 |
| llm_A_zeroshot | 0.0072 | 0.5206 | 0.2061 | 0.021 | 0.0 |
| llm_A_fewshot | 0.0054 | 0.5004 | 0.1767 | 0.0214 | 0.0 |
| llm_B_zeroshot | 0.0085 | 0.246 | 0.0552 | 0.0286 | 0.0 |
| llm_B_fewshot | 0.0099 | 0.37 | 0.1003 | 0.024 | 0.0 |
| lead3_baseline | 0.0105 | 0.0 | 0.0 | 0.0226 | 0.0 |

## Paired bootstrap vs. `lstm_beam`

Per-example differences, resampling articles once per replicate and applying the same resample to both systems (10,000 replicates). A positive difference means the system beats `lstm_beam`. Unlike comparing two independent CIs, this test accounts for the fact that a hard article is hard for every system.

| System | N | Δ ROUGE-1 [95% CI] | p | Δ ROUGE-2 [95% CI] | p | W/L |
|---|---|---|---|---|---|---|
| lstm_greedy | 500 | -2.40 [-3.23, -1.53] | 0.0000 | -1.69 [-2.44, -0.93] | 0.0000 | 204/291 |
| lstm_beam_norepeat | 500 | -5.35 [-6.03, -4.67] | 0.0000 | -2.83 [-3.34, -2.32] | 0.0000 | 62/330 |
| no_attention | 500 | -14.02 [-15.02, -13.00] | 0.0000 | -10.28 [-11.18, -9.35] | 0.0000 | 56/442 |
| unidirectional | 500 | -1.87 [-2.74, -0.98] | 0.0001 | -1.11 [-1.84, -0.35] | 0.0028 | 210/284 |
| short_context | 500 | -0.98 [-1.96, +0.02] | 0.0532 (n.s.) | -0.31 [-1.16, +0.58] | 0.4951 (n.s.) | 235/265 |
| llm_A_zeroshot | 500 | +3.52 [+2.52, +4.55] | 0.0000 | +1.17 [+0.25, +2.08] | 0.0124 | 314/185 |
| llm_A_fewshot | 500 | +4.40 [+3.42, +5.41] | 0.0000 | +1.16 [+0.23, +2.10] | 0.0143 | 329/171 |
| llm_B_zeroshot | 500 | +6.35 [+5.34, +7.39] | 0.0000 | +4.33 [+3.39, +5.28] | 0.0000 | 358/142 |
| llm_B_fewshot | 500 | +5.48 [+4.43, +6.54] | 0.0000 | +2.83 [+1.83, +3.84] | 0.0000 | 348/152 |
| lead3_baseline | 500 | +4.90 [+3.85, +5.94] | 0.0000 | +3.86 [+2.88, +4.87] | 0.0000 | 333/167 |

## ROUGE-1 by length

| System | long (>834 tok) | medium (534-834 tok) | short (<=534 tok) |
|---|---|---|---|
| lstm_beam | 33.15 | 36.16 | 35.69 |
| lstm_greedy | 30.47 | 32.54 | 34.79 |
| lstm_beam_norepeat | 27.43 | 31.01 | 30.52 |
| no_attention | 20.81 | 20.4 | 21.71 |
| unidirectional | 30.75 | 33.96 | 34.67 |
| short_context | 31.14 | 35.37 | 35.57 |
| llm_A_zeroshot | 37.34 | 38.0 | 40.2 |
| llm_A_fewshot | 37.64 | 39.49 | 41.07 |
| llm_B_zeroshot | 39.89 | 41.29 | 42.87 |
| llm_B_fewshot | 38.23 | 40.07 | 43.12 |
| lead3_baseline | 36.87 | 40.81 | 42.0 |

## ROUGE-1 by abstractiveness

| System | abstractive (high novelty) | extractive (low novelty) | mixed |
|---|---|---|---|
| lstm_beam | 30.52 | 40.04 | 34.44 |
| lstm_greedy | 28.18 | 37.39 | 32.23 |
| lstm_beam_norepeat | 26.71 | 33.52 | 28.71 |
| no_attention | 19.32 | 22.15 | 21.46 |
| unidirectional | 28.85 | 37.38 | 33.15 |
| short_context | 30.71 | 38.19 | 33.16 |
| llm_A_zeroshot | 34.3 | 42.55 | 38.7 |
| llm_A_fewshot | 35.52 | 43.22 | 39.47 |
| llm_B_zeroshot | 35.67 | 46.29 | 42.09 |
| llm_B_fewshot | 36.1 | 44.41 | 40.92 |
| lead3_baseline | 34.61 | 45.35 | 39.72 |
