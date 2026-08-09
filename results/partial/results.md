# Results

Test set: `data/processed/test_llm.jsonl` (500 examples)


## Overall (ROUGE F1 x100, 95% bootstrap CI)

| System | N | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Len |
|---|---|---|---|---|---|
| lead3_baseline | 500 | 39.89 [38.87, 40.93] | 17.6 [16.58, 18.6] | 36.35 [35.35, 37.38] | 86.224 |
| llm_A_fewshot | 500 | 39.4 [38.6, 40.21] | 14.9 [14.23, 15.58] | 36.15 [35.43, 36.91] | 81.106 |
| llm_A_zeroshot | 500 | 38.52 [37.63, 39.34] | 14.91 [14.28, 15.56] | 34.65 [33.83, 35.41] | 109.862 |
| lstm_beam | 500 | 35.0 [34.03, 36.04] | 13.75 [12.85, 14.64] | 32.25 [31.35, 33.27] | 48.188 |
| no_attention | 500 | 20.97 [20.32, 21.67] | 3.47 [3.15, 3.81] | 19.4 [18.82, 20.02] | 38.222 |

## Diagnostics (means)

| System | Dup-trigram | Novel-bigram | Unsupported content | OOV rate | Empty |
|---|---|---|---|---|---|
| lstm_beam | 0.0 | 0.0796 | 0.0152 | 0.0 | 0.0 |
| no_attention | 0.0 | 0.704 | 0.5297 | 0.0 | 0.0 |
| llm_A_zeroshot | 0.0072 | 0.5206 | 0.2061 | 0.021 | 0.0 |
| llm_A_fewshot | 0.0054 | 0.5004 | 0.1767 | 0.0214 | 0.0 |
| lead3_baseline | 0.0105 | 0.0 | 0.0 | 0.0226 | 0.0 |

## Paired bootstrap vs. `lstm_beam`

Per-example differences, resampling articles once per replicate and applying the same resample to both systems (10,000 replicates). A positive difference means the system beats `lstm_beam`. Unlike comparing two independent CIs, this test accounts for the fact that a hard article is hard for every system.

| System | N | Δ ROUGE-1 [95% CI] | p | Δ ROUGE-2 [95% CI] | p | W/L |
|---|---|---|---|---|---|---|
| no_attention | 500 | -14.02 [-15.02, -13.00] | 0.0000 | -10.28 [-11.18, -9.35] | 0.0000 | 56/442 |
| llm_A_zeroshot | 500 | +3.52 [+2.52, +4.55] | 0.0000 | +1.17 [+0.25, +2.08] | 0.0124 | 314/185 |
| llm_A_fewshot | 500 | +4.40 [+3.42, +5.41] | 0.0000 | +1.16 [+0.23, +2.10] | 0.0143 | 329/171 |
| lead3_baseline | 500 | +4.90 [+3.85, +5.94] | 0.0000 | +3.86 [+2.88, +4.87] | 0.0000 | 333/167 |

## ROUGE-1 by length

| System | long (>834 tok) | medium (534-834 tok) | short (<=534 tok) |
|---|---|---|---|
| lstm_beam | 33.15 | 36.16 | 35.69 |
| no_attention | 20.81 | 20.4 | 21.71 |
| llm_A_zeroshot | 37.34 | 38.0 | 40.2 |
| llm_A_fewshot | 37.64 | 39.49 | 41.07 |
| lead3_baseline | 36.87 | 40.81 | 42.0 |

## ROUGE-1 by abstractiveness

| System | abstractive (high novelty) | extractive (low novelty) | mixed |
|---|---|---|---|
| lstm_beam | 30.52 | 40.04 | 34.44 |
| no_attention | 19.32 | 22.15 | 21.46 |
| llm_A_zeroshot | 34.3 | 42.55 | 38.7 |
| llm_A_fewshot | 35.52 | 43.22 | 39.47 |
| lead3_baseline | 34.61 | 45.35 | 39.72 |
