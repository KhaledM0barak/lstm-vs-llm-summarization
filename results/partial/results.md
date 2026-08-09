# Results

Test set: `data/processed/test_llm.jsonl` (500 examples)


## Overall (ROUGE F1 x100, 95% bootstrap CI)

| System | N | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Len |
|---|---|---|---|---|---|
| lead3_baseline | 500 | 39.75 [38.74, 40.79] | 17.5 [16.55, 18.49] | 36.2 [35.19, 37.19] | 86.296 |
| llm_A_fewshot | 500 | 39.4 [38.6, 40.21] | 14.9 [14.23, 15.58] | 36.13 [35.41, 36.88] | 81.106 |
| lstm_beam | 500 | 35.0 [34.03, 36.04] | 13.75 [12.85, 14.64] | 32.23 [31.33, 33.25] | 48.188 |
| no_attention | 500 | 20.97 [20.32, 21.67] | 3.47 [3.15, 3.81] | 19.36 [18.8, 19.97] | 38.222 |

## Diagnostics (means)

| System | Dup-trigram | Novel-bigram | Unsupported content | OOV rate | Empty |
|---|---|---|---|---|---|
| lstm_beam | 0.0 | 0.0796 | 0.0152 | 0.0 | 0.0 |
| no_attention | 0.0 | 0.704 | 0.5297 | 0.0 | 0.0 |
| llm_A_fewshot | 0.0054 | 0.5004 | 0.1767 | 0.0214 | 0.0 |
| lead3_baseline | 0.0113 | 0.0 | 0.0 | 0.0225 | 0.0 |

## ROUGE-1 by length

| System | long (>834 tok) | medium (534-834 tok) | short (<=534 tok) |
|---|---|---|---|
| lstm_beam | 33.15 | 36.16 | 35.69 |
| no_attention | 20.81 | 20.4 | 21.71 |
| llm_A_fewshot | 37.64 | 39.49 | 41.07 |
| lead3_baseline | 36.86 | 40.39 | 41.99 |

## ROUGE-1 by abstractiveness

| System | abstractive (high novelty) | extractive (low novelty) | mixed |
|---|---|---|---|
| lstm_beam | 30.52 | 40.04 | 34.44 |
| no_attention | 19.32 | 22.15 | 21.46 |
| llm_A_fewshot | 35.52 | 43.22 | 39.47 |
| lead3_baseline | 34.57 | 45.04 | 39.63 |
