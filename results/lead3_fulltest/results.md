# Results

Test set: `data/processed/test.jsonl` (11490 examples)


## Overall (ROUGE F1 x100, 95% bootstrap CI)

| System | N | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Len |
|---|---|---|---|---|---|
| lead3_baseline | 11490 | 40.0 [39.79, 40.22] | 17.46 [17.26, 17.67] | 36.28 [36.09, 36.51] | 85.2959 |

## Diagnostics (means)

| System | Dup-trigram | Novel-bigram | Unsupported content | OOV rate | Empty |
|---|---|---|---|---|---|
| lead3_baseline | 0.0104 | 0.0 | 0.0 | 0.0209 | 0.0 |

## ROUGE-1 by length

| System | long (>874 tok) | medium (544-874 tok) | short (<=544 tok) |
|---|---|---|---|
| lead3_baseline | 37.68 | 40.14 | 42.16 |

## ROUGE-1 by abstractiveness

| System | abstractive (high novelty) | extractive (low novelty) | mixed |
|---|---|---|---|
| lead3_baseline | 33.93 | 45.64 | 40.4 |
