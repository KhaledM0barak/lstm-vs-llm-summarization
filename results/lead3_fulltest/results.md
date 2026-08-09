# Results

Test set: `data/processed/test.jsonl` (11490 examples)


## Overall (ROUGE F1 x100, 95% bootstrap CI)

| System | N | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Len |
|---|---|---|---|---|---|
| lead3_baseline | 11490 | 40.04 [39.83, 40.26] | 17.5 [17.3, 17.7] | 36.34 [36.14, 36.56] | 85.6223 |

## Diagnostics (means)

| System | Dup-trigram | Novel-bigram | Unsupported content | OOV rate | Empty |
|---|---|---|---|---|---|
| lead3_baseline | 0.0103 | 0.0 | 0.0 | 0.0209 | 0.0 |

## ROUGE-1 by length

| System | long (>874 tok) | medium (544-874 tok) | short (<=544 tok) |
|---|---|---|---|
| lead3_baseline | 37.72 | 40.19 | 42.19 |

## ROUGE-1 by abstractiveness

| System | abstractive (high novelty) | extractive (low novelty) | mixed |
|---|---|---|---|
| lead3_baseline | 33.95 | 45.74 | 40.41 |
