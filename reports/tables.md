# Generated result tables

Every number below is read from a run artifact. Regenerate with `python scripts/collect_results.py`.


## Table 1 — Dataset

| Split | Documents | Mean source tokens | Mean summary tokens |
|---|---|---|---|
| train | 79,996 | 785.0 | 55.0 |
| validation | 3,000 | 764.1 | 61.7 |
| test | 11,490 | 773.0 | 58.0 |
| test_llm | 500 | 747.4 | 58.2 |

Dataset: `abisee/cnn_dailymail` (3.0.0), license **Apache-2.0**.
Official split sizes: {'train': 287113, 'validation': 13368, 'test': 11490}.

Vocabulary: 50,000 types from 393,066 distinct training tokens; token coverage **98.17%** (**1.83%** OOV).

## Table 2 — Training runs

| Run | Parameters | Epochs run | Train hours | Best val loss | Best val PPL |
|---|---|---|---|---|---|
| base | 15,347,280 | 5 | 3.05 | 4.6219 | 35.6 |
| no_attention | 15,150,416 | 5 | 2.67 | 5.6369 | 121.7 |
| short_context | 15,347,280 | 5 | 1.11 | 5.1179 | 65.5 |
| unidirectional | 14,558,800 | 5 | 1.90 | 4.7204 | 40.0 |

Hardware: Apple M4 Pro, 24 GB RAM, GPU: Apple Silicon GPU (Metal / MPS); device `mps`, torch 2.13.0, Python 3.13.1.

## Not yet available

- results/results.json (run src.evaluate)
- runs/llm/cost_summary.json (run src.llm.baseline --all)
