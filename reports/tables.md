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

## Not yet available

- runs/*/train_summary.json (run scripts/train_all.sh)
- results/results.json (run src.evaluate)
- runs/llm/cost_summary.json (run src.llm.baseline --all)
