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

## Table 3 — Main results (ROUGE F1 x100, 95% bootstrap CI, n=500)

| System | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Mean length |
|---|---|---|---|---|
| Lead-3 baseline | 39.89 [38.87, 40.93] | 17.60 [16.58, 18.60] | 36.35 [35.35, 37.38] | 86.2 |
| LSTM + attention (beam 4) | 35.00 [34.03, 36.04] | 13.75 [12.85, 14.64] | 32.25 [31.35, 33.27] | 48.2 |
| LSTM + attention (greedy) | 32.60 [31.66, 33.62] | 12.05 [11.25, 12.85] | 30.57 [29.64, 31.53] | 48.3 |
| LSTM + attention (beam, no trigram block) | 29.65 [28.69, 30.71] | 10.92 [10.14, 11.76] | 26.99 [26.07, 28.03] | 51.8 |
| — ablation: no attention | 20.97 [20.32, 21.67] | 3.47 [3.15, 3.81] | 19.40 [18.82, 20.02] | 38.2 |
| — ablation: unidirectional encoder | 33.12 [32.21, 34.08] | 12.64 [11.84, 13.49] | 30.37 [29.49, 31.26] | 45.8 |
| — ablation: 100-token encoder window | 34.02 [33.10, 34.91] | 13.44 [12.59, 14.29] | 31.49 [30.61, 32.32] | 41.8 |
| LLM variant A (plain), zero-shot | 38.52 [37.63, 39.34] | 14.91 [14.28, 15.56] | 34.65 [33.83, 35.41] | 109.9 |
| LLM variant A (plain), few-shot k=4 | 39.40 [38.60, 40.21] | 14.90 [14.23, 15.58] | 36.15 [35.43, 36.91] | 81.1 |
| LLM variant B (style-matched), zero-shot | 41.35 [40.44, 42.20] | 18.07 [17.25, 18.84] | 38.14 [37.29, 39.02] | 82.1 |
| LLM variant B (style-matched), few-shot k=4 | 40.48 [39.62, 41.34] | 16.58 [15.82, 17.36] | 37.58 [36.74, 38.41] | 79.1 |
| LLM variant B, zero-shot, full article | 40.87 [39.95, 41.70] | 17.70 [16.92, 18.50] | 37.65 [36.80, 38.50] | 84.0 |

## Table 4 — Behavioral diagnostics (means)

| System | Dup-trigram | Novel-bigram | Unsupported content | OOV rate | Empty |
|---|---|---|---|---|---|
| Lead-3 baseline | 0.011 | 0.000 | 0.000 | 0.023 | 0.000 |
| LSTM + attention (beam 4) | 0.000 | 0.080 | 0.015 | 0.000 | 0.000 |
| LSTM + attention (greedy) | 0.000 | 0.218 | 0.065 | 0.000 | 0.000 |
| LSTM + attention (beam, no trigram block) | 0.282 | 0.061 | 0.007 | 0.000 | 0.000 |
| — ablation: no attention | 0.000 | 0.704 | 0.530 | 0.000 | 0.000 |
| — ablation: unidirectional encoder | 0.000 | 0.088 | 0.024 | 0.000 | 0.000 |
| — ablation: 100-token encoder window | 0.000 | 0.150 | 0.040 | 0.000 | 0.000 |
| LLM variant A (plain), zero-shot | 0.007 | 0.521 | 0.206 | 0.021 | 0.000 |
| LLM variant A (plain), few-shot k=4 | 0.005 | 0.500 | 0.177 | 0.021 | 0.000 |
| LLM variant B (style-matched), zero-shot | 0.009 | 0.246 | 0.055 | 0.029 | 0.000 |
| LLM variant B (style-matched), few-shot k=4 | 0.010 | 0.370 | 0.100 | 0.024 | 0.000 |
| LLM variant B, zero-shot, full article | 0.009 | 0.241 | 0.054 | 0.028 | 0.000 |

## Table 5 — ROUGE-1 by length

| System | long (>834 tok) | medium (534-834 tok) | short (<=534 tok) |
|---|---|---|---|
| Lead-3 baseline | 36.87 | 40.81 | 42.00 |
| LSTM + attention (beam 4) | 33.15 | 36.16 | 35.69 |
| LSTM + attention (greedy) | 30.47 | 32.54 | 34.79 |
| LSTM + attention (beam, no trigram block) | 27.43 | 31.01 | 30.52 |
| — ablation: no attention | 20.81 | 20.40 | 21.71 |
| — ablation: unidirectional encoder | 30.75 | 33.96 | 34.67 |
| — ablation: 100-token encoder window | 31.14 | 35.37 | 35.57 |
| LLM variant A (plain), zero-shot | 37.34 | 38.00 | 40.20 |
| LLM variant A (plain), few-shot k=4 | 37.64 | 39.49 | 41.07 |
| LLM variant B (style-matched), zero-shot | 39.89 | 41.29 | 42.87 |
| LLM variant B (style-matched), few-shot k=4 | 38.23 | 40.07 | 43.12 |
| LLM variant B, zero-shot, full article | 38.66 | 41.37 | 42.60 |

## Table 6 — ROUGE-1 by abstractiveness

| System | abstractive (high novelty) | extractive (low novelty) | mixed |
|---|---|---|---|
| Lead-3 baseline | 34.61 | 45.35 | 39.72 |
| LSTM + attention (beam 4) | 30.52 | 40.04 | 34.44 |
| LSTM + attention (greedy) | 28.18 | 37.39 | 32.23 |
| LSTM + attention (beam, no trigram block) | 26.71 | 33.52 | 28.71 |
| — ablation: no attention | 19.32 | 22.15 | 21.46 |
| — ablation: unidirectional encoder | 28.85 | 37.38 | 33.15 |
| — ablation: 100-token encoder window | 30.71 | 38.19 | 33.16 |
| LLM variant A (plain), zero-shot | 34.30 | 42.55 | 38.70 |
| LLM variant A (plain), few-shot k=4 | 35.52 | 43.22 | 39.47 |
| LLM variant B (style-matched), zero-shot | 35.67 | 46.29 | 42.09 |
| LLM variant B (style-matched), few-shot k=4 | 36.10 | 44.41 | 40.92 |
| LLM variant B, zero-shot, full article | 35.58 | 45.17 | 41.88 |

## Headline gap

Best LLM setting is `llm_B_zeroshot`. Absolute ROUGE gap over the LSTM: R1 +6.35, R2 +4.32, R-Lsum +5.89. 95% CIs on ROUGE-1 do not overlap.

## Table 7 — LLM baseline compute and latency

| Setting | Shots | Input tok | Output tok | Wall-clock (min) | GPU-h / 1k summaries | Summaries/min | p50 latency (s) | Errors |
|---|---|---|---|---|---|---|---|---|
| A_fewshot | 4 | 1,234,587 | 48,433 | 76.5 | 2.551 | 6.5 | 9.15 | 0 |
| A_zeroshot | 0 | 270,587 | 62,501 | 25.2 | 0.839 | 19.9 | 3.01 | 0 |
| B_fewshot | 4 | 1,328,087 | 47,914 | 76.5 | 2.550 | 6.5 | 9.05 | 0 |
| B_zeroshot | 0 | 318,087 | 48,387 | 23.7 | 0.789 | 21.1 | 2.8 | 0 |
| B_zeroshot_fullarticle | 0 | 500,578 | 49,471 | 42.0 | 1.401 | 11.9 | 5.07 | 0 |

Backend: **local open-weights** — `mlx-community/Llama-3.1-8B-Instruct-4bit` (4-bit, greedy (temp=0.0)) via mlx-lm on the Apple silicon GPU. Monetary cost **$0.00**; total compute **4.07 GPU-hours** over 2,500 summaries.

## Appendix A — Exact prompts


### A_fewshot (variant A — plain, 4-shot, truncated_400_words)

**System prompt:**
```
You are a helpful assistant that writes concise news summaries.
```

**User template:**
```
Summarize the following news article.

Article:
{article}

Summary:
```

### A_zeroshot (variant A — plain, 0-shot, truncated_400_words)

**System prompt:**
```
You are a helpful assistant that writes concise news summaries.
```

**User template:**
```
Summarize the following news article.

Article:
{article}

Summary:
```

### B_fewshot (variant B — style-matched, 4-shot, truncated_400_words)

**System prompt:**
```
You write summaries in the style of CNN/DailyMail article highlights: 3 to 4 short, declarative, self-contained sentences totalling roughly 55 words. Each sentence states one concrete fact from the article — a name, a number, a place, an action. You never editorialize, never add background the article does not contain, and never write an introductory clause such as 'This article discusses'.
```

**User template:**
```
Write the highlights for the following news article. Output only the highlight sentences as a single paragraph, with no bullet points, no labels, and no preamble.

Article:
{article}

Highlights:
```

### B_zeroshot (variant B — style-matched, 0-shot, truncated_400_words)

**System prompt:**
```
You write summaries in the style of CNN/DailyMail article highlights: 3 to 4 short, declarative, self-contained sentences totalling roughly 55 words. Each sentence states one concrete fact from the article — a name, a number, a place, an action. You never editorialize, never add background the article does not contain, and never write an introductory clause such as 'This article discusses'.
```

**User template:**
```
Write the highlights for the following news article. Output only the highlight sentences as a single paragraph, with no bullet points, no labels, and no preamble.

Article:
{article}

Highlights:
```

### B_zeroshot_fullarticle (variant B — style-matched, 0-shot, full_article)

**System prompt:**
```
You write summaries in the style of CNN/DailyMail article highlights: 3 to 4 short, declarative, self-contained sentences totalling roughly 55 words. Each sentence states one concrete fact from the article — a name, a number, a place, an action. You never editorialize, never add background the article does not contain, and never write an introductory clause such as 'This article discusses'.
```

**User template:**
```
Write the highlights for the following news article. Output only the highlight sentences as a single paragraph, with no bullet points, no labels, and no preamble.

Article:
{article}

Highlights:
```

## LSTM inference

Decoding: beam (beam 4), 500 examples in 115.6 s = **0.231 s/summary** on mps. Model size: 15,347,280 parameters (~61 MB in fp32).
