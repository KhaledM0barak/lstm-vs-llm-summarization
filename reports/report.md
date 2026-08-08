# LSTM Sequence-to-Sequence vs. a Pretrained LLM on Abstractive Summarization

*System report — 5 pages excluding references and appendices.*

> **Status:** sections 1–3 and 8 are final. Sections 4–7 are filled from
> `results/results.md`, `results/qualitative.md`, and the `train_summary.json` /
> `.meta.json` files once all runs complete. Placeholders are marked `[[FILL]]`.

---

## 1. Task and dataset

**Task.** Abstractive single-document summarization: given a news article,
generate a short multi-sentence summary.

**Dataset.** CNN/DailyMail 3.0.0, obtained from the Hugging Face Hub as
`abisee/cnn_dailymail` and distributed under the **Apache-2.0** license
(Hermann et al., 2015; See et al., 2017). The dataset's official splits are
article-disjoint by construction. We use them unchanged and add only
deterministic, seeded subsampling for compute tractability:

| Split | Documents | Mean source tokens | Mean summary tokens |
|---|---|---|---|
| Train (subsampled from 287,113) | 79,996 | 785.0 | 55.0 |
| Validation (subsampled from 13,368) | 3,000 | 764.1 | 61.7 |
| Test (full) | 11,490 | 773.0 | 58.0 |
| **Test-shared** (drawn once from test) | **500** | 747.4 | 58.2 |

**The shared test set.** Every system in this report — the LSTM, all ablations,
and all four LLM settings — is scored on the same 500-article subset. It exists
because scoring an API model on all 11,490 test articles exceeds the project's
budget. It was drawn once with seed 1234 before any model development, and its
indices are recorded in `data/processed/dataset_meta.json`.

**Leakage control.** The vocabulary is built from the training split only; the
LLM's few-shot exemplars are drawn from the training split only. No validation or
test text influences the vocabulary, the exemplars, or any hyperparameter.

**Pipeline validation.** Before comparing systems we verified the preprocessing
and ROUGE implementation by reproducing a published baseline. Our Lead-3 (first
three article sentences) scores **40.00 / 17.46 / 36.28** (ROUGE-1/2/Lsum) on the
full 11,490-article test set, against See et al. (2017)'s published
**40.34 / 17.70 / 36.57**. Agreement to within ~0.3 ROUGE indicates the
tokenization, sentence splitting, reference construction, and ROUGE configuration
are correct. Lead-3 is reported alongside every result below: on CNN/DailyMail it
is a famously strong baseline, and a summarization score is uninterpretable
without it.

---

## 2. System design

### 2.1 LSTM seq2seq with attention

Implemented directly in PyTorch using `nn.LSTM` and `nn.Embedding` only. No
prebuilt seq2seq framework is used anywhere in the model, training loop, or
decoding.

```
tokens → Embedding(50,000 × 256, shared encoder/decoder/output)
       → BiLSTM encoder (256 hidden per direction, 1 layer)
       → bridge: tanh(W·[h_fwd ; h_bwd]) → decoder initial state
       → Bahdanau additive attention, masked over padding
       → LSTM decoder (256 hidden, 1 layer, input feeding)
       → attentional vector tanh(W_c·[h_t ; c_t])
       → output projection (weight-tied to the embedding)
```

**Parameters: 15,347,280**, of which 12.8M (83%) are the embedding table.

**Preprocessing.** Word-level regex tokenizer with lowercasing, wire-service
preamble stripping, and whitespace normalization. A transparent word-level
tokenizer was chosen deliberately over subwords: it makes the out-of-vocabulary
failure mode directly observable, which a subword vocabulary would hide by
construction. Vocabulary: 50,000 types, min frequency 2, built from train only,
covering **98.17%** of training tokens (**1.83% OOV**). Sources are truncated to
400 tokens (following See et al., 2017) and targets to 100.

**Training.** Teacher forcing; label-smoothed cross-entropy (0.1); Adam at
lr 1e-3; gradient clipping at 5.0; `ReduceLROnPlateau` (factor 0.5); early
stopping on validation loss with patience 2; max 6 epochs; batch size 64 with
length-bucketed batching to limit padding waste.

**Decoding.** Beam search (beam 4) with the GNMT length penalty and
repeated-trigram blocking; `<pad>` and `<unk>` are suppressed at inference, since
an `<unk>` in a generated summary is a pure error. Greedy and no-blocking
variants are reported as decoding ablations.

**Two implementation details that mattered.** (i) Padding is masked before the
attention softmax; without it the decoder places probability mass on `<pad>` for
every short article batched with long ones. (ii) The vocabulary projection is
applied in time-chunks rather than to the full `(B, T, V)` tensor — materializing
it at batch 64 costs 1.28 GB in fp32 before the loss function copies it, which
drove the 24 GB machine into swap and cut throughput roughly 5×.

### 2.2 LLM baseline

**Model.** Claude Haiku 4.5 (`claude-haiku-4-5`) via the Anthropic API — the
low-cost tier, matching the assignment's "low-cost LLM API" option.

**Prompt variants.** Two, differing along the axis that most affects ROUGE on
this dataset — whether the prompt describes the *reference style*:

- **Variant A ("plain").** A natural summarization request, written as a user
  would write it without having seen the dataset.
- **Variant B ("style-matched").** Additionally specifies the length (~55 words),
  sentence count (3–4), and register of CNN/DailyMail highlights.

Both are run **zero-shot** and **few-shot (k = 4)**, giving four settings. The
exact prompts are reproduced in Appendix A and stored verbatim in the run
metadata. Testing both separates genuine summarization capability (A) from
fitting the metric's stylistic target (B), which is precisely why the assignment
requires more than one variant.

**Input parity.** By default the LLM receives the article **truncated to the same
400-word window the LSTM encoder sees**. Giving the LLM the untruncated article
while the LSTM sees 400 tokens would confound "better model" with "more input".
The unmatched full-article condition is run separately and reported as its own
row.

---

## 3. Experimental settings

| | |
|---|---|
| Hardware | Apple M4 Pro (12-core CPU, 16-core GPU), 24 GB unified memory, macOS 15 |
| Framework | PyTorch 2.13.0 (Metal/MPS backend), Python 3.13.1 |
| Seed | 1234 everywhere (data subsampling, vocabulary, batching, exemplar choice) |
| Metric | ROUGE-1/2/Lsum F1 (`rouge-score` 0.1.2, `use_stemmer=True`), 95% percentile bootstrap CIs over 1,000 resamples |

Training time, per-epoch losses, and full hardware provenance for every run are
recorded in `runs/<name>/train_summary.json`.

---

## 4. Results

`[[FILL: main results table from results/results.md — LSTM (beam/greedy),
three architecture ablations, Lead-3, and all four LLM settings, with CIs]]`

### 4.1 The quantitative gap

`[[FILL: size of the LSTM–LLM gap in ROUGE-1/2/Lsum, with CIs; whether the
intervals overlap; where each system sits relative to Lead-3]]`

### 4.2 Consistency across input length and difficulty

`[[FILL: ROUGE-1 by source-length tercile and by reference-abstractiveness
tercile; state whether the gap is constant or widens, and in which direction]]`

### 4.3 Ablations

`[[FILL: effect of removing attention, removing bidirectionality, and shrinking
the encoder window to 100 tokens; plus greedy-vs-beam and trigram-blocking]]`

### 4.4 Cost, latency, and compute

`[[FILL: measured USD cost per LLM setting and per 1,000 summaries; LSTM
training GPU-hours and inference latency; LLM p50/p95 latency]]`

---

## 5. Error analysis

`[[FILL: reference results/qualitative.md — 12 side-by-side examples selected by
behavior. Categorize errors using the measured diagnostics (duplicate-trigram
rate, OOV rate, unsupported-content rate, novel-bigram rate) rather than
asserting the textbook failure modes. Verify or refute: LSTM repetition,
fluent-but-wrong output, rare-word breakage; LLM hallucination, over-elaboration,
format-instruction drift.]]`

---

## 6. Discussion

### 6.1 Why the gap exists

`[[FILL: connect to model capacity (15.3M vs. a frontier-scale model), pretraining
data scale, transformer self-attention vs. the recurrent bottleneck — with the
short-context and no-attention ablations as direct evidence — and transfer
learning vs. 80k task-specific pairs]]`

### 6.2 Failure-mode contrast

`[[FILL: where each model fails *differently*, grounded in the measured
diagnostic rates rather than assertion]]`

### 6.3 Fairness of the comparison

The comparison is unfair in both directions, and both directions are measurable
here.

*Unfair to the LSTM:* the LLM was pretrained on a corpus many orders of magnitude
larger than 80k article–summary pairs, and CNN/DailyMail is a public benchmark
that has very likely appeared in that corpus — so its "zero-shot" performance may
partly reflect memorization of this dataset's style, or of these articles.

*Unfair to the LLM:* the LSTM is trained directly on this dataset's reference
distribution, so it optimizes the exact stylistic target ROUGE rewards, while the
LLM must be told about that target through a prompt. The gap between prompt
variants A and B quantifies exactly this penalty.

A fairer middle point would be fine-tuning a small pretrained transformer
(e.g. BART-base or T5-small, ~140M/60M parameters) on the same 80k pairs: it
isolates the contribution of pretraining from the contribution of architecture
and of task-specific supervision. `[[FILL: reference the measured A-vs-B gap]]`

### 6.4 Engineering trade-offs

`[[FILL: with the measured cost, latency, and size numbers, state the regimes
where the small trained model is still the right choice in 2026 — per-request
cost at scale, offline/air-gapped deployment, data-residency and privacy
constraints, fixed latency budgets, output controllability, and low-resource
languages or domains with no pretrained coverage]]`

### 6.5 Limitations and ethics

**Metric limitations.** ROUGE measures n-gram overlap with a single reference. It
rewards extractive copying — which is why Lead-3 scores 40 ROUGE-1 — and cannot
distinguish a factually wrong summary from a correct paraphrase. Our
unsupported-content diagnostic is a lexical proxy for faithfulness, not a
factuality judgment. No human evaluation was performed.

**Dataset bias and licensing.** CNN/DailyMail is English-only, US/UK news from a
narrow period, and its "summaries" are editor-written highlight bullets rather
than true summaries. Conclusions do not transfer to other languages, genres, or
summary styles. The dataset is Apache-2.0; the underlying articles remain the
publishers' property and are not redistributed in this repository.

**Contamination risk.** We cannot verify what was in the LLM's pretraining data.
CNN/DailyMail is among the most widely mirrored NLP benchmarks, so its test
articles were plausibly seen during pretraining. Any LLM number here should be
read as an upper bound on genuinely held-out performance. Our few-shot exemplars
are drawn from the training split, which controls the leakage we *can* control.

**Environmental and compute cost.** `[[FILL: measured training GPU-hours and API
token totals]]` These are small in absolute terms, but the LLM's per-request cost
is borne on inference hardware whose training cost is amortized across all users
and is invisible to us — a real asymmetry when comparing "compute used".

---

## 7. Conclusion

`[[FILL]]`

---

## 8. References

- Hermann, K. M., Kočiský, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman,
  M., & Blunsom, P. (2015). Teaching Machines to Read and Comprehend. *NeurIPS*.
- See, A., Liu, P. J., & Manning, C. D. (2017). Get To The Point: Summarization
  with Pointer-Generator Networks. *ACL*.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by
  Jointly Learning to Align and Translate. *ICLR*.
- Luong, M.-T., Pham, H., & Manning, C. D. (2015). Effective Approaches to
  Attention-based Neural Machine Translation. *EMNLP*.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning
  with Neural Networks. *NeurIPS*.
- Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
  *Text Summarization Branches Out*.
- Wu, Y., et al. (2016). Google's Neural Machine Translation System. *arXiv:1609.08144*.
  (GNMT length penalty.)

---

## Appendix A — Exact prompts

`[[FILL: paste verbatim from src/llm/prompts.py and the run .meta.json files —
both system prompts, both user templates, and the four few-shot exemplars]]`

## Appendix B — Contribution statement

`[[FILL: one line per team member describing their specific contribution.]]`

| Member | Contribution |
|---|---|
| `[[NAME]]` | `[[e.g. data pipeline, vocabulary, tokenization]]` |
| `[[NAME]]` | `[[e.g. encoder/decoder/attention implementation]]` |
| `[[NAME]]` | `[[e.g. LLM baseline, prompt design, cost accounting]]` |
| `[[NAME]]` | `[[e.g. evaluation harness, error analysis, report]]` |

## Appendix C — AI-use disclosure

`[[FILL — this must be an accurate account of your team's actual use. Template:]]`

This project used Claude (Anthropic) as a coding assistant during development.
Specifically, it was used to `[[e.g. draft the PyTorch model and training code,
the evaluation harness, and an initial draft of this report]]`. All generated code
was `[[reviewed / tested / modified]]` by the team; all reported numbers were
produced by executing the committed code, not written by a language model. Claude
Haiku 4.5 is additionally the *object of study* as the LLM baseline in Section
2.2 — that use is methodological rather than authorial and is described in full
in Sections 2.2 and 4.

## Appendix D — Repository and demo

- Code: `[[GITHUB URL]]`
- Demo video (8 min): `[[VIDEO URL]]`
