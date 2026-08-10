# Task split: CP-468-D · AI · 17

Five workstreams, one owner each. The assignment below is **arbitrary:** I had
no information about who did what, so I split the project into five roughly equal
areas and assigned them in roster order. **Swap names freely**; what matters is
that each area has exactly one owner and that owner can defend it.

Each section lists: the files you own, the numbers you should know cold, and the
question a grader is most likely to ask you.

---

## 1. Mohanad Bahammam: Data & preprocessing

**Files:** `src/data/prepare.py`, `tokenizer.py`, `vocab.py`, `build_vocab.py`, `dataset.py`

**Own these numbers**
- CNN/DailyMail 3.0.0, **Apache-2.0**, official splits 287,113 / 13,368 / 11,490
- We train on **79,996** (seeded subsample), validate on 3,000, and every system is scored on the same **500-article** shared subset
- Vocabulary **50,000** types, **98.17%** token coverage, **1.83% OOV**
- Sources truncated to **400** tokens (following See et al., 2017), targets to 100

**Be ready to explain**
- *"How do you know there's no leakage?"* Vocabulary is built from train only; few-shot exemplars are drawn from train only; the 500-article subset was drawn once with seed 1234 before any modeling, and its indices are recorded in `dataset_meta.json`.
- Why a word-level tokenizer instead of subwords: it makes the OOV failure mode **observable**. A subword vocabulary would hide it by construction.
- Why sentence splitting protects abbreviations: without it, "the u.s. said" splits into two sentences, which shortens Lead-3 and corrupts ROUGE-Lsum.

---

## 2. Yakup Bastug: Model implementation

**Files:** `src/models/encoder.py`, `attention.py`, `decoder.py`, `seq2seq.py`

**Own these numbers**
- **15,347,280** parameters; 12.8M (83%) is the embedding table
- Embedding 256 → BiLSTM encoder 256/direction → Bahdanau attention → LSTM decoder 256 with input feeding → tied output projection
- Beam 4, GNMT length penalty, repeated-trigram blocking

**Be ready to explain**
- *"Show me the attention."* Open `attention.py` and walk the additive score `v·tanh(W_dec h + W_enc m)`, then point at `masked_fill(~mask, NEG_INF)` and say why: without it the decoder puts probability mass on `<pad>` for every short article batched with long ones.
- Why `<unk>` is suppressed at inference: an `<unk>` in a generated summary is a pure error, never useful output.
- Why additive attention can't be batched over decoder steps but multiplicative can (the `(B,T,S,A)` tensor would be ~330 GB).

---

## 3. Orhan Gundogan: Training & performance

**Files:** `src/train.py`, `configs/*.yaml`, `scripts/train_all.sh`

**Own these numbers**
- 5 epochs each, batch 64, Adam 1e-3, label smoothing 0.1, grad clip 5.0
- Base: val loss **4.622**, val PPL **35.6**, **3.05** GPU-hours. Total across four runs: **8.73 GPU-hours**
- Ablation perplexities: no-attention **121.7**, unidirectional **40.0**, 100-token window **65.5**

**Be ready to explain:** this is the most interesting workstream to be asked about
- The **34-55x speedup** (Appendix E): MPS compiles a Metal kernel per distinct tensor shape, and length-bucketed batches produced a near-unique `(batch, src_len, tgt_len)` every step, so training spent most of its time in shader compilation. Quantizing padded lengths to multiples of 64/16 fixed it.
- The memory bug: projecting to the 50k vocabulary in one matmul makes a 1.28 GB tensor at batch 64, and the loss copied it twice more, the machine went 9.5 GB into swap. Fixed by chunking over timesteps.
- Both were found by **measuring throughput** rather than by reading the code.

---

## 4. Ayuub Hagi: LLM baseline

**Files:** `src/llm/prompts.py`, `backends.py`, `baseline.py`

**Own these numbers**
- **Llama 3.1 8B Instruct**, 4-bit, run locally via MLX. Greedy decoding, so it's deterministic. **$0.00:** cost reported as GPU-hours
- 2 prompt variants x {zero-shot, few-shot k=4} = 4 settings, plus a full-article condition
- Prompts are ~600 tokens zero-shot, ~2,800 few-shot; throughput ~9 s/article, prefill-bound

**Be ready to explain**
- Why **two** prompt variants: A is a natural request, B describes the CNN/DailyMail reference style. The gap between them separates *summarization ability* from *fitting the metric's stylistic target*. One prompt would have made the comparison an artifact of prompt quality.
- **Input parity:** the LLM sees the same 400-word window as the LSTM. Giving it the full article would confound "better model" with "more input". The unmatched condition is reported separately.
- Why few-shot exemplars come from the training split: leakage control.

---

## 5. Khaled Mobarak: Evaluation, analysis & report

**Files:** `src/evaluate.py`, `qualitative.py`, `scripts/collect_results.py`, `reports/*`

**Own these numbers**
- **Lead-3 validation: 40.04 / 17.50 / 36.34** vs. See et al.'s published **40.34 / 17.70 / 36.57**
- LSTM beam **35.00** R1; no-attention **20.97** (-14.03); no trigram blocking **29.65** (-5.35)
- ROUGE-1/2/Lsum with `use_stemmer=True`, 95% bootstrap CIs over 1,000 resamples

**Be ready to explain**
- Why Lead-3 is reported everywhere: it scores **39.89** on the shared set, **beats our LSTM** (35.00), and **outscores four of the five LLM configurations**. That's the honest framing, a statement about what ROUGE measures, not a failure of the model.
- The diagnostics: duplicate-trigram, novel-bigram, unsupported-content, OOV. These are how error categories were **verified** rather than asserted.
- That `collect_results.py` reads run artifacts directly and refuses to emit a number whose artifact is missing, so no figure in the report was typed by hand.

---

## Everyone, before the demo

1. Run the pipeline once yourself: `python -m src.train --config configs/smoke.yaml` then `python -m src.demo --example 3 --ablations`.
2. Read your own section of `reports/report.md` and the files you own.
3. Know the **three headline findings**, regardless of your section:
   - Attention is worth **14 ROUGE-1**; without it the model hallucinates (56% unsupported content, it invents "San Diego" for a Louisville fire).
   - Trigram blocking is worth **5.4 ROUGE-1**; without it 28% of generated trigrams are repeats.
   - **Lead-3 beats our LSTM**, which is a caution about the metric.
4. Be able to say what you personally did. This is the question that separates a group that built something from a group that submitted something.
