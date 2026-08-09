# LSTM seq2seq vs. LLM on abstractive summarization

Course project: a from-scratch LSTM encoder–decoder with attention, trained on
CNN/DailyMail, compared against a locally-run open-weights LLM baseline
(Llama 3.1 8B Instruct) on an identical test set.

The LSTM is implemented directly in PyTorch — `nn.LSTM` and `nn.Embedding` only.
No prebuilt seq2seq pipeline (Fairseq, OpenNMT, HuggingFace `Seq2SeqTrainer`) is
used anywhere in the model, training loop, or decoding.

---

## 1. Task and dataset

**Task.** Abstractive summarization: a news article → a short multi-sentence
summary.

**Dataset.** CNN/DailyMail 3.0.0, from the Hugging Face Hub as
[`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail).

| | |
|---|---|
| License | **Apache-2.0** (as published on the dataset's Hub card) |
| Citation | Hermann et al. (2015), *Teaching Machines to Read and Comprehend*; See et al. (2017), *Get To The Point: Summarization with Pointer-Generator Networks* |
| Official splits | 287,113 train / 13,368 validation / 11,490 test |
| Redistribution | Not redistributed in this repo. `src/data/prepare.py` downloads and rebuilds the exact splits used, deterministically. |

**Splits used here.** The dataset's official, article-disjoint splits are used
as-is; the only addition is deterministic, seeded subsampling for tractability:

| Split | Size | Purpose |
|---|---|---|
| `train.jsonl` | 79,996 | Model training (subsampled from 287,113 with `--seed 1234`) |
| `validation.jsonl` | 3,000 | Early stopping and LR scheduling |
| `test.jsonl` | 11,490 | Full test set — LSTM headline number |
| `test_llm.jsonl` | 500 | **The shared head-to-head set.** Every system — LSTM, every ablation, and all five LLM settings — is scored on exactly this file. |

`test_llm.jsonl` bounds the cost of the head-to-head comparison: generating
summaries for all 11,490 test articles across five LLM settings would take
roughly 35 GPU-hours rather than 3.4. It is drawn once, before any model
development, with a fixed seed, and every system is scored on exactly it.

**Leakage control.** The vocabulary is built from the training split only
(`src/data/build_vocab.py`); few-shot exemplars for the LLM are drawn from the
training split only. Neither validation nor test text influences the vocabulary,
the exemplars, or any hyperparameter chosen after the fact.

---

## 2. Pipeline validation

Before trusting any comparison, the preprocessing and ROUGE implementation were
checked against a published number. The Lead-3 baseline (first three sentences of
the article) on the **full 11,490-article test set**:

| | ROUGE-1 | ROUGE-2 | ROUGE-Lsum |
|---|---|---|---|
| This repo's Lead-3 | **40.04** | **17.50** | **36.34** |
| See et al. (2017), published Lead-3 | 40.34 | 17.70 | 36.57 |

Reproducing a published baseline to within ~0.3 ROUGE indicates the tokenization,
sentence splitting, reference construction, and ROUGE configuration are sound.
Lead-3 is reported alongside every result below, because on CNN/DailyMail it is a
famously strong baseline and a summarization score is hard to interpret without
it.

---

## 3. Setup

Requires Python 3.13 (3.10+ should work). Apple silicon (MPS), CUDA, and CPU are
all supported; the device is selected automatically.

```bash
git clone <this repo>
cd lstm-vs-llm-summarization

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The LLM baseline runs a **local open-weights model** by default — no API key and
no cost. The model (~4.5 GB) downloads automatically on first use.

`mlx` and `mlx-lm` are Apple-silicon only; the requirement markers skip them
elsewhere, so the install succeeds on any platform. To generate LLM summaries on
other hardware, use the hosted-API backend:

```bash
cp .env.example .env         # then set ANTHROPIC_API_KEY=sk-ant-...
python -m src.llm.baseline --all --backend anthropic
```

`.env` is gitignored. Every step other than the first dataset download runs
offline.

### Just want to run the demo?

You do not need the 400 MB dataset, 8.7 GPU-hours of training, or the 4.5 GB LLM.
On any platform:

```bash
pip install -r requirements-demo.txt
bash scripts/fetch_demo_bundle.sh    # 217 MB: trained checkpoints + the 500 test articles
python -m src.demo --example 3 --ablations
```

The LSTM runs locally. The LLM side replays responses recorded in
`examples/llm_cache.json` by `scripts/build_llm_cache.py` — produced through the
same code path a live run uses, and labelled `· replayed` on screen so nothing is
presented as live generation when it isn't. See
[`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).

---

## 4. Reproducing the results

All commands are run from the repository root, with the virtualenv active. Every
step is seeded with `1234`.

### Step 1 — Build the dataset (~2 min + download)

```bash
python -m src.data.prepare \
    --out-dir data/processed \
    --train-size 80000 --val-size 3000 --llm-test-size 500 --seed 1234
```

Writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, `test_llm.jsonl`, and
`dataset_meta.json` (which records split sizes, license, and the exact indices of
the head-to-head subset).

### Step 2 — Build the vocabulary (~1 min)

```bash
python -m src.data.build_vocab \
    --train-file data/processed/train.jsonl \
    --out data/processed/vocab.json --max-size 50000 --min-freq 2
```

Produces a 50,000-type shared vocabulary covering **98.17%** of training tokens
(1.83% OOV).

### Step 3 — Train the model and the ablations

```bash
python -m src.train --config configs/base.yaml
python -m src.train --config configs/ablation_no_attention.yaml
python -m src.train --config configs/ablation_unidirectional.yaml
python -m src.train --config configs/ablation_short_context.yaml
```

Each writes `runs/<name>/best.pt` plus `runs/<name>/train_summary.json`
(per-epoch losses, parameter count, wall-clock time, and full hardware
description). A fast end-to-end check of the whole pipeline:

```bash
python -m src.train --config configs/smoke.yaml     # ~30 s
```

### Step 4 — Generate LSTM summaries on the shared test set

```bash
python -m src.generate --checkpoint runs/base/best.pt \
    --test-file data/processed/test_llm.jsonl \
    --out runs/base/preds_test_llm.jsonl --decode beam --beam-size 4
```

Decoding ablations (greedy, and beam without repeated-trigram blocking):

```bash
python -m src.generate --checkpoint runs/base/best.pt \
    --test-file data/processed/test_llm.jsonl \
    --out runs/base/preds_greedy.jsonl --decode greedy

python -m src.generate --checkpoint runs/base/best.pt \
    --test-file data/processed/test_llm.jsonl \
    --out runs/base/preds_beam_norepeat.jsonl --decode beam --no-block-trigram
```

### Step 5 — Run the LLM baseline

Llama 3.1 8B Instruct (4-bit) run locally via MLX, greedy decoding. Two prompt
variants × {zero-shot, few-shot k=4} = four settings, all on the same 500
articles. Inspect the exact prompts first (loads nothing):

```bash
python -m src.llm.baseline --all --dry-run
```

Then run for real (~2 hours on an M4 Pro; resumable if interrupted):

```bash
python -m src.llm.baseline --all --batch-size 8
```

By default the model receives the article **truncated to the same 400-word window
the LSTM encoder sees**, so the comparison is not confounded by input length. The
unmatched condition is available separately and reported as its own row:

```bash
python -m src.llm.baseline --setting B_zeroshot --full-article --batch-size 4
```

Results are written per setting to `runs/llm/<setting>.jsonl` with a
`.meta.json` recording the verbatim prompts, token counts, latency percentiles,
and measured compute (GPU-hours locally, USD via the API backend).
`runs/llm/cost_summary.json` aggregates across settings. The runner is
resumable: rerunning after an interruption skips completed examples.

### Step 6 — Score everything

```bash
python -m src.evaluate \
    --test-file data/processed/test_llm.jsonl \
    --system lstm_beam=runs/base/preds_test_llm.jsonl \
    --system lstm_greedy=runs/base/preds_greedy.jsonl \
    --system no_attention=runs/no_attention/preds_test_llm.jsonl \
    --system unidirectional=runs/unidirectional/preds_test_llm.jsonl \
    --system short_context=runs/short_context/preds_test_llm.jsonl \
    --system llm_A_zeroshot=runs/llm/A_zeroshot.jsonl \
    --system llm_A_fewshot=runs/llm/A_fewshot.jsonl \
    --system llm_B_zeroshot=runs/llm/B_zeroshot.jsonl \
    --system llm_B_fewshot=runs/llm/B_fewshot.jsonl \
    --out-dir results/
```

Emits `results/results.md` (report tables), `results/results.json`, and
`results/per_example_scores.json`. Reports ROUGE-1/2/Lsum with 95% bootstrap
confidence intervals, the same metrics bucketed by input length and by reference
abstractiveness, and diagnostic rates (repetition, OOV, unsupported content).
A Lead-3 baseline is added automatically.

### Step 7 — Build the qualitative comparison

```bash
python -m src.qualitative \
    --scores results/per_example_scores.json \
    --system lstm_beam=runs/base/preds_test_llm.jsonl \
    --system llm_B_zeroshot=runs/llm/B_zeroshot.jsonl \
    --out results/qualitative.md
```

Selects test examples by *behavior* — most repetitive LSTM output, highest
unsupported-content LLM output, largest and smallest gaps, longest and shortest
articles — rather than by score, and emits them side by side with the measured
diagnostics attached.

---

## 4b. Tests

```bash
pip install pytest
python -m pytest tests/ -q          # 156 tests, ~70 s
```

The suite covers the places where a bug would be **silent** — a wrong attention
mask, a misaligned target shift, or a miscounted diagnostic changes every
reported number without raising anything. Notable checks:

| Test | Why it exists |
|---|---|
| `test_attention_assigns_zero_weight_to_padding` | Verifies the mask actually zeroes padded positions and that weights still sum to 1 |
| `test_attention_context_ignores_padded_content` | Writes garbage into padded positions and asserts the context vector is unchanged |
| `test_chunked_loss_matches_unchunked_reference` | The chunked projection (a memory optimization) must be numerically identical to projecting at once |
| `test_collate_target_shift_is_correct` | An off-by-one in the `<bos>`/`<eos>` shift trains the model to predict the wrong token |
| `test_bucket_sampler_covers_every_example_exactly_once` | Guards against silently losing or duplicating training data |
| `test_split_sentences_boundaries` | Asserts split *positions*, not just counts — this caught two real bugs |
| `test_paired_bootstrap_beats_independent_cis_on_correlated_data` | Demonstrates the case the paired test exists for |
| `tests/test_integration.py` | Full pipeline on a synthetic corpus: vocab → train → generate → evaluate → qualitative |

Two real bugs were found by writing these; both are described in the commit
history and in Appendix E of the report.

## 5. Model

```
tokens → Embedding(50k × 256, shared)
       → BiLSTM encoder (256 hidden/direction, 1 layer)
       → bridge: tanh(W·[h_fwd; h_bwd]) → decoder initial state
       → Bahdanau (additive) attention, masked over padding
       → LSTM decoder (256 hidden, 1 layer, input feeding)
       → attentional vector tanh(W_c·[h_t; c_t])
       → output projection (tied to the embedding matrix)
```

**15,347,280 parameters**, of which 12.8M are the embedding table.

Training uses teacher forcing, label-smoothed cross-entropy (0.1), Adam,
gradient clipping at 5.0, `ReduceLROnPlateau`, and early stopping on validation
loss. Batches are length-bucketed to cut padding waste. Inference supports greedy
and beam search with a GNMT length penalty and optional repeated-trigram
blocking.

Implementation notes worth knowing:

- Padding is masked before the attention softmax; without it the decoder puts
  probability mass on `<pad>` for every short article in a batch of long ones.
- The output projection is applied once to the stacked `(B, T, H)` decoder
  states rather than once per step — one large GEMM instead of 100 small ones,
  worth roughly a 3× speedup on MPS.
- `<unk>` and `<pad>` are suppressed at inference: an `<unk>` in a generated
  summary is a pure error rather than a useful token.

## 6. Repository layout

```
configs/                YAML configs: base + three ablations + smoke test
src/
  data/
    prepare.py          Download CNN/DailyMail, write frozen splits
    tokenizer.py        Word-level tokenizer, normalization, sentence splitting
    vocab.py            Vocabulary (train-split only)
    build_vocab.py      CLI for vocabulary construction
    dataset.py          Torch Dataset, padding/masking collate, length bucketing
  models/
    encoder.py          BiLSTM encoder + bridge to decoder state
    attention.py        Bahdanau / Luong / none (ablation), all masked
    decoder.py          LSTM decoder with input feeding
    seq2seq.py          Full model, greedy and beam search
  llm/
    prompts.py          The two prompt variants, verbatim
    backends.py         MLX (local, default) and Anthropic API backends
    baseline.py         LLM baseline runner: resume, usage, latency, cost accounting
  utils/
    seed.py             Deterministic seeding
    device.py           Device selection + hardware provenance
  train.py              Training loop
  generate.py           Batch decoding from a checkpoint
  evaluate.py           ROUGE + bootstrap CIs + buckets + diagnostics
  qualitative.py        Behavior-selected side-by-side comparison
  demo.py               Interactive side-by-side demo (LSTM vs LLM vs ablations)
scripts/
  train_all.sh          Train the base model and all three ablations
  finish.sh             Score everything and regenerate results/ and reports/
  collect_results.py    Assemble tables from run artifacts; never invents a number
  build_pdf.py          Render the report to PDF, flagging unfilled placeholders
  walkthrough.sh        Self-narrating 8-minute demo (see DEMO_SCRIPT.md)
  fetch_demo_bundle.sh  Download checkpoints + test data needed to run the demo
  build_llm_cache.py    Record LLM responses so the demo runs without the model
tests/                  172 tests, run with `pytest -q`
review/                 Per-component review packets, one per team member
results/                Generated tables and analyses
runs/                   Checkpoints, training logs, predictions (checkpoints gitignored)
```

Demo and recording: [`DEMO.md`](DEMO.md) (verification checklist + how to record),
[`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) (segment-by-segment timeline of the video).

## 7. Reproducibility

- Every entry point calls `set_seed(1234)` before touching data or parameters.
- Dataset subsampling, vocabulary tie-breaking, batch shuffling, and few-shot
  exemplar selection are all seeded and deterministic.
- Dependencies are pinned in `requirements.txt` to the exact versions used.
- `train_summary.json` and the generation/LLM `.meta.json` files record the
  hardware, device, wall-clock time, and full configuration of every run.

One caveat, stated plainly: cuDNN/MPS kernel nondeterminism means LSTM training
is not bit-for-bit reproducible across machines. Split construction, vocabulary,
and evaluation are exactly reproducible; trained model scores may vary by a small
amount on different hardware.
