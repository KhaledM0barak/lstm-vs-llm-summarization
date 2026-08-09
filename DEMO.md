# Testing the system and recording the demo

Two parts: a checklist to verify everything works, and a shot-by-shot script for
the 8-minute video.

---

## Part 1 — Verify it works (~10 minutes)

Run these in order from the repository root with the virtualenv active
(`source .venv/bin/activate`). Each has an expected result — if one doesn't
match, stop and fix it before recording.

### 1. Environment

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```
**Expect:** `2.13.0 True` (on Apple silicon; `False` elsewhere is fine — everything falls back to CPU).

### 2. Data is present

```bash
wc -l data/processed/*.jsonl
```
**Expect:** 79,996 train / 3,000 validation / 11,490 test / 500 test_llm.
If missing, rebuild: `python -m src.data.prepare` then `python -m src.data.build_vocab`.

### 3. Checkpoints are present

```bash
ls -la runs/*/best.pt
```
**Expect:** four files (`base`, `no_attention`, `unidirectional`, `short_context`), ~56–59 MB each.

### 4. The pipeline runs end to end on a tiny slice

```bash
python -m src.train --config configs/smoke.yaml
```
**Expect:** finishes in well under a minute and prints a JSON line with `val_loss`.
This proves training, batching, masking, and checkpointing all work.

### 5. The demo produces a sensible summary

```bash
python -m src.demo --example 3 --no-llm
```
**Expect:** a fluent summary of the Kentucky fire article, ROUGE-1 around 37.

> **Important:** if you see empty output or `the the the a the`, the GPU is out of
> memory — usually because a local LLM is still resident. The demo detects this
> and falls back to CPU automatically, printing a warning. If you want to force
> it: `--device cpu`. Do not record while another GPU job is running.

### 6. Reproduce the headline numbers

```bash
python scripts/collect_results.py
```
**Expect:** `reports/tables.md` regenerates with every table filled. Anything it
can't find is listed under "Not yet available" rather than guessed.

---

## Part 2 — Recording the 8-minute demo

**Setup before you hit record**

- Close the LLM sweep and any other GPU job (`pgrep -f src.llm.baseline`) — see the warning above.
- Terminal at ~110 columns, large readable font, dark theme.
- Have `reports/tables.md` and `results/qualitative.md` open in a second window.
- Do a dry run first. The commands below are fast, but rehearse once so you aren't reading them cold.

### Shot list (8:00 total)

| Time | Segment | What to show | What to say |
|---|---|---|---|
| **0:00–0:45** | **Problem & setup** | `README.md` §1 on screen | The task is abstractive summarization on CNN/DailyMail, Apache-2.0, official splits. Every system is scored on the same 500-article held-out subset, drawn once before any modeling. |
| **0:45–1:30** | **Pipeline is trustworthy** | `README.md` §2, the Lead-3 table | Before comparing anything, we validated the pipeline: our Lead-3 baseline scores 40.00 / 17.46 / 36.28 against See et al.'s published 40.34 / 17.70 / 36.57. That agreement is what makes the rest of the numbers believable. |
| **1:30–2:45** | **The model** | `src/models/encoder.py`, `attention.py`, `decoder.py` — scroll, don't read | Embedding → BiLSTM encoder → Bahdanau attention with source masking → LSTM decoder with input feeding → tied output projection. 15.3M parameters, written against `nn.LSTM` and `nn.Embedding` only. Point out the attention mask and say why it matters. |
| **2:45–4:15** | **Live demo — the payoff** | `python -m src.demo --example 3 --ablations` | Walk through: source, human reference, then the LSTM's summary with its ROUGE and diagnostics. Then the ablations on the *same* article. **Land on the no-attention output inventing "San Diego" for a Louisville fire** — 56% unsupported content. This is the fixed-vector bottleneck, visible in one example. |
| **4:15–5:00** | **Interactive** | `python -m src.demo --interactive`, paste a news article of your own | Shows it generalizes beyond the test set. Have the article in your clipboard ready. |
| **5:00–6:15** | **Results** | `reports/tables.md` | The ablation table: attention is worth 14 ROUGE-1 and collapses ROUGE-2 fourfold; trigram blocking is worth 5.4; a 100-token window costs only 1.0, which is the lead bias. Then the LSTM-vs-LLM gap and the two prompt variants. |
| **6:15–7:15** | **Error analysis** | `results/qualitative.md` | Two or three contrasting examples: where the LSTM is fluent-but-wrong, where the LLM over-elaborates or drifts from the format. Emphasize the categories come from measured diagnostics, not assertion. |
| **7:15–8:00** | **Trade-offs & close** | Cost/latency table | 15.3M parameters and ~0.2 s per summary locally vs. an 8B model. State the honest conclusion: the LLM wins on quality, the LSTM wins on cost, latency, and controllability, and Lead-3 beating both is a caution about what ROUGE measures. |

### The three moments that matter

If you're short on time, protect these:

1. **The Lead-3 validation** (1:30) — it's what separates a project with numbers from a project with *trustworthy* numbers.
2. **The San Diego hallucination** (2:45) — one screenshot that proves the central architectural claim.
3. **Lead-3 beating the LSTM** (5:00) — shows you understand the metric's limits rather than just optimizing it.

### Commands, copy-paste ready

```bash
# Main demo shot
python -m src.demo --example 3 --ablations

# Interactive shot
python -m src.demo --interactive

# A different article if example 3 feels stale
python -m src.demo --example 11 --ablations

# LSTM only, instant startup (no 4.5 GB model load)
python -m src.demo --example 3 --no-llm

# Force CPU if the GPU is busy
python -m src.demo --example 3 --device cpu
```

### Recording

macOS: **Shift-Cmd-5** → *Record Selected Portion* → select the terminal.
Record audio from the built-in mic in the same pass. Keep the mouse still while
talking. If you fluff a segment, pause, and re-record that segment separately —
stitching two clips is faster than restarting an 8-minute take.
