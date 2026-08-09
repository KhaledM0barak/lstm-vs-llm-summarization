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

### 6. The test suite passes

```bash
python -m pytest tests/ -q
```
**Expect:** `170 passed`. If anything fails, do not record — the failure is
either a real regression or a stale artifact.

### 7. The repository works from a clean clone

This is the one check that cannot be done from inside the working directory,
and it is the one that matters most for a "fully reproducible" requirement.

```bash
cd /tmp && rm -rf checkclone
git clone https://github.com/KhaledM0barak/lstm-vs-llm-summarization.git checkclone
cd checkclone && python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m pytest tests/ -q
```
**Expect:** `169 passed, 1 skipped`. This caught a real bug — an unanchored
`.gitignore` rule had excluded the entire `src/data/` package from the
repository, so the project ran perfectly here and did not import at all for
anyone who cloned it.

### 8. Reproduce the headline numbers

```bash
python scripts/collect_results.py
```
**Expect:** `reports/tables.md` regenerates with every table filled. Anything it
can't find is listed under "Not yet available" rather than guessed.

---

## Part 2 — Recording the 8-minute demo

There are two ways to record. Pick one.

**Option A — the self-narrating walkthrough (recommended, no voice-over).**

```bash
bash scripts/walkthrough.sh --fast    # rehearse: 30 seconds, no pauses
bash scripts/walkthrough.sh           # record this: ~7:05
```

It prints each explanation on screen, pauses long enough to read it, then runs the
command. Start the recording, run it, don't touch the keyboard. `--pace 0.85`
shortens everything, `--pace 1.1` lengthens it, `--step` advances on Enter, and
`--from N` starts at segment N if you're re-recording one part.
**[`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) is the segment-by-segment timeline** — read it
before recording, and pass it to whoever is holding the camera.

**Recording on a teammate's machine?** They don't need Apple silicon, the 4.5 GB
LLM, or the 400 MB dataset — see *Recording on someone else's machine* in
`DEMO_SCRIPT.md`. Short version:

```bash
git clone https://github.com/KhaledM0barak/lstm-vs-llm-summarization.git
cd lstm-vs-llm-summarization
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-demo.txt
bash scripts/fetch_demo_bundle.sh
bash scripts/walkthrough.sh --fast     # rehearse, then drop --fast to record
```

**Option B — narrate it yourself** using the shot list below. Same content, but
you speak over it and control the pace.

---

**Setup before you hit record**

- Close the LLM sweep and any other GPU job (`pgrep -f src.llm.baseline`) — see the warning above.
- Terminal at ~110–130 columns, large readable font, dark theme. The demo adapts
  its render width to the window (clamped 60–140), so nothing wraps — check with
  `tput cols`. To pin a width regardless of window size: `COLUMNS=120 python -m src.demo ...`
- Have `reports/tables.md` and `results/qualitative.md` open in a second window.
- Do a dry run first. The commands below are fast, but rehearse once so you aren't reading them cold.

### Shot list (8:00 total)

| Time | Segment | What to show | What to say |
|---|---|---|---|
| **0:00–0:45** | **Problem & setup** | `README.md` §1 on screen | The task is abstractive summarization on CNN/DailyMail, Apache-2.0, official splits. Every system is scored on the same 500-article held-out subset, drawn once before any modeling. |
| **0:45–1:30** | **Pipeline is trustworthy** | `README.md` §2, the Lead-3 table | Before comparing anything, we validated the pipeline: our Lead-3 baseline scores 40.04 / 17.50 / 36.34 against See et al.'s published 40.34 / 17.70 / 36.57. That agreement is what makes the rest of the numbers believable. |
| **1:30–2:45** | **The model** | `src/models/encoder.py`, `attention.py`, `decoder.py` — scroll, don't read | Embedding → BiLSTM encoder → Bahdanau attention with source masking → LSTM decoder with input feeding → tied output projection. 15.3M parameters, written against `nn.LSTM` and `nn.Embedding` only. Point out the attention mask and say why it matters. |
| **2:45–4:15** | **Live demo — the payoff** | `python -m src.demo --example 3 --ablations` | Source, reference, then the LSTM with its ROUGE and diagnostics, then the ablations on the *same* article. **Land on the no-attention output placing a Louisville fire in "San Diego"** — 56% unsupported content. Say plainly that the LSTM *beats* the LLM on this one example (36.9 vs 33.3) and that the aggregate (+6.35, p<0.0001) is what carries the claim. |
| **4:15–5:00** | **Out-of-domain** | `python -m src.demo --file examples/demo_article_battery.txt` | Battery-chemistry article, nothing like 2015 news. The LSTM says *"have developed a battery"* and stops — `electrolyte`, `anode`, `graphite` and the researcher's name are all outside its 50k vocabulary. OOV 5.3% vs 1.83% in-domain. The LLM handles it. This is the transfer-learning story, live. |
| **5:00–6:15** | **Results** | `reports/tables.md` | The ablation table: attention is worth 14 ROUGE-1 and collapses ROUGE-2 fourfold; trigram blocking is worth 5.4; a 100-token window costs only 1.0, which is the lead bias. Then the LSTM-vs-LLM gap and the two prompt variants. |
| **6:15–7:15** | **Error analysis** | `python -m src.demo --example 112`, then Appendix G | **Celtic vs Kilmarnock.** Four failures in two sentences: hat-trick then brace about the same player; near-duplicate structure (repetition evading the trigram filter); "to win a win"; nobody named, because the names are OOV. **ROUGE-2 exactly 0.0** — fluent English, zero shared bigrams. Then the reverse case (Sarah Stage): the LSTM scores 57.1, the LLM 26.3, and **the LLM is entirely correct** — it just chose different true facts. That's the metric critique. |
| **7:15–8:00** | **Trade-offs & close** | Cost/latency table | 15.3M parameters and ~0.2 s per summary locally vs. an 8B model. State the honest conclusion: the LLM wins on quality, the LSTM wins on cost, latency, and controllability, and Lead-3 beating both is a caution about what ROUGE measures. |

### The three moments that matter

If you're short on time, protect these:

1. **The Lead-3 validation** (1:30) — it's what separates a project with numbers from a project with *trustworthy* numbers.
2. **The San Diego hallucination** (2:45) — one screenshot that proves the central architectural claim.
3. **Lead-3 beating the LSTM** (5:00) — shows you understand the metric's limits rather than just optimizing it.

### Commands, copy-paste ready

```bash
# Main demo shot -- ablations, the San Diego hallucination
python -m src.demo --example 3 --ablations

# Out-of-domain shot -- the OOV failure, live
python -m src.demo --file examples/demo_article_battery.txt

# Error-analysis shot -- fluent-but-wrong, ROUGE-2 of zero
python -m src.demo --example 112

# Interactive, if you would rather paste text live
python -m src.demo --interactive

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
