# Driving the demo by hand

**You probably don't need this.** `bash scripts/walkthrough.sh` runs every command
below itself, in order, with the explanations already on screen. You type one
line and don't touch the keyboard for 7 minutes.

Two better options before you resort to typing:

| | |
|---|---|
| `bash scripts/walkthrough.sh` | Runs itself, paced for recording. 7:11. |
| `bash scripts/walkthrough.sh --step` | Same commands, same on-screen text, but **waits for Enter** between segments. You control the pace without typing anything. |

`--step` gives you everything manual typing gives you, control over when to move
on, with none of the risk of a typo on camera. Use it unless you specifically
want the audience to watch you type.

### What typing costs you

The walkthrough does not just run these commands, it prints **103 lines of
explanation** around them: 13 framed segment headers and the commentary after
each command ("See et al. published 40.34...", "the no-attention model places a
Louisville fire in SAN DIEGO", "62.8% of the attention mass lands on PADDING").

If you type the commands raw, **none of that appears.** The video becomes bare
terminal output with your voice over it. That is a legitimate demo, but it
changes two things:

- **Narration stops being optional.** With the script, the video explains itself
  in silence. Typed, the audio carries the entire argument, so if the sound fails,
  the recording is unusable.
- **The cue times in `NARRATION.md` no longer apply.** They were measured from a
  paced run of the script. Use the **Say:** lines below instead and ignore the
  clock.

`--step` keeps all 103 lines of on-screen explanation *and* gives you manual
pacing. It is strictly better than typing unless you specifically want the
commands typed on camera.

If you do want to type, this is the list. Put it on your second monitor.

---

## Before you start

```bash
cd ~/lstm-vs-llm-summarization
source .venv/bin/activate
clear
```

Warm the LLM once beforehand, off-camera, or step 5 stalls ~40 s loading the
model from disk:

```bash
python -m src.demo --example 3 >/dev/null
```

---

## 1. The task and the data

```bash
head -30 data/processed/dataset_meta.json
```

**Look for:** `"dataset": "abisee/cnn_dailymail"`, `"license": "Apache-2.0"`, and
`official_split_sizes` reading 287,113 / 13,368 / 11,490.

**Say:** the task, the licence, and that every system is scored on the same 500
held-out articles.

**Move on when:** the split sizes are visible. ~2 s.

---

## 2. Is the measurement trustworthy?

```bash
awk '/^## Overall/{f=1} /^## Diagnostics/{f=0} f' results/lead3_fulltest/results.md
```

**Look for:** one row, `lead3_baseline | 11490 | 40.04 ... 17.5 ... 36.34`.

**Say:** See et al. (2017) published 40.34 / 17.70 / 36.57. Agreement to ~0.3
ROUGE is what makes every other number believable.

> If this prints only the heading and no table, you have an older copy of the
> repo, `git pull`. That exact bug was fixed on 2026-08-09.

**Move on when:** the row is on screen. Instant.

---

## 3. The model, built from primitives

```bash
grep -rho "nn\.[A-Za-z]*" src/models/*.py | sort | uniq -c | sort -rn | head -6
```

**Look for:** counts of `nn.Linear`, `nn.Module`, `nn.LSTM`, `nn.Embedding`,
and nothing resembling a framework.

**Say:** no Fairseq, no OpenNMT, no HuggingFace trainer. 15,347,280 parameters.

**Move on when:** the six lines are up. Instant.

---

## 4. The one line that matters most

```bash
grep -n "masked_fill" src/models/attention.py
```

**Look for:** three hits, lines 48, 82 and 95 of `attention.py`. Bahdanau,
Luong and the batched path each mask independently; point at line 48.

**Say:** without it, 62.8% of the attention mass lands on padding, and it never
raises an error.

**Move on when:** the line is up. Instant.

---

## 5. Live: one article through every system

```bash
python -m src.demo --example 3 --ablations
```

**Look for, in order:** the source article, the reference summary, then

| Row | Expect |
|---|---|
| `LSTM + attention` | `R1=36.9 R2=22.2` |
| `LLM (Llama-3.1-8B-Instruct-4bit)` | `R1=33.3 R2=18.4` |
| `,  no_attention` | `R1=17.5`, `unsupported=0.56`, and the text says **San Diego** |
| `,  unidirectional` | `R1=44.0` |
| `,  short_context` | `R1=44.0` |

**Say:** the no-attention model relocates a Louisville fire to San Diego, the
fixed-vector bottleneck. And that the LSTM beats the LLM on *this* example,
which is why one example never carries a claim.

**Move on when:** all five rows have printed. ~10 s.

> If any output is repeated words (`the the the a the`) the GPU is out of memory.
> The demo detects this and falls back to CPU with a warning, stop and re-run
> rather than record it.

---

## 6. Live: text the model has never seen

```bash
python -m src.demo --file examples/demo_article_battery.txt
```

**Look for:** the LSTM output ending at *"have developed a battery"*, and the LLM
producing a full technical summary. No reference summary or ROUGE here, this
article has no gold answer.

**Say:** `electrolyte`, `anode`, `graphite` and the researcher's name are all
outside the 50k vocabulary. OOV 5.3% here vs 1.83% in domain.

**Move on when:** both summaries are up. ~8 s.

---

## 7. Results

```bash
sed -n '/## Overall/,/## Diagnostics/p' results/results.md
```

**Look for:** 12 rows, 11 systems plus `lead3_baseline`, with bootstrap CIs. `llm_B_zeroshot` top at 41.35,
`lstm_beam` at 35.00, `lead3_baseline` at 39.89, above four of five LLM rows.

**Move on when:** the whole table is up. Instant.

---

## 8. Three numbers *(nothing to type: talk over the table above)*

-14.02 · +2.83 · 39.89. See `NARRATION.md`.

---

## 9. Significance

```bash
sed -n '/## Paired bootstrap/,/## ROUGE-1 by/p' results/results.md | head -20
```

**Look for:** the `Δ ROUGE-1 [95% CI]` and `p` columns. `no_attention` at -14.02,
`llm_B_zeroshot` at +6.35, and `short_context` at **0.0532 (n.s.)**.

**Say:** the 100-token encoder window is the one we report as *not* significant.

**Move on when:** the table is up. Instant.

---

## 10. Error analysis

```bash
python -m src.demo --example 112
```

**Look for:** `LSTM + attention ... R1=10.5 R2=0.0`, ROUGE-2 exactly zero, and
an output naming nobody, saying both "hat-trick" and "brace", ending "to win a
win".

**Move on when:** both summaries are up. ~10 s.

---

## 11. The reverse case *(nothing to type)*

Test example 4, Sarah Stage: LSTM 57.1, LLM 26.3, and the LLM is entirely
correct. The numbers are in `NARRATION.md`; you can also show it live with
`python -m src.demo --example 4` if you have time, but it costs ~10 s.

---

## 12. The engineering trade-off

```bash
sed -n '/## Table 7/,/^## /p' reports/tables.md
```

**Look for:** GPU-hours per 1k summaries and p50 latency per setting, and the
line stating **$0.00** and 4.07 GPU-hours total.

**Say:** 15.3M parameters against ~8B; 61 MB against ~4.5 GB; 0.231 s per summary
against 2.85 s.

> The walkthrough script builds a cleaner side-by-side table here with an inline
> Python snippet. It is not worth typing on camera, this table carries the same
> point.

**Move on when:** the table is up. Instant.

---

## 13. Conclusion *(nothing to type)*

See `NARRATION.md`. Then stop the recording.

---

## Time budget

The typed path has almost no dead air, the three live commands are ~28 s total
and everything else is instant. That means **you are talking continuously for
about seven minutes**, with no pauses built in. The walkthrough script exists
precisely to give you those pauses.

If you type it manually, rehearse once with a timer. Target 7:00-7:45; do not
exceed 8:00.
