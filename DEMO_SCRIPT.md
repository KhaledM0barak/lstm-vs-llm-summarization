# Demo script and timeline

The recorded demo is **8 minutes maximum**. Everything below is produced by one
command, `bash scripts/walkthrough.sh`, which prints each explanation on screen,
pauses long enough to read it, then runs the command. **No voice-over is required.**

Measured runtime in paced mode: **7:05-7:10**, leaving ~50 s of headroom.
(The spread is whether the LLM generates live or replays a recorded response.)

Read this file before recording so you know what is coming and can stop the take
early if something looks wrong. If you would rather narrate,
[`NARRATION.md`](NARRATION.md) has a spoken script keyed to these same cue
times.

---

## Timeline

The **#** column is the segment number printed on screen, the same number
`--from N` takes. Cumulative marks are approximate (±10 s; the two live model runs vary).

| Mark | # | Segment | On screen | Why it's in the video |
|---|---|---|---|---|
| 0:00 | | Title | Project title, CP-468 Group 17, all five names, repo URL | Identifies the submission |
| 0:08 | **1** | Task and data | `dataset_meta.json`: CNN/DailyMail 3.0.0, Apache-2.0, official split sizes, per-split token stats | States the task and proves the splits are the official article-disjoint ones. Establishes that **every system is scored on the same 500 held-out articles**, drawn once with a fixed seed |
| 0:32 | **2** | Is the measurement trustworthy? | Lead-3 on the full 11,490-article test set vs. See et al. (2017) | The credibility anchor. We reproduce a published baseline to ~0.3 ROUGE before comparing anything of our own. If ROUGE were misconfigured, every other number would be wrong |
| 0:58 | **3** | The model | `nn.*` primitive counts in `src/models/`, then the architecture and parameter count | Shows the model is built from `nn.LSTM`/`nn.Linear`/`nn.Embedding`, no Fairseq, no OpenNMT, no `Seq2SeqTrainer`. This is the "implement from scratch" requirement, demonstrated rather than asserted |
| 1:20 | **4** | The line that matters most | `grep -n masked_fill src/models/attention.py` | Without it, 62.8% of attention mass lands on padding. It never raises an error, the single most instructive bug in the project |
| 1:46 | **5** | Live: one article, every system | `src.demo --example 3 --ablations`, LSTM, LLM, and three ablations side by side | The required live demonstration. The no-attention model relocates a Louisville fire to **San Diego:** the fixed-vector bottleneck, visible. Also notes the LSTM *beats* the LLM here, and that one example never carries a claim |
| 2:30 | **6** | Live: out-of-domain text | `src.demo --file examples/demo_article_battery.txt` | Battery chemistry, nothing like 2015 news wire. OOV 5.3% vs 1.83% in-domain; the model cannot say "electrolyte" so it stops the clause. Shows the closed-vocabulary limit directly |
| 3:10 | **7** | Results | Overall table: 11 systems + Lead-3 | The headline comparison |
| 3:36 | **8** | Three numbers | -14.02 (attention ablation), +2.83 (prompt sensitivity vs. a 6.35 model gap), 39.89 (Lead-3 beats 4 of 5 LLM settings) | The three findings that carry the analysis |
| 4:10 | **9** | Significance | Paired bootstrap table | Independent CIs overlap on ROUGE-2; the paired test is what licenses the claim. Also shows the 100-token encoder result at p = 0.053, reported as **not** significant |
| 4:40 | **10** | Error analysis | `src.demo --example 112`, a football match report | Four failures in two sentences: hat-trick→brace contradiction, near-duplicate repetition evading the trigram filter, "to win a win", and no player named (all OOV). **ROUGE-2 exactly 0.0:** fluent English sharing not one bigram with the reference |
| 5:16 | **11** | The reverse case | Test example 4 (Sarah Stage), LSTM 57.1 vs LLM 26.3 | The LLM's summary is entirely true and scores less than half, because it chose different true facts than the editor. Bounds what ROUGE can tell us |
| 6:00 | **12** | Trade-off table | Parameters, disk, training hours, latency, throughput, cost | 15.3M vs ~8B; 0.231 s vs 2.85 s per summary; both $0.00 (local) |
| 6:22 | **13** | Conclusion | The four claims we're prepared to defend | LLM wins by 6.35 ROUGE-1 (p < 0.0001), real, but 6 points against a model 500x smaller; 45% of the gap is prompt phrasing; the LSTM is 12x faster and 74x smaller; with no training data it has no answer at all, and that is what pretraining bought |
| ~7:08 | | Close | Closing rule | |

---

## How to record

**1. Rehearse first, it takes 30 seconds.**

```bash
cd ~/lstm-vs-llm-summarization && source .venv/bin/activate
bash scripts/walkthrough.sh --fast
```

This runs everything with the pauses collapsed. If any command errors, fix it
before recording. `--fast` output is otherwise identical to the real thing.

**2. Warm the model caches**, or the first live segment stalls for ~40 s while
MLX loads Llama from disk:

```bash
python -m src.demo --example 3 >/dev/null
```

**3. Set up the terminal.**

- Full-screen the terminal window. The script reads `tput cols` and draws to fit,
  so a wide window gives more readable tables.
- 100-120 columns is the sweet spot. Check with `tput cols`.
- Increase the font size until text is legible in a scaled-down video player,
  bigger than feels natural.
- Use a light-on-dark theme; the script uses bold/dim/cyan/yellow.

**4. Record.**

macOS: `Cmd+Shift+5` → *Record Selected Portion* → select the terminal window →
Record. Or QuickTime → File → New Screen Recording.

Then, in the terminal:

```bash
bash scripts/walkthrough.sh
```

Do not touch the keyboard until the closing rule prints. Stop the recording.

**5. If it needs to be shorter or longer**, scale every pause at once:

```bash
bash scripts/walkthrough.sh --pace 0.85   # ~6:10
bash scripts/walkthrough.sh --pace 1.1    # ~7:50
```

Do not exceed 8:00.

**Other modes**

```bash
bash scripts/walkthrough.sh --step        # advance manually with Enter
bash scripts/walkthrough.sh --from 10     # start at segment 10 (re-recording one part)
```

`--step` is the safe option if you would rather control the pace yourself or add
your own narration; the timeline above then becomes a rough guide rather than a
measurement.

---

## Before you upload

- [ ] Runtime is under 8:00
- [ ] Text is legible at half size
- [ ] No terminal error tracebacks anywhere in the take
- [ ] The live demo segments actually produced summaries (not repeated words,
      see the degenerate-output note below)
- [ ] Upload unlisted or public, and paste the URL into **Appendix D** of
      `reports/report.md`

**Degenerate output:** on a machine under GPU memory pressure, PyTorch's Metal
backend returns garbage instead of raising. `src/demo.py` detects this
(`looks_degenerate`) and warns. If you see that warning, close other applications
and re-run, do not record it.

---

## Recording on someone else's machine

Anything that runs Python works, Windows, Linux, Intel Mac, Apple silicon. Four
commands, about ten minutes, most of it a download:

```bash
git clone https://github.com/KhaledM0barak/lstm-vs-llm-summarization.git
cd lstm-vs-llm-summarization
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-demo.txt                   # ~2 min
bash scripts/fetch_demo_bundle.sh                      # ~217 MB, checksummed
```

Then rehearse and record exactly as above:

```bash
bash scripts/walkthrough.sh --fast     # 30 s check
bash scripts/walkthrough.sh            # record this
```

**What gets downloaded and why.** The four trained checkpoints (232 MB) and the
500 held-out test articles are not in git, binary artifacts, and we don't
redistribute the full dataset. `fetch_demo_bundle.sh` pulls them from the
[`demo-artifacts-v1`](https://github.com/KhaledM0barak/lstm-vs-llm-summarization/releases/tag/demo-artifacts-v1)
release, verifies the SHA-256, and unpacks them to the paths the code expects.
Re-running it is a no-op once the files are there.

**The LLM half is replayed, not generated.** Running Llama 3.1 8B live needs
Apple silicon, `mlx-lm`, and a 4.5 GB download. On any other machine the demo
falls back to `examples/llm_cache.json`, responses recorded by
`scripts/build_llm_cache.py` through the *same* code path a live run uses, so the
text and the ROUGE scores are identical. The demo says so on screen: the header
prints `replayed from cache, recorded <date>`, and the output is labelled
`LLM (Llama-3.1-8B-Instruct-4bit · replayed)`. Nothing on the recording claims to
be live generation when it isn't.

Force it on a machine that *could* run the model live (useful if you don't want
to wait for the 4.5 GB download):

```bash
bash scripts/walkthrough.sh --replay
```

**What replay does not cover.** The cache holds the walkthrough's four fixed
articles. `--interactive` and `--file yourown.txt` on a pasted article will raise
a clear error on the LLM side, those need the live backend. The LSTM side works
on any input anywhere, so `--no-llm` always works.

To re-record the cache after changing a demo example, on an Apple silicon
machine:

```bash
python -m scripts.build_llm_cache
```
