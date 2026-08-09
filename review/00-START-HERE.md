# CP-468 project review — start here

The code for the LSTM-vs-LLM summarization project was AI-generated. Before the
demo and before signing the AI-use disclosure, each of us reviews one part and
signs off on it.

**Repository:** `~/lstm-vs-llm-summarization`
**Public:** <https://github.com/KhaledM0barak/lstm-vs-llm-summarization>

## Why this matters

The disclosure in the report says *"All AI-assisted code was reviewed and tested
by the authors, who ran the full pipeline end to end and are responsible for its
correctness. Each author owns a specific component and can account for its design
decisions."*

That sentence is currently **not yet true**. These review files are what make it
true. It is also the single most checkable claim in the submission — one question
to the wrong person and it falls apart.

## Assignment

| File | Owner | Area | Lines to read |
|---|---|---|---|
| `01-data-preprocessing.md` | Mohanad Bahammam | Data & preprocessing | ~600 |
| `02-model.md` | Yakup Bastug | Model implementation | ~700 |
| `03-training.md` | Orhan Gundogan | Training & performance | ~300 |
| `04-llm-baseline.md` | Ayuub Hagi | LLM baseline | ~650 |
| `05-evaluation.md` | Khaled Mobarak | Evaluation & analysis | ~800 |

Swap freely — just make sure every file has exactly one owner.

## How to review (~60–90 minutes each)

1. **Run it first.** Each file starts with commands. Run them and confirm the output matches what's written. If it doesn't, that's a finding — write it down.
2. **Read the code you own.** Not every line — the functions each file points at.
3. **Answer the questions.** These are the ones a grader is most likely to ask. If you can't answer one from the code, that's a gap to close, not a box to tick.
4. **Check the "worth scrutinizing" section.** These are places where the implementation makes a debatable choice or where I'd look first for a bug. They are not known bugs — they're the parts that deserve a second pair of eyes.
5. **Sign off** at the bottom, honestly. "Reviewed, found X" is more valuable than a tick.

## Setup

```bash
cd ~/lstm-vs-llm-summarization
source .venv/bin/activate
python -m pytest tests/ -q          # expect: 149 passed, ~75 s
```

**Run the tests before you start.** There are 149 of them, aimed at the places
where a bug is silent rather than loud — a wrong attention mask, a misaligned
target shift, a miscounted diagnostic. Writing them found four real bugs:

1. The sentence splitter swallowed sentence-final digits, and the word "no"
   was treated as an abbreviation, so sentences ending in "no." never
   terminated. This changed Lead-3 and ROUGE-Lsum.
2. Paired significance tests silently did nothing when the reference system
   name was absent — no warning.
3. Resume treated a *failed* example as complete, so a transient error would
   permanently drop an article and leave systems scored on different subsets.
4. Resume reported token counts and GPU-hours for only the segment after the
   interruption, understating the totals quoted in the report.

If a test fails on your machine, that is a finding — write it down rather than
working around it.

**These files live in git** (`review/` in the repository). Edit them there and
commit your sign-off, so the review is part of the record rather than a local
file. The copy under `~/Documents/workspace/cp468/` is not tracked.

If the LLM sweep is still running (`pgrep -fl src.llm.baseline`), leave it alone —
it takes hours and is resumable but slow to redo. Use `--device cpu` for anything
that needs the GPU.

## The three findings everyone should know

Regardless of your section:

1. **Attention is worth 14 ROUGE-1.** Removing it drops the model from 35.00 to 20.97 and collapses ROUGE-2 from 13.75 to 3.47. Without attention the model produces summary-shaped text that isn't about the article — 53% of its content words don't appear in the source. In the demo it invents "San Diego" for a fire in Louisville, Kentucky.
2. **Trigram blocking is worth 5.4 ROUGE-1.** Without it, 28% of generated trigrams are repeats — the classic LSTM repetition failure, demonstrated rather than asserted.
3. **Lead-3 beats both our LSTM and the 8B LLM.** Copying the article's first three sentences scores 39.75 ROUGE-1 vs. our 35.00 and the LLM's 39.40. That is a statement about what ROUGE rewards, and it is the most intellectually honest thing in the report.
