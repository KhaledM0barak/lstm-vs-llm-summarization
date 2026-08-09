# Review 5 — Evaluation, analysis & report

**Owner:** Khaled Mobarak
**Files:** `src/evaluate.py` (361), `qualitative.py` (162), `scripts/collect_results.py` (285), `reports/report.md`

Everything reported rests on this component. If ROUGE is configured wrongly,
every number in the report is wrong and nothing else matters.

---

## Run this first

```bash
cd ~/lstm-vs-llm-summarization && source .venv/bin/activate

# THE credibility check: reproduce a published baseline
cat results/lead3_fulltest/results.md | head -8

# Final standings, all systems
sed -n '/## Overall/,/## Diagnostics/p' results/results.md

# Confirm the collector refuses to invent numbers
python scripts/collect_results.py | tail -20
```

**Expect:** Lead-3 on the full 11,490-article test set scores **40.04 / 17.50 /
36.34** against See et al. (2017)'s published **40.34 / 17.70 / 36.57**. Agreement
to ~0.3 ROUGE is what makes every other number in the report believable. If this
ever stops matching, stop and investigate before trusting anything else.

---

## Read these specifically

- `evaluate.py` → `score_system()`, `diagnostics()`, `bootstrap_ci()`, `assign_buckets()`, and `rougeLsum_ready()`
- `qualitative.py` → the `selectors` list — the examples are chosen by **behaviour**, not by score
- `collect_results.py` → the `missing` list logic

---

## Verify these claims

- [ ] ROUGE uses `use_stemmer=True` — the standard CNN/DailyMail configuration
- [ ] `rougeLsum` receives newline-separated sentences (`rougeLsum_ready`) — without this, ROUGE-Lsum is computed wrongly
- [ ] Confidence intervals are percentile bootstrap over **per-example** scores, 1,000 resamples, seeded
- [ ] Lead-3 is computed from the source articles automatically and appears in every table
- [ ] `collect_results.py` lists artifacts it can't find under "Not yet available" rather than emitting a placeholder number

---

## Worth scrutinizing

1. **Systems are scored on the IDs they have.** `score_system` iterates references present in the predictions file. If one system has 500 predictions and another 480, they're compared on **different subsets** and the `N` column differs. Check that every system reports `N = 500` in the final table; if one doesn't, the comparison is not like-for-like.

2. **The paired bootstrap — now implemented, verify it.** We report a CI per system, but non-overlapping CIs imply a difference while overlapping CIs do **not** imply the absence of one. `paired_bootstrap()` and `compare_systems()` in `evaluate.py` now resample article indices once per replicate and apply the same resample to both systems, preserving the correlation. Check: (a) the "Paired bootstrap" table appears in `results.md`; (b) `test_paired_bootstrap_beats_independent_cis_on_correlated_data` in `tests/test_evaluate.py` demonstrates exactly the case it exists for; (c) the p-value for the ROUGE-2 LSTM-vs-LLM gap — that gap was only +1.15 with CIs nearly touching, so this test decides whether we can claim it at all.

3. **"Unsupported content" is lexical, not factual.** It counts content words absent from the source. A correct paraphrase scores as unsupported; a fluent lie that reuses source words doesn't. It's a *proxy* for faithfulness. The report must say so — never call it a hallucination rate without qualification.

4. **OOV rate is 0.000 for all LSTM systems by construction**, because `<unk>` is suppressed at generation. It isn't evidence there's no OOV problem — the problem shows up as **substitution** instead. Make sure the error analysis says this rather than reporting 0.000 as a good result.

5. **Difficulty buckets are terciles of reference novelty**, which is one of several reasonable definitions. Be able to justify it: it measures how much genuine abstraction the example demands rather than how long it is.

---

## Your other job: the qualitative error categories

`results/qualitative.md` generates ~13 side-by-side examples with each error
category marked `TODO — verify against the text`. **These were deliberately left
unlabelled.** The assignment says:

> *"Categorize errors ... Verify against your data — do not just assert this."*

Filling these in by reading the actual outputs is the difference between an error
analysis and a recitation of the textbook. Use the diagnostics next to each
output as evidence, not as the answer.

---

## Be ready to answer

- **"How do you know your ROUGE implementation is right?"** — the Lead-3 reproduction. This is the strongest answer in the whole project.
- **"Lead-3 beats your model. Isn't that a failure?"** — It's the well-known CNN/DailyMail result and it's a statement about the metric: references are editor-written highlights that foreground the lead, so extraction is rewarded. Reporting it is the honest choice; hiding it would be the failure.
- *"What do your diagnostics actually measure?"* — duplicate-trigram (repetition), novel-bigram (abstractiveness), unsupported content (a lexical faithfulness proxy), OOV.

## Demo examples used in the video

Chosen so the walkthrough shows verified failure modes rather than a highlight
reel, and so nothing objectionable appears on a submitted recording.

**`--example 3`** (Kentucky fire) — the ablation shot. The no-attention model
places the fire in *San Diego* (unsupported-content 0.56) for an article about
Louisville, Kentucky. Note the LSTM *beats* the LLM here (36.9 vs 33.3): a single
example never carries the claim, the aggregate does.

**`--example 112`** (Celtic vs Kilmarnock) — the fluent-but-wrong shot. Four
failures in two sentences: hat-trick then brace about the same player; near-
duplicate sentence structure showing repetition evading the trigram filter;
"to win a win"; and no player named at all, because the names are out of
vocabulary. **ROUGE-2 exactly 0.0.**

**`examples/demo_article_battery.txt`** — the out-of-domain shot. OOV rate 5.3%
vs 1.83% in-domain; the model cannot emit *electrolyte*, so it stops the clause
at "have developed a battery".

Avoid test example 3's neighbours in the beauty-pageant story — the reference
contains a censored slur.

---

## Sign-off

```
Reviewed by: ______________________    Date: __________
Ran the commands above and output matched:   [ ] yes  [ ] no — differences:

Lead-3 reproduction confirmed (40.04 / 17.50 / 36.34):   [ ] yes  [ ] no

Findings / concerns:


I can explain this component and its design decisions:   [ ] yes  [ ] no
```
