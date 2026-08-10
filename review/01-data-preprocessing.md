# Review 1 — Data & preprocessing

**Owner:** Mohanad Bahammam
**Files:** `src/data/prepare.py` (166), `tokenizer.py` (91), `vocab.py` (93), `build_vocab.py` (53), `dataset.py` (199)

---

## Run this first

```bash
cd ~/lstm-vs-llm-summarization && source .venv/bin/activate

# Split sizes and dataset provenance
python -c "
import json; m=json.load(open('data/processed/dataset_meta.json'))
print('dataset:', m['dataset'], m['config'], '| license:', m['license'])
print('official:', m['official_split_sizes'])
for k,v in m['splits'].items(): print(f'  {k:12s} n={v[\"n\"]:>6,}  src={v[\"mean_src_tokens\"]}  tgt={v[\"mean_tgt_tokens\"]}')
"

# Vocabulary coverage
cat data/processed/vocab.stats.json

# Tokenizer behaviour on a hard case
python -c "
from src.data.tokenizer import tokenize, normalize, split_sentences
s = 'WASHINGTON (CNN) -- The U.S. said Mr. Smith paid \$3.5m on Jan. 4, 2015. He left.'
print('normalized:', normalize(s))
print('tokens    :', tokenize(s)[:20])
print('sentences :', split_sentences(normalize(s)))
"
```

**Expect:** Apache-2.0; 79,996 / 3,000 / 11,490 / 500; coverage 0.9817, OOV 0.01831.
The sentence splitter must return **2** sentences, not 5 — it must not break on
"U.S.", "Mr.", or "Jan.".

---

## Read these specifically

- `tokenizer.py` → `normalize()`, `_PREAMBLE_RE`, `split_sentences()` and `_ABBREVIATIONS`
- `vocab.py` → `Vocab.build()` — note the tie-breaking sort
- `prepare.py` → `_clean_summary()`, `_to_record()`, and the split logic in `main()`
- `dataset.py` → `collate_batch()` and `BucketBatchSampler`

---

## Verify these claims

- [ ] The vocabulary is built from **train only** (`build_vocab.py` reads `train.jsonl` and nothing else)
- [ ] The 500-article shared set is drawn from the **processed test split** with a fixed seed, and its indices are saved in `dataset_meta.json`
- [ ] Sorting in `Vocab.build()` breaks ties **alphabetically**, so the vocabulary is byte-identical across machines
- [ ] Articles shorter than 30 tokens or summaries shorter than 5 are dropped (`_to_record`)
- [ ] `collate_batch` builds `src_mask` from `src.ne(PAD_ID)` — the mask the attention depends on

---

## Worth scrutinizing

These are debatable choices, not known bugs. Form your own view.

1. **`_clean_summary` joins highlight bullets with `" . "`.** CNN/DailyMail references are separate bullets; we flatten them into one string. This is applied identically to every system so it can't favour one, but it does affect absolute ROUGE. Is the joining sensible? What would change if we used newlines?

2. **`_PREAMBLE_RE` doesn't catch everything.** It strips a leading `(cnn)` but not `los angeles (cnn)`, so some articles keep a dateline glued to the first word (`"(cnn)it's"`). Check how often this happens and whether it matters after tokenization.

3. **Shape quantization loses data.** `collate_batch` rounds padding up to multiples of 64/16, and the training sampler drops each pool's ragged final batch — under 2% of examples per epoch. Confirm the reshuffle means a *different* remainder is dropped each epoch (`BucketBatchSampler.set_epoch`), so no example is permanently excluded.

4. **Lowercasing is irreversible.** We lowercase everything, so the model can never produce a capital letter and `detokenize()` is only a best-effort inverse. Is that acceptable for the summaries we generate? What does it cost us on named entities?

---

## Be ready to answer

- *"How do you know there's no train/test leakage?"*
- *"Why a word-level tokenizer instead of subwords like BPE?"* (the real answer is about making OOV **observable** — a subword vocabulary hides it by construction)
- *"Your vocabulary covers 98.17% of tokens. What happens to the other 1.83%?"*
- *"Why truncate the source to 400 tokens when the mean article is 785?"*

---

## Sign-off

```
Reviewed by: Mohanad Bahammam            Date: 2026-08-10
Ran the commands above and output matched:   [x] yes  [ ] no — differences:

Findings / concerns:
  Three defects found in split_sentences() and fixed during review:
  sentence-final digits were swallowed ("jan. 4."); "no" in the abbreviation
  list broke every sentence ending in that word; "u.s." was unprotected.
  Lead-3 moved 40.00 -> 40.04, closer to the published 40.34. Sentence
  boundaries feed both Lead-3 and ROUGE-Lsum, so this affected every number
  in the report. Regression tests added in tests/test_data.py.

I can explain this component and its design decisions:   [x] yes  [ ] no
```
