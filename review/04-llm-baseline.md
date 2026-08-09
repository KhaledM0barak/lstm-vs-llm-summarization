# Review 4 — LLM baseline

**Owner:** Ayuub Hagi
**Files:** `src/llm/prompts.py` (89), `backends.py` (204), `baseline.py` (359)

The assignment is specific here: *at least two prompt variants*, zero-shot **and**
few-shot (k = 3–5), exact prompts in the report, and a cost estimate.

---

## Run this first

```bash
cd ~/lstm-vs-llm-summarization && source .venv/bin/activate

# See the exact prompts without generating anything
python -m src.llm.baseline --all --dry-run

# What's completed so far
wc -l runs/llm/*.jsonl

# Usage/cost accounting for a finished setting
python -c "
import json; m=json.load(open('runs/llm/A_fewshot.meta.json'))
print('model:', m['model'], '| backend:', m['backend'], m['backend_kind'])
print('variant:', m['prompt_variant'], m['prompt_variant_name'], '| shots:', m['shots'])
print('input condition:', m['input_condition'])
print(json.dumps(m['usage'], indent=2))"
```

**Expect:** four settings; zero-shot prompts ~600 tokens, few-shot ~2,800;
`input_condition` = `truncated_400_words`; cost `$0.00` with `gpu_hours` reported.

---

## Read these specifically

- `prompts.py` → both `SYSTEM_*` and `USER_*` strings, and `build_messages()`
- `backends.py` → `MLXBackend.generate_batch()` and `cost_usd()`
- `baseline.py` → `truncate_words()`, `sample_exemplars()`, `clean_output()`, and the resume logic in `run_setting()`

---

## Verify these claims

- [ ] Few-shot exemplars come from **`train.jsonl`**, never validation or test — this is the leakage control
- [ ] Exemplar choice is seeded, so the few-shot prompt is identical across runs
- [ ] The LLM receives the article truncated to the **same 400 words** the LSTM encoder sees
- [ ] Decoding is greedy (`temp=0.0`), so the baseline is deterministic
- [ ] `clean_output()` is applied identically to **every** setting — it can't favour one prompt variant
- [ ] Exemplars are passed as real prior conversation turns, not pasted into one user message

---

## Worth scrutinizing

1. **`clean_output()` strips preambles.** It removes leading `"Here are the highlights:"`, `"Summary:"`, etc. This is defensible only because it's symmetric across variants — but it *does* hide a real behaviour (the model ignoring "no preamble" instructions). Check how often it fires: compare the `raw` and `prediction` fields in the JSONL. If variant A triggers it far more than B, that's a finding worth reporting as format-adherence, not something to quietly normalize away.

2. **`max_tokens=200` may truncate.** Mean output is ~100 tokens so it rarely binds — but check `stop_reason` / output lengths for outputs sitting exactly at the cap.

3. **The model over-produces.** References average 58 tokens; the LLM produces ~81. Variant B explicitly asks for ~55 words and still overshoots. That's a real instruction-following finding — make sure it reaches the report rather than being treated as noise.

4. **Prompt caching isn't used.** The ~2,200-token few-shot prefix is identical across all 500 examples and is re-processed every time, which is why few-shot runs at ~9 s/article vs ~3.6 s zero-shot. MLX supports `prompt_caches`; we didn't implement it. Legitimate limitation to mention — quantify it if you want (roughly a 3× saving on few-shot).

5. **Batch latency is attributed evenly.** `generate_batch` divides one batch's wall-clock across its items, so per-item latency is a throughput figure, not a true single-request latency. Fine for cost accounting; don't call it "response time" in the report.

---

## Be ready to answer

- **"Why two prompt variants?"** — A is a natural request; B describes the CNN/DailyMail reference style (length, sentence count, register). The gap between them separates *summarization ability* from *fitting the metric's stylistic target*. With one prompt, the whole comparison could be an artifact of prompt quality.
- **"Isn't it unfair that the LLM only sees 400 words?"** — That's the point. Both systems get the same input window, or you're measuring "more input", not "better model". The full-article condition is run separately and reported as its own row.
- *"What did it cost?"* — $0.00. It's Llama 3.1 8B Instruct, 4-bit, running locally on the M4 Pro's GPU via MLX. Cost is reported as GPU-hours, which is what the assignment asks for on the local-model path.
- *"Why an 8B open model rather than GPT/Claude?"* — the assignment permits it, it's free, and it makes the comparison a *lower bound* on the frontier-model gap. The harness supports an API backend (`--backend anthropic`) if we want to re-run against a frontier model.

---

## Sign-off

```
Reviewed by: ______________________    Date: __________
Ran the commands above and output matched:   [ ] yes  [ ] no — differences:

Findings / concerns:


I can explain this component and its design decisions:   [ ] yes  [ ] no
```
