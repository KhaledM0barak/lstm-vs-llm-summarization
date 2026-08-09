# Review 3 — Training & performance

**Owner:** Orhan Gundogan
**Files:** `src/train.py` (231), `configs/*.yaml`, `scripts/train_all.sh`

This section has the most interesting story in the project. Two performance bugs
were found by *measuring* rather than by reading code, and one of them was a 34–55×
speedup.

---

## Run this first

```bash
cd ~/lstm-vs-llm-summarization && source .venv/bin/activate

# Every training run's real numbers
for d in base no_attention unidirectional short_context; do
python -c "
import json; s=json.load(open('runs/$d/train_summary.json')); h=s['history']
print(f\"{s['run_name']:16s} params={s['parameters']['total']:>10,} epochs={len(h)} hours={s['total_train_hours']:>5.2f} best_val={s['best_val_loss']:.4f} ppl={min(x['val_ppl'] for x in h):.1f}\")"
done

# Per-epoch curve for the main model
python -c "
import json
for h in json.load(open('runs/base/train_summary.json'))['history']:
    print(f\"epoch {h['epoch']}  train={h['train_loss']:.4f}  val={h['val_loss']:.4f}  ppl={h['val_ppl']:>6.1f}  {h['epoch_seconds']:.0f}s\")"

# End-to-end smoke test (~30s)
python -m src.train --config configs/smoke.yaml
```

**Expect:** base 15,347,280 params / 3.05 h / val 4.6219 / PPL 35.6; no_attention
PPL 121.7; unidirectional 40.0; short_context 65.5. Validation loss must fall
monotonically across all 5 epochs.

---

## Read these specifically

- `train.py` → the training loop, `evaluate_loss()`, and the early-stopping / checkpoint logic
- `seq2seq.py` → `loss_from_states()` — the chunked projection (this is fix #2 below)
- `dataset.py` → `SRC_LEN_MULTIPLE` / `TGT_LEN_MULTIPLE` and `_round_up` (fix #3)
- `configs/base.yaml` vs. the three ablation configs — diff them and confirm **exactly one thing** changes in each

---

## Verify these claims

- [ ] All four configs use identical hyperparameters, data, and seed — only the ablated component differs
- [ ] Early stopping is on **validation** loss, patience 2; the checkpoint saved is the best-validation one, not the last
- [ ] Loss is normalized per target token, so gradient scale doesn't depend on how many tokens land in a batch
- [ ] `train_summary.json` records hardware, device, wall-clock, and parameter count for every run
- [ ] Raw article text is dropped after encoding (`train_ds.records = []`) — memory, not correctness

---

## The two performance findings — know these cold

**Fix 1 — chunked vocabulary projection (memory).**
Projecting decoder states to the 50k vocabulary in one matmul makes a `(B, T, V)`
tensor: at batch 64 × 100 steps that's **1.28 GB in fp32**, and the original
loss function copied it twice more via boolean-mask indexing. The machine went
**9.5 GB into swap** and throughput fell from 1.1 s/batch to 3.2 s/batch and
worsening. Fixed by projecting in 16-step chunks with `F.cross_entropy` (native
label smoothing + `ignore_index`, so no masked copies). Peak RSS: **1.15 GB**.

**Fix 2 — padded-shape quantization (the big one).**
PyTorch's Metal (MPS) backend compiles a **kernel per distinct tensor shape**.
Length-bucketed batching produced a near-unique `(batch, src_len, tgt_len)` triple
almost every step, so training spent most of its wall-clock in shader compilation
— `MTLCompilerService` pinning two CPU cores — instead of arithmetic. Rounding
padded lengths to multiples of 64/16 and fixing the batch dimension collapsed
thousands of shapes into a few dozen:

| Configuration | Before | After | Speedup |
|---|---|---|---|
| Bahdanau + input feeding | 2537 s | 73.7 s | **34×** |
| Multiplicative attention, batched | 2695 s | 49.1 s | **55×** |

*(one epoch over 3,840 examples, batch 64, same hardware)*

Without this, the four training runs would have taken an estimated 20+ hours
instead of **8.73**.

**The transferable point:** neither bug is visible by reading the code. Both were
found by measuring throughput and then asking why the number was wrong.

---

## Worth scrutinizing

1. **5 epochs may be short.** Validation loss was **still falling** at epoch 5 (4.622 and decreasing) — early stopping never triggered. The model is undertrained; more epochs would likely improve ROUGE. Decide whether to say this plainly in the report as a limitation. I'd argue yes.
2. **`drop_last=True` discards data.** Under 2% per epoch, and reshuffling drops a different remainder each time. Confirm that reasoning holds.
3. **The LR scheduler barely fired.** `ReduceLROnPlateau` with patience 0 should halve the LR on any non-improving epoch — but loss improved every epoch, so LR stayed at 1e-3 throughout. Check `history` and confirm.
4. **Ablation training times differ a lot** (1.11 h to 3.05 h). Make sure you can explain why — shorter encoder window and no attention are genuinely cheaper — and that this doesn't confound the comparison (it doesn't; epochs and data are equal).

---

## Be ready to answer

- *"How long did training take and on what?"* — 8.73 GPU-hours total on an Apple M4 Pro (16-core GPU, 24 GB unified memory), MPS backend.
- *"Did the model converge?"* — Honest answer: not fully. Validation loss was still improving at the epoch cap.
- *"Why is the no-attention run's perplexity so much worse?"* — 121.7 vs 35.6; without attention the decoder has only a pooled encoder state and can't select what to talk about.

---

## Sign-off

```
Reviewed by: ______________________    Date: __________
Ran the commands above and output matched:   [ ] yes  [ ] no — differences:

Findings / concerns:


I can explain this component and its design decisions:   [ ] yes  [ ] no
```
