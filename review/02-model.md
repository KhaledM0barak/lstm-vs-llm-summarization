# Review 2 — Model implementation

**Owner:** Yakup Bastug
**Files:** `src/models/encoder.py` (84), `attention.py` (158), `decoder.py` (149), `seq2seq.py` (313)

This is the component the assignment cares most about: *"You must implement the
model yourself; do not use prebuilt seq2seq pipelines."*

---

## Run this first

```bash
cd ~/lstm-vs-llm-summarization && source .venv/bin/activate

# Parameter count and architecture
python -c "
import torch
from src.models.seq2seq import Seq2Seq, ModelConfig
ck = torch.load('runs/base/best.pt', map_location='cpu', weights_only=False)
m = Seq2Seq(ModelConfig(**ck['model_config']))
print(m.num_parameters())
print('attention:', ck['model_config']['attention'], '| bidirectional:', ck['model_config']['bidirectional'])
"

# Prove nothing prebuilt is imported
grep -rn "fairseq\|opennmt\|Seq2SeqTrainer\|transformers" src/models/ || echo "clean: no prebuilt seq2seq imports"

# The model actually working
python -m src.demo --example 3 --no-llm --device cpu
```

**Expect:** 15,347,280 total / 12,800,000 embedding; `bahdanau`, bidirectional
`True`; "clean"; and a fluent summary of the Kentucky fire article.

---

## Read these specifically

- `encoder.py` → `Encoder.forward()` — the `pack_padded_sequence` call and the bridge (`bridge_h`/`bridge_c`)
- `attention.py` → `BahdanauAttention.forward()` — **this is the line to know**: `scores.masked_fill(~mask, NEG_INF)`
- `decoder.py` → `Decoder.forward()` — all three paths (input feeding, batched, per-step) and `Decoder.step()`
- `seq2seq.py` → `generate_beam()` and `_block_repeat_trigrams()`

---

## Verify these claims

- [ ] Only `nn.LSTM`, `nn.Embedding`, `nn.Linear`, and `nn.Dropout` are used — no prebuilt seq2seq anything
- [ ] The encoder is bidirectional and the two directions are **concatenated** before the bridge projection
- [ ] Attention masks padding **before** the softmax, not after
- [ ] The output projection is **weight-tied** to the embedding (`out_proj.weight = embedding.weight`)
- [ ] `<pad>` and `<unk>` are set to `-inf` at every generation step
- [ ] Beam search applies the GNMT length penalty `((5+|Y|)/6)^alpha` when a hypothesis completes

---

## Worth scrutinizing

Not known bugs — the places I'd look first.

1. **Beam search re-padding.** In `generate_beam`, when a beam finishes, the survivors are padded back to `beam_size` by duplicating the last hypothesis with score `-inf`. Convince yourself this can't let a duplicate win, and that the `-inf` scores are actually excluded from the next `topk`.

2. **Trigram blocking is O(T) per step.** `_block_repeat_trigrams` rebuilds its `seen` dictionary from scratch on every decoding step. Correct but wasteful. Does it block the right thing — the token that would *complete* a repeated trigram, not the trigram itself?

3. **Input feeding forces a Python loop.** The decoder can't run as one fused LSTM call when input feeding is on, which is why training is slower than the batched path. Read the three branches in `Decoder.forward()` and be able to say why the additive score can't be batched over decoder steps (hint: the intermediate tensor would be `(B, T, S, attn_size)` ≈ 330 GB at our sizes).

4. **`NoAttention` isn't literally "no attention".** It returns the *mean* of unmasked encoder states, not the final hidden state. That's a defensible way to implement the fixed-vector bottleneck while keeping the module interface, but be ready to say so rather than being surprised by it.

5. **Enc/dec layer counts must match.** `Seq2Seq.__init__` raises if they differ, because the bridge maps per-layer states directly. Fine for our 1-layer config; know that it's a constraint.

---

## Be ready to answer

- **"Show me your attention mechanism."** — open `attention.py`, walk the additive score `v·tanh(W_dec h + W_enc m)`, then point at the `masked_fill` and explain *why*: without it the decoder puts probability mass on `<pad>` for every short article batched with long ones, quietly corrupting the context vector.
- *"Why suppress `<unk>` at inference?"* — an `<unk>` in a generated summary is never useful output; it's a pure error. Note the consequence: our OOV rate is 0.000 by construction, so the OOV failure shows up as **substitution**, not as visible `<unk>`.
- *"What does input feeding do?"* — feeds the previous step's attentional vector into the current input, so the decoder knows what it already attended to.
- *"Why tie the output projection to the embedding?"* — saves 12.8M parameters and regularizes; requires `hidden_size == emb_dim`.

---

## Sign-off

```
Reviewed by: Yakup Bastug                Date: 2026-08-10
Ran the commands above and output matched:   [x] yes  [ ] no — differences:

Findings / concerns:
  Verified: 15,347,280 total parameters / 12,800,000 embedding, attention
  'bahdanau', bidirectional True -- matches the report. grep over src/models/
  for fairseq/opennmt/Seq2SeqTrainer/transformers returns nothing, so the
  from-scratch requirement holds. Model decodes correctly on CPU as well as
  MPS (example 3, ROUGE-1 36.9, identical on both devices).
  No discrepancies found.

I can explain this component and its design decisions:   [x] yes  [ ] no
```
