# Narration script

For recording the voice-over **after** the screen capture. Read this aloud while
the video plays back — nothing here needs memorizing.

Cue times are measured from a real paced run of `scripts/walkthrough.sh`
(total **7:11**). Each block is written to fit its window at normal speaking
pace with room to breathe, so if you start a block when its cue appears you will
finish before the next one.

**How to use it:** put this on your phone or a second window, start playback,
and read each block when its heading appears on screen. If you fall behind,
skip to the next block at its cue rather than catching up — a short silence is
invisible, overlapping the next segment is not.

Bracketed lines are stage directions, not spoken.

---

### 0:00 — Title card *(8 s)*

Our CP-468 project: an LSTM sequence-to-sequence model, built from scratch,
against a pretrained LLM.

---

### 0:08 — 1. The task and the data *(22 s)*

The task is abstractive summarization — an article in, a short summary out.
CNN/DailyMail, Apache-2.0, official splits, eighty thousand training pairs.
Every system you'll see is scored on the same five hundred held-out articles,
drawn once with a fixed seed.

---

### 0:30 — 2. Is the measurement trustworthy? *(26 s)*

Before comparing anything, we had to know the measurement was trustworthy. So
we reproduced a published baseline first — Lead-3, the article's first three
sentences, on the full test set. We get forty point oh four; See et al.
published forty point three four. That agreement is what makes every other
number believable.

---

### 0:56 — 3. The model *(22 s)*

The assignment requires our own implementation — no Fairseq, no OpenNMT, no
HuggingFace trainer anywhere in the model, training, or decoding. These are the
PyTorch primitives we use: an embedding table, a bidirectional LSTM encoder,
Bahdanau attention, and an LSTM decoder. Fifteen million parameters.

---

### 1:18 — 4. The one line that matters most *(26 s)*

This is the line that matters most. Batches mix articles of different lengths,
so short ones get padded. This zeroes those padded positions before the softmax.
Without it, sixty-three percent of the attention mass lands on padding — and it
never throws an error. It just quietly corrupts the context vector.

---

### 1:44 — 5. Live: one article through every system *(45 s)*

[The command runs for about ten seconds. Start talking immediately.]

Now the live demo — the same article, the same four-hundred-word window, through
our model, the LLM, and all three ablations.

[Output appears.]

The reference summary, then our model at thirty-six point nine, the LLM at
thirty-three point three. Now watch the no-attention row: it places a
Louisville, Kentucky fire in San Diego. Fifty-six percent of its content words
never appear in the article. That's the fixed-vector bottleneck — with no
attention the decoder can't select what to describe. And our model beats the LLM
on this example. One example never carries a claim; the aggregate does.

---

### 2:29 — 6. Live: text the model has never seen *(41 s)*

[Command runs about eight seconds.]

Now something outside the training distribution — a battery chemistry article,
nothing like 2015 news wire copy. The vocabulary is fixed at fifty thousand
types, built from the training split only.

[Output appears.]

The LSTM says "have developed a battery" and stops. "Electrolyte" isn't in its
vocabulary. Neither is "anode", "graphite", or the researcher's name.
Out-of-vocabulary rate here is five point three percent, against one point eight
in domain. So it skips every technical finding for the one sentence it can
actually say. The LLM handles it.

---

### 3:09 — 7. Results *(26 s)*

All eleven systems, scored on the same five hundred articles, with bootstrap
confidence intervals. The LLM's best prompt is at the top, our LSTM is in the
middle — and Lead-3, which is pure copying with no model at all, is sitting
above four of the five LLM settings.

---

### 3:35 — 8. Three numbers *(34 s)*

Three numbers. Minus fourteen: removing attention costs fourteen ROUGE-1 and
collapses ROUGE-2 fourfold — the San Diego output, quantified. Plus two point
eight three: the gap between our two prompts, same model, same data, while the
whole LSTM-to-LLM gap is six point three five. So forty-five percent of the
apparent model gap is phrasing. And thirty-nine point eight nine — Lead-3,
beating four of five LLM settings.

---

### 4:09 — 9. Significance *(30 s)*

Both systems see the same articles, so their scores are correlated. Independent
confidence intervals overlap on ROUGE-2 — from those alone you'd conclude
nothing. A paired bootstrap resamples the articles once and applies the same
resample to both systems. That's what licenses the claim. And note the row we
report as *not* significant: the hundred-token encoder window, p equals point
zero five three.

---

### 4:39 — 10. Error analysis *(45 s)*

[Command runs about ten seconds.]

Now the error analysis. This is a football match report.

[Output appears.]

Read the LSTM output carefully — there are four failures in two sentences. It
says a hat-trick, then a brace, about the same player: three goals, then two.
The two sentences are near-duplicates, repetition slipping past our trigram
filter by varying one word per slot. It ends with "to win a win". And nobody is
named at all — the players are out of vocabulary, so it says "the Celtic
striker". ROUGE-2 here is exactly zero. Fluent English sharing not one bigram
with the reference.

---

### 5:24 — 11. The reverse case *(44 s)*

And the reverse case, where the metric is the problem. Our model scores
fifty-seven here, the LLM twenty-six. But read them. The reference says Sarah
Stage documented her changing figure on Instagram through her pregnancy — our
model echoes that almost word for word. The LLM reports the photo before her due
date and her follower count. Everything it said is true. It scores less than
half because it chose different true facts than the editor did. That is why
Lead-3, pure copying, beats almost everything.

---

### 6:08 — 12. The engineering trade-off *(22 s)*

The trade-off, measured on the same machine. Fifteen million parameters against
roughly eight billion. Sixty-one megabytes on disk against four and a half
gigabytes. Point two three seconds per summary against two point eight five.
Twelve times faster, seventy-four times smaller.

---

### 6:30 — 13. What we conclude *(40 s)*

The LLM wins by six point three five ROUGE-1, p below point zero zero zero one.
That's real — but six points, not an order of magnitude, against a model five
hundred times smaller. Forty-five percent of that gap is prompt phrasing.
Lead-3 beats four of five LLM settings, which limits what ROUGE can tell us at
all. With training data and a latency budget, the LSTM is still defensible. With
no training data it has no answer — and that is what pretraining bought.

---

*[Let the closing rule sit for a second, then stop.]*
