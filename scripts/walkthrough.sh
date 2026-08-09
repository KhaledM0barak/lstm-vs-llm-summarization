#!/usr/bin/env bash
# Self-narrating 8-minute demo walkthrough.
#
# Prints a framed explanation before each segment, pauses long enough to read it,
# then runs the command. Start a screen recording, run this, and let it play --
# no voice-over and no editing required.
#
#   bash scripts/walkthrough.sh              # ~8 minutes, paced for recording
#   bash scripts/walkthrough.sh --fast       # no pauses, for rehearsing
#   bash scripts/walkthrough.sh --step       # advance manually with Enter
#   bash scripts/walkthrough.sh --from 4     # start at segment 4
#   bash scripts/walkthrough.sh --replay     # replay recorded LLM responses
#                                            # (no Apple silicon / no 4.5 GB model)
#
# Pacing is tuned so the whole run lands near 8:00 including the ~8s each demo
# command takes. Use --fast first to check everything works, then record.
set -uo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
MODE="paced"
START_AT=1
PACE=1.0          # multiplier on every reading pause
DEMO_ARGS=()      # extra flags passed to every src.demo invocation

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast) MODE="fast"; shift ;;
        --step) MODE="step"; shift ;;
        --from) START_AT="$2"; shift 2 ;;
        --pace) PACE="$2"; shift 2 ;;
        --replay) DEMO_ARGS+=(--replay-llm); shift ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

COLS=$(tput cols 2>/dev/null || echo 100)
[[ $COLS -gt 120 ]] && COLS=120
BOLD=$'\033[1m'; DIM=$'\033[2m'; CYAN=$'\033[36m'; YELLOW=$'\033[33m'; OFF=$'\033[0m'

SEGMENT=0

rule() { printf "%s%s%s\n" "$CYAN" "$(printf '─%.0s' $(seq 1 "$COLS"))" "$OFF"; }

# Pause long enough to read. Base values are sized at roughly 2.5 words per
# second -- careful technical reading, not skimming -- and scaled by --pace.
pause() {
    case "$MODE" in
        fast) sleep 0.3 ;;
        step) printf "%s    [Enter to continue]%s" "$DIM" "$OFF"; read -r _ ;;
        *)    awk -v b="${1:-12}" -v p="$PACE" 'BEGIN{printf "%.1f", b*p}' \
                | xargs sleep ;;
    esac
}

# say <title> <line>...   -- a framed segment header
say() {
    local title="$1"; shift
    SEGMENT=$((SEGMENT + 1))
    echo; rule
    printf "%s  %d. %s%s\n" "$BOLD" "$SEGMENT" "$title" "$OFF"
    rule
    for line in "$@"; do
        printf "  %s\n" "$line"
    done
    echo
}

# run <command...>  -- echo the command, then execute it
run() {
    printf "%s  $ %s%s\n\n" "$YELLOW" "$*" "$OFF"
    "$@"
}

skip() { [[ $SEGMENT -lt $START_AT ]]; }

clear
cat <<BANNER

  ${BOLD}LSTM Sequence-to-Sequence vs. a Pretrained LLM${OFF}
  Abstractive summarization on CNN/DailyMail

  CP-468 · Group 17
  Bahammam · Bastug · Gundogan · Hagi · Mobarak

  ${DIM}github.com/KhaledM0barak/lstm-vs-llm-summarization${OFF}

BANNER
pause 8

# ─────────────────────────────────────────────────────────── 1. Task and data
say "The task and the data" \
  "Abstractive summarization: a news article in, a short summary out." \
  "CNN/DailyMail 3.0.0 — Apache-2.0, official article-disjoint splits." \
  "" \
  "80,000 training pairs. Every system in this project — the LSTM, all four" \
  "ablations, and all five LLM settings — is scored on the SAME 500 held-out" \
  "articles, drawn once with a fixed seed before any model was built."
if ! skip; then
    run $PY -c "
import json; m=json.load(open('data/processed/dataset_meta.json'))
print(f\"  dataset : {m['dataset']} ({m['config']})   license: {m['license']}\")
print(f\"  official splits: {m['official_split_sizes']}\")
for k,v in m['splits'].items():
    print(f\"    {k:<12} n={v['n']:>6,}   mean article {v['mean_src_tokens']:>5} tokens   mean summary {v['mean_tgt_tokens']}\")
"
fi
pause 22

# ────────────────────────────────────────────────── 2. Is the pipeline sound?
say "Before comparing anything: is the measurement trustworthy?" \
  "A summarization score means nothing if ROUGE is configured wrongly." \
  "" \
  "So we reproduced a published baseline first. Lead-3 — literally the article's" \
  "first three sentences — on the full 11,490-article test set:"
if ! skip; then
    run sed -n '/## Overall/,/^$/p' results/lead3_fulltest/results.md
    echo "  See et al. (2017) published:  40.34 / 17.70 / 36.57"
    echo
    echo "  Agreement to ~0.3 ROUGE. The tokenization, sentence splitting,"
    echo "  reference construction and ROUGE settings are right."
fi
pause 26

# ────────────────────────────────────────────────────────────── 3. The model
say "The model — built from primitives, not a framework" \
  "The assignment requires our own implementation. No Fairseq, no OpenNMT," \
  "no HuggingFace Seq2SeqTrainer anywhere in the model, training or decoding."
if ! skip; then
    run grep -rho "nn\.[A-Za-z]*" src/models/*.py | sort | uniq -c | sort -rn | head -6
    echo
    echo "  Embedding(50k x 256) -> BiLSTM encoder -> Bahdanau attention"
    echo "    -> LSTM decoder with input feeding -> tied output projection"
    echo "  15,347,280 parameters (83% of them the embedding table)"
fi
pause 22

say "The one line that matters most" \
  "Batches mix articles of different lengths, so short ones are padded." \
  "This line zeroes the padded positions before the softmax:"
if ! skip; then
    run grep -n "masked_fill" src/models/attention.py
    echo
    echo "  Without it, measured on a short article in a mixed batch:"
    echo "    62.8% of the attention mass lands on PADDING."
    echo "  It never raises an error. It just quietly corrupts the context vector,"
    echo "  and makes the output depend on which articles share the batch."
fi
pause 26

# ──────────────────────────────────────────────────── 4. Live demo, ablations
say "Live: one article through every system" \
  "Same article, same 400-word window, LSTM and LLM and all three ablations." \
  "" \
  "Watch the no-attention row."
if ! skip; then
    run $PY -m src.demo --example 3 --ablations ${DEMO_ARGS+"${DEMO_ARGS[@]}"}
    echo
    echo "  The no-attention model places a Louisville, Kentucky fire in SAN DIEGO."
    echo "  56% of its content words never appear in the article. That is the"
    echo "  fixed-vector bottleneck: with no attention the decoder cannot select"
    echo "  which part of a 400-token article to describe."
    echo
    echo "  Note the LSTM beats the LLM on THIS example, 36.9 to 33.3. One example"
    echo "  never carries a claim — the aggregate over 500 does."
fi
pause 34

# ─────────────────────────────────────────────────────────── 5. Out of domain
say "Live: text the model has never seen" \
  "A battery-chemistry article — nothing like 2015 news wire copy." \
  "The vocabulary is fixed at 50,000 types, built from the training split."
if ! skip; then
    run $PY -m src.demo --file examples/demo_article_battery.txt ${DEMO_ARGS+"${DEMO_ARGS[@]}"}
    echo
    echo "  The LSTM says 'have developed a battery' and stops."
    echo "  'electrolyte' is not in its vocabulary. Neither are 'anode',"
    echo "  'graphite', 'fluorinated', or the researcher's name."
    echo "  Out-of-vocabulary rate here: 5.3%, against 1.83% in-domain."
    echo "  So it skips every technical finding for the one sentence it can say."
fi
pause 32

# ───────────────────────────────────────────────────────────────── 6. Results
say "Results — and the finding we did not expect" \
  "All eleven systems, scored on the same 500 articles."
if ! skip; then
    run sed -n '/## Overall/,/## Diagnostics/p' results/results.md
fi
pause 26

say "Three numbers worth stopping on" \
  "" \
  "  -14.02   Removing attention costs 14 ROUGE-1 and collapses ROUGE-2" \
  "           fourfold. That is the San Diego output, quantified." \
  "" \
  "  +2.83    The two prompt variants differ by 2.83 ROUGE-1 — same model," \
  "           same data, same decoding. The LSTM-to-LLM gap is 6.35." \
  "           So 45% of the 'model gap' is how we phrased the request." \
  "" \
  "  39.89    Lead-3 — copying the first three sentences — beats our model AND" \
  "           four of the five LLM configurations. That is a statement about" \
  "           what ROUGE rewards, not about the systems."
pause 34

say "Significance, done properly" \
  "Both systems see the same articles, so the scores are correlated." \
  "Independent confidence intervals overlap on ROUGE-2 — you would conclude" \
  "nothing. A paired bootstrap resamples articles once and applies the same" \
  "resample to both systems:"
if ! skip; then
    run sed -n '/## Paired bootstrap/,/## ROUGE-1 by/p' results/results.md | head -20
    echo "  Note the 100-token encoder window: p = 0.053, NOT significant."
    echo "  Cutting the encoder's input fourfold changes nothing we can measure."
fi
pause 30

# ─────────────────────────────────────────────────────── 7. Error analysis
say "Error analysis — fluent, and completely wrong" \
  "A football match report. Read the LSTM output carefully."
if ! skip; then
    run $PY -m src.demo --example 112 ${DEMO_ARGS+"${DEMO_ARGS[@]}"}
    echo
    echo "  Four failures in two sentences:"
    echo "    - 'a hat-trick' then 'a brace' — three goals, then two, same player"
    echo "    - the two sentences are near-duplicates: repetition slipping past"
    echo "      our trigram filter by varying one word per slot"
    echo "    - it ends 'to win a win'"
    echo "    - nobody is named. Griffiths, Westlake, Commons are all OOV, so it"
    echo "      says 'the celtic striker', 'the celtic forward'"
    echo
    echo "  ROUGE-2 is exactly 0.0 — fluent English sharing not one bigram"
    echo "  with the reference. This is what 'fluent but wrong' looks like."
fi
pause 36

say "And the reverse case — where the metric is the problem" \
  "Test example 4. The LSTM scores 57.1, the LLM 26.3." \
  "" \
  "  Reference: sarah stage, 30, has documented her changing figure via her" \
  "             instagram page throughout her pregnancy ..." \
  "" \
  "  LSTM  57.1  sarah stage, a 30-year-old underwear model ... has documented" \
  "              her changing figure via her instagram page throughout her" \
  "              pregnancy." \
  "" \
  "  LLM   26.3  Sarah Stage ... has shared a photo of her barely-there baby" \
  "              bump 10 days before her due date. The model, who has 1.5" \
  "              million Instagram followers ..." \
  "" \
  "The LLM's summary is entirely TRUE. It scores less than half because it" \
  "chose different true facts than the editor did. Our model wins by echoing" \
  "the reference's framing — which it learned from this dataset." \
  "" \
  "That is why Lead-3, which is pure copying, beats almost everything."
pause 44

# ──────────────────────────────────────────────────── 8. Trade-offs and close
say "The engineering trade-off" \
  "Measured on the same machine, same GPU, no network."
if ! skip; then
    run $PY -c "
rows = [
 ('',                    'LSTM + attention', 'Llama 3.1 8B (4-bit)'),
 ('parameters',          '15.3M',            '~8B'),
 ('size on disk',        '61 MB',            '~4.5 GB'),
 ('one-time training',   '8.73 GPU-hours',   'pretrained'),
 ('per summary',         '0.231 s',          '2.85 s'),
 ('throughput',          '259 / min',        '21 / min'),
 ('monetary cost',       '\$0.00',            '\$0.00 (local)'),
]
for a,b,c in rows:
    print(f'  {a:<20} {b:<20} {c}')
"
fi
pause 22

say "What we conclude" \
  "The LLM wins by 6.35 ROUGE-1, p < 0.0001. Real — but 6 points, not an" \
  "order of magnitude, against a model 500x smaller trained for 8.7 GPU-hours." \
  "" \
  "45% of that gap is prompt phrasing, not capability." \
  "Lead-3 beats four of the five LLM settings, which limits what ROUGE can" \
  "tell us at all." \
  "And removing attention costs 14 points while cutting the encoder window" \
  "fourfold costs nothing measurable — so the recurrent bottleneck is not the" \
  "binding constraint here." \
  "" \
  "The LSTM is 12x faster, 74x smaller, runs offline, and its failure modes are" \
  "fixable at decoding time. For a task with training data and a latency budget," \
  "it is still defensible. For a task with no training data, it has no answer —" \
  "and that, rather than six ROUGE points, is what pretraining bought."
pause 40

echo
rule
printf "%s  Code, report and tests: github.com/KhaledM0barak/lstm-vs-llm-summarization%s\n" "$BOLD" "$OFF"
printf "  %s162 tests · full report in reports/report.pdf%s\n" "$DIM" "$OFF"
rule
echo
