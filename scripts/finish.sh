#!/usr/bin/env bash
# Everything after training: decode, run the LLM baseline, score, and build the
# report tables. Safe to re-run; the LLM step resumes rather than re-paying.
#
# Prerequisites:
#   - scripts/train_all.sh has finished (runs/*/best.pt exist)
#   - .env contains ANTHROPIC_API_KEY
#
# Usage:  bash scripts/finish.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
TEST=data/processed/test_llm.jsonl

step() { echo; echo "=============== $* ==============="; }

# ---------------------------------------------------------------- 1. decode
step "1/5  Decoding LSTM summaries"
for run in base no_attention unidirectional short_context; do
    if [[ -f "runs/$run/best.pt" ]]; then
        echo "--- $run (beam) ---"
        "$PY" -m src.generate --checkpoint "runs/$run/best.pt" \
            --test-file "$TEST" --out "runs/$run/preds_test_llm.jsonl" \
            --decode beam --beam-size 4
    else
        echo "--- skipping $run: no checkpoint yet ---"
    fi
done

# Decoding ablations on the main model only.
if [[ -f runs/base/best.pt ]]; then
    echo "--- base (greedy) ---"
    "$PY" -m src.generate --checkpoint runs/base/best.pt --test-file "$TEST" \
        --out runs/base/preds_greedy.jsonl --decode greedy
    echo "--- base (beam, no trigram blocking) ---"
    "$PY" -m src.generate --checkpoint runs/base/best.pt --test-file "$TEST" \
        --out runs/base/preds_beam_norepeat.jsonl --decode beam --no-block-trigram
fi

# ---------------------------------------------------------------- 2. LLM
# Local open-weights model on Apple silicon: free, and the assignment's stated
# alternative to a paid API. Run after training so neither measurement is taken
# under GPU contention. Resumable -- re-running skips completed examples.
step "2/5  LLM baseline (local open-weights via MLX)"
BACKEND="${BACKEND:-mlx}"
"$PY" -m src.llm.baseline --all --backend "$BACKEND" --batch-size 8
# Unmatched-input condition for the fairness discussion.
"$PY" -m src.llm.baseline --setting B_zeroshot --full-article \
    --backend "$BACKEND" --batch-size 4

# ---------------------------------------------------------------- 3. score
step "3/5  Scoring"
ARGS=()
add() { [[ -f "$2" ]] && ARGS+=(--system "$1=$2"); }
add lstm_beam                     runs/base/preds_test_llm.jsonl
add lstm_greedy                   runs/base/preds_greedy.jsonl
add lstm_beam_norepeat            runs/base/preds_beam_norepeat.jsonl
add no_attention                  runs/no_attention/preds_test_llm.jsonl
add unidirectional                runs/unidirectional/preds_test_llm.jsonl
add short_context                 runs/short_context/preds_test_llm.jsonl
add llm_A_zeroshot                runs/llm/A_zeroshot.jsonl
add llm_A_fewshot                 runs/llm/A_fewshot.jsonl
add llm_B_zeroshot                runs/llm/B_zeroshot.jsonl
add llm_B_fewshot                 runs/llm/B_fewshot.jsonl
add llm_B_zeroshot_fullarticle    runs/llm/B_zeroshot_fullarticle.jsonl

"$PY" -m src.evaluate --test-file "$TEST" "${ARGS[@]}" --out-dir results/

# ---------------------------------------------------------------- 4. qualitative
step "4/5  Qualitative comparison"
QARGS=()
[[ -f runs/base/preds_test_llm.jsonl ]] && QARGS+=(--system "lstm_beam=runs/base/preds_test_llm.jsonl")
[[ -f runs/llm/B_zeroshot.jsonl ]] && QARGS+=(--system "llm_B_zeroshot=runs/llm/B_zeroshot.jsonl")
if [[ ${#QARGS[@]} -eq 4 ]]; then
    "$PY" -m src.qualitative --scores results/per_example_scores.json "${QARGS[@]}" \
        --out results/qualitative.md
else
    echo "SKIPPED: need both LSTM and LLM predictions."
fi

# ---------------------------------------------------------------- 5. tables
step "5/5  Report tables"
"$PY" scripts/collect_results.py

echo
echo "Done. See results/results.md, results/qualitative.md, reports/tables.md"
