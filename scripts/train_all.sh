#!/usr/bin/env bash
# Train the main model and all three ablations, sequentially.
#
# Run sequentially on purpose: the configs share one GPU, and running them
# concurrently makes every run slower and the reported per-run training times
# meaningless.
#
# Usage:  bash scripts/train_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

for cfg in base ablation_no_attention ablation_unidirectional ablation_short_context; do
    echo "=============================================================="
    echo "training: configs/${cfg}.yaml    ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "=============================================================="
    "$PY" -m src.train --config "configs/${cfg}.yaml"
done

echo "all training runs complete ($(date '+%Y-%m-%d %H:%M:%S'))"
