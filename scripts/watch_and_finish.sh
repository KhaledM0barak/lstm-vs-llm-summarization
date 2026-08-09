#!/usr/bin/env bash
# Wait for scripts/train_all.sh to finish, then run the rest of the pipeline.
#
# Polls train_all.log for the completion marker rather than matching on process
# names -- this script's own command line would otherwise match a pgrep for
# "train_all.sh" and it would wait on itself forever.
#
# Usage:  nohup bash scripts/watch_and_finish.sh > finish.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/.."
LOG=train_all.log
MARKER="all training runs complete"

echo "waiting for training to finish (polling $LOG) ..."
while true; do
    if [[ -f "$LOG" ]] && grep -aq "$MARKER" "$LOG"; then
        echo "training finished at $(date '+%Y-%m-%d %H:%M:%S')"
        break
    fi
    # Bail out if the training driver died without writing the marker.
    if ! pgrep -f "bash scripts/train_all" > /dev/null 2>&1; then
        if [[ -f "$LOG" ]] && ! grep -aq "$MARKER" "$LOG"; then
            echo "WARNING: training driver is no longer running and the log has no"
            echo "completion marker. Continuing with whatever checkpoints exist."
            break
        fi
    fi
    sleep 60
done

exec bash scripts/finish.sh
