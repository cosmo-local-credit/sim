#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

JOB="${1:-validation-full}"
SESSION="${SESSION:-regenbond}"
LOG_DIR="${LOG_DIR:-analysis/monte_carlo}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$JOB.log}"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required to start detached batch jobs." >&2
  exit 127
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Check it with: tmux attach -t $SESSION" >&2
  echo "Or tail the log with: tail -f $LOG_FILE" >&2
  exit 2
fi

COMMAND="cd '$PWD' && ./scripts/run_regenbond_remote_batch.sh '$JOB' 2>&1 | tee '$LOG_FILE'"
tmux new-session -d -s "$SESSION" "$COMMAND"

echo "Started detached tmux session: $SESSION"
echo "Job: $JOB"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
echo "Attach: tmux attach -t $SESSION"
