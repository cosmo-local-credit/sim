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

FORWARDED_ENV_KEYS=(
  PYTHON_BIN
  CALIBRATION_DIR
  OUTPUT_ROOT
  OUTPUT
  WORKERS
  MONTE_CARLO_WORKERS
  RESUME
  SHARD_DIR
  PARTIAL_AGGREGATE_STRIDE
  ROUTE_SUCCESS_MODE
  ROUTE_SUCCESS_FLOOR
  RUNS
  TICKS
  SEED
  ANALYSIS_STRIDE
  POOL_METRICS_STRIDE
  PROGRESS_STRIDE
  NETWORK_SCALES
  PRINCIPAL_RATIOS
  COUPON_TARGETS
  BOND_FEE_SERVICE_SHARES
  CERTIFICATION_POLICY
  FRONTIER_MODE
  FRONTIER_REFINEMENT_ROUNDS
  BOND_TERM
)
ENV_PREFIX=""
for key in "${FORWARDED_ENV_KEYS[@]}"; do
  if [[ -v "$key" ]]; then
    printf -v quoted_env "%q" "$key=${!key}"
    ENV_PREFIX+=" $quoted_env"
  fi
done

printf -v quoted_pwd "%q" "$PWD"
printf -v quoted_job "%q" "$JOB"
printf -v quoted_log "%q" "$LOG_FILE"
COMMAND="cd $quoted_pwd && env$ENV_PREFIX ./scripts/run_regenbond_remote_batch.sh $quoted_job 2>&1 | tee $quoted_log"
tmux new-session -d -s "$SESSION" "$COMMAND"

echo "Started detached tmux session: $SESSION"
echo "Job: $JOB"
echo "Log: $LOG_FILE"
echo "Workers: ${WORKERS:-${MONTE_CARLO_WORKERS:-auto}}"
echo "Resume: ${RESUME:-1}"
echo "Tail: tail -f $LOG_FILE"
echo "Attach: tmux attach -t $SESSION"
