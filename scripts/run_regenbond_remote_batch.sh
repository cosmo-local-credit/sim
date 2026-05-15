#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

JOB="${1:-validation-full}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

CALIBRATION_DIR="${CALIBRATION_DIR:-analysis/sarafu_calibration}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis/monte_carlo}"
mkdir -p "$OUTPUT_ROOT"

echo "[batch] job=$JOB"
echo "[batch] python=$PYTHON_BIN"
echo "[batch] calibration_dir=$CALIBRATION_DIR"
echo "[batch] output_root=$OUTPUT_ROOT"

run_engine_validation() {
  local default_runs="$1"
  local default_ticks="$2"
  local default_seed="$3"
  local default_output="$4"
  local output_dir="${OUTPUT:-$OUTPUT_ROOT/$default_output}"
  echo "[batch] writing engine validation artifacts to $output_dir"
  "${PYTHON_BIN}" scripts/run_regenbond_monte_carlo.py \
    --scenario sarafu_engine_validation \
    --runs "${RUNS:-$default_runs}" \
    --ticks "${TICKS:-$default_ticks}" \
    --seed "${SEED:-$default_seed}" \
    --analysis-stride "${ANALYSIS_STRIDE:-13}" \
    --pool-metrics-stride "${POOL_METRICS_STRIDE:-13}" \
    --progress-stride "${PROGRESS_STRIDE:-13}" \
    --calibration-dir "$CALIBRATION_DIR" \
    --output "$output_dir"
}

run_frontier() {
  local default_runs="$1"
  local default_ticks="$2"
  local default_output="$3"
  local default_scales="$4"
  local default_ratios="$5"
  local default_coupons="$6"
  local default_shares="$7"
  local output_dir="${OUTPUT:-$OUTPUT_ROOT/$default_output}"
  echo "[batch] writing frontier artifacts to $output_dir"
  "${PYTHON_BIN}" scripts/run_regenbond_monte_carlo.py \
    --scenario bond_issuer_frontier \
    --network-scales "${NETWORK_SCALES:-$default_scales}" \
    --principal-ratios "${PRINCIPAL_RATIOS:-$default_ratios}" \
    --coupon-targets "${COUPON_TARGETS:-$default_coupons}" \
    --bond-fee-service-shares "${BOND_FEE_SERVICE_SHARES:-$default_shares}" \
    --certification-policy "${CERTIFICATION_POLICY:-strong_moderate_capped}" \
    --frontier-mode "${FRONTIER_MODE:-adaptive}" \
    --frontier-refinement-rounds "${FRONTIER_REFINEMENT_ROUNDS:-1}" \
    --runs "${RUNS:-$default_runs}" \
    --ticks "${TICKS:-$default_ticks}" \
    --term "${BOND_TERM:-260}" \
    --seed "${SEED:-1}" \
    --analysis-stride "${ANALYSIS_STRIDE:-13}" \
    --pool-metrics-stride "${POOL_METRICS_STRIDE:-13}" \
    --progress-stride "${PROGRESS_STRIDE:-13}" \
    --calibration-dir "$CALIBRATION_DIR" \
    --output "$output_dir"
}

case "$JOB" in
  validation-1mo)
    run_engine_validation 100 4 101 engine_validation_1mo_test
    ;;
  validation-smoke)
    run_engine_validation 5 52 11 engine_validation_smoke
    ;;
  validation-pilot)
    run_engine_validation 20 260 1 engine_validation_20run
    ;;
  validation-full)
    run_engine_validation 100 260 1 engine_validation
    ;;
  frontier-smoke)
    run_frontier 5 52 bond_issuer_frontier_smoke current 0.05,0.10 0,0.06 0.50
    ;;
  frontier-pilot)
    run_frontier 20 260 bond_issuer_frontier_pilot current,connected_2x,connected_5x 0.05,0.10,0.20,0.40,0.80,1.50 0,0.06,0.12 0.25,0.50,0.75
    ;;
  frontier-publication)
    run_frontier 100 260 bond_issuer_frontier current,connected_2x,connected_5x 0.05,0.10,0.20,0.40,0.60,0.80,1.00,1.50 0,0.03,0.06,0.09,0.12 0.25,0.50,0.75
    ;;
  *)
    echo "Unknown job: $JOB" >&2
    echo "Use one of: validation-1mo, validation-smoke, validation-pilot, validation-full, frontier-smoke, frontier-pilot, frontier-publication" >&2
    exit 2
    ;;
esac
