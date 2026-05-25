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
WORKERS_VALUE="${WORKERS:-${MONTE_CARLO_WORKERS:-auto}}"
PARTIAL_AGGREGATE_STRIDE_VALUE="${PARTIAL_AGGREGATE_STRIDE:-1}"
MONTE_CARLO_EXTRA_ARGS=(
  --workers "$WORKERS_VALUE"
  --partial-aggregate-stride "$PARTIAL_AGGREGATE_STRIDE_VALUE"
)
if [[ -n "${SHARD_DIR:-}" ]]; then
  MONTE_CARLO_EXTRA_ARGS+=(--shard-dir "$SHARD_DIR")
fi
case "${RESUME:-1}" in
  0|false|FALSE|no|NO)
    MONTE_CARLO_EXTRA_ARGS+=(--no-resume)
    ;;
  *)
    MONTE_CARLO_EXTRA_ARGS+=(--resume)
    ;;
esac

append_optional_arg() {
  local env_name="$1"
  local cli_name="$2"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    MONTE_CARLO_EXTRA_ARGS+=("$cli_name" "$value")
  fi
}

append_optional_arg MAX_HOPS --max-hops
append_optional_arg NOAM_MAX_HOPS --noam-max-hops
append_optional_arg NOAM_OVERLAY_ENABLED --noam-overlay-enabled
append_optional_arg NOAM_CLEARING_ENABLED --noam-clearing-enabled
append_optional_arg NOAM_CLEARING_STRIDE_TICKS --noam-clearing-stride-ticks
append_optional_arg NOAM_CLEARING_MAX_CYCLES --noam-clearing-max-cycles
append_optional_arg NOAM_CLEARING_MAX_HOPS --noam-clearing-max-hops
append_optional_arg NOAM_CLEARING_EDGE_CAP_PER_ASSET --noam-clearing-edge-cap-per-asset
append_optional_arg DECISION_BASED_ACTIVITY_ENABLED --decision-based-activity-enabled
append_optional_arg REPEAT_PARTNER_ROUTE_SHARE --repeat-partner-route-share
append_optional_arg AFFINITY_BUDDY_MIN_COUNT --affinity-buddy-min-count
append_optional_arg SWAP_SUSTAIN_MAX_EXTRA_ATTEMPTS --swap-sustain-max-extra-attempts
append_optional_arg SWAP_SUSTAIN_MAX_ROUNDS --swap-sustain-max-rounds
append_optional_arg SWAP_SUSTAIN_ATTEMPTS_PER_MISSING_SWAP --swap-sustain-attempts-per-missing-swap
append_optional_arg VOUCHER_FEE_CONVERSION_MAX_SWAPS_PER_EPOCH --voucher-fee-conversion-max-swaps-per-epoch
append_optional_arg VOUCHER_FEE_CONVERSION_MAX_USD_PER_EPOCH --voucher-fee-conversion-max-usd-per-epoch

mkdir -p "$OUTPUT_ROOT"

REQUIRED_CALIBRATION_FILES=(
  monte_carlo_calibration_parameters.csv
  repayment_calibration_by_tier_asset.csv
  pool_report_activity.csv
  impact_projection_by_activity.csv
  stable_dependency_anchors.csv
  producer_deposit_calibration.csv
  productive_credit_calibration.csv
  debt_removal_calibration.csv
  fee_conversion_calibration.csv
  quarterly_clearing_calibration.csv
  pool_cohort_exclusion_summary.csv
  stable_actor_demographics.csv
  settlement_reliability_anchors.csv
  route_substitution_diagnostics.csv
  unit_normalization_calibration.csv
)
for required_file in "${REQUIRED_CALIBRATION_FILES[@]}"; do
  if [[ ! -f "$CALIBRATION_DIR/$required_file" ]]; then
    echo "[batch] missing calibration file: $CALIBRATION_DIR/$required_file" >&2
    echo "[batch] rerun/export the public Sarafu calibration bundle before this batch." >&2
    exit 2
  fi
done

KES_PER_USD="$(awk -F, '$1 == "kes_per_usd" { print $2; exit }' "$CALIBRATION_DIR/unit_normalization_calibration.csv")"
VOUCHER_KES_VALUE="$(awk -F, '$1 == "individual_voucher_kes_value_default" { print $2; exit }' "$CALIBRATION_DIR/unit_normalization_calibration.csv")"

echo "[batch] job=$JOB"
echo "[batch] python=$PYTHON_BIN"
echo "[batch] calibration_dir=$CALIBRATION_DIR"
echo "[batch] output_root=$OUTPUT_ROOT"
echo "[batch] workers=$WORKERS_VALUE"
echo "[batch] resume=${RESUME:-1}"
echo "[batch] dry_run=${DRY_RUN:-0}"
echo "[batch] route_success_mode=${ROUTE_SUCCESS_MODE:-diagnostic}"
echo "[batch] routing_profile=max_hops:${MAX_HOPS:-3} noam_max_hops:${NOAM_MAX_HOPS:-3} overlay:${NOAM_OVERLAY_ENABLED:-1} clearing:${NOAM_CLEARING_ENABLED:-1} clearing_stride:${NOAM_CLEARING_STRIDE_TICKS:-13}"
echo "[batch] activity_profile=decision_based:${DECISION_BASED_ACTIVITY_ENABLED:-1} repeat_partner_share:${REPEAT_PARTNER_ROUTE_SHARE:-0.70} buddy_min:${AFFINITY_BUDDY_MIN_COUNT:-1}"
echo "[batch] kes_per_usd=${KES_PER_USD:-missing}"
echo "[batch] voucher_kes_value=${VOUCHER_KES_VALUE:-missing}"

is_dry_run() {
  case "${DRY_RUN:-0}" in
    1|true|TRUE|yes|YES)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_monte_carlo() {
  if is_dry_run; then
    printf '[batch] dry_run command:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

run_engine_validation() {
  local default_runs="$1"
  local default_ticks="$2"
  local default_seed="$3"
  local default_output="$4"
  local output_dir="${OUTPUT:-$OUTPUT_ROOT/$default_output}"
  echo "[batch] writing engine validation artifacts to $output_dir"
  run_monte_carlo "${PYTHON_BIN}" scripts/run_regenbond_monte_carlo.py \
    --scenario sarafu_engine_validation \
    --runs "${RUNS:-$default_runs}" \
    --ticks "${TICKS:-$default_ticks}" \
    --seed "${SEED:-$default_seed}" \
    --analysis-stride "${ANALYSIS_STRIDE:-13}" \
    --pool-metrics-stride "${POOL_METRICS_STRIDE:-13}" \
    --progress-stride "${PROGRESS_STRIDE:-13}" \
    --calibration-dir "$CALIBRATION_DIR" \
    --output "$output_dir" \
    "${MONTE_CARLO_EXTRA_ARGS[@]}"
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
  local frontier_extra_args=()
  if [[ -n "${PRODUCTIVE_CREDIT_VOUCHER_DEPOSIT_SHARE:-}" ]]; then
    frontier_extra_args+=(--productive-credit-voucher-deposit-share "$PRODUCTIVE_CREDIT_VOUCHER_DEPOSIT_SHARE")
  fi
  if [[ -n "${PRODUCTIVE_CREDIT_VOUCHER_DEPOSIT_CAP_RATE_PER_MONTH:-}" ]]; then
    frontier_extra_args+=(
      --productive-credit-voucher-deposit-cap-rate-per-month
      "$PRODUCTIVE_CREDIT_VOUCHER_DEPOSIT_CAP_RATE_PER_MONTH"
    )
  fi
  case "${DISABLE_PRODUCTIVE_CREDIT_VOUCHER_ACTIVITY_BOOST:-0}" in
    1|true|TRUE|yes|YES)
      frontier_extra_args+=(--disable-productive-credit-voucher-activity-boost)
      ;;
  esac
  case "${DISABLE_ORDINARY_STABLE_SPEND_PROTECTION:-0}" in
    1|true|TRUE|yes|YES)
      frontier_extra_args+=(--disable-ordinary-stable-spend-protection)
      ;;
  esac
  case "${ENABLE_ORDINARY_STABLE_SPEND_PROTECTION:-0}" in
    1|true|TRUE|yes|YES)
      frontier_extra_args+=(--enable-ordinary-stable-spend-protection)
      ;;
  esac
  case "${DISABLE_PRODUCER_LOAN_FAILURE_BACKFILL:-0}" in
    1|true|TRUE|yes|YES)
      frontier_extra_args+=(--disable-producer-loan-failure-backfill)
      ;;
  esac
  case "${ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK:-0}" in
    1|true|TRUE|yes|YES)
      frontier_extra_args+=(--enable-producer-voucher-loan-fallback)
      ;;
  esac
  case "${ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST:-0}" in
    1|true|TRUE|yes|YES)
      frontier_extra_args+=(--enable-producer-voucher-loan-activity-boost)
      ;;
  esac
  case "${ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING:-0}" in
    1|true|TRUE|yes|YES)
      frontier_extra_args+=(--enable-producer-primary-voucher-borrowing)
      ;;
  esac
  if [[ -n "${PRODUCER_PRIMARY_VOUCHER_BORROWING_ATTEMPT_SHARE:-}" ]]; then
    frontier_extra_args+=(
      --producer-primary-voucher-borrowing-attempt-share
      "$PRODUCER_PRIMARY_VOUCHER_BORROWING_ATTEMPT_SHARE"
    )
  fi
  if [[ -n "${PRODUCER_CREDIT_REQUEST_BUDGET_SHARE:-}" ]]; then
    frontier_extra_args+=(
      --producer-credit-request-budget-share
      "$PRODUCER_CREDIT_REQUEST_BUDGET_SHARE"
    )
  fi
  if [[ -n "${PRODUCER_VOUCHER_OVERLAP_MODE:-}" ]]; then
    frontier_extra_args+=(--producer-voucher-overlap-mode "$PRODUCER_VOUCHER_OVERLAP_MODE")
  fi
  if [[ -n "${PRODUCER_VOUCHER_LOAN_MAX_TARGET_CANDIDATES:-}" ]]; then
    frontier_extra_args+=(
      --producer-voucher-loan-max-target-candidates
      "$PRODUCER_VOUCHER_LOAN_MAX_TARGET_CANDIDATES"
    )
  fi
  case "${ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND:-0}" in
    1|true|TRUE|yes|YES)
      frontier_extra_args+=(--enable-lender-voucher-purchase-demand)
      ;;
  esac
  if [[ -n "${LENDER_VOUCHER_PURCHASE_ATTEMPTS_PER_TICK:-}" ]]; then
    frontier_extra_args+=(
      --lender-voucher-purchase-attempts-per-tick
      "$LENDER_VOUCHER_PURCHASE_ATTEMPTS_PER_TICK"
    )
  fi
  if [[ -n "${LENDER_VOUCHER_PURCHASE_CONSUMER_SHARE:-}" ]]; then
    frontier_extra_args+=(
      --lender-voucher-purchase-consumer-share
      "$LENDER_VOUCHER_PURCHASE_CONSUMER_SHARE"
    )
  fi
  if [[ -n "${LENDER_VOUCHER_PURCHASE_INVENTORY_SHARE:-}" ]]; then
    frontier_extra_args+=(
      --lender-voucher-purchase-inventory-share
      "$LENDER_VOUCHER_PURCHASE_INVENTORY_SHARE"
    )
  fi
  if [[ -n "${LENDER_VOUCHER_PURCHASE_STABLE_BUDGET_USD_PER_TICK:-}" ]]; then
    frontier_extra_args+=(
      --lender-voucher-purchase-stable-budget-usd-per-tick
      "$LENDER_VOUCHER_PURCHASE_STABLE_BUDGET_USD_PER_TICK"
    )
  fi
  shared_debt_margin="${PRODUCER_DEBT_CONTRACT_SERVICE_MARGIN_RATE:-0.0}"
  stable_debt_margin="${PRODUCER_STABLE_DEBT_CONTRACT_SERVICE_MARGIN_RATE:-$shared_debt_margin}"
  voucher_debt_margin="${PRODUCER_VOUCHER_DEBT_CONTRACT_SERVICE_MARGIN_RATE:-$shared_debt_margin}"
  echo "[batch] writing frontier artifacts to $output_dir"
  echo "[batch] bond service lockbox: mode=${BOND_SERVICE_LOCKBOX_MODE:-remaining_schedule} coverage=${BOND_SERVICE_LOCKBOX_COVERAGE_RATIO:-1.25}"
  echo "[batch] producer shared debt contract service margin: $shared_debt_margin"
  echo "[batch] producer stable debt contract service margin: $stable_debt_margin"
  echo "[batch] producer voucher debt contract service margin: $voucher_debt_margin"
  if [[ -n "${PRODUCER_DEBT_CONTRACT_SERVICE_MARGIN_RATE:-}" ]]; then
    echo "[batch] shared producer debt contract service margin override: $PRODUCER_DEBT_CONTRACT_SERVICE_MARGIN_RATE"
  fi
  echo "[batch] ablation/control flags: voucher_boost_disabled=${DISABLE_PRODUCTIVE_CREDIT_VOUCHER_ACTIVITY_BOOST:-0} stable_spend_protection_enabled=${ENABLE_ORDINARY_STABLE_SPEND_PROTECTION:-0} loan_backfill_disabled=${DISABLE_PRODUCER_LOAN_FAILURE_BACKFILL:-0}"
  echo "[batch] voucher-loan fallback: enabled=${ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK:-0} activity_boost=${ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST:-0} max_targets=${PRODUCER_VOUCHER_LOAN_MAX_TARGET_CANDIDATES:-3}"
  echo "[batch] primary voucher borrowing: enabled=${ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING:-0} attempt_share=${PRODUCER_PRIMARY_VOUCHER_BORROWING_ATTEMPT_SHARE:-calibrated} producer_credit_budget_share=${PRODUCER_CREDIT_REQUEST_BUDGET_SHARE:-calibrated}"
  echo "[batch] producer voucher overlap mode: ${PRODUCER_VOUCHER_OVERLAP_MODE:-single_lender}"
  echo "[batch] lender voucher purchase demand: enabled=${ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND:-0} attempts_per_tick=${LENDER_VOUCHER_PURCHASE_ATTEMPTS_PER_TICK:-calibrated} consumer_share=${LENDER_VOUCHER_PURCHASE_CONSUMER_SHARE:-calibrated} inventory_share=${LENDER_VOUCHER_PURCHASE_INVENTORY_SHARE:-calibrated} stable_budget_usd_per_tick=${LENDER_VOUCHER_PURCHASE_STABLE_BUDGET_USD_PER_TICK:-calibrated}"
  run_monte_carlo "${PYTHON_BIN}" scripts/run_regenbond_monte_carlo.py \
    --scenario bond_issuer_frontier \
    --network-scales "${NETWORK_SCALES:-$default_scales}" \
    --principal-ratios "${PRINCIPAL_RATIOS:-$default_ratios}" \
    --coupon-targets "${COUPON_TARGETS:-$default_coupons}" \
    --bond-fee-service-shares "${BOND_FEE_SERVICE_SHARES:-$default_shares}" \
    --certification-policy "${CERTIFICATION_POLICY:-strong_moderate_capped}" \
    --frontier-mode "${FRONTIER_MODE:-grid}" \
    --frontier-refinement-rounds "${FRONTIER_REFINEMENT_ROUNDS:-0}" \
    --route-success-mode "${ROUTE_SUCCESS_MODE:-diagnostic}" \
    --route-success-floor "${ROUTE_SUCCESS_FLOOR:-0.85}" \
    --bond-service-lockbox-mode "${BOND_SERVICE_LOCKBOX_MODE:-remaining_schedule}" \
    --bond-service-lockbox-coverage-ratio "${BOND_SERVICE_LOCKBOX_COVERAGE_RATIO:-1.25}" \
    --producer-debt-contract-service-margin-rate "$shared_debt_margin" \
    --producer-stable-debt-contract-service-margin-rate "$stable_debt_margin" \
    --producer-voucher-debt-contract-service-margin-rate "$voucher_debt_margin" \
    "${frontier_extra_args[@]}" \
    --runs "${RUNS:-$default_runs}" \
    --ticks "${TICKS:-$default_ticks}" \
    --term "${BOND_TERM:-260}" \
    --seed "${SEED:-1}" \
    --analysis-stride "${ANALYSIS_STRIDE:-13}" \
    --pool-metrics-stride "${POOL_METRICS_STRIDE:-13}" \
    --progress-stride "${PROGRESS_STRIDE:-13}" \
    --calibration-dir "$CALIBRATION_DIR" \
    --output "$output_dir" \
    "${MONTE_CARLO_EXTRA_ARGS[@]}"
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
    ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK="${ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK:-1}" \
      ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST="${ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST:-1}" \
      ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING="${ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING:-1}" \
      ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND="${ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND:-1}" \
      LENDER_VOUCHER_PURCHASE_ATTEMPTS_PER_TICK="${LENDER_VOUCHER_PURCHASE_ATTEMPTS_PER_TICK:-5}" \
      LENDER_VOUCHER_PURCHASE_CONSUMER_SHARE="${LENDER_VOUCHER_PURCHASE_CONSUMER_SHARE:-0.75}" \
      LENDER_VOUCHER_PURCHASE_INVENTORY_SHARE="${LENDER_VOUCHER_PURCHASE_INVENTORY_SHARE:-0.05}" \
      LENDER_VOUCHER_PURCHASE_STABLE_BUDGET_USD_PER_TICK="${LENDER_VOUCHER_PURCHASE_STABLE_BUDGET_USD_PER_TICK:-184.061305}" \
      PRODUCER_VOUCHER_OVERLAP_MODE="${PRODUCER_VOUCHER_OVERLAP_MODE:-empirical_overlap}" \
      run_frontier 5 52 bond_issuer_frontier_smoke current 0.05,0.10 0,0.06 1.0
    ;;
  frontier-maturity-smoke)
    ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK="${ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK:-1}" \
      ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST="${ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST:-1}" \
      ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING="${ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING:-1}" \
      ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND="${ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND:-1}" \
      LENDER_VOUCHER_PURCHASE_ATTEMPTS_PER_TICK="${LENDER_VOUCHER_PURCHASE_ATTEMPTS_PER_TICK:-5}" \
      LENDER_VOUCHER_PURCHASE_CONSUMER_SHARE="${LENDER_VOUCHER_PURCHASE_CONSUMER_SHARE:-0.75}" \
      LENDER_VOUCHER_PURCHASE_INVENTORY_SHARE="${LENDER_VOUCHER_PURCHASE_INVENTORY_SHARE:-0.05}" \
      LENDER_VOUCHER_PURCHASE_STABLE_BUDGET_USD_PER_TICK="${LENDER_VOUCHER_PURCHASE_STABLE_BUDGET_USD_PER_TICK:-184.061305}" \
      PRODUCER_VOUCHER_OVERLAP_MODE="${PRODUCER_VOUCHER_OVERLAP_MODE:-empirical_overlap}" \
      run_frontier 2 260 bond_issuer_frontier_maturity_smoke current 0.05 0 1.0
    ;;
  frontier-feedback-probe)
    FRONTIER_REFINEMENT_ROUNDS="${FRONTIER_REFINEMENT_ROUNDS:-0}" \
      run_frontier 2 260 bond_issuer_frontier_feedback_probe current 0,0.01,0.02,0.05,0.10,0.25,0.50,1.00,1.50 0 0.50
    ;;
  frontier-low-principal-probe)
    FRONTIER_REFINEMENT_ROUNDS="${FRONTIER_REFINEMENT_ROUNDS:-0}" \
      run_frontier 5 260 bond_issuer_frontier_low_principal_probe current 0,0.005,0.01,0.02,0.03,0.04,0.05 0 0.25,0.50,0.75
    ;;
  frontier-activity-ablation-probe)
    FRONTIER_REFINEMENT_ROUNDS="${FRONTIER_REFINEMENT_ROUNDS:-0}" \
      DISABLE_PRODUCTIVE_CREDIT_VOUCHER_ACTIVITY_BOOST="${DISABLE_PRODUCTIVE_CREDIT_VOUCHER_ACTIVITY_BOOST:-1}" \
      DISABLE_ORDINARY_STABLE_SPEND_PROTECTION="${DISABLE_ORDINARY_STABLE_SPEND_PROTECTION:-1}" \
      DISABLE_PRODUCER_LOAN_FAILURE_BACKFILL="${DISABLE_PRODUCER_LOAN_FAILURE_BACKFILL:-1}" \
      OUTPUT="${OUTPUT:-$OUTPUT_ROOT/bond_issuer_frontier_activity_ablation_probe}" \
      run_frontier 5 260 bond_issuer_frontier_activity_ablation_probe current 0,0.005,0.01,0.02,0.03,0.04,0.05 0 0.50
    ;;
  frontier-rola-regeneration-probe)
    FRONTIER_REFINEMENT_ROUNDS="${FRONTIER_REFINEMENT_ROUNDS:-0}" \
      ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK="${ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK:-1}" \
      ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST="${ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST:-1}" \
      ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING="${ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING:-1}" \
      ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND="${ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND:-1}" \
      PRODUCER_VOUCHER_OVERLAP_MODE="${PRODUCER_VOUCHER_OVERLAP_MODE:-empirical_overlap}" \
      OUTPUT="${OUTPUT:-$OUTPUT_ROOT/bond_issuer_frontier_rola_regeneration_probe}" \
      run_frontier 5 260 bond_issuer_frontier_rola_regeneration_probe current 0,0.005,0.01,0.02,0.03,0.04,0.05 0 0.50
    ;;
  frontier-pilot)
    ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK="${ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK:-1}" \
      ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST="${ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST:-1}" \
      ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING="${ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING:-1}" \
      ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND="${ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND:-1}" \
      PRODUCER_VOUCHER_OVERLAP_MODE="${PRODUCER_VOUCHER_OVERLAP_MODE:-empirical_overlap}" \
      run_frontier 20 260 bond_issuer_frontier_pilot current 0.05,0.10,0.15,0.20,0.25 0,0.02,0.04,0.06,0.08,0.10 1.0
    ;;
  frontier-publication)
    FRONTIER_MODE="${FRONTIER_MODE:-grid}" \
      FRONTIER_REFINEMENT_ROUNDS="${FRONTIER_REFINEMENT_ROUNDS:-0}" \
      ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK="${ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK:-1}" \
      ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST="${ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST:-1}" \
      ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING="${ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING:-1}" \
      ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND="${ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND:-1}" \
      PRODUCER_VOUCHER_OVERLAP_MODE="${PRODUCER_VOUCHER_OVERLAP_MODE:-empirical_overlap}" \
      run_frontier 100 260 bond_issuer_frontier current 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12 1.0
    ;;
  *)
    echo "Unknown job: $JOB" >&2
    echo "Use one of: validation-1mo, validation-smoke, validation-pilot, validation-full, frontier-smoke, frontier-maturity-smoke, frontier-feedback-probe, frontier-low-principal-probe, frontier-activity-ablation-probe, frontier-rola-regeneration-probe, frontier-pilot, frontier-publication" >&2
    exit 2
    ;;
esac
