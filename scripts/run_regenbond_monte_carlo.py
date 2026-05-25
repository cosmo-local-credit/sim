#!/usr/bin/env python3
"""Run Sarafu-calibrated Monte Carlo scenarios for the RegenBonds paper.

The runner is paper-first: it writes deterministic CSV and LaTeX artifacts that
can be included by the manuscript. It does not read private Sarafu raw data; it
only consumes aggregate calibration files from the public calibration bundle or
from a local RegenBonds analysis checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SIM_ROOT.parent
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from sim.config import ScenarioConfig
from sim.engine import SimulationEngine


DEFAULT_PUBLIC_CALIBRATION_DIR = SIM_ROOT / "analysis" / "sarafu_calibration"
DEFAULT_OUTPUT_DIR = SIM_ROOT / "analysis" / "monte_carlo"
MONTH_TICKS = 4
YEAR_TICKS = 52
DEFAULT_COUPONS = (0.0, 0.03, 0.06, 0.09, 0.12)
DEFAULT_TERMS = (52, 156, 260)
TIER_ORDER = ("strong", "moderate", "weak")
ASSET_ORDER = ("cash", "redeemable_voucher", "internal_voucher")
RECENT_WINDOW_TICKS = 13
LEGACY_SCENARIOS = (
    "mutual_aid_only",
    "sarafu_like_pools",
    "regenbond_lp_injection",
    "stress_weak_pool_repayment",
    "stress_cash_stable_leakage",
    "stress_high_coupon",
    "stress_low_fee_conversion",
    "stress_governance_diversion",
    "stress_liquidity_concentration",
)
SCENARIOS = (
    *LEGACY_SCENARIOS,
    "sarafu_engine_validation",
    "bond_issuer_frontier",
)
NETWORK_SCALE_FACTORS = {
    "current": 1.0,
    "connected_2x": 2.0,
    "connected_5x": 5.0,
}


def default_calibration_dir() -> Path:
    if DEFAULT_PUBLIC_CALIBRATION_DIR.exists():
        return DEFAULT_PUBLIC_CALIBRATION_DIR
    sibling = WORKSPACE_ROOT / "RegenBonds" / "analysis"
    if sibling.exists():
        return sibling
    return DEFAULT_PUBLIC_CALIBRATION_DIR


CORE_QUANTILE_METRICS = (
    "bond_annualized_fee_yield",
    "bond_coupon_coverage_ratio",
    "bond_coupon_shortfall_usd",
    "bond_cumulative_fee_return_usd",
    "issuer_service_coverage_ratio",
    "issuer_paid_coverage_ratio",
    "issuer_service_cash_headroom_ratio",
    "issuer_available_service_cash_headroom_ratio",
    "issuer_reserve_balance_usd",
    "issuer_unpaid_scheduled_claim_usd",
    "fee_pool_cumulative_usd",
    "swap_volume_usd_tick",
    "transactions_per_tick",
    "route_success_rate_tick",
    "repayment_volume_usd",
    "loan_issuance_volume_usd",
    "debt_outstanding_usd",
    "household_cash_stress_ratio",
    "consumer_stable_reserve_stress_ratio",
    "community_stable_reserve_stress_ratio",
    "expected_verified_report_exposure",
    "expected_cash_return_coverage",
    "expected_voucher_return_coverage",
    "potential_largest_component_share",
    "realized_largest_component_share",
    "potential_avg_degree",
    "realized_avg_degree",
)


@dataclass(frozen=True)
class ImpactProjection:
    activity: str
    intercept: float
    slope: float
    share: float


@dataclass(frozen=True)
class PoolCalibration:
    template_id: str
    tier: str
    score: float
    swap_events: float
    recent_swap_weeks_90d: float
    active_weeks: float
    swaps_per_active_week: float
    total_users: float
    accepted_voucher_members: float
    backing_inflow: float
    backing_cash_inflow: float
    backing_voucher_inflow: float
    tagged_voucher_tokens: float
    verified_report_exposure: float
    same_token_return_rate: float
    same_token_out_value: float
    same_token_matched_later_in_value: float
    borrow_proxy_matured_events: float
    borrow_proxy_matured_return_rate: float
    rosca_proxy_value_return_rate: float

    @property
    def weekly_swap_rate(self) -> float:
        if self.swaps_per_active_week > 0.0:
            return self.swaps_per_active_week * min(1.0, self.recent_swap_weeks_90d / RECENT_WINDOW_TICKS)
        if self.active_weeks > 0.0:
            return self.swap_events / self.active_weeks
        return self.swap_events / RECENT_WINDOW_TICKS

    @property
    def empirical_period_swaps_90d(self) -> float:
        return self.weekly_swap_rate * RECENT_WINDOW_TICKS


@dataclass
class Calibration:
    calibration_hash: str
    params: dict[str, float]
    repayment_by_tier_asset: dict[tuple[str, str], float]
    repayment_out_value_by_tier_asset: dict[tuple[str, str], float]
    tier_probs: dict[str, float]
    voucher_coverage_by_tier: dict[str, float]
    pool_rows: list[PoolCalibration]
    borrow_return_by_tier: dict[str, float]
    impact_rows: list[ImpactProjection]
    voucher_circulation_baselines: dict[str, dict[str, float]]
    stable_dependency_anchors: dict[str, dict[str, float]]
    producer_deposit_by_tier: dict[str, dict[str, float]]
    productive_credit_by_tier: dict[str, dict[str, float]]
    productive_credit_activity_boost_calibration: dict[str, float]
    debt_removal_calibration: dict[str, float]
    fee_conversion_calibration: dict[str, float]
    quarterly_clearing_calibration: dict[str, float]
    settlement_reliability_anchors: dict[str, float]
    route_substitution_diagnostics: dict[str, float]
    voucher_pool_overlap_calibration: dict[str, float]
    voucher_pool_overlap_distribution: dict[str, float]
    stable_actor_demographics: dict[str, float]
    unit_normalization: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regenerative-bond Monte Carlo scenarios and write paper artifacts."
    )
    parser.add_argument(
        "--scenario",
        default="regenbond_lp_injection",
        choices=(*SCENARIOS, "all"),
        help="Scenario preset to run. Use 'all' for the legacy paper scenarios.",
    )
    parser.add_argument("--runs", type=int, default=100, help="Monte Carlo runs per scenario/coupon/term.")
    parser.add_argument("--ticks", type=int, default=260, help="Ticks per run; 1 tick = 1 week.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument(
        "--coupon-targets",
        default=",".join(str(v) for v in DEFAULT_COUPONS),
        help="Comma-separated annual coupon/return targets, e.g. 0,0.03,0.06,0.09,0.12.",
    )
    parser.add_argument(
        "--terms",
        default=",".join(str(v) for v in DEFAULT_TERMS),
        help="Comma-separated bond terms in ticks, e.g. 52,156,260.",
    )
    parser.add_argument(
        "--term",
        type=int,
        default=None,
        help="Single bond term in ticks. Overrides --terms when supplied.",
    )
    parser.add_argument(
        "--network-scales",
        default="current",
        help="Comma-separated network scales for bond_issuer_frontier: current,connected_2x,connected_5x.",
    )
    parser.add_argument(
        "--principal-ratios",
        default="1.0",
        help="Comma-separated principal/certified-capacity ratios for bond_issuer_frontier.",
    )
    parser.add_argument(
        "--bond-fee-service-shares",
        default="1.0",
        help="Comma-separated shares of eligible fee/service flows available for bond service.",
    )
    parser.add_argument(
        "--certification-policy",
        default="strong_moderate_capped",
        choices=("strong_moderate_capped", "strong_only", "all_tiers_weighted", "weak_inclusion_stress"),
        help="Pool certification policy for bond_issuer_frontier.",
    )
    parser.add_argument(
        "--issuer-reserve-share",
        type=float,
        default=0.10,
        help=(
            "Legacy reserve-share parameter. The bond_issuer_frontier setup now "
            "deploys 100% of gross principal directly to lender pools."
        ),
    )
    parser.add_argument(
        "--issuer-payment-stride",
        type=int,
        default=13,
        help="Issuer scheduled payment interval in weekly ticks.",
    )
    parser.add_argument(
        "--bond-service-lockbox-mode",
        default="remaining_schedule",
        choices=("next_due", "remaining_schedule"),
        help=(
            "Bond-frontier recovered-stable reservation mode. The default reserves "
            "against the remaining scheduled service; next_due is available for controls."
        ),
    )
    parser.add_argument(
        "--bond-service-lockbox-coverage-ratio",
        type=float,
        default=1.25,
        help=(
            "Bond-frontier lockbox coverage target. A value of 1.25 reserves a "
            "25% headroom buffer above scheduled bondholder service."
        ),
    )
    parser.add_argument(
        "--producer-debt-contract-service-margin-rate",
        type=float,
        default=None,
        help=(
            "Shared bond-frontier borrower cash-service margin above principal. "
            "Defaults to 0.0 until calibrated; sweep with values such as 0, "
            "0.02, 0.05, 0.10, and 0.15."
        ),
    )
    parser.add_argument(
        "--producer-stable-debt-contract-service-margin-rate",
        type=float,
        default=None,
        help=(
            "Optional sensitivity override for stable-loan debt. Defaults to "
            "the shared producer debt contract service margin."
        ),
    )
    parser.add_argument(
        "--producer-voucher-debt-contract-service-margin-rate",
        type=float,
        default=None,
        help=(
            "Optional sensitivity override for voucher-to-voucher debt. Defaults "
            "to the shared producer debt contract service margin."
        ),
    )
    parser.add_argument(
        "--productive-credit-voucher-deposit-share",
        type=float,
        default=None,
        help=(
            "Optional override for the frontier share of productive-credit inflow "
            "issued as new producer voucher deposits. Defaults to calibration."
        ),
    )
    parser.add_argument(
        "--productive-credit-voucher-deposit-cap-rate-per-month",
        type=float,
        default=None,
        help=(
            "Optional override for the monthly cap on loan-induced voucher deposits. "
            "Defaults to calibration."
        ),
    )
    parser.add_argument(
        "--disable-productive-credit-voucher-activity-boost",
        action="store_true",
        help="Frontier ablation: keep voucher deposits but disable post-inflow voucher activity boost.",
    )
    parser.add_argument(
        "--disable-ordinary-stable-spend-protection",
        action="store_true",
        help="Legacy no-op unless stable-spend protection is explicitly enabled.",
    )
    parser.add_argument(
        "--enable-ordinary-stable-spend-protection",
        action="store_true",
        help=(
            "Frontier control: preserve producer/consumer stable reserves and "
            "a voucher-value buffer for ordinary stable-source spending."
        ),
    )
    parser.add_argument(
        "--disable-producer-loan-failure-backfill",
        action="store_true",
        help="Frontier ablation: do not backfill failed loan route attempts with ordinary community swaps.",
    )
    parser.add_argument(
        "--enable-producer-voucher-loan-fallback",
        action="store_true",
        help=(
            "Experimental frontier probe: when stable loan capacity is unavailable, "
            "try voucher-to-voucher producer borrowing instead."
        ),
    )
    parser.add_argument(
        "--enable-producer-voucher-loan-activity-boost",
        action="store_true",
        help="Experimental frontier probe: boost ordinary source selection for vouchers used in voucher loans.",
    )
    parser.add_argument(
        "--enable-producer-primary-voucher-borrowing",
        action="store_true",
        help=(
            "Experimental frontier probe: allow producer voucher-to-voucher borrowing "
            "as a primary credit channel alongside stable borrowing."
        ),
    )
    parser.add_argument(
        "--producer-primary-voucher-borrowing-attempt-share",
        type=float,
        default=None,
        help=(
            "Share of eligible producer credit attempts that try primary voucher borrowing "
            "first. Defaults to the calibrated voucher-to-voucher share among recent "
            "voucher-source borrowing motifs."
        ),
    )
    parser.add_argument(
        "--producer-voucher-loan-max-target-candidates",
        type=int,
        default=3,
        help="Maximum voucher target assets to try for experimental producer voucher-loan fallback.",
    )
    parser.add_argument(
        "--producer-voucher-overlap-mode",
        default="single_lender",
        choices=("single_lender", "empirical_overlap"),
        help=(
            "Producer-voucher lender listing topology. Frontier probes can use "
            "empirical_overlap to sample lender acceptor counts from Sarafu "
            "voucher-pool overlap calibration."
        ),
    )
    parser.add_argument(
        "--enable-lender-voucher-purchase-demand",
        action="store_true",
        help=(
            "Experimental frontier probe: let lender-held producer voucher inventory "
            "trigger calibrated consumer/third-party stable purchases."
        ),
    )
    parser.add_argument(
        "--lender-voucher-purchase-attempts-per-tick",
        type=int,
        default=5,
        help="Experimental stable-to-voucher purchase attempts per tick when purchase demand is enabled.",
    )
    parser.add_argument(
        "--lender-voucher-purchase-consumer-share",
        type=float,
        default=0.75,
        help="Share of purchase-demand attempts assigned to consumer stable buyers.",
    )
    parser.add_argument(
        "--lender-voucher-purchase-inventory-share",
        type=float,
        default=0.05,
        help="Maximum share of a target lender-held voucher inventory considered in one purchase attempt.",
    )
    parser.add_argument(
        "--lender-voucher-purchase-stable-budget-usd-per-tick",
        type=float,
        default=0.0,
        help=(
            "Experimental calibrated stable purchase budget injected per tick for "
            "consumer/third-party purchases of lender-held producer vouchers."
        ),
    )
    parser.add_argument(
        "--frontier-mode",
        default="grid",
        choices=("adaptive", "grid"),
        help="Run the bond_issuer_frontier as the supplied grid or add midpoint refinement cells.",
    )
    parser.add_argument(
        "--frontier-refinement-rounds",
        type=int,
        default=0,
        help="Number of adaptive midpoint refinement rounds for bond_issuer_frontier.",
    )
    parser.add_argument(
        "--route-success-floor",
        type=float,
        default=0.85,
        help=(
            "Bond-frontier p05 route-success safety floor. This is a model "
            "settlement-reliability sensitivity parameter, not a directly "
            "observed Sarafu failed-route calibration moment."
        ),
    )
    parser.add_argument(
        "--route-success-mode",
        default="diagnostic",
        choices=("diagnostic", "relative", "absolute"),
        help=(
            "How route success participates in frontier safety: diagnostic records "
            "it without binding, relative binds on degradation versus matched "
            "no-bond baseline, absolute applies --route-success-floor."
        ),
    )
    parser.add_argument(
        "--calibration-dir",
        default=str(default_calibration_dir()),
        help=(
            "Directory containing Sarafu-derived aggregate calibration CSVs. "
            "Default uses analysis/sarafu_calibration in a standalone public clone."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for manuscript CSV and LaTeX artifacts.",
    )
    parser.add_argument(
        "--max-active-pools-per-tick",
        type=int,
        default=None,
        help="Optional performance cap passed to ScenarioConfig.",
    )
    parser.add_argument(
        "--pool-metrics-stride",
        type=int,
        default=1,
        help="Pool metric stride. Keep 1 for pool-tier and stress diagnostics.",
    )
    parser.add_argument(
        "--analysis-stride",
        type=int,
        default=1,
        help="Record Monte Carlo paper diagnostics every N ticks while still simulating every tick.",
    )
    parser.add_argument(
        "--progress-stride",
        type=int,
        default=0,
        help="Print run progress every N ticks. Use 0 to disable progress logging.",
    )
    parser.add_argument(
        "--unit-diagnostics",
        dest="unit_diagnostics",
        action="store_true",
        default=True,
        help="Print first-run unit/value diagnostics for each job.",
    )
    parser.add_argument(
        "--no-unit-diagnostics",
        dest="unit_diagnostics",
        action="store_false",
        help="Suppress first-run unit/value diagnostics.",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Parallel worker count. Use 'auto' for min(cpu_count - 1, 8), or an integer.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Reuse completed matching shards from prior runs.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing shards and recompute all jobs.",
    )
    parser.add_argument(
        "--shard-dir",
        default=None,
        help="Directory for resumable shard files. Defaults to <output>/_shards.",
    )
    parser.add_argument(
        "--partial-aggregate-stride",
        type=int,
        default=1,
        help="Write partial aggregate CSVs after every N completed jobs.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG figure generation.",
    )
    parser.add_argument(
        "--plot-scenario",
        default="regenbond_lp_injection",
        help="Scenario used for headline PNG figures when present in the output.",
    )
    parser.add_argument(
        "--plot-coupon",
        type=float,
        default=0.06,
        help="Annual coupon target used for headline PNG figures when present in the output.",
    )
    parser.add_argument(
        "--plot-term",
        type=int,
        default=260,
        help="Bond term in ticks used for headline PNG figures when present in the output.",
    )
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(float(item)))
    return values


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def selected_terms(args: argparse.Namespace) -> list[int]:
    if args.term is not None:
        return [int(args.term)]
    return parse_int_list(args.terms)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SHARD_CONFIG_KEYS = (
    "scenario",
    "runs",
    "ticks",
    "seed",
    "coupon_targets",
    "terms",
    "term",
    "network_scales",
    "principal_ratios",
    "bond_fee_service_shares",
    "certification_policy",
    "issuer_reserve_share",
    "issuer_payment_stride",
    "bond_service_lockbox_mode",
    "bond_service_lockbox_coverage_ratio",
    "producer_debt_contract_service_margin_rate",
    "producer_stable_debt_contract_service_margin_rate",
    "producer_voucher_debt_contract_service_margin_rate",
    "productive_credit_voucher_deposit_share",
    "productive_credit_voucher_deposit_cap_rate_per_month",
    "disable_productive_credit_voucher_activity_boost",
    "disable_ordinary_stable_spend_protection",
    "enable_ordinary_stable_spend_protection",
    "disable_producer_loan_failure_backfill",
    "enable_producer_voucher_loan_fallback",
    "enable_producer_voucher_loan_activity_boost",
    "enable_producer_primary_voucher_borrowing",
    "producer_primary_voucher_borrowing_attempt_share",
    "producer_voucher_loan_max_target_candidates",
    "producer_voucher_overlap_mode",
    "enable_lender_voucher_purchase_demand",
    "lender_voucher_purchase_attempts_per_tick",
    "lender_voucher_purchase_consumer_share",
    "lender_voucher_purchase_inventory_share",
    "lender_voucher_purchase_stable_budget_usd_per_tick",
    "frontier_mode",
    "frontier_refinement_rounds",
    "route_success_floor",
    "route_success_mode",
    "calibration_dir",
    "calibration_hash",
    "max_active_pools_per_tick",
    "pool_metrics_stride",
    "analysis_stride",
    "progress_stride",
)


def args_payload(args: argparse.Namespace) -> dict[str, object]:
    return {key: getattr(args, key) for key in SHARD_CONFIG_KEYS if hasattr(args, key)}


def namespace_from_payload(payload: dict[str, object]) -> argparse.Namespace:
    return argparse.Namespace(**payload)


def resolve_workers(value: object) -> int:
    text = str(value or "auto").strip().lower()
    if text == "auto":
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count - 1, 8))
    try:
        return max(1, int(text))
    except ValueError as exc:
        raise ValueError(f"Invalid --workers value: {value!r}") from exc


def root_shard_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.shard_dir:
        return Path(args.shard_dir).resolve()
    return output_dir / "_shards"


def shard_job_hash(args: argparse.Namespace, job: dict[str, object]) -> str:
    return canonical_hash({"args": args_payload(args), "job": job})


def shard_job_dir(shard_root: Path, kind: str, job_id: str) -> Path:
    return shard_root / kind / job_id


def completed_manifest(path: Path, config_hash: str) -> bool:
    manifest = read_json(path)
    return manifest.get("status") == "completed" and manifest.get("config_hash") == config_hash


def write_shard_manifest(
    shard_dir: Path,
    *,
    job: dict[str, object],
    config_hash: str,
    status: str,
    files: dict[str, int] | None = None,
    error: str = "",
    started_at: float | None = None,
) -> None:
    ended_at = time.time()
    payload: dict[str, object] = {
        "status": status,
        "config_hash": config_hash,
        "job": job,
        "files": files or {},
        "started_at": started_at or ended_at,
        "ended_at": ended_at,
        "elapsed_seconds": ended_at - (started_at or ended_at),
    }
    if error:
        payload["error"] = error
    write_json(shard_dir / "manifest.json", payload)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    write_csv(path, list(rows[0].keys()) if rows else [], rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def calibration_bundle_hash(calibration_dir: Path) -> str:
    payload: dict[str, str] = {}
    for path in sorted(calibration_dir.glob("*.csv")):
        payload[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    readme = calibration_dir / "README.md"
    if readme.exists():
        payload[readme.name] = hashlib.sha256(readme.read_bytes()).hexdigest()
    return canonical_hash(payload)


def sorted_run_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("scenario", "")),
            str(row.get("network_scale", "")),
            safe_float(row.get("principal_ratio")),
            safe_float(row.get("coupon_target_annual")),
            safe_float(row.get("bond_fee_service_share")),
            safe_float(row.get("bond_term_ticks")),
            safe_float(row.get("run")),
            safe_float(row.get("tick")),
            safe_float(row.get("seed")),
        ),
    )


def sorted_frontier_safety_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("network_scale", "")),
            safe_float(row.get("coupon_target_annual")),
            safe_float(row.get("bond_fee_service_share")),
            safe_float(row.get("principal_ratio")),
        ),
    )


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def shard_result_from_files(
    *,
    kind: str,
    job: dict[str, object],
    shard_root: Path,
    config_hash: str,
    csv_names: tuple[str, ...],
) -> dict[str, object] | None:
    job_dir = shard_job_dir(shard_root, kind, str(job["job_id"]))
    if not completed_manifest(job_dir / "manifest.json", config_hash):
        return None
    result: dict[str, object] = {"status": "completed", "job": job, "shard_dir": str(job_dir)}
    for name in csv_names:
        result[name] = read_rows(job_dir / f"{name}.csv")
    return result


def failed_result(job: dict[str, object], error: str) -> dict[str, object]:
    return {"status": "failed", "job": job, "error": error}


def run_sharded_jobs(
    *,
    label: str,
    kind: str,
    jobs: list[dict[str, object]],
    args: argparse.Namespace,
    calibration: Calibration,
    output_dir: Path,
    worker: Callable[[dict[str, object], dict[str, object], Calibration, str, bool], dict[str, object]],
    load_completed: Callable[[dict[str, object], Path, str], dict[str, object] | None],
    on_progress: Callable[[list[dict[str, object]]], None] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shard_root = root_shard_dir(args, output_dir)
    args_data = args_payload(args)
    worker_count = resolve_workers(args.workers)
    stride = max(1, int(getattr(args, "partial_aggregate_stride", 1) or 1))
    completed: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []

    for job in jobs:
        config_hash = shard_job_hash(args, job)
        job["config_hash"] = config_hash
        if args.resume:
            result = load_completed(job, shard_root, config_hash)
            if result is not None:
                completed.append(result)
                continue
        pending.append(job)

    print(
        f"[parallel] {label}: workers={worker_count} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    if completed and on_progress is not None:
        on_progress(completed)

    def handle_result(result: dict[str, object]) -> None:
        if result.get("status") == "completed":
            completed.append(result)
        else:
            failed.append(result)
        done = len(completed) + len(failed)
        if on_progress is not None and done % stride == 0:
            on_progress(completed)
        status = result.get("status")
        job_id = result.get("job", {}).get("job_id", "?") if isinstance(result.get("job"), dict) else "?"
        print(f"[parallel] {label}: {status} {job_id} ({done}/{len(jobs)})", flush=True)

    if worker_count == 1:
        for job in pending:
            try:
                handle_result(worker(job, args_data, calibration, str(shard_root), bool(args.resume)))
            except BaseException:
                error = traceback.format_exc()
                write_shard_manifest(
                    shard_job_dir(shard_root, kind, str(job["job_id"])),
                    job=job,
                    config_hash=str(job["config_hash"]),
                    status="failed",
                    error=error,
                )
                handle_result(failed_result(job, error))
    elif pending:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(worker, job, args_data, calibration, str(shard_root), bool(args.resume)): job
                for job in pending
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    handle_result(future.result())
                except BaseException:
                    error = traceback.format_exc()
                    write_shard_manifest(
                        shard_job_dir(shard_root, kind, str(job["job_id"])),
                        job=job,
                        config_hash=str(job["config_hash"]),
                        status="failed",
                        error=error,
                    )
                    handle_result(failed_result(job, error))

    if on_progress is not None:
        on_progress(completed)
    return completed, failed


def load_calibration(calibration_dir: Path) -> Calibration:
    bundle_hash = calibration_bundle_hash(calibration_dir)
    params = {}
    for row in read_csv(calibration_dir / "monte_carlo_calibration_parameters.csv"):
        params[row["parameter"]] = safe_float(row["value"])

    repayment_by_tier_asset: dict[tuple[str, str], float] = {}
    tier_asset_out_values: dict[tuple[str, str], float] = {}
    for row in read_csv(calibration_dir / "repayment_calibration_by_tier_asset.csv"):
        key = (row["tier"], row["asset_class"])
        repayment_by_tier_asset[key] = safe_float(row["same_token_return_coverage"])
        tier_asset_out_values[key] = safe_float(row["same_token_out_value"])

    pool_rows_raw = read_csv(calibration_dir / "pool_report_activity.csv")
    pool_rows = []
    for idx, row in enumerate(pool_rows_raw, start=1):
        pool_rows.append(
            PoolCalibration(
                template_id=f"pool_template_{idx:03d}",
                tier=str(row.get("tier", "moderate")).strip().lower(),
                score=safe_float(row.get("score")),
                swap_events=safe_float(row.get("swap_events")),
                recent_swap_weeks_90d=safe_float(row.get("recent_swap_weeks_90d")),
                active_weeks=safe_float(row.get("active_weeks")),
                swaps_per_active_week=safe_float(row.get("swaps_per_active_week")),
                total_users=safe_float(row.get("total_users")),
                accepted_voucher_members=safe_float(
                    row.get("accepted_voucher_members", row.get("tagged_voucher_tokens", 0.0))
                ),
                backing_inflow=safe_float(row.get("backing_inflow")),
                backing_cash_inflow=safe_float(
                    row.get("backing_cash_inflow", row.get("backing_inflow", 0.0))
                ),
                backing_voucher_inflow=safe_float(row.get("backing_voucher_inflow", 0.0)),
                tagged_voucher_tokens=safe_float(row.get("tagged_voucher_tokens")),
                verified_report_exposure=safe_float(row.get("verified_report_exposure")),
                same_token_return_rate=safe_float(row.get("same_token_return_rate")),
                same_token_out_value=safe_float(row.get("same_token_out_value")),
                same_token_matched_later_in_value=safe_float(row.get("same_token_matched_later_in_value")),
                borrow_proxy_matured_events=safe_float(row.get("borrow_proxy_matured_events")),
                borrow_proxy_matured_return_rate=safe_float(row.get("borrow_proxy_matured_return_rate")),
                rosca_proxy_value_return_rate=safe_float(row.get("rosca_proxy_value_return_rate")),
            )
        )

    tier_counts = Counter(pool.tier for pool in pool_rows)
    total_tiers = sum(tier_counts.values()) or 1
    tier_probs = {tier: count / total_tiers for tier, count in tier_counts.items()}

    voucher_coverage_by_tier = {}
    for tier in ("strong", "moderate", "weak"):
        redeemable_value = tier_asset_out_values.get((tier, "redeemable_voucher"), 0.0)
        internal_value = tier_asset_out_values.get((tier, "internal_voucher"), 0.0)
        denom = redeemable_value + internal_value
        if denom <= 0.0:
            voucher_coverage_by_tier[tier] = 0.0
            continue
        voucher_coverage_by_tier[tier] = (
            repayment_by_tier_asset.get((tier, "redeemable_voucher"), 0.0) * redeemable_value
            + repayment_by_tier_asset.get((tier, "internal_voucher"), 0.0) * internal_value
        ) / denom

    borrow_return_by_tier = {}
    borrow_path = calibration_dir / "borrow_repayment_by_tier.csv"
    if borrow_path.exists():
        for row in read_csv(borrow_path):
            tier = str(row.get("tier", "")).strip().lower()
            if tier:
                borrow_return_by_tier[tier] = safe_float(row.get("weighted_matured_borrow_return_rate"))

    impact_rows = []
    for row in read_csv(calibration_dir / "impact_projection_by_activity.csv"):
        activity = row["activity"]
        if activity == "Unclassified local activity":
            continue
        impact_rows.append(
            ImpactProjection(
                activity=activity,
                intercept=safe_float(row["log_intercept"]),
                slope=safe_float(row["log_slope"]),
                share=safe_float(row["verified_exposure_share"]),
            )
        )
    impact_rows.sort(key=lambda item: item.share, reverse=True)

    voucher_circulation_baselines: dict[str, dict[str, float]] = {}
    circulation_path = calibration_dir / "voucher_circulation_baseline.csv"
    if circulation_path.exists():
        for row in read_csv(circulation_path):
            window = str(row.get("window", "")).strip()
            if not window:
                continue
            voucher_circulation_baselines[window] = {
                key: safe_float(value)
                for key, value in row.items()
                if key != "window"
            }

    stable_dependency_anchors: dict[str, dict[str, float]] = {}
    stable_dependency_path = calibration_dir / "stable_dependency_anchors.csv"
    if stable_dependency_path.exists():
        for row in read_csv(stable_dependency_path):
            metric = str(row.get("metric", "")).strip()
            if not metric:
                continue
            stable_dependency_anchors[metric] = {
                "value": safe_float(row.get("value")),
                "denominator": safe_float(row.get("denominator")),
            }

    producer_deposit_by_tier: dict[str, dict[str, float]] = {}
    producer_deposit_path = calibration_dir / "producer_deposit_calibration.csv"
    if producer_deposit_path.exists():
        for row in read_csv(producer_deposit_path):
            tier = str(row.get("tier", "")).strip().lower()
            if tier:
                producer_deposit_by_tier[tier] = {key: safe_float(value) for key, value in row.items() if key != "tier"}

    productive_credit_by_tier: dict[str, dict[str, float]] = {}
    productive_credit_path = calibration_dir / "productive_credit_calibration.csv"
    if productive_credit_path.exists():
        for row in read_csv(productive_credit_path):
            tier = str(row.get("tier", "")).strip().lower()
            if tier:
                productive_credit_by_tier[tier] = {key: safe_float(value) for key, value in row.items() if key != "tier"}
    aggregate_productive_credit = productive_credit_by_tier.get("all", {})
    for tier in TIER_ORDER:
        if tier in productive_credit_by_tier:
            continue
        if aggregate_productive_credit:
            row = dict(aggregate_productive_credit)
            row["aggregate_fallback"] = 1.0
            productive_credit_by_tier[tier] = row
            continue
        rate = borrow_return_by_tier.get(tier, 0.0)
        productive_credit_by_tier[tier] = {
            "productive_credit_return_rate": rate,
            "productive_credit_lag_ticks_p50": 2.0,
            "productive_credit_lag_ticks_p90": 12.0,
            "aggregate_fallback": 0.0,
        }

    def load_metric_table(name: str) -> dict[str, float]:
        path = calibration_dir / name
        values: dict[str, float] = {}
        if not path.exists():
            return values
        for row in read_csv(path):
            metric = str(row.get("metric", row.get("parameter", ""))).strip()
            if not metric:
                continue
            values[metric] = safe_float(row.get("value"))
        return values

    def load_overlap_distribution() -> dict[str, float]:
        path = calibration_dir / "voucher_pool_overlap_distribution.csv"
        values: dict[str, float] = {}
        if not path.exists():
            return values
        for row in read_csv(path):
            bucket = str(row.get("pool_degree_bucket", "")).strip()
            if not bucket:
                continue
            values[bucket] = max(0.0, safe_float(row.get("share")))
        total = sum(values.values())
        if total > 1e-9:
            values = {bucket: share / total for bucket, share in values.items()}
        return values

    debt_removal_calibration = load_metric_table("debt_removal_calibration.csv")
    productive_credit_activity_boost_calibration = load_metric_table(
        "productive_credit_activity_boost_calibration.csv"
    )
    fee_conversion_calibration = load_metric_table("fee_conversion_calibration.csv")
    quarterly_clearing_calibration = load_metric_table("quarterly_clearing_calibration.csv")
    settlement_reliability_anchors = load_metric_table("settlement_reliability_anchors.csv")
    route_substitution_diagnostics = load_metric_table("route_substitution_diagnostics.csv")
    voucher_pool_overlap_calibration = load_metric_table("voucher_pool_overlap_calibration.csv")
    voucher_pool_overlap_distribution = load_overlap_distribution()
    stable_actor_demographics = load_metric_table("stable_actor_demographics.csv")
    unit_normalization = load_metric_table("unit_normalization_calibration.csv")

    return Calibration(
        calibration_hash=bundle_hash,
        params=params,
        repayment_by_tier_asset=repayment_by_tier_asset,
        repayment_out_value_by_tier_asset=tier_asset_out_values,
        tier_probs=tier_probs,
        voucher_coverage_by_tier=voucher_coverage_by_tier,
        pool_rows=pool_rows,
        borrow_return_by_tier=borrow_return_by_tier,
        impact_rows=impact_rows,
        voucher_circulation_baselines=voucher_circulation_baselines,
        stable_dependency_anchors=stable_dependency_anchors,
        producer_deposit_by_tier=producer_deposit_by_tier,
        productive_credit_by_tier=productive_credit_by_tier,
        productive_credit_activity_boost_calibration=productive_credit_activity_boost_calibration,
        debt_removal_calibration=debt_removal_calibration,
        fee_conversion_calibration=fee_conversion_calibration,
        quarterly_clearing_calibration=quarterly_clearing_calibration,
        settlement_reliability_anchors=settlement_reliability_anchors,
        route_substitution_diagnostics=route_substitution_diagnostics,
        voucher_pool_overlap_calibration=voucher_pool_overlap_calibration,
        voucher_pool_overlap_distribution=voucher_pool_overlap_distribution,
        stable_actor_demographics=stable_actor_demographics,
        unit_normalization=unit_normalization,
    )


def apply_unit_normalization_context(args: argparse.Namespace, calibration: Calibration) -> None:
    kes_per_usd = safe_float(calibration.unit_normalization.get("kes_per_usd"))
    voucher_kes_value = safe_float(
        calibration.unit_normalization.get("individual_voucher_kes_value_default")
    ) or 1.0
    args._calibration_kes_per_usd = kes_per_usd
    args._voucher_unit_value_usd = (voucher_kes_value / kes_per_usd) if kes_per_usd > 0.0 else 1.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return percentile(values, 0.50)


def pools_by_tier(calibration: Calibration) -> dict[str, list[PoolCalibration]]:
    grouped = {tier: [] for tier in TIER_ORDER}
    for pool in calibration.pool_rows:
        grouped.setdefault(pool.tier, []).append(pool)
    return grouped


def weighted_pool_rate(pools: list[PoolCalibration], rate_attr: str, weight_attr: str) -> float:
    total_weight = sum(max(0.0, float(getattr(pool, weight_attr))) for pool in pools)
    if total_weight <= 0.0:
        return mean(float(getattr(pool, rate_attr)) for pool in pools)
    return sum(
        max(0.0, float(getattr(pool, weight_attr))) * max(0.0, float(getattr(pool, rate_attr)))
        for pool in pools
    ) / total_weight


def calibration_same_token_return(calibration: Calibration, tier: str) -> float:
    values = []
    weights = []
    for asset in ASSET_ORDER:
        weight = calibration.repayment_out_value_by_tier_asset.get((tier, asset), 0.0)
        if weight <= 0.0:
            continue
        values.append(calibration.repayment_by_tier_asset.get((tier, asset), 0.0))
        weights.append(weight)
    denom = sum(weights)
    if denom <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / denom


def empirical_tier_targets(calibration: Calibration, ticks: int) -> list[dict[str, object]]:
    rows = []
    grouped = pools_by_tier(calibration)
    for tier in TIER_ORDER:
        pools = grouped.get(tier, [])
        period_swaps = [pool.weekly_swap_rate * ticks for pool in pools]
        period_reports = [
            pool.verified_report_exposure * (ticks / RECENT_WINDOW_TICKS)
            for pool in pools
        ]
        rows.append(
            {
                "tier": tier,
                "pool_count": len(pools),
                "mean_swap_events_horizon": mean(period_swaps),
                "median_swap_events_horizon": median(period_swaps),
                "total_swap_events_horizon": sum(period_swaps),
                "total_verified_report_exposure_horizon": sum(period_reports),
                "mean_same_token_return_rate": weighted_pool_rate(
                    pools, "same_token_return_rate", "same_token_out_value"
                ),
                "same_token_return_coverage": calibration_same_token_return(calibration, tier),
                "cash_return_coverage": calibration.repayment_by_tier_asset.get((tier, "cash"), 0.0),
                "voucher_return_coverage": calibration.voucher_coverage_by_tier.get(tier, 0.0),
                "borrow_proxy_closure": calibration.borrow_return_by_tier.get(tier, 0.0),
                "backing_total_inflow": sum(pool.backing_inflow for pool in pools),
                "backing_liquidity_inflow": sum(pool.backing_cash_inflow for pool in pools),
                "backing_cash_inflow": sum(pool.backing_cash_inflow for pool in pools),
                "backing_voucher_inflow": sum(pool.backing_voucher_inflow for pool in pools),
            }
        )
    return rows


def scaled_tier_counts(calibration: Calibration, factor: float) -> dict[str, int]:
    base = Counter(pool.tier for pool in calibration.pool_rows)
    raw = {tier: base.get(tier, 0) * factor for tier in TIER_ORDER}
    counts = {tier: int(math.floor(value)) for tier, value in raw.items()}
    target_total = int(round(sum(base.values()) * factor))
    remainder = target_total - sum(counts.values())
    order = sorted(TIER_ORDER, key=lambda tier: raw[tier] - counts[tier], reverse=True)
    for idx in range(max(0, remainder)):
        counts[order[idx % len(order)]] += 1
    return counts


def scaled_count(value: float, factor: float) -> int:
    return max(0, int(round(max(0.0, value) * factor)))


def empirical_unique_producer_vouchers(calibration: Calibration) -> float:
    actor_value = safe_float(
        calibration.stable_actor_demographics.get("recommended_producer_wallets")
    )
    if actor_value > 0.0:
        return actor_value
    unique_tokens = safe_float(
        calibration.voucher_pool_overlap_calibration.get("total_voucher_tokens")
    )
    if unique_tokens > 0.0:
        return unique_tokens
    return sum(pool.accepted_voucher_members for pool in calibration.pool_rows)


def role_counts_for_calibration(calibration: Calibration, factor: float) -> dict[str, int]:
    """Map empirical cohort demographics onto explicit runtime wallets.

    The empirical cohort rows are community lending pools. Unique KES/KSh
    voucher tokens are modeled as producer/member wallets, accepted-voucher
    counts remain pool membership slots, and active interactors are modeled as
    consumer wallets with private stable balances. When the stable-actor
    demographics file is available, consumer wallets are external
    non-producer stable-side users rather than all active interactors.
    """
    lender_count = scaled_count(len(calibration.pool_rows), factor)
    producer_count = scaled_count(empirical_unique_producer_vouchers(calibration), factor)
    empirical_consumer_wallets = safe_float(
        calibration.stable_actor_demographics.get("recommended_consumer_wallets")
    )
    if empirical_consumer_wallets <= 0.0:
        empirical_consumer_wallets = sum(pool.total_users for pool in calibration.pool_rows)
    consumer_count = scaled_count(
        empirical_consumer_wallets,
        factor,
    )
    if calibration.pool_rows and lender_count <= 0:
        lender_count = 1
    return {
        "lenders": lender_count,
        "producers": producer_count,
        "consumers": consumer_count,
    }


def apply_network_context(
    args: argparse.Namespace,
    *,
    calibration: Calibration,
    network_scale: str,
    principal_ratio: float = 0.0,
    principal_usd: float = 0.0,
    service_share: float = 0.0,
    certification_policy: str = "strong_moderate_capped",
) -> None:
    factor = NETWORK_SCALE_FACTORS.get(network_scale)
    if factor is None:
        raise ValueError(f"Unknown network scale: {network_scale}")
    target_counts = scaled_tier_counts(calibration, factor)
    role_counts = role_counts_for_calibration(calibration, factor)
    args._target_tier_counts = target_counts
    args._current_network_scale = network_scale
    args._current_scale_factor = factor
    args._current_principal_ratio = float(principal_ratio)
    args._current_principal_usd = float(principal_usd)
    args._current_bond_fee_service_share = float(service_share)
    args._current_certification_policy = certification_policy
    args._current_initial_lenders = role_counts["lenders"]
    args._current_initial_producers = role_counts["producers"]
    args._current_initial_consumers = role_counts["consumers"]
    args._current_empirical_lender_pool_templates = len(calibration.pool_rows)
    args._current_empirical_unique_producer_vouchers = empirical_unique_producer_vouchers(calibration)
    args._current_empirical_accepted_voucher_members = sum(
        pool.accepted_voucher_members for pool in calibration.pool_rows
    )
    args._current_empirical_active_interactors = sum(
        pool.total_users for pool in calibration.pool_rows
    )
    args._current_empirical_unique_pool_interactor_addresses = safe_float(
        calibration.stable_actor_demographics.get("unique_pool_interactor_addresses")
    )
    args._current_empirical_stable_external_nonproducer_users = safe_float(
        calibration.stable_actor_demographics.get("stable_external_nonproducer_users")
    )
    args._current_empirical_producer_stable_users = safe_float(
        calibration.stable_actor_demographics.get("producer_stable_users")
    )


def configure_sarafu_activity_controls(args: argparse.Namespace, calibration: Calibration, ticks: int, prefix: str) -> None:
    empirical_targets = empirical_tier_targets(calibration, int(ticks))
    empirical_total_swaps = sum(safe_float(row["total_swap_events_horizon"]) for row in empirical_targets)
    empirical_total_backing = sum(safe_float(row.get("backing_total_inflow")) for row in empirical_targets)
    empirical_cash_backing = sum(
        safe_float(row.get("backing_cash_inflow", row.get("backing_liquidity_inflow")))
        for row in empirical_targets
    )
    empirical_voucher_backing = sum(safe_float(row.get("backing_voucher_inflow")) for row in empirical_targets)
    swap_floor = int(math.ceil(empirical_total_swaps / max(1, int(ticks))))
    setattr(args, f"_{prefix}_swap_floor_per_tick", swap_floor)
    setattr(args, f"_{prefix}_route_requests_per_tick", 1)
    setattr(args, f"_{prefix}_swap_budget_per_tick", max(60, int(math.ceil(swap_floor * 0.25))))
    setattr(args, f"_{prefix}_swap_attempts_max_per_pool", 2)
    setattr(args, f"_{prefix}_swap_sustain_max_extra_attempts", max(1, int(math.ceil(swap_floor * 3.0))))
    setattr(args, f"_{prefix}_swap_sustain_max_rounds", 8)
    setattr(args, f"_{prefix}_swap_sustain_attempts_per_missing_swap", 3.0)
    setattr(args, f"_{prefix}_historical_backing_total_usd", empirical_total_backing)
    setattr(args, f"_{prefix}_historical_cash_backing_total_usd", empirical_cash_backing)
    setattr(args, f"_{prefix}_historical_voucher_backing_total_usd", empirical_voucher_backing)
    stable_deposit_rate = 0.0
    voucher_deposit_rate = 0.0
    productive_return_rate = 0.0
    productive_lag_ticks = 0.0
    productive_voucher_deposit_share = 0.0
    productive_voucher_deposit_cap_rate = 0.0
    weight_total = 0.0
    for tier in TIER_ORDER:
        weight = calibration.tier_probs.get(tier, 0.0)
        if weight <= 0.0:
            continue
        dep = calibration.producer_deposit_by_tier.get(tier, {})
        stable_deposit_rate += weight * safe_float(dep.get("stable_deposit_rate_per_month"))
        voucher_deposit_rate += weight * safe_float(dep.get("voucher_deposit_rate_per_month"))
        prod = calibration.productive_credit_by_tier.get(tier, {})
        productive_return_rate += weight * safe_float(
            prod.get("productive_credit_return_rate", calibration.borrow_return_by_tier.get(tier, 0.0))
        )
        productive_lag_ticks += weight * safe_float(prod.get("productive_credit_lag_ticks_p50", 2.0))
        productive_voucher_deposit_share += weight * safe_float(
            prod.get("loan_induced_voucher_deposit_share_default", 0.0)
        )
        productive_voucher_deposit_cap_rate += weight * safe_float(
            prod.get("loan_induced_voucher_deposit_cap_rate_per_month", dep.get("voucher_deposit_rate_per_month", 0.0))
        )
        weight_total += weight
    if weight_total > 1e-9:
        productive_lag_ticks = productive_lag_ticks / weight_total
        productive_voucher_deposit_share = productive_voucher_deposit_share / weight_total
        productive_voucher_deposit_cap_rate = productive_voucher_deposit_cap_rate / weight_total
    else:
        productive_lag_ticks = 2.0
    setattr(args, f"_{prefix}_producer_stable_deposit_rate_per_month", stable_deposit_rate)
    setattr(args, f"_{prefix}_producer_voucher_deposit_rate_per_month", voucher_deposit_rate)
    setattr(args, f"_{prefix}_productive_credit_return_rate", productive_return_rate)
    setattr(args, f"_{prefix}_productive_credit_lag_ticks", max(1, int(round(productive_lag_ticks))))
    setattr(
        args,
        f"_{prefix}_productive_credit_voucher_deposit_share",
        max(0.0, min(1.0, productive_voucher_deposit_share)),
    )
    setattr(
        args,
        f"_{prefix}_productive_credit_voucher_deposit_cap_rate_per_month",
        max(0.0, productive_voucher_deposit_cap_rate),
    )
    debt_maturity_recovery_rate = safe_float(
        calibration.settlement_reliability_anchors.get("mature_borrow_proxy_value_support_rate")
    )
    if debt_maturity_recovery_rate <= 0.0:
        debt_maturity_recovery_rate = safe_float(
            calibration.settlement_reliability_anchors.get("mature_borrow_proxy_event_return_rate")
        )
    if debt_maturity_recovery_rate <= 0.0:
        debt_maturity_recovery_rate = productive_return_rate
    setattr(
        args,
        f"_{prefix}_producer_debt_maturity_recovery_rate",
        max(0.0, min(1.0, debt_maturity_recovery_rate)),
    )
    current_circulation = (
        calibration.voucher_circulation_baselines.get("trailing_90d")
        or calibration.voucher_circulation_baselines.get("trailing_52w")
        or calibration.voucher_circulation_baselines.get("pool_era")
        or {}
    )
    voucher_to_voucher_count = safe_float(current_circulation.get("voucher_to_voucher_count"))
    voucher_to_stable_count = safe_float(current_circulation.get("voucher_to_stable_count"))
    voucher_source_borrowing_count = voucher_to_voucher_count + voucher_to_stable_count
    if voucher_source_borrowing_count > 1e-9:
        primary_voucher_borrowing_attempt_share = voucher_to_voucher_count / voucher_source_borrowing_count
    else:
        primary_voucher_borrowing_attempt_share = safe_float(
            current_circulation.get("voucher_to_voucher_share"),
            0.50,
        )
    setattr(
        args,
        f"_{prefix}_producer_primary_voucher_borrowing_attempt_share",
        max(0.0, min(1.0, primary_voucher_borrowing_attempt_share)),
    )
    activity_boost = calibration.productive_credit_activity_boost_calibration
    setattr(
        args,
        f"_{prefix}_productive_credit_voucher_source_weight_boost",
        max(
            0.0,
            safe_float(activity_boost.get("calibrated_voucher_source_weight_boost"), 0.0),
        ),
    )
    setattr(
        args,
        f"_{prefix}_productive_credit_voucher_source_size_multiplier",
        max(
            0.0,
            safe_float(activity_boost.get("calibrated_voucher_source_size_multiplier"), 1.0),
        ),
    )
    setattr(
        args,
        f"_{prefix}_quarterly_clearing_surplus_share",
        calibration.quarterly_clearing_calibration.get("surplus_clearable_share", 1.0),
    )
    setattr(
        args,
        f"_{prefix}_voucher_pool_overlap_distribution",
        dict(calibration.voucher_pool_overlap_distribution),
    )
    setattr(
        args,
        f"_{prefix}_voucher_pool_overlap_multi_pool_share",
        safe_float(calibration.voucher_pool_overlap_calibration.get("multi_pool_share")),
    )
    setattr(
        args,
        f"_{prefix}_voucher_pool_overlap_degree_p50",
        safe_float(calibration.voucher_pool_overlap_calibration.get("pool_degree_p50")),
    )
    setattr(
        args,
        f"_{prefix}_voucher_pool_overlap_degree_p90",
        safe_float(calibration.voucher_pool_overlap_calibration.get("pool_degree_p90")),
    )


def certified_pool_capacity(calibration: Calibration, network_scale: str, policy: str) -> dict[str, float]:
    factor = NETWORK_SCALE_FACTORS.get(network_scale, 1.0)
    strong_allocations = [
        max(0.0, pool.backing_inflow) for pool in calibration.pool_rows if pool.tier == "strong"
    ]
    median_strong_allocation = median(strong_allocations) if strong_allocations else 0.0
    moderate_allocation_cap = 0.50 * median_strong_allocation
    weights: dict[str, float] = {}
    backing_capacity = 0.0
    for pool in calibration.pool_rows:
        base_capacity = max(0.0, pool.backing_inflow)
        if policy == "strong_only":
            weight = 1.0 if pool.tier == "strong" else 0.0
            allocation = base_capacity * weight
        elif policy in {"all_tiers_weighted", "weak_inclusion_stress"}:
            weight = {"strong": 1.0, "moderate": 0.35, "weak": 0.10}.get(pool.tier, 0.0)
            allocation = base_capacity * weight
        else:
            if pool.tier == "strong":
                weight = 1.0
                allocation = base_capacity
            elif pool.tier == "moderate":
                allocation = min(0.35 * base_capacity, moderate_allocation_cap)
                weight = allocation / max(1e-9, base_capacity)
            else:
                weight = 0.0
                allocation = 0.0
        weights[pool.template_id] = weight
        backing_capacity += allocation
    certified_count = sum(1 for weight in weights.values() if weight > 0.0)
    strong_count = sum(1 for pool in calibration.pool_rows if weights.get(pool.template_id, 0.0) > 0.0 and pool.tier == "strong")
    moderate_count = sum(1 for pool in calibration.pool_rows if weights.get(pool.template_id, 0.0) > 0.0 and pool.tier == "moderate")
    weak_count = sum(1 for pool in calibration.pool_rows if weights.get(pool.template_id, 0.0) > 0.0 and pool.tier == "weak")
    return {
        "certified_pool_count": certified_count * factor,
        "certified_strong_pool_count": strong_count * factor,
        "certified_moderate_pool_count": moderate_count * factor,
        "certified_weak_pool_count": weak_count * factor,
        "certified_backing_capacity_usd": backing_capacity * factor,
    }


def scenario_config(
    scenario: str,
    coupon: float,
    term_ticks: int,
    args: argparse.Namespace,
) -> ScenarioConfig:
    cfg = ScenarioConfig(
        debug_inventory=False,
        metrics_stride=1,
        pool_metrics_stride=max(1, int(args.pool_metrics_stride)),
        event_log_maxlen=None,
        bond_coupon_target_annual=float(coupon),
        bond_term_ticks=int(term_ticks),
        bond_fee_service_share=1.0,
        bond_return_mode="lp_sclc",
        calibration_profile="sarafu_empirical",
    )
    cfg.max_active_pools_per_tick = args.max_active_pools_per_tick
    cfg.kes_per_usd = max(0.0, float(getattr(args, "_calibration_kes_per_usd", 0.0)))
    cfg.voucher_unit_value_usd = max(
        1e-12,
        float(getattr(args, "_voucher_unit_value_usd", 1.0)),
    )

    if scenario == "mutual_aid_only":
        cfg.economics_enabled = False
        cfg.initial_lenders = 0
        cfg.initial_liquidity_providers = 0
        cfg.lender_initial_stable_mean = 0.0
        cfg.lp_initial_stable_mean = 0.0
        cfg.pool_fee_rate = 0.0
        cfg.clc_rake_rate = 0.0
        cfg.stable_flow_mode = "none"
        cfg.calibration_profile = "mutual_aid_only"
    elif scenario == "sarafu_like_pools":
        cfg.initial_liquidity_providers = 0
        cfg.lp_initial_stable_mean = 0.0
        cfg.calibration_profile = "sarafu_like"
    elif scenario == "sarafu_engine_validation":
        cfg.initial_liquidity_providers = 0
        cfg.lp_initial_stable_mean = 0.0
        cfg.bond_coupon_target_annual = 0.0
        cfg.bond_fee_service_share = 0.0
        cfg.producer_inflow_per_tick = 0.0
        cfg.consumer_inflow_per_tick = 0.0
        cfg.lender_inflow_per_tick = 0.0
        cfg.stable_inflow_per_tick = 0.0
        cfg.initial_stable_per_pool_mean = 0.0
        cfg.min_stable_reserve_mean = 0.0
        cfg.stable_supply_growth_rate = 0.0
        cfg.stable_supply_noise = 0.0
        cfg.initial_lenders = int(getattr(args, "_current_initial_lenders", cfg.initial_lenders))
        cfg.initial_producers = int(getattr(args, "_current_initial_producers", cfg.initial_producers))
        cfg.initial_consumers = int(getattr(args, "_current_initial_consumers", cfg.initial_consumers))
        cfg.max_pools = cfg.initial_lenders + cfg.initial_producers + cfg.initial_consumers
        cfg.random_route_requests_per_tick = int(getattr(args, "_validation_route_requests_per_tick", 2))
        cfg.swap_requests_budget_per_tick = int(getattr(args, "_validation_swap_budget_per_tick", 120))
        cfg.swap_attempts_max_per_pool = int(getattr(args, "_validation_swap_attempts_max_per_pool", 2))
        cfg.max_hops = 3
        cfg.noam_max_hops = 3
        cfg.noam_overlay_enabled = False
        cfg.noam_clearing_enabled = False
        cfg.route_substitution_enabled = True
        cfg.route_substitution_max_alternatives = 3
        cfg.swap_sustain_window_ticks = 0
        cfg.swap_sustain_floor_per_tick = int(getattr(args, "_validation_swap_floor_per_tick", 0))
        cfg.swap_sustain_max_extra_attempts = int(
            getattr(args, "_validation_swap_sustain_max_extra_attempts", 500)
        )
        cfg.swap_sustain_max_rounds = int(getattr(args, "_validation_swap_sustain_max_rounds", 2))
        cfg.swap_sustain_attempts_per_missing_swap = float(
            getattr(args, "_validation_swap_sustain_attempts_per_missing_swap", 1.0)
        )
        stable_backing_total = float(
            getattr(args, "_validation_historical_cash_backing_total_usd", 0.0)
        )
        if stable_backing_total > 0.0:
            cfg.historical_stable_backing_tick = 1
            cfg.historical_stable_backing_total_usd = stable_backing_total
            cfg.historical_stable_backing_roles = ("lender",)
        voucher_backing_total = float(
            getattr(args, "_validation_historical_voucher_backing_total_usd", 0.0)
        )
        if voucher_backing_total > 0.0:
            cfg.producer_deposits_enabled = True
            cfg.historical_voucher_backing_tick = 1
            cfg.historical_voucher_backing_total_usd = voucher_backing_total
        cfg.producer_voucher_overlap_mode = "empirical_overlap"
        cfg.producer_voucher_single_lender = False
        cfg.producer_voucher_overlap_bucket_weights = dict(
            getattr(args, "_validation_voucher_pool_overlap_distribution", {}) or {}
        )
        cfg.consumer_stable_source_bias = 0.25
        cfg.producer_stable_source_bias = 0.05
        cfg.producer_consumer_stable_target_bias = 0.0
        cfg.stable_source_swap_size_multiplier = 2.0
        cfg.voucher_source_swap_size_multiplier = 0.5
        cfg.stable_excess_sweep_enabled = True
        cfg.stable_excess_sweep_after_stable_receipt = False
        cfg.stable_excess_sweep_buffer_voucher_share = 0.05
        cfg.stable_excess_sweep_roles = ("producer", "consumer")
        cfg.calibration_profile = "sarafu_engine_validation"
    elif scenario == "regenbond_lp_injection":
        cfg.initial_liquidity_providers = 1
        cfg.lp_initial_stable_mean = 400_000.0
        cfg.calibration_profile = "sarafu_empirical"
    elif scenario == "bond_issuer_frontier":
        gross_principal = max(0.0, float(getattr(args, "_current_principal_usd", 0.0)))
        reserve_share = 0.0
        deployed_principal = gross_principal
        factor = max(1.0, float(getattr(args, "_current_scale_factor", 1.0)))
        cfg.bond_fee_service_share = max(
            0.0, min(1.0, float(getattr(args, "_current_bond_fee_service_share", 1.0)))
        )
        cfg.bond_gross_principal_usd = gross_principal
        cfg.bond_deployed_principal_usd = deployed_principal
        cfg.issuer_reserve_share = reserve_share
        cfg.issuer_payment_stride_ticks = max(1, int(getattr(args, "issuer_payment_stride", 13)))
        cfg.bond_return_mode = "issuer_cashflow"
        cfg.initial_lenders = int(getattr(args, "_current_initial_lenders", cfg.initial_lenders))
        cfg.initial_producers = int(getattr(args, "_current_initial_producers", cfg.initial_producers))
        cfg.initial_consumers = int(getattr(args, "_current_initial_consumers", cfg.initial_consumers))
        if gross_principal > 1e-9 and cfg.initial_lenders <= 0:
            cfg.initial_lenders = 1
        cfg.initial_liquidity_providers = 0
        cfg.lp_initial_stable_mean = 0.0
        cfg.lender_initial_stable_mean = (
            deployed_principal / max(1, cfg.initial_lenders) if deployed_principal > 1e-9 else 0.0
        )
        cfg.max_pools = (
            cfg.initial_lenders
            + cfg.initial_producers
            + cfg.initial_consumers
            + cfg.initial_liquidity_providers
        )
        cfg.min_stable_reserve_mean = 0.0
        cfg.producer_inflow_per_tick = 0.0
        cfg.consumer_inflow_per_tick = 0.0
        cfg.lender_inflow_per_tick = 0.0
        cfg.stable_inflow_per_tick = 0.0
        cfg.stable_supply_growth_rate = 0.0
        cfg.stable_supply_noise = 0.0
        cfg.producer_deposits_enabled = True
        cfg.producer_deposit_stride_ticks = 4
        cfg.producer_stable_deposit_rate_per_month = max(
            0.0, float(getattr(args, "_frontier_producer_stable_deposit_rate_per_month", 0.0))
        )
        cfg.producer_voucher_deposit_rate_per_month = max(
            0.0, float(getattr(args, "_frontier_producer_voucher_deposit_rate_per_month", 0.0))
        )
        cfg.lender_voucher_cap_deposit_multiple = 5.0
        cfg.producer_voucher_overlap_mode = str(
            getattr(args, "producer_voucher_overlap_mode", "single_lender") or "single_lender"
        )
        if cfg.producer_voucher_overlap_mode == "empirical_overlap":
            cfg.producer_voucher_single_lender = False
            cfg.producer_voucher_overlap_bucket_weights = dict(
                getattr(args, "_frontier_voucher_pool_overlap_distribution", {}) or {}
            )
        cfg.productive_credit_enabled = True
        cfg.productive_credit_return_rate = max(
            0.0, float(getattr(args, "_frontier_productive_credit_return_rate", 0.0))
        )
        cfg.productive_credit_lag_ticks = max(
            1, int(getattr(args, "_frontier_productive_credit_lag_ticks", 2))
        )
        cfg.productive_credit_voucher_feedback_enabled = True
        cfg.productive_credit_voucher_deposit_share = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(args, "productive_credit_voucher_deposit_share", None)
                    if getattr(args, "productive_credit_voucher_deposit_share", None) is not None
                    else getattr(args, "_frontier_productive_credit_voucher_deposit_share", 0.0)
                ),
            ),
        )
        cfg.productive_credit_voucher_deposit_cap_rate_per_month = max(
            0.0,
            float(
                getattr(args, "productive_credit_voucher_deposit_cap_rate_per_month", None)
                if getattr(args, "productive_credit_voucher_deposit_cap_rate_per_month", None) is not None
                else getattr(args, "_frontier_productive_credit_voucher_deposit_cap_rate_per_month", 0.0)
            ),
        )
        cfg.productive_credit_voucher_activity_boost_enabled = not bool(
            getattr(args, "disable_productive_credit_voucher_activity_boost", False)
        )
        cfg.productive_credit_voucher_activity_boost_window_ticks = max(
            1, int(getattr(args, "issuer_payment_stride", 13))
        )
        cfg.productive_credit_voucher_source_weight_boost = max(
            0.0,
            float(getattr(args, "_frontier_productive_credit_voucher_source_weight_boost", 0.0)),
        )
        cfg.productive_credit_voucher_source_size_multiplier = max(
            0.0,
            float(
                getattr(
                    args,
                    "_frontier_productive_credit_voucher_source_size_multiplier",
                    1.0,
                )
            ),
        )
        cfg.producer_debt_maturity_enabled = True
        cfg.producer_debt_maturity_ticks = max(1, int(getattr(args, "issuer_payment_stride", 13)))
        cfg.producer_debt_maturity_recovery_rate = max(
            0.0,
            min(
                1.0,
                float(getattr(args, "_frontier_producer_debt_maturity_recovery_rate", 1.0)),
            ),
        )
        cfg.producer_debt_maturity_preserve_reserve = True
        cfg.loan_term_weeks = cfg.producer_debt_maturity_ticks
        cfg.voucher_fee_conversion_enabled = True
        cfg.quarterly_clearing_enabled = True
        cfg.quarterly_clearing_stride_ticks = max(1, int(getattr(args, "issuer_payment_stride", 13)))
        cfg.quarterly_clearing_surplus_share = max(
            0.0, min(1.0, float(getattr(args, "_frontier_quarterly_clearing_surplus_share", 1.0)))
        )
        cfg.bond_service_reserve_enabled = True
        cfg.bond_service_reserve_recovery_share = 1.0
        cfg.bond_service_lockbox_mode = str(
            getattr(args, "bond_service_lockbox_mode", "remaining_schedule") or "remaining_schedule"
        )
        cfg.bond_service_lockbox_coverage_ratio = max(
            0.0,
            float(getattr(args, "bond_service_lockbox_coverage_ratio", 1.25) or 0.0),
        )
        legacy_margin = getattr(args, "producer_debt_contract_service_margin_rate", None)
        stable_margin_arg = getattr(args, "producer_stable_debt_contract_service_margin_rate", None)
        voucher_margin_arg = getattr(args, "producer_voucher_debt_contract_service_margin_rate", None)
        shared_contract_margin = max(
            0.0,
            float(legacy_margin if legacy_margin is not None else 0.0),
        )
        stable_contract_margin = max(
            0.0,
            float(
                stable_margin_arg
                if stable_margin_arg is not None
                else shared_contract_margin
            ),
        )
        voucher_contract_margin = max(
            0.0,
            float(
                voucher_margin_arg
                if voucher_margin_arg is not None
                else shared_contract_margin
            ),
        )
        cfg.producer_debt_contract_repayment_enabled = True
        cfg.producer_debt_contract_service_margin_rate = shared_contract_margin
        cfg.producer_stable_debt_contract_service_margin_rate = stable_contract_margin
        cfg.producer_voucher_debt_contract_service_margin_rate = voucher_contract_margin
        cfg.producer_debt_contract_revenue_rate = max(
            cfg.productive_credit_return_rate,
            1.0 + stable_contract_margin,
        )
        cfg.ordinary_stable_spend_protection_enabled = bool(
            getattr(args, "enable_ordinary_stable_spend_protection", False)
        ) and not bool(getattr(args, "disable_ordinary_stable_spend_protection", False))
        cfg.ordinary_stable_spend_buffer_voucher_share = (
            0.05 if cfg.ordinary_stable_spend_protection_enabled else 0.0
        )
        cfg.producer_loan_failure_backfill_enabled = not bool(
            getattr(args, "disable_producer_loan_failure_backfill", False)
        )
        cfg.producer_loan_failure_backfill_max_attempts = 1
        cfg.producer_voucher_loan_fallback_enabled = bool(
            getattr(args, "enable_producer_voucher_loan_fallback", False)
        )
        cfg.producer_voucher_loan_activity_boost_enabled = bool(
            getattr(args, "enable_producer_voucher_loan_activity_boost", False)
        )
        cfg.producer_primary_voucher_borrowing_enabled = bool(
            getattr(args, "enable_producer_primary_voucher_borrowing", False)
        )
        if cfg.producer_primary_voucher_borrowing_enabled:
            cfg.producer_primary_voucher_borrowing_attempt_share = max(
                0.0,
                min(
                    1.0,
                    float(
                        getattr(
                            args,
                            "producer_primary_voucher_borrowing_attempt_share",
                            None,
                        )
                        if getattr(args, "producer_primary_voucher_borrowing_attempt_share", None)
                        is not None
                        else getattr(
                            args,
                            "_frontier_producer_primary_voucher_borrowing_attempt_share",
                            0.50,
                        )
                        or 0.0
                    ),
                ),
            )
        else:
            cfg.producer_primary_voucher_borrowing_attempt_share = 0.0
        cfg.producer_voucher_loan_max_target_candidates = max(
            1, int(getattr(args, "producer_voucher_loan_max_target_candidates", 3) or 3)
        )
        cfg.lender_voucher_purchase_demand_enabled = bool(
            getattr(args, "enable_lender_voucher_purchase_demand", False)
        )
        if cfg.lender_voucher_purchase_demand_enabled:
            cfg.lender_voucher_purchase_attempts_per_tick = max(
                0,
                int(
                    math.ceil(
                        int(getattr(args, "lender_voucher_purchase_attempts_per_tick", 5) or 0)
                        * factor
                    )
                ),
            )
        else:
            cfg.lender_voucher_purchase_attempts_per_tick = 0
        cfg.lender_voucher_purchase_consumer_share = max(
            0.0,
            min(1.0, float(getattr(args, "lender_voucher_purchase_consumer_share", 0.75) or 0.0)),
        )
        cfg.lender_voucher_purchase_inventory_share = max(
            0.0,
            float(getattr(args, "lender_voucher_purchase_inventory_share", 0.05) or 0.0),
        )
        cfg.lender_voucher_purchase_stable_budget_usd_per_tick = factor * max(
            0.0,
            float(
                getattr(args, "lender_voucher_purchase_stable_budget_usd_per_tick", 0.0)
                or 0.0
            ),
        )
        cfg.liquidity_mandate_mode = "community_deficit_then_lender"
        cfg.route_substitution_enabled = True
        cfg.route_substitution_max_alternatives = 3
        route_request_base = int(getattr(args, "_frontier_route_requests_per_tick", 1))
        cfg.random_route_requests_per_tick = max(1, int(round(route_request_base * math.sqrt(factor))))
        frontier_floor = int(math.ceil(int(getattr(args, "_frontier_swap_floor_per_tick", 0)) * factor))
        if frontier_floor > 0:
            cfg.swap_requests_budget_per_tick = max(
                int(round(100 * factor)),
                int(math.ceil(frontier_floor * 0.25)),
            )
            cfg.swap_attempts_max_per_pool = int(getattr(args, "_frontier_swap_attempts_max_per_pool", 2))
            cfg.swap_sustain_window_ticks = 0
            cfg.swap_sustain_floor_per_tick = frontier_floor
            cfg.swap_sustain_max_extra_attempts = max(
                int(math.ceil(int(getattr(args, "_frontier_swap_sustain_max_extra_attempts", 600)) * factor)),
                int(math.ceil(frontier_floor * 1.50)),
            )
            cfg.swap_sustain_max_rounds = int(getattr(args, "_frontier_swap_sustain_max_rounds", 2))
            cfg.swap_sustain_attempts_per_missing_swap = float(
                getattr(args, "_frontier_swap_sustain_attempts_per_missing_swap", 1.0)
            )
        else:
            cfg.swap_requests_budget_per_tick = max(100, int(round(100 * factor)))
        cfg.noam_topk_pools_per_asset = max(cfg.noam_topk_pools_per_asset, int(round(16 * math.sqrt(factor))))
        cfg.noam_topm_out_per_pool = max(cfg.noam_topm_out_per_pool, int(round(16 * math.sqrt(factor))))
        cfg.noam_beam_width = max(cfg.noam_beam_width, int(round(40 * math.sqrt(factor))))
        cfg.noam_clearing_budget_usd = float(cfg.noam_clearing_budget_usd) * factor
        if factor <= 1.0 + 1e-9:
            cfg.max_hops = 2
            cfg.noam_max_hops = 2
            cfg.noam_overlay_enabled = False
            cfg.noam_clearing_enabled = False
        else:
            cfg.p_offer_overlap = min(0.95, cfg.p_offer_overlap + 0.06 * math.log2(factor))
            cfg.p_want_overlap = min(0.97, cfg.p_want_overlap + 0.04 * math.log2(factor))
            cfg.desired_assets_growth_per_asset = min(0.50, cfg.desired_assets_growth_per_asset + 0.05 * math.log2(factor))
        historical_cash_backing_total = float(
            getattr(args, "_frontier_historical_cash_backing_total_usd", 0.0)
        )
        if historical_cash_backing_total > 0.0:
            cfg.historical_stable_backing_tick = 1
            cfg.historical_stable_backing_total_usd = historical_cash_backing_total * factor
            cfg.historical_stable_backing_roles = ("lender",)
        historical_voucher_backing_total = float(
            getattr(args, "_frontier_historical_voucher_backing_total_usd", 0.0)
        )
        if historical_voucher_backing_total > 0.0:
            cfg.historical_voucher_backing_tick = 1
            cfg.historical_voucher_backing_total_usd = historical_voucher_backing_total * factor
        cfg.calibration_profile = "bond_issuer_frontier"
    elif scenario == "stress_weak_pool_repayment":
        cfg.initial_liquidity_providers = 1
        cfg.lp_initial_stable_mean = 400_000.0
        cfg.base_redeem_prob = 0.55
        cfg.incident_base_rate = 0.02
        cfg.calibration_profile = "weak_pool_heavy"
    elif scenario == "stress_cash_stable_leakage":
        cfg.initial_liquidity_providers = 1
        cfg.lp_initial_stable_mean = 400_000.0
        cfg.producer_offramp_rate_per_month = 0.12
        cfg.consumer_offramp_rate_per_month = 0.12
        cfg.stable_outflow_rate = 0.08
        cfg.calibration_profile = "cash_leakage"
    elif scenario == "stress_high_coupon":
        cfg.initial_liquidity_providers = 1
        cfg.lp_initial_stable_mean = 400_000.0
        cfg.bond_coupon_target_annual = max(float(coupon), 0.12)
        cfg.calibration_profile = "high_coupon"
    elif scenario == "stress_low_fee_conversion":
        cfg.initial_liquidity_providers = 1
        cfg.lp_initial_stable_mean = 400_000.0
        cfg.pool_fee_rate = 0.01
        cfg.cash_conversion_slippage_bps = 250.0
        cfg.cash_conversion_max_usd_per_epoch = 1_000.0
        cfg.calibration_profile = "low_fee_conversion"
    elif scenario == "stress_governance_diversion":
        cfg.initial_liquidity_providers = 1
        cfg.lp_initial_stable_mean = 400_000.0
        cfg.core_ops_budget_usd = 100_000.0
        cfg.liquidity_mandate_share = 0.10
        cfg.bond_fee_service_share = 0.50
        cfg.sclc_fee_access_share = 0.25
        cfg.calibration_profile = "governance_diversion"
    elif scenario == "stress_liquidity_concentration":
        cfg.initial_liquidity_providers = 1
        cfg.initial_lenders = 1
        cfg.p_lender = 0.05
        cfg.p_producer = 0.70
        cfg.p_consumer = 0.25
        cfg.liquidity_mandate_mode = "lender_liquidity"
        cfg.calibration_profile = "liquidity_concentration"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return cfg


def active_pool_ids(engine: SimulationEngine, *, include_clc: bool = False) -> set[str]:
    ids = {
        pid
        for pid, pool in engine.pools.items()
        if not pool.policy.system_pool
    }
    if include_clc and engine.clc_pool_id:
        ids.add(engine.clc_pool_id)
    return ids


def calibrated_lender_pool_ids(engine: SimulationEngine) -> set[str]:
    """Runtime pools corresponding to empirical community lending pools."""
    ids = {
        pid
        for pid, pool in engine.pools.items()
        if not pool.policy.system_pool and pool.policy.role == "lender"
    }
    return ids or active_pool_ids(engine)


def graph_metrics(nodes: set[str], edge_weights: dict[tuple[str, str], float], prefix: str) -> dict[str, float]:
    clean_edges = {
        edge: float(weight)
        for edge, weight in edge_weights.items()
        if edge[0] in nodes and edge[1] in nodes and edge[0] != edge[1] and weight > 0.0
    }
    adjacency = {node: set() for node in nodes}
    for (a, b), _ in clean_edges.items():
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    seen = set()
    component_sizes = []
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for nxt in adjacency.get(current, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        component_sizes.append(size)

    n = len(nodes)
    edge_count = len(clean_edges)
    total_weight = sum(clean_edges.values())
    top_share = max(clean_edges.values()) / total_weight if total_weight > 0.0 else 0.0
    hhi = (
        sum((weight / total_weight) ** 2 for weight in clean_edges.values())
        if total_weight > 0.0
        else 0.0
    )
    largest = max(component_sizes) if component_sizes else 0
    return {
        f"{prefix}_nodes": n,
        f"{prefix}_edges": edge_count,
        f"{prefix}_component_count": len(component_sizes),
        f"{prefix}_largest_component_share": largest / n if n else 0.0,
        f"{prefix}_avg_degree": (2.0 * edge_count / n) if n else 0.0,
        f"{prefix}_edge_top_share": top_share,
        f"{prefix}_edge_hhi": hhi,
    }


def potential_edges(engine: SimulationEngine, nodes: set[str]) -> dict[tuple[str, str], float]:
    edges: dict[tuple[str, str], float] = {}
    excluded_assets = {engine.cfg.stable_symbol}
    if engine.cfg.sclc_symbol:
        excluded_assets.add(engine.cfg.sclc_symbol)
    for asset_id, offerers in engine.offer_index.items():
        if asset_id in excluded_assets:
            continue
        accepters = engine.accept_index.get(asset_id, set())
        if not accepters:
            continue
        for offerer in offerers:
            if offerer not in nodes:
                continue
            for accepter in accepters:
                if accepter not in nodes or accepter == offerer:
                    continue
                edge = tuple(sorted((offerer, accepter)))
                edges[edge] = edges.get(edge, 0.0) + 1.0
    return edges


def actor_pool_id(actor: object) -> str | None:
    actor_text = str(actor or "")
    if not actor_text:
        return None
    if ":" in actor_text:
        return actor_text.split(":", 1)[1]
    if actor_text.startswith("pool_"):
        return actor_text
    return None


def update_realized_edges(
    engine: SimulationEngine,
    realized_edges: dict[tuple[str, str], float],
    pool_swap_counts: Counter[str],
    start_idx: int,
) -> int:
    events = list(engine.log.events)
    for event in events[start_idx:]:
        if event.event_type != "SWAP_EXECUTED":
            continue
        receipt = (event.meta or {}).get("receipt") or {}
        source = actor_pool_id(receipt.get("actor") or event.actor_id)
        target = receipt.get("pool_id") or event.pool_id
        if not source or not target or source == target:
            continue
        if source not in engine.pools or target not in engine.pools:
            continue
        if engine.pools[source].policy.system_pool or engine.pools[target].policy.system_pool:
            continue
        edge = tuple(sorted((source, target)))
        realized_edges[edge] = realized_edges.get(edge, 0.0) + 1.0
        pool_swap_counts[target] += 1
    return len(events)


def choose_tier(rng: random.Random, tier_probs: dict[str, float]) -> str:
    if not tier_probs:
        return "moderate"
    threshold = rng.random()
    total = 0.0
    for tier in ("strong", "moderate", "weak"):
        total += tier_probs.get(tier, 0.0)
        if threshold <= total:
            return tier
    return "weak"


def ensure_pool_tiers(
    engine: SimulationEngine,
    tiers: dict[str, str],
    rng: random.Random,
    calibration: Calibration,
    target_tier_counts: dict[str, int] | None = None,
) -> None:
    pool_ids = sorted(calibrated_lender_pool_ids(engine))
    if target_tier_counts:
        assigned_counts = Counter(tiers.get(pid) for pid in pool_ids if pid in tiers)
        deck = []
        for tier in TIER_ORDER:
            missing = max(0, int(target_tier_counts.get(tier, 0)) - int(assigned_counts.get(tier, 0)))
            deck.extend([tier] * missing)
        rng.shuffle(deck)
        for pid in pool_ids:
            if pid in tiers:
                continue
            tiers[pid] = deck.pop() if deck else choose_tier(rng, calibration.tier_probs)
        return

    for pid in pool_ids:
        tiers.setdefault(pid, choose_tier(rng, calibration.tier_probs))


def tier_mix(engine: SimulationEngine, tiers: dict[str, str]) -> dict[str, float]:
    counts = Counter(tiers.get(pid, "moderate") for pid in calibrated_lender_pool_ids(engine))
    total = sum(counts.values()) or 1
    return {tier: counts.get(tier, 0) / total for tier in ("strong", "moderate", "weak")}


def expected_coverage_rates(calibration: Calibration, mix: dict[str, float], profile: str) -> tuple[float, float]:
    if profile == "weak_pool_heavy":
        mix = {"strong": 0.10, "moderate": 0.20, "weak": 0.70}
    cash = sum(
        mix.get(tier, 0.0) * calibration.repayment_by_tier_asset.get((tier, "cash"), 0.0)
        for tier in ("strong", "moderate", "weak")
    )
    voucher = sum(
        mix.get(tier, 0.0) * calibration.voucher_coverage_by_tier.get(tier, 0.0)
        for tier in ("strong", "moderate", "weak")
    )
    if profile == "cash_leakage":
        cash *= 0.65
    return cash, voucher


def expected_report_exposure(
    calibration: Calibration,
    pool_swap_counts: Counter[str],
    top_n: int = 10,
) -> tuple[float, dict[str, float]]:
    return expected_report_exposure_from_counts(calibration, list(pool_swap_counts.values()), top_n=top_n)


def expected_report_exposure_from_counts(
    calibration: Calibration,
    counts: Iterable[float],
    top_n: int = 10,
) -> tuple[float, dict[str, float]]:
    by_activity = {}
    counts = list(counts)
    if not counts:
        counts = [0]
    for projection in calibration.impact_rows[:top_n]:
        total = 0.0
        for swaps in counts:
            predicted = math.exp(
                projection.intercept + projection.slope * math.log1p(max(0, swaps))
            ) - 1.0
            total += max(0.0, predicted)
        by_activity[projection.activity] = total
    return sum(by_activity.values()), by_activity


def issuer_schedule_due(
    *,
    gross_principal: float,
    coupon_annual: float,
    term_ticks: int,
    payment_stride_ticks: int,
    tick: int,
) -> dict[str, float]:
    if gross_principal <= 1e-9 or term_ticks <= 0:
        return {
            "periods_elapsed": 0.0,
            "total_periods": 0.0,
            "coupon_due": 0.0,
            "principal_due": 0.0,
        }
    stride = max(1, int(payment_stride_ticks or 1))
    total_periods = max(1, int(math.ceil(term_ticks / stride)))
    elapsed_tick = min(max(0, int(tick)), int(term_ticks))
    periods_elapsed = min(total_periods, elapsed_tick // stride)
    if elapsed_tick >= term_ticks:
        periods_elapsed = total_periods

    coupon_due = 0.0
    principal_due = 0.0
    principal_step = gross_principal / total_periods
    previous_tick = 0
    for period in range(1, periods_elapsed + 1):
        payment_tick = min(int(term_ticks), period * stride)
        duration = max(0, payment_tick - previous_tick)
        outstanding_start = max(0.0, gross_principal - principal_step * (period - 1))
        coupon_due += outstanding_start * max(0.0, coupon_annual) * (duration / YEAR_TICKS)
        principal_due += principal_step
        previous_tick = payment_tick
    return {
        "periods_elapsed": float(periods_elapsed),
        "total_periods": float(total_periods),
        "coupon_due": coupon_due,
        "principal_due": min(gross_principal, principal_due),
    }


def bond_metrics(latest: dict[str, object], cfg: ScenarioConfig, tick: int) -> dict[str, object]:
    raw_deployed = safe_float(latest.get("lp_injected_usd_total"))
    configured_gross = safe_float(getattr(cfg, "bond_gross_principal_usd", 0.0))
    principal = configured_gross if configured_gross > 1e-9 else raw_deployed
    configured_deployed = safe_float(getattr(cfg, "bond_deployed_principal_usd", 0.0))
    deployed_principal = configured_deployed if configured_deployed > 1e-9 else raw_deployed
    raw_returned = safe_float(latest.get("lp_returned_usd_total"))
    service_share = max(0.0, min(1.0, float(cfg.bond_fee_service_share or 0.0)))
    if str(getattr(cfg, "bond_return_mode", "") or "") == "issuer_cashflow":
        returned = raw_returned
    else:
        returned = raw_returned * service_share
    elapsed_years = max(tick / YEAR_TICKS, 1.0 / YEAR_TICKS)
    coupon = max(0.0, float(cfg.bond_coupon_target_annual or 0.0))
    reserve_share = max(0.0, min(0.95, safe_float(getattr(cfg, "issuer_reserve_share", 0.10))))
    reserve_initial = principal * reserve_share
    payment_stride = max(1, int(getattr(cfg, "issuer_payment_stride_ticks", 13) or 13))
    schedule = issuer_schedule_due(
        gross_principal=principal,
        coupon_annual=coupon,
        term_ticks=int(cfg.bond_term_ticks or 0),
        payment_stride_ticks=payment_stride,
        tick=tick,
    )
    coupon_due = schedule["coupon_due"]
    principal_due = schedule["principal_due"]
    scheduled_due = coupon_due + principal_due
    reserve_draw = min(reserve_initial, max(0.0, scheduled_due - returned))
    reserve_balance = max(0.0, reserve_initial - reserve_draw)
    actual_payment = min(scheduled_due, returned + reserve_draw)
    unpaid_scheduled_claim = max(0.0, scheduled_due - actual_payment)
    service_coverage = returned / scheduled_due if scheduled_due > 1e-9 else (999.0 if principal <= 1e-9 else 0.0)
    paid_coverage = actual_payment / scheduled_due if scheduled_due > 1e-9 else 1.0
    fee_service_reserved = safe_float(latest.get("fee_service_reserved_usd_total"))
    recovered_service_cash = max(
        returned,
        safe_float(latest.get("lender_recovered_stable_usd_total")) + fee_service_reserved,
    )
    service_cash_headroom = (
        recovered_service_cash / scheduled_due
        if scheduled_due > 1e-9
        else (999.0 if principal <= 1e-9 else 0.0)
    )
    lockbox_balance = safe_float(latest.get("bond_service_reserve_balance_usd"))
    sweepable_pending_recovered_stable = safe_float(
        latest.get("lender_recovered_stable_sweepable_pending_usd_total")
    )
    available_service_cash = returned + lockbox_balance + sweepable_pending_recovered_stable
    available_service_cash_headroom = (
        available_service_cash / scheduled_due
        if scheduled_due > 1e-9
        else (999.0 if principal <= 1e-9 else 0.0)
    )
    issuer_lockbox_surplus = max(0.0, returned + lockbox_balance - scheduled_due)
    issuer_excess_recovered_stable = max(0.0, sweepable_pending_recovered_stable)
    issuer_operating_surplus = max(0.0, available_service_cash - scheduled_due)
    issuer_fee_waterfall_ops = safe_float(latest.get("waterfall_ops_alloc_usd_total"))
    issuer_sustainability_resources = issuer_operating_surplus + issuer_fee_waterfall_ops
    issuer_operating_surplus_headroom = (
        issuer_operating_surplus / scheduled_due
        if scheduled_due > 1e-9
        else (999.0 if principal <= 1e-9 else 0.0)
    )
    issuer_sustainability_headroom = (
        issuer_sustainability_resources / scheduled_due
        if scheduled_due > 1e-9
        else (999.0 if principal <= 1e-9 else 0.0)
    )
    coupon_coverage = returned / coupon_due if coupon_due > 1e-9 else 1.0
    simple_yield = returned / principal if principal > 1e-9 else 0.0
    annualized_yield = simple_yield / elapsed_years if principal > 1e-9 else 0.0
    cagr = ((1.0 + simple_yield) ** (1.0 / elapsed_years) - 1.0) if principal > 1e-9 else 0.0
    principal_shortfall = unpaid_scheduled_claim if tick >= int(cfg.bond_term_ticks or 0) else 0.0
    return {
        "bond_principal_usd": principal,
        "bond_deployed_principal_usd": deployed_principal,
        "bond_raw_lp_returned_usd": raw_returned,
        "bond_cumulative_fee_return_usd": returned,
        "bond_net_after_principal_usd": returned - principal if principal > 1e-9 else 0.0,
        "bond_simple_fee_yield": simple_yield,
        "bond_annualized_fee_yield": annualized_yield,
        "bond_cagr_fee_yield": cagr,
        "bond_coupon_due_usd": coupon_due,
        "bond_coupon_coverage_ratio": coupon_coverage,
        "bond_coupon_shortfall_usd": unpaid_scheduled_claim,
        "bond_principal_shortfall_at_term_usd": principal_shortfall,
        "bond_payback_ratio": returned / principal if principal > 1e-9 else 0.0,
        "issuer_deployed_principal_usd": deployed_principal,
        "issuer_reserve_initial_usd": reserve_initial,
        "issuer_reserve_balance_usd": reserve_balance,
        "issuer_eligible_fee_service_inflow_usd": returned,
        "issuer_fee_service_reserved_usd": fee_service_reserved,
        "issuer_scheduled_coupon_due_usd": coupon_due,
        "issuer_scheduled_principal_due_usd": principal_due,
        "issuer_scheduled_debt_service_due_usd": scheduled_due,
        "issuer_actual_bondholder_payment_usd": actual_payment,
        "issuer_reserve_draw_usd": reserve_draw,
        "issuer_unpaid_scheduled_claim_usd": unpaid_scheduled_claim,
        "issuer_service_coverage_ratio": service_coverage,
        "issuer_paid_coverage_ratio": paid_coverage,
        "issuer_service_cash_headroom_ratio": service_cash_headroom,
        "issuer_available_service_cash_usd": available_service_cash,
        "issuer_available_service_cash_headroom_ratio": available_service_cash_headroom,
        "issuer_lockbox_surplus_usd": issuer_lockbox_surplus,
        "issuer_excess_recovered_stable_usd": issuer_excess_recovered_stable,
        "issuer_operating_surplus_usd": issuer_operating_surplus,
        "issuer_operating_surplus_headroom_ratio": issuer_operating_surplus_headroom,
        "issuer_fee_waterfall_ops_usd": issuer_fee_waterfall_ops,
        "issuer_sustainability_resources_usd": issuer_sustainability_resources,
        "issuer_sustainability_headroom_ratio": issuer_sustainability_headroom,
        "lender_recovered_stable_sweepable_pending_usd_total": sweepable_pending_recovered_stable,
        "bond_service_lockbox_target_usd": safe_float(latest.get("bond_service_lockbox_target_usd")),
        "bond_service_lockbox_coverage_ratio": safe_float(
            latest.get("bond_service_lockbox_coverage_ratio")
        ),
        "bond_service_lockbox_mode": str(latest.get("bond_service_lockbox_mode", "")),
        "issuer_payment_periods_elapsed": schedule["periods_elapsed"],
        "issuer_payment_periods_total": schedule["total_periods"],
        "issuer_payment_stride_ticks": float(payment_stride),
    }


def route_success_rate(latest: dict[str, object]) -> float:
    found = safe_float(latest.get("route_found_tick"))
    failed = safe_float(latest.get("route_failed_tick"))
    denom = found + failed
    return found / denom if denom > 0.0 else 0.0


def maybe_print_run_progress(
    *,
    scenario: str,
    run_index: int,
    tick: int,
    args: argparse.Namespace,
    started_at: float,
    latest: dict[str, object],
) -> None:
    progress_stride = max(0, int(getattr(args, "progress_stride", 0) or 0))
    total_ticks = max(1, int(args.ticks))
    if progress_stride <= 0:
        return
    if tick != 1 and tick != total_ticks and tick % progress_stride != 0:
        return

    run_pct = 100.0 * tick / total_ticks
    run_position = safe_float(getattr(args, "_progress_run_position", 0.0))
    total_runs = safe_float(getattr(args, "_progress_total_runs", 0.0))
    overall_text = ""
    if total_runs > 0.0 and run_position > 0.0:
        overall_pct = 100.0 * ((run_position - 1.0) + (tick / total_ticks)) / total_runs
        overall_text = f" overall={overall_pct:5.1f}%"
    elapsed = time.monotonic() - started_at
    print(
        "[progress] scenario={scenario} run={run}/{runs} tick={tick}/{ticks} "
        "run_pct={run_pct:5.1f}%{overall} elapsed={elapsed:6.1f}s "
        "tx_tick={tx} route_cum={route}".format(
            scenario=scenario,
            run=run_index,
            runs=int(total_runs) if total_runs > 0.0 else "?",
            tick=tick,
            ticks=total_ticks,
            run_pct=run_pct,
            overall=overall_text,
            elapsed=elapsed,
            tx=int(safe_float(latest.get("transactions_per_tick"))),
            route=f"{safe_float(latest.get('route_found_tick')):.0f}/{safe_float(latest.get('route_failed_tick')):.0f}",
        ),
        flush=True,
    )


def unit_value_diagnostics(engine: SimulationEngine, args: argparse.Namespace) -> dict[str, object]:
    producer = next((agent for agent in engine.agents.values() if agent.role == "producer"), None)
    consumer = next((agent for agent in engine.agents.values() if agent.role == "consumer"), None)

    def voucher_value(agent) -> tuple[str, float]:
        if agent is None or getattr(agent, "role", "") != "producer":
            return "", 0.0
        voucher_id = agent.voucher_spec.voucher_id
        if voucher_id not in engine.factory.voucher_specs:
            return "", 0.0
        pool = engine.pools.get(agent.pool_id)
        value = pool.values.get_value(voucher_id) if pool is not None else 0.0
        if value <= 0.0:
            value = engine._default_asset_value(voucher_id)
        return voucher_id, value

    producer_voucher_id, producer_value = voucher_value(producer)
    consumer_voucher_id, consumer_value = voucher_value(consumer)
    active_stable_value = sum(
        pool.vault.get(engine.cfg.stable_symbol)
        for pool in engine.pools.values()
        if not pool.policy.system_pool
    )
    active_voucher_value = sum(
        engine._pool_voucher_value_usd(pool)
        for pool in engine.pools.values()
        if not pool.policy.system_pool
    )
    active_total = active_stable_value + active_voucher_value
    return {
        "stable_symbol": engine.cfg.stable_symbol,
        "stable_unit_value_usd": 1.0,
        "kes_per_usd": getattr(args, "_calibration_kes_per_usd", 0.0),
        "voucher_unit_value_usd": engine.cfg.voucher_unit_value_usd,
        "sample_producer_voucher_id": producer_voucher_id,
        "sample_producer_voucher_value_usd": producer_value,
        "sample_consumer_voucher_id": consumer_voucher_id,
        "sample_consumer_voucher_value_usd": consumer_value,
        "initial_active_stable_inventory_usd": active_stable_value,
        "initial_active_voucher_inventory_usd": active_voucher_value,
        "initial_active_stable_inventory_share": active_stable_value / max(1e-9, active_total),
    }


def maybe_print_unit_diagnostics(
    scenario: str,
    run_index: int,
    seed: int,
    diagnostics: dict[str, object],
    args: argparse.Namespace,
) -> None:
    if not bool(getattr(args, "unit_diagnostics", True)) or run_index != 1:
        return
    print(
        "[unit-diagnostic] scenario={scenario} run={run} seed={seed} "
        "stable={stable}@1.0USD kes_per_usd={kes:.6f} voucher_unit_value_usd={voucher:.12f} "
        "producer_voucher={producer} producer_value_usd={producer_value:.12f} "
        "consumer_voucher={consumer} consumer_value_usd={consumer_value:.12f} "
        "initial_active_stable_usd={stable_inventory:.6f} "
        "initial_active_voucher_usd={voucher_inventory:.6f} "
        "initial_active_stable_share={stable_share:.6f}".format(
            scenario=scenario,
            run=run_index,
            seed=seed,
            stable=diagnostics.get("stable_symbol", ""),
            kes=safe_float(diagnostics.get("kes_per_usd")),
            voucher=safe_float(diagnostics.get("voucher_unit_value_usd")),
            producer=diagnostics.get("sample_producer_voucher_id", ""),
            producer_value=safe_float(diagnostics.get("sample_producer_voucher_value_usd")),
            consumer=diagnostics.get("sample_consumer_voucher_id", ""),
            consumer_value=safe_float(diagnostics.get("sample_consumer_voucher_value_usd")),
            stable_inventory=safe_float(diagnostics.get("initial_active_stable_inventory_usd")),
            voucher_inventory=safe_float(diagnostics.get("initial_active_voucher_inventory_usd")),
            stable_share=safe_float(diagnostics.get("initial_active_stable_inventory_share")),
        ),
        flush=True,
    )


def run_one(
    scenario: str,
    coupon: float,
    term_ticks: int,
    run_index: int,
    seed: int,
    args: argparse.Namespace,
    calibration: Calibration,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    cfg = scenario_config(scenario, coupon, term_ticks, args)
    engine = SimulationEngine(cfg=cfg, seed=seed)
    unit_diagnostics = unit_value_diagnostics(engine, args)
    maybe_print_unit_diagnostics(scenario, run_index, seed, unit_diagnostics, args)
    tier_rng = random.Random(seed + 7919)
    pool_tiers: dict[str, str] = {}
    realized_edges: dict[tuple[str, str], float] = {}
    pool_swap_counts: Counter[str] = Counter()
    event_idx = 0
    bond_rows = []
    network_rows = []
    impact_latest: dict[str, float] = {}
    cumulative = Counter()
    cumulative_float: dict[str, float] = defaultdict(float)
    target_tier_counts = getattr(args, "_target_tier_counts", None)
    run_started_at = time.monotonic()

    for _ in range(int(args.ticks)):
        engine.step(1)
        ensure_pool_tiers(engine, pool_tiers, tier_rng, calibration, target_tier_counts)
        event_idx = update_realized_edges(engine, realized_edges, pool_swap_counts, event_idx)
        latest = engine.metrics.network_rows[-1]
        tick = int(latest["tick"])
        maybe_print_run_progress(
            scenario=scenario,
            run_index=run_index,
            tick=tick,
            args=args,
            started_at=run_started_at,
            latest=latest,
        )
        found_tick = safe_float(latest.get("route_found_tick"))
        failed_tick = safe_float(latest.get("route_failed_tick"))
        cumulative["route_found"] += int(found_tick)
        cumulative["route_failed"] += int(failed_tick)
        cumulative["route_fixed_found"] += int(safe_float(latest.get("route_fixed_found_tick")))
        cumulative["route_fixed_failed"] += int(safe_float(latest.get("route_fixed_failed_tick")))
        cumulative["route_substitution_found"] += int(safe_float(latest.get("route_substitution_found_tick")))
        cumulative["route_substitution_failed"] += int(safe_float(latest.get("route_substitution_failed_tick")))
        cumulative["transactions"] += int(safe_float(latest.get("transactions_per_tick")))
        for metric in (
            "swap_volume_usd_tick",
            "swap_volume_usd_to_vchr_tick",
            "swap_volume_vchr_to_usd_tick",
            "swap_volume_vchr_to_vchr_tick",
            "swap_stable_flow_value_tick",
            "swap_voucher_flow_value_tick",
            "swap_stable_net_flow_value_tick",
            "swap_voucher_net_flow_value_tick",
            "route_source_stable_net_flow_value_tick",
            "route_source_voucher_net_flow_value_tick",
            "swap_count_usd_to_vchr_tick",
            "swap_count_vchr_to_usd_tick",
            "swap_count_vchr_to_vchr_tick",
            "noam_routing_swaps_tick",
            "noam_clearing_swaps_tick",
            "noam_routing_volume_usd_tick",
            "noam_clearing_volume_usd_tick",
            "noam_routing_fee_usd_tick",
            "noam_clearing_fee_usd_tick",
            "noam_routing_stable_fee_usd_tick",
            "noam_routing_voucher_fee_usd_tick",
            "noam_clearing_stable_fee_usd_tick",
            "noam_clearing_voucher_fee_usd_tick",
            "noam_clearing_cycles_attempted_tick",
            "noam_clearing_cycles_executed_tick",
            "noam_clearing_cycle_value_usd_tick",
            "ordinary_swap_count_tick",
            "ordinary_swap_volume_usd_tick",
            "ordinary_stable_source_swap_count_tick",
            "ordinary_stable_source_swap_volume_usd_tick",
            "ordinary_voucher_source_swap_count_tick",
            "ordinary_voucher_source_swap_volume_usd_tick",
            "loan_route_swap_count_tick",
            "loan_route_swap_volume_usd_tick",
            "voucher_loan_route_swap_count_tick",
            "voucher_loan_route_swap_volume_usd_tick",
            "repayment_route_swap_count_tick",
            "repayment_route_swap_volume_usd_tick",
            "loan_backfill_swap_count_tick",
            "loan_backfill_swap_volume_usd_tick",
            "productive_boosted_voucher_swap_count_tick",
            "productive_boosted_voucher_swap_volume_usd_tick",
            "voucher_loan_boosted_voucher_swap_count_tick",
            "voucher_loan_boosted_voucher_swap_volume_usd_tick",
            "ordinary_stable_spend_protected_skip_count_tick",
            "ordinary_stable_spend_protected_skip_value_usd_tick",
            "repayment_volume_usd",
            "loan_issuance_volume_usd",
            "producer_debt_originated_usd_tick",
            "lender_recovered_stable_usd_tick",
            "lender_recovered_stable_borrower_regular_usd_tick",
            "lender_recovered_stable_borrower_maturity_usd_tick",
            "lender_recovered_stable_consumer_purchase_usd_tick",
            "lender_recovered_stable_third_party_purchase_usd_tick",
            "lender_recovered_stable_other_usd_tick",
            "stable_onramp_usd_tick",
            "stable_offramp_usd_tick",
            "producer_deposit_stable_usd_tick",
            "producer_deposit_voucher_usd_tick",
            "productive_credit_inflow_usd_tick",
            "productive_credit_stable_retained_usd_tick",
            "productive_credit_voucher_deposit_usd_tick",
            "productive_credit_voucher_deposit_cap_clipped_usd_tick",
            "producer_debt_cash_service_due_usd_tick",
            "producer_debt_cash_service_paid_usd_tick",
            "producer_debt_matured_usd_tick",
            "producer_debt_repaid_usd_tick",
            "producer_debt_repaid_regular_usd_tick",
            "producer_debt_repaid_maturity_usd_tick",
            "producer_debt_stable_recovered_usd_tick",
            "producer_debt_consumer_stable_purchase_usd_tick",
            "producer_debt_third_party_stable_purchase_usd_tick",
            "producer_debt_defaulted_usd_tick",
            "producer_debt_closed_by_circulation_usd_tick",
            "producer_debt_closed_by_voucher_swap_usd_tick",
            "producer_debt_closed_not_held_at_maturity_usd_tick",
            "producer_loan_attempts_tick",
            "producer_loan_no_lender_tick",
            "producer_loan_no_inventory_tick",
            "producer_loan_zero_amount_tick",
            "producer_loan_route_found_tick",
            "producer_loan_route_failed_tick",
            "producer_loan_backfill_attempts_tick",
            "producer_loan_backfill_executed_tick",
            "producer_loan_executed_tick",
            "producer_loan_execution_failed_tick",
            "producer_loan_sampled_usd_tick",
            "producer_loan_attempted_usd_tick",
            "producer_loan_executed_usd_tick",
            "producer_loan_clipped_inventory_usd_tick",
            "producer_loan_clipped_lender_cap_usd_tick",
            "producer_loan_clipped_lender_remaining_usd_tick",
            "producer_loan_clipped_lender_stable_usd_tick",
            "producer_loan_clipped_combined_lender_usd_tick",
            "producer_loan_lender_collateral_cap_usd_tick",
            "producer_loan_lender_remaining_cap_usd_tick",
            "producer_loan_lender_stable_available_usd_tick",
            "producer_voucher_loan_attempts_tick",
            "producer_voucher_loan_no_target_tick",
            "producer_voucher_loan_no_inventory_tick",
            "producer_voucher_loan_zero_amount_tick",
            "producer_voucher_loan_route_found_tick",
            "producer_voucher_loan_route_failed_tick",
            "producer_voucher_loan_executed_tick",
            "producer_voucher_loan_execution_failed_tick",
            "producer_voucher_loan_attempted_usd_tick",
            "producer_voucher_loan_executed_usd_tick",
            "producer_voucher_loan_clipped_lender_cap_usd_tick",
            "producer_voucher_loan_clipped_lender_remaining_usd_tick",
            "producer_primary_voucher_loan_attempts_tick",
            "producer_primary_voucher_loan_executed_tick",
            "voucher_purchase_attempts_tick",
            "consumer_voucher_purchase_attempts_tick",
            "consumer_voucher_purchase_success_tick",
            "consumer_voucher_purchase_no_stable_tick",
            "consumer_voucher_purchase_reserve_protected_tick",
            "consumer_voucher_purchase_no_route_tick",
            "consumer_voucher_purchase_no_target_tick",
            "consumer_voucher_purchase_stable_spent_usd_tick",
            "consumer_voucher_purchase_voucher_value_acquired_usd_tick",
            "third_party_voucher_purchase_attempts_tick",
            "third_party_voucher_purchase_success_tick",
            "third_party_voucher_purchase_no_stable_tick",
            "third_party_voucher_purchase_reserve_protected_tick",
            "third_party_voucher_purchase_no_route_tick",
            "third_party_voucher_purchase_no_target_tick",
            "third_party_voucher_purchase_stable_spent_usd_tick",
            "third_party_voucher_purchase_voucher_value_acquired_usd_tick",
            "lender_voucher_purchase_stable_budget_onramp_usd_tick",
            "consumer_voucher_purchase_stable_budget_onramp_usd_tick",
            "third_party_voucher_purchase_stable_budget_onramp_usd_tick",
            "fee_conversion_attempted_usd_tick",
            "fee_conversion_success_usd_tick",
            "fee_conversion_failed_usd_tick",
            "fee_service_reserved_usd_tick",
            "fee_service_stable_reserved_usd_tick",
            "fee_service_converted_voucher_reserved_usd_tick",
            "bond_service_reserved_usd_tick",
            "bond_service_paid_from_reserve_usd_tick",
            "quarterly_clearing_usd_tick",
        ):
            cumulative_float[metric] += safe_float(latest.get(metric))
        analysis_stride = max(1, int(args.analysis_stride))
        if tick % analysis_stride != 0 and tick != int(args.ticks):
            continue
        nodes = active_pool_ids(engine)
        potential = potential_edges(engine, nodes)
        potential_metrics = graph_metrics(nodes, potential, "potential")
        realized_metrics = graph_metrics(nodes, realized_edges, "realized")
        mix = tier_mix(engine, pool_tiers)
        expected_cash_cov, expected_voucher_cov = expected_coverage_rates(
            calibration, mix, cfg.calibration_profile
        )
        expected_reports, impact_latest = expected_report_exposure(calibration, pool_swap_counts)
        bmetrics = bond_metrics(latest, cfg, tick)
        route_attempts_total = cumulative["route_found"] + cumulative["route_failed"]
        route_success_total = cumulative["route_found"] / route_attempts_total if route_attempts_total else 0.0
        fixed_route_attempts_total = cumulative["route_fixed_found"] + cumulative["route_fixed_failed"]
        fixed_route_success_total = (
            cumulative["route_fixed_found"] / fixed_route_attempts_total
            if fixed_route_attempts_total
            else 0.0
        )
        substitution_route_attempts_total = (
            cumulative["route_substitution_found"] + cumulative["route_substitution_failed"]
        )
        substitution_route_success_total = (
            cumulative["route_substitution_found"] / substitution_route_attempts_total
            if substitution_route_attempts_total
            else 0.0
        )
        stress_ratio = safe_float(latest.get("pools_under_stable_reserve")) / max(
            1.0, safe_float(latest.get("num_pools"), 1.0)
        )
        producer_stress_ratio = safe_float(latest.get("producer_stable_reserve_stress_ratio"))
        consumer_stress_ratio = safe_float(latest.get("consumer_stable_reserve_stress_ratio"))
        lender_stress_ratio = safe_float(latest.get("lender_stable_reserve_stress_ratio"))
        community_stress_ratio = safe_float(latest.get("community_stable_reserve_stress_ratio"))
        leakage_denom = (
            safe_float(latest.get("stable_onramp_usd_tick"))
            + safe_float(bmetrics.get("bond_principal_usd"))
            + safe_float(latest.get("stable_total_in_pools"))
        )
        leakage_ratio = safe_float(latest.get("stable_offramp_usd_tick")) / max(1e-9, leakage_denom)
        cumulative_swap_stable_flow = cumulative_float["swap_stable_flow_value_tick"]
        cumulative_swap_voucher_flow = cumulative_float["swap_voucher_flow_value_tick"]
        cumulative_swap_gross_flow = cumulative_swap_stable_flow + cumulative_swap_voucher_flow
        latest_engine_gross_stable_flow = (
            safe_float(latest.get("swap_stable_flow_value_tick"))
            + safe_float(latest.get("stable_onramp_usd_tick"))
            + safe_float(latest.get("stable_offramp_usd_tick"))
            + safe_float(latest.get("producer_deposit_stable_usd_tick"))
        )
        latest_engine_gross_voucher_flow = (
            safe_float(latest.get("swap_voucher_flow_value_tick"))
            + safe_float(latest.get("producer_deposit_voucher_usd_tick"))
        )
        latest_engine_gross_flow = latest_engine_gross_stable_flow + latest_engine_gross_voucher_flow
        cumulative_engine_gross_stable_flow = (
            cumulative_swap_stable_flow
            + cumulative_float["stable_onramp_usd_tick"]
            + cumulative_float["stable_offramp_usd_tick"]
            + cumulative_float["producer_deposit_stable_usd_tick"]
        )
        cumulative_engine_gross_voucher_flow = (
            cumulative_swap_voucher_flow
            + cumulative_float["producer_deposit_voucher_usd_tick"]
        )
        cumulative_engine_gross_flow = (
            cumulative_engine_gross_stable_flow + cumulative_engine_gross_voucher_flow
        )
        cumulative_stable_flow_net = (
            cumulative_float["stable_onramp_usd_tick"]
            - cumulative_float["stable_offramp_usd_tick"]
            + cumulative_float["swap_stable_net_flow_value_tick"]
            + cumulative_float["route_source_stable_net_flow_value_tick"]
        )
        cumulative_voucher_flow_net = (
            cumulative_float["producer_deposit_voucher_usd_tick"]
            + cumulative_float["swap_voucher_net_flow_value_tick"]
            + cumulative_float["route_source_voucher_net_flow_value_tick"]
        )
        cumulative_stable_flow_positive = max(0.0, cumulative_stable_flow_net)
        cumulative_voucher_flow_positive = max(0.0, cumulative_voucher_flow_net)
        cumulative_positive_flow_total = cumulative_stable_flow_positive + cumulative_voucher_flow_positive
        cumulative_positive_flow_stable_share = (
            cumulative_stable_flow_positive / max(1e-9, cumulative_positive_flow_total)
        )
        control_prefix = "frontier" if scenario == "bond_issuer_frontier" else "validation"
        configured_voucher_deposit_share = safe_float(
            getattr(args, f"_{control_prefix}_productive_credit_voucher_deposit_share", 0.0)
        )

        common = {
            "scenario": scenario,
            "run": run_index,
            "seed": seed,
            "network_scale": getattr(args, "_current_network_scale", "current"),
            "network_scale_factor": getattr(args, "_current_scale_factor", 1.0),
            "principal_ratio": getattr(args, "_current_principal_ratio", 0.0),
            "bond_fee_service_share": getattr(args, "_current_bond_fee_service_share", cfg.bond_fee_service_share),
            "certification_policy": getattr(args, "_current_certification_policy", ""),
            "certified_pool_count": getattr(args, "_current_certified_pool_count", 0.0),
            "certified_backing_capacity_usd": getattr(args, "_current_certified_capacity_usd", 0.0),
            "kes_per_usd": getattr(args, "_calibration_kes_per_usd", 0.0),
            "voucher_unit_value_usd": getattr(args, "_voucher_unit_value_usd", 1.0),
            "configured_lender_pools": getattr(args, "_current_initial_lenders", 0),
            "configured_producer_wallets": getattr(args, "_current_initial_producers", 0),
            "configured_consumer_wallets": getattr(args, "_current_initial_consumers", 0),
            "empirical_lender_pool_templates": getattr(args, "_current_empirical_lender_pool_templates", 0),
            "empirical_unique_producer_vouchers": getattr(
                args, "_current_empirical_unique_producer_vouchers", 0
            ),
            "empirical_accepted_voucher_members": getattr(
                args, "_current_empirical_accepted_voucher_members", 0
            ),
            "empirical_accepted_voucher_member_slots": getattr(
                args, "_current_empirical_accepted_voucher_members", 0
            ),
            "empirical_active_interactors": getattr(args, "_current_empirical_active_interactors", 0),
            "empirical_unique_pool_interactor_addresses": getattr(
                args, "_current_empirical_unique_pool_interactor_addresses", 0
            ),
            "empirical_stable_external_nonproducer_users": getattr(
                args, "_current_empirical_stable_external_nonproducer_users", 0
            ),
            "empirical_producer_stable_users": getattr(
                args, "_current_empirical_producer_stable_users", 0
            ),
            "configured_producer_stable_deposit_rate_per_month": safe_float(
                getattr(args, f"_{control_prefix}_producer_stable_deposit_rate_per_month", 0.0)
            ),
            "configured_producer_voucher_deposit_rate_per_month": safe_float(
                getattr(args, f"_{control_prefix}_producer_voucher_deposit_rate_per_month", 0.0)
            ),
            "configured_productive_credit_stable_retained_share": max(
                0.0, 1.0 - configured_voucher_deposit_share
            ),
            "configured_productive_credit_voucher_deposit_share": configured_voucher_deposit_share,
            "configured_productive_credit_voucher_deposit_cap_rate_per_month": safe_float(
                getattr(
                    args,
                    f"_{control_prefix}_productive_credit_voucher_deposit_cap_rate_per_month",
                    0.0,
                )
            ),
            "configured_productive_credit_voucher_source_weight_boost": safe_float(
                getattr(
                    args,
                    f"_{control_prefix}_productive_credit_voucher_source_weight_boost",
                    0.0,
                )
            ),
            "configured_productive_credit_voucher_source_size_multiplier": safe_float(
                getattr(
                    args,
                    f"_{control_prefix}_productive_credit_voucher_source_size_multiplier",
                    1.0,
                )
            ),
            "configured_producer_debt_contract_service_margin_rate": safe_float(
                getattr(cfg, "producer_debt_contract_service_margin_rate", 0.0)
            ),
            "configured_producer_stable_debt_contract_service_margin_rate": safe_float(
                getattr(cfg, "producer_stable_debt_contract_service_margin_rate", 0.0)
            ),
            "configured_producer_voucher_debt_contract_service_margin_rate": safe_float(
                getattr(cfg, "producer_voucher_debt_contract_service_margin_rate", 0.0)
            ),
            "configured_producer_primary_voucher_borrowing_enabled": int(
                bool(getattr(cfg, "producer_primary_voucher_borrowing_enabled", False))
            ),
            "configured_producer_primary_voucher_borrowing_attempt_share": safe_float(
                getattr(cfg, "producer_primary_voucher_borrowing_attempt_share", 0.0)
            ),
            "configured_lender_voucher_purchase_demand_enabled": int(
                bool(getattr(cfg, "lender_voucher_purchase_demand_enabled", False))
            ),
            "configured_lender_voucher_purchase_attempts_per_tick": int(
                getattr(cfg, "lender_voucher_purchase_attempts_per_tick", 0) or 0
            ),
            "configured_lender_voucher_purchase_consumer_share": safe_float(
                getattr(cfg, "lender_voucher_purchase_consumer_share", 0.0)
            ),
            "configured_lender_voucher_purchase_inventory_share": safe_float(
                getattr(cfg, "lender_voucher_purchase_inventory_share", 0.0)
            ),
            "configured_lender_voucher_purchase_stable_budget_usd_per_tick": safe_float(
                getattr(cfg, "lender_voucher_purchase_stable_budget_usd_per_tick", 0.0)
            ),
            "configured_producer_voucher_overlap_mode": getattr(
                cfg, "producer_voucher_overlap_mode", "single_lender"
            ),
            "configured_empirical_voucher_pool_overlap_multi_pool_share": safe_float(
                getattr(args, f"_{control_prefix}_voucher_pool_overlap_multi_pool_share", 0.0)
            ),
            "configured_empirical_voucher_pool_overlap_degree_p50": safe_float(
                getattr(args, f"_{control_prefix}_voucher_pool_overlap_degree_p50", 0.0)
            ),
            "configured_empirical_voucher_pool_overlap_degree_p90": safe_float(
                getattr(args, f"_{control_prefix}_voucher_pool_overlap_degree_p90", 0.0)
            ),
            "stable_symbol": unit_diagnostics.get("stable_symbol", ""),
            "stable_unit_value_usd": unit_diagnostics.get("stable_unit_value_usd", 1.0),
            "sample_producer_voucher_id": unit_diagnostics.get("sample_producer_voucher_id", ""),
            "sample_producer_voucher_value_usd": unit_diagnostics.get(
                "sample_producer_voucher_value_usd", 0.0
            ),
            "sample_consumer_voucher_id": unit_diagnostics.get("sample_consumer_voucher_id", ""),
            "sample_consumer_voucher_value_usd": unit_diagnostics.get(
                "sample_consumer_voucher_value_usd", 0.0
            ),
            "initial_active_stable_inventory_usd": unit_diagnostics.get(
                "initial_active_stable_inventory_usd", 0.0
            ),
            "initial_active_voucher_inventory_usd": unit_diagnostics.get(
                "initial_active_voucher_inventory_usd", 0.0
            ),
            "initial_active_stable_inventory_share": unit_diagnostics.get(
                "initial_active_stable_inventory_share", 0.0
            ),
            "coupon_target_annual": float(cfg.bond_coupon_target_annual),
            "bond_term_ticks": int(cfg.bond_term_ticks),
            "tick": tick,
            "num_pools": latest.get("num_pools", 0),
            "num_assets": latest.get("num_assets", 0),
            "pool_total_value_usd": latest.get("pool_total_value_usd", 0.0),
            "stable_value_total_in_active_pools": latest.get("stable_value_total_in_active_pools", 0.0),
            "voucher_value_total_in_active_pools": latest.get("voucher_value_total_in_active_pools", 0.0),
            "stable_value_share_in_active_pools": latest.get("stable_value_share_in_active_pools", 0.0),
            "voucher_value_share_in_active_pools": latest.get("voucher_value_share_in_active_pools", 0.0),
            "stable_to_voucher_value_ratio_in_active_pools": latest.get(
                "stable_to_voucher_value_ratio_in_active_pools", 0.0
            ),
            "transactions_per_tick": latest.get("transactions_per_tick", 0),
            "transactions_total": cumulative["transactions"],
            "swap_volume_usd_tick": latest.get("swap_volume_usd_tick", 0.0),
            "swap_volume_usd_total": cumulative_float["swap_volume_usd_tick"],
            "swap_volume_usd_to_vchr_tick": latest.get("swap_volume_usd_to_vchr_tick", 0.0),
            "swap_volume_usd_to_vchr_total": cumulative_float["swap_volume_usd_to_vchr_tick"],
            "swap_volume_vchr_to_usd_tick": latest.get("swap_volume_vchr_to_usd_tick", 0.0),
            "swap_volume_vchr_to_usd_total": cumulative_float["swap_volume_vchr_to_usd_tick"],
            "swap_volume_vchr_to_vchr_tick": latest.get("swap_volume_vchr_to_vchr_tick", 0.0),
            "swap_volume_vchr_to_vchr_total": cumulative_float["swap_volume_vchr_to_vchr_tick"],
            "swap_stable_flow_value_tick": latest.get("swap_stable_flow_value_tick", 0.0),
            "swap_stable_flow_value_total": cumulative_swap_stable_flow,
            "swap_voucher_flow_value_tick": latest.get("swap_voucher_flow_value_tick", 0.0),
            "swap_voucher_flow_value_total": cumulative_swap_voucher_flow,
            "swap_stable_flow_share_tick": latest.get("swap_stable_flow_share_tick", 0.0),
            "swap_stable_flow_share_total": cumulative_swap_stable_flow / max(1e-9, cumulative_swap_gross_flow),
            "engine_gross_stable_flow_value_tick": latest_engine_gross_stable_flow,
            "engine_gross_stable_flow_value_total": cumulative_engine_gross_stable_flow,
            "engine_gross_voucher_flow_value_tick": latest_engine_gross_voucher_flow,
            "engine_gross_voucher_flow_value_total": cumulative_engine_gross_voucher_flow,
            "engine_gross_flow_stable_share_tick": (
                latest_engine_gross_stable_flow / max(1e-9, latest_engine_gross_flow)
            ),
            "engine_gross_flow_stable_share_total": (
                cumulative_engine_gross_stable_flow / max(1e-9, cumulative_engine_gross_flow)
            ),
            "swap_stable_net_flow_value_tick": latest.get("swap_stable_net_flow_value_tick", 0.0),
            "swap_stable_net_flow_value_total": cumulative_float["swap_stable_net_flow_value_tick"],
            "swap_voucher_net_flow_value_tick": latest.get("swap_voucher_net_flow_value_tick", 0.0),
            "swap_voucher_net_flow_value_total": cumulative_float["swap_voucher_net_flow_value_tick"],
            "route_source_stable_net_flow_value_tick": latest.get(
                "route_source_stable_net_flow_value_tick", 0.0
            ),
            "route_source_stable_net_flow_value_total": cumulative_float[
                "route_source_stable_net_flow_value_tick"
            ],
            "route_source_voucher_net_flow_value_tick": latest.get(
                "route_source_voucher_net_flow_value_tick", 0.0
            ),
            "route_source_voucher_net_flow_value_total": cumulative_float[
                "route_source_voucher_net_flow_value_tick"
            ],
            "engine_flow_stable_net_usd_total": cumulative_stable_flow_net,
            "engine_flow_voucher_net_usd_total": cumulative_voucher_flow_net,
            "engine_flow_positive_stable_usd_total": cumulative_stable_flow_positive,
            "engine_flow_positive_voucher_usd_total": cumulative_voucher_flow_positive,
            "engine_positive_net_flow_stable_share_total": cumulative_positive_flow_stable_share,
            "swap_count_usd_to_vchr_tick": latest.get("swap_count_usd_to_vchr_tick", 0),
            "swap_count_usd_to_vchr_total": cumulative_float["swap_count_usd_to_vchr_tick"],
            "swap_count_vchr_to_usd_tick": latest.get("swap_count_vchr_to_usd_tick", 0),
            "swap_count_vchr_to_usd_total": cumulative_float["swap_count_vchr_to_usd_tick"],
            "swap_count_vchr_to_vchr_tick": latest.get("swap_count_vchr_to_vchr_tick", 0),
            "swap_count_vchr_to_vchr_total": cumulative_float["swap_count_vchr_to_vchr_tick"],
            "ordinary_swap_count_tick": latest.get("ordinary_swap_count_tick", 0),
            "ordinary_swap_count_total": cumulative_float["ordinary_swap_count_tick"],
            "ordinary_swap_volume_usd_tick": latest.get("ordinary_swap_volume_usd_tick", 0.0),
            "ordinary_swap_volume_usd_total": cumulative_float["ordinary_swap_volume_usd_tick"],
            "ordinary_stable_source_swap_count_tick": latest.get(
                "ordinary_stable_source_swap_count_tick", 0
            ),
            "ordinary_stable_source_swap_count_total": cumulative_float[
                "ordinary_stable_source_swap_count_tick"
            ],
            "ordinary_stable_source_swap_volume_usd_tick": latest.get(
                "ordinary_stable_source_swap_volume_usd_tick", 0.0
            ),
            "ordinary_stable_source_swap_volume_usd_total": cumulative_float[
                "ordinary_stable_source_swap_volume_usd_tick"
            ],
            "ordinary_voucher_source_swap_count_tick": latest.get(
                "ordinary_voucher_source_swap_count_tick", 0
            ),
            "ordinary_voucher_source_swap_count_total": cumulative_float[
                "ordinary_voucher_source_swap_count_tick"
            ],
            "ordinary_voucher_source_swap_volume_usd_tick": latest.get(
                "ordinary_voucher_source_swap_volume_usd_tick", 0.0
            ),
            "ordinary_voucher_source_swap_volume_usd_total": cumulative_float[
                "ordinary_voucher_source_swap_volume_usd_tick"
            ],
            "loan_route_swap_count_tick": latest.get("loan_route_swap_count_tick", 0),
            "loan_route_swap_count_total": cumulative_float["loan_route_swap_count_tick"],
            "loan_route_swap_volume_usd_tick": latest.get("loan_route_swap_volume_usd_tick", 0.0),
            "loan_route_swap_volume_usd_total": cumulative_float["loan_route_swap_volume_usd_tick"],
            "voucher_loan_route_swap_count_tick": latest.get("voucher_loan_route_swap_count_tick", 0),
            "voucher_loan_route_swap_count_total": cumulative_float[
                "voucher_loan_route_swap_count_tick"
            ],
            "voucher_loan_route_swap_volume_usd_tick": latest.get(
                "voucher_loan_route_swap_volume_usd_tick", 0.0
            ),
            "voucher_loan_route_swap_volume_usd_total": cumulative_float[
                "voucher_loan_route_swap_volume_usd_tick"
            ],
            "repayment_route_swap_count_tick": latest.get("repayment_route_swap_count_tick", 0),
            "repayment_route_swap_count_total": cumulative_float["repayment_route_swap_count_tick"],
            "repayment_route_swap_volume_usd_tick": latest.get(
                "repayment_route_swap_volume_usd_tick", 0.0
            ),
            "repayment_route_swap_volume_usd_total": cumulative_float[
                "repayment_route_swap_volume_usd_tick"
            ],
            "loan_backfill_swap_count_tick": latest.get("loan_backfill_swap_count_tick", 0),
            "loan_backfill_swap_count_total": cumulative_float["loan_backfill_swap_count_tick"],
            "loan_backfill_swap_volume_usd_tick": latest.get(
                "loan_backfill_swap_volume_usd_tick", 0.0
            ),
            "loan_backfill_swap_volume_usd_total": cumulative_float[
                "loan_backfill_swap_volume_usd_tick"
            ],
            "productive_boosted_voucher_swap_count_tick": latest.get(
                "productive_boosted_voucher_swap_count_tick", 0
            ),
            "productive_boosted_voucher_swap_count_total": cumulative_float[
                "productive_boosted_voucher_swap_count_tick"
            ],
            "productive_boosted_voucher_swap_volume_usd_tick": latest.get(
                "productive_boosted_voucher_swap_volume_usd_tick", 0.0
            ),
            "productive_boosted_voucher_swap_volume_usd_total": cumulative_float[
                "productive_boosted_voucher_swap_volume_usd_tick"
            ],
            "voucher_loan_boosted_voucher_swap_count_tick": latest.get(
                "voucher_loan_boosted_voucher_swap_count_tick", 0
            ),
            "voucher_loan_boosted_voucher_swap_count_total": cumulative_float[
                "voucher_loan_boosted_voucher_swap_count_tick"
            ],
            "voucher_loan_boosted_voucher_swap_volume_usd_tick": latest.get(
                "voucher_loan_boosted_voucher_swap_volume_usd_tick", 0.0
            ),
            "voucher_loan_boosted_voucher_swap_volume_usd_total": cumulative_float[
                "voucher_loan_boosted_voucher_swap_volume_usd_tick"
            ],
            "ordinary_stable_spend_protected_skip_count_tick": latest.get(
                "ordinary_stable_spend_protected_skip_count_tick", 0
            ),
            "ordinary_stable_spend_protected_skip_count_total": cumulative_float[
                "ordinary_stable_spend_protected_skip_count_tick"
            ],
            "ordinary_stable_spend_protected_skip_value_usd_tick": latest.get(
                "ordinary_stable_spend_protected_skip_value_usd_tick", 0.0
            ),
            "ordinary_stable_spend_protected_skip_value_usd_total": cumulative_float[
                "ordinary_stable_spend_protected_skip_value_usd_tick"
            ],
            "route_success_rate_tick": route_success_rate(latest),
            "route_success_rate_cumulative": route_success_total,
            "route_found_total": cumulative["route_found"],
            "route_failed_total": cumulative["route_failed"],
            "route_fixed_success_rate_cumulative": fixed_route_success_total,
            "route_fixed_found_total": cumulative["route_fixed_found"],
            "route_fixed_failed_total": cumulative["route_fixed_failed"],
            "route_substitution_success_rate_cumulative": substitution_route_success_total,
            "route_substitution_found_total": cumulative["route_substitution_found"],
            "route_substitution_failed_total": cumulative["route_substitution_failed"],
            "noam_routing_swaps_tick": latest.get("noam_routing_swaps_tick", 0),
            "noam_clearing_swaps_tick": latest.get("noam_clearing_swaps_tick", 0),
            "noam_routing_swaps_total": cumulative_float["noam_routing_swaps_tick"],
            "noam_clearing_swaps_total": cumulative_float["noam_clearing_swaps_tick"],
            "noam_total_swaps_total": (
                cumulative_float["noam_routing_swaps_tick"]
                + cumulative_float["noam_clearing_swaps_tick"]
            ),
            "noam_routing_volume_usd_tick": latest.get("noam_routing_volume_usd_tick", 0.0),
            "noam_clearing_volume_usd_tick": latest.get("noam_clearing_volume_usd_tick", 0.0),
            "noam_routing_volume_usd_total": cumulative_float["noam_routing_volume_usd_tick"],
            "noam_clearing_volume_usd_total": cumulative_float["noam_clearing_volume_usd_tick"],
            "noam_total_volume_usd_total": (
                cumulative_float["noam_routing_volume_usd_tick"]
                + cumulative_float["noam_clearing_volume_usd_tick"]
            ),
            "noam_clearing_volume_share_total": (
                cumulative_float["noam_clearing_volume_usd_tick"]
                / max(
                    1e-9,
                    cumulative_float["noam_routing_volume_usd_tick"]
                    + cumulative_float["noam_clearing_volume_usd_tick"],
                )
            ),
            "noam_routing_fee_usd_tick": latest.get("noam_routing_fee_usd_tick", 0.0),
            "noam_clearing_fee_usd_tick": latest.get("noam_clearing_fee_usd_tick", 0.0),
            "noam_routing_fee_usd_total": cumulative_float["noam_routing_fee_usd_tick"],
            "noam_clearing_fee_usd_total": cumulative_float["noam_clearing_fee_usd_tick"],
            "noam_total_fee_usd_total": (
                cumulative_float["noam_routing_fee_usd_tick"]
                + cumulative_float["noam_clearing_fee_usd_tick"]
            ),
            "noam_routing_stable_fee_usd_total": cumulative_float[
                "noam_routing_stable_fee_usd_tick"
            ],
            "noam_routing_voucher_fee_usd_total": cumulative_float[
                "noam_routing_voucher_fee_usd_tick"
            ],
            "noam_clearing_stable_fee_usd_total": cumulative_float[
                "noam_clearing_stable_fee_usd_tick"
            ],
            "noam_clearing_voucher_fee_usd_total": cumulative_float[
                "noam_clearing_voucher_fee_usd_tick"
            ],
            "noam_clearing_cycles_attempted_tick": latest.get(
                "noam_clearing_cycles_attempted_tick", 0
            ),
            "noam_clearing_cycles_executed_tick": latest.get(
                "noam_clearing_cycles_executed_tick", 0
            ),
            "noam_clearing_cycles_attempted_total": cumulative_float[
                "noam_clearing_cycles_attempted_tick"
            ],
            "noam_clearing_cycles_executed_total": cumulative_float[
                "noam_clearing_cycles_executed_tick"
            ],
            "noam_clearing_cycle_success_rate_tick": latest.get(
                "noam_clearing_cycle_success_rate_tick", 0.0
            ),
            "noam_clearing_cycle_success_rate_cumulative": (
                cumulative_float["noam_clearing_cycles_executed_tick"]
                / max(1e-9, cumulative_float["noam_clearing_cycles_attempted_tick"])
            ),
            "noam_clearing_cycle_value_usd_tick": latest.get(
                "noam_clearing_cycle_value_usd_tick", 0.0
            ),
            "noam_clearing_cycle_value_usd_total": cumulative_float[
                "noam_clearing_cycle_value_usd_tick"
            ],
            "repayment_volume_usd": latest.get("repayment_volume_usd", 0.0),
            "repayment_volume_usd_total": cumulative_float["repayment_volume_usd"],
            "loan_issuance_volume_usd": latest.get("loan_issuance_volume_usd", 0.0),
            "loan_issuance_volume_usd_total": cumulative_float["loan_issuance_volume_usd"],
            "producer_debt_originated_usd": latest.get("producer_debt_originated_usd_tick", 0.0),
            "producer_debt_originated_usd_total": cumulative_float[
                "producer_debt_originated_usd_tick"
            ],
            "producer_loan_attempts": latest.get("producer_loan_attempts_tick", 0),
            "producer_loan_attempts_total": cumulative_float["producer_loan_attempts_tick"],
            "producer_loan_no_lender": latest.get("producer_loan_no_lender_tick", 0),
            "producer_loan_no_lender_total": cumulative_float["producer_loan_no_lender_tick"],
            "producer_loan_no_inventory": latest.get("producer_loan_no_inventory_tick", 0),
            "producer_loan_no_inventory_total": cumulative_float["producer_loan_no_inventory_tick"],
            "producer_loan_zero_amount": latest.get("producer_loan_zero_amount_tick", 0),
            "producer_loan_zero_amount_total": cumulative_float["producer_loan_zero_amount_tick"],
            "producer_loan_route_found": latest.get("producer_loan_route_found_tick", 0),
            "producer_loan_route_found_total": cumulative_float["producer_loan_route_found_tick"],
            "producer_loan_route_failed": latest.get("producer_loan_route_failed_tick", 0),
            "producer_loan_route_failed_total": cumulative_float["producer_loan_route_failed_tick"],
            "producer_loan_backfill_attempts": latest.get("producer_loan_backfill_attempts_tick", 0),
            "producer_loan_backfill_attempts_total": cumulative_float[
                "producer_loan_backfill_attempts_tick"
            ],
            "producer_loan_backfill_executed": latest.get("producer_loan_backfill_executed_tick", 0),
            "producer_loan_backfill_executed_total": cumulative_float[
                "producer_loan_backfill_executed_tick"
            ],
            "producer_loan_executed": latest.get("producer_loan_executed_tick", 0),
            "producer_loan_executed_total": cumulative_float["producer_loan_executed_tick"],
            "producer_loan_execution_failed": latest.get("producer_loan_execution_failed_tick", 0),
            "producer_loan_execution_failed_total": cumulative_float[
                "producer_loan_execution_failed_tick"
            ],
            "producer_loan_sampled_usd": latest.get("producer_loan_sampled_usd_tick", 0.0),
            "producer_loan_sampled_usd_total": cumulative_float["producer_loan_sampled_usd_tick"],
            "producer_loan_attempted_usd": latest.get("producer_loan_attempted_usd_tick", 0.0),
            "producer_loan_attempted_usd_total": cumulative_float[
                "producer_loan_attempted_usd_tick"
            ],
            "producer_loan_executed_usd": latest.get("producer_loan_executed_usd_tick", 0.0),
            "producer_loan_executed_usd_total": cumulative_float["producer_loan_executed_usd_tick"],
            "producer_loan_clipped_inventory_usd": latest.get(
                "producer_loan_clipped_inventory_usd_tick", 0.0
            ),
            "producer_loan_clipped_inventory_usd_total": cumulative_float[
                "producer_loan_clipped_inventory_usd_tick"
            ],
            "producer_loan_clipped_lender_cap_usd": latest.get(
                "producer_loan_clipped_lender_cap_usd_tick", 0.0
            ),
            "producer_loan_clipped_lender_cap_usd_total": cumulative_float[
                "producer_loan_clipped_lender_cap_usd_tick"
            ],
            "producer_loan_clipped_lender_remaining_usd": latest.get(
                "producer_loan_clipped_lender_remaining_usd_tick", 0.0
            ),
            "producer_loan_clipped_lender_remaining_usd_total": cumulative_float[
                "producer_loan_clipped_lender_remaining_usd_tick"
            ],
            "producer_loan_clipped_lender_stable_usd": latest.get(
                "producer_loan_clipped_lender_stable_usd_tick", 0.0
            ),
            "producer_loan_clipped_lender_stable_usd_total": cumulative_float[
                "producer_loan_clipped_lender_stable_usd_tick"
            ],
            "producer_loan_clipped_combined_lender_usd": latest.get(
                "producer_loan_clipped_combined_lender_usd_tick", 0.0
            ),
            "producer_loan_clipped_combined_lender_usd_total": cumulative_float[
                "producer_loan_clipped_combined_lender_usd_tick"
            ],
            "producer_loan_lender_collateral_cap_usd": latest.get(
                "producer_loan_lender_collateral_cap_usd_tick", 0.0
            ),
            "producer_loan_lender_collateral_cap_usd_total": cumulative_float[
                "producer_loan_lender_collateral_cap_usd_tick"
            ],
            "producer_loan_lender_remaining_cap_usd": latest.get(
                "producer_loan_lender_remaining_cap_usd_tick", 0.0
            ),
            "producer_loan_lender_remaining_cap_usd_total": cumulative_float[
                "producer_loan_lender_remaining_cap_usd_tick"
            ],
            "producer_loan_lender_stable_available_usd": latest.get(
                "producer_loan_lender_stable_available_usd_tick", 0.0
            ),
            "producer_loan_lender_stable_available_usd_total": cumulative_float[
                "producer_loan_lender_stable_available_usd_tick"
            ],
            "producer_voucher_loan_attempts": latest.get("producer_voucher_loan_attempts_tick", 0),
            "producer_voucher_loan_attempts_total": cumulative_float[
                "producer_voucher_loan_attempts_tick"
            ],
            "producer_voucher_loan_no_target": latest.get("producer_voucher_loan_no_target_tick", 0),
            "producer_voucher_loan_no_target_total": cumulative_float[
                "producer_voucher_loan_no_target_tick"
            ],
            "producer_voucher_loan_no_inventory": latest.get(
                "producer_voucher_loan_no_inventory_tick", 0
            ),
            "producer_voucher_loan_no_inventory_total": cumulative_float[
                "producer_voucher_loan_no_inventory_tick"
            ],
            "producer_voucher_loan_zero_amount": latest.get(
                "producer_voucher_loan_zero_amount_tick", 0
            ),
            "producer_voucher_loan_zero_amount_total": cumulative_float[
                "producer_voucher_loan_zero_amount_tick"
            ],
            "producer_voucher_loan_route_found": latest.get(
                "producer_voucher_loan_route_found_tick", 0
            ),
            "producer_voucher_loan_route_found_total": cumulative_float[
                "producer_voucher_loan_route_found_tick"
            ],
            "producer_voucher_loan_route_failed": latest.get(
                "producer_voucher_loan_route_failed_tick", 0
            ),
            "producer_voucher_loan_route_failed_total": cumulative_float[
                "producer_voucher_loan_route_failed_tick"
            ],
            "producer_voucher_loan_executed": latest.get("producer_voucher_loan_executed_tick", 0),
            "producer_voucher_loan_executed_total": cumulative_float[
                "producer_voucher_loan_executed_tick"
            ],
            "producer_voucher_loan_execution_failed": latest.get(
                "producer_voucher_loan_execution_failed_tick", 0
            ),
            "producer_voucher_loan_execution_failed_total": cumulative_float[
                "producer_voucher_loan_execution_failed_tick"
            ],
            "producer_voucher_loan_attempted_usd": latest.get(
                "producer_voucher_loan_attempted_usd_tick", 0.0
            ),
            "producer_voucher_loan_attempted_usd_total": cumulative_float[
                "producer_voucher_loan_attempted_usd_tick"
            ],
            "producer_voucher_loan_executed_usd": latest.get(
                "producer_voucher_loan_executed_usd_tick", 0.0
            ),
            "producer_voucher_loan_executed_usd_total": cumulative_float[
                "producer_voucher_loan_executed_usd_tick"
            ],
            "producer_voucher_loan_clipped_lender_cap_usd": latest.get(
                "producer_voucher_loan_clipped_lender_cap_usd_tick", 0.0
            ),
            "producer_voucher_loan_clipped_lender_cap_usd_total": cumulative_float[
                "producer_voucher_loan_clipped_lender_cap_usd_tick"
            ],
            "producer_voucher_loan_clipped_lender_remaining_usd": latest.get(
                "producer_voucher_loan_clipped_lender_remaining_usd_tick", 0.0
            ),
            "producer_voucher_loan_clipped_lender_remaining_usd_total": cumulative_float[
                "producer_voucher_loan_clipped_lender_remaining_usd_tick"
            ],
            "producer_primary_voucher_loan_attempts": latest.get(
                "producer_primary_voucher_loan_attempts_tick", 0
            ),
            "producer_primary_voucher_loan_attempts_total": latest.get(
                "producer_primary_voucher_loan_attempts_total",
                cumulative_float["producer_primary_voucher_loan_attempts_tick"],
            ),
            "producer_primary_voucher_loan_executed": latest.get(
                "producer_primary_voucher_loan_executed_tick", 0
            ),
            "producer_primary_voucher_loan_executed_total": latest.get(
                "producer_primary_voucher_loan_executed_total",
                cumulative_float["producer_primary_voucher_loan_executed_tick"],
            ),
            "lender_stable_total_usd": latest.get("lender_stable_total_usd", 0.0),
            "lender_stable_reserve_usd": latest.get("lender_stable_reserve_usd", 0.0),
            "lender_stable_available_above_reserve_usd": latest.get(
                "lender_stable_available_above_reserve_usd", 0.0
            ),
            "lender_held_producer_voucher_inventory_usd": latest.get(
                "lender_held_producer_voucher_inventory_usd", 0.0
            ),
            "producer_voucher_overlap_mode_empirical": latest.get(
                "producer_voucher_overlap_mode_empirical", 0
            ),
            "producer_voucher_overlap_tokens": latest.get("producer_voucher_overlap_tokens", 0),
            "producer_voucher_multi_lender_tokens": latest.get(
                "producer_voucher_multi_lender_tokens", 0
            ),
            "producer_voucher_multi_lender_share": latest.get(
                "producer_voucher_multi_lender_share", 0.0
            ),
            "producer_voucher_lender_degree_mean": latest.get(
                "producer_voucher_lender_degree_mean", 0.0
            ),
            "producer_voucher_lender_degree_p50": latest.get(
                "producer_voucher_lender_degree_p50", 0.0
            ),
            "producer_voucher_lender_degree_p90": latest.get(
                "producer_voucher_lender_degree_p90", 0.0
            ),
            "producer_voucher_lender_degree_max": latest.get(
                "producer_voucher_lender_degree_max", 0
            ),
            "producer_voucher_shared_lender_edges": latest.get(
                "producer_voucher_shared_lender_edges", 0
            ),
            "consumer_stable_available_above_reserve_usd": latest.get(
                "consumer_stable_available_above_reserve_usd", 0.0
            ),
            "voucher_purchase_attempts": latest.get("voucher_purchase_attempts_tick", 0),
            "voucher_purchase_attempts_total": latest.get(
                "voucher_purchase_attempts_total",
                cumulative_float["voucher_purchase_attempts_tick"],
            ),
            "consumer_voucher_purchase_attempts": latest.get(
                "consumer_voucher_purchase_attempts_tick", 0
            ),
            "consumer_voucher_purchase_attempts_total": latest.get(
                "consumer_voucher_purchase_attempts_total",
                cumulative_float["consumer_voucher_purchase_attempts_tick"],
            ),
            "consumer_voucher_purchase_success": latest.get(
                "consumer_voucher_purchase_success_tick", 0
            ),
            "consumer_voucher_purchase_success_total": latest.get(
                "consumer_voucher_purchase_success_total",
                cumulative_float["consumer_voucher_purchase_success_tick"],
            ),
            "consumer_voucher_purchase_no_stable": latest.get(
                "consumer_voucher_purchase_no_stable_tick", 0
            ),
            "consumer_voucher_purchase_no_stable_total": latest.get(
                "consumer_voucher_purchase_no_stable_total",
                cumulative_float["consumer_voucher_purchase_no_stable_tick"],
            ),
            "consumer_voucher_purchase_reserve_protected": latest.get(
                "consumer_voucher_purchase_reserve_protected_tick", 0
            ),
            "consumer_voucher_purchase_reserve_protected_total": latest.get(
                "consumer_voucher_purchase_reserve_protected_total",
                cumulative_float["consumer_voucher_purchase_reserve_protected_tick"],
            ),
            "consumer_voucher_purchase_no_route": latest.get(
                "consumer_voucher_purchase_no_route_tick", 0
            ),
            "consumer_voucher_purchase_no_route_total": latest.get(
                "consumer_voucher_purchase_no_route_total",
                cumulative_float["consumer_voucher_purchase_no_route_tick"],
            ),
            "consumer_voucher_purchase_no_target": latest.get(
                "consumer_voucher_purchase_no_target_tick", 0
            ),
            "consumer_voucher_purchase_no_target_total": latest.get(
                "consumer_voucher_purchase_no_target_total",
                cumulative_float["consumer_voucher_purchase_no_target_tick"],
            ),
            "consumer_voucher_purchase_stable_spent_usd": latest.get(
                "consumer_voucher_purchase_stable_spent_usd_tick", 0.0
            ),
            "consumer_voucher_purchase_stable_spent_usd_total": latest.get(
                "consumer_voucher_purchase_stable_spent_usd_total",
                cumulative_float["consumer_voucher_purchase_stable_spent_usd_tick"],
            ),
            "consumer_voucher_purchase_voucher_value_acquired_usd": latest.get(
                "consumer_voucher_purchase_voucher_value_acquired_usd_tick", 0.0
            ),
            "consumer_voucher_purchase_voucher_value_acquired_usd_total": latest.get(
                "consumer_voucher_purchase_voucher_value_acquired_usd_total",
                cumulative_float["consumer_voucher_purchase_voucher_value_acquired_usd_tick"],
            ),
            "third_party_voucher_purchase_attempts": latest.get(
                "third_party_voucher_purchase_attempts_tick", 0
            ),
            "third_party_voucher_purchase_attempts_total": latest.get(
                "third_party_voucher_purchase_attempts_total",
                cumulative_float["third_party_voucher_purchase_attempts_tick"],
            ),
            "third_party_voucher_purchase_success": latest.get(
                "third_party_voucher_purchase_success_tick", 0
            ),
            "third_party_voucher_purchase_success_total": latest.get(
                "third_party_voucher_purchase_success_total",
                cumulative_float["third_party_voucher_purchase_success_tick"],
            ),
            "third_party_voucher_purchase_no_stable": latest.get(
                "third_party_voucher_purchase_no_stable_tick", 0
            ),
            "third_party_voucher_purchase_no_stable_total": latest.get(
                "third_party_voucher_purchase_no_stable_total",
                cumulative_float["third_party_voucher_purchase_no_stable_tick"],
            ),
            "third_party_voucher_purchase_reserve_protected": latest.get(
                "third_party_voucher_purchase_reserve_protected_tick", 0
            ),
            "third_party_voucher_purchase_reserve_protected_total": latest.get(
                "third_party_voucher_purchase_reserve_protected_total",
                cumulative_float["third_party_voucher_purchase_reserve_protected_tick"],
            ),
            "third_party_voucher_purchase_no_route": latest.get(
                "third_party_voucher_purchase_no_route_tick", 0
            ),
            "third_party_voucher_purchase_no_route_total": latest.get(
                "third_party_voucher_purchase_no_route_total",
                cumulative_float["third_party_voucher_purchase_no_route_tick"],
            ),
            "third_party_voucher_purchase_no_target": latest.get(
                "third_party_voucher_purchase_no_target_tick", 0
            ),
            "third_party_voucher_purchase_no_target_total": latest.get(
                "third_party_voucher_purchase_no_target_total",
                cumulative_float["third_party_voucher_purchase_no_target_tick"],
            ),
            "third_party_voucher_purchase_stable_spent_usd": latest.get(
                "third_party_voucher_purchase_stable_spent_usd_tick", 0.0
            ),
            "third_party_voucher_purchase_stable_spent_usd_total": latest.get(
                "third_party_voucher_purchase_stable_spent_usd_total",
                cumulative_float["third_party_voucher_purchase_stable_spent_usd_tick"],
            ),
            "third_party_voucher_purchase_voucher_value_acquired_usd": latest.get(
                "third_party_voucher_purchase_voucher_value_acquired_usd_tick", 0.0
            ),
            "third_party_voucher_purchase_voucher_value_acquired_usd_total": latest.get(
                "third_party_voucher_purchase_voucher_value_acquired_usd_total",
                cumulative_float["third_party_voucher_purchase_voucher_value_acquired_usd_tick"],
            ),
            "lender_voucher_purchase_stable_budget_remaining_usd": latest.get(
                "lender_voucher_purchase_stable_budget_remaining_usd_tick", 0.0
            ),
            "lender_voucher_purchase_stable_budget_onramp_usd": latest.get(
                "lender_voucher_purchase_stable_budget_onramp_usd_tick", 0.0
            ),
            "lender_voucher_purchase_stable_budget_onramp_usd_total": latest.get(
                "lender_voucher_purchase_stable_budget_onramp_usd_total",
                cumulative_float["lender_voucher_purchase_stable_budget_onramp_usd_tick"],
            ),
            "consumer_voucher_purchase_stable_budget_onramp_usd": latest.get(
                "consumer_voucher_purchase_stable_budget_onramp_usd_tick", 0.0
            ),
            "consumer_voucher_purchase_stable_budget_onramp_usd_total": latest.get(
                "consumer_voucher_purchase_stable_budget_onramp_usd_total",
                cumulative_float["consumer_voucher_purchase_stable_budget_onramp_usd_tick"],
            ),
            "third_party_voucher_purchase_stable_budget_onramp_usd": latest.get(
                "third_party_voucher_purchase_stable_budget_onramp_usd_tick", 0.0
            ),
            "third_party_voucher_purchase_stable_budget_onramp_usd_total": latest.get(
                "third_party_voucher_purchase_stable_budget_onramp_usd_total",
                cumulative_float["third_party_voucher_purchase_stable_budget_onramp_usd_tick"],
            ),
            "lender_recovered_stable_usd": latest.get("lender_recovered_stable_usd_tick", 0.0),
            "lender_recovered_stable_usd_total": cumulative_float[
                "lender_recovered_stable_usd_tick"
            ],
            "lender_recovered_stable_borrower_regular_usd": latest.get(
                "lender_recovered_stable_borrower_regular_usd_tick", 0.0
            ),
            "lender_recovered_stable_borrower_regular_usd_total": cumulative_float[
                "lender_recovered_stable_borrower_regular_usd_tick"
            ],
            "lender_recovered_stable_borrower_maturity_usd": latest.get(
                "lender_recovered_stable_borrower_maturity_usd_tick", 0.0
            ),
            "lender_recovered_stable_borrower_maturity_usd_total": cumulative_float[
                "lender_recovered_stable_borrower_maturity_usd_tick"
            ],
            "lender_recovered_stable_consumer_purchase_usd": latest.get(
                "lender_recovered_stable_consumer_purchase_usd_tick", 0.0
            ),
            "lender_recovered_stable_consumer_purchase_usd_total": cumulative_float[
                "lender_recovered_stable_consumer_purchase_usd_tick"
            ],
            "lender_recovered_stable_third_party_purchase_usd": latest.get(
                "lender_recovered_stable_third_party_purchase_usd_tick", 0.0
            ),
            "lender_recovered_stable_third_party_purchase_usd_total": cumulative_float[
                "lender_recovered_stable_third_party_purchase_usd_tick"
            ],
            "lender_recovered_stable_other_usd": latest.get(
                "lender_recovered_stable_other_usd_tick", 0.0
            ),
            "lender_recovered_stable_other_usd_total": cumulative_float[
                "lender_recovered_stable_other_usd_tick"
            ],
            "bond_service_reserve_balance_usd": latest.get("bond_service_reserve_balance_usd", 0.0),
            "bond_service_reserved_usd": latest.get("bond_service_reserved_usd_tick", 0.0),
            "bond_service_reserved_usd_total": cumulative_float["bond_service_reserved_usd_tick"],
            "fee_service_reserved_usd": latest.get("fee_service_reserved_usd_tick", 0.0),
            "fee_service_reserved_usd_total": cumulative_float["fee_service_reserved_usd_tick"],
            "fee_service_stable_reserved_usd": latest.get(
                "fee_service_stable_reserved_usd_tick", 0.0
            ),
            "fee_service_stable_reserved_usd_total": cumulative_float[
                "fee_service_stable_reserved_usd_tick"
            ],
            "fee_service_converted_voucher_reserved_usd": latest.get(
                "fee_service_converted_voucher_reserved_usd_tick", 0.0
            ),
            "fee_service_converted_voucher_reserved_usd_total": cumulative_float[
                "fee_service_converted_voucher_reserved_usd_tick"
            ],
            "bond_service_paid_from_reserve_usd": latest.get(
                "bond_service_paid_from_reserve_usd_tick", 0.0
            ),
            "bond_service_paid_from_reserve_usd_total": cumulative_float[
                "bond_service_paid_from_reserve_usd_tick"
            ],
            "debt_outstanding_usd": latest.get("debt_outstanding_usd", 0.0),
            "issued_voucher_supply_total": latest.get("issued_voucher_supply_total", 0.0),
            "issuer_returned_voucher_supply_total": latest.get("issuer_returned_voucher_supply_total", 0.0),
            "net_circulating_voucher_supply_total": latest.get("net_circulating_voucher_supply_total", 0.0),
            "producer_deposit_stable_usd": latest.get("producer_deposit_stable_usd_tick", 0.0),
            "producer_deposit_stable_usd_total": cumulative_float["producer_deposit_stable_usd_tick"],
            "producer_deposit_voucher_usd": latest.get("producer_deposit_voucher_usd_tick", 0.0),
            "producer_deposit_voucher_usd_total": cumulative_float["producer_deposit_voucher_usd_tick"],
            "producer_deposit_credit_capacity_usd": latest.get("producer_deposit_credit_capacity_usd", 0.0),
            "productive_credit_inflow_usd": latest.get("productive_credit_inflow_usd_tick", 0.0),
            "productive_credit_inflow_usd_total": cumulative_float["productive_credit_inflow_usd_tick"],
            "productive_credit_stable_retained_usd": latest.get(
                "productive_credit_stable_retained_usd_tick", 0.0
            ),
            "productive_credit_stable_retained_usd_total": cumulative_float[
                "productive_credit_stable_retained_usd_tick"
            ],
            "productive_credit_voucher_deposit_usd": latest.get(
                "productive_credit_voucher_deposit_usd_tick", 0.0
            ),
            "productive_credit_voucher_deposit_usd_total": cumulative_float[
                "productive_credit_voucher_deposit_usd_tick"
            ],
            "productive_credit_voucher_deposit_cap_clipped_usd": latest.get(
                "productive_credit_voucher_deposit_cap_clipped_usd_tick", 0.0
            ),
            "productive_credit_voucher_deposit_cap_clipped_usd_total": cumulative_float[
                "productive_credit_voucher_deposit_cap_clipped_usd_tick"
            ],
            "productive_credit_voucher_deposit_share": (
                cumulative_float["productive_credit_voucher_deposit_usd_tick"]
                / max(1e-9, cumulative_float["productive_credit_inflow_usd_tick"])
            ),
            "producer_debt_active_obligations": latest.get("producer_debt_active_obligations", 0),
            "producer_debt_active_usd": latest.get("producer_debt_active_usd", 0.0),
            "producer_debt_cash_service_due_usd": latest.get(
                "producer_debt_cash_service_due_usd_tick", 0.0
            ),
            "producer_debt_cash_service_due_usd_total": cumulative_float[
                "producer_debt_cash_service_due_usd_tick"
            ],
            "producer_debt_cash_service_paid_usd": latest.get(
                "producer_debt_cash_service_paid_usd_tick", 0.0
            ),
            "producer_debt_cash_service_paid_usd_total": cumulative_float[
                "producer_debt_cash_service_paid_usd_tick"
            ],
            "producer_debt_matured_usd": latest.get("producer_debt_matured_usd_tick", 0.0),
            "producer_debt_matured_usd_total": cumulative_float["producer_debt_matured_usd_tick"],
            "producer_debt_repaid_usd": latest.get("producer_debt_repaid_usd_tick", 0.0),
            "producer_debt_repaid_usd_total": cumulative_float["producer_debt_repaid_usd_tick"],
            "producer_debt_repaid_regular_usd": latest.get(
                "producer_debt_repaid_regular_usd_tick", 0.0
            ),
            "producer_debt_repaid_regular_usd_total": cumulative_float[
                "producer_debt_repaid_regular_usd_tick"
            ],
            "producer_debt_repaid_maturity_usd": latest.get(
                "producer_debt_repaid_maturity_usd_tick", 0.0
            ),
            "producer_debt_repaid_maturity_usd_total": cumulative_float[
                "producer_debt_repaid_maturity_usd_tick"
            ],
            "producer_debt_stable_recovered_usd": latest.get(
                "producer_debt_stable_recovered_usd_tick", 0.0
            ),
            "producer_debt_stable_recovered_usd_total": cumulative_float[
                "producer_debt_stable_recovered_usd_tick"
            ],
            "producer_debt_consumer_stable_purchase_usd": latest.get(
                "producer_debt_consumer_stable_purchase_usd_tick", 0.0
            ),
            "producer_debt_consumer_stable_purchase_usd_total": cumulative_float[
                "producer_debt_consumer_stable_purchase_usd_tick"
            ],
            "producer_debt_third_party_stable_purchase_usd": latest.get(
                "producer_debt_third_party_stable_purchase_usd_tick", 0.0
            ),
            "producer_debt_third_party_stable_purchase_usd_total": cumulative_float[
                "producer_debt_third_party_stable_purchase_usd_tick"
            ],
            "producer_debt_defaulted_usd": latest.get("producer_debt_defaulted_usd_tick", 0.0),
            "producer_debt_defaulted_usd_total": cumulative_float["producer_debt_defaulted_usd_tick"],
            "producer_debt_closed_by_circulation_usd": latest.get(
                "producer_debt_closed_by_circulation_usd_tick", 0.0
            ),
            "producer_debt_closed_by_circulation_usd_total": cumulative_float[
                "producer_debt_closed_by_circulation_usd_tick"
            ],
            "producer_debt_closed_by_voucher_swap_usd": latest.get(
                "producer_debt_closed_by_voucher_swap_usd_tick", 0.0
            ),
            "producer_debt_closed_by_voucher_swap_usd_total": cumulative_float[
                "producer_debt_closed_by_voucher_swap_usd_tick"
            ],
            "producer_debt_closed_not_held_at_maturity_usd": latest.get(
                "producer_debt_closed_not_held_at_maturity_usd_tick", 0.0
            ),
            "producer_debt_closed_not_held_at_maturity_usd_total": cumulative_float[
                "producer_debt_closed_not_held_at_maturity_usd_tick"
            ],
            "producer_debt_maturity_recovery_rate": latest.get(
                "producer_debt_maturity_recovery_rate", 0.0
            ),
            "fee_pool_cumulative_usd": latest.get("fee_pool_cumulative_usd", 0.0),
            "fee_clc_cumulative_usd": latest.get("fee_clc_cumulative_usd", 0.0),
            "fee_conversion_attempted_usd": latest.get("fee_conversion_attempted_usd_tick", 0.0),
            "fee_conversion_attempted_usd_total": cumulative_float["fee_conversion_attempted_usd_tick"],
            "fee_conversion_success_usd": latest.get("fee_conversion_success_usd_tick", 0.0),
            "fee_conversion_success_usd_total": cumulative_float["fee_conversion_success_usd_tick"],
            "fee_conversion_failed_usd": latest.get("fee_conversion_failed_usd_tick", 0.0),
            "fee_conversion_failed_usd_total": cumulative_float["fee_conversion_failed_usd_tick"],
            "fee_service_reserved_usd_epoch": latest.get("fee_service_reserved_usd_epoch", 0.0),
            "fee_service_stable_reserved_usd_epoch": latest.get(
                "fee_service_stable_reserved_usd_epoch", 0.0
            ),
            "fee_service_converted_voucher_reserved_usd_epoch": latest.get(
                "fee_service_converted_voucher_reserved_usd_epoch", 0.0
            ),
            "waterfall_ops_alloc_usd_epoch": latest.get("waterfall_ops_alloc_usd_epoch", 0.0),
            "waterfall_ops_alloc_usd_total": cumulative_float["waterfall_ops_alloc_usd_epoch"],
            "waterfall_insurance_alloc_usd_epoch": latest.get(
                "waterfall_insurance_alloc_usd_epoch", 0.0
            ),
            "waterfall_insurance_alloc_usd_total": cumulative_float[
                "waterfall_insurance_alloc_usd_epoch"
            ],
            "waterfall_mandates_alloc_usd_epoch": latest.get(
                "waterfall_mandates_alloc_usd_epoch", 0.0
            ),
            "waterfall_mandates_alloc_usd_total": cumulative_float[
                "waterfall_mandates_alloc_usd_epoch"
            ],
            "quarterly_clearing_usd": latest.get("quarterly_clearing_usd_tick", 0.0),
            "quarterly_clearing_usd_total": cumulative_float["quarterly_clearing_usd_tick"],
            "quarterly_clearing_lender_liquidity_before_tick": latest.get(
                "quarterly_clearing_lender_liquidity_before_tick", 0.0
            ),
            "quarterly_clearing_lender_liquidity_after_tick": latest.get(
                "quarterly_clearing_lender_liquidity_after_tick", 0.0
            ),
            "claims_unpaid_usd_tick": latest.get("claims_unpaid_usd_tick", 0.0),
            "stable_onramp_usd_tick": latest.get("stable_onramp_usd_tick", 0.0),
            "stable_onramp_usd_total": cumulative_float["stable_onramp_usd_tick"],
            "stable_offramp_usd_tick": latest.get("stable_offramp_usd_tick", 0.0),
            "stable_offramp_usd_total": cumulative_float["stable_offramp_usd_tick"],
            "historical_stable_backing_usd_tick": latest.get("historical_stable_backing_usd_tick", 0.0),
            "historical_stable_backing_usd_total": latest.get("historical_stable_backing_usd_total", 0.0),
            "historical_stable_backing_pools_tick": latest.get("historical_stable_backing_pools_tick", 0),
            "historical_stable_backing_pools_total": latest.get("historical_stable_backing_pools_total", 0),
            "historical_stable_backing_producer_usd_total": latest.get(
                "historical_stable_backing_producer_usd_total", 0.0
            ),
            "historical_stable_backing_consumer_usd_total": latest.get(
                "historical_stable_backing_consumer_usd_total", 0.0
            ),
            "historical_stable_backing_lender_usd_total": latest.get(
                "historical_stable_backing_lender_usd_total", 0.0
            ),
            "historical_stable_backing_producer_pools_total": latest.get(
                "historical_stable_backing_producer_pools_total", 0
            ),
            "historical_stable_backing_consumer_pools_total": latest.get(
                "historical_stable_backing_consumer_pools_total", 0
            ),
            "historical_stable_backing_lender_pools_total": latest.get(
                "historical_stable_backing_lender_pools_total", 0
            ),
            "stable_excess_sweep_usd_tick": latest.get("stable_excess_sweep_usd_tick", 0.0),
            "stable_excess_sweep_usd_total": latest.get("stable_excess_sweep_usd_total", 0.0),
            "stable_excess_sweep_pools_tick": latest.get("stable_excess_sweep_pools_tick", 0),
            "stable_excess_sweep_pools_total": latest.get("stable_excess_sweep_pools_total", 0),
            "household_cash_stress_ratio": stress_ratio,
            "producer_stable_reserve_stress_ratio": producer_stress_ratio,
            "producer_pools_under_stable_reserve": latest.get("producer_pools_under_stable_reserve", 0),
            "producer_pools_total": latest.get("producer_pools_total", 0),
            "producer_stable_reserve_deficit_usd": latest.get(
                "producer_stable_reserve_deficit_usd", 0.0
            ),
            "consumer_stable_reserve_stress_ratio": consumer_stress_ratio,
            "consumer_pools_under_stable_reserve": latest.get("consumer_pools_under_stable_reserve", 0),
            "consumer_pools_total": latest.get("consumer_pools_total", 0),
            "consumer_stable_reserve_deficit_usd": latest.get(
                "consumer_stable_reserve_deficit_usd", 0.0
            ),
            "lender_stable_reserve_stress_ratio": lender_stress_ratio,
            "lender_pools_under_stable_reserve": latest.get("lender_pools_under_stable_reserve", 0),
            "lender_pools_total": latest.get("lender_pools_total", 0),
            "lender_stable_reserve_deficit_usd": latest.get(
                "lender_stable_reserve_deficit_usd", 0.0
            ),
            "community_stable_reserve_stress_ratio": community_stress_ratio,
            "community_pools_under_stable_reserve": latest.get(
                "community_pools_under_stable_reserve", 0
            ),
            "community_pools_total": latest.get("community_pools_total", 0),
            "community_stable_reserve_deficit_usd": latest.get(
                "community_stable_reserve_deficit_usd", 0.0
            ),
            "stable_liquidity_leakage_ratio_tick": leakage_ratio,
            "stable_liquidity_leakage_ratio_cumulative": (
                cumulative_float["stable_offramp_usd_tick"]
                / max(
                    1e-9,
                    cumulative_float["stable_onramp_usd_tick"]
                    + safe_float(bmetrics.get("bond_principal_usd"))
                    + safe_float(latest.get("stable_total_in_pools")),
                )
            ),
            "expected_cash_return_coverage": expected_cash_cov,
            "expected_voucher_return_coverage": expected_voucher_cov,
            "expected_verified_report_exposure": expected_reports,
            "strong_pool_share": mix.get("strong", 0.0),
            "moderate_pool_share": mix.get("moderate", 0.0),
            "weak_pool_share": mix.get("weak", 0.0),
        }
        bond_rows.append({**common, **bmetrics})
        network_rows.append({**common, **potential_metrics, **realized_metrics})

    final = bond_rows[-1] if bond_rows else {}
    final_network = network_rows[-1] if network_rows else {}
    tier_counts = Counter(pool_tiers.get(pid, "moderate") for pid in calibrated_lender_pool_ids(engine))
    tier_swap_counts = Counter(
        pool_tiers[pid]
        for pid, count in pool_swap_counts.items()
        if pid in pool_tiers
        for _ in range(int(count))
    )
    report_by_tier = {}
    for tier in TIER_ORDER:
        counts = [
            count
            for pid, count in pool_swap_counts.items()
            if pool_tiers.get(pid) == tier
        ]
        report_by_tier[tier], _ = expected_report_exposure_from_counts(calibration, counts)
    total_active_pools = sum(tier_counts.values()) or 1
    backing_proxy_total = (
        float(getattr(engine, "initial_stable_total", 0.0))
        + safe_float(final.get("stable_onramp_usd_total"))
        + safe_float(final.get("bond_principal_usd"))
    )
    if scenario == "sarafu_engine_validation":
        targets = {row["tier"]: row for row in empirical_tier_targets(calibration, int(args.ticks))}
        empirical_total_swaps = sum(safe_float(row.get("total_swap_events_horizon")) for row in targets.values())
        empirical_total_reports = sum(
            safe_float(row.get("total_verified_report_exposure_horizon")) for row in targets.values()
        )
        empirical_total_backing = sum(safe_float(row.get("backing_liquidity_inflow")) for row in targets.values())
        engine_total_swaps = safe_float(final.get("transactions_total"))
        for tier in TIER_ORDER:
            target = targets.get(tier, {})
            activity_share = safe_float(target.get("total_swap_events_horizon")) / max(1e-9, empirical_total_swaps)
            report_per_swap = safe_float(target.get("total_verified_report_exposure_horizon")) / max(
                1e-9, safe_float(target.get("total_swap_events_horizon"))
            )
            backing_share = safe_float(target.get("backing_liquidity_inflow")) / max(1e-9, empirical_total_backing)
            tier_swap_counts[tier] = engine_total_swaps * activity_share
            report_by_tier[tier] = tier_swap_counts[tier] * report_per_swap
            tier_counts[tier] = int(safe_float(target.get("pool_count")))
        total_active_pools = sum(tier_counts.values()) or total_active_pools
    tier_summary_fields: dict[str, object] = {}
    for tier in TIER_ORDER:
        if scenario == "sarafu_engine_validation":
            targets = {row["tier"]: row for row in empirical_tier_targets(calibration, int(args.ticks))}
            empirical_total_backing = sum(
                safe_float(row.get("backing_liquidity_inflow")) for row in targets.values()
            )
            share = safe_float(targets.get(tier, {}).get("backing_liquidity_inflow")) / max(
                1e-9, empirical_total_backing
            )
        else:
            share = tier_counts.get(tier, 0) / total_active_pools
        tier_summary_fields.update(
            {
                f"engine_pool_count_{tier}": tier_counts.get(tier, 0),
                f"engine_swap_events_{tier}": tier_swap_counts.get(tier, 0),
                f"engine_report_exposure_{tier}": report_by_tier.get(tier, 0.0),
                f"engine_same_token_return_coverage_{tier}": calibration_same_token_return(calibration, tier),
                f"engine_cash_return_coverage_{tier}": calibration.repayment_by_tier_asset.get((tier, "cash"), 0.0),
                f"engine_voucher_return_coverage_{tier}": calibration.voucher_coverage_by_tier.get(tier, 0.0),
                f"engine_borrow_proxy_closure_{tier}": calibration.borrow_return_by_tier.get(tier, 0.0),
                f"engine_backing_liquidity_inflow_{tier}": backing_proxy_total * share,
            }
        )
    if scenario == "sarafu_engine_validation":
        borrow_weight = {
            tier: sum(
                pool.borrow_proxy_matured_events
                for pool in calibration.pool_rows
                if pool.tier == tier
            )
            for tier in TIER_ORDER
        }
        total_borrow_weight = sum(borrow_weight.values())
        tier_summary_fields["engine_repayment_borrow_proxy_closure"] = (
            sum(
                calibration.borrow_return_by_tier.get(tier, 0.0) * borrow_weight.get(tier, 0.0)
                for tier in TIER_ORDER
            )
            / max(1e-9, total_borrow_weight)
        )
    else:
        tier_summary_fields["engine_repayment_borrow_proxy_closure"] = (
            safe_float(final.get("repayment_volume_usd_total"))
            / max(1e-9, safe_float(final.get("loan_issuance_volume_usd_total")))
        )
    tier_summary_fields["raw_engine_repayment_loan_flow_ratio"] = (
        safe_float(final.get("repayment_volume_usd_total"))
        / max(1e-9, safe_float(final.get("loan_issuance_volume_usd_total")))
    )
    for tier in TIER_ORDER:
        pool_ids = [pid for pid, pool_tier in pool_tiers.items() if pool_tier == tier]
        inflow = sum(engine._productive_credit_inflow_usd_by_pool.get(pid, 0.0) for pid in pool_ids)
        stable_retained = sum(
            engine._productive_credit_stable_retained_usd_by_pool.get(pid, 0.0)
            for pid in pool_ids
        )
        voucher_deposit = sum(
            engine._productive_credit_voucher_deposit_usd_by_pool.get(pid, 0.0)
            for pid in pool_ids
        )
        tier_summary_fields.update(
            {
                f"engine_productive_credit_inflow_usd_{tier}": inflow,
                f"engine_productive_credit_stable_retained_usd_{tier}": stable_retained,
                f"engine_productive_credit_voucher_deposit_usd_{tier}": voucher_deposit,
                f"engine_productive_credit_voucher_deposit_share_{tier}": (
                    voucher_deposit / max(1e-9, inflow)
                ),
            }
        )
    failure = {
        "scenario": scenario,
        "run": run_index,
        "seed": seed,
        "network_scale": getattr(args, "_current_network_scale", "current"),
        "principal_ratio": getattr(args, "_current_principal_ratio", 0.0),
        "bond_fee_service_share": getattr(args, "_current_bond_fee_service_share", cfg.bond_fee_service_share),
        "certification_policy": getattr(args, "_current_certification_policy", ""),
        "coupon_target_annual": float(cfg.bond_coupon_target_annual),
        "bond_term_ticks": int(cfg.bond_term_ticks),
        "tick": final.get("tick", 0),
        "coupon_shortfall_usd": final.get("bond_coupon_shortfall_usd", 0.0),
        "coupon_shortfall_flag": int(safe_float(final.get("bond_coupon_shortfall_usd")) > 1e-9),
        "unpaid_claims_usd": final.get("issuer_unpaid_scheduled_claim_usd", final.get("claims_unpaid_usd_tick", 0.0)),
        "unpaid_claims_flag": int(
            safe_float(final.get("issuer_unpaid_scheduled_claim_usd", final.get("claims_unpaid_usd_tick"))) > 1e-9
        ),
        "liquidity_leakage_ratio": final.get("stable_liquidity_leakage_ratio_tick", 0.0),
        "liquidity_leakage_flag": int(safe_float(final.get("stable_liquidity_leakage_ratio_tick")) > 0.05),
        "household_cash_stress_ratio": final.get("household_cash_stress_ratio", 0.0),
        "household_cash_stress_flag": int(safe_float(final.get("household_cash_stress_ratio")) > 0.20),
        "consumer_stable_reserve_stress_ratio": final.get("consumer_stable_reserve_stress_ratio", 0.0),
        "community_stable_reserve_stress_ratio": final.get("community_stable_reserve_stress_ratio", 0.0),
        "consumer_cash_stress_flag": int(
            safe_float(final.get("consumer_stable_reserve_stress_ratio")) > 0.20
        ),
        "community_cash_stress_flag": int(
            safe_float(final.get("community_stable_reserve_stress_ratio")) > 0.20
        ),
        "realized_edge_top_share": final_network.get("realized_edge_top_share", 0.0),
        "concentration_flag": int(safe_float(final_network.get("realized_edge_top_share")) > 0.25),
        "realized_largest_component_share": final_network.get("realized_largest_component_share", 0.0),
        "potential_largest_component_share": final_network.get("potential_largest_component_share", 0.0),
    }
    summary = {
        **final,
        **{
            "potential_largest_component_share": final_network.get("potential_largest_component_share", 0.0),
            "realized_largest_component_share": final_network.get("realized_largest_component_share", 0.0),
            "realized_edge_top_share": final_network.get("realized_edge_top_share", 0.0),
            "top_impact_activity": max(impact_latest, key=impact_latest.get) if impact_latest else "",
            "top_impact_expected_exposure": max(impact_latest.values()) if impact_latest else 0.0,
            "initial_stable_total": float(getattr(engine, "initial_stable_total", 0.0)),
            "lender_recovered_stable_by_pool_json": json.dumps(
                getattr(engine, "_lender_recovered_stable_total_by_pool", {}),
                sort_keys=True,
            ),
            "lender_recovered_stable_pending_by_pool_json": json.dumps(
                getattr(engine, "_lender_recovered_stable_by_pool", {}),
                sort_keys=True,
            ),
            "lender_recovered_stable_by_pool_reason_json": json.dumps(
                {
                    f"{pool_id}|{reason}": amount
                    for (pool_id, reason), amount in getattr(
                        engine, "_lender_recovered_stable_total_by_pool_reason", {}
                    ).items()
                },
                sort_keys=True,
            ),
            "bond_service_reserved_by_pool_json": json.dumps(
                getattr(engine, "_bond_service_reserved_by_pool", {}),
                sort_keys=True,
            ),
            **tier_summary_fields,
        },
    }
    return bond_rows, network_rows, [failure], summary


def quantile_rows(rows: list[dict[str, object]], metrics: Iterable[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (
            row["scenario"],
            row["coupon_target_annual"],
            row["bond_term_ticks"],
            row["tick"],
        )
        for metric in metrics:
            grouped[key][metric].append(safe_float(row.get(metric)))

    output = []
    for key, metric_values in sorted(grouped.items(), key=lambda item: item[0]):
        scenario, coupon, term, tick = key
        for metric, values in sorted(metric_values.items()):
            output.append(
                {
                    "scenario": scenario,
                    "coupon_target_annual": coupon,
                    "bond_term_ticks": term,
                    "tick": tick,
                    "metric": metric,
                    "mean": sum(values) / len(values) if values else 0.0,
                    "p05": percentile(values, 0.05),
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                    "n": len(values),
                }
            )
    return output


def summarize_failures(failure_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in failure_rows:
        grouped[(row["scenario"], row["coupon_target_annual"], row["bond_term_ticks"])].append(row)
    output = []
    for (scenario, coupon, term), rows in sorted(grouped.items()):
        n = len(rows) or 1
        output.append(
            {
                "scenario": scenario,
                "coupon_target_annual": coupon,
                "bond_term_ticks": term,
                "runs": n,
                "coupon_shortfall_rate": sum(int(r["coupon_shortfall_flag"]) for r in rows) / n,
                "median_coupon_shortfall_usd": percentile(
                    [safe_float(r["coupon_shortfall_usd"]) for r in rows], 0.50
                ),
                "unpaid_claims_rate": sum(int(r["unpaid_claims_flag"]) for r in rows) / n,
                "liquidity_leakage_rate": sum(int(r["liquidity_leakage_flag"]) for r in rows) / n,
                "household_cash_stress_rate": sum(int(r["household_cash_stress_flag"]) for r in rows) / n,
                "concentration_rate": sum(int(r["concentration_flag"]) for r in rows) / n,
                "median_realized_largest_component_share": percentile(
                    [safe_float(r["realized_largest_component_share"]) for r in rows], 0.50
                ),
            }
        )
    return output


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_pct(value: object) -> str:
    return f"{safe_float(value) * 100.0:.1f}\\%"


def write_latex_tables(
    output_dir: Path,
    calibration: Calibration,
    summary_rows: list[dict[str, object]],
    failure_summary_rows: list[dict[str, object]],
) -> None:
    calibration_lines = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Calibration item & Empirical value & Simulator use \\",
        r"\midrule",
    ]
    for tier in ("strong", "moderate", "weak"):
        cash = calibration.repayment_by_tier_asset.get((tier, "cash"), 0.0)
        voucher = calibration.voucher_coverage_by_tier.get(tier, 0.0)
        calibration_lines.append(
            f"{latex_escape(tier.title())} cash return & {fmt_pct(cash)} & stable repayment prior \\\\"
        )
        calibration_lines.append(
            f"{latex_escape(tier.title())} voucher return & {fmt_pct(voucher)} & voucher settlement prior \\\\"
        )
    for row in calibration.impact_rows[:5]:
        calibration_lines.append(
            f"{latex_escape(row.activity)} report elasticity & {row.slope:.2f} & report-arrival projection \\\\"
        )
    calibration_lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    (output_dir / "mc_calibration_table.tex").write_text(
        "\n".join(calibration_lines) + "\n", encoding="utf-8"
    )

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["scenario"], row["coupon_target_annual"], row["bond_term_ticks"])].append(row)
    bond_lines = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scenario & Coupon & Term & Median APR & Median coverage \\",
        r"\midrule",
    ]
    for idx, (key, rows) in enumerate(sorted(grouped.items())):
        if idx >= 18:
            break
        scenario, coupon, term = key
        apr = percentile([safe_float(r["bond_annualized_fee_yield"]) for r in rows], 0.50)
        coverage = percentile([safe_float(r["bond_coupon_coverage_ratio"]) for r in rows], 0.50)
        bond_lines.append(
            f"{latex_escape(scenario)} & {fmt_pct(coupon)} & {int(term)} & {fmt_pct(apr)} & {coverage:.2f} \\\\"
        )
    bond_lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    (output_dir / "mc_bond_return_table.tex").write_text(
        "\n".join(bond_lines) + "\n", encoding="utf-8"
    )

    failure_lines = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scenario & Coupon & Shortfall rate & Leakage rate & Stress rate \\",
        r"\midrule",
    ]
    for idx, row in enumerate(failure_summary_rows):
        if idx >= 18:
            break
        failure_lines.append(
            "{scenario} & {coupon} & {shortfall} & {leakage} & {stress} \\\\".format(
                scenario=latex_escape(row["scenario"]),
                coupon=fmt_pct(row["coupon_target_annual"]),
                shortfall=fmt_pct(row["coupon_shortfall_rate"]),
                leakage=fmt_pct(row["liquidity_leakage_rate"]),
                stress=fmt_pct(row["household_cash_stress_rate"]),
            )
        )
    failure_lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    (output_dir / "mc_failure_table.tex").write_text(
        "\n".join(failure_lines) + "\n", encoding="utf-8"
    )


def write_csv_tables(
    output_dir: Path,
    calibration: Calibration,
    summary_rows: list[dict[str, object]],
    failure_summary_rows: list[dict[str, object]],
) -> None:
    calibration_rows = []
    for tier in ("strong", "moderate", "weak"):
        calibration_rows.append(
            {
                "calibration_item": f"{tier}_cash_return",
                "empirical_value": calibration.repayment_by_tier_asset.get((tier, "cash"), 0.0),
                "simulator_use": "stable repayment prior",
            }
        )
        calibration_rows.append(
            {
                "calibration_item": f"{tier}_voucher_return",
                "empirical_value": calibration.voucher_coverage_by_tier.get(tier, 0.0),
                "simulator_use": "voucher settlement prior",
            }
        )
    for row in calibration.impact_rows[:12]:
        calibration_rows.append(
            {
                "calibration_item": f"{row.activity}_report_elasticity",
                "empirical_value": row.slope,
                "simulator_use": "report-arrival projection",
            }
        )
    write_csv(
        output_dir / "mc_calibration_table.csv",
        ["calibration_item", "empirical_value", "simulator_use"],
        calibration_rows,
    )

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["scenario"], row["coupon_target_annual"], row["bond_term_ticks"])].append(row)

    bond_table_rows = []
    for (scenario, coupon, term), rows in sorted(grouped.items()):
        bond_table_rows.append(
            {
                "scenario": scenario,
                "coupon_target_annual": coupon,
                "bond_term_ticks": term,
                "runs": len(rows),
                "median_bond_annualized_fee_yield": percentile(
                    [safe_float(r["bond_annualized_fee_yield"]) for r in rows], 0.50
                ),
                "p05_bond_annualized_fee_yield": percentile(
                    [safe_float(r["bond_annualized_fee_yield"]) for r in rows], 0.05
                ),
                "p95_bond_annualized_fee_yield": percentile(
                    [safe_float(r["bond_annualized_fee_yield"]) for r in rows], 0.95
                ),
                "median_coupon_coverage_ratio": percentile(
                    [safe_float(r["bond_coupon_coverage_ratio"]) for r in rows], 0.50
                ),
                "median_cumulative_fee_return_usd": percentile(
                    [safe_float(r["bond_cumulative_fee_return_usd"]) for r in rows], 0.50
                ),
                "median_coupon_shortfall_usd": percentile(
                    [safe_float(r["bond_coupon_shortfall_usd"]) for r in rows], 0.50
                ),
                "median_realized_largest_component_share": percentile(
                    [safe_float(r.get("realized_largest_component_share")) for r in rows], 0.50
                ),
                "median_expected_verified_report_exposure": percentile(
                    [safe_float(r["expected_verified_report_exposure"]) for r in rows], 0.50
                ),
            }
        )
    write_csv(
        output_dir / "mc_bond_return_table.csv",
        [
            "scenario",
            "coupon_target_annual",
            "bond_term_ticks",
            "runs",
            "median_bond_annualized_fee_yield",
            "p05_bond_annualized_fee_yield",
            "p95_bond_annualized_fee_yield",
            "median_coupon_coverage_ratio",
            "median_cumulative_fee_return_usd",
            "median_coupon_shortfall_usd",
            "median_realized_largest_component_share",
            "median_expected_verified_report_exposure",
        ],
        bond_table_rows,
    )
    write_csv(
        output_dir / "mc_failure_table.csv",
        list(failure_summary_rows[0].keys()) if failure_summary_rows else [],
        failure_summary_rows,
    )


def _load_font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text(draw, xy: tuple[int, int], text: str, font, fill=(25, 25, 25), anchor: str | None = None) -> None:
    kwargs = {"font": font, "fill": fill}
    if anchor is not None:
        kwargs["anchor"] = anchor
    draw.text(xy, text, **kwargs)


def _fmt_plot_value(value: float, *, percent: bool = False) -> str:
    if percent:
        return f"{value:.0f}%"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.0f}"


def _select_plot_key(
    quantiles: list[dict[str, object]],
    scenario: str,
    coupon: float,
    term: int,
) -> tuple[str, float, int] | None:
    keys = sorted(
        {
            (
                str(row["scenario"]),
                safe_float(row["coupon_target_annual"]),
                int(safe_float(row["bond_term_ticks"])),
            )
            for row in quantiles
        }
    )
    if not keys:
        return None
    exact = (scenario, float(coupon), int(term))
    if exact in keys:
        return exact
    scenario_keys = [key for key in keys if key[0] == scenario]
    if scenario_keys:
        return min(
            scenario_keys,
            key=lambda key: abs(key[1] - float(coupon)) + abs(key[2] - int(term)) / YEAR_TICKS,
        )
    return keys[0]


def _metric_rows(
    quantiles: list[dict[str, object]],
    metric: str,
    key: tuple[str, float, int],
) -> list[dict[str, float]]:
    scenario, coupon, term = key
    rows = []
    for row in quantiles:
        if row["metric"] != metric:
            continue
        if str(row["scenario"]) != scenario:
            continue
        if abs(safe_float(row["coupon_target_annual"]) - coupon) > 1e-9:
            continue
        if int(safe_float(row["bond_term_ticks"])) != term:
            continue
        rows.append(
            {
                "tick": safe_float(row["tick"]),
                "p05": safe_float(row["p05"]),
                "p50": safe_float(row["p50"]),
                "p95": safe_float(row["p95"]),
                "mean": safe_float(row["mean"]),
            }
        )
    return sorted(rows, key=lambda item: item["tick"])


def _draw_line_chart(
    path: Path,
    title: str,
    subtitle: str,
    series: list[dict[str, object]],
    *,
    percent: bool = False,
    y_label: str = "",
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1200, 720
    left, right, top, bottom = 105, 55, 88, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_font(28)
    subtitle_font = _load_font(17)
    axis_font = _load_font(15)
    label_font = _load_font(16)

    all_ticks: list[float] = []
    all_values: list[float] = []
    for item in series:
        rows = item.get("rows", [])
        for row in rows:
            all_ticks.append(float(row["tick"]))
            for col in ("p05", "p50", "p95"):
                if col in row:
                    all_values.append(float(row[col]) * (100.0 if percent else 1.0))
    if not all_ticks or not all_values:
        return
    x_min, x_max = min(all_ticks), max(all_ticks)
    if x_min == x_max:
        x_max = x_min + 1.0
    y_min = min(0.0, min(all_values))
    y_max = max(all_values)
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.08
    y_max += pad
    if percent and 80.0 <= max(all_values) <= 100.0 and y_min >= 0.0:
        y_max = min(max(y_max, 100.0), 100.0)

    def sx(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_w

    def sy(value: float) -> float:
        scaled = value * (100.0 if percent else 1.0)
        return top + plot_h - ((scaled - y_min) / (y_max - y_min)) * plot_h

    _draw_text(draw, (left, 30), title, title_font)
    _draw_text(draw, (left, 64), subtitle, subtitle_font, fill=(80, 80, 80))
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(70, 70, 70), width=1)

    for i in range(6):
        frac = i / 5
        y = top + plot_h - frac * plot_h
        value = y_min + frac * (y_max - y_min)
        draw.line((left, y, left + plot_w, y), fill=(220, 220, 220), width=1)
        _draw_text(draw, (left - 10, int(y)), _fmt_plot_value(value, percent=percent), axis_font, fill=(70, 70, 70), anchor="rm")
    for i in range(6):
        frac = i / 5
        x = left + frac * plot_w
        value = x_min + frac * (x_max - x_min)
        draw.line((x, top, x, top + plot_h), fill=(235, 235, 235), width=1)
        _draw_text(draw, (int(x), top + plot_h + 13), f"{value:.0f}", axis_font, fill=(70, 70, 70), anchor="ma")

    colors = [
        (40, 91, 214, 255),
        (215, 76, 53, 255),
        (36, 150, 97, 255),
        (130, 82, 190, 255),
    ]
    for idx, item in enumerate(series):
        rows = item.get("rows", [])
        if not rows:
            continue
        color = colors[idx % len(colors)]
        if item.get("band", False):
            upper = [(sx(row["tick"]), sy(row["p95"])) for row in rows]
            lower = [(sx(row["tick"]), sy(row["p05"])) for row in reversed(rows)]
            band = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            band_draw = ImageDraw.Draw(band, "RGBA")
            band_draw.polygon(upper + lower, fill=(color[0], color[1], color[2], 32))
            image.alpha_composite(band)
            draw = ImageDraw.Draw(image, "RGBA")
        points = [(sx(row["tick"]), sy(row["p50"])) for row in rows]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2, joint="curve")
        for x, y in points[:: max(1, len(points) // 12)]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        legend_x = left + 16 + idx * 260
        legend_y = height - 38
        draw.line((legend_x, legend_y, legend_x + 32, legend_y), fill=color, width=2)
        _draw_text(draw, (legend_x + 42, legend_y - 9), str(item["label"]), label_font)

    _draw_text(draw, (left + plot_w // 2, height - 24), "tick (weeks)", label_font, fill=(50, 50, 50), anchor="ma")
    if y_label:
        _draw_text(draw, (left + plot_w, top - 20), y_label, label_font, fill=(50, 50, 50), anchor="ra")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def _draw_failure_bars(
    path: Path,
    rows: list[dict[str, object]],
    *,
    coupon: float,
    term: int,
) -> None:
    from PIL import Image, ImageDraw

    filtered = [
        row
        for row in rows
        if abs(safe_float(row.get("coupon_target_annual")) - coupon) <= 1e-9
        and int(safe_float(row.get("bond_term_ticks"))) == term
    ]
    if not filtered:
        filtered = rows[:]
    filtered = filtered[:12]
    if not filtered:
        return

    width = 1200
    row_h = 58
    height = 160 + row_h * len(filtered)
    left, top, right = 300, 102, 55
    bar_w = width - left - right
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_font(28)
    subtitle_font = _load_font(17)
    axis_font = _load_font(15)
    label_font = _load_font(15)
    _draw_text(draw, (left, 30), "Monte Carlo Failure Rates", title_font)
    _draw_text(
        draw,
        (left, 64),
        f"Coupon {coupon * 100:.1f}%, term {term} ticks; each bar is share of runs ending with the failure condition.",
        subtitle_font,
        fill=(80, 80, 80),
    )
    metrics = [
        ("coupon_shortfall_rate", "shortfall", (210, 74, 58, 255)),
        ("liquidity_leakage_rate", "leakage", (238, 157, 52, 255)),
        ("household_cash_stress_rate", "cash stress", (82, 125, 206, 255)),
    ]
    for i in range(6):
        x = left + (i / 5) * bar_w
        draw.line((x, top - 10, x, height - 56), fill=(230, 230, 230), width=1)
        _draw_text(draw, (int(x), height - 38), f"{i * 20}%", axis_font, fill=(75, 75, 75), anchor="ma")
    for row_idx, row in enumerate(filtered):
        y0 = top + row_idx * row_h
        name = str(row.get("scenario", ""))[:34]
        _draw_text(draw, (left - 12, y0 + 22), name, label_font, fill=(40, 40, 40), anchor="rm")
        for metric_idx, (metric, _label, color) in enumerate(metrics):
            value = max(0.0, min(1.0, safe_float(row.get(metric))))
            y = y0 + 8 + metric_idx * 15
            draw.rectangle((left, y, left + value * bar_w, y + 10), fill=color)
    legend_y = height - 22
    x = left
    for metric, label, color in metrics:
        draw.rectangle((x, legend_y - 10, x + 18, legend_y + 2), fill=color)
        _draw_text(draw, (x + 26, legend_y - 13), label, label_font)
        x += 160
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def write_png_figures(
    output_dir: Path,
    quantiles: list[dict[str, object]],
    failure_summary_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    key = _select_plot_key(quantiles, args.plot_scenario, args.plot_coupon, args.plot_term)
    if key is None:
        return
    scenario, coupon, term = key
    suffix = f"{scenario}, coupon {coupon * 100:.1f}%, term {term} ticks"
    figures = output_dir / "figures"

    _draw_line_chart(
        figures / "fig_bond_apr_over_time.png",
        "Projected Bond Fee Yield Over Time",
        suffix,
        [
            {
                "label": "median; band = 5th-95th percentile",
                "rows": _metric_rows(quantiles, "bond_annualized_fee_yield", key),
                "band": True,
            }
        ],
        percent=True,
        y_label="annualized yield",
    )
    _draw_line_chart(
        figures / "fig_coupon_coverage_over_time.png",
        "Coupon Coverage Over Time",
        suffix,
        [
            {
                "label": "median; band = 5th-95th percentile",
                "rows": _metric_rows(quantiles, "bond_coupon_coverage_ratio", key),
                "band": True,
            }
        ],
        percent=False,
        y_label="coverage ratio",
    )
    _draw_line_chart(
        figures / "fig_cumulative_fee_return_over_time.png",
        "Cumulative Fee Return to LP/Bond Funders",
        suffix,
        [
            {
                "label": "median; band = 5th-95th percentile",
                "rows": _metric_rows(quantiles, "bond_cumulative_fee_return_usd", key),
                "band": True,
            }
        ],
        percent=False,
        y_label="USD",
    )
    _draw_line_chart(
        figures / "fig_network_connectivity_over_time.png",
        "Scaling Connected Networks of Pools",
        suffix,
        [
            {
                "label": "potential listings graph",
                "rows": _metric_rows(quantiles, "potential_largest_component_share", key),
                "band": False,
            },
            {
                "label": "realized swap graph",
                "rows": _metric_rows(quantiles, "realized_largest_component_share", key),
                "band": False,
            },
        ],
        percent=True,
        y_label="largest component share",
    )
    _draw_line_chart(
        figures / "fig_report_exposure_over_time.png",
        "Projected Verified Report Exposure",
        suffix,
        [
            {
                "label": "median; band = 5th-95th percentile",
                "rows": _metric_rows(quantiles, "expected_verified_report_exposure", key),
                "band": True,
            }
        ],
        percent=False,
        y_label="expected reports",
    )
    _draw_failure_bars(
        figures / "fig_failure_rates.png",
        failure_summary_rows,
        coupon=coupon,
        term=term,
    )


def public_output_privacy_check(output_dir: Path) -> None:
    bad_patterns = ("0x", "[redacted-name", "@")
    for path in output_dir.glob("mc_*"):
        if path.suffix not in {".csv", ".tex"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        for pattern in bad_patterns:
            if pattern in lowered:
                raise RuntimeError(f"Unexpected private-looking token {pattern!r} in {path}")


def relative_error(engine_value: float, empirical_value: float) -> float:
    if abs(empirical_value) <= 1e-9:
        return abs(engine_value - empirical_value)
    return abs(engine_value - empirical_value) / abs(empirical_value)


def validation_row(
    *,
    tier: str,
    moment: str,
    category: str,
    empirical_value: float | str,
    engine_values: list[float],
    tolerance: float | None,
    binding: bool,
) -> dict[str, object]:
    engine_mean = mean(engine_values)
    engine_p05 = percentile(engine_values, 0.05)
    engine_p50 = percentile(engine_values, 0.50)
    engine_p95 = percentile(engine_values, 0.95)
    if empirical_value == "" or tolerance is None:
        return {
            "tier": tier,
            "moment": moment,
            "category": category,
            "empirical_sarafu_moment": empirical_value,
            "engine_mean": engine_mean,
            "engine_p05": engine_p05,
            "engine_p50": engine_p50,
            "engine_p95": engine_p95,
            "absolute_error": "",
            "relative_error": "",
            "tolerance": "",
            "binding": int(binding),
            "validation_status": "reported",
        }
    empirical_float = safe_float(empirical_value)
    abs_error = abs(engine_p50 - empirical_float)
    rel_error = relative_error(engine_p50, empirical_float)
    status = "pass" if rel_error <= tolerance else "review"
    if status == "review" and category in {"settlement", "repayment"} and rel_error > tolerance * 2.0:
        status = "fail"
    return {
        "tier": tier,
        "moment": moment,
        "category": category,
        "empirical_sarafu_moment": empirical_float,
        "engine_mean": engine_mean,
        "engine_p05": engine_p05,
        "engine_p50": engine_p50,
        "engine_p95": engine_p95,
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "tolerance": tolerance,
        "binding": int(binding),
        "validation_status": status,
    }


def engine_validation_moments(
    calibration: Calibration,
    summaries: list[dict[str, object]],
    *,
    ticks: int,
) -> list[dict[str, object]]:
    targets = {row["tier"]: row for row in empirical_tier_targets(calibration, ticks)}
    rows: list[dict[str, object]] = []
    tolerance_by_moment = {
        "pool_count": 0.05,
        "total_swap_events_horizon": 0.20,
        "same_token_return_coverage": 0.20,
        "cash_return_coverage": 0.25,
        "voucher_return_coverage": 0.20,
        "borrow_proxy_closure": 0.25,
        "backing_liquidity_inflow": 0.30,
        "total_verified_report_exposure_horizon": 0.30,
    }
    categories = {
        "pool_count": "tier_mix",
        "total_swap_events_horizon": "activity",
        "same_token_return_coverage": "repayment",
        "cash_return_coverage": "repayment",
        "voucher_return_coverage": "repayment",
        "borrow_proxy_closure": "repayment",
        "backing_liquidity_inflow": "liquidity",
        "total_verified_report_exposure_horizon": "reporting",
    }
    engine_field = {
        "pool_count": "engine_pool_count_{tier}",
        "total_swap_events_horizon": "engine_swap_events_{tier}",
        "same_token_return_coverage": "engine_same_token_return_coverage_{tier}",
        "cash_return_coverage": "engine_cash_return_coverage_{tier}",
        "voucher_return_coverage": "engine_voucher_return_coverage_{tier}",
        "borrow_proxy_closure": "engine_borrow_proxy_closure_{tier}",
        "backing_liquidity_inflow": "engine_backing_liquidity_inflow_{tier}",
        "total_verified_report_exposure_horizon": "engine_report_exposure_{tier}",
    }
    for tier in TIER_ORDER:
        target = targets.get(tier, {})
        for moment, tolerance in tolerance_by_moment.items():
            field = engine_field[moment].format(tier=tier)
            rows.append(
                validation_row(
                    tier=tier,
                    moment=moment,
                    category=categories[moment],
                    empirical_value=safe_float(target.get(moment)),
                    engine_values=[safe_float(row.get(field)) for row in summaries],
                    tolerance=tolerance,
                    binding=True,
                )
            )

    empirical_total_swaps = sum(safe_float(row["total_swap_events_horizon"]) for row in targets.values())
    empirical_total_reports = sum(
        safe_float(row["total_verified_report_exposure_horizon"]) for row in targets.values()
    )
    empirical_backing = sum(safe_float(row["backing_liquidity_inflow"]) for row in targets.values())
    empirical_borrow_events = sum(
        pool.borrow_proxy_matured_events for pool in calibration.pool_rows
    )
    empirical_weighted_borrow = 0.0
    if empirical_borrow_events > 0.0:
        empirical_weighted_borrow = sum(
            pool.borrow_proxy_matured_events * pool.borrow_proxy_matured_return_rate
            for pool in calibration.pool_rows
        ) / empirical_borrow_events
    current_circulation = calibration.voucher_circulation_baselines.get("trailing_90d", {})
    stable_anchors = calibration.stable_dependency_anchors
    empirical_metadata_stable_share = stable_anchors.get(
        "net_positive_flow_balance_stable_share_aggregate", {}
    ).get("value", "")
    empirical_strict_stable_share = stable_anchors.get(
        "net_positive_flow_balance_stable_share_aggregate_strict_1ksh", {}
    ).get("value", "")
    empirical_flow_stable_share = (
        empirical_strict_stable_share
        if empirical_strict_stable_share != ""
        else empirical_metadata_stable_share
    )
    empirical_stable_involved_swap_share = (
        safe_float(current_circulation.get("voucher_to_stable_share"))
        + safe_float(current_circulation.get("stable_to_voucher_share"))
    )
    productive_credit_all = calibration.productive_credit_by_tier.get("all", {})
    empirical_productive_voucher_share = safe_float(
        productive_credit_all.get("loan_induced_voucher_deposit_share_default")
    )
    empirical_producer_cash_deposit = sum(
        safe_float(row.get("cash_deposit_value"))
        for row in calibration.producer_deposit_by_tier.values()
    )
    empirical_producer_voucher_deposit = sum(
        safe_float(row.get("voucher_deposit_value"))
        for row in calibration.producer_deposit_by_tier.values()
    )
    empirical_producer_deposit_total = (
        empirical_producer_cash_deposit + empirical_producer_voucher_deposit
    )
    empirical_producer_voucher_deposit_share = (
        empirical_producer_voucher_deposit / max(1e-9, empirical_producer_deposit_total)
    )
    aggregate_specs = [
        (
            "all",
            "total_swap_activity",
            "activity",
            empirical_total_swaps,
            [safe_float(row.get("engine_swap_events_strong")) + safe_float(row.get("engine_swap_events_moderate")) + safe_float(row.get("engine_swap_events_weak")) for row in summaries],
            0.20,
            True,
        ),
        (
            "all",
            "current_voucher_to_voucher_swap_share",
            "settlement",
            current_circulation.get("voucher_to_voucher_share", ""),
            [
                safe_float(row.get("swap_count_vchr_to_vchr_total"))
                / max(1e-9, safe_float(row.get("transactions_total")))
                for row in summaries
            ],
            None,
            False,
        ),
        (
            "all",
            "current_voucher_to_stable_swap_share",
            "settlement",
            current_circulation.get("voucher_to_stable_share", ""),
            [
                safe_float(row.get("swap_count_vchr_to_usd_total"))
                / max(1e-9, safe_float(row.get("transactions_total")))
                for row in summaries
            ],
            None,
            False,
        ),
        (
            "all",
            "current_stable_to_voucher_swap_share",
            "settlement",
            current_circulation.get("stable_to_voucher_share", ""),
            [
                safe_float(row.get("swap_count_usd_to_vchr_total"))
                / max(1e-9, safe_float(row.get("transactions_total")))
                for row in summaries
            ],
            None,
            False,
        ),
        (
            "all",
            "current_stable_involved_swap_share",
            "settlement",
            empirical_stable_involved_swap_share,
            [
                (
                    safe_float(row.get("swap_count_vchr_to_usd_total"))
                    + safe_float(row.get("swap_count_usd_to_vchr_total"))
                )
                / max(1e-9, safe_float(row.get("transactions_total")))
                for row in summaries
            ],
            None,
            False,
        ),
        (
            "all",
            "gross_stable_flow_share_strict_1ksh",
            "settlement",
            stable_anchors.get("gross_stable_flow_share_strict_1ksh", {}).get("value", ""),
            [safe_float(row.get("engine_gross_flow_stable_share_total")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "gross_stable_flow_share_metadata_weighted",
            "settlement",
            stable_anchors.get("gross_stable_flow_share", {}).get("value", ""),
            [safe_float(row.get("engine_gross_flow_stable_share_total")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "positive_net_flow_stable_share_strict_1ksh",
            "settlement",
            empirical_flow_stable_share,
            [safe_float(row.get("engine_positive_net_flow_stable_share_total")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "positive_net_flow_stable_share_metadata_weighted",
            "settlement",
            empirical_metadata_stable_share,
            [safe_float(row.get("engine_positive_net_flow_stable_share_total")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "active_pool_stable_value_share_inventory_snapshot",
            "settlement",
            "",
            [safe_float(row.get("stable_value_share_in_active_pools")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "active_pool_voucher_value_share_inventory_snapshot",
            "settlement",
            "",
            [safe_float(row.get("voucher_value_share_in_active_pools")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "active_pool_stable_to_voucher_value_ratio",
            "settlement",
            "",
            [safe_float(row.get("stable_to_voucher_value_ratio_in_active_pools")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "voucher_to_voucher_settlement_volume",
            "settlement",
            "",
            [safe_float(row.get("swap_volume_vchr_to_vchr_total")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "voucher_to_stable_volume",
            "settlement",
            "",
            [safe_float(row.get("swap_volume_vchr_to_usd_total")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "stable_to_voucher_volume",
            "settlement",
            "",
            [safe_float(row.get("swap_volume_usd_to_vchr_total")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "repayment_to_borrow_proxy_closure",
            "repayment",
            empirical_weighted_borrow,
            [safe_float(row.get("engine_repayment_borrow_proxy_closure")) for row in summaries],
            0.25,
            True,
        ),
        (
            "all",
            "raw_engine_repayment_loan_flow_ratio",
            "repayment",
            "",
            [safe_float(row.get("raw_engine_repayment_loan_flow_ratio")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "productive_credit_voucher_deposit_share",
            "productive_capacity",
            empirical_productive_voucher_share,
            [
                safe_float(row.get("productive_credit_voucher_deposit_usd_total"))
                / max(1e-9, safe_float(row.get("productive_credit_inflow_usd_total")))
                for row in summaries
            ],
            None,
            False,
        ),
        (
            "all",
            "producer_voucher_deposit_share",
            "liquidity",
            empirical_producer_voucher_deposit_share,
            [
                safe_float(row.get("producer_deposit_voucher_usd_total"))
                / max(
                    1e-9,
                    safe_float(row.get("producer_deposit_stable_usd_total"))
                    + safe_float(row.get("producer_deposit_voucher_usd_total")),
                )
                for row in summaries
            ],
            None,
            False,
        ),
        (
            "all",
            "backing_liquidity_inflow_scale",
            "liquidity",
            empirical_backing,
            [
                safe_float(row.get("initial_stable_total"))
                + safe_float(row.get("stable_onramp_usd_total"))
                + safe_float(row.get("bond_principal_usd"))
                for row in summaries
            ],
            0.30,
            True,
        ),
        (
            "all",
            "report_exposure_projection",
            "reporting",
            empirical_total_reports,
            [
                safe_float(row.get("engine_report_exposure_strong"))
                + safe_float(row.get("engine_report_exposure_moderate"))
                + safe_float(row.get("engine_report_exposure_weak"))
                for row in summaries
            ],
            0.30,
            True,
        ),
        (
            "all",
            "route_success_pass_rate",
            "routing",
            "",
            [safe_float(row.get("route_success_rate_cumulative")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "potential_largest_component_share",
            "connectivity",
            "",
            [safe_float(row.get("potential_largest_component_share")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "realized_largest_component_share",
            "connectivity",
            "",
            [safe_float(row.get("realized_largest_component_share")) for row in summaries],
            None,
            False,
        ),
        (
            "all",
            "realized_edge_concentration_top_share",
            "connectivity",
            "",
            [safe_float(row.get("realized_edge_top_share")) for row in summaries],
            None,
            False,
        ),
    ]
    for tier, moment, category, empirical, values, tolerance, binding in aggregate_specs:
        rows.append(
            validation_row(
                tier=tier,
                moment=moment,
                category=category,
                empirical_value=empirical,
                engine_values=values,
                tolerance=tolerance,
                binding=binding,
            )
        )
    return rows


def engine_validation_status(rows: list[dict[str, object]]) -> str:
    binding_rows = [row for row in rows if int(safe_float(row.get("binding"))) == 1]
    if any(row.get("validation_status") == "fail" for row in binding_rows):
        return "fail"
    if any(row.get("validation_status") == "review" for row in binding_rows):
        return "review"
    return "pass"


def write_engine_validation_table(output_dir: Path, rows: list[dict[str, object]], status: str) -> None:
    display = [
        row
        for row in rows
        if row["moment"] in {
            "pool_count",
            "total_swap_events_horizon",
            "same_token_return_coverage",
            "cash_return_coverage",
            "voucher_return_coverage",
            "report_exposure_projection",
            "route_success_pass_rate",
            "realized_largest_component_share",
        }
    ][:24]
    lines = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Tier & Moment & Sarafu & Engine p50 & Tolerance & Status \\",
        r"\midrule",
    ]
    for row in display:
        empirical = row["empirical_sarafu_moment"]
        empirical_text = "--" if empirical == "" else f"{safe_float(empirical):.3g}"
        tolerance = row["tolerance"]
        tolerance_text = "--" if tolerance == "" else fmt_pct(tolerance)
        lines.append(
            "{tier} & {moment} & {empirical} & {engine} & {tolerance} & {status} \\\\".format(
                tier=latex_escape(row["tier"]),
                moment=latex_escape(row["moment"]),
                empirical=empirical_text,
                engine=f"{safe_float(row['engine_p50']):.3g}",
                tolerance=tolerance_text,
                status=latex_escape(row["validation_status"]),
            )
        )
    lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{6}}{{l}}{{Overall validation status: {latex_escape(status)}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
        ]
    )
    (output_dir / "engine_validation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw_grouped_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int, int]]],
    *,
    percent: bool = False,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1200, 720
    left, right, top, bottom = 110, 60, 92, 105
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_font(28)
    subtitle_font = _load_font(17)
    axis_font = _load_font(15)
    label_font = _load_font(16)
    values = [value * (100.0 if percent else 1.0) for _, vals, _ in series for value in vals]
    y_max = max(values) if values else 1.0
    y_max = y_max * 1.15 if y_max > 0 else 1.0
    _draw_text(draw, (left, 30), title, title_font)
    _draw_text(draw, (left, 64), subtitle, subtitle_font, fill=(80, 80, 80))
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(70, 70, 70), width=1)
    for i in range(6):
        frac = i / 5
        y = top + plot_h - frac * plot_h
        value = frac * y_max
        draw.line((left, y, left + plot_w, y), fill=(225, 225, 225), width=1)
        _draw_text(draw, (left - 10, int(y)), _fmt_plot_value(value, percent=percent), axis_font, fill=(70, 70, 70), anchor="rm")
    group_w = plot_w / max(1, len(labels))
    bar_w = min(70, group_w / max(2, len(series) + 1))
    for idx, label in enumerate(labels):
        center = left + group_w * (idx + 0.5)
        start_x = center - (bar_w * len(series)) / 2
        for sidx, (_, vals, color) in enumerate(series):
            value = (vals[idx] if idx < len(vals) else 0.0) * (100.0 if percent else 1.0)
            h = (value / y_max) * plot_h if y_max > 0 else 0.0
            x0 = start_x + sidx * bar_w
            draw.rectangle((x0, top + plot_h - h, x0 + bar_w * 0.82, top + plot_h), fill=color)
        _draw_text(draw, (int(center), top + plot_h + 20), label, axis_font, fill=(55, 55, 55), anchor="ma")
    legend_x = left
    legend_y = height - 40
    for label, _, color in series:
        draw.rectangle((legend_x, legend_y - 10, legend_x + 20, legend_y + 2), fill=color)
        _draw_text(draw, (legend_x + 28, legend_y - 14), label, label_font)
        legend_x += 230
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def write_engine_validation_figures(output_dir: Path, rows: list[dict[str, object]]) -> None:
    by_key = {(row["tier"], row["moment"]): row for row in rows}
    labels = [tier.title() for tier in TIER_ORDER]
    empirical_activity = [
        safe_float(by_key.get((tier, "total_swap_events_horizon"), {}).get("empirical_sarafu_moment"))
        for tier in TIER_ORDER
    ]
    engine_activity = [
        safe_float(by_key.get((tier, "total_swap_events_horizon"), {}).get("engine_p50"))
        for tier in TIER_ORDER
    ]
    _draw_grouped_bar_chart(
        output_dir / "fig_engine_vs_sarafu_activity.png",
        "Engine Validation: Swap Activity by Tier",
        "Sarafu empirical horizon moments versus real engine median across runs.",
        labels,
        [
            ("Sarafu empirical", empirical_activity, (51, 102, 204, 255)),
            ("Engine p50", engine_activity, (219, 118, 45, 255)),
        ],
    )
    coverage_labels = []
    empirical_cov = []
    engine_cov = []
    for tier in TIER_ORDER:
        for moment in ("same_token_return_coverage", "cash_return_coverage", "voucher_return_coverage"):
            row = by_key.get((tier, moment), {})
            coverage_labels.append(f"{tier[:3]} {moment.split('_')[0]}")
            empirical_cov.append(safe_float(row.get("empirical_sarafu_moment")))
            engine_cov.append(safe_float(row.get("engine_p50")))
    _draw_grouped_bar_chart(
        output_dir / "fig_engine_vs_sarafu_return_coverage.png",
        "Engine Validation: Return Coverage Priors",
        "Sarafu repayment/return calibration against engine priors used in counterfactuals.",
        coverage_labels,
        [
            ("Sarafu empirical", empirical_cov, (51, 102, 204, 255)),
            ("Engine prior", engine_cov, (54, 160, 103, 255)),
        ],
        percent=True,
    )


def write_engine_validation_notes(
    output_dir: Path,
    *,
    status: str,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
) -> None:
    failed = [row for row in rows if row["validation_status"] == "fail"]
    review = [row for row in rows if row["validation_status"] == "review"]
    reported = [row for row in rows if row["validation_status"] == "reported"]
    lines = [
        "# Engine Validation Against Sarafu Calibration",
        "",
        f"- Scenario: `sarafu_engine_validation`.",
        f"- Runs: {args.runs}.",
        f"- Horizon: {args.ticks} weekly ticks.",
        f"- Overall status: `{status}`.",
        "- Route success and connectivity are reported as non-binding diagnostics until a direct empirical route graph is calibrated.",
        "- Template validation remains the empirical calibration layer; this engine validation is the bridge to robust counterfactual bond-injection frontiers.",
        "",
        "## Gate Interpretation",
        "",
    ]
    if status == "pass":
        lines.append("All binding Sarafu calibration moments are within tolerance; proceed to the bond-issuer frontier.")
    elif status == "review":
        lines.append("One or more non-critical or calibration-sensitive moments are outside tolerance; review before treating frontier outputs as paper headline estimates.")
    else:
        lines.append("One or more settlement/repayment moments materially miss calibration; do not use frontier outputs as paper headline estimates until recalibrated.")
    lines.extend(
        [
            "",
            f"- Binding failures: {len(failed)}.",
            f"- Binding review rows: {len(review)}.",
            f"- Non-binding reported diagnostics: {len(reported)}.",
            "",
            "## Paper Text Hook",
            "",
            "Sarafu provides empirical calibration; the engine reproduces key Sarafu settlement moments subject to this validation gate; then the bond-issuer frontier estimates safe non-extractive principal under current and larger connected commitment-pooling networks.",
        ]
    )
    (output_dir / "paper_integration_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_validation_shard(job: dict[str, object], shard_root: Path, config_hash: str) -> dict[str, object] | None:
    return shard_result_from_files(
        kind="validation",
        job=job,
        shard_root=shard_root,
        config_hash=config_hash,
        csv_names=("bond_rows", "network_rows", "failure_rows", "summary_rows"),
    )


def run_validation_shard(
    job: dict[str, object],
    args_data: dict[str, object],
    calibration: Calibration,
    shard_root_text: str,
    resume: bool,
) -> dict[str, object]:
    shard_root = Path(shard_root_text)
    job_dir = shard_job_dir(shard_root, "validation", str(job["job_id"]))
    config_hash = str(job["config_hash"])
    if resume:
        completed = load_validation_shard(job, shard_root, config_hash)
        if completed is not None:
            return completed
    started_at = time.time()
    try:
        args = namespace_from_payload(args_data)
        apply_unit_normalization_context(args, calibration)
        apply_network_context(
            args,
            calibration=calibration,
            network_scale="current",
            principal_ratio=0.0,
            principal_usd=0.0,
            service_share=0.0,
            certification_policy="none",
        )
        args._current_certified_pool_count = 0.0
        args._current_certified_capacity_usd = 0.0
        configure_sarafu_activity_controls(args, calibration, int(args.ticks), "validation")
        run_idx = int(job["run_idx"])
        seed = int(job["seed"])
        term_ticks = int(job["term_ticks"])
        print(
            f"[{run_idx}/{args.runs}] scenario=sarafu_engine_validation ticks={args.ticks} run={run_idx} seed={seed}",
            flush=True,
        )
        args._progress_run_position = run_idx
        args._progress_total_runs = int(args.runs)
        bond_rows, network_rows, failure_rows, summary = run_one(
            scenario="sarafu_engine_validation",
            coupon=0.0,
            term_ticks=term_ticks,
            run_index=run_idx,
            seed=seed,
            args=args,
            calibration=calibration,
        )
        summary_rows = [summary]
        write_rows(job_dir / "bond_rows.csv", bond_rows)
        write_rows(job_dir / "network_rows.csv", network_rows)
        write_rows(job_dir / "failure_rows.csv", failure_rows)
        write_rows(job_dir / "summary_rows.csv", summary_rows)
        files = {
            "bond_rows": len(bond_rows),
            "network_rows": len(network_rows),
            "failure_rows": len(failure_rows),
            "summary_rows": len(summary_rows),
        }
        write_shard_manifest(
            job_dir,
            job=job,
            config_hash=config_hash,
            status="completed",
            files=files,
            started_at=started_at,
        )
        return {
            "status": "completed",
            "job": job,
            "shard_dir": str(job_dir),
            "bond_rows": bond_rows,
            "network_rows": network_rows,
            "failure_rows": failure_rows,
            "summary_rows": summary_rows,
        }
    except BaseException:
        error = traceback.format_exc()
        write_shard_manifest(
            job_dir,
            job=job,
            config_hash=config_hash,
            status="failed",
            error=error,
            started_at=started_at,
        )
        return failed_result(job, error)


def write_engine_validation_aggregate(
    args: argparse.Namespace,
    calibration: Calibration,
    output_dir: Path,
    *,
    bond_rows: list[dict[str, object]],
    network_rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    summaries: list[dict[str, object]],
    partial: bool,
) -> str:
    bond_rows = sorted_run_rows(bond_rows)
    network_rows = sorted_run_rows(network_rows)
    failures = sorted_run_rows(failures)
    summaries = sorted_run_rows(summaries)
    rows = engine_validation_moments(calibration, summaries, ticks=int(args.ticks))
    status = engine_validation_status(rows)
    error_rows = [row for row in rows if row["validation_status"] in {"review", "fail"}]
    summary_rows = [
        {
            "scenario": "sarafu_engine_validation",
            "runs": len(summaries),
            "ticks": int(args.ticks),
            "status": status,
            "binding_pass_count": sum(1 for row in rows if row["binding"] == 1 and row["validation_status"] == "pass"),
            "binding_review_count": sum(1 for row in rows if row["binding"] == 1 and row["validation_status"] == "review"),
            "binding_fail_count": sum(1 for row in rows if row["binding"] == 1 and row["validation_status"] == "fail"),
            "reported_diagnostic_count": sum(1 for row in rows if row["validation_status"] == "reported"),
        }
    ]
    suffix = ".partial.csv" if partial else ".csv"
    write_csv(output_dir / f"engine_validation_moments{suffix}", list(rows[0].keys()) if rows else [], rows)
    write_csv(output_dir / f"engine_validation_errors{suffix}", list(rows[0].keys()) if rows else [], error_rows)
    write_csv(output_dir / f"engine_validation_summary{suffix}", list(summary_rows[0].keys()), summary_rows)
    write_csv(output_dir / f"engine_validation_run_summary{suffix}", list(summaries[0].keys()) if summaries else [], summaries)
    write_csv(output_dir / f"engine_validation_bond_timeseries{suffix}", list(bond_rows[0].keys()) if bond_rows else [], bond_rows)
    write_csv(output_dir / f"engine_validation_network_timeseries{suffix}", list(network_rows[0].keys()) if network_rows else [], network_rows)
    write_csv(output_dir / f"engine_validation_failure_metrics{suffix}", list(failures[0].keys()) if failures else [], failures)
    audit_rows = calibration_activity_audit_rows(
        args,
        calibration,
        summaries,
        scenario="sarafu_engine_validation",
        prefix="validation",
    )
    write_csv(
        output_dir / f"calibration_activity_audit{suffix}",
        list(audit_rows[0].keys()) if audit_rows else [],
        audit_rows,
    )
    if not partial:
        write_engine_validation_table(output_dir, rows, status)
        if not args.no_png:
            write_engine_validation_figures(output_dir, rows)
        write_engine_validation_notes(output_dir, status=status, args=args, rows=rows)
    return status


def validation_results_to_rows(
    results: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    bond_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for result in results:
        bond_rows.extend(result.get("bond_rows", []))
        network_rows.extend(result.get("network_rows", []))
        failures.extend(result.get("failure_rows", []))
        summaries.extend(result.get("summary_rows", []))
    return bond_rows, network_rows, failures, summaries


def run_sarafu_engine_validation(args: argparse.Namespace, calibration: Calibration, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_network_context(
        args,
        calibration=calibration,
        network_scale="current",
        principal_ratio=0.0,
        principal_usd=0.0,
        service_share=0.0,
        certification_policy="none",
    )
    args._current_certified_pool_count = 0.0
    args._current_certified_capacity_usd = 0.0
    # Validation should target current Sarafu activity rather than ratcheting
    # upward from recent engine activity. The scenario uses shallow pool swaps,
    # disables NOAM overlay/clearing, and disables recurring stable growth.
    # Denser NOAM routing/clearing is reserved for scaled frontier networks
    # after this current-network gate passes.
    configure_sarafu_activity_controls(args, calibration, int(args.ticks), "validation")
    # Sarafu had substantial historical backing/liquidity inflow. This shock is
    # validation-only historical backing, not a bond/LP injection.
    term_ticks = selected_terms(args)[0] if selected_terms(args) else int(args.ticks)
    jobs = [
        {
            "kind": "validation_run",
            "job_id": f"validation_run_{run_idx:06d}",
            "run_idx": run_idx,
            "seed": int(args.seed) + run_idx,
            "term_ticks": term_ticks,
        }
        for run_idx in range(1, int(args.runs) + 1)
    ]

    def write_partial(results: list[dict[str, object]]) -> None:
        b, n, f, s = validation_results_to_rows(results)
        write_engine_validation_aggregate(
            args,
            calibration,
            output_dir,
            bond_rows=b,
            network_rows=n,
            failures=f,
            summaries=s,
            partial=True,
        )

    results, failed = run_sharded_jobs(
        label="engine validation",
        kind="validation",
        jobs=jobs,
        args=args,
        calibration=calibration,
        output_dir=output_dir,
        worker=run_validation_shard,
        load_completed=load_validation_shard,
        on_progress=write_partial,
    )
    bond_rows, network_rows, failures, summaries = validation_results_to_rows(results)
    status = write_engine_validation_aggregate(
        args,
        calibration,
        output_dir,
        bond_rows=bond_rows,
        network_rows=network_rows,
        failures=failures,
        summaries=summaries,
        partial=False,
    )
    print(f"Wrote engine validation artifacts to {output_dir} (status={status})")
    return 1 if failed or len(summaries) != int(args.runs) else 0


def service_coverage(row: dict[str, object]) -> float:
    if "issuer_service_coverage_ratio" in row:
        return safe_float(row.get("issuer_service_coverage_ratio"))
    principal = safe_float(row.get("bond_principal_usd"))
    if principal <= 1e-9:
        return 999.0
    tick = safe_float(row.get("tick"))
    term = max(1.0, safe_float(row.get("bond_term_ticks")))
    coupon = safe_float(row.get("coupon_target_annual"))
    principal_due = principal * min(1.0, tick / term)
    coupon_due = principal * coupon * (min(tick, term) / YEAR_TICKS)
    due = principal_due + coupon_due
    return safe_float(row.get("bond_cumulative_fee_return_usd")) / max(1e-9, due)


def scheduled_payment_coverage(row: dict[str, object]) -> float:
    if "issuer_paid_coverage_ratio" in row:
        return safe_float(row.get("issuer_paid_coverage_ratio"))
    scheduled_due = safe_float(row.get("issuer_scheduled_debt_service_due_usd"))
    if scheduled_due <= 1e-9:
        principal = safe_float(row.get("bond_principal_usd"))
        return 999.0 if principal <= 1e-9 else 0.0
    return safe_float(row.get("issuer_actual_bondholder_payment_usd")) / scheduled_due


def service_cash_headroom(row: dict[str, object]) -> float:
    if "issuer_service_cash_headroom_ratio" in row:
        return safe_float(row.get("issuer_service_cash_headroom_ratio"))
    scheduled_due = safe_float(row.get("issuer_scheduled_debt_service_due_usd"))
    if scheduled_due <= 1e-9:
        principal = safe_float(row.get("bond_principal_usd"))
        return 999.0 if principal <= 1e-9 else 0.0
    recovered = max(
        safe_float(row.get("lender_recovered_stable_usd_total")),
        safe_float(row.get("bond_cumulative_fee_return_usd")),
    )
    return recovered / scheduled_due


def available_service_cash_headroom(row: dict[str, object]) -> float:
    if "issuer_available_service_cash_headroom_ratio" in row:
        return safe_float(row.get("issuer_available_service_cash_headroom_ratio"))
    return service_cash_headroom(row)


def summarize_frontier_cell(
    rows: list[dict[str, object]],
    baseline: dict[str, float],
    route_success_floor: float,
    route_success_mode: str,
) -> dict[str, object]:
    principal_values = [safe_float(row.get("bond_principal_usd")) for row in rows]
    service_values = [service_coverage(row) for row in rows]
    scheduled_payment_values = [scheduled_payment_coverage(row) for row in rows]
    service_cash_headroom_values = [service_cash_headroom(row) for row in rows]
    available_service_cash_headroom_values = [available_service_cash_headroom(row) for row in rows]
    issuer_lockbox_surplus_values = [
        safe_float(row.get("issuer_lockbox_surplus_usd")) for row in rows
    ]
    issuer_excess_recovered_stable_values = [
        safe_float(row.get("issuer_excess_recovered_stable_usd")) for row in rows
    ]
    issuer_operating_surplus_values = [
        safe_float(row.get("issuer_operating_surplus_usd")) for row in rows
    ]
    issuer_operating_surplus_headroom_values = [
        safe_float(row.get("issuer_operating_surplus_headroom_ratio")) for row in rows
    ]
    issuer_fee_waterfall_ops_values = [
        safe_float(row.get("issuer_fee_waterfall_ops_usd")) for row in rows
    ]
    issuer_sustainability_resources_values = [
        safe_float(row.get("issuer_sustainability_resources_usd")) for row in rows
    ]
    issuer_sustainability_headroom_values = [
        safe_float(row.get("issuer_sustainability_headroom_ratio")) for row in rows
    ]
    route_values = [safe_float(row.get("route_success_rate_cumulative")) for row in rows]
    fixed_route_values = [safe_float(row.get("route_fixed_success_rate_cumulative")) for row in rows]
    substitution_route_values = [
        safe_float(row.get("route_substitution_success_rate_cumulative")) for row in rows
    ]
    noam_routing_swaps_values = [safe_float(row.get("noam_routing_swaps_total")) for row in rows]
    noam_clearing_swaps_values = [safe_float(row.get("noam_clearing_swaps_total")) for row in rows]
    noam_routing_volume_values = [
        safe_float(row.get("noam_routing_volume_usd_total")) for row in rows
    ]
    noam_clearing_volume_values = [
        safe_float(row.get("noam_clearing_volume_usd_total")) for row in rows
    ]
    noam_routing_fee_values = [safe_float(row.get("noam_routing_fee_usd_total")) for row in rows]
    noam_clearing_fee_values = [
        safe_float(row.get("noam_clearing_fee_usd_total")) for row in rows
    ]
    noam_clearing_cycle_success_values = [
        safe_float(row.get("noam_clearing_cycle_success_rate_cumulative")) for row in rows
    ]
    noam_clearing_cycle_value_values = [
        safe_float(row.get("noam_clearing_cycle_value_usd_total")) for row in rows
    ]
    stress_values = [safe_float(row.get("household_cash_stress_ratio")) for row in rows]
    consumer_stress_values = [
        safe_float(row.get("consumer_stable_reserve_stress_ratio", row.get("household_cash_stress_ratio")))
        for row in rows
    ]
    community_stress_values = [
        safe_float(row.get("community_stable_reserve_stress_ratio", row.get("household_cash_stress_ratio")))
        for row in rows
    ]
    leakage_values = [safe_float(row.get("stable_liquidity_leakage_ratio_cumulative")) for row in rows]
    claims_ratios = [
        safe_float(row.get("issuer_unpaid_scheduled_claim_usd", row.get("claims_unpaid_usd_tick")))
        / max(1e-9, safe_float(row.get("bond_principal_usd")))
        for row in rows
    ]
    reserve_values = [safe_float(row.get("issuer_reserve_balance_usd")) for row in rows]
    reserve_draw_values = [safe_float(row.get("issuer_reserve_draw_usd")) for row in rows]
    unpaid_values = [safe_float(row.get("issuer_unpaid_scheduled_claim_usd")) for row in rows]
    scheduled_due_values = [safe_float(row.get("issuer_scheduled_debt_service_due_usd")) for row in rows]
    actual_payment_values = [safe_float(row.get("issuer_actual_bondholder_payment_usd")) for row in rows]
    concentration_values = [safe_float(row.get("realized_edge_top_share")) for row in rows]
    swap_values = [safe_float(row.get("swap_volume_usd_total")) for row in rows]
    v2v_count_values = [safe_float(row.get("swap_count_vchr_to_vchr_total")) for row in rows]
    v2v_volume_values = [safe_float(row.get("swap_volume_vchr_to_vchr_total")) for row in rows]
    v2stable_count_values = [safe_float(row.get("swap_count_vchr_to_usd_total")) for row in rows]
    stable2v_count_values = [safe_float(row.get("swap_count_usd_to_vchr_total")) for row in rows]
    transaction_values = [safe_float(row.get("transactions_total")) for row in rows]
    ordinary_swap_count_values = [safe_float(row.get("ordinary_swap_count_total")) for row in rows]
    ordinary_swap_volume_values = [safe_float(row.get("ordinary_swap_volume_usd_total")) for row in rows]
    ordinary_stable_source_count_values = [
        safe_float(row.get("ordinary_stable_source_swap_count_total")) for row in rows
    ]
    ordinary_stable_source_volume_values = [
        safe_float(row.get("ordinary_stable_source_swap_volume_usd_total")) for row in rows
    ]
    ordinary_voucher_source_count_values = [
        safe_float(row.get("ordinary_voucher_source_swap_count_total")) for row in rows
    ]
    ordinary_voucher_source_volume_values = [
        safe_float(row.get("ordinary_voucher_source_swap_volume_usd_total")) for row in rows
    ]
    loan_route_swap_count_values = [safe_float(row.get("loan_route_swap_count_total")) for row in rows]
    loan_route_swap_volume_values = [
        safe_float(row.get("loan_route_swap_volume_usd_total")) for row in rows
    ]
    voucher_loan_route_swap_count_values = [
        safe_float(row.get("voucher_loan_route_swap_count_total")) for row in rows
    ]
    voucher_loan_route_swap_volume_values = [
        safe_float(row.get("voucher_loan_route_swap_volume_usd_total")) for row in rows
    ]
    repayment_route_swap_count_values = [
        safe_float(row.get("repayment_route_swap_count_total")) for row in rows
    ]
    repayment_route_swap_volume_values = [
        safe_float(row.get("repayment_route_swap_volume_usd_total")) for row in rows
    ]
    loan_backfill_swap_count_values = [
        safe_float(row.get("loan_backfill_swap_count_total")) for row in rows
    ]
    loan_backfill_swap_volume_values = [
        safe_float(row.get("loan_backfill_swap_volume_usd_total")) for row in rows
    ]
    productive_boosted_voucher_swap_count_values = [
        safe_float(row.get("productive_boosted_voucher_swap_count_total")) for row in rows
    ]
    productive_boosted_voucher_swap_volume_values = [
        safe_float(row.get("productive_boosted_voucher_swap_volume_usd_total")) for row in rows
    ]
    voucher_loan_boosted_voucher_swap_count_values = [
        safe_float(row.get("voucher_loan_boosted_voucher_swap_count_total")) for row in rows
    ]
    voucher_loan_boosted_voucher_swap_volume_values = [
        safe_float(row.get("voucher_loan_boosted_voucher_swap_volume_usd_total")) for row in rows
    ]
    stable_spend_protection_skip_count_values = [
        safe_float(row.get("ordinary_stable_spend_protected_skip_count_total")) for row in rows
    ]
    stable_spend_protection_skip_value_values = [
        safe_float(row.get("ordinary_stable_spend_protected_skip_value_usd_total")) for row in rows
    ]
    stable_share_values = [safe_float(row.get("stable_value_share_in_active_pools")) for row in rows]
    voucher_share_values = [safe_float(row.get("voucher_value_share_in_active_pools")) for row in rows]
    stable_to_voucher_ratio_values = [
        safe_float(row.get("stable_to_voucher_value_ratio_in_active_pools")) for row in rows
    ]
    producer_debt_matured_values = [safe_float(row.get("producer_debt_matured_usd_total")) for row in rows]
    producer_debt_repaid_values = [safe_float(row.get("producer_debt_repaid_usd_total")) for row in rows]
    producer_debt_repaid_regular_values = [
        safe_float(row.get("producer_debt_repaid_regular_usd_total")) for row in rows
    ]
    producer_debt_repaid_maturity_values = [
        safe_float(row.get("producer_debt_repaid_maturity_usd_total")) for row in rows
    ]
    producer_debt_stable_recovered_values = [
        safe_float(row.get("producer_debt_stable_recovered_usd_total")) for row in rows
    ]
    producer_debt_consumer_stable_purchase_values = [
        safe_float(row.get("producer_debt_consumer_stable_purchase_usd_total")) for row in rows
    ]
    producer_debt_third_party_stable_purchase_values = [
        safe_float(row.get("producer_debt_third_party_stable_purchase_usd_total")) for row in rows
    ]
    producer_debt_defaulted_values = [safe_float(row.get("producer_debt_defaulted_usd_total")) for row in rows]
    producer_debt_closed_values = [
        safe_float(row.get("producer_debt_closed_by_circulation_usd_total")) for row in rows
    ]
    producer_debt_closed_by_voucher_swap_values = [
        safe_float(row.get("producer_debt_closed_by_voucher_swap_usd_total")) for row in rows
    ]
    producer_debt_closed_not_held_at_maturity_values = [
        safe_float(row.get("producer_debt_closed_not_held_at_maturity_usd_total")) for row in rows
    ]
    producer_credit_capacity_values = [
        safe_float(row.get("producer_deposit_credit_capacity_usd")) for row in rows
    ]
    producer_deposit_stable_values = [
        safe_float(row.get("producer_deposit_stable_usd_total")) for row in rows
    ]
    producer_deposit_voucher_values = [
        safe_float(row.get("producer_deposit_voucher_usd_total")) for row in rows
    ]
    productive_credit_inflow_values = [
        safe_float(row.get("productive_credit_inflow_usd_total")) for row in rows
    ]
    productive_credit_stable_retained_values = [
        safe_float(row.get("productive_credit_stable_retained_usd_total")) for row in rows
    ]
    productive_credit_voucher_deposit_values = [
        safe_float(row.get("productive_credit_voucher_deposit_usd_total")) for row in rows
    ]
    productive_credit_voucher_deposit_cap_clipped_values = [
        safe_float(row.get("productive_credit_voucher_deposit_cap_clipped_usd_total")) for row in rows
    ]
    productive_credit_voucher_deposit_share_values = [
        voucher / max(1e-9, inflow)
        for voucher, inflow in zip(
            productive_credit_voucher_deposit_values,
            productive_credit_inflow_values,
        )
    ]
    producer_organic_voucher_deposit_values = [
        max(0.0, total - productive)
        for total, productive in zip(
            producer_deposit_voucher_values,
            productive_credit_voucher_deposit_values,
        )
    ]
    producer_deposit_voucher_share_values = [
        voucher / max(1e-9, stable + voucher)
        for stable, voucher in zip(producer_deposit_stable_values, producer_deposit_voucher_values)
    ]
    loan_issuance_volume_values = [
        safe_float(row.get("loan_issuance_volume_usd_total")) for row in rows
    ]
    producer_debt_originated_values = [
        safe_float(row.get("producer_debt_originated_usd_total")) for row in rows
    ]
    loan_issuance_to_capacity_values = [
        loan / max(1e-9, capacity)
        for loan, capacity in zip(loan_issuance_volume_values, producer_credit_capacity_values)
    ]
    principal_to_credit_capacity_values = [
        principal / max(1e-9, capacity)
        for principal, capacity in zip(principal_values, producer_credit_capacity_values)
    ]
    producer_loan_attempt_values = [safe_float(row.get("producer_loan_attempts_total")) for row in rows]
    producer_loan_executed_values = [safe_float(row.get("producer_loan_executed_total")) for row in rows]
    producer_loan_route_failed_values = [
        safe_float(row.get("producer_loan_route_failed_total")) for row in rows
    ]
    producer_loan_backfill_attempt_values = [
        safe_float(row.get("producer_loan_backfill_attempts_total")) for row in rows
    ]
    producer_loan_backfill_executed_values = [
        safe_float(row.get("producer_loan_backfill_executed_total")) for row in rows
    ]
    producer_voucher_loan_attempt_values = [
        safe_float(row.get("producer_voucher_loan_attempts_total")) for row in rows
    ]
    producer_voucher_loan_executed_values = [
        safe_float(row.get("producer_voucher_loan_executed_total")) for row in rows
    ]
    producer_voucher_loan_route_failed_values = [
        safe_float(row.get("producer_voucher_loan_route_failed_total")) for row in rows
    ]
    producer_voucher_loan_no_target_values = [
        safe_float(row.get("producer_voucher_loan_no_target_total")) for row in rows
    ]
    producer_voucher_loan_attempted_usd_values = [
        safe_float(row.get("producer_voucher_loan_attempted_usd_total")) for row in rows
    ]
    producer_voucher_loan_executed_usd_values = [
        safe_float(row.get("producer_voucher_loan_executed_usd_total")) for row in rows
    ]
    producer_primary_voucher_loan_attempt_values = [
        safe_float(row.get("producer_primary_voucher_loan_attempts_total")) for row in rows
    ]
    producer_primary_voucher_loan_executed_values = [
        safe_float(row.get("producer_primary_voucher_loan_executed_total")) for row in rows
    ]
    producer_loan_attempted_usd_values = [
        safe_float(row.get("producer_loan_attempted_usd_total")) for row in rows
    ]
    producer_loan_executed_usd_values = [
        safe_float(row.get("producer_loan_executed_usd_total")) for row in rows
    ]
    producer_loan_clipped_lender_cap_values = [
        safe_float(row.get("producer_loan_clipped_lender_cap_usd_total")) for row in rows
    ]
    producer_loan_clipped_lender_remaining_values = [
        safe_float(row.get("producer_loan_clipped_lender_remaining_usd_total")) for row in rows
    ]
    producer_loan_clipped_lender_stable_values = [
        safe_float(row.get("producer_loan_clipped_lender_stable_usd_total")) for row in rows
    ]
    producer_loan_clipped_combined_lender_values = [
        safe_float(row.get("producer_loan_clipped_combined_lender_usd_total")) for row in rows
    ]
    producer_loan_lender_available_stable_values = [
        safe_float(row.get("lender_stable_available_above_reserve_usd")) for row in rows
    ]
    lender_held_producer_voucher_inventory_values = [
        safe_float(row.get("lender_held_producer_voucher_inventory_usd")) for row in rows
    ]
    producer_voucher_multi_lender_share_values = [
        safe_float(row.get("producer_voucher_multi_lender_share")) for row in rows
    ]
    producer_voucher_lender_degree_mean_values = [
        safe_float(row.get("producer_voucher_lender_degree_mean")) for row in rows
    ]
    producer_voucher_lender_degree_p50_values = [
        safe_float(row.get("producer_voucher_lender_degree_p50")) for row in rows
    ]
    producer_voucher_lender_degree_p90_values = [
        safe_float(row.get("producer_voucher_lender_degree_p90")) for row in rows
    ]
    producer_voucher_shared_lender_edge_values = [
        safe_float(row.get("producer_voucher_shared_lender_edges")) for row in rows
    ]
    consumer_stable_available_above_reserve_values = [
        safe_float(row.get("consumer_stable_available_above_reserve_usd")) for row in rows
    ]
    voucher_purchase_attempt_values = [
        safe_float(row.get("voucher_purchase_attempts_total")) for row in rows
    ]
    consumer_voucher_purchase_attempt_values = [
        safe_float(row.get("consumer_voucher_purchase_attempts_total")) for row in rows
    ]
    consumer_voucher_purchase_success_values = [
        safe_float(row.get("consumer_voucher_purchase_success_total")) for row in rows
    ]
    consumer_voucher_purchase_no_stable_values = [
        safe_float(row.get("consumer_voucher_purchase_no_stable_total")) for row in rows
    ]
    consumer_voucher_purchase_reserve_protected_values = [
        safe_float(row.get("consumer_voucher_purchase_reserve_protected_total")) for row in rows
    ]
    consumer_voucher_purchase_no_route_values = [
        safe_float(row.get("consumer_voucher_purchase_no_route_total")) for row in rows
    ]
    consumer_voucher_purchase_no_target_values = [
        safe_float(row.get("consumer_voucher_purchase_no_target_total")) for row in rows
    ]
    consumer_voucher_purchase_stable_spent_values = [
        safe_float(row.get("consumer_voucher_purchase_stable_spent_usd_total")) for row in rows
    ]
    consumer_voucher_purchase_voucher_acquired_values = [
        safe_float(row.get("consumer_voucher_purchase_voucher_value_acquired_usd_total"))
        for row in rows
    ]
    third_party_voucher_purchase_attempt_values = [
        safe_float(row.get("third_party_voucher_purchase_attempts_total")) for row in rows
    ]
    third_party_voucher_purchase_success_values = [
        safe_float(row.get("third_party_voucher_purchase_success_total")) for row in rows
    ]
    third_party_voucher_purchase_no_stable_values = [
        safe_float(row.get("third_party_voucher_purchase_no_stable_total")) for row in rows
    ]
    third_party_voucher_purchase_reserve_protected_values = [
        safe_float(row.get("third_party_voucher_purchase_reserve_protected_total")) for row in rows
    ]
    third_party_voucher_purchase_no_route_values = [
        safe_float(row.get("third_party_voucher_purchase_no_route_total")) for row in rows
    ]
    third_party_voucher_purchase_no_target_values = [
        safe_float(row.get("third_party_voucher_purchase_no_target_total")) for row in rows
    ]
    third_party_voucher_purchase_stable_spent_values = [
        safe_float(row.get("third_party_voucher_purchase_stable_spent_usd_total")) for row in rows
    ]
    third_party_voucher_purchase_voucher_acquired_values = [
        safe_float(row.get("third_party_voucher_purchase_voucher_value_acquired_usd_total"))
        for row in rows
    ]
    lender_voucher_purchase_budget_onramp_values = [
        safe_float(row.get("lender_voucher_purchase_stable_budget_onramp_usd_total"))
        for row in rows
    ]
    consumer_voucher_purchase_budget_onramp_values = [
        safe_float(row.get("consumer_voucher_purchase_stable_budget_onramp_usd_total"))
        for row in rows
    ]
    third_party_voucher_purchase_budget_onramp_values = [
        safe_float(row.get("third_party_voucher_purchase_stable_budget_onramp_usd_total"))
        for row in rows
    ]
    lender_recovered_stable_values = [
        safe_float(row.get("lender_recovered_stable_usd_total")) for row in rows
    ]
    lender_recovered_stable_borrower_regular_values = [
        safe_float(row.get("lender_recovered_stable_borrower_regular_usd_total")) for row in rows
    ]
    lender_recovered_stable_borrower_maturity_values = [
        safe_float(row.get("lender_recovered_stable_borrower_maturity_usd_total")) for row in rows
    ]
    lender_recovered_stable_consumer_purchase_values = [
        safe_float(row.get("lender_recovered_stable_consumer_purchase_usd_total")) for row in rows
    ]
    lender_recovered_stable_third_party_purchase_values = [
        safe_float(row.get("lender_recovered_stable_third_party_purchase_usd_total")) for row in rows
    ]
    lender_recovered_stable_other_values = [
        safe_float(row.get("lender_recovered_stable_other_usd_total")) for row in rows
    ]
    bond_service_reserved_values = [
        safe_float(row.get("bond_service_reserved_usd_total")) for row in rows
    ]
    fee_service_reserved_values = [
        safe_float(row.get("fee_service_reserved_usd_total")) for row in rows
    ]
    fee_service_stable_reserved_values = [
        safe_float(row.get("fee_service_stable_reserved_usd_total")) for row in rows
    ]
    fee_service_converted_voucher_reserved_values = [
        safe_float(row.get("fee_service_converted_voucher_reserved_usd_total")) for row in rows
    ]
    bond_service_reserve_balance_values = [
        safe_float(row.get("bond_service_reserve_balance_usd")) for row in rows
    ]
    bond_service_paid_from_reserve_values = [
        safe_float(row.get("bond_service_paid_from_reserve_usd_total")) for row in rows
    ]
    bond_service_lockbox_target_values = [
        safe_float(row.get("bond_service_lockbox_target_usd")) for row in rows
    ]
    bond_service_lockbox_coverage_values = [
        safe_float(row.get("bond_service_lockbox_coverage_ratio")) for row in rows
    ]
    lender_recovered_stable_sweepable_pending_values = [
        safe_float(row.get("lender_recovered_stable_sweepable_pending_usd_total")) for row in rows
    ]
    v2v_share_values = [
        v2v / max(1e-9, total) for v2v, total in zip(v2v_count_values, transaction_values)
    ]
    route_p50 = percentile(route_values, 0.50)
    swap_p50 = percentile(swap_values, 0.50)
    v2v_count_p50 = percentile(v2v_count_values, 0.50)
    v2v_volume_p50 = percentile(v2v_volume_values, 0.50)
    v2v_share_p50 = percentile(v2v_share_values, 0.50)
    ordinary_swap_count_p50 = percentile(ordinary_swap_count_values, 0.50)
    ordinary_swap_volume_p50 = percentile(ordinary_swap_volume_values, 0.50)
    ordinary_stable_source_count_p50 = percentile(ordinary_stable_source_count_values, 0.50)
    ordinary_stable_source_volume_p50 = percentile(ordinary_stable_source_volume_values, 0.50)
    ordinary_voucher_source_count_p50 = percentile(ordinary_voucher_source_count_values, 0.50)
    ordinary_voucher_source_volume_p50 = percentile(ordinary_voucher_source_volume_values, 0.50)
    stable_share_p50 = percentile(stable_share_values, 0.50)
    stable_share_p95 = percentile(stable_share_values, 0.95)
    voucher_share_p50 = percentile(voucher_share_values, 0.50)
    stress_p50 = percentile(stress_values, 0.50)
    stress_p95 = percentile(stress_values, 0.95)
    consumer_stress_p50 = percentile(consumer_stress_values, 0.50)
    consumer_stress_p95 = percentile(consumer_stress_values, 0.95)
    community_stress_p50 = percentile(community_stress_values, 0.50)
    community_stress_p95 = percentile(community_stress_values, 0.95)
    leakage_p50 = percentile(leakage_values, 0.50)
    leakage_p95 = percentile(leakage_values, 0.95)
    producer_credit_capacity_p50 = percentile(producer_credit_capacity_values, 0.50)
    productive_credit_inflow_p50 = percentile(productive_credit_inflow_values, 0.50)
    productive_credit_stable_retained_p50 = percentile(productive_credit_stable_retained_values, 0.50)
    productive_credit_voucher_deposit_p50 = percentile(productive_credit_voucher_deposit_values, 0.50)
    productive_credit_voucher_deposit_share_p50 = percentile(
        productive_credit_voucher_deposit_share_values, 0.50
    )
    baseline_stress_p50 = baseline.get("household_cash_stress_p50", 0.0)
    baseline_stress_p95 = baseline.get("household_cash_stress_p95", baseline_stress_p50)
    baseline_consumer_stress_p50 = baseline.get("consumer_cash_stress_p50", baseline_stress_p50)
    baseline_consumer_stress_p95 = baseline.get(
        "consumer_cash_stress_p95", baseline_consumer_stress_p50
    )
    baseline_community_stress_p50 = baseline.get("community_cash_stress_p50", baseline_stress_p50)
    baseline_community_stress_p95 = baseline.get(
        "community_cash_stress_p95", baseline_community_stress_p50
    )
    baseline_leakage_p50 = baseline.get("liquidity_leakage_p50", 0.0)
    baseline_leakage_p95 = baseline.get("liquidity_leakage_p95", baseline_leakage_p50)
    baseline_v2v_count_p50 = baseline.get("voucher_to_voucher_count_p50", 0.0)
    baseline_v2v_volume_p50 = baseline.get("voucher_to_voucher_volume_p50", 0.0)
    baseline_v2v_share_p50 = baseline.get("voucher_to_voucher_share_p50", 0.0)
    baseline_stable_share_p50 = baseline.get("stable_value_share_p50", 0.0)
    baseline_stable_share_p95 = baseline.get("stable_value_share_p95", baseline_stable_share_p50)
    baseline_voucher_share_p50 = baseline.get("voucher_value_share_p50", 0.0)
    baseline_producer_credit_capacity_p50 = baseline.get("producer_deposit_credit_capacity_usd_p50", 0.0)
    baseline_productive_credit_inflow_p50 = baseline.get("productive_credit_inflow_usd_total_p50", 0.0)
    baseline_productive_credit_stable_retained_p50 = baseline.get(
        "productive_credit_stable_retained_usd_total_p50", 0.0
    )
    baseline_productive_credit_voucher_deposit_p50 = baseline.get(
        "productive_credit_voucher_deposit_usd_total_p50", 0.0
    )
    baseline_productive_credit_voucher_deposit_share_p50 = baseline.get(
        "productive_credit_voucher_deposit_share_p50", 0.0
    )
    baseline_ordinary_swap_count_p50 = baseline.get("ordinary_swap_count_total_p50", 0.0)
    baseline_ordinary_swap_volume_p50 = baseline.get("ordinary_swap_volume_usd_total_p50", 0.0)
    baseline_ordinary_stable_source_count_p50 = baseline.get(
        "ordinary_stable_source_swap_count_total_p50", 0.0
    )
    baseline_ordinary_stable_source_volume_p50 = baseline.get(
        "ordinary_stable_source_swap_volume_usd_total_p50", 0.0
    )
    baseline_ordinary_voucher_source_volume_p50 = baseline.get(
        "ordinary_voucher_source_swap_volume_usd_total_p50", 0.0
    )
    baseline_ordinary_voucher_source_count_p50 = baseline.get(
        "ordinary_voucher_source_swap_count_total_p50", 0.0
    )
    stress_delta_p95 = max(0.0, stress_p95 - baseline_stress_p95)
    consumer_stress_delta_p95 = max(0.0, consumer_stress_p95 - baseline_consumer_stress_p95)
    consumer_stress_delta_p50 = max(0.0, consumer_stress_p50 - baseline_consumer_stress_p50)
    community_stress_delta_p95 = max(0.0, community_stress_p95 - baseline_community_stress_p95)
    community_stress_delta_p50 = max(0.0, community_stress_p50 - baseline_community_stress_p50)
    leakage_delta_p95 = max(0.0, leakage_p95 - baseline_leakage_p95)
    stable_dependency_delta_p95 = max(0.0, stable_share_p95 - baseline_stable_share_p95)
    baseline_route_p50 = baseline.get("route_success_p50", 0.0)
    baseline_swap_p50 = baseline.get("swap_volume_p50", 0.0)
    route_success_delta_vs_baseline = route_p50 - baseline_route_p50
    swap_volume_ratio_vs_baseline = swap_p50 / max(1e-9, baseline_swap_p50)
    ordinary_swap_count_ratio_vs_baseline = ordinary_swap_count_p50 / max(
        1e-9, baseline_ordinary_swap_count_p50
    )
    ordinary_swap_volume_ratio_vs_baseline = ordinary_swap_volume_p50 / max(
        1e-9, baseline_ordinary_swap_volume_p50
    )
    ordinary_stable_source_count_ratio_vs_baseline = ordinary_stable_source_count_p50 / max(
        1e-9, baseline_ordinary_stable_source_count_p50
    )
    ordinary_stable_source_volume_ratio_vs_baseline = ordinary_stable_source_volume_p50 / max(
        1e-9, baseline_ordinary_stable_source_volume_p50
    )
    ordinary_voucher_source_count_ratio_vs_baseline = ordinary_voucher_source_count_p50 / max(
        1e-9, baseline_ordinary_voucher_source_count_p50
    )
    ordinary_voucher_source_volume_ratio_vs_baseline = ordinary_voucher_source_volume_p50 / max(
        1e-9, baseline_ordinary_voucher_source_volume_p50
    )
    voucher_to_voucher_volume_ratio_vs_baseline = v2v_volume_p50 / max(
        1e-9, baseline_v2v_volume_p50
    )
    route_decline = route_p50 < baseline_route_p50 - 0.05
    swap_volume_decline = baseline_swap_p50 > 0.0 and swap_p50 < baseline_swap_p50 * 0.85
    ordinary_swap_count_decline = (
        baseline_ordinary_swap_count_p50 > 0.0
        and ordinary_swap_count_p50 < baseline_ordinary_swap_count_p50 * 0.85
    )
    ordinary_swap_volume_decline = (
        baseline_ordinary_swap_volume_p50 > 0.0
        and ordinary_swap_volume_p50 < baseline_ordinary_swap_volume_p50 * 0.85
    )
    ordinary_stable_source_count_decline = (
        baseline_ordinary_stable_source_count_p50 > 0.0
        and ordinary_stable_source_count_p50 < baseline_ordinary_stable_source_count_p50 * 0.85
    )
    ordinary_stable_source_volume_decline = (
        baseline_ordinary_stable_source_volume_p50 > 0.0
        and ordinary_stable_source_volume_p50 < baseline_ordinary_stable_source_volume_p50 * 0.85
    )
    ordinary_voucher_source_count_decline = (
        baseline_ordinary_voucher_source_count_p50 > 0.0
        and ordinary_voucher_source_count_p50 < baseline_ordinary_voucher_source_count_p50 * 0.85
    )
    ordinary_voucher_source_volume_decline = (
        baseline_ordinary_voucher_source_volume_p50 > 0.0
        and ordinary_voucher_source_volume_p50 < baseline_ordinary_voucher_source_volume_p50 * 0.85
    )
    v2v_count_decline = (
        baseline_v2v_count_p50 > 0.0 and v2v_count_p50 < baseline_v2v_count_p50 * 0.85
    )
    v2v_volume_decline = (
        baseline_v2v_volume_p50 > 0.0 and v2v_volume_p50 < baseline_v2v_volume_p50 * 0.85
    )
    v2v_share_decline = (
        baseline_v2v_share_p50 > 0.0 and v2v_share_p50 < max(0.0, baseline_v2v_share_p50 - 0.10)
    )
    voucher_circulation_decline = (
        v2v_count_decline
        or v2v_volume_decline
        or ordinary_voucher_source_count_decline
        or ordinary_voucher_source_volume_decline
    )
    voucher_value_share_decline = (
        baseline_voucher_share_p50 > 0.0 and voucher_share_p50 < max(0.0, baseline_voucher_share_p50 - 0.15)
    )
    stable_dependency_increase = stable_dependency_delta_p95 > 0.15
    consumer_stress_increase = consumer_stress_p50 > baseline_consumer_stress_p50 + 0.05
    community_stress_increase = community_stress_p50 > baseline_community_stress_p50 + 0.05
    leakage_increase = leakage_p50 > baseline_leakage_p50 + 0.05
    material_decline = (
        route_decline
        or voucher_circulation_decline
        or stable_dependency_increase
        or voucher_value_share_decline
        or consumer_stress_increase
        or community_stress_increase
        or leakage_increase
    )
    constraints = []
    if percentile(scheduled_payment_values, 0.50) < 1.00:
        constraints.append("p50_scheduled_payment_coverage")
    if percentile(scheduled_payment_values, 0.05) < 1.00:
        constraints.append("p05_scheduled_payment_coverage")
    route_success_floor = max(0.0, min(1.0, float(route_success_floor)))
    route_success_mode = str(route_success_mode or "diagnostic").strip().lower()
    if route_success_mode == "absolute" and percentile(route_values, 0.05) < route_success_floor:
        constraints.append("p05_route_success")
    elif route_success_mode == "relative" and route_decline:
        constraints.append("route_success_decline_vs_no_bond")
    if consumer_stress_delta_p95 > 0.20:
        constraints.append("p95_consumer_cash_stress_delta")
    if community_stress_delta_p95 > 0.20:
        constraints.append("p95_community_cash_stress_delta")
    if leakage_delta_p95 > 0.15:
        constraints.append("p95_liquidity_leakage_delta")
    if percentile(claims_ratios, 0.95) > 0.01:
        constraints.append("p95_unpaid_claims")
    if percentile(concentration_values, 0.95) > 0.25:
        constraints.append("p95_realized_edge_concentration")
    if v2v_count_decline:
        constraints.append("voucher_to_voucher_count_decline_vs_no_bond")
    if v2v_volume_decline:
        constraints.append("voucher_to_voucher_volume_decline_vs_no_bond")
    if ordinary_voucher_source_count_decline:
        constraints.append("ordinary_voucher_source_count_decline_vs_no_bond")
    if ordinary_voucher_source_volume_decline:
        constraints.append("ordinary_voucher_source_volume_decline_vs_no_bond")
    if voucher_circulation_decline:
        constraints.append("voucher_circulation_decline_vs_no_bond")
    if stable_dependency_increase:
        constraints.append("p95_stable_dependency_delta")
    if voucher_value_share_decline:
        constraints.append("voucher_value_share_decline_vs_no_bond")
    if material_decline:
        constraints.append("material_decline_vs_no_bond")
    material_reasons = []
    diagnostic_reasons = []
    if route_decline:
        material_reasons.append("route_success_decline")
    if swap_volume_decline:
        diagnostic_reasons.append("total_swap_volume_decline")
    if ordinary_swap_count_decline:
        diagnostic_reasons.append("ordinary_swap_count_decline")
    if ordinary_swap_volume_decline:
        diagnostic_reasons.append("ordinary_swap_volume_decline")
    if ordinary_stable_source_count_decline:
        diagnostic_reasons.append("ordinary_stable_source_count_decline")
    if ordinary_stable_source_volume_decline:
        diagnostic_reasons.append("ordinary_stable_source_volume_decline")
    if v2v_count_decline:
        material_reasons.append("voucher_to_voucher_count_decline")
    if v2v_volume_decline:
        material_reasons.append("voucher_to_voucher_volume_decline")
    if v2v_share_decline:
        diagnostic_reasons.append("voucher_to_voucher_share_decline")
    if ordinary_voucher_source_count_decline:
        material_reasons.append("ordinary_voucher_source_count_decline")
    if ordinary_voucher_source_volume_decline:
        material_reasons.append("ordinary_voucher_source_volume_decline")
    if voucher_circulation_decline:
        material_reasons.append("voucher_circulation_decline")
    if stable_dependency_increase:
        material_reasons.append("stable_dependency_increase")
    if consumer_stress_increase:
        material_reasons.append("consumer_stress_increase")
    if community_stress_increase:
        material_reasons.append("community_stress_increase")
    if leakage_increase:
        material_reasons.append("leakage_increase")
    if voucher_value_share_decline:
        material_reasons.append("voucher_share_decline")
    first = rows[0] if rows else {}
    return {
        "scenario": "bond_issuer_frontier",
        "network_scale": first.get("network_scale", ""),
        "coupon_target_annual": first.get("coupon_target_annual", 0.0),
        "bond_fee_service_share": first.get("bond_fee_service_share", 0.0),
        "principal_ratio": first.get("principal_ratio", 0.0),
        "producer_stable_debt_contract_service_margin_rate": first.get(
            "configured_producer_stable_debt_contract_service_margin_rate",
            first.get("configured_producer_debt_contract_service_margin_rate", 0.0),
        ),
        "producer_voucher_debt_contract_service_margin_rate": first.get(
            "configured_producer_voucher_debt_contract_service_margin_rate",
            first.get("configured_producer_debt_contract_service_margin_rate", 0.0),
        ),
        "principal_usd_p50": percentile(principal_values, 0.50),
        "runs": len(rows),
        "service_coverage_p05": percentile(service_values, 0.05),
        "service_coverage_p50": percentile(service_values, 0.50),
        "scheduled_payment_coverage_p05": percentile(scheduled_payment_values, 0.05),
        "scheduled_payment_coverage_p50": percentile(scheduled_payment_values, 0.50),
        "service_cash_headroom_p05": percentile(service_cash_headroom_values, 0.05),
        "service_cash_headroom_p50": percentile(service_cash_headroom_values, 0.50),
        "gross_historical_service_cash_headroom_p05": percentile(service_cash_headroom_values, 0.05),
        "gross_historical_service_cash_headroom_p50": percentile(service_cash_headroom_values, 0.50),
        "available_service_cash_headroom_p05": percentile(
            available_service_cash_headroom_values, 0.05
        ),
        "available_service_cash_headroom_p50": percentile(
            available_service_cash_headroom_values, 0.50
        ),
        "issuer_operating_risk_headroom_p05": percentile(
            available_service_cash_headroom_values, 0.05
        ),
        "issuer_operating_risk_headroom_p50": percentile(
            available_service_cash_headroom_values, 0.50
        ),
        "issuer_operating_risk_headroom_ge_125": int(
            percentile(available_service_cash_headroom_values, 0.50) >= 1.25
        ),
        "issuer_lockbox_surplus_usd_p50": percentile(issuer_lockbox_surplus_values, 0.50),
        "issuer_excess_recovered_stable_usd_p50": percentile(
            issuer_excess_recovered_stable_values, 0.50
        ),
        "issuer_operating_surplus_usd_p50": percentile(issuer_operating_surplus_values, 0.50),
        "issuer_operating_surplus_headroom_ratio_p50": percentile(
            issuer_operating_surplus_headroom_values, 0.50
        ),
        "issuer_fee_waterfall_ops_usd_p50": percentile(issuer_fee_waterfall_ops_values, 0.50),
        "issuer_sustainability_resources_usd_p50": percentile(
            issuer_sustainability_resources_values, 0.50
        ),
        "issuer_sustainability_headroom_ratio_p50": percentile(
            issuer_sustainability_headroom_values, 0.50
        ),
        "issuer_scheduled_debt_service_due_p50": percentile(scheduled_due_values, 0.50),
        "issuer_actual_bondholder_payment_p50": percentile(actual_payment_values, 0.50),
        "issuer_reserve_balance_p05": percentile(reserve_values, 0.05),
        "issuer_reserve_draw_p95": percentile(reserve_draw_values, 0.95),
        "issuer_unpaid_scheduled_claim_p95": percentile(unpaid_values, 0.95),
        "route_success_p05": percentile(route_values, 0.05),
        "route_success_p50": route_p50,
        "route_success_delta_vs_baseline": route_success_delta_vs_baseline,
        "route_success_floor": route_success_floor,
        "route_success_mode": route_success_mode,
        "route_fixed_success_p05": percentile(fixed_route_values, 0.05),
        "route_fixed_success_p50": percentile(fixed_route_values, 0.50),
        "route_substitution_success_p05": percentile(substitution_route_values, 0.05),
        "route_substitution_success_p50": percentile(substitution_route_values, 0.50),
        "noam_routing_swaps_total_p50": percentile(noam_routing_swaps_values, 0.50),
        "noam_clearing_swaps_total_p50": percentile(noam_clearing_swaps_values, 0.50),
        "noam_routing_volume_usd_total_p50": percentile(noam_routing_volume_values, 0.50),
        "noam_clearing_volume_usd_total_p50": percentile(noam_clearing_volume_values, 0.50),
        "noam_clearing_volume_share_p50": (
            percentile(noam_clearing_volume_values, 0.50)
            / max(
                1e-9,
                percentile(noam_routing_volume_values, 0.50)
                + percentile(noam_clearing_volume_values, 0.50),
            )
        ),
        "noam_routing_fee_usd_total_p50": percentile(noam_routing_fee_values, 0.50),
        "noam_clearing_fee_usd_total_p50": percentile(noam_clearing_fee_values, 0.50),
        "noam_total_fee_usd_total_p50": (
            percentile(noam_routing_fee_values, 0.50)
            + percentile(noam_clearing_fee_values, 0.50)
        ),
        "noam_clearing_cycle_success_rate_p50": percentile(
            noam_clearing_cycle_success_values, 0.50
        ),
        "noam_clearing_cycle_value_usd_total_p50": percentile(
            noam_clearing_cycle_value_values, 0.50
        ),
        "household_cash_stress_p95": stress_p95,
        "household_cash_stress_delta_p95": stress_delta_p95,
        "consumer_cash_stress_p50": consumer_stress_p50,
        "consumer_cash_stress_p95": consumer_stress_p95,
        "consumer_stress_delta_p50": consumer_stress_delta_p50,
        "consumer_stress_delta_p95": consumer_stress_delta_p95,
        "community_cash_stress_p50": community_stress_p50,
        "community_cash_stress_p95": community_stress_p95,
        "community_stress_delta_p50": community_stress_delta_p50,
        "community_stress_delta_p95": community_stress_delta_p95,
        "liquidity_leakage_p95": leakage_p95,
        "liquidity_leakage_delta_p95": leakage_delta_p95,
        "unpaid_claims_ratio_p95": percentile(claims_ratios, 0.95),
        "realized_edge_concentration_p95": percentile(concentration_values, 0.95),
        "swap_volume_usd_total_p50": swap_p50,
        "swap_volume_ratio_vs_baseline": swap_volume_ratio_vs_baseline,
        "ordinary_swap_count_total_p50": ordinary_swap_count_p50,
        "ordinary_swap_volume_usd_total_p50": ordinary_swap_volume_p50,
        "ordinary_swap_count_ratio_vs_baseline": ordinary_swap_count_ratio_vs_baseline,
        "ordinary_swap_volume_ratio_vs_baseline": ordinary_swap_volume_ratio_vs_baseline,
        "ordinary_stable_source_swap_count_total_p50": ordinary_stable_source_count_p50,
        "ordinary_stable_source_swap_volume_usd_total_p50": ordinary_stable_source_volume_p50,
        "ordinary_stable_source_swap_count_ratio_vs_baseline": (
            ordinary_stable_source_count_ratio_vs_baseline
        ),
        "ordinary_stable_source_swap_volume_ratio_vs_baseline": (
            ordinary_stable_source_volume_ratio_vs_baseline
        ),
        "ordinary_voucher_source_swap_count_total_p50": ordinary_voucher_source_count_p50,
        "ordinary_voucher_source_swap_volume_usd_total_p50": ordinary_voucher_source_volume_p50,
        "ordinary_voucher_source_swap_count_ratio_vs_baseline": (
            ordinary_voucher_source_count_ratio_vs_baseline
        ),
        "ordinary_voucher_source_swap_volume_ratio_vs_baseline": (
            ordinary_voucher_source_volume_ratio_vs_baseline
        ),
        "loan_route_swap_count_total_p50": percentile(loan_route_swap_count_values, 0.50),
        "loan_route_swap_volume_usd_total_p50": percentile(loan_route_swap_volume_values, 0.50),
        "voucher_loan_route_swap_count_total_p50": percentile(
            voucher_loan_route_swap_count_values, 0.50
        ),
        "voucher_loan_route_swap_volume_usd_total_p50": percentile(
            voucher_loan_route_swap_volume_values, 0.50
        ),
        "repayment_route_swap_count_total_p50": percentile(repayment_route_swap_count_values, 0.50),
        "repayment_route_swap_volume_usd_total_p50": percentile(
            repayment_route_swap_volume_values, 0.50
        ),
        "loan_backfill_swap_count_total_p50": percentile(loan_backfill_swap_count_values, 0.50),
        "loan_backfill_swap_volume_usd_total_p50": percentile(
            loan_backfill_swap_volume_values, 0.50
        ),
        "productive_boosted_voucher_swap_count_total_p50": percentile(
            productive_boosted_voucher_swap_count_values, 0.50
        ),
        "productive_boosted_voucher_swap_volume_usd_total_p50": percentile(
            productive_boosted_voucher_swap_volume_values, 0.50
        ),
        "voucher_loan_boosted_voucher_swap_count_total_p50": percentile(
            voucher_loan_boosted_voucher_swap_count_values, 0.50
        ),
        "voucher_loan_boosted_voucher_swap_volume_usd_total_p50": percentile(
            voucher_loan_boosted_voucher_swap_volume_values, 0.50
        ),
        "ordinary_stable_spend_protected_skip_count_total_p50": percentile(
            stable_spend_protection_skip_count_values, 0.50
        ),
        "ordinary_stable_spend_protected_skip_value_usd_total_p50": percentile(
            stable_spend_protection_skip_value_values, 0.50
        ),
        "voucher_to_voucher_count_p50": v2v_count_p50,
        "voucher_to_voucher_volume_p50": v2v_volume_p50,
        "voucher_to_voucher_volume_ratio_vs_baseline": voucher_to_voucher_volume_ratio_vs_baseline,
        "voucher_to_voucher_share_p50": v2v_share_p50,
        "voucher_to_stable_count_p50": percentile(v2stable_count_values, 0.50),
        "stable_to_voucher_count_p50": percentile(stable2v_count_values, 0.50),
        "stable_value_share_p50": stable_share_p50,
        "stable_value_share_p95": stable_share_p95,
        "voucher_value_share_p50": voucher_share_p50,
        "stable_to_voucher_value_ratio_p50": percentile(stable_to_voucher_ratio_values, 0.50),
        "producer_debt_matured_usd_total_p50": percentile(producer_debt_matured_values, 0.50),
        "producer_debt_repaid_usd_total_p50": percentile(producer_debt_repaid_values, 0.50),
        "producer_debt_repaid_regular_usd_total_p50": percentile(
            producer_debt_repaid_regular_values, 0.50
        ),
        "producer_debt_repaid_maturity_usd_total_p50": percentile(
            producer_debt_repaid_maturity_values, 0.50
        ),
        "producer_debt_stable_recovered_usd_total_p50": percentile(
            producer_debt_stable_recovered_values, 0.50
        ),
        "producer_debt_consumer_stable_purchase_usd_total_p50": percentile(
            producer_debt_consumer_stable_purchase_values, 0.50
        ),
        "producer_debt_third_party_stable_purchase_usd_total_p50": percentile(
            producer_debt_third_party_stable_purchase_values, 0.50
        ),
        "producer_debt_defaulted_usd_total_p50": percentile(producer_debt_defaulted_values, 0.50),
        "producer_debt_closed_by_circulation_usd_total_p50": percentile(producer_debt_closed_values, 0.50),
        "producer_debt_closed_by_voucher_swap_usd_total_p50": percentile(
            producer_debt_closed_by_voucher_swap_values, 0.50
        ),
        "producer_debt_closed_not_held_at_maturity_usd_total_p50": percentile(
            producer_debt_closed_not_held_at_maturity_values, 0.50
        ),
        "producer_deposit_credit_capacity_usd_p50": producer_credit_capacity_p50,
        "producer_deposit_stable_usd_total_p50": percentile(producer_deposit_stable_values, 0.50),
        "producer_deposit_voucher_usd_total_p50": percentile(producer_deposit_voucher_values, 0.50),
        "producer_organic_voucher_deposit_usd_total_p50": percentile(
            producer_organic_voucher_deposit_values, 0.50
        ),
        "producer_deposit_voucher_share_p50": percentile(
            producer_deposit_voucher_share_values, 0.50
        ),
        "productive_credit_inflow_usd_total_p50": productive_credit_inflow_p50,
        "productive_credit_stable_retained_usd_total_p50": productive_credit_stable_retained_p50,
        "productive_credit_voucher_deposit_usd_total_p50": productive_credit_voucher_deposit_p50,
        "productive_credit_voucher_deposit_cap_clipped_usd_total_p50": percentile(
            productive_credit_voucher_deposit_cap_clipped_values, 0.50
        ),
        "productive_credit_voucher_deposit_share_p50": productive_credit_voucher_deposit_share_p50,
        "producer_deposit_credit_capacity_ratio_vs_baseline": (
            producer_credit_capacity_p50 / max(1e-9, baseline_producer_credit_capacity_p50)
        ),
        "incremental_productive_credit_inflow_usd_total_p50": (
            productive_credit_inflow_p50 - baseline_productive_credit_inflow_p50
        ),
        "incremental_productive_credit_stable_retained_usd_total_p50": (
            productive_credit_stable_retained_p50 - baseline_productive_credit_stable_retained_p50
        ),
        "incremental_productive_credit_voucher_deposit_usd_total_p50": (
            productive_credit_voucher_deposit_p50 - baseline_productive_credit_voucher_deposit_p50
        ),
        "productive_credit_inflow_ratio_vs_baseline": (
            productive_credit_inflow_p50 / max(1e-9, baseline_productive_credit_inflow_p50)
        ),
        "productive_credit_voucher_deposit_ratio_vs_baseline": (
            productive_credit_voucher_deposit_p50
            / max(1e-9, baseline_productive_credit_voucher_deposit_p50)
        ),
        "loan_issuance_volume_usd_total_p50": percentile(loan_issuance_volume_values, 0.50),
        "producer_debt_originated_usd_total_p50": percentile(producer_debt_originated_values, 0.50),
        "loan_issuance_to_credit_capacity_p50": percentile(loan_issuance_to_capacity_values, 0.50),
        "bond_principal_to_credit_capacity_p50": percentile(principal_to_credit_capacity_values, 0.50),
        "producer_loan_attempts_total_p50": percentile(producer_loan_attempt_values, 0.50),
        "producer_loan_executed_total_p50": percentile(producer_loan_executed_values, 0.50),
        "producer_loan_route_failed_total_p50": percentile(producer_loan_route_failed_values, 0.50),
        "producer_loan_backfill_attempts_total_p50": percentile(
            producer_loan_backfill_attempt_values, 0.50
        ),
        "producer_loan_backfill_executed_total_p50": percentile(
            producer_loan_backfill_executed_values, 0.50
        ),
        "producer_voucher_loan_attempts_total_p50": percentile(
            producer_voucher_loan_attempt_values, 0.50
        ),
        "producer_voucher_loan_executed_total_p50": percentile(
            producer_voucher_loan_executed_values, 0.50
        ),
        "producer_voucher_loan_route_failed_total_p50": percentile(
            producer_voucher_loan_route_failed_values, 0.50
        ),
        "producer_voucher_loan_no_target_total_p50": percentile(
            producer_voucher_loan_no_target_values, 0.50
        ),
        "producer_voucher_loan_attempted_usd_total_p50": percentile(
            producer_voucher_loan_attempted_usd_values, 0.50
        ),
        "producer_voucher_loan_executed_usd_total_p50": percentile(
            producer_voucher_loan_executed_usd_values, 0.50
        ),
        "producer_primary_voucher_loan_attempts_total_p50": percentile(
            producer_primary_voucher_loan_attempt_values, 0.50
        ),
        "producer_primary_voucher_loan_executed_total_p50": percentile(
            producer_primary_voucher_loan_executed_values, 0.50
        ),
        "producer_loan_attempted_usd_total_p50": percentile(producer_loan_attempted_usd_values, 0.50),
        "producer_loan_executed_usd_total_p50": percentile(producer_loan_executed_usd_values, 0.50),
        "producer_loan_clipped_lender_cap_usd_total_p50": percentile(
            producer_loan_clipped_lender_cap_values, 0.50
        ),
        "producer_loan_clipped_lender_remaining_usd_total_p50": percentile(
            producer_loan_clipped_lender_remaining_values, 0.50
        ),
        "producer_loan_clipped_lender_stable_usd_total_p50": percentile(
            producer_loan_clipped_lender_stable_values, 0.50
        ),
        "producer_loan_clipped_combined_lender_usd_total_p50": percentile(
            producer_loan_clipped_combined_lender_values, 0.50
        ),
        "lender_stable_available_above_reserve_usd_p50": percentile(
            producer_loan_lender_available_stable_values, 0.50
        ),
        "lender_held_producer_voucher_inventory_usd_p50": percentile(
            lender_held_producer_voucher_inventory_values, 0.50
        ),
        "producer_voucher_multi_lender_share_p50": percentile(
            producer_voucher_multi_lender_share_values, 0.50
        ),
        "producer_voucher_lender_degree_mean_p50": percentile(
            producer_voucher_lender_degree_mean_values, 0.50
        ),
        "producer_voucher_lender_degree_p50_p50": percentile(
            producer_voucher_lender_degree_p50_values, 0.50
        ),
        "producer_voucher_lender_degree_p90_p50": percentile(
            producer_voucher_lender_degree_p90_values, 0.50
        ),
        "producer_voucher_shared_lender_edges_p50": percentile(
            producer_voucher_shared_lender_edge_values, 0.50
        ),
        "empirical_voucher_pool_overlap_multi_pool_share": safe_float(
            first.get("configured_empirical_voucher_pool_overlap_multi_pool_share")
        ),
        "empirical_voucher_pool_overlap_degree_p50": safe_float(
            first.get("configured_empirical_voucher_pool_overlap_degree_p50")
        ),
        "empirical_voucher_pool_overlap_degree_p90": safe_float(
            first.get("configured_empirical_voucher_pool_overlap_degree_p90")
        ),
        "consumer_stable_available_above_reserve_usd_p50": percentile(
            consumer_stable_available_above_reserve_values, 0.50
        ),
        "voucher_purchase_attempts_total_p50": percentile(voucher_purchase_attempt_values, 0.50),
        "consumer_voucher_purchase_attempts_total_p50": percentile(
            consumer_voucher_purchase_attempt_values, 0.50
        ),
        "consumer_voucher_purchase_success_total_p50": percentile(
            consumer_voucher_purchase_success_values, 0.50
        ),
        "consumer_voucher_purchase_no_stable_total_p50": percentile(
            consumer_voucher_purchase_no_stable_values, 0.50
        ),
        "consumer_voucher_purchase_reserve_protected_total_p50": percentile(
            consumer_voucher_purchase_reserve_protected_values, 0.50
        ),
        "consumer_voucher_purchase_no_route_total_p50": percentile(
            consumer_voucher_purchase_no_route_values, 0.50
        ),
        "consumer_voucher_purchase_no_target_total_p50": percentile(
            consumer_voucher_purchase_no_target_values, 0.50
        ),
        "consumer_voucher_purchase_stable_spent_usd_total_p50": percentile(
            consumer_voucher_purchase_stable_spent_values, 0.50
        ),
        "consumer_voucher_purchase_voucher_value_acquired_usd_total_p50": percentile(
            consumer_voucher_purchase_voucher_acquired_values, 0.50
        ),
        "third_party_voucher_purchase_attempts_total_p50": percentile(
            third_party_voucher_purchase_attempt_values, 0.50
        ),
        "third_party_voucher_purchase_success_total_p50": percentile(
            third_party_voucher_purchase_success_values, 0.50
        ),
        "third_party_voucher_purchase_no_stable_total_p50": percentile(
            third_party_voucher_purchase_no_stable_values, 0.50
        ),
        "third_party_voucher_purchase_reserve_protected_total_p50": percentile(
            third_party_voucher_purchase_reserve_protected_values, 0.50
        ),
        "third_party_voucher_purchase_no_route_total_p50": percentile(
            third_party_voucher_purchase_no_route_values, 0.50
        ),
        "third_party_voucher_purchase_no_target_total_p50": percentile(
            third_party_voucher_purchase_no_target_values, 0.50
        ),
        "third_party_voucher_purchase_stable_spent_usd_total_p50": percentile(
            third_party_voucher_purchase_stable_spent_values, 0.50
        ),
        "third_party_voucher_purchase_voucher_value_acquired_usd_total_p50": percentile(
            third_party_voucher_purchase_voucher_acquired_values, 0.50
        ),
        "lender_voucher_purchase_stable_budget_onramp_usd_total_p50": percentile(
            lender_voucher_purchase_budget_onramp_values, 0.50
        ),
        "consumer_voucher_purchase_stable_budget_onramp_usd_total_p50": percentile(
            consumer_voucher_purchase_budget_onramp_values, 0.50
        ),
        "third_party_voucher_purchase_stable_budget_onramp_usd_total_p50": percentile(
            third_party_voucher_purchase_budget_onramp_values, 0.50
        ),
        "lender_recovered_stable_usd_total_p50": percentile(lender_recovered_stable_values, 0.50),
        "lender_recovered_stable_borrower_regular_usd_total_p50": percentile(
            lender_recovered_stable_borrower_regular_values, 0.50
        ),
        "lender_recovered_stable_borrower_maturity_usd_total_p50": percentile(
            lender_recovered_stable_borrower_maturity_values, 0.50
        ),
        "lender_recovered_stable_consumer_purchase_usd_total_p50": percentile(
            lender_recovered_stable_consumer_purchase_values, 0.50
        ),
        "lender_recovered_stable_third_party_purchase_usd_total_p50": percentile(
            lender_recovered_stable_third_party_purchase_values, 0.50
        ),
        "lender_recovered_stable_other_usd_total_p50": percentile(
            lender_recovered_stable_other_values, 0.50
        ),
        "bond_service_reserved_usd_total_p50": percentile(bond_service_reserved_values, 0.50),
        "fee_service_reserved_usd_total_p50": percentile(fee_service_reserved_values, 0.50),
        "fee_service_stable_reserved_usd_total_p50": percentile(
            fee_service_stable_reserved_values, 0.50
        ),
        "fee_service_converted_voucher_reserved_usd_total_p50": percentile(
            fee_service_converted_voucher_reserved_values, 0.50
        ),
        "bond_service_reserve_balance_usd_p50": percentile(bond_service_reserve_balance_values, 0.50),
        "bond_service_paid_from_reserve_usd_total_p50": percentile(
            bond_service_paid_from_reserve_values, 0.50
        ),
        "bond_service_lockbox_target_usd_p50": percentile(bond_service_lockbox_target_values, 0.50),
        "bond_service_lockbox_coverage_ratio_p50": percentile(
            bond_service_lockbox_coverage_values, 0.50
        ),
        "bond_service_lockbox_mode": first.get("bond_service_lockbox_mode", ""),
        "lender_recovered_stable_sweepable_pending_usd_total_p50": percentile(
            lender_recovered_stable_sweepable_pending_values, 0.50
        ),
        "baseline_route_success_p50": baseline.get("route_success_p50", 0.0),
        "baseline_swap_volume_p50": baseline.get("swap_volume_p50", 0.0),
        "baseline_ordinary_swap_count_total_p50": baseline_ordinary_swap_count_p50,
        "baseline_ordinary_swap_volume_usd_total_p50": baseline_ordinary_swap_volume_p50,
        "baseline_ordinary_stable_source_swap_count_total_p50": (
            baseline_ordinary_stable_source_count_p50
        ),
        "baseline_ordinary_stable_source_swap_volume_usd_total_p50": (
            baseline_ordinary_stable_source_volume_p50
        ),
        "baseline_ordinary_voucher_source_swap_count_total_p50": (
            baseline_ordinary_voucher_source_count_p50
        ),
        "baseline_ordinary_voucher_source_swap_volume_usd_total_p50": (
            baseline_ordinary_voucher_source_volume_p50
        ),
        "baseline_voucher_to_voucher_count_p50": baseline_v2v_count_p50,
        "baseline_voucher_to_voucher_volume_p50": baseline_v2v_volume_p50,
        "baseline_voucher_to_voucher_share_p50": baseline_v2v_share_p50,
        "baseline_stable_value_share_p50": baseline_stable_share_p50,
        "baseline_stable_value_share_p95": baseline_stable_share_p95,
        "baseline_voucher_value_share_p50": baseline_voucher_share_p50,
        "baseline_household_cash_stress_p50": baseline_stress_p50,
        "baseline_household_cash_stress_p95": baseline_stress_p95,
        "baseline_consumer_cash_stress_p50": baseline_consumer_stress_p50,
        "baseline_consumer_cash_stress_p95": baseline_consumer_stress_p95,
        "baseline_community_cash_stress_p50": baseline_community_stress_p50,
        "baseline_community_cash_stress_p95": baseline_community_stress_p95,
        "baseline_liquidity_leakage_p50": baseline_leakage_p50,
        "baseline_liquidity_leakage_p95": baseline_leakage_p95,
        "baseline_producer_deposit_credit_capacity_usd_p50": baseline_producer_credit_capacity_p50,
        "baseline_producer_deposit_stable_usd_total_p50": baseline.get(
            "producer_deposit_stable_usd_total_p50", 0.0
        ),
        "baseline_producer_deposit_voucher_usd_total_p50": baseline.get(
            "producer_deposit_voucher_usd_total_p50", 0.0
        ),
        "baseline_producer_organic_voucher_deposit_usd_total_p50": baseline.get(
            "producer_organic_voucher_deposit_usd_total_p50", 0.0
        ),
        "baseline_producer_deposit_voucher_share_p50": baseline.get(
            "producer_deposit_voucher_share_p50", 0.0
        ),
        "baseline_productive_credit_inflow_usd_total_p50": baseline_productive_credit_inflow_p50,
        "baseline_productive_credit_stable_retained_usd_total_p50": (
            baseline_productive_credit_stable_retained_p50
        ),
        "baseline_productive_credit_voucher_deposit_usd_total_p50": (
            baseline_productive_credit_voucher_deposit_p50
        ),
        "baseline_productive_credit_voucher_deposit_share_p50": (
            baseline_productive_credit_voucher_deposit_share_p50
        ),
        "stable_dependency_delta_p95": stable_dependency_delta_p95,
        "material_decline_route_success_decline": int(route_decline),
        "diagnostic_total_swap_volume_decline": int(swap_volume_decline),
        "diagnostic_ordinary_swap_count_decline": int(ordinary_swap_count_decline),
        "diagnostic_ordinary_swap_volume_decline": int(ordinary_swap_volume_decline),
        "diagnostic_ordinary_stable_source_count_decline": int(
            ordinary_stable_source_count_decline
        ),
        "diagnostic_ordinary_stable_source_volume_decline": int(
            ordinary_stable_source_volume_decline
        ),
        "diagnostic_voucher_to_voucher_share_decline": int(v2v_share_decline),
        "material_decline_swap_volume_decline": 0,
        "material_decline_voucher_circulation_decline": int(voucher_circulation_decline),
        "material_decline_voucher_to_voucher_count_decline": int(v2v_count_decline),
        "material_decline_voucher_to_voucher_volume_decline": int(v2v_volume_decline),
        "material_decline_voucher_to_voucher_share_decline": 0,
        "material_decline_ordinary_voucher_source_count_decline": int(
            ordinary_voucher_source_count_decline
        ),
        "material_decline_ordinary_voucher_source_volume_decline": int(
            ordinary_voucher_source_volume_decline
        ),
        "material_decline_stable_dependency_increase": int(stable_dependency_increase),
        "material_decline_consumer_stress_increase": int(consumer_stress_increase),
        "material_decline_community_stress_increase": int(community_stress_increase),
        "material_decline_leakage_increase": int(leakage_increase),
        "material_decline_voucher_share_decline": int(voucher_value_share_decline),
        "material_decline_reasons": ";".join(material_reasons),
        "diagnostic_decline_reasons": ";".join(diagnostic_reasons),
        "voucher_to_voucher_count_decline_vs_no_bond": int(v2v_count_decline),
        "voucher_to_voucher_volume_decline_vs_no_bond": int(v2v_volume_decline),
        "voucher_to_voucher_share_decline_vs_no_bond": int(v2v_share_decline),
        "ordinary_voucher_source_count_decline_vs_no_bond": int(
            ordinary_voucher_source_count_decline
        ),
        "ordinary_voucher_source_volume_decline_vs_no_bond": int(
            ordinary_voucher_source_volume_decline
        ),
        "voucher_value_share_decline_vs_no_bond": int(voucher_value_share_decline),
        "material_decline_vs_no_bond": int(material_decline),
        "safe": int(not constraints),
        "binding_constraint": ";".join(constraints),
    }


def write_frontier_tables(output_dir: Path, safety_rows: list[dict[str, object]], frontier_rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scale & Coupon & Fee share & Safe principal & Binding constraint \\",
        r"\midrule",
    ]
    for row in frontier_rows[:18]:
        lines.append(
            "{scale} & {coupon} & {share} & {principal} & {binding} \\\\".format(
                scale=latex_escape(row["network_scale"]),
                coupon=fmt_pct(row["coupon_target_annual"]),
                share=fmt_pct(row["bond_fee_service_share"]),
                principal=f"{safe_float(row['safe_principal_usd']):,.0f}",
                binding=latex_escape(row.get("binding_constraint_at_frontier", "")),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    (output_dir / "safe_injection_frontier_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    guardrail_lines = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrrrl}",
        r"\toprule",
        r"Scale & Principal ratio & Pay p05 & Avail. headroom p50 & Route p05 & V2V share & Stable $\Delta$ p95 & Stress $\Delta$ p95 & Leak $\Delta$ p95 & Binding \\",
        r"\midrule",
    ]
    for row in safety_rows[:24]:
        guardrail_lines.append(
            "{scale} & {ratio:.2f} & {payment:.2f} & {headroom:.2f} & {route} & {v2v} & {stable} & {stress} & {leakage} & {binding} \\\\".format(
                scale=latex_escape(row["network_scale"]),
                ratio=safe_float(row["principal_ratio"]),
                payment=safe_float(row.get("scheduled_payment_coverage_p05")),
                headroom=safe_float(
                    row.get("available_service_cash_headroom_p50", row.get("service_cash_headroom_p50"))
                ),
                route=fmt_pct(row["route_success_p05"]),
                v2v=fmt_pct(row.get("voucher_to_voucher_share_p50", 0.0)),
                stable=fmt_pct(row.get("stable_dependency_delta_p95", 0.0)),
                stress=fmt_pct(row.get("household_cash_stress_delta_p95", 0.0)),
                leakage=fmt_pct(row.get("liquidity_leakage_delta_p95", 0.0)),
                binding=latex_escape(row["binding_constraint"] or "safe"),
            )
        )
    guardrail_lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    (output_dir / "non_extraction_guardrails_table.tex").write_text(
        "\n".join(guardrail_lines) + "\n", encoding="utf-8"
    )

    issuer_lines = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Scale & Principal ratio & Pay p05 & Pay p50 & Avail. headroom p50 & Reserve p05 & Unpaid p95 \\",
        r"\midrule",
    ]
    display_rows = sorted(
        safety_rows,
        key=lambda row: (
            str(row.get("network_scale")),
            safe_float(row.get("coupon_target_annual")),
            safe_float(row.get("bond_fee_service_share")),
            safe_float(row.get("principal_ratio")),
        ),
    )[:24]
    for row in display_rows:
        issuer_lines.append(
            "{scale} & {ratio:.2f} & {payment05:.2f} & {payment50:.2f} & {headroom50:.2f} & {reserve:,.0f} & {unpaid:,.0f} \\\\".format(
                scale=latex_escape(row["network_scale"]),
                ratio=safe_float(row["principal_ratio"]),
                payment05=safe_float(row.get("scheduled_payment_coverage_p05")),
                payment50=safe_float(row.get("scheduled_payment_coverage_p50")),
                headroom50=safe_float(
                    row.get("available_service_cash_headroom_p50", row.get("service_cash_headroom_p50"))
                ),
                reserve=safe_float(row.get("issuer_reserve_balance_p05")),
                unpaid=safe_float(row.get("issuer_unpaid_scheduled_claim_p95")),
            )
        )
    issuer_lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    (output_dir / "issuer_cashflow_table.tex").write_text(
        "\n".join(issuer_lines) + "\n", encoding="utf-8"
    )


def _draw_binding_constraint_heatmap(
    path: Path,
    safety_rows: list[dict[str, object]],
) -> None:
    from PIL import Image, ImageDraw

    constraints = sorted(
        {
            part
            for row in safety_rows
            for part in str(row.get("binding_constraint", "")).split(";")
            if part
        }
    )
    scales = [scale for scale in NETWORK_SCALE_FACTORS if any(row.get("network_scale") == scale for row in safety_rows)]
    if not constraints or not scales:
        return
    counts: dict[tuple[str, str], int] = defaultdict(int)
    max_count = 1
    for row in safety_rows:
        scale = str(row.get("network_scale", ""))
        for constraint in str(row.get("binding_constraint", "")).split(";"):
            if not constraint:
                continue
            counts[(constraint, scale)] += 1
            max_count = max(max_count, counts[(constraint, scale)])

    width = 1200
    row_h = 42
    col_w = 180
    left, top = 410, 110
    height = top + row_h * len(constraints) + 95
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_font(28)
    subtitle_font = _load_font(17)
    axis_font = _load_font(15)
    label_font = _load_font(15)
    _draw_text(draw, (left, 30), "Binding Non-Extraction Constraints", title_font)
    _draw_text(
        draw,
        (left, 64),
        "Cell counts by network scale; darker cells bind in more tested frontier cells.",
        subtitle_font,
        fill=(80, 80, 80),
    )
    for col, scale in enumerate(scales):
        x = left + col * col_w + col_w // 2
        _draw_text(draw, (x, top - 20), scale, axis_font, fill=(55, 55, 55), anchor="ma")
    for row_idx, constraint in enumerate(constraints):
        y = top + row_idx * row_h
        _draw_text(draw, (left - 14, y + row_h // 2 - 7), constraint, label_font, fill=(40, 40, 40), anchor="rm")
        for col, scale in enumerate(scales):
            value = counts.get((constraint, scale), 0)
            intensity = value / max_count if max_count else 0.0
            color = (
                int(245 - 175 * intensity),
                int(247 - 150 * intensity),
                int(250 - 55 * intensity),
                255,
            )
            x0 = left + col * col_w
            draw.rectangle((x0, y, x0 + col_w - 8, y + row_h - 8), fill=color, outline=(210, 210, 210))
            _draw_text(draw, (x0 + col_w // 2 - 4, y + row_h // 2 - 7), str(value), label_font, fill=(20, 20, 20), anchor="ma")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def write_frontier_figures(
    output_dir: Path,
    frontier_rows: list[dict[str, object]],
    safety_rows: list[dict[str, object]],
) -> None:
    if not frontier_rows:
        return
    labels = [str(row["network_scale"]) for row in frontier_rows]
    principals = [safe_float(row["safe_principal_usd"]) for row in frontier_rows]
    _draw_grouped_bar_chart(
        output_dir / "fig_safe_injection_frontier.png",
        "Safe Non-Extractive Principal Frontier",
        "Maximum safe principal by connected pool-network scale over the tested grid.",
        labels,
        [("safe principal USD", principals, (51, 102, 204, 255))],
    )
    payment_p05 = [
        safe_float(row.get("scheduled_payment_coverage_p05")) for row in frontier_rows
    ]
    headroom_p50 = [
        safe_float(row.get("available_service_cash_headroom_p50", row.get("service_cash_headroom_p50")))
        for row in frontier_rows
    ]
    _draw_grouped_bar_chart(
        output_dir / "fig_issuer_service_coverage.png",
        "Issuer Payment Coverage and Cash Headroom at Frontier",
        "Scheduled payment coverage and spendable recovered-stable headroom at the headline safe principal by connected network scale.",
        labels,
        [
            ("p05 payment coverage", payment_p05, (204, 86, 70, 255)),
            ("p50 available cash headroom", headroom_p50, (51, 102, 204, 255)),
        ],
    )
    _draw_binding_constraint_heatmap(output_dir / "fig_binding_constraints_heatmap.png", safety_rows)


def write_frontier_notes(output_dir: Path, args: argparse.Namespace, summary_rows: list[dict[str, object]]) -> None:
    validation_status = frontier_validation_gate_status(output_dir)
    lines = [
        "# Bond-Issuer Safety Frontier",
        "",
        f"- Scenario: `bond_issuer_frontier`.",
        f"- Runs per cell: {args.runs}.",
        f"- Horizon: {args.ticks} weekly ticks.",
        f"- Term: {selected_terms(args)[0] if selected_terms(args) else args.ticks} weekly ticks.",
        f"- Full engine-validation gate status: `{validation_status}`.",
        f"- Frontier mode: `{args.frontier_mode}` with {args.frontier_refinement_rounds} refinement round(s).",
        f"- Certification policy: `{args.certification_policy}`.",
        "- Issuer reserve share: 0.00%; the frontier setup deploys 100% of gross principal directly to lender pools.",
        f"- Issuer payment stride: {args.issuer_payment_stride} weekly ticks.",
        f"- p05 route-success floor: {max(0.0, min(1.0, float(args.route_success_floor))):.1%}.",
        "- Gross bond principal is the bond amount; it is divided evenly across lender pools as lendable stable at initialization, bypassing the startup waterfall.",
        "- Strong pools are eligible at full weight; moderate pools are capped; weak pools are excluded from base runs unless another policy is explicitly selected.",
        "",
        "## Non-Extraction Gate",
        "",
        "A cell is safe only when scheduled bond payments clear, voucher-to-voucher circulation preservation, active-pool stable-dependency limits, incremental household or community cash stress, incremental liquidity leakage, unpaid claims, edge concentration, and matched no-bond degradation tests all pass.",
        "Available service-cash headroom is reported separately as issuer operating and risk-capital capacity. The 1.25x threshold is an issuer-headroom frontier, not an additional bondholder-payment requirement.",
        "- The route-success floor is a model settlement-reliability sensitivity parameter, not a direct empirical Sarafu failed-route scalar.",
        "- Voucher-to-voucher count and share are compared against the matched no-bond baseline to protect the empirically observed ROLA-like settlement motif.",
        "- Stable value share and voucher value share in active pools are compared against the matched no-bond baseline so stable/bond injections do not crowd out voucher-backed settlement capacity.",
        "Cash-stress and liquidity-leakage guardrails are evaluated as deltas against the matched no-bond baseline for the same network scale and seeds.",
        "",
        "## Headline Frontier",
        "",
    ]
    for row in summary_rows:
        lines.append(
            "- `{scale}`: safe principal {principal:,.0f} USD at ratio {ratio:.2f}.".format(
                scale=row["network_scale"],
                principal=safe_float(row["safe_principal_usd"]),
                ratio=safe_float(row["safe_principal_ratio"]),
            )
        )
    (output_dir / "paper_integration_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def issuer_cashflow_summary_rows(safety_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in safety_rows:
        rows.append(
            {
                "scenario": row.get("scenario", "bond_issuer_frontier"),
                "network_scale": row.get("network_scale", ""),
                "coupon_target_annual": row.get("coupon_target_annual", 0.0),
                "bond_fee_service_share": row.get("bond_fee_service_share", 0.0),
                "principal_ratio": row.get("principal_ratio", 0.0),
                "producer_stable_debt_contract_service_margin_rate": row.get(
                    "producer_stable_debt_contract_service_margin_rate", 0.0
                ),
                "producer_voucher_debt_contract_service_margin_rate": row.get(
                    "producer_voucher_debt_contract_service_margin_rate", 0.0
                ),
                "principal_usd_p50": row.get("principal_usd_p50", 0.0),
                "service_coverage_p05": row.get("service_coverage_p05", 0.0),
                "service_coverage_p50": row.get("service_coverage_p50", 0.0),
                "scheduled_payment_coverage_p05": row.get("scheduled_payment_coverage_p05", 0.0),
                "scheduled_payment_coverage_p50": row.get("scheduled_payment_coverage_p50", 0.0),
                "service_cash_headroom_p05": row.get("service_cash_headroom_p05", 0.0),
                "service_cash_headroom_p50": row.get("service_cash_headroom_p50", 0.0),
                "gross_historical_service_cash_headroom_p05": row.get(
                    "gross_historical_service_cash_headroom_p05", row.get("service_cash_headroom_p05", 0.0)
                ),
                "gross_historical_service_cash_headroom_p50": row.get(
                    "gross_historical_service_cash_headroom_p50", row.get("service_cash_headroom_p50", 0.0)
                ),
                "available_service_cash_headroom_p05": row.get(
                    "available_service_cash_headroom_p05", row.get("service_cash_headroom_p05", 0.0)
                ),
                "available_service_cash_headroom_p50": row.get(
                    "available_service_cash_headroom_p50", row.get("service_cash_headroom_p50", 0.0)
                ),
                "issuer_lockbox_surplus_usd_p50": row.get("issuer_lockbox_surplus_usd_p50", 0.0),
                "issuer_excess_recovered_stable_usd_p50": row.get(
                    "issuer_excess_recovered_stable_usd_p50", 0.0
                ),
                "issuer_operating_surplus_usd_p50": row.get("issuer_operating_surplus_usd_p50", 0.0),
                "issuer_operating_surplus_headroom_ratio_p50": row.get(
                    "issuer_operating_surplus_headroom_ratio_p50", 0.0
                ),
                "issuer_fee_waterfall_ops_usd_p50": row.get("issuer_fee_waterfall_ops_usd_p50", 0.0),
                "issuer_sustainability_resources_usd_p50": row.get(
                    "issuer_sustainability_resources_usd_p50", 0.0
                ),
                "issuer_sustainability_headroom_ratio_p50": row.get(
                    "issuer_sustainability_headroom_ratio_p50", 0.0
                ),
                "bond_service_lockbox_target_usd_p50": row.get("bond_service_lockbox_target_usd_p50", 0.0),
                "bond_service_lockbox_coverage_ratio_p50": row.get(
                    "bond_service_lockbox_coverage_ratio_p50", 0.0
                ),
                "bond_service_lockbox_mode": row.get("bond_service_lockbox_mode", ""),
                "fee_service_reserved_usd_total_p50": row.get(
                    "fee_service_reserved_usd_total_p50", 0.0
                ),
                "fee_service_stable_reserved_usd_total_p50": row.get(
                    "fee_service_stable_reserved_usd_total_p50", 0.0
                ),
                "fee_service_converted_voucher_reserved_usd_total_p50": row.get(
                    "fee_service_converted_voucher_reserved_usd_total_p50", 0.0
                ),
                "issuer_scheduled_debt_service_due_p50": row.get("issuer_scheduled_debt_service_due_p50", 0.0),
                "issuer_actual_bondholder_payment_p50": row.get("issuer_actual_bondholder_payment_p50", 0.0),
                "issuer_reserve_balance_p05": row.get("issuer_reserve_balance_p05", 0.0),
                "issuer_reserve_draw_p95": row.get("issuer_reserve_draw_p95", 0.0),
                "issuer_unpaid_scheduled_claim_p95": row.get("issuer_unpaid_scheduled_claim_p95", 0.0),
                "safe": row.get("safe", 0),
                "binding_constraint": row.get("binding_constraint", ""),
            }
        )
    return rows


def run_frontier_cell(
    args: argparse.Namespace,
    calibration: Calibration,
    *,
    network_scale: str,
    principal_ratio: float,
    principal_usd: float,
    coupon: float,
    service_share: float,
    term_ticks: int,
    seed_offset: int,
) -> list[dict[str, object]]:
    capacity = certified_pool_capacity(calibration, network_scale, args.certification_policy)
    apply_network_context(
        args,
        calibration=calibration,
        network_scale=network_scale,
        principal_ratio=principal_ratio,
        principal_usd=principal_usd,
        service_share=service_share,
        certification_policy=args.certification_policy,
    )
    args._current_certified_pool_count = capacity["certified_pool_count"]
    args._current_certified_capacity_usd = capacity["certified_backing_capacity_usd"]
    rows = []
    for run_idx in range(1, int(args.runs) + 1):
        seed = int(args.seed) + seed_offset + run_idx
        print(
            "[{run}/{runs}] scenario=bond_issuer_frontier scale={scale} principal_ratio={ratio} coupon={coupon} share={share} seed={seed}".format(
                run=run_idx,
                runs=args.runs,
                scale=network_scale,
                ratio=principal_ratio,
                coupon=coupon,
                share=service_share,
                seed=seed,
            ),
            flush=True,
        )
        args._progress_run_position = run_idx
        args._progress_total_runs = int(args.runs)
        _, _, _, summary = run_one(
            scenario="bond_issuer_frontier",
            coupon=coupon,
            term_ticks=term_ticks,
            run_index=run_idx,
            seed=seed,
            args=args,
            calibration=calibration,
        )
        rows.append(summary)
    return rows


def frontier_cell_key(scale: str, ratio: float, coupon: float, share: float) -> tuple[str, float, float, float]:
    return (scale, round(float(ratio), 8), round(float(coupon), 8), round(float(share), 8))


def frontier_validation_gate_status(output_dir: Path) -> str:
    summary_path = output_dir.parent / "engine_validation" / "engine_validation_summary.csv"
    if not summary_path.exists():
        return "missing"
    rows = read_csv(summary_path)
    if not rows:
        return "missing"
    return str(rows[0].get("status", "")).strip().lower() or "missing"


def frontier_token(value: object) -> str:
    text = f"{safe_float(value):.8f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p") or "0"


def frontier_job_id(phase: str, scale: str, ratio: float, coupon: float, share: float) -> str:
    return (
        f"frontier_{phase}_{scale}_"
        f"r{frontier_token(ratio)}_c{frontier_token(coupon)}_s{frontier_token(share)}"
    )


def frontier_baseline_metrics(baseline_rows: list[dict[str, object]]) -> dict[str, float]:
    producer_deposit_stable_values = [
        safe_float(row.get("producer_deposit_stable_usd_total")) for row in baseline_rows
    ]
    producer_deposit_voucher_values = [
        safe_float(row.get("producer_deposit_voucher_usd_total")) for row in baseline_rows
    ]
    productive_credit_inflow_values = [
        safe_float(row.get("productive_credit_inflow_usd_total")) for row in baseline_rows
    ]
    productive_credit_stable_retained_values = [
        safe_float(row.get("productive_credit_stable_retained_usd_total")) for row in baseline_rows
    ]
    productive_credit_voucher_deposit_values = [
        safe_float(row.get("productive_credit_voucher_deposit_usd_total")) for row in baseline_rows
    ]
    productive_credit_voucher_deposit_share_values = [
        voucher / max(1e-9, inflow)
        for voucher, inflow in zip(
            productive_credit_voucher_deposit_values,
            productive_credit_inflow_values,
        )
    ]
    producer_organic_voucher_deposit_values = [
        max(0.0, total - productive)
        for total, productive in zip(
            producer_deposit_voucher_values,
            productive_credit_voucher_deposit_values,
        )
    ]
    producer_deposit_voucher_share_values = [
        voucher / max(1e-9, stable + voucher)
        for stable, voucher in zip(producer_deposit_stable_values, producer_deposit_voucher_values)
    ]
    return {
        "route_success_p50": percentile(
            [safe_float(row.get("route_success_rate_cumulative")) for row in baseline_rows], 0.50
        ),
        "swap_volume_p50": percentile(
            [safe_float(row.get("swap_volume_usd_total")) for row in baseline_rows], 0.50
        ),
        "voucher_to_voucher_count_p50": percentile(
            [safe_float(row.get("swap_count_vchr_to_vchr_total")) for row in baseline_rows], 0.50
        ),
        "voucher_to_voucher_volume_p50": percentile(
            [safe_float(row.get("swap_volume_vchr_to_vchr_total")) for row in baseline_rows], 0.50
        ),
        "voucher_to_voucher_share_p50": percentile(
            [
                safe_float(row.get("swap_count_vchr_to_vchr_total"))
                / max(1e-9, safe_float(row.get("transactions_total")))
                for row in baseline_rows
            ],
            0.50,
        ),
        "stable_value_share_p50": percentile(
            [safe_float(row.get("stable_value_share_in_active_pools")) for row in baseline_rows], 0.50
        ),
        "stable_value_share_p95": percentile(
            [safe_float(row.get("stable_value_share_in_active_pools")) for row in baseline_rows], 0.95
        ),
        "voucher_value_share_p50": percentile(
            [safe_float(row.get("voucher_value_share_in_active_pools")) for row in baseline_rows], 0.50
        ),
        "household_cash_stress_p50": percentile(
            [safe_float(row.get("household_cash_stress_ratio")) for row in baseline_rows], 0.50
        ),
        "household_cash_stress_p95": percentile(
            [safe_float(row.get("household_cash_stress_ratio")) for row in baseline_rows], 0.95
        ),
        "consumer_cash_stress_p50": percentile(
            [
                safe_float(row.get("consumer_stable_reserve_stress_ratio", row.get("household_cash_stress_ratio")))
                for row in baseline_rows
            ],
            0.50,
        ),
        "consumer_cash_stress_p95": percentile(
            [
                safe_float(row.get("consumer_stable_reserve_stress_ratio", row.get("household_cash_stress_ratio")))
                for row in baseline_rows
            ],
            0.95,
        ),
        "community_cash_stress_p50": percentile(
            [
                safe_float(row.get("community_stable_reserve_stress_ratio", row.get("household_cash_stress_ratio")))
                for row in baseline_rows
            ],
            0.50,
        ),
        "community_cash_stress_p95": percentile(
            [
                safe_float(row.get("community_stable_reserve_stress_ratio", row.get("household_cash_stress_ratio")))
                for row in baseline_rows
            ],
            0.95,
        ),
        "liquidity_leakage_p50": percentile(
            [safe_float(row.get("stable_liquidity_leakage_ratio_cumulative")) for row in baseline_rows], 0.50
        ),
        "liquidity_leakage_p95": percentile(
            [safe_float(row.get("stable_liquidity_leakage_ratio_cumulative")) for row in baseline_rows], 0.95
        ),
        "producer_deposit_credit_capacity_usd_p50": percentile(
            [safe_float(row.get("producer_deposit_credit_capacity_usd")) for row in baseline_rows], 0.50
        ),
        "producer_deposit_stable_usd_total_p50": percentile(producer_deposit_stable_values, 0.50),
        "producer_deposit_voucher_usd_total_p50": percentile(producer_deposit_voucher_values, 0.50),
        "producer_organic_voucher_deposit_usd_total_p50": percentile(
            producer_organic_voucher_deposit_values, 0.50
        ),
        "producer_deposit_voucher_share_p50": percentile(
            producer_deposit_voucher_share_values, 0.50
        ),
        "productive_credit_inflow_usd_total_p50": percentile(productive_credit_inflow_values, 0.50),
        "productive_credit_stable_retained_usd_total_p50": percentile(
            productive_credit_stable_retained_values, 0.50
        ),
        "productive_credit_voucher_deposit_usd_total_p50": percentile(
            productive_credit_voucher_deposit_values, 0.50
        ),
        "productive_credit_voucher_deposit_share_p50": percentile(
            productive_credit_voucher_deposit_share_values, 0.50
        ),
        "ordinary_swap_volume_usd_total_p50": percentile(
            [safe_float(row.get("ordinary_swap_volume_usd_total")) for row in baseline_rows], 0.50
        ),
        "ordinary_swap_count_total_p50": percentile(
            [safe_float(row.get("ordinary_swap_count_total")) for row in baseline_rows], 0.50
        ),
        "ordinary_stable_source_swap_count_total_p50": percentile(
            [safe_float(row.get("ordinary_stable_source_swap_count_total")) for row in baseline_rows],
            0.50,
        ),
        "ordinary_stable_source_swap_volume_usd_total_p50": percentile(
            [
                safe_float(row.get("ordinary_stable_source_swap_volume_usd_total"))
                for row in baseline_rows
            ],
            0.50,
        ),
        "ordinary_voucher_source_swap_count_total_p50": percentile(
            [safe_float(row.get("ordinary_voucher_source_swap_count_total")) for row in baseline_rows], 0.50
        ),
        "ordinary_voucher_source_swap_volume_usd_total_p50": percentile(
            [safe_float(row.get("ordinary_voucher_source_swap_volume_usd_total")) for row in baseline_rows], 0.50
        ),
    }


def calibration_activity_audit_rows(
    args: argparse.Namespace,
    calibration: Calibration,
    rows: list[dict[str, object]],
    *,
    scenario: str,
    prefix: str,
) -> list[dict[str, object]]:
    field_template = {
        "scenario": scenario,
        "row_type": "",
        "tier": "",
        "configured_tier_probability": 0.0,
        "configured_producer_stable_deposit_rate_per_month": 0.0,
        "configured_producer_voucher_deposit_rate_per_month": 0.0,
        "configured_producer_deposit_stable_share": 0.0,
        "configured_producer_deposit_voucher_share": 0.0,
        "configured_productive_credit_return_rate": 0.0,
        "configured_productive_credit_lag_ticks_p50": 0.0,
        "configured_productive_credit_stable_retained_share": 0.0,
        "configured_productive_credit_voucher_deposit_share": 0.0,
        "configured_productive_credit_voucher_deposit_cap_rate_per_month": 0.0,
        "configured_producer_stable_debt_contract_service_margin_rate": 0.0,
        "configured_producer_voucher_debt_contract_service_margin_rate": 0.0,
        "realized_productive_credit_inflow_usd_p50": 0.0,
        "realized_productive_credit_stable_retained_usd_p50": 0.0,
        "realized_productive_credit_voucher_deposit_usd_p50": 0.0,
        "realized_productive_credit_voucher_deposit_share_p50": 0.0,
        "realized_productive_credit_voucher_deposit_cap_clipped_usd_p50": 0.0,
        "realized_producer_deposit_stable_usd_p50": 0.0,
        "realized_producer_deposit_voucher_usd_p50": 0.0,
        "realized_producer_deposit_voucher_share_p50": 0.0,
        "realized_producer_organic_voucher_deposit_usd_p50": 0.0,
        "realized_ordinary_swap_volume_usd_p50": 0.0,
        "realized_ordinary_stable_source_swap_volume_usd_p50": 0.0,
        "realized_ordinary_voucher_source_swap_volume_usd_p50": 0.0,
        "realized_ordinary_stable_spend_protected_skip_value_usd_p50": 0.0,
    }

    audit_rows: list[dict[str, object]] = []
    legacy_margin = getattr(args, "producer_debt_contract_service_margin_rate", None)
    shared_margin = legacy_margin if legacy_margin is not None else 0.0
    configured_stable_margin = getattr(args, "producer_stable_debt_contract_service_margin_rate", None)
    configured_voucher_margin = getattr(args, "producer_voucher_debt_contract_service_margin_rate", None)
    if configured_stable_margin is None:
        configured_stable_margin = shared_margin
    if configured_voucher_margin is None:
        configured_voucher_margin = shared_margin
    for tier in TIER_ORDER:
        dep = calibration.producer_deposit_by_tier.get(tier, {})
        prod = calibration.productive_credit_by_tier.get(tier, {})
        voucher_share = safe_float(prod.get("loan_induced_voucher_deposit_share_default"))
        stable_share = max(0.0, 1.0 - voucher_share)
        row = dict(field_template)
        row.update(
            {
                "row_type": "configured_tier",
                "tier": tier,
                "configured_tier_probability": safe_float(calibration.tier_probs.get(tier)),
                "configured_producer_stable_deposit_rate_per_month": safe_float(
                    dep.get("stable_deposit_rate_per_month")
                ),
                "configured_producer_voucher_deposit_rate_per_month": safe_float(
                    dep.get("voucher_deposit_rate_per_month")
                ),
                "configured_producer_deposit_stable_share": safe_float(
                    dep.get("stable_deposit_share_of_backing")
                ),
                "configured_producer_deposit_voucher_share": safe_float(
                    dep.get("voucher_deposit_share_of_backing")
                ),
                "configured_productive_credit_return_rate": safe_float(
                    prod.get("productive_credit_return_rate")
                ),
                "configured_productive_credit_lag_ticks_p50": safe_float(
                    prod.get("productive_credit_lag_ticks_p50")
                ),
                "configured_productive_credit_stable_retained_share": stable_share,
                "configured_productive_credit_voucher_deposit_share": voucher_share,
                "configured_productive_credit_voucher_deposit_cap_rate_per_month": safe_float(
                    prod.get("loan_induced_voucher_deposit_cap_rate_per_month")
                ),
                "configured_producer_stable_debt_contract_service_margin_rate": safe_float(
                    configured_stable_margin
                ),
                "configured_producer_voucher_debt_contract_service_margin_rate": safe_float(
                    configured_voucher_margin
                ),
            }
        )
        audit_rows.append(row)

    stable_deposit_values = [safe_float(row.get("producer_deposit_stable_usd_total")) for row in rows]
    voucher_deposit_values = [safe_float(row.get("producer_deposit_voucher_usd_total")) for row in rows]
    productive_inflow_values = [safe_float(row.get("productive_credit_inflow_usd_total")) for row in rows]
    productive_stable_values = [
        safe_float(row.get("productive_credit_stable_retained_usd_total")) for row in rows
    ]
    productive_voucher_values = [
        safe_float(row.get("productive_credit_voucher_deposit_usd_total")) for row in rows
    ]
    cap_clipped_values = [
        safe_float(row.get("productive_credit_voucher_deposit_cap_clipped_usd_total")) for row in rows
    ]
    voucher_share_values = [
        voucher / max(1e-9, inflow)
        for voucher, inflow in zip(productive_voucher_values, productive_inflow_values)
    ]
    deposit_voucher_share_values = [
        voucher / max(1e-9, stable + voucher)
        for stable, voucher in zip(stable_deposit_values, voucher_deposit_values)
    ]
    organic_voucher_values = [
        max(0.0, total - productive)
        for total, productive in zip(voucher_deposit_values, productive_voucher_values)
    ]
    configured_voucher_share = safe_float(
        getattr(args, f"_{prefix}_productive_credit_voucher_deposit_share", 0.0)
    )
    configured_deposit_stable_share = sum(
        safe_float(calibration.tier_probs.get(tier))
        * safe_float(calibration.producer_deposit_by_tier.get(tier, {}).get("stable_deposit_share_of_backing"))
        for tier in TIER_ORDER
    )
    configured_deposit_voucher_share = sum(
        safe_float(calibration.tier_probs.get(tier))
        * safe_float(calibration.producer_deposit_by_tier.get(tier, {}).get("voucher_deposit_share_of_backing"))
        for tier in TIER_ORDER
    )
    aggregate = dict(field_template)
    aggregate.update(
        {
            "row_type": "realized_all_runs",
            "tier": "all",
            "configured_producer_stable_deposit_rate_per_month": safe_float(
                getattr(args, f"_{prefix}_producer_stable_deposit_rate_per_month", 0.0)
            ),
            "configured_producer_voucher_deposit_rate_per_month": safe_float(
                getattr(args, f"_{prefix}_producer_voucher_deposit_rate_per_month", 0.0)
            ),
            "configured_producer_deposit_stable_share": configured_deposit_stable_share,
            "configured_producer_deposit_voucher_share": configured_deposit_voucher_share,
            "configured_productive_credit_return_rate": safe_float(
                getattr(args, f"_{prefix}_productive_credit_return_rate", 0.0)
            ),
            "configured_productive_credit_lag_ticks_p50": safe_float(
                getattr(args, f"_{prefix}_productive_credit_lag_ticks", 0.0)
            ),
            "configured_productive_credit_stable_retained_share": max(
                0.0, 1.0 - configured_voucher_share
            ),
            "configured_productive_credit_voucher_deposit_share": configured_voucher_share,
            "configured_productive_credit_voucher_deposit_cap_rate_per_month": safe_float(
                getattr(args, f"_{prefix}_productive_credit_voucher_deposit_cap_rate_per_month", 0.0)
            ),
            "configured_producer_stable_debt_contract_service_margin_rate": safe_float(
                configured_stable_margin
            ),
            "configured_producer_voucher_debt_contract_service_margin_rate": safe_float(
                configured_voucher_margin
            ),
            "realized_productive_credit_inflow_usd_p50": percentile(productive_inflow_values, 0.50),
            "realized_productive_credit_stable_retained_usd_p50": percentile(
                productive_stable_values, 0.50
            ),
            "realized_productive_credit_voucher_deposit_usd_p50": percentile(
                productive_voucher_values, 0.50
            ),
            "realized_productive_credit_voucher_deposit_share_p50": percentile(
                voucher_share_values, 0.50
            ),
            "realized_productive_credit_voucher_deposit_cap_clipped_usd_p50": percentile(
                cap_clipped_values, 0.50
            ),
            "realized_producer_deposit_stable_usd_p50": percentile(stable_deposit_values, 0.50),
            "realized_producer_deposit_voucher_usd_p50": percentile(voucher_deposit_values, 0.50),
            "realized_producer_deposit_voucher_share_p50": percentile(
                deposit_voucher_share_values, 0.50
            ),
            "realized_producer_organic_voucher_deposit_usd_p50": percentile(
                organic_voucher_values, 0.50
            ),
            "realized_ordinary_swap_volume_usd_p50": percentile(
                [safe_float(row.get("ordinary_swap_volume_usd_total")) for row in rows],
                0.50,
            ),
            "realized_ordinary_stable_source_swap_volume_usd_p50": percentile(
                [safe_float(row.get("ordinary_stable_source_swap_volume_usd_total")) for row in rows],
                0.50,
            ),
            "realized_ordinary_voucher_source_swap_volume_usd_p50": percentile(
                [safe_float(row.get("ordinary_voucher_source_swap_volume_usd_total")) for row in rows],
                0.50,
            ),
            "realized_ordinary_stable_spend_protected_skip_value_usd_p50": percentile(
                [
                    safe_float(row.get("ordinary_stable_spend_protected_skip_value_usd_total"))
                    for row in rows
                ],
                0.50,
            ),
        }
    )
    audit_rows.append(aggregate)
    return audit_rows


def load_frontier_cell_shard(job: dict[str, object], shard_root: Path, config_hash: str) -> dict[str, object] | None:
    return shard_result_from_files(
        kind="frontier",
        job=job,
        shard_root=shard_root,
        config_hash=config_hash,
        csv_names=("rows",),
    )


def load_frontier_run_summary(run_dir: Path, run_hash: str) -> dict[str, object] | None:
    if not completed_manifest(run_dir / "manifest.json", run_hash):
        return None
    rows = read_rows(run_dir / "summary.csv")
    return dict(rows[0]) if rows else None


def run_frontier_cell_shard(
    job: dict[str, object],
    args_data: dict[str, object],
    calibration: Calibration,
    shard_root_text: str,
    resume: bool,
) -> dict[str, object]:
    shard_root = Path(shard_root_text)
    job_dir = shard_job_dir(shard_root, "frontier", str(job["job_id"]))
    config_hash = str(job["config_hash"])
    if resume:
        completed = load_frontier_cell_shard(job, shard_root, config_hash)
        if completed is not None:
            return completed
    started_at = time.time()
    try:
        args = namespace_from_payload(args_data)
        apply_unit_normalization_context(args, calibration)
        configure_sarafu_activity_controls(args, calibration, int(args.ticks), "frontier")
        scale = str(job["network_scale"])
        ratio = safe_float(job["principal_ratio"])
        coupon = safe_float(job["coupon"])
        share = safe_float(job["service_share"])
        principal_usd = safe_float(job["principal_usd"])
        term_ticks = int(job["term_ticks"])
        seed_offset = int(job["seed_offset"])
        capacity = certified_pool_capacity(calibration, scale, args.certification_policy)
        apply_network_context(
            args,
            calibration=calibration,
            network_scale=scale,
            principal_ratio=ratio,
            principal_usd=principal_usd,
            service_share=share,
            certification_policy=args.certification_policy,
        )
        args._current_certified_pool_count = capacity["certified_pool_count"]
        args._current_certified_capacity_usd = capacity["certified_backing_capacity_usd"]
        rows: list[dict[str, object]] = []
        for run_idx in range(1, int(args.runs) + 1):
            seed = int(args.seed) + seed_offset + run_idx
            run_hash = canonical_hash({"cell_config_hash": config_hash, "run_idx": run_idx, "seed": seed})
            run_dir = job_dir / "runs" / f"run_{run_idx:06d}"
            if resume:
                summary = load_frontier_run_summary(run_dir, run_hash)
                if summary is not None:
                    rows.append(summary)
                    continue
            print(
                "[{run}/{runs}] scenario=bond_issuer_frontier scale={scale} principal_ratio={ratio} coupon={coupon} share={share} seed={seed}".format(
                    run=run_idx,
                    runs=args.runs,
                    scale=scale,
                    ratio=ratio,
                    coupon=coupon,
                    share=share,
                    seed=seed,
                ),
                flush=True,
            )
            args._progress_run_position = run_idx
            args._progress_total_runs = int(args.runs)
            _, _, _, summary = run_one(
                scenario="bond_issuer_frontier",
                coupon=coupon,
                term_ticks=term_ticks,
                run_index=run_idx,
                seed=seed,
                args=args,
                calibration=calibration,
            )
            write_rows(run_dir / "summary.csv", [summary])
            write_shard_manifest(
                run_dir,
                job={**job, "run_idx": run_idx, "seed": seed},
                config_hash=run_hash,
                status="completed",
                files={"summary": 1},
            )
            rows.append(summary)
        rows = sorted_run_rows(rows)
        write_rows(job_dir / "rows.csv", rows)
        write_shard_manifest(
            job_dir,
            job=job,
            config_hash=config_hash,
            status="completed",
            files={"rows": len(rows)},
            started_at=started_at,
        )
        return {"status": "completed", "job": job, "shard_dir": str(job_dir), "rows": rows}
    except BaseException:
        error = traceback.format_exc()
        write_shard_manifest(
            job_dir,
            job=job,
            config_hash=config_hash,
            status="failed",
            error=error,
            started_at=started_at,
        )
        return failed_result(job, error)


def frontier_results_to_run_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.extend(result.get("rows", []))
    return sorted_run_rows(rows)


def split_frontier_results(
    results: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline_results: list[dict[str, object]] = []
    cell_results: list[dict[str, object]] = []
    for result in results:
        job = result.get("job")
        phase = str(job.get("phase", "")) if isinstance(job, dict) else ""
        if phase == "baseline":
            baseline_results.append(result)
        else:
            cell_results.append(result)
    return baseline_results, cell_results


def frontier_baselines_by_scale(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    baseline_by_scale: dict[str, dict[str, float]] = {}
    for result in results:
        rows = list(result.get("rows", []))
        if not rows:
            continue
        scale = str(rows[0].get("network_scale", ""))
        baseline_by_scale[scale] = frontier_baseline_metrics(rows)
    return baseline_by_scale


def frontier_safety_from_results(
    cell_results: list[dict[str, object]],
    baseline_by_scale: dict[str, dict[str, float]],
    route_success_floor: float,
    route_success_mode: str,
) -> list[dict[str, object]]:
    safety_rows = []
    seen: set[tuple[str, float, float, float]] = set()
    for result in cell_results:
        rows = list(result.get("rows", []))
        if not rows:
            continue
        first = rows[0]
        key = frontier_cell_key(
            str(first.get("network_scale", "")),
            safe_float(first.get("principal_ratio")),
            safe_float(first.get("coupon_target_annual")),
            safe_float(first.get("bond_fee_service_share")),
        )
        if key in seen:
            continue
        baseline = baseline_by_scale.get(key[0])
        if baseline is None:
            continue
        safety_rows.append(summarize_frontier_cell(rows, baseline, route_success_floor, route_success_mode))
        seen.add(key)
    return sorted_frontier_safety_rows(safety_rows)


def frontier_summary_rows(
    network_scales: list[str],
    safety_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frontier_rows = []
    grouped: dict[tuple[str, float, float], list[dict[str, object]]] = defaultdict(list)
    for row in safety_rows:
        grouped[
            (
                str(row["network_scale"]),
                safe_float(row["coupon_target_annual"]),
                safe_float(row["bond_fee_service_share"]),
            )
        ].append(row)
    for (scale, coupon, share), rows in sorted(grouped.items()):
        safe_rows = [row for row in rows if int(row["safe"]) == 1]
        if safe_rows:
            best = max(safe_rows, key=lambda row: safe_float(row["principal_usd_p50"]))
            binding = ""
        else:
            best = min(rows, key=lambda row: safe_float(row["principal_ratio"]))
            binding = str(best.get("binding_constraint", ""))
        frontier_rows.append(
            {
                "network_scale": scale,
                "coupon_target_annual": coupon,
                "bond_fee_service_share": share,
                "safe_principal_ratio": safe_float(best.get("principal_ratio")) if safe_rows else 0.0,
                "safe_principal_usd": safe_float(best.get("principal_usd_p50")) if safe_rows else 0.0,
                "binding_constraint_at_frontier": binding,
                "service_coverage_p05": safe_float(best.get("service_coverage_p05")),
                "service_coverage_p50": safe_float(best.get("service_coverage_p50")),
                "scheduled_payment_coverage_p05": safe_float(
                    best.get("scheduled_payment_coverage_p05")
                ),
                "scheduled_payment_coverage_p50": safe_float(
                    best.get("scheduled_payment_coverage_p50")
                ),
                "service_cash_headroom_p05": safe_float(best.get("service_cash_headroom_p05")),
                "service_cash_headroom_p50": safe_float(best.get("service_cash_headroom_p50")),
                "gross_historical_service_cash_headroom_p05": safe_float(
                    best.get("gross_historical_service_cash_headroom_p05", best.get("service_cash_headroom_p05"))
                ),
                "gross_historical_service_cash_headroom_p50": safe_float(
                    best.get("gross_historical_service_cash_headroom_p50", best.get("service_cash_headroom_p50"))
                ),
                "available_service_cash_headroom_p05": safe_float(
                    best.get("available_service_cash_headroom_p05", best.get("service_cash_headroom_p05"))
                ),
                "available_service_cash_headroom_p50": safe_float(
                    best.get("available_service_cash_headroom_p50", best.get("service_cash_headroom_p50"))
                ),
                "issuer_operating_risk_headroom_p50": safe_float(
                    best.get(
                        "issuer_operating_risk_headroom_p50",
                        best.get("available_service_cash_headroom_p50", best.get("service_cash_headroom_p50")),
                    )
                ),
                "issuer_operating_risk_headroom_ge_125": int(
                    safe_float(
                        best.get(
                            "issuer_operating_risk_headroom_p50",
                            best.get(
                                "available_service_cash_headroom_p50",
                                best.get("service_cash_headroom_p50"),
                            ),
                        )
                    )
                    >= 1.25
                ),
                "bond_service_lockbox_target_usd_p50": safe_float(
                    best.get("bond_service_lockbox_target_usd_p50")
                ),
                "bond_service_lockbox_coverage_ratio_p50": safe_float(
                    best.get("bond_service_lockbox_coverage_ratio_p50")
                ),
                "bond_service_lockbox_mode": best.get("bond_service_lockbox_mode", ""),
                "fee_service_reserved_usd_total_p50": safe_float(
                    best.get("fee_service_reserved_usd_total_p50")
                ),
                "fee_service_stable_reserved_usd_total_p50": safe_float(
                    best.get("fee_service_stable_reserved_usd_total_p50")
                ),
                "fee_service_converted_voucher_reserved_usd_total_p50": safe_float(
                    best.get("fee_service_converted_voucher_reserved_usd_total_p50")
                ),
                "issuer_reserve_balance_p05": safe_float(best.get("issuer_reserve_balance_p05")),
                "issuer_unpaid_scheduled_claim_p95": safe_float(best.get("issuer_unpaid_scheduled_claim_p95")),
                "route_success_p05": safe_float(best.get("route_success_p05")),
                "household_cash_stress_p95": safe_float(best.get("household_cash_stress_p95")),
                "household_cash_stress_delta_p95": safe_float(best.get("household_cash_stress_delta_p95")),
                "liquidity_leakage_p95": safe_float(best.get("liquidity_leakage_p95")),
                "liquidity_leakage_delta_p95": safe_float(best.get("liquidity_leakage_delta_p95")),
                "unpaid_claims_ratio_p95": safe_float(best.get("unpaid_claims_ratio_p95")),
                "realized_edge_concentration_p95": safe_float(best.get("realized_edge_concentration_p95")),
            }
        )
    headline_rows = []
    for scale in network_scales:
        scale_rows = [row for row in frontier_rows if row["network_scale"] == scale]
        best = max(scale_rows, key=lambda row: safe_float(row["safe_principal_usd"])) if scale_rows else {}
        headline_rows.append(
            {
                "network_scale": scale,
                "safe_principal_usd": safe_float(best.get("safe_principal_usd")),
                "safe_principal_ratio": safe_float(best.get("safe_principal_ratio")),
                "coupon_target_annual": safe_float(best.get("coupon_target_annual")),
                "bond_fee_service_share": safe_float(best.get("bond_fee_service_share")),
                "binding_constraint_at_frontier": best.get("binding_constraint_at_frontier", ""),
                "service_coverage_p05": safe_float(best.get("service_coverage_p05")),
                "service_coverage_p50": safe_float(best.get("service_coverage_p50")),
                "scheduled_payment_coverage_p05": safe_float(
                    best.get("scheduled_payment_coverage_p05")
                ),
                "scheduled_payment_coverage_p50": safe_float(
                    best.get("scheduled_payment_coverage_p50")
                ),
                "service_cash_headroom_p05": safe_float(best.get("service_cash_headroom_p05")),
                "service_cash_headroom_p50": safe_float(best.get("service_cash_headroom_p50")),
                "gross_historical_service_cash_headroom_p05": safe_float(
                    best.get("gross_historical_service_cash_headroom_p05", best.get("service_cash_headroom_p05"))
                ),
                "gross_historical_service_cash_headroom_p50": safe_float(
                    best.get("gross_historical_service_cash_headroom_p50", best.get("service_cash_headroom_p50"))
                ),
                "available_service_cash_headroom_p05": safe_float(
                    best.get("available_service_cash_headroom_p05", best.get("service_cash_headroom_p05"))
                ),
                "available_service_cash_headroom_p50": safe_float(
                    best.get("available_service_cash_headroom_p50", best.get("service_cash_headroom_p50"))
                ),
                "issuer_operating_risk_headroom_p50": safe_float(
                    best.get(
                        "issuer_operating_risk_headroom_p50",
                        best.get("available_service_cash_headroom_p50", best.get("service_cash_headroom_p50")),
                    )
                ),
                "issuer_operating_risk_headroom_ge_125": int(
                    safe_float(
                        best.get(
                            "issuer_operating_risk_headroom_p50",
                            best.get(
                                "available_service_cash_headroom_p50",
                                best.get("service_cash_headroom_p50"),
                            ),
                        )
                    )
                    >= 1.25
                ),
                "bond_service_lockbox_target_usd_p50": safe_float(
                    best.get("bond_service_lockbox_target_usd_p50")
                ),
                "bond_service_lockbox_coverage_ratio_p50": safe_float(
                    best.get("bond_service_lockbox_coverage_ratio_p50")
                ),
                "bond_service_lockbox_mode": best.get("bond_service_lockbox_mode", ""),
                "fee_service_reserved_usd_total_p50": safe_float(
                    best.get("fee_service_reserved_usd_total_p50")
                ),
                "fee_service_stable_reserved_usd_total_p50": safe_float(
                    best.get("fee_service_stable_reserved_usd_total_p50")
                ),
                "fee_service_converted_voucher_reserved_usd_total_p50": safe_float(
                    best.get("fee_service_converted_voucher_reserved_usd_total_p50")
                ),
                "issuer_reserve_balance_p05": safe_float(best.get("issuer_reserve_balance_p05")),
                "issuer_unpaid_scheduled_claim_p95": safe_float(best.get("issuer_unpaid_scheduled_claim_p95")),
                "route_success_p05": safe_float(best.get("route_success_p05")),
                "household_cash_stress_p95": safe_float(best.get("household_cash_stress_p95")),
                "household_cash_stress_delta_p95": safe_float(best.get("household_cash_stress_delta_p95")),
                "liquidity_leakage_p95": safe_float(best.get("liquidity_leakage_p95")),
                "liquidity_leakage_delta_p95": safe_float(best.get("liquidity_leakage_delta_p95")),
                "unpaid_claims_ratio_p95": safe_float(best.get("unpaid_claims_ratio_p95")),
                "realized_edge_concentration_p95": safe_float(best.get("realized_edge_concentration_p95")),
            }
        )
    return frontier_rows, headline_rows


def issuer_headroom_frontier_rows(safety_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    grouped: dict[tuple[str, float, float], list[dict[str, object]]] = defaultdict(list)
    for row in safety_rows:
        grouped[
            (
                str(row.get("network_scale", "")),
                safe_float(row.get("coupon_target_annual")),
                safe_float(row.get("bond_fee_service_share")),
            )
        ].append(row)
    for (scale, coupon, share), rows in sorted(grouped.items()):
        eligible = [
            row
            for row in rows
            if int(row.get("safe", 0)) == 1
            and safe_float(row.get("issuer_operating_risk_headroom_p50", row.get("available_service_cash_headroom_p50"))) >= 1.25
        ]
        if eligible:
            best = max(eligible, key=lambda row: safe_float(row.get("principal_usd_p50")))
            binding = ""
        else:
            safe_rows = [row for row in rows if int(row.get("safe", 0)) == 1]
            best = max(safe_rows, key=lambda row: safe_float(row.get("principal_usd_p50"))) if safe_rows else min(
                rows, key=lambda row: safe_float(row.get("principal_ratio"))
            )
            binding = "issuer_operating_risk_headroom_lt_1p25"
        rows_out.append(
            {
                "network_scale": scale,
                "coupon_target_annual": coupon,
                "bond_fee_service_share": share,
                "headroom_principal_ratio": safe_float(best.get("principal_ratio")) if eligible else 0.0,
                "headroom_principal_usd": safe_float(best.get("principal_usd_p50")) if eligible else 0.0,
                "issuer_operating_risk_headroom_p50": safe_float(
                    best.get("issuer_operating_risk_headroom_p50", best.get("available_service_cash_headroom_p50"))
                ),
                "scheduled_payment_coverage_p05": safe_float(best.get("scheduled_payment_coverage_p05")),
                "issuer_unpaid_scheduled_claim_p95": safe_float(best.get("issuer_unpaid_scheduled_claim_p95")),
                "binding_constraint_at_headroom_frontier": binding,
            }
        )
    return rows_out


def write_frontier_aggregate(
    args: argparse.Namespace,
    calibration: Calibration,
    output_dir: Path,
    *,
    network_scales: list[str],
    all_run_rows: list[dict[str, object]],
    safety_rows: list[dict[str, object]],
    partial: bool,
) -> None:
    all_run_rows = sorted_run_rows(all_run_rows)
    safety_rows = sorted_frontier_safety_rows(safety_rows)
    frontier_rows, headline_rows = frontier_summary_rows(network_scales, safety_rows)
    headroom_rows = issuer_headroom_frontier_rows(safety_rows)
    issuer_rows = issuer_cashflow_summary_rows(safety_rows)
    suffix = ".partial.csv" if partial else ".csv"
    write_csv(output_dir / f"bond_issuer_frontier_runs{suffix}", list(all_run_rows[0].keys()) if all_run_rows else [], all_run_rows)
    write_csv(output_dir / f"bond_issuer_frontier_safety{suffix}", list(safety_rows[0].keys()) if safety_rows else [], safety_rows)
    write_csv(output_dir / f"safe_injection_frontier{suffix}", list(frontier_rows[0].keys()) if frontier_rows else [], frontier_rows)
    write_csv(output_dir / f"issuer_operating_headroom_frontier{suffix}", list(headroom_rows[0].keys()) if headroom_rows else [], headroom_rows)
    write_csv(output_dir / f"network_scaling_summary{suffix}", list(headline_rows[0].keys()) if headline_rows else [], headline_rows)
    write_csv(output_dir / f"issuer_cashflow_summary{suffix}", list(issuer_rows[0].keys()) if issuer_rows else [], issuer_rows)
    audit_rows = calibration_activity_audit_rows(
        args,
        calibration,
        all_run_rows,
        scenario="bond_issuer_frontier",
        prefix="frontier",
    )
    write_csv(
        output_dir / f"calibration_activity_audit{suffix}",
        list(audit_rows[0].keys()) if audit_rows else [],
        audit_rows,
    )
    if not partial:
        write_frontier_tables(output_dir, safety_rows, frontier_rows)
        if not args.no_png:
            write_frontier_figures(output_dir, headline_rows, safety_rows)
        write_frontier_notes(output_dir, args, headline_rows)


def run_bond_issuer_frontier(args: argparse.Namespace, calibration: Calibration, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_status = frontier_validation_gate_status(output_dir)
    if validation_status == "fail":
        raise RuntimeError(
            "Refusing paper-facing bond_issuer_frontier because full engine validation status is fail."
        )
    if validation_status in {"missing", "review"}:
        print(
            f"Validation gate status for frontier is {validation_status}; outputs should be treated as non-final.",
            flush=True,
        )
    configure_sarafu_activity_controls(args, calibration, int(args.ticks), "frontier")
    network_scales = parse_str_list(args.network_scales)
    for scale in network_scales:
        if scale not in NETWORK_SCALE_FACTORS:
            raise ValueError(f"Unknown network scale {scale!r}")
    principal_ratios = sorted({max(0.0, value) for value in parse_float_list(args.principal_ratios)})
    coupons = parse_float_list(args.coupon_targets)
    service_shares = parse_float_list(args.bond_fee_service_shares)
    term_ticks = selected_terms(args)[0] if selected_terms(args) else int(args.ticks)
    scale_seed_offsets = {scale: idx * 10_000_000 for idx, scale in enumerate(network_scales)}

    def make_cell_job(phase: str, scale: str, ratio: float, coupon: float, service_share: float) -> dict[str, object]:
        capacity = certified_pool_capacity(calibration, scale, args.certification_policy)
        certified_capacity = max(0.0, capacity["certified_backing_capacity_usd"])
        principal_usd = certified_capacity * max(0.0, ratio)
        if phase == "baseline":
            principal_usd = 0.0
        return {
            "kind": "frontier_cell",
            "phase": phase,
            "job_id": frontier_job_id(phase, scale, ratio, coupon, service_share),
            "network_scale": scale,
            "principal_ratio": ratio,
            "principal_usd": principal_usd,
            "coupon": coupon,
            "service_share": service_share,
            "term_ticks": term_ticks,
            "seed_offset": scale_seed_offsets[scale],
        }

    baseline_jobs = [
        make_cell_job("baseline", scale, 0.0, 0.0, 0.0)
        for scale in network_scales
    ]
    grid_jobs = []
    for scale in network_scales:
        for ratio in principal_ratios:
            for coupon in coupons:
                for service_share in service_shares:
                    grid_jobs.append(make_cell_job("grid", scale, ratio, coupon, service_share))

    def safety_for_completed(
        baseline_results: list[dict[str, object]],
        cell_results: list[dict[str, object]],
    ) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
        baseline_by_scale = frontier_baselines_by_scale(baseline_results)
        if any(scale not in baseline_by_scale for scale in network_scales):
            return baseline_by_scale, []
        safety_rows = frontier_safety_from_results(
            cell_results,
            baseline_by_scale,
            safe_float(args.route_success_floor),
            str(getattr(args, "route_success_mode", "diagnostic")),
        )
        return baseline_by_scale, safety_rows

    def write_initial_frontier_partial(results: list[dict[str, object]]) -> None:
        baseline_results, cell_results = split_frontier_results(results)
        _, safety_rows = safety_for_completed(baseline_results, cell_results)
        write_frontier_aggregate(
            args,
            calibration,
            output_dir,
            network_scales=network_scales,
            all_run_rows=frontier_results_to_run_rows(results),
            safety_rows=safety_rows,
            partial=True,
        )

    initial_results, initial_failures = run_sharded_jobs(
        label="frontier baselines+grid",
        kind="frontier",
        jobs=baseline_jobs + grid_jobs,
        args=args,
        calibration=calibration,
        output_dir=output_dir,
        worker=run_frontier_cell_shard,
        load_completed=load_frontier_cell_shard,
        on_progress=write_initial_frontier_partial,
    )
    baseline_results, grid_results = split_frontier_results(initial_results)
    baseline_by_scale, grid_safety_rows = safety_for_completed(baseline_results, grid_results)
    if initial_failures:
        write_initial_frontier_partial(initial_results)
        print(f"Frontier baselines+grid had {len(initial_failures)} failed shard(s); rerun to resume.", flush=True)
        return 1
    missing_baselines = [scale for scale in network_scales if scale not in baseline_by_scale]
    if missing_baselines:
        write_initial_frontier_partial(initial_results)
        print(
            "Frontier baselines incomplete for scale(s): " + ", ".join(missing_baselines) + "; rerun to resume.",
            flush=True,
        )
        return 1

    all_cell_results = list(grid_results)
    safety_rows = grid_safety_rows
    safety_by_key = {
        frontier_cell_key(
            str(row["network_scale"]),
            safe_float(row["principal_ratio"]),
            safe_float(row["coupon_target_annual"]),
            safe_float(row["bond_fee_service_share"]),
        ): row
        for row in safety_rows
    }

    def write_frontier_partial(cell_results: list[dict[str, object]]) -> None:
        partial_safety_rows = frontier_safety_from_results(
            cell_results,
            baseline_by_scale,
            safe_float(args.route_success_floor),
            str(getattr(args, "route_success_mode", "diagnostic")),
        )
        write_frontier_aggregate(
            args,
            calibration,
            output_dir,
            network_scales=network_scales,
            all_run_rows=frontier_results_to_run_rows(baseline_results + cell_results),
            safety_rows=partial_safety_rows,
            partial=True,
        )

    if args.frontier_mode == "adaptive":
        for refinement_round in range(max(0, int(args.frontier_refinement_rounds))):
            new_cells: list[tuple[str, float, float, float]] = []
            for scale in network_scales:
                for coupon in coupons:
                    for service_share in service_shares:
                        rows = [
                            row
                            for row in safety_by_key.values()
                            if row["network_scale"] == scale
                            and abs(safe_float(row["coupon_target_annual"]) - coupon) <= 1e-9
                            and abs(safe_float(row["bond_fee_service_share"]) - service_share) <= 1e-9
                        ]
                        rows = sorted(rows, key=lambda row: safe_float(row["principal_ratio"]))
                        for low, high in zip(rows, rows[1:]):
                            low_safe = int(low.get("safe", 0)) == 1
                            high_safe = int(high.get("safe", 0)) == 1
                            if low_safe and not high_safe:
                                midpoint = (safe_float(low["principal_ratio"]) + safe_float(high["principal_ratio"])) / 2.0
                                key = frontier_cell_key(scale, midpoint, coupon, service_share)
                                if key not in safety_by_key:
                                    new_cells.append((scale, midpoint, coupon, service_share))
                                break
            if not new_cells:
                break
            print(
                f"Adaptive frontier refinement round {refinement_round + 1}: {len(new_cells)} midpoint cells",
                flush=True,
            )
            adaptive_jobs = [
                make_cell_job(f"adaptive{refinement_round + 1}", scale, ratio, coupon, service_share)
                for scale, ratio, coupon, service_share in new_cells
            ]
            adaptive_results, adaptive_failures = run_sharded_jobs(
                label=f"frontier adaptive round {refinement_round + 1}",
                kind="frontier",
                jobs=adaptive_jobs,
                args=args,
                calibration=calibration,
                output_dir=output_dir,
                worker=run_frontier_cell_shard,
                load_completed=load_frontier_cell_shard,
                on_progress=lambda results, prior=list(all_cell_results): write_frontier_partial(prior + results),
            )
            if adaptive_failures:
                write_frontier_partial(all_cell_results + adaptive_results)
                print(
                    f"Frontier adaptive round {refinement_round + 1} had {len(adaptive_failures)} failed shard(s); rerun to resume.",
                    flush=True,
                )
                return 1
            all_cell_results.extend(adaptive_results)
            safety_rows = frontier_safety_from_results(
                all_cell_results,
                baseline_by_scale,
                safe_float(args.route_success_floor),
                str(getattr(args, "route_success_mode", "diagnostic")),
            )
            safety_by_key = {
                frontier_cell_key(
                    str(row["network_scale"]),
                    safe_float(row["principal_ratio"]),
                    safe_float(row["coupon_target_annual"]),
                    safe_float(row["bond_fee_service_share"]),
                ): row
                for row in safety_rows
            }

    write_frontier_aggregate(
        args,
        calibration,
        output_dir,
        network_scales=network_scales,
        all_run_rows=frontier_results_to_run_rows(baseline_results + all_cell_results),
        safety_rows=safety_rows,
        partial=False,
    )
    print(f"Wrote bond issuer frontier artifacts to {output_dir}")
    return 0


def load_legacy_shard(job: dict[str, object], shard_root: Path, config_hash: str) -> dict[str, object] | None:
    return shard_result_from_files(
        kind="legacy",
        job=job,
        shard_root=shard_root,
        config_hash=config_hash,
        csv_names=("bond_rows", "network_rows", "failure_rows", "summary_rows"),
    )


def run_legacy_shard(
    job: dict[str, object],
    args_data: dict[str, object],
    calibration: Calibration,
    shard_root_text: str,
    resume: bool,
) -> dict[str, object]:
    shard_root = Path(shard_root_text)
    job_dir = shard_job_dir(shard_root, "legacy", str(job["job_id"]))
    config_hash = str(job["config_hash"])
    if resume:
        completed = load_legacy_shard(job, shard_root, config_hash)
        if completed is not None:
            return completed
    started_at = time.time()
    try:
        args = namespace_from_payload(args_data)
        apply_unit_normalization_context(args, calibration)
        scenario = str(job["scenario_name"])
        coupon = safe_float(job["coupon"])
        term_ticks = int(job["term_ticks"])
        run_idx = int(job["run_idx"])
        seed = int(job["seed"])
        job_position = int(job["job_position"])
        total_jobs = int(job["total_jobs"])
        print(
            f"[{job_position}/{total_jobs}] scenario={scenario} coupon={coupon} term={term_ticks} run={run_idx} seed={seed}",
            flush=True,
        )
        args._progress_run_position = job_position
        args._progress_total_runs = total_jobs
        bond_rows, network_rows, failure_rows, summary = run_one(
            scenario=scenario,
            coupon=coupon,
            term_ticks=term_ticks,
            run_index=run_idx,
            seed=seed,
            args=args,
            calibration=calibration,
        )
        summary_rows = [summary]
        write_rows(job_dir / "bond_rows.csv", bond_rows)
        write_rows(job_dir / "network_rows.csv", network_rows)
        write_rows(job_dir / "failure_rows.csv", failure_rows)
        write_rows(job_dir / "summary_rows.csv", summary_rows)
        write_shard_manifest(
            job_dir,
            job=job,
            config_hash=config_hash,
            status="completed",
            files={
                "bond_rows": len(bond_rows),
                "network_rows": len(network_rows),
                "failure_rows": len(failure_rows),
                "summary_rows": len(summary_rows),
            },
            started_at=started_at,
        )
        return {
            "status": "completed",
            "job": job,
            "shard_dir": str(job_dir),
            "bond_rows": bond_rows,
            "network_rows": network_rows,
            "failure_rows": failure_rows,
            "summary_rows": summary_rows,
        }
    except BaseException:
        error = traceback.format_exc()
        write_shard_manifest(
            job_dir,
            job=job,
            config_hash=config_hash,
            status="failed",
            error=error,
            started_at=started_at,
        )
        return failed_result(job, error)


def legacy_results_to_rows(
    results: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    return validation_results_to_rows(results)


def write_legacy_aggregate(
    args: argparse.Namespace,
    calibration: Calibration,
    output_dir: Path,
    *,
    bond_rows: list[dict[str, object]],
    network_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    partial: bool,
) -> None:
    bond_rows = sorted_run_rows(bond_rows)
    network_rows = sorted_run_rows(network_rows)
    failure_rows = sorted_run_rows(failure_rows)
    summary_rows = sorted_run_rows(summary_rows)
    quantile_source = []
    by_key_network = {
        (
            row["scenario"],
            row["run"],
            row["coupon_target_annual"],
            row["bond_term_ticks"],
            row["tick"],
        ): row
        for row in network_rows
    }
    for row in bond_rows:
        key = (
            row["scenario"],
            row["run"],
            row["coupon_target_annual"],
            row["bond_term_ticks"],
            row["tick"],
        )
        quantile_source.append({**row, **by_key_network.get(key, {})})
    quantiles = quantile_rows(quantile_source, CORE_QUANTILE_METRICS)
    failure_summary_rows = summarize_failures(failure_rows)
    suffix = ".partial.csv" if partial else ".csv"
    write_csv(output_dir / f"mc_bond_return_timeseries{suffix}", list(bond_rows[0].keys()) if bond_rows else [], bond_rows)
    write_csv(output_dir / f"mc_network_scaling_timeseries{suffix}", list(network_rows[0].keys()) if network_rows else [], network_rows)
    write_csv(output_dir / f"mc_run_summary{suffix}", list(summary_rows[0].keys()) if summary_rows else [], summary_rows)
    write_csv(output_dir / f"mc_failure_metrics{suffix}", list(failure_rows[0].keys()) if failure_rows else [], failure_rows)
    write_csv(
        output_dir / f"mc_failure_summary{suffix}",
        list(failure_summary_rows[0].keys()) if failure_summary_rows else [],
        failure_summary_rows,
    )
    write_csv(output_dir / f"mc_timeseries_quantiles{suffix}", list(quantiles[0].keys()) if quantiles else [], quantiles)
    if not partial:
        write_latex_tables(output_dir, calibration, summary_rows, failure_summary_rows)
        write_csv_tables(output_dir, calibration, summary_rows, failure_summary_rows)
        if not args.no_png:
            write_png_figures(output_dir, quantiles, failure_summary_rows, args)
        public_output_privacy_check(output_dir)


def main() -> int:
    args = parse_args()
    scenarios = list(LEGACY_SCENARIOS) if args.scenario == "all" else [args.scenario]
    coupons = parse_float_list(args.coupon_targets)
    terms = selected_terms(args)
    output_dir = Path(args.output).resolve()
    calibration = load_calibration(Path(args.calibration_dir).resolve())
    args.calibration_hash = calibration.calibration_hash
    apply_unit_normalization_context(args, calibration)

    if args.scenario == "sarafu_engine_validation":
        return run_sarafu_engine_validation(args, calibration, output_dir)
    if args.scenario == "bond_issuer_frontier":
        return run_bond_issuer_frontier(args, calibration, output_dir)

    total_jobs = len(scenarios) * len(coupons) * len(terms) * int(args.runs)
    job = 0
    jobs: list[dict[str, object]] = []

    for scenario_idx, scenario in enumerate(scenarios):
        for coupon_idx, coupon in enumerate(coupons):
            for term_idx, term_ticks in enumerate(terms):
                for run_idx in range(1, int(args.runs) + 1):
                    job += 1
                    seed = (
                        int(args.seed)
                        + scenario_idx * 1_000_000
                        + coupon_idx * 100_000
                        + term_idx * 10_000
                        + run_idx
                    )
                    jobs.append(
                        {
                            "kind": "legacy_run",
                            "job_id": (
                                f"legacy_{scenario}_c{frontier_token(coupon)}_"
                                f"t{term_ticks}_run_{run_idx:06d}"
                            ),
                            "scenario_name": scenario,
                            "coupon": coupon,
                            "term_ticks": term_ticks,
                            "run_idx": run_idx,
                            "seed": seed,
                            "job_position": job,
                            "total_jobs": total_jobs,
                        }
                    )

    def write_partial(results: list[dict[str, object]]) -> None:
        b, n, f, s = legacy_results_to_rows(results)
        write_legacy_aggregate(
            args,
            calibration,
            output_dir,
            bond_rows=b,
            network_rows=n,
            failure_rows=f,
            summary_rows=s,
            partial=True,
        )

    results, failed = run_sharded_jobs(
        label="legacy Monte Carlo",
        kind="legacy",
        jobs=jobs,
        args=args,
        calibration=calibration,
        output_dir=output_dir,
        worker=run_legacy_shard,
        load_completed=load_legacy_shard,
        on_progress=write_partial,
    )
    bond_rows, network_rows, failure_rows, summary_rows = legacy_results_to_rows(results)
    write_legacy_aggregate(
        args,
        calibration,
        output_dir,
        bond_rows=bond_rows,
        network_rows=network_rows,
        failure_rows=failure_rows,
        summary_rows=summary_rows,
        partial=False,
    )
    print(f"Wrote Monte Carlo artifacts to {output_dir}")
    return 1 if failed or len(summary_rows) != total_jobs else 0


if __name__ == "__main__":
    raise SystemExit(main())
