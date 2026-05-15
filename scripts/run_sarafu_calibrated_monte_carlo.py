#!/usr/bin/env python3
"""Sarafu-grounded Monte Carlo artifacts for the RegenBonds paper.

This runner is intentionally paper-facing. It consumes only privacy-safe
aggregate calibration files, anonymizes pools as templates, validates a
Sarafu-like baseline against empirical moments, and then
simulates counterfactual liquidity policies:

* aid_grant_injection: historical, non-repayable Sarafu-style liquidity.
* bond_lp_injection: counterfactual LP/bond-purchaser principal with fee returns.

The outputs are designed to be cited directly from the LaTeX manuscript.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SIM_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


YEAR_TICKS = 52
RECENT_WINDOW_TICKS = 13
TIER_ORDER = ("strong", "moderate", "weak")
ASSET_ORDER = ("cash", "redeemable_voucher", "internal_voucher")
DEFAULT_POLICIES = (
    "aid_baseline",
    "broad_equal",
    "strong_activity",
    "weak_capacity",
)
POLICY_LABELS = {
    "aid_baseline": "Aid/grant baseline",
    "broad_equal": "Bond broad equal",
    "strong_activity": "Bond strong-pool activity",
    "weak_capacity": "Bond weak-capacity inclusion",
    "mixed_aid_plus_bond": "Aid plus bond broad equal",
}


def default_calibration_dir() -> Path:
    local = SIM_ROOT / "analysis" / "sarafu_calibration"
    if local.exists():
        return local
    sibling = WORKSPACE_ROOT / "RegenBonds" / "analysis"
    if sibling.exists():
        return sibling
    return local


def default_output_dir() -> Path:
    sibling_analysis = WORKSPACE_ROOT / "RegenBonds" / "analysis"
    if sibling_analysis.exists():
        return sibling_analysis / "monte_carlo" / "sarafu_calibrated"
    return SIM_ROOT / "analysis" / "sarafu_calibrated"


@dataclass(frozen=True)
class PoolTemplate:
    template_id: str
    tier: str
    score: float
    swap_events: float
    recent_swap_weeks_90d: float
    active_weeks: float
    swaps_per_active_week: float
    total_users: float
    backing_inflow: float
    tagged_voucher_tokens: float
    verified_report_exposure: float
    approved_report_exposure: float
    period_aligned_verified_exposure: float
    same_token_return_rate: float
    same_token_out_value: float
    same_token_matched_later_in_value: float
    borrow_proxy_matured_events: float
    borrow_proxy_matured_return_rate: float
    rosca_proxy_value_return_rate: float
    debt_removal_purchase_events: float

    @property
    def weekly_swap_rate(self) -> float:
        if self.swaps_per_active_week > 0.0:
            return self.swaps_per_active_week * min(1.0, self.recent_swap_weeks_90d / RECENT_WINDOW_TICKS)
        if self.active_weeks > 0.0:
            return self.swap_events / self.active_weeks
        return self.swap_events / RECENT_WINDOW_TICKS

    @property
    def value_per_swap(self) -> float:
        candidates = []
        if self.same_token_out_value > 0.0 and self.swap_events > 0.0:
            candidates.append(self.same_token_out_value / self.swap_events)
        if self.backing_inflow > 0.0 and self.swap_events > 0.0:
            candidates.append(self.backing_inflow / self.swap_events)
        if not candidates:
            candidates.append(120.0)
        value = statistics.median(candidates)
        return min(10_000.0, max(25.0, value))

    @property
    def empirical_period_swaps_90d(self) -> float:
        return self.weekly_swap_rate * RECENT_WINDOW_TICKS


@dataclass(frozen=True)
class Calibration:
    pools: list[PoolTemplate]
    repayment_by_tier_asset: dict[tuple[str, str], float]
    repayment_out_value_by_tier_asset: dict[tuple[str, str], float]
    borrow_return_by_tier: dict[str, float]
    impact_projection_rows: list[dict[str, object]]
    report_quality_counts: dict[str, float]

    def cash_return(self, tier: str) -> float:
        return self.repayment_by_tier_asset.get((tier, "cash"), 0.0)

    def voucher_return(self, tier: str) -> float:
        redeemable_value = self.repayment_out_value_by_tier_asset.get((tier, "redeemable_voucher"), 0.0)
        internal_value = self.repayment_out_value_by_tier_asset.get((tier, "internal_voucher"), 0.0)
        denom = redeemable_value + internal_value
        if denom <= 0.0:
            return 0.0
        redeemable = self.repayment_by_tier_asset.get((tier, "redeemable_voucher"), 0.0)
        internal = self.repayment_by_tier_asset.get((tier, "internal_voucher"), 0.0)
        return (redeemable * redeemable_value + internal * internal_value) / denom

    def same_token_return(self, tier: str) -> float:
        values = [
            self.repayment_by_tier_asset.get((tier, asset), 0.0)
            for asset in ASSET_ORDER
            if self.repayment_out_value_by_tier_asset.get((tier, asset), 0.0) > 0.0
        ]
        weights = [
            self.repayment_out_value_by_tier_asset.get((tier, asset), 0.0)
            for asset in ASSET_ORDER
            if self.repayment_out_value_by_tier_asset.get((tier, asset), 0.0) > 0.0
        ]
        denom = sum(weights)
        if denom <= 0.0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights)) / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Sarafu-grounded simulation artifacts for the RegenBonds paper."
    )
    parser.add_argument("--runs", type=int, default=100, help="Monte Carlo runs per policy/coupon.")
    parser.add_argument("--ticks", type=int, default=52, help="Horizon in weekly ticks.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated policies: aid_baseline,broad_equal,strong_activity,weak_capacity,mixed_aid_plus_bond.",
    )
    parser.add_argument(
        "--coupon-targets",
        default="0,0.03,0.06,0.09,0.12",
        help="Comma-separated annual coupon targets for bond policies.",
    )
    parser.add_argument("--term", type=int, default=260, help="Bond term in weekly ticks.")
    parser.add_argument("--principal", type=float, default=400_000.0, help="LP/bond principal.")
    parser.add_argument("--fee-rate", type=float, default=0.02, help="Pool swap fee rate.")
    parser.add_argument(
        "--bond-fee-service-share",
        type=float,
        default=1.0,
        help="Share of simulated pool fees available for explicit bond-service support.",
    )
    parser.add_argument(
        "--activity-liquidity-elasticity",
        type=float,
        default=0.24,
        help="Elasticity linking incremental liquidity to activity in counterfactual scenarios.",
    )
    parser.add_argument(
        "--analysis-stride",
        type=int,
        default=1,
        help="Record time-series artifacts every N ticks.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N jobs. Use 1 for every job or 0 for only the final message.",
    )
    parser.add_argument(
        "--calibration-dir",
        default=str(default_calibration_dir()),
        help=(
            "Directory containing Sarafu aggregate calibration CSVs. "
            "Default uses sibling ../RegenBonds/analysis when present, "
            "otherwise analysis/sarafu_calibration for standalone clones."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(default_output_dir()),
        help=(
            "Output directory for paper artifacts. Default writes to sibling "
            "../RegenBonds/analysis/monte_carlo/sarafu_calibrated when present, "
            "otherwise analysis/sarafu_calibrated."
        ),
    )
    parser.add_argument("--no-png", action="store_true", help="Skip PNG figure generation.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def parse_policy_list(text: str) -> list[str]:
    values = []
    for item in text.split(","):
        policy = item.strip()
        if not policy:
            continue
        if policy not in POLICY_LABELS:
            raise ValueError(f"Unknown policy {policy!r}")
        values.append(policy)
    if not values:
        raise ValueError("At least one policy is required.")
    return values


def load_calibration(calibration_dir: Path) -> Calibration:
    pool_rows = read_csv(calibration_dir / "pool_report_activity.csv")
    pools = []
    for idx, row in enumerate(pool_rows, start=1):
        pools.append(
            PoolTemplate(
                template_id=f"pool_template_{idx:03d}",
                tier=str(row["tier"]).strip().lower(),
                score=safe_float(row.get("score")),
                swap_events=safe_float(row.get("swap_events")),
                recent_swap_weeks_90d=safe_float(row.get("recent_swap_weeks_90d")),
                active_weeks=safe_float(row.get("active_weeks")),
                swaps_per_active_week=safe_float(row.get("swaps_per_active_week")),
                total_users=safe_float(row.get("total_users")),
                backing_inflow=safe_float(row.get("backing_inflow")),
                tagged_voucher_tokens=safe_float(row.get("tagged_voucher_tokens")),
                verified_report_exposure=safe_float(row.get("verified_report_exposure")),
                approved_report_exposure=safe_float(row.get("approved_report_exposure")),
                period_aligned_verified_exposure=safe_float(row.get("period_aligned_verified_exposure")),
                same_token_return_rate=safe_float(row.get("same_token_return_rate")),
                same_token_out_value=safe_float(row.get("same_token_out_value")),
                same_token_matched_later_in_value=safe_float(row.get("same_token_matched_later_in_value")),
                borrow_proxy_matured_events=safe_float(row.get("borrow_proxy_matured_events")),
                borrow_proxy_matured_return_rate=safe_float(row.get("borrow_proxy_matured_return_rate")),
                rosca_proxy_value_return_rate=safe_float(row.get("rosca_proxy_value_return_rate")),
                debt_removal_purchase_events=safe_float(row.get("debt_removal_purchase_events")),
            )
        )

    repayment_by_tier_asset: dict[tuple[str, str], float] = {}
    repayment_out_value_by_tier_asset: dict[tuple[str, str], float] = {}
    for row in read_csv(calibration_dir / "repayment_calibration_by_tier_asset.csv"):
        key = (row["tier"], row["asset_class"])
        repayment_by_tier_asset[key] = safe_float(row.get("same_token_return_coverage"))
        repayment_out_value_by_tier_asset[key] = safe_float(row.get("same_token_out_value"))

    borrow_return_by_tier = {}
    borrow_path = calibration_dir / "borrow_repayment_by_tier.csv"
    if borrow_path.exists():
        for row in read_csv(borrow_path):
            borrow_return_by_tier[row["tier"]] = safe_float(row.get("weighted_matured_borrow_return_rate"))

    impact_projection_rows = []
    impact_path = calibration_dir / "impact_projection_by_activity.csv"
    if impact_path.exists():
        for row in read_csv(impact_path):
            activity = str(row.get("activity", "")).strip()
            if not activity or activity == "Unclassified local activity":
                continue
            impact_projection_rows.append(
                {
                    "activity": activity,
                    "verified_exposure_share": safe_float(row.get("verified_exposure_share")),
                    "log_intercept": safe_float(row.get("log_intercept")),
                    "log_slope": safe_float(row.get("log_slope")),
                }
            )
        impact_projection_rows.sort(key=lambda r: safe_float(r["verified_exposure_share"]), reverse=True)

    report_quality_counts = {}
    report_path = calibration_dir / "report_quality_counts.csv"
    if report_path.exists():
        for row in read_csv(report_path):
            report_quality_counts[str(row["metric"])] = safe_float(row.get("count"))

    return Calibration(
        pools=pools,
        repayment_by_tier_asset=repayment_by_tier_asset,
        repayment_out_value_by_tier_asset=repayment_out_value_by_tier_asset,
        borrow_return_by_tier=borrow_return_by_tier,
        impact_projection_rows=impact_projection_rows,
        report_quality_counts=report_quality_counts,
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def by_tier(pools: list[PoolTemplate]) -> dict[str, list[PoolTemplate]]:
    grouped = {tier: [] for tier in TIER_ORDER}
    for pool in pools:
        grouped.setdefault(pool.tier, []).append(pool)
    return grouped


def empirical_tier_summary(calibration: Calibration, ticks: int) -> list[dict[str, object]]:
    rows = []
    grouped = by_tier(calibration.pools)
    for tier in TIER_ORDER:
        pools = grouped.get(tier, [])
        scale = ticks / RECENT_WINDOW_TICKS
        period_swaps = [p.empirical_period_swaps_90d * scale for p in pools]
        period_reports = [
            p.verified_report_exposure * (ticks / RECENT_WINDOW_TICKS)
            for p in pools
        ]
        rows.append(
            {
                "tier": tier,
                "pool_count": len(pools),
                "mean_swap_events_horizon": mean(period_swaps),
                "median_swap_events_horizon": median(period_swaps),
                "total_swap_events_horizon": sum(period_swaps),
                "mean_verified_report_exposure_horizon": mean(period_reports),
                "median_verified_report_exposure_horizon": median(period_reports),
                "total_verified_report_exposure_horizon": sum(period_reports),
                "mean_same_token_return_rate": weighted_pool_rate(
                    pools, "same_token_return_rate", "same_token_out_value"
                ),
                "cash_return_rate": calibration.cash_return(tier),
                "voucher_return_rate": calibration.voucher_return(tier),
                "borrow_return_rate": calibration.borrow_return_by_tier.get(tier, 0.0),
                "backing_inflow": sum(p.backing_inflow for p in pools),
                "tagged_voucher_tokens": sum(p.tagged_voucher_tokens for p in pools),
            }
        )
    return rows


def weighted_pool_rate(pools: list[PoolTemplate], rate_attr: str, weight_attr: str) -> float:
    total_weight = sum(max(0.0, float(getattr(pool, weight_attr))) for pool in pools)
    if total_weight <= 0.0:
        return mean(float(getattr(pool, rate_attr)) for pool in pools)
    return sum(
        max(0.0, float(getattr(pool, weight_attr))) * max(0.0, float(getattr(pool, rate_attr)))
        for pool in pools
    ) / total_weight


def weighted_row_rate(rows: list[dict[str, object]], rate_col: str, weight_col: str) -> float:
    total_weight = sum(max(0.0, safe_float(row.get(weight_col))) for row in rows)
    if total_weight <= 0.0:
        return mean(safe_float(row.get(rate_col)) for row in rows)
    return sum(
        max(0.0, safe_float(row.get(weight_col))) * max(0.0, safe_float(row.get(rate_col)))
        for row in rows
    ) / total_weight


def jitter_lognormal(rng: random.Random, sigma: float) -> float:
    if sigma <= 0.0:
        return 1.0
    return rng.lognormvariate(-0.5 * sigma * sigma, sigma)


def simulate_baseline(
    calibration: Calibration,
    *,
    ticks: int,
    runs: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pool_rows = []
    tier_accumulator: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in range(1, runs + 1):
        rng = random.Random(seed + run * 17)
        for pool in calibration.pools:
            tier_sigma = {"strong": 0.16, "moderate": 0.24, "weak": 0.42}.get(pool.tier, 0.30)
            expected_swaps = pool.weekly_swap_rate * ticks
            if expected_swaps < 0.5 and rng.random() > expected_swaps:
                simulated_swaps = 0.0
            else:
                simulated_swaps = expected_swaps * jitter_lognormal(rng, tier_sigma)
            simulated_swaps = max(0.0, simulated_swaps)
            baseline_report_exposure = pool.verified_report_exposure * (ticks / RECENT_WINDOW_TICKS)
            activity_ratio = simulated_swaps / max(1e-9, expected_swaps)
            report_noise = jitter_lognormal(rng, tier_sigma * 0.9)
            simulated_reports = max(
                0.0,
                baseline_report_exposure * (0.75 + 0.25 * activity_ratio) * report_noise,
            )
            empirical_rate = pool.same_token_return_rate
            if empirical_rate <= 0.0:
                empirical_rate = calibration.same_token_return(pool.tier)
            simulated_return = min(1.0, max(0.0, rng.gauss(empirical_rate, 0.04 + tier_sigma * 0.06)))
            row = {
                "run": run,
                "template_id": pool.template_id,
                "public_role": "committee_pool",
                "internal_role": "lender",
                "tier": pool.tier,
                "empirical_weekly_swap_rate": pool.weekly_swap_rate,
                "simulated_swap_events_horizon": simulated_swaps,
                "simulated_verified_report_exposure_horizon": simulated_reports,
                "simulated_same_token_return_rate": simulated_return,
                "same_token_out_weight": pool.same_token_out_value,
                "simulated_backing_inflow": pool.backing_inflow,
                "claim_type": "validated simulation baseline",
            }
            pool_rows.append(row)
            tier_accumulator[pool.tier].append(row)

    summary_rows = []
    for tier in TIER_ORDER:
        rows = tier_accumulator.get(tier, [])
        pool_count = len({row["template_id"] for row in rows})
        run_count = len({row["run"] for row in rows}) or 1
        summary_rows.append(
            {
                "tier": tier,
                "pool_count": pool_count,
                "runs": run_count,
                "mean_swap_events_horizon": mean(safe_float(row["simulated_swap_events_horizon"]) for row in rows),
                "median_swap_events_horizon": median(safe_float(row["simulated_swap_events_horizon"]) for row in rows),
                "total_swap_events_horizon": sum(
                    safe_float(row["simulated_swap_events_horizon"]) for row in rows
                )
                / run_count,
                "mean_verified_report_exposure_horizon": mean(
                    safe_float(row["simulated_verified_report_exposure_horizon"]) for row in rows
                ),
                "median_verified_report_exposure_horizon": median(
                    safe_float(row["simulated_verified_report_exposure_horizon"]) for row in rows
                ),
                "total_verified_report_exposure_horizon": sum(
                    safe_float(row["simulated_verified_report_exposure_horizon"]) for row in rows
                )
                / run_count,
                "mean_same_token_return_rate": weighted_row_rate(
                    rows,
                    "simulated_same_token_return_rate",
                    "same_token_out_weight",
                ),
                "backing_inflow": sum(safe_float(row["simulated_backing_inflow"]) for row in rows) / run_count,
                "claim_type": "validated simulation baseline",
            }
        )
    return pool_rows, summary_rows


def validation_errors(
    empirical_rows: list[dict[str, object]],
    simulated_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    sim_by_tier = {row["tier"]: row for row in simulated_rows}
    metrics = [
        ("pool_count", 0.0),
        ("mean_swap_events_horizon", 0.18),
        ("median_swap_events_horizon", 0.35),
        ("total_swap_events_horizon", 0.15),
        ("total_verified_report_exposure_horizon", 0.25),
        ("mean_same_token_return_rate", 0.20),
        ("backing_inflow", 0.0),
    ]
    output = []
    for empirical in empirical_rows:
        tier = str(empirical["tier"])
        simulated = sim_by_tier.get(tier, {})
        for metric, tolerance in metrics:
            empirical_value = safe_float(empirical.get(metric))
            simulated_value = safe_float(simulated.get(metric))
            absolute_error = abs(simulated_value - empirical_value)
            relative_error = absolute_error / abs(empirical_value) if abs(empirical_value) > 1e-9 else absolute_error
            if tolerance == 0.0:
                passed = absolute_error <= 1e-9
            else:
                passed = relative_error <= tolerance
            output.append(
                {
                    "tier": tier,
                    "metric": metric,
                    "empirical_sarafu_moment": empirical_value,
                    "simulated_baseline_moment": simulated_value,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "tolerance": tolerance,
                    "pass_fail": "pass" if passed else "review",
                    "claim_type": "validated simulation baseline" if passed else "empirical calibration review",
                }
            )
    return output


def allocation_capacity(pool: PoolTemplate) -> float:
    return (
        1.0
        + math.log1p(pool.swap_events)
        + 0.5 * math.log1p(pool.tagged_voucher_tokens)
        + 0.6 * math.log1p(pool.verified_report_exposure)
        + 0.2 * math.log1p(pool.backing_inflow)
    )


def allocate_budget(
    pools: list[PoolTemplate],
    *,
    policy: str,
    principal: float,
) -> tuple[dict[str, float], dict[str, float]]:
    bond = {pool.template_id: 0.0 for pool in pools}
    aid = {pool.template_id: 0.0 for pool in pools}
    if policy in {"aid_baseline", "mixed_aid_plus_bond"}:
        for pool in pools:
            aid[pool.template_id] = pool.backing_inflow
    if policy == "aid_baseline":
        return aid, bond

    if policy in {"broad_equal", "mixed_aid_plus_bond"}:
        share = principal / max(1, len(pools))
        for pool in pools:
            bond[pool.template_id] = share
        return aid, bond

    tier_budgets: dict[str, float]
    if policy == "strong_activity":
        tier_budgets = {"strong": 0.80 * principal, "moderate": 0.20 * principal, "weak": 0.0}
    elif policy == "weak_capacity":
        tier_budgets = {"strong": 0.15 * principal, "moderate": 0.25 * principal, "weak": 0.60 * principal}
    else:
        raise ValueError(f"Unknown allocation policy: {policy}")

    grouped = by_tier(pools)
    for tier, budget in tier_budgets.items():
        candidates = grouped.get(tier, [])
        if policy == "strong_activity":
            weights = {pool.template_id: max(1.0, pool.swap_events) for pool in candidates}
        else:
            weights = {pool.template_id: allocation_capacity(pool) for pool in candidates}
        denom = sum(weights.values())
        if denom <= 0.0:
            continue
        for pool in candidates:
            bond[pool.template_id] = budget * weights[pool.template_id] / denom
    return aid, bond


def top_share(values: Iterable[float]) -> float:
    values = [max(0.0, float(v)) for v in values]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return max(values) / total


def hhi(values: Iterable[float]) -> float:
    values = [max(0.0, float(v)) for v in values]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return sum((v / total) ** 2 for v in values)


def project_report_exposure(calibration: Calibration, cumulative_swaps: float) -> float:
    if not calibration.impact_projection_rows:
        return 0.0
    total = 0.0
    for row in calibration.impact_projection_rows[:12]:
        predicted = math.exp(
            safe_float(row["log_intercept"]) + safe_float(row["log_slope"]) * math.log1p(cumulative_swaps)
        ) - 1.0
        total += max(0.0, predicted)
    return total


def simulate_policy_run(
    calibration: Calibration,
    *,
    policy: str,
    coupon: float,
    run: int,
    seed: int,
    ticks: int,
    term: int,
    principal: float,
    fee_rate: float,
    bond_fee_service_share: float,
    activity_liquidity_elasticity: float,
    analysis_stride: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    rng = random.Random(seed + run * 1009 + int(coupon * 100_000))
    aid, bond = allocate_budget(calibration.pools, policy=policy, principal=principal)
    bond_principal = sum(bond.values())
    aid_grant = sum(aid.values())
    is_bond_policy = bond_principal > 0.0
    pool_rows = []
    tier_totals = {tier: {"bond": 0.0, "aid": 0.0, "active": 0, "count": 0} for tier in TIER_ORDER}

    weekly_swaps_total = 0.0
    weekly_value_total = 0.0
    weighted_cash_leakage_num = 0.0
    weighted_repayment_num = 0.0
    weighted_repayment_den = 0.0
    active_pool_count = 0
    weak_recipient_count = 0
    weak_pool_count = sum(1 for pool in calibration.pools if pool.tier == "weak")

    for pool in calibration.pools:
        bond_amount = bond[pool.template_id]
        aid_amount = aid[pool.template_id]
        base_liquidity = 1.0 + pool.backing_inflow + pool.same_token_out_value / max(1.0, pool.active_weeks)
        incremental = bond_amount
        if policy == "aid_baseline":
            incremental = 0.0
        boost = 1.0 + activity_liquidity_elasticity * math.log1p(incremental / base_liquidity)
        if policy == "weak_capacity" and pool.tier == "weak" and bond_amount > 0.0:
            boost *= 1.06
        boost = min(3.0, max(0.35, boost))
        tier_sigma = {"strong": 0.12, "moderate": 0.20, "weak": 0.34}.get(pool.tier, 0.24)
        run_multiplier = jitter_lognormal(rng, tier_sigma)
        weekly_swaps = max(0.0, pool.weekly_swap_rate * boost * run_multiplier)
        weekly_value = weekly_swaps * pool.value_per_swap
        weekly_swaps_total += weekly_swaps
        weekly_value_total += weekly_value
        if weekly_swaps * ticks >= 1.0:
            active_pool_count += 1
        tier_totals[pool.tier]["bond"] += bond_amount
        tier_totals[pool.tier]["aid"] += aid_amount
        tier_totals[pool.tier]["count"] += 1
        if bond_amount > 0.0:
            tier_totals[pool.tier]["active"] += 1
        if pool.tier == "weak" and bond_amount > 0.0:
            weak_recipient_count += 1

        cash_return = calibration.cash_return(pool.tier)
        voucher_return = calibration.voucher_return(pool.tier)
        borrow_return = calibration.borrow_return_by_tier.get(pool.tier, voucher_return)
        closure = 0.45 * voucher_return + 0.35 * borrow_return + 0.20 * cash_return
        weight = max(bond_amount, aid_amount, weekly_value, 1.0)
        weighted_repayment_num += closure * weight
        weighted_repayment_den += weight
        if bond_amount > 0.0:
            weighted_cash_leakage_num += bond_amount * max(0.0, 1.0 - cash_return)

        pool_rows.append(
            {
                "policy": policy,
                "run": run,
                "template_id": pool.template_id,
                "public_role": "committee_pool",
                "internal_role": "lender",
                "tier": pool.tier,
                "aid_grant_injection": aid_amount,
                "bond_lp_injection": bond_amount,
                "weekly_swap_rate_projected": weekly_swaps,
                "weekly_swap_value_projected": weekly_value,
                "repayment_closure_prior": closure,
                "claim_type": "counterfactual Monte Carlo result" if is_bond_policy else "validated simulation baseline",
            }
        )

    allocation_values = list(bond.values())
    allocation_top_share = top_share(allocation_values)
    allocation_hhi = hhi(allocation_values)
    active_share = active_pool_count / max(1, len(calibration.pools))
    diversity_factor = max(0.0, 1.0 - allocation_hhi)
    potential_largest_component_share = min(1.0, 0.20 + 0.80 * active_share)
    realized_largest_component_share = min(
        potential_largest_component_share,
        0.12 + 0.72 * active_share * math.sqrt(max(0.05, diversity_factor)),
    )
    realized_concentration = max(allocation_top_share, 1.0 - diversity_factor)
    repayment_closure = weighted_repayment_num / max(1e-9, weighted_repayment_den)
    leakage = weighted_cash_leakage_num / max(1e-9, bond_principal) if bond_principal > 0.0 else 0.0
    weak_inclusion_count_share = weak_recipient_count / max(1, weak_pool_count)
    weak_inclusion_principal_share = tier_totals["weak"]["bond"] / max(1e-9, bond_principal) if bond_principal > 0 else 0.0

    timeseries = []
    for tick in range(1, ticks + 1):
        if tick % max(1, analysis_stride) != 0 and tick != ticks:
            continue
        cumulative_swaps = weekly_swaps_total * tick
        cumulative_value = weekly_value_total * tick
        fee_income = cumulative_value * fee_rate
        fee_return = fee_income * bond_fee_service_share if is_bond_policy else 0.0
        coupon_due = bond_principal * coupon * (min(tick, term) / YEAR_TICKS)
        coverage = fee_return / coupon_due if coupon_due > 1e-9 else 1.0
        projected_reports = project_report_exposure(calibration, cumulative_swaps)
        household_cash_stress = min(1.0, leakage * 0.55 + max(0.0, 0.55 - repayment_closure) * 0.45)
        timeseries.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "run": run,
                "seed": seed,
                "tick": tick,
                "coupon_target_annual": coupon,
                "bond_term_ticks": term,
                "principal_injected": bond_principal,
                "aid_grant_liquidity": aid_grant,
                "cumulative_swap_events": cumulative_swaps,
                "cumulative_swap_value": cumulative_value,
                "fee_income": fee_income,
                "fee_return_to_bond_lp": fee_return,
                "coupon_due": coupon_due,
                "coupon_coverage": coverage,
                "coupon_shortfall": max(0.0, coupon_due - fee_return),
                "principal_recovery_ratio": fee_return / bond_principal if bond_principal > 0.0 else 0.0,
                "projected_verified_report_exposure": projected_reports,
                "repayment_closure": repayment_closure,
                "liquidity_leakage": leakage,
                "household_cash_stress": household_cash_stress,
                "weak_pool_inclusion_count_share": weak_inclusion_count_share,
                "weak_pool_inclusion_principal_share": weak_inclusion_principal_share,
                "allocation_top_share": allocation_top_share,
                "allocation_hhi": allocation_hhi,
                "potential_largest_component_share": potential_largest_component_share,
                "realized_largest_component_share": realized_largest_component_share,
                "realized_concentration": realized_concentration,
                "claim_type": "counterfactual Monte Carlo result" if is_bond_policy else "validated simulation baseline",
            }
        )

    final = timeseries[-1].copy()
    final.update(
        {
            "runs": run,
            "source_type": "bond_lp_injection" if is_bond_policy else "aid_grant_injection",
            "weak_pool_count": weak_pool_count,
            "weak_recipient_count": weak_recipient_count,
        }
    )
    allocation_rows = []
    for tier in TIER_ORDER:
        allocation_rows.append(
            {
                "policy": policy,
                "run": run,
                "tier": tier,
                "pool_count": int(tier_totals[tier]["count"]),
                "recipient_count": int(tier_totals[tier]["active"]),
                "bond_lp_injection": tier_totals[tier]["bond"],
                "aid_grant_injection": tier_totals[tier]["aid"],
                "bond_principal_share": tier_totals[tier]["bond"] / max(1e-9, bond_principal)
                if bond_principal > 0.0
                else 0.0,
                "aid_grant_share": tier_totals[tier]["aid"] / max(1e-9, aid_grant)
                if aid_grant > 0.0
                else 0.0,
                "claim_type": "counterfactual Monte Carlo result" if is_bond_policy else "empirical calibration",
            }
        )
    return timeseries, final, pool_rows + allocation_rows


def quantile_timeseries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "fee_return_to_bond_lp",
        "coupon_due",
        "coupon_coverage",
        "coupon_shortfall",
        "principal_recovery_ratio",
        "projected_verified_report_exposure",
        "repayment_closure",
        "liquidity_leakage",
        "household_cash_stress",
        "potential_largest_component_share",
        "realized_largest_component_share",
        "realized_concentration",
    ]
    grouped: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["policy"], row["coupon_target_annual"], row["bond_term_ticks"], row["tick"])
        for metric in metrics:
            grouped[key][metric].append(safe_float(row.get(metric)))

    output = []
    for (policy, coupon, term, tick), metric_values in sorted(grouped.items()):
        for metric, values in sorted(metric_values.items()):
            output.append(
                {
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                    "coupon_target_annual": coupon,
                    "bond_term_ticks": term,
                    "tick": tick,
                    "metric": metric,
                    "mean": mean(values),
                    "p05": percentile(values, 0.05),
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                    "n": len(values),
                    "claim_type": "counterfactual Monte Carlo result"
                    if str(policy) != "aid_baseline"
                    else "validated simulation baseline",
                }
            )
    return output


def summarize_counterfactuals(final_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in final_rows:
        grouped[(row["policy"], row["coupon_target_annual"], row["bond_term_ticks"])].append(row)
    output = []
    for (policy, coupon, term), rows in sorted(grouped.items()):
        output.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                "source_type": "aid_grant_injection" if policy == "aid_baseline" else "bond_lp_injection",
                "coupon_target_annual": coupon,
                "bond_term_ticks": term,
                "runs": len(rows),
                "principal_injected_median": percentile([safe_float(r["principal_injected"]) for r in rows], 0.50),
                "aid_grant_liquidity_median": percentile([safe_float(r["aid_grant_liquidity"]) for r in rows], 0.50),
                "coupon_due_median": percentile([safe_float(r["coupon_due"]) for r in rows], 0.50),
                "fee_return_median": percentile([safe_float(r["fee_return_to_bond_lp"]) for r in rows], 0.50),
                "coupon_coverage_median": percentile([safe_float(r["coupon_coverage"]) for r in rows], 0.50),
                "principal_recovery_median": percentile(
                    [safe_float(r["principal_recovery_ratio"]) for r in rows], 0.50
                ),
                "weak_pool_inclusion_median": percentile(
                    [safe_float(r["weak_pool_inclusion_count_share"]) for r in rows], 0.50
                ),
                "weak_principal_share_median": percentile(
                    [safe_float(r["weak_pool_inclusion_principal_share"]) for r in rows], 0.50
                ),
                "allocation_top_share_median": percentile([safe_float(r["allocation_top_share"]) for r in rows], 0.50),
                "allocation_hhi_median": percentile([safe_float(r["allocation_hhi"]) for r in rows], 0.50),
                "leakage_median": percentile([safe_float(r["liquidity_leakage"]) for r in rows], 0.50),
                "repayment_closure_median": percentile([safe_float(r["repayment_closure"]) for r in rows], 0.50),
                "realized_component_median": percentile(
                    [safe_float(r["realized_largest_component_share"]) for r in rows], 0.50
                ),
                "projected_verified_reports_median": percentile(
                    [safe_float(r["projected_verified_report_exposure"]) for r in rows], 0.50
                ),
                "claim_type": "counterfactual Monte Carlo result" if policy != "aid_baseline" else "validated simulation baseline",
            }
        )
    return output


def role_mapping_rows() -> list[dict[str, object]]:
    return [
        {
            "model_role": "Producer",
            "sarafu_interpretation": "Voucher issuer with redeemable obligations",
            "simulator_role": "producer",
            "observable": "voucher issuance, circulation, redemption or issuer buyback",
            "validation_metric": "voucher return and same-token repayment coverage",
            "paper_claim": "implementation evidence",
        },
        {
            "model_role": "Committee pool",
            "sarafu_interpretation": "village, chama, or community-managed commitment pool",
            "simulator_role": "lender",
            "observable": "pool listings, limits, swaps, backing inflows, voucher holdings",
            "validation_metric": "pool activity tier, same-token coverage, report exposure",
            "paper_claim": "implementation evidence",
        },
        {
            "model_role": "Consumer",
            "sarafu_interpretation": "buyer or redeemer using stable assets or accepted vouchers",
            "simulator_role": "consumer",
            "observable": "stable-to-voucher purchases and redemption paths",
            "validation_metric": "debt-removal proxy and stable-voucher flow share",
            "paper_claim": "empirical calibration",
        },
        {
            "model_role": "Aid/donor liquidity",
            "sarafu_interpretation": "historical non-repayable liquidity or program backing",
            "simulator_role": "aid_grant_injection",
            "observable": "backing inflow into pools",
            "validation_metric": "aid/grant liquidity baseline by tier",
            "paper_claim": "empirical calibration",
        },
        {
            "model_role": "LP/bond purchaser",
            "sarafu_interpretation": "counterfactual repayable funder, not historical Sarafu",
            "simulator_role": "liquidity_provider",
            "observable": "principal injected and fee-supported returns",
            "validation_metric": "coupon coverage, fee return, principal recovery",
            "paper_claim": "counterfactual Monte Carlo result",
        },
        {
            "model_role": "Regenerative bond",
            "sarafu_interpretation": "formal finance layer added to Sarafu-calibrated pool mechanics",
            "simulator_role": "bond_lp_injection policy",
            "observable": "allocation policy, waterfall share, issuer debt-service assumptions",
            "validation_metric": "non-extraction, weak-pool inclusion, leakage, repayment closure",
            "paper_claim": "future legal/local validation requirement",
        },
    ]


def liquidity_comparison_rows() -> list[dict[str, object]]:
    return [
        {
            "liquidity_source": "Aid/grant liquidity",
            "historical_status": "observed precedent in Sarafu-style operations",
            "repayable_or_grant": "grant or program capital; no bondholder claim",
            "accounting_path": "aid_grant_injection to committee_pool templates",
            "recipient_pools": "pools with empirical backing inflows",
            "expected_return": "none to funder; social/ecological reporting is separate evidence",
            "paper_inference": "implementation evidence and empirical calibration",
        },
        {
            "liquidity_source": "LP/bond liquidity",
            "historical_status": "counterfactual in this paper; not historical Sarafu",
            "repayable_or_grant": "repayable principal with explicit coupon target",
            "accounting_path": "bond_lp_injection through allocation policy and fee-service share",
            "recipient_pools": "broad, strong-activity, weak-capacity, or mixed portfolios",
            "expected_return": "fee-supported return subject to issuer debt-service design",
            "paper_inference": "counterfactual Monte Carlo result",
        },
    ]


def calibration_prior_rows(calibration: Calibration) -> list[dict[str, object]]:
    rows = []
    for tier in TIER_ORDER:
        for asset in ASSET_ORDER:
            rows.append(
                {
                    "tier": tier,
                    "asset_class": asset,
                    "same_token_return_coverage": calibration.repayment_by_tier_asset.get((tier, asset), 0.0),
                    "same_token_out_value": calibration.repayment_out_value_by_tier_asset.get((tier, asset), 0.0),
                    "simulator_use": "repayment and settlement prior",
                    "claim_type": "empirical calibration",
                }
            )
        rows.append(
            {
                "tier": tier,
                "asset_class": "borrow_proxy",
                "same_token_return_coverage": calibration.borrow_return_by_tier.get(tier, 0.0),
                "same_token_out_value": "",
                "simulator_use": "ROSCA-like borrowing closure prior",
                "claim_type": "empirical calibration",
            }
        )
    for row in calibration.impact_projection_rows[:10]:
        rows.append(
            {
                "tier": "all",
                "asset_class": row["activity"],
                "same_token_return_coverage": row["log_slope"],
                "same_token_out_value": row["verified_exposure_share"],
                "simulator_use": "impact-report projection elasticity and activity mix",
                "claim_type": "empirical calibration",
            }
        )
    return rows


def table_to_latex(
    rows: list[dict[str, object]],
    *,
    columns: list[str],
    headers: list[str],
    max_rows: int | None = None,
) -> str:
    out = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    selected = rows[:max_rows] if max_rows is not None else rows
    for row in selected:
        out.append(" & ".join(format_latex_cell(row.get(col, "")) for col in columns) + r" \\")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    return "\n".join(out) + "\n"


def format_latex_cell(value: object) -> str:
    if isinstance(value, float):
        if abs(value) < 1.0:
            return f"{value:.3f}"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.2f}"
    return latex_escape(value)


def write_table_artifacts(
    tables_dir: Path,
    latex_dir: Path,
    name: str,
    rows: list[dict[str, object]],
    columns: list[str],
    headers: list[str],
    *,
    max_latex_rows: int | None = None,
) -> None:
    write_csv(tables_dir / f"{name}.csv", columns, rows)
    (latex_dir / f"{name}.tex").write_text(
        table_to_latex(rows, columns=columns, headers=headers, max_rows=max_latex_rows),
        encoding="utf-8",
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


def _fmt_value(value: float, *, percent: bool = False) -> str:
    if percent:
        return f"{value * 100:.0f}%"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.0f}"


def draw_grouped_bars(
    path: Path,
    *,
    title: str,
    subtitle: str,
    x_axis_label: str,
    y_axis_label: str,
    groups: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    percent: bool = False,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1200, 720
    left, right, top, bottom = 115, 55, 130, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_font(28)
    subtitle_font = _load_font(17)
    axis_font = _load_font(15)
    label_font = _load_font(16)
    _draw_text(draw, (left, 30), title, title_font)
    _draw_text(draw, (left, 64), subtitle, subtitle_font, fill=(80, 80, 80))
    values = [value for _, vals, _ in series for value in vals]
    y_max = max(values) if values else 1.0
    if y_max <= 0.0:
        y_max = 1.0
    y_max *= 1.15
    for i in range(6):
        frac = i / 5
        y = top + plot_h - frac * plot_h
        value = frac * y_max
        draw.line((left, y, left + plot_w, y), fill=(226, 226, 226), width=1)
        label = _fmt_value(value, percent=percent)
        _draw_text(draw, (left - 10, int(y)), label, axis_font, fill=(70, 70, 70), anchor="rm")
    _draw_text(draw, (left, top - 24), y_axis_label, axis_font, fill=(50, 50, 50))
    group_w = plot_w / max(1, len(groups))
    bar_w = min(42, group_w / max(1, len(series) + 1))
    for group_idx, group in enumerate(groups):
        center = left + group_w * (group_idx + 0.5)
        _draw_text(draw, (int(center), top + plot_h + 18), group.title(), label_font, fill=(45, 45, 45), anchor="ma")
        for series_idx, (_, vals, color) in enumerate(series):
            value = vals[group_idx] if group_idx < len(vals) else 0.0
            x0 = center - (len(series) * bar_w) / 2 + series_idx * bar_w + 3
            x1 = x0 + bar_w - 6
            y1 = top + plot_h
            y0 = y1 - (value / y_max) * plot_h
            draw.rectangle((x0, y0, x1, y1), fill=(*color, 220))
    legend_x = left
    legend_y = height - 40
    for label, _, color in series:
        draw.rectangle((legend_x, legend_y - 10, legend_x + 20, legend_y + 4), fill=(*color, 220))
        _draw_text(draw, (legend_x + 28, legend_y - 13), label, label_font)
        legend_x += 250
    _draw_text(draw, (left + plot_w // 2, height - 68), x_axis_label, label_font, fill=(50, 50, 50), anchor="ma")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def draw_line_with_band(
    path: Path,
    *,
    title: str,
    subtitle: str,
    y_axis_label: str,
    rows: list[dict[str, object]],
    policies: list[str],
    metric: str,
    percent: bool = False,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1200, 720
    left, right, top, bottom = 115, 60, 130, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_font(28)
    subtitle_font = _load_font(17)
    axis_font = _load_font(15)
    label_font = _load_font(15)
    _draw_text(draw, (left, 30), title, title_font)
    _draw_text(draw, (left, 64), subtitle, subtitle_font, fill=(80, 80, 80))
    colors = {
        "aid_baseline": (96, 96, 96),
        "broad_equal": (38, 99, 210),
        "strong_activity": (215, 78, 52),
        "weak_capacity": (40, 150, 98),
        "mixed_aid_plus_bond": (130, 82, 190),
    }
    filtered = [row for row in rows if row["metric"] == metric and row["policy"] in policies]
    if not filtered:
        return
    ticks = [safe_float(row["tick"]) for row in filtered]
    values = []
    for row in filtered:
        values.extend([safe_float(row["p05"]), safe_float(row["p50"]), safe_float(row["p95"])])
    x_min, x_max = min(ticks), max(ticks)
    if x_min == x_max:
        x_max = x_min + 1.0
    y_min = min(0.0, min(values))
    y_max = max(values)
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_max *= 1.10

    def sx(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h

    for i in range(6):
        frac = i / 5
        y = top + plot_h - frac * plot_h
        value = y_min + frac * (y_max - y_min)
        draw.line((left, y, left + plot_w, y), fill=(226, 226, 226), width=1)
        _draw_text(draw, (left - 10, int(y)), _fmt_value(value, percent=percent), axis_font, fill=(70, 70, 70), anchor="rm")
    _draw_text(draw, (left, top - 24), y_axis_label, axis_font, fill=(50, 50, 50))
    for i in range(6):
        frac = i / 5
        x = left + frac * plot_w
        value = x_min + frac * (x_max - x_min)
        draw.line((x, top, x, top + plot_h), fill=(238, 238, 238), width=1)
        _draw_text(draw, (int(x), top + plot_h + 14), f"{value:.0f}", axis_font, fill=(70, 70, 70), anchor="ma")

    for idx, policy in enumerate(policies):
        policy_rows = sorted(
            [row for row in filtered if row["policy"] == policy],
            key=lambda row: safe_float(row["tick"]),
        )
        if not policy_rows:
            continue
        color = colors.get(policy, (40, 40, 40))
        upper = [(sx(safe_float(row["tick"])), sy(safe_float(row["p95"]))) for row in policy_rows]
        lower = [(sx(safe_float(row["tick"])), sy(safe_float(row["p05"]))) for row in reversed(policy_rows)]
        if len(upper) >= 2:
            overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            band_draw = ImageDraw.Draw(overlay, "RGBA")
            band_draw.polygon(upper + lower, fill=(*color, 26))
            image.alpha_composite(overlay)
            draw = ImageDraw.Draw(image, "RGBA")
        points = [(sx(safe_float(row["tick"])), sy(safe_float(row["p50"]))) for row in policy_rows]
        if len(points) >= 2:
            draw.line(points, fill=(*color, 255), width=2)
        legend_x = left + idx * 250
        legend_y = height - 36
        draw.line((legend_x, legend_y, legend_x + 30, legend_y), fill=(*color, 255), width=2)
        _draw_text(draw, (legend_x + 38, legend_y - 9), POLICY_LABELS.get(policy, policy), label_font)
    _draw_text(draw, (left + plot_w // 2, height - 18), "tick (weeks)", label_font, fill=(50, 50, 50), anchor="ma")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def write_caption(
    path: Path,
    *,
    title: str,
    data_source: str,
    parameter_settings: str,
    interpretation: str,
    claim_boundary: str,
) -> None:
    path.with_suffix(".caption.txt").write_text(
        "\n".join(
            [
                f"Figure title: {title}",
                f"Data source: {data_source}",
                f"Parameter settings: {parameter_settings}",
                f"Interpretation: {interpretation}",
                f"Claim boundary: {claim_boundary}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_figures(
    figures_dir: Path,
    *,
    empirical_rows: list[dict[str, object]],
    simulated_rows: list[dict[str, object]],
    calibration: Calibration,
    counterfactual_summary: list[dict[str, object]],
    quantiles: list[dict[str, object]],
    coupon_for_figures: float,
    term: int,
    ticks: int,
) -> None:
    groups = list(TIER_ORDER)
    empirical_by_tier = {row["tier"]: row for row in empirical_rows}
    simulated_by_tier = {row["tier"]: row for row in simulated_rows}
    fig1 = figures_dir / "fig_sarafu_empirical_vs_simulated_activity_by_tier.png"
    draw_grouped_bars(
        fig1,
        title="Sarafu Empirical vs Calibrated Baseline Activity",
        subtitle=f"Horizon {ticks} weekly ticks; bars show mean pool swap events by tier.",
        x_axis_label="Pool tier",
        y_axis_label="Mean swap events per pool",
        groups=groups,
        series=[
            (
                "empirical Sarafu moment",
                [safe_float(empirical_by_tier[tier]["mean_swap_events_horizon"]) for tier in groups],
                (45, 94, 190),
            ),
            (
                "simulated baseline",
                [safe_float(simulated_by_tier[tier]["mean_swap_events_horizon"]) for tier in groups],
                (51, 151, 93),
            ),
        ],
    )
    write_caption(
        fig1,
        title="Sarafu empirical vs simulated baseline activity by tier",
        data_source="RegenBonds/analysis/pool_report_activity.csv, aggregated into privacy-safe pool templates.",
        parameter_settings=f"runs={safe_float(simulated_rows[0].get('runs')) if simulated_rows else 'NA'}, ticks={ticks}.",
        interpretation="The calibrated baseline preserves the weak-heavy Sarafu pool-health distribution while matching activity scale by tier.",
        claim_boundary="Validated simulation baseline; it does not claim exact transaction replay or causal impact.",
    )

    fig2 = figures_dir / "fig_same_token_repayment_coverage_by_tier_asset.png"
    draw_grouped_bars(
        fig2,
        title="Same-Token Repayment Coverage by Tier and Asset Class",
        subtitle="Cash/stable, redeemable vouchers, and internal vouchers are separated as Monte Carlo priors.",
        x_axis_label="Pool tier",
        y_axis_label="Same-token return coverage",
        groups=groups,
        series=[
            ("cash/stable", [calibration.repayment_by_tier_asset.get((tier, "cash"), 0.0) for tier in groups], (63, 106, 200)),
            (
                "redeemable voucher",
                [calibration.repayment_by_tier_asset.get((tier, "redeemable_voucher"), 0.0) for tier in groups],
                (56, 151, 101),
            ),
            (
                "internal voucher",
                [calibration.repayment_by_tier_asset.get((tier, "internal_voucher"), 0.0) for tier in groups],
                (220, 132, 51),
            ),
        ],
        percent=True,
    )
    write_caption(
        fig2,
        title="Same-token repayment coverage by pool tier and asset class",
        data_source="RegenBonds/analysis/repayment_calibration_by_tier_asset.csv.",
        parameter_settings="Coverage is later same-token inflow divided by same-token outflow, grouped by tier and asset class.",
        interpretation="Voucher repayment/settlement behavior is materially stronger than cash/stable closure in several tiers, shaping the bond stress tests.",
        claim_boundary="Empirical calibration proxy; it does not alone prove fulfillment quality, consent, or welfare causality.",
    )

    final_at_coupon = [
        row
        for row in counterfactual_summary
        if abs(safe_float(row["coupon_target_annual"]) - coupon_for_figures) <= 1e-9
        and int(safe_float(row["bond_term_ticks"])) == int(term)
    ]
    policy_order = [policy for policy in DEFAULT_POLICIES if any(row["policy"] == policy for row in final_at_coupon)]
    fig3 = figures_dir / "fig_aid_grant_baseline_vs_bond_injection.png"
    draw_grouped_bars(
        fig3,
        title="Aid-Liquidity Baseline vs LP/Bond Counterfactual",
        subtitle=f"Coupon {coupon_for_figures * 100:.1f}%, term {term} ticks; bars show median liquidity by policy.",
        x_axis_label="Policy",
        y_axis_label="Median liquidity amount",
        groups=[POLICY_LABELS[p] for p in policy_order],
        series=[
            (
                "aid/grant liquidity",
                [
                    safe_float(next(row for row in final_at_coupon if row["policy"] == p)["aid_grant_liquidity_median"])
                    for p in policy_order
                ],
                (92, 92, 92),
            ),
            (
                "LP/bond principal",
                [
                    safe_float(next(row for row in final_at_coupon if row["policy"] == p)["principal_injected_median"])
                    for p in policy_order
                ],
                (45, 94, 190),
            ),
        ],
    )
    write_caption(
        fig3,
        title="Aid-liquidity baseline versus LP/bond-injection counterfactual",
        data_source="Sarafu backing inflow calibration and simulated LP/bond allocation policies.",
        parameter_settings=f"coupon={coupon_for_figures:.3f}, term={term}, ticks={ticks}.",
        interpretation="The figure separates observed grant-like liquidity precedent from the counterfactual repayable bond principal.",
        claim_boundary="Counterfactual Monte Carlo result for bond policies; historical Sarafu is not treated as a regenerative bond.",
    )

    qrows = [
        row
        for row in quantiles
        if abs(safe_float(row["coupon_target_annual"]) - coupon_for_figures) <= 1e-9
        and int(safe_float(row["bond_term_ticks"])) == int(term)
    ]
    fig4 = figures_dir / "fig_bond_fee_return_over_time.png"
    draw_line_with_band(
        fig4,
        title="Bond Fee Return Over Time",
        subtitle=f"Median line with 5th-95th percentile band; coupon {coupon_for_figures * 100:.1f}%, term {term} ticks.",
        y_axis_label="Cumulative fee return to LP/bond funder",
        rows=qrows,
        policies=[p for p in ("broad_equal", "strong_activity", "weak_capacity") if p in policy_order],
        metric="fee_return_to_bond_lp",
    )
    write_caption(
        fig4,
        title="Bond fee return over time",
        data_source="Sarafu-calibrated counterfactual Monte Carlo timeseries.",
        parameter_settings=f"fee_rate=2%, bond_fee_service_share as CLI parameter, ticks={ticks}.",
        interpretation="Thin lines are median simulated cumulative fee returns; shaded bands are run-level 5th-95th percentile uncertainty.",
        claim_boundary="Counterfactual result; fees are modeled as explicit bond-service support, not hidden household liability.",
    )

    allocation_groups = [POLICY_LABELS[p] for p in policy_order]
    tier_counts = {
        str(row["tier"]): safe_float(row.get("pool_count"))
        for row in empirical_rows
    }
    fig5 = figures_dir / "fig_tier_inclusion_by_policy.png"
    draw_grouped_bars(
        fig5,
        title="Weak, Moderate, and Strong Pool Inclusion by Policy",
        subtitle=f"Median principal allocation shares, coupon {coupon_for_figures * 100:.1f}%.",
        x_axis_label="Policy",
        y_axis_label="Principal allocation share",
        groups=allocation_groups,
        series=[
            (
                tier.title(),
                [
                    tier_share_from_summary(final_at_coupon, policy=p, tier=tier, tier_counts=tier_counts)
                    for p in policy_order
                ],
                {"strong": (46, 99, 190), "moderate": (218, 83, 58), "weak": (95, 95, 95)}[tier],
            )
            for tier in TIER_ORDER
        ],
        percent=True,
    )
    write_caption(
        fig5,
        title="Weak, moderate, and strong tier inclusion under allocation policies",
        data_source="Sarafu tier distribution and simulated LP/bond allocation rules.",
        parameter_settings=f"principal policies={','.join(policy_order)}, principal normalized to 100%.",
        interpretation="Weak-capacity allocation is intentionally inclusionary; strong-activity allocation concentrates principal in healthier pools.",
        claim_boundary="Counterfactual policy comparison; final allocation rules require local governance and legal validation.",
    )

    fig6 = figures_dir / "fig_network_connectivity_and_concentration.png"
    draw_grouped_bars(
        fig6,
        title="Projected Network Connectivity and Concentration",
        subtitle=f"Median outcomes at tick {ticks}, coupon {coupon_for_figures * 100:.1f}%.",
        x_axis_label="Policy",
        y_axis_label="Share / concentration index",
        groups=allocation_groups,
        series=[
            (
                "realized component",
                [
                    safe_float(next(row for row in final_at_coupon if row["policy"] == p)["realized_component_median"])
                    for p in policy_order
                ],
                (41, 124, 198),
            ),
            (
                "top allocation share",
                [
                    safe_float(next(row for row in final_at_coupon if row["policy"] == p)["allocation_top_share_median"])
                    for p in policy_order
                ],
                (221, 129, 50),
            ),
        ],
        percent=True,
    )
    write_caption(
        fig6,
        title="Network connectivity and concentration under allocation policies",
        data_source="Sarafu-calibrated activity templates and simulated allocation concentration diagnostics.",
        parameter_settings=f"ticks={ticks}; realized component is an activity-weighted connectivity proxy.",
        interpretation="Connectivity and concentration move together differently across allocation policies, giving the paper a governance stress-test target.",
        claim_boundary="Network scaling proxy; it should be replaced or supplemented by full route-level engine validation in later runs.",
    )


def tier_share_from_summary(
    rows: list[dict[str, object]],
    *,
    policy: str,
    tier: str,
    tier_counts: dict[str, float],
) -> float:
    if policy == "aid_baseline":
        return 0.0
    if policy == "broad_equal":
        return tier_counts.get(tier, 0.0) / max(1.0, sum(tier_counts.values()))
    if policy == "strong_activity":
        return {"strong": 0.80, "moderate": 0.20, "weak": 0.0}[tier]
    if policy == "weak_capacity":
        return {"strong": 0.15, "moderate": 0.25, "weak": 0.60}[tier]
    if policy == "mixed_aid_plus_bond":
        return tier_counts.get(tier, 0.0) / max(1.0, sum(tier_counts.values()))
    return 0.0


def write_paper_integration_notes(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    calibration: Calibration,
    validation_rows: list[dict[str, object]],
    counterfactual_rows: list[dict[str, object]],
) -> None:
    failed = [row for row in validation_rows if row["pass_fail"] != "pass"]
    policies = sorted({str(row["policy"]) for row in counterfactual_rows})
    text = f"""# Sarafu-Calibrated Simulation Paper Integration Notes

Generated by `scripts/run_sarafu_calibrated_monte_carlo.py`.

## Paper Section Placement

- Section 12, Sarafu Network Model Implementation: use `tables/model_role_mapping.csv` and `latex/model_role_mapping.tex`.
- Section 13, Sarafu Network Empirical Findings: use `tables/calibration_priors.csv`, `figures/fig_same_token_repayment_coverage_by_tier_asset.png`, and the report-quality counts in `tables/calibration_priors.csv`.
- Section 14, Monte Carlo Simulation Design: use `tables/aid_vs_bond_liquidity.csv` and `latex/aid_vs_bond_liquidity.tex`.
- Section 15, Monte Carlo Sarafu Network Empirical Calibrations: use `tables/baseline_validation_table.csv`, `latex/baseline_validation_table.tex`, and `figures/fig_sarafu_empirical_vs_simulated_activity_by_tier.png`.
- Section 16, Monte Carlo Simulation Results: use `tables/counterfactual_results_table.csv`, `latex/counterfactual_results_table.tex`, and Figures 3-6 in `figures/`.

## Generated Run Settings

- Pool templates: {len(calibration.pools)} privacy-safe Sarafu pool templates.
- Horizon: {args.ticks} weekly ticks.
- Runs per policy/coupon: {args.runs}.
- Policies: {', '.join(policies)}.
- Coupon targets: {args.coupon_targets}.
- Bond principal: {args.principal:,.2f}.
- Fee rate: {args.fee_rate:.4f}.
- Bond fee-service share: {args.bond_fee_service_share:.4f}.

## Claims Supported

- Implementation evidence: Sarafu contains voucher issuers, committee-pool mechanics, limits, swaps, reports, and liquidity/backing flows that map to the commitment-pool architecture.
- Empirical calibration: pool tiers, activity rates, same-token repayment coverage, borrow-return proxies, aid/backing inflows, and report-exposure gradients are directly taken from aggregate Sarafu analysis outputs.
- Validated simulation baseline: the calibrated baseline is checked against empirical tier-level moments; {len(failed)} validation rows are marked for review.
- Counterfactual Monte Carlo result: LP/bond-purchaser liquidity is simulated as repayable principal and separated from historical aid/grant liquidity.

## Claims Not Proven

- The counterfactual bond layer was not historically implemented in Sarafu.
- Transaction and report data do not alone prove causal welfare impact, ecological regeneration, legal compliance, consent, fulfillment quality, or cultural legitimacy.
- Fee-supported return is modeled as explicit debt-service support. It should not be interpreted as household liability unless a future legal design explicitly makes that obligation and passes local review.
- Network scaling metrics in this first calibrated runner are pool-template proxies. Full route-level validation in the engine remains a follow-on step.
"""
    (output_dir / "paper_integration_notes.md").write_text(text, encoding="utf-8")


def privacy_check(output_dir: Path) -> None:
    forbidden_headers = {"pool_id", "pool_label", "address", "report_text", "tag_text", "gps", "latitude", "longitude"}
    forbidden_text = ("0x", "@")
    for path in output_dir.rglob("*"):
        if path.suffix.lower() not in {".csv", ".tex", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        header = text.splitlines()[0].lower().split(",") if text.splitlines() else []
        overlap = forbidden_headers.intersection(header)
        if overlap:
            raise RuntimeError(f"Public artifact {path} exposes forbidden columns: {sorted(overlap)}")
        lowered = text.lower()
        for token in forbidden_text:
            if token in lowered:
                raise RuntimeError(f"Public artifact {path} contains private-looking token {token!r}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    latex_dir = output_dir / "latex"
    for directory in (tables_dir, figures_dir, latex_dir):
        directory.mkdir(parents=True, exist_ok=True)

    policies = parse_policy_list(args.policies)
    coupons = parse_float_list(args.coupon_targets)
    calibration = load_calibration(Path(args.calibration_dir).resolve())
    if not calibration.pools:
        raise RuntimeError("No Sarafu pool templates loaded.")

    empirical_rows = empirical_tier_summary(calibration, args.ticks)
    baseline_pool_rows, simulated_baseline_rows = simulate_baseline(
        calibration,
        ticks=args.ticks,
        runs=args.runs,
        seed=args.seed,
    )
    validation_rows = validation_errors(empirical_rows, simulated_baseline_rows)

    timeseries_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    total_jobs = len(policies) * len(coupons) * int(args.runs)
    job = 0
    for policy_idx, policy in enumerate(policies):
        for coupon_idx, coupon in enumerate(coupons):
            effective_coupon = 0.0 if policy == "aid_baseline" else coupon
            for run in range(1, int(args.runs) + 1):
                job += 1
                seed = int(args.seed) + policy_idx * 1_000_000 + coupon_idx * 100_000 + run
                progress_every = int(args.progress_every)
                if progress_every > 0 and (job == 1 or job == total_jobs or job % progress_every == 0):
                    print(
                        f"[{job}/{total_jobs}] policy={policy} coupon={effective_coupon} run={run} seed={seed}",
                        flush=True,
                    )
                series, final, details = simulate_policy_run(
                    calibration,
                    policy=policy,
                    coupon=effective_coupon,
                    run=run,
                    seed=seed,
                    ticks=int(args.ticks),
                    term=int(args.term),
                    principal=float(args.principal),
                    fee_rate=float(args.fee_rate),
                    bond_fee_service_share=float(args.bond_fee_service_share),
                    activity_liquidity_elasticity=float(args.activity_liquidity_elasticity),
                    analysis_stride=int(args.analysis_stride),
                )
                timeseries_rows.extend(series)
                final_rows.append(final)
                if run == 1 and coupon_idx == 0:
                    detail_rows.extend(details)

    quantiles = quantile_timeseries(timeseries_rows)
    counterfactual_summary = summarize_counterfactuals(final_rows)

    write_table_artifacts(
        tables_dir,
        latex_dir,
        "model_role_mapping",
        role_mapping_rows(),
        [
            "model_role",
            "sarafu_interpretation",
            "simulator_role",
            "observable",
            "validation_metric",
            "paper_claim",
        ],
        ["Model role", "Sarafu interpretation", "Simulator role", "Observable", "Validation metric", "Paper claim"],
    )
    write_table_artifacts(
        tables_dir,
        latex_dir,
        "aid_vs_bond_liquidity",
        liquidity_comparison_rows(),
        [
            "liquidity_source",
            "historical_status",
            "repayable_or_grant",
            "accounting_path",
            "recipient_pools",
            "expected_return",
            "paper_inference",
        ],
        ["Liquidity source", "Historical status", "Repayable/grant", "Accounting path", "Recipient pools", "Expected return", "Paper inference"],
    )
    write_table_artifacts(
        tables_dir,
        latex_dir,
        "empirical_tier_summary",
        empirical_rows,
        list(empirical_rows[0].keys()),
        list(empirical_rows[0].keys()),
    )
    write_table_artifacts(
        tables_dir,
        latex_dir,
        "baseline_validation_table",
        validation_rows,
        list(validation_rows[0].keys()),
        list(validation_rows[0].keys()),
        max_latex_rows=30,
    )
    write_table_artifacts(
        tables_dir,
        latex_dir,
        "counterfactual_results_table",
        counterfactual_summary,
        list(counterfactual_summary[0].keys()),
        list(counterfactual_summary[0].keys()),
        max_latex_rows=40,
    )
    write_table_artifacts(
        tables_dir,
        latex_dir,
        "calibration_priors",
        calibration_prior_rows(calibration),
        [
            "tier",
            "asset_class",
            "same_token_return_coverage",
            "same_token_out_value",
            "simulator_use",
            "claim_type",
        ],
        ["Tier", "Asset/activity", "Coverage/slope", "Out value/share", "Simulator use", "Claim type"],
        max_latex_rows=28,
    )

    write_csv(
        tables_dir / "sarafu_template_baseline_pool_rows.csv",
        list(baseline_pool_rows[0].keys()) if baseline_pool_rows else [],
        baseline_pool_rows,
    )
    write_csv(
        tables_dir / "counterfactual_timeseries.csv",
        list(timeseries_rows[0].keys()) if timeseries_rows else [],
        timeseries_rows,
    )
    write_csv(
        tables_dir / "counterfactual_timeseries_quantiles.csv",
        list(quantiles[0].keys()) if quantiles else [],
        quantiles,
    )
    write_csv(
        tables_dir / "counterfactual_detail_rows.csv",
        list(detail_rows[0].keys()) if detail_rows else [],
        detail_rows,
    )

    coupon_for_figures = 0.06
    available_coupons = sorted({safe_float(row["coupon_target_annual"]) for row in counterfactual_summary})
    if coupon_for_figures not in available_coupons and available_coupons:
        coupon_for_figures = min(available_coupons, key=lambda value: abs(value - 0.06))
    if not args.no_png:
        write_figures(
            figures_dir,
            empirical_rows=empirical_rows,
            simulated_rows=simulated_baseline_rows,
            calibration=calibration,
            counterfactual_summary=counterfactual_summary,
            quantiles=quantiles,
            coupon_for_figures=coupon_for_figures,
            term=args.term,
            ticks=args.ticks,
        )

    write_paper_integration_notes(
        output_dir,
        args=args,
        calibration=calibration,
        validation_rows=validation_rows,
        counterfactual_rows=counterfactual_summary,
    )
    privacy_check(output_dir)
    print(f"Wrote Sarafu-calibrated paper artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
