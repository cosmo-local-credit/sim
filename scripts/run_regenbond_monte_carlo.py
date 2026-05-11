#!/usr/bin/env python3
"""Run Sarafu-calibrated Monte Carlo scenarios for the RegenBonds paper.

The runner is paper-first: it writes deterministic CSV and LaTeX artifacts that
can be included by the manuscript. It does not read private Sarafu raw data; it
only consumes aggregate calibration files from RegenBonds/analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SIM_ROOT.parent
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from sim.config import ScenarioConfig
from sim.engine import SimulationEngine


MONTH_TICKS = 4
YEAR_TICKS = 52
DEFAULT_COUPONS = (0.0, 0.03, 0.06, 0.09, 0.12)
DEFAULT_TERMS = (52, 156, 260)
SCENARIOS = (
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
CORE_QUANTILE_METRICS = (
    "bond_annualized_fee_yield",
    "bond_coupon_coverage_ratio",
    "bond_coupon_shortfall_usd",
    "bond_cumulative_fee_return_usd",
    "fee_pool_cumulative_usd",
    "swap_volume_usd_tick",
    "transactions_per_tick",
    "route_success_rate_tick",
    "repayment_volume_usd",
    "loan_issuance_volume_usd",
    "debt_outstanding_usd",
    "household_cash_stress_ratio",
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


@dataclass
class Calibration:
    params: dict[str, float]
    repayment_by_tier_asset: dict[tuple[str, str], float]
    tier_probs: dict[str, float]
    voucher_coverage_by_tier: dict[str, float]
    impact_rows: list[ImpactProjection]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regenerative-bond Monte Carlo scenarios and write paper artifacts."
    )
    parser.add_argument(
        "--scenario",
        default="regenbond_lp_injection",
        choices=(*SCENARIOS, "all"),
        help="Scenario preset to run. Use 'all' for all paper scenarios.",
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
        "--calibration-dir",
        default=str(WORKSPACE_ROOT / "RegenBonds" / "analysis"),
        help="Directory containing Sarafu-derived aggregate calibration CSVs.",
    )
    parser.add_argument(
        "--output",
        default=str(WORKSPACE_ROOT / "RegenBonds" / "analysis" / "monte_carlo"),
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def load_calibration(calibration_dir: Path) -> Calibration:
    params = {}
    for row in read_csv(calibration_dir / "monte_carlo_calibration_parameters.csv"):
        params[row["parameter"]] = safe_float(row["value"])

    repayment_by_tier_asset: dict[tuple[str, str], float] = {}
    tier_asset_out_values: dict[tuple[str, str], float] = {}
    for row in read_csv(calibration_dir / "repayment_calibration_by_tier_asset.csv"):
        key = (row["tier"], row["asset_class"])
        repayment_by_tier_asset[key] = safe_float(row["same_token_return_coverage"])
        tier_asset_out_values[key] = safe_float(row["same_token_out_value"])

    tier_counts = Counter(row["tier"] for row in read_csv(calibration_dir / "pool_report_activity.csv"))
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

    return Calibration(
        params=params,
        repayment_by_tier_asset=repayment_by_tier_asset,
        tier_probs=tier_probs,
        voucher_coverage_by_tier=voucher_coverage_by_tier,
        impact_rows=impact_rows,
    )


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
    elif scenario == "regenbond_lp_injection":
        cfg.initial_liquidity_providers = 1
        cfg.lp_initial_stable_mean = 400_000.0
        cfg.calibration_profile = "sarafu_empirical"
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
) -> None:
    for pid in active_pool_ids(engine):
        tiers.setdefault(pid, choose_tier(rng, calibration.tier_probs))


def tier_mix(engine: SimulationEngine, tiers: dict[str, str]) -> dict[str, float]:
    counts = Counter(tiers.get(pid, "moderate") for pid in active_pool_ids(engine))
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
    by_activity = {}
    counts = list(pool_swap_counts.values())
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


def bond_metrics(latest: dict[str, object], cfg: ScenarioConfig, tick: int) -> dict[str, float]:
    principal = safe_float(latest.get("lp_injected_usd_total"))
    raw_returned = safe_float(latest.get("lp_returned_usd_total"))
    service_share = max(0.0, min(1.0, float(cfg.bond_fee_service_share or 0.0)))
    returned = raw_returned * service_share
    elapsed_years = max(tick / YEAR_TICKS, 1.0 / YEAR_TICKS)
    coupon = max(0.0, float(cfg.bond_coupon_target_annual or 0.0))
    term_elapsed_ticks = min(max(0, tick), max(0, int(cfg.bond_term_ticks or 0)))
    coupon_due = principal * coupon * (term_elapsed_ticks / YEAR_TICKS)
    coupon_coverage = returned / coupon_due if coupon_due > 1e-9 else 1.0
    simple_yield = returned / principal if principal > 1e-9 else 0.0
    annualized_yield = simple_yield / elapsed_years if principal > 1e-9 else 0.0
    cagr = ((1.0 + simple_yield) ** (1.0 / elapsed_years) - 1.0) if principal > 1e-9 else 0.0
    principal_shortfall = max(0.0, principal - returned) if tick >= int(cfg.bond_term_ticks or 0) else 0.0
    return {
        "bond_principal_usd": principal,
        "bond_raw_lp_returned_usd": raw_returned,
        "bond_cumulative_fee_return_usd": returned,
        "bond_net_after_principal_usd": returned - principal if principal > 1e-9 else 0.0,
        "bond_simple_fee_yield": simple_yield,
        "bond_annualized_fee_yield": annualized_yield,
        "bond_cagr_fee_yield": cagr,
        "bond_coupon_due_usd": coupon_due,
        "bond_coupon_coverage_ratio": coupon_coverage,
        "bond_coupon_shortfall_usd": max(0.0, coupon_due - returned),
        "bond_principal_shortfall_at_term_usd": principal_shortfall,
        "bond_payback_ratio": returned / principal if principal > 1e-9 else 0.0,
    }


def route_success_rate(latest: dict[str, object]) -> float:
    found = safe_float(latest.get("route_found_tick"))
    failed = safe_float(latest.get("route_failed_tick"))
    denom = found + failed
    return found / denom if denom > 0.0 else 0.0


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
    tier_rng = random.Random(seed + 7919)
    pool_tiers: dict[str, str] = {}
    realized_edges: dict[tuple[str, str], float] = {}
    pool_swap_counts: Counter[str] = Counter()
    event_idx = 0
    bond_rows = []
    network_rows = []
    impact_latest: dict[str, float] = {}

    for _ in range(int(args.ticks)):
        engine.step(1)
        ensure_pool_tiers(engine, pool_tiers, tier_rng, calibration)
        event_idx = update_realized_edges(engine, realized_edges, pool_swap_counts, event_idx)
        latest = engine.metrics.network_rows[-1]
        tick = int(latest["tick"])
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
        stress_ratio = safe_float(latest.get("pools_under_stable_reserve")) / max(
            1.0, safe_float(latest.get("num_pools"), 1.0)
        )
        leakage_denom = (
            safe_float(latest.get("stable_onramp_usd_tick"))
            + safe_float(bmetrics.get("bond_principal_usd"))
            + safe_float(latest.get("stable_total_in_pools"))
        )
        leakage_ratio = safe_float(latest.get("stable_offramp_usd_tick")) / max(1e-9, leakage_denom)

        common = {
            "scenario": scenario,
            "run": run_index,
            "seed": seed,
            "coupon_target_annual": float(cfg.bond_coupon_target_annual),
            "bond_term_ticks": int(cfg.bond_term_ticks),
            "tick": tick,
            "num_pools": latest.get("num_pools", 0),
            "num_assets": latest.get("num_assets", 0),
            "transactions_per_tick": latest.get("transactions_per_tick", 0),
            "swap_volume_usd_tick": latest.get("swap_volume_usd_tick", 0.0),
            "route_success_rate_tick": route_success_rate(latest),
            "repayment_volume_usd": latest.get("repayment_volume_usd", 0.0),
            "loan_issuance_volume_usd": latest.get("loan_issuance_volume_usd", 0.0),
            "debt_outstanding_usd": latest.get("debt_outstanding_usd", 0.0),
            "fee_pool_cumulative_usd": latest.get("fee_pool_cumulative_usd", 0.0),
            "fee_clc_cumulative_usd": latest.get("fee_clc_cumulative_usd", 0.0),
            "claims_unpaid_usd_tick": latest.get("claims_unpaid_usd_tick", 0.0),
            "household_cash_stress_ratio": stress_ratio,
            "stable_liquidity_leakage_ratio_tick": leakage_ratio,
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
    failure = {
        "scenario": scenario,
        "run": run_index,
        "seed": seed,
        "coupon_target_annual": float(cfg.bond_coupon_target_annual),
        "bond_term_ticks": int(cfg.bond_term_ticks),
        "tick": final.get("tick", 0),
        "coupon_shortfall_usd": final.get("bond_coupon_shortfall_usd", 0.0),
        "coupon_shortfall_flag": int(safe_float(final.get("bond_coupon_shortfall_usd")) > 1e-9),
        "unpaid_claims_usd": final.get("claims_unpaid_usd_tick", 0.0),
        "unpaid_claims_flag": int(safe_float(final.get("claims_unpaid_usd_tick")) > 1e-9),
        "liquidity_leakage_ratio": final.get("stable_liquidity_leakage_ratio_tick", 0.0),
        "liquidity_leakage_flag": int(safe_float(final.get("stable_liquidity_leakage_ratio_tick")) > 0.05),
        "household_cash_stress_ratio": final.get("household_cash_stress_ratio", 0.0),
        "household_cash_stress_flag": int(safe_float(final.get("household_cash_stress_ratio")) > 0.20),
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


def main() -> int:
    args = parse_args()
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    coupons = parse_float_list(args.coupon_targets)
    terms = parse_int_list(args.terms)
    output_dir = Path(args.output).resolve()
    calibration = load_calibration(Path(args.calibration_dir).resolve())

    bond_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    total_jobs = len(scenarios) * len(coupons) * len(terms) * int(args.runs)
    job = 0

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
                    print(
                        f"[{job}/{total_jobs}] scenario={scenario} coupon={coupon} term={term_ticks} run={run_idx} seed={seed}",
                        flush=True,
                    )
                    b, n, f, s = run_one(
                        scenario=scenario,
                        coupon=coupon,
                        term_ticks=term_ticks,
                        run_index=run_idx,
                        seed=seed,
                        args=args,
                        calibration=calibration,
                    )
                    bond_rows.extend(b)
                    network_rows.extend(n)
                    failure_rows.extend(f)
                    summary_rows.append(s)

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

    write_csv(
        output_dir / "mc_bond_return_timeseries.csv",
        list(bond_rows[0].keys()) if bond_rows else [],
        bond_rows,
    )
    write_csv(
        output_dir / "mc_network_scaling_timeseries.csv",
        list(network_rows[0].keys()) if network_rows else [],
        network_rows,
    )
    write_csv(
        output_dir / "mc_run_summary.csv",
        list(summary_rows[0].keys()) if summary_rows else [],
        summary_rows,
    )
    write_csv(
        output_dir / "mc_failure_metrics.csv",
        list(failure_rows[0].keys()) if failure_rows else [],
        failure_rows,
    )
    write_csv(
        output_dir / "mc_failure_summary.csv",
        list(failure_summary_rows[0].keys()) if failure_summary_rows else [],
        failure_summary_rows,
    )
    write_csv(
        output_dir / "mc_timeseries_quantiles.csv",
        list(quantiles[0].keys()) if quantiles else [],
        quantiles,
    )
    write_latex_tables(output_dir, calibration, summary_rows, failure_summary_rows)
    public_output_privacy_check(output_dir)
    print(f"Wrote Monte Carlo artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
