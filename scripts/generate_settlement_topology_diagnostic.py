#!/usr/bin/env python3
"""Generate a matched-seed settlement topology diagnostic for the supplement.

The diagnostic reruns two trajectories with the same seed and traces executed
swaps into an actor/pool/asset interaction graph. It does not change Monte
Carlo economics; it only observes the event log produced by the existing
SimulationEngine.
"""

from __future__ import annotations

import argparse
import copy
import csv
import html
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SIM_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_regenbond_monte_carlo as rmc  # noqa: E402


FRONTIER_DIR = SIM_ROOT / "analysis" / "monte_carlo" / "bond_issuer_frontier_publication_v2"
TOPOLOGY_DIRNAME = "topology_diagnostic"
PANEL_BASELINE = "matched_no_bond_baseline"
PANEL_FAILED = "failed_stress_cell"
REQUIRED_NODE_TYPES = {"producer", "consumer", "pool", "voucher", "stable"}


@dataclass
class NodeAgg:
    panel: str
    node_id: str
    label: str
    node_type: str
    shape: str
    color: str
    weighted_degree: float = 0.0
    volume_usd: float = 0.0
    swap_count: int = 0
    displayed: int = 0


@dataclass
class EdgeAgg:
    panel: str
    source: str
    target: str
    edge_kind: str
    route_context: str
    route_source_role: str
    motif_class: str
    asset_in: str
    asset_out: str
    swap_count: int = 0
    volume_usd: float = 0.0
    displayed: int = 0

    @property
    def spring_weight(self) -> float:
        return math.log1p(max(0.0, self.volume_usd)) + 0.25 * max(0, self.swap_count)


@dataclass
class TraceGraph:
    panel: str
    coupon: float
    principal_ratio: float
    seed: int
    nodes: dict[str, NodeAgg] = field(default_factory=dict)
    edges: dict[tuple[str, str, str, str, str, str, str, str], EdgeAgg] = field(default_factory=dict)
    motif_edge_weights: Counter[str] = field(default_factory=Counter)
    context_edge_weights: Counter[str] = field(default_factory=Counter)
    event_count: int = 0
    swap_count: int = 0
    total_volume_usd: float = 0.0


@dataclass
class V2VSettlementEdge:
    panel: str
    source: str
    source_label: str
    source_type: str
    pool: str
    pool_label: str
    output_producer: str
    output_producer_label: str
    input_producer: str
    input_producer_label: str
    route_context: str
    route_source_role: str
    asset_in: str
    asset_out: str
    swap_count: int
    volume_usd: float
    displayed: int = 0

    @property
    def key(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.panel,
            self.source,
            self.pool,
            self.output_producer,
            self.input_producer,
            self.asset_in,
            self.asset_out,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the matched-seed settlement topology diagnostic for the RegenBonds supplement."
    )
    parser.add_argument(
        "--frontier-output",
        type=Path,
        default=FRONTIER_DIR,
        help="Final frontier-publication output directory containing bond_issuer_frontier_runs.csv.",
    )
    parser.add_argument("--ticks", type=int, default=260, help="Weekly ticks per traced trajectory.")
    parser.add_argument("--network-scale", default="current", help="Network scale to trace.")
    parser.add_argument("--failed-coupon", type=float, default=0.45, help="Failed stress-cell annual coupon.")
    parser.add_argument("--failed-principal-ratio", type=float, default=2.0, help="Failed stress-cell principal ratio.")
    parser.add_argument("--service-share", type=float, default=1.0, help="Bond fee-service share for the failed cell.")
    parser.add_argument("--seed", type=int, default=None, help="Override representative failed-cell seed.")
    parser.add_argument(
        "--regenbonds-root",
        type=Path,
        default=WORKSPACE_ROOT / "RegenBonds",
        help="Sibling RegenBonds repo. Outputs are copied there unless --skip-paper-copy is used.",
    )
    parser.add_argument("--skip-paper-copy", action="store_true", help="Do not copy outputs into RegenBonds analysis/.")
    parser.add_argument("--allow-svg-only", action="store_true", help="Do not fail if ImageMagick cannot write PNG.")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse topology_nodes.csv and topology_edges.csv to regenerate display flags and figures without rerunning trajectories.",
    )
    return parser.parse_args()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def short_label(value: str, max_len: int = 18) -> str:
    text = str(value)
    for prefix in ("VCHR:", "pool_", "agent_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if len(text) > max_len:
        return text[: max_len - 1] + "."
    return text


def asset_class(asset_id: str, stable_symbol: str) -> str:
    if asset_id == stable_symbol:
        return "stable"
    if str(asset_id).startswith("VCHR:"):
        return "voucher"
    return "other"


def motif_class(asset_in: str, asset_out: str, stable_symbol: str) -> str:
    in_class = asset_class(asset_in, stable_symbol)
    out_class = asset_class(asset_out, stable_symbol)
    if in_class == "voucher" and out_class == "voucher":
        return "voucher_to_voucher"
    if in_class == "voucher" and out_class == "stable":
        return "voucher_to_stable"
    if in_class == "stable" and out_class == "voucher":
        return "stable_to_voucher"
    if in_class == "stable" or out_class == "stable":
        return "stable_involved_other"
    return "other"


def is_stable_involved(motif: str) -> bool:
    return motif in {"voucher_to_stable", "stable_to_voucher", "stable_involved_other"}


def node_style(node_type: str) -> tuple[str, str]:
    return {
        "producer": ("square", "#2563eb"),
        "consumer": ("triangle", "#7c3aed"),
        "pool": ("star", "#b7791f"),
        "voucher": ("circle", "#1f78b4"),
        "stable": ("double_circle", "#d73027"),
        "other": ("circle", "#6b7280"),
    }.get(node_type, ("circle", "#6b7280"))


def add_node(graph: TraceGraph, node_id: str, label: str, node_type: str) -> None:
    if node_id in graph.nodes:
        return
    shape, color = node_style(node_type)
    graph.nodes[node_id] = NodeAgg(
        panel=graph.panel,
        node_id=node_id,
        label=label,
        node_type=node_type,
        shape=shape,
        color=color,
    )


def endpoint_node_for_pool(engine: rmc.SimulationEngine, pool_id: str | None) -> tuple[str, str, str]:
    if not pool_id or pool_id not in engine.pools:
        return ("pool:unknown", "unknown", "pool")
    pool = engine.pools[pool_id]
    role = str(pool.policy.role or "pool")
    if role == "producer":
        return (f"producer:{pool_id}", f"producer {short_label(pool_id, 10)}", "producer")
    if role == "consumer":
        return (f"consumer:{pool_id}", f"consumer {short_label(pool_id, 10)}", "consumer")
    return (f"pool:{pool_id}", f"pool {short_label(pool_id, 10)}", "pool")


def add_engine_static_nodes(graph: TraceGraph, engine: rmc.SimulationEngine) -> None:
    stable_id = engine.cfg.stable_symbol
    add_node(graph, f"asset:{stable_id}", stable_id, "stable")
    for pool_id, pool in engine.pools.items():
        if pool.policy.system_pool:
            continue
        endpoint_id, label, node_type = endpoint_node_for_pool(engine, pool_id)
        add_node(graph, endpoint_id, label, node_type)
    for voucher_id in sorted(engine.factory.voucher_specs):
        add_node(graph, f"asset:{voucher_id}", f"voucher {short_label(voucher_id, 10)}", "voucher")


def pool_value(engine: rmc.SimulationEngine, pool_id: str | None, asset_id: str) -> float:
    if pool_id and pool_id in engine.pools:
        pool = engine.pools[pool_id]
        value = pool.values.get_value(asset_id)
        if value > 0.0:
            return value
    if asset_id == engine.cfg.stable_symbol:
        return 1.0
    if str(asset_id).startswith("VCHR:"):
        return max(1e-12, float(engine.cfg.voucher_unit_value_usd or 1.0))
    return 1.0


def parse_escrow_actor_pool(actor: str) -> str | None:
    text = str(actor or "")
    for prefix in ("escrow:", "producer_debt_pressure:"):
        if text.startswith(prefix):
            return text.split(":", 1)[1]
    return None


def add_edge(
    graph: TraceGraph,
    *,
    source: str,
    target: str,
    edge_kind: str,
    route_context: str,
    route_source_role: str,
    motif: str,
    asset_in: str,
    asset_out: str,
    volume_usd: float,
) -> None:
    if source == target:
        return
    if source not in graph.nodes or target not in graph.nodes:
        return
    key = (
        graph.panel,
        source,
        target,
        edge_kind,
        route_context,
        motif,
        asset_in,
        asset_out,
    )
    edge = graph.edges.get(key)
    if edge is None:
        edge = EdgeAgg(
            panel=graph.panel,
            source=source,
            target=target,
            edge_kind=edge_kind,
            route_context=route_context,
            route_source_role=route_source_role,
            motif_class=motif,
            asset_in=asset_in,
            asset_out=asset_out,
        )
        graph.edges[key] = edge
    edge.swap_count += 1
    edge.volume_usd += max(0.0, volume_usd)


def trace_swap_event(graph: TraceGraph, engine: rmc.SimulationEngine, event: object) -> None:
    meta = getattr(event, "meta", {}) or {}
    receipt = meta.get("receipt") or {}
    if not isinstance(receipt, dict):
        return
    if str(receipt.get("status", "executed")) != "executed":
        return

    asset_in = str(receipt.get("asset_in", ""))
    asset_out = str(receipt.get("asset_out", ""))
    if not asset_in or not asset_out:
        return

    target_pool_id = str(getattr(event, "pool_id", "") or receipt.get("pool_id", "") or "")
    source_pool_id = meta.get("route_source_pool_id") or parse_escrow_actor_pool(str(receipt.get("actor", "")))
    if source_pool_id is None:
        source_pool_id = target_pool_id
    source_pool_id = str(source_pool_id)

    route_context = str(meta.get("route_context", "direct") or "direct")
    route_source_role = str(meta.get("route_source_role", "") or "")
    if not route_source_role and source_pool_id in engine.pools:
        route_source_role = str(engine.pools[source_pool_id].policy.role or "")

    source_node, source_label, source_type = endpoint_node_for_pool(engine, source_pool_id)
    add_node(graph, source_node, source_label, source_type)
    target_pool_node = f"pool:{target_pool_id}"
    add_node(graph, target_pool_node, f"pool {short_label(target_pool_id, 10)}", "pool")

    asset_in_node = f"asset:{asset_in}"
    asset_out_node = f"asset:{asset_out}"
    add_node(graph, asset_in_node, short_label(asset_in), asset_class(asset_in, engine.cfg.stable_symbol))
    add_node(graph, asset_out_node, short_label(asset_out), asset_class(asset_out, engine.cfg.stable_symbol))

    amount_in = safe_float(receipt.get("amount_in"))
    volume_usd = amount_in * pool_value(engine, target_pool_id, asset_in)
    motif = motif_class(asset_in, asset_out, engine.cfg.stable_symbol)
    graph.event_count += 1
    graph.swap_count += 1
    graph.total_volume_usd += volume_usd

    add_edge(
        graph,
        source=source_node,
        target=target_pool_node,
        edge_kind="source_uses_pool",
        route_context=route_context,
        route_source_role=route_source_role,
        motif=motif,
        asset_in=asset_in,
        asset_out=asset_out,
        volume_usd=volume_usd,
    )
    add_edge(
        graph,
        source=source_node,
        target=asset_in_node,
        edge_kind="source_offers_asset",
        route_context=route_context,
        route_source_role=route_source_role,
        motif=motif,
        asset_in=asset_in,
        asset_out=asset_out,
        volume_usd=volume_usd,
    )
    add_edge(
        graph,
        source=target_pool_node,
        target=asset_in_node,
        edge_kind="pool_receives_asset",
        route_context=route_context,
        route_source_role=route_source_role,
        motif=motif,
        asset_in=asset_in,
        asset_out=asset_out,
        volume_usd=volume_usd,
    )
    add_edge(
        graph,
        source=target_pool_node,
        target=asset_out_node,
        edge_kind="pool_pays_asset",
        route_context=route_context,
        route_source_role=route_source_role,
        motif=motif,
        asset_in=asset_in,
        asset_out=asset_out,
        volume_usd=volume_usd,
    )
    add_edge(
        graph,
        source=source_node,
        target=asset_out_node,
        edge_kind="source_receives_asset",
        route_context=route_context,
        route_source_role=route_source_role,
        motif=motif,
        asset_in=asset_in,
        asset_out=asset_out,
        volume_usd=volume_usd,
    )


def finalize_graph_degrees(graph: TraceGraph) -> None:
    for node in graph.nodes.values():
        node.weighted_degree = 0.0
        node.volume_usd = 0.0
        node.swap_count = 0
    graph.motif_edge_weights.clear()
    graph.context_edge_weights.clear()
    for edge in graph.edges.values():
        spring = edge.spring_weight
        graph.nodes[edge.source].weighted_degree += spring
        graph.nodes[edge.target].weighted_degree += spring
        graph.nodes[edge.source].volume_usd += edge.volume_usd
        graph.nodes[edge.target].volume_usd += edge.volume_usd
        graph.nodes[edge.source].swap_count += edge.swap_count
        graph.nodes[edge.target].swap_count += edge.swap_count
        graph.motif_edge_weights[edge.motif_class] += spring
        graph.context_edge_weights[edge.route_context] += spring


def choose_representative_failed_seed(
    runs_csv: Path,
    *,
    network_scale: str,
    coupon: float,
    principal_ratio: float,
    service_share: float,
) -> tuple[int, int, dict[str, float]]:
    rows: list[dict[str, str]] = []
    with runs_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("network_scale", "")) != network_scale:
                continue
            if abs(safe_float(row.get("coupon_target_annual")) - coupon) > 1e-9:
                continue
            if abs(safe_float(row.get("principal_ratio")) - principal_ratio) > 1e-9:
                continue
            if abs(safe_float(row.get("bond_fee_service_share"), service_share) - service_share) > 1e-9:
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(
            f"No failed-cell rows found in {runs_csv} for coupon={coupon}, principal_ratio={principal_ratio}."
        )

    candidate_metrics = [
        "issuer_paid_coverage_ratio",
        "issuer_available_service_cash_headroom_ratio",
        "observed_route_motif_voucher_to_voucher_share_total",
        "observed_route_motif_voucher_to_stable_share_total",
        "observed_route_motif_stable_to_voucher_share_total",
        "observed_route_motif_stable_involved_share_total",
        "route_success_rate_cumulative",
        "producer_debt_arrears_usd",
        "stable_receipts_waiting_for_repayment_usd",
    ]
    metric_values: dict[str, list[float]] = {
        metric: [safe_float(row.get(metric)) for row in rows if row.get(metric, "") != ""]
        for metric in candidate_metrics
    }
    medians = {metric: median(values) for metric, values in metric_values.items() if values}
    scales = {
        metric: max(1e-9, max(values) - min(values))
        for metric, values in metric_values.items()
        if values
    }
    best_row = rows[0]
    best_score = float("inf")
    for row in rows:
        score = 0.0
        used = 0
        for metric, metric_median in medians.items():
            score += abs(safe_float(row.get(metric)) - metric_median) / scales[metric]
            used += 1
        if used and score < best_score:
            best_score = score
            best_row = row

    diagnostics = {f"median_{metric}": value for metric, value in medians.items()}
    diagnostics["selection_distance"] = best_score
    return int(safe_float(best_row.get("seed"))), int(safe_float(best_row.get("run"))), diagnostics


def build_runner_args(args: argparse.Namespace, calibration: rmc.Calibration) -> argparse.Namespace:
    runner_args = rmc.parse_args(
        [
            "--scenario",
            "bond_issuer_frontier",
            "--runs",
            "1",
            "--ticks",
            str(args.ticks),
            "--term",
            str(args.ticks),
            "--network-scales",
            args.network_scale,
            "--principal-ratios",
            str(args.failed_principal_ratio),
            "--coupon-targets",
            str(args.failed_coupon),
            "--bond-fee-service-shares",
            str(args.service_share),
            "--voucher-settlement-mode",
            "redeem_outputs",
            "--issuer-payment-stride",
            "26",
            "--pool-clearing-stride",
            "13",
            "--producer-debt-pressure-period-ticks",
            "4",
            "--lender-voucher-cap-deposit-multiple",
            "5.0",
            "--bond-service-lockbox-mode",
            "remaining_schedule",
            "--route-success-mode",
            "diagnostic",
            "--calibration-dir",
            str(args.frontier_output.parents[1] / "sarafu_calibration"),
            "--output",
            str(args.frontier_output),
        ]
    )
    if not Path(runner_args.calibration_dir).exists():
        runner_args.calibration_dir = str(rmc.default_calibration_dir())
    runner_args.calibration_hash = calibration.calibration_hash
    rmc.apply_unit_normalization_context(runner_args, calibration)
    rmc.configure_sarafu_activity_controls(runner_args, calibration, int(runner_args.ticks), "frontier")
    return runner_args


def configure_panel_args(
    base_args: argparse.Namespace,
    calibration: rmc.Calibration,
    *,
    network_scale: str,
    coupon: float,
    principal_ratio: float,
    service_share: float,
) -> argparse.Namespace:
    runner_args = copy.deepcopy(base_args)
    capacity = rmc.certified_pool_capacity(calibration, network_scale, runner_args.certification_policy)
    certified_capacity = max(0.0, float(capacity["certified_backing_capacity_usd"]))
    principal_usd = certified_capacity * max(0.0, principal_ratio)
    rmc.apply_network_context(
        runner_args,
        calibration=calibration,
        network_scale=network_scale,
        principal_ratio=principal_ratio,
        principal_usd=principal_usd,
        service_share=service_share,
        certification_policy=runner_args.certification_policy,
    )
    runner_args._current_certified_pool_count = capacity["certified_pool_count"]
    runner_args._current_certified_capacity_usd = capacity["certified_backing_capacity_usd"]
    return runner_args


def run_traced_trajectory(
    *,
    panel: str,
    coupon: float,
    principal_ratio: float,
    service_share: float,
    seed: int,
    base_args: argparse.Namespace,
    calibration: rmc.Calibration,
    network_scale: str,
) -> TraceGraph:
    runner_args = configure_panel_args(
        base_args,
        calibration,
        network_scale=network_scale,
        coupon=coupon,
        principal_ratio=principal_ratio,
        service_share=service_share,
    )
    cfg = rmc.scenario_config("bond_issuer_frontier", coupon, int(runner_args.ticks), runner_args)
    engine = rmc.SimulationEngine(cfg=cfg, seed=seed)
    graph = TraceGraph(panel=panel, coupon=coupon, principal_ratio=principal_ratio, seed=seed)
    add_engine_static_nodes(graph, engine)

    tier_rng = random.Random(seed + 7919)
    pool_tiers: dict[str, str] = {}
    target_tier_counts = getattr(runner_args, "_target_tier_counts", None)
    event_idx = 0
    for _ in range(int(runner_args.ticks)):
        engine.step(1)
        rmc.ensure_pool_tiers(engine, pool_tiers, tier_rng, calibration, target_tier_counts)
        events = list(engine.log.events)[event_idx:]
        event_idx += len(events)
        for event in events:
            if getattr(event, "event_type", "") == "SWAP_EXECUTED":
                trace_swap_event(graph, engine, event)

    finalize_graph_degrees(graph)
    return graph


def pair_edges_for_display(graph: TraceGraph) -> dict[tuple[str, str], dict[str, object]]:
    pairs: dict[tuple[str, str], dict[str, object]] = {}
    for edge in graph.edges.values():
        a, b = sorted((edge.source, edge.target))
        key = (a, b)
        rec = pairs.setdefault(
            key,
            {
                "source": a,
                "target": b,
                "spring_weight": 0.0,
                "layout_weight": 0.0,
                "v2v_weight": 0.0,
                "v2s_weight": 0.0,
                "s2v_weight": 0.0,
                "stable_weight": 0.0,
                "other_weight": 0.0,
                "volume_usd": 0.0,
                "swap_count": 0,
                "motifs": Counter(),
            },
        )
        spring = edge.spring_weight
        rec["spring_weight"] = float(rec["spring_weight"]) + spring
        if edge.motif_class == "voucher_to_voucher":
            rec["v2v_weight"] = float(rec["v2v_weight"]) + spring
        elif edge.motif_class == "voucher_to_stable":
            rec["v2s_weight"] = float(rec["v2s_weight"]) + spring
            rec["stable_weight"] = float(rec["stable_weight"]) + spring
        elif edge.motif_class == "stable_to_voucher":
            rec["s2v_weight"] = float(rec["s2v_weight"]) + spring
            rec["stable_weight"] = float(rec["stable_weight"]) + spring
        elif is_stable_involved(edge.motif_class):
            rec["stable_weight"] = float(rec["stable_weight"]) + spring
        else:
            rec["other_weight"] = float(rec["other_weight"]) + spring
        rec["volume_usd"] = float(rec["volume_usd"]) + edge.volume_usd
        rec["swap_count"] = int(rec["swap_count"]) + edge.swap_count
        rec["motifs"][edge.motif_class] += spring
    return pairs


def node_family_degrees(
    graph: TraceGraph,
    pairs: dict[tuple[str, str], dict[str, object]] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if pairs is None:
        pairs = pair_edges_for_display(graph)
    v2v_degree: dict[str, float] = defaultdict(float)
    stable_degree: dict[str, float] = defaultdict(float)
    total_degree: dict[str, float] = defaultdict(float)
    for (a, b), rec in pairs.items():
        v2v_weight = safe_float(rec.get("v2v_weight"))
        stable_weight = safe_float(rec.get("stable_weight"))
        total_weight = safe_float(rec.get("spring_weight"))
        for node_id in (a, b):
            v2v_degree[node_id] += v2v_weight
            stable_degree[node_id] += stable_weight
            total_degree[node_id] += total_weight
    return v2v_degree, stable_degree, total_degree


def select_display_nodes(graph: TraceGraph, max_edges: int = 720) -> tuple[set[str], set[tuple[str, str]]]:
    """Select a V2V-centered publication subgraph.

    The full trace graph remains the audit object. This display selector gives
    voucher-to-voucher interaction the main visual weight, then overlays the
    strongest stable-involved pressure edges and enough actors to make the
    endpoint layer visible.
    """
    for node in graph.nodes.values():
        node.displayed = 0
    for edge in graph.edges.values():
        edge.displayed = 0

    pairs = pair_edges_for_display(graph)
    v2v_degree, stable_degree, total_degree = node_family_degrees(graph, pairs)
    sorted_pairs = sorted(pairs.items(), key=lambda item: safe_float(item[1]["spring_weight"]), reverse=True)
    sorted_v2v_pairs = sorted(pairs.items(), key=lambda item: safe_float(item[1]["v2v_weight"]), reverse=True)
    sorted_stable_pairs = sorted(pairs.items(), key=lambda item: safe_float(item[1]["stable_weight"]), reverse=True)
    selected: set[str] = set()
    selected.update(node_id for node_id, node in graph.nodes.items() if node.node_type == "stable")

    limits = {"voucher": 76, "pool": 52, "producer": 68, "consumer": 32}
    minimums = {"producer": 60, "consumer": 20}
    selected_by_type: dict[str, list[str]] = {}
    for node_type, limit in limits.items():
        candidates = [
            node for node in graph.nodes.values() if node.node_type == node_type and node.weighted_degree > 0.0
        ]
        candidates.sort(
            key=lambda node: (
                v2v_degree.get(node.node_id, 0.0),
                stable_degree.get(node.node_id, 0.0),
                node.weighted_degree,
            ),
            reverse=True,
        )
        chosen = [node.node_id for node in candidates[:limit]]
        if len(chosen) < minimums.get(node_type, 0):
            fallback = sorted(
                (node for node in candidates if node.node_id not in chosen),
                key=lambda node: (
                    stable_degree.get(node.node_id, 0.0),
                    total_degree.get(node.node_id, 0.0),
                ),
                reverse=True,
            )
            need = minimums[node_type] - len(chosen)
            chosen.extend(node.node_id for node in fallback[:need])
        selected_by_type[node_type] = chosen
        selected.update(chosen)

    for (a, b), rec in sorted_v2v_pairs[:220]:
        if safe_float(rec["v2v_weight"]) <= 0.0:
            break
        selected.add(a)
        selected.add(b)
    for (a, b), rec in sorted_stable_pairs[:100]:
        if safe_float(rec["stable_weight"]) <= 0.0:
            break
        selected.add(a)
        selected.add(b)

    display_edges: set[tuple[str, str]] = set()
    for (a, b), rec in sorted_v2v_pairs[:360]:
        if safe_float(rec["v2v_weight"]) <= 0.0:
            break
        if a in selected and b in selected:
            display_edges.add((a, b))
    for (a, b), rec in sorted_stable_pairs[:260]:
        if safe_float(rec["stable_weight"]) <= 0.0:
            break
        if a in selected and b in selected:
            display_edges.add((a, b))

    actor_nodes = selected_by_type.get("producer", []) + selected_by_type.get("consumer", [])
    for actor_id in actor_nodes:
        incident_v2v = [
            (pair, rec)
            for pair, rec in sorted_v2v_pairs
            if actor_id in pair
        ]
        incident_stable = [
            (pair, rec)
            for pair, rec in sorted_stable_pairs
            if actor_id in pair
        ]
        for (a, b), rec in incident_v2v[:4]:
            if safe_float(rec["v2v_weight"]) <= 0.0:
                continue
            selected.add(a)
            selected.add(b)
            display_edges.add((a, b))
        for (a, b), rec in incident_stable[:3]:
            if safe_float(rec["stable_weight"]) <= 0.0:
                continue
            selected.add(a)
            selected.add(b)
            display_edges.add((a, b))

    # Keep explicit actor, pool, voucher, and stable edge families visible.
    family_pairs: dict[str, list[tuple[tuple[str, str], dict[str, object]]]] = defaultdict(list)
    for pair, rec in sorted_pairs:
        a, b = pair
        ta = graph.nodes.get(a).node_type if a in graph.nodes else ""
        tb = graph.nodes.get(b).node_type if b in graph.nodes else ""
        types = {ta, tb}
        if types & {"producer", "consumer"} and "voucher" in types:
            family_pairs["actor_voucher"].append((pair, rec))
        if types & {"producer", "consumer"} and "pool" in types:
            family_pairs["actor_pool"].append((pair, rec))
        if types & {"producer", "consumer"} and "stable" in types:
            family_pairs["actor_stable"].append((pair, rec))
        if types == {"pool", "voucher"}:
            family_pairs["pool_voucher"].append((pair, rec))
        if types == {"pool", "stable"}:
            family_pairs["pool_stable"].append((pair, rec))

    family_limits = {
        "actor_voucher": 220,
        "actor_pool": 180,
        "actor_stable": 120,
        "pool_voucher": 160,
        "pool_stable": 90,
    }
    required_family_edges: set[tuple[str, str]] = set()
    for family, candidates in family_pairs.items():
        candidates.sort(
            key=lambda item: (
                safe_float(item[1]["v2v_weight"]),
                safe_float(item[1]["stable_weight"]),
                safe_float(item[1]["spring_weight"]),
            ),
            reverse=True,
        )
        required_family_edges.update(pair for pair, _rec in candidates[:12])
        for (a, b), _rec in candidates[: family_limits[family]]:
            selected.add(a)
            selected.add(b)
            display_edges.add((a, b))

    # Bound the figure density while protecting V2V and actor incident links.
    if len(display_edges) > max_edges:
        protected: set[tuple[str, str]] = set()
        actor_set = set(actor_nodes)
        for pair in display_edges:
            if pair in required_family_edges or pair[0] in actor_set or pair[1] in actor_set:
                protected.add(pair)
        ranked = sorted(
            display_edges - protected,
            key=lambda pair: (
                safe_float(pairs[pair]["v2v_weight"]),
                0.35 * safe_float(pairs[pair]["stable_weight"]),
                safe_float(pairs[pair]["spring_weight"]),
            ),
            reverse=True,
        )
        remaining_slots = max(0, max_edges - len(protected))
        display_edges = protected | set(ranked[:remaining_slots])

    connected_nodes = {
        node_id for node_id, node in graph.nodes.items() if node.node_type == "stable"
    }
    for a, b in display_edges:
        connected_nodes.add(a)
        connected_nodes.add(b)
    selected = connected_nodes
    for node in graph.nodes.values():
        node.displayed = 1 if node.node_id in selected else 0
    for edge in graph.edges.values():
        a, b = sorted((edge.source, edge.target))
        edge.displayed = 1 if (a, b) in display_edges and edge.source in selected and edge.target in selected else 0
    return selected, display_edges


def apply_v2v_centered_layout_weights(pair_edges: dict[tuple[str, str], dict[str, object]]) -> None:
    max_v2v = max((safe_float(rec.get("v2v_weight")) for rec in pair_edges.values()), default=0.0)
    max_stable = max((safe_float(rec.get("stable_weight")) for rec in pair_edges.values()), default=0.0)
    max_total = max((safe_float(rec.get("spring_weight")) for rec in pair_edges.values()), default=0.0)
    for rec in pair_edges.values():
        v2v_component = math.sqrt(safe_float(rec.get("v2v_weight")) / max(1e-9, max_v2v))
        stable_component = math.sqrt(safe_float(rec.get("stable_weight")) / max(1e-9, max_stable))
        residual_component = math.sqrt(safe_float(rec.get("spring_weight")) / max(1e-9, max_total))
        rec["layout_weight"] = 0.04 + v2v_component + 0.16 * stable_component + 0.03 * residual_component


def spring_layout(
    node_ids: list[str],
    pair_edges: dict[tuple[str, str], dict[str, object]],
    *,
    seed: int,
    iterations: int = 320,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    if not node_ids:
        return {}
    positions = {node_id: [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)] for node_id in node_ids}
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    n = max(1, len(node_ids))
    k = math.sqrt(1.0 / n)
    max_weight = max((safe_float(rec.get("layout_weight", rec["spring_weight"])) for rec in pair_edges.values()), default=1.0)
    max_weight = max(1e-9, max_weight)

    for step in range(iterations):
        disp = {node_id: [0.0, 0.0] for node_id in node_ids}
        temp = 0.08 * (1.0 - step / max(1, iterations)) + 0.004
        for i, a in enumerate(node_ids):
            ax, ay = positions[a]
            for b in node_ids[i + 1 :]:
                bx, by = positions[b]
                dx = ax - bx
                dy = ay - by
                dist = math.hypot(dx, dy) + 1e-6
                force = (k * k) / dist
                fx = (dx / dist) * force
                fy = (dy / dist) * force
                disp[a][0] += fx
                disp[a][1] += fy
                disp[b][0] -= fx
                disp[b][1] -= fy

        for (a, b), rec in pair_edges.items():
            if a not in index or b not in index:
                continue
            ax, ay = positions[a]
            bx, by = positions[b]
            dx = ax - bx
            dy = ay - by
            dist = math.hypot(dx, dy) + 1e-6
            layout_weight = safe_float(rec.get("layout_weight", rec["spring_weight"]))
            normalized = 0.25 + 1.75 * math.sqrt(layout_weight / max_weight)
            force = (dist * dist / k) * normalized
            fx = (dx / dist) * force
            fy = (dy / dist) * force
            disp[a][0] -= fx
            disp[a][1] -= fy
            disp[b][0] += fx
            disp[b][1] += fy

        for node_id in node_ids:
            dx, dy = disp[node_id]
            length = math.hypot(dx, dy)
            if length > 0.0:
                positions[node_id][0] += (dx / length) * min(length, temp)
                positions[node_id][1] += (dy / length) * min(length, temp)
            positions[node_id][0] = max(-1.4, min(1.4, positions[node_id][0]))
            positions[node_id][1] = max(-1.1, min(1.1, positions[node_id][1]))

    return {node_id: (xy[0], xy[1]) for node_id, xy in positions.items()}


def scale_positions(
    positions: dict[str, tuple[float, float]],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> dict[str, tuple[float, float]]:
    if not positions:
        return {}
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(1e-9, max_x - min_x)
    dy = max(1e-9, max_y - min_y)
    pad = 34.0
    out = {}
    for node_id, (px, py) in positions.items():
        sx = x + pad + ((px - min_x) / dx) * max(1.0, width - 2 * pad)
        sy = y + pad + ((py - min_y) / dy) * max(1.0, height - 2 * pad)
        out[node_id] = (sx, sy)
    return out


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    weight: str = "normal",
    anchor: str = "start",
    fill: str = "#111827",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{esc(text)}</text>'
    )


def star_points(cx: float, cy: float, radius: float) -> str:
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.44
        points.append(f"{cx + math.cos(angle) * r:.2f},{cy + math.sin(angle) * r:.2f}")
    return " ".join(points)


def triangle_points(cx: float, cy: float, radius: float) -> str:
    return " ".join(
        f"{cx + math.cos(-math.pi / 2 + i * 2 * math.pi / 3) * radius:.2f},"
        f"{cy + math.sin(-math.pi / 2 + i * 2 * math.pi / 3) * radius:.2f}"
        for i in range(3)
    )


def draw_node(node: NodeAgg, x: float, y: float, radius: float) -> list[str]:
    color = node.color
    stroke = "#1f2937"
    if node.node_type == "stable":
        return [
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="#ffffff" stroke="{color}" stroke-width="3.0"/>',
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{max(2.0, radius - 5):.2f}" fill="none" stroke="{color}" stroke-width="2.0"/>',
        ]
    if node.shape == "square":
        side = radius * 1.78
        return [
            f'<rect x="{x - side/2:.2f}" y="{y - side/2:.2f}" width="{side:.2f}" height="{side:.2f}" '
            f'fill="{color}" fill-opacity="0.90" stroke="{stroke}" stroke-width="1.1"/>'
        ]
    if node.shape == "triangle":
        return [
            f'<polygon points="{triangle_points(x, y, radius)}" fill="{color}" fill-opacity="0.86" '
            f'stroke="{stroke}" stroke-width="1.1"/>'
        ]
    if node.shape == "star":
        return [
            f'<polygon points="{star_points(x, y, radius)}" fill="{color}" fill-opacity="0.88" '
            f'stroke="{stroke}" stroke-width="0.8"/>'
        ]
    return [
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="0.84" '
        f'stroke="{stroke}" stroke-width="0.8"/>'
    ]


def edge_color(motif: str) -> str:
    if motif == "voucher_to_voucher":
        return "#2c7fb8"
    if motif == "voucher_to_stable":
        return "#b23a48"
    if motif == "stable_to_voucher":
        return "#d97706"
    if motif == "stable_involved_other":
        return "#ef4444"
    return "#6b7280"


def edge_family_components(rec: dict[str, object]) -> list[tuple[str, float]]:
    v2v = safe_float(rec.get("v2v_weight"))
    v2s = safe_float(rec.get("v2s_weight"))
    s2v = safe_float(rec.get("s2v_weight"))
    stable_other = max(0.0, safe_float(rec.get("stable_weight")) - v2s - s2v)
    components: list[tuple[str, float]] = []
    if stable_other > 0.0:
        components.append(("stable_involved_other", stable_other))
    if v2s > 0.0:
        components.append(("voucher_to_stable", v2s))
    if s2v > 0.0:
        components.append(("stable_to_voucher", s2v))
    if v2v > 0.0:
        components.append(("voucher_to_voucher", v2v))
    return components


def panel_summary_metrics(graph: TraceGraph) -> dict[str, float]:
    total_node_degree = sum(node.weighted_degree for node in graph.nodes.values())
    stable_degree = sum(node.weighted_degree for node in graph.nodes.values() if node.node_type == "stable")
    voucher_degree = sum(node.weighted_degree for node in graph.nodes.values() if node.node_type == "voucher")
    total_edge_weight = sum(edge.spring_weight for edge in graph.edges.values())
    v2v_weight = sum(edge.spring_weight for edge in graph.edges.values() if edge.motif_class == "voucher_to_voucher")
    stable_involved_weight = sum(edge.spring_weight for edge in graph.edges.values() if is_stable_involved(edge.motif_class))
    return {
        "stable_weighted_degree_share": stable_degree / total_node_degree if total_node_degree > 0 else 0.0,
        "stable_to_voucher_weighted_degree_ratio": stable_degree / max(1e-9, voucher_degree),
        "v2v_edge_weight_share": v2v_weight / total_edge_weight if total_edge_weight > 0 else 0.0,
        "stable_involved_edge_weight_share": stable_involved_weight / total_edge_weight if total_edge_weight > 0 else 0.0,
    }


def producer_from_voucher_node(graph: TraceGraph, asset_id: str) -> tuple[str, str]:
    text = str(asset_id)
    if text.startswith("VCHR:agent_"):
        suffix = text.split("agent_", 1)[1]
        producer_id = f"producer:pool_{suffix}"
        if producer_id in graph.nodes:
            return producer_id, graph.nodes[producer_id].label
        return f"producer_issuer:agent_{suffix}", f"issuer {short_label('agent_' + suffix, 12)}"
    if text.startswith("VCHR:"):
        suffix = text.split(":", 1)[1]
        return f"producer_issuer:{suffix}", f"issuer {short_label(suffix, 12)}"
    return "producer_issuer:unknown", "issuer unknown"


def build_v2v_settlement_edges(graph: TraceGraph) -> list[V2VSettlementEdge]:
    aggregated: dict[tuple[str, str, str, str, str, str, str], V2VSettlementEdge] = {}
    for edge in graph.edges.values():
        if edge.edge_kind != "source_uses_pool" or edge.motif_class != "voucher_to_voucher":
            continue
        source_node = graph.nodes.get(edge.source)
        pool_node = graph.nodes.get(edge.target)
        if source_node is None or pool_node is None:
            continue
        if source_node.node_type not in {"producer", "consumer"} or pool_node.node_type != "pool":
            continue
        output_producer, output_label = producer_from_voucher_node(graph, edge.asset_out)
        input_producer, input_label = producer_from_voucher_node(graph, edge.asset_in)
        key = (
            graph.panel,
            edge.source,
            edge.target,
            output_producer,
            input_producer,
            edge.asset_in,
            edge.asset_out,
        )
        record = aggregated.get(key)
        if record is None:
            record = V2VSettlementEdge(
                panel=graph.panel,
                source=edge.source,
                source_label=source_node.label,
                source_type=source_node.node_type,
                pool=edge.target,
                pool_label=pool_node.label,
                output_producer=output_producer,
                output_producer_label=output_label,
                input_producer=input_producer,
                input_producer_label=input_label,
                route_context=edge.route_context,
                route_source_role=edge.route_source_role,
                asset_in=edge.asset_in,
                asset_out=edge.asset_out,
                swap_count=0,
                volume_usd=0.0,
            )
            aggregated[key] = record
        record.swap_count += edge.swap_count
        record.volume_usd += edge.volume_usd
    return sorted(aggregated.values(), key=lambda rec: rec.volume_usd, reverse=True)


def route_motif_source_use_metrics(graph: TraceGraph) -> dict[str, float]:
    counts: Counter[str] = Counter()
    volumes: Counter[str] = Counter()
    for edge in graph.edges.values():
        if edge.edge_kind != "source_uses_pool":
            continue
        counts[edge.motif_class] += edge.swap_count
        volumes[edge.motif_class] += edge.volume_usd
    total_count = sum(counts.values())
    total_volume = sum(volumes.values())
    stable_count = sum(count for motif, count in counts.items() if is_stable_involved(motif))
    stable_volume = sum(volume for motif, volume in volumes.items() if is_stable_involved(motif))
    return {
        "source_use_route_count": float(total_count),
        "source_use_route_volume_usd": float(total_volume),
        "v2v_route_count": float(counts["voucher_to_voucher"]),
        "v2v_route_volume_usd": float(volumes["voucher_to_voucher"]),
        "v2v_route_count_share": counts["voucher_to_voucher"] / total_count if total_count else 0.0,
        "v2v_route_volume_share": volumes["voucher_to_voucher"] / total_volume if total_volume else 0.0,
        "stable_involved_route_count_share": stable_count / total_count if total_count else 0.0,
        "stable_involved_route_volume_share": stable_volume / total_volume if total_volume else 0.0,
    }


def select_v2v_settlement_display(
    baseline_records: list[V2VSettlementEdge],
    failed_records: list[V2VSettlementEdge],
    *,
    source_limit: int = 28,
    pool_limit: int = 26,
    issuer_limit: int = 34,
    path_limit_per_panel: int = 96,
) -> dict[str, object]:
    for record in baseline_records + failed_records:
        record.displayed = 0

    all_records = baseline_records + failed_records
    source_volume: Counter[str] = Counter()
    pool_volume: Counter[str] = Counter()
    issuer_volume: Counter[str] = Counter()
    source_labels: dict[str, str] = {}
    pool_labels: dict[str, str] = {}
    issuer_labels: dict[str, str] = {}
    source_types: dict[str, str] = {}
    for record in all_records:
        source_volume[record.source] += record.volume_usd
        pool_volume[record.pool] += record.volume_usd
        issuer_volume[record.output_producer] += record.volume_usd
        source_labels[record.source] = record.source_label
        pool_labels[record.pool] = record.pool_label
        issuer_labels[record.output_producer] = record.output_producer_label
        source_types[record.source] = record.source_type

    selected_sources = [node for node, _ in source_volume.most_common(source_limit)]
    selected_pools = [node for node, _ in pool_volume.most_common(pool_limit)]
    selected_issuers = [node for node, _ in issuer_volume.most_common(issuer_limit)]

    # Preserve consumers if they appear at all; otherwise a pure volume ranking
    # can hide the fact that consumer V2V is nearly absent in the failed trace.
    consumer_sources = [
        node for node, _volume in source_volume.most_common() if source_types.get(node) == "consumer"
    ]
    for node in consumer_sources[:6]:
        if node not in selected_sources:
            selected_sources.append(node)

    selected_sets = {
        "source": set(selected_sources),
        "pool": set(selected_pools),
        "issuer": set(selected_issuers),
    }
    panel_totals = {
        PANEL_BASELINE: sum(record.volume_usd for record in baseline_records),
        PANEL_FAILED: sum(record.volume_usd for record in failed_records),
    }
    max_panel_total = max(1e-9, max(panel_totals.values()))
    panel_limits = {
        panel: max(42, int(round(path_limit_per_panel * total / max_panel_total)))
        for panel, total in panel_totals.items()
    }
    displayed_by_panel: dict[str, set[tuple[str, str, str, str, str, str, str]]] = defaultdict(set)
    for records in (baseline_records, failed_records):
        panel_limit = panel_limits.get(records[0].panel if records else "", path_limit_per_panel)
        panel_top = sorted(records, key=lambda rec: rec.volume_usd, reverse=True)[:panel_limit]
        for record in panel_top:
            selected_sets["source"].add(record.source)
            selected_sets["pool"].add(record.pool)
            selected_sets["issuer"].add(record.output_producer)
    for records in (baseline_records, failed_records):
        candidates = [
            rec
            for rec in sorted(records, key=lambda item: item.volume_usd, reverse=True)
            if rec.source in selected_sets["source"]
            and rec.pool in selected_sets["pool"]
            and rec.output_producer in selected_sets["issuer"]
        ]
        panel_limit = panel_limits.get(records[0].panel if records else "", path_limit_per_panel)
        for record in candidates[:panel_limit]:
            record.displayed = 1
            displayed_by_panel[record.panel].add(record.key)

    source_order = [node for node, _ in source_volume.most_common() if node in selected_sets["source"]]
    pool_order = [node for node, _ in pool_volume.most_common() if node in selected_sets["pool"]]
    issuer_order = [node for node, _ in issuer_volume.most_common() if node in selected_sets["issuer"]]
    return {
        "source_order": source_order,
        "pool_order": pool_order,
        "issuer_order": issuer_order,
        "source_labels": source_labels,
        "pool_labels": pool_labels,
        "issuer_labels": issuer_labels,
        "source_types": source_types,
        "displayed_by_panel": displayed_by_panel,
        "panel_limits": panel_limits,
    }


def v2v_settlement_summary_rows(
    baseline: TraceGraph,
    failed: TraceGraph,
    baseline_records: list[V2VSettlementEdge],
    failed_records: list[V2VSettlementEdge],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for graph, records in ((baseline, baseline_records), (failed, failed_records)):
        metrics = route_motif_source_use_metrics(graph)
        displayed_records = [record for record in records if record.displayed]
        rows.append(
            {
                "panel": graph.panel,
                "matched_seed": graph.seed,
                "coupon_target_annual": graph.coupon,
                "principal_ratio": graph.principal_ratio,
                "v2v_settlement_record_count": len(records),
                "v2v_settlement_record_count_displayed": len(displayed_records),
                "v2v_settlement_swap_count": sum(record.swap_count for record in records),
                "v2v_settlement_volume_usd": f"{sum(record.volume_usd for record in records):.9f}",
                "displayed_v2v_settlement_swap_count": sum(record.swap_count for record in displayed_records),
                "displayed_v2v_settlement_volume_usd": f"{sum(record.volume_usd for record in displayed_records):.9f}",
                "source_use_route_count": int(metrics["source_use_route_count"]),
                "source_use_route_volume_usd": f"{metrics['source_use_route_volume_usd']:.9f}",
                "v2v_route_count": int(metrics["v2v_route_count"]),
                "v2v_route_volume_usd": f"{metrics['v2v_route_volume_usd']:.9f}",
                "v2v_route_count_share": f"{metrics['v2v_route_count_share']:.9f}",
                "v2v_route_volume_share": f"{metrics['v2v_route_volume_share']:.9f}",
                "stable_involved_route_count_share": f"{metrics['stable_involved_route_count_share']:.9f}",
                "stable_involved_route_volume_share": f"{metrics['stable_involved_route_volume_share']:.9f}",
                "consumer_source_v2v_swap_count": sum(
                    record.swap_count for record in records if record.source_type == "consumer"
                ),
                "producer_source_v2v_swap_count": sum(
                    record.swap_count for record in records if record.source_type == "producer"
                ),
            }
        )
    base_metrics = route_motif_source_use_metrics(baseline)
    failed_metrics = route_motif_source_use_metrics(failed)
    rows.append(
        {
            "panel": "failed_minus_baseline",
            "matched_seed": failed.seed,
            "coupon_target_annual": failed.coupon,
            "principal_ratio": failed.principal_ratio,
            "v2v_settlement_record_count": len(failed_records) - len(baseline_records),
            "v2v_settlement_record_count_displayed": (
                sum(record.displayed for record in failed_records)
                - sum(record.displayed for record in baseline_records)
            ),
            "v2v_settlement_swap_count": (
                sum(record.swap_count for record in failed_records)
                - sum(record.swap_count for record in baseline_records)
            ),
            "v2v_settlement_volume_usd": (
                f"{sum(record.volume_usd for record in failed_records) - sum(record.volume_usd for record in baseline_records):.9f}"
            ),
            "displayed_v2v_settlement_swap_count": (
                sum(record.swap_count for record in failed_records if record.displayed)
                - sum(record.swap_count for record in baseline_records if record.displayed)
            ),
            "displayed_v2v_settlement_volume_usd": (
                f"{sum(record.volume_usd for record in failed_records if record.displayed) - sum(record.volume_usd for record in baseline_records if record.displayed):.9f}"
            ),
            "source_use_route_count": int(failed_metrics["source_use_route_count"] - base_metrics["source_use_route_count"]),
            "source_use_route_volume_usd": (
                f"{failed_metrics['source_use_route_volume_usd'] - base_metrics['source_use_route_volume_usd']:.9f}"
            ),
            "v2v_route_count": int(failed_metrics["v2v_route_count"] - base_metrics["v2v_route_count"]),
            "v2v_route_volume_usd": f"{failed_metrics['v2v_route_volume_usd'] - base_metrics['v2v_route_volume_usd']:.9f}",
            "v2v_route_count_share": f"{failed_metrics['v2v_route_count_share'] - base_metrics['v2v_route_count_share']:.9f}",
            "v2v_route_volume_share": f"{failed_metrics['v2v_route_volume_share'] - base_metrics['v2v_route_volume_share']:.9f}",
            "stable_involved_route_count_share": (
                f"{failed_metrics['stable_involved_route_count_share'] - base_metrics['stable_involved_route_count_share']:.9f}"
            ),
            "stable_involved_route_volume_share": (
                f"{failed_metrics['stable_involved_route_volume_share'] - base_metrics['stable_involved_route_volume_share']:.9f}"
            ),
            "consumer_source_v2v_swap_count": (
                sum(record.swap_count for record in failed_records if record.source_type == "consumer")
                - sum(record.swap_count for record in baseline_records if record.source_type == "consumer")
            ),
            "producer_source_v2v_swap_count": (
                sum(record.swap_count for record in failed_records if record.source_type == "producer")
                - sum(record.swap_count for record in baseline_records if record.source_type == "producer")
            ),
        }
    )
    return rows


def v2v_settlement_edge_rows(records: list[V2VSettlementEdge]) -> list[dict[str, object]]:
    return [
        {
            "panel": record.panel,
            "source": record.source,
            "source_label": record.source_label,
            "source_type": record.source_type,
            "pool": record.pool,
            "pool_label": record.pool_label,
            "output_producer": record.output_producer,
            "output_producer_label": record.output_producer_label,
            "input_producer": record.input_producer,
            "input_producer_label": record.input_producer_label,
            "route_context": record.route_context,
            "route_source_role": record.route_source_role,
            "asset_in": record.asset_in,
            "asset_out": record.asset_out,
            "swap_count": record.swap_count,
            "volume_usd": f"{record.volume_usd:.9f}",
            "displayed": record.displayed,
        }
        for record in records
    ]


def draw_panel(
    graph: TraceGraph,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    layout_seed: int,
) -> list[str]:
    selected_nodes, display_edges = select_display_nodes(graph)
    pair_edges_all = pair_edges_for_display(graph)
    pair_edges = {key: value for key, value in pair_edges_all.items() if key in display_edges}
    apply_v2v_centered_layout_weights(pair_edges)
    node_ids = sorted(selected_nodes)
    positions = spring_layout(node_ids, pair_edges, seed=layout_seed)
    scaled = scale_positions(positions, x=x, y=y + 54, width=width, height=height - 80)
    max_family_weight = {
        "voucher_to_voucher": max((safe_float(rec.get("v2v_weight")) for rec in pair_edges.values()), default=1.0),
        "voucher_to_stable": max((safe_float(rec.get("v2s_weight")) for rec in pair_edges.values()), default=1.0),
        "stable_to_voucher": max((safe_float(rec.get("s2v_weight")) for rec in pair_edges.values()), default=1.0),
        "stable_involved_other": max((safe_float(rec.get("stable_weight")) for rec in pair_edges.values()), default=1.0),
    }
    max_by_type: dict[str, float] = defaultdict(lambda: 1.0)
    for node_id in node_ids:
        node = graph.nodes[node_id]
        max_by_type[node.node_type] = max(max_by_type[node.node_type], node.weighted_degree)
    visible_counts = Counter(graph.nodes[node_id].node_type for node_id in node_ids)
    metrics = panel_summary_metrics(graph)
    elements = [
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>',
        svg_text(x + 14, y + 26, title, size=17, weight="bold"),
        svg_text(
            x + 14,
            y + 48,
            (
                f"V2V share {metrics['v2v_edge_weight_share']:.3f}; "
                f"stable-involved share {metrics['stable_involved_edge_weight_share']:.3f}; "
                f"actors shown P{visible_counts.get('producer', 0)}/C{visible_counts.get('consumer', 0)}"
            ),
            size=11,
            fill="#4b5563",
        ),
    ]

    draw_specs: list[tuple[int, float, str, str, str, float]] = []
    for (a, b), rec in pair_edges.items():
        if a not in scaled or b not in scaled:
            continue
        for motif, weight in edge_family_components(rec):
            # Stable edges are an overlay; V2V edges are drawn last and remain
            # visually primary in the publication subgraph.
            layer = 2 if motif == "voucher_to_voucher" else 1
            draw_specs.append((layer, weight, a, b, motif, weight))

    for _layer, _rank_weight, a, b, motif, weight in sorted(draw_specs, key=lambda item: (item[0], item[1])):
        family_max = max(1e-9, max_family_weight.get(motif, max_family_weight["stable_involved_other"]))
        norm = math.sqrt(weight / family_max)
        if motif == "voucher_to_voucher":
            width_px = 0.75 + 3.8 * norm
            opacity = 0.18 + 0.56 * norm
        else:
            width_px = 0.45 + 2.3 * norm
            opacity = 0.06 + 0.28 * norm
        ax, ay = scaled[a]
        bx, by = scaled[b]
        elements.append(
            f'<line x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" y2="{by:.2f}" '
            f'stroke="{edge_color(motif)}" stroke-opacity="{opacity:.3f}" stroke-width="{width_px:.2f}"/>'
        )

    label_candidates = sorted(
        (graph.nodes[node_id] for node_id in node_ids),
        key=lambda node: (node.node_type != "stable", -node.weighted_degree),
    )[:22]
    label_ids = {node.node_id for node in label_candidates}
    draw_order = {"voucher": 0, "pool": 1, "producer": 2, "consumer": 3, "stable": 4}
    for node_id in sorted(
        node_ids,
        key=lambda nid: (draw_order.get(graph.nodes[nid].node_type, 0), graph.nodes[nid].weighted_degree),
    ):
        node = graph.nodes[node_id]
        if node_id not in scaled:
            continue
        nx, ny = scaled[node_id]
        local_norm = math.sqrt(max(0.0, node.weighted_degree) / max(1e-9, max_by_type[node.node_type]))
        if node.node_type == "stable":
            radius = 14.0 + 10.0 * local_norm
        elif node.node_type == "pool":
            radius = 7.0 + 13.0 * local_norm
        elif node.node_type == "voucher":
            radius = 6.5 + 11.0 * local_norm
        elif node.node_type in {"producer", "consumer"}:
            radius = 7.0 + 7.0 * local_norm
        else:
            radius = 5.0 + 8.0 * local_norm
        elements.extend(draw_node(node, nx, ny, radius))
        if node_id in label_ids:
            elements.append(svg_text(nx + radius + 3, ny + 3, node.label, size=8, fill="#374151"))
    return elements


def draw_legend(x: float, y: float) -> list[str]:
    items = [
        ("producer", "#2563eb", "square"),
        ("consumer", "#7c3aed", "triangle"),
        ("pool", "#b7791f", "star"),
        ("voucher", "#1f78b4", "circle"),
        ("stable", "#d73027", "double_circle"),
    ]
    edge_items = [
        ("V2V", "#2c7fb8"),
        ("V2S", "#b23a48"),
        ("S2V", "#d97706"),
    ]
    elements = [
        svg_text(x, y, "Legend", size=14, weight="bold"),
    ]
    yy = y + 22
    for label, color, shape in items:
        node = NodeAgg("", "", label, label, shape, color)
        elements.extend(draw_node(node, x + 9, yy - 4, 7))
        elements.append(svg_text(x + 24, yy, label, size=10, fill="#374151"))
        yy += 22
    yy += 6
    elements.append(svg_text(x, yy, "Dominant edge motif", size=11, weight="bold", fill="#374151"))
    yy += 18
    for label, color in edge_items:
        elements.append(
            f'<line x1="{x:.2f}" y1="{yy - 4:.2f}" x2="{x + 18:.2f}" y2="{yy - 4:.2f}" '
            f'stroke="{color}" stroke-width="3" stroke-opacity="0.55"/>'
        )
        elements.append(svg_text(x + 24, yy, label, size=10, fill="#374151"))
        yy += 20
    return elements


def horizontal_bar(
    *,
    x: float,
    y: float,
    width: float,
    label: str,
    value: float,
    color: str,
    suffix: str = "%",
) -> list[str]:
    value = max(0.0, min(1.0, value))
    return [
        svg_text(x, y - 5, label, size=10, weight="bold", fill="#374151"),
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="8" fill="#e5e7eb" rx="2"/>',
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width * value:.2f}" height="8" fill="{color}" rx="2"/>',
        svg_text(x + width + 8, y + 8, f"{100.0 * value:.1f}{suffix}", size=10, fill="#374151"),
    ]


def money_short(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}k"
    return f"${value:.0f}"


def node_y_positions(order: list[str], top: float, bottom: float) -> dict[str, float]:
    if not order:
        return {}
    if len(order) == 1:
        return {order[0]: 0.5 * (top + bottom)}
    span = max(1.0, bottom - top)
    return {node_id: top + span * idx / (len(order) - 1) for idx, node_id in enumerate(order)}


def draw_layered_symbol(node_type: str, x: float, y: float, *, role: str) -> list[str]:
    if role == "pool":
        node = NodeAgg("", "", "pool", "pool", "star", "#b7791f")
        return draw_node(node, x, y, 8.0)
    if node_type == "consumer":
        node = NodeAgg("", "", "consumer", "consumer", "triangle", "#7c3aed")
        return draw_node(node, x, y, 7.0)
    if role == "issuer":
        return [
            f'<rect x="{x - 6.5:.2f}" y="{y - 6.5:.2f}" width="13" height="13" fill="#ffffff" '
            f'stroke="#1f78b4" stroke-width="2.0"/>'
        ]
    node = NodeAgg("", "", "producer", "producer", "square", "#2563eb")
    return draw_node(node, x, y, 7.0)


def active_v2v_nodes(records: list[V2VSettlementEdge]) -> tuple[set[str], set[str], set[str]]:
    displayed = [record for record in records if record.displayed]
    return (
        {record.source for record in displayed},
        {record.pool for record in displayed},
        {record.output_producer for record in displayed},
    )


def draw_v2v_settlement_panel(
    graph: TraceGraph,
    records: list[V2VSettlementEdge],
    selection: dict[str, object],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    global_max_volume: float,
    reference_v2v_volume: float,
) -> list[str]:
    metrics = route_motif_source_use_metrics(graph)
    source_order = selection["source_order"]  # type: ignore[assignment]
    pool_order = selection["pool_order"]  # type: ignore[assignment]
    issuer_order = selection["issuer_order"]  # type: ignore[assignment]
    source_labels = selection["source_labels"]  # type: ignore[assignment]
    pool_labels = selection["pool_labels"]  # type: ignore[assignment]
    issuer_labels = selection["issuer_labels"]  # type: ignore[assignment]
    source_types = selection["source_types"]  # type: ignore[assignment]
    source_active, pool_active, issuer_active = active_v2v_nodes(records)

    top = y + 150
    bottom = y + height - 42
    x_source = x + 92
    x_pool = x + width * 0.50
    x_issuer = x + width - 92
    source_y = node_y_positions([node for node in source_order if node in source_active], top, bottom)
    pool_y = node_y_positions([node for node in pool_order if node in pool_active], top, bottom)
    issuer_y = node_y_positions([node for node in issuer_order if node in issuer_active], top, bottom)

    elements = [
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>',
        svg_text(x + 14, y + 25, title, size=17, weight="bold"),
        svg_text(
            x + 14,
            y + 49,
            f"V2V volume {money_short(metrics['v2v_route_volume_usd'])}; displayed paths {sum(record.displayed for record in records)}; source actors -> pools -> output voucher issuers",
            size=11,
            fill="#4b5563",
        ),
        svg_text(x_source, y + 116, "source actor", size=11, weight="bold", anchor="middle", fill="#374151"),
        svg_text(x_pool, y + 116, "pool", size=11, weight="bold", anchor="middle", fill="#374151"),
        svg_text(x_issuer, y + 116, "output voucher issuer", size=11, weight="bold", anchor="middle", fill="#374151"),
        f'<line x1="{x_source:.2f}" y1="{top - 10:.2f}" x2="{x_source:.2f}" y2="{bottom + 10:.2f}" stroke="#e5e7eb" stroke-width="1"/>',
        f'<line x1="{x_pool:.2f}" y1="{top - 10:.2f}" x2="{x_pool:.2f}" y2="{bottom + 10:.2f}" stroke="#e5e7eb" stroke-width="1"/>',
        f'<line x1="{x_issuer:.2f}" y1="{top - 10:.2f}" x2="{x_issuer:.2f}" y2="{bottom + 10:.2f}" stroke="#e5e7eb" stroke-width="1"/>',
    ]
    elements.extend(
        horizontal_bar(
            x=x + 14,
            y=y + 74,
            width=130,
            label="relative V2V volume",
            value=metrics["v2v_route_volume_usd"] / max(1e-9, reference_v2v_volume),
            color="#2c7fb8",
        )
    )
    elements.extend(
        horizontal_bar(
            x=x + 220,
            y=y + 74,
            width=110,
            label="V2V share",
            value=metrics["v2v_route_volume_share"],
            color="#2c7fb8",
        )
    )
    elements.extend(
        horizontal_bar(
            x=x + 390,
            y=y + 74,
            width=110,
            label="stable-involved share",
            value=metrics["stable_involved_route_volume_share"],
            color="#b23a48",
        )
    )
    elements.append(
        svg_text(
            x + 570,
            y + 82,
            f"V2V count share {100.0 * metrics['v2v_route_count_share']:.1f}%",
            size=10,
            fill="#374151",
        )
    )

    displayed_records = [record for record in records if record.displayed]
    for record in sorted(displayed_records, key=lambda item: item.volume_usd):
        if record.source not in source_y or record.pool not in pool_y or record.output_producer not in issuer_y:
            continue
        norm = math.sqrt(record.volume_usd / max(1e-9, global_max_volume))
        width_px = 0.55 + 6.0 * norm
        opacity = 0.14 + 0.58 * norm
        sy = source_y[record.source]
        py = pool_y[record.pool]
        iy = issuer_y[record.output_producer]
        elements.append(
            f'<path d="M {x_source:.2f},{sy:.2f} C {x_source + 80:.2f},{sy:.2f} {x_pool - 80:.2f},{py:.2f} {x_pool:.2f},{py:.2f}" '
            f'fill="none" stroke="#2c7fb8" stroke-width="{width_px:.2f}" stroke-opacity="{opacity:.3f}"/>'
        )
        elements.append(
            f'<path d="M {x_pool:.2f},{py:.2f} C {x_pool + 80:.2f},{py:.2f} {x_issuer - 80:.2f},{iy:.2f} {x_issuer:.2f},{iy:.2f}" '
            f'fill="none" stroke="#1f78b4" stroke-width="{width_px:.2f}" stroke-opacity="{opacity:.3f}"/>'
        )

    label_limit = 10
    for node_id, yy in source_y.items():
        elements.extend(draw_layered_symbol(source_types.get(node_id, "producer"), x_source, yy, role="source"))
    for idx, (node_id, yy) in enumerate(source_y.items()):
        if idx < label_limit:
            elements.append(svg_text(x_source + 12, yy + 3, short_label(source_labels.get(node_id, node_id), 14), size=7, fill="#374151"))
    for node_id, yy in pool_y.items():
        elements.extend(draw_layered_symbol("pool", x_pool, yy, role="pool"))
    for idx, (node_id, yy) in enumerate(pool_y.items()):
        if idx < label_limit:
            elements.append(svg_text(x_pool + 12, yy + 3, short_label(pool_labels.get(node_id, node_id), 12), size=7, fill="#374151"))
    for node_id, yy in issuer_y.items():
        elements.extend(draw_layered_symbol("producer", x_issuer, yy, role="issuer"))
    for idx, (node_id, yy) in enumerate(issuer_y.items()):
        if idx < label_limit:
            elements.append(svg_text(x_issuer + 12, yy + 3, short_label(issuer_labels.get(node_id, node_id), 14), size=7, fill="#374151"))
    return elements


def draw_v2v_settlement_legend(x: float, y: float) -> list[str]:
    elements = [svg_text(x, y, "Legend", size=14, weight="bold")]
    yy = y + 26
    for label, node_type, role in [
        ("producer source", "producer", "source"),
        ("consumer source", "consumer", "source"),
        ("pool", "pool", "pool"),
        ("output issuer", "producer", "issuer"),
    ]:
        elements.extend(draw_layered_symbol(node_type, x + 10, yy - 5, role=role))
        elements.append(svg_text(x + 28, yy, label, size=10, fill="#374151"))
        yy += 26
    yy += 8
    elements.append(svg_text(x, yy, "V2V settlement path", size=11, weight="bold", fill="#374151"))
    yy += 18
    elements.append(
        f'<path d="M {x:.2f},{yy:.2f} C {x + 22:.2f},{yy:.2f} {x + 38:.2f},{yy:.2f} {x + 60:.2f},{yy:.2f}" '
        f'fill="none" stroke="#2c7fb8" stroke-width="4" stroke-opacity="0.70"/>'
    )
    elements.append(svg_text(x + 68, yy + 4, "source -> pool -> issuer", size=10, fill="#374151"))
    return elements


def write_svg_figure(path: Path, baseline: TraceGraph, failed: TraceGraph) -> None:
    baseline_records = build_v2v_settlement_edges(baseline)
    failed_records = build_v2v_settlement_edges(failed)
    selection = select_v2v_settlement_display(baseline_records, failed_records)
    displayed_records = [
        record for record in baseline_records + failed_records if record.displayed
    ]
    global_max_volume = max((record.volume_usd for record in displayed_records), default=1.0)
    reference_v2v_volume = route_motif_source_use_metrics(baseline)["v2v_route_volume_usd"]
    width = 1800
    height = 930
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(48, 42, "Matched-Seed V2V Settlement-Path Diagnostic", size=28, weight="bold"),
        svg_text(
            48,
            68,
            "Layered view of producer/consumer source -> pool -> output-voucher issuer paths; stable pressure is shown as metrics, not layout force.",
            size=14,
            fill="#4b5563",
        ),
    ]
    elements.extend(
        draw_v2v_settlement_panel(
            baseline,
            baseline_records,
            selection,
            x=48,
            y=100,
            width=780,
            height=760,
            title="No-bond baseline, same seed",
            global_max_volume=global_max_volume,
            reference_v2v_volume=reference_v2v_volume,
        )
    )
    elements.extend(
        draw_v2v_settlement_panel(
            failed,
            failed_records,
            selection,
            x=880,
            y=100,
            width=780,
            height=760,
            title="Failed stress cell: 45% coupon, 2.00 principal ratio",
            global_max_volume=global_max_volume,
            reference_v2v_volume=reference_v2v_volume,
        )
    )
    elements.extend(draw_v2v_settlement_legend(1688, 132))
    elements.append(
        svg_text(
            48,
            900,
            "Edge width is normalized across both panels by V2V settlement-path volume. Derived CSVs contain the displayed producer-pool-producer paths; full trace CSVs remain audit outputs.",
            size=12,
            fill="#4b5563",
        )
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def maybe_write_png(svg_path: Path, *, allow_svg_only: bool) -> None:
    converter = shutil.which("convert")
    if converter is None:
        if allow_svg_only:
            return
        raise RuntimeError("ImageMagick convert is required for PNG output; rerun with --allow-svg-only to skip PNG.")
    png_path = svg_path.with_suffix(".png")
    subprocess.run(
        [converter, "-density", "180", str(svg_path), "-background", "white", "-alpha", "remove", str(png_path)],
        check=True,
    )


def graph_node_rows(graphs: list[TraceGraph]) -> list[dict[str, object]]:
    rows = []
    for graph in graphs:
        for node in graph.nodes.values():
            rows.append(
                {
                    "panel": graph.panel,
                    "seed": graph.seed,
                    "coupon_target_annual": graph.coupon,
                    "principal_ratio": graph.principal_ratio,
                    "node_id": node.node_id,
                    "label": node.label,
                    "node_type": node.node_type,
                    "shape": node.shape,
                    "color": node.color,
                    "weighted_degree": f"{node.weighted_degree:.9f}",
                    "volume_usd": f"{node.volume_usd:.9f}",
                    "swap_count": node.swap_count,
                    "displayed": node.displayed,
                }
            )
    return rows


def graph_edge_rows(graphs: list[TraceGraph]) -> list[dict[str, object]]:
    rows = []
    for graph in graphs:
        for edge in graph.edges.values():
            rows.append(
                {
                    "panel": graph.panel,
                    "seed": graph.seed,
                    "coupon_target_annual": graph.coupon,
                    "principal_ratio": graph.principal_ratio,
                    "source": edge.source,
                    "target": edge.target,
                    "source_type": graph.nodes[edge.source].node_type,
                    "target_type": graph.nodes[edge.target].node_type,
                    "edge_kind": edge.edge_kind,
                    "route_context": edge.route_context,
                    "route_source_role": edge.route_source_role,
                    "motif_class": edge.motif_class,
                    "asset_in": edge.asset_in,
                    "asset_out": edge.asset_out,
                    "swap_count": edge.swap_count,
                    "volume_usd": f"{edge.volume_usd:.9f}",
                    "spring_weight": f"{edge.spring_weight:.9f}",
                    "displayed": edge.displayed,
                }
            )
    return rows


def graph_display_node_rows(graphs: list[TraceGraph]) -> list[dict[str, object]]:
    rows = []
    for graph in graphs:
        pairs = pair_edges_for_display(graph)
        v2v_degree, stable_degree, total_degree = node_family_degrees(graph, pairs)
        for node in graph.nodes.values():
            if not node.displayed:
                continue
            rows.append(
                {
                    "panel": graph.panel,
                    "seed": graph.seed,
                    "coupon_target_annual": graph.coupon,
                    "principal_ratio": graph.principal_ratio,
                    "node_id": node.node_id,
                    "label": node.label,
                    "node_type": node.node_type,
                    "shape": node.shape,
                    "color": node.color,
                    "weighted_degree": f"{node.weighted_degree:.9f}",
                    "v2v_weighted_degree": f"{v2v_degree.get(node.node_id, 0.0):.9f}",
                    "stable_involved_weighted_degree": f"{stable_degree.get(node.node_id, 0.0):.9f}",
                    "total_pair_weighted_degree": f"{total_degree.get(node.node_id, 0.0):.9f}",
                    "volume_usd": f"{node.volume_usd:.9f}",
                    "swap_count": node.swap_count,
                }
            )
    return rows


def graph_display_edge_rows(graphs: list[TraceGraph]) -> list[dict[str, object]]:
    rows = []
    for graph in graphs:
        for edge in graph.edges.values():
            if not edge.displayed:
                continue
            rows.append(
                {
                    "panel": graph.panel,
                    "seed": graph.seed,
                    "coupon_target_annual": graph.coupon,
                    "principal_ratio": graph.principal_ratio,
                    "source": edge.source,
                    "target": edge.target,
                    "source_type": graph.nodes[edge.source].node_type,
                    "target_type": graph.nodes[edge.target].node_type,
                    "edge_kind": edge.edge_kind,
                    "route_context": edge.route_context,
                    "route_source_role": edge.route_source_role,
                    "motif_class": edge.motif_class,
                    "asset_in": edge.asset_in,
                    "asset_out": edge.asset_out,
                    "swap_count": edge.swap_count,
                    "volume_usd": f"{edge.volume_usd:.9f}",
                    "spring_weight": f"{edge.spring_weight:.9f}",
                }
            )
    return rows


def load_existing_graphs(output_dir: Path) -> tuple[TraceGraph, TraceGraph, int, dict[str, float]]:
    nodes_path = output_dir / "topology_nodes.csv"
    edges_path = output_dir / "topology_edges.csv"
    summary_path = output_dir / "topology_summary.csv"
    if not nodes_path.exists() or not edges_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Existing topology_nodes.csv, topology_edges.csv, and topology_summary.csv are required.")

    graphs: dict[str, TraceGraph] = {}
    with nodes_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            panel = str(row["panel"])
            graph = graphs.setdefault(
                panel,
                TraceGraph(
                    panel=panel,
                    coupon=safe_float(row.get("coupon_target_annual")),
                    principal_ratio=safe_float(row.get("principal_ratio")),
                    seed=int(safe_float(row.get("seed"))),
                ),
            )
            graph.nodes[row["node_id"]] = NodeAgg(
                panel=panel,
                node_id=row["node_id"],
                label=row["label"],
                node_type=row["node_type"],
                shape=row["shape"],
                color=row["color"],
            )

    with edges_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            panel = str(row["panel"])
            if panel not in graphs:
                continue
            graph = graphs[panel]
            key = (
                panel,
                row["source"],
                row["target"],
                row["edge_kind"],
                row["route_context"],
                row["motif_class"],
                row["asset_in"],
                row["asset_out"],
            )
            graph.edges[key] = EdgeAgg(
                panel=panel,
                source=row["source"],
                target=row["target"],
                edge_kind=row["edge_kind"],
                route_context=row["route_context"],
                route_source_role=row.get("route_source_role", ""),
                motif_class=row["motif_class"],
                asset_in=row["asset_in"],
                asset_out=row["asset_out"],
                swap_count=int(safe_float(row.get("swap_count"))),
                volume_usd=safe_float(row.get("volume_usd")),
            )

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    failed_summary = next((row for row in summary if row.get("panel") == PANEL_FAILED), {})
    selected_failed_run = int(safe_float(failed_summary.get("selected_failed_run"), -1))
    selection_diagnostics = {
        "selection_distance": safe_float(failed_summary.get("selection_distance_to_failed_cell_medians"))
    }

    baseline = graphs.get(PANEL_BASELINE)
    failed = graphs.get(PANEL_FAILED)
    if baseline is None or failed is None:
        raise RuntimeError("Existing topology CSVs must contain baseline and failed panels.")
    for graph in (baseline, failed):
        graph.swap_count = sum(edge.swap_count for edge in graph.edges.values()) // 5
        graph.total_volume_usd = sum(edge.volume_usd for edge in graph.edges.values()) / 5.0
        finalize_graph_degrees(graph)
    return baseline, failed, selected_failed_run, selection_diagnostics


def summary_rows(
    baseline: TraceGraph,
    failed: TraceGraph,
    *,
    selected_failed_run: int,
    selection_diagnostics: dict[str, float],
) -> list[dict[str, object]]:
    metrics_by_panel = {graph.panel: panel_summary_metrics(graph) for graph in (baseline, failed)}
    baseline_metrics = metrics_by_panel[baseline.panel]
    failed_metrics = metrics_by_panel[failed.panel]
    failed_has_higher_stable_concentration = (
        failed_metrics["stable_weighted_degree_share"] > baseline_metrics["stable_weighted_degree_share"]
        and failed_metrics["stable_to_voucher_weighted_degree_ratio"]
        > baseline_metrics["stable_to_voucher_weighted_degree_ratio"]
    )
    rows = []
    for graph in (baseline, failed):
        metrics = metrics_by_panel[graph.panel]
        node_types = {node.node_type for node in graph.nodes.values()}
        displayed_nodes = sum(1 for node in graph.nodes.values() if node.displayed)
        displayed_edges = sum(1 for edge in graph.edges.values() if edge.displayed)
        rows.append(
            {
                "panel": graph.panel,
                "matched_seed": graph.seed,
                "selected_failed_run": selected_failed_run,
                "coupon_target_annual": graph.coupon,
                "principal_ratio": graph.principal_ratio,
                "swap_count": graph.swap_count,
                "total_volume_usd": f"{graph.total_volume_usd:.9f}",
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "displayed_node_count": displayed_nodes,
                "displayed_edge_count": displayed_edges,
                "required_node_types_present": int(REQUIRED_NODE_TYPES.issubset(node_types)),
                "stable_weighted_degree_share": f"{metrics['stable_weighted_degree_share']:.9f}",
                "stable_to_voucher_weighted_degree_ratio": f"{metrics['stable_to_voucher_weighted_degree_ratio']:.9f}",
                "v2v_edge_weight_share": f"{metrics['v2v_edge_weight_share']:.9f}",
                "stable_involved_edge_weight_share": f"{metrics['stable_involved_edge_weight_share']:.9f}",
                "failed_has_higher_stable_concentration": int(failed_has_higher_stable_concentration),
                "selection_distance_to_failed_cell_medians": f"{selection_diagnostics.get('selection_distance', 0.0):.9f}",
            }
        )
    rows.append(
        {
            "panel": "failed_minus_baseline",
            "matched_seed": failed.seed,
            "selected_failed_run": selected_failed_run,
            "coupon_target_annual": failed.coupon,
            "principal_ratio": failed.principal_ratio,
            "swap_count": failed.swap_count - baseline.swap_count,
            "total_volume_usd": f"{failed.total_volume_usd - baseline.total_volume_usd:.9f}",
            "node_count": len(failed.nodes) - len(baseline.nodes),
            "edge_count": len(failed.edges) - len(baseline.edges),
            "displayed_node_count": "",
            "displayed_edge_count": "",
            "required_node_types_present": "",
            "stable_weighted_degree_share": (
                f"{failed_metrics['stable_weighted_degree_share'] - baseline_metrics['stable_weighted_degree_share']:.9f}"
            ),
            "stable_to_voucher_weighted_degree_ratio": (
                f"{failed_metrics['stable_to_voucher_weighted_degree_ratio'] - baseline_metrics['stable_to_voucher_weighted_degree_ratio']:.9f}"
            ),
            "v2v_edge_weight_share": (
                f"{failed_metrics['v2v_edge_weight_share'] - baseline_metrics['v2v_edge_weight_share']:.9f}"
            ),
            "stable_involved_edge_weight_share": (
                f"{failed_metrics['stable_involved_edge_weight_share'] - baseline_metrics['stable_involved_edge_weight_share']:.9f}"
            ),
            "failed_has_higher_stable_concentration": int(failed_has_higher_stable_concentration),
            "selection_distance_to_failed_cell_medians": f"{selection_diagnostics.get('selection_distance', 0.0):.9f}",
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_notes(path: Path, summary: list[dict[str, object]]) -> None:
    by_panel = {row["panel"]: row for row in summary}
    base = by_panel[PANEL_BASELINE]
    fail = by_panel[PANEL_FAILED]
    comparison = by_panel["failed_minus_baseline"]
    evidence = "higher stable concentration" if int(fail["failed_has_higher_stable_concentration"]) else "mixed stable-clustering evidence"
    lines = [
        "# Settlement Topology Diagnostic",
        "",
        "This V2V settlement-path diagnostic reruns one matched seed from the final 100-run settlement-capacity frontier.",
        "It is a representative trace diagnostic, not a binding aggregate result.",
        "The figure derives producer/consumer source -> pool -> output-voucher issuer paths from V2V source-use-pool trace rows. Stable-involved settlement pressure is reported as panel metrics rather than used as layout force.",
        "",
        f"- Matched seed: `{base['matched_seed']}`.",
        f"- Baseline stable weighted-degree share: `{base['stable_weighted_degree_share']}`.",
        f"- Failed-cell stable weighted-degree share: `{fail['stable_weighted_degree_share']}`.",
        f"- Failed minus baseline stable weighted-degree share: `{comparison['stable_weighted_degree_share']}`.",
        f"- Baseline V2V edge-weight share: `{base['v2v_edge_weight_share']}`.",
        f"- Failed-cell V2V edge-weight share: `{fail['v2v_edge_weight_share']}`.",
        f"- Summary evidence label: `{evidence}`.",
        "",
        "Layout distances are generated from weighted simulated interactions and are not geographic distances, causal proof, or participant-level empirical network claims.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_outputs_to_regenbonds(output_dir: Path, regenbonds_root: Path) -> None:
    if not regenbonds_root.exists():
        return
    destination = regenbonds_root / "analysis" / "monte_carlo" / "bond_issuer_frontier_publication_v2" / TOPOLOGY_DIRNAME
    destination.mkdir(parents=True, exist_ok=True)
    for name in [
        "topology_nodes.csv",
        "topology_edges.csv",
        "topology_display_nodes.csv",
        "topology_display_edges.csv",
        "topology_v2v_settlement_edges.csv",
        "topology_v2v_settlement_summary.csv",
        "topology_summary.csv",
        "topology_notes.md",
        "fig_topology_baseline_vs_failed.svg",
        "fig_topology_baseline_vs_failed.png",
    ]:
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, destination / name)


def validate_outputs(output_dir: Path, baseline: TraceGraph, failed: TraceGraph) -> None:
    if baseline.seed != failed.seed:
        raise RuntimeError("Topology diagnostic baseline and failed panels do not use the same seed.")
    node_types = {node.node_type for graph in (baseline, failed) for node in graph.nodes.values()}
    missing = REQUIRED_NODE_TYPES - node_types
    if missing:
        raise RuntimeError(f"Topology diagnostic is missing node type(s): {sorted(missing)}")
    if not baseline.edges or not failed.edges:
        raise RuntimeError("Topology diagnostic edge CSV would be empty for at least one panel.")
    v2v_summary_path = output_dir / "topology_v2v_settlement_summary.csv"
    v2v_edges_path = output_dir / "topology_v2v_settlement_edges.csv"
    if not v2v_edges_path.exists() or not v2v_summary_path.exists():
        raise RuntimeError("Missing derived V2V settlement diagnostic CSV outputs.")
    with v2v_edges_path.open(newline="", encoding="utf-8") as handle:
        v2v_edges = list(csv.DictReader(handle))
    if not any(row.get("panel") == PANEL_BASELINE for row in v2v_edges) or not any(
        row.get("panel") == PANEL_FAILED for row in v2v_edges
    ):
        raise RuntimeError("Derived V2V settlement diagnostic CSV is missing a panel.")
    with v2v_summary_path.open(newline="", encoding="utf-8") as handle:
        v2v_rows = {row["panel"]: row for row in csv.DictReader(handle)}
    if PANEL_BASELINE in v2v_rows and PANEL_FAILED in v2v_rows:
        baseline_v2v_share = safe_float(v2v_rows[PANEL_BASELINE].get("v2v_route_volume_share"))
        failed_v2v_share = safe_float(v2v_rows[PANEL_FAILED].get("v2v_route_volume_share"))
        baseline_stable_share = safe_float(v2v_rows[PANEL_BASELINE].get("stable_involved_route_volume_share"))
        failed_stable_share = safe_float(v2v_rows[PANEL_FAILED].get("stable_involved_route_volume_share"))
        if not baseline_v2v_share > failed_v2v_share:
            raise RuntimeError("Expected baseline V2V route volume share to exceed failed panel.")
        if not failed_stable_share > baseline_stable_share:
            raise RuntimeError("Expected failed stable-involved route volume share to exceed baseline panel.")
    for graph in (baseline, failed):
        displayed_nodes = [node for node in graph.nodes.values() if node.displayed]
        displayed_types = {node.node_type for node in displayed_nodes}
        missing_displayed = REQUIRED_NODE_TYPES - displayed_types
        if missing_displayed:
            raise RuntimeError(
                f"Displayed topology panel {graph.panel} is missing node type(s): {sorted(missing_displayed)}"
            )
        if sum(1 for node in displayed_nodes if node.node_type == "producer") < 60:
            raise RuntimeError(f"Displayed topology panel {graph.panel} has fewer than 60 producers.")
        if sum(1 for node in displayed_nodes if node.node_type == "consumer") < 20:
            raise RuntimeError(f"Displayed topology panel {graph.panel} has fewer than 20 consumers.")

        edge_families: set[str] = set()
        for edge in graph.edges.values():
            if not edge.displayed:
                continue
            source_type = graph.nodes[edge.source].node_type
            target_type = graph.nodes[edge.target].node_type
            types = {source_type, target_type}
            if types & {"producer", "consumer"} and "voucher" in types:
                edge_families.add("actor_voucher")
            if types & {"producer", "consumer"} and "pool" in types:
                edge_families.add("actor_pool")
            if types & {"producer", "consumer"} and "stable" in types:
                edge_families.add("actor_stable")
            if types == {"pool", "voucher"}:
                edge_families.add("pool_voucher")
            if types == {"pool", "stable"}:
                edge_families.add("pool_stable")
        missing_families = {"actor_voucher", "actor_pool", "actor_stable", "pool_voucher", "pool_stable"} - edge_families
        if missing_families:
            raise RuntimeError(
                f"Displayed topology panel {graph.panel} is missing edge family/families: {sorted(missing_families)}"
            )
    for name in [
        "topology_nodes.csv",
        "topology_edges.csv",
        "topology_display_nodes.csv",
        "topology_display_edges.csv",
        "topology_v2v_settlement_edges.csv",
        "topology_v2v_settlement_summary.csv",
        "topology_summary.csv",
        "fig_topology_baseline_vs_failed.svg",
        "fig_topology_baseline_vs_failed.png",
    ]:
        if not (output_dir / name).exists():
            raise RuntimeError(f"Missing topology diagnostic output: {name}")


def main() -> int:
    args = parse_args()
    frontier_dir = args.frontier_output.resolve()
    runs_csv = frontier_dir / "bond_issuer_frontier_runs.csv"
    output_dir = frontier_dir / TOPOLOGY_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing:
        baseline, failed, failed_run, selection_diagnostics = load_existing_graphs(output_dir)
        print(f"[topology] reusing existing CSV traces from {output_dir}", flush=True)
    elif not runs_csv.exists():
        raise FileNotFoundError(f"Missing final frontier run CSV: {runs_csv}")
    else:
        if args.seed is None:
            seed, failed_run, selection_diagnostics = choose_representative_failed_seed(
                runs_csv,
                network_scale=args.network_scale,
                coupon=args.failed_coupon,
                principal_ratio=args.failed_principal_ratio,
                service_share=args.service_share,
            )
        else:
            seed = int(args.seed)
            failed_run = -1
            selection_diagnostics = {"selection_distance": 0.0}

        calibration = rmc.load_calibration(Path(rmc.default_calibration_dir()).resolve())
        base_runner_args = build_runner_args(args, calibration)

        print(f"[topology] selected matched seed={seed} failed_run={failed_run}", flush=True)
        baseline = run_traced_trajectory(
            panel=PANEL_BASELINE,
            coupon=0.0,
            principal_ratio=0.0,
            service_share=0.0,
            seed=seed,
            base_args=base_runner_args,
            calibration=calibration,
            network_scale=args.network_scale,
        )
        print(f"[topology] baseline swaps={baseline.swap_count} edges={len(baseline.edges)}", flush=True)
        failed = run_traced_trajectory(
            panel=PANEL_FAILED,
            coupon=args.failed_coupon,
            principal_ratio=args.failed_principal_ratio,
            service_share=args.service_share,
            seed=seed,
            base_args=base_runner_args,
            calibration=calibration,
            network_scale=args.network_scale,
        )
        print(f"[topology] failed swaps={failed.swap_count} edges={len(failed.edges)}", flush=True)

    # Mark display flags before writing tables.
    select_display_nodes(baseline)
    select_display_nodes(failed)
    summary = summary_rows(
        baseline,
        failed,
        selected_failed_run=failed_run,
        selection_diagnostics=selection_diagnostics,
    )
    if not args.reuse_existing:
        write_csv(output_dir / "topology_nodes.csv", graph_node_rows([baseline, failed]))
        write_csv(output_dir / "topology_edges.csv", graph_edge_rows([baseline, failed]))
    write_csv(output_dir / "topology_display_nodes.csv", graph_display_node_rows([baseline, failed]))
    write_csv(output_dir / "topology_display_edges.csv", graph_display_edge_rows([baseline, failed]))
    baseline_v2v_records = build_v2v_settlement_edges(baseline)
    failed_v2v_records = build_v2v_settlement_edges(failed)
    select_v2v_settlement_display(baseline_v2v_records, failed_v2v_records)
    write_csv(
        output_dir / "topology_v2v_settlement_edges.csv",
        v2v_settlement_edge_rows(baseline_v2v_records + failed_v2v_records),
    )
    write_csv(
        output_dir / "topology_v2v_settlement_summary.csv",
        v2v_settlement_summary_rows(baseline, failed, baseline_v2v_records, failed_v2v_records),
    )
    write_csv(output_dir / "topology_summary.csv", summary)
    write_notes(output_dir / "topology_notes.md", summary)

    svg_path = output_dir / "fig_topology_baseline_vs_failed.svg"
    write_svg_figure(svg_path, baseline, failed)
    maybe_write_png(svg_path, allow_svg_only=bool(args.allow_svg_only))
    validate_outputs(output_dir, baseline, failed)
    if not args.skip_paper_copy:
        copy_outputs_to_regenbonds(output_dir, args.regenbonds_root.resolve())
    print(f"[topology] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
