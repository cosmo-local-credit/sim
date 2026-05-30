#!/usr/bin/env python3
"""Generate a single-panel topology diagnostic with explicit USD flows.

This reads the existing matched-seed topology CSVs produced by
generate_settlement_topology_diagnostic.py. It does not rerun the simulator.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SIM_ROOT.parent
DEFAULT_TOPOLOGY_DIR = (
    WORKSPACE_ROOT
    / "RegenBonds"
    / "analysis"
    / "monte_carlo"
    / "bond_issuer_frontier_publication_v2"
    / "topology_diagnostic"
)

PANEL_BASELINE = "matched_no_bond_baseline"
PANEL_FAILED = "failed_stress_cell"

TEXT = "#111827"
MUTED = "#4b5563"
GRID = "#e5e7eb"
BORDER = "#d1d5db"
V2V = "#2c7fb8"
S2V = "#d97706"
V2S = "#b23a48"
PRODUCER = "#2563eb"
CONSUMER = "#7c3aed"
POOL = "#b7791f"
VOUCHER = "#1f78b4"
USD = "#d73027"
OTHER = "#6b7280"

FLOW_COLORS = {
    "voucher_to_voucher": V2V,
    "stable_to_voucher": S2V,
    "voucher_to_stable": V2S,
}


@dataclass
class NodeInfo:
    node_id: str
    label: str
    node_type: str
    seed: int = 0
    coupon: float = 0.0
    principal_ratio: float = 0.0


@dataclass
class FlowRecord:
    panel: str
    source: str
    source_label: str
    source_type: str
    pool: str
    pool_label: str
    destination: str
    destination_label: str
    destination_type: str
    motif: str
    usd_direction: str
    asset_in: str
    asset_out: str
    route_context: str
    route_source_role: str
    swap_count: int = 0
    volume_usd: float = 0.0
    displayed: int = 0

    @property
    def key(self) -> tuple[str, str, str, str, str, str, str, str]:
        return (
            self.panel,
            self.source,
            self.pool,
            self.destination,
            self.motif,
            self.asset_in,
            self.asset_out,
            self.route_context,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology-dir",
        type=Path,
        default=DEFAULT_TOPOLOGY_DIR,
        help="Directory containing topology_nodes.csv and topology_edges.csv.",
    )
    parser.add_argument(
        "--panel",
        choices=(PANEL_FAILED, PANEL_BASELINE),
        default=PANEL_FAILED,
        help="Panel to draw as the single figure.",
    )
    parser.add_argument(
        "--output-prefix",
        default="fig_topology_failed_single_with_usd_flows",
        help="Output filename prefix in --topology-dir.",
    )
    parser.add_argument(
        "--flow-mode",
        choices=("all", "v2v_usd_in"),
        default="all",
        help="Which source-use motifs to display. v2v_usd_in excludes voucher-to-USD-out paths.",
    )
    parser.add_argument("--path-limit", type=int, default=170, help="Maximum displayed source-pool-destination paths.")
    parser.add_argument("--write-converted", action="store_true", help="Also write PNG/PDF using Chrome or ImageMagick.")
    return parser.parse_args()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def safe_int(value: object, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    weight: str = "normal",
    anchor: str = "start",
    fill: str = TEXT,
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{esc(text)}</text>'
    )


def short_label(value: str, max_len: int = 18) -> str:
    text = str(value)
    for prefix in ("asset:VCHR:", "asset:", "producer:", "consumer:", "pool:", "VCHR:", "pool_", "agent_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("sys_clc", "sys CLC")
    if len(text) > max_len:
        return text[: max_len - 1] + "."
    return text


def money_short(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}k"
    return f"${value:.0f}"


def node_y_positions(order: list[str], top: float, bottom: float, *, usd_center: bool = False) -> dict[str, float]:
    if not order:
        return {}
    if usd_center and "asset:USD" in order and len(order) > 1:
        remaining = [node for node in order if node != "asset:USD"]
        positions = node_y_positions(remaining, top, bottom, usd_center=False)
        positions["asset:USD"] = top + 0.22 * (bottom - top)
        return positions
    if len(order) == 1:
        return {order[0]: 0.5 * (top + bottom)}
    span = max(1.0, bottom - top)
    return {node_id: top + span * idx / (len(order) - 1) for idx, node_id in enumerate(order)}


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


def draw_source_symbol(node_type: str, x: float, y: float) -> list[str]:
    if node_type == "pool":
        return draw_pool_symbol(x, y)
    if node_type == "consumer":
        return [
            f'<polygon points="{triangle_points(x, y, 7.0)}" fill="{CONSUMER}" fill-opacity="0.86" '
            'stroke="#1f2937" stroke-width="1.1"/>'
        ]
    side = 12.5
    return [
        f'<rect x="{x - side / 2:.2f}" y="{y - side / 2:.2f}" width="{side:.2f}" height="{side:.2f}" '
        f'fill="{PRODUCER}" fill-opacity="0.90" stroke="#1f2937" stroke-width="1.1"/>'
    ]


def draw_pool_symbol(x: float, y: float) -> list[str]:
    return [
        f'<polygon points="{star_points(x, y, 8.0)}" fill="{POOL}" fill-opacity="0.88" '
        'stroke="#1f2937" stroke-width="0.8"/>'
    ]


def draw_destination_symbol(destination_type: str, x: float, y: float) -> list[str]:
    if destination_type == "stable":
        return [
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="11.0" fill="#ffffff" stroke="{USD}" stroke-width="3.0"/>',
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.0" fill="none" stroke="{USD}" stroke-width="2.0"/>',
        ]
    if destination_type == "producer":
        return [
            f'<rect x="{x - 6.5:.2f}" y="{y - 6.5:.2f}" width="13" height="13" fill="#ffffff" '
            f'stroke="{VOUCHER}" stroke-width="2.0"/>'
        ]
    return [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7.0" fill="{OTHER}" fill-opacity="0.80"/>']


def horizontal_bar(
    *,
    x: float,
    y: float,
    width: float,
    label: str,
    value: float,
    color: str,
) -> list[str]:
    value = max(0.0, min(1.0, value))
    return [
        svg_text(x, y - 5, label, size=10, weight="bold", fill="#374151"),
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="8" fill="#e5e7eb" rx="2"/>',
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width * value:.2f}" height="8" fill="{color}" rx="2"/>',
        svg_text(x + width + 8, y + 8, f"{100.0 * value:.1f}%", size=10, fill="#374151"),
    ]


def load_nodes(path: Path, panel: str) -> dict[str, NodeInfo]:
    nodes: dict[str, NodeInfo] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["panel"] != panel:
                continue
            nodes[row["node_id"]] = NodeInfo(
                node_id=row["node_id"],
                label=row.get("label") or short_label(row["node_id"]),
                node_type=row.get("node_type") or "other",
                seed=safe_int(row.get("seed")),
                coupon=safe_float(row.get("coupon_target_annual")),
                principal_ratio=safe_float(row.get("principal_ratio")),
            )
    if not nodes:
        raise RuntimeError(f"No topology nodes found for panel {panel!r}.")
    return nodes


def producer_from_voucher(nodes: dict[str, NodeInfo], asset_id: str) -> tuple[str, str]:
    text = str(asset_id)
    if text.startswith("VCHR:agent_"):
        suffix = text.split("agent_", 1)[1]
        producer_id = f"producer:pool_{suffix}"
        if producer_id in nodes:
            return producer_id, nodes[producer_id].label
        return f"producer_issuer:agent_{suffix}", f"issuer {short_label('agent_' + suffix, 12)}"
    if text.startswith("VCHR:"):
        suffix = text.split(":", 1)[1]
        return f"producer_issuer:{suffix}", f"issuer {short_label(suffix, 12)}"
    return "producer_issuer:unknown", "issuer unknown"


def destination_for_edge(nodes: dict[str, NodeInfo], motif: str, asset_in: str, asset_out: str) -> tuple[str, str, str, str]:
    if motif == "voucher_to_stable" or asset_out == "USD":
        return ("asset:USD", "USD", "stable", "usd_out")
    if str(asset_out).startswith("VCHR:"):
        producer_id, label = producer_from_voucher(nodes, asset_out)
        return (producer_id, label, "producer", "usd_in" if asset_in == "USD" else "none")
    return (f"asset:{asset_out}", short_label(asset_out), "other", "none")


def build_flow_records(
    edges_path: Path,
    nodes: dict[str, NodeInfo],
    panel: str,
    *,
    flow_mode: str,
) -> list[FlowRecord]:
    included_motifs = {"voucher_to_voucher", "stable_to_voucher", "voucher_to_stable"}
    if flow_mode == "v2v_usd_in":
        included_motifs = {"voucher_to_voucher", "stable_to_voucher"}
    records: dict[tuple[str, str, str, str, str, str, str, str], FlowRecord] = {}
    with edges_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["panel"] != panel or row["edge_kind"] != "source_uses_pool":
                continue
            motif = row.get("motif_class") or "other"
            if motif not in included_motifs:
                continue
            source = row["source"]
            pool = row["target"]
            source_node = nodes.get(source)
            pool_node = nodes.get(pool)
            if source_node is None or pool_node is None:
                continue
            asset_in = row.get("asset_in", "")
            asset_out = row.get("asset_out", "")
            destination, destination_label, destination_type, usd_direction = destination_for_edge(
                nodes, motif, asset_in, asset_out
            )
            record = FlowRecord(
                panel=panel,
                source=source,
                source_label=source_node.label,
                source_type=source_node.node_type,
                pool=pool,
                pool_label=pool_node.label,
                destination=destination,
                destination_label=destination_label,
                destination_type=destination_type,
                motif=motif,
                usd_direction=usd_direction,
                asset_in=asset_in,
                asset_out=asset_out,
                route_context=row.get("route_context", ""),
                route_source_role=row.get("route_source_role", ""),
            )
            existing = records.get(record.key)
            if existing is None:
                records[record.key] = record
                existing = record
            existing.swap_count += safe_int(row.get("swap_count"))
            existing.volume_usd += safe_float(row.get("volume_usd"))
    return sorted(records.values(), key=lambda rec: rec.volume_usd, reverse=True)


def select_display_records(
    records: list[FlowRecord],
    *,
    source_limit: int = 38,
    pool_limit: int = 32,
    destination_limit: int = 42,
    path_limit: int = 170,
) -> dict[str, object]:
    for record in records:
        record.displayed = 0

    source_volume: Counter[str] = Counter()
    pool_volume: Counter[str] = Counter()
    destination_volume: Counter[str] = Counter()
    source_labels: dict[str, str] = {}
    pool_labels: dict[str, str] = {}
    destination_labels: dict[str, str] = {}
    source_types: dict[str, str] = {}
    destination_types: dict[str, str] = {}
    for record in records:
        source_volume[record.source] += record.volume_usd
        pool_volume[record.pool] += record.volume_usd
        destination_volume[record.destination] += record.volume_usd
        source_labels[record.source] = record.source_label
        pool_labels[record.pool] = record.pool_label
        destination_labels[record.destination] = record.destination_label
        source_types[record.source] = record.source_type
        destination_types[record.destination] = record.destination_type

    selected_sources = {node for node, _ in source_volume.most_common(source_limit)}
    selected_pools = {node for node, _ in pool_volume.most_common(pool_limit)}
    selected_destinations = {node for node, _ in destination_volume.most_common(destination_limit)}
    if any(record.destination == "asset:USD" for record in records):
        selected_destinations.add("asset:USD")

    protected: list[FlowRecord] = []
    per_motif = max(18, path_limit // 5)
    for motif in ("voucher_to_voucher", "stable_to_voucher", "voucher_to_stable"):
        if not any(record.motif == motif for record in records):
            continue
        for record in [rec for rec in records if rec.motif == motif][:per_motif]:
            selected_sources.add(record.source)
            selected_pools.add(record.pool)
            selected_destinations.add(record.destination)
            protected.append(record)

    candidates = [
        record
        for record in records
        if record.source in selected_sources
        and record.pool in selected_pools
        and record.destination in selected_destinations
    ]
    selected_keys = {record.key for record in protected}
    for record in candidates:
        if len(selected_keys) >= path_limit:
            break
        selected_keys.add(record.key)
    for record in records:
        record.displayed = 1 if record.key in selected_keys else 0

    source_order = [node for node, _ in source_volume.most_common() if node in selected_sources]
    pool_order = [node for node, _ in pool_volume.most_common() if node in selected_pools]
    destination_order = [node for node, _ in destination_volume.most_common() if node in selected_destinations]
    if "asset:USD" in destination_order:
        destination_order.remove("asset:USD")
        destination_order.insert(0, "asset:USD")
    return {
        "source_order": source_order,
        "pool_order": pool_order,
        "destination_order": destination_order,
        "source_labels": source_labels,
        "pool_labels": pool_labels,
        "destination_labels": destination_labels,
        "source_types": source_types,
        "destination_types": destination_types,
    }


def flow_metrics(records: list[FlowRecord]) -> dict[str, float]:
    volume_by_motif: Counter[str] = Counter()
    count_by_motif: Counter[str] = Counter()
    for record in records:
        volume_by_motif[record.motif] += record.volume_usd
        count_by_motif[record.motif] += record.swap_count
    total_volume = sum(volume_by_motif.values())
    total_count = sum(count_by_motif.values())
    stable_volume = volume_by_motif["stable_to_voucher"] + volume_by_motif["voucher_to_stable"]
    stable_count = count_by_motif["stable_to_voucher"] + count_by_motif["voucher_to_stable"]
    return {
        "total_volume": total_volume,
        "total_count": float(total_count),
        "v2v_volume": volume_by_motif["voucher_to_voucher"],
        "s2v_volume": volume_by_motif["stable_to_voucher"],
        "v2s_volume": volume_by_motif["voucher_to_stable"],
        "stable_volume": stable_volume,
        "v2v_share": volume_by_motif["voucher_to_voucher"] / total_volume if total_volume else 0.0,
        "s2v_share": volume_by_motif["stable_to_voucher"] / total_volume if total_volume else 0.0,
        "v2s_share": volume_by_motif["voucher_to_stable"] / total_volume if total_volume else 0.0,
        "stable_share": stable_volume / total_volume if total_volume else 0.0,
        "v2v_count_share": count_by_motif["voucher_to_voucher"] / total_count if total_count else 0.0,
        "stable_count_share": stable_count / total_count if total_count else 0.0,
    }


def draw_single_panel(
    records: list[FlowRecord],
    selection: dict[str, object],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    flow_mode: str,
) -> list[str]:
    metrics = flow_metrics(records)
    source_order = selection["source_order"]  # type: ignore[assignment]
    pool_order = selection["pool_order"]  # type: ignore[assignment]
    destination_order = selection["destination_order"]  # type: ignore[assignment]
    source_labels = selection["source_labels"]  # type: ignore[assignment]
    pool_labels = selection["pool_labels"]  # type: ignore[assignment]
    destination_labels = selection["destination_labels"]  # type: ignore[assignment]
    source_types = selection["source_types"]  # type: ignore[assignment]
    destination_types = selection["destination_types"]  # type: ignore[assignment]
    displayed = [record for record in records if record.displayed]
    active_sources = {record.source for record in displayed}
    active_pools = {record.pool for record in displayed}
    active_destinations = {record.destination for record in displayed}

    top = y + 155
    bottom = y + height - 42
    x_source = x + 110
    x_pool = x + width * 0.47
    x_destination = x + width - 126
    source_y = node_y_positions([node for node in source_order if node in active_sources], top, bottom)
    pool_y = node_y_positions([node for node in pool_order if node in active_pools], top, bottom)
    destination_y = node_y_positions(
        [node for node in destination_order if node in active_destinations],
        top,
        bottom,
        usd_center=True,
    )
    max_volume = max((record.volume_usd for record in displayed), default=1.0)

    elements = [
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="#ffffff" stroke="{BORDER}" stroke-width="1"/>',
        svg_text(x + 14, y + 25, title, size=17, weight="bold"),
        svg_text(
            x + 14,
            y + 49,
            (
                f"route volume {money_short(metrics['total_volume'])}; displayed paths {len(displayed)}; "
                "source actors -> pools -> output voucher issuers"
                if flow_mode == "v2v_usd_in"
                else f"route volume {money_short(metrics['total_volume'])}; displayed paths {len(displayed)}; source actors -> pools -> voucher issuers or USD"
            ),
            size=11,
            fill=MUTED,
        ),
        svg_text(x_source, y + 120, "source actor / pool", size=11, weight="bold", anchor="middle", fill="#374151"),
        svg_text(x_pool, y + 120, "pool", size=11, weight="bold", anchor="middle", fill="#374151"),
        svg_text(
            x_destination,
            y + 120,
            "output voucher issuer" if flow_mode == "v2v_usd_in" else "output voucher issuer / USD",
            size=11,
            weight="bold",
            anchor="middle",
            fill="#374151",
        ),
        f'<line x1="{x_source:.2f}" y1="{top - 10:.2f}" x2="{x_source:.2f}" y2="{bottom + 10:.2f}" stroke="{GRID}" stroke-width="1"/>',
        f'<line x1="{x_pool:.2f}" y1="{top - 10:.2f}" x2="{x_pool:.2f}" y2="{bottom + 10:.2f}" stroke="{GRID}" stroke-width="1"/>',
        f'<line x1="{x_destination:.2f}" y1="{top - 10:.2f}" x2="{x_destination:.2f}" y2="{bottom + 10:.2f}" stroke="{GRID}" stroke-width="1"/>',
    ]
    elements.extend(horizontal_bar(x=x + 14, y=y + 76, width=125, label="V2V volume share", value=metrics["v2v_share"], color=V2V))
    elements.extend(horizontal_bar(x=x + 220, y=y + 76, width=115, label="USD-in S2V share", value=metrics["s2v_share"], color=S2V))
    if flow_mode == "all":
        elements.extend(horizontal_bar(x=x + 405, y=y + 76, width=115, label="USD-out V2S share", value=metrics["v2s_share"], color=V2S))
        count_x = x + 600
    else:
        count_x = x + 405
    elements.append(
        svg_text(
            count_x,
            y + 84,
            f"USD-in count share {100.0 * metrics['stable_count_share']:.1f}%",
            size=10,
            fill="#374151",
        )
    )

    for record in sorted(displayed, key=lambda item: item.volume_usd):
        if record.source not in source_y or record.pool not in pool_y or record.destination not in destination_y:
            continue
        norm = math.sqrt(record.volume_usd / max(1e-9, max_volume))
        width_px = 0.50 + 5.6 * norm
        opacity = 0.12 + 0.54 * norm
        color = FLOW_COLORS.get(record.motif, OTHER)
        sy = source_y[record.source]
        py = pool_y[record.pool]
        dy = destination_y[record.destination]
        elements.append(
            f'<path d="M {x_source:.2f},{sy:.2f} C {x_source + 110:.2f},{sy:.2f} {x_pool - 110:.2f},{py:.2f} {x_pool:.2f},{py:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="{width_px:.2f}" stroke-opacity="{opacity:.3f}"/>'
        )
        elements.append(
            f'<path d="M {x_pool:.2f},{py:.2f} C {x_pool + 110:.2f},{py:.2f} {x_destination - 110:.2f},{dy:.2f} {x_destination:.2f},{dy:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="{width_px:.2f}" stroke-opacity="{opacity:.3f}"/>'
        )

    label_limit = 15
    for node_id, yy in source_y.items():
        elements.extend(draw_source_symbol(source_types.get(node_id, "producer"), x_source, yy))
    for idx, (node_id, yy) in enumerate(source_y.items()):
        if idx < label_limit:
            elements.append(svg_text(x_source + 12, yy + 3, short_label(source_labels.get(node_id, node_id), 14), size=7, fill="#374151"))
    for node_id, yy in pool_y.items():
        elements.extend(draw_pool_symbol(x_pool, yy))
    for idx, (node_id, yy) in enumerate(pool_y.items()):
        if idx < label_limit:
            elements.append(svg_text(x_pool + 12, yy + 3, short_label(pool_labels.get(node_id, node_id), 12), size=7, fill="#374151"))
    for node_id, yy in destination_y.items():
        elements.extend(draw_destination_symbol(destination_types.get(node_id, "producer"), x_destination, yy))
    for idx, (node_id, yy) in enumerate(destination_y.items()):
        if idx < label_limit or node_id == "asset:USD":
            elements.append(
                svg_text(
                    x_destination + 14,
                    yy + 3,
                    short_label(destination_labels.get(node_id, node_id), 15),
                    size=7 if node_id != "asset:USD" else 9,
                    weight="bold" if node_id == "asset:USD" else "normal",
                    fill="#374151",
                )
            )
    return elements


def draw_single_legend(x: float, y: float, *, flow_mode: str) -> list[str]:
    elements = [svg_text(x, y, "Legend", size=14, weight="bold")]
    yy = y + 26
    for label, kind in [
        ("producer source", "producer"),
        ("consumer source", "consumer"),
        ("pool source", "pool"),
        ("pool", "pool"),
        ("voucher issuer", "issuer"),
    ]:
        if kind == "pool":
            elements.extend(draw_pool_symbol(x + 10, yy - 5))
        elif kind == "issuer":
            elements.extend(draw_destination_symbol("producer", x + 10, yy - 5))
        else:
            elements.extend(draw_source_symbol(kind, x + 10, yy - 5))
        elements.append(svg_text(x + 28, yy, label, size=10, fill="#374151"))
        yy += 26
    if flow_mode == "all":
        elements.extend(draw_destination_symbol("stable", x + 10, yy - 5))
        elements.append(svg_text(x + 28, yy, "USD", size=10, fill="#374151"))
        yy += 34
    elements.append(svg_text(x, yy, "Route / USD flow", size=11, weight="bold", fill="#374151"))
    yy += 20
    flow_items = [
        ("V2V voucher path", V2V),
        ("USD in -> voucher", S2V),
    ]
    if flow_mode == "all":
        flow_items.append(("voucher -> USD out", V2S))
    for label, color in flow_items:
        elements.append(
            f'<path d="M {x:.2f},{yy:.2f} C {x + 22:.2f},{yy:.2f} {x + 38:.2f},{yy:.2f} {x + 60:.2f},{yy:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="4" stroke-opacity="0.70"/>'
        )
        elements.append(svg_text(x + 68, yy + 4, label, size=10, fill="#374151"))
        yy += 24
    yy += 6
    elements.append(svg_text(x, yy, "Reading", size=11, weight="bold", fill="#374151"))
    yy += 18
    reading_lines = ["USD-in paths are", "stable-to-voucher routes.", "No voucher-to-USD shown."]
    if flow_mode == "all":
        reading_lines = ["USD paths are source-use", "routes touching the USD", "asset, not extra reruns."]
    for line in reading_lines:
        elements.append(svg_text(x, yy, line, size=10, fill=MUTED))
        yy += 15
    return elements


def write_flow_rows(path: Path, records: list[FlowRecord]) -> None:
    rows = [
        {
            "panel": record.panel,
            "source": record.source,
            "source_label": record.source_label,
            "source_type": record.source_type,
            "pool": record.pool,
            "pool_label": record.pool_label,
            "destination": record.destination,
            "destination_label": record.destination_label,
            "destination_type": record.destination_type,
            "motif": record.motif,
            "usd_direction": record.usd_direction,
            "asset_in": record.asset_in,
            "asset_out": record.asset_out,
            "route_context": record.route_context,
            "route_source_role": record.route_source_role,
            "swap_count": record.swap_count,
            "volume_usd": f"{record.volume_usd:.9f}",
            "displayed": record.displayed,
        }
        for record in records
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_notes(path: Path, graph_meta: NodeInfo, records: list[FlowRecord], *, flow_mode: str) -> None:
    metrics = flow_metrics(records)
    displayed = [record for record in records if record.displayed]
    lines = [
        "# Single-Panel Topology With USD Flows",
        "",
        "Generated from the existing topology diagnostic CSVs; no simulation was rerun.",
        f"Flow mode: `{flow_mode}`.",
        (
            f"Panel: `{graph_meta.node_id}`; matched seed `{graph_meta.seed}`; "
            f"coupon `{graph_meta.coupon:.2f}`; principal ratio `{graph_meta.principal_ratio:.2f}`."
        ),
        f"Displayed source-pool-destination paths: `{len(displayed)}` of `{len(records)}` aggregated source-use records.",
        f"V2V volume share: `{metrics['v2v_share']:.9f}`.",
        f"USD-in stable-to-voucher volume share: `{metrics['s2v_share']:.9f}`.",
        f"USD-out voucher-to-stable volume share: `{metrics['v2s_share']:.9f}`.",
        "",
        (
            "USD-in flows are derived from source-use routes with `asset_in == USD`; voucher-to-USD routes are excluded."
            if flow_mode == "v2v_usd_in"
            else "USD flows are derived from source-use routes with `asset_in == USD` or `asset_out == USD`."
        ),
        "The figure is a display diagnostic; full trace CSVs remain the audit objects.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_svg(records: list[FlowRecord], selection: dict[str, object], panel: str, *, flow_mode: str) -> str:
    title = "Failed stress cell: 45% coupon, 2.00 principal ratio"
    subtitle = "Layered source -> pool -> voucher/USD routes from the matched-seed failed stress-cell trace."
    main_title = "Single-Panel Settlement Topology With USD Flows"
    footer = "Edge width is normalized within this single panel by displayed source-use route volume; USD-in and USD-out paths are shown explicitly."
    if flow_mode == "v2v_usd_in":
        main_title = "Single-Panel Settlement Topology With USD-In Flows"
        subtitle = "Layered V2V and USD-in -> voucher-out routes from the matched-seed failed stress-cell trace."
        footer = "Edge width is normalized within this single panel by displayed source-use route volume; voucher-to-USD-out paths are excluded."
    if panel == PANEL_BASELINE:
        title = "No-bond baseline, same seed"
        subtitle = "Layered source -> pool -> voucher/USD routes from the matched-seed no-bond baseline trace."
        if flow_mode == "v2v_usd_in":
            subtitle = "Layered V2V and USD-in -> voucher-out routes from the matched-seed no-bond baseline trace."
    width = 1800
    height = 930
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(48, 42, main_title, size=28, weight="bold"),
        svg_text(48, 68, subtitle, size=14, fill=MUTED),
    ]
    elements.extend(draw_single_panel(records, selection, x=48, y=100, width=1495, height=760, title=title, flow_mode=flow_mode))
    elements.extend(draw_single_legend(1582, 132, flow_mode=flow_mode))
    elements.append(
        svg_text(
            48,
            900,
            footer,
            size=12,
            fill=MUTED,
        )
    )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def write_converted(svg_path: Path, *, write_converted: bool) -> None:
    if not write_converted:
        return
    png_path = svg_path.with_suffix(".png")
    pdf_path = svg_path.with_suffix(".pdf")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is not None:
        uri = svg_path.resolve().as_uri()
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-crash-reporter",
                "--disable-crashpad",
                f"--screenshot={png_path}",
                "--window-size=1800,930",
                uri,
            ],
            check=True,
        )
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-crash-reporter",
                "--disable-crashpad",
                f"--print-to-pdf={pdf_path}",
                uri,
            ],
            check=True,
        )
        return
    converter = shutil.which("convert")
    if converter is None:
        print("[single-topology] no Chrome or ImageMagick converter found; wrote SVG only.")
        return
    subprocess.run(
        [converter, "-density", "180", str(svg_path), "-background", "white", "-alpha", "remove", str(png_path)],
        check=True,
    )
    subprocess.run(
        [converter, "-density", "180", str(svg_path), "-background", "white", "-alpha", "remove", str(pdf_path)],
        check=True,
    )


def main() -> int:
    args = parse_args()
    topology_dir = args.topology_dir.resolve()
    nodes = load_nodes(topology_dir / "topology_nodes.csv", args.panel)
    graph_meta = next(iter(nodes.values()))
    graph_meta.node_id = args.panel
    records = build_flow_records(topology_dir / "topology_edges.csv", nodes, args.panel, flow_mode=args.flow_mode)
    selection = select_display_records(records, path_limit=args.path_limit)

    svg_path = topology_dir / f"{args.output_prefix}.svg"
    svg_path.write_text(build_svg(records, selection, args.panel, flow_mode=args.flow_mode), encoding="utf-8")
    write_flow_rows(topology_dir / f"{args.output_prefix}_flows.csv", records)
    write_notes(topology_dir / f"{args.output_prefix}_notes.md", graph_meta, records, flow_mode=args.flow_mode)
    write_converted(svg_path, write_converted=bool(args.write_converted))
    print(f"[single-topology] wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
