#!/usr/bin/env python3
"""Generate a full-network k-cycle centrality diagnostic from topology CSVs.

This script does not rerun the simulator. It reads the matched-seed topology
trace CSVs, collapses all traced swap and asset-transfer legs into undirected
node-pair edges, computes exact short-cycle participation up to k=4, and writes
a raw SVG plus PNG/PDF siblings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
PANELS = (PANEL_BASELINE, PANEL_FAILED)

WIDTH = 1800
HEIGHT = 930

TEXT = "#111827"
MUTED = "#4b5563"
BORDER = "#d1d5db"
GRID = "#e5e7eb"
V2V = "#2c7fb8"
V2S = "#b23a48"
S2V = "#d97706"
STABLE_OTHER = "#ef4444"
OTHER = "#6b7280"

NODE_COLORS = {
    "producer": "#2563eb",
    "consumer": "#7c3aed",
    "pool": "#b7791f",
    "voucher": "#1f78b4",
    "stable": "#d73027",
    "other": "#6b7280",
}


@dataclass
class Node:
    panel: str
    node_id: str
    label: str
    node_type: str
    weighted_degree: float = 0.0
    volume_usd: float = 0.0
    swap_count: int = 0
    record_count: int = 0
    triangle_participation: float = 0.0
    four_cycle_participation: float = 0.0
    cycle_score: float = 0.0
    rank: int = 0


@dataclass
class PairEdge:
    panel: str
    source: str
    target: str
    spring_weight: float = 0.0
    volume_usd: float = 0.0
    swap_count: int = 0
    record_count: int = 0
    motif_weights: Counter[str] = field(default_factory=Counter)
    edge_kind_counts: Counter[str] = field(default_factory=Counter)

    @property
    def dominant_motif(self) -> str:
        if not self.motif_weights:
            return "other"
        return self.motif_weights.most_common(1)[0][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology-dir",
        type=Path,
        default=DEFAULT_TOPOLOGY_DIR,
        help="Directory containing topology_nodes.csv and topology_edges.csv.",
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=(3, 4),
        default=4,
        help="Compute exact simple-cycle participation for lengths 3..k. k=4 is the default.",
    )
    parser.add_argument(
        "--output-prefix",
        default="fig_kcycle_centrality_network",
        help="Output filename prefix in --topology-dir.",
    )
    parser.add_argument(
        "--centrality-prefix",
        default=None,
        help="Output prefix for the centrality CSV and notes. Defaults to kcycle_centrality_nodes/notes for the original figure.",
    )
    parser.add_argument(
        "--exclude-node",
        action="append",
        default=[],
        help="Node id to remove before computing centrality and layout. May be repeated.",
    )
    parser.add_argument(
        "--exclude-usd",
        action="store_true",
        help="Remove the USD asset node and all USD-touching transfer legs before recomputing centrality.",
    )
    parser.add_argument(
        "--layout",
        choices=("radial", "spring"),
        default="radial",
        help="Layout mode. spring uses deterministic spring attraction with type-cluster anchors.",
    )
    parser.add_argument(
        "--write-converted",
        action="store_true",
        help="Also write PNG/PDF with ImageMagick. Dense all-edge SVG conversion can be slow.",
    )
    parser.add_argument("--svg-only", action="store_true", help="Deprecated alias for the default SVG-only behavior.")
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


def label_box(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 8,
    anchor: str = "start",
    fill: str = "#374151",
) -> list[str]:
    width = max(28.0, len(text) * size * 0.56 + 8.0)
    height = size + 7.0
    rx = x - width / 2 if anchor == "middle" else x - 3
    tx = x if anchor == "middle" else x + 1
    return [
        f'<rect x="{rx:.2f}" y="{y - size - 4:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="#ffffff" fill-opacity="0.88" stroke="#ffffff" stroke-width="0.5" rx="2"/>',
        svg_text(tx, y, text, size=size, anchor=anchor, fill=fill),
    ]


def marker_defs() -> str:
    return """
<defs>
  <marker id="arrow-edge" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
  </marker>
</defs>""".strip()


def triangle_points(cx: float, cy: float, radius: float) -> str:
    return " ".join(
        f"{cx + math.cos(-math.pi / 2 + i * 2 * math.pi / 3) * radius:.2f},"
        f"{cy + math.sin(-math.pi / 2 + i * 2 * math.pi / 3) * radius:.2f}"
        for i in range(3)
    )


def star_points(cx: float, cy: float, radius: float) -> str:
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.44
        points.append(f"{cx + math.cos(angle) * r:.2f},{cy + math.sin(angle) * r:.2f}")
    return " ".join(points)


def draw_node_symbol(node_type: str, x: float, y: float, radius: float, *, opacity: float = 0.92) -> list[str]:
    color = NODE_COLORS.get(node_type, NODE_COLORS["other"])
    stroke = "#1f2937"
    if node_type == "stable":
        return [
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="#ffffff" stroke="{color}" stroke-width="2.4" opacity="{opacity:.3f}"/>',
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{max(2.0, radius - 4):.2f}" fill="none" stroke="{color}" stroke-width="1.6" opacity="{opacity:.3f}"/>',
        ]
    if node_type == "producer":
        side = radius * 1.78
        return [
            f'<rect x="{x - side / 2:.2f}" y="{y - side / 2:.2f}" width="{side:.2f}" height="{side:.2f}" '
            f'fill="{color}" fill-opacity="{opacity:.3f}" stroke="{stroke}" stroke-width="0.7"/>'
        ]
    if node_type == "consumer":
        return [
            f'<polygon points="{triangle_points(x, y, radius)}" fill="{color}" fill-opacity="{opacity:.3f}" '
            f'stroke="{stroke}" stroke-width="0.7"/>'
        ]
    if node_type == "pool":
        return [
            f'<polygon points="{star_points(x, y, radius)}" fill="{color}" fill-opacity="{opacity:.3f}" '
            f'stroke="{stroke}" stroke-width="0.55"/>'
        ]
    if node_type == "voucher":
        return [
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="{opacity:.3f}" '
            f'stroke="{stroke}" stroke-width="0.45"/>'
        ]
    return [
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="{opacity:.3f}" '
        f'stroke="{stroke}" stroke-width="0.45"/>'
    ]


def motif_color(motif: str) -> str:
    return {
        "voucher_to_voucher": V2V,
        "voucher_to_stable": V2S,
        "stable_to_voucher": S2V,
        "stable_involved_other": STABLE_OTHER,
    }.get(motif, OTHER)


def horizontal_bar(
    *,
    x: float,
    y: float,
    width: float,
    label: str,
    value: float,
    color: str,
    value_text: str,
) -> list[str]:
    value = max(0.0, min(1.0, value))
    return [
        svg_text(x, y - 5, label, size=10, weight="bold", fill="#374151"),
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="8" fill="#e5e7eb" rx="2"/>',
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width * value:.2f}" height="8" fill="{color}" rx="2"/>',
        svg_text(x + width + 8, y + 8, value_text, size=10, fill="#374151"),
    ]


def load_nodes(path: Path) -> dict[str, dict[str, Node]]:
    nodes: dict[str, dict[str, Node]] = {panel: {} for panel in PANELS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            panel = row["panel"]
            if panel not in nodes:
                continue
            node = Node(
                panel=panel,
                node_id=row["node_id"],
                label=row.get("label") or short_label(row["node_id"]),
                node_type=row.get("node_type") or "other",
                weighted_degree=safe_float(row.get("weighted_degree")),
                volume_usd=safe_float(row.get("volume_usd")),
                swap_count=safe_int(row.get("swap_count")),
            )
            nodes[panel][node.node_id] = node
    return nodes


def load_pair_edges(path: Path, nodes: dict[str, dict[str, Node]]) -> dict[str, dict[tuple[str, str], PairEdge]]:
    pair_edges: dict[str, dict[tuple[str, str], PairEdge]] = {panel: {} for panel in PANELS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            panel = row["panel"]
            if panel not in pair_edges:
                continue
            source = row["source"]
            target = row["target"]
            if source == target:
                continue
            if source not in nodes[panel]:
                nodes[panel][source] = Node(panel, source, short_label(source), row.get("source_type") or "other")
            if target not in nodes[panel]:
                nodes[panel][target] = Node(panel, target, short_label(target), row.get("target_type") or "other")
            a, b = sorted((source, target))
            edge = pair_edges[panel].get((a, b))
            if edge is None:
                edge = PairEdge(panel=panel, source=a, target=b)
                pair_edges[panel][(a, b)] = edge
            spring = safe_float(row.get("spring_weight"))
            edge.spring_weight += spring
            edge.volume_usd += safe_float(row.get("volume_usd"))
            edge.swap_count += safe_int(row.get("swap_count"))
            edge.record_count += 1
            edge.motif_weights[row.get("motif_class") or "other"] += spring
            edge.edge_kind_counts[row.get("edge_kind") or "unknown"] += safe_int(row.get("swap_count"), 1)
    return pair_edges


def apply_node_exclusions(
    nodes_by_panel: dict[str, dict[str, Node]],
    edges_by_panel: dict[str, dict[tuple[str, str], PairEdge]],
    excluded_nodes: set[str],
) -> tuple[dict[str, dict[str, Node]], dict[str, dict[tuple[str, str], PairEdge]]]:
    if not excluded_nodes:
        return nodes_by_panel, edges_by_panel
    filtered_nodes: dict[str, dict[str, Node]] = {}
    filtered_edges: dict[str, dict[tuple[str, str], PairEdge]] = {}
    for panel in PANELS:
        filtered_nodes[panel] = {
            node_id: node for node_id, node in nodes_by_panel[panel].items() if node_id not in excluded_nodes
        }
        filtered_edges[panel] = {
            pair: edge
            for pair, edge in edges_by_panel[panel].items()
            if edge.source not in excluded_nodes and edge.target not in excluded_nodes
        }
        recompute_node_activity(filtered_nodes[panel], filtered_edges[panel])
    return filtered_nodes, filtered_edges


def recompute_node_activity(nodes: dict[str, Node], pair_edges: dict[tuple[str, str], PairEdge]) -> None:
    for node in nodes.values():
        node.weighted_degree = 0.0
        node.volume_usd = 0.0
        node.swap_count = 0
        node.record_count = 0
        node.triangle_participation = 0.0
        node.four_cycle_participation = 0.0
        node.cycle_score = 0.0
        node.rank = 0
    for edge in pair_edges.values():
        if edge.source not in nodes or edge.target not in nodes:
            continue
        for node_id in (edge.source, edge.target):
            nodes[node_id].weighted_degree += edge.spring_weight
            nodes[node_id].volume_usd += edge.volume_usd
            nodes[node_id].swap_count += edge.swap_count
            nodes[node_id].record_count += edge.record_count


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            panel = row.get("panel", "")
            out[panel] = {key: safe_float(value) for key, value in row.items() if key != "panel"}
    return out


def adjacency_from_edges(pair_edges: dict[tuple[str, str], PairEdge]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in pair_edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def compute_cycle_centrality(
    nodes: dict[str, Node],
    pair_edges: dict[tuple[str, str], PairEdge],
    *,
    k: int,
) -> dict[str, float]:
    adj = adjacency_from_edges(pair_edges)
    triangle_counts: dict[str, float] = defaultdict(float)
    four_counts: dict[str, float] = defaultdict(float)

    if k >= 3:
        ordered = sorted(adj, key=lambda node_id: (len(adj[node_id]), node_id))
        rank = {node_id: idx for idx, node_id in enumerate(ordered)}
        forward = {node_id: {n for n in adj[node_id] if rank[node_id] < rank[n]} for node_id in adj}
        for u in ordered:
            fwd_u = forward[u]
            for v in fwd_u:
                common = fwd_u.intersection(forward[v])
                for w in common:
                    triangle_counts[u] += 1.0
                    triangle_counts[v] += 1.0
                    triangle_counts[w] += 1.0

    if k >= 4:
        pair_common_counts: dict[tuple[str, str], int] = defaultdict(int)
        for center, neighbors in adj.items():
            ordered_neighbors = sorted(neighbors)
            for i, u in enumerate(ordered_neighbors):
                for v in ordered_neighbors[i + 1 :]:
                    pair_common_counts[(u, v)] += 1
        for center, neighbors in adj.items():
            ordered_neighbors = sorted(neighbors)
            for i, u in enumerate(ordered_neighbors):
                for v in ordered_neighbors[i + 1 :]:
                    other_common = pair_common_counts.get((u, v), 0) - 1
                    if other_common > 0:
                        four_counts[center] += float(other_common)

    scores: dict[str, float] = {}
    for node_id, node in nodes.items():
        node.triangle_participation = triangle_counts.get(node_id, 0.0)
        node.four_cycle_participation = four_counts.get(node_id, 0.0)
        node.cycle_score = node.triangle_participation + node.four_cycle_participation
        scores[node_id] = node.cycle_score

    ranked = sorted(nodes.values(), key=lambda node: (-node.cycle_score, -node.weighted_degree, node.node_id))
    for idx, node in enumerate(ranked, start=1):
        node.rank = idx
    return scores


def stable_hash_fraction(text: str) -> float:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def layout_nodes(
    nodes: dict[str, Node],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    global_max_score: float,
) -> dict[str, tuple[float, float]]:
    cx = x + width / 2
    cy = y + height / 2 + 38
    max_log = math.log1p(max(1.0, global_max_score))
    bands = {
        "stable": (0.0, 0.0),
        "pool": (82.0, 178.0),
        "voucher": (185.0, 300.0),
        "producer": (285.0, 360.0),
        "consumer": (300.0, 372.0),
        "other": (235.0, 340.0),
    }
    positions: dict[str, tuple[float, float]] = {}
    for node in nodes.values():
        norm = math.log1p(max(0.0, node.cycle_score)) / max_log if max_log else 0.0
        inner, outer = bands.get(node.node_type, bands["other"])
        if node.node_type == "stable":
            radius = 0.0
        else:
            radius = inner + (outer - inner) * (1.0 - norm)
            radius += (stable_hash_fraction(node.node_id + ":r") - 0.5) * 16.0
        angle = 2.0 * math.pi * stable_hash_fraction(node.node_id)
        if node.node_type == "producer":
            angle = math.pi * (0.62 + 0.76 * stable_hash_fraction(node.node_id))
        elif node.node_type == "consumer":
            angle = math.pi * (1.62 + 0.58 * stable_hash_fraction(node.node_id))
        elif node.node_type == "pool":
            angle = 2.0 * math.pi * stable_hash_fraction("pool:" + node.node_id)
        elif node.node_type == "voucher":
            angle = 2.0 * math.pi * stable_hash_fraction("voucher:" + node.node_id)
        sx = cx + math.cos(angle) * radius * 1.05
        sy = cy + math.sin(angle) * radius * 0.82
        positions[node.node_id] = (sx, sy)
    return positions


def panel_edge_metrics(pair_edges: dict[tuple[str, str], PairEdge]) -> dict[str, float]:
    total = sum(edge.spring_weight for edge in pair_edges.values())
    v2v = sum(edge.motif_weights.get("voucher_to_voucher", 0.0) for edge in pair_edges.values())
    stable = sum(
        edge.motif_weights.get(motif, 0.0)
        for edge in pair_edges.values()
        for motif in ("voucher_to_stable", "stable_to_voucher", "stable_involved_other")
    )
    return {
        "v2v_edge_weight_share": v2v / total if total > 0.0 else 0.0,
        "stable_involved_edge_weight_share": stable / total if total > 0.0 else 0.0,
    }


def spring_cluster_layout(
    nodes: dict[str, Node],
    pair_edges: dict[tuple[str, str], PairEdge],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    global_max_score: float,
    seed: int,
    iterations: int = 180,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    node_ids = sorted(nodes)
    if not node_ids:
        return {}
    max_log_score = math.log1p(max(1.0, global_max_score))
    max_weight = max((edge.spring_weight for edge in pair_edges.values()), default=1.0)
    anchors = {
        "pool": (0.00, 0.00),
        "voucher": (0.10, 0.10),
        "producer": (-0.78, -0.12),
        "consumer": (0.78, 0.12),
        "stable": (0.00, -0.42),
        "other": (0.00, 0.58),
    }
    positions: dict[str, list[float]] = {}
    for node_id in node_ids:
        node = nodes[node_id]
        norm = math.log1p(max(0.0, node.cycle_score)) / max_log_score if max_log_score else 0.0
        ax, ay = anchors.get(node.node_type, anchors["other"])
        # High-cycle nodes are allowed to move toward the center; low-cycle
        # nodes stay closer to their type cluster.
        ax *= 1.0 - 0.55 * norm
        ay *= 1.0 - 0.55 * norm
        spread = 0.40 if node.node_type in {"producer", "consumer", "voucher"} else 0.26
        positions[node_id] = [
            ax + (stable_hash_fraction(node_id + ":x") - 0.5) * spread + rng.uniform(-0.012, 0.012),
            ay + (stable_hash_fraction(node_id + ":y") - 0.5) * spread + rng.uniform(-0.012, 0.012),
        ]

    edge_items = [
        (edge.source, edge.target, max(0.0, edge.spring_weight), edge.dominant_motif)
        for edge in pair_edges.values()
        if edge.source in positions and edge.target in positions
    ]

    for step in range(iterations):
        disp = {node_id: [0.0, 0.0] for node_id in node_ids}
        cooling = 1.0 - step / max(1, iterations)
        temperature = 0.030 * cooling + 0.003

        for source, target, weight, motif in edge_items:
            ax, ay = positions[source]
            bx, by = positions[target]
            dx = bx - ax
            dy = by - ay
            dist = math.hypot(dx, dy) + 1e-6
            w = math.sqrt(weight / max(1e-9, max_weight))
            target_len = 0.050 + 0.190 * (1.0 - w)
            if motif == "voucher_to_voucher":
                target_len *= 0.82
            force = (dist - target_len) * (0.016 + 0.042 * w)
            fx = (dx / dist) * force
            fy = (dy / dist) * force
            disp[source][0] += fx
            disp[source][1] += fy
            disp[target][0] -= fx
            disp[target][1] -= fy

        # Local-only repulsion prevents complete overplotting without the O(n^2)
        # cost of a full Fruchterman-Reingold pass.
        cell_size = 0.090
        grid: dict[tuple[int, int], list[str]] = defaultdict(list)
        for node_id, (px, py) in positions.items():
            grid[(math.floor(px / cell_size), math.floor(py / cell_size))].append(node_id)
        for (gx, gy), bucket in grid.items():
            nearby: list[str] = []
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    nearby.extend(grid.get((gx + ox, gy + oy), []))
            bucket = sorted(bucket)
            nearby = sorted(nearby)
            for a in bucket:
                ax, ay = positions[a]
                for b in nearby:
                    if a >= b:
                        continue
                    bx, by = positions[b]
                    dx = ax - bx
                    dy = ay - by
                    dist = math.hypot(dx, dy) + 1e-6
                    if dist > 0.16:
                        continue
                    force = 0.00085 / (dist * dist)
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    disp[a][0] += fx
                    disp[a][1] += fy
                    disp[b][0] -= fx
                    disp[b][1] -= fy

        for node_id in node_ids:
            node = nodes[node_id]
            norm = math.log1p(max(0.0, node.cycle_score)) / max_log_score if max_log_score else 0.0
            ax, ay = anchors.get(node.node_type, anchors["other"])
            ax *= 1.0 - 0.55 * norm
            ay *= 1.0 - 0.55 * norm
            px, py = positions[node_id]
            anchor_strength = 0.010 + 0.020 * (1.0 - norm)
            disp[node_id][0] += (ax - px) * anchor_strength
            disp[node_id][1] += (ay - py) * anchor_strength
            dx, dy = disp[node_id]
            length = math.hypot(dx, dy)
            if length > 0.0:
                positions[node_id][0] += (dx / length) * min(length, temperature)
                positions[node_id][1] += (dy / length) * min(length, temperature)
            positions[node_id][0] = max(-1.25, min(1.25, positions[node_id][0]))
            positions[node_id][1] = max(-1.05, min(1.05, positions[node_id][1]))

    cx = x + width / 2
    cy = y + height / 2 + 42
    sx = width * 0.39
    sy = height * 0.38
    return {node_id: (cx + px * sx, cy + py * sy) for node_id, (px, py) in positions.items()}


def edge_draw_order(edge: PairEdge) -> tuple[int, float]:
    motif = edge.dominant_motif
    layer = 2 if motif == "voucher_to_voucher" else 1
    return (layer, edge.spring_weight)


def draw_panel(
    panel: str,
    nodes: dict[str, Node],
    pair_edges: dict[tuple[str, str], PairEdge],
    summary: dict[str, float],
    *,
    k: int,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    global_max_score: float,
    global_max_edge_weight: float,
    layout: str,
    excluded_nodes: set[str],
) -> list[str]:
    if layout == "spring":
        positions = spring_cluster_layout(
            nodes,
            pair_edges,
            x=x,
            y=y,
            width=width,
            height=height,
            global_max_score=global_max_score,
            seed=7919 if panel == PANEL_BASELINE else 104729,
        )
    else:
        positions = layout_nodes(nodes, x=x, y=y, width=width, height=height, global_max_score=global_max_score)
    cycle_active = sum(1 for node in nodes.values() if node.cycle_score > 0)
    total_score = sum(node.cycle_score for node in nodes.values())
    top_n = max(1, math.ceil(0.01 * len(nodes)))
    top_score = sum(node.cycle_score for node in sorted(nodes.values(), key=lambda n: n.cycle_score, reverse=True)[:top_n])
    edge_count = len(pair_edges)
    record_count = sum(edge.record_count for edge in pair_edges.values())
    event_leg_count = sum(edge.edge_kind_counts.total() for edge in pair_edges.values())
    metrics = panel_edge_metrics(pair_edges)
    exclusion_text = "; USD asset removed" if "asset:USD" in excluded_nodes else ""

    elements = [
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="#ffffff" stroke="{BORDER}" stroke-width="1"/>',
        svg_text(x + 14, y + 25, title, size=17, weight="bold"),
        svg_text(
            x + 14,
            y + 49,
            f"Nodes {len(nodes):,}; pair edges {edge_count:,}; trace rows {record_count:,}; event legs {event_leg_count:,}; cycles 3..{k}{exclusion_text}",
            size=11,
            fill=MUTED,
        ),
    ]
    elements.extend(
        horizontal_bar(
            x=x + 14,
            y=y + 74,
            width=130,
            label="cycle-active nodes",
            value=cycle_active / max(1, len(nodes)),
            color=V2V,
            value_text=f"{cycle_active:,}/{len(nodes):,}",
        )
    )
    elements.extend(
        horizontal_bar(
            x=x + 220,
            y=y + 74,
            width=110,
            label="top 1% cycle mass",
            value=top_score / max(1.0, total_score),
            color=S2V,
            value_text=f"{100.0 * top_score / max(1.0, total_score):.1f}%",
        )
    )
    elements.extend(
        horizontal_bar(
            x=x + 390,
            y=y + 74,
            width=110,
            label="stable-involved share",
            value=metrics.get("stable_involved_edge_weight_share", 0.0),
            color=V2S,
            value_text=f"{100.0 * metrics.get('stable_involved_edge_weight_share', 0.0):.1f}%",
        )
    )
    elements.append(
        svg_text(
            x + 570,
            y + 82,
            f"V2V edge-weight share {100.0 * metrics.get('v2v_edge_weight_share', 0.0):.1f}%",
            size=10,
            fill="#374151",
        )
    )

    graph_top = y + 122
    graph_bottom = y + height - 44
    graph_left = x + 36
    graph_right = x + width - 36
    elements.extend(
        [
            f'<line x1="{graph_left:.2f}" y1="{(graph_top + graph_bottom) / 2:.2f}" x2="{graph_right:.2f}" y2="{(graph_top + graph_bottom) / 2:.2f}" stroke="{GRID}" stroke-width="1"/>',
            f'<line x1="{(graph_left + graph_right) / 2:.2f}" y1="{graph_top:.2f}" x2="{(graph_left + graph_right) / 2:.2f}" y2="{graph_bottom:.2f}" stroke="{GRID}" stroke-width="1"/>',
        ]
    )

    for edge in sorted(pair_edges.values(), key=edge_draw_order):
        if edge.source not in positions or edge.target not in positions:
            continue
        ax, ay = positions[edge.source]
        bx, by = positions[edge.target]
        norm = math.sqrt(max(0.0, edge.spring_weight) / max(1e-9, global_max_edge_weight))
        stroke_width = 0.10 + 1.65 * norm
        opacity = 0.018 + 0.180 * norm
        color = motif_color(edge.dominant_motif)
        elements.append(
            f'<line x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" y2="{by:.2f}" '
            f'stroke="{color}" stroke-width="{stroke_width:.3f}" stroke-opacity="{opacity:.3f}" stroke-linecap="round"/>'
        )

    max_log = math.log1p(max(1.0, global_max_score))
    top_label_nodes = sorted(nodes.values(), key=lambda node: (-node.cycle_score, -node.weighted_degree, node.node_id))[:14]
    top_label_ids = {node.node_id for node in top_label_nodes}
    top_halo_ids = {
        node.node_id
        for node in sorted(nodes.values(), key=lambda item: item.cycle_score, reverse=True)[: max(10, top_n)]
        if node.cycle_score > 0
    }
    draw_order = {"voucher": 0, "producer": 1, "consumer": 2, "pool": 3, "stable": 4}
    for node in sorted(nodes.values(), key=lambda item: (draw_order.get(item.node_type, 0), item.cycle_score)):
        if node.node_id not in positions:
            continue
        nx, ny = positions[node.node_id]
        norm = math.log1p(max(0.0, node.cycle_score)) / max_log if max_log else 0.0
        if node.node_type == "stable":
            radius = 7.0 + 13.0 * norm
        elif node.node_type == "pool":
            radius = 2.3 + 9.5 * norm
        elif node.node_type == "voucher":
            radius = 1.4 + 7.2 * norm
        else:
            radius = 1.3 + 6.2 * norm
        opacity = 0.26 + 0.70 * norm if node.cycle_score > 0 else 0.18
        if node.node_id in top_halo_ids:
            elements.append(
                f'<circle cx="{nx:.2f}" cy="{ny:.2f}" r="{radius + 5.5:.2f}" fill="none" '
                f'stroke="{S2V}" stroke-width="1.2" stroke-opacity="0.45"/>'
            )
        elements.extend(draw_node_symbol(node.node_type, nx, ny, radius, opacity=opacity))
        if node.node_id in top_label_ids:
            elements.extend(label_box(nx + radius + 4, ny + 3, short_label(node.label or node.node_id, 16), size=7))
    return elements


def draw_legend(x: float, y: float, *, k: int, excluded_nodes: set[str], layout: str) -> list[str]:
    elements = [svg_text(x, y, "Legend", size=14, weight="bold")]
    yy = y + 26
    node_items = [
        ("producer", "producer"),
        ("consumer", "consumer"),
        ("pool", "pool"),
        ("voucher", "voucher asset"),
    ]
    if "asset:USD" not in excluded_nodes:
        node_items.append(("stable", "stable asset"))
    for node_type, label in node_items:
        elements.extend(draw_node_symbol(node_type, x + 10, yy - 5, 7.0))
        elements.append(svg_text(x + 28, yy, label, size=10, fill="#374151"))
        yy += 26
    if "asset:USD" in excluded_nodes:
        elements.append(svg_text(x, yy, "USD asset node removed", size=10, weight="bold", fill=V2S))
        yy += 24
    yy += 8
    elements.append(svg_text(x, yy, "Dominant edge motif", size=11, weight="bold", fill="#374151"))
    yy += 20
    for label, color in [
        ("V2V", V2V),
        ("V2S", V2S),
        ("S2V", S2V),
        ("stable-other", STABLE_OTHER),
    ]:
        elements.append(
            f'<line x1="{x:.2f}" y1="{yy - 4:.2f}" x2="{x + 60:.2f}" y2="{yy - 4:.2f}" '
            f'stroke="{color}" stroke-width="3.5" stroke-opacity="0.70"/>'
        )
        elements.append(svg_text(x + 72, yy, label, size=10, fill="#374151"))
        yy += 22
    yy += 10
    elements.append(svg_text(x, yy, "k-cycle centrality", size=11, weight="bold", fill="#374151"))
    yy += 20
    elements.append(f'<circle cx="{x + 8:.2f}" cy="{yy - 4:.2f}" r="3" fill="{NODE_COLORS["pool"]}" opacity="0.45"/>')
    elements.append(f'<circle cx="{x + 38:.2f}" cy="{yy - 4:.2f}" r="9" fill="{NODE_COLORS["pool"]}" opacity="0.90"/>')
    elements.append(
        f'<circle cx="{x + 38:.2f}" cy="{yy - 4:.2f}" r="14" fill="none" stroke="{S2V}" stroke-width="1.2" stroke-opacity="0.45"/>'
    )
    elements.append(svg_text(x + 60, yy, f"node radius/halo = cycles 3..{k}", size=10, fill="#374151"))
    yy += 28
    elements.append(svg_text(x, yy, "Layout", size=11, weight="bold", fill="#374151"))
    yy += 18
    layout_text = "spring + type-cluster anchors" if layout == "spring" else "deterministic radial clusters"
    elements.append(svg_text(x, yy, layout_text, size=10, fill=MUTED))
    yy += 24
    elements.append(svg_text(x, yy, "Interpretation", size=11, weight="bold", fill="#374151"))
    yy += 18
    for line in [
        "High centrality marks nodes",
        "embedded in many short",
        "closed settlement loops.",
        "Edges aggregate all traced",
        "swap and asset-transfer legs.",
    ]:
        elements.append(svg_text(x, yy, line, size=10, fill=MUTED))
        yy += 15
    return elements


def build_svg(
    nodes_by_panel: dict[str, dict[str, Node]],
    edges_by_panel: dict[str, dict[tuple[str, str], PairEdge]],
    summary_by_panel: dict[str, dict[str, float]],
    *,
    k: int,
    layout: str,
    excluded_nodes: set[str],
) -> str:
    global_max_score = max((node.cycle_score for nodes in nodes_by_panel.values() for node in nodes.values()), default=1.0)
    global_max_edge_weight = max(
        (edge.spring_weight for edges in edges_by_panel.values() for edge in edges.values()),
        default=1.0,
    )
    title = "Full-Network k-Cycle Centrality Diagnostic"
    subtitle = "Entire matched-seed swap/transfer graph; node radius highlights participation in short closed settlement cycles."
    if "asset:USD" in excluded_nodes and layout == "spring":
        title = "No-USD Spring k-Cycle Centrality Diagnostic"
        subtitle = (
            "USD asset node and USD-touching transfer legs removed; spring clusters show remaining short settlement loops."
        )
    elif "asset:USD" in excluded_nodes:
        title = "No-USD k-Cycle Centrality Diagnostic"
        subtitle = "USD asset node and USD-touching transfer legs removed before recomputing short-cycle centrality."
    elif layout == "spring":
        title = "Spring k-Cycle Centrality Diagnostic"
        subtitle = "Entire matched-seed swap/transfer graph in a spring layout with type-cluster anchors."

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        marker_defs(),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(48, 42, title, size=28, weight="bold"),
        svg_text(
            48,
            68,
            subtitle,
            size=14,
            fill=MUTED,
        ),
    ]
    elements.extend(
        draw_panel(
            PANEL_BASELINE,
            nodes_by_panel[PANEL_BASELINE],
            edges_by_panel[PANEL_BASELINE],
            summary_by_panel.get(PANEL_BASELINE, {}),
            k=k,
            x=48,
            y=100,
            width=780,
            height=760,
            title="No-bond baseline, same seed",
            global_max_score=global_max_score,
            global_max_edge_weight=global_max_edge_weight,
            layout=layout,
            excluded_nodes=excluded_nodes,
        )
    )
    elements.extend(
        draw_panel(
            PANEL_FAILED,
            nodes_by_panel[PANEL_FAILED],
            edges_by_panel[PANEL_FAILED],
            summary_by_panel.get(PANEL_FAILED, {}),
            k=k,
            x=880,
            y=100,
            width=780,
            height=760,
            title="Failed stress cell: 45% coupon, 2.00 principal ratio",
            global_max_score=global_max_score,
            global_max_edge_weight=global_max_edge_weight,
            layout=layout,
            excluded_nodes=excluded_nodes,
        )
    )
    elements.extend(draw_legend(1688, 132, k=k, excluded_nodes=excluded_nodes, layout=layout))
    footer = "Cycle centrality is exact for simple cycles of length 3..k on the collapsed undirected trace graph. Layout is diagnostic, not geographic or causal proof."
    if "asset:USD" in excluded_nodes:
        footer = (
            "USD asset node and incident transfer legs are excluded; remaining stable-involved motifs are swap-context labels, not a visible USD node."
        )
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


def write_node_centrality(path: Path, nodes_by_panel: dict[str, dict[str, Node]], *, k: int) -> None:
    rows: list[dict[str, object]] = []
    for panel in PANELS:
        for node in sorted(nodes_by_panel[panel].values(), key=lambda n: (n.panel, n.rank, n.node_id)):
            rows.append(
                {
                    "panel": panel,
                    "k": k,
                    "rank": node.rank,
                    "node_id": node.node_id,
                    "label": node.label,
                    "node_type": node.node_type,
                    "cycle_score": f"{node.cycle_score:.9f}",
                    "triangle_participation": f"{node.triangle_participation:.9f}",
                    "four_cycle_participation": f"{node.four_cycle_participation:.9f}",
                    "weighted_degree": f"{node.weighted_degree:.9f}",
                    "volume_usd": f"{node.volume_usd:.9f}",
                    "swap_count": node.swap_count,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_notes(
    path: Path,
    nodes_by_panel: dict[str, dict[str, Node]],
    *,
    k: int,
    layout: str,
    excluded_nodes: set[str],
) -> None:
    exclusion_line = (
        "The USD asset node (`asset:USD`) and all incident transfer legs are removed before recomputing centrality."
        if "asset:USD" in excluded_nodes
        else "No nodes are excluded before computing centrality."
    )
    lines = [
        "# Full-Network k-Cycle Centrality Diagnostic",
        "",
        "This diagnostic is generated from the existing matched-seed topology CSVs; no simulation was rerun.",
        f"Centrality is exact simple-cycle participation for cycle lengths `3..{k}` on the collapsed undirected trace graph.",
        "All traced swap and asset-transfer edge records are included, then aggregated to unique node pairs for display.",
        f"Layout mode: `{layout}`.",
        exclusion_line,
        "",
    ]
    for panel in PANELS:
        top = sorted(nodes_by_panel[panel].values(), key=lambda node: (-node.cycle_score, -node.weighted_degree))[:10]
        lines.append(f"## {panel}")
        for node in top:
            lines.append(
                f"- rank `{node.rank}` `{node.label}` ({node.node_type}): "
                f"cycle_score `{node.cycle_score:.0f}`, triangles `{node.triangle_participation:.0f}`, "
                f"four-cycles `{node.four_cycle_participation:.0f}`."
            )
        lines.append("")
    lines.extend(
        [
            "Interpretation: high k-cycle centrality means a node lies in many short closed settlement loops in the traced graph.",
            "It is a structural diagnostic, not a claim that the node is economically beneficial, extractive, or causally responsible for the frontier result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        print("[kcycle] no Chrome or ImageMagick converter found; wrote SVG only.")
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
    nodes_path = topology_dir / "topology_nodes.csv"
    edges_path = topology_dir / "topology_edges.csv"
    summary_path = topology_dir / "topology_summary.csv"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Missing topology CSVs in {topology_dir}")

    nodes_by_panel = load_nodes(nodes_path)
    edges_by_panel = load_pair_edges(edges_path, nodes_by_panel)
    summary_by_panel = load_summary(summary_path)
    excluded_nodes = set(args.exclude_node)
    if args.exclude_usd:
        excluded_nodes.add("asset:USD")
    nodes_by_panel, edges_by_panel = apply_node_exclusions(nodes_by_panel, edges_by_panel, excluded_nodes)

    for panel in PANELS:
        compute_cycle_centrality(nodes_by_panel[panel], edges_by_panel[panel], k=args.k)

    svg_path = topology_dir / f"{args.output_prefix}.svg"
    svg_path.write_text(
        build_svg(
            nodes_by_panel,
            edges_by_panel,
            summary_by_panel,
            k=args.k,
            layout=args.layout,
            excluded_nodes=excluded_nodes,
        ),
        encoding="utf-8",
    )
    if args.centrality_prefix is None:
        if args.output_prefix == "fig_kcycle_centrality_network":
            centrality_nodes_path = topology_dir / "kcycle_centrality_nodes.csv"
            centrality_notes_path = topology_dir / "kcycle_centrality_notes.md"
        else:
            stem = args.output_prefix.removeprefix("fig_")
            centrality_nodes_path = topology_dir / f"{stem}_nodes.csv"
            centrality_notes_path = topology_dir / f"{stem}_notes.md"
    else:
        centrality_nodes_path = topology_dir / f"{args.centrality_prefix}_nodes.csv"
        centrality_notes_path = topology_dir / f"{args.centrality_prefix}_notes.md"
    write_node_centrality(centrality_nodes_path, nodes_by_panel, k=args.k)
    write_notes(centrality_notes_path, nodes_by_panel, k=args.k, layout=args.layout, excluded_nodes=excluded_nodes)
    write_converted(svg_path, write_converted=bool(args.write_converted and not args.svg_only))
    print(f"[kcycle] wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
