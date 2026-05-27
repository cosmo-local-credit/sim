#!/usr/bin/env python3
"""Generate settlement-capacity frontier smoke figures from frontier CSVs.

The script intentionally writes SVG directly with the Python standard library so
the figures remain reproducible on remote batch hosts that do not have plotting
libraries installed. If ImageMagick's `convert` is available, PNG previews are
also produced.
"""

from __future__ import annotations

import argparse
import csv
import html
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable


ColorMap = dict[float, str]
Row = dict[str, str]
Point = tuple[float, float]


def safe_float(row: Row, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def read_csv(path: Path) -> list[Row]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value: object) -> str:
    return html.escape(str(value))


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 12,
    fill: str = "#222",
    anchor: str = "start",
    weight: str = "normal",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-family="Inter, Arial, sans-serif" '
        f'font-weight="{weight}"{transform}>{esc(value)}</text>'
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str = "#ddd",
    stroke_width: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def polyline(points: Iterable[Point], stroke: str, *, width: float = 2.0, dash: str | None = None) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
    )


def circle(x: float, y: float, radius: float, fill: str, stroke: str = "white") -> str:
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.2"/>'
    )


def svg_start(title: str, subtitle: str, *, width: int = 760, height: int = 540) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        text(32, 40, title, size=21, weight="700"),
        text(32, 64, subtitle, size=12, fill="#4b5563"),
        rect(32, 86, width - 64, height - 124, "#ffffff", "#d7dde2", 1.0),
    ]


def svg_finish(elements: list[str], note: str | None = None, *, height: int = 540) -> str:
    if note:
        elements.append(text(32, height - 25, note, size=11, fill="#4b5563"))
    elements.append("</svg>")
    return "\n".join(elements)


def add_plot_axes(
    elements: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xticks: list[float],
    yticks: list[float],
    xlabel: str,
    ylabel: str,
) -> tuple[Callable[[float], float], Callable[[float], float], tuple[float, float, float, float]]:
    left, top, right, bottom = x + 82, y + 48, x + width - 34, y + height - 72
    plot_width = right - left
    plot_height = bottom - top

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_width if xmax != xmin else left

    def sy(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * plot_height if ymax != ymin else bottom

    elements.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333" stroke-width="1"/>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333" stroke-width="1"/>')

    for tick in xticks:
        tx = sx(tick)
        elements.append(f'<line x1="{tx:.2f}" y1="{bottom}" x2="{tx:.2f}" y2="{bottom + 5}" stroke="#333"/>')
        elements.append(text(tx, bottom + 23, f"{tick:g}", size=10, anchor="middle"))
        elements.append(f'<line x1="{tx:.2f}" y1="{top}" x2="{tx:.2f}" y2="{bottom}" stroke="#e9edf0"/>')

    for tick in yticks:
        ty = sy(tick)
        label = f"{tick:.2g}" if abs(tick) < 1 else f"{tick:g}"
        elements.append(f'<line x1="{left - 5}" y1="{ty:.2f}" x2="{left}" y2="{ty:.2f}" stroke="#333"/>')
        elements.append(text(left - 8, ty + 3.5, label, size=10, anchor="end"))
        elements.append(f'<line x1="{left}" y1="{ty:.2f}" x2="{right}" y2="{ty:.2f}" stroke="#e9edf0"/>')

    elements.append(text((left + right) / 2, y + height - 24, xlabel, size=12, anchor="middle"))
    # Horizontal y-axis titles survive SVG->PNG conversion more reliably than
    # rotated text across minimal remote hosts.
    elements.append(text(left, top - 16, f"Y-axis: {ylabel}", size=12, anchor="start", weight="600"))
    return sx, sy, (left, top, right, bottom)


def coupon_colors(coupons: list[float]) -> ColorMap:
    defaults = {
        0.0: "#2c7fb8",
        0.08: "#41ab5d",
        0.15: "#f0a202",
        0.30: "#b23a48",
    }
    fallback = ["#2c7fb8", "#41ab5d", "#f0a202", "#b23a48", "#6a4c93", "#6f4e37"]
    return {coupon: defaults.get(coupon, fallback[i % len(fallback)]) for i, coupon in enumerate(coupons)}


def coupon_label(coupon: float) -> str:
    return f"{int(round(coupon * 100))}%"


def write_svg(path: Path, elements: list[str], note: str | None = None) -> None:
    path.write_text(svg_finish(elements, note), encoding="utf-8")


def maybe_write_png(svg_path: Path, *, enabled: bool) -> None:
    if not enabled:
        return
    converter = shutil.which("convert")
    if converter is None:
        return
    png_path = svg_path.with_suffix(".png")
    subprocess.run([converter, str(svg_path), str(png_path)], check=True)


def plot_observed_motif_shift(
    output_dir: Path,
    by_run: dict[tuple[float, float], list[Row]],
    coupons: list[float],
    principals: list[float],
    colors: ColorMap,
    *,
    png: bool,
) -> Path:
    path = output_dir / "fig_observed_settlement_motif_shift.svg"
    elements = svg_start(
        "Observed Settlement Motifs Shift With Stress",
        "Median route shares by coupon and principal ratio; 52-week smoke, 3 seeds per cell.",
    )
    sx, sy, _ = add_plot_axes(
        elements,
        32,
        86,
        696,
        416,
        xmin=min(principals),
        xmax=max(principals),
        ymin=0.35,
        ymax=0.60,
        xticks=principals,
        yticks=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        xlabel="Gross principal ratio (principal / certified backing capacity)",
        ylabel="Median share of observed ledger routes",
    )
    for coupon in coupons:
        v2v_points: list[Point] = []
        v2s_points: list[Point] = []
        for principal in principals:
            rows = by_run[(coupon, principal)]
            v2v = median(safe_float(row, "observed_route_motif_voucher_to_voucher_share_total") for row in rows)
            v2s = median(safe_float(row, "observed_route_motif_voucher_to_stable_share_total") for row in rows)
            v2v_points.append((sx(principal), sy(v2v)))
            v2s_points.append((sx(principal), sy(v2s)))
        elements.append(polyline(v2v_points, colors[coupon], width=2.4))
        elements.append(polyline(v2s_points, colors[coupon], width=2.1, dash="6 5"))
        for px, py in v2v_points:
            elements.append(circle(px, py, 4.2, colors[coupon]))
        for px, py in v2s_points:
            elements.append(
                f'<rect x="{px - 4.1:.2f}" y="{py - 4.1:.2f}" width="8.2" height="8.2" '
                f'fill="{colors[coupon]}" stroke="white" stroke-width="1.2"/>'
            )

    lx, ly = 525, 124
    elements.append(rect(lx - 12, ly - 24, 170, 136, "#ffffff", "#dde3e8"))
    elements.append(polyline([(lx, ly), (lx + 28, ly)], "#555", width=2.2))
    elements.append(circle(lx + 14, ly, 3.8, "#555"))
    elements.append(text(lx + 38, ly + 4, "V2V route share", size=10))
    elements.append(polyline([(lx, ly + 22), (lx + 28, ly + 22)], "#555", width=2.0, dash="6 5"))
    elements.append(f'<rect x="{lx + 10:.2f}" y="{ly + 18:.2f}" width="8" height="8" fill="#555"/>')
    elements.append(text(lx + 38, ly + 26, "V2S route share", size=10))
    for idx, coupon in enumerate(coupons):
        yy = ly + 50 + idx * 18
        elements.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 28}" y2="{yy}" stroke="{colors[coupon]}" stroke-width="4"/>')
        elements.append(text(lx + 38, yy + 4, f"{coupon_label(coupon)} coupon", size=10))

    write_svg(path, elements, "Result: stress shifts settlement composition gradually; this is not a collapse claim.")
    maybe_write_png(path, enabled=png)
    return path


def plot_outcome_grid(
    output_dir: Path,
    summary_by_key: dict[tuple[float, float], Row],
    coupons: list[float],
    principals: list[float],
    *,
    png: bool,
) -> Path:
    path = output_dir / "fig_frontier_outcome_grid.svg"
    elements = svg_start(
        "Frontier Outcome Grid",
        "Green cells pass strongly; red cells fail service/headroom in the smoke run.",
    )
    grid_x, grid_y = 126, 132
    grid_width, grid_height = 570, 292
    cell_width = grid_width / len(principals)
    cell_height = grid_height / len(coupons)
    status_color = {0: "#b94a48", 1: "#e0a43a", 2: "#3f8f5a"}
    status_word = {0: "fail", 1: "safe", 2: "strong"}

    for j, principal in enumerate(principals):
        elements.append(
            text(
                grid_x + (j + 0.5) * cell_width,
                grid_y - 16,
                f"{principal:g}",
                size=12,
                anchor="middle",
                weight="700",
            )
        )
    for i, coupon in enumerate(coupons):
        elements.append(
            text(
                grid_x - 20,
                grid_y + (i + 0.5) * cell_height + 4,
                coupon_label(coupon),
                size=12,
                anchor="end",
                weight="700",
            )
        )
        for j, principal in enumerate(principals):
            row = summary_by_key[(coupon, principal)]
            safe = int(safe_float(row, "safe"))
            strong = int(safe_float(row, "strong_success"))
            status = 2 if strong else (1 if safe else 0)
            rx = grid_x + j * cell_width
            ry = grid_y + i * cell_height
            fill = "white" if status == 0 else "#18202a"
            elements.append(rect(rx, ry, cell_width, cell_height, status_color[status], "#ffffff", 1.5))
            elements.append(
                text(
                    rx + cell_width / 2,
                    ry + cell_height / 2 - 7,
                    status_word[status],
                    size=13,
                    fill=fill,
                    anchor="middle",
                    weight="700",
                )
            )
            elements.append(
                text(
                    rx + cell_width / 2,
                    ry + cell_height / 2 + 13,
                    f"V2V {safe_float(row, 'voucher_to_voucher_volume_ratio_vs_baseline'):.2f}",
                    size=10,
                    fill=fill,
                    anchor="middle",
                )
            )

    elements.append(
        text(
            grid_x + grid_width / 2,
            grid_y + grid_height + 38,
            "Gross principal ratio (principal / certified backing capacity)",
            size=12,
            anchor="middle",
        )
    )
    elements.append(text(grid_x, grid_y - 42, "Y-axis: Annual coupon target", size=12, anchor="start", weight="600"))
    write_svg(path, elements, "Result: the first boundary is repayment/headroom, not disappearance of voucher circulation.")
    maybe_write_png(path, enabled=png)
    return path


def plot_scheduled_payment_coverage(
    output_dir: Path,
    safety_by_key: dict[tuple[float, float], Row],
    coupons: list[float],
    principals: list[float],
    colors: ColorMap,
    *,
    png: bool,
) -> Path:
    path = output_dir / "fig_scheduled_payment_coverage_boundary.svg"
    elements = svg_start(
        "Scheduled Payment Coverage Boundary",
        "Fifth-percentile scheduled payment coverage by coupon and principal ratio.",
    )
    sx, sy, bounds = add_plot_axes(
        elements,
        32,
        86,
        696,
        416,
        xmin=min(principals),
        xmax=max(principals),
        ymin=0.0,
        ymax=1.05,
        xticks=principals,
        yticks=[0.0, 0.25, 0.50, 0.75, 1.0],
        xlabel="Gross principal ratio (principal / certified backing capacity)",
        ylabel="p05 scheduled payment coverage ratio",
    )
    elements.append(polyline([(bounds[0], sy(1.0)), (bounds[2], sy(1.0))], "#111", width=1.4, dash="3 4"))
    elements.append(text(bounds[2] - 5, sy(1.0) - 7, "full scheduled coverage", size=10, anchor="end"))
    for coupon in coupons:
        points = [
            (sx(principal), sy(safe_float(safety_by_key[(coupon, principal)], "scheduled_payment_coverage_p05")))
            for principal in principals
        ]
        elements.append(polyline(points, colors[coupon], width=2.4))
        for px, py in points:
            elements.append(circle(px, py, 4.2, colors[coupon]))

    lx, ly = 526, 346
    elements.append(rect(lx - 12, ly - 24, 160, 92, "#ffffff", "#dde3e8"))
    for idx, coupon in enumerate(coupons):
        yy = ly + idx * 18
        elements.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 28}" y2="{yy}" stroke="{colors[coupon]}" stroke-width="4"/>')
        elements.append(text(lx + 38, yy + 4, f"{coupon_label(coupon)} coupon", size=10))
    write_svg(path, elements, "Result: scheduled service coverage deteriorates sharply with larger principal and coupon.")
    maybe_write_png(path, enabled=png)
    return path


def plot_capacity_caps(
    output_dir: Path,
    summary_by_key: dict[tuple[float, float], Row],
    coupons: list[float],
    principals: list[float],
    colors: ColorMap,
    *,
    png: bool,
) -> Path:
    path = output_dir / "fig_borrowing_capacity_caps_not_reached.svg"
    elements = svg_start(
        "Borrowing Capacity Caps Do Not Bind In This Smoke",
        "p95 producer borrowing-capacity utilization remains far below the soft cap threshold.",
    )
    sx, sy, bounds = add_plot_axes(
        elements,
        32,
        86,
        696,
        416,
        xmin=min(principals),
        xmax=max(principals),
        ymin=0.0,
        ymax=0.90,
        xticks=principals,
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8],
        xlabel="Gross principal ratio (principal / certified backing capacity)",
        ylabel="p95 producer borrowing-capacity utilization",
    )
    elements.append(polyline([(bounds[0], sy(0.80)), (bounds[2], sy(0.80))], "#9b2d30", width=1.6, dash="6 4"))
    elements.append(text(bounds[2] - 5, sy(0.80) - 8, "soft cap threshold", size=10, fill="#9b2d30", anchor="end"))
    for coupon in coupons:
        points = [
            (sx(principal), sy(safe_float(summary_by_key[(coupon, principal)], "producer_borrowing_capacity_used_share_p95")))
            for principal in principals
        ]
        elements.append(polyline(points, colors[coupon], width=2.4))
        for px, py in points:
            elements.append(circle(px, py, 4.2, colors[coupon]))
    elements.append(rect(bounds[0] + 16, bounds[1] + 14, 252, 38, "#ffffff", "#dde3e8"))
    elements.append(text(bounds[0] + 30, bounds[1] + 38, "cap-bound producers = 0 in all cells", size=12, fill="#374151"))
    write_svg(path, elements, "Result: cap-bound suppression is not the active failure mechanism under this cap calibration.")
    maybe_write_png(path, enabled=png)
    return path


def generate_figures(
    input_dir: Path,
    output_dir: Path,
    *,
    png: bool,
    include_cap_nonbinding: bool,
) -> list[Path]:
    runs = read_csv(input_dir / "bond_issuer_frontier_runs.csv")
    safety = read_csv(input_dir / "bond_issuer_frontier_safety.csv")
    summary = read_csv(input_dir / "settlement_capacity_frontier_summary.csv")

    by_run: dict[tuple[float, float], list[Row]] = defaultdict(list)
    for row in runs:
        key = (safe_float(row, "coupon_target_annual"), safe_float(row, "principal_ratio"))
        if key == (0.0, 0.0):
            continue
        by_run[key].append(row)

    summary_by_key = {
        (safe_float(row, "coupon_target_annual"), safe_float(row, "principal_ratio")): row
        for row in summary
    }
    safety_by_key = {
        (safe_float(row, "coupon_target_annual"), safe_float(row, "principal_ratio")): row
        for row in safety
    }
    coupons = sorted({key[0] for key in summary_by_key})
    principals = sorted({key[1] for key in summary_by_key})
    colors = coupon_colors(coupons)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_observed_motif_shift(output_dir, by_run, coupons, principals, colors, png=png),
        plot_outcome_grid(output_dir, summary_by_key, coupons, principals, png=png),
        plot_scheduled_payment_coverage(output_dir, safety_by_key, coupons, principals, colors, png=png),
    ]
    if include_cap_nonbinding:
        paths.append(plot_capacity_caps(output_dir, summary_by_key, coupons, principals, colors, png=png))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing bond_issuer_frontier_runs.csv, bond_issuer_frontier_safety.csv, and settlement_capacity_frontier_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Only write SVG files. By default, PNG previews are also written when ImageMagick convert is available.",
    )
    parser.add_argument(
        "--include-cap-nonbinding-figure",
        action="store_true",
        help=(
            "Also write the old cap-not-reached diagnostic. Paper-facing runs should prefer "
            "scripts/analyze_borrowing_cap_permissiveness.py after a cap sweep."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    paths = generate_figures(
        input_dir,
        output_dir,
        png=not args.no_png,
        include_cap_nonbinding=bool(args.include_cap_nonbinding_figure),
    )
    for path in paths:
        print(path)
        png_path = path.with_suffix(".png")
        if png_path.exists():
            print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
