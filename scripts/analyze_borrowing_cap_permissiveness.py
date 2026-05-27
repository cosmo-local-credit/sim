#!/usr/bin/env python3
"""Analyze whether producer borrowing caps are too permissive.

The analysis compares the current summed-per-lender cap denominator with stricter
aggregate producer-deposit denominators. It can be run on one frontier output
directory or on several cap-sensitivity output directories.
"""

from __future__ import annotations

import argparse
import csv
import html
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable


Row = dict[str, str]


SUMMARY_FIELDS = [
    "source_dir",
    "lender_voucher_cap_deposit_multiple",
    "coupon_target_annual",
    "principal_ratio",
    "safe_p50",
    "strong_success_p50",
    "scheduled_payment_coverage_p05_p50",
    "voucher_to_voucher_volume_ratio_vs_baseline_p50",
    "observed_v2v_share_p50",
    "observed_v2s_share_p50",
    "current_summed_lender_cap_usd_p50",
    "aggregate_deposit_cap_usd_p50",
    "active_aggregate_deposit_cap_usd_p50",
    "current_exposure_usd_p50",
    "cap_multiplicity_factor_p50",
    "exposure_over_current_summed_lender_cap_p50",
    "exposure_over_aggregate_deposit_cap_p50",
    "exposure_over_active_aggregate_deposit_cap_p50",
    "p95_current_summed_lender_cap_utilization_p50",
    "p95_aggregate_deposit_cap_utilization_exact_or_rescaled_p50",
    "p95_active_aggregate_deposit_cap_utilization_exact_or_rescaled_p50",
    "active_borrower_count_p50",
    "cap_bound_producer_count_p50",
    "cap_bound_own_voucher_route_suppressed_count_total_p50",
    "producer_loan_clipped_lender_cap_usd_total_p50",
    "producer_voucher_loan_clipped_lender_cap_usd_total_p50",
]


MULTIPLE_FIELDS = [
    "lender_voucher_cap_deposit_multiple",
    "cells",
    "safe_cells",
    "strong_success_cells",
    "max_p95_current_summed_lender_cap_utilization",
    "max_p95_aggregate_deposit_cap_utilization_exact_or_rescaled",
    "max_p95_active_aggregate_deposit_cap_utilization_exact_or_rescaled",
    "max_exposure_over_current_summed_lender_cap",
    "max_exposure_over_aggregate_deposit_cap",
    "max_exposure_over_active_aggregate_deposit_cap",
    "max_cap_bound_producer_count",
    "max_cap_bound_suppressed_count",
    "total_clipped_lender_cap_usd_p50_sum",
    "median_v2v_ratio",
    "median_observed_v2v_share",
    "median_observed_v2s_share",
]


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


def write_csv(path: Path, rows: list[Row], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def derive_run_metrics(row: Row) -> dict[str, float]:
    exposure = safe_float(
        row,
        "producer_borrowing_current_exposure_usd",
        safe_float(row, "lender_held_producer_voucher_inventory_usd"),
    )
    current_cap = safe_float(
        row,
        "producer_borrowing_current_summed_lender_cap_usd",
        safe_float(row, "producer_borrowing_capacity_usd"),
    )
    aggregate_cap = safe_float(
        row,
        "producer_borrowing_aggregate_deposit_cap_usd",
        safe_float(row, "producer_deposit_credit_capacity_usd"),
    )
    active_cap = safe_float(row, "producer_borrowing_active_aggregate_deposit_cap_usd")
    if active_cap <= 1e-9:
        active_cap = aggregate_cap
    multiplicity = safe_float(row, "producer_borrowing_cap_multiplicity_factor")
    if multiplicity <= 1e-9 and aggregate_cap > 1e-9:
        multiplicity = current_cap / aggregate_cap
    current_p95 = safe_float(row, "producer_borrowing_capacity_used_share_p95")
    aggregate_p95 = safe_float(row, "producer_borrowing_aggregate_deposit_cap_utilization_p95")
    if aggregate_p95 <= 1e-9:
        aggregate_p95 = current_p95 * multiplicity
    active_p95 = safe_float(row, "producer_borrowing_active_aggregate_deposit_cap_utilization_p95")
    if active_p95 <= 1e-9:
        active_p95 = aggregate_p95
    return {
        "exposure": exposure,
        "current_cap": current_cap,
        "aggregate_cap": aggregate_cap,
        "active_cap": active_cap,
        "multiplicity": multiplicity,
        "exposure_over_current_cap": exposure / current_cap if current_cap > 1e-9 else 0.0,
        "exposure_over_aggregate_cap": exposure / aggregate_cap if aggregate_cap > 1e-9 else 0.0,
        "exposure_over_active_cap": exposure / active_cap if active_cap > 1e-9 else 0.0,
        "current_p95": current_p95,
        "aggregate_p95": aggregate_p95,
        "active_p95": active_p95,
    }


def cell_key(row: Row) -> tuple[float, float]:
    return (safe_float(row, "coupon_target_annual"), safe_float(row, "principal_ratio"))


def summarize_input_dir(input_dir: Path) -> list[Row]:
    runs_path = input_dir / "bond_issuer_frontier_runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"missing {runs_path}")
    runs = [row for row in read_csv(runs_path) if safe_float(row, "principal_ratio") > 0.0]
    safety_path = input_dir / "bond_issuer_frontier_safety.csv"
    safety_by_cell = {
        cell_key(row): row
        for row in read_csv(safety_path)
    } if safety_path.exists() else {}
    by_cell: dict[tuple[float, float], list[Row]] = defaultdict(list)
    for row in runs:
        by_cell[cell_key(row)].append(row)

    summary_rows: list[Row] = []
    for (coupon, principal), rows in sorted(by_cell.items()):
        safety_row = safety_by_cell.get((coupon, principal), {})
        derived = [derive_run_metrics(row) for row in rows]
        cap_multiple = median(
            safe_float(row, "configured_lender_voucher_cap_deposit_multiple") for row in rows
        )
        clipped_lender_cap = [
            safe_float(row, "producer_loan_clipped_lender_cap_usd_total")
            + safe_float(row, "producer_voucher_loan_clipped_lender_cap_usd_total")
            for row in rows
        ]
        summary_rows.append({
            "source_dir": input_dir.name,
            "lender_voucher_cap_deposit_multiple": f"{cap_multiple:.6g}",
            "coupon_target_annual": f"{coupon:.6g}",
            "principal_ratio": f"{principal:.6g}",
            "safe_p50": f"{safe_float(safety_row, 'safe', median(safe_float(row, 'safe') for row in rows)):.6g}",
            "strong_success_p50": f"{safe_float(safety_row, 'strong_success', median(safe_float(row, 'strong_success') for row in rows)):.6g}",
            "scheduled_payment_coverage_p05_p50": f"{safe_float(safety_row, 'scheduled_payment_coverage_p05', median(safe_float(row, 'scheduled_payment_coverage_p05') for row in rows)):.6g}",
            "voucher_to_voucher_volume_ratio_vs_baseline_p50": f"{safe_float(safety_row, 'voucher_to_voucher_volume_ratio_vs_baseline', median(safe_float(row, 'voucher_to_voucher_volume_ratio_vs_baseline') for row in rows)):.6g}",
            "observed_v2v_share_p50": f"{median(safe_float(row, 'observed_route_motif_voucher_to_voucher_share_total') for row in rows):.6g}",
            "observed_v2s_share_p50": f"{median(safe_float(row, 'observed_route_motif_voucher_to_stable_share_total') for row in rows):.6g}",
            "current_summed_lender_cap_usd_p50": f"{median(item['current_cap'] for item in derived):.6g}",
            "aggregate_deposit_cap_usd_p50": f"{median(item['aggregate_cap'] for item in derived):.6g}",
            "active_aggregate_deposit_cap_usd_p50": f"{median(item['active_cap'] for item in derived):.6g}",
            "current_exposure_usd_p50": f"{median(item['exposure'] for item in derived):.6g}",
            "cap_multiplicity_factor_p50": f"{median(item['multiplicity'] for item in derived):.6g}",
            "exposure_over_current_summed_lender_cap_p50": f"{median(item['exposure_over_current_cap'] for item in derived):.6g}",
            "exposure_over_aggregate_deposit_cap_p50": f"{median(item['exposure_over_aggregate_cap'] for item in derived):.6g}",
            "exposure_over_active_aggregate_deposit_cap_p50": f"{median(item['exposure_over_active_cap'] for item in derived):.6g}",
            "p95_current_summed_lender_cap_utilization_p50": f"{median(item['current_p95'] for item in derived):.6g}",
            "p95_aggregate_deposit_cap_utilization_exact_or_rescaled_p50": f"{median(item['aggregate_p95'] for item in derived):.6g}",
            "p95_active_aggregate_deposit_cap_utilization_exact_or_rescaled_p50": f"{median(item['active_p95'] for item in derived):.6g}",
            "active_borrower_count_p50": f"{median(safe_float(row, 'producer_borrowing_active_borrower_count') for row in rows):.6g}",
            "cap_bound_producer_count_p50": f"{median(safe_float(row, 'cap_bound_producer_count') for row in rows):.6g}",
            "cap_bound_own_voucher_route_suppressed_count_total_p50": f"{median(safe_float(row, 'cap_bound_own_voucher_route_suppressed_count_total') for row in rows):.6g}",
            "producer_loan_clipped_lender_cap_usd_total_p50": f"{median(safe_float(row, 'producer_loan_clipped_lender_cap_usd_total') for row in rows):.6g}",
            "producer_voucher_loan_clipped_lender_cap_usd_total_p50": f"{median(safe_float(row, 'producer_voucher_loan_clipped_lender_cap_usd_total') for row in rows):.6g}",
        })
        summary_rows[-1]["total_clipped_lender_cap_usd_p50"] = f"{median(clipped_lender_cap):.6g}"
    return summary_rows


def summarize_by_multiple(cell_rows: list[Row]) -> list[Row]:
    grouped: dict[float, list[Row]] = defaultdict(list)
    for row in cell_rows:
        grouped[safe_float(row, "lender_voucher_cap_deposit_multiple")].append(row)

    multiple_rows: list[Row] = []
    for multiple, rows in sorted(grouped.items(), reverse=True):
        multiple_rows.append({
            "lender_voucher_cap_deposit_multiple": f"{multiple:.6g}",
            "cells": str(len(rows)),
            "safe_cells": str(sum(1 for row in rows if safe_float(row, "safe_p50") >= 0.5)),
            "strong_success_cells": str(
                sum(1 for row in rows if safe_float(row, "strong_success_p50") >= 0.5)
            ),
            "max_p95_current_summed_lender_cap_utilization": f"{max(safe_float(row, 'p95_current_summed_lender_cap_utilization_p50') for row in rows):.6g}",
            "max_p95_aggregate_deposit_cap_utilization_exact_or_rescaled": f"{max(safe_float(row, 'p95_aggregate_deposit_cap_utilization_exact_or_rescaled_p50') for row in rows):.6g}",
            "max_p95_active_aggregate_deposit_cap_utilization_exact_or_rescaled": f"{max(safe_float(row, 'p95_active_aggregate_deposit_cap_utilization_exact_or_rescaled_p50') for row in rows):.6g}",
            "max_exposure_over_current_summed_lender_cap": f"{max(safe_float(row, 'exposure_over_current_summed_lender_cap_p50') for row in rows):.6g}",
            "max_exposure_over_aggregate_deposit_cap": f"{max(safe_float(row, 'exposure_over_aggregate_deposit_cap_p50') for row in rows):.6g}",
            "max_exposure_over_active_aggregate_deposit_cap": f"{max(safe_float(row, 'exposure_over_active_aggregate_deposit_cap_p50') for row in rows):.6g}",
            "max_cap_bound_producer_count": f"{max(safe_float(row, 'cap_bound_producer_count_p50') for row in rows):.6g}",
            "max_cap_bound_suppressed_count": f"{max(safe_float(row, 'cap_bound_own_voucher_route_suppressed_count_total_p50') for row in rows):.6g}",
            "total_clipped_lender_cap_usd_p50_sum": f"{sum(safe_float(row, 'total_clipped_lender_cap_usd_p50') for row in rows):.6g}",
            "median_v2v_ratio": f"{median(safe_float(row, 'voucher_to_voucher_volume_ratio_vs_baseline_p50') for row in rows):.6g}",
            "median_observed_v2v_share": f"{median(safe_float(row, 'observed_v2v_share_p50') for row in rows):.6g}",
            "median_observed_v2s_share": f"{median(safe_float(row, 'observed_v2s_share_p50') for row in rows):.6g}",
        })
    return multiple_rows


def svg_text(x: float, y: float, value: object, *, size: int = 12, anchor: str = "start", fill: str = "#222", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-family="Inter, Arial, sans-serif" font-weight="{weight}">'
        f'{html.escape(str(value))}</text>'
    )


def svg_polyline(points: list[tuple[float, float]], color: str, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'


def write_cap_figure(path: Path, multiple_rows: list[Row]) -> None:
    width, height = 760, 520
    left, top, right, bottom = 96, 112, 710, 420
    rows = sorted(multiple_rows, key=lambda row: safe_float(row, "lender_voucher_cap_deposit_multiple"))
    xs = [safe_float(row, "lender_voucher_cap_deposit_multiple") for row in rows]
    if not xs:
        return
    xmin, xmax = min(xs), max(xs)
    xmax = xmax if xmax > xmin else xmin + 1.0
    ymax = max(
        1.0,
        max(safe_float(row, "max_p95_aggregate_deposit_cap_utilization_exact_or_rescaled") for row in rows),
        max(safe_float(row, "max_p95_active_aggregate_deposit_cap_utilization_exact_or_rescaled") for row in rows),
    )
    ymax = min(1.5, max(1.0, ymax * 1.08))

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * (right - left)

    def sy(value: float) -> float:
        return bottom - value / ymax * (bottom - top)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        svg_text(32, 40, "Borrowing Cap Permissiveness Sensitivity", size=21, weight="700"),
        svg_text(32, 64, "Maximum utilization across smoke-grid cells by lender voucher cap deposit multiple.", size=12, fill="#4b5563"),
        '<rect x="32" y="86" width="696" height="376" fill="#ffffff" stroke="#d7dde2"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    for tick in sorted(set(xs)):
        tx = sx(tick)
        elements.append(f'<line x1="{tx:.2f}" y1="{top}" x2="{tx:.2f}" y2="{bottom}" stroke="#e9edf0"/>')
        elements.append(f'<line x1="{tx:.2f}" y1="{bottom}" x2="{tx:.2f}" y2="{bottom + 5}" stroke="#333"/>')
        elements.append(svg_text(tx, bottom + 22, f"{tick:g}", size=10, anchor="middle"))
    for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        ty = sy(tick)
        elements.append(f'<line x1="{left}" y1="{ty:.2f}" x2="{right}" y2="{ty:.2f}" stroke="#e9edf0"/>')
        elements.append(f'<line x1="{left - 5}" y1="{ty:.2f}" x2="{left}" y2="{ty:.2f}" stroke="#333"/>')
        elements.append(svg_text(left - 9, ty + 3.5, f"{tick:.2g}", size=10, anchor="end"))
    elements.append(f'<line x1="{left}" y1="{sy(0.80):.2f}" x2="{right}" y2="{sy(0.80):.2f}" stroke="#9b2d30" stroke-width="1.4" stroke-dasharray="6 4"/>')
    elements.append(svg_text(right - 4, sy(0.80) - 8, "soft cap threshold", size=10, anchor="end", fill="#9b2d30"))

    series = [
        ("current summed-lender p95", "max_p95_current_summed_lender_cap_utilization", "#2c7fb8", None),
        ("aggregate-deposit p95", "max_p95_aggregate_deposit_cap_utilization_exact_or_rescaled", "#f0a202", "6 5"),
        ("active aggregate p95", "max_p95_active_aggregate_deposit_cap_utilization_exact_or_rescaled", "#b23a48", "3 4"),
    ]
    for label, field, color, dash in series:
        points = [(sx(safe_float(row, "lender_voucher_cap_deposit_multiple")), sy(safe_float(row, field))) for row in rows]
        elements.append(svg_polyline(points, color, dash))
        for px, py in points:
            elements.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{color}" stroke="white" stroke-width="1.2"/>')

    lx, ly = 465, 126
    elements.append('<rect x="452" y="104" width="240" height="82" fill="#ffffff" stroke="#dde3e8"/>')
    for idx, (label, _field, color, dash) in enumerate(series):
        yy = ly + idx * 20
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 30}" y2="{yy}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        elements.append(svg_text(lx + 40, yy + 4, label, size=10))
    elements.append(svg_text((left + right) / 2, 474, "Lender voucher cap deposit multiple", size=12, anchor="middle"))
    elements.append(svg_text(left, top - 16, "Y-axis: maximum utilization ratio across cells", size=12, weight="600"))
    elements.append(svg_text(32, 500, "Interpretation: if aggregate-cap utilization rises sharply as the multiple falls, the previous non-binding cap was calibration/definition-sensitive.", size=10, fill="#4b5563"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def maybe_write_png(svg_path: Path) -> None:
    converter = shutil.which("convert")
    if converter is None:
        return
    subprocess.run([converter, str(svg_path), str(svg_path.with_suffix(".png"))], check=True)


def write_notes(path: Path, multiple_rows: list[Row]) -> None:
    if not multiple_rows:
        return
    highest = max(multiple_rows, key=lambda row: safe_float(row, "lender_voucher_cap_deposit_multiple"))
    strongest_agg = max(
        multiple_rows,
        key=lambda row: safe_float(row, "max_p95_aggregate_deposit_cap_utilization_exact_or_rescaled"),
    )
    notes = [
        "# Borrowing Cap Permissiveness Diagnostic",
        "",
        "This diagnostic compares the current summed-per-lender cap denominator with stricter aggregate producer-deposit denominators. It is a calibration/model check, not a paper result by itself.",
        "",
        "## Main Read",
        "",
        f"- Highest tested cap multiple: `{highest['lender_voucher_cap_deposit_multiple']}`.",
        f"- At that multiple, max current summed-lender p95 utilization is `{highest['max_p95_current_summed_lender_cap_utilization']}`.",
        f"- At that multiple, max aggregate-deposit p95 utilization is `{highest['max_p95_aggregate_deposit_cap_utilization_exact_or_rescaled']}`.",
        f"- Largest aggregate-deposit p95 utilization across tested multiples is `{strongest_agg['max_p95_aggregate_deposit_cap_utilization_exact_or_rescaled']}` at multiple `{strongest_agg['lender_voucher_cap_deposit_multiple']}`.",
        "",
        "## Interpretation Rules",
        "",
        "- If lower multiples create cap-bound suppression while service/circulation remains coherent, the mechanism works and the default cap is too loose.",
        "- If aggregate utilization is much higher than current utilization, the current cap definition is permissive.",
        "- If neither aggregate utilization nor suppression rises, the tested frontier grid is still too small relative to borrowing capacity.",
    ]
    path.write_text("\n".join(notes) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="Frontier output directory. Pass once per cap-multiple output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic CSVs and figure. Defaults to the first input dir for one input, otherwise common parent/borrowing_cap_permissiveness.",
    )
    parser.add_argument("--no-png", action="store_true", help="Do not render PNG preview even if ImageMagick is installed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.input_dir]
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif len(input_dirs) == 1:
        output_dir = input_dirs[0]
    else:
        output_dir = input_dirs[0].parent / "borrowing_cap_permissiveness"
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_rows: list[Row] = []
    for input_dir in input_dirs:
        cell_rows.extend(summarize_input_dir(input_dir))
    multiple_rows = summarize_by_multiple(cell_rows)

    write_csv(output_dir / "borrowing_cap_permissiveness_summary.csv", cell_rows, SUMMARY_FIELDS + ["total_clipped_lender_cap_usd_p50"])
    write_csv(output_dir / "borrowing_cap_permissiveness_by_multiple.csv", multiple_rows, MULTIPLE_FIELDS)
    write_notes(output_dir / "borrowing_cap_permissiveness_notes.md", multiple_rows)
    figure_path = output_dir / "fig_borrowing_cap_permissiveness.svg"
    write_cap_figure(figure_path, multiple_rows)
    if not args.no_png:
        maybe_write_png(figure_path)

    print(output_dir / "borrowing_cap_permissiveness_summary.csv")
    print(output_dir / "borrowing_cap_permissiveness_by_multiple.csv")
    print(output_dir / "borrowing_cap_permissiveness_notes.md")
    print(figure_path)
    if figure_path.with_suffix(".png").exists():
        print(figure_path.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
