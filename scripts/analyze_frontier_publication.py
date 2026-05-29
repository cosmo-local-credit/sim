#!/usr/bin/env python3
"""Summarize paper-facing bond frontier outputs.

This script intentionally depends only on the Python standard library so it can
run on local laptops and remote batch hosts without extra packages. It reads the
frontier safety CSV written by run_regenbond_monte_carlo.py and writes compact
paper-analysis artifacts next to the run outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median
from typing import Iterable


SAFETY_FILE = "bond_issuer_frontier_safety.csv"
RUNS_FILE = "bond_issuer_frontier_runs.csv"


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def fmt_pct(value: float, digits: int = 1) -> str:
    if math.isnan(value):
        return ""
    return f"{100.0 * value:.{digits}f}%"


def fmt_money(value: float, digits: int = 0) -> str:
    if math.isnan(value):
        return ""
    return f"${value:,.{digits}f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def outcome_label(row: dict[str, str]) -> str:
    if as_bool(row, "strong_success"):
        return "strong"
    if as_bool(row, "safe") and not as_bool(row, "headroom_pass"):
        return "safe_low_headroom"
    if as_bool(row, "safe"):
        return "safe"
    return "fail"


def row_sort_key(row: dict[str, str]) -> tuple[str, float, float]:
    return (
        str(row.get("network_scale", "")),
        as_float(row, "coupon_target_annual"),
        as_float(row, "principal_ratio"),
    )


def max_row(rows: Iterable[dict[str, str]], predicate: str) -> dict[str, str] | None:
    filtered = [row for row in rows if as_bool(row, predicate)]
    if not filtered:
        return None
    return max(filtered, key=lambda row: as_float(row, "principal_ratio"))


def metric_range(rows: list[dict[str, str]], key: str) -> dict[str, object]:
    pairs = [(as_float(row, key), row) for row in rows if not math.isnan(as_float(row, key))]
    if not pairs:
        return {
            "metric": key,
            "min": "",
            "min_coupon": "",
            "min_principal_ratio": "",
            "max": "",
            "max_coupon": "",
            "max_principal_ratio": "",
            "median": "",
        }
    values = [value for value, _row in pairs]
    lo_value, lo_row = min(pairs, key=lambda pair: pair[0])
    hi_value, hi_row = max(pairs, key=lambda pair: pair[0])
    return {
        "metric": key,
        "min": lo_value,
        "min_coupon": as_float(lo_row, "coupon_target_annual"),
        "min_principal_ratio": as_float(lo_row, "principal_ratio"),
        "max": hi_value,
        "max_coupon": as_float(hi_row, "coupon_target_annual"),
        "max_principal_ratio": as_float(hi_row, "principal_ratio"),
        "median": median(values),
    }


def build_coupon_frontier(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    scales = sorted({row.get("network_scale", "") for row in rows})
    for scale in scales:
        scale_rows = [row for row in rows if row.get("network_scale", "") == scale]
        coupons = sorted({as_float(row, "coupon_target_annual") for row in scale_rows})
        for coupon in coupons:
            coupon_rows = [
                row for row in scale_rows if as_float(row, "coupon_target_annual") == coupon
            ]
            safe = max_row(coupon_rows, "safe")
            strong = max_row(coupon_rows, "strong_success")
            headroom = max_row(coupon_rows, "headroom_pass")
            first_fail = next(
                (
                    row
                    for row in sorted(coupon_rows, key=lambda r: as_float(r, "principal_ratio"))
                    if not as_bool(row, "safe")
                ),
                None,
            )
            out.append(
                {
                    "network_scale": scale,
                    "coupon_target_annual": coupon,
                    "max_safe_principal_ratio": as_float(safe, "principal_ratio") if safe else "",
                    "max_safe_principal_usd": as_float(safe, "principal_usd_p50") if safe else "",
                    "max_strong_principal_ratio": as_float(strong, "principal_ratio") if strong else "",
                    "max_strong_principal_usd": as_float(strong, "principal_usd_p50") if strong else "",
                    "max_headroom_principal_ratio": as_float(headroom, "principal_ratio")
                    if headroom
                    else "",
                    "max_headroom_principal_usd": as_float(headroom, "principal_usd_p50")
                    if headroom
                    else "",
                    "first_failed_principal_ratio": as_float(first_fail, "principal_ratio")
                    if first_fail
                    else "",
                    "first_failed_binding_constraint": first_fail.get("binding_constraint", "")
                    if first_fail
                    else "",
                }
            )
    return out


def build_outcome_matrix(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    scales = sorted({row.get("network_scale", "") for row in rows})
    ratios = sorted({as_float(row, "principal_ratio") for row in rows})
    for scale in scales:
        for coupon in sorted({as_float(r, "coupon_target_annual") for r in rows if r.get("network_scale", "") == scale}):
            row_out: dict[str, object] = {
                "network_scale": scale,
                "coupon_target_annual": coupon,
            }
            for ratio in ratios:
                match = next(
                    (
                        row
                        for row in rows
                        if row.get("network_scale", "") == scale
                        and as_float(row, "coupon_target_annual") == coupon
                        and as_float(row, "principal_ratio") == ratio
                    ),
                    None,
                )
                row_out[f"principal_ratio_{ratio:g}"] = outcome_label(match) if match else ""
            out.append(row_out)
    return out


def write_markdown(
    path: Path,
    rows: list[dict[str, str]],
    frontier_rows: list[dict[str, object]],
    binding_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    run_rows: int | None,
) -> None:
    total = len(rows)
    safe_count = sum(as_bool(row, "safe") for row in rows)
    strong_count = sum(as_bool(row, "strong_success") for row in rows)
    headroom_count = sum(as_bool(row, "headroom_pass") for row in rows)
    repayment_count = sum(as_bool(row, "repayment_pass") for row in rows)
    v2v_float_count = sum(as_bool(row, "v2v_float_pass") for row in rows)
    runs_values = sorted({int(as_float(row, "runs", 0)) for row in rows})
    coupons = sorted({as_float(row, "coupon_target_annual") for row in rows})
    ratios = sorted({as_float(row, "principal_ratio") for row in rows})
    scales = sorted({row.get("network_scale", "") for row in rows})

    v2v = metric_range(rows, "voucher_to_voucher_volume_ratio_vs_baseline")
    coverage = metric_range(rows, "scheduled_payment_coverage_p05")
    route = metric_range(rows, "route_success_p05")
    headroom = metric_range(rows, "issuer_operating_risk_headroom_p50")
    float_ratio = metric_range(rows, "active_routable_producer_voucher_float_ratio_vs_baseline")

    lines = [
        "# Frontier Publication Analysis",
        "",
        "## Audit",
        "",
        f"- Safety rows: {total}.",
        f"- Run rows: {run_rows if run_rows is not None else 'not checked'}.",
        f"- Runs per cell: {', '.join(str(v) for v in runs_values)}.",
        f"- Network scales: {', '.join(scales)}.",
        f"- Coupons: {', '.join(fmt_pct(c, 0) for c in coupons)}.",
        f"- Principal ratios: {', '.join(f'{r:g}' for r in ratios)}.",
        "",
        "## Headline Counts",
        "",
        f"- Safe cells: {safe_count}/{total}.",
        f"- Strong-success cells: {strong_count}/{total}.",
        f"- Repayment-pass cells: {repayment_count}/{total}.",
        f"- Issuer-headroom-pass cells: {headroom_count}/{total}.",
        f"- V2V-float-pass cells: {v2v_float_count}/{total}.",
        f"- Binding/failing cells: {len(binding_rows)}/{total}.",
        "",
        "## Frontier By Coupon",
        "",
        "| Coupon | Max safe ratio | Max safe principal | Max strong ratio | Max strong principal | First failed ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frontier_rows:
        lines.append(
            "| {coupon} | {safe_ratio} | {safe_usd} | {strong_ratio} | {strong_usd} | {fail_ratio} |".format(
                coupon=fmt_pct(float(row["coupon_target_annual"]), 0),
                safe_ratio=row["max_safe_principal_ratio"],
                safe_usd=fmt_money(float(row["max_safe_principal_usd"]))
                if row["max_safe_principal_usd"] != ""
                else "",
                strong_ratio=row["max_strong_principal_ratio"],
                strong_usd=fmt_money(float(row["max_strong_principal_usd"]))
                if row["max_strong_principal_usd"] != ""
                else "",
                fail_ratio=row["first_failed_principal_ratio"],
            )
        )

    lines.extend(
        [
            "",
            "## Metric Ranges",
            "",
            f"- Scheduled-payment coverage p05 ranges from {coverage['min']:.3f} to {coverage['max']:.3f}.",
            f"- Issuer operating/risk headroom p50 ranges from {headroom['min']:.3f} to {headroom['max']:.3f}.",
            f"- V2V volume ratio vs baseline ranges from {v2v['min']:.3f} to {v2v['max']:.3f}.",
            f"- Active routable producer-voucher float ratio ranges from {float_ratio['min']:.3f} to {float_ratio['max']:.3f}.",
            f"- Route success p05 ranges from {fmt_pct(route['min'], 1)} to {fmt_pct(route['max'], 1)}.",
            "",
            "## Binding Cells",
            "",
        ]
    )
    if binding_rows:
        lines.append(
            "| Coupon | Principal ratio | Scheduled coverage p05 | Scheduled coverage p50 | Headroom p50 | Binding constraint |"
        )
        lines.append("|---:|---:|---:|---:|---:|---|")
        for row in binding_rows:
            lines.append(
                "| {coupon} | {ratio:g} | {p05:.3f} | {p50:.3f} | {head:.3f} | {binding} |".format(
                    coupon=fmt_pct(float(row["coupon_target_annual"]), 0),
                    ratio=float(row["principal_ratio"]),
                    p05=float(row["scheduled_payment_coverage_p05"]),
                    p50=float(row["scheduled_payment_coverage_p50"]),
                    head=float(row["issuer_operating_risk_headroom_p50"]),
                    binding=row["binding_constraint"],
                )
            )
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The final run supports a settlement-capacity frontier framing: the visible boundary is scheduled-payment coverage and issuer headroom, not V2V collapse.",
            "- Voucher-to-voucher volume is broadly preserved across the tested grid, including high stress cells; failures occur where scheduled service cannot clear reliably.",
            "- Borrowing-cap utilization remains an internal diagnostic under the static calibrated cap and should not be reported as a binding frontier result from this run.",
            "",
            "## Generated Files",
            "",
            "- `frontier_publication_coupon_frontier.csv`",
            "- `frontier_publication_outcome_matrix.csv`",
            "- `frontier_publication_binding_cells.csv`",
            "- `frontier_publication_metric_ranges.csv`",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(input_dir: Path, output_dir: Path) -> None:
    safety_path = input_dir / SAFETY_FILE
    if not safety_path.exists():
        raise SystemExit(f"Missing required file: {safety_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(read_csv(safety_path), key=row_sort_key)
    frontier_rows = build_coupon_frontier(rows)
    matrix_rows = build_outcome_matrix(rows)
    binding_rows = [
        {
            "network_scale": row.get("network_scale", ""),
            "coupon_target_annual": as_float(row, "coupon_target_annual"),
            "principal_ratio": as_float(row, "principal_ratio"),
            "principal_usd_p50": as_float(row, "principal_usd_p50"),
            "scheduled_payment_coverage_p05": as_float(row, "scheduled_payment_coverage_p05"),
            "scheduled_payment_coverage_p50": as_float(row, "scheduled_payment_coverage_p50"),
            "issuer_operating_risk_headroom_p50": as_float(row, "issuer_operating_risk_headroom_p50"),
            "voucher_to_voucher_volume_ratio_vs_baseline": as_float(
                row, "voucher_to_voucher_volume_ratio_vs_baseline"
            ),
            "binding_constraint": row.get("binding_constraint", ""),
        }
        for row in rows
        if not as_bool(row, "safe")
    ]
    metrics = [
        "scheduled_payment_coverage_p05",
        "scheduled_payment_coverage_p50",
        "issuer_operating_risk_headroom_p50",
        "voucher_to_voucher_volume_ratio_vs_baseline",
        "voucher_to_voucher_share_p50",
        "active_routable_producer_voucher_float_ratio_vs_baseline",
        "route_success_p05",
        "producer_borrowing_capacity_used_share_p95",
        "stable_receipts_waiting_for_repayment_usd_p50",
        "producer_debt_arrears_usd_p50",
    ]
    metric_rows = [metric_range(rows, key) for key in metrics]

    run_rows = None
    runs_path = input_dir / RUNS_FILE
    if runs_path.exists():
        run_rows = max(0, line_count(runs_path) - 1)

    write_csv(output_dir / "frontier_publication_coupon_frontier.csv", frontier_rows)
    write_csv(output_dir / "frontier_publication_outcome_matrix.csv", matrix_rows)
    write_csv(output_dir / "frontier_publication_binding_cells.csv", binding_rows)
    write_csv(output_dir / "frontier_publication_metric_ranges.csv", metric_rows)
    write_markdown(
        output_dir / "frontier_publication_analysis_summary.md",
        rows,
        frontier_rows,
        binding_rows,
        metric_rows,
        run_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    analyze(args.input_dir, output_dir)
    print(f"Wrote frontier publication analysis artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
