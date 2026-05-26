#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


MOMENTS = {
    "market_v2v": ("all", "market_route_motif_voucher_to_voucher_share"),
    "market_v2s": ("all", "market_route_motif_voucher_to_stable_share"),
    "market_s2v": ("all", "market_route_motif_stable_to_voucher_share"),
    "market_stable_involved": ("all", "market_route_motif_stable_involved_share"),
    "total_swap_activity": ("all", "total_swap_activity"),
    "repayment_closure": ("all", "repayment_to_borrow_proxy_closure"),
}

RUN_METRICS = (
    "route_success_rate_cumulative",
    "swap_volume_usd_total",
    "producer_debt_originated_usd_total",
    "producer_debt_repaid_usd_total",
    "producer_debt_stable_recovered_usd_total",
    "producer_debt_service_capacity_balance_usd",
    "producer_debt_service_capacity_credited_usd_total",
    "producer_debt_service_capacity_onramp_usd_total",
    "producer_self_repayment_swap_volume_usd_total",
    "producer_self_repayment_voucher_removed_usd_total",
    "producer_debt_pressure_prepayment_usd_total",
    "producer_debt_penalty_accrued_usd_total",
    "producer_debt_penalty_paid_usd_total",
    "producer_debt_arrears_usd",
    "producer_stable_exited_usd_total",
    "producer_stable_reuse_budget_usd_total",
    "market_route_motif_voucher_to_voucher_share_total",
    "market_route_motif_voucher_to_stable_share_total",
    "market_route_motif_stable_to_voucher_share_total",
    "loan_route_motif_count_total",
    "loan_route_motif_voucher_to_voucher_share_total",
    "loan_route_motif_voucher_to_stable_share_total",
    "loan_route_motif_voucher_to_voucher_volume_usd_total",
    "loan_route_motif_voucher_to_stable_volume_usd_total",
    "repayment_route_motif_count_total",
    "repayment_route_motif_stable_to_voucher_share_total",
    "repayment_route_motif_stable_involved_share_total",
    "repayment_route_motif_stable_to_voucher_volume_usd_total",
)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], q: float) -> float:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def read_first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return next(reader, {})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def moment_lookup(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row.get("tier", ""), row.get("moment", "")): row
        for row in read_csv_rows(path)
    }


def summarize_cell(cell_dir: Path) -> dict[str, object] | None:
    summary = read_first_csv_row(cell_dir / "engine_validation_summary.csv")
    run_rows = read_csv_rows(cell_dir / "engine_validation_run_summary.csv")
    moments = moment_lookup(cell_dir / "engine_validation_moments.csv")
    if not summary or not run_rows or not moments:
        return None

    first_run = run_rows[0]
    out: dict[str, object] = {
        "cell": cell_dir.name,
        "output_dir": str(cell_dir),
        "status": summary.get("status", ""),
        "runs": summary.get("runs", ""),
        "ticks": summary.get("ticks", ""),
        "binding_pass_count": summary.get("binding_pass_count", ""),
        "binding_review_count": summary.get("binding_review_count", ""),
        "binding_fail_count": summary.get("binding_fail_count", ""),
        "capacity_share": safe_float(
            first_run.get("configured_producer_debt_pressure_capacity_share")
        ),
        "prepay_share": safe_float(
            first_run.get("configured_producer_debt_pressure_prepay_share")
        ),
        "pressure_enabled": int(
            safe_float(first_run.get("configured_producer_debt_pressure_enabled")) > 0.5
        ),
        "penalty_enabled": int(
            safe_float(first_run.get("configured_producer_debt_penalty_enabled")) > 0.5
        ),
        "penalty_rate": safe_float(
            first_run.get("configured_producer_debt_penalty_rate_per_period")
        ),
    }

    for label, key in MOMENTS.items():
        row = moments.get(key, {})
        out[f"{label}_target"] = safe_float(row.get("empirical_sarafu_moment"))
        out[f"{label}_p50"] = safe_float(row.get("engine_p50"))
        out[f"{label}_relative_error"] = safe_float(row.get("relative_error"))
        out[f"{label}_tolerance"] = safe_float(row.get("tolerance"))
        out[f"{label}_status"] = row.get("validation_status", "")

    for metric in RUN_METRICS:
        values = [safe_float(row.get(metric)) for row in run_rows]
        out[f"{metric}_p50"] = percentile(values, 0.50)
        out[f"{metric}_p05"] = percentile(values, 0.05)
        out[f"{metric}_p95"] = percentile(values, 0.95)

    out["v2s_accept"] = int(
        safe_float(out.get("market_v2s_relative_error")) <= safe_float(out.get("market_v2s_tolerance"))
    )
    out["v2v_accept"] = int(
        safe_float(out.get("market_v2v_relative_error")) <= safe_float(out.get("market_v2v_tolerance"))
    )
    out["s2v_accept"] = int(
        safe_float(out.get("market_s2v_relative_error")) <= safe_float(out.get("market_s2v_tolerance"))
    )
    out["self_repayment_present"] = int(
        safe_float(out.get("producer_self_repayment_swap_volume_usd_total_p50")) > 1e-9
    )
    out["debt_closure_present"] = int(
        safe_float(out.get("producer_debt_repaid_usd_total_p50")) > 1e-9
    )
    out["no_binding_validation_failures"] = int(safe_float(out.get("binding_fail_count")) == 0.0)
    out["candidate"] = int(
        out["v2s_accept"]
        and out["v2v_accept"]
        and out["s2v_accept"]
        and out["self_repayment_present"]
        and out["debt_closure_present"]
        and out["no_binding_validation_failures"]
    )
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: summarize_debt_pressure_sensitivity.py OUTPUT_BASE", file=sys.stderr)
        return 2
    base = Path(argv[1])
    if not base.exists():
        print(f"missing output base: {base}", file=sys.stderr)
        return 2

    rows = []
    for cell_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        row = summarize_cell(cell_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        print(f"no completed validation cells found under {base}", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (safe_float(r["capacity_share"]), safe_float(r["prepay_share"])))
    output = base / "debt_pressure_sensitivity_summary.csv"
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[sensitivity-summary] wrote {output}")
    print(
        "capacity prepay status market_v2s loan_v2s repay_s2v self_repay_p50 "
        "debt_repaid_p50 arrears_p50 capacity_balance_p50 candidate"
    )
    for row in rows:
        print(
            f"{safe_float(row['capacity_share']):7.2f} "
            f"{safe_float(row['prepay_share']):6.2f} "
            f"{str(row['status']):6s} "
            f"{safe_float(row['market_v2s_p50']):8.4f} "
            f"{safe_float(row['loan_route_motif_voucher_to_stable_volume_usd_total_p50']):8.2f} "
            f"{safe_float(row['repayment_route_motif_stable_to_voucher_volume_usd_total_p50']):9.2f} "
            f"{safe_float(row['producer_self_repayment_swap_volume_usd_total_p50']):15.2f} "
            f"{safe_float(row['producer_debt_repaid_usd_total_p50']):15.2f} "
            f"{safe_float(row['producer_debt_arrears_usd_p50']):10.2f} "
            f"{safe_float(row['producer_debt_service_capacity_balance_usd_p50']):20.2f} "
            f"{int(row['candidate'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
