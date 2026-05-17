#!/usr/bin/env python3
"""Export privacy-safe Sarafu calibration inputs for public simulation runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE = SIM_ROOT.parent / "RegenBonds" / "analysis"
DEFAULT_OUTPUT = SIM_ROOT / "analysis" / "sarafu_calibration"

COPY_FILES = (
    "monte_carlo_calibration_parameters.csv",
    "repayment_calibration_by_tier_asset.csv",
    "borrow_repayment_by_tier.csv",
    "settlement_reliability_anchors.csv",
    "unit_normalization_calibration.csv",
    "voucher_circulation_baseline.csv",
    "stable_dependency_anchors.csv",
    "producer_deposit_calibration.csv",
    "productive_credit_calibration.csv",
    "debt_removal_calibration.csv",
    "fee_conversion_calibration.csv",
    "quarterly_clearing_calibration.csv",
    "route_substitution_diagnostics.csv",
    "impact_projection_by_activity.csv",
    "report_quality_counts.csv",
)

POOL_COLUMNS = (
    "template_id",
    "tier",
    "score",
    "swap_events",
    "recent_swap_weeks_90d",
    "active_weeks",
    "swaps_per_active_week",
    "total_users",
    "backing_inflow",
    "tagged_voucher_tokens",
    "verified_report_exposure",
    "approved_report_exposure",
    "period_aligned_verified_exposure",
    "same_token_return_rate",
    "same_token_out_value",
    "same_token_matched_later_in_value",
    "borrow_proxy_matured_events",
    "borrow_proxy_matured_return_rate",
    "rosca_proxy_value_return_rate",
    "debt_removal_purchase_events",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_pool_templates(source: Path, output: Path) -> None:
    rows = []
    for idx, row in enumerate(read_csv(source / "pool_report_activity.csv"), start=1):
        public_row = {column: row.get(column, "") for column in POOL_COLUMNS if column != "template_id"}
        public_row["template_id"] = f"pool_template_{idx:03d}"
        rows.append(public_row)
    write_csv(output / "pool_report_activity.csv", POOL_COLUMNS, rows)


def copy_public_csvs(source: Path, output: Path) -> None:
    for name in COPY_FILES:
        rows = read_csv(source / name)
        if not rows:
            continue
        write_csv(output / name, tuple(rows[0].keys()), rows)


def write_readme(output: Path) -> None:
    text = """# Public Sarafu Calibration Bundle

This directory contains the aggregate, privacy-safe calibration inputs needed
to run the RegenBond Monte Carlo workflows from a standalone public clone of
the `sim` repository.

The per-pool calibration file is anonymized into synthetic template rows. It
does not include raw transactions, addresses, report text, pool labels, or pool
IDs. The simulator uses these templates only for tier mix, activity rates,
repayment/return priors, backing-liquidity scale, and impact-report exposure.
The settlement reliability anchor file adds aggregate ROLA-like voucher
circulation, ROSCA-like stable-credit, same-token return, submitted-swap
execution, and current cluster-topology metrics. The unit-normalization file
records how KES/KSh voucher obligations are converted to USD-equivalent for
bond-accounting columns, using successful KES/USD pool swaps and the current
simulator convention that individual voucher units are 1 voucher = 1 KSh.
The voucher circulation
baseline file records the same motifs over the pool era and recent trailing
windows. The stable dependency anchor file records stable/cash flow shares,
voucher flow shares, and stable-to-voucher dependency proxies. Additional
aggregate tables calibrate producer deposit proxies, productive-credit timing,
debt-removal purchases, voucher-fee conversion, quarterly clearing, and route
substitution diagnostics. These are empirical settlement motifs and scenario
anchors, not a direct failed-route denominator.

The bond-frontier safety tests use these files in three ways:

- engine validation reports voucher-to-voucher and stable-flow shares as
  calibration diagnostics;
- frontier cells are rejected when bond liquidity materially degrades
  voucher-to-voucher circulation versus the matched no-bond baseline;
- frontier cells are rejected when stable/bond liquidity materially increases
  active-pool stable dependency or reduces active-pool voucher value share
  versus the matched no-bond baseline.

Regenerate from the research workspace with:

```bash
python scripts/export_public_sarafu_calibration.py \\
  --source ../RegenBonds/analysis \\
  --output analysis/sarafu_calibration
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source RegenBonds analysis directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output public calibration directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not source.exists():
        raise SystemExit(f"Source calibration directory does not exist: {source}")
    output.mkdir(parents=True, exist_ok=True)
    copy_public_csvs(source, output)
    export_pool_templates(source, output)
    write_readme(output)
    print(f"Wrote public Sarafu calibration bundle to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
