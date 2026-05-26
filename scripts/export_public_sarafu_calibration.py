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
    "productive_credit_activity_boost_calibration.csv",
    "debt_removal_calibration.csv",
    "debt_removal_purchase_weekly_distribution.csv",
    "debt_removal_purchase_weekly_summary.csv",
    "stable_deposit_weekly_distribution.csv",
    "stable_deposit_weekly_summary.csv",
    "pool_swap_weekly_distribution.csv",
    "pool_swap_weekly_summary.csv",
    "fee_conversion_calibration.csv",
    "quarterly_clearing_calibration.csv",
    "route_substitution_diagnostics.csv",
    "voucher_pool_overlap_calibration.csv",
    "voucher_pool_overlap_distribution.csv",
    "impact_projection_by_activity.csv",
    "report_quality_counts.csv",
    "pool_cohort_exclusion_summary.csv",
    "stable_actor_demographics.csv",
    "stable_to_voucher_actor_split.csv",
    "producer_stable_reuse_calibration.csv",
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
    "accepted_voucher_members",
    "backing_inflow",
    "backing_cash_inflow",
    "backing_voucher_inflow",
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


def _metric_value(source: Path, filename: str, metric: str, default: str = "n/a") -> str:
    path = source / filename
    if not path.exists():
        return default
    for row in read_csv(path):
        if row.get("metric") == metric:
            value = row.get("value", default)
            try:
                number = float(value)
            except (TypeError, ValueError):
                return str(value)
            if abs(number - round(number)) < 1e-9:
                return str(int(round(number)))
            return f"{number:.6g}"
    return default


def write_readme(source: Path, output: Path) -> None:
    cohort_pools = _metric_value(source, "pool_cohort_exclusion_summary.csv", "included_pools")
    accepted_members = _metric_value(
        source, "pool_cohort_exclusion_summary.csv", "included_accepted_voucher_members"
    )
    producer_wallets = _metric_value(
        source, "stable_actor_demographics.csv", "unique_producer_voucher_wallets"
    )
    consumer_wallets = _metric_value(
        source, "stable_actor_demographics.csv", "recommended_consumer_wallets"
    )
    stable_pool_interactor_slots = _metric_value(
        source, "stable_actor_demographics.csv", "pool_interactor_address_slots"
    )
    text = f"""# Public Sarafu Calibration Bundle

This directory contains the aggregate, privacy-safe calibration inputs needed
to run the RegenBond Monte Carlo workflows from a standalone public clone of
the `sim` repository.

The per-pool calibration file is anonymized into synthetic template rows. It
does not include raw transactions, addresses, report text, pool labels, or pool
IDs. The simulator uses these templates only for tier mix, activity rates,
repayment/return priors, backing-liquidity scale, and impact-report exposure.
The paper calibration cohort is Kenyan KSh/KES community lending pools. Pools
must have KSh/KES accepted-voucher membership evidence, must not be identified
as non-Kenya by metadata, must not be in the exact named non-community
exclusion list, and must have at least four distinct active swap weeks.
The current exported cohort has {cohort_pools} lender-pool templates,
{accepted_members} accepted-voucher member slots, {producer_wallets} unique
producer-voucher wallets, and {consumer_wallets} recommended external
non-producer consumer wallets. Stable-side pool interaction has
{stable_pool_interactor_slots} address-pool slots; this is an interaction
count, not the consumer-wallet count used by the Monte Carlo topology.
Accepted voucher counts are exported as `accepted_voucher_members`; these are
pool membership slots. Unique producer-voucher wallet counts and multi-pool
membership degree are exported in the voucher-pool overlap calibration. Active
interactors remain `total_users`. The stable actor demographics file separates
external non-producer stable-side users from producer addresses that also use
stable-side interactions. Raw included/excluded pool-address audits remain
local-only in the research workspace.
In the simulator, these empirical pools are open lender pools. Producer and
consumer wallets are private source/sink wallets: they can start or receive a
route, but other agents cannot traverse or clear through them. A producer's
voucher can be accepted by multiple lender pools according to the exported
overlap calibration.
The settlement reliability anchor file adds aggregate ROLA-like voucher
circulation, ROSCA-like stable-credit, same-token return, submitted-swap
execution, and current cluster-topology metrics. The unit-normalization file
records how KES/KSh voucher obligations are converted to USD-equivalent for
bond-accounting columns, using successful KES/USD pool swaps and the current
simulator convention that individual voucher units are 1 voucher = 1 KSh.
Pool backing is split into cash/stable backing and KSh/KES voucher backing so
cash injections can be calibrated separately from voucher-denominated credit
capacity.
The voucher circulation
baseline file records the same motifs over the pool era and recent trailing
windows. The stable dependency anchor file records stable/cash flow shares,
voucher flow shares, and stable-to-voucher dependency proxies under both the
legacy metadata-weighted voucher valuation and the simulator-comparable strict
1 voucher = 1 KSh convention. Additional aggregate tables calibrate producer
deposit proxies, productive-credit timing, debt-removal purchases, voucher-fee
conversion, quarterly clearing, and route substitution diagnostics. These are
empirical settlement motifs and scenario anchors, not direct pool inventory
snapshots or a failed-route denominator.
Voucher-pool overlap calibration is also exported as aggregate topology only:
it reports how many KSh/KES voucher tokens appear in one or more pools, a
bucketed degree distribution, and shared-voucher pool-pool edge counts. It does
not include raw token or pool addresses and should be read as route potential,
not participant intent.
The current ROLA/frontier purchase-demand setting uses network-level stable
budgets, not per-consumer allowances. The paper-facing default uses the
trailing-52-week actor split: original-issuer self purchases, other-producer
stable-to-voucher purchases, and external non-producer purchases are exported
separately. external_nonproducer is not automatically "consumer"; it only means
the initiating wallet is not linked to a cohort producer voucher. Producer
stable reuse is separately calibrated from producer voucher-to-stable receipts
versus producer stable-to-voucher spending, with the residual treated as stable
that exits the local network.
The productive-credit voucher activity boost calibration uses privacy-safe
post-borrow event windows around voucher-to-stable borrow-proxy events. It
reports same-voucher source activity and target-side voucher demand before and
after borrowing, and converts only the source-side motif into the simulator's
voucher-source boost coefficients. In the current artifact, same-voucher
source activity does not rise after borrow-proxy events, so the calibrated
source-weight boost is 0.0 and the source-size multiplier is 1.0. Target-side
voucher demand rises, which supports the purchase-demand channel rather than an
extra uncalibrated producer source boost.
The public bundle also exports a standalone stable/cash deposit weekly series.
That series counts ordinary successful pool-era stable/cash deposit logs and
excludes swap-associated deposit logs, while reporting the inclusive
deposit-log total separately. It is intended to distinguish ordinary stable
backing/deposit timing from stable used directly in swaps.
An all-pool weekly swap series is exported as well, with successful pool-swap
log counts and one-sided normalized input value by week. Output value and
stable/voucher source-pair counts are included for audit.

The bond-frontier safety tests use these files in three ways:

- engine validation reports voucher-to-voucher motifs, strict 1 KSh stable-flow
  shares, and separate active-pool inventory snapshots as calibration
  diagnostics;
- frontier cells are rejected when bond liquidity materially degrades
  voucher-to-voucher circulation versus the matched no-bond baseline;
- frontier cells are rejected when stable/bond liquidity materially increases
  active-pool stable inventory dependency or reduces active-pool voucher
  inventory value share versus the matched no-bond baseline.

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
    write_readme(source, output)
    print(f"Wrote public Sarafu calibration bundle to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
