# Public Sarafu Calibration Bundle

This directory contains the aggregate, privacy-safe calibration inputs needed
to run the RegenBond Monte Carlo workflows from a standalone public clone of
the `sim` repository.

The per-pool calibration file is anonymized into synthetic template rows. It
does not include raw transactions, addresses, report text, pool labels, or pool
IDs. The simulator uses these templates only for tier mix, activity rates,
repayment/return priors, backing-liquidity scale, and impact-report exposure.
The paper calibration cohort is Kenyan KSh/KES individual voucher pools:
cash/stable tokens and KSh/KES-denominated individual vouchers are retained,
while USD/non-KSh/global voucher systems are excluded from the public paper
cohort or handled separately in private review files.
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
The current ROLA/frontier-pilot purchase-demand setting uses a network-level
weekly stable budget, not a per-consumer allowance. In the research workspace,
the aggregate stable-to-voucher purchase value is also exported as a weekly
distribution and figure so the flat budget can be replaced by a dynamic
purchase-timing model once buyer-income moments are added. The current flat
purchase budget is a conservative 260-week spreading of the aggregate purchase
value, not a dynamic buyer-income process.
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
python scripts/export_public_sarafu_calibration.py \
  --source ../RegenBonds/analysis \
  --output analysis/sarafu_calibration
```
