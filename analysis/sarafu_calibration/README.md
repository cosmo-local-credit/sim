# Public Sarafu Calibration Bundle

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
python scripts/export_public_sarafu_calibration.py \
  --source ../RegenBonds/analysis \
  --output analysis/sarafu_calibration
```
