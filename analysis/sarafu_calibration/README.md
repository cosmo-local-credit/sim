# Public Sarafu Calibration Bundle

This directory contains the aggregate, privacy-safe calibration inputs needed
to run the RegenBond Monte Carlo workflows from a standalone public clone of
the `sim` repository.

The per-pool calibration file is anonymized into synthetic template rows. It
does not include raw transactions, addresses, report text, pool labels, or pool
IDs. The simulator uses these templates only for tier mix, activity rates,
repayment/return priors, backing-liquidity scale, and impact-report exposure.

Regenerate from the research workspace with:

```bash
python scripts/export_public_sarafu_calibration.py \
  --source ../RegenBonds/analysis \
  --output analysis/sarafu_calibration
```
