# Frontier Publication Analysis

## Audit

- Safety rows: 64.
- Run rows: 6500.
- Runs per cell: 100.
- Network scales: current.
- Coupons: 0%, 4%, 8%, 12%, 15%, 25%, 35%, 45%.
- Principal ratios: 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.

## Headline Counts

- Safe cells: 57/64.
- Strong-success cells: 52/64.
- Repayment-pass cells: 57/64.
- Issuer-headroom-pass cells: 52/64.
- V2V-float-pass cells: 64/64.
- Binding/failing cells: 7/64.

## Frontier By Coupon

| Coupon | Max safe ratio | Max safe principal | Max strong ratio | Max strong principal | First failed ratio |
|---:|---:|---:|---:|---:|---:|
| 0% | 2.0 | $36,472 | 2.0 | $36,472 |  |
| 4% | 2.0 | $36,472 | 1.5 | $27,354 |  |
| 8% | 2.0 | $36,472 | 1.5 | $27,354 |  |
| 12% | 1.5 | $27,354 | 1.5 | $27,354 | 2.0 |
| 15% | 1.5 | $27,354 | 1.0 | $18,236 | 2.0 |
| 25% | 1.5 | $27,354 | 1.0 | $18,236 | 2.0 |
| 35% | 1.0 | $18,236 | 1.0 | $18,236 | 1.5 |
| 45% | 1.0 | $18,236 | 0.75 | $13,677 | 1.5 |

## Metric Ranges

- Scheduled-payment coverage p05 ranges from 0.580 to 1.000.
- Issuer operating/risk headroom p50 ranges from 0.593 to 26.907.
- V2V volume ratio vs baseline ranges from 0.989 to 1.006.
- Active routable producer-voucher float ratio ranges from 0.998 to 1.004.
- Route success p05 ranges from 92.1% to 92.6%.

## Binding Cells

| Coupon | Principal ratio | Scheduled coverage p05 | Scheduled coverage p50 | Headroom p50 | Binding constraint |
|---:|---:|---:|---:|---:|---|
| 12% | 2 | 0.976 | 0.999 | 0.999 | p50_scheduled_payment_coverage;p05_scheduled_payment_coverage;p95_unpaid_claims |
| 15% | 2 | 0.917 | 0.940 | 0.940 | p50_scheduled_payment_coverage;p05_scheduled_payment_coverage;p95_unpaid_claims |
| 25% | 2 | 0.770 | 0.787 | 0.787 | p50_scheduled_payment_coverage;p05_scheduled_payment_coverage;p95_unpaid_claims |
| 35% | 1.5 | 0.880 | 0.900 | 0.900 | p50_scheduled_payment_coverage;p05_scheduled_payment_coverage;p95_unpaid_claims |
| 35% | 2 | 0.665 | 0.679 | 0.679 | p50_scheduled_payment_coverage;p05_scheduled_payment_coverage;p95_unpaid_claims |
| 45% | 1.5 | 0.772 | 0.789 | 0.789 | p50_scheduled_payment_coverage;p05_scheduled_payment_coverage;p95_unpaid_claims |
| 45% | 2 | 0.580 | 0.593 | 0.593 | p50_scheduled_payment_coverage;p05_scheduled_payment_coverage;p95_unpaid_claims |

## Interpretation

- The final run supports a settlement-capacity frontier framing: the visible boundary is scheduled-payment coverage and issuer headroom, not V2V collapse.
- Voucher-to-voucher volume is broadly preserved across the tested grid, including high stress cells; failures occur where scheduled service cannot clear reliably.
- Borrowing-cap utilization remains an internal diagnostic under the static calibrated cap and should not be reported as a binding frontier result from this run.

## Generated Files

- `frontier_publication_coupon_frontier.csv`
- `frontier_publication_outcome_matrix.csv`
- `frontier_publication_binding_cells.csv`
- `frontier_publication_metric_ranges.csv`
