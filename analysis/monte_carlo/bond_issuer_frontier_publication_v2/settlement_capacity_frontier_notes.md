# Settlement and Productive-Credit Diagnostics

These diagnostics support frontier interpretation but are not a borrowing-cap edge result.
The paper-facing causal chain is: catalytic or bond liquidity expands lender liquidity; producers borrow through own-voucher V2S or V2V routes; borrowing creates delayed productive stable receipts; repayment pressure and stable availability then change producer attention, target choice, and capacity to continue ROLA-like voucher-to-voucher exchange.
Borrowing-cap utilization is retained as an internal guardrail diagnostic under the static calibrated cap. It should not be reported as a headline frontier unless a separate cap-sensitivity study shows an empirically grounded binding edge.

## Directional Smoke Checks

- V2V decline is present when voucher-to-voucher volume falls below 98% of the matched no-bond baseline.
- Repayment waiting is present when stable receipts or off-network debt-service capacity are reserved for borrower self-repayment.
- Cap utilization remains an audit field only; under the current static multiplier it is not a pass/fail frontier criterion.

## Low-Stress Reference

- coupon=0.00%, principal_ratio=0.05, V2V ratio=0.999, capacity used p95=0.031.

## High-Stress Reference

- coupon=45.00%, principal_ratio=2.00, V2V ratio=0.989, capacity used p95=0.031, stable waiting=2,925.50.

Interpretation should be updated after larger frontier runs; this file is designed for fast mechanism smoke tests and edge finding.
