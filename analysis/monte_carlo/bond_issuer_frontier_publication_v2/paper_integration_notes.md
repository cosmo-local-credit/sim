# Bond-Issuer Safety Frontier

- Scenario: `bond_issuer_frontier`.
- Runs per cell: 100.
- Horizon: 260 weekly ticks.
- Term: 260 weekly ticks.
- Full engine-validation gate status: `pass`.
- Frontier mode: `grid` with 0 refinement round(s).
- Voucher settlement mode: `redeem_outputs`.
- Certification policy: `strong_moderate_capped`.
- Issuer reserve share: 0.00%; the frontier setup deploys 100% of gross principal directly to lender pools.
- Issuer payment stride: 26 weekly ticks.
- p05 route-success floor: 85.0%.
- Gross bond principal is the bond amount; it is divided evenly across lender pools as lendable stable at initialization, bypassing the startup waterfall.
- Strong pools are eligible at full weight; moderate pools are capped; weak pools are excluded from base runs unless another policy is explicitly selected.
- Producer own-voucher V2S and V2V borrowing are productive-credit events in paper-facing frontier runs; delayed stable returns can fund monthly stable-to-own-voucher self-repayment.
- Producer borrowing caps remain active as internal guardrails. The paper-facing frontier uses the static calibrated cap and does not report borrowing-cap edges unless an explicit cap-sensitivity study is being run.

## Non-Extraction Gate

A cell is safe only when scheduled bond payments clear, voucher-to-voucher circulation preservation, active routable producer-voucher float, active-pool stable-dependency limits, incremental household or community cash stress, incremental liquidity leakage, unpaid claims, edge concentration, and matched no-bond degradation tests all pass.
Available service-cash headroom is reported separately as issuer operating and risk-capital capacity. The 1.25x threshold is an issuer-headroom frontier, not an additional bondholder-payment requirement.
In `redeem_outputs` mode, net final voucher outputs leave routing liquidity and are returned to issuers; voucher-denominated swap fees remain in fee ledgers for the existing fee-conversion service path.
- The route-success floor is a model settlement-reliability sensitivity parameter, not a direct empirical Sarafu failed-route scalar.
- Voucher-to-voucher count and share are compared against the matched no-bond baseline to protect the empirically observed ROLA-like settlement motif.
- Stable value share and voucher value share in active pools are compared against the matched no-bond baseline so stable/bond injections do not crowd out voucher-backed settlement capacity.
Cash-stress and liquidity-leakage guardrails are evaluated as deltas against the matched no-bond baseline for the same network scale and seeds.

## Headline Frontier

- `current`: safe principal 36,472 USD at ratio 2.00.
