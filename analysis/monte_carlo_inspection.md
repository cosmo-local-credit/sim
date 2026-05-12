# RegenBond Monte Carlo Inspection

Generated from the CLI runner in `scripts/run_regenbond_monte_carlo.py`.

## Runs

All-scenario comparison:

```bash
.venv/bin/python scripts/run_regenbond_monte_carlo.py \
  --scenario all \
  --runs 2 \
  --ticks 52 \
  --seed 101 \
  --coupon-targets 0,0.06,0.12 \
  --terms 52 \
  --output analysis \
  --analysis-stride 4 \
  --pool-metrics-stride 4 \
  --plot-scenario regenbond_lp_injection \
  --plot-coupon 0.06 \
  --plot-term 52
```

Five-year headline regenbond trajectory:

```bash
.venv/bin/python scripts/run_regenbond_monte_carlo.py \
  --scenario regenbond_lp_injection \
  --runs 3 \
  --ticks 260 \
  --seed 501 \
  --coupon-targets 0.06 \
  --terms 260 \
  --output analysis/headline_regenbond_5y \
  --analysis-stride 13 \
  --pool-metrics-stride 13 \
  --plot-scenario regenbond_lp_injection \
  --plot-coupon 0.06 \
  --plot-term 260
```

## Headline Results

- One-year regenbond scenario at 6 percent coupon: median annualized fee yield `19.99%`, median coupon coverage `3.33`, median cumulative fee return `79,945`, median coupon shortfall `0`.
- Five-year regenbond headline at 6 percent coupon: median annualized fee yield `19.25%`, median coupon coverage `3.21`, median cumulative fee return `384,973`, median coupon shortfall `0`.
- Potential and realized pool connectivity are both near a single giant component in the headline runs: median realized largest component share is about `0.992`.
- Expected verified report exposure rises with simulated activity: five-year headline median is `1,719` expected verified report exposures.

## Figure Interpretation

For line charts, the thin line is the Monte Carlo median (`p50`) at each tick.
The shaded band is the 5th--95th percentile range across stochastic seeds,
holding the scenario parameters fixed.

`figures/fig_cumulative_fee_return_over_time.png` shows cumulative stable fee
return reaching a one-year median of about `79,945` under the
`regenbond_lp_injection` scenario at a `6%` annual coupon target. With
`400,000` of LP/bond-like principal, the one-year coupon due is `24,000`, so
the inspection run implies `3.33x` median coupon coverage. This is evidence
that fee flow can service coupon-like return targets in the calibrated scenario.
It is not evidence that principal is fully repaid within one year.

The five-year headline figure reaches a median cumulative fee return of about
`384,973`, close to but still below the `400,000` principal baseline. That
result should be framed as strong fee-service capacity and near-principal
recovery under this inspection run, not a final claim of guaranteed payback.

## Key RegenBond Scenario Parameters

- Time step: `1 tick = 1 week`.
- Main one-year figure: `52` ticks, `2` stochastic runs, seed base `101`,
  analysis diagnostics recorded every `4` ticks.
- Five-year headline figure: `260` ticks, `3` stochastic runs, seed base `501`,
  analysis diagnostics recorded every `13` ticks.
- Scenario: `regenbond_lp_injection`.
- LP/bond-like principal: `400,000` stable units from one LP.
- Annual coupon target: `6%`.
- Bond fee-service share: `1.0`, meaning all simulated LP/sCLC stable return is
  counted toward bond service in this first-pass experiment.
- Initial network: `100` producer pools, `20` consumer pools, `4` lender pools,
  and `1` LP pool.
- Pool fee rate: `2%` of swap output.
- CLC rake: `100%` of pool fees routed into the CLC/waterfall accounting layer
  in this scenario.
- Waterfall cadence: every `4` ticks.
- Waterfall inputs: pool fee ledger included; stable is cash-eligible.
- Fee conversion slippage: `25` basis points.
- Waterfall policy: insurance top-up, operations budget, liquidity mandates,
  then CLC pool allocation under current engine logic.
- Operations cap: `20,000` stable units per epoch.
- Insurance top-up cap: `100,000` stable units per epoch.
- Liquidity mandate share: `50%` of remaining cash after prior waterfall steps.
- Bootstrap liquidity mandate share: `100%` for `1` epoch.
- sCLC fee access: enabled; `50%` of CLC stable is eligible for sCLC access,
  conditional on insurance and operations requirements.
- CLC pool: always open; CLC rebalancing enabled with target stable ratio `1.0`.
- Routing: NOAM routing, top `16` pools per asset, top `16` outputs per
  pool/asset, beam width `40`, max hops `5`.
- NOAM clearing: enabled every `4` ticks.
- Swap activity: max `4` swap attempts per active pool, global request budget
  `100` per tick, mean swap size fraction `0.02`, minimum swap size `1`.
- Producer and consumer off-ramp rate: `5%` per month.
- Redemption probability: `0.85`.
- Loan term: `4` weeks, with loan activity spread over `4` ticks.
- Sarafu empirical repayment priors: strong cash return `22.96%`, strong
  voucher return `66.14%`; moderate cash `20.15%`, moderate voucher `76.39%`;
  weak cash `7.07%`, weak voucher `50.20%`.

## Interpretation Cautions

- The all-scenario pass uses only two runs per cell. It is a mechanics and artifact inspection run, not a publication-grade uncertainty estimate.
- The current stress flags are deliberately blunt. Household cash stress is triggered in most scenarios because the threshold is defined as pools under stable reserve over all pools. This should be refined before using the failure table as a substantive result.
- Liquidity leakage flags are also sensitive to the current denominator and should be interpreted as a diagnostic trigger, not a calibrated welfare claim.
- `stress_high_coupon` collapses all requested coupon levels to at least 12 percent by design, so its table row aggregates six runs at 12 percent in this inspection pass.

## Artifacts

- CSV tables: `mc_bond_return_table.csv`, `mc_failure_table.csv`, `mc_calibration_table.csv`.
- Full CSV outputs: `mc_run_summary.csv`, `mc_timeseries_quantiles.csv`, `mc_bond_return_timeseries.csv`, `mc_network_scaling_timeseries.csv`, `mc_failure_metrics.csv`, `mc_failure_summary.csv`.
- PNG figures: `figures/fig_bond_apr_over_time.png`, `figures/fig_coupon_coverage_over_time.png`, `figures/fig_cumulative_fee_return_over_time.png`, `figures/fig_network_connectivity_over_time.png`, `figures/fig_report_exposure_over_time.png`, `figures/fig_failure_rates.png`.
