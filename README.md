# CLC Pool Network Simulator (MVP-1)

This repo simulates a network of agent-owned liquidity pools exchanging a stablecoin and agent-issued vouchers.
It includes multi-hop routing, loan issuance/repayment via voucher swaps, stable supply growth, NOAM routing,
NOAM clearing, and a CLC economics layer (fees + waterfall). A Streamlit UI (`app.py`) lets you run ticks,
inspect metrics, and sweep parameters.

**NOAM Routing (online, per request)**
1) Build/refresh the **working set** (Top‑K pools per asset, Top‑M outs per pool/asset).
2) (Optional) **Overlay**: find hub paths when enabled and the network is large enough.
3) **Beam‑A*** over the working set with network‑aware scoring:
   success probability + fee penalty + scarcity penalty + inventory‑rebalance benefit + dead‑end penalty,
   plus a CLC edge bonus.
4) **Validate** the best route with live quotes; if it fails, try the next best.
5) **Execute**, update success/λ (scarcity) states, and cache outcomes.

**NOAM Clearing (periodic batch, every `noam_clearing_stride_ticks`)**
1) Build the same **working set** (Top‑K/Top‑M).
2) Enumerate candidate **cycles** within caps (max hops, per‑asset edge cap, budget).
3) **Score** cycles with the same multi‑objective weights as routing.
4) **Validate & execute** cycles with live quotes until budget/limits hit.
5) **Update** success/λ states and metrics.

Time model: **1 tick = 1 week** (4 ticks = 1 month). Many rates are defined "per month" and applied per tick.

---

## Running the app

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Enable debug inventory logs:

```bash
streamlit run app.py --logger.level=debug
```

---

## Core model (agentic logic)

### Agents and pools
- **One agent = one pool**. Each agent issues its own voucher asset `VCHR:<agent_id>` tracked by an issuer ledger.
- Pools maintain:
  - **Vault** (inventory), **listing registry** (what assets they accept), **value index** (prices in USD),
  - **swap limiter** (cap-in per window), **fee registry**, and **fee ledgers**.

### Bootstrap (initial network)
- If `economics_enabled=True`, **system pools** are created first: ops, insurance, mandates, CLC.
- Initial agent pools are created in this order (up to `initial_pools`):
  **producer, producer, lender, liquidity_provider**. Additional pools follow role mix probabilities.

### Roles and policies
**Producer pools**
- **Goal**: sell their voucher / accept stable inflows, repay debt.
- **Listings (wants)**: stable + own voucher + random wanted assets (Poisson mean `add_pool_want_assets_mean`).
- **Inventory (offers)**: own voucher seed + offered assets seed.
- **Restrictions**: cannot swap out stable (`producer_no_stable_outflow`).
- **Starts with**:
  - Stable seed: `0`
  - Own voucher seed: `exp(mean=10000)`
  - Offered asset seeds: `exp(mean=10000)` per offered asset

**Lender pools**
- **Goal**: provide stable loans; hold voucher collateral.
- **Listings (wants)**: stable + random wanted assets.
- **Inventory (offers)**: stable + offered assets seed.
- **Restrictions**: swaps must include a stable leg (`lender_requires_stable_leg`).
- **Starts with**:
  - Stable seed: **fixed** `lender_initial_stable_mean` (default `100000`)
  - Offered asset seeds: `exp(mean=250)` per offered asset

**Consumer pools**
- **Goal**: spend stable to acquire vouchers and redeem.
- **Listings (wants)**: stable + own voucher + random wanted assets.
- **Inventory (offers)**: stable + own voucher + offered assets seed.
- **Restrictions**: cannot swap out stable (`consumer_no_stable_outflow`).
- **Starts with** (current implementation constants in `sim/engine.py`):
  - Stable seed: `exp(mean=initial_stable_per_pool_mean * 0.25)`
  - Own voucher seed: `exp(mean=200)`
  - Offered asset seeds: `exp(mean=150)` per offered asset

**Liquidity Provider (LP) pools**
- **Goal**: contribute stable to the waterfall in exchange for sCLC.
- **Listings**: stable only.
- **Inventory**: stable seed only.
- **Starts with**: **fixed** `lp_initial_stable_mean` (default `100000`).
- **Contribution**: each tick contributes `lp_waterfall_contribution_rate` of its stable to the waterfall, minting sCLC
  until `lp_sclc_supply_cap` is exhausted.

**System pools**
- **ops**, **insurance**, **mandates**, **clc**. These are non-agent pools controlled by policy:
  - System pools are **paused** by default and do not trade.
  - **CLC pool** is special (see CLC section), and is **always open** when `clc_pool_always_open=True`.

### Swap rules and limits
- **All pools list USD**; each listing has a cap-in per rolling window (`default_window_len`).
- **Cap-in limits** apply only if `swap_limits_enabled=True`.
- **Stable cap-in** uses role-specific caps (`lender_stable_cap_in`, `producer_stable_cap_in`, else `default_cap_in`).
- **Stable reserve guardrail**: swaps that take stable below `min_stable_reserve` are blocked.
- **Role constraints**:
  - Lenders require a stable leg.
  - Producers/Consumers cannot swap out stable.
  - CLC pool requires a stable/sCLC leg.

### Redemption
- **Consumers** auto-redeem vouchers they receive (if issued by someone else).
- **Final hop vouchers** are redeemed to the issuer when the route completes.

---

## Routing and trading

### Swap request generation (per tick)
- Every **non-system, non-LP pool** tries to execute swaps.
- Per pool attempts are based on:
  - `random_route_requests_per_tick`
  - value-scaled attempts `swap_attempts_value_scale_usd`
  - capped by `swap_attempts_max_per_pool`
  - and constrained by `swap_requests_budget_per_tick` across all pools.
- Attempts are multiplied by a **utilization boost** when network volume is below `utilization_target_rate`
  (capped by `utilization_boost_max`).
- Producers also attempt **loan repayments** and **new loan issuance** according to `loan_activity_period_ticks`.
- Target assets are chosen by `swap_target_selection_mode` and retried up to `swap_target_retry_count` times.
  Producers/consumers **will not target stable**.
- If a route fails at the chosen amount, the engine retries once with a smaller **fallback amount** before giving up.

### Loan mechanics (producer ↔ lender)
- **Issuance**: producers mint their voucher as needed and route it to lenders to receive USD stable.
- **Repayment**: producers route USD (or another asset) to acquire their voucher from lenders.
  The repayment amount amortizes `loan_term_weeks` and is spread by `loan_activity_period_ticks`.

### NOAM Routing (default)
NOAM is the default router (`routing_mode=noam`). It is a **network-aware overlay + beam search**:
1) **Working set** (Top-K/Top-M)
   - `noam_topk_pools_per_asset` pools per asset
   - `noam_topm_out_per_pool` outputs per pool/asset
   - Adaptive caps can shrink these when the network is large.
2) **Overlay routing** (optional)
   - Uses hub assets and precomputed overlay paths to reduce search.
   - Only used when `noam_overlay_enabled=True` **and** pool count ≥ `noam_overlay_min_pools`.
3) **Beam search** with scoring:
   - Success probability (`noam_weight_success`)
   - Fees (`noam_weight_fee`)
   - Scarcity (`noam_weight_lambda`, `noam_scarcity_eta`)
   - Inventory benefit (`noam_weight_benefit`)
   - Dead-end penalty (`noam_weight_deadend`)
   - **CLC edge bonus** (`noam_clc_edge_bonus`) favors CLC pool edges
   - Amount propagation is **carry-only** (no rate propagation); validation happens at execution.
   - Benefit mode uses **inventory rebalance** (per-hop imbalance reduction).
4) **Caching + failure TTL**
   - Route cache (`noam_route_cache_*`), failure TTL (`noam_failure_ttl_ticks`)

If NOAM is disabled, the legacy BFS router can be used (`routing_mode=bfs`).

---

## NOAM Clearing (batch cycles)
NOAM Clearing runs periodically to clear feasible cycles and rebalance the network.

- **Stride**: every `noam_clearing_stride_ticks` ticks.
- **Working set**: same Top-K/Top-M graph as routing.
- **Cycles**: up to `noam_clearing_max_cycles`, with max length `noam_clearing_max_hops`.
- **Edge caps**: per asset `noam_clearing_edge_cap_per_asset`.
- **Budget**: max of
  - `noam_clearing_budget_usd`, and
  - `noam_clearing_budget_share * network_value`.
- **Minimum cycle value**: `noam_clearing_min_cycle_value_usd`.
- **Execution**: quotes are validated live; fees and edge states update on success/failure.
- **Scoring**: uses the same success/fee/scarcity/benefit/dead-end weights as NOAM routing.

---

## CLC economics and waterfall

### Fees
- **Pool fee**: `pool_fee_rate` applied to gross output.
- **CLC rake**: `clc_rake_rate` is taken **from** pool fee (not additive).
- **All executed swaps** (routing and clearing) accrue pool fees and CLC rake.
- System pools have zero fees by default; agent pools use the configured pool fee + CLC rake.
- Fee ledgers are swept into the waterfall each epoch.
- Cumulative fee totals are tracked separately in metrics.
  - If `waterfall_include_pool_fees=False`, pool fees stay in their pool (only CLC fees flow to the waterfall).

### Waterfall inflows
- **CLC fee ledger + pool fee ledger** (if `waterfall_include_pool_fees=True`)
- **LP contributions** (stable inflow + sCLC mint)
- **External inflows** (used for mandates / policy injections)

Before allocation, fee assets are **converted to cash** when eligible:
- Assets in `cash_eligible_assets` are converted to USD using the pool’s value index,
  with `cash_conversion_slippage_bps` and optional `cash_conversion_max_usd_per_epoch`.
- Non-convertible assets are deposited to the CLC pool **in-kind**.

### Waterfall order (current implementation)
1) **Insurance top-up**
   - Target = `insurance_target_multiplier * total_voucher_value * risk_weight`
   - Cap per epoch: `insurance_max_topup_usd`
2) **Ops budget**
   - Cap per epoch: `core_ops_budget_usd`
3) **Liquidity mandates**
   - Share of remaining cash: `liquidity_mandate_share`
   - **No cap** when `liquidity_mandate_max_usd = 0`
4) **CLC pool**
   - Receives all remaining cash

> Note: `waterfall_alpha_ops_share`, `waterfall_beta_liquidity_share`, and `waterfall_gamma_insurance_share`
> remain in config for legacy experiments, but the current waterfall logic routes all remaining cash to CLC.

### Liquidity mandates (distribution)
Default mode: `lender_liquidity`
- **Only lenders** are eligible.
- Weights are based on **low stable liquidity** (deficit, or inverse stable if no deficits).
- The full mandate budget is distributed; any remainder is assigned to the lowest-liquidity lender.

Other modes (configurable):
- `activity_weighted`, `deficit_weighted`, `utilization_weighted`.

### CLC pool behavior
- **Always open** for swaps when `clc_pool_always_open=True` (pool is unpaused and `min_stable_reserve=0`).
- **Preferred in routing/clearing** via `noam_clc_edge_bonus` + Top-K inclusion.
- **Rebalancing**: periodically swaps vouchers → stable to maintain target ratio.

### sCLC fee access
- sCLC access budget depends on ops/insurance targets unless `clc_pool_always_open` is used.
- Controlled by `sclc_fee_access_*` parameters.

---

## Insurance & incidents
- Incidents occur with probability `incident_base_rate * pool_risk_weight`.
- Losses are capped by `incident_haircut_cap` and `incident_max_per_tick`.
- **Eligibility**: pools must have paid at least `insurance_min_fee_usd` in CLC fees over the last
  `insurance_fee_window_ticks` ticks.
- Payouts are drawn from the insurance pool; unpaid portions are tracked.

---

## Parameters (ScenarioConfig)

All parameters live in `sim/config.py`. Defaults shown below.

### Network growth
| Parameter | Default | Meaning |
| --- | --- | --- |
| `initial_pools` | `4` | Initial pool count at boot. |
| `pool_growth_rate_per_tick` | `0.02` | Pool growth rate per tick. |
| `pool_growth_stride_ticks` | `4` | Stride for pool growth. |
| `max_pools` | `2000` | Hard cap on active pools. |
| `add_pool_offer_assets_mean` | `4` | Mean offered assets per new pool. |
| `add_pool_want_assets_mean` | `6` | Mean wanted assets per new pool. |
| `p_offer_overlap` | `0.75` | Offered assets sampled from existing universe. |
| `p_want_overlap` | `0.85` | Wanted assets sampled from existing universe. |
| `desired_assets_min_per_pool` | `8` | Minimum desired assets per pool. |
| `desired_assets_max_per_pool` | `100` | Maximum desired assets per pool. |
| `desired_assets_growth_per_asset` | `0.2` | Growth factor vs total asset universe. |
| `desired_assets_add_per_tick` | `2` | Desired assets added per stride. |
| `desired_assets_stride_ticks` | `4` | Stride for desired asset growth. |

### Agent role mix
| Parameter | Default | Meaning |
| --- | --- | --- |
| `p_liquidity_provider` | `0.02` | Probability a new agent is an LP. |
| `p_lender` | `0.25` | Probability a new agent is a lender. |
| `p_producer` | `0.50` | Probability a new agent is a producer. |
| `p_consumer` | `0.25` | Probability a new agent is a consumer. |

### Stable supply & flow
| Parameter | Default | Meaning |
| --- | --- | --- |
| `stable_symbol` | `USD` | Stablecoin asset id. |
| `initial_stable_per_pool_mean` | `2000.0` | Baseline for consumer stable seed (exp mean * 0.25). |
| `lender_initial_stable_mean` | `100000.0` | Fixed stable seed for lenders. |
| `lp_initial_stable_mean` | `100000.0` | Fixed stable seed for LPs. |
| `stable_inflow_per_tick` | `0.0` | Generic per-pool inflow (per month). |
| `producer_inflow_per_tick` | `0.05` | Producer inflow rate (per month). |
| `consumer_inflow_per_tick` | `0.05` | Consumer inflow rate (per month). |
| `lender_inflow_per_tick` | `0.0` | Lender inflow rate (per month). |
| `liquidity_provider_inflow_per_tick` | `0.0` | LP inflow rate (per month). |
| `stable_shock_tick` | `None` | One-time shock tick. |
| `stable_shock_amount` | `0.0` | Shock amount (positive adds, negative drains). |
| `stable_growth_mode` | `per_pool` | Growth regime (`per_pool` or `network_target`). |
| `stable_growth_stride_ticks` | `4` | Stride for stable growth. |
| `stable_supply_cap` | `100000000.0` | Network cap for `network_target` mode. |
| `stable_supply_growth_rate` | `0.15` | Growth rate toward cap (per month). |
| `stable_supply_noise` | `0.05` | Relative noise on target. |
| `stable_outflow_rate` | `0.02` | Stable outflow rate (per month). |
| `stable_growth_smoothing` | `0.25` | Fraction of target gap applied per tick. |
| `stable_flow_mode` | `none` | Activity-driven flow mode (`none`, `loan`, `swap`, `both`). |
| `stable_flow_loan_scale` | `1.0` | Loan-based flow scaling. |
| `stable_flow_swap_scale` | `0.05` | Swap-based flow scaling. |
| `stable_flow_swap_target_usd` | `0.0` | Swap flow target per window. |
| `stable_flow_window_ticks` | `4` | Flow window length (ticks). |
| `stable_inflow_activity_share` | `0.6` | Blend of activity vs deficit weights. |
| `stable_inflow_activity_window_ticks` | `12` | Activity window. |
| `voucher_inflow_share` | `0.5` | Voucher USD minted per USD stable inflow. |
| `offramps_enabled` | `True` | Enable stable offramps. |
| `offramp_rate_min_per_tick` | `0.0` | Min offramp rate. |
| `offramp_rate_max_per_tick` | `0.02` | Max offramp rate. |
| `offramp_success_ema_alpha` | `0.2` | EMA alpha for success/failure. |
| `offramp_min_attempts` | `2` | Min attempts before offramps apply. |

### Metrics & performance
| Parameter | Default | Meaning |
| --- | --- | --- |
| `metrics_stride` | `5` | Network metrics stride. |
| `pool_metrics_stride` | `10` | Pool metrics stride. |
| `max_active_pools_per_tick` | `100` | Cap active pools sampled per tick. |
| `max_candidate_pools_per_hop` | `None` | Cap candidate pools per hop. |
| `event_log_maxlen` | `None` | Event log max length. |

### Routing & NOAM
| Parameter | Default | Meaning |
| --- | --- | --- |
| `routing_mode` | `noam` | `noam` or `bfs`. |
| `max_hops` | `4` | Max hops in routing (legacy BFS). |
| `sticky_route_bias` | `0.6` | Bias toward known counterparties. |
| `sticky_affinity_decay` | `0.02` | Per-tick decay. |
| `sticky_affinity_gain` | `0.15` | Gain per successful swap. |
| `sticky_affinity_cap` | `50.0` | Affinity cap. |
| `noam_topk_pools_per_asset` | `16` | Top-K pools per asset. |
| `noam_topm_out_per_pool` | `16` | Top-M outs per pool/asset. |
| `noam_beam_width` | `40` | Beam width. |
| `noam_max_hops` | `5` | Max hops in NOAM. |
| `noam_topk_refresh_ticks` | `50` | Refresh Top-K/Top-M. |
| `noam_dynamic_caps_enabled` | `True` | Adaptive caps. |
| `noam_dynamic_cap_reference_pools` | `50` | Reference pool count for scaling. |
| `noam_dynamic_min_topk` | `2` | Min Top-K. |
| `noam_dynamic_min_topm` | `2` | Min Top-M. |
| `noam_dynamic_min_beam` | `8` | Min beam width. |
| `noam_edge_cap_per_state` | `30` | Edge expansion cap per state. |
| `noam_dynamic_min_edge_cap` | `10` | Min edge cap. |
| `noam_overlay_enabled` | `True` | Enable overlay graph. |
| `noam_hub_asset_count` | `60` | Hub asset count. |
| `noam_hub_depth` | `2` | Hub depth. |
| `noam_hub_candidate_limit` | `10` | Hub candidate limit. |
| `noam_overlay_top_r_paths` | `3` | Overlay paths per query. |
| `noam_overlay_max_hops` | `3` | Max overlay hops. |
| `noam_overlay_refresh_ticks` | `200` | Overlay refresh stride. |
| `noam_overlay_min_pools` | `200` | Overlay min pools. |
| `noam_clearing_enabled` | `True` | Enable NOAM clearing. |
| `noam_clearing_stride_ticks` | `2` | Clearing cadence. |
| `noam_clearing_max_cycles` | `200` | Max cycles per run. |
| `noam_clearing_max_hops` | `4` | Max hops per cycle. |
| `noam_clearing_edge_cap_per_asset` | `16` | Edge cap per asset in clearing. |
| `noam_clearing_budget_usd` | `25000.0` | Base clearing budget. |
| `noam_clearing_budget_share` | `0.01` | Budget share of network value. |
| `noam_clearing_min_cycle_value_usd` | `1.0` | Min cycle value. |
| `noam_success_ema_alpha` | `0.2` | EMA alpha for success. |
| `noam_success_min` | `0.05` | Min success. |
| `noam_success_max` | `0.98` | Max success. |
| `noam_weight_success` | `1.2` | Success weight. |
| `noam_weight_fee` | `1.0` | Fee weight. |
| `noam_weight_lambda` | `1.2` | Scarcity weight. |
| `noam_weight_benefit` | `1.5` | Benefit weight. |
| `noam_weight_deadend` | `1.0` | Dead-end weight. |
| `noam_clc_edge_bonus` | `0.75` | CLC edge bonus. |
| `noam_scarcity_eta` | `0.1` | Scarcity update rate. |
| `noam_safe_budget_fraction` | `0.2` | Safe usage threshold. |
| `noam_lambda_decay` | `0.1` | Lambda decay. |
| `noam_usage_cap` | `5.0` | Usage cap. |
| `noam_failure_ttl_ticks` | `4` | Failure cache TTL. |
| `noam_route_cache_ttl_ticks` | `6` | Route cache TTL. |
| `noam_route_cache_bucket_usd` | `100.0` | Route cache bucket size. |

### Limits & fees
| Parameter | Default | Meaning |
| --- | --- | --- |
| `default_window_len` | `10` | Default cap window length. |
| `default_cap_in` | `10000.0` | Default cap-in per window. |
| `lender_voucher_cap_in` | `2000.0` | Cap-in for lender vouchers. |
| `lender_stable_cap_in` | `25000.0` | Cap-in for lender stable. |
| `producer_voucher_cap_in` | `15000.0` | Cap-in for producer vouchers. |
| `producer_stable_cap_in` | `1000000000.0` | Cap-in for producer stable. |
| `pool_fee_rate` | `0.02` | Pool fee rate. |
| `clc_rake_rate` | `1.0` | CLC rake (share of pool fee). |

### Economics / Waterfall
| Parameter | Default | Meaning |
| --- | --- | --- |
| `economics_enabled` | `True` | Enable economics layer. |
| `waterfall_epoch_ticks` | `4` | Waterfall cadence. |
| `waterfall_include_pool_fees` | `True` | Include pool fees in waterfall. |
| `cash_eligible_assets` | `["USD"]` | Assets eligible for cash conversion. |
| `cash_conversion_slippage_bps` | `25.0` | Conversion slippage. |
| `cash_conversion_max_usd_per_epoch` | `None` | Max conversion per epoch. |
| `core_ops_budget_usd` | `2000.0` | Ops cap per epoch. |
| `insurance_max_topup_usd` | `10000.0` | Insurance cap per epoch. |
| `liquidity_mandate_share` | `0.50` | Share of remaining cash to mandates. |
| `liquidity_mandate_max_usd` | `0.0` | Mandate cap per epoch (0 = no cap). |
| `liquidity_mandate_mode` | `lender_liquidity` | Mandate distribution mode. |
| `liquidity_mandate_activity_window_ticks` | `12` | Activity window for mandates. |
| `liquidity_mandate_max_per_pool_usd` | `2000.0` | Per-pool cap (ignored in lender_liquidity). |
| `waterfall_alpha_ops_share` | `0.20` | Legacy remainder share (ops). |
| `waterfall_beta_liquidity_share` | `0.40` | Legacy remainder share (liquidity). |
| `waterfall_gamma_insurance_share` | `0.40` | Legacy remainder share (insurance). |
| `lp_waterfall_contribution_rate` | `1.0` | LP contribution rate per tick. |
| `lp_sclc_supply_cap` | `100000000.0` | sCLC supply cap for LPs. |
| `sclc_symbol` | `sCLC` | sCLC asset id. |
| `sclc_fee_access_enabled` | `True` | Enable fee access for sCLC. |
| `sclc_fee_access_share` | `0.50` | Share of CLC stable eligible for sCLC. |
| `sclc_emission_cap_usd` | `2000.0` | sCLC emission cap per epoch. |
| `sclc_requires_insurance_target` | `True` | Require insurance target. |
| `sclc_requires_core_ops` | `True` | Require ops target. |
| `sclc_swap_window_ticks` | `4` | sCLC access cadence. |
| `sclc_swap_window_open_ticks` | `1` | sCLC access open ticks. |
| `clc_pool_always_open` | `True` | Keep CLC pool open for swaps. |

### CLC rebalancing
| Parameter | Default | Meaning |
| --- | --- | --- |
| `clc_rebalance_enabled` | `True` | Enable CLC rebalancing. |
| `clc_rebalance_interval_ticks` | `1` | Rebalance cadence. |
| `clc_rebalance_max_swaps_per_tick` | `2` | Max rebalance swaps per tick. |
| `clc_rebalance_target_stable_ratio` | `0.50` | Target stable ratio. |
| `clc_rebalance_swap_size_frac` | `0.05` | Swap size as share of CLC value. |
| `clc_rebalance_min_usd` | `25.0` | Minimum rebalance size. |

### Insurance & incidents
| Parameter | Default | Meaning |
| --- | --- | --- |
| `insurance_target_multiplier` | `0.02` | Insurance target vs voucher value. |
| `insurance_risk_weight_base` | `1.0` | Risk weight base. |
| `insurance_risk_weight_reserve_scale` | `1.0` | Reserve shortfall scale. |
| `insurance_risk_weight_min` | `0.5` | Minimum risk weight. |
| `insurance_risk_weight_max` | `3.0` | Maximum risk weight. |
| `incident_base_rate` | `0.01` | Base incident rate. |
| `incident_loss_rate` | `0.05` | Loss rate vs voucher value. |
| `incident_min_loss_usd` | `100.0` | Minimum loss. |
| `incident_haircut_cap` | `0.10` | Claim cap vs voucher value. |
| `incident_max_per_tick` | `1` | Max incidents per tick. |
| `insurance_fee_window_ticks` | `12` | Fee eligibility window. |
| `insurance_min_fee_usd` | `25.0` | Min CLC fees to be eligible. |

### Redemption
| Parameter | Default | Meaning |
| --- | --- | --- |
| `base_redeem_prob` | `0.85` | Base redemption probability. |
| `redeem_bias_swap_only` | `0.10` | Reserved (future). |
| `redeem_bias_mixed` | `0.05` | Reserved (future). |
| `redeem_bias_borrow_only` | `0.00` | Reserved (future). |

### Activity & loans
| Parameter | Default | Meaning |
| --- | --- | --- |
| `random_route_requests_per_tick` | `4` | Base attempts per pool per tick. |
| `swap_requests_budget_per_tick` | `100` | Global request budget per tick. |
| `random_request_amount_mean` | `200.0` | Legacy (unused). |
| `loan_term_weeks` | `12` | Loan term. |
| `loan_activity_period_ticks` | `12` | Spread loan activity over period. |

### Swap sizing & targeting
| Parameter | Default | Meaning |
| --- | --- | --- |
| `swap_size_mean_frac` | `0.02` | Mean swap size as share of pool value. |
| `swap_size_min_usd` | `1.0` | Minimum swap size. |
| `swap_size_max_usd` | `None` | Max swap size. |
| `swap_asset_selection_mode` | `value_weighted` | Asset_in selection. |
| `swap_limits_enabled` | `False` | Enforce cap limits. |
| `swap_target_selection_mode` | `liquidity_weighted` | Asset_out selection. |
| `swap_target_retry_count` | `2` | Target retries per asset. |
| `swap_attempts_value_scale_usd` | `500000.0` | Value-scaled attempts. |
| `swap_attempts_max_per_pool` | `4` | Attempts per pool cap. |
| `utilization_target_rate` | `0.02` | Target utilization. |
| `utilization_boost_max` | `3.0` | Max utilization boost. |

### Debug
| Parameter | Default | Meaning |
| --- | --- | --- |
| `debug_inventory` | `True` | Enable debug inventory diffs. |
