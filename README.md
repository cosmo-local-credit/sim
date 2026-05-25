# CLC Pool Network Simulator (MVP-1.1)

This repo simulates a network of agent-owned liquidity pools exchanging a stablecoin and agent-issued vouchers.
It includes multi-hop routing, loan issuance/repayment via voucher swaps, stable supply growth, NOAM routing,
NOAM clearing, and a CLC economics layer (fees + waterfall). NOAM means Network-Overlay Adaptive
Multiobjective Routing/Clearing: a network-aware scoring layer over bounded overlay/beam route search
and optional cycle clearing. A Streamlit UI (`app.py`) lets you run ticks,
inspect metrics, and sweep parameters.

The generic UI defaults are small demonstration settings. The paper-facing
RegenBond runner is calibrated from the exported Kenyan KSh/KES community
lending-pool bundle: `current` scale is 73 open lender pools, 996 producer
wallets with producer vouchers, and 462 external non-producer consumer wallets.
The empirical producer-voucher membership count is 1,247 pool slots because a
single producer voucher can be accepted in more than one pool. In paper-facing
runs, `pool` means an open automatic-swap venue with listings, limits,
inventory, fees, and routing visibility; producer, consumer, issuer, and
bondholder holdings are private wallets that can initiate or receive routes but
cannot be traversed by other agents.
The matching `connected_2x` scale doubles those counts while preserving the
empirical role mix and producer-voucher pool-overlap distribution.

**NOAM Routing (online, per request)**
1) Build/refresh the **working set** (Top‑K pools per asset, Top‑M outs per pool/asset).
2) (Optional) **Overlay**: find hub paths when enabled and the network is large enough.
3) **Beam‑A*** over the working set with network‑aware scoring:
   success probability + fee penalty + scarcity penalty + inventory‑rebalance benefit + dead‑end penalty,
   over open lender-pool venues.
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
git clone https://github.com/cosmo-local-credit/sim.git
cd sim
```

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

## Regenerative-bond Monte Carlo runner

The paper-facing Monte Carlo workflow has two layers:

- `sarafu_engine_validation` is the no-bond validation gate. It checks whether
  the real engine reproduces Sarafu-calibrated settlement, backing, return, and
  report-exposure moments before any bond claim is interpreted.
- `bond_issuer_frontier` is the current bond-issuer model. Bondholders fund
  stable principal to an issuer; the issuer deploys gross principal into
  eligible lender pools; producer own-voucher borrowing draws stable from those
  pools; bounded productive-credit feedback can create additional producer
  voucher deposits; recovered lender stable and the configured service share
  of eligible fee cash are reserved first for scheduled bond service.

Older `regenbond_lp_injection` and `--scenario all` runs remain useful for
mechanics inspection, but they are legacy evidence for the paper-facing bond
issuer frontier. They should not be read as the current bondholder return
model.

From the `sim` repo root:

```bash
python scripts/run_regenbond_monte_carlo.py \
  --scenario sarafu_engine_validation \
  --runs 100 \
  --ticks 260 \
  --seed 1 \
  --analysis-stride 13 \
  --pool-metrics-stride 13 \
  --progress-stride 13 \
  --output analysis/monte_carlo/engine_validation
```

The runner consumes the public aggregate Sarafu calibration bundle in
`analysis/sarafu_calibration/` by default and writes CSV, LaTeX, Markdown, and
PNG artifacts under the output directory. It does not require a sibling
`RegenBonds` checkout. Use `--calibration-dir ../RegenBonds/analysis` only when
intentionally testing against a local regenerated paper-analysis bundle.

For SSH/server batch verification, use:

```bash
./scripts/start_regenbond_batch_tmux.sh validation-full
tail -f analysis/monte_carlo/validation-full.log
./scripts/run_regenbond_remote_batch.sh frontier-maturity-smoke
./scripts/run_regenbond_remote_batch.sh frontier-rola-regeneration-probe
./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

The full runbook, including `tmux`, `tail -f`, and expected output files, is in
`docs/regenbond_remote_batches.md`.

Use `--scenario all` for the older baseline, regenerative-bond, and stress-test
suite. For longer runs, `--analysis-stride N` records expensive paper
diagnostics every `N` ticks while still simulating every tick.

Current bond-frontier outputs distinguish capped scheduled-payment coverage
from uncapped cash headroom. `scheduled_payment_coverage` asks whether the
bondholder received scheduled principal plus coupon due. `service_cash_headroom`
is gross historical recovery relative to scheduled due.
`available_service_cash_headroom` is the spendable proxy: paid scheduled
service plus lockbox balance plus sweepable pending recovered stable, divided
by scheduled due. These service-cash metrics include recovered lender stable
plus fee cash reserved into the bond-service lockbox. Any available headroom
above scheduled service is candidate issuer operating and risk headroom, not
proven net profit until explicit issuer costs and first-loss capital are
modeled.

The paper-facing no-bond validation gate is `sarafu_engine_validation` with
100 runs over 260 weekly ticks. The previous full gate passed before the
2026-05-25 Kenya community-pool calibration and private-wallet routing update.
After this update, rerun `validation-full` before citing full validation
statistics; use `validation-smoke` only as a fast sanity check.

Frontier runs also enable a producer debt contract cash-service layer. Producer
borrowing still starts as own-voucher-in/stable-out through an eligible lender
pool, but the borrower now owes contract cash service in addition to the
tradable voucher exposure. The current frontier default is principal-only
contract cash service (`0%` service margin) until an empirical debt-service
margin is calibrated. Stable and voucher-to-voucher obligations use the same
shared margin by default; channel-specific margin flags are available only for
explicit sensitivity/ablation runs. Issuer sustainability is reported
separately through fee service, excess recovered stable, lockbox surplus, and
available operating-surplus diagnostics.

The current `frontier-pilot` and `frontier-publication` batch targets enable
the ROLA mechanism that previously passed the 20-run low-principal probe:
primary producer voucher borrowing, voucher-loan fallback, voucher-loan
activity boost, and bounded consumer or third-party stable purchases of visible
lender-held producer vouchers. This is the current frontier configuration;
disable those flags only for explicit control or ablation runs.

The current frontier is a voucher-capable ROSCA-like credit-pool test. It does
not replay the literal Sarafu ledger, and it is not the future voucher-free
ROSCA-to-ROLA regeneration counterfactual. It starts from a Sarafu-calibrated
credit-pool substrate with stable-credit logic, borrowing rights, credit
limits, repayment obligations, producer voucher identities, lender acceptance
rules, and routing. Under those rules, the matched no-bond baseline already
produces ROLA-like voucher exchange; the frontier asks whether bond-backed
lending preserves or amplifies voucher-to-voucher value while repayment and
non-extraction guardrails hold. Field-history claims about the prior weakening
of ROLA-like practice come from practitioner/training context and literature;
the transaction data supplies the separate revival evidence. The historical
batch name `frontier-rola-regeneration-probe` should not be confused with the
future no-voucher ROSCA-to-ROLA regeneration counterfactual, which is not yet
implemented.

Conceptually, the simulator keeps two obligation layers separate. The formal
bond is a tradable debt instrument owed by the issuer to bondholders. A
producer voucher in a lender pool is a transferable redeemable commitment owed
through local fulfillment, redemption, return, or repair. Frontier tests ask
whether the formal debt layer can add catalytic liquidity while preserving or
regenerating the local voucher-obligation layer.

The current frontier defaults are calibration driven where possible:
producer debt maturity recovery uses the mature borrow-proxy value support rate
(`0.673` in the current bundle); primary producer voucher-borrowing attempts
use the recent voucher-source settlement motif share (`0.868845`, computed as
voucher-to-voucher events divided by voucher-to-voucher plus voucher-to-stable
events) as a behavioral prior, not as a controller that forces realized route
composition; visible lender-held voucher purchase demand uses a network-level
stable purchase budget of `$184.061305` per weekly tick, the direct pool-era
stable-to-voucher purchase cash mean; productive-credit voucher-source boost
coefficients are loaded from post-borrow event-window calibration. The current
activity-boost artifact estimates a neutral same-voucher source boost
(`0.0`) and neutral source-size multiplier (`1.0`): same-voucher
voucher-to-voucher source use falls in the 91-day post-borrow window, while
target-side voucher demand rises. That means the empirically supported near-term
frontier channel is stronger visible-voucher stable purchase demand, not an
extra uncalibrated producer voucher-source boost. Producer-voucher overlap uses
the empirical aggregate pool-overlap distribution. The `frontier-publication`
job is the paper-facing expansion of the earlier focused `frontier-pilot`
setup: it uses `current` and `connected_2x`, 13 coupon targets from `0%` to
`12%`, 10 positive principal ratios from `0.05` to `0.50`, and built-in
matched no-bond baselines. Earlier focused-pilot pass counts should be treated
as historical pilot evidence until the expanded grid has been reviewed under
the current calibration and private-wallet routing rules.

The Streamlit app includes a **RegenBond MC** tab that runs this same script as
a subprocess and displays the exact CLI-equivalent command. For identical
results between terminal and UI, keep the scenario, runs, ticks, seed, coupon
targets, terms, output directory, analysis stride, and optional performance caps
identical.

Frontier capacity-feedback behavior is controlled by calibration-backed config
knobs. The current aggregate productive-credit calibration splits loan-enabled
productive inflow into a stable retained share of `0.244881` and a voucher
deposit share of `0.755119`, capped at `0.189536` voucher-deposit growth per
month. This mechanism is enabled for frontier runs and is evaluated against the
matched no-bond baseline; it is disabled for the no-bond validation gate except
as reported calibration diagnostics.

## Sarafu-calibrated paper workflow

The default Streamlit tab is **Sarafu Calibrated**. It starts from the
privacy-safe Sarafu pool calibration outputs in `analysis/sarafu_calibration/`,
validates a Sarafu-like baseline, separates observed aid/grant liquidity from
counterfactual repayable-liquidity template policies, and writes
manuscript-ready tables, LaTeX snippets, PNG figures, and captions. This
template runner is background evidence; the current paper-facing bondholder
model is `bond_issuer_frontier`. In this research workspace the
authoritative paper output can still be set explicitly:

```text
../RegenBonds/analysis/monte_carlo/sarafu_calibrated/
```

When the simulator is cloned standalone, the runner uses the local public
calibration bundle and falls back to local `analysis/sarafu_calibrated/` output.

CLI equivalent:

```bash
python scripts/run_sarafu_calibrated_monte_carlo.py \
  --runs 100 \
  --ticks 52 \
  --seed 1 \
  --policies aid_baseline,broad_equal,strong_activity,weak_capacity \
  --coupon-targets 0,0.03,0.06,0.09,0.12 \
  --term 260 \
  --output ../RegenBonds/analysis/monte_carlo/sarafu_calibrated
```

The UX step buttons rerun this same command with a longer tick horizon, so
terminal and Streamlit outputs match when the displayed command is identical.

---

## Core model (agentic logic)

### Agents and pools
- The generic engine stores every actor as a `Pool` object, but the
  paper-facing topology distinguishes open pools from private wallets.
- **Open pools** are the empirical community lending pools. They are the only
  automatic-swap, routing, and NOAM-clearing venues in paper-facing runs.
- **Private wallets** belong to producers, consumers, issuers, or bondholders.
  They can initiate or receive swaps/routes, but other agents cannot traverse
  or clear through them.
- Producer wallets each issue one producer voucher asset `VCHR:<agent_id>`
  tracked by the issuer ledger. That voucher can be accepted by multiple open
  pools according to the empirical overlap calibration.
- Open pools maintain:
  - **Vault** (inventory), **listing registry** (what assets they accept), **value index** (prices in USD),
  - **swap limiter** (cap-in per window), **fee registry**, and **fee ledgers**.

### Bootstrap (initial network)
- If `economics_enabled=True`, **system pools** are created first: ops, insurance, mandates, CLC.
- If any `initial_*` counts are set (default), those exact counts are created:
  `initial_producers`, `initial_consumers`, `initial_lenders`, `initial_liquidity_providers`.
  Otherwise, `initial_pools` is used with the role mix probabilities.

### Roles and policies
**Producer wallets**
- **Goal**: hold the producer's private wallet, issue one producer voucher, use
  that voucher for borrowing, and hold stable for repayment or ordinary use.
- **Listings**: stable + the producer's own voucher. Producer wallets are not
  open swap venues for other agents.
- **Inventory**: own voucher seed; stable arrives through routes, borrowing,
  productive-credit feedback, or other explicit inflows.
- **Restrictions**: producers avoid ordinary stable-sourced swaps while debt is
  outstanding; stable can still be used for repayment and maturity settlement.
- **Redemption**: auto‑redeem any **foreign** voucher received. Producer
  wallets retain the producer's own voucher and stable.
- **Stable usage**: ordinary stable-source selection is governed by the
  configured producer stable bias and stable-reserve protection.
- **Starts with**:
  - Stable seed: `0`
  - Own voucher seed: `exp(mean=10000)`

**Lender pools**
- **Goal**: act as open community swap/lending venues; provide stable loans,
  hold producer voucher exposure, and route visible voucher and stable
  liquidity under pool limits.
- **Listings (wants)**: stable plus producer vouchers assigned by the empirical
  overlap calibration.
- **Inventory (offers)**: stable + offered assets seed.
- **Restrictions**: none on voucher↔voucher; stable can flow in/out.
- **No mint/off‑ramp**: lenders do **not** mint or burn stables/vouchers.
- **Starts with**:
  - Stable seed: **fixed** `lender_initial_stable_mean` (default `0`).
  - Offered asset seeds: `exp(mean=250)` per offered asset
- **Liquidity**: lenders only gain stable via **liquidity mandates** and **repayment swaps**;
  vouchers via **borrowing swaps** and **voucher↔voucher** swaps.

**Consumer wallets**
- **Goal**: hold a private stable wallet, spend stable to acquire producer
  vouchers, and redeem acquired vouchers.
- **Listings**: stable only. Consumers do not create tradable consumer vouchers
  in the paper-facing topology.
- **Inventory**: stable seed only; acquired producer vouchers are final route
  outputs and can be redeemed.
- **Restrictions**: ordinary stable spending is governed by source-selection
  bias and stable-reserve protection; consumers are no longer hard-forced to
  spend stable whenever they hold it.
- **Redemption**: auto‑redeem acquired producer vouchers with their issuers.
- **Starts with** (current implementation constants in `sim/engine.py`):
  - Stable seed: `exp(mean=initial_stable_per_pool_mean * 0.25)`

**Liquidity Provider (LP) pools**
- This LP/sCLC role is part of the generic CLC economics layer and legacy
  inspection scenarios. The current `bond_issuer_frontier` does not model
  bondholders as ordinary LP pools.
- **Goal**: contribute stable to the waterfall in exchange for sCLC.
- **Listings**: stable only.
- **Inventory**: stable seed only.
- **Starts with**: **fixed** `lp_initial_stable_mean` (default `400000`).
- **Contribution**: **one‑shot** (all stable) on the first waterfall tick after creation, minting sCLC
  until `lp_sclc_supply_cap` is exhausted. `lp_waterfall_contribution_rate` is ignored in current behavior.

**System pools**
- **ops**, **insurance**, **mandates**, **clc**. These are non-agent pools controlled by policy:
  - System pools are **paused** by default and do not trade.
  - **CLC pool** is a system settlement pool for explicit CLC mechanisms, not
    an open NOAM route/clearing venue.
  - **CLC stable outflow** in explicit CLC mechanisms is only allowed when the
    input is **sCLC**.

### Swap rules and limits
- **All pools list USD**; each listing has a cap-in per rolling window (`default_window_len`).
- **Cap-in limits** apply only if `swap_limits_enabled=True`.
- **Stable cap-in** uses role-specific caps (`lender_stable_cap_in`, `producer_stable_cap_in`, else `default_cap_in`).
- **Stable reserve guardrail**: swaps that take stable below `min_stable_reserve` are blocked.
- **Role constraints**:
  - Lenders are the only open swap venues for ordinary routing and NOAM
    clearing.
  - Producers with outstanding debt avoid ordinary stable-sourced swaps so
    stable remains available for repayment; repayment routes can still spend
    producer stable.
  - Consumers can spend stable under the configured source-selection bias. In
    validation/frontier-style runs this is value-weighted with stable bias and
    reserve protection rather than a hard "always spend stable" rule.
  - Producer and consumer private wallets can start or receive routes, but
    other agents cannot traverse or clear through them.
  - CLC pool requires a stable/sCLC leg in explicit CLC mechanisms, and stable
    outflow requires **sCLC** in.

### Redemption
- **Producers/Consumers** auto‑redeem any **foreign** voucher they receive.
- **Final hop vouchers** are redeemed to the issuer **only** for producer/consumer source pools.
- Redemptions **return vouchers to the issuer’s pool** and **do not burn supply** (issuer ledgers track `redeemed_total`).

---

## Routing and trading

### Swap request generation (per tick)
- Generic demo runs can let every **non-system, non-LP** actor submit route
  requests. In paper-facing runs, open lender pools are the automatic-swap
  venues; producer and consumer wallets submit their own borrowing, repayment,
  purchase, redemption, or ordinary source requests but are not intermediate
  venues.
- Per pool attempts are based on:
  - `random_route_requests_per_tick`
  - value-scaled attempts `swap_attempts_value_scale_usd`
  - capped by `swap_attempts_max_per_pool`
  - and constrained by `swap_requests_budget_per_tick` across all pools.
- Attempts are multiplied by a **utilization boost** when network volume is below `utilization_target_rate`
  (capped by `utilization_boost_max`).
- Producers also attempt **loan repayments** and **new loan issuance** according to `loan_activity_period_ticks`.
- Target assets are chosen by `swap_target_selection_mode` and retried up to `swap_target_retry_count` times.
- Producers avoid ordinary stable spending until debt is cleared; consumers use
  the configured stable-source bias. Frontier runs can preserve each
  producer/consumer wallet's stable reserve plus a voucher-value buffer before
  ordinary stable-sourced swaps.
- If a route fails at the chosen amount, the engine retries once with a smaller **fallback amount** before giving up.

### Loan mechanics (producer ↔ lender)
- **Issuance**: producers **do not mint** as part of a swap; they route
  existing own vouchers to lenders to receive USD stable or, in current
  frontier-pilot ROLA settings, other useful producer vouchers.
- **Repayment**: producers route **stable** to acquire their voucher from lenders.
  The repayment amount amortizes `loan_term_weeks` and is spread by `loan_activity_period_ticks`.
- **Bond-frontier producer debt**: producer own-voucher-in/stable-out creates a
  dated lender-pool exposure plus a contract cash-service obligation. Ordinary
  circulation can reduce the pool-level voucher exposure before a 13-week
  maturity, but it does not by itself erase cash service unless stable is
  recovered through borrower repayment, consumer purchase, third-party stable
  purchase, or contract cash-service payment. At maturity, remaining cash
  service is attempted from available producer stable and unrecovered service is
  written off under the configured recovery/default behavior. There is no
  explicit one-third monthly installment schedule in the current implementation.

### NOAM Routing (default)
NOAM is the default router (`routing_mode=noam`). It is **Network-Overlay Adaptive
Multiobjective Routing/Clearing**: a network-aware overlay + beam-style route search with
live pool validation:
Producer and consumer wallets are private source/sink wallets. They can initiate
a route and receive the final output or redeemed voucher, but they are not open
swap venues. The executable NOAM graph traverses lender pools only.
When configured for the Sarafu validation/frontier profiles, open lender pools
may execute direct voucher-to-voucher swaps if both vouchers are listed and
limits/inventory allow. If a producer or consumer route ends in a voucher, that
voucher is redeemed to the issuer's private producer wallet; it is not retained
as an open venue balance.
1) **Working set** (Top-K/Top-M)
   - `noam_topk_pools_per_asset` lender pools per asset
   - `noam_topm_out_per_pool` outputs per pool/asset
   - Adaptive caps can shrink these when the network is large.
2) **Overlay routing** (optional)
   - Uses hub assets and precomputed overlay paths to reduce search.
   - Only used when `noam_overlay_enabled=True` **and** pool count ≥ `noam_overlay_min_pools`.
3) **Bounded beam-style search** with multiobjective scoring:
   - Success probability (`noam_weight_success`)
   - Fees (`noam_weight_fee`)
   - Scarcity (`noam_weight_lambda`, `noam_scarcity_eta`)
   - Inventory benefit (`noam_weight_benefit`)
   - Dead-end penalty (`noam_weight_deadend`)
   - Amount propagation is **carry-only** (no rate propagation); validation happens at execution.
   - Benefit mode uses **inventory rebalance** (per-hop imbalance reduction).
4) **Caching + failure TTL**
   - Route cache (`noam_route_cache_*`), failure TTL (`noam_failure_ttl_ticks`)

If NOAM is disabled, the legacy BFS router can be used (`routing_mode=bfs`).

---

## NOAM Clearing (batch cycles)
NOAM Clearing runs periodically to clear feasible cycles and rebalance the network.
It is conceptually related to network obligation-clearing or netting, but in
the simulator it executes only feasible pool-exchange cycles that pass live
quotes, inventory, limits, budget, and scoring constraints.

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
- **Venue restriction**: clearing cycles execute only through lender pools. Producer and
  consumer wallets are never clearing venues.

NOAM clearing is separate from the bond-issuer quarterly clearing mechanism.
NOAM clearing is route/cycle execution inside the pool network. Bond-issuer
clearing moves eligible recovered lender stable toward scheduled issuer service
accounting, constrained by lender surplus and scheduled bondholder need.

---

## CLC economics and waterfall

### Fees
- **Pool fee**: `pool_fee_rate` applied to gross output; the fee is denominated
  in the outgoing asset, which can be stable or a voucher.
- **CLC rake**: `clc_rake_rate` is taken **from** pool fee (not additive).
- **All executed swaps** (routing and clearing) accrue pool fees and CLC rake.
- System pools have zero fees by default; agent pools use the configured pool fee + CLC rake.
- Fee ledgers are swept into the waterfall each epoch.
- Cumulative fee totals are tracked separately in metrics.
  - If `waterfall_include_pool_fees=False`, **CLC fees** flow to the waterfall; otherwise **pool fees** do.

### Waterfall inflows
- **Either pool fee ledger or CLC fee ledger** (controlled by `waterfall_include_pool_fees`; not both)
- **LP contributions** (stable inflow + sCLC mint)
- **External inflows** (used for mandates / policy injections)

Before allocation, fee assets are **converted to cash** when eligible:
- Assets in `cash_eligible_assets` are converted to USD using the pool’s value index,
  with `cash_conversion_slippage_bps` and optional `cash_conversion_max_usd_per_epoch`.
- Converted **vouchers remain in-system** (deposited to CLC); non‑convertible assets are deposited to CLC **in‑kind**.
- Converted cash is treated as a **fiat on‑ramp** for KPI accounting.

For Regenerative Bond frontier runs with `bond_return_mode=issuer_cashflow`
and `bond_service_reserve_enabled=True`, fee cash has a senior service path
before the ordinary waterfall:
- Stable-denominated fees reserve `bond_fee_service_share` into the
  bond-service lockbox until the configured lockbox target is met.
- Successfully converted voucher-denominated fees reserve the same configured
  share into the lockbox.
- Fee cash above the lockbox need proceeds through the normal waterfall;
  unconverted voucher fees remain in-kind inventory.

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
- **System settlement pool**, not an open NOAM venue.
- **Not traversed by routing/clearing**; NOAM route and clearing venues are lender pools.
- **Accepts stable for vouchers** only in explicit CLC mechanisms outside the open lending-pool route graph.
- **Rebalancing**: periodically swaps vouchers -> stable to maintain target ratio.
- **Stable outflow**: only allowed when the input asset is **sCLC**.

### sCLC fee access
- sCLC can swap into CLC at any time (always-open pool); eligibility and fee-access budgeting are tracked.
- Fee‑access minting targets **all available CLC stable** (subject to cap).
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
| `initial_pools` | `10` | Initial pool count (used only when `initial_*` counts are unset). |
| `initial_lenders` | `4` | Initial lender count. |
| `initial_producers` | `100` | Generic demo producer count; paper-facing runs override from calibration. |
| `initial_consumers` | `20` | Initial consumer count. |
| `initial_liquidity_providers` | `1` | Initial LP count. |
| `pool_growth_rate_per_tick` | `0.0` | Pool growth rate per tick. |
| `pool_growth_stride_ticks` | `4` | Stride for pool growth. |
| `max_pools` | `500` | Hard cap on active pools. |
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
| `p_liquidity_provider` | `0.0` | Probability a new agent is an LP. |
| `p_lender` | `0.25` | Probability a new agent is a lender. |
| `p_producer` | `0.50` | Probability a new agent is a producer. |
| `p_consumer` | `0.25` | Probability a new agent is a consumer. |

### Stable supply & flow
| Parameter | Default | Meaning |
| --- | --- | --- |
| `stable_symbol` | `USD` | Stablecoin asset id. |
| `initial_stable_per_pool_mean` | `2000.0` | Baseline for consumer stable seed (exp mean * 0.25). |
| `lender_initial_stable_mean` | `0.0` | Fixed stable seed for lenders (lenders get stable via mandates/repayments). |
| `lp_initial_stable_mean` | `400000.0` | Fixed stable seed for LPs. |
| `stable_inflow_per_tick` | `0.0` | Generic per-pool inflow (per month). |
| `producer_inflow_per_tick` | `0.05` | Producer inflow rate (per month). |
| `consumer_inflow_per_tick` | `0.05` | Consumer inflow rate (per month). |
| `lender_inflow_per_tick` | `0.05` | Lender inflow rate (ignored; lenders do not mint). |
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
| `voucher_inflow_share` | `0.5` | Unused (voucher inflow minting is disabled). |
| `offramps_enabled` | `True` | Enable stable offramps. |
| `offramp_rate_min_per_tick` | `0.0` | Min offramp rate. |
| `offramp_rate_max_per_tick` | `0.02` | Max offramp rate. |
| `offramp_success_ema_alpha` | `0.2` | EMA alpha for success/failure. |
| `offramp_min_attempts` | `2` | Min attempts before offramps apply. |
| `producer_offramp_rate_per_month` | `0.05` | Monthly stable cash‑out fraction (producers). |
| `consumer_offramp_rate_per_month` | `0.05` | Monthly stable cash‑out fraction (consumers). |

Notes:
- Monthly off‑ramping is **auto‑balanced** against net on‑ramping at the end of each month to stabilize total stable supply.
- Stable on‑ramps are treated as **fiat inflows**; off‑ramps (including ops burns) are treated as **fiat cash‑outs**.
- Lenders and system pools do **not** mint or off‑ramp stables or vouchers.

### Metrics & performance
| Parameter | Default | Meaning |
| --- | --- | --- |
| `metrics_stride` | `1` | Network metrics stride. |
| `pool_metrics_stride` | `1` | Pool metrics stride. |
| `max_active_pools_per_tick` | `None` | Cap active pools sampled per tick (None = no cap). |
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
| `noam_topk_refresh_ticks` | `4` | Refresh Top-K/Top-M. |
| `noam_dynamic_caps_enabled` | `True` | Adaptive caps. |
| `noam_dynamic_cap_reference_pools` | `50` | Reference pool count for scaling. |
| `noam_dynamic_min_topk` | `4` | Min Top-K. |
| `noam_dynamic_min_topm` | `4` | Min Top-M. |
| `noam_dynamic_min_beam` | `16` | Min beam width. |
| `noam_edge_cap_per_state` | `30` | Edge expansion cap per state. |
| `noam_dynamic_min_edge_cap` | `20` | Min edge cap. |
| `noam_overlay_enabled` | `True` | Enable overlay graph. |
| `noam_hub_asset_count` | `60` | Hub asset count. |
| `noam_hub_depth` | `2` | Hub depth. |
| `noam_hub_candidate_limit` | `10` | Hub candidate limit. |
| `noam_overlay_top_r_paths` | `3` | Overlay paths per query. |
| `noam_overlay_max_hops` | `3` | Max overlay hops. |
| `noam_overlay_refresh_ticks` | `200` | Overlay refresh stride. |
| `noam_overlay_min_pools` | `200` | Overlay min pools. |
| `noam_clearing_enabled` | `True` | Enable NOAM clearing. |
| `noam_clearing_stride_ticks` | `4` | Clearing cadence. |
| `noam_clearing_max_cycles` | `200` | Max cycles per run. |
| `noam_clearing_max_hops` | `4` | Max hops per cycle. |
| `noam_clearing_edge_cap_per_asset` | `16` | Edge cap per asset in clearing. |
| `noam_clearing_safety_factor` | `0.8` | Safety multiplier on clearing cycle sizes. |
| `noam_clearing_budget_usd` | `25000.0` | Base clearing budget. |
| `noam_clearing_budget_share` | `0.01` | Budget share of network value. |
| `noam_clearing_min_cycle_value_usd` | `1.0` | Min cycle value. |
| `noam_clearing_lenders_only` | `True` | Restrict clearing to lender pools. |
| `noam_clearing_include_clc` | `False` | Legacy option; CLC is not part of the open NOAM venue set. |
| `noam_clearing_lender_edge_bonus` | `0.5` | Bonus to lender edges in clearing scoring. |
| `noam_success_ema_alpha` | `0.2` | EMA alpha for success. |
| `noam_success_min` | `0.05` | Min success. |
| `noam_success_max` | `0.98` | Max success. |
| `noam_weight_success` | `1.2` | Success weight. |
| `noam_weight_fee` | `1.0` | Fee weight. |
| `noam_weight_lambda` | `1.2` | Scarcity weight. |
| `noam_weight_benefit` | `1.5` | Benefit weight. |
| `noam_weight_deadend` | `1.0` | Dead-end weight. |
| `noam_clc_edge_bonus` | `0.75` | Legacy CLC edge bonus; inactive when CLC is excluded from NOAM venues. |
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
| `lender_voucher_cap_in` | `10000.0` | Cap-in for lender vouchers. |
| `lender_stable_cap_in` | `100000.0` | Cap-in for lender stable. |
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
| `core_ops_budget_usd` | `20000.0` | Ops cap per epoch. |
| `insurance_max_topup_usd` | `100000.0` | Insurance cap per epoch. |
| `liquidity_mandate_share` | `0.50` | Share of remaining cash to mandates. |
| `liquidity_mandate_bootstrap_share` | `1.0` | Bootstrap mandate share before normal waterfall. |
| `liquidity_mandate_bootstrap_epochs` | `1` | Bootstrap epochs for full mandate routing. |
| `liquidity_mandate_max_usd` | `0.0` | Mandate cap per epoch (0 = no cap). |
| `liquidity_mandate_mode` | `lender_liquidity` | Mandate distribution mode. |
| `liquidity_mandate_activity_window_ticks` | `12` | Activity window for mandates. |
| `liquidity_mandate_max_per_pool_usd` | `2000.0` | Per-pool cap (ignored in lender_liquidity). |
| `waterfall_alpha_ops_share` | `0.20` | Legacy remainder share (ops). |
| `waterfall_beta_liquidity_share` | `0.40` | Legacy remainder share (liquidity). |
| `waterfall_gamma_insurance_share` | `0.40` | Legacy remainder share (insurance). |
| `lp_waterfall_contribution_rate` | `1.0` | LP contribution rate (ignored; LP contributes once). |
| `lp_sclc_supply_cap` | `100000000.0` | sCLC supply cap for LPs. |
| `sclc_symbol` | `sCLC` | sCLC asset id. |
| `sclc_fee_access_enabled` | `True` | Enable fee access for sCLC. |
| `sclc_fee_access_share` | `0.50` | Share of CLC stable eligible for sCLC. |
| `sclc_emission_cap_usd` | `1_000_000_000.0` | sCLC emission cap per epoch. |
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
| `clc_rebalance_max_swaps_per_tick` | `10` | Max rebalance swaps per tick. |
| `clc_rebalance_target_stable_ratio` | `1.0` | Target stable ratio. |
| `clc_rebalance_swap_size_frac` | `0.25` | Swap size as share of CLC value. |
| `clc_rebalance_min_usd` | `1.0` | Minimum rebalance size. |

### Insurance & incidents
| Parameter | Default | Meaning |
| --- | --- | --- |
| `insurance_target_multiplier` | `0.05` | Insurance target vs voucher value. |
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
| `loan_term_weeks` | `4` | Loan term. |
| `loan_activity_period_ticks` | `4` | Spread loan activity over period. |

### Swap sizing & targeting
| Parameter | Default | Meaning |
| --- | --- | --- |
| `swap_size_mean_frac` | `0.02` | Mean swap size as share of pool value. |
| `swap_size_min_usd` | `1.0` | Minimum swap size. |
| `swap_size_max_usd` | `None` | Max swap size. |
| `swap_asset_selection_mode` | `value_weighted` | Asset_in selection. |
| `swap_limits_enabled` | `True` | Enforce cap limits. |
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
