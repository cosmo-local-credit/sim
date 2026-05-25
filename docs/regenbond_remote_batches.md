# RegenBond Remote Batch Verification

This repo is self-contained for public RegenBond Monte Carlo verification. A
clean clone contains the simulator, the paper-facing runners, and the
privacy-safe Sarafu calibration bundle under `analysis/sarafu_calibration/`.

The calibration bundle is aggregate/anonymized. It does not include raw Sarafu
transactions, addresses, report text, GPS data, pool labels, or pool IDs.

## Quick Start

Run this once on the remote server:

```bash
git clone https://github.com/cosmo-local-credit/sim.git
cd sim
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

After a fresh clone, confirm that the public calibration bundle includes the
revised aggregate tables:

```bash
ls analysis/sarafu_calibration/producer_deposit_calibration.csv
ls analysis/sarafu_calibration/productive_credit_calibration.csv
ls analysis/sarafu_calibration/debt_removal_calibration.csv
ls analysis/sarafu_calibration/fee_conversion_calibration.csv
ls analysis/sarafu_calibration/quarterly_clearing_calibration.csv
ls analysis/sarafu_calibration/route_substitution_diagnostics.csv
ls analysis/sarafu_calibration/unit_normalization_calibration.csv
```

The remote server normally does not run the private empirical calibration
pipeline. That pipeline uses `RegenBonds/cleaned_data/data/csv` and should be
run in the private research workspace, then exported into this repo as the
public-safe `analysis/sarafu_calibration/` bundle before pushing.

The current exported calibration is the Kenya KES/KSh community-pool cohort:
`73` open pools, `996` unique producer-voucher wallets,
`1,247` accepted-voucher member slots, and `462` recommended external
non-producer consumer wallets. Stable-side pool interaction has `950`
address-pool slots; that is an interaction count, not the consumer-wallet
count. Producer and consumer wallets are private source/sink wallets, while
open routing and NOAM clearing venues are pools only. In code, these open
automatic-swap venues still use the `lender` role; in paper-facing language
they are simply `pools`. Producer, consumer, issuer, and bondholder holdings
are `wallets`: they can initiate or receive swaps and routes, but other agents
cannot swap through them.

Current paper-facing interpretation uses `sarafu_engine_validation` as the
no-bond gate and `bond_issuer_frontier` as the issuer/pool frontier.
The frontier deploys gross bond principal directly into eligible pools,
uses producer own-voucher-in/stable-out borrowing, and evaluates scheduled
bondholder payment separately from recovered-stable cash headroom. Current
frontier runs also include bounded productive-credit feedback: loan-enabled
productive inflow can increase stable retained and producer voucher deposits
within aggregate calibration shares and growth caps.

Validation, matched no-bond frontier baselines, `frontier-pilot`, and
`frontier-publication` now use the same current-network routing profile:
`max_hops=3`, `noam_max_hops=3`, NOAM overlay enabled, NOAM clearing enabled,
and quarterly NOAM clearing every 13 weekly ticks. `connected_2x` remains
available through `NETWORK_SCALES=connected_2x`, but it is no longer a
paper-facing default.

The current ROLA/frontier-pilot configuration also enables producer primary
voucher borrowing, voucher-loan fallback, voucher-loan activity boost, and
bounded consumer/third-party stable purchases of pool-held producer vouchers.
The earlier 20-run `frontier-rola-regeneration-probe` passed all tested
current-scale low-principal cells from `0` to `0.05` principal ratio with
scheduled bond service paid and voucher-to-voucher circulation preserved. That
result remains useful model-development evidence, but validation and frontier
artifacts should be regenerated after the current cohort and private-wallet
routing update.

Terminology note: `frontier-rola-regeneration-probe` is a historical batch
name. In the current documentation it means a voucher-capable ROSCA-like
credit-pool mechanism probe. It starts from a Sarafu-calibrated substrate with
stable-credit logic, borrowing rights, credit limits, repayment obligations,
producer voucher identities, pool acceptance rules, and routing. The
separate no-voucher ROSCA-to-ROLA regeneration counterfactual is future work
and is not run by this target.

Current frontier defaults are calibration-backed where possible:

- producer debt maturity recovery uses the mature borrow-proxy value support
  rate, currently `0.673`;
- producer credit attempts reserve a calibrated slice of the route-request
  budget for producer wallets before ordinary pool traffic. The current
  voucher-source budget share is `0.936905`;
- producer primary voucher-borrowing attempts use the recent voucher-source
  motif share, currently `0.868845`, as a decision prior unless explicitly
  overridden. It does not force the realized route mix to match empirical
  motif shares;
- producer-voucher overlap uses the empirical aggregate pool-degree
  distribution;
- external stable-to-voucher purchase capacity is seeded in private
  producer/consumer wallets from the empirical stable-to-voucher purchase
  value, not injected into pools. The current 260-week default is `2`
  purchase attempts per tick, external-consumer share `0.977324`, inventory
  share `0.05`, and no per-tick purchase-budget onramp unless explicitly
  overridden. Private producer/consumer stable is not swept out in validation;
  frontier runs scale the initial private stable balances by network size;
- productive-credit voucher-source boost coefficients are loaded from the
  post-borrow event-window calibration artifact. In the current artifact the
  same-voucher source boost is `0.0` and the source-size multiplier is `1.0`,
  because same-voucher voucher-to-voucher source activity does not increase in
  the 91-day post-borrow window. Target-side voucher demand does increase, so
  the calibrated purchase-demand budget carries that empirical signal instead
  of adding an unsupported source boost;
- frontier runs treat producer/consumer stable balances as transactional
  money, not savings reserves, so ordinary stable-spend reserve protection is
  off by default;
- the main pilot uses fee-service share `1.0`: stable swap fees reserve into the
  bond-service lockbox first; voucher swap fees move to the CLC fee-conversion
  wallet, which attempts to route them through open pools into stable for
  additional bond-service reservation. Only post-service headroom should be read
  as available for mandates, insurance, operations, or other discretionary uses.

For later runs:

```bash
cd ~/sim
git pull
.venv/bin/python -m pip install -r requirements.txt
```

## Fresh Remote Start For A Model Redo

If you are starting over on the remote, a clean clone is usually simpler than
cleaning old output directories:

```bash
cd ~
git clone https://github.com/cosmo-local-credit/sim.git sim
cd ~/sim
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If `~/sim` already exists and you want to keep the old run for comparison, use
a separate directory:

```bash
cd ~
git clone https://github.com/cosmo-local-credit/sim.git sim-revised
cd ~/sim-revised
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run a local smoke on the remote checkout before long jobs:

```bash
.venv/bin/python -m unittest discover -s tests
RUNS=2 TICKS=4 WORKERS=2 RESUME=0 ./scripts/run_regenbond_remote_batch.sh validation-smoke
```

For the official redo, use the standard `~/sim` checkout and standard output
directories because frontier jobs read the validation gate from
`analysis/monte_carlo/engine_validation/engine_validation_summary.csv`.

Recommended redo sequence:

```bash
WORKERS=15 RESUME=0 ./scripts/start_regenbond_batch_tmux.sh validation-1mo
WORKERS=15 RESUME=0 ./scripts/start_regenbond_batch_tmux.sh validation-smoke
WORKERS=15 RESUME=0 ./scripts/start_regenbond_batch_tmux.sh validation-pilot
WORKERS=15 RESUME=0 ./scripts/start_regenbond_batch_tmux.sh validation-full
WORKERS=15 RESUME=0 ./scripts/start_regenbond_batch_tmux.sh frontier-smoke
WORKERS=15 RESUME=0 ./scripts/start_regenbond_batch_tmux.sh frontier-maturity-smoke
WORKERS=15 RESUME=0 ./scripts/start_regenbond_batch_tmux.sh frontier-rola-regeneration-probe
WORKERS=15 RESUME=0 ./scripts/start_regenbond_batch_tmux.sh frontier-pilot
```

Run `frontier-pilot` only after `frontier-maturity-smoke` and
`frontier-rola-regeneration-probe` pass. Run `frontier-publication` after the
validation gate and pilot checks look correct. `frontier-publication` is the
paper-facing expansion of the `frontier-pilot` setup used so far, with a denser
coupon/principal grid.

Run batches from the repo root. The wrapper defaults to parallel, resumable
execution:

```bash
./scripts/run_regenbond_remote_batch.sh validation-smoke
./scripts/run_regenbond_remote_batch.sh frontier-smoke
```

Detached `tmux` execution is usually better for long jobs:

```bash
./scripts/start_regenbond_batch_tmux.sh validation-full
tail -f analysis/monte_carlo/validation-full.log
```

## Job Sequence

Use this order for paper-facing work:

```bash
./scripts/run_regenbond_remote_batch.sh validation-1mo
./scripts/run_regenbond_remote_batch.sh validation-smoke
./scripts/run_regenbond_remote_batch.sh validation-pilot
./scripts/run_regenbond_remote_batch.sh validation-full
./scripts/run_regenbond_remote_batch.sh frontier-smoke
./scripts/run_regenbond_remote_batch.sh frontier-maturity-smoke
./scripts/run_regenbond_remote_batch.sh frontier-rola-regeneration-probe
./scripts/run_regenbond_remote_batch.sh frontier-pilot
./scripts/run_regenbond_remote_batch.sh frontier-publication
```

Recommended gate:

1. Run `validation-smoke` after code or dependency changes.
2. Run `validation-pilot` before spending time on `validation-full`.
3. Run `validation-full` and confirm
   `analysis/monte_carlo/engine_validation/engine_validation_summary.csv`
   reports `status=pass`.
4. Run `frontier-smoke` and inspect frontier output structure.
5. Run `frontier-maturity-smoke` and confirm the full-term lockbox guard passes.
   If the two-run smoke is borderline, rerun it with `RUNS=10` into a separate
   output directory before spending time on the full pilot.
6. Run `frontier-rola-regeneration-probe` and inspect productive-credit
   feedback, primary voucher borrowing, stable purchase demand,
   matched-baseline voucher-to-voucher circulation, and consumer/community
   stress deltas.
7. Run `frontier-pilot`; this target now enables the same ROLA/purchase
   mechanism by default.
8. Run `frontier-publication` only after validation and pilot outputs look
   correct. `frontier-publication` uses the same paper `frontier-pilot` setup
   with more coupon/principal cells.

After any calibration, accounting, routing, cashflow, or credit-behavior model
change, rerun all validation jobs before treating frontier outputs as
paper-facing. Old frontier outputs are not comparable across those model
revisions.

## Batch Jobs

All jobs write under `analysis/monte_carlo/` unless `OUTPUT_ROOT` or `OUTPUT`
is set.

| Job | Purpose | Default size | Output directory | Log when using tmux helper |
| --- | --- | ---: | --- | --- |
| `validation-1mo` | Fast current-engine sanity check | 100 runs, 4 ticks | `analysis/monte_carlo/engine_validation_1mo_test/` | `analysis/monte_carlo/validation-1mo.log` |
| `validation-smoke` | Short validation smoke | 5 runs, 52 ticks | `analysis/monte_carlo/engine_validation_smoke/` | `analysis/monte_carlo/validation-smoke.log` |
| `validation-pilot` | Full-horizon validation pilot | 20 runs, 260 ticks | `analysis/monte_carlo/engine_validation_20run/` | `analysis/monte_carlo/validation-pilot.log` |
| `validation-full` | Paper-facing validation gate | 100 runs, 260 ticks | `analysis/monte_carlo/engine_validation/` | `analysis/monte_carlo/validation-full.log` |
| `frontier-smoke` | Small bond-frontier structure check | 5 runs, 52 ticks, current scale | `analysis/monte_carlo/bond_issuer_frontier_smoke/` | `analysis/monte_carlo/frontier-smoke.log` |
| `frontier-maturity-smoke` | Fast full-maturity lockbox check | 2 runs by default, 260 ticks, current scale; use `RUNS=10` for the pre-pilot gate | `analysis/monte_carlo/bond_issuer_frontier_maturity_smoke/` | `analysis/monte_carlo/frontier-maturity-smoke.log` |
| `frontier-feedback-probe` | Focused capacity-feedback principal probe | 2 runs, 260 ticks, current scale | `analysis/monte_carlo/bond_issuer_frontier_feedback_probe/` | `analysis/monte_carlo/frontier-feedback-probe.log` |
| `frontier-rola-regeneration-probe` | Current-scale low-principal ROLA/voucher-purchase probe | 5 runs by default, 260 ticks, current scale | `analysis/monte_carlo/bond_issuer_frontier_rola_regeneration_probe/` | `analysis/monte_carlo/frontier-rola-regeneration-probe.log` |
| `frontier-pilot` | Full-horizon focused frontier pilot | 20 runs, 260 ticks, current scale | `analysis/monte_carlo/bond_issuer_frontier_pilot/` | `analysis/monte_carlo/frontier-pilot.log` |
| `frontier-publication` | 100-run paper-facing expansion of the `frontier-pilot` setup with a denser coupon/principal grid | 100 runs, 260 ticks, current scale, 130 positive-principal cells | `analysis/monte_carlo/bond_issuer_frontier/` | `analysis/monte_carlo/frontier-publication.log` |

`frontier-*` jobs read the full validation gate at
`analysis/monte_carlo/engine_validation/engine_validation_summary.csv`. If it
is missing or `review`, the frontier still runs but marks outputs non-final. If
it is `fail`, paper-facing frontier execution is refused.

Current frontier outputs should include `scheduled_payment_coverage_*`,
`service_cash_headroom_*`, and `available_service_cash_headroom_*`.
Scheduled-payment coverage is capped at 1.0 because it measures paid scheduled
principal plus coupon divided by scheduled due. Gross service-cash headroom is
uncapped because it measures cumulative historical eligible recovery relative
to scheduled due. Available service-cash headroom is the spendable proxy:
scheduled service paid plus lockbox balance plus sweepable pending recovered
stable, divided by scheduled due. Excess available headroom above scheduled
service is candidate issuer operating and risk-capital headroom, not proven
profit until explicit issuer costs are modeled.

Current frontier defaults also enable producer debt contract cash service.
Until an empirical debt-service margin is calibrated, the default shared margin
is `PRODUCER_DEBT_CONTRACT_SERVICE_MARGIN_RATE=0.0`, meaning borrowers owe
principal cash service but no additional modeled interest/service charge.
Stable and voucher-to-voucher obligations use the same shared margin by
default. Channel-specific overrides remain available only for explicit
sensitivity/ablation runs. Issuer sustainability is measured separately from
borrower service margins through fee service, excess recovered stable, lockbox
surplus, and operating-surplus diagnostics.

The current `frontier-pilot` defaults also enable the ROLA mechanism that
passed the low-principal probe:

```text
ENABLE_PRODUCER_VOUCHER_LOAN_FALLBACK=1
ENABLE_PRODUCER_VOUCHER_LOAN_ACTIVITY_BOOST=1
ENABLE_PRODUCER_PRIMARY_VOUCHER_BORROWING=1
# Leave unset for the calibrated defaults, currently 0.868845 primary
# voucher-borrowing share and 0.936905 producer-credit route-budget share.
# PRODUCER_PRIMARY_VOUCHER_BORROWING_ATTEMPT_SHARE=0.868845
# PRODUCER_CREDIT_REQUEST_BUDGET_SHARE=0.936905
ENABLE_LENDER_VOUCHER_PURCHASE_DEMAND=1
# Leave these unset for calibrated defaults: currently 2 attempts per tick,
# consumer share 0.977324, inventory share 0.05, and no per-tick purchase
# budget onramp. Private consumer/producer stable balances are seeded from
# empirical stable-to-voucher purchase value instead.
# LENDER_VOUCHER_PURCHASE_ATTEMPTS_PER_TICK=2
# LENDER_VOUCHER_PURCHASE_CONSUMER_SHARE=0.977324
# LENDER_VOUCHER_PURCHASE_INVENTORY_SHARE=0.05
# LENDER_VOUCHER_PURCHASE_STABLE_BUDGET_USD_PER_TICK=0
```

These remain ordinary environment overrides. Set them explicitly only for
control/ablation runs or if you intentionally want to override the current
frontier-pilot mechanism.

## Calibration Bundle

Public runs use privacy-safe aggregate calibration files in
`analysis/sarafu_calibration/`. The bundle contains no raw Sarafu IDs,
addresses, transaction hashes, GPS data, report text, or pool labels.

The revised Monte Carlo also reads aggregate tables for:

- producer stable and own-voucher deposit proxies;
- productive-credit return timing and repayment lag;
- productive-credit voucher-deposit feedback shares and growth caps;
- stable-to-voucher debt-removal purchase motifs;
- voucher-fee-to-stable conversion capacity;
- fee-service reservation into the bond-service lockbox;
- quarterly pool clearing capacity;
- route-substitution diagnostics.
- unit normalization for KES/KSh vouchers against USD stable, including the
  simulator convention `1 voucher = 1 KSh`.

Current topology and flow semantics:

- empirical community pools are modeled as open pools;
- producers and consumers have private wallets that can initiate or receive
  routes but are not traversable NOAM/clearing venues;
- open pools can execute direct voucher-to-voucher swaps when both vouchers are
  listed and limits/inventory allow; private wallets remain non-routable;
- ordinary activity is decision-based: repeat-partner attempts first try known
  sticky routes or top affinity buddies, while exploration attempts search for
  new targets;
- consumer stable-to-voucher purchases are generated by the calibrated purchase
  process, not by generic ordinary stable-spend route attempts;
- producer wallets start with their own vouchers; those private seed vouchers
  do not create credit capacity until deposited into pools;
- producer deposits in pools create credit capacity and lending limits;
  they are not treated as outstanding debt until the producer executes a
  stable-loan or voucher-loan route against the pool;
- calibrated producer voucher and stable deposits are transferred into assigned
  pools, with producer voucher acceptance following the empirical
  multi-pool overlap distribution;
- private producer/consumer stable balances are seeded from empirical
  stable-to-voucher purchase value, split by the calibrated external-consumer
  versus producer-self event mix;
- validation/no-bond runs include historical philanthropic/programmatic stable
  backing as pool deposits, while bond-frontier runs omit that
  historical stable backing and use bond principal as the modeled pool
  stable injection;
- the validation backing-liquidity moment is compared against actual historical
  stable backing injected into open pools. Private producer/consumer stable
  seeded for voucher-purchase behavior remains a separate diagnostic and is
  not pool backing;
- the frontier keeps the same private-wallet, deposit, swap, and 2% fee
  mechanics, plus issuer/bondholder accounting and bond capital deployment.

Empirical integrity audit:

```bash
cd /home/wor/src/ge/clc
python3 RegenBonds/analysis/producer_voucher_integrity_audit.py \
  --sim-output-dir sim/analysis/monte_carlo/behavior_alignment_smoke52
```

The tracked summary is
`RegenBonds/analysis/producer_voucher_integrity_summary.csv`. The local detail
file is `RegenBonds/analysis_outputs/producer_voucher_integrity_detail.local.csv`
and contains raw voucher addresses for audit only. The audit makes the
pool/wallet boundary explicit: mints start in producer wallets, deposits move
assets into pools, credit limits are pool-token records, and debt is proxied by
swap-originated producer voucher exposure above deposit backing.

These route-substitution diagnostics are scenario anchors, not observed
failed-route denominators. Regenerate the public bundle from the private
research workspace after any empirical calibration change:

```bash
python scripts/export_public_sarafu_calibration.py \
  --source ../RegenBonds/analysis \
  --output analysis/sarafu_calibration
```

When the calibration bundle or model-relevant CLI flags change, old shards are
ignored automatically because the shard config hash no longer matches.

## Parallelism And Resume

The batch wrapper passes these options to `scripts/run_regenbond_monte_carlo.py`
for every job:

```text
--workers auto
--resume
--partial-aggregate-stride 1
```

`--workers auto` means:

```text
min(cpu_count - 1, 8), with a floor of 1
```

The cap at `8` is deliberately conservative for shared or memory-constrained
VPS runs, especially for `connected_5x` frontier cells. On a dedicated 16-CPU
server with enough RAM, use an explicit worker count instead of `auto`:

```bash
WORKERS=12 ./scripts/start_regenbond_batch_tmux.sh frontier-pilot
WORKERS=15 ./scripts/start_regenbond_batch_tmux.sh frontier-pilot
```

`WORKERS=12` leaves more headroom for SSH, tmux, logging, and aggregation.
`WORKERS=15` is reasonable when the server is dedicated to the batch and memory
use remains comfortable in `htop` or `free -h`.

Override worker count:

```bash
WORKERS=4 ./scripts/run_regenbond_remote_batch.sh frontier-pilot
MONTE_CARLO_WORKERS=4 ./scripts/run_regenbond_remote_batch.sh validation-full
```

Disable resume and recompute all shards:

```bash
RESUME=0 ./scripts/run_regenbond_remote_batch.sh frontier-smoke
```

Write shards somewhere else:

```bash
SHARD_DIR=/mnt/batch/shards/frontier-pilot \
  ./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Write partial aggregate CSVs less often:

```bash
PARTIAL_AGGREGATE_STRIDE=10 ./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Resume behavior:

- Completed matching shards are skipped on rerun.
- Missing, corrupt, failed, or config-mismatched shards are recomputed.
- Failed worker shards write a failed `manifest.json` with the traceback.
- The process exits nonzero if required shards remain failed or incomplete.
- Existing pre-shard logs cannot be converted into resumable shards; resume
  applies to runs started after the shard runner was introduced.

## Output Files During Runs

Each output directory contains final files and, while the job is running,
partial files:

```text
*.partial.csv
_shards/
```

Examples:

```text
analysis/monte_carlo/engine_validation_20run/engine_validation_run_summary.partial.csv
analysis/monte_carlo/bond_issuer_frontier_pilot/bond_issuer_frontier_safety.partial.csv
analysis/monte_carlo/bond_issuer_frontier_pilot/_shards/
```

Final CSV/LaTeX/PNG/Markdown filenames are unchanged. Final PNG, LaTeX, and
Markdown artifacts are written only after all required shards complete.

Validation shard layout:

```text
<output>/_shards/validation/validation_run_000001/
  manifest.json
  bond_rows.csv
  network_rows.csv
  failure_rows.csv
  summary_rows.csv
```

Frontier shard layout:

```text
<output>/_shards/frontier/frontier_grid_current_r0p05_c0_s0p5/
  manifest.json
  rows.csv
  runs/run_000001/manifest.json
  runs/run_000001/summary.csv
```

Frontier cells are split by:

```text
(network_scale, principal_ratio, coupon_target, bond_fee_service_share)
```

The parent process aggregates completed cell shards into the existing frontier
CSV outputs.

## Running Detached

Start a detached job:

```bash
./scripts/start_regenbond_batch_tmux.sh validation-full
```

The helper starts a detached `tmux` session named `regenbond` and writes:

```text
analysis/monte_carlo/<job>.log
```

The helper explicitly forwards the same batch environment used by the direct
runner, including `WORKERS`, `RESUME`, `CALIBRATION_DIR`, `OUTPUT_ROOT`,
frontier grid overrides, and route-success settings.

Check progress:

```bash
tail -f analysis/monte_carlo/validation-full.log
tail -f analysis/monte_carlo/frontier-pilot.log
```

Attach only if needed:

```bash
tmux attach -t regenbond
```

Use another session or log:

```bash
SESSION=regenbond-frontier LOG_FILE=analysis/monte_carlo/frontier-pilot.log \
  ./scripts/start_regenbond_batch_tmux.sh frontier-pilot
```

Manual `tmux` equivalent:

```bash
tmux new -s regenbond
./scripts/run_regenbond_remote_batch.sh validation-full 2>&1 | tee analysis/monte_carlo/validation-full.log
```

Detach with `Ctrl-b d`; reattach with:

```bash
tmux attach -t regenbond
```

If SSH drops:

```bash
cd ~/sim
tmux ls
tail -f analysis/monte_carlo/validation-full.log
```

## Parameters

Common wrapper parameters:

| Env var | Meaning | Default |
| --- | --- | --- |
| `WORKERS` | Worker count passed as `--workers`; use `auto` or an integer. | `auto` |
| `MONTE_CARLO_WORKERS` | Alias used only when `WORKERS` is unset. | unset |
| `RESUME` | `1`/default resumes matching shards; `0`, `false`, or `no` recomputes. | `1` |
| `SHARD_DIR` | Override shard root; otherwise each output uses `<output>/_shards`. | unset |
| `PARTIAL_AGGREGATE_STRIDE` | Write partial aggregates after every N completed jobs. | `1` |
| `RUNS` | Override default runs for the selected batch job. | job-specific |
| `TICKS` | Override horizon ticks. One tick is one week. | job-specific |
| `SEED` | Base random seed. | validation default or `1` for frontier |
| `OUTPUT_ROOT` | Parent directory for default job outputs. | `analysis/monte_carlo` |
| `OUTPUT` | Exact output directory for this run. | job-specific under `OUTPUT_ROOT` |
| `PYTHON_BIN` | Python interpreter. Wrapper prefers `.venv/bin/python`. | auto |
| `CALIBRATION_DIR` | Calibration bundle directory. | `analysis/sarafu_calibration` |
| `ANALYSIS_STRIDE` | Diagnostic recording stride. | `13` in wrapper |
| `POOL_METRICS_STRIDE` | Pool metric recording stride. | `13` in wrapper |
| `PROGRESS_STRIDE` | Per-run progress log stride. | `13` in wrapper |
| `DRY_RUN` | `1` prints the resolved Python command after calibration-file checks without running it. | `0` |

Routing and hotspot controls are common to validation and frontier. Leave them
unset for the paper-facing current-network profile.

| Env var | Meaning | Default |
| --- | --- | --- |
| `MAX_HOPS` | Generic route max-hop override. | `3` |
| `NOAM_MAX_HOPS` | NOAM route max-hop override. | `3` |
| `NOAM_OVERLAY_ENABLED` | `1` enables the NOAM overlay, `0` disables it. | `1` |
| `NOAM_CLEARING_ENABLED` | `1` enables NOAM clearing, `0` disables it. | `1` |
| `NOAM_CLEARING_STRIDE_TICKS` | NOAM clearing cadence in weekly ticks. | `13` |
| `NOAM_CLEARING_MAX_CYCLES` | Cap on clearing cycles per clearing epoch. | config default |
| `NOAM_CLEARING_MAX_HOPS` | Cap on clearing cycle path length. | config default |
| `NOAM_CLEARING_EDGE_CAP_PER_ASSET` | Cap on clearing candidate edges per asset. | config default |
| `DECISION_BASED_ACTIVITY_ENABLED` | `1` uses repeat-partner/exploration route-attempt scheduling; `0` restores the older generic route-attempt behavior. | `1` |
| `REPEAT_PARTNER_ROUTE_SHARE` | Share of ordinary route attempts that prefer known sticky/buddy partners when available. | `0.70` |
| `AFFINITY_BUDDY_MIN_COUNT` | Minimum known partners required before buddy-direct repeat routing can engage. | `1` |
| `SWAP_SUSTAIN_MAX_EXTRA_ATTEMPTS` | Cap on extra sustain attempts. | calibrated |
| `SWAP_SUSTAIN_MAX_ROUNDS` | Cap on sustain passes per tick. | calibrated |
| `SWAP_SUSTAIN_ATTEMPTS_PER_MISSING_SWAP` | Missing-swap sustain multiplier. | calibrated |
| `VOUCHER_FEE_CONVERSION_MAX_SWAPS_PER_EPOCH` | Cap on routed voucher-fee conversion swaps per waterfall epoch. | config default |
| `VOUCHER_FEE_CONVERSION_MAX_USD_PER_EPOCH` | Optional USD cap on routed voucher-fee conversion per waterfall epoch. | unset |

Frontier-specific parameters:

| Env var | Meaning | Default |
| --- | --- | --- |
| `NETWORK_SCALES` | Comma-separated frontier scales. | job-specific |
| `PRINCIPAL_RATIOS` | Comma-separated principal/eligible-capacity ratios. | job-specific |
| `COUPON_TARGETS` | Comma-separated annual coupon targets. | job-specific |
| `BOND_FEE_SERVICE_SHARES` | Comma-separated eligible fee/service shares. | job-specific |
| `CERTIFICATION_POLICY` | Eligible-pool policy used by the current frontier code. | `strong_moderate_capped` |
| `FRONTIER_MODE` | `grid` or `adaptive`. | `grid` |
| `FRONTIER_REFINEMENT_ROUNDS` | Adaptive midpoint rounds; ignored unless `FRONTIER_MODE=adaptive`. | `0` |
| `ROUTE_SUCCESS_MODE` | `diagnostic`, `relative`, or `absolute`. | `diagnostic` |
| `ROUTE_SUCCESS_FLOOR` | p05 route-success safety floor, binding only when `ROUTE_SUCCESS_MODE=absolute`. | `0.85` |
| `BOND_TERM` | Frontier term in ticks. Do not use `TERM`. | `260` |
| `ENABLE_ORDINARY_STABLE_SPEND_PROTECTION` | Optional control run: restore producer/consumer stable reserve and voucher-buffer preservation for ordinary stable-source spending. | `0` |
| `PRODUCER_CREDIT_REQUEST_BUDGET_SHARE` | Optional control run: override the calibrated producer-wallet credit-attempt share of each tick's route-request budget. | calibrated |

Do not use `TERM` for bond terms; shells and `tmux` use `TERM` for terminal
type values such as `tmux-256color`.

## Route Success Modes

`ROUTE_SUCCESS_MODE=diagnostic` is the revised frontier default. It records
route reliability but does not reject a cell solely because the old absolute
`0.85` p05 route-success floor is missed. This is useful because real users may
substitute a different ordinary purchase when one target route fails, while
borrowing, producer self-repayment, and fee conversion remain fixed-target
routes.

`ROUTE_SUCCESS_MODE=relative` treats route success as a degradation check
against the matched no-bond baseline. This is useful when scaled networks have
different baseline route reliability.

`ROUTE_SUCCESS_MODE=absolute` restores the older hard floor behavior:

```bash
ROUTE_SUCCESS_MODE=absolute ROUTE_SUCCESS_FLOOR=0.85 \
  ./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Frontier run/safety CSVs now report fixed-target route success, substituted
route success, and final operational route success separately.

## Step Commands

### `validation-1mo`

Fast 4-week engine sanity check.

```bash
./scripts/run_regenbond_remote_batch.sh validation-1mo
```

Detached:

```bash
./scripts/start_regenbond_batch_tmux.sh validation-1mo
tail -f analysis/monte_carlo/validation-1mo.log
```

Inspect:

```bash
cat analysis/monte_carlo/engine_validation_1mo_test/engine_validation_summary.csv
ls -lh analysis/monte_carlo/engine_validation_1mo_test
```

### `validation-smoke`

Short validation smoke before longer validation.

```bash
./scripts/run_regenbond_remote_batch.sh validation-smoke
```

Detached:

```bash
./scripts/start_regenbond_batch_tmux.sh validation-smoke
tail -f analysis/monte_carlo/validation-smoke.log
```

Inspect:

```bash
cat analysis/monte_carlo/engine_validation_smoke/engine_validation_summary.csv
wc -l analysis/monte_carlo/engine_validation_smoke/engine_validation_errors.csv
```

### `validation-pilot`

Full-horizon 20-run validation pilot.

```bash
./scripts/run_regenbond_remote_batch.sh validation-pilot
```

Detached:

```bash
./scripts/start_regenbond_batch_tmux.sh validation-pilot
tail -f analysis/monte_carlo/validation-pilot.log
```

Inspect:

```bash
cat analysis/monte_carlo/engine_validation_20run/engine_validation_summary.csv
wc -l analysis/monte_carlo/engine_validation_20run/engine_validation_errors.csv
ls -lh analysis/monte_carlo/engine_validation_20run
```

### `validation-full`

Paper-facing full validation gate. This is the validation directory that
frontier runs use as their gate.

```bash
./scripts/run_regenbond_remote_batch.sh validation-full
```

Detached:

```bash
./scripts/start_regenbond_batch_tmux.sh validation-full
tail -f analysis/monte_carlo/validation-full.log
```

Inspect:

```bash
cat analysis/monte_carlo/engine_validation/engine_validation_summary.csv
wc -l analysis/monte_carlo/engine_validation/engine_validation_errors.csv
grep -E 'voucher_to_voucher|stable_involved|gross_stable|active_pool_(stable|voucher)' \
  analysis/monte_carlo/engine_validation/engine_validation_moments.csv
```

For paper-facing use, `engine_validation_summary.csv` should report
`status=pass`. If it reports `review`, inspect errors before using frontier
outputs as headline evidence. If it reports `fail`, fix calibration before
frontier publication runs.

### `frontier-smoke`

Small bond-issuer frontier structure check.

```bash
./scripts/run_regenbond_remote_batch.sh frontier-smoke
```

Detached:

```bash
./scripts/start_regenbond_batch_tmux.sh frontier-smoke
tail -f analysis/monte_carlo/frontier-smoke.log
```

Inspect:

```bash
ls -lh analysis/monte_carlo/bond_issuer_frontier_smoke
head analysis/monte_carlo/bond_issuer_frontier_smoke/bond_issuer_frontier_safety.csv
cat analysis/monte_carlo/bond_issuer_frontier_smoke/paper_integration_notes.md
```

### `frontier-maturity-smoke`

Fast full-term bond-service lockbox check. This is the local guard against
hidden 260-week repayment failures before spending time on `frontier-pilot`.

```bash
./scripts/run_regenbond_remote_batch.sh frontier-maturity-smoke
```

Ten-run pre-pilot version:

```bash
SESSION=regenbond-maturity-smoke10 \
LOG_FILE=analysis/monte_carlo/frontier-maturity-smoke-10run.log \
OUTPUT=analysis/monte_carlo/bond_issuer_frontier_maturity_smoke_10run \
RUNS=10 WORKERS=15 RESUME=0 \
  ./scripts/start_regenbond_batch_tmux.sh frontier-maturity-smoke
```

Sensitivity run with an explicit shared producer debt contract service margin:

```bash
PRODUCER_DEBT_CONTRACT_SERVICE_MARGIN_RATE=0.05 \
  ./scripts/run_regenbond_remote_batch.sh frontier-maturity-smoke
```

Channel-specific overrides can still be used for ablation runs, but they should
not be interpreted as current default assumptions without empirical support.

Control run with the old next-payment reserve:

```bash
BOND_SERVICE_LOCKBOX_MODE=next_due BOND_SERVICE_LOCKBOX_COVERAGE_RATIO=1.0 \
  ./scripts/run_regenbond_remote_batch.sh frontier-maturity-smoke
```

Inspect:

```bash
head analysis/monte_carlo/bond_issuer_frontier_maturity_smoke/bond_issuer_frontier_safety.csv
```

### `frontier-feedback-probe`

Historical focused full-horizon current-scale probe for the capacity-feedback
mechanism. This job remains available as an ablation/control target, but the
current pre-pilot gate is `frontier-rola-regeneration-probe`.

```bash
./scripts/run_regenbond_remote_batch.sh frontier-feedback-probe
```

Detached:

```bash
./scripts/start_regenbond_batch_tmux.sh frontier-feedback-probe
tail -f analysis/monte_carlo/frontier-feedback-probe.log
```

Inspect:

```bash
head analysis/monte_carlo/bond_issuer_frontier_feedback_probe/bond_issuer_frontier_safety.csv
head analysis/monte_carlo/bond_issuer_frontier_feedback_probe/safe_injection_frontier.csv
```

The key columns to inspect are the scheduled-payment coverage fields, unpaid
claims, swap-volume ratio versus baseline, voucher-to-voucher count/volume
preservation, voucher-to-voucher share as a diagnostic composition metric,
baseline productive-credit inflow/deposits, and incremental productive-credit
inflow/deposits.

### `frontier-rola-regeneration-probe`

Current full-horizon low-principal probe for the ROLA/voucher-purchase
mechanism. This is the preferred current-scale gate before the focused pilot.
Despite the target name, it is not the no-voucher ROSCA-to-ROLA regeneration
counterfactual; it is a voucher-capable ROSCA-like credit-pool mechanism
check.

```bash
./scripts/run_regenbond_remote_batch.sh frontier-rola-regeneration-probe
```

Detached:

```bash
./scripts/start_regenbond_batch_tmux.sh frontier-rola-regeneration-probe
tail -f analysis/monte_carlo/frontier-rola-regeneration-probe.log
```

Inspect:

```bash
head analysis/monte_carlo/bond_issuer_frontier_rola_regeneration_probe/bond_issuer_frontier_safety.csv
head analysis/monte_carlo/bond_issuer_frontier_rola_regeneration_probe/safe_injection_frontier.csv
cat analysis/monte_carlo/bond_issuer_frontier_rola_regeneration_probe/paper_integration_notes.md
```

The key columns to inspect are scheduled-payment coverage, available
service-cash headroom, unpaid claims, voucher-to-voucher volume ratio,
ordinary voucher-source volume ratio, consumer/community stress deltas,
producer voucher-loan execution, consumer/third-party purchase successes, and
realized productive-credit voucher-deposit share.

### `frontier-pilot`

Full-horizon 20-run focused frontier pilot on the `current` network. The
default grid is principal ratios
`0.05,0.10,0.15,0.20,0.25`, coupon targets
`0,0.02,0.04,0.06,0.08,0.10`, and fee-service share `1.0`. `connected_2x` and
`connected_5x` are no longer part of the main pilot because they are too slow
for routine paper-facing iteration; use them only as explicit scaling stress
tests.

The previously reviewed focused pilot is a guardrail frontier, not a blanket
safety or pricing recommendation. It was generated before the current Kenya
cohort and private-wallet routing update, so treat it as historical pilot
evidence until the current validation and publication-grid artifacts are
regenerated.

```bash
./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Detached:

```bash
./scripts/start_regenbond_batch_tmux.sh frontier-pilot
tail -f analysis/monte_carlo/frontier-pilot.log
```

Inspect while running:

```bash
tail -f analysis/monte_carlo/frontier-pilot.log
ls -lh analysis/monte_carlo/bond_issuer_frontier_pilot
find analysis/monte_carlo/bond_issuer_frontier_pilot/_shards -name manifest.json | wc -l
head analysis/monte_carlo/bond_issuer_frontier_pilot/bond_issuer_frontier_safety.partial.csv
```

Inspect after completion:

```bash
ls -lh analysis/monte_carlo/bond_issuer_frontier_pilot
head analysis/monte_carlo/bond_issuer_frontier_pilot/bond_issuer_frontier_safety.csv
cat analysis/monte_carlo/bond_issuer_frontier_pilot/paper_integration_notes.md
```

### `frontier-publication`

Paper-facing frontier grid. Run only after `validation-full` passes and the
`frontier-pilot` output is acceptable. This job is the 100-run expansion of the
`frontier-pilot` setup used so far in the paper: the same `current` network
routing/clearing profile, but a denser coupon/principal grid. It excludes
`principal_ratio=0` because matched no-bond baselines are generated separately,
and tests 13 annual coupon targets from 0% through 12% against 10 positive
principal ratios from 0.05 through 0.50. The resulting grid has 130 frontier
cells, plus built-in no-bond baselines.

```bash
./scripts/run_regenbond_remote_batch.sh frontier-publication
```

Detached:

```bash
WORKERS=15 RESUME=0 ./scripts/start_regenbond_batch_tmux.sh frontier-publication
tail -f analysis/monte_carlo/frontier-publication.log
```

If interrupted, rerun the same job with `RESUME=1`.

Inspect:

```bash
ls -lh analysis/monte_carlo/bond_issuer_frontier
head analysis/monte_carlo/bond_issuer_frontier/safe_injection_frontier.csv
head analysis/monte_carlo/bond_issuer_frontier/network_scaling_summary.csv
cat analysis/monte_carlo/bond_issuer_frontier/paper_integration_notes.md
```

Check the grid dimensions:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

out = Path("analysis/monte_carlo/bond_issuer_frontier")
rows = list(csv.DictReader((out / "bond_issuer_frontier_safety.csv").open()))

print("safety rows:", len(rows))
print("scales:", sorted({r["network_scale"] for r in rows}))
print("coupon count:", len({r["coupon_target_annual"] for r in rows}))
print("principal ratio count:", len({r["principal_ratio"] for r in rows}))
print("min principal ratio:", min(float(r["principal_ratio"]) for r in rows))
PY
```

Expected: 130 safety rows, scale `current`, 13 coupon targets, 10 principal
ratios, and minimum principal ratio 0.05.

## Sensitivity Runs

Use a distinct `OUTPUT` for runs you want to keep side by side:

```bash
ROUTE_SUCCESS_MODE=absolute ROUTE_SUCCESS_FLOOR=0.80 \
OUTPUT=analysis/monte_carlo/bond_issuer_frontier_pilot_route080 \
  ./scripts/run_regenbond_remote_batch.sh frontier-pilot

ROUTE_SUCCESS_MODE=absolute ROUTE_SUCCESS_FLOOR=0.90 \
OUTPUT=analysis/monte_carlo/bond_issuer_frontier_pilot_route090 \
  ./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Compare diagnostic and relative route treatment:

```bash
ROUTE_SUCCESS_MODE=diagnostic OUTPUT=analysis/monte_carlo/bond_issuer_frontier_pilot_route_diag \
  ./scripts/run_regenbond_remote_batch.sh frontier-pilot

ROUTE_SUCCESS_MODE=relative OUTPUT=analysis/monte_carlo/bond_issuer_frontier_pilot_route_relative \
  ./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Compare the frontier lockbox against the old next-payment reserve behavior:

```bash
BOND_SERVICE_LOCKBOX_MODE=next_due BOND_SERVICE_LOCKBOX_COVERAGE_RATIO=1.0 \
OUTPUT=analysis/monte_carlo/bond_issuer_frontier_maturity_smoke_next_due \
  ./scripts/run_regenbond_remote_batch.sh frontier-maturity-smoke
```

Adaptive midpoint refinement is opt-in. The default frontier pilot now runs the
fixed configured grid only:

```bash
./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Enable adaptive midpoint refinement explicitly when needed:

```bash
FRONTIER_MODE=adaptive FRONTIER_REFINEMENT_ROUNDS=1 \
  ./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Run only one scale:

```bash
NETWORK_SCALES=current ./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Run a small custom frontier for debugging:

```bash
RUNS=2 TICKS=4 BOND_TERM=4 NETWORK_SCALES=current PRINCIPAL_RATIOS=0.05 \
COUPON_TARGETS=0 BOND_FEE_SERVICE_SHARES=0.5 FRONTIER_MODE=grid WORKERS=2 \
OUTPUT=/tmp/regenbond_frontier_debug \
  ./scripts/run_regenbond_remote_batch.sh frontier-smoke
```

## Pulling Outputs Locally

Run these from your local machine after each remote job finishes. Create the
local parent directory first:

```bash
mkdir -p /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo
```

Use `rsync` instead of `scp -r` for normal review/paper pulls. The commands
below use your SSH key and exclude `_shards/` plus `*.partial.csv`, so they do
not download resumable worker shards:

```bash
rsync -av \
  --exclude '_shards/' \
  --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/engine_validation_20run/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/engine_validation_20run/
```

Only use `scp -r` if you intentionally want to copy `_shards/` too.

Pull the output directory that matches the job you just ran:

| Job | Remote output directory | Remote log | Local destination |
| --- | --- | --- | --- |
| `validation-1mo` | `~/sim/analysis/monte_carlo/engine_validation_1mo_test/` | `~/sim/analysis/monte_carlo/validation-1mo.log` | `RegenBonds/analysis/monte_carlo/engine_validation_1mo_test/` |
| `validation-smoke` | `~/sim/analysis/monte_carlo/engine_validation_smoke/` | `~/sim/analysis/monte_carlo/validation-smoke.log` | `RegenBonds/analysis/monte_carlo/engine_validation_smoke/` |
| `validation-pilot` | `~/sim/analysis/monte_carlo/engine_validation_20run/` | `~/sim/analysis/monte_carlo/validation-pilot.log` | `RegenBonds/analysis/monte_carlo/engine_validation_20run/` |
| `validation-full` | `~/sim/analysis/monte_carlo/engine_validation/` | `~/sim/analysis/monte_carlo/validation-full.log` | `RegenBonds/analysis/monte_carlo/engine_validation/` |
| `frontier-smoke` | `~/sim/analysis/monte_carlo/bond_issuer_frontier_smoke/` | `~/sim/analysis/monte_carlo/frontier-smoke.log` | `RegenBonds/analysis/monte_carlo/bond_issuer_frontier_smoke/` |
| `frontier-maturity-smoke` | `~/sim/analysis/monte_carlo/bond_issuer_frontier_maturity_smoke/` | `~/sim/analysis/monte_carlo/frontier-maturity-smoke.log` | `RegenBonds/analysis/monte_carlo/bond_issuer_frontier_maturity_smoke/` |
| `frontier-feedback-probe` | `~/sim/analysis/monte_carlo/bond_issuer_frontier_feedback_probe/` | `~/sim/analysis/monte_carlo/frontier-feedback-probe.log` | `RegenBonds/analysis/monte_carlo/bond_issuer_frontier_feedback_probe/` |
| `frontier-rola-regeneration-probe` | `~/sim/analysis/monte_carlo/bond_issuer_frontier_rola_regeneration_probe/` | `~/sim/analysis/monte_carlo/frontier-rola-regeneration-probe.log` | `RegenBonds/analysis/monte_carlo/bond_issuer_frontier_rola_regeneration_probe/` |
| `frontier-pilot` | `~/sim/analysis/monte_carlo/bond_issuer_frontier_pilot/` | `~/sim/analysis/monte_carlo/frontier-pilot.log` | `RegenBonds/analysis/monte_carlo/bond_issuer_frontier_pilot/` |
| `frontier-publication` | `~/sim/analysis/monte_carlo/bond_issuer_frontier/` | `~/sim/analysis/monte_carlo/frontier-publication.log` | `RegenBonds/analysis/monte_carlo/bond_issuer_frontier/` |

Keep the trailing slash on remote output directories. It copies the contents of
the job directory into the matching local directory and avoids creating nested
duplicate directories.

### Validation Outputs

Pull `validation-1mo`:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/engine_validation_1mo_test/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/engine_validation_1mo_test/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/validation-1mo.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/engine_validation_1mo_test/
```

Pull `validation-smoke`:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/engine_validation_smoke/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/engine_validation_smoke/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/validation-smoke.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/engine_validation_smoke/
```

Pull `validation-pilot`:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/engine_validation_20run/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/engine_validation_20run/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/validation-pilot.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/engine_validation_20run/
```

Pull `validation-full`:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/engine_validation/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/engine_validation/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/validation-full.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/engine_validation/
```

### Frontier Outputs

Pull `frontier-smoke`:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/bond_issuer_frontier_smoke/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_smoke/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/frontier-smoke.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_smoke/
```

Pull `frontier-maturity-smoke`:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/bond_issuer_frontier_maturity_smoke/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_maturity_smoke/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/frontier-maturity-smoke.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_maturity_smoke/
```

Pull `frontier-feedback-probe`:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/bond_issuer_frontier_feedback_probe/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_feedback_probe/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/frontier-feedback-probe.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_feedback_probe/
```

Pull `frontier-rola-regeneration-probe`:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/bond_issuer_frontier_rola_regeneration_probe/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_rola_regeneration_probe/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/frontier-rola-regeneration-probe.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_rola_regeneration_probe/
```

Pull `frontier-pilot`:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/bond_issuer_frontier_pilot/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_pilot/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/frontier-pilot.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_pilot/
```

Pull `frontier-publication`:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/bond_issuer_frontier/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519' \
  root@128.140.120.36:~/sim/analysis/monte_carlo/frontier-publication.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier/
```

### SSH Alias Form

If `wor-testing` is configured as an SSH alias for
`root@128.140.120.36`, the same commands can be shortened. Example:

```bash
rsync -av --exclude '_shards/' --exclude '*.partial.csv' \
  root@wor-testing:~/sim/analysis/monte_carlo/bond_issuer_frontier_pilot/ \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_pilot/

rsync -av root@wor-testing:~/sim/analysis/monte_carlo/frontier-pilot.log \
  /home/wor/src/ge/clc/RegenBonds/analysis/monte_carlo/bond_issuer_frontier_pilot/
```

## Expected Outputs

Validation outputs:

```text
engine_validation_moments.csv
engine_validation_errors.csv
engine_validation_summary.csv
engine_validation_run_summary.csv
engine_validation_bond_timeseries.csv
engine_validation_network_timeseries.csv
engine_validation_failure_metrics.csv
engine_validation_table.tex
fig_engine_vs_sarafu_activity.png
fig_engine_vs_sarafu_return_coverage.png
paper_integration_notes.md
```

Frontier outputs:

```text
bond_issuer_frontier_runs.csv
bond_issuer_frontier_safety.csv
safe_injection_frontier.csv
network_scaling_summary.csv
issuer_cashflow_summary.csv
safe_injection_frontier_table.tex
non_extraction_guardrails_table.tex
issuer_cashflow_table.tex
fig_safe_injection_frontier.png
fig_issuer_service_coverage.png
fig_binding_constraints_heatmap.png
paper_integration_notes.md
```

Revised validation and frontier CSVs also include columns for producer stable
and voucher deposits, deposit-based credit capacity, productive-credit inflow,
net circulating voucher obligation, producer-debt maturity/repaid/defaulted
principal, contract cash-service due/paid, producer debt closed by circulation,
voucher-fee conversion, fee-service lockbox reservation, quarterly clearing,
pool liquidity impact, fixed-target route success, and substituted route
success.

`--no-png` skips PNG generation but still writes CSV, LaTeX, Markdown, partial
CSV, and shard files.

## Troubleshooting

Check running sessions:

```bash
tmux ls
ps -eo pid,etimes,pcpu,pmem,cmd | grep run_regenbond
```

Check shard progress:

```bash
find analysis/monte_carlo/bond_issuer_frontier_pilot/_shards -name manifest.json | wc -l
grep -R '"status": "failed"' analysis/monte_carlo/bond_issuer_frontier_pilot/_shards
```

Rerun the same command to resume failed or missing shards:

```bash
./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Restart a tmux batch with more workers:

```bash
tmux send-keys -t regenbond C-c
WORKERS=15 ./scripts/start_regenbond_batch_tmux.sh frontier-pilot
```

Do not delete the output directory when increasing workers. The rerun uses
`_shards/` to skip completed cells and resume missing or interrupted work.

If the machine is memory constrained, reduce workers:

```bash
WORKERS=2 ./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

If output files look stale, check both partial and final files:

```bash
ls -lh analysis/monte_carlo/bond_issuer_frontier_pilot/*.partial.csv
ls -lh analysis/monte_carlo/bond_issuer_frontier_pilot/*.csv
```

The route-success floor in the frontier is a settlement-reliability sensitivity
parameter, not a directly observed Sarafu failed-route scalar. In the revised
default `ROUTE_SUCCESS_MODE=diagnostic`, route reliability is reported but is
not a primary binding guardrail. Frontier safety rows still report and guard
against voucher-circulation deterioration versus the matched no-bond baseline,
with paper interpretation focused on voucher-to-voucher value and ordinary
voucher-source activity. Voucher-to-voucher count and share are reported as
diagnostic composition metrics, but they are not growth headlines.
Stable-dependency anchors come from
`analysis/sarafu_calibration/stable_dependency_anchors.csv`.
