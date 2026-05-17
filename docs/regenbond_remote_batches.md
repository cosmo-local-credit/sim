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

For later runs:

```bash
cd ~/sim
git pull
.venv/bin/python -m pip install -r requirements.txt
```

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
5. Run `frontier-pilot`.
6. Run `frontier-publication` only after validation and pilot outputs look
   correct.

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
| `frontier-pilot` | Full-horizon frontier pilot | 20 runs, 260 ticks, 3 scales | `analysis/monte_carlo/bond_issuer_frontier_pilot/` | `analysis/monte_carlo/frontier-pilot.log` |
| `frontier-publication` | Paper-facing frontier grid | 100 runs, 260 ticks, 3 scales | `analysis/monte_carlo/bond_issuer_frontier/` | `analysis/monte_carlo/frontier-publication.log` |

`frontier-*` jobs read the full validation gate at
`analysis/monte_carlo/engine_validation/engine_validation_summary.csv`. If it
is missing or `review`, the frontier still runs but marks outputs non-final. If
it is `fail`, paper-facing frontier execution is refused.

## Calibration Bundle

Public runs use privacy-safe aggregate calibration files in
`analysis/sarafu_calibration/`. The bundle contains no raw Sarafu IDs,
addresses, transaction hashes, GPS data, report text, or pool labels.

The revised Monte Carlo also reads aggregate tables for:

- producer stable and own-voucher deposit proxies;
- productive-credit return timing and repayment lag;
- stable-to-voucher debt-removal purchase motifs;
- voucher-fee-to-stable conversion capacity;
- quarterly lender-pool clearing capacity;
- route-substitution diagnostics.

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

Frontier-specific parameters:

| Env var | Meaning | Default |
| --- | --- | --- |
| `NETWORK_SCALES` | Comma-separated frontier scales. | job-specific |
| `PRINCIPAL_RATIOS` | Comma-separated principal/certified-capacity ratios. | job-specific |
| `COUPON_TARGETS` | Comma-separated annual coupon targets. | job-specific |
| `BOND_FEE_SERVICE_SHARES` | Comma-separated eligible fee/service shares. | job-specific |
| `CERTIFICATION_POLICY` | Certified pool policy. | `strong_moderate_capped` |
| `FRONTIER_MODE` | `adaptive` or `grid`. | `adaptive` |
| `FRONTIER_REFINEMENT_ROUNDS` | Adaptive midpoint rounds. | `1` |
| `ROUTE_SUCCESS_MODE` | `diagnostic`, `relative`, or `absolute`. | `diagnostic` |
| `ROUTE_SUCCESS_FLOOR` | p05 route-success safety floor, binding only when `ROUTE_SUCCESS_MODE=absolute`. | `0.85` |
| `BOND_TERM` | Frontier term in ticks. Do not use `TERM`. | `260` |

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

### `frontier-pilot`

Full-horizon 20-run frontier pilot across `current`, `connected_2x`, and
`connected_5x`.

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
pilot output is acceptable.

```bash
./scripts/run_regenbond_remote_batch.sh frontier-publication
```

Detached:

```bash
./scripts/start_regenbond_batch_tmux.sh frontier-publication
tail -f analysis/monte_carlo/frontier-publication.log
```

Inspect:

```bash
ls -lh analysis/monte_carlo/bond_issuer_frontier
head analysis/monte_carlo/bond_issuer_frontier/safe_injection_frontier.csv
head analysis/monte_carlo/bond_issuer_frontier/network_scaling_summary.csv
cat analysis/monte_carlo/bond_issuer_frontier/paper_integration_notes.md
```

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

Run a fixed grid without adaptive midpoint refinement:

```bash
FRONTIER_MODE=grid ./scripts/run_regenbond_remote_batch.sh frontier-pilot
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
net circulating voucher obligation, voucher-fee conversion, quarterly clearing,
lender liquidity impact, fixed-target route success, and substituted route
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
against voucher-to-voucher decline versus the matched no-bond baseline.
Stable-dependency anchors come from
`analysis/sarafu_calibration/stable_dependency_anchors.csv`.
