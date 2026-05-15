# RegenBond Remote Batch Verification

This repo is self-contained for public RegenBond Monte Carlo verification. A
clean clone contains the simulator, the paper-facing runners, and the
privacy-safe Sarafu calibration bundle under `analysis/sarafu_calibration/`.

The calibration bundle is aggregate/anonymized. It does not include raw Sarafu
transactions, addresses, report text, GPS data, pool labels, or pool IDs.

## Server Setup

```bash
git clone https://github.com/cosmo-local-credit/sim.git
cd sim
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Batch Wrapper

Use the wrapper from the repo root:

```bash
./scripts/run_regenbond_remote_batch.sh validation-1mo
./scripts/run_regenbond_remote_batch.sh validation-smoke
./scripts/run_regenbond_remote_batch.sh validation-full
./scripts/run_regenbond_remote_batch.sh frontier-smoke
./scripts/run_regenbond_remote_batch.sh frontier-pilot
./scripts/run_regenbond_remote_batch.sh frontier-publication
```

Default outputs are written under `analysis/monte_carlo/`. The full validation
gate writes to `analysis/monte_carlo/engine_validation/`; frontier jobs look for
that directory before treating outputs as paper-facing.

Useful overrides:

```bash
RUNS=20 TICKS=260 ./scripts/run_regenbond_remote_batch.sh validation-full
OUTPUT_ROOT=/mnt/batch/regenbond ./scripts/run_regenbond_remote_batch.sh frontier-pilot
PYTHON_BIN=/opt/venvs/sim/bin/python ./scripts/run_regenbond_remote_batch.sh validation-full
```

## Start Detached

Start the full validation already detached:

```bash
./scripts/start_regenbond_batch_tmux.sh validation-full
```

The launcher starts a detached `tmux` session named `regenbond` and writes:

```text
analysis/monte_carlo/validation-full.log
```

Check progress:

```bash
tail -f analysis/monte_carlo/validation-full.log
```

Attach only if you want an interactive view:

```bash
tmux attach -t regenbond
```

Use a different session name or log path:

```bash
SESSION=regenbond2 LOG_FILE=analysis/monte_carlo/validation-full-2.log \
  ./scripts/start_regenbond_batch_tmux.sh validation-full
```

## Manual Tmux

```bash
tmux new -s regenbond
./scripts/run_regenbond_remote_batch.sh validation-full 2>&1 | tee analysis/monte_carlo/validation-full.log
```

Detach with `Ctrl-b d`; reattach with:

```bash
tmux attach -t regenbond
```

Or start manually detached without the helper:

```bash
tmux new-session -d -s regenbond \
  "cd $PWD && ./scripts/run_regenbond_remote_batch.sh validation-full 2>&1 | tee analysis/monte_carlo/validation-full.log"
```

The log directory exists in the repo after `git pull` because
`analysis/monte_carlo/.gitkeep` is tracked. If you change `OUTPUT_ROOT`, create
that directory before using `tee`.

Check progress without attaching:

```bash
tail -n 80 analysis/monte_carlo/validation-full.log
tail -f analysis/monte_carlo/validation-full.log
```

The progress lines look like:

```text
[progress] scenario=sarafu_engine_validation run=12/100 tick=130/260 run_pct= 50.0% overall= 11.5% elapsed= ...
```

If the SSH connection drops, reconnect and run:

```bash
cd sim
tmux ls
tail -f analysis/monte_carlo/validation-full.log
```

## Paper-Facing Sequence

1. Run `validation-full`.
2. Confirm `analysis/monte_carlo/engine_validation/engine_validation_summary.csv`
   has `status=pass`.
3. Run `frontier-pilot`.
4. Inspect `bond_issuer_frontier_safety.csv`, `safe_injection_frontier.csv`, and
   `paper_integration_notes.md`.
5. Run `frontier-publication` only after the validation gate and pilot outputs
   look correct.

## Expected Validation Outputs

When `validation-full` finishes, the final log line should look like:

```text
Wrote engine validation artifacts to .../analysis/monte_carlo/engine_validation (status=pass)
```

Expected files:

```text
analysis/monte_carlo/engine_validation/engine_validation_moments.csv
analysis/monte_carlo/engine_validation/engine_validation_errors.csv
analysis/monte_carlo/engine_validation/engine_validation_summary.csv
analysis/monte_carlo/engine_validation/engine_validation_run_summary.csv
analysis/monte_carlo/engine_validation/engine_validation_bond_timeseries.csv
analysis/monte_carlo/engine_validation/engine_validation_network_timeseries.csv
analysis/monte_carlo/engine_validation/engine_validation_failure_metrics.csv
analysis/monte_carlo/engine_validation/engine_validation_table.tex
analysis/monte_carlo/engine_validation/fig_engine_vs_sarafu_activity.png
analysis/monte_carlo/engine_validation/fig_engine_vs_sarafu_return_coverage.png
analysis/monte_carlo/engine_validation/paper_integration_notes.md
```

Quick checks:

```bash
cat analysis/monte_carlo/engine_validation/engine_validation_summary.csv
wc -l analysis/monte_carlo/engine_validation/engine_validation_errors.csv
ls -lh analysis/monte_carlo/engine_validation
```

For a clean paper-facing pass, `engine_validation_summary.csv` should report
`status=pass`, and `engine_validation_errors.csv` should contain only its header
row. If the status is `review`, inspect the error rows before using frontier
outputs as headline paper evidence. If the status is `fail`, do not use frontier
outputs for the paper until the calibration miss is fixed.
