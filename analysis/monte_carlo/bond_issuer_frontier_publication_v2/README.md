# Regenerative Bonds Publication Frontier

This directory contains the privacy-safe aggregate simulator artifacts for the
reported Regenerative Bonds settlement-capacity frontier. It is intended to be
referenced from the paper's arXiv/Overleaf source package and from the public
repository:

```text
https://github.com/cosmo-local-credit/sim
```

The paper source package is self-contained for LaTeX compilation. This
directory provides the simulator-side code/data provenance for the Monte Carlo
frontier and selected aggregate outputs used in the paper.

## Reported Scenario

- Scenario: `bond_issuer_frontier`
- Network scale: `current`
- Runs per positive-principal cell: 100
- Horizon and term: 260 weekly ticks
- Issuer payment stride: 26 weekly ticks
- Pool clearing stride: 13 weekly ticks
- Principal ratios: `0.05,0.10,0.25,0.50,0.75,1.00,1.50,2.00`
- Annual coupon targets: `0,0.04,0.08,0.12,0.15,0.25,0.35,0.45`
- Certification policy: `strong_moderate_capped`
- Bond-service lockbox mode: `remaining_schedule`
- Bond-service lockbox coverage ratio: `1.25`
- Base seed: `1`

The runner derives per-cell and per-run seeds from the base seed, network-scale
offset, run index, and cell configuration. Matched no-bond baselines use the
same scale/cell seed policy where applicable.

## Reproduction Command

From a public clone of `cosmo-local-credit/sim`:

```bash
python scripts/run_regenbond_monte_carlo.py \
  --scenario bond_issuer_frontier \
  --network-scales current \
  --principal-ratios 0.05,0.10,0.25,0.50,0.75,1.00,1.50,2.00 \
  --coupon-targets 0,0.04,0.08,0.12,0.15,0.25,0.35,0.45 \
  --bond-fee-service-shares 1.0 \
  --certification-policy strong_moderate_capped \
  --frontier-mode grid \
  --frontier-refinement-rounds 0 \
  --issuer-payment-stride 26 \
  --pool-clearing-stride 13 \
  --bond-service-lockbox-mode remaining_schedule \
  --bond-service-lockbox-coverage-ratio 1.25 \
  --runs 100 \
  --ticks 260 \
  --term 260 \
  --seed 1 \
  --output analysis/monte_carlo/bond_issuer_frontier_publication_v2
```

The runner uses the public aggregate calibration bundle in
`analysis/sarafu_calibration/` by default. Raw Sarafu transaction records,
participant identifiers, raw report text, and sensitive linkage tables are not
needed for this public reproduction path.

After rerunning the frontier, regenerate the summary tables and publication
figures with the analysis/plotting scripts documented in the repository
README and runbook:

```bash
python scripts/analyze_frontier_publication.py \
  --input-dir analysis/monte_carlo/bond_issuer_frontier_publication_v2

python scripts/plot_settlement_capacity_frontier.py \
  --input-dir analysis/monte_carlo/bond_issuer_frontier_publication_v2
```

## Tracked Publication Artifacts

Tracked files in this directory are aggregate outputs and figures sufficient to
audit the reported paper tables/figures without committing the large per-run
CSV. The full run-level file `bond_issuer_frontier_runs.csv` is reproducible
from the command above but is intentionally not tracked because it is large.

Core tracked artifacts include:

- `bond_issuer_frontier_safety.csv`
- `frontier_publication_analysis_summary.md`
- `frontier_publication_outcome_matrix.csv`
- `frontier_publication_coupon_frontier.csv`
- `frontier_publication_binding_cells.csv`
- `frontier_publication_metric_ranges.csv`
- `settlement_capacity_frontier_summary.csv`
- `issuer_cashflow_summary.csv`
- `safe_injection_frontier.csv`
- `fig_frontier_outcome_grid.png`
- `fig_scheduled_payment_coverage_boundary.png`
- `fig_observed_settlement_motif_shift.png`
- `output_hashes.csv`

The checksum file records SHA-256 hashes for the tracked publication artifacts.

## Interpretation Boundary

These artifacts support the paper's calibrated settlement-capacity frontier.
They are simulation guardrail outputs, not investment advice, proposed
securities terms, legal compliance findings, or causal welfare/ecological
effect estimates.
