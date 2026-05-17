# RegenBond Model Review Plan

This document tracks simulator questions raised during manuscript review. It is a planning document only: do not change model behavior until the current `frontier-pilot` outputs have been inspected.

## Current Status

- The validation runner has produced a passing no-bond Sarafu calibration gate.
- The `frontier-pilot` batch is still expected to produce the first usable bond-issuer safety-frontier outputs.
- Paper text should remain conditional until those outputs are reviewed.
- Existing model changes should be judged against the paper's non-extraction frame: issuer responsibility must remain explicit, and local pools must not become hidden guarantors.

## First Frontier Review

Inspect these files after `frontier-pilot` finishes and is copied back:

- `bond_issuer_frontier_safety.csv`
- `safe_injection_frontier.csv`
- `network_scaling_summary.csv`
- `issuer_cashflow_summary.csv`
- `paper_integration_notes.md`
- `frontier-pilot.log`

Answer these questions before changing model parameters:

- Which cells are safe, and which are unsafe?
- Which binding constraints appear most often?
- Does safe principal fall as coupon rises?
- Does safe principal change monotonically with principal ratio?
- Does `connected_5x` improve route success without excessive concentration?
- Do safe cells preserve voucher-to-voucher count, volume, and share relative to matched no-bond baselines?
- Do safe cells avoid rising stable dependency, measured by active-pool stable value share and stable-to-voucher value ratio?
- Do higher fee-service shares improve service coverage while degrading local settlement diagnostics?
- Are unpaid scheduled claims attributed to issuer reserve exhaustion and cashflow failure rather than to local pools?
- Is any result surprising enough to require code inspection before paper text is finalized?

## Model Semantics To Document

### Pools

A pool is a commitment grammar, not a fixed legal actor. It can represent a personal wallet, lending group, village association, institution, CLC pool, insurance pool, or mandate pool depending on the steward and policy layer.

Documentation should avoid implying that all pools are people, firms, or community groups. The output role label is a behavioral simplification for a run.

### Agent Roles

`lender`, `producer`, `consumer`, and `liquidity_provider` are primary simulation roles. In the real system these roles can overlap. Future model variants should allow mixed or changing roles if calibration requires it.

Review tasks:

- Compare role proportions against Sarafu calibration artifacts.
- Mark each role-mix parameter as empirical calibration, derived calibration, scenario assumption, stress parameter, or policy parameter.

### Vouchers

The headline model uses one primary voucher per agent, following the dominant Sarafu demographic pattern. That voucher can still represent multiple goods or services. Multi-voucher issuers exist and should be treated as a later sensitivity if they materially affect results.

Review tasks:

- Check whether calibration outputs distinguish voucher issuers from listed offerings.
- Add multi-voucher issuer sensitivity only if frontier results depend on issuer concentration or voucher diversity.

### Credit And Repayment

Producer borrowing is modeled as swapping the producer's own voucher for an accepted asset. The creditor pool then holds the producer's obligation. Repayment or closure occurs when the producer voucher leaves the creditor pool through an accepted return asset, acquisition by another participant, issuer return, burn, or redemption proxy.

Review tasks:

- Keep `LOAN_ISSUED`, `REPAYMENT_EXECUTED`, issuer-return, burn, and redemption as related but distinct metrics.
- Record repayment-failure reasons where possible: liquidity scarcity, route failure, cap or limit failure, redemption failure, voucher-circulation breakdown, shock, or curation decision.
- Treat fixed loan cadence and term as calibrated values if supported by Sarafu evidence; otherwise label them as scenario assumptions.

### Single-Lender Assignment

`producer_voucher_single_lender=True` is a relationship-credit control. It does not claim that real producer vouchers are accepted by only one pool.

Review tasks:

- Test whether single-lender assignment creates artificial bottlenecks.
- Later compare against multi-lender voucher acceptance to see whether route success improves or risk becomes less accountable.

### Routing

NOAM routing models marketplace-assisted route discovery over published pools, accepted vouchers, limits, and inventories. It is not perfect hidden knowledge. BFS is useful only as a diagnostic on very small graphs because exhaustive search is too slow at Sarafu-like scale.

Review tasks:

- Reduce NOAM caps, route cache, or beam width and check whether the safe frontier shrinks.
- Treat route-success floors as stress parameters, not directly observed Sarafu facts.
- Compare route success with voucher-to-voucher preservation, stable-credit closure, and concentration diagnostics.

### Affinity And Stickiness

Sticky routes and affinity buddies represent repeated local trust, neighborhood exchange, and village or group membership. They should be calibrated or tested rather than assumed.

Review tasks:

- Disable or reduce sticky affinity and compare Sarafu validation moments.
- If validation breaks, document affinity as a necessary behavioral mechanism.

### Redemption And Voucher Holding

Automatic third-party voucher redemption is a settlement-timing assumption for face-to-face exchange. It should not imply that every voucher is immediately redeemed in reality.

Review tasks:

- Add delayed or probabilistic redemption sensitivity.
- Compare voucher circulation, stale voucher inventory, route success, same-token return, and voucher-return coverage.
- Track voucher dumping through rising voucher-to-stable exits, declining acceptance, declining voucher-to-voucher share, concentration, or leakage.

### Fees And Bond-Service Share

Swap fees and bond-service share are distributional parameters. They can help pay operations, reserves, maintenance, and bondholders, but high bond-service allocation can become extractive if it starves repair, curation, local reinvestment, or household resilience.

Review tasks:

- Compare service coverage, voucher circulation, leakage, concentration, and stress across fee-service shares.
- Never treat higher bondholder payment as safe unless settlement guardrails also pass.

## Planned Sensitivity Matrix

Run these only after the pilot frontier has been reviewed:

- Lower route visibility: reduce NOAM beam width, top-k/top-m, edge caps, or cache.
- Lower route-success floor: test whether safe principal depends on an arbitrary floor.
- Higher off-ramp pressure: increase stable leakage and check whether liquidity leakage binds.
- Higher coupon: check whether service coverage and unpaid claims bind earlier.
- Slower redemption: delay or probabilistically gate third-party voucher redemption.
- Lower affinity: disable or weaken sticky route bias and buddy pools.
- Multi-lender acceptance: allow producer vouchers to be accepted by multiple pools with distinct limits.
- Role mixing: allow producers, lenders, and consumers to overlap where calibration supports it.
- Pool splitting or replication: test whether strong-pool concentration can decentralize into newly curated pools.

## Paper Integration Gate

Do not promote frontier results into headline paper claims until:

- The validation gate remains passing for the relevant code version.
- Frontier outputs are reconstructed from complete shards.
- No failed or stale shard manifests are included.
- Safe cells pass issuer service, unpaid claim, route success, leakage, concentration, cash stress, stable dependency, and matched no-bond degradation guardrails.
- Binding constraints are summarized by scale, coupon, principal ratio, and fee-service share.
- Surprising results are checked against code and calibration assumptions.

