# RegenBond Model Review Plan

This document tracks simulator questions raised during manuscript review. It is
a planning document for model revision and validation. Do not promote revised
frontier results into the paper until the validation sequence and pilot review
below pass.

## Current Status

- The validation runner has produced a passing no-bond Sarafu calibration gate.
- The `frontier-pilot` batch completed and produced the first usable bond-issuer safety-frontier outputs.
- The pilot rejected all tested repayable-principal cells under the current guardrails.
- The calibration-first model revision has now been implemented in code and has passed unit/tiny smoke checks locally.
- Paper text should remain conditional until the revised model is fully revalidated and rerun remotely.
- Existing model changes should be judged against the paper's non-extraction frame: issuer responsibility must remain explicit, and local pools must not become hidden guarantors.

## Calibration-First Revision

Implemented changes:

- Regenerated private empirical calibration aggregates and exported public-safe CSVs for producer deposits, productive-credit timing, debt-removal purchases, fee conversion, quarterly clearing, and route-substitution diagnostics.
- Updated voucher ledger accounting so net circulating obligation is tracked separately from cumulative issued, returned, and redeemed totals.
- Added producer deposits of stable and own vouchers, with producer credit capacity based on deposited value using the default `5x` multiple.
- Added productive-credit stable inflow after producer borrowing, calibrated from aggregate borrow-return timing where available.
- Added routed conversion attempts for voucher-denominated fees, with successful stable conversion entering issuer/CLC service capacity and failures retained as voucher fee inventory.
- Added quarterly clearing of eligible recovered stable from lender pools, capped by scheduled issuer need and lender surplus.
- Added route-substitution diagnostics for ordinary purchase/exchange attempts while keeping borrowing, repayment, and fee conversion as fixed-target routes.
- Changed the frontier route-success default to `diagnostic`; `absolute` mode keeps the old hard floor behavior, and `relative` mode compares against the matched no-bond baseline.

Validation required before using revised frontier results:

```bash
./scripts/run_regenbond_remote_batch.sh validation-1mo
./scripts/run_regenbond_remote_batch.sh validation-smoke
./scripts/run_regenbond_remote_batch.sh validation-pilot
./scripts/run_regenbond_remote_batch.sh validation-full
./scripts/run_regenbond_remote_batch.sh frontier-smoke
./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Run `frontier-publication` only after the revised pilot is reviewed.

## Empirical Grounding Discipline

Every model revision should be traced through this chain before it becomes a
paper claim:

1. Sarafu Network empirical observation.
2. Privacy-safe aggregate calibration artifact.
3. Explicit simulator mechanism or scenario parameter.
4. No-bond engine validation against Sarafu-calibrated moments.
5. Bond-frontier counterfactual with a matched no-bond baseline.
6. Paper claim, limited to the strength of the prior steps.

Use these labels in review notes and paper drafts:

- `observed`: directly measured in Sarafu aggregate data.
- `calibrated proxy`: inferred from observed motifs but not direct ground truth.
- `scenario parameter`: chosen for stress testing or policy design.
- `validation result`: accepted only after the revised no-bond validation passes.
- `counterfactual result`: accepted only after matched-baseline frontier review.

Mechanism grounding:

- Voucher ledger accounting is an accounting invariant grounded in Sarafu issue,
  transfer, burn, redemption, and issuer-return traces.
- Producer deposits and deposit-based credit capacity are calibrated from pool
  deposits, backing inflows, and deposit-to-limit ratios.
- Productive-credit inflow is a calibrated proxy from borrow-proxy events with
  later issuer returns; it is not causal proof of business growth.
- Stable-to-voucher debt removal is calibrated from stable/cash-in and
  voucher-out motifs with later return support.
- Voucher-fee conversion is calibrated from observed voucher-to-stable
  conversion capacity; failed conversion must remain visible.
- Quarterly clearing is a scenario mechanism bounded by recovered stable,
  lender surplus, and issuer scheduled need. It must not become a hidden local
  pool guarantee.
- Route substitution is a scenario diagnostic grounded in shared voucher-pool
  topology, not an observed denominator of failed searches.

## First Frontier Review

The first pilot review inspected these files:

- `bond_issuer_frontier_safety.csv`
- `safe_injection_frontier.csv`
- `network_scaling_summary.csv`
- `issuer_cashflow_summary.csv`
- `paper_integration_notes.md`
- `frontier-pilot.log`

The review found that issuer cashflow, voucher-to-voucher motif preservation,
and route reliability are binding in the current implementation. Before
changing headline parameters, the simulator needs semantic corrections for
productive credit, voucher redemption, fee conversion, and route substitution.

Answer these questions again after the next model revision:

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
- Model productive credit: a producer loan should be able to increase future producer stable income or productive capacity according to calibrated or explicit scenario assumptions.
- Model regular producer deposits of stable and vouchers into the producer's pool.
- Review credit-limit logic so producer borrowing capacity can depend on deposited stable/voucher value, with an explicit proposed multiple of `5x` deposited value.
- Treat consumer voucher purchases and redemption as settlement/closure of producer obligations even when they are not counted as producer self-repayment.
- Preserve ROLA-like voucher borrowing and redemption as a primary strength mechanism: producer vouchers can be used to acquire other producers' vouchers, which are then redeemed for goods, services, production inputs, or cash savings.

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
- Add route-substitution behavior: if the desired target cannot be routed because of liquidity or limits, agents should be able to choose an alternative acceptable purchase or redemption route where calibration supports it.
- Reconsider the fixed `0.85` p05 route-success safety floor. It may be removed from the primary safe frontier or replaced with degradation-versus-baseline diagnostics after substitution behavior is implemented.

### Affinity And Stickiness

Sticky routes and affinity buddies represent repeated local trust, neighborhood exchange, and village or group membership. They should be calibrated or tested rather than assumed.

Review tasks:

- Disable or reduce sticky affinity and compare Sarafu validation moments.
- If validation breaks, document affinity as a necessary behavioral mechanism.

### Redemption And Voucher Holding

Automatic third-party voucher redemption is a settlement-timing assumption for face-to-face exchange. It should not imply that every voucher is immediately redeemed in reality.

Review tasks:

- Fix issuer-ledger semantics so redeemed vouchers reduce the relevant outstanding obligation while preserving auditable totals.
- When someone redeems another issuer's voucher, the voucher amount should leave the holder/pool inventory and return to the issuer's pool or issuer ledger consistently.
- Add delayed or probabilistic redemption sensitivity.
- Compare voucher circulation, stale voucher inventory, route success, same-token return, and voucher-return coverage.
- Track voucher dumping through rising voucher-to-stable exits, declining acceptance, declining voucher-to-voucher share, concentration, or leakage.

### Fees And Bond-Service Share

Swap fees and bond-service share are distributional parameters. They can help pay operations, reserves, maintenance, and bondholders, but high bond-service allocation can become extractive if it starves repair, curation, local reinvestment, or household resilience.

Review tasks:

- Compare service coverage, voucher circulation, leakage, concentration, and stress across fee-service shares.
- Never treat higher bondholder payment as safe unless settlement guardrails also pass.
- Voucher-denominated fees should not be ignored for bond service. They should enter a conversion process where the bond issuer attempts to swap them through pools for stable when liquidity permits.
- Track failed fee conversion separately from absence of fees.
- Do not mechanically assume that higher bond-service share reduces local reinvestment unless the modeled waterfall diverts scarce cash from those uses. Instead, report the opportunity cost explicitly: if the same fee cash can fund bondholders, liquidity mandates, repair, insurance, or operations, the model must show which bucket gives way.

### Bond Principal Clearing

The current pilot repays bondholders primarily through fee/service flows and the
issuer reserve. Producer repayment restores lender-pool liquidity, but it does
not automatically return principal to the bond issuer.

Review tasks:

- Consider a quarterly credit-clearing event in which eligible repaid stable or surplus lender-pool liquidity can return to the bond issuer for bondholder repayment.
- Keep this distinct from hidden community guarantees: only explicitly eligible cleared amounts should service bond principal.
- Track lender-pool liquidity before and after clearing so credit capacity is not silently drained.

## Planned Sensitivity Matrix

Run these only after the model-semantics changes above have been implemented
and the validation gate is rerun:

- Lower route visibility: reduce NOAM beam width, top-k/top-m, edge caps, or cache.
- Route-substitution sensitivity: compare fixed-target route success, substituted route success, and degradation versus matched no-bond baselines.
- Higher off-ramp pressure: increase stable leakage and check whether liquidity leakage binds.
- Higher coupon: check whether service coverage and unpaid claims bind earlier.
- Slower redemption: delay or probabilistically gate third-party voucher redemption.
- Lower affinity: disable or weaken sticky route bias and buddy pools.
- Multi-lender acceptance: allow producer vouchers to be accepted by multiple pools with distinct limits.
- Role mixing: allow producers, lenders, and consumers to overlap where calibration supports it.
- Pool splitting or replication: test whether strong-pool concentration can decentralize into newly curated pools.
- Productive-credit sensitivity: vary the strength and timing of producer stable-income growth after borrowing.
- Quarterly clearing sensitivity: vary the share and timing of lender-pool surplus that can return to the bond issuer.
- Fee-conversion sensitivity: vary how reliably voucher-denominated fees can be converted into stable.

## Paper Integration Gate

Do not promote frontier results into headline paper claims until:

- The validation gate remains passing for the relevant code version.
- Frontier outputs are reconstructed from complete shards.
- No failed or stale shard manifests are included.
- Safe cells pass issuer service, unpaid claim, route success, leakage, concentration, cash stress, stable dependency, and matched no-bond degradation guardrails.
- Binding constraints are summarized by scale, coupon, principal ratio, and fee-service share.
- Surprising results are checked against code and calibration assumptions.
