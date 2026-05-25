# RegenBond Model Review Plan

This document tracks simulator questions raised during manuscript review. It is
a planning document for model revision and validation. Historical sections are
kept for auditability; current paper-facing interpretation should follow the
latest validation and frontier artifacts.

## Current Status

- The empirical calibration has been redone for Kenya KES/KSh community
  lending pools only: `73` open lender pools, `996` unique producer-voucher
  wallets, `1,247` accepted-voucher member slots, and `462` recommended
  external non-producer consumer wallets.
- The current simulation topology treats those empirical pools as open lender
  pools. Producer and consumer accounts are private source/sink wallets: they
  can start or receive routes, but routing and NOAM clearing may traverse only
  open lender pools. A producer voucher can be accepted in multiple lender
  pools according to the exported overlap calibration.
- The current full no-bond Sarafu engine-validation gate passes after the
  2026-05-25 cohort, private-wallet routing, route-motif accounting, and
  stable-to-voucher purchase calibration updates: 100 runs, 260 weekly ticks,
  32 binding pass rows, zero review rows, zero failures, and 22 non-binding
  diagnostic rows.
- The current bond-issuer frontier deploys 100% of gross bond principal as
  stable liquidity into eligible lender pools rather than withholding an
  initial reserve-waterfall split.
- The current safety gate separates capped scheduled-payment coverage from
  gross and available service-cash headroom. Smoke frontier outputs are quick
  structural checks; the larger pilot/publication grids remain the
  paper-facing frontier evidence.
- The latest revision adds bounded productive-capacity feedback: frontier
  producer borrowing can create additional voucher deposits and voucher-source
  activity, but only through aggregate calibration shares and growth caps, and
  always against a matched no-bond baseline.
- The 20-run `frontier-rola-regeneration-probe` passed all tested
  current-scale low-principal cells from `0` to `0.05` principal ratio. The
  current `frontier-pilot` and `frontier-publication` targets use the same
  primary producer voucher borrowing and bounded lender-held-voucher purchase
  demand.
- The `frontier-rola-regeneration-probe` is a voucher-capable ROSCA-like
  credit-pool mechanism check: it starts from a Sarafu-calibrated substrate
  with stable-credit logic, borrowing rights, credit limits, repayment
  obligations, producer voucher identities, lender acceptance rules, and
  routing.
- The 10-run `frontier-maturity-smoke` passed the full-term check: scheduled
  bondholder payment cleared, unpaid scheduled claims were
  zero, voucher circulation was preserved, and the `1.25x` available
  service-cash threshold was reported separately as issuer operating/risk
  headroom.
- The current `frontier-pilot` target is the focused current-scale grid:
  principal ratios `0.05,0.10,0.15,0.20,0.25`; coupon targets
  `0,0.02,0.04,0.06,0.08,0.10`; fee-service share `1.0`; frontier mode
  `grid`; refinement rounds `0`.
- The paper-facing next step is `frontier-publication`, the 100-run expansion
  of the same mechanism at current scale only: 13 coupon targets from `0%`
  through `12%`, 10 positive principal ratios from `0.05` through `0.50`, and
  one matched no-bond baseline.
- Existing model changes should be judged against the paper's non-extraction frame: issuer responsibility must remain explicit, and local pools must not become hidden guarantors.

## Calibration-First Revision

Implemented changes:

- Regenerated private empirical calibration aggregates and exported public-safe CSVs for producer deposits, productive-credit timing, debt-removal purchases, fee conversion, quarterly clearing, and route-substitution diagnostics.
- Updated voucher ledger accounting so net circulating obligation is tracked separately from cumulative issued, returned, and redeemed totals.
- Added producer deposits of stable and own vouchers, with producer credit capacity based on deposited value using the default `5x` multiple.
- Added productive-credit stable inflow after producer borrowing, calibrated from aggregate borrow-return timing where available.
- Added loan-induced voucher-deposit feedback from productive-credit inflow.
  The current aggregate calibration retains `0.244881` as stable, allocates
  `0.755119` to voucher deposits, and caps loan-induced voucher-deposit growth
  at `0.189536` per month.
- Added dated producer-debt obligations for frontier runs: producer voucher borrowing creates a maturity record and a contract cash-service obligation. Lender-held vouchers can close through normal circulation, but circulation alone only closes the pool-level voucher exposure; stable bond-service recovery requires borrower repayment, consumer or third-party stable purchase, or contract cash-service payment. Any remaining cash service at the 13-tick maturity is attempted from producer stable and then written off under the configured recovery/default rate.
- Calibrated producer debt maturity recovery from the mature borrow-proxy
  value support rate. The current bundle sets this at `0.673`, replacing the
  earlier full-recovery frontier assumption.
- Added producer debt contract cash-service accounting for frontier runs. The
  current default is principal-only cash service with a shared `0%` margin over
  borrowed principal until an empirical debt-service margin is calibrated.
  Margins such as `2%`, `5%`, `10%`, `15%`, or `50%` are sensitivity/stress
  parameters, not deployed lending-price claims. Issuer operating and risk
  sustainability are measured separately through fee service, excess recovered
  stable, lockbox surplus, and operating-surplus diagnostics.
- Added primary producer voucher borrowing and voucher-loan diagnostics.
  Producer voucher-to-voucher borrowing is treated as a ROLA/marketplace
  circulation channel; it can shift lender-pool exposure and still carries
  cash-service accounting, but it is not counted as bond-service cash until
  stable is recovered.
- Calibrated the default primary voucher-borrowing attempt share from the
  recent voucher-source settlement motif mix. The current value is `0.868845`,
  replacing the earlier fixed `0.50` probe assumption. This is a decision
  prior: producers may seek stable credit, but when redeemable goods/services
  are available through listed vouchers, the model allows direct
  voucher-to-voucher borrowing through open lender pools instead of forcing a
  stable leg first.
- Added bounded lender-held-voucher purchase demand. Consumers and third-party
  buyers can use calibrated stable purchase budgets to buy visible producer
  vouchers from lender pools, subject to route success, target inventory, and
  reserve protection. Consumer stable-to-voucher purchases are generated here,
  not duplicated as generic ordinary stable-spend routes.
- Added routed conversion attempts for voucher-denominated fees. Successful stable conversion can reserve the configured fee-service share into the bond lockbox before excess fee cash enters the CLC waterfall; failures are retained as voucher fee inventory.
- Added quarterly clearing of eligible recovered stable from lender pools, capped by scheduled issuer need and lender surplus.
- Added a configurable bond-service lockbox. Frontier jobs reserve recovered lender stable and eligible fee-service cash against `1.25x` remaining scheduled principal plus coupon before cash can recirculate; `next_due` remains available as a control mode. The `1.25x` value is reported as issuer operating/risk headroom and as a separate headroom frontier, not as extra scheduled bondholder service.
- Added route-substitution diagnostics for ordinary purchase/exchange attempts while keeping borrowing, repayment, and fee conversion as fixed-target routes.
- Changed the frontier route-success default to `diagnostic`; `absolute` mode keeps the old hard floor behavior, and `relative` mode compares against the matched no-bond baseline.

Validation and frontier checks required before using revised frontier results:

```bash
./scripts/run_regenbond_remote_batch.sh validation-1mo
./scripts/run_regenbond_remote_batch.sh validation-smoke
./scripts/run_regenbond_remote_batch.sh validation-pilot
./scripts/run_regenbond_remote_batch.sh validation-full
./scripts/run_regenbond_remote_batch.sh frontier-smoke
./scripts/run_regenbond_remote_batch.sh frontier-maturity-smoke
./scripts/run_regenbond_remote_batch.sh frontier-rola-regeneration-probe
./scripts/run_regenbond_remote_batch.sh frontier-pilot
```

Run `frontier-pilot` only after `frontier-maturity-smoke` and
`frontier-rola-regeneration-probe` pass repayment, voucher-circulation, and
stress checks. Run `frontier-publication` only after the revised pilot is
reviewed.

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
- Loan-induced voucher deposits are a bounded productive-capacity proxy. They
  must be reported separately from empirical baseline deposits and must not be
  double counted as both revenue and collateral without the explicit split.
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

This section is historical. It refers to the earlier frontier pilot before
primary voucher borrowing and bounded lender-held-voucher purchase demand were
enabled. The current paper-facing pilot should be judged against the
ROLA-enabled frontier configuration.

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

- Which cells pass current guardrails, and which fail?
- Which binding constraints appear most often?
- Does guardrail-passing principal fall as coupon rises?
- Does guardrail-passing principal change monotonically with principal ratio?
- Does `connected_5x` improve route success without excessive concentration?
- Do guardrail-passing cells preserve voucher-to-voucher volume and ordinary
  voucher-source activity relative to matched no-bond baselines, while
  reporting voucher-to-voucher count and share as diagnostic composition
  metrics?
- Do guardrail-passing cells avoid rising stable dependency, measured by active-pool stable value share and stable-to-voucher value ratio?
- Which cells both increase voucher-to-voucher volume and maintain at least
  `1.25x` issuer operating/risk headroom?
- Do higher service shares improve scheduled-payment coverage or cash headroom
  while degrading local settlement diagnostics?
- Are unpaid scheduled claims attributed to issuer cashflow, recovered-stable
  headroom, reserve exhaustion, or actual lender-pool liquidity scarcity rather
  than to local pools as hidden guarantors?
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

Producer borrowing is currently modeled as swapping the producer's own voucher
into a lender pool for stable out. The lender pool then holds the producer's
pool-level obligation marker. In bond-frontier runs this also creates a dated
contract cash-service obligation. Repayment or pool-level closure can occur when
the producer voucher leaves the lender pool through borrower stable repayment,
stable purchase by another participant, issuer return, burn, redemption proxy,
or ordinary voucher circulation. Ordinary circulation closes the pool-level
voucher exposure, but it does not automatically create serviceable stable for
bondholders. Stable bond-service recovery is recorded only when stable comes
back through borrower repayment, consumer or third-party stable purchase, or
contract cash-service payment. At the 13-week maturity, remaining cash service
is attempted from producer stable subject to the calibrated recovery/default
rate; unrecovered cash service is written off as defaulted producer debt rather
than silently remaining as serviceable bond cashflow.

Review tasks:

- Keep `LOAN_ISSUED`, `REPAYMENT_EXECUTED`, issuer-return, burn, and redemption as related but distinct metrics.
- Record repayment-failure reasons where possible: liquidity scarcity, route failure, cap or limit failure, redemption failure, voucher-circulation breakdown, shock, or curation decision.
- Treat fixed loan cadence and term as calibrated values if supported by Sarafu evidence; otherwise label them as scenario assumptions.
- Model productive credit: a producer loan should be able to increase future producer stable income or productive capacity according to calibrated or explicit scenario assumptions.
- Model regular producer deposits of stable and vouchers into the producer's pool.
- Review credit-limit logic so producer borrowing capacity can depend on deposited stable/voucher value, with an explicit proposed multiple of `5x` deposited value.
- Track `producer_debt_matured`, `producer_debt_repaid`, `producer_debt_defaulted`, and `producer_debt_closed_by_circulation` alongside quarterly clearing so the bond-service channel can be separated into fees, recovered principal, and defaults.
- Treat consumer voucher purchases and redemption as settlement/closure of producer obligations even when they are not counted as producer self-repayment.
- Preserve ROLA-like voucher borrowing and redemption as a primary strength mechanism: producer vouchers can be used to acquire other producers' vouchers, which are then redeemed for goods, services, production inputs, or cash savings.

### Single-Lender Assignment

`producer_voucher_single_lender=True` is a relationship-credit control. It does not claim that real producer vouchers are accepted by only one pool.

Review tasks:

- Test whether single-lender assignment creates artificial bottlenecks.
- Later compare against multi-lender voucher acceptance to see whether route success improves or risk becomes less accountable.

### Routing

NOAM routing models marketplace-assisted route discovery over published pools, accepted vouchers, limits, and inventories. It is not perfect hidden knowledge. BFS is useful only as a diagnostic on very small graphs because exhaustive search is too slow at Sarafu-like scale.
For the current paper-facing topology, published open venues are lender pools.
Producer and consumer private wallets may be route sources or sinks, but they
are not intermediate venues. The CLC system pool is excluded from the open NOAM
venue set.

Review tasks:

- Reduce NOAM caps, route cache, or beam width and check whether the guardrail frontier shrinks.
- Treat route-success floors as stress parameters, not directly observed Sarafu facts.
- Compare route success with voucher-to-voucher preservation, stable-credit closure, and concentration diagnostics.
- Add route-substitution behavior: if the desired target cannot be routed because of liquidity or limits, agents should be able to choose an alternative acceptable purchase or redemption route where calibration supports it.
- Reconsider the fixed `0.85` p05 route-success floor. It may be removed from
  the primary guardrail frontier or replaced with
  degradation-versus-baseline diagnostics after substitution behavior is
  implemented.

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

- Compare scheduled-payment coverage, service-cash headroom, voucher
  circulation, leakage, concentration, and stress across service-share settings.
- Never treat higher bondholder payment as safe unless settlement guardrails also pass.
- Voucher-denominated fees should not be ignored for bond service. They enter a conversion process where the bond issuer attempts to swap them through pools for stable when liquidity permits; successfully converted stable reserves the configured fee-service share into the lockbox before excess cash goes to the waterfall.
- Track failed fee conversion separately from absence of fees.
- Do not mechanically assume that higher bond-service share reduces local reinvestment unless the modeled waterfall diverts scarce cash from those uses. Instead, report the opportunity cost explicitly: if the same fee cash can fund bondholders, liquidity mandates, repair, insurance, or operations, the model must show which bucket gives way.

### Bond Principal Clearing

The current frontier treats gross bond principal as issuer-raised stable that is
deployed directly to eligible lender pools. Producer borrowing draws from that
lendable stable. Stable recovered by lender pools through borrower repayment,
consumer or third-party stable purchases, and maturity cash-service settlement
is reserved first for scheduled bondholder service through explicit issuer
clearing accounting.

Review tasks:

- Keep quarterly credit clearing explicit: only eligible recovered stable or
  surplus lender-pool liquidity can return to issuer service accounting.
- Keep this distinct from hidden community guarantees: only explicitly eligible cleared amounts should service bond principal.
- Track lender-pool liquidity before and after clearing so credit capacity is not silently drained.
- Report scheduled-payment coverage separately from service-cash headroom. The
  latter is the candidate issuer operating and risk headroom after scheduled
  bondholder service, not proven net surplus.

## Planned Sensitivity Matrix

Run these as future sensitivity checks after the current publication frontier
artifacts are complete and the validation gate remains passing:

- Lower route visibility: reduce NOAM beam width, top-k/top-m, edge caps, or cache.
- Route-substitution sensitivity: compare fixed-target route success, substituted route success, and degradation versus matched no-bond baselines.
- Higher off-ramp pressure: increase stable leakage and check whether liquidity leakage binds.
- Higher coupon: check whether scheduled-payment coverage, service-cash
  headroom, or unpaid claims bind earlier.
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
- Guardrail-passing cells pass issuer service, unpaid claim, route success,
  leakage, concentration, cash stress, stable dependency, and matched no-bond
  degradation guardrails.
- Binding constraints are summarized by scale, coupon, principal ratio, service
  share, scheduled-payment coverage, and cash headroom.
- Surprising results are checked against code and calibration assumptions.
