# Public Sarafu Calibration Bundle

This directory contains the aggregate, privacy-safe calibration inputs needed
to run the RegenBond Monte Carlo workflows from a standalone public clone of
the `sim` repository.

The per-pool calibration file is anonymized into synthetic template rows. It
does not include raw transactions, addresses, report text, pool labels, or pool
IDs. The simulator uses these templates only for tier mix, activity rates,
repayment/return priors, backing-liquidity scale, and impact-report exposure.
The paper calibration cohort is Kenyan KSh/KES community lending pools. Pools
must have KSh/KES accepted-voucher membership evidence, must not be identified
as non-Kenya by metadata, must not be in the exact named non-community
exclusion list, and must have at least four distinct active swap weeks.
The current exported cohort has 73 lender-pool templates,
1247 accepted-voucher member slots, 996 unique
producer-voucher wallets, and 462 recommended external
non-producer consumer wallets. Stable-side pool interaction has
950 address-pool slots; this is an interaction
count, not the consumer-wallet count used by the Monte Carlo topology.
Accepted voucher counts are exported as `accepted_voucher_members`; these are
pool membership slots. Unique producer-voucher wallet counts and multi-pool
membership degree are exported in the voucher-pool overlap calibration. Active
interactors remain `total_users`. The stable actor demographics file separates
external non-producer stable-side users from producer addresses that also use
stable-side interactions. Raw included/excluded pool-address audits remain
local-only in the research workspace.
In the simulator, these empirical pools are open lender pools. Producer and
consumer wallets are private source/sink wallets: they can start or receive a
route, but other agents cannot traverse or clear through them. A producer's
voucher can be accepted by multiple lender pools according to the exported
overlap calibration.

## Paper-Facing Agent And Flow Semantics

The calibration bundle maps empirical Sarafu records into four paper-facing
roles:

| Role | Calibrated count or source | Model semantics |
| --- | ---: | --- |
| Open pools | 73 | Kenya KES/KSh community lending pools. These are the only open automatic-swap, routing, and clearing venues. Internally they use the code role `lender`. |
| Producer wallets | 996 | Unique KES/KSh producer-voucher issuers. Each wallet has one own voucher and private stable holdings; its voucher may be accepted by multiple open pools. |
| Accepted producer-voucher slots | 1,247 | Pool membership/listing slots, not extra producer wallets. These calibrate multi-pool acceptance and route potential. |
| Consumer wallets | 462 | External non-producer stable-side users recommended for the topology. Stable-side pool interaction has 950 address-pool slots, which is an interaction count rather than the consumer-wallet count. |

Producer voucher and stable deposits move assets from private producer wallets
into open pools. Deposits increase pool-side credit capacity and limits, but
they are not outstanding debt until a producer later swaps its own voucher into
a pool for stable or another useful voucher. Stable-to-voucher purchase value
seeds private producer/consumer stable balances for purchase behavior; it is
not treated as philanthropic pool backing.

Validation/no-bond runs include historical stable backing into open pools as a
calibration moment. In the current bundle, the inclusive pool stable backing
target is `6,574.127191`, all assigned to the 73 open pools. The broader
private-stable plus on-ramp accounting proxy is reported separately and should
not be compared to this pool-backing target. Bond-frontier runs omit the
historical philanthropic/programmatic backing event and instead use bond
principal as the modeled stable liquidity injection into eligible pools.
The settlement reliability anchor file adds aggregate ROLA-like voucher
circulation, ROSCA-like stable-credit, same-token return, submitted-swap
execution, and current cluster-topology metrics. The unit-normalization file
records how KES/KSh voucher obligations are converted to USD-equivalent for
bond-accounting columns, using successful KES/USD pool swaps and the current
simulator convention that individual voucher units are 1 voucher = 1 KSh.
Pool backing is split into cash/stable backing and KSh/KES voucher backing so
cash injections can be calibrated separately from voucher-denominated credit
capacity.
The voucher circulation
baseline file records the same motifs over the pool era and recent trailing
windows. The stable dependency anchor file records stable/cash flow shares,
voucher flow shares, and stable-to-voucher dependency proxies under both the
legacy metadata-weighted voucher valuation and the simulator-comparable strict
1 voucher = 1 KSh convention. Additional aggregate tables calibrate producer
deposit proxies, productive-credit timing, debt-removal purchases, voucher-fee
conversion, quarterly clearing, and route substitution diagnostics. These are
empirical settlement motifs and scenario anchors, not direct pool inventory
snapshots or a failed-route denominator.
Validation/frontier scenarios use these motif shares as decision priors and
diagnostics. They do not mechanically rebalance successful routes to hit the
empirical shares; realized composition still depends on wallet balances, lender
pool listings, limits, inventory, and route feasibility. Consumer
stable-to-voucher behavior is represented by the calibrated purchase-demand
channel, while generic ordinary wallet routes are kept separate from that
purchase process.
Voucher-pool overlap calibration is also exported as aggregate topology only:
it reports how many KSh/KES voucher tokens appear in one or more pools, a
bucketed degree distribution, and shared-voucher pool-pool edge counts. It does
not include raw token or pool addresses and should be read as route potential,
not participant intent.
The current ROLA/frontier purchase-demand setting uses network-level stable
budgets, not per-consumer allowances. The paper-facing default uses the
trailing-52-week actor split: original-issuer self purchases, other-producer
stable-to-voucher purchases, and external non-producer purchases are exported
separately. external_nonproducer is not automatically "consumer"; it only means
the initiating wallet is not linked to a cohort producer voucher. Producer
stable reuse is separately calibrated from producer voucher-to-stable receipts
versus producer stable-to-voucher spending, with the residual treated as stable
that exits the local network.
The productive-credit voucher activity boost calibration uses privacy-safe
post-borrow event windows around voucher-to-stable borrow-proxy events. It
reports same-voucher source activity and target-side voucher demand before and
after borrowing, and converts only the source-side motif into the simulator's
voucher-source boost coefficients. In the current artifact, same-voucher
source activity does not rise after borrow-proxy events, so the calibrated
source-weight boost is 0.0 and the source-size multiplier is 1.0. Target-side
voucher demand rises, which supports the purchase-demand channel rather than an
extra uncalibrated producer source boost.
The aggregate productive-credit calibration splits loan-enabled inflow into
`0.244881` stable retained and `0.755119` new producer-voucher deposit capacity,
capped at `0.189536` voucher-deposit growth per month. This is a bounded
capacity proxy, not causal proof that every loan increases production.
The public bundle also exports a standalone stable/cash deposit weekly series.
That series counts ordinary successful pool-era stable/cash deposit logs and
excludes swap-associated deposit logs, while reporting the inclusive
deposit-log total separately. It is intended to distinguish ordinary stable
backing/deposit timing from stable used directly in swaps.
An all-pool weekly swap series is exported as well, with successful pool-swap
log counts and one-sided normalized input value by week. Output value and
stable/voucher source-pair counts are included for audit.

The bond-frontier safety tests use these files in three ways:

- engine validation reports voucher-to-voucher motifs, strict 1 KSh stable-flow
  shares, and separate active-pool inventory snapshots as calibration
  diagnostics;
- frontier cells are rejected when bond liquidity materially degrades
  voucher-to-voucher circulation versus the matched no-bond baseline;
- frontier cells are rejected when stable/bond liquidity materially increases
  active-pool stable inventory dependency or reduces active-pool voucher
  inventory value share versus the matched no-bond baseline.

Regenerate from the research workspace with:

```bash
python scripts/export_public_sarafu_calibration.py \
  --source ../RegenBonds/analysis \
  --output analysis/sarafu_calibration
```
