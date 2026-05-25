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
Voucher-pool overlap calibration is also exported as aggregate topology only:
it reports how many KSh/KES voucher tokens appear in one or more pools, a
bucketed degree distribution, and shared-voucher pool-pool edge counts. It does
not include raw token or pool addresses and should be read as route potential,
not participant intent.
The current ROLA/frontier-pilot purchase-demand setting uses a network-level
weekly stable budget, not a per-consumer allowance. It is set to the direct
pool-era empirical mean stable-to-voucher purchase cash value per week. In the
research workspace, the aggregate stable-to-voucher purchase value is also
exported as a weekly distribution and figure so the flat budget can later be
replaced by a dynamic purchase-timing model once buyer-income moments are
added.
The productive-credit voucher activity boost calibration uses privacy-safe
post-borrow event windows around voucher-to-stable borrow-proxy events. It
reports same-voucher source activity and target-side voucher demand before and
after borrowing, and converts only the source-side motif into the simulator's
voucher-source boost coefficients. In the current artifact, same-voucher
source activity does not rise after borrow-proxy events, so the calibrated
source-weight boost is 0.0 and the source-size multiplier is 1.0. Target-side
voucher demand rises, which supports the purchase-demand channel rather than an
extra uncalibrated producer source boost.
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
