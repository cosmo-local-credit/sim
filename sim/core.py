from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Literal, List
from collections import deque
import logging

logger = logging.getLogger(__name__)
import math
import random

AssetKind = Literal["stable", "voucher", "other"]
PoolMode = Literal["swap_only", "borrow_only", "mixed", "none"]
AgentRole = Literal["lender", "producer", "consumer", "liquidity_provider", "generic", "ops", "insurance", "clc", "mandates"]

def format_inventory(inv: Dict[str, float]) -> str:
    if not inv:
        return "(empty)"
    items = sorted(inv.items(), key=lambda kv: kv[0])
    return ", ".join(f"{asset}:{amount:.2f}" for asset, amount in items)

# -----------------------------
# Events
# -----------------------------
@dataclass
class Event:
    tick: int
    event_type: str
    actor_id: Optional[str] = None
    pool_id: Optional[str] = None
    asset_id: Optional[str] = None
    amount: Optional[float] = None
    meta: dict = field(default_factory=dict)

class EventLog:
    def __init__(self, maxlen: Optional[int] = None) -> None:
        self.events = deque(maxlen=maxlen)

    def add(self, e: Event) -> None:
        self.events.append(e)

    def tail(self, n: int = 200) -> List[Event]:
        if n <= 0:
            return []
        if n >= len(self.events):
            return list(self.events)
        return list(self.events)[-n:]


# -----------------------------
# Assets / Vouchers
# -----------------------------
@dataclass(frozen=True)
class Asset:
    asset_id: str
    kind: AssetKind
    issuer_id: Optional[str] = None

@dataclass
class VoucherSpec:
    voucher_id: str
    issuer_id: str
    redeem_prob_base: float = 0.85

@dataclass
class IssuerLedger:
    voucher_id: str
    outstanding_supply: float = 0.0
    redeemed_total: float = 0.0

    def issue(self, amount: float) -> None:
        self.outstanding_supply += amount

    def redeem(self, amount: float) -> None:
        amt = min(amount, self.outstanding_supply)
        self.outstanding_supply -= amt
        self.redeemed_total += amt


# -----------------------------
# Pool components
# -----------------------------
@dataclass
class ListingPolicy:
    enabled: bool = True
    routing_allowed: bool = True
    risk_tier: int = 1

class CommitmentRegistry:
    def __init__(self) -> None:
        self.listings: Dict[str, ListingPolicy] = {}

    def list_asset(self, asset_id: str, policy: Optional[ListingPolicy] = None) -> None:
        self.listings[asset_id] = policy or ListingPolicy()

    def is_listed(self, asset_id: str) -> bool:
        pol = self.listings.get(asset_id)
        return bool(pol and pol.enabled)

class ValueIndex:
    def __init__(self, ref_unit: str = "USD") -> None:
        self.ref_unit = ref_unit
        self.values: Dict[str, float] = {}
        self.version: int = 1

    def set_value(self, asset_id: str, value: float) -> None:
        self.values[asset_id] = max(0.0, float(value))
        self.version += 1

    def get_value(self, asset_id: str) -> float:
        return float(self.values.get(asset_id, 0.0))

@dataclass
class LimitRule:
    window_len_ticks: int
    cap_in_global: float

class SwapLimiter:
    def __init__(self) -> None:
        self.rules: Dict[str, LimitRule] = {}
        self.usage: Dict[Tuple[str, int], float] = {}  # (asset_id, bucket) -> used

    def set_rule(self, asset_id: str, rule: LimitRule) -> None:
        self.rules[asset_id] = rule

    def bucket(self, tick: int, asset_id: str) -> int:
        r = self.rules.get(asset_id)
        w = r.window_len_ticks if r else 1
        return tick // max(1, w)

    def check(self, tick: int, asset_id: str, amount_in: float) -> Tuple[bool, str]:
        r = self.rules.get(asset_id)
        if r is None:
            return True, "ok"
        b = self.bucket(tick, asset_id)
        used = self.usage.get((asset_id, b), 0.0)
        if used + amount_in > r.cap_in_global + 1e-9:
            return False, "limit_hit"
        return True, "ok"

    def consume(self, tick: int, asset_id: str, amount_in: float) -> None:
        r = self.rules.get(asset_id)
        if r is None:
            return
        b = self.bucket(tick, asset_id)
        self.usage[(asset_id, b)] = self.usage.get((asset_id, b), 0.0) + amount_in

    def remaining(self, tick: int, asset_id: str) -> float:
        r = self.rules.get(asset_id)
        if r is None:
            return math.inf
        b = self.bucket(tick, asset_id)
        used = self.usage.get((asset_id, b), 0.0)
        return max(0.0, r.cap_in_global - used)

@dataclass
class FeeBreakdown:
    pool_fee: float
    clc_fee: float
    total_fee: float

    def to_dict(self) -> dict:
        return {
            "pool_fee": float(self.pool_fee),
            "clc_fee": float(self.clc_fee),
            "total_fee": float(self.total_fee),
        }
    
class FeeRegistry:
    def __init__(self, pool_fee_rate: float, clc_rake_rate: float) -> None:
        self.pool_fee_rate = float(pool_fee_rate)
        self.clc_rake_rate = float(clc_rake_rate)

    def compute(self, amount_out: float) -> FeeBreakdown:
        pool_fee = amount_out * self.pool_fee_rate
        clc_fee = pool_fee * self.clc_rake_rate
        total_fee = pool_fee
        return FeeBreakdown(pool_fee=pool_fee, clc_fee=clc_fee, total_fee=total_fee)

class Vault:
    def __init__(self) -> None:
        self.inventory: Dict[str, float] = {}

    def get(self, asset_id: str) -> float:
        return float(self.inventory.get(asset_id, 0.0))

    def add(self, asset_id: str, amount: float) -> None:
        self.inventory[asset_id] = self.get(asset_id) + float(amount)

    def sub(self, asset_id: str, amount: float) -> bool:
        amt = float(amount)
        if self.get(asset_id) + 1e-9 < amt:
            return False
        self.inventory[asset_id] = self.get(asset_id) - amt
        if self.inventory[asset_id] <= 1e-12:
            self.inventory.pop(asset_id, None)
        return True


# -----------------------------
# Receipts
# -----------------------------
@dataclass
class SwapReceipt:
    tick: int
    pool_id: str
    actor: str
    asset_in: str
    amount_in: float
    asset_out: str
    amount_out: float
    value_version: int
    fees: FeeBreakdown
    status: Literal["executed", "failed"]
    fail_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "pool_id": self.pool_id,
            "actor": self.actor,
            "asset_in": self.asset_in,
            "amount_in": float(self.amount_in),
            "asset_out": self.asset_out,
            "amount_out": float(self.amount_out),
            "value_version": int(self.value_version),
            "fees": self.fees.to_dict(),
            "status": self.status,
            "fail_reason": self.fail_reason,
        }
class ReceiptStore:
    def __init__(self) -> None:
        self.receipts: List[SwapReceipt] = []

    def add(self, r: SwapReceipt) -> None:
        self.receipts.append(r)

    def tail(self, n: int = 200) -> List[SwapReceipt]:
        return self.receipts[-n:]


# -----------------------------
# Pool policy + Pool
# -----------------------------
@dataclass
class PoolPolicy:
    mode: PoolMode = "mixed"
    role: AgentRole = "generic"
    min_stable_reserve: float = 0.0
    redemption_bias: float = 0.0
    paused: bool = False
    limits_enabled: bool = True
    system_pool: bool = False
    clc_liquidity_symbol: Optional[str] = None

class Pool:
    def __init__(self, pool_id: str, steward_id: str, stable_id: str, policy: PoolPolicy, fees: FeeRegistry) -> None:
        self.pool_id = pool_id
        self.steward_id = steward_id
        self.stable_id = stable_id
        self.policy = policy
        self.debug_inventory: bool = False

        self.registry = CommitmentRegistry()
        self.values = ValueIndex(ref_unit=stable_id)
        self.limiter = SwapLimiter()
        self.fees = fees
        self.vault = Vault()
        self.receipts = ReceiptStore()

        # simple fee ledgers
        self.fee_ledger_pool: Dict[str, float] = {}  # by asset
        self.fee_ledger_clc: Dict[str, float] = {}

    def _debug_inventory_change(self, action: str, counterparty: str, asset: str, amount: float,
                                before: Dict[str, float], after: Dict[str, float]) -> None:
        if not self.debug_inventory or not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "[INV] pool=%s role=%s action=%s counterparty=%s asset=%s amount=%.2f before={ %s } after={ %s }",
            self.pool_id,
            self.policy.role,
            action,
            counterparty,
            asset,
            amount,
            format_inventory(before),
            format_inventory(after),
        )

    def list_asset_with_value_and_limit(self, asset_id: str, value: float, window_len: int, cap_in: float) -> None:
        self.registry.list_asset(asset_id)
        self.values.set_value(asset_id, value)
        self.limiter.set_rule(asset_id, LimitRule(window_len_ticks=window_len, cap_in_global=cap_in))

    def can_swap(self, tick: int, asset_in: str, amount_in: float, asset_out: str, amount_out: float) -> Tuple[bool, str]:
        if self.policy.paused:
            return False, "paused"
        if not self.registry.is_listed(asset_in) or not self.registry.is_listed(asset_out):
            return False, "not_listed"
        if self.policy.limits_enabled:
            ok, reason = self.limiter.check(tick, asset_in, amount_in)
            if not ok:
                return False, reason
        if self.vault.get(asset_out) + 1e-9 < amount_out:
            return False, "insufficient_inventory"
        # guardrail: stable reserve (for stable outflow)
        if asset_out == self.stable_id and self.vault.get(self.stable_id) - amount_out < self.policy.min_stable_reserve - 1e-9:
            return False, "reserve_guardrail"
        # policy mode: stable outflow via swaps allowed?
        if self.policy.role == "lender":
            if asset_in != self.stable_id and asset_out != self.stable_id:
                return False, "lender_requires_stable_leg"
        if self.policy.role == "liquidity_provider":
            if asset_in.startswith("VCHR:") or asset_out.startswith("VCHR:"):
                return False, "lp_no_voucher_swaps"
        # Producers/consumers do not allow others to swap out their stables.
        if self.policy.role == "consumer":
            if asset_out == self.stable_id:
                return False, "consumer_no_stable_outflow"
        if self.policy.role == "producer":
            if asset_out == self.stable_id:
                return False, "producer_no_stable_outflow"
        if self.policy.role == "clc":
            allowed = {self.stable_id}
            if self.policy.clc_liquidity_symbol:
                allowed.add(self.policy.clc_liquidity_symbol)
            if asset_in not in allowed and asset_out not in allowed:
                return False, "clc_requires_stable_leg"
        if asset_out == self.stable_id and self.policy.mode == "borrow_only":
            return False, "stable_outflow_not_allowed"
        if asset_out == self.stable_id and self.policy.mode == "none":
            return False, "stable_outflow_not_allowed"
        if asset_out != self.stable_id and not self.policy.mode in ("mixed", "swap_only", "borrow_only", "none"):
            return False, "policy_error"
        if self.policy.mode == "none":
            return False, "swaps_disabled"
        if self.policy.mode == "borrow_only":
            # in MVP-1: treat borrow_only as swaps allowed except stable cannot leave by swap
            return False, "swaps_disabled_for_borrow_only"
        return True, "ok"

    def quote_swap(self, asset_in: str, amount_in: float, asset_out: str) -> Tuple[bool, str, float, float]:
        vin = self.values.get_value(asset_in)
        vout = self.values.get_value(asset_out)
        if vin <= 0 or vout <= 0:
            return False, "missing_price", 0.0, 0.0
        value_in_ref = amount_in * vin
        amount_out_gross = value_in_ref / vout
        fees = self.fees.compute(amount_out_gross)
        amount_out_net = max(0.0, amount_out_gross - fees.total_fee)
        return True, "ok", amount_out_net, fees.total_fee

    def execute_swap(self, tick: int, actor: str, asset_in: str, amount_in: float, asset_out: str) -> SwapReceipt:
        debug = self.debug_inventory and logger.isEnabledFor(logging.DEBUG)
        okq, reasonq, amount_out, _fee_amt = self.quote_swap(asset_in, amount_in, asset_out)
        if not okq:
            r = SwapReceipt(
                tick=tick, pool_id=self.pool_id, actor=actor,
                asset_in=asset_in, amount_in=amount_in, asset_out=asset_out, amount_out=0.0,
                value_version=self.values.version, fees=FeeBreakdown(0.0, 0.0, 0.0),
                status="failed", fail_reason=reasonq
            )
            self.receipts.add(r)
            return r

        fees = self.fees.compute(amount_out + _fee_amt)  # compute on gross
        gross_out = amount_out + fees.total_fee

        ok, reason = self.can_swap(tick, asset_in, amount_in, asset_out, gross_out)
        if not ok:
            r = SwapReceipt(
                tick=tick, pool_id=self.pool_id, actor=actor,
                asset_in=asset_in, amount_in=amount_in, asset_out=asset_out, amount_out=0.0,
                value_version=self.values.version, fees=fees,
                status="failed", fail_reason=reason
            )
            self.receipts.add(r)
            return r

        # pool takes in asset_in
        if debug:
            before_in = dict(self.vault.inventory)
        self.vault.add(asset_in, amount_in)
        if debug:
            after_in = dict(self.vault.inventory)
            self._debug_inventory_change("swap_in", actor, asset_in, amount_in, before_in, after_in)
        # pool pays out asset_out
        if debug:
            before_out = dict(self.vault.inventory)
        if not self.vault.sub(asset_out, gross_out):
            # rollback asset_in on failure
            self.vault.sub(asset_in, amount_in)
            r = SwapReceipt(
                tick=tick, pool_id=self.pool_id, actor=actor,
                asset_in=asset_in, amount_in=amount_in, asset_out=asset_out, amount_out=0.0,
                value_version=self.values.version, fees=fees,
                status="failed", fail_reason="insufficient_inventory"
            )
            self.receipts.add(r)
            return r
        if debug:
            after_out = dict(self.vault.inventory)
            self._debug_inventory_change("swap_out", actor, asset_out, gross_out, before_out, after_out)
        # consume limiter on success
        if self.policy.limits_enabled:
            self.limiter.consume(tick, asset_in, amount_in)

        # accrue fees in asset_out (fee taken from gross out)
        self.fee_ledger_pool[asset_out] = self.fee_ledger_pool.get(asset_out, 0.0) + fees.pool_fee
        self.fee_ledger_clc[asset_out] = self.fee_ledger_clc.get(asset_out, 0.0) + fees.clc_fee

        r = SwapReceipt(
            tick=tick, pool_id=self.pool_id, actor=actor,
            asset_in=asset_in, amount_in=amount_in, asset_out=asset_out, amount_out=amount_out,
            value_version=self.values.version, fees=fees,
            status="executed", fail_reason=None
        )
        self.receipts.add(r)
        return r
