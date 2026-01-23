from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Optional, Tuple
from collections import deque
import heapq
import logging
import math
import numpy as np
import random

from .config import ScenarioConfig
from .core import Event, EventLog, format_inventory, Pool, PoolPolicy, FeeRegistry
from .factory import PoolFactory, Agent
from .router import Router, RoutePlan, Hop
from .metrics import MetricsStore

logger = logging.getLogger(__name__)

MONTH_TICKS = 4

@dataclass
class NoamEdgeState:
    p_success: float
    lambda_price: float
    last_fail_reason: Optional[str] = None

@dataclass(slots=True)
class NoamState:
    asset_id: str
    score: float
    path: list[Hop]

@dataclass(slots=True)
class NoamRouteCache:
    remaining: Dict[Tuple[str, str], float]
    inventory: Dict[Tuple[str, str], float]
    value: Dict[Tuple[str, str], float]

@dataclass(slots=True)
class NoamClearingEdge:
    pool_id: str
    asset_in: str
    asset_out: str
    score: float
    cap_value: float
    value_in: float
    value_out: float

@dataclass(slots=True)
class NoamClearingCycle:
    edges: list[NoamClearingEdge]
    score: float
    cap_value: float

class SimulationEngine:
    def __init__(self, cfg: ScenarioConfig, seed: int = 1) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.tick: int = 0
        self.log = EventLog(maxlen=cfg.event_log_maxlen)
        self.metrics = MetricsStore()

        self.factory = PoolFactory(cfg)
        self.router = Router(max_hops=cfg.max_hops)

        self.agents: Dict[str, Agent] = {}
        self.pools: Dict[str, "Pool"] = {}

        # indexes
        self.accept_index: Dict[str, Set[str]] = {}
        self.offer_index: Dict[str, Set[str]] = {}
        self.pool_affinity: Dict[Tuple[str, str], float] = {}
        self._swap_volume_usd_tick: float = 0.0
        self._swap_volume_usd_by_pool: Dict[str, float] = {}
        self._noam_routing_swaps_tick: int = 0
        self._noam_clearing_swaps_tick: int = 0
        self._liquidity_tick: int = -1
        self._liquidity_by_asset: Dict[str, float] = {}
        self._liquidity_initialized: bool = False
        self._utilization_boost: float = 1.0
        self._last_utilization_rate: float = 0.0
        self._loan_phase_by_agent: Dict[str, int] = {}
        self.system_pools: Set[str] = set()
        self.ops_pool_id: Optional[str] = None
        self.insurance_pool_id: Optional[str] = None
        self.clc_pool_id: Optional[str] = None
        self.mandates_pool_id: Optional[str] = None
        self._claims_paid_usd_tick: float = 0.0
        self._claims_unpaid_usd_tick: float = 0.0
        self._incidents_tick: int = 0
        self._fee_access_budget_usd: float = 0.0
        self._clc_access_open: bool = False
        self._waterfall_last: Dict[str, float] = {}
        self._waterfall_external_inflows: Dict[str, float] = {}
        self._fee_pool_cumulative_usd: float = 0.0
        self._fee_clc_cumulative_usd: float = 0.0
        self._fee_pool_cumulative_voucher: float = 0.0
        self._fee_clc_cumulative_voucher: float = 0.0
        self._pool_growth_remainder: float = 0.0
        self._sclc_minted_total: float = 0.0
        self._stable_onramp_usd_tick: float = 0.0
        self._stable_offramp_usd_tick: float = 0.0
        self._pool_value_cache: Dict[str, float] = {}
        self._pool_voucher_value_cache: Dict[str, float] = {}
        self._swap_success_ema: Dict[str, float] = {}
        self._swap_attempt_counts: Dict[str, int] = {}
        self._noam_top_pools: Dict[str, list[str]] = {}
        self._noam_top_out: Dict[Tuple[str, str], list[str]] = {}
        self._noam_edge_state: Dict[Tuple[str, str, str], NoamEdgeState] = {}
        self._noam_escape_score: Dict[str, float] = {}
        self._noam_hubs: Set[str] = set()
        self._noam_adj_forward: Dict[str, Set[str]] = {}
        self._noam_adj_reverse: Dict[str, Set[str]] = {}
        self._noam_overlay_graph: Dict[str, list[Tuple[str, float]]] = {}
        self._noam_overlay_paths_cache: Dict[Tuple[str, str], Tuple[float, list[str]]] = {}
        self._noam_last_refresh_tick: int = -1
        self._noam_last_overlay_tick: int = -1
        self._noam_failure_cache: Dict[Tuple[str, str, str], int] = {}
        self._noam_route_cache: Dict[Tuple[str, str, str, int], Tuple[int, RoutePlan]] = {}
        self._noam_distance_cache_tick: int = -1
        self._noam_distance_cache: Dict[Tuple[str, int], Dict[str, int]] = {}
        self._noam_near_hubs_cache_tick: int = -1
        self._noam_near_hubs_cache: Dict[Tuple[str, bool], list[str]] = {}
        self._noam_cap_scale_cache_tick: int = -1
        self._noam_cap_scale_cache_value: float = 1.0

        self.initial_stable_total: float = 0.0
        self._bootstrap()

    def _bootstrap(self) -> None:
        if self.cfg.economics_enabled:
            self._bootstrap_system_pools()
        initial_roles = ["producer", "producer", "lender", "liquidity_provider"]
        for idx in range(self.cfg.initial_pools):
            role = initial_roles[idx] if idx < len(initial_roles) else None
            self.add_pool(role=role)

        self.snapshot_metrics()

    def _bootstrap_system_pools(self) -> None:
        self.ops_pool_id = self._create_system_pool("ops", "sys_ops")
        self.insurance_pool_id = self._create_system_pool("insurance", "sys_insurance")
        self.mandates_pool_id = self._create_system_pool("mandates", "sys_mandates")
        self.clc_pool_id = self._create_system_pool("clc", "sys_clc")

    def _create_system_pool(self, role: str, pool_id: str) -> str:
        policy = PoolPolicy(
            mode="swap_only" if role == "clc" else "none",
            role=role,
            min_stable_reserve=0.0,
            redemption_bias=0.0,
            paused=True,
            limits_enabled=False,
            system_pool=True,
        )
        fees = FeeRegistry(pool_fee_rate=0.0, clc_rake_rate=0.0)
        pool = Pool(pool_id=pool_id, steward_id=f"system:{role}", stable_id=self.cfg.stable_symbol, policy=policy, fees=fees)
        pool.debug_inventory = self.cfg.debug_inventory
        pool.list_asset_with_value_and_limit(self.cfg.stable_symbol, value=1.0, window_len=1, cap_in=1e12)
        if role == "clc" and self.cfg.sclc_symbol:
            pool.list_asset_with_value_and_limit(self.cfg.sclc_symbol, value=1.0, window_len=1, cap_in=1e12)
            pool.policy.clc_liquidity_symbol = self.cfg.sclc_symbol
        self.pools[pool_id] = pool
        self.system_pools.add(pool_id)
        self.log.add(Event(self.tick, "SYSTEM_POOL_CREATED", pool_id=pool_id, meta={"role": role}))
        return pool_id

    def _debug_inventory_change(self, pool: "Pool", action: str, counterparty: str,
                                asset: str, amount: float,
                                before: Dict[str, float], after: Dict[str, float]) -> None:
        if not self.cfg.debug_inventory or not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "[INV] pool=%s role=%s action=%s counterparty=%s asset=%s amount=%.2f before={ %s } after={ %s }",
            pool.pool_id,
            pool.policy.role,
            action,
            counterparty,
            asset,
            amount,
            format_inventory(before),
            format_inventory(after),
        )

    def _vault_add(self, pool: "Pool", asset: str, amount: float,
                   action: str, counterparty: str) -> None:
        debug = self.cfg.debug_inventory and logger.isEnabledFor(logging.DEBUG)
        if debug:
            before = dict(pool.vault.inventory)
        pool.vault.add(asset, amount)
        self._update_pool_caches(pool, asset, float(amount))
        if debug:
            after = dict(pool.vault.inventory)
            self._debug_inventory_change(pool, action, counterparty, asset, amount, before, after)

    def _vault_sub(self, pool: "Pool", asset: str, amount: float,
                   action: str, counterparty: str) -> bool:
        debug = self.cfg.debug_inventory and logger.isEnabledFor(logging.DEBUG)
        if debug:
            before = dict(pool.vault.inventory)
        ok = pool.vault.sub(asset, amount)
        if ok:
            self._update_pool_caches(pool, asset, -float(amount))
        if ok and debug:
            after = dict(pool.vault.inventory)
            self._debug_inventory_change(pool, action, counterparty, asset, amount, before, after)
        return ok

    def _asset_value(self, pool: "Pool", asset_id: str) -> float:
        v = pool.values.get_value(asset_id)
        return v if v > 0.0 else 1.0

    def _rebuild_pool_value_cache(self, pool: "Pool") -> float:
        total = 0.0
        for asset, amt in pool.vault.inventory.items():
            if amt <= 1e-12:
                continue
            value = pool.values.get_value(asset)
            if value <= 0.0:
                continue
            total += amt * value
        self._pool_value_cache[pool.pool_id] = total
        return total

    def _rebuild_pool_voucher_cache(self, pool: "Pool") -> float:
        total = 0.0
        for asset, amt in pool.vault.inventory.items():
            if amt <= 1e-12 or not asset.startswith("VCHR:"):
                continue
            value = pool.values.get_value(asset)
            if value <= 0.0:
                value = 1.0
            total += amt * value
        self._pool_voucher_value_cache[pool.pool_id] = total
        return total

    def _update_pool_caches(self, pool: "Pool", asset: str, delta_amount: float) -> None:
        if abs(delta_amount) <= 1e-12:
            return
        pool_id = pool.pool_id
        if pool_id not in self._pool_value_cache:
            self._rebuild_pool_value_cache(pool)
        else:
            value = pool.values.get_value(asset)
            if value > 0.0:
                total = self._pool_value_cache.get(pool_id, 0.0) + (delta_amount * value)
                if total <= 1e-9:
                    total = 0.0
                self._pool_value_cache[pool_id] = total

        if asset.startswith("VCHR:"):
            if pool_id not in self._pool_voucher_value_cache:
                self._rebuild_pool_voucher_cache(pool)
            else:
                value = pool.values.get_value(asset)
                if value <= 0.0:
                    value = 1.0
                total = self._pool_voucher_value_cache.get(pool_id, 0.0) + (delta_amount * value)
                if total <= 1e-9:
                    total = 0.0
                self._pool_voucher_value_cache[pool_id] = total

        if asset != self.cfg.sclc_symbol and self._is_routable_pool(pool):
            usd = delta_amount * self._asset_value(pool, asset)
            if abs(usd) > 1e-12:
                total = self._liquidity_by_asset.get(asset, 0.0) + usd
                if total <= 1e-9:
                    self._liquidity_by_asset.pop(asset, None)
                else:
                    self._liquidity_by_asset[asset] = total
                self._liquidity_initialized = True

    def _pool_total_value(self, pool: "Pool") -> float:
        cached = self._pool_value_cache.get(pool.pool_id)
        if cached is None:
            return self._rebuild_pool_value_cache(pool)
        return cached

    def _pool_voucher_value_usd(self, pool: "Pool") -> float:
        cached = self._pool_voucher_value_cache.get(pool.pool_id)
        if cached is None:
            return self._rebuild_pool_voucher_cache(pool)
        return cached

    def _noam_assets(self) -> list[str]:
        stable_id = self.cfg.stable_symbol
        assets = [stable_id]
        for asset_id in self.factory.asset_universe.keys():
            if asset_id.startswith("VCHR:"):
                assets.append(asset_id)
        return list(dict.fromkeys(assets))

    def _noam_pool_score(self, pool: "Pool") -> float:
        value = self._pool_total_value(pool)
        listings = len(pool.registry.listings)
        score = math.log1p(max(0.0, value)) * (1.0 + math.log1p(max(1, listings)))
        reliability = float(self._swap_success_ema.get(pool.pool_id, 1.0))
        if reliability <= 0.0:
            reliability = 1e-3
        return score * reliability

    def _noam_cap_scale(self) -> float:
        if not self.cfg.noam_dynamic_caps_enabled:
            return 1.0
        if self._noam_cap_scale_cache_tick == self.tick:
            return self._noam_cap_scale_cache_value
        ref = max(1, int(self.cfg.noam_dynamic_cap_reference_pools or 1))
        pool_count = sum(1 for p in self.pools.values() if not p.policy.system_pool)
        if pool_count <= ref:
            scale = 1.0
        else:
            scale = math.sqrt(ref / max(1, pool_count))
        self._noam_cap_scale_cache_tick = self.tick
        self._noam_cap_scale_cache_value = scale
        return scale

    def _noam_scaled_cap(self, base: int, min_value: int) -> int:
        if base <= 0:
            return 0
        scale = self._noam_cap_scale()
        if scale >= 1.0:
            return base
        min_value = min(min_value, base)
        return max(min_value, int(math.ceil(base * scale)))

    def _noam_cached_remaining(self, pool: "Pool", asset_in: str, cache: Optional[NoamRouteCache]) -> float:
        if not pool.policy.limits_enabled or cache is None:
            return pool.limiter.remaining(self.tick, asset_in)
        key = (pool.pool_id, asset_in)
        if key not in cache.remaining:
            cache.remaining[key] = pool.limiter.remaining(self.tick, asset_in)
        return cache.remaining[key]

    def _noam_cached_inventory(self, pool: "Pool", asset_id: str, cache: Optional[NoamRouteCache]) -> float:
        if cache is None:
            return pool.vault.get(asset_id)
        key = (pool.pool_id, asset_id)
        if key not in cache.inventory:
            cache.inventory[key] = pool.vault.get(asset_id)
        return cache.inventory[key]

    def _noam_cached_value(self, pool: "Pool", asset_id: str, cache: Optional[NoamRouteCache]) -> float:
        if cache is None:
            return self._asset_value(pool, asset_id)
        key = (pool.pool_id, asset_id)
        if key not in cache.value:
            cache.value[key] = self._asset_value(pool, asset_id)
        return cache.value[key]

    def _noam_distance_to_target(self, target_asset: str, max_hops: int) -> Dict[str, int]:
        if self._noam_distance_cache_tick != self.tick:
            self._noam_distance_cache = {}
            self._noam_distance_cache_tick = self.tick
        key = (target_asset, max_hops)
        cached = self._noam_distance_cache.get(key)
        if cached is not None:
            return cached
        dist: Dict[str, int] = {target_asset: 0}
        queue = deque([target_asset])
        while queue:
            asset = queue.popleft()
            depth = dist[asset]
            if depth >= max_hops:
                continue
            for prev_asset in self._noam_adj_reverse.get(asset, set()):
                if prev_asset not in dist:
                    dist[prev_asset] = depth + 1
                    queue.append(prev_asset)
        self._noam_distance_cache[key] = dist
        return dist

    def _noam_failure_active(self, pool_id: str, asset_in: str, asset_out: str) -> bool:
        ttl = int(self.cfg.noam_failure_ttl_ticks or 0)
        if ttl <= 0:
            return False
        key = (pool_id, asset_in, asset_out)
        expires = self._noam_failure_cache.get(key)
        if expires is None:
            return False
        if self.tick < expires:
            return True
        self._noam_failure_cache.pop(key, None)
        return False

    def _noam_record_failure(self, pool_id: str, asset_in: str, asset_out: str) -> None:
        ttl = int(self.cfg.noam_failure_ttl_ticks or 0)
        if ttl <= 0:
            return
        self._noam_failure_cache[(pool_id, asset_in, asset_out)] = self.tick + ttl

    def _noam_route_cache_key(
        self,
        source_pool: "Pool",
        start_asset: str,
        target_asset: str,
        amount_in: float,
        target_pools: Optional[Set[str]],
    ) -> Optional[Tuple[str, str, str, int]]:
        ttl = int(self.cfg.noam_route_cache_ttl_ticks or 0)
        bucket = float(self.cfg.noam_route_cache_bucket_usd or 0.0)
        if ttl <= 0 or bucket <= 0.0 or target_pools is not None:
            return None
        amount_usd = amount_in * self._asset_value(source_pool, start_asset)
        bucket_id = int(amount_usd / bucket)
        return (source_pool.pool_id, start_asset, target_asset, bucket_id)

    def _noam_route_cache_get(self, key: Tuple[str, str, str, int]) -> Optional[RoutePlan]:
        cached = self._noam_route_cache.get(key)
        if cached is None:
            return None
        expires, plan = cached
        if self.tick < expires:
            return plan
        self._noam_route_cache.pop(key, None)
        return None

    def _noam_route_cache_store(self, source_pool_id: str, plan: RoutePlan, amount_in: float) -> None:
        if self.cfg.routing_mode != "noam":
            return
        ttl = int(self.cfg.noam_route_cache_ttl_ticks or 0)
        bucket = float(self.cfg.noam_route_cache_bucket_usd or 0.0)
        if ttl <= 0 or bucket <= 0.0:
            return
        if not plan.hops:
            return
        source_pool = self.pools.get(source_pool_id)
        if source_pool is None:
            return
        start_asset = plan.hops[0].asset_in
        target_asset = plan.hops[-1].asset_out
        amount_usd = amount_in * self._asset_value(source_pool, start_asset)
        bucket_id = int(amount_usd / bucket)
        key = (source_pool_id, start_asset, target_asset, bucket_id)
        self._noam_route_cache[key] = (self.tick + ttl, plan)

    def _maybe_refresh_noam_working_set(self) -> None:
        if self.cfg.routing_mode != "noam":
            return
        interval = max(1, int(self.cfg.noam_topk_refresh_ticks or 1))
        if self._noam_last_refresh_tick >= 0:
            if (self.tick - self._noam_last_refresh_tick) < interval:
                if self.cfg.noam_overlay_enabled:
                    self._maybe_refresh_noam_overlay()
                return
        self._refresh_noam_working_set()
        if self.cfg.noam_overlay_enabled:
            self._maybe_refresh_noam_overlay()

    def _maybe_refresh_noam_overlay(self) -> None:
        if not self.cfg.noam_overlay_enabled:
            self._noam_overlay_graph = {}
            return
        interval = max(1, int(self.cfg.noam_overlay_refresh_ticks or 1))
        if self._noam_last_overlay_tick >= 0:
            if (self.tick - self._noam_last_overlay_tick) < interval:
                return
        self._refresh_noam_overlay_graph()

    def _refresh_noam_working_set(self) -> None:
        assets = self._noam_assets()
        base_k = max(1, int(self.cfg.noam_topk_pools_per_asset or 1))
        base_m = max(1, int(self.cfg.noam_topm_out_per_pool or 1))
        min_k = max(1, int(self.cfg.noam_dynamic_min_topk or 1))
        min_m = max(1, int(self.cfg.noam_dynamic_min_topm or 1))
        top_k = max(1, self._noam_scaled_cap(base_k, min_k))
        top_m = max(1, self._noam_scaled_cap(base_m, min_m))

        top_pools: Dict[str, list[str]] = {}
        for asset_id in assets:
            pool_ids = list(self.accept_index.get(asset_id, set()))
            if not pool_ids:
                continue
            scored = []
            for pid in pool_ids:
                pool = self.pools.get(pid)
                if pool is None:
                    continue
                scored.append((self._noam_pool_score(pool), pid))
            scored.sort(reverse=True)
            top_pools[asset_id] = [pid for _, pid in scored[:top_k]]

        if self.clc_pool_id and self.clc_pool_id in self.pools:
            clc_pool = self.pools[self.clc_pool_id]
            for asset_id in assets:
                if not clc_pool.registry.is_listed(asset_id):
                    continue
                pools = top_pools.get(asset_id, [])
                if self.clc_pool_id in pools:
                    continue
                if not pools:
                    top_pools[asset_id] = [self.clc_pool_id]
                    continue
                if len(pools) >= top_k:
                    pools = list(pools)
                    pools[-1] = self.clc_pool_id
                    top_pools[asset_id] = pools
                else:
                    pools = list(pools)
                    pools.append(self.clc_pool_id)
                    top_pools[asset_id] = pools

        top_out: Dict[Tuple[str, str], list[str]] = {}
        for asset_in, pool_ids in top_pools.items():
            for pid in pool_ids:
                pool = self.pools.get(pid)
                if pool is None:
                    continue
                candidates = []
                for asset_out, amt in pool.vault.inventory.items():
                    if amt <= 1e-9:
                        continue
                    if asset_out == asset_in:
                        continue
                    if asset_out != self.cfg.stable_symbol and not asset_out.startswith("VCHR:"):
                        continue
                    if not pool.registry.is_listed(asset_out):
                        continue
                    value = pool.values.get_value(asset_out)
                    if value <= 0.0:
                        value = 1.0
                    candidates.append((amt * value, asset_out))
                if not candidates:
                    continue
                candidates.sort(reverse=True)
                top_out[(pid, asset_in)] = [asset_out for _, asset_out in candidates[:top_m]]

        adj_forward: Dict[str, Set[str]] = {}
        adj_reverse: Dict[str, Set[str]] = {}
        for (pid, asset_in), outs in top_out.items():
            for asset_out in outs:
                adj_forward.setdefault(asset_in, set()).add(asset_out)
                adj_reverse.setdefault(asset_out, set()).add(asset_in)

        escape_score: Dict[str, float] = {}
        for asset_id in assets:
            deg = len(top_pools.get(asset_id, []))
            escape_score[asset_id] = math.log1p(deg)

        hub_count = max(1, int(self.cfg.noam_hub_asset_count or 1))
        hub_scores = []
        for asset_id in assets:
            deg = len(top_pools.get(asset_id, []))
            score = math.log1p(deg) * (1.0 + escape_score.get(asset_id, 0.0))
            hub_scores.append((score, asset_id))
        hub_scores.sort(reverse=True)
        hubs = {asset_id for _, asset_id in hub_scores[:hub_count]}

        p_min = float(self.cfg.noam_success_min or 0.0)
        p_max = float(self.cfg.noam_success_max or 1.0)
        new_edge_state: Dict[Tuple[str, str, str], NoamEdgeState] = {}
        for (pid, asset_in), outs in top_out.items():
            pool = self.pools.get(pid)
            if pool is None:
                continue
            reliability = float(self._swap_success_ema.get(pid, 1.0))
            if reliability <= 0.0:
                reliability = 1e-3
            for asset_out in outs:
                key = (pid, asset_in, asset_out)
                existing = self._noam_edge_state.get(key)
                if existing is None:
                    p_success = min(p_max, max(p_min, reliability))
                    existing = NoamEdgeState(p_success=p_success, lambda_price=0.0)
                else:
                    existing.p_success = min(p_max, max(p_min, existing.p_success))
                existing.lambda_price *= max(0.0, 1.0 - float(self.cfg.noam_lambda_decay or 0.0))
                new_edge_state[key] = existing

        self._noam_top_pools = top_pools
        self._noam_top_out = top_out
        self._noam_edge_state = new_edge_state
        self._noam_escape_score = escape_score
        self._noam_hubs = hubs
        self._noam_adj_forward = adj_forward
        self._noam_adj_reverse = adj_reverse
        self._noam_last_refresh_tick = self.tick

    def _refresh_noam_overlay_graph(self) -> None:
        hubs = self._noam_hubs
        adj = self._noam_adj_forward
        overlay_max = max(1, int(self.cfg.noam_overlay_max_hops or 1))
        candidate_limit = max(1, int(self.cfg.noam_hub_candidate_limit or 1))
        overlay_graph: Dict[str, list[Tuple[str, float]]] = {}
        overlay_paths: Dict[Tuple[str, str], Tuple[float, list[str]]] = {}
        if not hubs or not adj:
            self._noam_overlay_graph = overlay_graph
            self._noam_overlay_paths_cache = overlay_paths
            self._noam_last_overlay_tick = self.tick
            return
        for hub in hubs:
            distances: Dict[str, int] = {}
            visited = {hub}
            queue = deque([(hub, 0)])
            while queue:
                node, depth = queue.popleft()
                if depth >= overlay_max:
                    continue
                for nxt in adj.get(node, set()):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, depth + 1))
                    if nxt in hubs and nxt != hub:
                        prev = distances.get(nxt)
                        if prev is None or (depth + 1) < prev:
                            distances[nxt] = depth + 1
            if not distances:
                continue
            ranked = sorted(
                distances.items(),
                key=lambda item: (item[1], -self._noam_escape_score.get(item[0], 0.0)),
            )
            if candidate_limit > 0 and len(ranked) > candidate_limit:
                ranked = ranked[:candidate_limit]
            overlay_graph[hub] = [(target, float(dist)) for target, dist in ranked]
        self._noam_overlay_graph = overlay_graph
        for hub in hubs:
            dist, prev = self._noam_overlay_dijkstra(hub)
            for target, cost in dist.items():
                if target == hub:
                    continue
                path = self._noam_overlay_reconstruct(prev, hub, target)
                if path:
                    overlay_paths[(hub, target)] = (cost, path)
        self._noam_overlay_paths_cache = overlay_paths
        self._noam_last_overlay_tick = self.tick

    def _noam_near_hubs(self, asset_id: str, *, forward: bool) -> list[str]:
        if not self._noam_hubs:
            return []
        if self._noam_near_hubs_cache_tick != self.tick:
            self._noam_near_hubs_cache = {}
            self._noam_near_hubs_cache_tick = self.tick
        cache_key = (asset_id, forward)
        cached = self._noam_near_hubs_cache.get(cache_key)
        if cached is not None:
            return cached
        depth_max = max(0, int(self.cfg.noam_hub_depth or 0))
        candidate_limit = max(1, int(self.cfg.noam_hub_candidate_limit or 1))
        adj = self._noam_adj_forward if forward else self._noam_adj_reverse
        if not adj:
            return []
        found: Dict[str, int] = {}
        visited = {asset_id}
        queue = deque([(asset_id, 0)])
        while queue:
            node, depth = queue.popleft()
            if node in self._noam_hubs:
                prev = found.get(node)
                if prev is None or depth < prev:
                    found[node] = depth
            if depth >= depth_max:
                continue
            for nxt in adj.get(node, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))
        ranked = sorted(
            found.items(),
            key=lambda item: (item[1], -self._noam_escape_score.get(item[0], 0.0)),
        )
        if candidate_limit > 0 and len(ranked) > candidate_limit:
            ranked = ranked[:candidate_limit]
        result = [hub for hub, _ in ranked]
        self._noam_near_hubs_cache[cache_key] = result
        return result

    def _noam_overlay_paths(self, hubs_start: list[str], hubs_target: list[str]) -> list[list[str]]:
        if not self._noam_overlay_graph:
            return []
        if not hubs_start or not hubs_target:
            return []
        top_r = max(1, int(self.cfg.noam_overlay_top_r_paths or 1))
        candidates: list[Tuple[float, list[str]]] = []
        target_set = set(hubs_target)
        for start in hubs_start:
            if start in target_set:
                candidates.append((0.0, [start]))
            for target in hubs_target:
                if target == start:
                    continue
                cached = self._noam_overlay_paths_cache.get((start, target))
                if cached:
                    cost, path = cached
                    candidates.append((cost, path))
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0])
        seen: Set[Tuple[str, ...]] = set()
        result: list[list[str]] = []
        for _, path in candidates:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            result.append(path)
            if len(result) >= top_r:
                break
        return result

    def _noam_overlay_dijkstra(self, start: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        dist: Dict[str, float] = {start: 0.0}
        prev: Dict[str, str] = {}
        heap: list[Tuple[float, str]] = [(0.0, start)]
        while heap:
            cost, node = heapq.heappop(heap)
            if cost > dist.get(node, float("inf")):
                continue
            for nxt, weight in self._noam_overlay_graph.get(node, []):
                new_cost = cost + weight
                if new_cost < dist.get(nxt, float("inf")):
                    dist[nxt] = new_cost
                    prev[nxt] = node
                    heapq.heappush(heap, (new_cost, nxt))
        return dist, prev

    def _noam_overlay_reconstruct(self, prev: Dict[str, str], start: str, target: str) -> list[str]:
        if start == target:
            return [start]
        if target not in prev:
            return []
        path = [target]
        while path[-1] != start:
            parent = prev.get(path[-1])
            if parent is None:
                return []
            path.append(parent)
        path.reverse()
        return path

    def _noam_clearing_edges(self) -> Dict[str, list[NoamClearingEdge]]:
        self._maybe_refresh_noam_working_set()
        min_value = float(self.cfg.noam_clearing_min_cycle_value_usd or 0.0)
        edge_cap = max(0, int(self.cfg.noam_clearing_edge_cap_per_asset or 0))
        p_min = float(self.cfg.noam_success_min or 1e-6)
        p_max = float(self.cfg.noam_success_max or 1.0)
        w_p = float(self.cfg.noam_weight_success or 0.0)
        w_f = float(self.cfg.noam_weight_fee or 0.0)
        w_l = float(self.cfg.noam_weight_lambda or 0.0)
        w_b = float(self.cfg.noam_weight_benefit or 0.0)
        w_d = float(self.cfg.noam_weight_deadend or 0.0)

        edges_by_asset: Dict[str, list[NoamClearingEdge]] = {}
        nominal_value = max(1.0, min_value)

        for (pid, asset_in), outs in self._noam_top_out.items():
            pool = self.pools.get(pid)
            if pool is None:
                continue
            value_in = pool.values.get_value(asset_in)
            if value_in <= 0.0:
                continue
            amount_in_nominal = nominal_value / value_in
            for asset_out in outs:
                if asset_out == asset_in:
                    continue
                value_out = pool.values.get_value(asset_out)
                if value_out <= 0.0:
                    continue
                if not self._noam_edge_allowed(pool, asset_in, asset_out, amount_in_nominal):
                    continue
                remaining = float("inf")
                if pool.policy.limits_enabled:
                    remaining = pool.limiter.remaining(self.tick, asset_in)
                inventory_out = pool.vault.get(asset_out)
                if asset_out == self.cfg.stable_symbol:
                    inventory_out = max(0.0, inventory_out - pool.policy.min_stable_reserve)
                cap_value = min(remaining * value_in, inventory_out * value_out)
                if cap_value <= min_value + 1e-9:
                    continue
                state = self._noam_edge_state_for(pid, asset_in, asset_out)
                if state is None:
                    continue
                p = min(p_max, max(p_min, state.p_success))
                fee_rate = float(pool.fees.pool_fee_rate or 0.0)
                benefit = self._noam_edge_benefit(pool, asset_in, asset_out)
                deadend = self._noam_deadend_penalty(asset_out)
                score = (w_p * math.log(p)) - (w_f * fee_rate) - (w_l * state.lambda_price)
                score += (w_b * benefit) - (w_d * deadend)
                if pool.policy.role == "clc":
                    score += float(self.cfg.noam_clc_edge_bonus or 0.0)
                edge = NoamClearingEdge(
                    pool_id=pid,
                    asset_in=asset_in,
                    asset_out=asset_out,
                    score=score,
                    cap_value=cap_value,
                    value_in=value_in,
                    value_out=value_out,
                )
                edges_by_asset.setdefault(asset_in, []).append(edge)

        if edge_cap > 0:
            for asset_in, edges in list(edges_by_asset.items()):
                edges.sort(key=lambda e: e.score, reverse=True)
                edges_by_asset[asset_in] = edges[:edge_cap]

        return edges_by_asset

    def _noam_find_clearing_cycles(self, edges_by_asset: Dict[str, list[NoamClearingEdge]]) -> list[NoamClearingCycle]:
        max_hops = max(2, int(self.cfg.noam_clearing_max_hops or 2))
        max_cycles = max(1, int(self.cfg.noam_clearing_max_cycles or 1))
        cycles: list[NoamClearingCycle] = []
        max_search = max_cycles * 3

        for start_asset in list(edges_by_asset.keys()):
            if len(cycles) >= max_search:
                break

            def dfs(
                current_asset: str,
                depth: int,
                path: list[NoamClearingEdge],
                used_assets: Set[str],
                used_pools: Set[str],
            ) -> None:
                if len(cycles) >= max_search:
                    return
                if depth >= max_hops:
                    return
                for edge in edges_by_asset.get(current_asset, []):
                    if edge.pool_id in used_pools:
                        continue
                    next_asset = edge.asset_out
                    new_path = path + [edge]
                    if next_asset == start_asset and len(new_path) >= 2:
                        score = sum(e.score for e in new_path)
                        if score > 0.0:
                            cap_value = min(e.cap_value for e in new_path)
                            cycles.append(NoamClearingCycle(edges=new_path, score=score, cap_value=cap_value))
                        continue
                    if next_asset in used_assets:
                        continue
                    dfs(
                        next_asset,
                        depth + 1,
                        new_path,
                        used_assets | {next_asset},
                        used_pools | {edge.pool_id},
                    )

            dfs(start_asset, 0, [], {start_asset}, set())

        if not cycles:
            return []
        cycles.sort(key=lambda c: (c.score * c.cap_value), reverse=True)
        return cycles[:max_cycles]

    def _noam_execute_clearing_cycle(
        self,
        cycle: NoamClearingCycle,
        budget_remaining: Optional[float],
    ) -> Tuple[bool, Optional[float], float]:
        min_value = float(self.cfg.noam_clearing_min_cycle_value_usd or 0.0)
        cycle_value = cycle.cap_value
        if budget_remaining is not None:
            cycle_value = min(cycle_value, budget_remaining)
        if cycle_value < min_value:
            return False, budget_remaining, 0.0

        hop_amounts: list[Tuple[NoamClearingEdge, float, float]] = []
        amount_in = cycle_value / max(1e-9, cycle.edges[0].value_in)
        net_value = 0.0

        for edge in cycle.edges:
            pool = self.pools.get(edge.pool_id)
            if pool is None:
                return False, budget_remaining, 0.0
            okq, reasonq, amount_out, fee_amt = pool.quote_swap(edge.asset_in, amount_in, edge.asset_out)
            if not okq or amount_out <= 1e-12:
                return False, budget_remaining, 0.0
            gross_out = amount_out + fee_amt
            ok, _reason = pool.can_swap(self.tick, edge.asset_in, amount_in, edge.asset_out, gross_out)
            if not ok:
                return False, budget_remaining, 0.0
            net_value += (amount_out * edge.value_out) - (amount_in * edge.value_in)
            hop_amounts.append((edge, amount_in, amount_out))
            amount_in = amount_out

        cost = max(0.0, -net_value)
        if budget_remaining is not None and cost > budget_remaining + 1e-9:
            return False, budget_remaining, 0.0

        for edge, amount_in, _amount_out in hop_amounts:
            pool = self.pools.get(edge.pool_id)
            if pool is None:
                return False, budget_remaining, 0.0
            receipt = pool.execute_swap(
                self.tick,
                actor="clearing",
                asset_in=edge.asset_in,
                amount_in=amount_in,
                asset_out=edge.asset_out,
            )
            if receipt.status != "executed":
                self._noam_update_edge_after_swap(
                    pool,
                    receipt.asset_in,
                    receipt.asset_out,
                    float(receipt.amount_in),
                    success=False,
                    fail_reason=receipt.fail_reason,
                )
                return False, budget_remaining, 0.0

            gross_out = receipt.amount_out + float(receipt.fees.total_fee)
            self._update_pool_caches(pool, receipt.asset_in, float(receipt.amount_in))
            self._update_pool_caches(pool, receipt.asset_out, -float(gross_out))
            self._record_fee_cumulative(receipt)
            self._noam_update_edge_after_swap(
                pool,
                receipt.asset_in,
                receipt.asset_out,
                float(receipt.amount_in),
                success=True,
            )
            self._noam_clearing_swaps_tick += 1
            swap_usd = receipt.amount_in * self._asset_value(pool, receipt.asset_in)
            self._swap_volume_usd_tick += swap_usd
            self._swap_volume_usd_by_pool[pool.pool_id] = (
                self._swap_volume_usd_by_pool.get(pool.pool_id, 0.0) + swap_usd
            )

        if budget_remaining is not None:
            budget_remaining = max(0.0, budget_remaining - cost)
        return True, budget_remaining, cycle_value

    def _run_noam_clearing(self) -> None:
        if not self.cfg.noam_clearing_enabled or self.cfg.routing_mode != "noam":
            return
        stride = max(1, int(self.cfg.noam_clearing_stride_ticks or 1))
        if self.tick % stride != 0:
            return
        edges_by_asset = self._noam_clearing_edges()
        if not edges_by_asset:
            return
        cycles = self._noam_find_clearing_cycles(edges_by_asset)
        if not cycles:
            return
        budget = float(self.cfg.noam_clearing_budget_usd or 0.0)
        budget_share = float(self.cfg.noam_clearing_budget_share or 0.0)
        if budget_share > 0.0:
            budget = max(budget, budget_share * self._network_total_value())
        budget_remaining = budget if budget > 0.0 else None

        executed = 0
        attempted = 0
        total_value = 0.0
        for cycle in cycles:
            attempted += 1
            ok, budget_remaining, value_used = self._noam_execute_clearing_cycle(cycle, budget_remaining)
            if ok:
                executed += 1
                total_value += value_used
            if budget_remaining is not None and budget_remaining <= 1e-9:
                break

        if attempted > 0:
            self.log.add(Event(
                self.tick,
                "NOAM_CLEARING_RUN",
                amount=total_value,
                meta={
                    "cycles_attempted": attempted,
                    "cycles_executed": executed,
                    "budget_start": budget,
                    "budget_remaining": budget_remaining if budget_remaining is not None else None,
                },
            ))

    def _noam_edge_state_for(self, pool_id: str, asset_in: str, asset_out: str) -> Optional[NoamEdgeState]:
        return self._noam_edge_state.get((pool_id, asset_in, asset_out))

    def _noam_deadend_penalty(self, asset_id: str) -> float:
        deg = len(self._noam_top_pools.get(asset_id, []))
        return 1.0 / (deg + 1.0)

    def _noam_edge_benefit(
        self,
        pool: "Pool",
        asset_in: str,
        asset_out: str,
        cache: Optional[NoamRouteCache] = None,
    ) -> float:
        total_value = self._pool_total_value(pool)
        if total_value <= 1e-9:
            return 0.0
        listings = len(pool.vault.inventory)
        if listings <= 0:
            return 0.0
        avg_value = total_value / max(1, listings)
        vin = self._noam_cached_inventory(pool, asset_in, cache) * self._noam_cached_value(pool, asset_in, cache)
        vout = self._noam_cached_inventory(pool, asset_out, cache) * self._noam_cached_value(pool, asset_out, cache)
        z_in = (vin - avg_value) / (avg_value + 1e-9)
        z_out = (vout - avg_value) / (avg_value + 1e-9)
        return max(0.0, z_in - z_out)

    def _noam_edge_allowed(
        self,
        pool: "Pool",
        asset_in: str,
        asset_out: str,
        amount_in: float,
        cache: Optional[NoamRouteCache] = None,
    ) -> bool:
        if self._noam_failure_active(pool.pool_id, asset_in, asset_out):
            return False
        if pool.policy.paused:
            return False
        if not pool.registry.is_listed(asset_in) or not pool.registry.is_listed(asset_out):
            return False
        if self._noam_cached_inventory(pool, asset_out, cache) <= 1e-9:
            return False
        if pool.policy.role == "lender":
            if asset_in != self.cfg.stable_symbol and asset_out != self.cfg.stable_symbol:
                return False
        if pool.policy.role in ("consumer", "producer") and asset_out == self.cfg.stable_symbol:
            return False
        if pool.policy.role == "clc":
            allowed = {self.cfg.stable_symbol}
            if pool.policy.clc_liquidity_symbol:
                allowed.add(pool.policy.clc_liquidity_symbol)
            if asset_in not in allowed and asset_out not in allowed:
                return False
        if asset_out == self.cfg.stable_symbol and pool.policy.mode in ("borrow_only", "none"):
            return False
        if pool.policy.mode in ("none", "borrow_only"):
            return False
        if pool.policy.limits_enabled:
            remaining = self._noam_cached_remaining(pool, asset_in, cache)
            if remaining <= 1e-9 or amount_in > remaining + 1e-9:
                return False
        if asset_out == self.cfg.stable_symbol:
            value_in = self._noam_cached_value(pool, asset_in, cache)
            value_out = self._noam_cached_value(pool, asset_out, cache)
            if value_in > 0.0 and value_out > 0.0:
                amount_out_est = (amount_in * value_in / value_out)
                stable_available = self._noam_cached_inventory(pool, self.cfg.stable_symbol, cache)
                if stable_available - amount_out_est < pool.policy.min_stable_reserve - 1e-9:
                    return False
        return True

    def _noam_edge_score(
        self,
        pool: "Pool",
        asset_in: str,
        asset_out: str,
        amount_in: float,
        cache: Optional[NoamRouteCache] = None,
    ) -> float:
        state = self._noam_edge_state_for(pool.pool_id, asset_in, asset_out)
        if state is None:
            return -1e9

        w_p = float(self.cfg.noam_weight_success or 0.0)
        w_f = float(self.cfg.noam_weight_fee or 0.0)
        w_l = float(self.cfg.noam_weight_lambda or 0.0)
        w_b = float(self.cfg.noam_weight_benefit or 0.0)
        w_d = float(self.cfg.noam_weight_deadend or 0.0)

        p_min = float(self.cfg.noam_success_min or 1e-6)
        p_max = float(self.cfg.noam_success_max or 1.0)
        p = min(p_max, max(p_min, state.p_success))
        score = w_p * math.log(p)

        fee = amount_in * float(pool.fees.pool_fee_rate or 0.0)
        score -= w_f * fee

        if pool.policy.limits_enabled:
            remaining = self._noam_cached_remaining(pool, asset_in, cache)
        else:
            remaining = float("inf")
        inventory_out = self._noam_cached_inventory(pool, asset_out, cache)
        denom = max(1e-9, min(remaining, inventory_out))
        usage = amount_in / denom
        usage_cap = float(self.cfg.noam_usage_cap or 0.0)
        if usage_cap > 0.0:
            usage = min(usage, usage_cap)
        score -= w_l * state.lambda_price * usage

        benefit = self._noam_edge_benefit(pool, asset_in, asset_out, cache)
        score += w_b * benefit

        deadend = self._noam_deadend_penalty(asset_out)
        score -= w_d * deadend

        if pool.policy.role == "clc":
            score += float(self.cfg.noam_clc_edge_bonus or 0.0)

        return score

    def _noam_update_edge_after_swap(
        self,
        pool: "Pool",
        asset_in: str,
        asset_out: str,
        amount_in: float,
        success: bool,
        fail_reason: Optional[str] = None,
    ) -> None:
        if self.cfg.routing_mode != "noam":
            return
        state = self._noam_edge_state_for(pool.pool_id, asset_in, asset_out)
        if state is None:
            return
        alpha = float(self.cfg.noam_success_ema_alpha or 0.0)
        if alpha > 0.0:
            target = 1.0 if success else 0.0
            state.p_success = (1.0 - alpha) * state.p_success + alpha * target
            state.p_success = min(float(self.cfg.noam_success_max), max(float(self.cfg.noam_success_min), state.p_success))

        remaining = pool.limiter.remaining(self.tick, asset_in)
        inventory_out = pool.vault.get(asset_out)
        denom = max(1e-9, min(remaining, inventory_out))
        usage = amount_in / denom
        usage_cap = float(self.cfg.noam_usage_cap or 0.0)
        if usage_cap > 0.0:
            usage = min(usage, usage_cap)
        eta = float(self.cfg.noam_scarcity_eta or 0.0)
        safe_budget = float(self.cfg.noam_safe_budget_fraction or 0.0)
        if eta > 0.0:
            state.lambda_price = max(0.0, state.lambda_price + eta * (usage - safe_budget))
        if not success and fail_reason:
            state.last_fail_reason = fail_reason
        if not success:
            self._noam_record_failure(pool.pool_id, asset_in, asset_out)

    def _lp_sclc_remaining(self) -> Optional[float]:
        cap = float(self.cfg.lp_sclc_supply_cap or 0.0)
        if cap <= 0.0:
            return None
        remaining = cap - self._sclc_minted_total
        return max(0.0, remaining)

    def _pool_reserve_shortfall_ratio(self, pool: "Pool") -> float:
        reserve = float(pool.policy.min_stable_reserve or 0.0)
        if reserve <= 1e-9:
            return 0.0
        shortfall = max(0.0, reserve - pool.vault.get(self.cfg.stable_symbol))
        return shortfall / reserve

    def _pool_risk_weight(self, pool: "Pool") -> float:
        cfg = self.cfg
        base = float(cfg.insurance_risk_weight_base or 0.0)
        reserve_scale = float(cfg.insurance_risk_weight_reserve_scale or 0.0)
        weight = base + self._pool_reserve_shortfall_ratio(pool) * reserve_scale
        weight = max(float(cfg.insurance_risk_weight_min or 0.0), weight)
        return min(float(cfg.insurance_risk_weight_max or weight), weight)

    def _insurance_target_usd(self) -> float:
        cfg = self.cfg
        total = 0.0
        for p in self.pools.values():
            if p.policy.system_pool:
                continue
            voucher_value = self._pool_voucher_value_usd(p)
            if voucher_value <= 1e-9:
                continue
            total += voucher_value * self._pool_risk_weight(p)
        return total * float(cfg.insurance_target_multiplier or 0.0)

    def _sample_amount_in(self, pool: "Pool", asset_in: str) -> float:
        total_value = self._pool_total_value(pool)
        mean_usd = total_value * float(self.cfg.swap_size_mean_frac or 0.0)
        if mean_usd <= 0.0:
            return 0.0
        amount_usd = float(np.random.exponential(mean_usd))
        amount_usd = max(amount_usd, float(self.cfg.swap_size_min_usd or 0.0))
        if self.cfg.swap_size_max_usd is not None:
            amount_usd = min(amount_usd, float(self.cfg.swap_size_max_usd))
        value_in = self._asset_value(pool, asset_in)
        amount_in = amount_usd / value_in
        return min(amount_in, pool.vault.get(asset_in))

    def _swap_attempts_for_pool(self, pool: "Pool") -> int:
        attempts = int(self.cfg.random_route_requests_per_tick)
        scale = float(self.cfg.swap_attempts_value_scale_usd or 0.0)
        if scale > 0.0:
            pool_value = self._pool_total_value(pool)
            attempts += int(pool_value / scale)
        boost = float(self._utilization_boost or 1.0)
        if boost > 1.0 and attempts > 0:
            attempts = int(math.ceil(attempts * boost))
        max_attempts = int(self.cfg.swap_attempts_max_per_pool or 0)
        if max_attempts > 0:
            attempts = min(attempts, max_attempts)
        return max(0, attempts)

    def _record_swap_attempt(self, pool_id: str, success: bool) -> None:
        if not pool_id:
            return
        self._swap_attempt_counts[pool_id] = self._swap_attempt_counts.get(pool_id, 0) + 1
        alpha = float(self.cfg.offramp_success_ema_alpha or 0.0)
        if alpha <= 0.0:
            return
        prev = float(self._swap_success_ema.get(pool_id, 1.0))
        target = 1.0 if success else 0.0
        ema = (alpha * target) + ((1.0 - alpha) * prev)
        if ema < 0.0:
            ema = 0.0
        elif ema > 1.0:
            ema = 1.0
        self._swap_success_ema[pool_id] = ema

    def _record_fee_cumulative(self, receipt: SwapReceipt) -> None:
        if receipt.status != "executed":
            return
        pool_fee = float(receipt.fees.pool_fee)
        clc_fee = float(receipt.fees.clc_fee)
        if pool_fee <= 0.0 and clc_fee <= 0.0:
            return
        asset_out = receipt.asset_out
        if asset_out == self.cfg.stable_symbol:
            self._fee_pool_cumulative_usd += pool_fee
            self._fee_clc_cumulative_usd += clc_fee
        elif asset_out.startswith("VCHR:"):
            self._fee_pool_cumulative_voucher += pool_fee
            self._fee_clc_cumulative_voucher += clc_fee

    def _is_routable_pool(self, pool: "Pool") -> bool:
        if not pool.policy.system_pool:
            return True
        if pool.policy.role == "clc":
            return True
        return False

    def _refresh_liquidity_cache(self) -> None:
        if self._liquidity_tick == self.tick:
            return
        if self._liquidity_initialized:
            self._liquidity_tick = self.tick
            return
        weights: Dict[str, float] = {}
        for p in self.pools.values():
            if not self._is_routable_pool(p):
                continue
            for asset_id, amt in p.vault.inventory.items():
                if asset_id == self.cfg.sclc_symbol:
                    continue
                if amt <= 1e-9:
                    continue
                usd = amt * self._asset_value(p, asset_id)
                if usd <= 0.0:
                    continue
                weights[asset_id] = weights.get(asset_id, 0.0) + usd
        self._liquidity_by_asset = weights
        self._liquidity_tick = self.tick
        self._liquidity_initialized = True

    def _choose_target_asset(
        self,
        asset_in: str,
        source_pool: Optional["Pool"] = None,
        exclude: Optional[Set[str]] = None,
    ) -> Optional[str]:
        mode = self.cfg.swap_target_selection_mode
        restrict_stable = False
        if source_pool is not None and source_pool.policy.role in ("producer", "consumer"):
            restrict_stable = True
        if mode == "liquidity_weighted":
            self._refresh_liquidity_cache()
            candidates = [
                (a, w)
                for a, w in self._liquidity_by_asset.items()
                if a != asset_in
                and w > 0.0
                and (not restrict_stable or a != self.cfg.stable_symbol)
                and (exclude is None or a not in exclude)
            ]
            if not candidates:
                return None
            assets, weights = zip(*candidates)
            weights = np.array(weights, dtype=float)
            total = float(weights.sum())
            if total <= 0.0:
                return None
            probs = weights / total
            return str(np.random.choice(list(assets), p=probs))

        universe = [a for a in self.factory.asset_universe.keys() if a != self.cfg.sclc_symbol]
        if restrict_stable:
            universe = [a for a in universe if a != self.cfg.stable_symbol]
        if exclude:
            universe = [a for a in universe if a not in exclude]
        if len(universe) < 2:
            return None
        asset_out = random.choice(universe)
        if asset_out == asset_in and len(universe) > 1:
            asset_out = random.choice([a for a in universe if a != asset_in])
        return asset_out

    def _affinity_key(self, a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def _affinity_score(self, a: str, b: str) -> float:
        return float(self.pool_affinity.get(self._affinity_key(a, b), 0.0))

    def _decay_affinity(self) -> None:
        decay = float(self.cfg.sticky_affinity_decay or 0.0)
        if decay <= 0.0 or not self.pool_affinity:
            return
        keep = {}
        for key, score in self.pool_affinity.items():
            score *= (1.0 - decay)
            if score > 1e-6:
                keep[key] = score
        self.pool_affinity = keep

    def _update_affinity(self, a: str, b: str, amount_usd: float) -> None:
        gain = float(self.cfg.sticky_affinity_gain or 0.0)
        if gain <= 0.0:
            return
        inc = math.log1p(max(0.0, amount_usd)) * gain
        if inc <= 0.0:
            return
        key = self._affinity_key(a, b)
        cap = float(self.cfg.sticky_affinity_cap or 0.0)
        updated = self.pool_affinity.get(key, 0.0) + inc
        if cap > 0.0:
            updated = min(cap, updated)
        self.pool_affinity[key] = updated

    def _recent_pool_activity(self, window_ticks: int) -> Dict[str, float]:
        tick_min = max(1, self.tick - window_ticks + 1)
        activity: Dict[str, float] = {}
        for e in reversed(self.log.events):
            if e.tick < tick_min:
                break
            if e.event_type != "SWAP_EXECUTED":
                continue
            receipt = e.meta.get("receipt", {})
            pool_id = receipt.get("pool_id") or e.pool_id
            asset_in = receipt.get("asset_in")
            amount_in = receipt.get("amount_in")
            if not pool_id or not asset_in or amount_in is None:
                continue
            pool = self.pools.get(pool_id)
            if pool is None:
                continue
            if pool.policy.system_pool:
                continue
            usd = float(amount_in) * self._asset_value(pool, asset_in)
            if usd <= 0.0:
                continue
            activity[pool_id] = activity.get(pool_id, 0.0) + usd
        return activity

    def _recent_pool_clc_fees(self, window_ticks: int) -> Dict[str, float]:
        tick_min = max(1, self.tick - window_ticks + 1)
        fees: Dict[str, float] = {}
        for e in reversed(self.log.events):
            if e.tick < tick_min:
                break
            if e.event_type != "SWAP_EXECUTED":
                continue
            receipt = e.meta.get("receipt", {})
            pool_id = receipt.get("pool_id") or e.pool_id
            asset_out = receipt.get("asset_out")
            fees_meta = receipt.get("fees") or {}
            clc_fee = fees_meta.get("clc_fee")
            if not pool_id or clc_fee is None:
                continue
            pool = self.pools.get(pool_id)
            if pool is None or pool.policy.system_pool:
                continue
            value = self._asset_value(pool, asset_out) if asset_out else 0.0
            usd = float(clc_fee) * value
            if usd <= 0.0:
                continue
            fees[pool_id] = fees.get(pool_id, 0.0) + usd
        return fees

    def _loan_phase_for(self, agent_id: str, period: int) -> int:
        if period <= 1:
            return 0
        phase = self._loan_phase_by_agent.get(agent_id)
        if phase is None:
            phase = self.rng.randrange(period)
            self._loan_phase_by_agent[agent_id] = phase
        else:
            phase = phase % period
            self._loan_phase_by_agent[agent_id] = phase
        return phase

    def _choose_voucher_for_pool(self, pool: "Pool") -> Optional[str]:
        candidates = []
        weights = []
        for asset_id, pol in pool.registry.listings.items():
            if not pol.enabled or not asset_id.startswith("VCHR:"):
                continue
            value = pool.values.get_value(asset_id)
            if value <= 0.0:
                continue
            inv_value = pool.vault.get(asset_id) * value
            weight = inv_value if inv_value > 0.0 else value
            candidates.append(asset_id)
            weights.append(weight)
        if not candidates:
            return None
        total = float(np.sum(weights))
        if total <= 0.0:
            return random.choice(candidates)
        probs = np.array(weights, dtype=float) / total
        return str(np.random.choice(candidates, p=probs))

    def _mint_vouchers_for_pool(self, pool: "Pool", usd_value: float) -> float:
        if usd_value <= 1e-9:
            return 0.0
        voucher_id = self._choose_voucher_for_pool(pool)
        if voucher_id is None:
            return 0.0
        value = pool.values.get_value(voucher_id)
        if value <= 0.0:
            return 0.0
        amount = usd_value / value
        if amount <= 1e-9:
            return 0.0
        self._vault_add(pool, voucher_id, amount, "voucher_inflow", "system")
        spec = self.factory.voucher_specs.get(voucher_id)
        if spec:
            issuer = self.agents.get(spec.issuer_id)
            if issuer:
                issuer.issuer.issue(amount)
        return usd_value

    def _min_route_amount_in(self, pool: "Pool", asset_in: str, amount_in: float) -> Optional[float]:
        min_usd = float(self.cfg.swap_size_min_usd or 0.0)
        if min_usd <= 0.0:
            min_usd = 1.0
        value_in = self._asset_value(pool, asset_in)
        min_amount = min_usd / max(1e-9, value_in)
        min_amount = min(min_amount, pool.vault.get(asset_in))
        if min_amount <= 1e-9 or min_amount >= amount_in - 1e-9:
            return None
        return min_amount

    def _find_route_noam(
        self,
        *,
        tick: int,
        start_asset: str,
        target_asset: str,
        amount_in: float,
        source_pool: "Pool",
        target_pools: Optional[Set[str]] = None,
    ) -> RoutePlan:
        if start_asset == target_asset:
            return RoutePlan(ok=True, reason="trivial", hops=[], expected_amount_out=amount_in)

        self._maybe_refresh_noam_working_set()
        max_hops = max(1, int(self.cfg.noam_max_hops or self.cfg.max_hops))
        base_beam = max(1, int(self.cfg.noam_beam_width or 1))
        min_beam = max(1, int(self.cfg.noam_dynamic_min_beam or 1))
        beam_width = max(1, self._noam_scaled_cap(base_beam, min_beam))
        base_edge_cap = int(self.cfg.noam_edge_cap_per_state or 0)
        min_edge_cap = max(1, int(self.cfg.noam_dynamic_min_edge_cap or 1))
        edge_cap = self._noam_scaled_cap(base_edge_cap, min_edge_cap) if base_edge_cap > 0 else 0
        route_cache = NoamRouteCache(remaining={}, inventory={}, value={})
        dist_to_target = self._noam_distance_to_target(target_asset, max_hops)
        if start_asset not in dist_to_target:
            return RoutePlan(ok=False, reason="no_path_found", hops=[])

        best_path: Optional[list] = None
        best_score = -1e18
        states = [NoamState(asset_id=start_asset, score=0.0, path=[])]

        for depth in range(max_hops):
            next_states: list[NoamState] = []
            best_by_asset: Dict[str, float] = {}
            for state in states:
                asset_in = state.asset_id
                remaining_hops = max_hops - depth
                dist_here = dist_to_target.get(asset_in)
                if dist_here is None or dist_here > remaining_hops:
                    continue
                pool_ids = self._noam_top_pools.get(asset_in, [])
                if not pool_ids:
                    continue
                edges_scanned = 0
                edge_cap_reached = False
                for pid in pool_ids:
                    if pid == source_pool.pool_id:
                        continue
                    pool = self.pools.get(pid)
                    if pool is None:
                        continue
                    outs = self._noam_top_out.get((pid, asset_in), [])
                    if not outs:
                        continue
                    for asset_out in outs:
                        if edge_cap > 0 and edges_scanned >= edge_cap:
                            edge_cap_reached = True
                            break
                        if asset_out == asset_in:
                            continue
                        dist_out = dist_to_target.get(asset_out)
                        if dist_out is None or dist_out > (remaining_hops - 1):
                            continue
                        if not self._noam_edge_allowed(pool, asset_in, asset_out, amount_in, route_cache):
                            continue
                        edges_scanned += 1
                        edge_score = self._noam_edge_score(pool, asset_in, asset_out, amount_in, route_cache)
                        if edge_score <= -1e8:
                            continue
                        hop = Hop(pool_id=pid, asset_in=asset_in, asset_out=asset_out, amount_in=amount_in)
                        new_score = state.score + edge_score
                        new_path = state.path + [hop]

                        if asset_out == target_asset and (target_pools is None or pid in target_pools):
                            if new_score > best_score:
                                best_score = new_score
                                best_path = new_path

                        if depth + 1 < max_hops:
                            prev_best = best_by_asset.get(asset_out, -1e18)
                            if new_score > prev_best:
                                best_by_asset[asset_out] = new_score
                                next_states.append(NoamState(asset_id=asset_out, score=new_score, path=new_path))
                    if edge_cap_reached:
                        break

            if not next_states:
                break
            next_states.sort(key=lambda s: s.score, reverse=True)
            states = next_states[:beam_width]

        if best_path:
            return RoutePlan(ok=True, reason="ok", hops=best_path, expected_amount_out=amount_in)
        return RoutePlan(ok=False, reason="no_path_found", hops=[])

    def _find_route_noam_overlay(
        self,
        *,
        tick: int,
        start_asset: str,
        target_asset: str,
        amount_in: float,
        source_pool: "Pool",
        target_pools: Optional[Set[str]] = None,
    ) -> RoutePlan:
        if start_asset == target_asset:
            return RoutePlan(ok=True, reason="trivial", hops=[], expected_amount_out=amount_in)

        self._maybe_refresh_noam_working_set()
        pool_count = sum(1 for p in self.pools.values() if not p.policy.system_pool)
        min_pools = int(self.cfg.noam_overlay_min_pools or 0)
        if (
            not self.cfg.noam_overlay_enabled
            or not self._noam_overlay_graph
            or (min_pools > 0 and pool_count < min_pools)
        ):
            return self._find_route_noam(
                tick=tick,
                start_asset=start_asset,
                target_asset=target_asset,
                amount_in=amount_in,
                source_pool=source_pool,
                target_pools=target_pools,
            )

        hubs_start = self._noam_near_hubs(start_asset, forward=True)
        hubs_target = self._noam_near_hubs(target_asset, forward=False)
        if not hubs_start or not hubs_target:
            return self._find_route_noam(
                tick=tick,
                start_asset=start_asset,
                target_asset=target_asset,
                amount_in=amount_in,
                source_pool=source_pool,
                target_pools=target_pools,
            )

        overlay_paths = self._noam_overlay_paths(hubs_start, hubs_target)
        if not overlay_paths:
            return self._find_route_noam(
                tick=tick,
                start_asset=start_asset,
                target_asset=target_asset,
                amount_in=amount_in,
                source_pool=source_pool,
                target_pools=target_pools,
            )

        for hub_path in overlay_paths:
            current_asset = start_asset
            full_hops: list[Hop] = []
            ok = True
            for hub_asset in hub_path:
                if current_asset == hub_asset:
                    current_asset = hub_asset
                    continue
                segment_target_pools = target_pools if hub_asset == target_asset else None
                segment = self._find_route_noam(
                    tick=tick,
                    start_asset=current_asset,
                    target_asset=hub_asset,
                    amount_in=amount_in,
                    source_pool=source_pool,
                    target_pools=segment_target_pools,
                )
                if not segment.ok:
                    ok = False
                    break
                full_hops.extend(segment.hops)
                current_asset = hub_asset
            if not ok:
                continue
            if current_asset != target_asset:
                segment = self._find_route_noam(
                    tick=tick,
                    start_asset=current_asset,
                    target_asset=target_asset,
                    amount_in=amount_in,
                    source_pool=source_pool,
                    target_pools=target_pools,
                )
                if not segment.ok:
                    continue
                full_hops.extend(segment.hops)
            if full_hops:
                return RoutePlan(ok=True, reason="ok", hops=full_hops, expected_amount_out=amount_in)

        return self._find_route_noam(
            tick=tick,
            start_asset=start_asset,
            target_asset=target_asset,
            amount_in=amount_in,
            source_pool=source_pool,
            target_pools=target_pools,
        )

    def _find_route_with_fallback(
        self,
        *,
        tick: int,
        start_asset: str,
        target_asset: str,
        amount_in: float,
        source_pool: "Pool",
        target_pools: Optional[Set[str]] = None,
    ) -> Tuple[RoutePlan, float, bool]:
        cache_key: Optional[Tuple[str, str, str, int]] = None
        if self.cfg.routing_mode == "noam":
            cache_key = self._noam_route_cache_key(
                source_pool,
                start_asset,
                target_asset,
                amount_in,
                target_pools,
            )
            if cache_key is not None:
                cached = self._noam_route_cache_get(cache_key)
                if cached is not None and cached.ok and cached.hops:
                    blocked = any(
                        self._noam_failure_active(hop.pool_id, hop.asset_in, hop.asset_out)
                        for hop in cached.hops
                    )
                    if not blocked:
                        return cached, amount_in, False
                    self._noam_route_cache.pop(cache_key, None)
        if self.cfg.routing_mode == "noam":
            if self.cfg.noam_overlay_enabled:
                plan = self._find_route_noam_overlay(
                    tick=tick,
                    start_asset=start_asset,
                    target_asset=target_asset,
                    amount_in=amount_in,
                    source_pool=source_pool,
                    target_pools=target_pools,
                )
            else:
                plan = self._find_route_noam(
                    tick=tick,
                    start_asset=start_asset,
                    target_asset=target_asset,
                    amount_in=amount_in,
                    source_pool=source_pool,
                    target_pools=target_pools,
                )
        else:
            plan = self.router.find_route(
                tick=tick,
                start_asset=start_asset,
                target_asset=target_asset,
                amount_in=amount_in,
                pools=self.pools,
                accept_index=self.accept_index,
                source_pool_id=source_pool.pool_id,
                target_pools=target_pools,
                pool_affinity=self.pool_affinity,
                affinity_bias=float(self.cfg.sticky_route_bias or 0.0),
                max_candidate_pools=self.cfg.max_candidate_pools_per_hop,
            )
        if plan.ok:
            return plan, amount_in, False

        fallback_amount = self._min_route_amount_in(source_pool, start_asset, amount_in)
        if fallback_amount is None:
            return plan, amount_in, False

        if self.cfg.routing_mode == "noam":
            if self.cfg.noam_overlay_enabled:
                retry_plan = self._find_route_noam_overlay(
                    tick=tick,
                    start_asset=start_asset,
                    target_asset=target_asset,
                    amount_in=fallback_amount,
                    source_pool=source_pool,
                    target_pools=target_pools,
                )
            else:
                retry_plan = self._find_route_noam(
                    tick=tick,
                    start_asset=start_asset,
                    target_asset=target_asset,
                    amount_in=fallback_amount,
                    source_pool=source_pool,
                    target_pools=target_pools,
                )
        else:
            retry_plan = self.router.find_route(
                tick=tick,
                start_asset=start_asset,
                target_asset=target_asset,
                amount_in=fallback_amount,
                pools=self.pools,
                accept_index=self.accept_index,
                source_pool_id=source_pool.pool_id,
                target_pools=target_pools,
                pool_affinity=self.pool_affinity,
                affinity_bias=float(self.cfg.sticky_route_bias or 0.0),
                max_candidate_pools=self.cfg.max_candidate_pools_per_hop,
            )
        if retry_plan.ok:
            return retry_plan, fallback_amount, True
        return plan, amount_in, False

    def _network_total_value(self) -> float:
        if not self.pools:
            return 0.0
        if self.cfg.swap_target_selection_mode == "liquidity_weighted":
            self._refresh_liquidity_cache()
            return float(sum(self._liquidity_by_asset.values()))
        return sum(self._pool_total_value(p) for p in self.pools.values() if self._is_routable_pool(p))

    def _update_utilization_boost(self) -> None:
        target = float(self.cfg.utilization_target_rate or 0.0)
        if target <= 0.0:
            self._utilization_boost = 1.0
            self._last_utilization_rate = 0.0
            return
        total_value = self._network_total_value()
        if total_value <= 1e-9:
            self._utilization_boost = 1.0
            self._last_utilization_rate = 0.0
            return
        utilization = self._swap_volume_usd_tick / total_value
        self._last_utilization_rate = utilization
        max_mult = max(1.0, float(self.cfg.utilization_boost_max or 1.0))
        if utilization >= target:
            self._utilization_boost = 1.0
            return
        ratio = (target - utilization) / max(target, 1e-9)
        ratio = min(1.0, max(0.0, ratio))
        self._utilization_boost = 1.0 + ratio * (max_mult - 1.0)

    def rebuild_indexes(self) -> None:
        self.accept_index = {}
        self.offer_index = {}
        for pid, p in self.pools.items():
            if not self._is_routable_pool(p):
                continue
            # accept: listed assets
            for asset_id, pol in p.registry.listings.items():
                if pol.enabled:
                    self.accept_index.setdefault(asset_id, set()).add(pid)
            # offer: inventory > 0
            for asset_id, amt in p.vault.inventory.items():
                if amt > 1e-9:
                    self.offer_index.setdefault(asset_id, set()).add(pid)
        self.log.add(Event(self.tick, "INDEX_UPDATED"))

    def add_pool(self, *, snapshot: bool = True, rebuild_indexes: bool = True,
                 role: Optional[str] = None) -> None:
        cfg = self.cfg
        max_pools = cfg.max_pools
        if max_pools is not None and max_pools > 0:
            active = sum(1 for p in self.pools.values() if not p.policy.system_pool)
            if active >= max_pools:
                self.log.add(Event(self.tick, "POOL_CAP_REACHED", amount=max_pools))
                return
        remaining_sclc = self._lp_sclc_remaining()
        allow_lp = True
        if remaining_sclc is not None and remaining_sclc <= 1e-9:
            allow_lp = False
        if role == "liquidity_provider" and not allow_lp:
            self.log.add(Event(self.tick, "LP_SUPPLY_CAP_REACHED", amount=float(self.cfg.lp_sclc_supply_cap)))
            return
        agent, pool = self.factory.create_agent_and_pool(
            role=role,
            allow_liquidity_provider=allow_lp,
        )
        role = pool.policy.role

        self.agents[agent.agent_id] = agent
        self.pools[pool.pool_id] = pool

        self.log.add(Event(self.tick, "POOL_CREATED", actor_id=agent.agent_id, pool_id=pool.pool_id))

        def list_voucher(asset_id: str, cap_in: float) -> None:
            v = float(max(0.05, np.random.lognormal(mean=0.0, sigma=0.35)))
            pool.list_asset_with_value_and_limit(asset_id, value=v, window_len=cfg.default_window_len, cap_in=cap_in)

        stable_cap_in = cfg.default_cap_in
        if role == "lender":
            stable_cap_in = cfg.lender_stable_cap_in
        elif role == "producer":
            stable_cap_in = cfg.producer_stable_cap_in

        # Always list stable in every pool (universal accept)
        pool.list_asset_with_value_and_limit(cfg.stable_symbol, value=1.0, window_len=cfg.default_window_len, cap_in=stable_cap_in)

        wanted: list[str] = []
        offered: list[str] = []
        stable_seed = 0.0

        if role == "lender":
            want_k = max(1, int(np.random.poisson(cfg.add_pool_want_assets_mean)))
            wanted = [a for a in self.factory.sample_assets(want_k, p_overlap=cfg.p_want_overlap) if a != cfg.stable_symbol]
            for a in wanted:
                list_voucher(a, cap_in=cfg.lender_voucher_cap_in)

            offer_k = max(1, int(np.random.poisson(cfg.add_pool_offer_assets_mean)))
            offered = [a for a in self.factory.sample_assets(offer_k, p_overlap=cfg.p_offer_overlap) if a != cfg.stable_symbol]
            for a in offered:
                if a not in wanted:
                    list_voucher(a, cap_in=cfg.lender_voucher_cap_in)

            stable_seed = float(max(0.0, cfg.lender_initial_stable_mean))
            self._vault_add(pool, cfg.stable_symbol, stable_seed, "seed_stable", "system")

            for a in offered:
                amt = float(max(0.0, np.random.exponential(250.0)))
                if amt <= 0:
                    continue
                self._vault_add(pool, a, amt, "seed_asset", "system")
                spec = self.factory.voucher_specs.get(a)
                if spec:
                    self.agents[spec.issuer_id].issuer.issue(amt)

        elif role == "producer":
            own_v = agent.voucher_spec.voucher_id
            list_voucher(own_v, cap_in=cfg.producer_voucher_cap_in)

            want_k = max(1, int(np.random.poisson(cfg.add_pool_want_assets_mean)))
            wanted = [
                a for a in self.factory.sample_assets(want_k, p_overlap=cfg.p_want_overlap)
                if a not in (cfg.stable_symbol, own_v)
            ]
            for a in wanted:
                list_voucher(a, cap_in=cfg.producer_voucher_cap_in)

            offer_k = max(1, int(np.random.poisson(cfg.add_pool_offer_assets_mean)))
            offered = [
                a for a in self.factory.sample_assets(offer_k, p_overlap=cfg.p_offer_overlap)
                if a not in (cfg.stable_symbol, own_v)
            ]
            for a in offered:
                if a not in wanted:
                    list_voucher(a, cap_in=cfg.producer_voucher_cap_in)

            stable_seed = 0.0
            if stable_seed > 0.0:
                self._vault_add(pool, cfg.stable_symbol, stable_seed, "seed_stable", "system")

            own_amt = float(max(0.0, np.random.exponential(10000.0)))
            if own_amt > 0:
                self._vault_add(pool, own_v, own_amt, "seed_voucher", agent.agent_id)
                agent.issuer.issue(own_amt)

            for a in offered:
                amt = float(max(0.0, np.random.exponential(10000.0)))
                if amt <= 0:
                    continue
                self._vault_add(pool, a, amt, "seed_asset", "system")
                spec = self.factory.voucher_specs.get(a)
                if spec:
                    self.agents[spec.issuer_id].issuer.issue(amt)

        elif role == "liquidity_provider":
            stable_seed = float(max(0.0, cfg.lp_initial_stable_mean))
            if stable_seed > 0.0:
                self._vault_add(pool, cfg.stable_symbol, stable_seed, "seed_stable", "system")

        else:  # consumer
            own_v = agent.voucher_spec.voucher_id
            list_voucher(own_v, cap_in=cfg.default_cap_in)

            want_k = max(1, int(np.random.poisson(cfg.add_pool_want_assets_mean)))
            wanted = [a for a in self.factory.sample_assets(want_k, p_overlap=cfg.p_want_overlap) if a != cfg.stable_symbol]
            for a in wanted:
                list_voucher(a, cap_in=cfg.default_cap_in)

            offer_k = max(1, int(np.random.poisson(cfg.add_pool_offer_assets_mean)))
            offered = [a for a in self.factory.sample_assets(offer_k, p_overlap=cfg.p_offer_overlap) if a != cfg.stable_symbol]
            for a in offered:
                if a not in wanted:
                    list_voucher(a, cap_in=cfg.default_cap_in)

            stable_seed = float(max(0.0, np.random.exponential(cfg.initial_stable_per_pool_mean * 0.25)))
            self._vault_add(pool, cfg.stable_symbol, stable_seed, "seed_stable", "system")

            own_amt = float(max(0.0, np.random.exponential(200.0)))
            if own_amt > 0:
                self._vault_add(pool, own_v, own_amt, "seed_voucher", agent.agent_id)
                agent.issuer.issue(own_amt)

            for a in offered:
                amt = float(max(0.0, np.random.exponential(150.0)))
                if amt <= 0:
                    continue
                self._vault_add(pool, a, amt, "seed_asset", "system")
                spec = self.factory.voucher_specs.get(a)
                if spec:
                    self.agents[spec.issuer_id].issuer.issue(amt)

        self.log.add(Event(self.tick, "POOL_CONFIGURED", actor_id=agent.agent_id, pool_id=pool.pool_id,
                           meta={"mode": pool.policy.mode, "role": pool.policy.role, "min_stable_reserve": pool.policy.min_stable_reserve}))
        self.log.add(Event(self.tick, "POOL_SEEDED", pool_id=pool.pool_id,
                           meta={"stable_seed": stable_seed, "offered": offered, "wanted": wanted}))

        if stable_seed > 0.0:
            self.initial_stable_total += stable_seed

        if rebuild_indexes:
            self.rebuild_indexes()
        if snapshot:
            self.snapshot_metrics()

    def _apply_pool_growth(self, multiplier: int = 1) -> None:
        rate = float(self.cfg.pool_growth_rate_per_tick or 0.0)
        if rate <= 0.0:
            return
        scale = max(1, int(multiplier or 1))
        rate *= scale
        active = [p for p in self.pools.values() if not p.policy.system_pool]
        base_count = len(active)
        if base_count <= 0:
            return
        max_pools = self.cfg.max_pools
        if max_pools is not None and max_pools > 0:
            available = max_pools - base_count
            if available <= 0:
                self._pool_growth_remainder = 0.0
                return
        desired = (base_count * rate) + self._pool_growth_remainder
        add_count = int(desired)
        self._pool_growth_remainder = desired - add_count
        if max_pools is not None and max_pools > 0:
            if add_count > available:
                add_count = available
                self._pool_growth_remainder = 0.0
        if add_count <= 0:
            return
        for _ in range(add_count):
            self.add_pool(snapshot=False, rebuild_indexes=False)
        self.rebuild_indexes()
        self.log.add(Event(self.tick, "POOL_GROWTH_APPLIED", amount=add_count,
                           meta={"base": base_count, "rate": rate}))

    def _grow_pool_desired_assets(self) -> None:
        max_per_pool = int(self.cfg.desired_assets_max_per_pool or 0)
        add_per_tick = int(self.cfg.desired_assets_add_per_tick or 0)
        if max_per_pool <= 0 or add_per_tick <= 0:
            return
        min_per_pool = max(0, int(self.cfg.desired_assets_min_per_pool or 0))
        growth_per_asset = max(0.0, float(self.cfg.desired_assets_growth_per_asset or 0.0))
        stable_id = self.cfg.stable_symbol
        sclc_id = self.cfg.sclc_symbol
        asset_universe = [
            a for a in self.factory.asset_universe.keys()
            if a not in (stable_id, sclc_id)
        ]
        if not asset_universe:
            return
        total_assets = len(asset_universe)
        target = int(total_assets * growth_per_asset)
        if target < min_per_pool:
            target = min_per_pool
        if target > max_per_pool:
            target = max_per_pool
        if target > total_assets:
            target = total_assets
        if target <= 0:
            return

        total_added = 0
        pools_updated = 0
        for p in self.pools.values():
            if p.policy.system_pool or p.policy.role == "liquidity_provider":
                continue
            current = sum(
                1
                for asset_id, pol in p.registry.listings.items()
                if pol.enabled and asset_id not in (stable_id, sclc_id)
            )
            if current >= target:
                continue
            need = min(add_per_tick, target - current)
            if need <= 0:
                continue
            candidates = [a for a in asset_universe if not p.registry.is_listed(a)]
            if not candidates:
                continue
            if len(candidates) > need:
                chosen = self.rng.sample(candidates, k=need)
            else:
                chosen = candidates

            cap_in = float(self.cfg.default_cap_in)
            if p.policy.role == "lender":
                cap_in = float(self.cfg.lender_voucher_cap_in)
            elif p.policy.role == "producer":
                cap_in = float(self.cfg.producer_voucher_cap_in)

            for asset_id in chosen:
                value = float(max(0.05, np.random.lognormal(mean=0.0, sigma=0.35)))
                p.list_asset_with_value_and_limit(
                    asset_id,
                    value=value,
                    window_len=self.cfg.default_window_len,
                    cap_in=cap_in,
                )
                if self._is_routable_pool(p):
                    self.accept_index.setdefault(asset_id, set()).add(p.pool_id)
            total_added += len(chosen)
            pools_updated += 1
        if total_added > 0:
            self.log.add(Event(
                self.tick,
                "DESIRED_ASSETS_GROWN",
                amount=total_added,
                meta={"pools_updated": pools_updated, "target_per_pool": target},
            ))

    def _apply_stable_growth_per_pool(self, multiplier: int = 1) -> None:
        voucher_share = max(0.0, float(self.cfg.voucher_inflow_share or 0.0))
        scale = max(1, int(multiplier or 1))
        minted_usd_total = 0.0
        minted_pools = 0
        onramp_total = 0.0
        offramp_total = 0.0
        for p in self.pools.values():
            if p.policy.system_pool:
                continue
            inflow = 0.0
            base = p.vault.get(self.cfg.stable_symbol)
            if p.policy.role == "producer":
                inflow = base * float(self.cfg.producer_inflow_per_tick or 0.0)
            elif p.policy.role == "consumer":
                inflow = base * float(self.cfg.consumer_inflow_per_tick or 0.0)
            elif p.policy.role == "lender":
                inflow = base * float(self.cfg.lender_inflow_per_tick or 0.0)
            elif p.policy.role == "liquidity_provider":
                inflow = base * float(self.cfg.liquidity_provider_inflow_per_tick or 0.0)
            else:
                inflow = base * float(self.cfg.stable_inflow_per_tick or 0.0)
            inflow = (inflow / float(MONTH_TICKS)) * scale
            if inflow != 0.0:
                self._vault_add(p, self.cfg.stable_symbol, inflow, "stable_inflow", "system")
                if inflow > 0.0:
                    onramp_total += inflow
                else:
                    offramp_total += -inflow
                if inflow > 0.0 and voucher_share > 0.0:
                    minted = self._mint_vouchers_for_pool(p, inflow * voucher_share)
                    if minted > 0.0:
                        minted_usd_total += minted
                        minted_pools += 1
        if minted_usd_total > 1e-9:
            self.log.add(Event(
                self.tick,
                "VOUCHER_MINTED",
                amount=minted_usd_total,
                meta={"pools": minted_pools, "share": voucher_share},
            ))
        if onramp_total > 0.0:
            self._stable_onramp_usd_tick += onramp_total
        if offramp_total > 0.0:
            self._stable_offramp_usd_tick += offramp_total

    def _apply_liquidity_provider_contributions(self) -> None:
        if not self.cfg.economics_enabled:
            return
        rate = float(self.cfg.lp_waterfall_contribution_rate or 0.0)
        if rate <= 0.0 or not self.cfg.sclc_symbol:
            return
        remaining_sclc = self._lp_sclc_remaining()
        if remaining_sclc is not None and remaining_sclc <= 1e-9:
            return
        stable_id = self.cfg.stable_symbol
        sclc_id = self.cfg.sclc_symbol
        total = 0.0
        pool_count = 0
        for p in self.pools.values():
            if p.policy.system_pool or p.policy.role != "liquidity_provider":
                continue
            stable = p.vault.get(stable_id)
            if stable <= 1e-9:
                continue
            contrib = min(stable, stable * rate)
            if remaining_sclc is not None:
                if remaining_sclc <= 1e-9:
                    break
                if contrib > remaining_sclc:
                    contrib = remaining_sclc
            if contrib <= 1e-9:
                continue
            if not self._vault_sub(p, stable_id, contrib, "lp_waterfall_out", "waterfall"):
                continue
            self._waterfall_external_inflows[stable_id] = (
                self._waterfall_external_inflows.get(stable_id, 0.0) + contrib
            )
            if p.values.get_value(sclc_id) <= 0.0:
                p.values.set_value(sclc_id, 1.0)
            self._vault_add(p, sclc_id, contrib, "sclc_mint", "waterfall")
            self._sclc_minted_total += contrib
            if remaining_sclc is not None:
                remaining_sclc = max(0.0, remaining_sclc - contrib)
            total += contrib
            pool_count += 1
        if total > 0.0:
            self.log.add(Event(
                self.tick,
                "LP_WATERFALL_CONTRIBUTED",
                amount=total,
                meta={"pools": pool_count, "rate": rate},
            ))

    def _stable_supply_target(self) -> float:
        cfg = self.cfg
        baseline = max(0.0, self.initial_stable_total)
        cap = max(baseline, float(cfg.stable_supply_cap or 0.0))
        growth_rate = float(cfg.stable_supply_growth_rate or 0.0)
        months = self.tick / float(MONTH_TICKS)
        if growth_rate <= 0.0 or cap <= baseline + 1e-9:
            target = baseline
        else:
            target = baseline + (cap - baseline) * (1.0 - math.exp(-growth_rate * months))
        noise = float(cfg.stable_supply_noise or 0.0)
        if noise > 0.0:
            per_tick_noise = noise / math.sqrt(MONTH_TICKS)
            target = max(0.0, target * (1.0 + np.random.normal(0.0, per_tick_noise)))
        return target

    def _apply_stable_inflow(self, amount: float) -> None:
        if amount <= 1e-9 or not self.pools:
            return
        self._stable_onramp_usd_tick += amount
        stable_id = self.cfg.stable_symbol
        voucher_share = max(0.0, float(self.cfg.voucher_inflow_share or 0.0))
        minted_usd_total = 0.0
        minted_pools = 0
        weights: Dict[str, float] = {}
        total_weight = 0.0
        deficit_weights: Dict[str, float] = {}
        total_deficit = 0.0
        for pid, p in self.pools.items():
            if p.policy.system_pool or p.policy.role == "liquidity_provider":
                continue
            deficit = max(0.0, p.policy.min_stable_reserve - p.vault.get(stable_id))
            if deficit > 1e-9:
                deficit_weights[pid] = deficit
                total_deficit += deficit

        activity_share = float(self.cfg.stable_inflow_activity_share or 0.0)
        activity_share = min(1.0, max(0.0, activity_share))
        activity_weights: Dict[str, float] = {}
        total_activity = 0.0
        if activity_share > 0.0:
            window = max(1, int(self.cfg.stable_inflow_activity_window_ticks or 1))
            activity_weights = self._recent_pool_activity(window)
            total_activity = sum(activity_weights.values())

        if total_deficit > 1e-9 and total_activity > 1e-9 and activity_share > 0.0:
            for pid in self.pools:
                if self.pools[pid].policy.system_pool or self.pools[pid].policy.role == "liquidity_provider":
                    continue
                d = deficit_weights.get(pid, 0.0) / total_deficit
                a = activity_weights.get(pid, 0.0) / total_activity
                w = (1.0 - activity_share) * d + activity_share * a
                if w > 0.0:
                    weights[pid] = w
                    total_weight += w
        elif total_deficit > 1e-9:
            weights = deficit_weights
            total_weight = total_deficit
        elif total_activity > 1e-9 and activity_share > 0.0:
            weights = activity_weights
            total_weight = total_activity
        else:
            role_weights = {"lender": 1.2, "producer": 1.0, "consumer": 0.8}
            for pid, p in self.pools.items():
                if p.policy.system_pool or p.policy.role == "liquidity_provider":
                    continue
                w = role_weights.get(p.policy.role, 1.0)
                weights[pid] = w
                total_weight += w

        if total_weight <= 1e-9:
            return
        for pid, w in weights.items():
            if w <= 0.0:
                continue
            inflow = amount * (w / total_weight)
            if inflow <= 1e-9:
                continue
            self._vault_add(self.pools[pid], stable_id, inflow, "stable_inflow", "system")
            if voucher_share > 0.0:
                minted = self._mint_vouchers_for_pool(self.pools[pid], inflow * voucher_share)
                if minted > 0.0:
                    minted_usd_total += minted
                    minted_pools += 1

        if minted_usd_total > 1e-9:
            self.log.add(Event(
                self.tick,
                "VOUCHER_MINTED",
                amount=minted_usd_total,
                meta={"pools": minted_pools, "share": voucher_share},
            ))

    def _apply_stable_outflow(self, amount: float) -> None:
        if amount <= 1e-9 or not self.pools:
            return
        stable_id = self.cfg.stable_symbol
        available: Dict[str, float] = {}
        total_available = 0.0
        for pid, p in self.pools.items():
            if p.policy.system_pool or p.policy.role == "liquidity_provider":
                continue
            excess = max(0.0, p.vault.get(stable_id) - p.policy.min_stable_reserve)
            if excess > 1e-9:
                available[pid] = excess
                total_available += excess
        if total_available <= 1e-9:
            for pid, p in self.pools.items():
                if p.policy.system_pool or p.policy.role == "liquidity_provider":
                    continue
                bal = p.vault.get(stable_id)
                if bal > 1e-9:
                    available[pid] = bal
                    total_available += bal
        if total_available <= 1e-9:
            return
        amount = min(amount, total_available)
        if amount > 0.0:
            self._stable_offramp_usd_tick += amount
        for pid, avail in available.items():
            take = amount * (avail / total_available)
            if take <= 1e-9:
                continue
            self._vault_sub(self.pools[pid], stable_id, take, "stable_outflow", "system")

    def _apply_stable_growth_network(self, multiplier: int = 1) -> None:
        if not self.pools:
            return
        scale = max(1, int(multiplier or 1))
        stable_id = self.cfg.stable_symbol
        total_stable = sum(
            p.vault.get(stable_id)
            for p in self.pools.values()
            if not p.policy.system_pool and p.policy.role != "liquidity_provider"
        )

        outflow_rate = float(self.cfg.stable_outflow_rate or 0.0)
        if outflow_rate > 0.0 and total_stable > 1e-9:
            outflow = total_stable * (outflow_rate / float(MONTH_TICKS)) * scale
            self._apply_stable_outflow(outflow)
            total_stable = sum(
                p.vault.get(stable_id)
                for p in self.pools.values()
                if not p.policy.system_pool and p.policy.role != "liquidity_provider"
            )

        target = self._stable_supply_target()
        raw_delta = target - total_stable
        smoothing = float(self.cfg.stable_growth_smoothing or 0.0)
        smoothing = min(1.0, max(0.0, smoothing))
        cap_smoothing = 1.0
        max_inflow = None
        if raw_delta > 1e-9:
            total_pool_value = self._network_total_value()
            max_total_inflow = 0.10 * total_pool_value
            voucher_share = max(0.0, float(self.cfg.voucher_inflow_share or 0.0))
            max_inflow = max_total_inflow / (1.0 + voucher_share) if max_total_inflow > 0.0 else 0.0
            if max_inflow > 0.0:
                cap_smoothing = min(1.0, max_inflow / raw_delta)
        effective_smoothing = min(smoothing, cap_smoothing)
        delta = raw_delta * effective_smoothing * scale
        if abs(delta) <= 1e-9:
            return
        if delta > 0.0:
            self._apply_stable_inflow(delta)
        else:
            self._apply_stable_outflow(-delta)
        self.log.add(Event(self.tick, "STABLE_SUPPLY_ADJUSTED", amount=delta,
                           meta={
                               "target": target,
                               "total_before": total_stable,
                               "raw_delta": raw_delta,
                               "smoothing": smoothing,
                               "effective_smoothing": effective_smoothing,
                               "max_inflow": max_inflow,
                           }))

    def _recent_activity_totals(self, window_ticks: int) -> Tuple[float, float, float]:
        tick_min = max(1, self.tick - window_ticks + 1)
        loan_issued = 0.0
        loan_repaid = 0.0
        swap_volume_usd = 0.0

        for e in reversed(self.log.events):
            if e.tick < tick_min:
                break
            if e.event_type == "LOAN_ISSUED":
                loan_issued += float(e.amount or 0.0)
            elif e.event_type == "REPAYMENT_EXECUTED":
                loan_repaid += float(e.amount or 0.0)
            elif e.event_type == "SWAP_EXECUTED":
                receipt = e.meta.get("receipt", {})
                pool_id = receipt.get("pool_id") or e.pool_id
                asset_in = receipt.get("asset_in")
                amount_in = receipt.get("amount_in")
                if not pool_id or not asset_in or amount_in is None:
                    continue
                pool = self.pools.get(pool_id)
                if pool is None:
                    continue
                swap_volume_usd += float(amount_in) * self._asset_value(pool, asset_in)

        return loan_issued, loan_repaid, swap_volume_usd

    def _apply_activity_stable_flow(self, window_ticks: int) -> None:
        mode = self.cfg.stable_flow_mode
        if mode == "none":
            return
        loan_issued, loan_repaid, swap_volume = self._recent_activity_totals(window_ticks)

        net_flow = 0.0
        if mode in ("loan", "both"):
            net_flow += (loan_issued - loan_repaid) * float(self.cfg.stable_flow_loan_scale or 0.0)
        if mode in ("swap", "both"):
            target = float(self.cfg.stable_flow_swap_target_usd or 0.0)
            net_flow += (swap_volume - target) * float(self.cfg.stable_flow_swap_scale or 0.0)

        if abs(net_flow) <= 1e-9:
            return
        if net_flow > 0.0:
            self._apply_stable_inflow(net_flow)
        else:
            self._apply_stable_outflow(-net_flow)
        self.log.add(Event(
            self.tick,
            "STABLE_FLOW_ADJUSTED",
            amount=net_flow,
            meta={
                "window_ticks": window_ticks,
                "loan_issued_usd": loan_issued,
                "loan_repaid_usd": loan_repaid,
                "swap_volume_usd": swap_volume,
            "mode": mode,
            },
        ))

    def _apply_offramp_behavior(self) -> None:
        if not bool(self.cfg.offramps_enabled):
            return
        min_rate = max(0.0, float(self.cfg.offramp_rate_min_per_tick or 0.0))
        max_rate = max(min_rate, float(self.cfg.offramp_rate_max_per_tick or 0.0))
        min_attempts = max(0, int(self.cfg.offramp_min_attempts or 0))
        if max_rate <= 0.0 and min_rate <= 0.0:
            return
        total_offramp = 0.0
        stable_id = self.cfg.stable_symbol
        for pool_id, attempts in self._swap_attempt_counts.items():
            if attempts < min_attempts:
                continue
            pool = self.pools.get(pool_id)
            if pool is None or pool.policy.system_pool or pool.policy.role == "liquidity_provider":
                continue
            ema = float(self._swap_success_ema.get(pool_id, 1.0))
            fail_rate = 1.0 - min(1.0, max(0.0, ema))
            rate = min_rate + (max_rate - min_rate) * fail_rate
            if rate <= 0.0:
                continue
            available = pool.vault.get(stable_id)
            if available <= 1e-9:
                continue
            amount = available * rate
            if amount <= 1e-9:
                continue
            if self._vault_sub(pool, stable_id, amount, "offramp", "fiat"):
                self._stable_offramp_usd_tick += amount
                total_offramp += amount
        if total_offramp > 0.0:
            self.log.add(Event(self.tick, "OFFRAMP_APPLIED", amount=total_offramp))

    def _deposit_fee_asset_to_pool(self, pool: Pool, asset_id: str, amount: float, value: float, action: str) -> None:
        if amount <= 1e-12:
            return
        if not pool.registry.is_listed(asset_id):
            pool.list_asset_with_value_and_limit(asset_id, value=value, window_len=1, cap_in=1e12)
        else:
            if pool.values.get_value(asset_id) <= 0.0:
                pool.values.set_value(asset_id, value)
                self._rebuild_pool_value_cache(pool)
                if asset_id.startswith("VCHR:"):
                    self._rebuild_pool_voucher_cache(pool)
        self._vault_add(pool, asset_id, amount, action, "waterfall")

    def _collect_fee_inflows(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        cfg = self.cfg
        amounts: Dict[str, float] = {}
        value_totals: Dict[str, float] = {}
        pool_fee_usd = 0.0
        clc_fee_usd = 0.0
        for p in self.pools.values():
            if p.policy.system_pool:
                continue
            ledgers = [p.fee_ledger_clc]
            if cfg.waterfall_include_pool_fees:
                ledgers.append(p.fee_ledger_pool)
            for ledger in ledgers:
                for asset_id, amt in ledger.items():
                    if amt <= 1e-12:
                        continue
                    value = p.values.get_value(asset_id)
                    if value <= 0.0:
                        value = 1.0
                    amounts[asset_id] = amounts.get(asset_id, 0.0) + amt
                    value_totals[asset_id] = value_totals.get(asset_id, 0.0) + amt * value
                    if ledger is p.fee_ledger_clc:
                        clc_fee_usd += amt * value
                    else:
                        pool_fee_usd += amt * value
            if cfg.waterfall_include_pool_fees:
                p.fee_ledger_pool.clear()
            p.fee_ledger_clc.clear()
        if self._waterfall_external_inflows:
            stable_id = self.cfg.stable_symbol
            for asset_id, amt in self._waterfall_external_inflows.items():
                if amt <= 1e-12:
                    continue
                value = 1.0
                if asset_id != stable_id and self.clc_pool_id and self.clc_pool_id in self.pools:
                    v = self.pools[self.clc_pool_id].values.get_value(asset_id)
                    if v > 0.0:
                        value = v
                amounts[asset_id] = amounts.get(asset_id, 0.0) + amt
                value_totals[asset_id] = value_totals.get(asset_id, 0.0) + amt * value
            self._waterfall_external_inflows.clear()
        avg_values = {}
        for asset_id, amt in amounts.items():
            if amt <= 1e-12:
                continue
            avg_values[asset_id] = value_totals.get(asset_id, 0.0) / amt
        return amounts, value_totals, avg_values

    def _distribute_liquidity_mandates(self, budget_usd: float) -> float:
        if budget_usd <= 1e-9 or not self.mandates_pool_id:
            return 0.0
        cfg = self.cfg
        mandates_pool = self.pools.get(self.mandates_pool_id)
        if mandates_pool is None:
            return 0.0
        eligible = [p for p in self.pools.values() if not p.policy.system_pool]
        if cfg.liquidity_mandate_mode == "lender_liquidity":
            eligible = [p for p in eligible if p.policy.role == "lender"]
        if not eligible:
            return 0.0
        weights: Dict[str, float] = {}
        if cfg.liquidity_mandate_mode == "lender_liquidity":
            stable_id = cfg.stable_symbol
            for p in eligible:
                stable = p.vault.get(stable_id)
                deficit = max(0.0, p.policy.min_stable_reserve - stable)
                if deficit > 1e-9:
                    weights[p.pool_id] = deficit
            if not weights:
                for p in eligible:
                    stable = max(1.0, p.vault.get(stable_id))
                    weights[p.pool_id] = 1.0 / stable
        elif cfg.liquidity_mandate_mode == "deficit_weighted":
            for p in eligible:
                deficit = max(0.0, p.policy.min_stable_reserve - p.vault.get(cfg.stable_symbol))
                if deficit > 1e-9:
                    weights[p.pool_id] = deficit
        elif cfg.liquidity_mandate_mode == "utilization_weighted":
            for p in eligible:
                weights[p.pool_id] = max(0.0, float(self._swap_volume_usd_by_pool.get(p.pool_id, 0.0)))
        else:
            window = max(1, int(cfg.liquidity_mandate_activity_window_ticks or 1))
            weights = self._recent_pool_activity(window)

        if not weights:
            weights = {p.pool_id: 1.0 for p in eligible}

        total_weight = sum(weights.values())
        if total_weight <= 1e-9:
            return 0.0

        distributed = 0.0
        max_per_pool = float(cfg.liquidity_mandate_max_per_pool_usd or 0.0)
        if cfg.liquidity_mandate_mode == "lender_liquidity":
            max_per_pool = 0.0
        for p in eligible:
            weight = weights.get(p.pool_id, 0.0)
            if weight <= 0.0:
                continue
            alloc = budget_usd * (weight / total_weight)
            if max_per_pool > 0.0:
                alloc = min(alloc, max_per_pool)
            if alloc <= 1e-9:
                continue
            if not self._vault_sub(mandates_pool, cfg.stable_symbol, alloc, "mandate_out", p.pool_id):
                continue
            self._vault_add(p, cfg.stable_symbol, alloc, "mandate_in", mandates_pool.pool_id)
            distributed += alloc

        if cfg.liquidity_mandate_mode == "lender_liquidity":
            remaining = budget_usd - distributed
            if remaining > 1e-9 and eligible:
                top_pool = max(eligible, key=lambda pool: weights.get(pool.pool_id, 0.0))
                if self._vault_sub(mandates_pool, cfg.stable_symbol, remaining, "mandate_out", top_pool.pool_id):
                    self._vault_add(top_pool, cfg.stable_symbol, remaining, "mandate_in", mandates_pool.pool_id)
                    distributed += remaining

        if distributed > 0.0:
            self.log.add(Event(self.tick, "LIQUIDITY_MANDATE_DISTRIBUTED", amount=distributed,
                               meta={"mode": cfg.liquidity_mandate_mode, "pool_count": len(eligible)}))
        return distributed

    def _apply_waterfall(self) -> None:
        if not self.cfg.economics_enabled:
            return
        if not self.ops_pool_id or not self.insurance_pool_id or not self.clc_pool_id or not self.mandates_pool_id:
            return
        ops_pool = self.pools.get(self.ops_pool_id)
        insurance_pool = self.pools.get(self.insurance_pool_id)
        clc_pool = self.pools.get(self.clc_pool_id)
        mandates_pool = self.pools.get(self.mandates_pool_id)
        if ops_pool is None or insurance_pool is None or clc_pool is None or mandates_pool is None:
            return

        amounts, value_totals, avg_values = self._collect_fee_inflows()
        total_fee_usd = sum(value_totals.values())
        cash_usd = 0.0
        kind_usd = 0.0
        conversion_used_usd = 0.0

        if total_fee_usd <= 1e-9:
            self._waterfall_last = {
                "fee_in_usd": 0.0,
                "fee_cash_usd": 0.0,
                "fee_kind_usd": 0.0,
                "chi": 0.0,
                "insurance_alloc_usd": 0.0,
                "ops_alloc_usd": 0.0,
                "mandates_alloc_usd": 0.0,
                "mandates_distributed_usd": 0.0,
                "clc_alloc_usd": 0.0,
                "conversion_used_usd": 0.0,
                "insurance_target_usd": self._insurance_target_usd(),
            }
            self._fee_access_budget_usd = self._compute_fee_access_budget()
            self._waterfall_last["fee_access_budget_usd"] = self._fee_access_budget_usd
            return

        stable_id = self.cfg.stable_symbol
        eligible = set(self.cfg.cash_eligible_assets or [])
        slippage = max(0.0, float(self.cfg.cash_conversion_slippage_bps or 0.0)) / 10000.0
        convertible_usd_total = sum(
            value_totals[a] for a in value_totals
            if a != stable_id and a in eligible
        )
        conversion_cap = self.cfg.cash_conversion_max_usd_per_epoch
        convertible_usd = convertible_usd_total
        if conversion_cap is not None:
            convertible_usd = min(convertible_usd_total, float(conversion_cap))
        conversion_ratio = 0.0
        if convertible_usd_total > 1e-9:
            conversion_ratio = convertible_usd / convertible_usd_total

        for asset_id, amt in amounts.items():
            if amt <= 1e-12:
                continue
            avg_value = avg_values.get(asset_id, 1.0)
            asset_usd = amt * avg_value
            if asset_id == stable_id:
                cash_usd += asset_usd
                continue
            if asset_id in eligible:
                convert_amt = amt * conversion_ratio
                convert_usd = convert_amt * avg_value
                if convert_usd > 0.0:
                    conversion_used_usd += convert_usd
                    cash_usd += convert_usd * (1.0 - slippage)
                remain_amt = amt - convert_amt
                if remain_amt > 1e-12:
                    kind_usd += remain_amt * avg_value
                    self._deposit_fee_asset_to_pool(clc_pool, asset_id, remain_amt, avg_value, "fee_kind")
                continue
            kind_usd += asset_usd
            self._deposit_fee_asset_to_pool(clc_pool, asset_id, amt, avg_value, "fee_kind")

        cash_remaining = cash_usd
        insurance_target = self._insurance_target_usd()
        insurance_fund = insurance_pool.vault.get(stable_id)
        insurance_need = max(0.0, insurance_target - insurance_fund)
        insurance_alloc = min(
            cash_remaining,
            max(0.0, float(self.cfg.insurance_max_topup_usd or 0.0)),
            insurance_need,
        )
        if insurance_alloc > 1e-9:
            self._vault_add(insurance_pool, stable_id, insurance_alloc, "waterfall_insurance", "waterfall")
            cash_remaining -= insurance_alloc

        ops_alloc = min(cash_remaining, max(0.0, float(self.cfg.core_ops_budget_usd or 0.0)))
        if ops_alloc > 1e-9:
            self._vault_add(ops_pool, stable_id, ops_alloc, "waterfall_ops", "waterfall")
            cash_remaining -= ops_alloc

        mandate_budget = 0.0
        if cash_remaining > 1e-9:
            mandate_budget = cash_remaining * max(0.0, float(self.cfg.liquidity_mandate_share or 0.0))
        if mandate_budget > 1e-9:
            self._vault_add(mandates_pool, stable_id, mandate_budget, "waterfall_mandates", "waterfall")
            cash_remaining -= mandate_budget
        mandates_distributed = self._distribute_liquidity_mandates(mandate_budget)

        ops_extra = 0.0
        insurance_extra = 0.0
        clc_alloc = cash_remaining
        if clc_alloc > 1e-9:
            self._vault_add(clc_pool, stable_id, clc_alloc, "waterfall_clc", "waterfall")

        chi = cash_usd / max(1e-9, cash_usd + kind_usd)
        self._waterfall_last = {
            "fee_in_usd": total_fee_usd,
            "fee_cash_usd": cash_usd,
            "fee_kind_usd": kind_usd,
            "chi": chi,
            "insurance_alloc_usd": insurance_alloc + insurance_extra,
            "ops_alloc_usd": ops_alloc + ops_extra,
            "mandates_alloc_usd": mandate_budget,
            "mandates_distributed_usd": mandates_distributed,
            "clc_alloc_usd": clc_alloc,
            "conversion_used_usd": conversion_used_usd,
            "insurance_target_usd": insurance_target,
        }

        self.log.add(Event(self.tick, "WATERFALL_EXECUTED", amount=total_fee_usd,
                           meta={"cash_usd": cash_usd, "kind_usd": kind_usd, "chi": chi}))

        self._fee_access_budget_usd = self._compute_fee_access_budget()
        self._waterfall_last["fee_access_budget_usd"] = self._fee_access_budget_usd
        self.rebuild_indexes()

    def _compute_fee_access_budget(self) -> float:
        if not self.cfg.sclc_fee_access_enabled:
            return 0.0
        if not self.ops_pool_id or not self.insurance_pool_id or not self.clc_pool_id:
            return 0.0
        ops_pool = self.pools.get(self.ops_pool_id)
        insurance_pool = self.pools.get(self.insurance_pool_id)
        clc_pool = self.pools.get(self.clc_pool_id)
        if ops_pool is None or insurance_pool is None or clc_pool is None:
            return 0.0
        ops_ok = True
        ins_ok = True
        if self.cfg.sclc_requires_core_ops:
            ops_ok = ops_pool.vault.get(self.cfg.stable_symbol) >= float(self.cfg.core_ops_budget_usd or 0.0)
        if self.cfg.sclc_requires_insurance_target:
            ins_ok = insurance_pool.vault.get(self.cfg.stable_symbol) >= self._insurance_target_usd()
        if not (ops_ok and ins_ok):
            return 0.0
        clc_stable = clc_pool.vault.get(self.cfg.stable_symbol)
        share = max(0.0, float(self.cfg.sclc_fee_access_share or 0.0))
        cap = max(0.0, float(self.cfg.sclc_emission_cap_usd or 0.0))
        return min(clc_stable * share, cap)

    def _simulate_incidents(self) -> None:
        if not self.cfg.economics_enabled:
            return
        if not self.insurance_pool_id:
            return
        insurance_pool = self.pools.get(self.insurance_pool_id)
        if insurance_pool is None:
            return
        fee_window = int(self.cfg.insurance_fee_window_ticks or 0)
        min_fee_usd = float(self.cfg.insurance_min_fee_usd or 0.0)
        fee_by_pool: Dict[str, float] = {}
        if fee_window > 0 and min_fee_usd > 0.0:
            fee_by_pool = self._recent_pool_clc_fees(fee_window)
        stable_id = self.cfg.stable_symbol
        incident_cap = max(0, int(self.cfg.incident_max_per_tick or 0))
        if incident_cap == 0:
            return
        candidates = [p for p in self.pools.values() if not p.policy.system_pool]
        self.rng.shuffle(candidates)
        incidents = 0
        for p in candidates:
            if incidents >= incident_cap:
                break
            voucher_value = self._pool_voucher_value_usd(p)
            if voucher_value <= 1e-9:
                continue
            if min_fee_usd > 0.0 and fee_window > 0:
                if fee_by_pool.get(p.pool_id, 0.0) < min_fee_usd:
                    continue
            prob = float(self.cfg.incident_base_rate or 0.0) * self._pool_risk_weight(p)
            prob = min(1.0, max(0.0, prob))
            if prob <= 0.0 or self.rng.random() > prob:
                continue
            loss = max(float(self.cfg.incident_min_loss_usd or 0.0),
                       voucher_value * float(self.cfg.incident_loss_rate or 0.0))
            claim_cap = voucher_value * float(self.cfg.incident_haircut_cap or 0.0)
            claim = min(loss, claim_cap) if claim_cap > 0.0 else loss
            if claim <= 1e-9:
                continue
            available = insurance_pool.vault.get(stable_id)
            paid = min(claim, available)
            unpaid = claim - paid
            if paid > 1e-9:
                self._vault_sub(insurance_pool, stable_id, paid, "insurance_payout", p.pool_id)
                self._vault_add(p, stable_id, paid, "insurance_payout", insurance_pool.pool_id)
            self._claims_paid_usd_tick += paid
            self._claims_unpaid_usd_tick += unpaid
            self._incidents_tick += 1
            incidents += 1
            self.log.add(Event(self.tick, "INSURANCE_CLAIM", pool_id=p.pool_id, amount=claim,
                               meta={"paid": paid, "unpaid": unpaid}))

    def _clc_rebalance(self) -> None:
        if not self.cfg.economics_enabled or not self.cfg.clc_rebalance_enabled:
            return
        if not self.clc_pool_id:
            return
        clc_pool = self.pools.get(self.clc_pool_id)
        if clc_pool is None:
            return
        interval = max(1, int(self.cfg.clc_rebalance_interval_ticks or 1))
        if self.tick % interval != 0:
            return
        stable_id = self.cfg.stable_symbol
        total_value = self._pool_total_value(clc_pool)
        if total_value <= 1e-9:
            return
        stable_value = clc_pool.vault.get(stable_id) * self._asset_value(clc_pool, stable_id)
        target_ratio = max(0.0, min(1.0, float(self.cfg.clc_rebalance_target_stable_ratio or 0.0)))
        if stable_value / total_value >= target_ratio:
            return

        max_swaps = max(0, int(self.cfg.clc_rebalance_max_swaps_per_tick or 0))
        if max_swaps == 0:
            return
        swap_frac = max(0.0, float(self.cfg.clc_rebalance_swap_size_frac or 0.0))
        min_usd = max(0.0, float(self.cfg.clc_rebalance_min_usd or 0.0))

        for _ in range(max_swaps):
            voucher_id = self._choose_voucher_for_pool(clc_pool)
            if voucher_id is None:
                break
            value = self._asset_value(clc_pool, voucher_id)
            if value <= 0.0:
                break
            target_usd = max(min_usd, total_value * swap_frac)
            amount_in = min(clc_pool.vault.get(voucher_id), target_usd / value)
            if amount_in <= 1e-9:
                break
            plan, amount_used, used_fallback = self._find_route_with_fallback(
                tick=self.tick,
                start_asset=voucher_id,
                target_asset=stable_id,
                amount_in=amount_in,
                source_pool=clc_pool,
            )
            if used_fallback:
                self.log.add(Event(self.tick, "CLC_REBALANCE_REQUESTED", pool_id=clc_pool.pool_id,
                                   asset_id=voucher_id, amount=amount_used,
                                   meta={"target_asset": stable_id, "fallback": True}))
            if not plan.ok:
                self.log.add(Event(self.tick, "CLC_REBALANCE_FAILED", pool_id=clc_pool.pool_id,
                                   asset_id=voucher_id, amount=amount_used, meta={"reason": plan.reason}))
                continue
            if self.execute_route_from_pool(clc_pool.pool_id, plan, amount_used):
                amount_usd = amount_used * value
                self.log.add(Event(self.tick, "CLC_REBALANCE_EXECUTED", pool_id=clc_pool.pool_id,
                                   asset_id=voucher_id, amount=amount_usd))

    def _update_clc_swap_window(self) -> None:
        if not self.clc_pool_id:
            return
        clc_pool = self.pools.get(self.clc_pool_id)
        if clc_pool is None:
            return
        self._fee_access_budget_usd = self._compute_fee_access_budget()
        if self._waterfall_last:
            self._waterfall_last["fee_access_budget_usd"] = self._fee_access_budget_usd
        if self.cfg.clc_pool_always_open:
            self._clc_access_open = True
            clc_pool.policy.paused = False
            clc_pool.policy.min_stable_reserve = 0.0
            return
        window = max(1, int(self.cfg.sclc_swap_window_ticks or 1))
        open_ticks = max(0, int(self.cfg.sclc_swap_window_open_ticks or 0))
        should_open = False
        if self.cfg.sclc_fee_access_enabled and open_ticks > 0 and self._fee_access_budget_usd > 0.0:
            should_open = (self.tick % window) < open_ticks
        self._clc_access_open = should_open
        clc_pool.policy.paused = False
        stable = clc_pool.vault.get(self.cfg.stable_symbol)
        if should_open:
            budget = min(stable, self._fee_access_budget_usd)
            clc_pool.policy.min_stable_reserve = max(0.0, stable - budget)
        else:
            clc_pool.policy.min_stable_reserve = stable + 1e-6

    def step(self, n_ticks: int = 1) -> None:
        for _ in range(n_ticks):
            self.tick += 1
            self._decay_affinity()
            self._swap_volume_usd_tick = 0.0
            self._swap_volume_usd_by_pool = {}
            self._noam_routing_swaps_tick = 0
            self._noam_clearing_swaps_tick = 0
            self._claims_paid_usd_tick = 0.0
            self._claims_unpaid_usd_tick = 0.0
            self._incidents_tick = 0
            self._stable_onramp_usd_tick = 0.0
            self._stable_offramp_usd_tick = 0.0
            self._swap_attempt_counts = {}

            # exogenous stable inflow
            stable_stride = max(1, int(self.cfg.stable_growth_stride_ticks or 1))
            if self.tick % stable_stride == 0:
                if self.cfg.stable_growth_mode == "network_target":
                    self._apply_stable_growth_network(multiplier=stable_stride)
                else:
                    self._apply_stable_growth_per_pool(multiplier=stable_stride)
            pool_growth_stride = max(1, int(self.cfg.pool_growth_stride_ticks or 1))
            if self.tick % pool_growth_stride == 0:
                self._apply_pool_growth(multiplier=pool_growth_stride)
            desired_stride = max(1, int(self.cfg.desired_assets_stride_ticks or 1))
            if self.tick % desired_stride == 0:
                self._grow_pool_desired_assets()
            self._maybe_refresh_noam_working_set()

            # shock
            if self.cfg.stable_shock_tick is not None and self.tick == self.cfg.stable_shock_tick:
                shock = self.cfg.stable_shock_amount
                shock_pools = 0
                for p in self.pools.values():
                    if p.policy.system_pool:
                        continue
                    self._vault_add(p, self.cfg.stable_symbol, shock, "stable_shock", "system")
                    shock_pools += 1
                shock_total = shock * shock_pools
                if shock_total > 0.0:
                    self._stable_onramp_usd_tick += shock_total
                elif shock_total < 0.0:
                    self._stable_offramp_usd_tick += -shock_total
                self.log.add(Event(self.tick, "STABLE_SHOCK", amount=shock))

            # per-pool swap attempts
            if self.cfg.swap_target_selection_mode == "liquidity_weighted":
                self._refresh_liquidity_cache()
            pools = [
                p for p in self.pools.values()
                if not p.policy.system_pool and p.policy.role != "liquidity_provider"
            ]
            max_active = int(self.cfg.max_active_pools_per_tick or 0)
            if max_active > 0 and max_active < len(pools):
                pools = self.rng.sample(pools, k=max_active)
            if pools:
                self.rng.shuffle(pools)
            remaining_requests = int(self.cfg.swap_requests_budget_per_tick or 0)
            if remaining_requests <= 0:
                remaining_requests = None
            for p in pools:
                if remaining_requests is not None and remaining_requests <= 0:
                    break
                remaining = self._swap_attempts_for_pool(p)
                if remaining <= 0:
                    continue
                if p.policy.role == "producer":
                    attempted = self._attempt_repayment(p)
                    if attempted:
                        remaining -= 1
                        if remaining_requests is not None:
                            remaining_requests -= 1
                if remaining > 0:
                    if remaining_requests is not None:
                        remaining = min(remaining, remaining_requests)
                    attempted = self._random_route_request(source_pool=p, max_assets=remaining)
                    if remaining_requests is not None:
                        remaining_requests = max(0, remaining_requests - attempted)

            self._run_noam_clearing()

            self._apply_offramp_behavior()

            window_ticks = max(1, int(self.cfg.stable_flow_window_ticks or 1))
            if self.cfg.stable_flow_mode != "none" and self.tick % window_ticks == 0:
                self._apply_activity_stable_flow(window_ticks)

            self._simulate_incidents()
            self._apply_liquidity_provider_contributions()

            if self.cfg.economics_enabled:
                epoch = max(1, int(self.cfg.waterfall_epoch_ticks or 1))
                if self.tick % epoch == 0:
                    self._apply_waterfall()
                self._clc_rebalance()
            self._update_clc_swap_window()

            self._update_utilization_boost()
            self.snapshot_metrics()

    def _random_route_request(self, source_pool: Optional["Pool"] = None, max_assets: Optional[int] = None) -> int:
        """
        Choose a random source pool and try to swap some asset it holds into a different asset.
        Executes via escrow so multi-hop can touch multiple pools.
        """
        if not self.pools:
            return 0

        if source_pool is None:
            candidates = [p for p in self.pools.values() if not p.policy.system_pool]
            if not candidates:
                return 0
            source_pool = random.choice(candidates)
        if source_pool.policy.system_pool:
            return 0
        if not source_pool.vault.inventory:
            return 0

        # try each asset_in with positive inventory
        asset_candidates = [a for a, amt in source_pool.vault.inventory.items() if amt > 1e-9]
        if not asset_candidates:
            return 0

        if max_assets is not None and max_assets > 0:
            if len(asset_candidates) > max_assets:
                mode = self.cfg.swap_asset_selection_mode
                if mode == "value_weighted":
                    weights = np.array(
                        [source_pool.vault.get(a) * self._asset_value(source_pool, a) for a in asset_candidates],
                        dtype=float,
                    )
                    total = float(weights.sum())
                    if total > 0.0:
                        probs = weights / total
                        asset_candidates = list(
                            np.random.choice(asset_candidates, size=max_assets, replace=False, p=probs)
                        )
                    else:
                        asset_candidates = random.sample(asset_candidates, k=max_assets)
                else:
                    asset_candidates = random.sample(asset_candidates, k=max_assets)

        attempted = 0

        for asset_in in asset_candidates:
            max_targets = max(1, int(self.cfg.swap_target_retry_count or 1))
            targets_tried: Set[str] = set()
            for _ in range(max_targets):
                if max_assets is not None and max_assets > 0 and attempted >= max_assets:
                    break
                asset_out = self._choose_target_asset(asset_in, source_pool, exclude=targets_tried)
                if not asset_out or asset_out == asset_in:
                    break
                targets_tried.add(asset_out)

                amount_in = self._sample_amount_in(source_pool, asset_in)
                if amount_in <= 1e-9:
                    break

                attempted += 1
                self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                                   asset_id=asset_in, amount=amount_in, meta={"target_asset": asset_out}))
                plan, amount_used, used_fallback = self._find_route_with_fallback(
                    tick=self.tick,
                    start_asset=asset_in,
                    target_asset=asset_out,
                    amount_in=amount_in,
                    source_pool=source_pool,
                )
                if used_fallback:
                    self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                                       asset_id=asset_in, amount=amount_used,
                                       meta={"target_asset": asset_out, "fallback": True}))

                if not plan.ok:
                    self.log.add(Event(self.tick, "ROUTE_FAILED", pool_id=source_pool.pool_id,
                                       asset_id=asset_in, amount=amount_used, meta={"reason": plan.reason, "target": asset_out}))
                    self._record_swap_attempt(source_pool.pool_id, success=False)
                    continue

                self.log.add(Event(self.tick, "ROUTE_FOUND", pool_id=source_pool.pool_id,
                                   meta={"hops": [h.__dict__ for h in plan.hops], "target": asset_out}))

                # Execute with escrow:
                ok = self.execute_route_from_pool(source_pool.pool_id, plan, amount_used)
                self._record_swap_attempt(source_pool.pool_id, success=ok)
                if ok:
                    break

        return attempted

    def _attempt_repayment(self, source_pool: "Pool") -> bool:
        if source_pool.policy.role != "producer":
            return False
        agent = self.agents.get(source_pool.steward_id)
        if agent is None:
            return False
        voucher_id = agent.voucher_spec.voucher_id
        period = max(1, int(self.cfg.loan_activity_period_ticks or 1))
        if period > 1:
            phase = self._loan_phase_for(agent.agent_id, period)
            if (self.tick + phase) % period != 0:
                return False
        lender_pools = {
            pid for pid, p in self.pools.items()
            if p.policy.role == "lender" and p.vault.get(voucher_id) > 1e-9 and p.registry.is_listed(voucher_id)
        }
        debt = sum(self.pools[pid].vault.get(voucher_id) for pid in lender_pools)
        if debt <= 1e-9:
            return self._attempt_new_loan(source_pool, voucher_id)
        if not lender_pools:
            return False

        # choose an input asset (USD preferred)
        inv = source_pool.vault.inventory
        asset_in = self.cfg.stable_symbol if inv.get(self.cfg.stable_symbol, 0.0) > 1e-9 else None
        if asset_in is None:
            candidates = [(a, amt) for a, amt in inv.items() if a != voucher_id and amt > 1e-9]
            if not candidates:
                return False
            candidates.sort(key=lambda kv: kv[1], reverse=True)
            asset_in = candidates[0][0]

        available = inv.get(asset_in, 0.0)
        debt_value = debt * self._asset_value(source_pool, voucher_id)
        payment_usd = debt_value / max(1, self.cfg.loan_term_weeks)
        if period > 1:
            payment_usd *= period
        payment_usd = min(payment_usd, debt_value)
        amount_in = min(available, payment_usd / self._asset_value(source_pool, asset_in))
        if amount_in <= 1e-9:
            return False

        self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                           asset_id=asset_in, amount=amount_in,
                           meta={"target_asset": voucher_id, "repayment": True}))
        plan, amount_used, used_fallback = self._find_route_with_fallback(
            tick=self.tick,
            start_asset=asset_in,
            target_asset=voucher_id,
            amount_in=amount_in,
            source_pool=source_pool,
            target_pools=lender_pools,
        )
        if used_fallback:
            self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                               asset_id=asset_in, amount=amount_used,
                               meta={"target_asset": voucher_id, "repayment": True, "fallback": True}))

        if not plan.ok:
            self.log.add(Event(self.tick, "ROUTE_FAILED", pool_id=source_pool.pool_id,
                               asset_id=asset_in, amount=amount_used,
                               meta={"reason": plan.reason, "target": voucher_id, "repayment": True}))
            self._record_swap_attempt(source_pool.pool_id, success=False)
            return True

        self.log.add(Event(self.tick, "ROUTE_FOUND", pool_id=source_pool.pool_id,
                           meta={"hops": [h.__dict__ for h in plan.hops], "target": voucher_id, "repayment": True}))
        ok = self.execute_route_from_pool(source_pool.pool_id, plan, amount_used)
        self._record_swap_attempt(source_pool.pool_id, success=ok)
        if ok:
            amount_usd = amount_used * self._asset_value(source_pool, asset_in)
            self.log.add(Event(self.tick, "REPAYMENT_EXECUTED", pool_id=source_pool.pool_id,
                               asset_id=self.cfg.stable_symbol, amount=amount_usd,
                               meta={"asset_in": asset_in, "amount_in": amount_used}))
        return True

    def _attempt_new_loan(self, source_pool: "Pool", voucher_id: str) -> bool:
        lenders = {
            pid for pid, p in self.pools.items()
            if p.policy.role == "lender"
            and p.vault.get(self.cfg.stable_symbol) > 1e-9
            and p.registry.is_listed(voucher_id)
        }
        if not lenders:
            return False

        amount_in = self._sample_amount_in(source_pool, voucher_id)
        if amount_in <= 1e-9:
            return False

        have = source_pool.vault.get(voucher_id)
        if have + 1e-9 < amount_in:
            mint = amount_in - have
            self._vault_add(source_pool, voucher_id, mint, "loan_issue", source_pool.steward_id)
            agent = self.agents.get(source_pool.steward_id)
            if agent is not None:
                agent.issuer.issue(mint)

        self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                           asset_id=voucher_id, amount=amount_in,
                           meta={"target_asset": self.cfg.stable_symbol, "borrow": True}))
        plan, amount_used, used_fallback = self._find_route_with_fallback(
            tick=self.tick,
            start_asset=voucher_id,
            target_asset=self.cfg.stable_symbol,
            amount_in=amount_in,
            source_pool=source_pool,
            target_pools=lenders,
        )
        if used_fallback:
            self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                               asset_id=voucher_id, amount=amount_used,
                               meta={"target_asset": self.cfg.stable_symbol, "borrow": True, "fallback": True}))

        if not plan.ok:
            self.log.add(Event(self.tick, "ROUTE_FAILED", pool_id=source_pool.pool_id,
                               asset_id=voucher_id, amount=amount_used,
                               meta={"reason": plan.reason, "target": self.cfg.stable_symbol, "borrow": True}))
            self._record_swap_attempt(source_pool.pool_id, success=False)
            return True

        self.log.add(Event(self.tick, "ROUTE_FOUND", pool_id=source_pool.pool_id,
                           meta={"hops": [h.__dict__ for h in plan.hops],
                                 "target": self.cfg.stable_symbol, "borrow": True}))
        ok = self.execute_route_from_pool(source_pool.pool_id, plan, amount_used)
        self._record_swap_attempt(source_pool.pool_id, success=ok)
        if ok:
            amount_usd = amount_used * self._asset_value(source_pool, voucher_id)
            self.log.add(Event(self.tick, "LOAN_ISSUED", pool_id=source_pool.pool_id,
                               asset_id=self.cfg.stable_symbol, amount=amount_usd,
                               meta={"asset_in": voucher_id, "amount_in": amount_used}))
        return True

    def execute_route_from_pool(self, source_pool_id: str, plan: RoutePlan, amount_in: float) -> bool:
        """
        Withdraw asset_in from source pool into escrow, execute hop swaps, deposit output back to source.
        If output is voucher, it may exit and redeem (your 'final settlement sink').
        """
        if not plan.hops:
            return False

        source_pool = self.pools[source_pool_id]
        asset_in = plan.hops[0].asset_in
        if not self._vault_sub(source_pool, asset_in, amount_in, "route_withdraw", f"escrow:{source_pool_id}"):
            self.log.add(Event(self.tick, "EXEC_ROUTE_FAILED", pool_id=source_pool_id, meta={"reason": "source_insufficient"}))
            return False

        escrow: Dict[str, float] = {asset_in: amount_in}
        actor = f"escrow:{source_pool_id}"

        current_amount = amount_in
        current_asset = asset_in

        for hop in plan.hops:
            if hop.pool_id == source_pool_id:
                self.log.add(Event(self.tick, "SWAP_FAILED", pool_id=hop.pool_id, asset_id=hop.asset_in, amount=hop.amount_in,
                                   meta={"reason": "self_swap_not_allowed"}))
                return False
            pool = self.pools[hop.pool_id]
            if escrow.get(current_asset, 0.0) <= 1e-9:
                self.log.add(Event(self.tick, "EXEC_ROUTE_FAILED", pool_id=source_pool_id, meta={"reason": "escrow_empty"}))
                return False

            amt_in = min(current_amount, escrow[current_asset])
            receipt = pool.execute_swap(self.tick, actor=actor, asset_in=current_asset, amount_in=amt_in, asset_out=hop.asset_out)

            if receipt.status != "executed":
                self._noam_update_edge_after_swap(
                    pool,
                    receipt.asset_in,
                    receipt.asset_out,
                    float(receipt.amount_in),
                    success=False,
                    fail_reason=receipt.fail_reason,
                )
                # refund remaining escrow to source pool (best-effort)
                for a, amt in list(escrow.items()):
                    if amt > 1e-9:
                        self._vault_add(source_pool, a, amt, "route_refund", "escrow")
                self.log.add(Event(self.tick, "SWAP_FAILED", pool_id=pool.pool_id, asset_id=current_asset, amount=amt_in,
                                   meta={"reason": receipt.fail_reason, "hop": hop.__dict__}))
                return False

            # update escrow balances
            escrow[current_asset] = escrow.get(current_asset, 0.0) - amt_in
            if escrow[current_asset] <= 1e-12:
                escrow.pop(current_asset, None)
            escrow[hop.asset_out] = escrow.get(hop.asset_out, 0.0) + receipt.amount_out

            self.log.add(Event(self.tick, "SWAP_EXECUTED", pool_id=pool.pool_id,
                   meta={"receipt": receipt.to_dict()}))
            gross_out = receipt.amount_out + float(receipt.fees.total_fee)
            self._update_pool_caches(pool, receipt.asset_in, float(receipt.amount_in))
            self._update_pool_caches(pool, receipt.asset_out, -float(gross_out))
            self._record_fee_cumulative(receipt)
            self._noam_update_edge_after_swap(
                pool,
                receipt.asset_in,
                receipt.asset_out,
                float(receipt.amount_in),
                success=True,
            )
            self._noam_routing_swaps_tick += 1
            swap_usd = receipt.amount_in * self._asset_value(pool, receipt.asset_in)
            self._swap_volume_usd_tick += swap_usd
            self._swap_volume_usd_by_pool[pool.pool_id] = (
                self._swap_volume_usd_by_pool.get(pool.pool_id, 0.0) + swap_usd
            )
            self._update_affinity(source_pool_id, pool.pool_id, swap_usd)

            if pool.policy.role == "consumer" and current_asset.startswith("VCHR:"):
                spec = self.factory.voucher_specs.get(current_asset)
                if spec and spec.issuer_id != pool.steward_id:
                    self._redeem_voucher_from_pool(pool, current_asset, amt_in, spec.issuer_id)

            current_asset = hop.asset_out
            current_amount = receipt.amount_out

        # Decide: deposit back or redeem if voucher
        out_asset = current_asset
        out_amount = escrow.get(out_asset, 0.0)

        if out_amount <= 1e-9:
            return False

        if out_asset.startswith("VCHR:"):
            # voucher exit
            self.log.add(Event(self.tick, "VOUCHER_EXIT_NETWORK", pool_id=source_pool_id, asset_id=out_asset, amount=out_amount))

            spec = self.factory.voucher_specs.get(out_asset)
            issuer_id = spec.issuer_id if spec else None
            if issuer_id in self.agents:
                self.agents[issuer_id].issuer.redeem(out_amount)
                issuer_pool_id = self.agents[issuer_id].pool_id
                issuer_pool = self.pools.get(issuer_pool_id)
                if issuer_pool is not None:
                    self._vault_add(issuer_pool, out_asset, out_amount, "redeem_receive", source_pool_id)
                escrow[out_asset] = 0.0
                self.log.add(Event(self.tick, "VOUCHER_REDEEMED", actor_id=issuer_id, asset_id=out_asset, amount=out_amount))
                self._noam_route_cache_store(source_pool_id, plan, amount_in)
                return True
            self.log.add(Event(self.tick, "VOUCHER_REDEEM_FAILED", pool_id=source_pool_id, asset_id=out_asset, amount=out_amount,
                               meta={"reason": "missing_issuer", "level": "error"}))
            return False

        # normal asset: deposit back to source pool
        self._vault_add(source_pool, out_asset, out_amount, "route_deposit", "escrow")
        self._noam_route_cache_store(source_pool_id, plan, amount_in)
        return True

    def _redeem_voucher_from_pool(self, holder_pool: "Pool", voucher_id: str, amount: float, issuer_id: str) -> None:
        if amount <= 1e-9:
            return
        issuer = self.agents.get(issuer_id)
        if issuer is None:
            return
        issuer_pool = self.pools.get(issuer.pool_id)
        if issuer_pool is None:
            return
        if not self._vault_sub(holder_pool, voucher_id, amount, "redeem_send", issuer.pool_id):
            return
        self._vault_add(issuer_pool, voucher_id, amount, "redeem_receive", holder_pool.pool_id)
        issuer.issuer.redeem(amount)
        self.log.add(Event(self.tick, "VOUCHER_REDEEMED_CONSUMER", actor_id=issuer_id,
                           pool_id=holder_pool.pool_id, asset_id=voucher_id, amount=amount))

    def snapshot_metrics(self) -> None:
        cfg = self.cfg
        metrics_stride = int(cfg.metrics_stride or 0)
        pool_stride = int(cfg.pool_metrics_stride or 0)
        do_network = metrics_stride > 0 and self.tick % metrics_stride == 0
        do_pool = pool_stride > 0 and self.tick % pool_stride == 0
        if not do_network and not do_pool:
            return
        # network-level
        total_pool_value = 0.0
        fee_pool_total_usd = 0.0
        fee_clc_total_usd = 0.0
        pool_rows = []
        if do_pool or do_network:
            for pid, p in self.pools.items():
                pool_value = self._pool_total_value(p)
                if do_network and not p.policy.system_pool:
                    for asset_id, amt in p.fee_ledger_pool.items():
                        value = p.values.get_value(asset_id)
                        if value > 0.0 and amt > 0.0:
                            fee_pool_total_usd += amt * value
                    for asset_id, amt in p.fee_ledger_clc.items():
                        value = p.values.get_value(asset_id)
                        if value > 0.0 and amt > 0.0:
                            fee_clc_total_usd += amt * value
                if do_network and not p.policy.system_pool:
                    total_pool_value += pool_value
                if do_pool:
                    swap_volume_usd = float(self._swap_volume_usd_by_pool.get(pid, 0.0))
                    utilization = swap_volume_usd / max(1e-9, pool_value)
                    fee_pool_usd = float(p.fee_ledger_pool.get(cfg.stable_symbol, 0.0))
                    fee_clc_usd = float(p.fee_ledger_clc.get(cfg.stable_symbol, 0.0))
                    fee_pool_voucher = sum(
                        amt for asset_id, amt in p.fee_ledger_pool.items()
                        if asset_id.startswith("VCHR:")
                    )
                    fee_clc_voucher = sum(
                        amt for asset_id, amt in p.fee_ledger_clc.items()
                        if asset_id.startswith("VCHR:")
                    )
                    pool_rows.append({
                        "tick": self.tick,
                        "pool_id": pid,
                        "mode": p.policy.mode,
                        "role": p.policy.role,
                        "system_pool": bool(p.policy.system_pool),
                        "stable": p.vault.get(cfg.stable_symbol),
                        "vouchers": self._pool_voucher_value_usd(p),
                        "fee_pool_USD": fee_pool_usd,
                        "fee_pool_VCHR": fee_pool_voucher,
                        "fee_clc_USD": fee_clc_usd,
                        "fee_clc_VCHR": fee_clc_voucher,
                        "inventory_assets_count": len(p.vault.inventory),
                        "swap_volume_usd_tick": swap_volume_usd,
                        "utilization_rate": utilization,
                    })
            if do_pool:
                self.metrics.add_pool_rows(pool_rows)

        if do_network:
            swap_receipts = sum(len(p.receipts.receipts) for p in self.pools.values() if not p.policy.system_pool)
            stable_total = sum(
                p.vault.get(cfg.stable_symbol) for p in self.pools.values() if not p.policy.system_pool
            )
            voucher_total = sum(
                amt
                for p in self.pools.values()
                if not p.policy.system_pool
                for asset_id, amt in p.vault.inventory.items()
                if asset_id.startswith("VCHR:")
            )
            pools_under_reserve = sum(
                1 for p in self.pools.values()
                if not p.policy.system_pool and p.vault.get(cfg.stable_symbol) < p.policy.min_stable_reserve
            )
            debt_outstanding_units = 0.0
            debt_outstanding_usd = 0.0
            for p in self.pools.values():
                if p.policy.role != "lender":
                    continue
                for asset_id, amt in p.vault.inventory.items():
                    if not asset_id.startswith("VCHR:"):
                        continue
                    if amt <= 0.0:
                        continue
                    debt_outstanding_units += amt
                    value = p.values.get_value(asset_id)
                    if value <= 0.0:
                        value = 1.0
                    debt_outstanding_usd += amt * value

            redeemed_total = 0.0
            outstanding_total = sum(a.issuer.outstanding_supply for a in self.agents.values())
            outstanding_value_usd = 0.0
            for agent in self.agents.values():
                outstanding = agent.issuer.outstanding_supply
                if outstanding <= 1e-9:
                    continue
                issuer_pool = self.pools.get(agent.pool_id)
                value = 0.0
                if issuer_pool is not None:
                    value = issuer_pool.values.get_value(agent.voucher_spec.voucher_id)
                if value <= 0.0:
                    value = 1.0
                outstanding_value_usd += outstanding * value

            repayment_volume_usd = 0.0
            loan_issuance_usd = 0.0
            transactions_per_tick = 0
            route_requested = 0
            route_found = 0
            route_failed = 0
            for e in reversed(self.log.events):
                if e.tick != self.tick:
                    if e.tick < self.tick:
                        break
                    continue
                if e.event_type == "REPAYMENT_EXECUTED":
                    repayment_volume_usd += float(e.amount or 0.0)
                elif e.event_type == "LOAN_ISSUED":
                    loan_issuance_usd += float(e.amount or 0.0)
                elif e.event_type in ("VOUCHER_REDEEMED", "VOUCHER_REDEEMED_CONSUMER"):
                    redeemed_total += float(e.amount or 0.0)
                elif e.event_type == "SWAP_EXECUTED":
                    transactions_per_tick += 1
                elif e.event_type == "ROUTE_REQUESTED":
                    route_requested += 1
                elif e.event_type == "ROUTE_FOUND":
                    route_found += 1
                elif e.event_type == "ROUTE_FAILED":
                    route_failed += 1

            swap_volume_usd_tick = float(self._swap_volume_usd_tick or 0.0)
            utilization_rate = swap_volume_usd_tick / max(1e-9, total_pool_value)

            num_system = sum(1 for p in self.pools.values() if p.policy.system_pool)
            num_active = len(self.pools) - num_system
            insurance_target = self._insurance_target_usd() if self.cfg.economics_enabled else 0.0
            ops_usd = 0.0
            insurance_usd = 0.0
            clc_usd = 0.0
            mandates_usd = 0.0
            if self.ops_pool_id and self.ops_pool_id in self.pools:
                ops_usd = self.pools[self.ops_pool_id].vault.get(cfg.stable_symbol)
            if self.insurance_pool_id and self.insurance_pool_id in self.pools:
                insurance_usd = self.pools[self.insurance_pool_id].vault.get(cfg.stable_symbol)
            if self.clc_pool_id and self.clc_pool_id in self.pools:
                clc_usd = self.pools[self.clc_pool_id].vault.get(cfg.stable_symbol)
            if self.mandates_pool_id and self.mandates_pool_id in self.pools:
                mandates_usd = self.pools[self.mandates_pool_id].vault.get(cfg.stable_symbol)
            insurance_coverage = insurance_usd / insurance_target if insurance_target > 1e-9 else 0.0

            fee_pool_cumulative = self._fee_pool_cumulative_usd
            self.metrics.add_network({
                "tick": self.tick,
                "num_pools": num_active,
                "num_system_pools": num_system,
                "num_assets": len(self.factory.asset_universe),
                "swap_receipts_total": swap_receipts,
                "stable_total_in_pools": stable_total,
                "voucher_total_in_pools": voucher_total,
                "pools_under_stable_reserve": pools_under_reserve,
                "debt_outstanding_units": debt_outstanding_units,
                "debt_outstanding_usd": debt_outstanding_usd,
                "redeemed_total": redeemed_total,
                "outstanding_voucher_supply_total": outstanding_total,
                "outstanding_voucher_value_usd": outstanding_value_usd,
                "repayment_volume_usd": repayment_volume_usd,
                "loan_issuance_volume_usd": loan_issuance_usd,
                "swap_volume_usd_tick": swap_volume_usd_tick,
                "utilization_rate": utilization_rate,
                "transactions_per_tick": transactions_per_tick,
                "route_requested_tick": int(route_requested),
                "route_found_tick": int(route_found),
                "route_failed_tick": int(route_failed),
                "noam_routing_swaps_tick": int(self._noam_routing_swaps_tick),
                "noam_clearing_swaps_tick": int(self._noam_clearing_swaps_tick),
                "fee_pool_total_usd": fee_pool_total_usd,
                "fee_clc_total_usd": fee_clc_total_usd,
                "fee_pool_cumulative_usd": float(fee_pool_cumulative),
                "fee_clc_cumulative_usd": float(self._fee_clc_cumulative_usd),
                "fee_pool_cumulative_voucher": float(self._fee_pool_cumulative_voucher),
                "fee_clc_cumulative_voucher": float(self._fee_clc_cumulative_voucher),
                "fee_in_usd_epoch": float(self._waterfall_last.get("fee_in_usd", 0.0)),
                "fee_cash_usd_epoch": float(self._waterfall_last.get("fee_cash_usd", 0.0)),
                "fee_kind_usd_epoch": float(self._waterfall_last.get("fee_kind_usd", 0.0)),
                "fee_chi": float(self._waterfall_last.get("chi", 0.0)),
                "fee_conversion_used_usd_epoch": float(self._waterfall_last.get("conversion_used_usd", 0.0)),
                "waterfall_ops_alloc_usd_epoch": float(self._waterfall_last.get("ops_alloc_usd", 0.0)),
                "waterfall_insurance_alloc_usd_epoch": float(self._waterfall_last.get("insurance_alloc_usd", 0.0)),
                "waterfall_mandates_alloc_usd_epoch": float(self._waterfall_last.get("mandates_alloc_usd", 0.0)),
                "waterfall_clc_alloc_usd_epoch": float(self._waterfall_last.get("clc_alloc_usd", 0.0)),
                "ops_pool_usd": ops_usd,
                "insurance_fund_usd": insurance_usd,
                "insurance_target_usd": insurance_target,
                "insurance_coverage_ratio": insurance_coverage,
                "mandates_pool_usd": mandates_usd,
                "clc_pool_usd": clc_usd,
                "mandates_distributed_usd_epoch": float(self._waterfall_last.get("mandates_distributed_usd", 0.0)),
                "fee_access_budget_usd": float(self._waterfall_last.get("fee_access_budget_usd", 0.0)),
                "claims_paid_usd_tick": float(self._claims_paid_usd_tick),
                "claims_unpaid_usd_tick": float(self._claims_unpaid_usd_tick),
                "incidents_tick": int(self._incidents_tick),
                "stable_onramp_usd_tick": float(self._stable_onramp_usd_tick),
                "stable_offramp_usd_tick": float(self._stable_offramp_usd_tick),
            })
