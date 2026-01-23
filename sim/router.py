from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from collections import deque
import random

@dataclass
class Hop:
    pool_id: str
    asset_in: str
    asset_out: str
    amount_in: float

@dataclass
class RoutePlan:
    ok: bool
    reason: str
    hops: List[Hop]
    expected_amount_out: float = 0.0

class Router:
    """
    BFS on (current_asset) where each hop chooses a pool that accepts current_asset
    and swaps to some next_asset that pool can pay out (inventory>0 and listed).
    """
    def __init__(self, max_hops: int = 3) -> None:
        self.max_hops = max_hops

    def find_route(
        self,
        tick: int,
        start_asset: str,
        target_asset: str,
        amount_in: float,
        pools: Dict[str, "Pool"],
        accept_index: Dict[str, Set[str]],
        source_pool_id: Optional[str] = None,
        target_pools: Optional[Set[str]] = None,
        pool_affinity: Optional[Dict[Tuple[str, str], float]] = None,
        affinity_bias: float = 0.0,
        max_candidate_pools: Optional[int] = None,
    ) -> RoutePlan:
        if start_asset == target_asset:
            return RoutePlan(ok=True, reason="trivial", hops=[], expected_amount_out=amount_in)

        # state: (asset, depth)
        q = deque()
        q.append((start_asset, 0))
        # parent pointers: (asset)->(prev_asset, hop)
        parent: Dict[Tuple[str,int], Tuple[Tuple[str,int], Hop]] = {}
        visited: Set[Tuple[str,int]] = set()
        visited.add((start_asset, 0))

        # We'll allow any intermediate asset, bounded by max_hops
        # Also we approximate amounts: assume same amount_in at each stage for search;
        # actual execution recomputes amounts hop-by-hop.
        while q:
            asset, depth = q.popleft()
            if depth >= self.max_hops:
                continue

            candidate_pools = list(accept_index.get(asset, set()))
            if pool_affinity and source_pool_id and affinity_bias > 0.0 and len(candidate_pools) > 1:
                bias = min(1.0, max(0.0, affinity_bias))
                scored = []
                for pid in candidate_pools:
                    key = (source_pool_id, pid) if source_pool_id < pid else (pid, source_pool_id)
                    score = float(pool_affinity.get(key, 0.0))
                    if bias < 1.0:
                        score = (score * bias) + (random.random() * (1.0 - bias))
                    scored.append((score, pid))
                scored.sort(reverse=True)
                candidate_pools = [pid for _, pid in scored]
            max_candidates = int(max_candidate_pools or 0)
            if max_candidates > 0 and len(candidate_pools) > max_candidates:
                if pool_affinity and source_pool_id and affinity_bias > 0.0:
                    candidate_pools = candidate_pools[:max_candidates]
                else:
                    candidate_pools = random.sample(candidate_pools, k=max_candidates)

            for pid in candidate_pools:
                if source_pool_id is not None and pid == source_pool_id:
                    continue
                pool = pools[pid]
                if pool.policy.paused:
                    continue
                # try swapping into any listed asset_out that pool has inventory for
                for asset_out, amt_out_avail in list(pool.vault.inventory.items()):
                    if amt_out_avail <= 1e-9:
                        continue
                    if not pool.registry.is_listed(asset_out):
                        continue
                    if asset_out == asset:
                        continue
                    okq, reasonq, amt_out_net, _ = pool.quote_swap(asset, amount_in, asset_out)
                    if not okq or amt_out_net <= 1e-9:
                        continue

                    # simple feasibility check (inventory + limiter + reserve guardrail)
                    ok, _reason = pool.can_swap(tick, asset, amount_in, asset_out, amt_out_net)
                    if not ok:
                        continue

                    next_state = (asset_out, depth + 1)
                    if next_state in visited:
                        continue

                    hop = Hop(pool_id=pid, asset_in=asset, asset_out=asset_out, amount_in=amount_in)
                    parent[next_state] = ((asset, depth), hop)
                    visited.add(next_state)

                    if asset_out == target_asset:
                        if target_pools is not None and pid not in target_pools:
                            continue
                        # reconstruct
                        hops: List[Hop] = []
                        cur = next_state
                        while cur != (start_asset, 0):
                            prev, h = parent[cur]
                            hops.append(h)
                            cur = prev
                        hops.reverse()
                        return RoutePlan(ok=True, reason="ok", hops=hops, expected_amount_out=amt_out_net)

                    q.append(next_state)

        return RoutePlan(ok=False, reason="no_path_found", hops=[])
