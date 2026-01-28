from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import random

from .config import ScenarioConfig
from .core import Asset, VoucherSpec, IssuerLedger, Pool, PoolPolicy, FeeRegistry, AgentRole

@dataclass
class Agent:
    agent_id: str
    pool_id: str
    role: AgentRole
    issuer: IssuerLedger
    voucher_spec: VoucherSpec

class PoolFactory:
    def __init__(self, cfg: ScenarioConfig) -> None:
        self.cfg = cfg
        self.asset_universe: Dict[str, Asset] = {}
        self.voucher_specs: Dict[str, VoucherSpec] = {}

        # stable asset
        self.asset_universe[cfg.stable_symbol] = Asset(cfg.stable_symbol, "stable", None)
        if cfg.sclc_symbol:
            self.asset_universe[cfg.sclc_symbol] = Asset(cfg.sclc_symbol, "other", None)

        self.agent_counter = 0
        self.pool_counter = 0

    def _new_agent_id(self) -> str:
        self.agent_counter += 1
        return f"agent_{self.agent_counter:04d}"

    def _new_pool_id(self) -> str:
        self.pool_counter += 1
        return f"pool_{self.pool_counter:04d}"

    def sample_agent_role(self, *, allow_liquidity_provider: bool = True) -> AgentRole:
        p_lp = max(0.0, float(self.cfg.p_liquidity_provider)) if allow_liquidity_provider else 0.0
        p_lender = max(0.0, float(self.cfg.p_lender))
        p_producer = max(0.0, float(self.cfg.p_producer))
        p_consumer = max(0.0, float(self.cfg.p_consumer))
        total = p_lp + p_lender + p_producer + p_consumer
        if total <= 0.0:
            return "consumer"
        r = random.random() * total
        if r < p_lp:
            return "liquidity_provider"
        r -= p_lp
        if r < p_lender:
            return "lender"
        r -= p_lender
        if r < p_producer:
            return "producer"
        return "consumer"

    def create_agent_and_pool(
        self,
        role: Optional[AgentRole] = None,
        *,
        allow_liquidity_provider: bool = True,
    ) -> tuple[Agent, Pool]:
        cfg = self.cfg
        agent_id = self._new_agent_id()
        pool_id = self._new_pool_id()

        # each agent issues its own voucher
        voucher_id = f"VCHR:{agent_id}"
        v_spec = VoucherSpec(voucher_id=voucher_id, issuer_id=agent_id, redeem_prob_base=cfg.base_redeem_prob)
        self.voucher_specs[voucher_id] = v_spec
        self.asset_universe[voucher_id] = Asset(voucher_id, "voucher", issuer_id=agent_id)

        issuer = IssuerLedger(voucher_id=voucher_id)
        role = role or self.sample_agent_role(allow_liquidity_provider=allow_liquidity_provider)
        agent = Agent(agent_id=agent_id, pool_id=pool_id, role=role, issuer=issuer, voucher_spec=v_spec)

        if role == "liquidity_provider":
            mode = "none"
        elif role == "consumer":
            mode = "swap_only"
        else:
            mode = "mixed"
        redeem_bias = 0.0

        policy = PoolPolicy(
            mode=mode,
            role=role,
            min_stable_reserve=max(0.0, np.random.exponential(200.0)),
            redemption_bias=redeem_bias,
            paused=False,
            limits_enabled=cfg.swap_limits_enabled,
        )
        if role == "lender":
            policy.min_stable_reserve = 0.0
        fees = FeeRegistry(pool_fee_rate=cfg.pool_fee_rate, clc_rake_rate=cfg.clc_rake_rate)
        pool = Pool(pool_id=pool_id, steward_id=agent_id, stable_id=cfg.stable_symbol, policy=policy, fees=fees)
        pool.debug_inventory = cfg.debug_inventory

        return agent, pool

    def sample_assets(self, k: int, p_overlap: float) -> List[str]:
        """
        Choose k assets to reference, biased toward overlap with existing universe.
        Always includes stable indirectly via general listing later.
        """
        assets = [a for a in self.asset_universe.keys() if a != self.cfg.sclc_symbol]
        if not assets:
            return [self.cfg.stable_symbol]

        chosen: List[str] = []
        for _ in range(max(0, k)):
            if random.random() < p_overlap and len(assets) > 0:
                chosen.append(random.choice(assets))
            else:
                # in MVP we don't create brand-new non-voucher assets; keep overlap.
                chosen.append(random.choice(assets))
        # dedupe
        return list(dict.fromkeys(chosen))
