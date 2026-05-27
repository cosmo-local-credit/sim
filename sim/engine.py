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
from .core import Event, EventLog, format_inventory, Pool, PoolPolicy, FeeRegistry, LimitRule
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

@dataclass(slots=True)
class ProducerDebtObligation:
    obligation_id: int
    producer_pool_id: str
    lender_pool_id: str
    voucher_id: str
    issued_tick: int
    due_tick: int
    original_voucher_units: float
    remaining_voucher_units: float
    borrowed_usd: float
    debt_kind: str = "stable"
    cash_service_due_usd: float = 0.0
    cash_service_remaining_usd: float = 0.0
    cash_service_arrears_usd: float = 0.0
    cash_service_penalty_remaining_usd: float = 0.0
    pressure_deferred_usd: float = 0.0
    last_pressure_due_tick: int = -1

class SimulationEngine:
    def __init__(self, cfg: ScenarioConfig, seed: int = 1) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        random.seed(seed)
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
        self._sticky_target_by_pool: Dict[Tuple[str, str], str] = {}
        self._sticky_plan_by_pool: Dict[Tuple[str, str, str], RoutePlan] = {}
        self._sticky_failures: Dict[Tuple[str, str, str], int] = {}
        self._affinity_buddies: Dict[str, Set[str]] = {}
        self._noam_edge_cache_tick: int = -1
        self._noam_edge_cache: Dict[Tuple[str, str, str, int], Tuple[bool, float]] = {}
        self._noam_top_pools_active: Dict[str, list[str]] = {}
        self._swap_volume_usd_tick: float = 0.0
        self._swap_volume_usd_by_pool: Dict[str, float] = {}
        self._noam_routing_swaps_tick: int = 0
        self._noam_clearing_swaps_tick: int = 0
        self._noam_routing_volume_usd_tick: float = 0.0
        self._noam_clearing_volume_usd_tick: float = 0.0
        self._noam_routing_fee_usd_tick: float = 0.0
        self._noam_clearing_fee_usd_tick: float = 0.0
        self._noam_routing_stable_fee_usd_tick: float = 0.0
        self._noam_routing_voucher_fee_usd_tick: float = 0.0
        self._noam_clearing_stable_fee_usd_tick: float = 0.0
        self._noam_clearing_voucher_fee_usd_tick: float = 0.0
        self._noam_clearing_cycles_attempted_tick: int = 0
        self._noam_clearing_cycles_executed_tick: int = 0
        self._noam_clearing_cycle_value_usd_tick: float = 0.0
        self._liquidity_tick: int = -1
        self._liquidity_by_asset: Dict[str, float] = {}
        self._liquidity_initialized: bool = False
        self._liquidity_version: int = 0
        self._target_asset_candidate_cache_tick: int = -1
        self._target_asset_candidate_cache: Dict[Tuple[Optional[str], bool], Tuple[str, ...]] = {}
        self._target_asset_weight_cache_tick: int = -1
        self._target_asset_weight_cache: Dict[
            Tuple[Optional[str], bool, float, int],
            Tuple[Tuple[str, ...], np.ndarray],
        ] = {}
        self._utilization_boost: float = 1.0
        self._last_utilization_rate: float = 0.0
        self._loan_phase_by_agent: Dict[str, int] = {}
        self._producer_voucher_assignments: Dict[str, str] = {}
        self._producer_voucher_lender_assignments: Dict[str, Set[str]] = {}
        self._producer_voucher_target_lender_counts: Dict[str, int] = {}
        self._producer_voucher_lender_target_cache: Dict[str, Tuple[str, ...]] = {}
        self._producer_voucher_ids: Set[str] = set()
        self._lender_producer_voucher_count_cache: Dict[str, int] = {}
        self._pending_producer_vouchers: list[str] = []
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
        self._waterfall_bootstrap_remaining: int = max(0, int(cfg.liquidity_mandate_bootstrap_epochs or 0))
        self._fee_pool_cumulative_usd: float = 0.0
        self._fee_clc_cumulative_usd: float = 0.0
        self._fee_pool_cumulative_voucher: float = 0.0
        self._fee_clc_cumulative_voucher: float = 0.0
        self._clc_fee_window_entries: deque[Tuple[int, str, float]] = deque()
        self._clc_fee_window_totals: Dict[str, float] = {}
        self._clc_fee_window_ticks: Optional[int] = None
        self._clc_pool_injected_usd_total: float = 0.0
        self._clc_pool_swapped_out_stable_total: float = 0.0
        self._clc_pool_swapped_out_voucher_total: float = 0.0
        self._mandates_allocated_usd_total: float = 0.0
        self._mandates_distributed_usd_total: float = 0.0
        self._pool_growth_remainder: float = 0.0
        self._sclc_minted_total: float = 0.0
        self._lp_injected_usd_by_pool: Dict[str, float] = {}
        self._lp_returned_usd_by_pool: Dict[str, float] = {}
        self._lp_injected_usd_total: float = 0.0
        self._lp_returned_usd_total: float = 0.0
        self._bond_service_reserve_usd_balance: float = 0.0
        self._bond_service_reserved_usd_tick: float = 0.0
        self._bond_service_reserved_usd_total: float = 0.0
        self._bond_service_paid_from_reserve_usd_tick: float = 0.0
        self._bond_service_paid_from_reserve_usd_total: float = 0.0
        self._bond_service_reserved_by_pool: Dict[str, float] = {}
        self._lp_pending_contribution_tick: Dict[str, int] = {}
        self._stable_onramp_usd_tick: float = 0.0
        self._stable_offramp_usd_tick: float = 0.0
        self._stable_onramp_usd_month: float = 0.0
        self._stable_offramp_usd_month: float = 0.0
        self._historical_stable_backing_usd_tick: float = 0.0
        self._historical_stable_backing_usd_total: float = 0.0
        self._historical_stable_backing_pools_tick: int = 0
        self._historical_stable_backing_pools_total: int = 0
        self._historical_stable_backing_usd_by_role: Dict[str, float] = {}
        self._historical_stable_backing_pools_by_role: Dict[str, int] = {}
        self._stable_excess_sweep_usd_tick: float = 0.0
        self._stable_excess_sweep_usd_total: float = 0.0
        self._stable_excess_sweep_pools_tick: int = 0
        self._stable_excess_sweep_pools_total: int = 0
        self._producer_deposit_stable_usd_tick: float = 0.0
        self._producer_deposit_voucher_usd_tick: float = 0.0
        self._producer_deposit_stable_usd_total: float = 0.0
        self._producer_deposit_voucher_usd_total: float = 0.0
        self._producer_deposit_value_by_voucher: Dict[str, float] = {}
        self._dirty_lender_voucher_limit_assets: Set[str] = set()
        self._route_source_stable_net_flow_value_tick: float = 0.0
        self._route_source_voucher_net_flow_value_tick: float = 0.0
        self._productive_credit_inflow_usd_tick: float = 0.0
        self._productive_credit_inflow_usd_total: float = 0.0
        self._productive_credit_stable_retained_usd_tick: float = 0.0
        self._productive_credit_stable_retained_usd_total: float = 0.0
        self._productive_credit_voucher_deposit_usd_tick: float = 0.0
        self._productive_credit_voucher_deposit_usd_total: float = 0.0
        self._productive_credit_voucher_deposit_cap_clipped_usd_tick: float = 0.0
        self._productive_credit_voucher_deposit_cap_clipped_usd_total: float = 0.0
        self._productive_credit_inflow_usd_by_pool: Dict[str, float] = {}
        self._productive_credit_stable_retained_usd_by_pool: Dict[str, float] = {}
        self._productive_credit_voucher_deposit_usd_by_pool: Dict[str, float] = {}
        self._productive_credit_voucher_deposit_usd_by_pool_tick: Dict[str, float] = {}
        self._productive_credit_voucher_activity_until: Dict[Tuple[str, str], int] = {}
        self._producer_voucher_loan_activity_until: Dict[Tuple[str, str], int] = {}
        self._productive_credit_queue: Dict[int, list[Tuple[str, float, str]]] = {}
        self._producer_debt_obligations: list[ProducerDebtObligation] = []
        self._producer_debt_obligations_by_lender_voucher: Dict[
            Tuple[str, str],
            list[ProducerDebtObligation],
        ] = {}
        self._producer_debt_obligation_index_dirty: Set[Tuple[str, str]] = set()
        self._next_producer_debt_obligation_id: int = 1
        self._producer_debt_originated_usd_tick: float = 0.0
        self._producer_debt_originated_usd_total: float = 0.0
        self._producer_debt_cash_service_due_usd_tick: float = 0.0
        self._producer_debt_cash_service_due_usd_total: float = 0.0
        self._producer_debt_cash_service_paid_usd_tick: float = 0.0
        self._producer_debt_cash_service_paid_usd_total: float = 0.0
        self._producer_debt_matured_usd_tick: float = 0.0
        self._producer_debt_matured_usd_total: float = 0.0
        self._producer_debt_repaid_usd_tick: float = 0.0
        self._producer_debt_repaid_usd_total: float = 0.0
        self._producer_debt_repaid_regular_usd_tick: float = 0.0
        self._producer_debt_repaid_regular_usd_total: float = 0.0
        self._producer_debt_repaid_maturity_usd_tick: float = 0.0
        self._producer_debt_repaid_maturity_usd_total: float = 0.0
        self._producer_debt_stable_recovered_usd_tick: float = 0.0
        self._producer_debt_stable_recovered_usd_total: float = 0.0
        self._producer_debt_consumer_stable_purchase_usd_tick: float = 0.0
        self._producer_debt_consumer_stable_purchase_usd_total: float = 0.0
        self._producer_debt_third_party_stable_purchase_usd_tick: float = 0.0
        self._producer_debt_third_party_stable_purchase_usd_total: float = 0.0
        self._producer_debt_defaulted_usd_tick: float = 0.0
        self._producer_debt_defaulted_usd_total: float = 0.0
        self._producer_debt_closed_by_circulation_usd_tick: float = 0.0
        self._producer_debt_closed_by_circulation_usd_total: float = 0.0
        self._producer_debt_closed_by_voucher_swap_usd_tick: float = 0.0
        self._producer_debt_closed_by_voucher_swap_usd_total: float = 0.0
        self._producer_debt_closed_not_held_at_maturity_usd_tick: float = 0.0
        self._producer_debt_closed_not_held_at_maturity_usd_total: float = 0.0
        self._producer_debt_service_capacity_by_pool: Dict[str, float] = {}
        self._producer_debt_service_capacity_credited_usd_tick: float = 0.0
        self._producer_debt_service_capacity_credited_usd_total: float = 0.0
        self._producer_debt_service_capacity_onramp_usd_tick: float = 0.0
        self._producer_debt_service_capacity_onramp_usd_total: float = 0.0
        self._producer_self_repayment_swap_volume_usd_tick: float = 0.0
        self._producer_self_repayment_swap_volume_usd_total: float = 0.0
        self._producer_self_repayment_voucher_removed_usd_tick: float = 0.0
        self._producer_self_repayment_voucher_removed_usd_total: float = 0.0
        self._producer_debt_pressure_prepayment_usd_tick: float = 0.0
        self._producer_debt_pressure_prepayment_usd_total: float = 0.0
        self._producer_debt_pressure_deferred_usd_tick: float = 0.0
        self._producer_debt_pressure_deferred_usd_total: float = 0.0
        self._producer_debt_pressure_batched_swap_count_tick: int = 0
        self._producer_debt_pressure_batched_swap_count_total: int = 0
        self._producer_debt_pressure_batched_swap_volume_usd_tick: float = 0.0
        self._producer_debt_pressure_batched_swap_volume_usd_total: float = 0.0
        self._producer_debt_attention_pressure_usd_tick: float = 0.0
        self._producer_debt_attention_pressure_usd_total: float = 0.0
        self._producer_debt_attention_suppressed_attempts_tick: int = 0
        self._producer_debt_attention_suppressed_attempts_total: int = 0
        self._producer_debt_attention_suppressed_v2v_attempts_tick: int = 0
        self._producer_debt_attention_suppressed_v2v_attempts_total: int = 0
        self._producer_debt_attention_share_sum_tick: float = 0.0
        self._producer_debt_attention_share_count_tick: int = 0
        self._producer_debt_attention_share_max_tick: float = 0.0
        self._producer_debt_attention_reference_usd_sum_tick: float = 0.0
        self._producer_debt_attention_reference_count_tick: int = 0
        self._producer_bond_assessment_pressure_usd_tick: float = 0.0
        self._producer_bond_assessment_pressure_usd_total: float = 0.0
        self._producer_bond_assessment_sustain_offset_attempts_tick: float = 0.0
        self._producer_bond_assessment_sustain_offset_attempts_total: float = 0.0
        self._producer_bond_assessment_sustain_offset_v2v_attempts_tick: float = 0.0
        self._producer_bond_assessment_sustain_offset_v2v_attempts_total: float = 0.0
        self._producer_bond_assessment_sustain_target_reduction_tick: int = 0
        self._producer_bond_assessment_sustain_target_reduction_total: int = 0
        self._producer_activity_composition_pressure_usd_tick: float = 0.0
        self._producer_activity_composition_pressure_usd_total: float = 0.0
        self._producer_activity_composition_shift_share_sum_tick: float = 0.0
        self._producer_activity_composition_shift_share_count_tick: int = 0
        self._producer_activity_composition_shift_share_max_tick: float = 0.0
        self._producer_activity_composition_reference_usd_sum_tick: float = 0.0
        self._producer_activity_composition_reference_count_tick: int = 0
        self._producer_activity_composition_v2v_weight_removed_tick: float = 0.0
        self._producer_activity_composition_v2v_weight_removed_total: float = 0.0
        self._producer_activity_composition_v2s_weight_added_tick: float = 0.0
        self._producer_activity_composition_v2s_weight_added_total: float = 0.0
        self._producer_activity_composition_shifted_route_attempts_tick: int = 0
        self._producer_activity_composition_shifted_route_attempts_total: int = 0
        self._producer_activity_composition_shifted_v2s_attempts_tick: int = 0
        self._producer_activity_composition_shifted_v2s_attempts_total: int = 0
        self._producer_activity_composition_own_voucher_stable_probability_sum_tick: float = 0.0
        self._producer_activity_composition_own_voucher_stable_probability_count_tick: int = 0
        self._producer_activity_composition_own_voucher_stable_probability_max_tick: float = 0.0
        self._producer_activity_composition_share_cache: Dict[str, tuple[int, float, float, float]] = {}
        self._producer_ordinary_v2v_volume_history: Dict[str, deque[Tuple[int, float]]] = {}
        self._producer_debt_penalty_accrued_usd_tick: float = 0.0
        self._producer_debt_penalty_accrued_usd_total: float = 0.0
        self._producer_debt_penalty_paid_usd_tick: float = 0.0
        self._producer_debt_penalty_paid_usd_total: float = 0.0
        self._producer_loan_attempts_tick: int = 0
        self._producer_loan_no_lender_tick: int = 0
        self._producer_loan_no_inventory_tick: int = 0
        self._producer_loan_zero_amount_tick: int = 0
        self._producer_loan_route_found_tick: int = 0
        self._producer_loan_route_failed_tick: int = 0
        self._producer_loan_backfill_attempts_tick: int = 0
        self._producer_loan_backfill_executed_tick: int = 0
        self._producer_loan_executed_tick: int = 0
        self._producer_loan_execution_failed_tick: int = 0
        self._producer_loan_sampled_usd_tick: float = 0.0
        self._producer_loan_attempted_usd_tick: float = 0.0
        self._producer_loan_executed_usd_tick: float = 0.0
        self._producer_loan_clipped_inventory_usd_tick: float = 0.0
        self._producer_loan_clipped_lender_cap_usd_tick: float = 0.0
        self._producer_loan_clipped_lender_remaining_usd_tick: float = 0.0
        self._producer_loan_clipped_lender_stable_usd_tick: float = 0.0
        self._producer_loan_clipped_combined_lender_usd_tick: float = 0.0
        self._producer_loan_lender_collateral_cap_usd_tick: float = 0.0
        self._producer_loan_lender_remaining_cap_usd_tick: float = 0.0
        self._producer_loan_lender_stable_available_usd_tick: float = 0.0
        self._producer_voucher_loan_attempts_tick: int = 0
        self._producer_voucher_loan_no_target_tick: int = 0
        self._producer_voucher_loan_no_inventory_tick: int = 0
        self._producer_voucher_loan_zero_amount_tick: int = 0
        self._producer_voucher_loan_route_found_tick: int = 0
        self._producer_voucher_loan_route_failed_tick: int = 0
        self._producer_voucher_loan_executed_tick: int = 0
        self._producer_voucher_loan_execution_failed_tick: int = 0
        self._producer_voucher_loan_attempted_usd_tick: float = 0.0
        self._producer_voucher_loan_executed_usd_tick: float = 0.0
        self._producer_voucher_loan_clipped_lender_cap_usd_tick: float = 0.0
        self._producer_voucher_loan_clipped_lender_remaining_usd_tick: float = 0.0
        self._producer_primary_voucher_loan_attempts_tick: int = 0
        self._producer_primary_voucher_loan_attempts_total: int = 0
        self._producer_primary_voucher_loan_executed_tick: int = 0
        self._producer_primary_voucher_loan_executed_total: int = 0
        self._voucher_purchase_attempts_tick: int = 0
        self._voucher_purchase_attempts_total: int = 0
        self._consumer_voucher_purchase_attempts_tick: int = 0
        self._consumer_voucher_purchase_attempts_total: int = 0
        self._consumer_voucher_purchase_success_tick: int = 0
        self._consumer_voucher_purchase_success_total: int = 0
        self._consumer_voucher_purchase_no_stable_tick: int = 0
        self._consumer_voucher_purchase_no_stable_total: int = 0
        self._consumer_voucher_purchase_reserve_protected_tick: int = 0
        self._consumer_voucher_purchase_reserve_protected_total: int = 0
        self._consumer_voucher_purchase_no_route_tick: int = 0
        self._consumer_voucher_purchase_no_route_total: int = 0
        self._consumer_voucher_purchase_no_target_tick: int = 0
        self._consumer_voucher_purchase_no_target_total: int = 0
        self._consumer_voucher_purchase_stable_spent_usd_tick: float = 0.0
        self._consumer_voucher_purchase_stable_spent_usd_total: float = 0.0
        self._consumer_voucher_purchase_voucher_value_acquired_usd_tick: float = 0.0
        self._consumer_voucher_purchase_voucher_value_acquired_usd_total: float = 0.0
        self._third_party_voucher_purchase_attempts_tick: int = 0
        self._third_party_voucher_purchase_attempts_total: int = 0
        self._third_party_voucher_purchase_success_tick: int = 0
        self._third_party_voucher_purchase_success_total: int = 0
        self._third_party_voucher_purchase_no_stable_tick: int = 0
        self._third_party_voucher_purchase_no_stable_total: int = 0
        self._third_party_voucher_purchase_reserve_protected_tick: int = 0
        self._third_party_voucher_purchase_reserve_protected_total: int = 0
        self._third_party_voucher_purchase_no_route_tick: int = 0
        self._third_party_voucher_purchase_no_route_total: int = 0
        self._third_party_voucher_purchase_no_target_tick: int = 0
        self._third_party_voucher_purchase_no_target_total: int = 0
        self._third_party_voucher_purchase_stable_spent_usd_tick: float = 0.0
        self._third_party_voucher_purchase_stable_spent_usd_total: float = 0.0
        self._third_party_voucher_purchase_voucher_value_acquired_usd_tick: float = 0.0
        self._third_party_voucher_purchase_voucher_value_acquired_usd_total: float = 0.0
        self._lender_voucher_purchase_stable_budget_remaining_usd_tick: float = 0.0
        self._lender_voucher_purchase_stable_budget_remaining_by_kind_tick: Dict[str, float] = {}
        self._lender_voucher_purchase_stable_budget_onramp_usd_tick: float = 0.0
        self._lender_voucher_purchase_stable_budget_onramp_usd_total: float = 0.0
        self._consumer_voucher_purchase_stable_budget_onramp_usd_tick: float = 0.0
        self._consumer_voucher_purchase_stable_budget_onramp_usd_total: float = 0.0
        self._third_party_voucher_purchase_stable_budget_onramp_usd_tick: float = 0.0
        self._third_party_voucher_purchase_stable_budget_onramp_usd_total: float = 0.0
        self._producer_stable_exited_usd_tick: float = 0.0
        self._producer_stable_exited_usd_total: float = 0.0
        self._producer_stable_reuse_budget_usd_tick: float = 0.0
        self._producer_stable_reuse_budget_usd_total: float = 0.0
        self._net_redeemed_voucher_usd_tick: float = 0.0
        self._net_redeemed_voucher_usd_total: float = 0.0
        self._voucher_redeemed_to_issuer_usd_tick: float = 0.0
        self._voucher_redeemed_to_issuer_usd_total: float = 0.0
        self._voucher_fee_retained_for_service_usd_tick: float = 0.0
        self._voucher_fee_retained_for_service_usd_total: float = 0.0
        self._voucher_reintroduced_by_deposit_usd_tick: float = 0.0
        self._voucher_reintroduced_by_deposit_usd_total: float = 0.0
        self._voucher_new_issuance_deposit_usd_tick: float = 0.0
        self._voucher_new_issuance_deposit_usd_total: float = 0.0
        self._debt_removal_voucher_redeemed_usd_tick: float = 0.0
        self._debt_removal_voucher_redeemed_usd_total: float = 0.0
        self._route_context_count_tick: Dict[str, int] = {}
        self._route_context_count_total: Dict[str, int] = {}
        self._route_context_volume_usd_tick: Dict[str, float] = {}
        self._route_context_volume_usd_total: Dict[str, float] = {}
        self._route_context_source_stable_count_tick: Dict[str, int] = {}
        self._route_context_source_stable_count_total: Dict[str, int] = {}
        self._route_context_source_stable_volume_usd_tick: Dict[str, float] = {}
        self._route_context_source_stable_volume_usd_total: Dict[str, float] = {}
        self._route_context_source_voucher_count_tick: Dict[str, int] = {}
        self._route_context_source_voucher_count_total: Dict[str, int] = {}
        self._route_context_source_voucher_volume_usd_tick: Dict[str, float] = {}
        self._route_context_source_voucher_volume_usd_total: Dict[str, float] = {}
        self._route_motif_count_tick: Dict[str, int] = {}
        self._route_motif_count_total: Dict[str, int] = {}
        self._route_motif_volume_usd_tick: Dict[str, float] = {}
        self._route_motif_volume_usd_total: Dict[str, float] = {}
        self._route_motif_stable_intermediate_count_tick: int = 0
        self._route_motif_stable_intermediate_count_total: int = 0
        self._route_motif_stable_intermediate_volume_usd_tick: float = 0.0
        self._route_motif_stable_intermediate_volume_usd_total: float = 0.0
        self._ordinary_route_motif_count_tick: Dict[str, int] = {}
        self._ordinary_route_motif_count_total: Dict[str, int] = {}
        self._ordinary_route_motif_volume_usd_tick: Dict[str, float] = {}
        self._ordinary_route_motif_volume_usd_total: Dict[str, float] = {}
        self._market_route_motif_count_tick: Dict[str, int] = {}
        self._market_route_motif_count_total: Dict[str, int] = {}
        self._market_route_motif_volume_usd_tick: Dict[str, float] = {}
        self._market_route_motif_volume_usd_total: Dict[str, float] = {}
        self._repayment_route_motif_count_tick: Dict[str, int] = {}
        self._repayment_route_motif_count_total: Dict[str, int] = {}
        self._repayment_route_motif_volume_usd_tick: Dict[str, float] = {}
        self._repayment_route_motif_volume_usd_total: Dict[str, float] = {}
        self._loan_route_motif_count_tick: Dict[str, int] = {}
        self._loan_route_motif_count_total: Dict[str, int] = {}
        self._loan_route_motif_volume_usd_tick: Dict[str, float] = {}
        self._loan_route_motif_volume_usd_total: Dict[str, float] = {}
        self._productive_boosted_voucher_swap_count_tick: int = 0
        self._productive_boosted_voucher_swap_count_total: int = 0
        self._productive_boosted_voucher_swap_volume_usd_tick: float = 0.0
        self._productive_boosted_voucher_swap_volume_usd_total: float = 0.0
        self._voucher_loan_boosted_voucher_swap_count_tick: int = 0
        self._voucher_loan_boosted_voucher_swap_count_total: int = 0
        self._voucher_loan_boosted_voucher_swap_volume_usd_tick: float = 0.0
        self._voucher_loan_boosted_voucher_swap_volume_usd_total: float = 0.0
        self._ordinary_stable_spend_protected_skip_count_tick: int = 0
        self._ordinary_stable_spend_protected_skip_count_total: int = 0
        self._ordinary_stable_spend_protected_skip_value_usd_tick: float = 0.0
        self._ordinary_stable_spend_protected_skip_value_usd_total: float = 0.0
        self._fee_conversion_attempted_usd_tick: float = 0.0
        self._fee_conversion_success_usd_tick: float = 0.0
        self._fee_conversion_failed_usd_tick: float = 0.0
        self._fee_conversion_attempted_usd_total: float = 0.0
        self._fee_conversion_success_usd_total: float = 0.0
        self._fee_conversion_failed_usd_total: float = 0.0
        self._fee_service_reserved_usd_tick: float = 0.0
        self._fee_service_reserved_usd_total: float = 0.0
        self._fee_service_stable_reserved_usd_tick: float = 0.0
        self._fee_service_stable_reserved_usd_total: float = 0.0
        self._fee_service_converted_voucher_reserved_usd_tick: float = 0.0
        self._fee_service_converted_voucher_reserved_usd_total: float = 0.0
        self._lender_recovered_stable_by_pool: Dict[str, float] = {}
        self._lender_recovered_stable_total_by_pool: Dict[str, float] = {}
        self._lender_recovered_stable_total_by_pool_reason: Dict[Tuple[str, str], float] = {}
        self._lender_producer_voucher_exposure_usd_by_pool_voucher: Dict[Tuple[str, str], float] = {}
        self._bond_recovery_budget_remaining_by_reason_tick: Dict[str, float] = {}
        self._lender_recovered_stable_usd_tick: float = 0.0
        self._lender_recovered_stable_usd_total: float = 0.0
        self._bond_eligible_pool_exposure_recovered_stable_usd_tick: float = 0.0
        self._bond_eligible_pool_exposure_recovered_stable_usd_total: float = 0.0
        self._lender_inventory_turnover_stable_usd_tick: float = 0.0
        self._lender_inventory_turnover_stable_usd_total: float = 0.0
        self._lender_recovered_stable_borrower_self_usd_tick: float = 0.0
        self._lender_recovered_stable_borrower_self_usd_total: float = 0.0
        self._lender_recovered_stable_borrower_regular_usd_tick: float = 0.0
        self._lender_recovered_stable_borrower_regular_usd_total: float = 0.0
        self._lender_recovered_stable_borrower_maturity_usd_tick: float = 0.0
        self._lender_recovered_stable_borrower_maturity_usd_total: float = 0.0
        self._lender_recovered_stable_consumer_purchase_usd_tick: float = 0.0
        self._lender_recovered_stable_consumer_purchase_usd_total: float = 0.0
        self._lender_recovered_stable_external_nonproducer_purchase_usd_tick: float = 0.0
        self._lender_recovered_stable_external_nonproducer_purchase_usd_total: float = 0.0
        self._lender_recovered_stable_other_producer_purchase_usd_tick: float = 0.0
        self._lender_recovered_stable_other_producer_purchase_usd_total: float = 0.0
        self._lender_recovered_stable_third_party_purchase_usd_tick: float = 0.0
        self._lender_recovered_stable_third_party_purchase_usd_total: float = 0.0
        self._lender_recovered_stable_other_usd_tick: float = 0.0
        self._lender_recovered_stable_other_usd_total: float = 0.0
        self._quarterly_clearing_usd_tick: float = 0.0
        self._quarterly_clearing_usd_total: float = 0.0
        self._quarterly_clearing_lender_liquidity_before_tick: float = 0.0
        self._quarterly_clearing_lender_liquidity_after_tick: float = 0.0
        self._route_fixed_requested_tick: int = 0
        self._route_fixed_found_tick: int = 0
        self._route_fixed_failed_tick: int = 0
        self._route_substitution_requested_tick: int = 0
        self._route_substitution_found_tick: int = 0
        self._route_substitution_failed_tick: int = 0
        self._route_repeat_partner_requested_tick: int = 0
        self._route_exploration_requested_tick: int = 0
        self._route_sticky_used_tick: int = 0
        self._route_buddy_direct_used_tick: int = 0
        self._route_new_target_search_tick: int = 0
        self._route_search_fallback_used_tick: int = 0
        self._frontier_relationship_candidates: Dict[str, Set[str]] = {}
        self._frontier_relationship_last_refresh_tick: int = -1
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
        self._noam_distance_cache_graph_version: int = 0
        self._noam_distance_cache: Dict[Tuple[str, int], Dict[str, int]] = {}
        self._noam_graph_version: int = 0
        self._noam_graph_changed_tick: int = -1
        self._noam_near_hubs_cache_tick: int = -1
        self._noam_near_hubs_cache: Dict[Tuple[str, bool], list[str]] = {}
        self._noam_cap_scale_cache_tick: int = -1
        self._noam_cap_scale_cache_value: float = 1.0
        self._non_system_pool_count_cache: Optional[int] = None
        self._recent_swap_counts: list[int] = []

        self.initial_stable_total: float = 0.0
        self._bootstrap()

    def _bootstrap(self) -> None:
        if self.cfg.economics_enabled:
            self._bootstrap_system_pools()
        cfg = self.cfg
        roles: list[str] = []
        if cfg.initial_liquidity_providers is not None:
            roles.extend(["liquidity_provider"] * max(0, int(cfg.initial_liquidity_providers)))
        if cfg.initial_lenders is not None:
            roles.extend(["lender"] * max(0, int(cfg.initial_lenders)))
        if cfg.initial_producers is not None:
            roles.extend(["producer"] * max(0, int(cfg.initial_producers)))
        if cfg.initial_consumers is not None:
            roles.extend(["consumer"] * max(0, int(cfg.initial_consumers)))
        if not roles:
            initial_roles = [
                "lender",
                "liquidity_provider",
                "producer",
                "producer",
                "producer",
                "producer",
                "producer",
                "producer",
                "producer",
                "producer",
            ]
            for idx in range(self.cfg.initial_pools):
                role = initial_roles[idx] if idx < len(initial_roles) else None
                self.add_pool(role=role)
        else:
            for role in roles:
                self.add_pool(role=role, snapshot=False, rebuild_indexes=False)
            self.rebuild_indexes()

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
        if asset.startswith("VCHR:") and pool.policy.role in ("producer", "consumer"):
            agent = self.agents.get(pool.steward_id)
            if agent and asset != agent.voucher_spec.voucher_id:
                spec = self.factory.voucher_specs.get(asset)
                if spec and spec.issuer_id in self.agents:
                    self._redeem_voucher_from_pool(pool, asset, amount, spec.issuer_id)

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
        return v if v > 0.0 else self._default_asset_value(asset_id)

    def _default_asset_value(self, asset_id: str) -> float:
        if asset_id == self.cfg.stable_symbol or asset_id == self.cfg.sclc_symbol:
            return 1.0
        if asset_id.startswith("VCHR:"):
            return max(1e-12, float(self.cfg.voucher_unit_value_usd or 1.0))
        return 1.0

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
                value = self._default_asset_value(asset)
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
                    value = self._default_asset_value(asset)
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
                self._liquidity_version += 1
                self._target_asset_weight_cache = {}

    def _voucher_outstanding_supply(self, voucher_id: str) -> float:
        spec = self.factory.voucher_specs.get(voucher_id)
        if spec is None:
            return 0.0
        agent = self.agents.get(spec.issuer_id)
        if agent is None:
            return 0.0
        return float(agent.issuer.outstanding_supply or 0.0)

    def _lender_voucher_cap(
        self,
        voucher_id: str,
        lender_pool: Optional["Pool"] = None,
        value_override: Optional[float] = None,
    ) -> float:
        deposit_multiple = max(0.0, float(self.cfg.lender_voucher_cap_deposit_multiple or 0.0))
        deposit_value = float(self._producer_deposit_value_by_voucher.get(voucher_id, 0.0))
        if bool(self.cfg.producer_deposits_enabled) and deposit_multiple > 0.0 and deposit_value > 0.0:
            value = value_override
            if value is None or value <= 0.0:
                if lender_pool is not None:
                    value = lender_pool.values.get_value(voucher_id)
                if value is None or value <= 0.0:
                    spec = self.factory.voucher_specs.get(voucher_id)
                    issuer = self.agents.get(spec.issuer_id) if spec else None
                    issuer_pool = self.pools.get(issuer.pool_id) if issuer else None
                    value = issuer_pool.values.get_value(voucher_id) if issuer_pool is not None else self._default_asset_value(voucher_id)
            value = value if value and value > 0.0 else self._default_asset_value(voucher_id)
            return max(0.0, (deposit_value * deposit_multiple) / value)
        fraction = max(0.0, float(self.cfg.lender_voucher_cap_supply_fraction or 0.0))
        if fraction <= 0.0:
            return 0.0
        supply = self._voucher_outstanding_supply(voucher_id)
        if supply <= 0.0:
            return 0.0
        return max(0.0, supply * fraction)

    def _is_producer_voucher(self, asset_id: str) -> bool:
        if not asset_id.startswith("VCHR:"):
            return False
        if asset_id in self._producer_voucher_ids:
            return True
        spec = self.factory.voucher_specs.get(asset_id)
        if spec is None:
            return False
        issuer = self.agents.get(spec.issuer_id)
        is_producer = bool(issuer and issuer.role == "producer")
        if is_producer:
            self._producer_voucher_ids.add(asset_id)
        return is_producer

    def _invalidate_lender_producer_voucher_count(self, pool_id: Optional[str] = None) -> None:
        if pool_id is None:
            self._lender_producer_voucher_count_cache = {}
        else:
            self._lender_producer_voucher_count_cache.pop(pool_id, None)

    def _invalidate_producer_voucher_lender_targets(self, voucher_id: str) -> None:
        self._producer_voucher_lender_target_cache.pop(voucher_id, None)

    def _lender_producer_voucher_count(self, pool: "Pool") -> int:
        cached = self._lender_producer_voucher_count_cache.get(pool.pool_id)
        if cached is not None:
            return cached
        count = 0
        for asset_id, pol in pool.registry.listings.items():
            if not pol.enabled:
                continue
            if self._is_producer_voucher(asset_id):
                count += 1
        self._lender_producer_voucher_count_cache[pool.pool_id] = count
        return count

    def _list_producer_voucher_on_lender(self, lender_pool: "Pool", voucher_id: str) -> bool:
        if lender_pool.registry.is_listed(voucher_id):
            return False
        value = 0.0
        spec = self.factory.voucher_specs.get(voucher_id)
        if spec:
            issuer_pool = self.pools.get(self.agents[spec.issuer_id].pool_id) if spec.issuer_id in self.agents else None
            if issuer_pool is not None:
                value = issuer_pool.values.get_value(voucher_id)
        if value <= 0.0:
            value = self._default_asset_value(voucher_id)
        cap_in = self._lender_voucher_cap(voucher_id, lender_pool=lender_pool, value_override=value)
        lender_pool.list_asset_with_value_and_limit(
            voucher_id,
            value=value,
            window_len=self.cfg.default_window_len,
            cap_in=cap_in,
        )
        self._producer_voucher_ids.add(voucher_id)
        self._invalidate_lender_producer_voucher_count(lender_pool.pool_id)
        self._invalidate_producer_voucher_lender_targets(voucher_id)
        return True

    def _producer_voucher_overlap_enabled(self) -> bool:
        mode = str(getattr(self.cfg, "producer_voucher_overlap_mode", "single_lender") or "single_lender")
        return mode == "empirical_overlap"

    def _bucketed_producer_voucher_lender_count(self, lender_count: int) -> int:
        if lender_count <= 0:
            return 0
        if not self._producer_voucher_overlap_enabled():
            return 1 if bool(self.cfg.producer_voucher_single_lender) else lender_count
        weights = getattr(self.cfg, "producer_voucher_overlap_bucket_weights", {}) or {}
        clean_weights = {
            str(bucket): max(0.0, float(weight))
            for bucket, weight in dict(weights).items()
            if max(0.0, float(weight)) > 0.0
        }
        if not clean_weights:
            return 1
        total = sum(clean_weights.values())
        draw = self.rng.random() * total
        running = 0.0
        selected = "1"
        for bucket, weight in sorted(clean_weights.items()):
            running += weight
            if draw <= running:
                selected = bucket
                break
        if "-" in selected:
            low_text, high_text = selected.split("-", 1)
            low = max(1, int(float(low_text)))
            high = max(low, int(float(high_text)))
            target = self.rng.randint(low, high)
        elif selected.endswith("+"):
            low = max(1, int(float(selected[:-1])))
            high = max(low, lender_count)
            target = self.rng.randint(low, high)
        else:
            target = max(1, int(float(selected)))
        max_acceptors = getattr(self.cfg, "producer_voucher_overlap_max_lender_acceptors", None)
        if max_acceptors is not None:
            target = min(target, max(1, int(max_acceptors)))
        return max(1, min(int(target), lender_count))

    def _producer_voucher_overlap_diagnostics(self) -> dict[str, float]:
        counts = [
            len(assignments)
            for assignments in self._producer_voucher_lender_assignments.values()
            if assignments
        ]
        if not counts:
            return {
                "producer_voucher_overlap_mode_empirical": int(self._producer_voucher_overlap_enabled()),
                "producer_voucher_overlap_tokens": 0,
                "producer_voucher_multi_lender_tokens": 0,
                "producer_voucher_multi_lender_share": 0.0,
                "producer_voucher_lender_degree_mean": 0.0,
                "producer_voucher_lender_degree_p50": 0.0,
                "producer_voucher_lender_degree_p90": 0.0,
                "producer_voucher_lender_degree_max": 0,
                "producer_voucher_shared_lender_edges": 0,
            }
        clean = sorted(float(count) for count in counts)

        def q(prob: float) -> float:
            if len(clean) == 1:
                return clean[0]
            pos = (len(clean) - 1) * max(0.0, min(1.0, prob))
            lower = math.floor(pos)
            upper = math.ceil(pos)
            if lower == upper:
                return clean[int(pos)]
            return clean[lower] + (clean[upper] - clean[lower]) * (pos - lower)

        edge_count = 0
        for assignments in self._producer_voucher_lender_assignments.values():
            n = len(assignments)
            if n > 1:
                edge_count += n * (n - 1) // 2
        multi = sum(1 for count in counts if count > 1)
        return {
            "producer_voucher_overlap_mode_empirical": int(self._producer_voucher_overlap_enabled()),
            "producer_voucher_overlap_tokens": len(counts),
            "producer_voucher_multi_lender_tokens": multi,
            "producer_voucher_multi_lender_share": multi / max(1, len(counts)),
            "producer_voucher_lender_degree_mean": sum(counts) / max(1, len(counts)),
            "producer_voucher_lender_degree_p50": q(0.50),
            "producer_voucher_lender_degree_p90": q(0.90),
            "producer_voucher_lender_degree_max": max(counts),
            "producer_voucher_shared_lender_edges": edge_count,
        }

    def _assign_producer_voucher_to_lender(self, voucher_id: str) -> None:
        stored_target = self._producer_voucher_target_lender_counts.get(voucher_id)
        stored_assignments = self._producer_voucher_lender_assignments.get(voucher_id)
        if (
            self._frontier_shortlist_enabled()
            and stored_target is not None
            and stored_target > 0
            and stored_assignments
        ):
            acceptors = self.accept_index.get(voucher_id, set())
            if len(stored_assignments) == int(stored_target) and all(
                (lender_id in self.pools)
                and self.pools[lender_id].policy.role == "lender"
                and self.pools[lender_id].registry.is_listed(voucher_id)
                and lender_id in acceptors
                for lender_id in stored_assignments
            ):
                return
        lenders = [
            p for p in self.pools.values()
            if not p.policy.system_pool and p.policy.role == "lender"
        ]
        if not lenders:
            if voucher_id not in self._pending_producer_vouchers:
                self._pending_producer_vouchers.append(voucher_id)
            return
        current = {
            lender_id
            for lender_id in self._producer_voucher_lender_assignments.get(voucher_id, set())
            if lender_id in self.pools and self.pools[lender_id].policy.role == "lender"
        }
        target_count = self._producer_voucher_target_lender_counts.get(voucher_id)
        if target_count is None or target_count <= 0:
            target_count = self._bucketed_producer_voucher_lender_count(len(lenders))
            self._producer_voucher_target_lender_counts[voucher_id] = target_count
        target_count = max(1, min(int(target_count), len(lenders)))
        if self._frontier_shortlist_enabled() and len(current) >= target_count:
            assigned = sorted(current)[:target_count]
            acceptors = self.accept_index.get(voucher_id, set())
            if (
                len(current) == target_count
                and self._producer_voucher_lender_assignments.get(voucher_id) == set(current)
                and all(
                    (lender_id in self.pools)
                    and self.pools[lender_id].registry.is_listed(voucher_id)
                    and lender_id in acceptors
                    for lender_id in assigned
                )
            ):
                return
        if len(current) < target_count:
            stable_id = self.cfg.stable_symbol
            available = [p for p in lenders if p.pool_id not in current]
            load_counts = {p.pool_id: self._lender_producer_voucher_count(p) for p in lenders}
            available.sort(
                key=lambda p: (
                    load_counts.get(p.pool_id, 0),
                    -(p.vault.get(stable_id) - p.policy.min_stable_reserve),
                    p.pool_id,
                )
            )
            for pool in available[: max(0, target_count - len(current))]:
                current.add(pool.pool_id)
        if not current:
            return
        self._producer_voucher_lender_assignments[voucher_id] = set(current)
        self._invalidate_producer_voucher_lender_targets(voucher_id)
        primary_id = sorted(
            current,
            key=lambda pid: (
                self._lender_producer_voucher_count(self.pools[pid]) if pid in self.pools else 0,
                pid,
            ),
        )[0]
        self._producer_voucher_assignments[voucher_id] = primary_id
        listed_any = False
        for lender_id in sorted(current):
            assigned_pool = self.pools.get(lender_id)
            if assigned_pool is None:
                continue
            listed = self._list_producer_voucher_on_lender(assigned_pool, voucher_id)
            listed_any = listed_any or listed
            if self._is_routable_pool(assigned_pool):
                self.accept_index.setdefault(voucher_id, set()).add(assigned_pool.pool_id)
        if listed_any:
            self._refresh_lender_voucher_limits({voucher_id})
        if bool(self.cfg.producer_voucher_single_lender) and not self._producer_voucher_overlap_enabled():
            for lender in lenders:
                if lender.pool_id in current:
                    continue
                pol = lender.registry.listings.get(voucher_id)
                if pol and pol.enabled:
                    pol.enabled = False
                    self._invalidate_lender_producer_voucher_count(lender.pool_id)
                    self._invalidate_producer_voucher_lender_targets(voucher_id)
                    pools = self.accept_index.get(voucher_id)
                    if pools is not None:
                        pools.discard(lender.pool_id)
                        if not pools:
                            self.accept_index.pop(voucher_id, None)

    def _assign_pending_producer_vouchers(self) -> None:
        if not self._pending_producer_vouchers:
            return
        pending = list(self._pending_producer_vouchers)
        self._pending_producer_vouchers.clear()
        for voucher_id in pending:
            self._assign_producer_voucher_to_lender(voucher_id)

    def _refresh_lender_voucher_limits(self, voucher_ids: Optional[Set[str]] = None) -> None:
        target_vouchers = set(voucher_ids) if voucher_ids is not None else None
        if not self.cfg.swap_limits_enabled:
            if target_vouchers is None:
                self._dirty_lender_voucher_limit_assets.clear()
            else:
                self._dirty_lender_voucher_limit_assets.difference_update(target_vouchers)
            return

        def refresh_rule(p: "Pool", asset_id: str) -> None:
            cap_in = self._lender_voucher_cap(asset_id, lender_pool=p)
            rule = p.limiter.rules.get(asset_id)
            window_len = rule.window_len_ticks if rule else int(self.cfg.default_window_len or 1)
            p.limiter.set_rule(asset_id, LimitRule(window_len_ticks=window_len, cap_in_global=cap_in))

        if target_vouchers is not None:
            for p in self.pools.values():
                if p.policy.system_pool or p.policy.role != "lender":
                    continue
                for asset_id in target_vouchers:
                    pol = p.registry.listings.get(asset_id)
                    if pol and pol.enabled and asset_id.startswith("VCHR:"):
                        refresh_rule(p, asset_id)
            self._dirty_lender_voucher_limit_assets.difference_update(target_vouchers)
            return

        for p in self.pools.values():
            if p.policy.system_pool or p.policy.role != "lender":
                continue
            for asset_id, pol in p.registry.listings.items():
                if not pol.enabled:
                    continue
                if not asset_id.startswith("VCHR:"):
                    continue
                refresh_rule(p, asset_id)
        self._dirty_lender_voucher_limit_assets.clear()

    def _mark_lender_voucher_limits_dirty(self, voucher_id: str) -> None:
        if voucher_id.startswith("VCHR:"):
            self._dirty_lender_voucher_limit_assets.add(voucher_id)

    def _refresh_dirty_lender_voucher_limits(self) -> None:
        if self._dirty_lender_voucher_limit_assets:
            self._refresh_lender_voucher_limits(set(self._dirty_lender_voucher_limit_assets))

    def _producer_voucher_lender_deposit_targets(self, voucher_id: str) -> list["Pool"]:
        cached_ids = (
            self._producer_voucher_lender_target_cache.get(voucher_id)
            if self._frontier_shortlist_enabled()
            else None
        )
        if cached_ids is not None:
            cached_targets: list["Pool"] = []
            cache_valid = True
            for lender_id in cached_ids:
                pool = self.pools.get(lender_id)
                if (
                    pool is None
                    or pool.policy.system_pool
                    or pool.policy.role != "lender"
                    or not pool.registry.is_listed(voucher_id)
                ):
                    cache_valid = False
                    break
                cached_targets.append(pool)
            if cache_valid:
                return cached_targets
            self._producer_voucher_lender_target_cache.pop(voucher_id, None)

        self._assign_producer_voucher_to_lender(voucher_id)
        lender_ids = sorted(self._producer_voucher_lender_assignments.get(voucher_id, set()))
        targets: list["Pool"] = []
        for lender_id in lender_ids:
            pool = self.pools.get(lender_id)
            if pool is None or pool.policy.system_pool or pool.policy.role != "lender":
                continue
            if not pool.registry.is_listed(voucher_id):
                self._list_producer_voucher_on_lender(pool, voucher_id)
            targets.append(pool)
        if self._frontier_shortlist_enabled() and targets:
            self._producer_voucher_lender_target_cache[voucher_id] = tuple(pool.pool_id for pool in targets)
        return targets

    def _deposit_producer_voucher_with_lenders(
        self,
        *,
        producer_pool: "Pool",
        agent: "Agent",
        voucher_id: str,
        voucher_value_usd: float,
        source: str,
    ) -> bool:
        voucher_value_usd = max(0.0, float(voucher_value_usd))
        if voucher_value_usd <= 1e-9:
            return False
        value = self._asset_value(producer_pool, voucher_id)
        if value <= 1e-12:
            return False
        targets = self._producer_voucher_lender_deposit_targets(voucher_id)
        if not targets:
            targets = [producer_pool]
            if not producer_pool.registry.is_listed(voucher_id):
                producer_pool.list_asset_with_value_and_limit(
                    voucher_id,
                    value=value,
                    window_len=self.cfg.default_window_len,
                    cap_in=self.cfg.producer_voucher_cap_in,
                )

        total_units = voucher_value_usd / value
        transferred_units = 0.0
        if all(target.pool_id != producer_pool.pool_id for target in targets):
            available_units = max(0.0, producer_pool.vault.get(voucher_id))
            transferred_units = min(available_units, total_units)
            if transferred_units > 1e-9:
                self._vault_sub(
                    producer_pool,
                    voucher_id,
                    transferred_units,
                    f"{source}_transfer_out",
                    agent.agent_id,
                )
            issued_units = total_units - transferred_units
        else:
            issued_units = total_units
        if issued_units > 1e-9:
            agent.issuer.issue(issued_units)
            self._mark_lender_voucher_limits_dirty(voucher_id)
        reintroduced_usd = transferred_units * value
        new_issuance_usd = issued_units * value
        if reintroduced_usd > 1e-9:
            self._voucher_reintroduced_by_deposit_usd_tick += reintroduced_usd
            self._voucher_reintroduced_by_deposit_usd_total += reintroduced_usd
        if new_issuance_usd > 1e-9:
            self._voucher_new_issuance_deposit_usd_tick += new_issuance_usd
            self._voucher_new_issuance_deposit_usd_total += new_issuance_usd
        units_per_target = total_units / float(len(targets))
        value_per_target = voucher_value_usd / float(len(targets))
        for target in targets:
            self._vault_add(target, voucher_id, units_per_target, source, agent.agent_id)
            if target.policy.role == "lender":
                self._increase_lender_producer_voucher_exposure(
                    target.pool_id,
                    voucher_id,
                    value_per_target,
                    source,
                )
            self.log.add(Event(
                self.tick,
                "PRODUCER_VOUCHER_DEPOSIT",
                actor_id=agent.agent_id,
                pool_id=target.pool_id,
                asset_id=voucher_id,
                amount=value_per_target,
                meta={
                    "source": source,
                    "producer_pool_id": producer_pool.pool_id,
                    "voucher_units": units_per_target,
                },
            ))
        self._record_producer_deposit_value(voucher_id, voucher_value_usd)
        self._producer_deposit_voucher_usd_tick += voucher_value_usd
        self._producer_deposit_voucher_usd_total += voucher_value_usd
        return True

    def _deposit_producer_stable_with_lenders(
        self,
        *,
        producer_pool: "Pool",
        agent: "Agent",
        voucher_id: str,
        stable_value_usd: float,
        source: str,
    ) -> bool:
        stable_value_usd = max(0.0, float(stable_value_usd))
        if stable_value_usd <= 1e-9:
            return False
        stable_id = self.cfg.stable_symbol
        stable_value = self._asset_value(producer_pool, stable_id)
        if stable_value <= 1e-12:
            stable_value = 1.0
        stable_units = stable_value_usd / stable_value
        targets = self._producer_voucher_lender_deposit_targets(voucher_id)
        if not targets:
            targets = [producer_pool]

        if all(target.pool_id != producer_pool.pool_id for target in targets):
            available_units = max(0.0, producer_pool.vault.get(stable_id))
            transferred_units = min(available_units, stable_units)
            if transferred_units > 1e-9:
                self._vault_sub(
                    producer_pool,
                    stable_id,
                    transferred_units,
                    f"{source}_transfer_out",
                    agent.agent_id,
                )
            onramp_units = stable_units - transferred_units
        else:
            onramp_units = stable_units
        if onramp_units > 1e-9:
            self._stable_onramp_usd_tick += onramp_units * stable_value

        units_per_target = stable_units / float(len(targets))
        value_per_target = stable_value_usd / float(len(targets))
        for target in targets:
            self._vault_add(target, stable_id, units_per_target, source, agent.agent_id)
            self.log.add(Event(
                self.tick,
                "PRODUCER_STABLE_DEPOSIT",
                actor_id=agent.agent_id,
                pool_id=target.pool_id,
                asset_id=stable_id,
                amount=value_per_target,
                meta={
                    "source": source,
                    "producer_pool_id": producer_pool.pool_id,
                },
            ))
        self._record_producer_deposit_value(voucher_id, stable_value_usd)
        self._producer_deposit_stable_usd_tick += stable_value_usd
        self._producer_deposit_stable_usd_total += stable_value_usd
        return True

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
        same_tick_cache = self._noam_distance_cache_tick == self.tick
        if (
            self._noam_distance_cache_graph_version != self._noam_graph_version
            and not same_tick_cache
        ):
            self._noam_distance_cache = {}
            self._noam_distance_cache_graph_version = self._noam_graph_version
        if self._noam_distance_cache_tick != self.tick:
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
        pool_whitelist: Optional[Set[str]],
    ) -> Optional[Tuple[str, str, str, int]]:
        ttl = int(self.cfg.noam_route_cache_ttl_ticks or 0)
        bucket = float(self.cfg.noam_route_cache_bucket_usd or 0.0)
        if ttl <= 0 or bucket <= 0.0 or target_pools is not None or pool_whitelist is not None:
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

    def _noam_edge_cache_reset(self) -> None:
        if self._noam_edge_cache_tick != self.tick:
            self._noam_edge_cache_tick = self.tick
            self._noam_edge_cache = {}
            active: Dict[str, list[str]] = {}
            for asset_id, pool_ids in self._noam_top_pools.items():
                filtered: list[str] = []
                for pid in pool_ids:
                    pool = self.pools.get(pid)
                    if pool is None:
                        continue
                    if not self._is_routable_pool(pool):
                        continue
                    if pool.policy.paused:
                        continue
                    if pool.policy.mode in ("none", "borrow_only"):
                        continue
                    filtered.append(pid)
                if filtered:
                    active[asset_id] = filtered
            self._noam_top_pools_active = active

    def _noam_edge_cache_key(
        self,
        pool: "Pool",
        asset_in: str,
        asset_out: str,
        amount_in: float,
        cache: Optional[NoamRouteCache],
    ) -> Tuple[str, str, str, int]:
        bucket = float(self.cfg.noam_route_cache_bucket_usd or 0.0)
        if bucket <= 0.0:
            bucket = 1.0
        value_in = self._noam_cached_value(pool, asset_in, cache)
        if value_in <= 0.0:
            value_in = 1.0
        amount_usd = amount_in * value_in
        bucket_id = int(amount_usd / bucket)
        return (pool.pool_id, asset_in, asset_out, bucket_id)

    def _validate_route_plan(self, plan: RoutePlan, amount_in: float, source_pool: "Pool") -> bool:
        if not plan.hops:
            return False
        current_amount = amount_in
        current_asset = plan.hops[0].asset_in
        for hop in plan.hops:
            if hop.asset_in != current_asset:
                return False
            pool = self.pools.get(hop.pool_id)
            if pool is None or pool.policy.paused:
                return False
            if not self._is_routable_pool(pool):
                return False
            okq, reasonq, amount_out, fee_amt = pool.quote_swap(current_asset, current_amount, hop.asset_out)
            if not okq or amount_out <= 1e-9:
                return False
            gross_out = amount_out + fee_amt
            ok, _ = pool.can_swap(self.tick, current_asset, current_amount, hop.asset_out, gross_out)
            if not ok:
                return False
            current_asset = hop.asset_out
            current_amount = amount_out
        return True

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
        enabled_listings_by_pool: Dict[str, Set[str]] = {
            pid: {
                asset_id
                for asset_id, pol in pool.registry.listings.items()
                if pol.enabled
            }
            for pid, pool in self.pools.items()
        }

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
                if not self._is_routable_pool(pool):
                    continue
                scored.append((self._noam_pool_score(pool), pid))
            scored.sort(reverse=True)
            top_pools[asset_id] = [pid for _, pid in scored[:top_k]]

        if self.clc_pool_id and self.clc_pool_id in self.pools:
            clc_pool = self.pools[self.clc_pool_id]
            if self._is_routable_pool(clc_pool):
                clc_enabled = enabled_listings_by_pool.get(self.clc_pool_id, set())
                for asset_id in assets:
                    if asset_id not in clc_enabled:
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

        # Ensure assigned lenders for each producer voucher are included in the NOAM working set.
        for voucher_id, lender_ids in self._producer_voucher_lender_assignments.items():
            for lender_id in sorted(lender_ids):
                lender_pool = self.pools.get(lender_id)
                if lender_pool is None:
                    continue
                if voucher_id not in enabled_listings_by_pool.get(lender_id, set()):
                    continue
                pools = top_pools.get(voucher_id, [])
                if lender_id in pools:
                    continue
                if not pools:
                    top_pools[voucher_id] = [lender_id]
                    continue
                pools = list(pools)
                if len(pools) >= top_k:
                    pools[-1] = lender_id
                else:
                    pools.append(lender_id)
                top_pools[voucher_id] = pools

        # Always include lenders in routing/clearing for assets they list.
        lender_ids = [
            pid for pid, p in self.pools.items()
            if not p.policy.system_pool and p.policy.role == "lender"
        ]
        if lender_ids:
            for asset_id in assets:
                pools = top_pools.get(asset_id, [])
                if pools is None:
                    pools = []
                for pid in lender_ids:
                    pool = self.pools.get(pid)
                    if pool is None:
                        continue
                    if asset_id not in enabled_listings_by_pool.get(pid, set()):
                        continue
                    if pid not in pools:
                        pools.append(pid)
                if pools:
                    top_pools[asset_id] = pools

        top_out: Dict[Tuple[str, str], list[str]] = {}
        for asset_in, pool_ids in top_pools.items():
            for pid in pool_ids:
                pool = self.pools.get(pid)
                if pool is None:
                    continue
                if not self._is_routable_pool(pool):
                    continue
                candidates = []
                for asset_out, amt in pool.vault.inventory.items():
                    if amt <= 1e-9:
                        continue
                    if asset_out == asset_in:
                        continue
                    if asset_out != self.cfg.stable_symbol and not asset_out.startswith("VCHR:"):
                        continue
                    if asset_out not in enabled_listings_by_pool.get(pid, set()):
                        continue
                    value = pool.values.get_value(asset_out)
                    if value <= 0.0:
                        value = self._default_asset_value(asset_out)
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
            if not self._is_routable_pool(pool):
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
        if adj_forward != self._noam_adj_forward or adj_reverse != self._noam_adj_reverse:
            self._noam_graph_version += 1
            self._noam_graph_changed_tick = self.tick
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
        safety = float(self.cfg.noam_clearing_safety_factor or 1.0)
        if safety <= 0.0:
            safety = 1.0
        safety = min(1.0, safety)
        p_min = float(self.cfg.noam_success_min or 1e-6)
        p_max = float(self.cfg.noam_success_max or 1.0)
        w_p = float(self.cfg.noam_weight_success or 0.0)
        w_f = float(self.cfg.noam_weight_fee or 0.0)
        w_l = float(self.cfg.noam_weight_lambda or 0.0)
        w_b = float(self.cfg.noam_weight_benefit or 0.0)
        w_d = float(self.cfg.noam_weight_deadend or 0.0)
        lender_bonus = float(self.cfg.noam_clearing_lender_edge_bonus or 0.0)

        edges_by_asset: Dict[str, list[NoamClearingEdge]] = {}
        nominal_value = max(1.0, min_value)
        restrict_lenders = bool(self.cfg.noam_clearing_lenders_only)
        include_clc = bool(self.cfg.noam_clearing_include_clc)

        for (pid, asset_in), outs in self._noam_top_out.items():
            pool = self.pools.get(pid)
            if pool is None:
                continue
            if not self._is_routable_pool(pool):
                continue
            if restrict_lenders:
                if pool.policy.role != "lender":
                    if not (include_clc and self.clc_pool_id and pid == self.clc_pool_id):
                        continue
            if pool.policy.paused or pool.policy.mode in ("none", "borrow_only"):
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
                remaining = float("inf")
                if pool.policy.limits_enabled:
                    remaining = pool.limiter.remaining(self.tick, asset_in)
                inventory_out = pool.vault.get(asset_out)
                if asset_out == self.cfg.stable_symbol:
                    inventory_out = max(0.0, inventory_out - pool.policy.min_stable_reserve)
                cap_amount_in = min(remaining, (inventory_out * value_out) / max(1e-9, value_in))
                cap_value = cap_amount_in * value_in
                cap_value *= safety
                if cap_value <= min_value + 1e-9:
                    continue
                amount_in_check = cap_value / value_in
                if amount_in_check <= 1e-12:
                    continue
                if not self._noam_edge_allowed(pool, asset_in, asset_out, amount_in_check):
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
                elif pool.policy.role == "lender":
                    score += lender_bonus
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
                edges.sort(key=lambda e: (e.score * e.cap_value), reverse=True)
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
            if not self._is_routable_pool(pool):
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
            if not self._is_routable_pool(pool):
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
            self._record_clc_swap_cumulative(receipt)
            if receipt.asset_in.startswith("VCHR:") and pool.policy.role in ("consumer", "producer"):
                spec = self.factory.voucher_specs.get(receipt.asset_in)
                if spec and spec.issuer_id != pool.steward_id:
                    self._redeem_voucher_from_pool(pool, receipt.asset_in, float(receipt.amount_in), spec.issuer_id)
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
            self._record_noam_fee_diagnostics(
                pool,
                receipt,
                kind="clearing",
                swap_usd=swap_usd,
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
        budget_scale = float(stride) if bool(self.cfg.noam_clearing_budget_scale_by_stride) else 1.0
        budget = float(self.cfg.noam_clearing_budget_usd or 0.0) * budget_scale
        budget_share = float(self.cfg.noam_clearing_budget_share or 0.0) * budget_scale
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
            self._noam_clearing_cycles_attempted_tick += int(attempted)
            self._noam_clearing_cycles_executed_tick += int(executed)
            self._noam_clearing_cycle_value_usd_tick += float(total_value)
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
        if not self._is_routable_pool(pool):
            return False
        if not pool.registry.is_listed(asset_in) or not pool.registry.is_listed(asset_out):
            return False
        if self._noam_cached_inventory(pool, asset_out, cache) <= 1e-9:
            return False
        if pool.policy.role == "lender":
            direct_voucher_to_voucher = (
                asset_in != self.cfg.stable_symbol
                and asset_out != self.cfg.stable_symbol
                and bool(getattr(self.cfg, "open_pool_direct_voucher_to_voucher_enabled", False))
            )
            if (
                asset_in != self.cfg.stable_symbol
                and asset_out != self.cfg.stable_symbol
                and not direct_voucher_to_voucher
            ):
                return False
        if (
            pool.policy.role in ("consumer", "producer")
            and asset_out == self.cfg.stable_symbol
            and max(0.0, float(self.cfg.producer_consumer_stable_target_bias or 0.0)) <= 0.0
        ):
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
        multiplier = 1.0
        if asset_in == self.cfg.stable_symbol:
            multiplier = max(0.0, float(self.cfg.stable_source_swap_size_multiplier or 0.0))
        elif asset_in.startswith("VCHR:"):
            multiplier = max(0.0, float(self.cfg.voucher_source_swap_size_multiplier or 0.0))
            multiplier *= self._productive_credit_voucher_source_size_boost(pool, asset_in)
        mean_usd = total_value * float(self.cfg.swap_size_mean_frac or 0.0) * multiplier
        if mean_usd <= 0.0:
            return 0.0
        amount_usd = float(np.random.exponential(mean_usd))
        amount_usd = max(amount_usd, float(self.cfg.swap_size_min_usd or 0.0))
        if self.cfg.swap_size_max_usd is not None:
            amount_usd = min(amount_usd, float(self.cfg.swap_size_max_usd))
        value_in = self._asset_value(pool, asset_in)
        amount_in = amount_usd / value_in
        return min(amount_in, self._ordinary_source_spendable_amount(pool, asset_in))

    def _ordinary_source_stable_preserve_units(self, pool: "Pool") -> float:
        cfg = self.cfg
        if not bool(cfg.ordinary_stable_spend_protection_enabled):
            return 0.0
        if pool.policy.role not in ("producer", "consumer"):
            return 0.0
        reserve = max(0.0, float(pool.policy.min_stable_reserve or 0.0))
        buffer_share = max(0.0, float(cfg.ordinary_stable_spend_buffer_voucher_share or 0.0))
        voucher_buffer_usd = self._pool_voucher_value_usd(pool) * buffer_share
        stable_value = self._asset_value(pool, cfg.stable_symbol)
        if stable_value <= 1e-12:
            stable_value = 1.0
        return reserve + voucher_buffer_usd / stable_value

    def _ordinary_source_spendable_amount(self, pool: "Pool", asset_in: str) -> float:
        available = max(0.0, pool.vault.get(asset_in))
        if asset_in != self.cfg.stable_symbol:
            return available
        preserve = self._ordinary_source_stable_preserve_units(pool)
        if preserve <= 0.0:
            return available
        return max(0.0, available - preserve)

    def _source_asset_selection_weights(self, pool: "Pool", assets: list[str]) -> np.ndarray:
        weights = np.array(
            [
                self._ordinary_source_spendable_amount(pool, asset) * self._asset_value(pool, asset)
                for asset in assets
            ],
            dtype=float,
        )
        stable_bias_config: float | None = None
        if pool.policy.role == "consumer":
            stable_bias_config = self.cfg.consumer_stable_source_bias
        elif pool.policy.role == "producer":
            stable_bias_config = self.cfg.producer_stable_source_bias
        if stable_bias_config is not None:
            stable_bias = max(0.0, float(stable_bias_config or 0.0))
            for idx, asset in enumerate(assets):
                if asset == self.cfg.stable_symbol:
                    weights[idx] *= stable_bias
        if bool(self.cfg.productive_credit_voucher_activity_boost_enabled):
            boost = max(0.0, float(self.cfg.productive_credit_voucher_source_weight_boost or 0.0))
            if boost > 0.0:
                for idx, asset in enumerate(assets):
                    if asset.startswith("VCHR:") and self._voucher_source_activity_active(pool, asset):
                        weights[idx] *= 1.0 + boost
        return weights

    @staticmethod
    def _increment_float_map(store: Dict[str, float], key: str, amount: float) -> None:
        store[key] = store.get(key, 0.0) + float(amount)

    @staticmethod
    def _increment_int_map(store: Dict[str, int], key: str, amount: int = 1) -> None:
        store[key] = store.get(key, 0) + int(amount)

    def _record_ordinary_stable_spend_protection_skip(self, pool: "Pool") -> None:
        if pool.policy.role not in ("producer", "consumer"):
            return
        stable_id = self.cfg.stable_symbol
        available = max(0.0, pool.vault.get(stable_id))
        if available <= 1e-9:
            return
        preserve = self._ordinary_source_stable_preserve_units(pool)
        if preserve <= 1e-9 or available > preserve + 1e-9:
            return
        blocked_usd = available * max(0.0, self._asset_value(pool, stable_id))
        if blocked_usd <= 1e-9:
            return
        self._ordinary_stable_spend_protected_skip_count_tick += 1
        self._ordinary_stable_spend_protected_skip_count_total += 1
        self._ordinary_stable_spend_protected_skip_value_usd_tick += blocked_usd
        self._ordinary_stable_spend_protected_skip_value_usd_total += blocked_usd

    def _record_route_context_swap(
        self,
        route_context: str,
        source_pool: "Pool",
        source_asset: str,
        swap_usd: float,
    ) -> None:
        context = str(route_context or "ordinary")
        self._increment_int_map(self._route_context_count_tick, context)
        self._increment_int_map(self._route_context_count_total, context)
        self._increment_float_map(self._route_context_volume_usd_tick, context, swap_usd)
        self._increment_float_map(self._route_context_volume_usd_total, context, swap_usd)
        if source_asset == self.cfg.stable_symbol:
            self._increment_int_map(self._route_context_source_stable_count_tick, context)
            self._increment_int_map(self._route_context_source_stable_count_total, context)
            self._increment_float_map(
                self._route_context_source_stable_volume_usd_tick, context, swap_usd
            )
            self._increment_float_map(
                self._route_context_source_stable_volume_usd_total, context, swap_usd
            )
        elif source_asset.startswith("VCHR:"):
            self._increment_int_map(self._route_context_source_voucher_count_tick, context)
            self._increment_int_map(self._route_context_source_voucher_count_total, context)
            self._increment_float_map(
                self._route_context_source_voucher_volume_usd_tick, context, swap_usd
            )
            self._increment_float_map(
                self._route_context_source_voucher_volume_usd_total, context, swap_usd
            )
            if context == "ordinary" and self._productive_credit_voucher_activity_active(
                source_pool, source_asset
            ):
                self._productive_boosted_voucher_swap_count_tick += 1
                self._productive_boosted_voucher_swap_count_total += 1
                self._productive_boosted_voucher_swap_volume_usd_tick += swap_usd
                self._productive_boosted_voucher_swap_volume_usd_total += swap_usd
            if context == "ordinary" and self._producer_voucher_loan_activity_active(
                source_pool, source_asset
            ):
                self._voucher_loan_boosted_voucher_swap_count_tick += 1
                self._voucher_loan_boosted_voucher_swap_count_total += 1
                self._voucher_loan_boosted_voucher_swap_volume_usd_tick += swap_usd
                self._voucher_loan_boosted_voucher_swap_volume_usd_total += swap_usd

    def _productive_credit_voucher_activity_active(self, pool: "Pool", voucher_id: str) -> bool:
        if not bool(self.cfg.productive_credit_voucher_activity_boost_enabled):
            return False
        if pool.policy.role != "producer":
            return False
        until = self._productive_credit_voucher_activity_until.get((pool.pool_id, voucher_id), -1)
        return until >= self.tick

    def _producer_voucher_loan_activity_active(self, pool: "Pool", voucher_id: str) -> bool:
        if not bool(getattr(self.cfg, "producer_voucher_loan_activity_boost_enabled", False)):
            return False
        if pool.policy.role != "producer":
            return False
        until = self._producer_voucher_loan_activity_until.get((pool.pool_id, voucher_id), -1)
        return until >= self.tick

    def _voucher_source_activity_active(self, pool: "Pool", voucher_id: str) -> bool:
        return (
            self._productive_credit_voucher_activity_active(pool, voucher_id)
            or self._producer_voucher_loan_activity_active(pool, voucher_id)
        )

    def _productive_credit_voucher_source_size_boost(self, pool: "Pool", voucher_id: str) -> float:
        if not self._voucher_source_activity_active(pool, voucher_id):
            return 1.0
        multiplier = float(self.cfg.productive_credit_voucher_source_size_multiplier or 1.0)
        return max(0.0, multiplier)

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

    def _record_producer_ordinary_v2v_volume(self, pool_id: str, volume_usd: float) -> None:
        value = max(0.0, float(volume_usd))
        if value <= 1e-9:
            return
        history = self._producer_ordinary_v2v_volume_history.setdefault(pool_id, deque())
        history.append((self.tick, value))
        period = max(1, int(getattr(self.cfg, "producer_debt_pressure_period_ticks", MONTH_TICKS) or MONTH_TICKS))
        cutoff = self.tick - period
        while history and history[0][0] <= cutoff:
            history.popleft()

    def _producer_recent_ordinary_v2v_volume_usd(self, pool_id: str) -> float:
        history = self._producer_ordinary_v2v_volume_history.get(pool_id)
        if not history:
            return 0.0
        period = max(1, int(getattr(self.cfg, "producer_debt_pressure_period_ticks", MONTH_TICKS) or MONTH_TICKS))
        cutoff = self.tick - period
        while history and history[0][0] <= cutoff:
            history.popleft()
        return sum(value for _tick, value in history)

    def _producer_debt_attention_fallback_reference_usd(self, pool: "Pool") -> float:
        total_value = self._pool_total_value(pool)
        multiplier = max(0.0, float(getattr(self.cfg, "voucher_source_swap_size_multiplier", 1.0) or 0.0))
        reference = total_value * float(getattr(self.cfg, "swap_size_mean_frac", 0.0) or 0.0) * multiplier
        if reference <= 1e-9:
            reference = float(getattr(self.cfg, "random_request_amount_mean", 1.0) or 1.0)
        return max(float(getattr(self.cfg, "swap_size_min_usd", 1.0) or 1.0), reference)

    def _producer_debt_attention_reference_usd(self, pool: "Pool") -> float:
        configured = getattr(self.cfg, "producer_debt_attention_reference_usd", None)
        if configured is not None:
            return max(1e-9, float(configured))
        recent_v2v = self._producer_recent_ordinary_v2v_volume_usd(pool.pool_id)
        if recent_v2v > 1e-9:
            return recent_v2v
        return self._producer_debt_attention_fallback_reference_usd(pool)

    def _bond_assessment_service_need_usd(self) -> float:
        if not bool(getattr(self.cfg, "producer_bond_assessment_pressure_enabled", False)):
            return 0.0
        principal = max(0.0, float(getattr(self.cfg, "bond_gross_principal_usd", 0.0) or 0.0))
        if principal <= 1e-9:
            return 0.0
        scheduled_due = self._next_issuer_service_due_target()
        if scheduled_due <= 1e-9:
            return 0.0
        paid_or_reserved = (
            float(self._lp_returned_usd_total or 0.0)
            + float(self._bond_service_reserve_usd_balance or 0.0)
        )
        remaining_need = max(0.0, scheduled_due - paid_or_reserved)
        scale = max(
            0.0,
            float(getattr(self.cfg, "producer_bond_assessment_pressure_scale", 1.0) or 0.0),
        )
        return remaining_need * scale

    def _producer_bond_assessment_pressure_usd(self, pool: "Pool") -> float:
        if pool.policy.role != "producer":
            return 0.0
        service_need = self._bond_assessment_service_need_usd()
        if service_need <= 1e-9:
            return 0.0
        voucher_id = self._producer_own_voucher_id(pool)
        if not voucher_id:
            return 0.0
        active_lenders = {
            obligation.lender_pool_id
            for obligation in self._producer_debt_obligations
            if obligation.producer_pool_id == pool.pool_id
            and obligation.voucher_id == voucher_id
            and (
                obligation.remaining_voucher_units > 1e-9
                or obligation.cash_service_remaining_usd > 1e-9
                or max(0.0, float(getattr(obligation, "pressure_deferred_usd", 0.0) or 0.0)) > 1e-9
            )
        }
        if not active_lenders:
            return 0.0
        producer_exposure = 0.0
        total_exposure = 0.0
        for (lender_pool_id, exposure_voucher_id), exposure_usd in (
            self._lender_producer_voucher_exposure_usd_by_pool_voucher.items()
        ):
            exposure = max(0.0, float(exposure_usd))
            if exposure <= 1e-9:
                continue
            total_exposure += exposure
            if exposure_voucher_id == voucher_id and lender_pool_id in active_lenders:
                producer_exposure += exposure
        if total_exposure <= 1e-9 or producer_exposure <= 1e-9:
            return 0.0
        return service_need * min(1.0, producer_exposure / total_exposure)

    def _producer_debt_attention_pressure_usd(self, pool: "Pool") -> float:
        if pool.policy.role != "producer":
            return 0.0
        voucher_id = self._producer_own_voucher_id(pool)
        if not voucher_id:
            return 0.0
        period = max(1, int(getattr(self.cfg, "producer_debt_pressure_period_ticks", MONTH_TICKS) or MONTH_TICKS))
        pressure = 0.0
        for obligation in self._producer_debt_obligations:
            if obligation.producer_pool_id != pool.pool_id or obligation.voucher_id != voucher_id:
                continue
            remaining_cash = max(0.0, float(obligation.cash_service_remaining_usd))
            remaining_voucher_usd = max(
                0.0,
                float(obligation.remaining_voucher_units) * self._producer_debt_unit_value(obligation),
            )
            remaining_usd = max(remaining_cash, remaining_voucher_usd)
            deferred = max(0.0, float(getattr(obligation, "pressure_deferred_usd", 0.0) or 0.0))
            arrears = max(0.0, float(getattr(obligation, "cash_service_arrears_usd", 0.0) or 0.0))
            if remaining_usd <= 1e-9 and deferred <= 1e-9 and arrears <= 1e-9:
                continue
            arrears = min(arrears, max(remaining_usd, arrears))
            scheduled_base = max(0.0, remaining_usd - arrears - deferred)
            remaining_ticks = max(1, obligation.due_tick - self.tick + 1)
            remaining_periods = max(1, int(math.ceil(remaining_ticks / period)))
            scheduled_due = scheduled_base / remaining_periods
            pressure += deferred + arrears + scheduled_due
        pressure += self._producer_bond_assessment_pressure_usd(pool)
        return max(0.0, pressure)

    def _producer_debt_attention_share(self, pool: "Pool") -> tuple[float, float, float]:
        if not bool(getattr(self.cfg, "producer_debt_attention_crowdout_enabled", False)):
            return 0.0, 0.0, 0.0
        if pool.policy.role != "producer":
            return 0.0, 0.0, 0.0
        pressure = self._producer_debt_attention_pressure_usd(pool)
        min_pressure = max(
            0.0,
            float(getattr(self.cfg, "producer_debt_attention_min_pressure_usd", 0.0) or 0.0),
        )
        reference = self._producer_debt_attention_reference_usd(pool)
        if pressure <= 1e-9 or pressure < min_pressure:
            return 0.0, pressure, reference
        scale = max(
            0.0,
            float(getattr(self.cfg, "producer_debt_attention_crowdout_scale", 1.0) or 0.0),
        )
        max_share = max(
            0.0,
            min(1.0, float(getattr(self.cfg, "producer_debt_attention_crowdout_max_share", 0.90) or 0.0)),
        )
        share = scale * pressure / max(1e-9, pressure + reference)
        return min(max_share, max(0.0, share)), pressure, reference

    def _producer_activity_composition_shift_share(self, pool: "Pool") -> tuple[float, float, float]:
        if not bool(getattr(self.cfg, "producer_activity_composition_shift_enabled", False)):
            return 0.0, 0.0, 0.0
        if pool.policy.role != "producer":
            return 0.0, 0.0, 0.0
        cached = self._producer_activity_composition_share_cache.get(pool.pool_id)
        if cached is not None and cached[0] == self.tick:
            return cached[1], cached[2], cached[3]
        pressure = self._producer_debt_attention_pressure_usd(pool)
        reference = self._producer_debt_attention_reference_usd(pool)
        min_pressure = max(
            0.0,
            float(
                getattr(
                    self.cfg,
                    "producer_activity_composition_shift_min_pressure_usd",
                    0.0,
                )
                or 0.0
            ),
        )
        if pressure <= 1e-9 or pressure < min_pressure:
            result = (0.0, pressure, reference)
            self._producer_activity_composition_share_cache[pool.pool_id] = (
                self.tick,
                result[0],
                result[1],
                result[2],
            )
            return result
        scale = max(
            0.0,
            float(getattr(self.cfg, "producer_activity_composition_shift_scale", 1.0) or 0.0),
        )
        max_share = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.cfg,
                        "producer_activity_composition_shift_max_share",
                        0.60,
                    )
                    or 0.0
                ),
            ),
        )
        share = scale * pressure / max(1e-9, pressure + reference)
        result = (min(max_share, max(0.0, share)), pressure, reference)
        self._producer_activity_composition_share_cache[pool.pool_id] = (
            self.tick,
            result[0],
            result[1],
            result[2],
        )
        return result

    def _record_producer_activity_composition_shift(
        self,
        *,
        share: float,
        pressure: float,
        reference: float,
        v2v_removed: float,
        v2s_added: float,
        motif: Optional[str],
    ) -> None:
        self._producer_activity_composition_reference_usd_sum_tick += reference
        self._producer_activity_composition_reference_count_tick += 1
        if pressure > 1e-9:
            self._producer_activity_composition_pressure_usd_tick += pressure
            self._producer_activity_composition_pressure_usd_total += pressure
        self._producer_activity_composition_shift_share_sum_tick += share
        self._producer_activity_composition_shift_share_count_tick += 1
        self._producer_activity_composition_shift_share_max_tick = max(
            self._producer_activity_composition_shift_share_max_tick,
            share,
        )
        if share <= 1e-12:
            return
        self._producer_activity_composition_shifted_route_attempts_tick += 1
        self._producer_activity_composition_shifted_route_attempts_total += 1
        if motif == "voucher_to_stable":
            self._producer_activity_composition_shifted_v2s_attempts_tick += 1
            self._producer_activity_composition_shifted_v2s_attempts_total += 1
        if v2v_removed > 1e-12:
            self._producer_activity_composition_v2v_weight_removed_tick += v2v_removed
            self._producer_activity_composition_v2v_weight_removed_total += v2v_removed
        if v2s_added > 1e-12:
            self._producer_activity_composition_v2s_weight_added_tick += v2s_added
            self._producer_activity_composition_v2s_weight_added_total += v2s_added

    def _invalidate_producer_activity_composition_share_cache(self, pool_id: str) -> None:
        self._producer_activity_composition_share_cache.pop(pool_id, None)

    def _apply_producer_debt_attention_crowdout(self, pool: "Pool", remaining_attempts: int) -> int:
        if remaining_attempts <= 0:
            return 0
        share, pressure, reference = self._producer_debt_attention_share(pool)
        self._producer_debt_attention_reference_usd_sum_tick += reference
        self._producer_debt_attention_reference_count_tick += 1
        bond_pressure = 0.0
        if pressure > 1e-9:
            self._producer_debt_attention_pressure_usd_tick += pressure
            self._producer_debt_attention_pressure_usd_total += pressure
            bond_pressure = self._producer_bond_assessment_pressure_usd(pool)
            if bond_pressure > 1e-9:
                self._producer_bond_assessment_pressure_usd_tick += bond_pressure
                self._producer_bond_assessment_pressure_usd_total += bond_pressure
        if share <= 1e-12:
            return 0
        self._producer_debt_attention_share_sum_tick += share
        self._producer_debt_attention_share_count_tick += 1
        self._producer_debt_attention_share_max_tick = max(
            self._producer_debt_attention_share_max_tick,
            share,
        )
        suppressed = min(remaining_attempts, int(math.ceil(remaining_attempts * share)))
        if suppressed <= 0:
            return 0
        self._producer_debt_attention_suppressed_attempts_tick += suppressed
        self._producer_debt_attention_suppressed_attempts_total += suppressed
        candidates = self._route_source_asset_candidates(pool)
        has_voucher_source = any(self._settlement_asset_class(asset) == "voucher" for asset in candidates)
        if has_voucher_source:
            self._producer_debt_attention_suppressed_v2v_attempts_tick += suppressed
            self._producer_debt_attention_suppressed_v2v_attempts_total += suppressed
        assessment_offset_attempts = 0.0
        if bond_pressure > 1e-9 and pressure > 1e-9:
            assessment_offset_attempts = suppressed * min(1.0, bond_pressure / pressure)
            self._producer_bond_assessment_sustain_offset_attempts_tick += assessment_offset_attempts
            self._producer_bond_assessment_sustain_offset_attempts_total += assessment_offset_attempts
            if has_voucher_source:
                self._producer_bond_assessment_sustain_offset_v2v_attempts_tick += assessment_offset_attempts
                self._producer_bond_assessment_sustain_offset_v2v_attempts_total += assessment_offset_attempts
        self.log.add(Event(
            self.tick,
            "PRODUCER_DEBT_ATTENTION_CROWDOUT",
            pool_id=pool.pool_id,
            amount=float(suppressed),
            meta={
                "pressure_usd": pressure,
                "reference_usd": reference,
                "attention_share": share,
                "suppressed_attempts": suppressed,
                "voucher_source_available": has_voucher_source,
                "bond_assessment_pressure_usd": bond_pressure,
                "bond_assessment_sustain_offset_attempts": assessment_offset_attempts,
            },
        ))
        return suppressed

    def _apply_producer_credit_request_budget(
        self,
        pools: list["Pool"],
        remaining_requests: int | None,
    ) -> int | None:
        if remaining_requests is None or remaining_requests <= 0:
            return remaining_requests
        share = max(0.0, min(1.0, float(self.cfg.producer_credit_request_budget_share or 0.0)))
        if share <= 0.0:
            return remaining_requests
        producers = [
            pool for pool in pools
            if not pool.policy.system_pool and pool.policy.role == "producer"
        ]
        if not producers:
            return remaining_requests
        budget = min(len(producers), max(1, int(math.ceil(remaining_requests * share))))
        if budget <= 0:
            return remaining_requests
        sampled = self.rng.sample(producers, k=budget) if budget < len(producers) else list(producers)
        for pool in sampled:
            if remaining_requests <= 0:
                break
            self._attempt_repayment(pool)
            remaining_requests -= 1
        return remaining_requests

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

    def _record_route_source_net_flow(self, pool: "Pool", asset_id: str, amount: float, sign: float) -> None:
        if amount <= 1e-12:
            return
        value = self._asset_value(pool, asset_id)
        flow_value = amount * value * sign
        if asset_id == self.cfg.stable_symbol:
            self._route_source_stable_net_flow_value_tick += flow_value
        elif asset_id.startswith("VCHR:"):
            self._route_source_voucher_net_flow_value_tick += flow_value

    def _route_motif_key(self, asset_in: str, asset_out: str) -> str:
        source_class = self._settlement_asset_class(asset_in)
        target_class = self._settlement_asset_class(asset_out)
        if source_class == "voucher" and target_class == "voucher":
            return "voucher_to_voucher"
        if source_class == "voucher" and target_class == "stable":
            return "voucher_to_stable"
        if source_class == "stable" and target_class == "voucher":
            return "stable_to_voucher"
        if source_class == "stable" and target_class == "stable":
            return "stable_to_stable"
        return "other"

    def _is_market_route_context(self, route_context: str) -> bool:
        context = str(route_context or "ordinary")
        return context in {"ordinary", "consumer_purchase", "third_party_purchase"}

    def _is_loan_route_context(self, route_context: str) -> bool:
        context = str(route_context or "ordinary")
        return context in {"loan", "voucher_loan"}

    def _record_route_motif_bucket(
        self,
        *,
        motif: str,
        volume_usd: float,
        count_tick: Dict[str, int],
        count_total: Dict[str, int],
        volume_tick: Dict[str, float],
        volume_total: Dict[str, float],
    ) -> None:
        count_tick[motif] = count_tick.get(motif, 0) + 1
        count_total[motif] = count_total.get(motif, 0) + 1
        volume_tick[motif] = volume_tick.get(motif, 0.0) + volume_usd
        volume_total[motif] = volume_total.get(motif, 0.0) + volume_usd

    def _record_route_motif(
        self,
        *,
        route_context: str,
        source_pool: "Pool",
        asset_in: str,
        asset_out: str,
        amount_in: float,
        plan: RoutePlan,
    ) -> None:
        motif = self._route_motif_key(asset_in, asset_out)
        value = self._asset_value(source_pool, asset_in)
        volume_usd = max(0.0, amount_in * value)
        self._record_route_motif_bucket(
            motif=motif,
            volume_usd=volume_usd,
            count_tick=self._route_motif_count_tick,
            count_total=self._route_motif_count_total,
            volume_tick=self._route_motif_volume_usd_tick,
            volume_total=self._route_motif_volume_usd_total,
        )
        if str(route_context or "ordinary") == "ordinary":
            self._record_route_motif_bucket(
                motif=motif,
                volume_usd=volume_usd,
                count_tick=self._ordinary_route_motif_count_tick,
                count_total=self._ordinary_route_motif_count_total,
                volume_tick=self._ordinary_route_motif_volume_usd_tick,
                volume_total=self._ordinary_route_motif_volume_usd_total,
            )
            if motif == "voucher_to_voucher" and source_pool.policy.role == "producer":
                self._record_producer_ordinary_v2v_volume(source_pool.pool_id, volume_usd)
        if self._is_market_route_context(route_context):
            self._record_route_motif_bucket(
                motif=motif,
                volume_usd=volume_usd,
                count_tick=self._market_route_motif_count_tick,
                count_total=self._market_route_motif_count_total,
                volume_tick=self._market_route_motif_volume_usd_tick,
                volume_total=self._market_route_motif_volume_usd_total,
            )
        if str(route_context or "ordinary") == "repayment":
            self._record_route_motif_bucket(
                motif=motif,
                volume_usd=volume_usd,
                count_tick=self._repayment_route_motif_count_tick,
                count_total=self._repayment_route_motif_count_total,
                volume_tick=self._repayment_route_motif_volume_usd_tick,
                volume_total=self._repayment_route_motif_volume_usd_total,
            )
        if self._is_loan_route_context(route_context):
            self._record_route_motif_bucket(
                motif=motif,
                volume_usd=volume_usd,
                count_tick=self._loan_route_motif_count_tick,
                count_total=self._loan_route_motif_count_total,
                volume_tick=self._loan_route_motif_volume_usd_tick,
                volume_total=self._loan_route_motif_volume_usd_total,
            )
        stable_id = self.cfg.stable_symbol
        stable_touched = any(
            hop.asset_in == stable_id or hop.asset_out == stable_id
            for hop in plan.hops
        )
        stable_endpoint = asset_in == stable_id or asset_out == stable_id
        if stable_touched and not stable_endpoint:
            self._route_motif_stable_intermediate_count_tick += 1
            self._route_motif_stable_intermediate_count_total += 1
            self._route_motif_stable_intermediate_volume_usd_tick += volume_usd
            self._route_motif_stable_intermediate_volume_usd_total += volume_usd
        self.log.add(Event(
            self.tick,
            "ROUTE_MOTIF_RECORDED",
            pool_id=source_pool.pool_id,
            asset_id=asset_in,
            amount=amount_in,
            meta={
                "motif": motif,
                "target_asset": asset_out,
                "route_context": str(route_context or "ordinary"),
                "stable_intermediate_only": bool(stable_touched and not stable_endpoint),
            },
        ))

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

    def _record_recent_clc_fee(self, pool: "Pool", receipt: SwapReceipt) -> None:
        if receipt.status != "executed":
            return
        if pool.policy.system_pool:
            return
        clc_fee = float(receipt.fees.clc_fee)
        if clc_fee <= 0.0:
            return
        value = self._asset_value(pool, receipt.asset_out)
        usd = clc_fee * value
        if usd <= 0.0:
            return
        self._clc_fee_window_entries.append((self.tick, pool.pool_id, usd))
        self._clc_fee_window_totals[pool.pool_id] = (
            self._clc_fee_window_totals.get(pool.pool_id, 0.0) + usd
        )

    def _prune_recent_clc_fee_window(self, window_ticks: int) -> None:
        tick_min = max(1, self.tick - max(1, int(window_ticks or 1)) + 1)
        while self._clc_fee_window_entries and self._clc_fee_window_entries[0][0] < tick_min:
            _tick, pool_id, usd = self._clc_fee_window_entries.popleft()
            remaining = self._clc_fee_window_totals.get(pool_id, 0.0) - usd
            if remaining > 1e-12:
                self._clc_fee_window_totals[pool_id] = remaining
            else:
                self._clc_fee_window_totals.pop(pool_id, None)

    def _record_noam_fee_diagnostics(
        self,
        pool: "Pool",
        receipt: SwapReceipt,
        *,
        kind: str,
        swap_usd: float,
    ) -> None:
        if receipt.status != "executed":
            return
        fee_units = float(receipt.fees.total_fee)
        asset_out = receipt.asset_out
        value_out = pool.values.get_value(asset_out)
        if value_out <= 0.0:
            value_out = self._default_asset_value(asset_out)
        fee_usd = max(0.0, fee_units * value_out)
        if kind == "clearing":
            self._noam_clearing_volume_usd_tick += max(0.0, float(swap_usd))
            self._noam_clearing_fee_usd_tick += fee_usd
            if asset_out == self.cfg.stable_symbol:
                self._noam_clearing_stable_fee_usd_tick += fee_usd
            elif asset_out.startswith("VCHR:"):
                self._noam_clearing_voucher_fee_usd_tick += fee_usd
        else:
            self._noam_routing_volume_usd_tick += max(0.0, float(swap_usd))
            self._noam_routing_fee_usd_tick += fee_usd
            if asset_out == self.cfg.stable_symbol:
                self._noam_routing_stable_fee_usd_tick += fee_usd
            elif asset_out.startswith("VCHR:"):
                self._noam_routing_voucher_fee_usd_tick += fee_usd

    def _record_clc_swap_cumulative(self, receipt: SwapReceipt) -> None:
        if receipt.status != "executed":
            return
        if not self.clc_pool_id:
            return
        if receipt.pool_id != self.clc_pool_id:
            return
        asset_out = receipt.asset_out
        if asset_out == self.cfg.stable_symbol:
            self._clc_pool_swapped_out_stable_total += float(receipt.amount_out)
        elif asset_out.startswith("VCHR:"):
            self._clc_pool_swapped_out_voucher_total += float(receipt.amount_out)

    def _voucher_settlement_mode(self) -> str:
        mode = str(getattr(self.cfg, "voucher_settlement_mode", "legacy") or "legacy")
        mode = mode.strip().lower()
        if mode not in {"legacy", "redeem_outputs"}:
            return "legacy"
        return mode

    def _redeem_final_voucher_outputs_enabled(self) -> bool:
        return self._voucher_settlement_mode() == "redeem_outputs"

    def _record_voucher_fee_retained_for_service(self, pool: "Pool", receipt: SwapReceipt) -> None:
        if receipt.status != "executed" or not receipt.asset_out.startswith("VCHR:"):
            return
        fee_units = max(0.0, float(receipt.fees.total_fee or 0.0))
        if fee_units <= 1e-9:
            return
        value = self._asset_value(pool, receipt.asset_out)
        fee_usd = fee_units * value
        self._voucher_fee_retained_for_service_usd_tick += fee_usd
        self._voucher_fee_retained_for_service_usd_total += fee_usd

    def _is_routable_pool(self, pool: "Pool") -> bool:
        return (not pool.policy.system_pool) and pool.policy.role == "lender"

    def _non_system_pool_count(self) -> int:
        cached = self._non_system_pool_count_cache
        if cached is not None:
            return cached
        count = sum(1 for p in self.pools.values() if not p.policy.system_pool)
        self._non_system_pool_count_cache = count
        return count

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
        self._liquidity_version += 1
        self._target_asset_candidate_cache = {}
        self._target_asset_weight_cache = {}

    def _settlement_asset_class(self, asset_id: str) -> str:
        if asset_id == self.cfg.stable_symbol:
            return "stable"
        if asset_id.startswith("VCHR:"):
            return "voucher"
        return "other"

    def _settlement_motif_weights(self) -> tuple[float, float, float]:
        v2v = max(0.0, float(getattr(self.cfg, "settlement_motif_voucher_to_voucher_share", 0.0) or 0.0))
        v2s = max(0.0, float(getattr(self.cfg, "settlement_motif_voucher_to_stable_share", 0.0) or 0.0))
        s2v = max(0.0, float(getattr(self.cfg, "settlement_motif_stable_to_voucher_share", 0.0) or 0.0))
        if (
            bool(getattr(self.cfg, "settlement_motif_purchase_lane_adjustment_enabled", False))
            and bool(getattr(self.cfg, "lender_voucher_purchase_demand_enabled", False))
            and s2v > 0.0
        ):
            v2s *= max(0.0, 1.0 - min(1.0, s2v))
        return v2v, v2s, s2v

    def _settlement_motif_targets(
        self,
        source_pool: Optional["Pool"] = None,
        route_context: str = "ordinary",
    ) -> Dict[str, float]:
        if not bool(getattr(self.cfg, "settlement_motif_targeting_enabled", False)):
            return {}
        v2v, v2s, s2v = self._settlement_motif_weights()
        weights = {
            "voucher_to_voucher": v2v,
            "voucher_to_stable": v2s,
            "stable_to_voucher": s2v,
        }
        total = sum(weights.values())
        if total <= 1e-12:
            return {}
        targets = {k: v / total for k, v in weights.items() if v > 0.0}
        if (
            source_pool is None
            or str(route_context or "ordinary") != "ordinary"
            or source_pool.policy.role != "producer"
            or not bool(getattr(self.cfg, "producer_activity_composition_shift_enabled", False))
        ):
            return targets

        share, _pressure, _reference = self._producer_activity_composition_shift_share(source_pool)
        if share <= 1e-12:
            return targets

        transfer_share = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.cfg,
                        "producer_activity_composition_shift_to_v2s_share",
                        1.0,
                    )
                    or 0.0
                ),
            ),
        )
        v2v_weight = targets.get("voucher_to_voucher", 0.0)
        v2v_removed = min(v2v_weight, v2v_weight * share * transfer_share)
        if v2v_removed <= 1e-12:
            return targets
        adjusted = dict(targets)
        adjusted["voucher_to_voucher"] = max(0.0, v2v_weight - v2v_removed)
        adjusted["voucher_to_stable"] = adjusted.get("voucher_to_stable", 0.0) + v2v_removed
        return {k: v for k, v in adjusted.items() if v > 1e-12}

    def _settlement_motif_choice(
        self,
        source_pool: Optional["Pool"] = None,
        route_context: str = "ordinary",
    ) -> Optional[str]:
        targets = self._settlement_motif_targets(source_pool=source_pool, route_context=route_context)
        if not targets:
            return None
        # These shares are behavioral priors from the empirical settlement mix,
        # not a controller that forces realized routes to hit the target.
        total = sum(targets.values())
        draw = self.rng.random() * total
        cumulative = 0.0
        for motif, weight in targets.items():
            cumulative += weight
            if draw <= cumulative:
                choice = motif
                break
        else:
            choice = next(iter(targets))
        if (
            source_pool is not None
            and str(route_context or "ordinary") == "ordinary"
            and source_pool.policy.role == "producer"
            and bool(getattr(self.cfg, "producer_activity_composition_shift_enabled", False))
        ):
            base_targets = self._settlement_motif_targets()
            share, pressure, reference = self._producer_activity_composition_shift_share(source_pool)
            base_v2v = base_targets.get("voucher_to_voucher", 0.0)
            base_v2s = base_targets.get("voucher_to_stable", 0.0)
            adjusted_v2v = targets.get("voucher_to_voucher", 0.0)
            adjusted_v2s = targets.get("voucher_to_stable", 0.0)
            self._record_producer_activity_composition_shift(
                share=share,
                pressure=pressure,
                reference=reference,
                v2v_removed=max(0.0, base_v2v - adjusted_v2v),
                v2s_added=max(0.0, adjusted_v2s - base_v2s),
                motif=choice,
            )
        return choice

    def _settlement_motif_classes(self, motif: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if motif == "voucher_to_voucher":
            return "voucher", "voucher"
        if motif == "voucher_to_stable":
            return "voucher", "stable"
        if motif == "stable_to_voucher":
            return "stable", "voucher"
        return None, None

    def _producer_own_voucher_id(self, pool: "Pool") -> Optional[str]:
        if pool.policy.role != "producer":
            return None
        agent = self.agents.get(pool.steward_id)
        if agent is None or getattr(agent, "voucher_spec", None) is None:
            return None
        return agent.voucher_spec.voucher_id

    def _is_producer_own_voucher(self, pool: "Pool", asset_id: str) -> bool:
        voucher_id = self._producer_own_voucher_id(pool)
        return bool(voucher_id and asset_id == voucher_id)

    def _blocks_ordinary_producer_own_voucher_to_stable(
        self,
        source_pool: "Pool",
        asset_in: str,
        route_context: str,
    ) -> bool:
        if bool(getattr(self.cfg, "ordinary_own_voucher_stable_borrowing_enabled", False)):
            return False
        return (
            str(route_context or "ordinary") == "ordinary"
            and self._is_producer_own_voucher(source_pool, asset_in)
        )

    def _ordinary_own_voucher_stable_borrowing_probability(
        self,
        source_pool: Optional["Pool"] = None,
    ) -> float:
        base_probability = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.cfg,
                        "ordinary_own_voucher_stable_borrowing_probability",
                        1.0,
                    )
                    or 0.0
                ),
            ),
        )
        if (
            source_pool is None
            or source_pool.policy.role != "producer"
            or not bool(getattr(self.cfg, "producer_activity_composition_shift_enabled", False))
        ):
            return base_probability
        share, _pressure, _reference = self._producer_activity_composition_shift_share(source_pool)
        if share <= 1e-12:
            return base_probability
        max_probability = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.cfg,
                        "producer_activity_composition_own_voucher_stable_probability_max",
                        0.95,
                    )
                    or 0.0
                ),
            ),
        )
        if max_probability <= base_probability:
            return base_probability
        return base_probability + (max_probability - base_probability) * share

    def _ordinary_own_voucher_stable_target_blocked_for_attempt(
        self,
        source_pool: "Pool",
        asset_in: str,
        route_context: str,
    ) -> bool:
        if (
            str(route_context or "ordinary") != "ordinary"
            or not self._is_producer_own_voucher(source_pool, asset_in)
        ):
            return False
        if not bool(getattr(self.cfg, "ordinary_own_voucher_stable_borrowing_enabled", False)):
            return True
        probability = self._ordinary_own_voucher_stable_borrowing_probability(source_pool)
        if bool(getattr(self.cfg, "producer_activity_composition_shift_enabled", False)):
            self._producer_activity_composition_own_voucher_stable_probability_sum_tick += probability
            self._producer_activity_composition_own_voucher_stable_probability_count_tick += 1
            self._producer_activity_composition_own_voucher_stable_probability_max_tick = max(
                self._producer_activity_composition_own_voucher_stable_probability_max_tick,
                probability,
            )
        if probability >= 1.0:
            return False
        if probability <= 0.0:
            return True
        return self.rng.random() >= probability

    def _effective_ordinary_target_class(
        self,
        source_pool: "Pool",
        asset_in: str,
        route_context: str,
        preferred_class: Optional[str],
        stable_target_blocked: Optional[bool] = None,
    ) -> Optional[str]:
        if stable_target_blocked is None:
            stable_target_blocked = self._blocks_ordinary_producer_own_voucher_to_stable(
                source_pool,
                asset_in,
                route_context,
            )
        if (
            stable_target_blocked
            and preferred_class in (None, "stable")
        ):
            return "voucher"
        return preferred_class

    def _route_context_for_ordinary_own_voucher_source(
        self,
        source_pool: "Pool",
        asset_in: str,
        asset_out: Optional[str],
        route_context: str,
    ) -> str:
        context = str(route_context or "ordinary")
        if context != "ordinary" or not self._is_producer_own_voucher(source_pool, asset_in):
            return context
        if asset_out == self.cfg.stable_symbol:
            return "loan"
        if asset_out and asset_out.startswith("VCHR:"):
            return "voucher_loan"
        return context

    def _settlement_motif_target_class(self, asset_in: str) -> Optional[str]:
        if not bool(getattr(self.cfg, "settlement_motif_targeting_enabled", False)):
            return None
        source_class = self._settlement_asset_class(asset_in)
        if source_class == "stable":
            return "voucher"
        if source_class != "voucher":
            return None
        v2v, v2s, _s2v = self._settlement_motif_weights()
        total = v2v + v2s
        if total <= 1e-12:
            return None
        return "stable" if self.rng.random() < (v2s / total) else "voucher"

    def _choose_source_asset_candidate(
        self,
        source_pool: "Pool",
        candidates: list[str],
        *,
        preferred_class: Optional[str] = None,
    ) -> list[str]:
        if not candidates:
            return []
        filtered = candidates
        if preferred_class:
            preferred = [
                asset_id for asset_id in candidates
                if self._settlement_asset_class(asset_id) == preferred_class
            ]
            if not preferred:
                return []
            filtered = preferred
        if len(filtered) == 1:
            return filtered
        mode = self.cfg.swap_asset_selection_mode
        if mode == "value_weighted":
            weights = self._source_asset_selection_weights(source_pool, filtered)
            total = float(weights.sum())
            if total > 0.0:
                probs = weights / total
                return [str(np.random.choice(filtered, p=probs))]
        return [self.rng.choice(filtered)]

    def _target_asset_candidate_universe(
        self,
        preferred_class: Optional[str],
        restrict_stable: bool,
    ) -> Tuple[str, ...]:
        if not self._frontier_shortlist_enabled() and self._target_asset_candidate_cache_tick != self.tick:
            self._target_asset_candidate_cache_tick = self.tick
            self._target_asset_candidate_cache = {}
        key = (preferred_class, bool(restrict_stable))
        cached = self._target_asset_candidate_cache.get(key)
        if cached is not None:
            return cached

        self._refresh_liquidity_cache()
        candidates = []
        for asset_id, weight in self._liquidity_by_asset.items():
            if weight <= 0.0:
                continue
            if preferred_class and self._settlement_asset_class(asset_id) != preferred_class:
                continue
            if restrict_stable and asset_id == self.cfg.stable_symbol:
                continue
            candidates.append(asset_id)
        universe = tuple(candidates)
        self._target_asset_candidate_cache[key] = universe
        return universe

    def _target_asset_weighted_candidates(
        self,
        preferred_class: Optional[str],
        restrict_stable: bool,
        stable_target_bias: float,
    ) -> Tuple[Tuple[str, ...], np.ndarray]:
        if not self._frontier_shortlist_enabled() and self._target_asset_weight_cache_tick != self.tick:
            self._target_asset_weight_cache_tick = self.tick
            self._target_asset_weight_cache = {}
        key = (preferred_class, bool(restrict_stable), float(stable_target_bias), self._liquidity_version)
        cached = self._target_asset_weight_cache.get(key)
        if cached is not None:
            return cached

        weighted: list[Tuple[str, float]] = []
        for asset_id in self._target_asset_candidate_universe(preferred_class, restrict_stable):
            weight = self._liquidity_by_asset.get(asset_id, 0.0)
            if asset_id == self.cfg.stable_symbol:
                weight *= stable_target_bias
            if weight > 0.0:
                weighted.append((asset_id, weight))
        if weighted:
            assets, weights = zip(*weighted)
            result = (tuple(assets), np.array(weights, dtype=float))
        else:
            result = ((), np.array([], dtype=float))
        self._target_asset_weight_cache[key] = result
        return result

    def _route_source_asset_candidates(self, source_pool: "Pool") -> list[str]:
        if source_pool.policy.system_pool:
            return []
        if not source_pool.vault.inventory:
            return []
        asset_candidates = [
            asset
            for asset in source_pool.vault.inventory
            if self._ordinary_source_spendable_amount(source_pool, asset) > 1e-9
        ]
        if source_pool.policy.role == "lender":
            asset_candidates = [a for a in asset_candidates if a != self.cfg.stable_symbol]
        elif source_pool.policy.role == "producer":
            if self._producer_debt_outstanding(source_pool) > 1e-9:
                asset_candidates = [a for a in asset_candidates if a != self.cfg.stable_symbol]
        elif source_pool.policy.role == "consumer" and self.cfg.consumer_stable_source_bias is None:
            if self.cfg.stable_symbol in asset_candidates:
                asset_candidates = [self.cfg.stable_symbol]
        return asset_candidates

    def _choose_target_asset(
        self,
        asset_in: str,
        source_pool: Optional["Pool"] = None,
        exclude: Optional[Set[str]] = None,
        preferred_class: Optional[str] = None,
    ) -> Optional[str]:
        mode = self.cfg.swap_target_selection_mode
        stable_target_bias = 1.0
        restrict_stable = False
        if source_pool is not None and source_pool.policy.role in ("producer", "consumer"):
            stable_target_bias = max(0.0, float(self.cfg.producer_consumer_stable_target_bias or 0.0))
            restrict_stable = stable_target_bias <= 0.0
        if preferred_class == "stable":
            restrict_stable = False
            stable_target_bias = max(1.0, stable_target_bias)
        if mode == "liquidity_weighted":
            assets, weights = self._target_asset_weighted_candidates(
                preferred_class,
                restrict_stable,
                stable_target_bias,
            )
            candidates: list[str] = []
            candidate_weights: list[float] = []
            for idx, asset_id in enumerate(assets):
                if asset_id == asset_in:
                    continue
                if exclude is not None and asset_id in exclude:
                    continue
                weight = float(weights[idx])
                if weight <= 0.0:
                    continue
                candidates.append(asset_id)
                candidate_weights.append(weight)
            if not candidates:
                return None
            weights_array = np.array(candidate_weights, dtype=float)
            total = float(weights_array.sum())
            if total <= 0.0:
                return None
            probs = weights_array / total
            return str(np.random.choice(candidates, p=probs))

        universe = [a for a in self.factory.asset_universe.keys() if a != self.cfg.sclc_symbol]
        if preferred_class:
            universe = [a for a in universe if self._settlement_asset_class(a) == preferred_class]
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
        self._affinity_buddies = {}

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
        self._affinity_buddies.pop(a, None)
        self._affinity_buddies.pop(b, None)

    def _affinity_buddies_for_pool(self, pool_id: str) -> Optional[Set[str]]:
        buddy_count = max(0, int(self.cfg.affinity_buddy_count or 0))
        if buddy_count <= 0:
            return None
        min_count = max(1, int(getattr(self.cfg, "affinity_buddy_min_count", 1) or 1))
        existing = self._affinity_buddies.get(pool_id)
        if existing is not None:
            return existing if existing else None
        scores: Dict[str, float] = {}
        for (a, b), score in self.pool_affinity.items():
            if score <= 0.0:
                continue
            if a == pool_id:
                scores[b] = max(scores.get(b, 0.0), score)
            elif b == pool_id:
                scores[a] = max(scores.get(a, 0.0), score)
        if len(scores) < min_count:
            if self._frontier_shortlist_enabled():
                self._affinity_buddies[pool_id] = set()
            return None
        top_count = min(buddy_count, len(scores))
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_count]
        buddies = {pid for pid, _ in top}
        if len(buddies) < min_count:
            if self._frontier_shortlist_enabled():
                self._affinity_buddies[pool_id] = set()
            return None
        self._affinity_buddies[pool_id] = buddies
        return buddies

    def _frontier_shortlist_enabled(self) -> bool:
        return str(getattr(self.cfg, "frontier_routing_abstraction", "full") or "full") == "steward_shortlist"

    def _frontier_steward_referral_pools(self) -> list[str]:
        referral_count = max(0, int(getattr(self.cfg, "frontier_steward_referral_count", 4) or 0))
        if referral_count <= 0:
            return []
        scored: list[Tuple[float, str]] = []
        for pool in self.pools.values():
            if not self._is_routable_pool(pool) or pool.policy.paused:
                continue
            liquidity = 0.0
            for asset_id, amount in pool.vault.inventory.items():
                if amount <= 1e-9 or asset_id == self.cfg.sclc_symbol:
                    continue
                liquidity += amount * self._asset_value(pool, asset_id)
            if liquidity <= 1e-9:
                continue
            reliability = float(self._swap_success_ema.get(pool.pool_id, 1.0))
            scored.append((reliability * math.log1p(liquidity), pool.pool_id))
        scored.sort(reverse=True)
        return [pid for _score, pid in scored[:referral_count]]

    def _refresh_frontier_relationship_candidates(self) -> None:
        referrals = set(self._frontier_steward_referral_pools())
        affinity_by_pool: Dict[str, Set[str]] = {}
        for (a, b), score in self.pool_affinity.items():
            if score <= 0.0:
                continue
            affinity_by_pool.setdefault(a, set()).add(b)
            affinity_by_pool.setdefault(b, set()).add(a)

        candidates_by_pool: Dict[str, Set[str]] = {}
        for pool_id, pool in self.pools.items():
            if pool.policy.system_pool or pool.policy.role == "liquidity_provider":
                continue
            candidates: Set[str] = set(referrals)
            candidates.update(affinity_by_pool.get(pool_id, set()))

            for (source_id, _asset_in, _asset_out), plan in self._sticky_plan_by_pool.items():
                if source_id != pool_id:
                    continue
                candidates.update(hop.pool_id for hop in plan.hops)

            if pool.policy.role == "producer":
                agent = self.agents.get(pool.steward_id)
                voucher_id = agent.voucher_spec.voucher_id if agent is not None else None
                if voucher_id:
                    candidates.update(self._producer_voucher_lender_assignments.get(voucher_id, set()))

            for asset_id, amount in pool.vault.inventory.items():
                if amount <= 1e-9:
                    continue
                if self._is_producer_voucher(asset_id):
                    candidates.update(self._producer_voucher_lender_assignments.get(asset_id, set()))

            filtered = {
                pid
                for pid in candidates
                if pid != pool_id
                and pid in self.pools
                and self._is_routable_pool(self.pools[pid])
                and not self.pools[pid].policy.paused
            }
            if filtered:
                candidates_by_pool[pool_id] = filtered
        self._frontier_relationship_candidates = candidates_by_pool
        self._frontier_relationship_last_refresh_tick = self.tick

    def _frontier_relationship_pool_whitelist(self, source_pool: "Pool") -> Optional[Set[str]]:
        if not self._frontier_shortlist_enabled():
            return None
        interval = max(1, int(getattr(self.cfg, "frontier_relationship_refresh_ticks", 13) or 13))
        if (
            self._frontier_relationship_last_refresh_tick < 0
            or (self.tick - self._frontier_relationship_last_refresh_tick) >= interval
        ):
            self._refresh_frontier_relationship_candidates()
        candidates = self._frontier_relationship_candidates.get(source_pool.pool_id)
        return candidates if candidates else None

    def _choose_frontier_relationship_target_asset(
        self,
        asset_in: str,
        source_pool: "Pool",
        pool_whitelist: Set[str],
        *,
        exclude: Optional[Set[str]] = None,
        preferred_class: Optional[str] = None,
    ) -> Optional[str]:
        if not pool_whitelist:
            return None
        stable_target_bias = 1.0
        restrict_stable = False
        if source_pool.policy.role in ("producer", "consumer"):
            stable_target_bias = max(0.0, float(self.cfg.producer_consumer_stable_target_bias or 0.0))
            restrict_stable = stable_target_bias <= 0.0
        if preferred_class == "stable":
            restrict_stable = False
            stable_target_bias = max(1.0, stable_target_bias)

        ordered_assets: list[str] = []
        weights_by_asset: Dict[str, float] = {}
        for pid in sorted(pool_whitelist):
            if pid == source_pool.pool_id:
                continue
            pool = self.pools.get(pid)
            if pool is None or pool.policy.paused or not self._is_routable_pool(pool):
                continue
            if not pool.registry.is_listed(asset_in):
                continue
            for asset_out, amount in pool.vault.inventory.items():
                if amount <= 1e-9 or asset_out == asset_in:
                    continue
                if exclude is not None and asset_out in exclude:
                    continue
                if preferred_class and self._settlement_asset_class(asset_out) != preferred_class:
                    continue
                if restrict_stable and asset_out == self.cfg.stable_symbol:
                    continue
                if asset_out == self.cfg.sclc_symbol:
                    continue
                if not pool.registry.is_listed(asset_out):
                    continue
                weight = amount * self._asset_value(pool, asset_out)
                if asset_out == self.cfg.stable_symbol:
                    weight *= stable_target_bias
                if weight <= 0.0:
                    continue
                if asset_out not in weights_by_asset:
                    ordered_assets.append(asset_out)
                    weights_by_asset[asset_out] = weight
                else:
                    weights_by_asset[asset_out] += weight
        if not ordered_assets:
            return None
        weights = np.array([weights_by_asset[asset_id] for asset_id in ordered_assets], dtype=float)
        total = float(weights.sum())
        if total <= 0.0:
            return None
        return str(np.random.choice(ordered_assets, p=weights / total))

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
        window_ticks = max(1, int(window_ticks or 1))
        if self._clc_fee_window_ticks is None:
            self._clc_fee_window_ticks = window_ticks
        if self._clc_fee_window_ticks == window_ticks:
            self._prune_recent_clc_fee_window(window_ticks)
            return {
                pool_id: amount
                for pool_id, amount in self._clc_fee_window_totals.items()
                if amount > 0.0
                and pool_id in self.pools
                and not self.pools[pool_id].policy.system_pool
            }

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
                value = self._default_asset_value(asset_id)
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
        agent = self.agents.get(pool.steward_id)
        if agent is None:
            return 0.0
        voucher_id = agent.voucher_spec.voucher_id
        if voucher_id is None:
            return 0.0
        value = self._asset_value(pool, voucher_id)
        if value <= 0.0:
            return 0.0
        amount = usd_value / value
        if amount <= 1e-9:
            return 0.0
        self._vault_add(pool, voucher_id, amount, "voucher_inflow", "system")
        agent.issuer.issue(amount)
        self._mark_lender_voucher_limits_dirty(voucher_id)
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

    def _swap_sustain_target(self) -> int:
        if not bool(self.cfg.swap_sustain_enabled):
            return 0
        window = max(0, int(self.cfg.swap_sustain_window_ticks or 0))
        floor = max(0, int(self.cfg.swap_sustain_floor_per_tick or 0))
        if window <= 0:
            return floor
        counts = self._recent_swap_counts[-window:]
        if not counts:
            return floor
        avg = sum(counts) / len(counts)
        target = int(math.ceil(avg))
        return max(target, floor)

    def _sustain_swap_activity(self) -> None:
        if not bool(self.cfg.swap_sustain_enabled):
            return
        target = self._swap_sustain_target()
        if target <= 0:
            return
        if bool(getattr(self.cfg, "producer_bond_assessment_sustain_offset_enabled", False)):
            offset = max(0.0, float(self._producer_bond_assessment_sustain_offset_attempts_tick or 0.0))
            if offset > 1e-9:
                adjusted_target = max(0, int(math.ceil(max(0.0, target - offset))))
                target_reduction = max(0, target - adjusted_target)
                if target_reduction > 0:
                    self._producer_bond_assessment_sustain_target_reduction_tick += target_reduction
                    self._producer_bond_assessment_sustain_target_reduction_total += target_reduction
                target = adjusted_target
                if target <= 0:
                    return
        executed = self._noam_routing_swaps_tick + self._noam_clearing_swaps_tick
        if executed >= target:
            return
        shortfall = max(0, target - executed)
        attempts_per_missing = max(1.0, float(self.cfg.swap_sustain_attempts_per_missing_swap or 1.0))
        max_extra = max(
            int(math.ceil(shortfall * attempts_per_missing)),
            int(self.cfg.swap_sustain_max_extra_attempts or 0),
        )
        if max_extra <= 0:
            return
        max_passes = max(1, int(self.cfg.swap_sustain_max_rounds or 1))
        pools = [
            p for p in self.pools.values()
            if not p.policy.system_pool and p.policy.role != "liquidity_provider" and p.vault.inventory
        ]
        if not pools:
            return
        remaining = max_extra
        idle_passes = 0
        for _ in range(max_passes):
            if executed >= target or remaining <= 0:
                break
            self.rng.shuffle(pools)
            progressed = False
            for p in pools:
                if executed >= target or remaining <= 0:
                    break
                attempted = self._random_route_request(source_pool=p, max_assets=1)
                if attempted <= 0:
                    remaining -= 1
                    continue
                progressed = True
                remaining -= attempted
                executed = self._noam_routing_swaps_tick + self._noam_clearing_swaps_tick
            if not progressed:
                idle_passes += 1
                if idle_passes >= 2:
                    break
            else:
                idle_passes = 0

    def _record_swap_history(self) -> None:
        if not bool(self.cfg.swap_sustain_enabled):
            return
        window = max(0, int(self.cfg.swap_sustain_window_ticks or 0))
        if window <= 0:
            return
        count = int(self._noam_routing_swaps_tick + self._noam_clearing_swaps_tick)
        self._recent_swap_counts.append(count)
        if len(self._recent_swap_counts) > window:
            self._recent_swap_counts = self._recent_swap_counts[-window:]

    def _find_buddy_direct_plan(
        self,
        *,
        source_pool: "Pool",
        asset_in: str,
        amount_in: float,
        buddy_pools: Set[str],
        target_asset: Optional[str] = None,
        target_class: Optional[str] = None,
    ) -> Tuple[RoutePlan, float, bool, Optional[str]]:
        if not buddy_pools:
            return RoutePlan(ok=False, reason="buddy_empty", hops=[]), amount_in, False, None

        def pick_candidate(amount: float) -> Optional[Tuple[str, str]]:
            candidates: list[Tuple[float, str, str]] = []
            for pid in buddy_pools:
                if pid == source_pool.pool_id:
                    continue
                pool = self.pools.get(pid)
                if pool is None or pool.policy.paused:
                    continue
                if not self._is_routable_pool(pool):
                    continue
                if not pool.registry.is_listed(asset_in):
                    continue
                if target_asset:
                    if not pool.registry.is_listed(target_asset):
                        continue
                    if pool.vault.get(target_asset) <= 1e-9:
                        continue
                    okq, _reason, amt_out_net, _ = pool.quote_swap(asset_in, amount, target_asset)
                    if not okq or amt_out_net <= 1e-9:
                        continue
                    ok, _ = pool.can_swap(self.tick, asset_in, amount, target_asset, amt_out_net)
                    if not ok:
                        continue
                    value_out = pool.values.get_value(target_asset)
                    if value_out <= 0.0:
                        value_out = 1.0
                    score = pool.vault.get(target_asset) * value_out
                    candidates.append((score, pid, target_asset))
                    continue

                best_asset = None
                best_score = 0.0
                for asset_out, amt_out in pool.vault.inventory.items():
                    if amt_out <= 1e-9 or asset_out == asset_in:
                        continue
                    if target_class and self._settlement_asset_class(asset_out) != target_class:
                        continue
                    if not pool.registry.is_listed(asset_out):
                        continue
                    okq, _reason, amt_out_net, _ = pool.quote_swap(asset_in, amount, asset_out)
                    if not okq or amt_out_net <= 1e-9:
                        continue
                    ok, _ = pool.can_swap(self.tick, asset_in, amount, asset_out, amt_out_net)
                    if not ok:
                        continue
                    value_out = pool.values.get_value(asset_out)
                    if value_out <= 0.0:
                        value_out = 1.0
                    score = amt_out * value_out
                    if score > best_score:
                        best_score = score
                        best_asset = asset_out
                if best_asset is not None:
                    candidates.append((best_score, pid, best_asset))

            if not candidates:
                return None
            candidates.sort(reverse=True)
            _score, pid, asset_out = candidates[0]
            return (pid, asset_out)

        candidate = pick_candidate(amount_in)
        used_fallback = False
        amount_used = amount_in
        if candidate is None:
            min_amount = self._min_route_amount_in(source_pool, asset_in, amount_in)
            if min_amount is not None:
                candidate = pick_candidate(min_amount)
                if candidate is not None:
                    used_fallback = True
                    amount_used = min_amount

        if candidate is None:
            return RoutePlan(ok=False, reason="buddy_no_path", hops=[]), amount_in, False, None

        pid, asset_out = candidate
        plan = RoutePlan(
            ok=True,
            reason="buddy_direct",
            hops=[Hop(pool_id=pid, asset_in=asset_in, asset_out=asset_out, amount_in=amount_used)],
        )
        return plan, amount_used, used_fallback, asset_out

    def _find_route_noam(
        self,
        *,
        tick: int,
        start_asset: str,
        target_asset: str,
        amount_in: float,
        source_pool: "Pool",
        target_pools: Optional[Set[str]] = None,
        pool_whitelist: Optional[Set[str]] = None,
    ) -> RoutePlan:
        if start_asset == target_asset:
            return RoutePlan(ok=True, reason="trivial", hops=[], expected_amount_out=amount_in)

        self._maybe_refresh_noam_working_set()
        self._noam_edge_cache_reset()
        max_hops = max(1, int(self.cfg.noam_max_hops or self.cfg.max_hops))
        base_beam = max(1, int(self.cfg.noam_beam_width or 1))
        min_beam = max(1, int(self.cfg.noam_dynamic_min_beam or 1))
        beam_width = max(1, self._noam_scaled_cap(base_beam, min_beam))
        base_edge_cap = int(self.cfg.noam_edge_cap_per_state or 0)
        min_edge_cap = max(1, int(self.cfg.noam_dynamic_min_edge_cap or 1))
        edge_cap = self._noam_scaled_cap(base_edge_cap, min_edge_cap) if base_edge_cap > 0 else 0
        route_cache = NoamRouteCache(remaining={}, inventory={}, value={})
        edge_cache = self._noam_edge_cache
        affinity_bias = float(self.cfg.sticky_route_bias or 0.0)
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
                pool_ids = self._noam_top_pools_active.get(asset_in)
                if not pool_ids:
                    pool_ids = self._noam_top_pools.get(asset_in, [])
                if not pool_ids:
                    continue
                if pool_whitelist is not None:
                    pool_ids = [pid for pid in pool_ids if pid in pool_whitelist]
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
                    if not self._is_routable_pool(pool):
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
                        edge_key = self._noam_edge_cache_key(pool, asset_in, asset_out, amount_in, route_cache)
                        cached_edge = edge_cache.get(edge_key)
                        if cached_edge is None:
                            allowed = self._noam_edge_allowed(pool, asset_in, asset_out, amount_in, route_cache)
                            if not allowed:
                                edge_cache[edge_key] = (False, -1e9)
                                continue
                            base_score = self._noam_edge_score(
                                pool,
                                asset_in,
                                asset_out,
                                amount_in,
                                cache=route_cache,
                            )
                            edge_cache[edge_key] = (True, base_score)
                        else:
                            allowed, base_score = cached_edge
                            if not allowed:
                                continue
                        edges_scanned += 1
                        edge_score = base_score
                        if affinity_bias > 0.0:
                            affinity = self._affinity_score(source_pool.pool_id, pid)
                            if affinity > 0.0:
                                edge_score += affinity_bias * affinity
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
        pool_whitelist: Optional[Set[str]] = None,
    ) -> RoutePlan:
        if start_asset == target_asset:
            return RoutePlan(ok=True, reason="trivial", hops=[], expected_amount_out=amount_in)

        self._maybe_refresh_noam_working_set()
        pool_count = self._non_system_pool_count()
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
                pool_whitelist=pool_whitelist,
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
                pool_whitelist=pool_whitelist,
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
                pool_whitelist=pool_whitelist,
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
                    pool_whitelist=pool_whitelist,
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
                    pool_whitelist=pool_whitelist,
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
            pool_whitelist=pool_whitelist,
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
        pool_whitelist: Optional[Set[str]] = None,
    ) -> Tuple[RoutePlan, float, bool]:
        cache_key: Optional[Tuple[str, str, str, int]] = None
        if self.cfg.routing_mode == "noam":
            cache_key = self._noam_route_cache_key(
                source_pool,
                start_asset,
                target_asset,
                amount_in,
                target_pools,
                pool_whitelist,
            )
            if cache_key is not None:
                cached = self._noam_route_cache_get(cache_key)
                if cached is not None and cached.ok and cached.hops:
                    blocked = any(
                        self._noam_failure_active(hop.pool_id, hop.asset_in, hop.asset_out)
                        for hop in cached.hops
                    )
                    if not blocked and self._validate_route_plan(cached, amount_in, source_pool):
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
                    pool_whitelist=pool_whitelist,
                )
            else:
                plan = self._find_route_noam(
                    tick=tick,
                    start_asset=start_asset,
                    target_asset=target_asset,
                    amount_in=amount_in,
                    source_pool=source_pool,
                    target_pools=target_pools,
                    pool_whitelist=pool_whitelist,
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
                pool_whitelist=pool_whitelist,
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
                    pool_whitelist=pool_whitelist,
                )
            else:
                retry_plan = self._find_route_noam(
                    tick=tick,
                    start_asset=start_asset,
                    target_asset=target_asset,
                    amount_in=fallback_amount,
                    source_pool=source_pool,
                    target_pools=target_pools,
                    pool_whitelist=pool_whitelist,
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
                pool_whitelist=pool_whitelist,
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
        self._non_system_pool_count_cache = None

        self.log.add(Event(self.tick, "POOL_CREATED", actor_id=agent.agent_id, pool_id=pool.pool_id))

        def list_voucher(asset_id: str, cap_in: float, *, value_override: Optional[float] = None) -> None:
            v = value_override if value_override is not None else self._default_asset_value(asset_id)
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
        supply_updates: Set[str] = set()

        if role == "lender":
            want_k = max(1, int(np.random.poisson(cfg.add_pool_want_assets_mean)))
            wanted = [
                a for a in self.factory.sample_assets(want_k, p_overlap=cfg.p_want_overlap)
                if a != cfg.stable_symbol and not self._is_producer_voucher(a)
            ]
            for a in wanted:
                v = self._default_asset_value(a)
                list_voucher(a, cap_in=self._lender_voucher_cap(a, lender_pool=pool, value_override=v), value_override=v)

            offer_k = max(1, int(np.random.poisson(cfg.add_pool_offer_assets_mean)))
            offered = [
                a for a in self.factory.sample_assets(offer_k, p_overlap=cfg.p_offer_overlap)
                if a != cfg.stable_symbol and not self._is_producer_voucher(a)
            ]
            for a in offered:
                if a not in wanted:
                    v = self._default_asset_value(a)
                    list_voucher(a, cap_in=self._lender_voucher_cap(a, lender_pool=pool, value_override=v), value_override=v)

            stable_seed = float(max(0.0, cfg.lender_initial_stable_mean))
            self._vault_add(pool, cfg.stable_symbol, stable_seed, "seed_stable", "system")

            for a in offered:
                amt = float(max(0.0, np.random.exponential(250.0)))
                if amt <= 0:
                    continue
                spec = self.factory.voucher_specs.get(a)
                if spec and spec.issuer_id != agent.agent_id:
                    continue
                self._vault_add(pool, a, amt, "seed_asset", "system")
                if spec:
                    agent.issuer.issue(amt)
                    self._mark_lender_voucher_limits_dirty(a)
                    supply_updates.add(a)
            self._assign_pending_producer_vouchers()
            producer_listed: Set[str] = set()
            for voucher_id, lender_ids in self._producer_voucher_lender_assignments.items():
                if pool.pool_id not in lender_ids:
                    continue
                if self._list_producer_voucher_on_lender(pool, voucher_id):
                    producer_listed.add(voucher_id)
            if producer_listed:
                self._refresh_lender_voucher_limits(producer_listed)

        elif role == "producer":
            own_v = agent.voucher_spec.voucher_id
            self._producer_voucher_ids.add(own_v)
            self._invalidate_lender_producer_voucher_count()
            list_voucher(own_v, cap_in=cfg.producer_voucher_cap_in)

            stable_seed = 0.0
            if stable_seed > 0.0:
                self._vault_add(pool, cfg.stable_symbol, stable_seed, "seed_stable", "system")

            own_amt = float(max(0.0, np.random.exponential(10000.0)))
            if own_amt > 0:
                self._vault_add(pool, own_v, own_amt, "seed_voucher", agent.agent_id)
                agent.issuer.issue(own_amt)
                self._mark_lender_voucher_limits_dirty(own_v)
                supply_updates.add(own_v)
            self._assign_producer_voucher_to_lender(own_v)
            producer_count = max(1, int(cfg.initial_producers or 1))
            stable_seed = max(0.0, float(cfg.producer_initial_stable_total_usd or 0.0)) / producer_count
            if stable_seed > 0.0:
                self._vault_add(pool, cfg.stable_symbol, stable_seed, "seed_stable", "system")

        elif role == "liquidity_provider":
            self._lp_pending_contribution_tick[pool.pool_id] = self.tick + 1
            stable_seed = float(max(0.0, cfg.lp_initial_stable_mean))
            if stable_seed > 0.0:
                self._vault_add(pool, cfg.stable_symbol, stable_seed, "seed_stable", "system")

        else:  # consumer
            consumer_count = max(1, int(cfg.initial_consumers or 1))
            consumer_seed_total = max(0.0, float(cfg.consumer_initial_stable_total_usd or 0.0))
            if consumer_seed_total > 0.0:
                stable_seed = consumer_seed_total / consumer_count
            else:
                stable_seed = float(max(0.0, np.random.exponential(cfg.initial_stable_per_pool_mean * 0.25)))
            self._vault_add(pool, cfg.stable_symbol, stable_seed, "seed_stable", "system")

        self.log.add(Event(self.tick, "POOL_CONFIGURED", actor_id=agent.agent_id, pool_id=pool.pool_id,
                           meta={"mode": pool.policy.mode, "role": pool.policy.role, "min_stable_reserve": pool.policy.min_stable_reserve}))
        self.log.add(Event(self.tick, "POOL_SEEDED", pool_id=pool.pool_id,
                           meta={"stable_seed": stable_seed, "offered": offered, "wanted": wanted}))

        if stable_seed > 0.0:
            self.initial_stable_total += stable_seed

        if supply_updates:
            self._refresh_lender_voucher_limits(supply_updates)

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
            enabled_assets = {
                asset_id
                for asset_id, pol in p.registry.listings.items()
                if pol.enabled
            }
            current = sum(
                1
                for asset_id in enabled_assets
                if asset_id not in (stable_id, sclc_id)
            )
            if current >= target:
                continue
            need = min(add_per_tick, target - current)
            if need <= 0:
                continue
            candidates = [a for a in asset_universe if a not in enabled_assets]
            if not candidates:
                continue
            if len(candidates) > need:
                chosen = self.rng.sample(candidates, k=need)
            else:
                chosen = candidates

            for asset_id in chosen:
                value = self._default_asset_value(asset_id)
                cap_in = float(self.cfg.default_cap_in)
                if p.policy.role == "lender":
                    cap_in = self._lender_voucher_cap(asset_id, lender_pool=p, value_override=value)
                elif p.policy.role == "producer":
                    cap_in = float(self.cfg.producer_voucher_cap_in)
                p.list_asset_with_value_and_limit(
                    asset_id,
                    value=value,
                    window_len=self.cfg.default_window_len,
                    cap_in=cap_in,
                )
                if self._is_routable_pool(p):
                    self.accept_index.setdefault(asset_id, set()).add(p.pool_id)
                if p.policy.role == "lender" and self._is_producer_voucher(asset_id):
                    self._invalidate_lender_producer_voucher_count(p.pool_id)
            total_added += len(chosen)
            pools_updated += 1
        if total_added > 0:
            self.log.add(Event(
                self.tick,
                "DESIRED_ASSETS_GROWN",
                amount=total_added,
                meta={"pools_updated": pools_updated, "target_per_pool": target},
            ))

    def _record_producer_deposit_value(self, voucher_id: str, amount_usd: float) -> None:
        if amount_usd <= 1e-9:
            return
        self._producer_deposit_value_by_voucher[voucher_id] = (
            self._producer_deposit_value_by_voucher.get(voucher_id, 0.0) + float(amount_usd)
        )
        self._mark_lender_voucher_limits_dirty(voucher_id)

    def _producer_deposit_credit_capacity_usd(self) -> float:
        multiple = max(0.0, float(self.cfg.lender_voucher_cap_deposit_multiple or 0.0))
        return sum(self._producer_deposit_value_by_voucher.values()) * multiple

    def _producer_debt_contract_repayment_enabled(self) -> bool:
        return bool(getattr(self.cfg, "producer_debt_contract_repayment_enabled", False))

    def _producer_debt_contract_service_margin(self, debt_kind: str = "stable") -> float:
        if not self._producer_debt_contract_repayment_enabled():
            return 0.0
        kind = str(debt_kind or "stable").lower()
        specific_margin = None
        if kind == "voucher":
            specific_margin = getattr(
                self.cfg,
                "producer_voucher_debt_contract_service_margin_rate",
                None,
            )
        elif kind == "stable":
            specific_margin = getattr(
                self.cfg,
                "producer_stable_debt_contract_service_margin_rate",
                None,
            )
        if specific_margin is None:
            specific_margin = getattr(self.cfg, "producer_debt_contract_service_margin_rate", 0.0)
        margin = max(
            0.0,
            float(specific_margin or 0.0),
        )
        return margin

    def _producer_debt_contract_service_multiplier(self, debt_kind: str = "stable") -> float:
        if not self._producer_debt_contract_repayment_enabled():
            return 1.0
        margin = self._producer_debt_contract_service_margin(debt_kind)
        return 1.0 + margin

    def _producer_debt_contract_revenue_rate(self) -> float:
        base_rate = max(0.0, float(self.cfg.productive_credit_return_rate or 0.0))
        if not self._producer_debt_contract_repayment_enabled():
            return base_rate
        contract_rate = max(
            0.0,
            float(getattr(self.cfg, "producer_debt_contract_revenue_rate", 0.0) or 0.0),
        )
        return max(base_rate, contract_rate)

    def _increase_lender_producer_voucher_exposure(
        self,
        pool_id: str,
        voucher_id: str,
        amount_usd: float,
        reason: str,
    ) -> None:
        amount = max(0.0, float(amount_usd))
        if amount <= 1e-9:
            return
        pool = self.pools.get(pool_id)
        if pool is None or pool.policy.role != "lender" or not self._is_producer_voucher(voucher_id):
            return
        key = (pool_id, voucher_id)
        self._lender_producer_voucher_exposure_usd_by_pool_voucher[key] = (
            self._lender_producer_voucher_exposure_usd_by_pool_voucher.get(key, 0.0) + amount
        )
        self.log.add(Event(
            self.tick,
            "LENDER_PRODUCER_VOUCHER_EXPOSURE_INCREASED",
            pool_id=pool_id,
            asset_id=voucher_id,
            amount=amount,
            meta={"reason": reason},
        ))

    def _reduce_lender_producer_voucher_exposure(
        self,
        pool_id: str,
        voucher_id: str,
        amount_usd: float,
        reason: str,
    ) -> float:
        amount = max(0.0, float(amount_usd))
        if amount <= 1e-9:
            return 0.0
        key = (pool_id, voucher_id)
        outstanding = max(0.0, self._lender_producer_voucher_exposure_usd_by_pool_voucher.get(key, 0.0))
        reduced = min(outstanding, amount)
        if reduced <= 1e-9:
            return 0.0
        remaining = outstanding - reduced
        if remaining > 1e-9:
            self._lender_producer_voucher_exposure_usd_by_pool_voucher[key] = remaining
        else:
            self._lender_producer_voucher_exposure_usd_by_pool_voucher.pop(key, None)
        self.log.add(Event(
            self.tick,
            "LENDER_PRODUCER_VOUCHER_EXPOSURE_REDUCED",
            pool_id=pool_id,
            asset_id=voucher_id,
            amount=reduced,
            meta={"reason": reason},
        ))
        return reduced

    def _stable_purchase_recovery_reason(self, source_pool: "Pool", voucher_id: str) -> str:
        spec = self.factory.voucher_specs.get(voucher_id)
        if (
            source_pool.policy.role == "producer"
            and spec is not None
            and source_pool.steward_id == spec.issuer_id
        ):
            return "borrower_stable_repayment"
        if source_pool.policy.role == "consumer":
            return "external_nonproducer_stable_purchase"
        if source_pool.policy.role == "producer":
            return "other_producer_stable_purchase"
        return "inventory_turnover_stable_purchase"

    def _producer_debt_pressure_enabled(self) -> bool:
        return bool(getattr(self.cfg, "producer_debt_pressure_enabled", False))

    def _producer_debt_pressure_period_ticks(self) -> int:
        configured = int(getattr(self.cfg, "producer_debt_pressure_period_ticks", 0) or 0)
        if configured <= 0:
            configured = int(getattr(self.cfg, "loan_activity_period_ticks", MONTH_TICKS) or MONTH_TICKS)
        return max(1, configured)

    def _producer_debt_pressure_batching_enabled(self) -> bool:
        return bool(getattr(self.cfg, "producer_debt_pressure_batching_enabled", False))

    def _producer_debt_pressure_min_swap_usd(self) -> float:
        return max(0.0, float(getattr(self.cfg, "producer_debt_pressure_min_swap_usd", 0.0) or 0.0))

    def _producer_debt_arrears_usd(self) -> float:
        return sum(
            max(0.0, float(getattr(obligation, "cash_service_arrears_usd", 0.0) or 0.0))
            for obligation in self._producer_debt_obligations
        )

    def _producer_debt_pressure_deferred_balance_usd(self) -> float:
        return sum(
            max(0.0, float(getattr(obligation, "pressure_deferred_usd", 0.0) or 0.0))
            for obligation in self._producer_debt_obligations
        )

    def _producer_debt_service_capacity_balance_usd(self) -> float:
        return sum(max(0.0, amount) for amount in self._producer_debt_service_capacity_by_pool.values())

    def _producer_has_active_own_voucher_debt(self, producer_pool: "Pool") -> bool:
        agent = self.agents.get(producer_pool.steward_id)
        if agent is None:
            return False
        voucher_id = agent.voucher_spec.voucher_id
        for obligation in self._producer_debt_obligations:
            if obligation.producer_pool_id != producer_pool.pool_id:
                continue
            if obligation.voucher_id != voucher_id:
                continue
            if obligation.remaining_voucher_units > 1e-9:
                return True
            if (
                self._producer_debt_contract_repayment_enabled()
                and max(0.0, obligation.cash_service_remaining_usd) > 1e-9
            ):
                return True
        return False

    def _credit_producer_debt_service_capacity(
        self,
        producer_pool: "Pool",
        amount_usd: float,
        reason: str,
    ) -> float:
        if not self._producer_debt_pressure_enabled():
            return 0.0
        if producer_pool.policy.role != "producer" or amount_usd <= 1e-9:
            return 0.0
        if not self._producer_has_active_own_voucher_debt(producer_pool):
            return 0.0
        share = max(
            0.0,
            min(1.0, float(getattr(self.cfg, "producer_debt_pressure_capacity_share", 1.0) or 0.0)),
        )
        credited = float(amount_usd) * share
        if credited <= 1e-9:
            return 0.0
        self._producer_debt_service_capacity_by_pool[producer_pool.pool_id] = (
            self._producer_debt_service_capacity_by_pool.get(producer_pool.pool_id, 0.0)
            + credited
        )
        self._producer_debt_service_capacity_credited_usd_tick += credited
        self._producer_debt_service_capacity_credited_usd_total += credited
        self.log.add(Event(
            self.tick,
            "PRODUCER_DEBT_SERVICE_CAPACITY_CREDITED",
            pool_id=producer_pool.pool_id,
            asset_id=self.cfg.stable_symbol,
            amount=credited,
            meta={"reason": reason, "capacity_share": share},
        ))
        return credited

    def _apply_producer_stable_exit(
        self,
        producer_pool: "Pool",
        amount_usd: float,
        reason: str,
    ) -> float:
        if producer_pool.policy.role != "producer":
            return 0.0
        exit_share = max(
            0.0,
            min(1.0, float(getattr(self.cfg, "producer_stable_exit_share", 0.0) or 0.0)),
        )
        if exit_share <= 1e-9 or amount_usd <= 1e-9:
            return 0.0
        stable_id = self.cfg.stable_symbol
        stable_value = self._asset_value(producer_pool, stable_id)
        if stable_value <= 1e-12:
            stable_value = 1.0
        exit_usd = max(0.0, float(amount_usd) * exit_share)
        exit_units = min(producer_pool.vault.get(stable_id), exit_usd / stable_value)
        if exit_units <= 1e-9:
            return 0.0
        if not self._vault_sub(producer_pool, stable_id, exit_units, "producer_stable_exit", "outside_network"):
            return 0.0
        exited_usd = exit_units * stable_value
        self._producer_stable_exited_usd_tick += exited_usd
        self._producer_stable_exited_usd_total += exited_usd
        self._stable_offramp_usd_tick += exited_usd
        self._credit_producer_debt_service_capacity(producer_pool, exited_usd, reason)
        self.log.add(Event(
            self.tick,
            "PRODUCER_STABLE_EXITED",
            pool_id=producer_pool.pool_id,
            asset_id=stable_id,
            amount=exited_usd,
            meta={"reason": reason, "exit_share": exit_share},
        ))
        return exited_usd

    def _refresh_bond_recovery_budget_caps(self) -> None:
        external_budget = max(
            0.0,
            float(
                getattr(
                    self.cfg,
                    "external_nonproducer_stable_to_voucher_budget_usd_per_tick",
                    0.0,
                )
                or 0.0
            ),
        )
        other_producer_budget = max(
            0.0,
            float(
                getattr(
                    self.cfg,
                    "other_producer_stable_to_voucher_budget_usd_per_tick",
                    0.0,
                )
                or 0.0
            ),
        )
        if external_budget <= 1e-9 and other_producer_budget <= 1e-9:
            self._bond_recovery_budget_remaining_by_reason_tick = {}
            return
        self._bond_recovery_budget_remaining_by_reason_tick = {
            "consumer_stable_purchase": external_budget,
            "external_nonproducer_stable_purchase": external_budget,
            "other_producer_stable_purchase": other_producer_budget,
            "third_party_stable_purchase": 0.0,
            "inventory_turnover_stable_purchase": 0.0,
        }

    def _cap_lender_recovered_stable_eligibility(self, reason: str, eligible_amount: float) -> float:
        eligible = max(0.0, float(eligible_amount))
        if eligible <= 1e-9 or not self._bond_recovery_budget_remaining_by_reason_tick:
            return eligible
        if reason not in self._bond_recovery_budget_remaining_by_reason_tick:
            return eligible
        remaining = max(0.0, self._bond_recovery_budget_remaining_by_reason_tick.get(reason, 0.0))
        capped = min(eligible, remaining)
        self._bond_recovery_budget_remaining_by_reason_tick[reason] = max(0.0, remaining - capped)
        return capped

    def _record_lender_recovered_stable(
        self,
        pool_id: str,
        amount_usd: float,
        reason: str,
        eligible_amount_usd: float | None = None,
    ) -> None:
        if amount_usd <= 1e-9:
            return
        pool = self.pools.get(pool_id)
        if pool is None or pool.policy.role != "lender":
            return
        amount = float(amount_usd)
        eligible_amount = amount if eligible_amount_usd is None else max(0.0, min(amount, float(eligible_amount_usd)))
        eligible_amount = self._cap_lender_recovered_stable_eligibility(reason, eligible_amount)
        inventory_turnover = max(0.0, amount - eligible_amount)
        reserved = self._reserve_lender_recovered_stable_for_bond_service(pool, eligible_amount, reason)
        pending = max(0.0, eligible_amount - reserved)
        if pending > 1e-9:
            self._lender_recovered_stable_by_pool[pool_id] = (
                self._lender_recovered_stable_by_pool.get(pool_id, 0.0) + pending
            )
        self._lender_recovered_stable_total_by_pool[pool_id] = (
            self._lender_recovered_stable_total_by_pool.get(pool_id, 0.0) + amount
        )
        pool_reason_key = (pool_id, str(reason))
        self._lender_recovered_stable_total_by_pool_reason[pool_reason_key] = (
            self._lender_recovered_stable_total_by_pool_reason.get(pool_reason_key, 0.0) + amount
        )
        self._lender_recovered_stable_usd_tick += amount
        self._lender_recovered_stable_usd_total += amount
        self._bond_eligible_pool_exposure_recovered_stable_usd_tick += eligible_amount
        self._bond_eligible_pool_exposure_recovered_stable_usd_total += eligible_amount
        self._lender_inventory_turnover_stable_usd_tick += inventory_turnover
        self._lender_inventory_turnover_stable_usd_total += inventory_turnover
        if reason == "borrower_stable_repayment":
            self._lender_recovered_stable_borrower_self_usd_tick += amount
            self._lender_recovered_stable_borrower_self_usd_total += amount
            self._lender_recovered_stable_borrower_regular_usd_tick += amount
            self._lender_recovered_stable_borrower_regular_usd_total += amount
        elif reason == "producer_debt_maturity_repayment":
            self._lender_recovered_stable_borrower_maturity_usd_tick += amount
            self._lender_recovered_stable_borrower_maturity_usd_total += amount
        elif reason in {"consumer_stable_purchase", "external_nonproducer_stable_purchase"}:
            self._lender_recovered_stable_consumer_purchase_usd_tick += amount
            self._lender_recovered_stable_consumer_purchase_usd_total += amount
            self._lender_recovered_stable_external_nonproducer_purchase_usd_tick += amount
            self._lender_recovered_stable_external_nonproducer_purchase_usd_total += amount
        elif reason == "other_producer_stable_purchase":
            self._lender_recovered_stable_other_producer_purchase_usd_tick += amount
            self._lender_recovered_stable_other_producer_purchase_usd_total += amount
        elif reason == "third_party_stable_purchase":
            self._lender_recovered_stable_third_party_purchase_usd_tick += amount
            self._lender_recovered_stable_third_party_purchase_usd_total += amount
        else:
            self._lender_recovered_stable_other_usd_tick += amount
            self._lender_recovered_stable_other_usd_total += amount
        self.log.add(Event(
            self.tick,
            "LENDER_STABLE_RECOVERED",
            pool_id=pool_id,
            amount=amount,
            meta={
                "reason": reason,
                "eligible_for_bond_service": eligible_amount,
                "inventory_turnover_usd": inventory_turnover,
                "reserved_for_bond_service": reserved,
            },
        ))

    def _producer_debt_obligation_sort_key(self, obligation: ProducerDebtObligation) -> Tuple[int, int, int]:
        return (obligation.due_tick, obligation.issued_tick, obligation.obligation_id)

    def _index_producer_debt_obligation(self, obligation: ProducerDebtObligation) -> None:
        key = (obligation.lender_pool_id, obligation.voucher_id)
        self._producer_debt_obligations_by_lender_voucher.setdefault(key, []).append(obligation)
        self._producer_debt_obligation_index_dirty.add(key)

    def _producer_debt_for_lender_voucher(
        self,
        lender_pool_id: str,
        voucher_id: str,
    ) -> list[ProducerDebtObligation]:
        key = (lender_pool_id, voucher_id)
        obligations = self._producer_debt_obligations_by_lender_voucher.get(key, [])
        if key in self._producer_debt_obligation_index_dirty:
            obligations.sort(key=self._producer_debt_obligation_sort_key)
            self._producer_debt_obligation_index_dirty.discard(key)
        return obligations

    def _rebuild_producer_debt_obligation_index(self) -> None:
        indexed: Dict[Tuple[str, str], list[ProducerDebtObligation]] = {}
        for obligation in self._producer_debt_obligations:
            key = (obligation.lender_pool_id, obligation.voucher_id)
            indexed.setdefault(key, []).append(obligation)
        for obligations in indexed.values():
            obligations.sort(key=self._producer_debt_obligation_sort_key)
        self._producer_debt_obligations_by_lender_voucher = indexed
        self._producer_debt_obligation_index_dirty = set()

    def _register_producer_debt_obligation(
        self,
        producer_pool_id: str,
        lender_pool_id: str,
        voucher_id: str,
        voucher_units: float,
        borrowed_usd: float,
        contract_cash_service: bool = True,
        debt_kind: str = "stable",
    ) -> None:
        if not bool(self.cfg.producer_debt_maturity_enabled):
            return
        if voucher_units <= 1e-9 or borrowed_usd <= 1e-9:
            return
        producer_pool = self.pools.get(producer_pool_id)
        lender_pool = self.pools.get(lender_pool_id)
        if producer_pool is None or lender_pool is None:
            return
        if producer_pool.policy.role != "producer" or lender_pool.policy.role != "lender":
            return
        spec = self.factory.voucher_specs.get(voucher_id)
        if spec is None or spec.issuer_id != producer_pool.steward_id:
            return
        maturity = max(1, int(self.cfg.producer_debt_maturity_ticks or 1))
        normalized_debt_kind = "voucher" if str(debt_kind or "").lower() == "voucher" else "stable"
        cash_service_due = 0.0
        if contract_cash_service and self._producer_debt_contract_repayment_enabled():
            cash_service_due = float(borrowed_usd) * self._producer_debt_contract_service_multiplier(
                normalized_debt_kind
            )
        obligation = ProducerDebtObligation(
            obligation_id=self._next_producer_debt_obligation_id,
            producer_pool_id=producer_pool_id,
            lender_pool_id=lender_pool_id,
            voucher_id=voucher_id,
            issued_tick=self.tick,
            due_tick=self.tick + maturity,
            original_voucher_units=float(voucher_units),
            remaining_voucher_units=float(voucher_units),
            borrowed_usd=float(borrowed_usd),
            debt_kind=normalized_debt_kind,
            cash_service_due_usd=cash_service_due,
            cash_service_remaining_usd=cash_service_due,
        )
        self._next_producer_debt_obligation_id += 1
        self._producer_debt_obligations.append(obligation)
        self._index_producer_debt_obligation(obligation)
        self._invalidate_producer_activity_composition_share_cache(producer_pool_id)
        self._producer_debt_originated_usd_tick += float(borrowed_usd)
        self._producer_debt_originated_usd_total += float(borrowed_usd)
        self._producer_debt_cash_service_due_usd_tick += cash_service_due
        self._producer_debt_cash_service_due_usd_total += cash_service_due
        self._increase_lender_producer_voucher_exposure(
            lender_pool_id,
            voucher_id,
            float(borrowed_usd),
            f"producer_debt_{normalized_debt_kind}",
        )
        self.log.add(Event(
            self.tick,
            "PRODUCER_DEBT_OBLIGATION_CREATED",
            pool_id=lender_pool_id,
            asset_id=voucher_id,
            amount=float(borrowed_usd),
            meta={
                "obligation_id": obligation.obligation_id,
                "producer_pool_id": producer_pool_id,
                "voucher_units": float(voucher_units),
                "due_tick": obligation.due_tick,
                "debt_kind": normalized_debt_kind,
                "cash_service_due_usd": cash_service_due,
                "contract_cash_service": bool(contract_cash_service),
            },
        ))

    def _reduce_producer_debt_obligations(
        self,
        lender_pool_id: str,
        voucher_id: str,
        voucher_units: float,
        reason: str,
        source_pool_id: str | None = None,
        source_role: str | None = None,
    ) -> float:
        if voucher_units <= 1e-9:
            return 0.0
        lender_pool = self.pools.get(lender_pool_id)
        if lender_pool is None:
            return 0.0
        remaining = float(voucher_units)
        reduced_units = 0.0
        unit_value = self._asset_value(lender_pool, voucher_id)
        for obligation in self._producer_debt_for_lender_voucher(lender_pool_id, voucher_id):
            if remaining <= 1e-9:
                break
            if obligation.remaining_voucher_units <= 1e-9:
                continue
            used = min(remaining, obligation.remaining_voucher_units)
            obligation.remaining_voucher_units = max(0.0, obligation.remaining_voucher_units - used)
            self._invalidate_producer_activity_composition_share_cache(obligation.producer_pool_id)
            remaining -= used
            reduced_units += used
            reduced_usd = used * unit_value
            cash_service_reduced_usd = 0.0
            if self._producer_debt_contract_repayment_enabled() and reason in {
                "borrower_stable_repayment",
                "consumer_stable_purchase",
                "external_nonproducer_stable_purchase",
                "other_producer_stable_purchase",
                "third_party_stable_purchase",
                "inventory_turnover_stable_purchase",
            }:
                cash_service_reduced_usd = min(
                    max(0.0, obligation.cash_service_remaining_usd),
                    reduced_usd,
                )
                obligation.cash_service_remaining_usd = max(
                    0.0,
                    obligation.cash_service_remaining_usd - cash_service_reduced_usd,
                )
                self._record_producer_debt_cash_service_paid(cash_service_reduced_usd)
            meta = {
                "obligation_id": obligation.obligation_id,
                "producer_pool_id": obligation.producer_pool_id,
                "source_pool_id": source_pool_id,
                "source_role": source_role,
                "voucher_units": used,
                "reason": reason,
                "cash_service_reduced_usd": cash_service_reduced_usd,
            }
            if reason == "borrower_stable_repayment":
                self._producer_debt_repaid_usd_tick += reduced_usd
                self._producer_debt_repaid_usd_total += reduced_usd
                self._producer_debt_repaid_regular_usd_tick += reduced_usd
                self._producer_debt_repaid_regular_usd_total += reduced_usd
                self._producer_debt_stable_recovered_usd_tick += reduced_usd
                self._producer_debt_stable_recovered_usd_total += reduced_usd
                event_type = "PRODUCER_DEBT_REPAID_BY_BORROWER_STABLE"
            elif reason in {"consumer_stable_purchase", "external_nonproducer_stable_purchase"}:
                self._producer_debt_stable_recovered_usd_tick += reduced_usd
                self._producer_debt_stable_recovered_usd_total += reduced_usd
                self._producer_debt_consumer_stable_purchase_usd_tick += reduced_usd
                self._producer_debt_consumer_stable_purchase_usd_total += reduced_usd
                event_type = (
                    "PRODUCER_DEBT_RECOVERED_BY_EXTERNAL_NONPRODUCER_STABLE"
                    if reason == "external_nonproducer_stable_purchase"
                    else "PRODUCER_DEBT_RECOVERED_BY_CONSUMER_STABLE"
                )
            elif reason in {"third_party_stable_purchase", "other_producer_stable_purchase"}:
                self._producer_debt_stable_recovered_usd_tick += reduced_usd
                self._producer_debt_stable_recovered_usd_total += reduced_usd
                self._producer_debt_third_party_stable_purchase_usd_tick += reduced_usd
                self._producer_debt_third_party_stable_purchase_usd_total += reduced_usd
                event_type = (
                    "PRODUCER_DEBT_RECOVERED_BY_OTHER_PRODUCER_STABLE"
                    if reason == "other_producer_stable_purchase"
                    else "PRODUCER_DEBT_RECOVERED_BY_THIRD_PARTY_STABLE"
                )
            elif reason == "inventory_turnover_stable_purchase":
                self._producer_debt_stable_recovered_usd_tick += reduced_usd
                self._producer_debt_stable_recovered_usd_total += reduced_usd
                event_type = "PRODUCER_DEBT_RECOVERED_BY_INVENTORY_TURNOVER_STABLE"
            elif reason == "producer_voucher_swap_out":
                self._producer_debt_closed_by_circulation_usd_tick += reduced_usd
                self._producer_debt_closed_by_circulation_usd_total += reduced_usd
                self._producer_debt_closed_by_voucher_swap_usd_tick += reduced_usd
                self._producer_debt_closed_by_voucher_swap_usd_total += reduced_usd
                event_type = "PRODUCER_DEBT_CLOSED_BY_VOUCHER_SWAP"
            else:
                self._producer_debt_closed_by_circulation_usd_tick += reduced_usd
                self._producer_debt_closed_by_circulation_usd_total += reduced_usd
                event_type = "PRODUCER_DEBT_CLOSED_BY_CIRCULATION"
            self.log.add(Event(
                self.tick,
                event_type,
                pool_id=lender_pool_id,
                asset_id=voucher_id,
                amount=reduced_usd,
                meta=meta,
            ))
        if reduced_units <= 1e-9:
            return 0.0
        return reduced_units

    def _schedule_productive_credit_inflow(self, pool_id: str, borrowed_usd: float, voucher_id: str) -> None:
        if not bool(self.cfg.productive_credit_enabled):
            return
        rate = self._producer_debt_contract_revenue_rate()
        if borrowed_usd <= 1e-9 or rate <= 0.0:
            return
        lag = max(0, int(self.cfg.productive_credit_lag_ticks or 0))
        due_tick = self.tick + lag
        amount = borrowed_usd * rate
        self._productive_credit_queue.setdefault(due_tick, []).append((pool_id, amount, voucher_id))
        self.log.add(Event(
            self.tick,
            "PRODUCTIVE_CREDIT_SCHEDULED",
            pool_id=pool_id,
            asset_id=self.cfg.stable_symbol,
            amount=amount,
            meta={"borrowed_usd": borrowed_usd, "lag_ticks": lag, "voucher_id": voucher_id},
        ))

    def _productive_credit_voucher_deposit_cap_remaining(self, pool: "Pool") -> float | None:
        cap_rate = max(
            0.0,
            float(getattr(self.cfg, "productive_credit_voucher_deposit_cap_rate_per_month", 0.0) or 0.0),
        )
        if cap_rate <= 0.0:
            return None
        base_value = max(self._pool_total_value(pool), float(self.cfg.random_request_amount_mean or 1.0))
        tick_cap = (base_value * cap_rate) / float(MONTH_TICKS)
        used = self._productive_credit_voucher_deposit_usd_by_pool_tick.get(pool.pool_id, 0.0)
        return max(0.0, tick_cap - used)

    def _productive_credit_allocation(
        self,
        pool: "Pool",
        amount_usd: float,
        voucher_id: str,
    ) -> tuple[float, float, float]:
        amount_usd = max(0.0, float(amount_usd))
        if amount_usd <= 1e-9:
            return 0.0, 0.0, 0.0
        if not bool(getattr(self.cfg, "productive_credit_voucher_feedback_enabled", False)):
            return amount_usd, 0.0, 0.0
        if pool.policy.role != "producer":
            return amount_usd, 0.0, 0.0
        agent = self.agents.get(pool.steward_id)
        if agent is None or agent.voucher_spec.voucher_id != voucher_id:
            return amount_usd, 0.0, 0.0
        share = max(
            0.0,
            min(1.0, float(getattr(self.cfg, "productive_credit_voucher_deposit_share", 0.0) or 0.0)),
        )
        requested_voucher_value = amount_usd * share
        cap_remaining = self._productive_credit_voucher_deposit_cap_remaining(pool)
        voucher_value = requested_voucher_value
        if cap_remaining is not None:
            voucher_value = min(voucher_value, cap_remaining)
        voucher_value = max(0.0, min(amount_usd, voucher_value))
        clipped_value = max(0.0, requested_voucher_value - voucher_value)
        stable_value = max(0.0, amount_usd - voucher_value)
        return stable_value, voucher_value, clipped_value

    def _mark_productive_credit_voucher_activity(self, pool_id: str, voucher_id: str) -> None:
        if not bool(getattr(self.cfg, "productive_credit_voucher_activity_boost_enabled", False)):
            return
        window = max(1, int(getattr(self.cfg, "productive_credit_voucher_activity_boost_window_ticks", 1) or 1))
        key = (pool_id, voucher_id)
        self._productive_credit_voucher_activity_until[key] = max(
            self._productive_credit_voucher_activity_until.get(key, 0),
            self.tick + window,
        )

    def _mark_producer_voucher_loan_activity(self, pool_id: str, voucher_id: str) -> None:
        if not bool(getattr(self.cfg, "producer_voucher_loan_activity_boost_enabled", False)):
            return
        window = max(1, int(getattr(self.cfg, "productive_credit_voucher_activity_boost_window_ticks", 1) or 1))
        key = (pool_id, voucher_id)
        self._producer_voucher_loan_activity_until[key] = max(
            self._producer_voucher_loan_activity_until.get(key, 0),
            self.tick + window,
        )

    def _apply_productive_credit_inflows(self) -> None:
        due = self._productive_credit_queue.pop(self.tick, [])
        if not due:
            return
        stable_id = self.cfg.stable_symbol
        for pool_id, amount, voucher_id in due:
            pool = self.pools.get(pool_id)
            if pool is None or pool.policy.system_pool:
                continue
            amount = max(0.0, float(amount))
            if amount <= 1e-9:
                continue
            stable_amount, voucher_value, clipped_value = self._productive_credit_allocation(
                pool, amount, voucher_id
            )
            if stable_amount > 1e-9:
                self._vault_add(pool, stable_id, stable_amount, "productive_credit_inflow", "business")
                self._stable_onramp_usd_tick += stable_amount
                self._productive_credit_stable_retained_usd_tick += stable_amount
                self._productive_credit_stable_retained_usd_total += stable_amount
                self._productive_credit_stable_retained_usd_by_pool[pool_id] = (
                    self._productive_credit_stable_retained_usd_by_pool.get(pool_id, 0.0)
                    + stable_amount
                )
            if voucher_value > 1e-9:
                agent = self.agents.get(pool.steward_id)
                if agent is not None:
                    deposited = self._deposit_producer_voucher_with_lenders(
                        producer_pool=pool,
                        agent=agent,
                        voucher_id=voucher_id,
                        voucher_value_usd=voucher_value,
                        source="productive_credit_voucher_deposit",
                    )
                    if deposited:
                        self._productive_credit_voucher_deposit_usd_tick += voucher_value
                        self._productive_credit_voucher_deposit_usd_total += voucher_value
                        self._productive_credit_voucher_deposit_usd_by_pool[pool_id] = (
                            self._productive_credit_voucher_deposit_usd_by_pool.get(pool_id, 0.0)
                            + voucher_value
                        )
                        self._productive_credit_voucher_deposit_usd_by_pool_tick[pool_id] = (
                            self._productive_credit_voucher_deposit_usd_by_pool_tick.get(pool_id, 0.0)
                            + voucher_value
                        )
                        self._mark_productive_credit_voucher_activity(pool_id, voucher_id)
            if clipped_value > 1e-9:
                self._productive_credit_voucher_deposit_cap_clipped_usd_tick += clipped_value
                self._productive_credit_voucher_deposit_cap_clipped_usd_total += clipped_value
            self._productive_credit_inflow_usd_tick += amount
            self._productive_credit_inflow_usd_total += amount
            self._productive_credit_inflow_usd_by_pool[pool_id] = (
                self._productive_credit_inflow_usd_by_pool.get(pool_id, 0.0) + amount
            )
            self.log.add(Event(
                self.tick,
                "PRODUCTIVE_CREDIT_INFLOW",
                pool_id=pool_id,
                asset_id=stable_id,
                amount=amount,
                meta={
                    "voucher_id": voucher_id,
                    "stable_retained_usd": stable_amount,
                    "voucher_deposit_usd": voucher_value,
                    "voucher_deposit_cap_clipped_usd": clipped_value,
                },
            ))
        self._refresh_dirty_lender_voucher_limits()
        self.rebuild_indexes()

    def _producer_debt_unit_value(self, obligation: ProducerDebtObligation) -> float:
        pool = self.pools.get(obligation.lender_pool_id) or self.pools.get(obligation.producer_pool_id)
        if pool is None:
            return self._default_asset_value(obligation.voucher_id)
        return self._asset_value(pool, obligation.voucher_id)

    def _producer_debt_active_usd(self) -> float:
        total = 0.0
        for obligation in self._producer_debt_obligations:
            voucher_exposure_usd = 0.0
            if obligation.remaining_voucher_units > 1e-9:
                voucher_exposure_usd = obligation.remaining_voucher_units * self._producer_debt_unit_value(obligation)
            cash_service_usd = (
                max(0.0, obligation.cash_service_remaining_usd)
                if self._producer_debt_contract_repayment_enabled()
                else 0.0
            )
            active_usd = max(voucher_exposure_usd, cash_service_usd)
            if active_usd <= 1e-9:
                continue
            total += active_usd
        return total

    def _producer_debt_stable_available(self, producer_pool: "Pool") -> float:
        stable_id = self.cfg.stable_symbol
        stable_available = producer_pool.vault.get(stable_id)
        if bool(self.cfg.producer_debt_maturity_preserve_reserve):
            stable_available -= max(0.0, float(producer_pool.policy.min_stable_reserve or 0.0))
        return max(0.0, stable_available)

    def _record_producer_debt_cash_service_paid(self, amount_usd: float) -> None:
        amount = max(0.0, float(amount_usd))
        if amount <= 1e-9:
            return
        self._producer_debt_cash_service_paid_usd_tick += amount
        self._producer_debt_cash_service_paid_usd_total += amount

    def _obligation_cash_service_multiplier(self, obligation: ProducerDebtObligation) -> float:
        if obligation.borrowed_usd <= 1e-9:
            return self._producer_debt_contract_service_multiplier(obligation.debt_kind)
        if obligation.cash_service_due_usd <= 1e-9:
            return 1.0
        return max(1.0, obligation.cash_service_due_usd / obligation.borrowed_usd)

    def _close_producer_debt_voucher_units(
        self,
        obligation: ProducerDebtObligation,
        units: float,
        reason: str,
    ) -> float:
        close_units = max(0.0, min(float(units), obligation.remaining_voucher_units))
        if close_units <= 1e-9:
            return 0.0
        producer_pool = self.pools.get(obligation.producer_pool_id)
        lender_pool = self.pools.get(obligation.lender_pool_id)
        if lender_pool is None:
            obligation.remaining_voucher_units = max(0.0, obligation.remaining_voucher_units - close_units)
            self._invalidate_producer_activity_composition_share_cache(obligation.producer_pool_id)
            return close_units

        held_units = min(close_units, lender_pool.vault.get(obligation.voucher_id))
        if held_units > 1e-9 and self._vault_sub(
            lender_pool,
            obligation.voucher_id,
            held_units,
            "producer_debt_cash_service_voucher_return",
            obligation.producer_pool_id,
        ):
            spec = self.factory.voucher_specs.get(obligation.voucher_id)
            if spec and spec.issuer_id in self.agents:
                issuer_agent = self.agents[spec.issuer_id]
                issuer_agent.issuer.return_to_issuer(held_units)
                self._mark_lender_voucher_limits_dirty(obligation.voucher_id)
                issuer_pool = self.pools.get(issuer_agent.pool_id)
                if issuer_pool is not None:
                    self._vault_add(
                        issuer_pool,
                        obligation.voucher_id,
                        held_units,
                        "producer_debt_cash_service_redeem_receive",
                        lender_pool.pool_id,
                    )
            elif producer_pool is not None:
                self._vault_add(
                    producer_pool,
                    obligation.voucher_id,
                    held_units,
                    "producer_debt_cash_service_redeem_receive",
                    lender_pool.pool_id,
                )

        obligation.remaining_voucher_units = max(0.0, obligation.remaining_voucher_units - close_units)
        self._invalidate_producer_activity_composition_share_cache(obligation.producer_pool_id)
        self._reduce_lender_producer_voucher_exposure(
            obligation.lender_pool_id,
            obligation.voucher_id,
            close_units * self._producer_debt_unit_value(obligation),
            reason,
        )
        self.log.add(Event(
            self.tick,
            "PRODUCER_DEBT_VOUCHER_EXPOSURE_CLOSED_BY_CASH_SERVICE",
            pool_id=obligation.lender_pool_id,
            asset_id=obligation.voucher_id,
            amount=close_units * self._producer_debt_unit_value(obligation),
            meta={
                "obligation_id": obligation.obligation_id,
                "producer_pool_id": obligation.producer_pool_id,
                "voucher_units": close_units,
                "held_units_returned": held_units,
                "reason": reason,
            },
        ))
        return close_units

    def _execute_producer_debt_cash_service_payment(
        self,
        obligation: ProducerDebtObligation,
        target_usd: float,
        reason: str,
    ) -> float:
        producer_pool = self.pools.get(obligation.producer_pool_id)
        lender_pool = self.pools.get(obligation.lender_pool_id)
        if producer_pool is None or lender_pool is None:
            return 0.0
        cash_remaining = max(0.0, float(obligation.cash_service_remaining_usd or 0.0))
        target = max(0.0, min(float(target_usd), cash_remaining))
        if target <= 1e-9:
            return 0.0
        stable_id = self.cfg.stable_symbol
        stable_value = self._asset_value(lender_pool, stable_id)
        if stable_value <= 0.0:
            stable_value = 1.0
        available_usd = self._producer_debt_stable_available(producer_pool) * stable_value
        payment_usd = min(target, available_usd)
        if payment_usd <= 1e-9:
            return 0.0
        amount_units = payment_usd / stable_value
        if not self._vault_sub(
            producer_pool,
            stable_id,
            amount_units,
            "producer_debt_contract_service_out",
            obligation.lender_pool_id,
        ):
            return 0.0
        self._vault_add(
            lender_pool,
            stable_id,
            amount_units,
            "producer_debt_contract_service_in",
            obligation.producer_pool_id,
        )
        unit_value = self._producer_debt_unit_value(obligation)
        eligible_basis_usd = max(0.0, obligation.remaining_voucher_units * unit_value)
        self._record_lender_recovered_stable(
            lender_pool.pool_id,
            payment_usd,
            reason,
            eligible_amount_usd=min(payment_usd, eligible_basis_usd),
        )
        obligation.cash_service_remaining_usd = max(0.0, cash_remaining - payment_usd)
        self._invalidate_producer_activity_composition_share_cache(obligation.producer_pool_id)
        self._record_producer_debt_cash_service_paid(payment_usd)

        service_multiplier = self._obligation_cash_service_multiplier(obligation)
        if unit_value > 0.0 and service_multiplier > 0.0:
            principal_closed_usd = min(
                obligation.remaining_voucher_units * unit_value,
                payment_usd / service_multiplier,
            )
            if principal_closed_usd > 1e-9:
                self._close_producer_debt_voucher_units(
                    obligation,
                    principal_closed_usd / unit_value,
                    reason,
                )

        self._producer_debt_repaid_usd_tick += payment_usd
        self._producer_debt_repaid_usd_total += payment_usd
        if reason == "producer_debt_maturity_repayment":
            self._producer_debt_repaid_maturity_usd_tick += payment_usd
            self._producer_debt_repaid_maturity_usd_total += payment_usd
        else:
            self._producer_debt_repaid_regular_usd_tick += payment_usd
            self._producer_debt_repaid_regular_usd_total += payment_usd
        self._producer_debt_stable_recovered_usd_tick += payment_usd
        self._producer_debt_stable_recovered_usd_total += payment_usd
        self.log.add(Event(
            self.tick,
            "PRODUCER_DEBT_CASH_SERVICE_PAID",
            pool_id=lender_pool.pool_id,
            asset_id=stable_id,
            amount=payment_usd,
            meta={
                "obligation_id": obligation.obligation_id,
                "producer_pool_id": producer_pool.pool_id,
                "reason": reason,
                "cash_service_remaining_usd": obligation.cash_service_remaining_usd,
            },
        ))
        return payment_usd

    def _write_off_producer_debt_cash_default(
        self,
        obligation: ProducerDebtObligation,
        default_usd: float,
        reason: str,
    ) -> None:
        amount = max(0.0, min(float(default_usd), obligation.cash_service_remaining_usd))
        if amount <= 1e-9:
            return
        obligation.cash_service_remaining_usd = max(0.0, obligation.cash_service_remaining_usd - amount)
        self._producer_debt_defaulted_usd_tick += amount
        self._producer_debt_defaulted_usd_total += amount
        self.log.add(Event(
            self.tick,
            "PRODUCER_DEBT_CASH_SERVICE_DEFAULTED",
            pool_id=obligation.lender_pool_id,
            asset_id=self.cfg.stable_symbol,
            amount=amount,
            meta={
                "obligation_id": obligation.obligation_id,
                "producer_pool_id": obligation.producer_pool_id,
                "reason": reason,
            },
        ))

    def _write_off_producer_debt_default(
        self,
        lender_pool: "Pool",
        obligation: ProducerDebtObligation,
        default_units: float,
        unit_value: float,
        reason: str,
    ) -> None:
        default_units = max(0.0, min(float(default_units), obligation.remaining_voucher_units))
        if default_units <= 1e-9:
            return
        available_units = lender_pool.vault.get(obligation.voucher_id)
        writeoff_units = min(default_units, available_units)
        if writeoff_units > 1e-9:
            self._vault_sub(
                lender_pool,
                obligation.voucher_id,
                writeoff_units,
                "producer_debt_default_writeoff",
                obligation.producer_pool_id,
            )
        default_usd = default_units * unit_value
        obligation.remaining_voucher_units = max(0.0, obligation.remaining_voucher_units - default_units)
        self._producer_debt_defaulted_usd_tick += default_usd
        self._producer_debt_defaulted_usd_total += default_usd
        self._reduce_lender_producer_voucher_exposure(
            lender_pool.pool_id,
            obligation.voucher_id,
            default_usd,
            reason,
        )
        self.log.add(Event(
            self.tick,
            "PRODUCER_DEBT_DEFAULTED",
            pool_id=obligation.lender_pool_id,
            asset_id=obligation.voucher_id,
            amount=default_usd,
            meta={
                "obligation_id": obligation.obligation_id,
                "producer_pool_id": obligation.producer_pool_id,
                "default_units": default_units,
                "writeoff_units": writeoff_units,
                "reason": reason,
            },
        ))

    def _execute_producer_debt_maturity_repayment(
        self,
        obligation: ProducerDebtObligation,
        target_units: float,
        unit_value: float,
    ) -> float:
        producer_pool = self.pools.get(obligation.producer_pool_id)
        lender_pool = self.pools.get(obligation.lender_pool_id)
        if producer_pool is None or lender_pool is None:
            return 0.0
        target_units = max(0.0, min(float(target_units), obligation.remaining_voucher_units))
        if target_units <= 1e-9 or unit_value <= 0.0:
            return 0.0
        stable_id = self.cfg.stable_symbol
        stable_value = self._asset_value(lender_pool, stable_id)
        if stable_value <= 0.0:
            stable_value = 1.0
        stable_needed = (target_units * unit_value) / stable_value
        amount_in = min(stable_needed, self._producer_debt_stable_available(producer_pool))
        if amount_in <= 1e-9:
            return 0.0
        if not self._vault_sub(
            producer_pool,
            stable_id,
            amount_in,
            "producer_debt_maturity_repayment_out",
            obligation.lender_pool_id,
        ):
            return 0.0

        receipt = lender_pool.execute_swap(
            self.tick,
            actor=f"producer_debt_maturity:{producer_pool.pool_id}",
            asset_in=stable_id,
            amount_in=amount_in,
            asset_out=obligation.voucher_id,
        )
        if receipt.status != "executed":
            self._vault_add(producer_pool, stable_id, amount_in, "producer_debt_maturity_refund", lender_pool.pool_id)
            self._noam_update_edge_after_swap(
                lender_pool,
                receipt.asset_in,
                receipt.asset_out,
                float(receipt.amount_in),
                success=False,
                fail_reason=receipt.fail_reason,
            )
            self.log.add(Event(
                self.tick,
                "PRODUCER_DEBT_MATURITY_REPAYMENT_FAILED",
                pool_id=lender_pool.pool_id,
                asset_id=obligation.voucher_id,
                amount=amount_in * stable_value,
                meta={
                    "obligation_id": obligation.obligation_id,
                    "producer_pool_id": producer_pool.pool_id,
                    "reason": receipt.fail_reason,
                },
            ))
            return 0.0

        self.log.add(Event(self.tick, "SWAP_EXECUTED", pool_id=lender_pool.pool_id, meta={"receipt": receipt.to_dict()}))
        gross_voucher_units = float(receipt.amount_out) + float(receipt.fees.total_fee)
        self._update_pool_caches(lender_pool, receipt.asset_in, float(receipt.amount_in))
        self._update_pool_caches(lender_pool, receipt.asset_out, -float(gross_voucher_units))
        self._record_fee_cumulative(receipt)
        self._record_recent_clc_fee(lender_pool, receipt)
        self._record_clc_swap_cumulative(receipt)
        self._noam_update_edge_after_swap(
            lender_pool,
            receipt.asset_in,
            receipt.asset_out,
            float(receipt.amount_in),
            success=True,
        )
        self._noam_routing_swaps_tick += 1
        swap_usd = float(receipt.amount_in) * self._asset_value(lender_pool, receipt.asset_in)
        self._swap_volume_usd_tick += swap_usd
        self._swap_volume_usd_by_pool[lender_pool.pool_id] = (
            self._swap_volume_usd_by_pool.get(lender_pool.pool_id, 0.0) + swap_usd
        )
        self._record_noam_fee_diagnostics(
            lender_pool,
            receipt,
            kind="routing",
            swap_usd=swap_usd,
        )
        exposure_reduced_usd = self._reduce_lender_producer_voucher_exposure(
            lender_pool.pool_id,
            obligation.voucher_id,
            gross_voucher_units * unit_value,
            "producer_debt_maturity_repayment",
        )
        self._record_lender_recovered_stable(
            lender_pool.pool_id,
            swap_usd,
            "producer_debt_maturity_repayment",
            eligible_amount_usd=min(swap_usd, exposure_reduced_usd),
        )

        spec = self.factory.voucher_specs.get(obligation.voucher_id)
        if spec and spec.issuer_id in self.agents and receipt.amount_out > 1e-9:
            issuer_agent = self.agents[spec.issuer_id]
            issuer_pool = self.pools.get(issuer_agent.pool_id)
            issuer_agent.issuer.return_to_issuer(float(receipt.amount_out))
            self._mark_lender_voucher_limits_dirty(obligation.voucher_id)
            if issuer_pool is not None:
                self._vault_add(
                    issuer_pool,
                    obligation.voucher_id,
                    float(receipt.amount_out),
                    "producer_debt_maturity_redeem_receive",
                    lender_pool.pool_id,
                )
                self.log.add(Event(
                    self.tick,
                    "VOUCHER_REDEEMED",
                    actor_id=spec.issuer_id,
                    asset_id=obligation.voucher_id,
                    amount=float(receipt.amount_out),
                    meta={"reason": "producer_debt_maturity_repayment"},
                ))

        repaid_units = min(obligation.remaining_voucher_units, gross_voucher_units)
        repaid_usd = repaid_units * unit_value
        obligation.remaining_voucher_units = max(0.0, obligation.remaining_voucher_units - repaid_units)
        self._producer_debt_repaid_usd_tick += repaid_usd
        self._producer_debt_repaid_usd_total += repaid_usd
        self._producer_debt_repaid_maturity_usd_tick += repaid_usd
        self._producer_debt_repaid_maturity_usd_total += repaid_usd
        self._producer_debt_stable_recovered_usd_tick += repaid_usd
        self._producer_debt_stable_recovered_usd_total += repaid_usd
        self.log.add(Event(
            self.tick,
            "PRODUCER_DEBT_MATURITY_REPAID",
            pool_id=lender_pool.pool_id,
            asset_id=obligation.voucher_id,
            amount=repaid_usd,
            meta={
                "obligation_id": obligation.obligation_id,
                "producer_pool_id": producer_pool.pool_id,
                "repaid_units": repaid_units,
                "stable_in": float(receipt.amount_in),
            },
        ))
        return repaid_units

    def _apply_producer_debt_maturities(self) -> None:
        if not bool(self.cfg.producer_debt_maturity_enabled):
            return
        if not self._producer_debt_obligations:
            return
        recovery_rate = max(0.0, min(1.0, float(self.cfg.producer_debt_maturity_recovery_rate or 0.0)))
        contract_repayment = self._producer_debt_contract_repayment_enabled()
        active: list[ProducerDebtObligation] = []
        for obligation in sorted(
            self._producer_debt_obligations,
            key=self._producer_debt_obligation_sort_key,
        ):
            has_voucher_exposure = obligation.remaining_voucher_units > 1e-9
            has_cash_service = (
                contract_repayment
                and max(0.0, obligation.cash_service_remaining_usd) > 1e-9
            )
            if not has_voucher_exposure and not has_cash_service:
                continue
            if obligation.due_tick > self.tick:
                active.append(obligation)
                continue
            producer_pool = self.pools.get(obligation.producer_pool_id)
            lender_pool = self.pools.get(obligation.lender_pool_id)
            unit_value = self._producer_debt_unit_value(obligation)
            if lender_pool is None or producer_pool is None or unit_value <= 0.0:
                default_usd = max(
                    obligation.remaining_voucher_units
                    * max(unit_value, self._default_asset_value(obligation.voucher_id)),
                    obligation.cash_service_remaining_usd if contract_repayment else 0.0,
                )
                self._producer_debt_defaulted_usd_tick += default_usd
                self._producer_debt_defaulted_usd_total += default_usd
                self.log.add(Event(
                    self.tick,
                    "PRODUCER_DEBT_DEFAULTED",
                    pool_id=obligation.lender_pool_id,
                    asset_id=obligation.voucher_id,
                    amount=default_usd,
                    meta={
                        "obligation_id": obligation.obligation_id,
                        "producer_pool_id": obligation.producer_pool_id,
                        "default_units": obligation.remaining_voucher_units,
                        "reason": "missing_pool",
                    },
                ))
                continue

            if contract_repayment:
                matured_usd = max(
                    obligation.remaining_voucher_units * unit_value,
                    max(0.0, obligation.cash_service_remaining_usd),
                )
                self._producer_debt_matured_usd_tick += matured_usd
                self._producer_debt_matured_usd_total += matured_usd
                recoverable_usd = max(0.0, obligation.cash_service_remaining_usd) * recovery_rate
                if recoverable_usd > 1e-9:
                    self._execute_producer_debt_cash_service_payment(
                        obligation,
                        recoverable_usd,
                        "producer_debt_maturity_repayment",
                    )
                if obligation.cash_service_remaining_usd > 1e-9:
                    self._write_off_producer_debt_cash_default(
                        obligation,
                        obligation.cash_service_remaining_usd,
                        "maturity_unrecovered",
                    )
                if obligation.remaining_voucher_units > 1e-9:
                    self._close_producer_debt_voucher_units(
                        obligation,
                        obligation.remaining_voucher_units,
                        "producer_debt_maturity_cash_service_close",
                    )
                continue

            held_units = min(obligation.remaining_voucher_units, lender_pool.vault.get(obligation.voucher_id))
            if held_units + 1e-9 < obligation.remaining_voucher_units:
                closed_units = obligation.remaining_voucher_units - held_units
                closed_usd = closed_units * unit_value
                self._producer_debt_closed_by_circulation_usd_tick += closed_usd
                self._producer_debt_closed_by_circulation_usd_total += closed_usd
                self._producer_debt_closed_not_held_at_maturity_usd_tick += closed_usd
                self._producer_debt_closed_not_held_at_maturity_usd_total += closed_usd
                self.log.add(Event(
                    self.tick,
                    "PRODUCER_DEBT_CLOSED_NOT_HELD_AT_MATURITY",
                    pool_id=obligation.lender_pool_id,
                    asset_id=obligation.voucher_id,
                    amount=closed_usd,
                    meta={
                        "obligation_id": obligation.obligation_id,
                        "producer_pool_id": obligation.producer_pool_id,
                        "voucher_units": closed_units,
                        "reason": "not_held_at_maturity",
                    },
                ))
                obligation.remaining_voucher_units = held_units
            if obligation.remaining_voucher_units <= 1e-9:
                continue

            matured_usd = obligation.remaining_voucher_units * unit_value
            self._producer_debt_matured_usd_tick += matured_usd
            self._producer_debt_matured_usd_total += matured_usd
            recoverable_units = obligation.remaining_voucher_units * recovery_rate
            if recoverable_units > 1e-9:
                self._execute_producer_debt_maturity_repayment(obligation, recoverable_units, unit_value)
            if obligation.remaining_voucher_units > 1e-9:
                self._write_off_producer_debt_default(
                    lender_pool,
                    obligation,
                    obligation.remaining_voucher_units,
                    unit_value,
                    "maturity_unrecovered",
                )
        self._producer_debt_obligations = active
        self._rebuild_producer_debt_obligation_index()

    def _apply_producer_deposits(self) -> None:
        if not bool(self.cfg.producer_deposits_enabled):
            return
        stride = max(1, int(self.cfg.producer_deposit_stride_ticks or 1))
        if self.tick % stride != 0:
            return
        stable_rate = max(0.0, float(self.cfg.producer_stable_deposit_rate_per_month or 0.0))
        voucher_rate = max(0.0, float(self.cfg.producer_voucher_deposit_rate_per_month or 0.0))
        if stable_rate <= 0.0 and voucher_rate <= 0.0:
            return
        stable_id = self.cfg.stable_symbol
        updated_vouchers: Set[str] = set()
        for pool in self.pools.values():
            if pool.policy.system_pool or pool.policy.role != "producer":
                continue
            agent = self.agents.get(pool.steward_id)
            if agent is None:
                continue
            voucher_id = agent.voucher_spec.voucher_id
            base_value = max(self._pool_total_value(pool), float(self.cfg.random_request_amount_mean or 1.0))
            stable_amount = (base_value * stable_rate / float(MONTH_TICKS)) * stride
            if stable_amount > 1e-9:
                self._deposit_producer_stable_with_lenders(
                    producer_pool=pool,
                    agent=agent,
                    voucher_id=voucher_id,
                    stable_value_usd=stable_amount,
                    source="producer_stable_deposit",
                )
            voucher_value = (base_value * voucher_rate / float(MONTH_TICKS)) * stride
            if voucher_value > 1e-9:
                deposited = self._deposit_producer_voucher_with_lenders(
                    producer_pool=pool,
                    agent=agent,
                    voucher_id=voucher_id,
                    voucher_value_usd=voucher_value,
                    source="producer_voucher_deposit",
                )
                if deposited:
                    updated_vouchers.add(voucher_id)
        if updated_vouchers:
            self._refresh_lender_voucher_limits(updated_vouchers)
            self.rebuild_indexes()

    def _apply_historical_voucher_backing(self) -> None:
        target_tick = self.cfg.historical_voucher_backing_tick
        if target_tick is None or self.tick != int(target_tick):
            return
        total = max(0.0, float(self.cfg.historical_voucher_backing_total_usd or 0.0))
        if total <= 1e-9:
            return

        producer_entries: list[tuple[Pool, Agent, str, float]] = []
        for pool in self.pools.values():
            if pool.policy.system_pool or pool.policy.role != "producer":
                continue
            agent = self.agents.get(pool.steward_id)
            if agent is None:
                continue
            voucher_id = agent.voucher_spec.voucher_id
            weight = max(self._pool_total_value(pool), float(self.cfg.random_request_amount_mean or 1.0))
            producer_entries.append((pool, agent, voucher_id, weight))
        if not producer_entries:
            return

        weight_total = sum(weight for *_rest, weight in producer_entries)
        if weight_total <= 1e-9:
            weight_total = float(len(producer_entries))
            producer_entries = [
                (pool, agent, voucher_id, 1.0)
                for pool, agent, voucher_id, _weight in producer_entries
            ]

        updated_vouchers: Set[str] = set()
        for pool, agent, voucher_id, weight in producer_entries:
            voucher_value = total * (weight / weight_total)
            if voucher_value <= 1e-9:
                continue
            deposited = self._deposit_producer_voucher_with_lenders(
                producer_pool=pool,
                agent=agent,
                voucher_id=voucher_id,
                voucher_value_usd=voucher_value,
                source="historical_voucher_backing",
            )
            if deposited:
                updated_vouchers.add(voucher_id)
        if updated_vouchers:
            self._refresh_lender_voucher_limits(updated_vouchers)
            self.rebuild_indexes()

    def _apply_historical_stable_backing(self) -> None:
        target_tick = self.cfg.historical_stable_backing_tick
        if target_tick is None or self.tick != int(target_tick):
            return
        total = max(0.0, float(self.cfg.historical_stable_backing_total_usd or 0.0))
        if total <= 1e-9:
            return

        role_filter = {
            str(role)
            for role in (self.cfg.historical_stable_backing_roles or ())
            if str(role)
        }
        eligible = [
            pool
            for pool in self.pools.values()
            if not pool.policy.system_pool
            and (not role_filter or pool.policy.role in role_filter)
        ]
        if not eligible:
            return

        amount_per_pool = total / float(len(eligible))
        stable_id = self.cfg.stable_symbol
        role_usd: Dict[str, float] = {}
        role_pools: Dict[str, int] = {}
        total_applied = 0.0
        for pool in eligible:
            if amount_per_pool <= 1e-9:
                continue
            self._vault_add(pool, stable_id, amount_per_pool, "historical_stable_backing", "historical")
            role = str(pool.policy.role)
            role_usd[role] = role_usd.get(role, 0.0) + amount_per_pool
            role_pools[role] = role_pools.get(role, 0) + 1
            total_applied += amount_per_pool

        if total_applied <= 1e-9:
            return
        self._stable_onramp_usd_tick += total_applied
        self._historical_stable_backing_usd_tick += total_applied
        self._historical_stable_backing_usd_total += total_applied
        self._historical_stable_backing_pools_tick += len(eligible)
        self._historical_stable_backing_pools_total += len(eligible)
        for role, value in role_usd.items():
            self._historical_stable_backing_usd_by_role[role] = (
                self._historical_stable_backing_usd_by_role.get(role, 0.0) + value
            )
        for role, count in role_pools.items():
            self._historical_stable_backing_pools_by_role[role] = (
                self._historical_stable_backing_pools_by_role.get(role, 0) + count
            )
        self.log.add(Event(
            self.tick,
            "HISTORICAL_STABLE_BACKING",
            asset_id=stable_id,
            amount=total_applied,
            meta={"roles": role_usd, "recipient_pools": len(eligible)},
        ))

    def _apply_stable_growth_per_pool(self, multiplier: int = 1) -> None:
        scale = max(1, int(multiplier or 1))
        onramp_total = 0.0
        offramp_total = 0.0
        for p in self.pools.values():
            if p.policy.system_pool or p.policy.role == "lender":
                continue
            inflow = 0.0
            base = p.vault.get(self.cfg.stable_symbol) + self._pool_voucher_value_usd(p)
            if p.policy.role == "producer":
                inflow = base * float(self.cfg.producer_inflow_per_tick or 0.0)
            elif p.policy.role == "consumer":
                inflow = base * float(self.cfg.producer_inflow_per_tick or 0.0)
            elif p.policy.role == "lender":
                inflow = 0.0
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
        if onramp_total > 0.0:
            self._stable_onramp_usd_tick += onramp_total
        if offramp_total > 0.0:
            self._stable_offramp_usd_tick += offramp_total

    def _apply_liquidity_provider_contributions(self) -> None:
        if not self.cfg.economics_enabled:
            return
        if not self.cfg.sclc_symbol:
            return
        pending = {
            pid: due for pid, due in self._lp_pending_contribution_tick.items()
            if self.tick >= due
        }
        if not pending:
            return
        remaining_sclc = self._lp_sclc_remaining()
        stable_id = self.cfg.stable_symbol
        sclc_id = self.cfg.sclc_symbol
        total = 0.0
        pool_count = 0
        for pid in list(pending.keys()):
            p = self.pools.get(pid)
            if p is None or p.policy.system_pool or p.policy.role != "liquidity_provider":
                self._lp_pending_contribution_tick.pop(pid, None)
                continue
            stable = p.vault.get(stable_id)
            contrib = stable
            if remaining_sclc is not None:
                if remaining_sclc <= 1e-9:
                    contrib = 0.0
                if contrib > remaining_sclc:
                    contrib = remaining_sclc
            if contrib <= 1e-9:
                self._lp_pending_contribution_tick.pop(pid, None)
                continue
            if not self._vault_sub(p, stable_id, contrib, "lp_waterfall_out", "waterfall"):
                self._lp_pending_contribution_tick.pop(pid, None)
                continue
            self._waterfall_external_inflows[stable_id] = (
                self._waterfall_external_inflows.get(stable_id, 0.0) + contrib
            )
            self._lp_injected_usd_by_pool[p.pool_id] = self._lp_injected_usd_by_pool.get(p.pool_id, 0.0) + contrib
            self._lp_injected_usd_total += contrib
            if p.values.get_value(sclc_id) <= 0.0:
                p.values.set_value(sclc_id, 1.0)
            self._vault_add(p, sclc_id, contrib, "sclc_mint", "waterfall")
            self._sclc_minted_total += contrib
            if remaining_sclc is not None:
                remaining_sclc = max(0.0, remaining_sclc - contrib)
            total += contrib
            pool_count += 1
            self._lp_pending_contribution_tick.pop(pid, None)
        if total > 0.0:
            self.log.add(Event(
                self.tick,
                "LP_WATERFALL_CONTRIBUTED",
                amount=total,
                meta={"pools": pool_count, "one_shot": True},
            ))

    def _issuer_schedule_due_at_tick(self, tick: Optional[int] = None) -> float:
        principal = max(0.0, float(self.cfg.bond_gross_principal_usd or 0.0))
        if principal <= 1e-9:
            return 0.0
        term_ticks = max(1, int(self.cfg.bond_term_ticks or 1))
        stride = max(1, int(self.cfg.issuer_payment_stride_ticks or 13))
        coupon_annual = max(0.0, float(self.cfg.bond_coupon_target_annual or 0.0))
        year_ticks = 52.0
        total_periods = max(1, math.ceil(term_ticks / stride))
        schedule_tick = self.tick if tick is None else max(0, int(tick))
        periods_elapsed = min(total_periods, schedule_tick // stride)
        if schedule_tick >= term_ticks:
            periods_elapsed = total_periods
        if periods_elapsed <= 0:
            return 0.0
        principal_step = principal / total_periods
        coupon_due = 0.0
        principal_due = 0.0
        previous_tick = 0
        for period in range(1, periods_elapsed + 1):
            payment_tick = min(term_ticks, period * stride)
            duration = max(0, payment_tick - previous_tick)
            outstanding_start = max(0.0, principal - principal_step * (period - 1))
            coupon_due += outstanding_start * coupon_annual * (duration / year_ticks)
            principal_due += principal_step
            previous_tick = payment_tick
        return coupon_due + min(principal, principal_due)

    def _next_issuer_service_due_target(self) -> float:
        principal = max(0.0, float(self.cfg.bond_gross_principal_usd or 0.0))
        if principal <= 1e-9:
            return 0.0
        term_ticks = max(1, int(self.cfg.bond_term_ticks or 1))
        stride = max(1, int(self.cfg.issuer_payment_stride_ticks or 13))
        if self.tick >= term_ticks:
            target_tick = term_ticks
        else:
            periods_elapsed = self.tick // stride
            if self.tick % stride == 0 and self.tick > 0:
                target_tick = min(term_ticks, periods_elapsed * stride)
            else:
                target_tick = min(term_ticks, (periods_elapsed + 1) * stride)
        return self._issuer_schedule_due_at_tick(target_tick)

    def _bond_service_lockbox_mode(self) -> str:
        mode = str(getattr(self.cfg, "bond_service_lockbox_mode", "next_due") or "next_due")
        mode = mode.strip().lower()
        if mode not in {"next_due", "remaining_schedule"}:
            return "next_due"
        return mode

    def _bond_service_lockbox_coverage_ratio(self) -> float:
        return max(
            0.0,
            float(getattr(self.cfg, "bond_service_lockbox_coverage_ratio", 1.0) or 0.0),
        )

    def _bond_service_lockbox_target_usd(self) -> float:
        mode = self._bond_service_lockbox_mode()
        if mode == "remaining_schedule":
            base_target = self._issuer_schedule_due_at_tick(int(self.cfg.bond_term_ticks or 0))
        else:
            base_target = self._next_issuer_service_due_target()
        return max(0.0, base_target * self._bond_service_lockbox_coverage_ratio())

    def _bond_service_remaining_lockbox_need_usd(self) -> float:
        return max(
            0.0,
            self._bond_service_lockbox_target_usd()
            - float(self._lp_returned_usd_total or 0.0)
            - float(self._bond_service_reserve_usd_balance or 0.0),
        )

    def _bond_fee_service_share(self) -> float:
        return max(0.0, min(1.0, float(self.cfg.bond_fee_service_share or 0.0)))

    def _lender_recovered_stable_pending_usd_total(self) -> float:
        return sum(max(0.0, float(amount or 0.0)) for amount in self._lender_recovered_stable_by_pool.values())

    def _lender_recovered_stable_sweepable_pending_usd_total(self) -> float:
        stable_id = self.cfg.stable_symbol
        total = 0.0
        for pool_id, pending in self._lender_recovered_stable_by_pool.items():
            if pending <= 1e-9:
                continue
            pool = self.pools.get(pool_id)
            if pool is None or pool.policy.system_pool or pool.policy.role != "lender":
                continue
            surplus = max(0.0, pool.vault.get(stable_id) - pool.policy.min_stable_reserve)
            total += min(float(pending), surplus)
        return total

    def _reserve_lender_recovered_stable_for_bond_service(
        self,
        pool: Pool,
        amount_usd: float,
        reason: str,
    ) -> float:
        if not bool(self.cfg.bond_service_reserve_enabled):
            return 0.0
        if str(self.cfg.bond_return_mode or "") != "issuer_cashflow":
            return 0.0
        if amount_usd <= 1e-9 or pool.policy.role != "lender":
            return 0.0
        share = max(0.0, min(1.0, float(self.cfg.bond_service_reserve_recovery_share or 0.0)))
        if share <= 1e-9:
            return 0.0
        target_due = self._bond_service_lockbox_target_usd()
        remaining_need = self._bond_service_remaining_lockbox_need_usd()
        if remaining_need <= 1e-9:
            return 0.0
        stable_id = self.cfg.stable_symbol
        amount = min(float(amount_usd) * share, remaining_need, pool.vault.get(stable_id))
        if amount <= 1e-9:
            return 0.0
        if not self._vault_sub(pool, stable_id, amount, "bond_service_reserve_out", "issuer_service_reserve"):
            return 0.0
        self._bond_service_reserve_usd_balance += amount
        self._bond_service_reserved_usd_tick += amount
        self._bond_service_reserved_usd_total += amount
        self._bond_service_reserved_by_pool[pool.pool_id] = (
            self._bond_service_reserved_by_pool.get(pool.pool_id, 0.0) + amount
        )
        self._stable_offramp_usd_tick += amount
        self.log.add(Event(
            self.tick,
            "BOND_SERVICE_RESERVED",
            pool_id=pool.pool_id,
            amount=amount,
            meta={
                "reason": reason,
                "target_due": target_due,
                "remaining_need_before": remaining_need,
                "lockbox_mode": self._bond_service_lockbox_mode(),
                "lockbox_coverage_ratio": self._bond_service_lockbox_coverage_ratio(),
            },
        ))
        return amount

    def _reserve_fee_service_cash_for_bond_service(
        self,
        amount_usd: float,
        source: str,
        source_pool: Optional[Pool] = None,
    ) -> float:
        if not bool(self.cfg.bond_service_reserve_enabled):
            return 0.0
        if str(self.cfg.bond_return_mode or "") != "issuer_cashflow":
            return 0.0
        if amount_usd <= 1e-9:
            return 0.0
        share = self._bond_fee_service_share()
        if share <= 1e-9:
            return 0.0
        target_due = self._bond_service_lockbox_target_usd()
        remaining_need = self._bond_service_remaining_lockbox_need_usd()
        if remaining_need <= 1e-9:
            return 0.0
        stable_id = self.cfg.stable_symbol
        available = float(amount_usd)
        if source_pool is not None:
            available = min(available, source_pool.vault.get(stable_id))
        amount = min(float(amount_usd) * share, remaining_need, available)
        if amount <= 1e-9:
            return 0.0
        if source_pool is not None:
            if not self._vault_sub(
                source_pool,
                stable_id,
                amount,
                "fee_service_reserve_out",
                "issuer_service_reserve",
            ):
                return 0.0
        self._bond_service_reserve_usd_balance += amount
        self._bond_service_reserved_usd_tick += amount
        self._bond_service_reserved_usd_total += amount
        self._bond_service_reserved_by_pool["fee_service"] = (
            self._bond_service_reserved_by_pool.get("fee_service", 0.0) + amount
        )
        self._fee_service_reserved_usd_tick += amount
        self._fee_service_reserved_usd_total += amount
        if source == "converted_voucher_fee":
            self._fee_service_converted_voucher_reserved_usd_tick += amount
            self._fee_service_converted_voucher_reserved_usd_total += amount
        else:
            self._fee_service_stable_reserved_usd_tick += amount
            self._fee_service_stable_reserved_usd_total += amount
        self._stable_offramp_usd_tick += amount
        self.log.add(Event(
            self.tick,
            "FEE_SERVICE_RESERVED",
            pool_id=source_pool.pool_id if source_pool is not None else None,
            amount=amount,
            meta={
                "source": source,
                "fee_service_share": share,
                "target_due": target_due,
                "remaining_need_before": remaining_need,
                "lockbox_mode": self._bond_service_lockbox_mode(),
                "lockbox_coverage_ratio": self._bond_service_lockbox_coverage_ratio(),
            },
        ))
        return amount

    def _pay_bond_service_from_reserve(self) -> float:
        if not bool(self.cfg.bond_service_reserve_enabled):
            return 0.0
        if str(self.cfg.bond_return_mode or "") != "issuer_cashflow":
            return 0.0
        scheduled_due = self._issuer_schedule_due_at_tick()
        remaining_need = max(0.0, scheduled_due - float(self._lp_returned_usd_total or 0.0))
        amount = min(float(self._bond_service_reserve_usd_balance or 0.0), remaining_need)
        if amount <= 1e-9:
            return 0.0
        self._bond_service_reserve_usd_balance -= amount
        self._bond_service_paid_from_reserve_usd_tick += amount
        self._bond_service_paid_from_reserve_usd_total += amount
        self._lp_returned_usd_by_pool["bond_service_reserve"] = (
            self._lp_returned_usd_by_pool.get("bond_service_reserve", 0.0) + amount
        )
        self._lp_returned_usd_total += amount
        self.log.add(Event(
            self.tick,
            "BOND_SERVICE_RESERVE_PAID",
            amount=amount,
            meta={"scheduled_due": scheduled_due, "remaining_need_before": remaining_need},
        ))
        return amount

    def _apply_quarterly_clearing(self) -> None:
        if not bool(self.cfg.quarterly_clearing_enabled):
            return
        if not self.clc_pool_id or self.clc_pool_id not in self.pools:
            return
        stride = max(1, int(self.cfg.quarterly_clearing_stride_ticks or 13))
        if self.tick % stride != 0:
            return
        self._pay_bond_service_from_reserve()
        scheduled_due = self._issuer_schedule_due_at_tick()
        remaining_need = max(0.0, scheduled_due - float(self._lp_returned_usd_total or 0.0))
        if remaining_need <= 1e-9:
            return
        share = max(0.0, min(1.0, float(self.cfg.quarterly_clearing_surplus_share or 0.0)))
        if share <= 0.0:
            return
        clc_pool = self.pools[self.clc_pool_id]
        stable_id = self.cfg.stable_symbol
        lender_ids = [
            pid for pid, amount in self._lender_recovered_stable_by_pool.items()
            if amount > 1e-9 and pid in self.pools and self.pools[pid].policy.role == "lender"
        ]
        lender_ids.sort()
        if not lender_ids:
            return
        direct_issuer_cashflow = str(self.cfg.bond_return_mode or "") == "issuer_cashflow"
        before_liquidity = sum(self.pools[pid].vault.get(stable_id) for pid in lender_ids)
        cleared = 0.0
        pools_cleared = 0
        for pid in lender_ids:
            if remaining_need <= 1e-9:
                break
            lender = self.pools[pid]
            eligible = self._lender_recovered_stable_by_pool.get(pid, 0.0) * share
            surplus = max(0.0, lender.vault.get(stable_id) - lender.policy.min_stable_reserve)
            amount = min(eligible, surplus, remaining_need)
            if amount <= 1e-9:
                continue
            if not self._vault_sub(lender, stable_id, amount, "quarterly_clearing_out", clc_pool.pool_id):
                continue
            if direct_issuer_cashflow:
                self._lp_returned_usd_by_pool["issuer_cashflow"] = (
                    self._lp_returned_usd_by_pool.get("issuer_cashflow", 0.0) + amount
                )
                self._lp_returned_usd_total += amount
                self._stable_offramp_usd_tick += amount
            else:
                self._vault_add(clc_pool, stable_id, amount, "quarterly_clearing_in", lender.pool_id)
            self._lender_recovered_stable_by_pool[pid] = max(
                0.0,
                self._lender_recovered_stable_by_pool.get(pid, 0.0) - amount,
            )
            remaining_need -= amount
            cleared += amount
            pools_cleared += 1
        if cleared <= 1e-9:
            return
        after_liquidity = sum(self.pools[pid].vault.get(stable_id) for pid in lender_ids)
        self._quarterly_clearing_usd_tick += cleared
        self._quarterly_clearing_usd_total += cleared
        self._quarterly_clearing_lender_liquidity_before_tick = before_liquidity
        self._quarterly_clearing_lender_liquidity_after_tick = after_liquidity
        self.log.add(Event(
            self.tick,
            "QUARTERLY_CLEARING_EXECUTED",
            pool_id=clc_pool.pool_id,
            asset_id=stable_id,
            amount=cleared,
            meta={
                "pools": pools_cleared,
                "scheduled_due": scheduled_due,
                "remaining_need_after": remaining_need,
                "lender_liquidity_before": before_liquidity,
                "lender_liquidity_after": after_liquidity,
                "direct_issuer_cashflow": direct_issuer_cashflow,
            },
        ))

    def _apply_lp_sclc_redemptions(self) -> None:
        if not self.cfg.economics_enabled:
            return
        if not self.clc_pool_id or not self.cfg.sclc_symbol:
            return
        clc_pool = self.pools.get(self.clc_pool_id)
        if clc_pool is None:
            return
        stable_id = self.cfg.stable_symbol
        sclc_id = self.cfg.sclc_symbol
        if not clc_pool.registry.is_listed(stable_id) or not clc_pool.registry.is_listed(sclc_id):
            return

        available_stable = clc_pool.vault.get(stable_id) - clc_pool.policy.min_stable_reserve
        if available_stable <= 1e-9:
            return

        lp_pools = [
            p for p in self.pools.values()
            if not p.policy.system_pool and p.policy.role == "liquidity_provider"
        ]
        if not lp_pools:
            return
        total_sclc = sum(p.vault.get(sclc_id) for p in lp_pools)
        if total_sclc <= 1e-9:
            return

        remaining_budget = available_stable
        swaps = 0
        for p in lp_pools:
            if remaining_budget <= 1e-9:
                break
            lp_sclc = p.vault.get(sclc_id)
            if lp_sclc <= 1e-9:
                continue
            alloc = available_stable * (lp_sclc / total_sclc)
            amount_in = min(lp_sclc, alloc, remaining_budget)
            if amount_in <= 1e-9:
                continue
            if not self._vault_sub(p, sclc_id, amount_in, "lp_sclc_exit", clc_pool.pool_id):
                continue

            receipt = clc_pool.execute_swap(
                self.tick,
                actor=f"lp_exit:{p.pool_id}",
                asset_in=sclc_id,
                amount_in=amount_in,
                asset_out=stable_id,
            )
            if receipt.status != "executed":
                self._vault_add(p, sclc_id, amount_in, "lp_sclc_refund", clc_pool.pool_id)
                continue

            self._vault_add(p, stable_id, receipt.amount_out, "lp_sclc_payout", clc_pool.pool_id)
            self._lp_returned_usd_by_pool[p.pool_id] = (
                self._lp_returned_usd_by_pool.get(p.pool_id, 0.0) + receipt.amount_out
            )
            self._lp_returned_usd_total += receipt.amount_out

            self.log.add(Event(self.tick, "SWAP_EXECUTED", pool_id=clc_pool.pool_id,
                               meta={"receipt": receipt.to_dict(), "lp_pool_id": p.pool_id, "lp_exit": True}))

            gross_out = receipt.amount_out + float(receipt.fees.total_fee)
            self._update_pool_caches(clc_pool, receipt.asset_in, float(receipt.amount_in))
            self._update_pool_caches(clc_pool, receipt.asset_out, -float(gross_out))
            self._record_fee_cumulative(receipt)
            self._record_recent_clc_fee(clc_pool, receipt)
            self._record_clc_swap_cumulative(receipt)
            self._noam_update_edge_after_swap(
                clc_pool,
                receipt.asset_in,
                receipt.asset_out,
                float(receipt.amount_in),
                success=True,
            )
            swap_usd = receipt.amount_in * self._asset_value(clc_pool, receipt.asset_in)
            self._swap_volume_usd_tick += swap_usd
            self._swap_volume_usd_by_pool[clc_pool.pool_id] = (
                self._swap_volume_usd_by_pool.get(clc_pool.pool_id, 0.0) + swap_usd
            )
            remaining_budget -= receipt.amount_out
            swaps += 1

        if swaps > 0:
            self.log.add(Event(self.tick, "LP_SCLC_REDEEMED", amount=available_stable - remaining_budget,
                               meta={"swaps": swaps, "capacity": available_stable}))

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
        weights: Dict[str, float] = {}
        total_weight = 0.0
        deficit_weights: Dict[str, float] = {}
        total_deficit = 0.0
        for pid, p in self.pools.items():
            if p.policy.system_pool or p.policy.role in ("liquidity_provider", "lender"):
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
            activity_weights = {
                pid: usd for pid, usd in self._recent_pool_activity(window).items()
                if pid in self.pools and not self.pools[pid].policy.system_pool
                and self.pools[pid].policy.role not in ("liquidity_provider", "lender")
            }
            total_activity = sum(activity_weights.values())

        if total_deficit > 1e-9 and total_activity > 1e-9 and activity_share > 0.0:
            for pid in self.pools:
                if self.pools[pid].policy.system_pool or self.pools[pid].policy.role in ("liquidity_provider", "lender"):
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
                if p.policy.system_pool or p.policy.role in ("liquidity_provider", "lender"):
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

    def _apply_stable_outflow(self, amount: float) -> None:
        if amount <= 1e-9 or not self.pools:
            return
        stable_id = self.cfg.stable_symbol
        available: Dict[str, float] = {}
        total_available = 0.0
        for pid, p in self.pools.items():
            if p.policy.system_pool or p.policy.role in ("liquidity_provider", "lender"):
                continue
            excess = max(0.0, p.vault.get(stable_id) - p.policy.min_stable_reserve)
            if excess > 1e-9:
                available[pid] = excess
                total_available += excess
        if total_available <= 1e-9:
            for pid, p in self.pools.items():
                if p.policy.system_pool or p.policy.role in ("liquidity_provider", "lender"):
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
            if not p.policy.system_pool and p.policy.role not in ("liquidity_provider", "lender")
        )

        outflow_rate = float(self.cfg.stable_outflow_rate or 0.0)
        if outflow_rate > 0.0 and total_stable > 1e-9:
            outflow = total_stable * (outflow_rate / float(MONTH_TICKS)) * scale
            self._apply_stable_outflow(outflow)
            total_stable = sum(
                p.vault.get(stable_id)
                for p in self.pools.values()
                if not p.policy.system_pool and p.policy.role not in ("liquidity_provider", "lender")
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
            if pool is None or pool.policy.system_pool or pool.policy.role in ("liquidity_provider", "lender"):
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

    def _apply_monthly_pool_offramp(self) -> None:
        if not bool(self.cfg.offramps_enabled):
            return
        rate_producer = max(0.0, float(self.cfg.producer_offramp_rate_per_month or 0.0))
        rate_consumer = max(0.0, float(self.cfg.consumer_offramp_rate_per_month or 0.0))
        if self.cfg.stable_growth_mode == "per_pool":
            def _effective_monthly_rate(r: float) -> float:
                if r <= 0.0:
                    return 0.0
                step = r / float(MONTH_TICKS)
                return max(0.0, (1.0 + step) ** float(MONTH_TICKS) - 1.0)
            rate_producer = _effective_monthly_rate(float(self.cfg.producer_inflow_per_tick or 0.0))
            rate_consumer = _effective_monthly_rate(float(self.cfg.producer_inflow_per_tick or 0.0))
        if rate_producer <= 0.0 and rate_consumer <= 0.0:
            return
        stable_id = self.cfg.stable_symbol
        total_offramp = 0.0
        producer_offramp = 0.0
        consumer_offramp = 0.0
        for pool in self.pools.values():
            if pool.policy.system_pool or pool.policy.role not in ("producer", "consumer"):
                continue
            if pool.policy.role == "producer":
                rate = rate_producer
            elif pool.policy.role == "consumer":
                rate = rate_consumer
            if rate <= 0.0:
                continue
            available = pool.vault.get(stable_id)
            if available <= 1e-9:
                continue
            reserve = max(0.0, float(pool.policy.min_stable_reserve or 0.0))
            available = max(0.0, available - reserve)
            if available <= 1e-9:
                continue
            amount = available * rate
            if amount <= 1e-9:
                continue
            if self._vault_sub(pool, stable_id, amount, "offramp_monthly", "fiat"):
                self._stable_offramp_usd_tick += amount
                total_offramp += amount
                if pool.policy.role == "producer":
                    producer_offramp += amount
                elif pool.policy.role == "consumer":
                    consumer_offramp += amount
        if total_offramp > 0.0:
            self.log.add(Event(
                self.tick,
                "MONTHLY_OFFRAMP_APPLIED",
                amount=total_offramp,
                meta={"producer_usd": producer_offramp, "consumer_usd": consumer_offramp},
            ))

    def _apply_monthly_offramp_balancer(self) -> None:
        if not bool(self.cfg.offramps_enabled):
            return
        net_onramp = self._stable_onramp_usd_month - self._stable_offramp_usd_month
        if net_onramp <= 1e-9:
            return
        stable_id = self.cfg.stable_symbol
        eligible: list[Tuple["Pool", float]] = []
        total_available = 0.0
        for pool in self.pools.values():
            if pool.policy.system_pool or pool.policy.role not in ("producer", "consumer"):
                continue
            available = pool.vault.get(stable_id)
            reserve = max(0.0, float(pool.policy.min_stable_reserve or 0.0))
            available = max(0.0, available - reserve)
            if available <= 1e-9:
                continue
            eligible.append((pool, available))
            total_available += available
        if total_available <= 1e-9:
            return
        target = min(net_onramp, total_available)
        total_offramp = 0.0
        for pool, available in eligible:
            share = available / total_available
            amount = target * share
            if amount <= 1e-9:
                continue
            if self._vault_sub(pool, stable_id, amount, "offramp_balance", "fiat"):
                self._stable_offramp_usd_tick += amount
                self._stable_offramp_usd_month += amount
                total_offramp += amount
        if total_offramp > 0.0:
            self.log.add(Event(
                self.tick,
                "MONTHLY_OFFRAMP_BALANCED",
                amount=total_offramp,
                meta={"net_onramp_usd": net_onramp, "used_usd": total_offramp},
            ))

    def _sweep_pool_stable_excess(self, pool: "Pool", action: str = "stable_excess_sweep") -> float:
        stable_id = self.cfg.stable_symbol
        buffer_share = max(0.0, float(self.cfg.stable_excess_sweep_buffer_voucher_share or 0.0))
        stable_available = pool.vault.get(stable_id)
        if stable_available <= 1e-9:
            return 0.0
        reserve = max(0.0, float(pool.policy.min_stable_reserve or 0.0))
        working_buffer = self._pool_voucher_value_usd(pool) * buffer_share
        preserved = reserve + working_buffer
        amount = max(0.0, stable_available - preserved)
        if amount <= 1e-9:
            return 0.0
        if not self._vault_sub(pool, stable_id, amount, action, "fiat"):
            return 0.0
        self._stable_offramp_usd_tick += amount
        self._stable_offramp_usd_month += amount
        self._stable_excess_sweep_usd_tick += amount
        self._stable_excess_sweep_usd_total += amount
        self._stable_excess_sweep_pools_tick += 1
        self._stable_excess_sweep_pools_total += 1
        return amount

    def _apply_stable_excess_sweep(self) -> None:
        if not bool(self.cfg.stable_excess_sweep_enabled):
            return
        role_filter = {
            str(role)
            for role in (self.cfg.stable_excess_sweep_roles or ())
            if str(role)
        }
        total_swept = 0.0
        pools_swept = 0
        role_usd: Dict[str, float] = {}
        for pool in self.pools.values():
            if pool.policy.system_pool:
                continue
            if role_filter and pool.policy.role not in role_filter:
                continue
            amount = self._sweep_pool_stable_excess(pool)
            if amount <= 1e-9:
                continue
            total_swept += amount
            pools_swept += 1
            role = str(pool.policy.role)
            role_usd[role] = role_usd.get(role, 0.0) + amount
        if total_swept > 0.0:
            self.log.add(Event(
                self.tick,
                "STABLE_EXCESS_SWEEP",
                asset_id=self.cfg.stable_symbol,
                amount=total_swept,
                meta={"roles": role_usd, "recipient_pools": pools_swept},
            ))

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
            # Always compute fee totals for KPIs.
            for asset_id, amt in p.fee_ledger_clc.items():
                if amt <= 1e-12:
                    continue
                value = self._asset_value(p, asset_id)
                clc_fee_usd += amt * value
            for asset_id, amt in p.fee_ledger_pool.items():
                if amt <= 1e-12:
                    continue
                value = self._asset_value(p, asset_id)
                pool_fee_usd += amt * value

            # For waterfall inflows, use either pool fees or CLC fees (not both).
            inflow_ledger = p.fee_ledger_pool if cfg.waterfall_include_pool_fees else p.fee_ledger_clc
            for asset_id, amt in inflow_ledger.items():
                if amt <= 1e-12:
                    continue
                value = self._asset_value(p, asset_id)
                amounts[asset_id] = amounts.get(asset_id, 0.0) + amt
                value_totals[asset_id] = value_totals.get(asset_id, 0.0) + amt * value

            p.fee_ledger_pool.clear()
            p.fee_ledger_clc.clear()
        if self._waterfall_external_inflows:
            stable_id = self.cfg.stable_symbol
            for asset_id, amt in self._waterfall_external_inflows.items():
                if amt <= 1e-12:
                    continue
                value = self._default_asset_value(asset_id)
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

    def _attempt_fee_voucher_conversion(
        self,
        clc_pool: "Pool",
        asset_id: str,
        amount: float,
        avg_value: float,
        remaining_cap_usd: Optional[float],
    ) -> tuple[float, float, float, Optional[float]]:
        """Materialize voucher fees in the CLC pool and try routing them to stable."""
        if amount <= 1e-12 or avg_value <= 0.0:
            return 0.0, 0.0, 0.0, remaining_cap_usd
        if not bool(self.cfg.voucher_fee_conversion_enabled):
            return 0.0, 0.0, amount * avg_value, remaining_cap_usd
        max_swaps = max(0, int(self.cfg.voucher_fee_conversion_max_swaps_per_epoch or 0))
        if max_swaps <= 0:
            return 0.0, 0.0, amount * avg_value, remaining_cap_usd
        min_usd = max(0.0, float(self.cfg.voucher_fee_conversion_min_usd or 0.0))
        asset_usd = amount * avg_value
        convert_usd = asset_usd
        if remaining_cap_usd is not None:
            convert_usd = min(convert_usd, max(0.0, remaining_cap_usd))
        if convert_usd < min_usd:
            self._deposit_fee_asset_to_pool(clc_pool, asset_id, amount, avg_value, "fee_kind")
            return 0.0, 0.0, asset_usd, remaining_cap_usd
        convert_amount = min(amount, convert_usd / avg_value)
        remainder_amount = max(0.0, amount - convert_amount)
        if convert_amount > 1e-12:
            self._deposit_fee_asset_to_pool(clc_pool, asset_id, convert_amount, avg_value, "fee_convert_source")
        if remainder_amount > 1e-12:
            self._deposit_fee_asset_to_pool(clc_pool, asset_id, remainder_amount, avg_value, "fee_kind")

        attempted_usd = convert_amount * avg_value
        self._fee_conversion_attempted_usd_tick += attempted_usd
        self._fee_conversion_attempted_usd_total += attempted_usd
        success_usd = 0.0
        failed_usd = attempted_usd
        swaps = 0
        amount_remaining = convert_amount
        stable_id = self.cfg.stable_symbol
        while swaps < max_swaps and amount_remaining * avg_value >= min_usd:
            amount_in = amount_remaining / max(1, max_swaps - swaps)
            before_stable = clc_pool.vault.get(stable_id)
            plan, amount_used, used_fallback = self._find_route_with_fallback(
                tick=self.tick,
                start_asset=asset_id,
                target_asset=stable_id,
                amount_in=amount_in,
                source_pool=clc_pool,
            )
            self.log.add(Event(
                self.tick,
                "FEE_CONVERSION_REQUESTED",
                pool_id=clc_pool.pool_id,
                asset_id=asset_id,
                amount=amount_used * avg_value,
                meta={"target_asset": stable_id, "fallback": used_fallback},
            ))
            if not plan.ok:
                break
            ok = self.execute_route_from_pool(
                clc_pool.pool_id, plan, amount_used, route_context="fee_conversion"
            )
            if not ok:
                break
            after_stable = clc_pool.vault.get(stable_id)
            gained = max(0.0, after_stable - before_stable)
            success_usd += gained
            amount_remaining = max(0.0, amount_remaining - amount_used)
            swaps += 1
        failed_usd = max(0.0, attempted_usd - success_usd)
        self._fee_conversion_success_usd_tick += success_usd
        self._fee_conversion_failed_usd_tick += failed_usd
        self._fee_conversion_success_usd_total += success_usd
        self._fee_conversion_failed_usd_total += failed_usd
        self.log.add(Event(
            self.tick,
            "FEE_CONVERSION_SUMMARY",
            pool_id=clc_pool.pool_id,
            asset_id=asset_id,
            amount=success_usd,
            meta={"attempted_usd": attempted_usd, "failed_usd": failed_usd, "swaps": swaps},
        ))
        if remaining_cap_usd is not None:
            remaining_cap_usd = max(0.0, remaining_cap_usd - attempted_usd)
        kind_usd = failed_usd + (remainder_amount * avg_value)
        return attempted_usd, success_usd, kind_usd, remaining_cap_usd

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
        stable_id = cfg.stable_symbol

        if cfg.liquidity_mandate_mode == "community_deficit_then_lender":
            distributed = 0.0
            community = [p for p in eligible if p.policy.role in ("producer", "consumer")]
            community_deficits = {
                p.pool_id: max(0.0, p.policy.min_stable_reserve - p.vault.get(stable_id))
                for p in community
            }
            community_deficits = {
                pool_id: deficit
                for pool_id, deficit in community_deficits.items()
                if deficit > 1e-9
            }
            total_deficit = sum(community_deficits.values())
            if total_deficit > 1e-9:
                fill_scale = min(1.0, budget_usd / total_deficit)
                for p in community:
                    deficit = community_deficits.get(p.pool_id, 0.0)
                    if deficit <= 0.0:
                        continue
                    alloc = deficit * fill_scale
                    if alloc <= 1e-9:
                        continue
                    if not self._vault_sub(mandates_pool, stable_id, alloc, "mandate_out", p.pool_id):
                        continue
                    self._vault_add(p, stable_id, alloc, "mandate_in", mandates_pool.pool_id)
                    distributed += alloc

            remaining_budget = max(0.0, budget_usd - distributed)
            lenders = [p for p in eligible if p.policy.role == "lender"]
            if remaining_budget > 1e-9 and lenders:
                weights = {
                    p.pool_id: 1.0 / max(1.0, p.vault.get(stable_id))
                    for p in lenders
                }
                total_weight = sum(weights.values())
                if total_weight > 1e-9:
                    for p in lenders:
                        alloc = remaining_budget * (weights.get(p.pool_id, 0.0) / total_weight)
                        if alloc <= 1e-9:
                            continue
                        if not self._vault_sub(mandates_pool, stable_id, alloc, "mandate_out", p.pool_id):
                            continue
                        self._vault_add(p, stable_id, alloc, "mandate_in", mandates_pool.pool_id)
                        distributed += alloc

            if distributed > 0.0:
                self.log.add(Event(self.tick, "LIQUIDITY_MANDATE_DISTRIBUTED", amount=distributed,
                                   meta={"mode": cfg.liquidity_mandate_mode, "pool_count": len(eligible)}))
                self._mandates_distributed_usd_total += distributed
            return distributed

        weights: Dict[str, float] = {}
        deficits: Dict[str, float] = {}

        if cfg.liquidity_mandate_mode == "lender_liquidity":
            for p in eligible:
                stable = p.vault.get(stable_id)
                deficit = max(0.0, p.policy.min_stable_reserve - stable)
                if deficit > 1e-9:
                    deficits[p.pool_id] = deficit
            if deficits:
                weights = dict(deficits)
            else:
                for p in eligible:
                    stable = max(1.0, p.vault.get(stable_id))
                    weights[p.pool_id] = 1.0 / stable
        elif cfg.liquidity_mandate_mode == "deficit_weighted":
            for p in eligible:
                deficit = max(0.0, p.policy.min_stable_reserve - p.vault.get(stable_id))
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

        # If we have explicit deficits, fill them first (pro‑rata if budget is short).
        if deficits:
            total_deficit = sum(deficits.values())
            if total_deficit > 1e-9:
                fill_scale = min(1.0, budget_usd / total_deficit)
                for p in eligible:
                    deficit = deficits.get(p.pool_id, 0.0)
                    if deficit <= 0.0:
                        continue
                    alloc = deficit * fill_scale
                    if alloc <= 1e-9:
                        continue
                    if not self._vault_sub(mandates_pool, stable_id, alloc, "mandate_out", p.pool_id):
                        continue
                    self._vault_add(p, stable_id, alloc, "mandate_in", mandates_pool.pool_id)
                    distributed += alloc

        remaining_budget = budget_usd - distributed

        # Distribute any remaining budget by weights (inverse stable, utilization, etc.).
        if remaining_budget > 1e-9:
            for p in eligible:
                weight = weights.get(p.pool_id, 0.0)
                if weight <= 0.0:
                    continue
                alloc = remaining_budget * (weight / total_weight)
                if max_per_pool > 0.0:
                    alloc = min(alloc, max_per_pool)
                if alloc <= 1e-9:
                    continue
                if not self._vault_sub(mandates_pool, stable_id, alloc, "mandate_out", p.pool_id):
                    continue
                self._vault_add(p, stable_id, alloc, "mandate_in", mandates_pool.pool_id)
                distributed += alloc

        if distributed > 0.0:
            self.log.add(Event(self.tick, "LIQUIDITY_MANDATE_DISTRIBUTED", amount=distributed,
                               meta={"mode": cfg.liquidity_mandate_mode, "pool_count": len(eligible)}))
            self._mandates_distributed_usd_total += distributed
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
        fee_service_reserved_usd = 0.0
        fee_service_stable_reserved_usd = 0.0
        fee_service_converted_voucher_reserved_usd = 0.0
        fee_converted_voucher_cash_usd = 0.0

        if total_fee_usd <= 1e-9:
            self._waterfall_last = {
                "fee_in_usd": 0.0,
                "fee_cash_usd": 0.0,
                "fee_cash_waterfall_usd": 0.0,
                "fee_kind_usd": 0.0,
                "fee_service_reserved_usd": 0.0,
                "fee_service_stable_reserved_usd": 0.0,
                "fee_service_converted_voucher_reserved_usd": 0.0,
                "fee_converted_voucher_cash_usd": 0.0,
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
            self._mint_sclc_fee_access()
            self._burn_ops_pool_usd(ops_pool)
            return

        stable_id = self.cfg.stable_symbol
        eligible = set(self.cfg.cash_eligible_assets or [])
        slippage = max(0.0, float(self.cfg.cash_conversion_slippage_bps or 0.0)) / 10000.0
        voucher_conversion_cap = self.cfg.voucher_fee_conversion_max_usd_per_epoch
        remaining_voucher_conversion_cap = (
            None if voucher_conversion_cap is None else max(0.0, float(voucher_conversion_cap))
        )
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
                reserved = self._reserve_fee_service_cash_for_bond_service(
                    asset_usd,
                    "stable_fee",
                )
                fee_service_reserved_usd += reserved
                fee_service_stable_reserved_usd += reserved
                cash_usd += max(0.0, asset_usd - reserved)
                continue
            if asset_id.startswith("VCHR:") and bool(self.cfg.voucher_fee_conversion_enabled):
                _, converted_usd, remaining_kind_usd, remaining_voucher_conversion_cap = (
                    self._attempt_fee_voucher_conversion(
                        clc_pool,
                        asset_id,
                        amt,
                        avg_value,
                        remaining_voucher_conversion_cap,
                    )
                )
                reserved = self._reserve_fee_service_cash_for_bond_service(
                    converted_usd,
                    "converted_voucher_fee",
                    source_pool=clc_pool,
                )
                fee_service_reserved_usd += reserved
                fee_service_converted_voucher_reserved_usd += reserved
                converted_for_waterfall = max(0.0, converted_usd - reserved)
                if converted_for_waterfall > 1e-9:
                    converted_for_waterfall = min(converted_for_waterfall, clc_pool.vault.get(stable_id))
                    if converted_for_waterfall > 1e-9 and self._vault_sub(
                        clc_pool,
                        stable_id,
                        converted_for_waterfall,
                        "fee_conversion_cash_waterfall_out",
                        "waterfall",
                    ):
                        cash_usd += converted_for_waterfall
                        fee_converted_voucher_cash_usd += converted_for_waterfall
                kind_usd += remaining_kind_usd
                continue
            if asset_id in eligible:
                convert_amt = amt * conversion_ratio
                convert_usd = convert_amt * avg_value
                if convert_usd > 0.0:
                    conversion_used_usd += convert_usd
                    cash_usd += convert_usd * (1.0 - slippage)
                    # keep converted vouchers in-system (no supply burn)
                    self._deposit_fee_asset_to_pool(clc_pool, asset_id, convert_amt, avg_value, "fee_convert")
                    self._stable_onramp_usd_tick += convert_usd * (1.0 - slippage)
                remain_amt = amt - convert_amt
                if remain_amt > 1e-12:
                    kind_usd += remain_amt * avg_value
                    self._deposit_fee_asset_to_pool(clc_pool, asset_id, remain_amt, avg_value, "fee_kind")
                continue
            kind_usd += asset_usd
            self._deposit_fee_asset_to_pool(clc_pool, asset_id, amt, avg_value, "fee_kind")

        cash_remaining = cash_usd
        gross_cash_usd = cash_usd + fee_service_reserved_usd
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
        mandate_share = max(0.0, float(self.cfg.liquidity_mandate_share or 0.0))
        if self._waterfall_bootstrap_remaining > 0:
            mandate_share = max(0.0, float(self.cfg.liquidity_mandate_bootstrap_share or 0.0))
        if cash_remaining > 1e-9:
            mandate_budget = cash_remaining * mandate_share
        if mandate_budget > 1e-9:
            self._vault_add(mandates_pool, stable_id, mandate_budget, "waterfall_mandates", "waterfall")
            cash_remaining -= mandate_budget
            self._mandates_allocated_usd_total += mandate_budget
        mandates_distributed = self._distribute_liquidity_mandates(mandate_budget)

        ops_extra = 0.0
        insurance_extra = 0.0
        clc_alloc = cash_remaining
        if clc_alloc > 1e-9:
            self._vault_add(clc_pool, stable_id, clc_alloc, "waterfall_clc", "waterfall")
            self._clc_pool_injected_usd_total += clc_alloc

        if total_fee_usd > 1e-9 and self._waterfall_bootstrap_remaining > 0:
            self._waterfall_bootstrap_remaining -= 1

        chi = gross_cash_usd / max(1e-9, gross_cash_usd + kind_usd)
        self._waterfall_last = {
            "fee_in_usd": total_fee_usd,
            "fee_cash_usd": gross_cash_usd,
            "fee_cash_waterfall_usd": cash_usd,
            "fee_kind_usd": kind_usd,
            "fee_service_reserved_usd": fee_service_reserved_usd,
            "fee_service_stable_reserved_usd": fee_service_stable_reserved_usd,
            "fee_service_converted_voucher_reserved_usd": fee_service_converted_voucher_reserved_usd,
            "fee_converted_voucher_cash_usd": fee_converted_voucher_cash_usd,
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
                           meta={
                               "cash_usd": gross_cash_usd,
                               "cash_waterfall_usd": cash_usd,
                               "kind_usd": kind_usd,
                               "fee_service_reserved_usd": fee_service_reserved_usd,
                               "chi": chi,
                           }))

        self._fee_access_budget_usd = self._compute_fee_access_budget()
        self._waterfall_last["fee_access_budget_usd"] = self._fee_access_budget_usd
        self._mint_sclc_fee_access()
        self._burn_ops_pool_usd(ops_pool)
        self.rebuild_indexes()
        if self.cfg.routing_mode == "noam":
            self._refresh_noam_working_set()
            if self.cfg.noam_overlay_enabled:
                self._maybe_refresh_noam_overlay()

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

    def _mint_sclc_fee_access(self) -> None:
        if not self.cfg.sclc_fee_access_enabled:
            return
        if not self.cfg.sclc_symbol:
            return
        if not self.clc_pool_id:
            return
        clc_pool = self.pools.get(self.clc_pool_id)
        if clc_pool is None:
            return
        lp_pools = [
            p for p in self.pools.values()
            if not p.policy.system_pool and p.policy.role == "liquidity_provider"
        ]
        if not lp_pools:
            return
        stable_id = self.cfg.stable_symbol
        available_stable = clc_pool.vault.get(stable_id) - clc_pool.policy.min_stable_reserve
        available_stable = max(0.0, available_stable)
        if available_stable <= 1e-9:
            return
        budget = available_stable
        remaining_sclc = self._lp_sclc_remaining()
        if remaining_sclc is not None:
            budget = min(budget, remaining_sclc)
        if budget <= 1e-9:
            return
        self._fee_access_budget_usd = budget
        if self._waterfall_last:
            self._waterfall_last["fee_access_budget_usd"] = self._fee_access_budget_usd
        total_injected = sum(self._lp_injected_usd_by_pool.get(p.pool_id, 0.0) for p in lp_pools)
        equal_share = total_injected <= 1e-9
        minted = 0.0
        for p in lp_pools:
            if equal_share:
                share = 1.0 / len(lp_pools)
            else:
                share = self._lp_injected_usd_by_pool.get(p.pool_id, 0.0) / total_injected
            if share <= 0.0:
                continue
            amt = budget * share
            if amt <= 1e-9:
                continue
            if p.values.get_value(self.cfg.sclc_symbol) <= 0.0:
                p.values.set_value(self.cfg.sclc_symbol, 1.0)
            self._vault_add(p, self.cfg.sclc_symbol, amt, "sclc_fee_access", "waterfall")
            self._sclc_minted_total += amt
            minted += amt
        if minted > 0.0:
            self.log.add(Event(
                self.tick,
                "SCLC_FEE_ACCESS_MINTED",
                amount=minted,
                meta={"budget": float(self._fee_access_budget_usd or 0.0), "target_stable": available_stable, "pools": len(lp_pools)},
            ))

    def _burn_ops_pool_usd(self, ops_pool: "Pool") -> None:
        stable_id = self.cfg.stable_symbol
        balance = ops_pool.vault.get(stable_id)
        if balance <= 1e-9:
            return
        if not self._vault_sub(ops_pool, stable_id, balance, "ops_burn", "system"):
            return
        self._stable_offramp_usd_tick += balance
        self.log.add(Event(self.tick, "OPS_BURNED", amount=balance))

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
            if self.execute_route_from_pool(
                clc_pool.pool_id, plan, amount_used, route_context="clc_rebalance"
            ):
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
        self._clc_access_open = True
        clc_pool.policy.paused = False
        clc_pool.policy.min_stable_reserve = 0.0

    def step(self, n_ticks: int = 1) -> None:
        for _ in range(n_ticks):
            self.tick += 1
            self._decay_affinity()
            self._swap_volume_usd_tick = 0.0
            self._swap_volume_usd_by_pool = {}
            self._noam_routing_swaps_tick = 0
            self._noam_clearing_swaps_tick = 0
            self._noam_routing_volume_usd_tick = 0.0
            self._noam_clearing_volume_usd_tick = 0.0
            self._noam_routing_fee_usd_tick = 0.0
            self._noam_clearing_fee_usd_tick = 0.0
            self._noam_routing_stable_fee_usd_tick = 0.0
            self._noam_routing_voucher_fee_usd_tick = 0.0
            self._noam_clearing_stable_fee_usd_tick = 0.0
            self._noam_clearing_voucher_fee_usd_tick = 0.0
            self._noam_clearing_cycles_attempted_tick = 0
            self._noam_clearing_cycles_executed_tick = 0
            self._noam_clearing_cycle_value_usd_tick = 0.0
            self._claims_paid_usd_tick = 0.0
            self._claims_unpaid_usd_tick = 0.0
            self._incidents_tick = 0
            self._stable_onramp_usd_tick = 0.0
            self._stable_offramp_usd_tick = 0.0
            self._historical_stable_backing_usd_tick = 0.0
            self._historical_stable_backing_pools_tick = 0
            self._stable_excess_sweep_usd_tick = 0.0
            self._stable_excess_sweep_pools_tick = 0
            self._producer_deposit_stable_usd_tick = 0.0
            self._producer_deposit_voucher_usd_tick = 0.0
            self._route_source_stable_net_flow_value_tick = 0.0
            self._route_source_voucher_net_flow_value_tick = 0.0
            self._productive_credit_inflow_usd_tick = 0.0
            self._productive_credit_stable_retained_usd_tick = 0.0
            self._productive_credit_voucher_deposit_usd_tick = 0.0
            self._productive_credit_voucher_deposit_cap_clipped_usd_tick = 0.0
            self._productive_credit_voucher_deposit_usd_by_pool_tick = {}
            self._producer_debt_originated_usd_tick = 0.0
            self._producer_debt_cash_service_due_usd_tick = 0.0
            self._producer_debt_cash_service_paid_usd_tick = 0.0
            self._producer_debt_matured_usd_tick = 0.0
            self._producer_debt_repaid_usd_tick = 0.0
            self._producer_debt_repaid_regular_usd_tick = 0.0
            self._producer_debt_repaid_maturity_usd_tick = 0.0
            self._producer_debt_stable_recovered_usd_tick = 0.0
            self._producer_debt_consumer_stable_purchase_usd_tick = 0.0
            self._producer_debt_third_party_stable_purchase_usd_tick = 0.0
            self._producer_debt_defaulted_usd_tick = 0.0
            self._producer_debt_closed_by_circulation_usd_tick = 0.0
            self._producer_debt_closed_by_voucher_swap_usd_tick = 0.0
            self._producer_debt_closed_not_held_at_maturity_usd_tick = 0.0
            self._producer_debt_service_capacity_credited_usd_tick = 0.0
            self._producer_debt_service_capacity_onramp_usd_tick = 0.0
            self._producer_self_repayment_swap_volume_usd_tick = 0.0
            self._producer_self_repayment_voucher_removed_usd_tick = 0.0
            self._producer_debt_pressure_prepayment_usd_tick = 0.0
            self._producer_debt_pressure_deferred_usd_tick = 0.0
            self._producer_debt_pressure_batched_swap_count_tick = 0
            self._producer_debt_pressure_batched_swap_volume_usd_tick = 0.0
            self._producer_debt_attention_pressure_usd_tick = 0.0
            self._producer_debt_attention_suppressed_attempts_tick = 0
            self._producer_debt_attention_suppressed_v2v_attempts_tick = 0
            self._producer_debt_attention_share_sum_tick = 0.0
            self._producer_debt_attention_share_count_tick = 0
            self._producer_debt_attention_share_max_tick = 0.0
            self._producer_debt_attention_reference_usd_sum_tick = 0.0
            self._producer_debt_attention_reference_count_tick = 0
            self._producer_bond_assessment_pressure_usd_tick = 0.0
            self._producer_bond_assessment_sustain_offset_attempts_tick = 0.0
            self._producer_bond_assessment_sustain_offset_v2v_attempts_tick = 0.0
            self._producer_bond_assessment_sustain_target_reduction_tick = 0
            self._producer_activity_composition_pressure_usd_tick = 0.0
            self._producer_activity_composition_shift_share_sum_tick = 0.0
            self._producer_activity_composition_shift_share_count_tick = 0
            self._producer_activity_composition_shift_share_max_tick = 0.0
            self._producer_activity_composition_reference_usd_sum_tick = 0.0
            self._producer_activity_composition_reference_count_tick = 0
            self._producer_activity_composition_v2v_weight_removed_tick = 0.0
            self._producer_activity_composition_v2s_weight_added_tick = 0.0
            self._producer_activity_composition_shifted_route_attempts_tick = 0
            self._producer_activity_composition_shifted_v2s_attempts_tick = 0
            self._producer_activity_composition_own_voucher_stable_probability_sum_tick = 0.0
            self._producer_activity_composition_own_voucher_stable_probability_count_tick = 0
            self._producer_activity_composition_own_voucher_stable_probability_max_tick = 0.0
            self._producer_activity_composition_share_cache = {}
            self._producer_debt_penalty_accrued_usd_tick = 0.0
            self._producer_debt_penalty_paid_usd_tick = 0.0
            self._producer_loan_attempts_tick = 0
            self._producer_loan_no_lender_tick = 0
            self._producer_loan_no_inventory_tick = 0
            self._producer_loan_zero_amount_tick = 0
            self._producer_loan_route_found_tick = 0
            self._producer_loan_route_failed_tick = 0
            self._producer_loan_backfill_attempts_tick = 0
            self._producer_loan_backfill_executed_tick = 0
            self._producer_loan_executed_tick = 0
            self._producer_loan_execution_failed_tick = 0
            self._producer_loan_sampled_usd_tick = 0.0
            self._producer_loan_attempted_usd_tick = 0.0
            self._producer_loan_executed_usd_tick = 0.0
            self._producer_loan_clipped_inventory_usd_tick = 0.0
            self._producer_loan_clipped_lender_cap_usd_tick = 0.0
            self._producer_loan_clipped_lender_remaining_usd_tick = 0.0
            self._producer_loan_clipped_lender_stable_usd_tick = 0.0
            self._producer_loan_clipped_combined_lender_usd_tick = 0.0
            self._producer_loan_lender_collateral_cap_usd_tick = 0.0
            self._producer_loan_lender_remaining_cap_usd_tick = 0.0
            self._producer_loan_lender_stable_available_usd_tick = 0.0
            self._producer_voucher_loan_attempts_tick = 0
            self._producer_voucher_loan_no_target_tick = 0
            self._producer_voucher_loan_no_inventory_tick = 0
            self._producer_voucher_loan_zero_amount_tick = 0
            self._producer_voucher_loan_route_found_tick = 0
            self._producer_voucher_loan_route_failed_tick = 0
            self._producer_voucher_loan_executed_tick = 0
            self._producer_voucher_loan_execution_failed_tick = 0
            self._producer_voucher_loan_attempted_usd_tick = 0.0
            self._producer_voucher_loan_executed_usd_tick = 0.0
            self._producer_voucher_loan_clipped_lender_cap_usd_tick = 0.0
            self._producer_voucher_loan_clipped_lender_remaining_usd_tick = 0.0
            self._producer_primary_voucher_loan_attempts_tick = 0
            self._producer_primary_voucher_loan_executed_tick = 0
            self._voucher_purchase_attempts_tick = 0
            self._consumer_voucher_purchase_attempts_tick = 0
            self._consumer_voucher_purchase_success_tick = 0
            self._consumer_voucher_purchase_no_stable_tick = 0
            self._consumer_voucher_purchase_reserve_protected_tick = 0
            self._consumer_voucher_purchase_no_route_tick = 0
            self._consumer_voucher_purchase_no_target_tick = 0
            self._consumer_voucher_purchase_stable_spent_usd_tick = 0.0
            self._consumer_voucher_purchase_voucher_value_acquired_usd_tick = 0.0
            self._third_party_voucher_purchase_attempts_tick = 0
            self._third_party_voucher_purchase_success_tick = 0
            self._third_party_voucher_purchase_no_stable_tick = 0
            self._third_party_voucher_purchase_reserve_protected_tick = 0
            self._third_party_voucher_purchase_no_route_tick = 0
            self._third_party_voucher_purchase_no_target_tick = 0
            self._third_party_voucher_purchase_stable_spent_usd_tick = 0.0
            self._third_party_voucher_purchase_voucher_value_acquired_usd_tick = 0.0
            self._lender_voucher_purchase_stable_budget_remaining_usd_tick = 0.0
            self._lender_voucher_purchase_stable_budget_remaining_by_kind_tick = {}
            self._lender_voucher_purchase_stable_budget_onramp_usd_tick = 0.0
            self._consumer_voucher_purchase_stable_budget_onramp_usd_tick = 0.0
            self._third_party_voucher_purchase_stable_budget_onramp_usd_tick = 0.0
            self._producer_stable_exited_usd_tick = 0.0
            self._producer_stable_reuse_budget_usd_tick = 0.0
            self._net_redeemed_voucher_usd_tick = 0.0
            self._voucher_redeemed_to_issuer_usd_tick = 0.0
            self._voucher_fee_retained_for_service_usd_tick = 0.0
            self._voucher_reintroduced_by_deposit_usd_tick = 0.0
            self._voucher_new_issuance_deposit_usd_tick = 0.0
            self._debt_removal_voucher_redeemed_usd_tick = 0.0
            self._route_context_count_tick = {}
            self._route_context_volume_usd_tick = {}
            self._route_context_source_stable_count_tick = {}
            self._route_context_source_stable_volume_usd_tick = {}
            self._route_context_source_voucher_count_tick = {}
            self._route_context_source_voucher_volume_usd_tick = {}
            self._route_motif_count_tick = {}
            self._route_motif_volume_usd_tick = {}
            self._route_motif_stable_intermediate_count_tick = 0
            self._route_motif_stable_intermediate_volume_usd_tick = 0.0
            self._ordinary_route_motif_count_tick = {}
            self._ordinary_route_motif_volume_usd_tick = {}
            self._market_route_motif_count_tick = {}
            self._market_route_motif_volume_usd_tick = {}
            self._repayment_route_motif_count_tick = {}
            self._repayment_route_motif_volume_usd_tick = {}
            self._loan_route_motif_count_tick = {}
            self._loan_route_motif_volume_usd_tick = {}
            self._productive_boosted_voucher_swap_count_tick = 0
            self._productive_boosted_voucher_swap_volume_usd_tick = 0.0
            self._voucher_loan_boosted_voucher_swap_count_tick = 0
            self._voucher_loan_boosted_voucher_swap_volume_usd_tick = 0.0
            self._ordinary_stable_spend_protected_skip_count_tick = 0
            self._ordinary_stable_spend_protected_skip_value_usd_tick = 0.0
            self._fee_conversion_attempted_usd_tick = 0.0
            self._fee_conversion_success_usd_tick = 0.0
            self._fee_conversion_failed_usd_tick = 0.0
            self._fee_service_reserved_usd_tick = 0.0
            self._fee_service_stable_reserved_usd_tick = 0.0
            self._fee_service_converted_voucher_reserved_usd_tick = 0.0
            self._bond_service_reserved_usd_tick = 0.0
            self._bond_service_paid_from_reserve_usd_tick = 0.0
            self._lender_recovered_stable_usd_tick = 0.0
            self._bond_eligible_pool_exposure_recovered_stable_usd_tick = 0.0
            self._lender_inventory_turnover_stable_usd_tick = 0.0
            self._lender_recovered_stable_borrower_self_usd_tick = 0.0
            self._lender_recovered_stable_borrower_regular_usd_tick = 0.0
            self._lender_recovered_stable_borrower_maturity_usd_tick = 0.0
            self._lender_recovered_stable_consumer_purchase_usd_tick = 0.0
            self._lender_recovered_stable_external_nonproducer_purchase_usd_tick = 0.0
            self._lender_recovered_stable_other_producer_purchase_usd_tick = 0.0
            self._lender_recovered_stable_third_party_purchase_usd_tick = 0.0
            self._lender_recovered_stable_other_usd_tick = 0.0
            self._refresh_bond_recovery_budget_caps()
            self._quarterly_clearing_usd_tick = 0.0
            self._quarterly_clearing_lender_liquidity_before_tick = 0.0
            self._quarterly_clearing_lender_liquidity_after_tick = 0.0
            self._route_fixed_requested_tick = 0
            self._route_fixed_found_tick = 0
            self._route_fixed_failed_tick = 0
            self._route_substitution_requested_tick = 0
            self._route_substitution_found_tick = 0
            self._route_substitution_failed_tick = 0
            self._route_repeat_partner_requested_tick = 0
            self._route_exploration_requested_tick = 0
            self._route_sticky_used_tick = 0
            self._route_buddy_direct_used_tick = 0
            self._route_new_target_search_tick = 0
            self._route_search_fallback_used_tick = 0
            self._swap_attempt_counts = {}

            self._apply_productive_credit_inflows()
            self._apply_producer_deposits()
            self._apply_historical_stable_backing()
            self._apply_historical_voucher_backing()
            self._apply_producer_debt_maturities()

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
            self._refresh_dirty_lender_voucher_limits()

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
            if max_active > 0:
                pool_count = len(pools)
                if pool_count > 0:
                    benchmark = int(self.cfg.max_pools or 500)
                    if benchmark <= 0:
                        benchmark = pool_count
                    scale = math.sqrt(benchmark / max(1, pool_count))
                    scaled_max = int(math.ceil(max_active * scale))
                    max_active = min(pool_count, max(1, scaled_max))
            if max_active > 0 and max_active < len(pools):
                pools = self.rng.sample(pools, k=max_active)
            if pools:
                self.rng.shuffle(pools)
            remaining_requests = int(self.cfg.swap_requests_budget_per_tick or 0)
            if remaining_requests <= 0:
                remaining_requests = None
            producer_credit_budget_reserved = (
                max(0.0, float(self.cfg.producer_credit_request_budget_share or 0.0)) > 0.0
            )
            remaining_requests = self._apply_producer_credit_request_budget(pools, remaining_requests)
            for p in pools:
                if remaining_requests is not None and remaining_requests <= 0:
                    break
                remaining = self._swap_attempts_for_pool(p)
                if remaining <= 0:
                    continue
                if p.policy.role == "producer" and not producer_credit_budget_reserved:
                    attempted = self._attempt_repayment(p)
                    if attempted:
                        remaining -= 1
                        if remaining_requests is not None:
                            remaining_requests -= 1
                if p.policy.role == "producer" and remaining > 0:
                    suppressed = self._apply_producer_debt_attention_crowdout(p, remaining)
                    if suppressed > 0:
                        remaining -= suppressed
                        if remaining_requests is not None:
                            remaining_requests = max(0, remaining_requests - suppressed)
                if remaining > 0:
                    if remaining_requests is not None:
                        remaining = min(remaining, remaining_requests)
                    if remaining <= 0:
                        continue
                    attempted = self._random_route_request(source_pool=p, max_assets=remaining)
                    if remaining_requests is not None:
                        remaining_requests = max(0, remaining_requests - attempted)

            self._apply_lender_voucher_purchase_demand()
            self._run_noam_clearing()
            self._sustain_swap_activity()

            self._apply_offramp_behavior()

            window_ticks = max(1, int(self.cfg.stable_flow_window_ticks or 1))
            if self.cfg.stable_flow_mode != "none" and self.tick % window_ticks == 0:
                self._apply_activity_stable_flow(window_ticks)

            self._simulate_incidents()
            self._apply_liquidity_provider_contributions()

            if self.cfg.economics_enabled:
                epoch = max(1, int(self.cfg.waterfall_epoch_ticks or 1))
                if self.tick == 1 or self.tick % epoch == 0:
                    self._apply_waterfall()
                    self._apply_monthly_pool_offramp()
                self._apply_quarterly_clearing()
                self._clc_rebalance()
            self._update_clc_swap_window()
            self._apply_lp_sclc_redemptions()

            self._update_utilization_boost()
            self._record_swap_history()
            self._stable_onramp_usd_month += self._stable_onramp_usd_tick
            self._stable_offramp_usd_month += self._stable_offramp_usd_tick
            epoch = max(1, int(self.cfg.waterfall_epoch_ticks or 1))
            if self.tick % epoch == 0:
                self._apply_monthly_offramp_balancer()
                self._apply_stable_excess_sweep()
                self._stable_onramp_usd_month = 0.0
                self._stable_offramp_usd_month = 0.0
            self.snapshot_metrics()

    def _has_repeat_partner_candidate(self, source_pool: "Pool", asset_candidates: list[str]) -> bool:
        for asset_id in asset_candidates:
            if self._sticky_target_by_pool.get((source_pool.pool_id, asset_id)):
                return True
            for key in self._sticky_plan_by_pool:
                if key[0] == source_pool.pool_id and key[1] == asset_id:
                    return True
        if source_pool.policy.role in ("producer", "consumer"):
            return self._affinity_buddies_for_pool(source_pool.pool_id) is not None
        return False

    def _choose_route_activity_mode(
        self,
        source_pool: "Pool",
        asset_candidates: list[str],
        route_context: str,
        requested_mode: str,
    ) -> str:
        mode = str(requested_mode or "auto")
        if mode in {"repeat_partner", "exploration"}:
            return mode
        if route_context != "ordinary":
            return "exploration"
        if not bool(getattr(self.cfg, "decision_based_activity_enabled", True)):
            return "exploration"
        if not self._has_repeat_partner_candidate(source_pool, asset_candidates):
            return "exploration"
        repeat_share = max(0.0, min(1.0, float(getattr(self.cfg, "repeat_partner_route_share", 0.0) or 0.0)))
        return "repeat_partner" if self.rng.random() < repeat_share else "exploration"

    def _record_route_decision_attempt(
        self,
        activity_mode: str,
        *,
        sticky: bool = False,
        buddy_direct: bool = False,
        new_target_search: bool = False,
        fallback: bool = False,
    ) -> None:
        if activity_mode == "repeat_partner":
            self._route_repeat_partner_requested_tick += 1
        else:
            self._route_exploration_requested_tick += 1
        if sticky:
            self._route_sticky_used_tick += 1
        if buddy_direct:
            self._route_buddy_direct_used_tick += 1
        if new_target_search:
            self._route_new_target_search_tick += 1
        if fallback:
            self._route_search_fallback_used_tick += 1

    def _random_route_request(
        self,
        source_pool: Optional["Pool"] = None,
        max_assets: Optional[int] = None,
        route_context: str = "ordinary",
        activity_mode: str = "auto",
    ) -> int:
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

        if route_context == "ordinary":
            self._record_ordinary_stable_spend_protection_skip(source_pool)

        motif_source_class: Optional[str] = None
        motif_target_class: Optional[str] = None
        if route_context == "ordinary":
            motif = self._settlement_motif_choice(
                source_pool=source_pool,
                route_context=route_context,
            )
            motif_source_class, motif_target_class = self._settlement_motif_classes(motif)

        # try each asset_in with positive ordinary-spendable inventory
        asset_candidates = self._route_source_asset_candidates(source_pool)
        if (
            route_context == "ordinary"
            and bool(getattr(self.cfg, "lender_voucher_purchase_demand_enabled", False))
            and source_pool.policy.role == "consumer"
        ):
            # Consumer stable-to-voucher purchases are modeled by the calibrated
            # purchase process, not by generic wallet route attempts.
            asset_candidates = [
                asset for asset in asset_candidates
                if asset != self.cfg.stable_symbol
            ]
        if not asset_candidates:
            return 0

        if motif_source_class is not None:
            preferred_assets = [
                asset
                for asset in asset_candidates
                if self._settlement_asset_class(asset) == motif_source_class
            ]
            # If this wallet does not currently hold the preferred source class,
            # local balance-sheet reality overrides the empirical prior.
            if not preferred_assets:
                preferred_assets = asset_candidates
            asset_candidates = self._choose_source_asset_candidate(
                source_pool,
                preferred_assets,
            )
            if not asset_candidates:
                return 0

        if max_assets is not None and max_assets > 0 and motif_source_class is None:
            if len(asset_candidates) > max_assets:
                mode = self.cfg.swap_asset_selection_mode
                if mode == "value_weighted":
                    weights = self._source_asset_selection_weights(source_pool, asset_candidates)
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

        selected_activity_mode = self._choose_route_activity_mode(
            source_pool,
            asset_candidates,
            route_context,
            activity_mode,
        )
        decision_mode_active = (
            bool(getattr(self.cfg, "decision_based_activity_enabled", True))
            and route_context == "ordinary"
        )
        attempted = 0
        buddy_pools: Optional[Set[str]] = None
        if source_pool.policy.role in ("producer", "consumer"):
            buddy_pools = self._affinity_buddies_for_pool(source_pool.pool_id)
        frontier_pool_whitelist: Optional[Set[str]] = None
        if route_context == "ordinary" and self._frontier_shortlist_enabled():
            frontier_pool_whitelist = self._frontier_relationship_pool_whitelist(source_pool)
            if frontier_pool_whitelist:
                if buddy_pools is None:
                    buddy_pools = set(frontier_pool_whitelist)
                else:
                    buddy_pools = set(buddy_pools) | set(frontier_pool_whitelist)
        if decision_mode_active:
            buddy_direct_only = (
                selected_activity_mode == "repeat_partner"
                and bool(self.cfg.affinity_buddy_direct_only)
                and buddy_pools is not None
            )
            sticky_bias = 1.0 if selected_activity_mode == "repeat_partner" else 0.0
        else:
            buddy_direct_only = bool(self.cfg.affinity_buddy_direct_only) and buddy_pools is not None
            sticky_bias = max(0.0, float(self.cfg.sticky_route_bias or 0.0))
        sticky_fail_threshold = max(1, int(self.cfg.sticky_fail_threshold or 1))

        for asset_in in asset_candidates:
            stable_target_blocked = self._ordinary_own_voucher_stable_target_blocked_for_attempt(
                source_pool,
                asset_in,
                route_context,
            )
            target_class = self._effective_ordinary_target_class(
                source_pool,
                asset_in,
                route_context,
                motif_target_class,
                stable_target_blocked=stable_target_blocked,
            )
            if route_context == "ordinary" and self._frontier_shortlist_enabled():
                max_targets = 1
            elif bool(self.cfg.route_substitution_enabled):
                max_targets = 1 + max(0, int(self.cfg.route_substitution_max_alternatives or 0))
            else:
                max_targets = max(1, int(self.cfg.swap_target_retry_count or 1))
            targets_tried: Set[str] = (
                {self.cfg.stable_symbol} if stable_target_blocked else set()
            )
            sticky_target = None
            if sticky_bias > 0.0:
                sticky_target = self._sticky_target_by_pool.get((source_pool.pool_id, asset_in))
            for attempt_index in range(max_targets):
                if max_assets is not None and max_assets > 0 and attempted >= max_assets:
                    break
                attempt_kind = "substitution" if attempt_index > 0 else "fixed"
                sticky_target_allowed = (
                    sticky_target
                    and sticky_target not in targets_tried
                    and sticky_target != asset_in
                    and (
                        target_class is None
                        or self._settlement_asset_class(sticky_target) == target_class
                    )
                )
                if buddy_direct_only and not sticky_target_allowed:
                    amount_in = self._sample_amount_in(source_pool, asset_in)
                    if amount_in <= 1e-9:
                        break
                    attempted += 1
                    plan, amount_used, used_fallback, asset_out = self._find_buddy_direct_plan(
                        source_pool=source_pool,
                        asset_in=asset_in,
                        amount_in=amount_in,
                        buddy_pools=buddy_pools or set(),
                        target_class=target_class,
                    )
                    self._record_route_decision_attempt(
                        selected_activity_mode,
                        buddy_direct=True,
                        fallback=used_fallback,
                    )
                    meta = {
                        "target_asset": asset_out,
                        "buddy_direct": True,
                        "route_attempt_kind": attempt_kind,
                        "route_activity_mode": selected_activity_mode,
                        "target_selection": "buddy_direct",
                    }
                    if used_fallback:
                        meta["fallback"] = True
                    self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                                       asset_id=asset_in, amount=amount_used, meta=meta))
                    if not plan.ok or asset_out is None:
                        self.log.add(Event(self.tick, "ROUTE_FAILED", pool_id=source_pool.pool_id,
                                           asset_id=asset_in, amount=amount_used,
                                           meta={
                                               "reason": plan.reason,
                                               "target": asset_out,
                                               "buddy_direct": True,
                                               "route_attempt_kind": attempt_kind,
                                               "route_activity_mode": selected_activity_mode,
                                               "target_selection": "buddy_direct",
                                           }))
                        self._record_swap_attempt(source_pool.pool_id, success=False)
                        continue
                    if stable_target_blocked and asset_out == self.cfg.stable_symbol:
                        self._record_swap_attempt(source_pool.pool_id, success=False)
                        continue
                    self.log.add(Event(self.tick, "ROUTE_FOUND", pool_id=source_pool.pool_id,
                                       meta={
                                           "hops": [h.__dict__ for h in plan.hops],
                                           "target": asset_out,
                                           "buddy_direct": True,
                                           "route_attempt_kind": attempt_kind,
                                           "route_activity_mode": selected_activity_mode,
                                           "target_selection": "buddy_direct",
                                       }))
                    execution_context = self._route_context_for_ordinary_own_voucher_source(
                        source_pool,
                        asset_in,
                        asset_out,
                        route_context,
                    )
                    ok = self.execute_route_from_pool(
                        source_pool.pool_id, plan, amount_used, route_context=execution_context
                    )
                    self._record_swap_attempt(source_pool.pool_id, success=ok)
                    if ok:
                        self._sticky_target_by_pool[(source_pool.pool_id, asset_in)] = asset_out
                        self._sticky_plan_by_pool[(source_pool.pool_id, asset_in, asset_out)] = plan
                        break
                    continue
                if sticky_target_allowed and self.rng.random() < sticky_bias:
                    asset_out = sticky_target
                    target_selection = "sticky"
                else:
                    asset_out = None
                    target_selection = "new_search"
                    if frontier_pool_whitelist:
                        asset_out = self._choose_frontier_relationship_target_asset(
                            asset_in,
                            source_pool,
                            frontier_pool_whitelist,
                            exclude=targets_tried,
                            preferred_class=target_class,
                        )
                        target_selection = "steward_shortlist"
                        if not asset_out and target_class is not None and not stable_target_blocked:
                            asset_out = self._choose_frontier_relationship_target_asset(
                                asset_in,
                                source_pool,
                                frontier_pool_whitelist,
                                exclude=targets_tried,
                            )
                    if asset_out is None:
                        asset_out = self._choose_target_asset(
                            asset_in,
                            source_pool,
                            exclude=targets_tried,
                            preferred_class=target_class,
                        )
                        target_selection = "new_search"
                        if not asset_out and target_class is None:
                            asset_out = self._choose_target_asset(
                                asset_in,
                                source_pool,
                                exclude=targets_tried,
                            )
                if not asset_out or asset_out == asset_in:
                    break
                if stable_target_blocked and asset_out == self.cfg.stable_symbol:
                    targets_tried.add(asset_out)
                    continue
                targets_tried.add(asset_out)

                amount_in = self._sample_amount_in(source_pool, asset_in)
                if amount_in <= 1e-9:
                    break

                attempted += 1
                self._record_route_decision_attempt(
                    selected_activity_mode,
                    new_target_search=(target_selection == "new_search"),
                )
                self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                                   asset_id=asset_in, amount=amount_in,
                                   meta={
                                       "target_asset": asset_out,
                                       "route_attempt_kind": attempt_kind,
                                       "route_activity_mode": selected_activity_mode,
                                       "target_selection": target_selection,
                                   }))
                sticky_key = (source_pool.pool_id, asset_in, asset_out)
                sticky_plan = self._sticky_plan_by_pool.get(sticky_key)
                if frontier_pool_whitelist is not None:
                    active_buddy_pools = frontier_pool_whitelist
                else:
                    active_buddy_pools = buddy_pools if selected_activity_mode == "repeat_partner" else None
                if sticky_plan and active_buddy_pools is not None:
                    if any(h.pool_id not in active_buddy_pools for h in sticky_plan.hops):
                        sticky_plan = None
                if sticky_plan:
                    failures = self._sticky_failures.get(sticky_key, 0)
                    if self._validate_route_plan(sticky_plan, amount_in, source_pool):
                        self._route_sticky_used_tick += 1
                        self.log.add(Event(self.tick, "ROUTE_FOUND", pool_id=source_pool.pool_id,
                                           meta={
                                               "hops": [h.__dict__ for h in sticky_plan.hops],
                                               "target": asset_out,
                                               "sticky": True,
                                               "route_attempt_kind": attempt_kind,
                                               "route_activity_mode": selected_activity_mode,
                                               "target_selection": target_selection,
                                           }))
                        execution_context = self._route_context_for_ordinary_own_voucher_source(
                            source_pool,
                            asset_in,
                            asset_out,
                            route_context,
                        )
                        ok = self.execute_route_from_pool(
                            source_pool.pool_id, sticky_plan, amount_in, route_context=execution_context
                        )
                        self._record_swap_attempt(source_pool.pool_id, success=ok)
                        if ok:
                            self._sticky_target_by_pool[(source_pool.pool_id, asset_in)] = asset_out
                            self._sticky_plan_by_pool[sticky_key] = sticky_plan
                            self._sticky_failures.pop(sticky_key, None)
                            break
                        failures += 1
                    else:
                        failures += 1

                    if failures >= sticky_fail_threshold:
                        self._sticky_plan_by_pool.pop(sticky_key, None)
                        self._sticky_target_by_pool.pop((source_pool.pool_id, asset_in), None)
                        self._sticky_failures.pop(sticky_key, None)
                        continue

                    self._sticky_failures[sticky_key] = failures

                if frontier_pool_whitelist is not None and target_selection == "steward_shortlist":
                    direct_plan, direct_amount, direct_fallback, direct_asset_out = self._find_buddy_direct_plan(
                        source_pool=source_pool,
                        asset_in=asset_in,
                        amount_in=amount_in,
                        buddy_pools=frontier_pool_whitelist,
                        target_asset=asset_out,
                    )
                    if direct_plan.ok and direct_asset_out is not None:
                        self._route_buddy_direct_used_tick += 1
                        if direct_fallback:
                            self._route_search_fallback_used_tick += 1
                            self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                                               asset_id=asset_in, amount=direct_amount,
                                               meta={
                                                   "target_asset": direct_asset_out,
                                                   "fallback": True,
                                                   "buddy_direct": True,
                                                   "route_attempt_kind": attempt_kind,
                                                   "route_activity_mode": selected_activity_mode,
                                                   "target_selection": target_selection,
                                               }))
                        self.log.add(Event(self.tick, "ROUTE_FOUND", pool_id=source_pool.pool_id,
                                           meta={
                                               "hops": [h.__dict__ for h in direct_plan.hops],
                                               "target": direct_asset_out,
                                               "buddy_direct": True,
                                               "route_attempt_kind": attempt_kind,
                                               "route_activity_mode": selected_activity_mode,
                                               "target_selection": target_selection,
                                           }))
                        execution_context = self._route_context_for_ordinary_own_voucher_source(
                            source_pool,
                            asset_in,
                            direct_asset_out,
                            route_context,
                        )
                        ok = self.execute_route_from_pool(
                            source_pool.pool_id,
                            direct_plan,
                            direct_amount,
                            route_context=execution_context,
                        )
                        self._record_swap_attempt(source_pool.pool_id, success=ok)
                        if ok:
                            self._sticky_target_by_pool[(source_pool.pool_id, asset_in)] = direct_asset_out
                            self._sticky_plan_by_pool[(source_pool.pool_id, asset_in, direct_asset_out)] = direct_plan
                            break
                        continue

                plan, amount_used, used_fallback = self._find_route_with_fallback(
                    tick=self.tick,
                    start_asset=asset_in,
                    target_asset=asset_out,
                    amount_in=amount_in,
                    source_pool=source_pool,
                    pool_whitelist=active_buddy_pools,
                )
                if used_fallback:
                    self._route_search_fallback_used_tick += 1
                    self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                                       asset_id=asset_in, amount=amount_used,
                                       meta={
                                           "target_asset": asset_out,
                                           "fallback": True,
                                           "route_attempt_kind": attempt_kind,
                                           "route_activity_mode": selected_activity_mode,
                                           "target_selection": target_selection,
                                       }))

                if not plan.ok:
                    self.log.add(Event(self.tick, "ROUTE_FAILED", pool_id=source_pool.pool_id,
                                       asset_id=asset_in, amount=amount_used,
                                       meta={
                                           "reason": plan.reason,
                                           "target": asset_out,
                                           "route_attempt_kind": attempt_kind,
                                           "route_activity_mode": selected_activity_mode,
                                           "target_selection": target_selection,
                                       }))
                    self._record_swap_attempt(source_pool.pool_id, success=False)
                    continue

                self.log.add(Event(self.tick, "ROUTE_FOUND", pool_id=source_pool.pool_id,
                                   meta={
                                       "hops": [h.__dict__ for h in plan.hops],
                                       "target": asset_out,
                                       "route_attempt_kind": attempt_kind,
                                       "route_activity_mode": selected_activity_mode,
                                       "target_selection": target_selection,
                                   }))

                # Execute with escrow:
                execution_context = self._route_context_for_ordinary_own_voucher_source(
                    source_pool,
                    asset_in,
                    asset_out,
                    route_context,
                )
                ok = self.execute_route_from_pool(
                    source_pool.pool_id, plan, amount_used, route_context=execution_context
                )
                self._record_swap_attempt(source_pool.pool_id, success=ok)
                if ok:
                    self._sticky_target_by_pool[(source_pool.pool_id, asset_in)] = asset_out
                    self._sticky_plan_by_pool[(source_pool.pool_id, asset_in, asset_out)] = plan
                    break

        return attempted

    def _backfill_failed_loan_route(self, source_pool: "Pool") -> None:
        cfg = self.cfg
        if not bool(cfg.producer_loan_failure_backfill_enabled):
            return
        attempts = max(0, int(cfg.producer_loan_failure_backfill_max_attempts or 0))
        if attempts <= 0:
            return
        before_swaps = int(self._noam_routing_swaps_tick + self._noam_clearing_swaps_tick)
        for _ in range(attempts):
            self._producer_loan_backfill_attempts_tick += 1
            attempted = self._random_route_request(
                source_pool=source_pool, max_assets=1, route_context="loan_backfill"
            )
            after_swaps = int(self._noam_routing_swaps_tick + self._noam_clearing_swaps_tick)
            if after_swaps > before_swaps:
                self._producer_loan_backfill_executed_tick += 1
                break
            before_swaps = after_swaps
            if attempted > 0:
                break

    def _lender_held_producer_voucher_inventory_usd(self) -> float:
        total = 0.0
        for pool in self.pools.values():
            if pool.policy.role != "lender":
                continue
            for asset_id, amount in pool.vault.inventory.items():
                if amount <= 1e-9 or not self._is_producer_voucher(asset_id):
                    continue
                total += amount * self._asset_value(pool, asset_id)
        return total

    def _active_routable_producer_voucher_float_usd(self) -> float:
        total = 0.0
        for pool in self.pools.values():
            if not self._is_routable_pool(pool):
                continue
            for asset_id, amount in pool.vault.inventory.items():
                if amount <= 1e-9 or not self._is_producer_voucher(asset_id):
                    continue
                if not pool.registry.is_listed(asset_id):
                    continue
                total += amount * self._asset_value(pool, asset_id)
        return total

    def _consumer_stable_available_above_reserve_usd(self) -> float:
        stable_id = self.cfg.stable_symbol
        total = 0.0
        for pool in self.pools.values():
            if pool.policy.role != "consumer" or pool.policy.system_pool:
                continue
            spendable = self._ordinary_source_spendable_amount(pool, stable_id)
            if spendable > 1e-9:
                total += spendable * self._asset_value(pool, stable_id)
        return total

    def _lender_producer_voucher_purchase_targets(self) -> list[tuple[float, str, str, float, float]]:
        stable_id = self.cfg.stable_symbol
        targets: list[tuple[float, str, str, float, float]] = []
        for pool in self.pools.values():
            if pool.policy.role != "lender" or pool.policy.paused:
                continue
            if not pool.registry.is_listed(stable_id):
                continue
            for voucher_id, amount in pool.vault.inventory.items():
                if amount <= 1e-9 or not self._is_producer_voucher(voucher_id):
                    continue
                if not pool.registry.is_listed(voucher_id):
                    continue
                value = self._asset_value(pool, voucher_id)
                score = amount * value
                if score > 1e-9:
                    targets.append((score, pool.pool_id, voucher_id, amount, value))
        targets.sort(key=lambda item: item[0], reverse=True)
        limit = max(
            1,
            int(getattr(self.cfg, "lender_voucher_purchase_max_target_candidates", 5) or 5),
        )
        return targets[:limit]

    def _lender_voucher_purchase_budget_remaining_for_kind(self, buyer_kind: str) -> float:
        if self._lender_voucher_purchase_stable_budget_remaining_by_kind_tick:
            return max(
                0.0,
                float(self._lender_voucher_purchase_stable_budget_remaining_by_kind_tick.get(buyer_kind, 0.0)),
            )
        return max(0.0, float(self._lender_voucher_purchase_stable_budget_remaining_usd_tick))

    def _voucher_purchase_source_candidates(
        self,
        buyer_kind: str,
        voucher_id: str,
    ) -> tuple[list["Pool"], bool, bool]:
        stable_id = self.cfg.stable_symbol
        spec = self.factory.voucher_specs.get(voucher_id)
        sources: list["Pool"] = []
        saw_stable = False
        saw_reserve_blocked = False
        for pool in self.pools.values():
            if pool.policy.system_pool or pool.policy.paused:
                continue
            if not pool.registry.is_listed(stable_id):
                continue
            if buyer_kind == "consumer":
                if pool.policy.role != "consumer":
                    continue
            else:
                if pool.policy.role != "producer":
                    continue
                if spec is not None and pool.steward_id == spec.issuer_id:
                    continue
                if self._producer_debt_outstanding(pool) > 1e-9:
                    continue
            stable_have = pool.vault.get(stable_id)
            if stable_have > 1e-9:
                saw_stable = True
            spendable = self._ordinary_source_spendable_amount(pool, stable_id)
            if spendable > 1e-9:
                sources.append(pool)
            elif stable_have > 1e-9:
                saw_reserve_blocked = True
                if self._lender_voucher_purchase_budget_remaining_for_kind(buyer_kind) > 1e-9:
                    sources.append(pool)
            elif self._lender_voucher_purchase_budget_remaining_for_kind(buyer_kind) > 1e-9:
                sources.append(pool)
        return sources, saw_stable, saw_reserve_blocked

    def _apply_voucher_purchase_stable_budget(
        self,
        buyer_kind: str,
        source_pool: "Pool",
        needed_usd: float,
    ) -> float:
        needed_usd = max(0.0, float(needed_usd))
        budget_usd = self._lender_voucher_purchase_budget_remaining_for_kind(buyer_kind)
        if needed_usd <= 1e-9 or budget_usd <= 1e-9:
            return 0.0
        stable_id = self.cfg.stable_symbol
        stable_value = self._asset_value(source_pool, stable_id)
        if stable_value <= 1e-12:
            stable_value = 1.0
        applied_usd = min(needed_usd, budget_usd)
        self._vault_add(
            source_pool,
            stable_id,
            applied_usd / stable_value,
            "lender_voucher_purchase_budget_income",
            "purchase_budget",
        )
        if self._lender_voucher_purchase_stable_budget_remaining_by_kind_tick:
            self._lender_voucher_purchase_stable_budget_remaining_by_kind_tick[buyer_kind] = max(
                0.0,
                self._lender_voucher_purchase_stable_budget_remaining_by_kind_tick.get(buyer_kind, 0.0) - applied_usd,
            )
        self._lender_voucher_purchase_stable_budget_remaining_usd_tick = max(
            0.0,
            self._lender_voucher_purchase_stable_budget_remaining_usd_tick - applied_usd,
        )
        self._lender_voucher_purchase_stable_budget_onramp_usd_tick += applied_usd
        self._lender_voucher_purchase_stable_budget_onramp_usd_total += applied_usd
        if buyer_kind == "consumer":
            self._consumer_voucher_purchase_stable_budget_onramp_usd_tick += applied_usd
            self._consumer_voucher_purchase_stable_budget_onramp_usd_total += applied_usd
        else:
            self._third_party_voucher_purchase_stable_budget_onramp_usd_tick += applied_usd
            self._third_party_voucher_purchase_stable_budget_onramp_usd_total += applied_usd
            self._producer_stable_reuse_budget_usd_tick += applied_usd
            self._producer_stable_reuse_budget_usd_total += applied_usd
        self._stable_onramp_usd_tick += applied_usd
        return applied_usd

    def _record_voucher_purchase_failure(self, buyer_kind: str, reason: str) -> None:
        if buyer_kind == "consumer":
            if reason == "no_target":
                self._consumer_voucher_purchase_no_target_tick += 1
                self._consumer_voucher_purchase_no_target_total += 1
            elif reason == "reserve_protected":
                self._consumer_voucher_purchase_reserve_protected_tick += 1
                self._consumer_voucher_purchase_reserve_protected_total += 1
            elif reason == "no_route":
                self._consumer_voucher_purchase_no_route_tick += 1
                self._consumer_voucher_purchase_no_route_total += 1
            else:
                self._consumer_voucher_purchase_no_stable_tick += 1
                self._consumer_voucher_purchase_no_stable_total += 1
            return
        if reason == "no_target":
            self._third_party_voucher_purchase_no_target_tick += 1
            self._third_party_voucher_purchase_no_target_total += 1
        elif reason == "reserve_protected":
            self._third_party_voucher_purchase_reserve_protected_tick += 1
            self._third_party_voucher_purchase_reserve_protected_total += 1
        elif reason == "no_route":
            self._third_party_voucher_purchase_no_route_tick += 1
            self._third_party_voucher_purchase_no_route_total += 1
        else:
            self._third_party_voucher_purchase_no_stable_tick += 1
            self._third_party_voucher_purchase_no_stable_total += 1

    def _record_voucher_purchase_success(
        self,
        buyer_kind: str,
        stable_spent_usd: float,
        voucher_value_acquired_usd: float,
    ) -> None:
        if buyer_kind == "consumer":
            self._consumer_voucher_purchase_success_tick += 1
            self._consumer_voucher_purchase_success_total += 1
            self._consumer_voucher_purchase_stable_spent_usd_tick += stable_spent_usd
            self._consumer_voucher_purchase_stable_spent_usd_total += stable_spent_usd
            self._consumer_voucher_purchase_voucher_value_acquired_usd_tick += voucher_value_acquired_usd
            self._consumer_voucher_purchase_voucher_value_acquired_usd_total += (
                voucher_value_acquired_usd
            )
            return
        self._third_party_voucher_purchase_success_tick += 1
        self._third_party_voucher_purchase_success_total += 1
        self._third_party_voucher_purchase_stable_spent_usd_tick += stable_spent_usd
        self._third_party_voucher_purchase_stable_spent_usd_total += stable_spent_usd
        self._third_party_voucher_purchase_voucher_value_acquired_usd_tick += voucher_value_acquired_usd
        self._third_party_voucher_purchase_voucher_value_acquired_usd_total += (
            voucher_value_acquired_usd
        )

    def _attempt_lender_voucher_purchase(self, buyer_kind: str) -> bool:
        self._voucher_purchase_attempts_tick += 1
        self._voucher_purchase_attempts_total += 1
        if buyer_kind == "consumer":
            self._consumer_voucher_purchase_attempts_tick += 1
            self._consumer_voucher_purchase_attempts_total += 1
        else:
            self._third_party_voucher_purchase_attempts_tick += 1
            self._third_party_voucher_purchase_attempts_total += 1

        targets = self._lender_producer_voucher_purchase_targets()
        if not targets:
            self._record_voucher_purchase_failure(buyer_kind, "no_target")
            return False
        weights = np.array([score for score, *_ in targets], dtype=float)
        total = float(weights.sum())
        target_index = int(np.random.choice(len(targets), p=weights / total)) if total > 0.0 else 0
        target_score, lender_pool_id, voucher_id, _voucher_units, voucher_value = targets[target_index]

        sources, saw_stable, saw_reserve_blocked = self._voucher_purchase_source_candidates(
            buyer_kind,
            voucher_id,
        )
        if not sources:
            reason = "reserve_protected" if saw_stable and saw_reserve_blocked else "no_stable"
            self._record_voucher_purchase_failure(buyer_kind, reason)
            return False
        source_pool = self.rng.choice(sources)
        stable_id = self.cfg.stable_symbol
        stable_value = self._asset_value(source_pool, stable_id)
        if stable_value <= 1e-12:
            stable_value = 1.0
        spendable_usd = self._ordinary_source_spendable_amount(source_pool, stable_id) * stable_value
        inventory_share = max(
            0.0,
            float(getattr(self.cfg, "lender_voucher_purchase_inventory_share", 0.05) or 0.0),
        )
        ratio = max(
            0.0,
            float(getattr(self.cfg, "lender_voucher_purchase_stable_to_voucher_value_ratio", 0.563188) or 0.0),
        )
        target_spend_usd = target_score * inventory_share * ratio
        target_usd = getattr(self.cfg, "lender_voucher_purchase_target_usd", None)
        if target_usd is not None:
            target_spend_usd = max(0.0, float(target_usd))
        max_usd = getattr(self.cfg, "lender_voucher_purchase_max_usd", None)
        if max_usd is not None:
            target_spend_usd = min(target_spend_usd, max(0.0, float(max_usd)))
        if target_spend_usd > spendable_usd + 1e-9:
            self._apply_voucher_purchase_stable_budget(
                buyer_kind,
                source_pool,
                target_spend_usd - spendable_usd,
            )
            spendable_usd = (
                self._ordinary_source_spendable_amount(source_pool, stable_id) * stable_value
            )
        stable_spend_usd = min(spendable_usd, target_spend_usd)
        min_usd = max(0.0, float(getattr(self.cfg, "lender_voucher_purchase_min_usd", 1.0) or 0.0))
        if stable_spend_usd < min_usd or stable_spend_usd <= 1e-9:
            reason = "reserve_protected" if spendable_usd <= 1e-9 and saw_reserve_blocked else "no_stable"
            self._record_voucher_purchase_failure(buyer_kind, reason)
            return False
        amount_in = stable_spend_usd / stable_value
        plan, amount_used, used_fallback, _asset_out = self._find_buddy_direct_plan(
            source_pool=source_pool,
            asset_in=stable_id,
            amount_in=amount_in,
            buddy_pools={lender_pool_id},
            target_asset=voucher_id,
        )
        if not plan.ok:
            self._record_voucher_purchase_failure(buyer_kind, "no_route")
            self._record_swap_attempt(source_pool.pool_id, success=False)
            return False
        lender_pool = self.pools[lender_pool_id]
        okq, _reason, amount_out_net, _fee = lender_pool.quote_swap(stable_id, amount_used, voucher_id)
        voucher_value_acquired_usd = amount_out_net * voucher_value if okq else 0.0
        ok = self.execute_route_from_pool(
            source_pool.pool_id,
            plan,
            amount_used,
            route_context=f"{buyer_kind}_purchase",
        )
        self._record_swap_attempt(source_pool.pool_id, success=ok)
        if not ok:
            self._record_voucher_purchase_failure(buyer_kind, "no_route")
            return False
        self._record_voucher_purchase_success(
            buyer_kind,
            amount_used * stable_value,
            voucher_value_acquired_usd,
        )
        self.log.add(Event(
            self.tick,
            "LENDER_VOUCHER_PURCHASE_EXECUTED",
            pool_id=source_pool.pool_id,
            asset_id=stable_id,
            amount=amount_used * stable_value,
            meta={
                "buyer_kind": buyer_kind,
                "target_voucher": voucher_id,
                "lender_pool_id": lender_pool_id,
                "voucher_value_acquired_usd": voucher_value_acquired_usd,
                "fallback": bool(used_fallback),
            },
        ))
        return True

    def _apply_lender_voucher_purchase_demand(self) -> None:
        if not bool(getattr(self.cfg, "lender_voucher_purchase_demand_enabled", False)):
            return
        attempts = max(0, int(getattr(self.cfg, "lender_voucher_purchase_attempts_per_tick", 0) or 0))
        if attempts <= 0:
            return
        total_budget = max(
            0.0,
            float(getattr(self.cfg, "lender_voucher_purchase_stable_budget_usd_per_tick", 0.0) or 0.0),
        )
        external_budget = max(
            0.0,
            float(
                getattr(
                    self.cfg,
                    "external_nonproducer_stable_to_voucher_budget_usd_per_tick",
                    0.0,
                )
                or 0.0
            ),
        )
        other_producer_budget = max(
            0.0,
            float(
                getattr(
                    self.cfg,
                    "other_producer_stable_to_voucher_budget_usd_per_tick",
                    0.0,
                )
                or 0.0
            ),
        )
        consumer_share = max(
            0.0,
            min(1.0, float(getattr(self.cfg, "lender_voucher_purchase_consumer_share", 0.75) or 0.0)),
        )
        if external_budget > 1e-9 or other_producer_budget > 1e-9:
            self._lender_voucher_purchase_stable_budget_remaining_by_kind_tick = {
                "consumer": external_budget,
                "third_party": other_producer_budget,
            }
            self._lender_voucher_purchase_stable_budget_remaining_usd_tick = external_budget + other_producer_budget
        else:
            self._lender_voucher_purchase_stable_budget_remaining_by_kind_tick = {}
            self._lender_voucher_purchase_stable_budget_remaining_usd_tick = total_budget
        for _ in range(attempts):
            buyer_kind = "consumer" if self.rng.random() < consumer_share else "third_party"
            self._attempt_lender_voucher_purchase(buyer_kind)

    def _producer_debt_pressure_capacity_available_usd(self, producer_pool: "Pool") -> float:
        return max(
            0.0,
            float(self._producer_debt_service_capacity_by_pool.get(producer_pool.pool_id, 0.0)),
        )

    def _producer_debt_pressure_stable_available_usd(self, producer_pool: "Pool") -> float:
        stable_id = self.cfg.stable_symbol
        stable_value = self._asset_value(producer_pool, stable_id)
        if stable_value <= 0.0:
            stable_value = 1.0
        local_usd = self._producer_debt_stable_available(producer_pool) * stable_value
        return max(0.0, local_usd) + self._producer_debt_pressure_capacity_available_usd(producer_pool)

    def _consume_producer_debt_pressure_stable(
        self,
        producer_pool: "Pool",
        amount_usd: float,
        counterparty: str,
    ) -> float:
        amount = max(0.0, float(amount_usd))
        if amount <= 1e-9:
            return 0.0
        stable_id = self.cfg.stable_symbol
        stable_value = self._asset_value(producer_pool, stable_id)
        if stable_value <= 0.0:
            stable_value = 1.0
        local_available_usd = self._producer_debt_stable_available(producer_pool) * stable_value
        local_usd = min(amount, max(0.0, local_available_usd))
        consumed = 0.0
        if local_usd > 1e-9:
            local_units = local_usd / stable_value
            if self._vault_sub(
                producer_pool,
                stable_id,
                local_units,
                "producer_debt_pressure_repayment_out",
                counterparty,
            ):
                consumed += local_usd
            else:
                local_usd = 0.0
        capacity_needed = max(0.0, amount - consumed)
        capacity_available = self._producer_debt_pressure_capacity_available_usd(producer_pool)
        capacity_usd = min(capacity_needed, capacity_available)
        if capacity_usd > 1e-9:
            self._producer_debt_service_capacity_by_pool[producer_pool.pool_id] = max(
                0.0,
                capacity_available - capacity_usd,
            )
            self._producer_debt_service_capacity_onramp_usd_tick += capacity_usd
            self._producer_debt_service_capacity_onramp_usd_total += capacity_usd
            self._stable_onramp_usd_tick += capacity_usd
            consumed += capacity_usd
            self.log.add(Event(
                self.tick,
                "PRODUCER_DEBT_SERVICE_CAPACITY_ONRAMPED",
                pool_id=producer_pool.pool_id,
                asset_id=stable_id,
                amount=capacity_usd,
                meta={"counterparty": counterparty},
            ))
        return consumed

    def _producer_debt_penalty_rate_per_period(self) -> float:
        configured = getattr(self.cfg, "producer_debt_penalty_rate_per_period", None)
        if configured is None:
            configured = getattr(self.cfg, "pool_fee_rate", 0.0)
        return max(0.0, float(configured or 0.0))

    def _apply_producer_debt_pressure_penalty(
        self,
        obligation: ProducerDebtObligation,
        missed_usd: float,
    ) -> float:
        missed = max(0.0, float(missed_usd))
        if missed <= 1e-9 or not bool(getattr(self.cfg, "producer_debt_penalty_enabled", True)):
            return 0.0
        rate = self._producer_debt_penalty_rate_per_period()
        penalty = missed * rate
        if penalty <= 1e-9:
            return 0.0
        obligation.cash_service_remaining_usd += penalty
        obligation.cash_service_arrears_usd += penalty
        obligation.cash_service_penalty_remaining_usd += penalty
        self._invalidate_producer_activity_composition_share_cache(obligation.producer_pool_id)
        self._producer_debt_penalty_accrued_usd_tick += penalty
        self._producer_debt_penalty_accrued_usd_total += penalty
        self.log.add(Event(
            self.tick,
            "PRODUCER_DEBT_PENALTY_ACCRUED",
            pool_id=obligation.lender_pool_id,
            asset_id=self.cfg.stable_symbol,
            amount=penalty,
            meta={
                "obligation_id": obligation.obligation_id,
                "producer_pool_id": obligation.producer_pool_id,
                "missed_usd": missed,
                "rate": rate,
            },
        ))
        return penalty

    def _producer_debt_pressure_due_usd(
        self,
        obligation: ProducerDebtObligation,
        period: int,
    ) -> float:
        if self._producer_debt_contract_repayment_enabled():
            remaining_cash = max(0.0, float(obligation.cash_service_remaining_usd))
            arrears = min(
                remaining_cash,
                max(0.0, float(getattr(obligation, "cash_service_arrears_usd", 0.0) or 0.0)),
            )
            non_arrears = max(0.0, remaining_cash - arrears)
            remaining_ticks = max(1, obligation.due_tick - self.tick + 1)
            remaining_periods = max(1, int(math.ceil(remaining_ticks / max(1, period))))
            return min(remaining_cash, arrears + (non_arrears / remaining_periods))
        unit_value = self._producer_debt_unit_value(obligation)
        remaining_usd = max(0.0, obligation.remaining_voucher_units * unit_value)
        remaining_ticks = max(1, obligation.due_tick - self.tick + 1)
        remaining_periods = max(1, int(math.ceil(remaining_ticks / max(1, period))))
        arrears = min(
            remaining_usd,
            max(0.0, float(getattr(obligation, "cash_service_arrears_usd", 0.0) or 0.0)),
        )
        return min(remaining_usd, arrears + (max(0.0, remaining_usd - arrears) / remaining_periods))

    def _record_producer_debt_pressure_payment_allocation(
        self,
        obligation: ProducerDebtObligation,
        paid_usd: float,
        due_usd: float,
    ) -> None:
        paid = max(0.0, float(paid_usd))
        due = max(0.0, float(due_usd))
        penalty_paid = min(
            max(0.0, float(getattr(obligation, "cash_service_penalty_remaining_usd", 0.0) or 0.0)),
            paid,
        )
        if penalty_paid > 1e-9:
            obligation.cash_service_penalty_remaining_usd = max(
                0.0,
                obligation.cash_service_penalty_remaining_usd - penalty_paid,
            )
            self._producer_debt_penalty_paid_usd_tick += penalty_paid
            self._producer_debt_penalty_paid_usd_total += penalty_paid
        missed = max(0.0, due - min(paid, due))
        obligation.cash_service_arrears_usd = missed
        self._apply_producer_debt_pressure_penalty(obligation, missed)
        self._invalidate_producer_activity_composition_share_cache(obligation.producer_pool_id)

    def _execute_producer_self_repayment_swap(
        self,
        producer_pool: "Pool",
        obligation: ProducerDebtObligation,
        target_usd: float,
        purpose: str,
    ) -> float:
        lender_pool = self.pools.get(obligation.lender_pool_id)
        if lender_pool is None or lender_pool.policy.role != "lender":
            return 0.0
        stable_id = self.cfg.stable_symbol
        voucher_id = obligation.voucher_id
        stable_value = self._asset_value(lender_pool, stable_id)
        voucher_value = self._asset_value(lender_pool, voucher_id)
        if stable_value <= 0.0:
            stable_value = 1.0
        if voucher_value <= 0.0:
            voucher_value = self._producer_debt_unit_value(obligation)
        if voucher_value <= 0.0:
            return 0.0
        available_usd = self._producer_debt_pressure_stable_available_usd(producer_pool)
        if available_usd <= 1e-9:
            return 0.0
        max_by_inventory_usd = lender_pool.vault.get(voucher_id) * voucher_value
        exposure_key = (lender_pool.pool_id, voucher_id)
        max_by_exposure_usd = max(
            0.0,
            self._lender_producer_voucher_exposure_usd_by_pool_voucher.get(exposure_key, 0.0),
        )
        max_swappable_usd = max_by_inventory_usd
        if max_by_exposure_usd > 1e-9:
            max_swappable_usd = min(max_swappable_usd, max_by_exposure_usd)
        spend_usd = min(max(0.0, float(target_usd)), available_usd, max_swappable_usd)
        if spend_usd <= 1e-9:
            return 0.0
        amount_in = spend_usd / stable_value
        receipt = lender_pool.execute_swap(
            self.tick,
            actor=f"producer_debt_pressure:{producer_pool.pool_id}",
            asset_in=stable_id,
            amount_in=amount_in,
            asset_out=voucher_id,
        )
        if receipt.status != "executed":
            self._noam_update_edge_after_swap(
                lender_pool,
                receipt.asset_in,
                receipt.asset_out,
                float(receipt.amount_in),
                success=False,
                fail_reason=receipt.fail_reason,
            )
            self.log.add(Event(
                self.tick,
                "PRODUCER_SELF_REPAYMENT_SWAP_FAILED",
                pool_id=lender_pool.pool_id,
                asset_id=stable_id,
                amount=spend_usd,
                meta={
                    "obligation_id": obligation.obligation_id,
                    "producer_pool_id": producer_pool.pool_id,
                    "reason": receipt.fail_reason,
                    "purpose": purpose,
                },
            ))
            return 0.0
        consumed_usd = self._consume_producer_debt_pressure_stable(
            producer_pool,
            spend_usd,
            lender_pool.pool_id,
        )
        if consumed_usd + 1e-6 < spend_usd:
            spend_usd = consumed_usd
        gross_voucher_units = float(receipt.amount_out) + float(receipt.fees.total_fee)
        voucher_removed_usd = gross_voucher_units * voucher_value
        net_voucher_removed_usd = float(receipt.amount_out) * voucher_value
        self.log.add(Event(
            self.tick,
            "SWAP_EXECUTED",
            pool_id=lender_pool.pool_id,
            meta={
                "receipt": receipt.to_dict(),
                "route_context": "repayment",
                "route_source_pool_id": producer_pool.pool_id,
                "route_source_role": producer_pool.policy.role,
                "route_source_asset": stable_id,
                "producer_self_repayment": True,
            },
        ))
        self._update_pool_caches(lender_pool, receipt.asset_in, float(receipt.amount_in))
        self._update_pool_caches(lender_pool, receipt.asset_out, -float(gross_voucher_units))
        self._record_fee_cumulative(receipt)
        self._record_voucher_fee_retained_for_service(lender_pool, receipt)
        self._record_recent_clc_fee(lender_pool, receipt)
        self._record_clc_swap_cumulative(receipt)
        self._noam_update_edge_after_swap(
            lender_pool,
            receipt.asset_in,
            receipt.asset_out,
            float(receipt.amount_in),
            success=True,
        )
        self._noam_routing_swaps_tick += 1
        self._swap_volume_usd_tick += spend_usd
        self._swap_volume_usd_by_pool[lender_pool.pool_id] = (
            self._swap_volume_usd_by_pool.get(lender_pool.pool_id, 0.0) + spend_usd
        )
        self._record_noam_fee_diagnostics(
            lender_pool,
            receipt,
            kind="routing",
            swap_usd=spend_usd,
        )
        self._record_route_context_swap("repayment", producer_pool, stable_id, spend_usd)
        self._record_route_motif(
            route_context="repayment",
            source_pool=producer_pool,
            asset_in=stable_id,
            asset_out=voucher_id,
            amount_in=amount_in,
            plan=RoutePlan(
                ok=True,
                reason="producer_self_repayment",
                hops=[
                    Hop(
                        pool_id=lender_pool.pool_id,
                        asset_in=stable_id,
                        asset_out=voucher_id,
                        amount_in=amount_in,
                    )
                ],
            ),
        )
        self._update_affinity(producer_pool.pool_id, lender_pool.pool_id, spend_usd)
        eligible_recovery_usd = self._reduce_lender_producer_voucher_exposure(
            lender_pool.pool_id,
            voucher_id,
            voucher_removed_usd,
            "borrower_stable_repayment",
        )
        self._record_lender_recovered_stable(
            lender_pool.pool_id,
            spend_usd,
            "borrower_stable_repayment",
            eligible_amount_usd=min(spend_usd, eligible_recovery_usd),
        )
        self._reduce_producer_debt_obligations(
            lender_pool.pool_id,
            voucher_id,
            gross_voucher_units,
            "borrower_stable_repayment",
            source_pool_id=producer_pool.pool_id,
            source_role=producer_pool.policy.role,
        )
        agent = self.agents.get(producer_pool.steward_id)
        if agent is not None and float(receipt.amount_out) > 1e-9:
            agent.issuer.return_to_issuer(float(receipt.amount_out))
            self._mark_lender_voucher_limits_dirty(voucher_id)
        self._producer_self_repayment_swap_volume_usd_tick += spend_usd
        self._producer_self_repayment_swap_volume_usd_total += spend_usd
        self._producer_self_repayment_voucher_removed_usd_tick += net_voucher_removed_usd
        self._producer_self_repayment_voucher_removed_usd_total += net_voucher_removed_usd
        self.log.add(Event(
            self.tick,
            "PRODUCER_SELF_REPAYMENT_SWAP_EXECUTED",
            pool_id=lender_pool.pool_id,
            asset_id=stable_id,
            amount=spend_usd,
            meta={
                "obligation_id": obligation.obligation_id,
                "producer_pool_id": producer_pool.pool_id,
                "voucher_id": voucher_id,
                "voucher_removed_usd": net_voucher_removed_usd,
                "gross_voucher_removed_usd": voucher_removed_usd,
                "purpose": purpose,
            },
        ))
        return spend_usd

    def _producer_debt_pressure_group_remaining_usd(
        self,
        obligations: list[ProducerDebtObligation],
    ) -> float:
        total = 0.0
        for obligation in obligations:
            remaining_cash = max(0.0, float(obligation.cash_service_remaining_usd))
            remaining_voucher_usd = max(
                0.0,
                obligation.remaining_voucher_units * self._producer_debt_unit_value(obligation),
            )
            total += max(remaining_cash, remaining_voucher_usd)
        return total

    def _producer_debt_pressure_add_due_to_deferred(
        self,
        obligation: ProducerDebtObligation,
        due_usd: float,
    ) -> None:
        due = max(0.0, float(due_usd))
        if due <= 1e-9:
            return
        obligation.pressure_deferred_usd = (
            max(0.0, float(getattr(obligation, "pressure_deferred_usd", 0.0) or 0.0))
            + due
        )
        self._invalidate_producer_activity_composition_share_cache(obligation.producer_pool_id)
        self._producer_debt_pressure_deferred_usd_tick += due
        self._producer_debt_pressure_deferred_usd_total += due
        self.log.add(Event(
            self.tick,
            "PRODUCER_DEBT_PRESSURE_DEFERRED",
            pool_id=obligation.producer_pool_id,
            asset_id=obligation.voucher_id,
            amount=due,
            meta={
                "obligation_id": obligation.obligation_id,
                "lender_pool_id": obligation.lender_pool_id,
                "deferred_balance_usd": obligation.pressure_deferred_usd,
            },
        ))

    def _attempt_debt_pressure_repayment_batched(
        self,
        source_pool: "Pool",
        obligations: list[ProducerDebtObligation],
        period: int,
    ) -> bool:
        min_swap_usd = self._producer_debt_pressure_min_swap_usd()
        obligations_by_lender: Dict[str, list[ProducerDebtObligation]] = {}
        attempted = False
        paid_any = False
        for obligation in obligations:
            if obligation.last_pressure_due_tick == self.tick:
                continue
            due_usd = self._producer_debt_pressure_due_usd(obligation, period)
            obligation.last_pressure_due_tick = self.tick
            if due_usd <= 1e-9:
                continue
            attempted = True
            self._producer_debt_pressure_add_due_to_deferred(obligation, due_usd)
            obligations_by_lender.setdefault(obligation.lender_pool_id, []).append(obligation)

        for obligation in obligations:
            if max(0.0, float(getattr(obligation, "pressure_deferred_usd", 0.0) or 0.0)) <= 1e-9:
                continue
            obligations_by_lender.setdefault(obligation.lender_pool_id, []).append(obligation)

        for lender_pool_id, lender_obligations in obligations_by_lender.items():
            unique: dict[int, ProducerDebtObligation] = {
                obligation.obligation_id: obligation for obligation in lender_obligations
            }
            group = sorted(unique.values(), key=self._producer_debt_obligation_sort_key)
            group_deferred_usd = sum(
                max(0.0, float(getattr(obligation, "pressure_deferred_usd", 0.0) or 0.0))
                for obligation in group
            )
            if group_deferred_usd <= 1e-9:
                continue
            force_settle = any(obligation.due_tick <= self.tick for obligation in group)
            if min_swap_usd > 1e-9 and group_deferred_usd + 1e-9 < min_swap_usd and not force_settle:
                continue

            available_usd = self._producer_debt_pressure_stable_available_usd(source_pool)
            prepay_share = max(
                0.0,
                min(1.0, float(getattr(self.cfg, "producer_debt_pressure_prepay_share", 0.10) or 0.0)),
            )
            remaining_after_deferred = max(
                0.0,
                self._producer_debt_pressure_group_remaining_usd(group) - group_deferred_usd,
            )
            prepay_target_usd = min(
                remaining_after_deferred,
                max(0.0, available_usd - group_deferred_usd) * prepay_share,
            )
            target_usd = group_deferred_usd + prepay_target_usd
            representative = group[0]
            paid = self._execute_producer_self_repayment_swap(
                source_pool,
                representative,
                target_usd,
                "batched_due_and_prepayment" if prepay_target_usd > 1e-9 else "batched_scheduled_due",
            )
            if paid > 1e-9:
                paid_any = True
                self._producer_debt_pressure_batched_swap_count_tick += 1
                self._producer_debt_pressure_batched_swap_count_total += 1
                self._producer_debt_pressure_batched_swap_volume_usd_tick += paid
                self._producer_debt_pressure_batched_swap_volume_usd_total += paid

            remaining_paid = paid
            for obligation in group:
                deferred = max(
                    0.0,
                    float(getattr(obligation, "pressure_deferred_usd", 0.0) or 0.0),
                )
                if deferred <= 1e-9:
                    continue
                applied = min(deferred, remaining_paid)
                remaining_paid = max(0.0, remaining_paid - applied)
                obligation.pressure_deferred_usd = 0.0
                self._invalidate_producer_activity_composition_share_cache(obligation.producer_pool_id)
                self._record_producer_debt_pressure_payment_allocation(
                    obligation,
                    applied,
                    deferred,
                )

            prepay_paid = max(0.0, paid - group_deferred_usd)
            if prepay_paid > 1e-9:
                self._producer_debt_pressure_prepayment_usd_tick += prepay_paid
                self._producer_debt_pressure_prepayment_usd_total += prepay_paid

            self.log.add(Event(
                self.tick,
                "PRODUCER_DEBT_PRESSURE_BATCH_SETTLED",
                pool_id=lender_pool_id,
                asset_id=representative.voucher_id,
                amount=paid,
                meta={
                    "producer_pool_id": source_pool.pool_id,
                    "scheduled_deferred_usd": group_deferred_usd,
                    "prepay_target_usd": prepay_target_usd,
                    "min_swap_usd": min_swap_usd,
                    "force_settle": force_settle,
                    "obligation_count": len(group),
                },
            ))

        return attempted or paid_any

    def _attempt_debt_pressure_repayment(
        self,
        source_pool: "Pool",
        voucher_id: str,
        period: int,
    ) -> bool:
        if not self._producer_debt_pressure_enabled():
            return False
        obligations = [
            obligation
            for obligation in sorted(self._producer_debt_obligations, key=self._producer_debt_obligation_sort_key)
            if obligation.producer_pool_id == source_pool.pool_id
            and obligation.voucher_id == voucher_id
            and (
                obligation.remaining_voucher_units > 1e-9
                or max(0.0, obligation.cash_service_remaining_usd) > 1e-9
            )
        ]
        if not obligations:
            return False
        if self._producer_debt_pressure_batching_enabled():
            return self._attempt_debt_pressure_repayment_batched(source_pool, obligations, period)
        attempted = False
        paid_any = False
        for obligation in obligations:
            if obligation.last_pressure_due_tick == self.tick:
                continue
            due_usd = self._producer_debt_pressure_due_usd(obligation, period)
            obligation.last_pressure_due_tick = self.tick
            if due_usd <= 1e-9:
                continue
            attempted = True
            paid = self._execute_producer_self_repayment_swap(
                source_pool,
                obligation,
                due_usd,
                "scheduled_due",
            )
            if paid > 1e-9:
                paid_any = True
            self._record_producer_debt_pressure_payment_allocation(obligation, paid, due_usd)
        if paid_any:
            prepay_share = max(
                0.0,
                min(1.0, float(getattr(self.cfg, "producer_debt_pressure_prepay_share", 0.10) or 0.0)),
            )
            prepay_budget = self._producer_debt_pressure_stable_available_usd(source_pool) * prepay_share
            for obligation in obligations:
                if prepay_budget <= 1e-9:
                    break
                remaining_cash = max(0.0, float(obligation.cash_service_remaining_usd))
                remaining_voucher_usd = max(
                    0.0,
                    obligation.remaining_voucher_units * self._producer_debt_unit_value(obligation),
                )
                remaining_usd = max(remaining_cash, remaining_voucher_usd)
                if remaining_usd <= 1e-9:
                    continue
                paid = self._execute_producer_self_repayment_swap(
                    source_pool,
                    obligation,
                    min(prepay_budget, remaining_usd),
                    "prepayment",
                )
                if paid <= 1e-9:
                    continue
                paid_any = True
                prepay_budget = max(0.0, prepay_budget - paid)
                self._producer_debt_pressure_prepayment_usd_tick += paid
                self._producer_debt_pressure_prepayment_usd_total += paid
        return attempted or paid_any

    def _producer_debt_contract_cash_due_usd(self, producer_pool_id: str, voucher_id: str) -> float:
        if not self._producer_debt_contract_repayment_enabled():
            return 0.0
        total = 0.0
        for obligation in self._producer_debt_obligations:
            if obligation.producer_pool_id != producer_pool_id or obligation.voucher_id != voucher_id:
                continue
            total += max(0.0, obligation.cash_service_remaining_usd)
        return total

    def _attempt_contract_cash_service_repayment(
        self,
        source_pool: "Pool",
        voucher_id: str,
        period: int,
    ) -> bool:
        stable_id = self.cfg.stable_symbol
        stable_value = self._asset_value(source_pool, stable_id)
        if stable_value <= 0.0:
            stable_value = 1.0
        available_usd = self._producer_debt_stable_available(source_pool) * stable_value
        if available_usd <= 1e-9:
            return False

        obligations = [
            obligation
            for obligation in sorted(
                self._producer_debt_obligations,
                key=lambda item: (item.due_tick, item.issued_tick, item.obligation_id),
            )
            if obligation.producer_pool_id == source_pool.pool_id
            and obligation.voucher_id == voucher_id
            and obligation.cash_service_remaining_usd > 1e-9
        ]
        if not obligations:
            return False

        scheduled_payment_usd = 0.0
        for obligation in obligations:
            remaining_ticks = max(1, obligation.due_tick - self.tick + 1)
            remaining_periods = max(1, int(math.ceil(remaining_ticks / max(1, period))))
            scheduled_payment_usd += obligation.cash_service_remaining_usd / remaining_periods
        target_usd = min(available_usd, scheduled_payment_usd)
        if target_usd <= 1e-9:
            return False

        paid_usd = 0.0
        for obligation in obligations:
            if paid_usd + 1e-9 >= target_usd:
                break
            need = min(obligation.cash_service_remaining_usd, target_usd - paid_usd)
            paid_usd += self._execute_producer_debt_cash_service_payment(
                obligation,
                need,
                "borrower_stable_repayment",
            )

        if paid_usd > 1e-9:
            self.log.add(Event(
                self.tick,
                "CONTRACT_REPAYMENT_EXECUTED",
                pool_id=source_pool.pool_id,
                asset_id=stable_id,
                amount=paid_usd,
                meta={"voucher_id": voucher_id, "scheduled_payment_usd": scheduled_payment_usd},
            ))
            return True
        return False

    def _attempt_repayment(self, source_pool: "Pool") -> bool:
        if source_pool.policy.role != "producer":
            return False
        agent = self.agents.get(source_pool.steward_id)
        if agent is None:
            return False
        voucher_id = agent.voucher_spec.voucher_id
        period = (
            self._producer_debt_pressure_period_ticks()
            if self._producer_debt_pressure_enabled()
            else max(1, int(self.cfg.loan_activity_period_ticks or 1))
        )
        in_phase = True
        if period > 1:
            phase = self._loan_phase_for(agent.agent_id, period)
            in_phase = (self.tick + phase) % period == 0
        if self._producer_debt_pressure_enabled() and in_phase:
            pressure_attempted = self._attempt_debt_pressure_repayment(source_pool, voucher_id, period)
            if pressure_attempted:
                return True
        if self._producer_debt_contract_cash_due_usd(source_pool.pool_id, voucher_id) > 1e-9:
            if not in_phase:
                return False
            return self._attempt_contract_cash_service_repayment(source_pool, voucher_id, period)
        active_obligations = [
            obligation
            for obligation in self._producer_debt_obligations
            if obligation.producer_pool_id == source_pool.pool_id
            and obligation.voucher_id == voucher_id
            and obligation.remaining_voucher_units > 1e-9
        ]
        lender_pools = {
            obligation.lender_pool_id
            for obligation in active_obligations
            if obligation.lender_pool_id in self.pools
            and self.pools[obligation.lender_pool_id].policy.role == "lender"
        }
        debt = sum(obligation.remaining_voucher_units for obligation in active_obligations)
        if debt <= 1e-9:
            if not in_phase:
                return False
            if bool(getattr(self.cfg, "producer_primary_voucher_borrowing_enabled", False)):
                share = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            getattr(
                                self.cfg,
                                "producer_primary_voucher_borrowing_attempt_share",
                                0.0,
                            )
                            or 0.0
                        ),
                    ),
                )
                if share > 0.0 and self.rng.random() < share:
                    self._producer_primary_voucher_loan_attempts_tick += 1
                    self._producer_primary_voucher_loan_attempts_total += 1
                    attempted = self._attempt_voucher_loan_fallback(
                        source_pool,
                        voucher_id,
                        require_fallback_enabled=False,
                    )
                    if attempted:
                        self._producer_primary_voucher_loan_executed_tick += 1
                        self._producer_primary_voucher_loan_executed_total += 1
                        return True
            return self._attempt_new_loan(source_pool, voucher_id)
        if not in_phase:
            return False
        if not lender_pools:
            return False

        # repay in stable only
        inv = source_pool.vault.inventory
        asset_in = self.cfg.stable_symbol if inv.get(self.cfg.stable_symbol, 0.0) > 1e-9 else None
        if asset_in is None:
            return False

        available = inv.get(asset_in, 0.0)
        scheduled_payment_usd = 0.0
        for obligation in self._producer_debt_obligations:
            if obligation.producer_pool_id != source_pool.pool_id:
                continue
            if obligation.voucher_id != voucher_id or obligation.lender_pool_id not in lender_pools:
                continue
            if obligation.remaining_voucher_units <= 1e-9:
                continue
            remaining_ticks = max(1, obligation.due_tick - self.tick + 1)
            remaining_periods = max(1, int(math.ceil(remaining_ticks / period)))
            scheduled_payment_usd += (
                obligation.remaining_voucher_units
                * self._producer_debt_unit_value(obligation)
                / remaining_periods
            )
        debt_value = debt * self._asset_value(source_pool, voucher_id)
        if scheduled_payment_usd > 1e-9:
            payment_usd = scheduled_payment_usd
        else:
            payment_usd = debt_value / max(1, self.cfg.loan_term_weeks)
            if period > 1:
                payment_usd *= period
        payment_usd = min(payment_usd, debt_value)
        amount_in = min(available, payment_usd / self._asset_value(source_pool, asset_in))
        if amount_in <= 1e-9:
            return False

        buddy_pools = None
        if bool(self.cfg.affinity_buddy_direct_only):
            buddy_pools = self._affinity_buddies_for_pool(source_pool.pool_id)
        if buddy_pools:
            plan, amount_used, used_fallback, asset_out = self._find_buddy_direct_plan(
                source_pool=source_pool,
                asset_in=asset_in,
                amount_in=amount_in,
                buddy_pools=buddy_pools,
                target_asset=voucher_id,
            )
            if plan.ok and asset_out is not None:
                meta = {"target_asset": asset_out, "repayment": True, "buddy_direct": True}
                if used_fallback:
                    meta["fallback"] = True
                self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                                   asset_id=asset_in, amount=amount_used, meta=meta))
                self.log.add(Event(self.tick, "ROUTE_FOUND", pool_id=source_pool.pool_id,
                                   meta={"hops": [h.__dict__ for h in plan.hops], "target": asset_out, "repayment": True, "buddy_direct": True}))
                ok = self.execute_route_from_pool(
                    source_pool.pool_id, plan, amount_used, route_context="repayment"
                )
                self._record_swap_attempt(source_pool.pool_id, success=ok)
                if ok:
                    amount_usd = amount_used * self._asset_value(source_pool, asset_in)
                    self.log.add(Event(self.tick, "REPAYMENT_EXECUTED", pool_id=source_pool.pool_id,
                                       asset_id=self.cfg.stable_symbol, amount=amount_usd,
                                       meta={"asset_in": asset_in, "amount_in": amount_used}))
                return True

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
            pool_whitelist=buddy_pools,
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
        ok = self.execute_route_from_pool(
            source_pool.pool_id, plan, amount_used, route_context="repayment"
        )
        self._record_swap_attempt(source_pool.pool_id, success=ok)
        if ok:
            amount_usd = amount_used * self._asset_value(source_pool, asset_in)
            self.log.add(Event(self.tick, "REPAYMENT_EXECUTED", pool_id=source_pool.pool_id,
                               asset_id=self.cfg.stable_symbol, amount=amount_usd,
                               meta={"asset_in": asset_in, "amount_in": amount_used}))
        return True

    def _producer_debt_outstanding(self, source_pool: "Pool") -> float:
        if source_pool.policy.role != "producer":
            return 0.0
        agent = self.agents.get(source_pool.steward_id)
        if agent is None:
            return 0.0
        voucher_id = agent.voucher_spec.voucher_id
        debt = 0.0
        for obligation in self._producer_debt_obligations:
            if obligation.producer_pool_id != source_pool.pool_id:
                continue
            if obligation.voucher_id != voucher_id:
                continue
            if obligation.remaining_voucher_units <= 1e-9:
                continue
            debt += obligation.remaining_voucher_units
        return debt

    def _voucher_loan_target_candidates(
        self,
        voucher_id: str,
    ) -> list[tuple[float, str, Set[str], float, float]]:
        candidates_by_asset: Dict[str, tuple[float, Set[str], float, float]] = {}
        for pool in self.pools.values():
            if pool.policy.role != "lender" or not pool.registry.is_listed(voucher_id):
                continue
            cap = self._lender_voucher_cap(voucher_id, lender_pool=pool)
            if pool.policy.limits_enabled:
                remaining = pool.limiter.remaining(self.tick, voucher_id)
            else:
                remaining = math.inf
            finite_remaining = remaining if math.isfinite(remaining) else cap
            borrowable = min(max(0.0, cap), max(0.0, finite_remaining))
            if borrowable <= 1e-9:
                continue
            for asset_id, amount in pool.vault.inventory.items():
                if asset_id == voucher_id or not asset_id.startswith("VCHR:") or amount <= 1e-9:
                    continue
                if not pool.registry.is_listed(asset_id):
                    continue
                score, pools, max_cap, max_remaining = candidates_by_asset.get(
                    asset_id, (0.0, set(), 0.0, 0.0)
                )
                value = self._asset_value(pool, asset_id)
                score += max(0.0, amount * value)
                pools.add(pool.pool_id)
                max_cap = max(max_cap, cap)
                max_remaining = max(max_remaining, finite_remaining)
                candidates_by_asset[asset_id] = (score, pools, max_cap, max_remaining)
        candidates = [
            (score, asset_id, pools, max_cap, max_remaining)
            for asset_id, (score, pools, max_cap, max_remaining) in candidates_by_asset.items()
            if score > 0.0 and pools
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)
        limit = max(1, int(getattr(self.cfg, "producer_voucher_loan_max_target_candidates", 3) or 3))
        return candidates[:limit]

    def _attempt_voucher_loan_fallback(
        self,
        source_pool: "Pool",
        voucher_id: str,
        *,
        sampled_amount_in: float | None = None,
        value: float | None = None,
        require_fallback_enabled: bool = True,
    ) -> bool:
        if require_fallback_enabled and not bool(
            getattr(self.cfg, "producer_voucher_loan_fallback_enabled", False)
        ):
            return False
        self._producer_voucher_loan_attempts_tick += 1
        value = float(value if value is not None else self._asset_value(source_pool, voucher_id))
        amount_in = (
            float(sampled_amount_in)
            if sampled_amount_in is not None
            else self._sample_amount_in(source_pool, voucher_id)
        )
        have = source_pool.vault.get(voucher_id)
        if have <= 1e-9:
            self._producer_voucher_loan_no_inventory_tick += 1
            return False
        if amount_in <= 1e-9 or value <= 1e-12:
            self._producer_voucher_loan_zero_amount_tick += 1
            return False
        if amount_in > have:
            amount_in = have
        candidates = self._voucher_loan_target_candidates(voucher_id)
        if not candidates:
            self._producer_voucher_loan_no_target_tick += 1
            return False

        attempted_any = False
        for _score, target_asset, target_pools, max_cap, max_remaining in candidates:
            candidate_amount = amount_in
            if max_cap > 1e-9 and candidate_amount > max_cap:
                self._producer_voucher_loan_clipped_lender_cap_usd_tick += (
                    candidate_amount - max_cap
                ) * value
                candidate_amount = max_cap
            if max_remaining > 1e-9 and candidate_amount > max_remaining:
                self._producer_voucher_loan_clipped_lender_remaining_usd_tick += (
                    candidate_amount - max_remaining
                ) * value
                candidate_amount = max_remaining
            if candidate_amount <= 1e-9:
                continue
            attempted_any = True
            self._producer_voucher_loan_attempted_usd_tick += candidate_amount * value
            self.log.add(Event(
                self.tick,
                "ROUTE_REQUESTED",
                pool_id=source_pool.pool_id,
                asset_id=voucher_id,
                amount=candidate_amount,
                meta={"target_asset": target_asset, "borrow_voucher": True},
            ))
            plan, amount_used, used_fallback, _direct_asset = self._find_buddy_direct_plan(
                source_pool=source_pool,
                asset_in=voucher_id,
                amount_in=candidate_amount,
                buddy_pools=target_pools,
                target_asset=target_asset,
            )
            if not plan.ok:
                plan, amount_used, used_fallback = self._find_route_with_fallback(
                    tick=self.tick,
                    start_asset=voucher_id,
                    target_asset=target_asset,
                    amount_in=candidate_amount,
                    source_pool=source_pool,
                    target_pools=target_pools,
                )
            if used_fallback:
                self.log.add(Event(
                    self.tick,
                    "ROUTE_REQUESTED",
                    pool_id=source_pool.pool_id,
                    asset_id=voucher_id,
                    amount=amount_used,
                    meta={"target_asset": target_asset, "borrow_voucher": True, "fallback": True},
                ))
            if not plan.ok:
                self._producer_voucher_loan_route_failed_tick += 1
                self.log.add(Event(
                    self.tick,
                    "ROUTE_FAILED",
                    pool_id=source_pool.pool_id,
                    asset_id=voucher_id,
                    amount=amount_used,
                    meta={
                        "reason": plan.reason,
                        "target": target_asset,
                        "borrow_voucher": True,
                    },
                ))
                self._record_swap_attempt(source_pool.pool_id, success=False)
                continue

            self._producer_voucher_loan_route_found_tick += 1
            self.log.add(Event(
                self.tick,
                "ROUTE_FOUND",
                pool_id=source_pool.pool_id,
                meta={
                    "hops": [h.__dict__ for h in plan.hops],
                    "target": target_asset,
                    "borrow_voucher": True,
                },
            ))
            ok = self.execute_route_from_pool(
                source_pool.pool_id,
                plan,
                amount_used,
                route_context="voucher_loan",
            )
            self._record_swap_attempt(source_pool.pool_id, success=ok)
            if not ok:
                self._producer_voucher_loan_execution_failed_tick += 1
                continue
            amount_usd = amount_used * value
            self._producer_voucher_loan_executed_tick += 1
            self._producer_voucher_loan_executed_usd_tick += amount_usd
            self._mark_producer_voucher_loan_activity(source_pool.pool_id, voucher_id)
            self.log.add(Event(
                self.tick,
                "VOUCHER_LOAN_ISSUED",
                pool_id=source_pool.pool_id,
                asset_id=target_asset,
                amount=amount_usd,
                meta={"asset_in": voucher_id, "amount_in": amount_used},
            ))
            return True

        if not attempted_any:
            self._producer_voucher_loan_zero_amount_tick += 1
            return False
        return False

    def _attempt_new_loan(self, source_pool: "Pool", voucher_id: str) -> bool:
        self._producer_loan_attempts_tick += 1
        lenders = {
            pid for pid, p in self.pools.items()
            if p.policy.role == "lender"
            and p.vault.get(self.cfg.stable_symbol) > 1e-9
            and p.registry.is_listed(voucher_id)
        }
        if not lenders:
            self._producer_loan_no_lender_tick += 1
            return self._attempt_voucher_loan_fallback(source_pool, voucher_id)

        value = self._asset_value(source_pool, voucher_id)
        amount_in = self._sample_amount_in(source_pool, voucher_id)
        self._producer_loan_sampled_usd_tick += amount_in * value
        have = source_pool.vault.get(voucher_id)
        if have <= 1e-9:
            self._producer_loan_no_inventory_tick += 1
            return False
        if amount_in <= 1e-9 or value <= 1e-12:
            self._producer_loan_zero_amount_tick += 1
            return False
        if amount_in > have:
            self._producer_loan_clipped_inventory_usd_tick += (amount_in - have) * value
            amount_in = have

        max_cap = 0.0
        max_remaining = 0.0
        max_by_stable = 0.0
        max_borrowable = 0.0
        stable_id = self.cfg.stable_symbol
        for pid in lenders:
            pool = self.pools.get(pid)
            if pool is None:
                continue
            cap = self._lender_voucher_cap(voucher_id, lender_pool=pool)
            if cap > max_cap:
                max_cap = cap
            if pool.policy.limits_enabled:
                remaining = pool.limiter.remaining(self.tick, voucher_id)
            else:
                remaining = math.inf
            finite_remaining = remaining if math.isfinite(remaining) else cap
            if finite_remaining > max_remaining:
                max_remaining = finite_remaining
            available_stable = pool.vault.get(stable_id) - pool.policy.min_stable_reserve
            stable_limited_amount = 0.0
            if available_stable > 1e-9:
                lender_value = self._asset_value(pool, voucher_id)
                if lender_value <= 1e-12:
                    lender_value = value
                stable_limited_amount = available_stable / lender_value
                if stable_limited_amount > max_by_stable:
                    max_by_stable = stable_limited_amount
            lender_borrowable = min(max(0.0, cap), max(0.0, finite_remaining), max(0.0, stable_limited_amount))
            if lender_borrowable > max_borrowable:
                max_borrowable = lender_borrowable

        self._producer_loan_lender_collateral_cap_usd_tick += max_cap * value
        self._producer_loan_lender_remaining_cap_usd_tick += max_remaining * value
        self._producer_loan_lender_stable_available_usd_tick += max_by_stable * value
        if max_cap > 1e-9 and amount_in > max_cap:
            self._producer_loan_clipped_lender_cap_usd_tick += (amount_in - max_cap) * value
            amount_in = max_cap
        if max_remaining > 1e-9 and amount_in > max_remaining:
            self._producer_loan_clipped_lender_remaining_usd_tick += (amount_in - max_remaining) * value
            amount_in = max_remaining
        if max_by_stable > 1e-9 and amount_in > max_by_stable:
            self._producer_loan_clipped_lender_stable_usd_tick += (amount_in - max_by_stable) * value
            amount_in = max_by_stable
        if max_borrowable <= 1e-9:
            self._producer_loan_clipped_combined_lender_usd_tick += amount_in * value
            self._producer_loan_zero_amount_tick += 1
            return self._attempt_voucher_loan_fallback(
                source_pool,
                voucher_id,
                sampled_amount_in=amount_in,
                value=value,
            )
        if amount_in > max_borrowable:
            self._producer_loan_clipped_combined_lender_usd_tick += (amount_in - max_borrowable) * value
            amount_in = max_borrowable
        if amount_in <= 1e-9:
            self._producer_loan_zero_amount_tick += 1
            return False
        self._producer_loan_attempted_usd_tick += amount_in * value

        buddy_pools = None
        if bool(self.cfg.affinity_buddy_direct_only):
            buddy_pools = self._affinity_buddies_for_pool(source_pool.pool_id)
        if buddy_pools:
            plan, amount_used, used_fallback, asset_out = self._find_buddy_direct_plan(
                source_pool=source_pool,
                asset_in=voucher_id,
                amount_in=amount_in,
                buddy_pools=buddy_pools,
                target_asset=self.cfg.stable_symbol,
            )
            if plan.ok and asset_out is not None:
                meta = {"target_asset": asset_out, "borrow": True, "buddy_direct": True}
                if used_fallback:
                    meta["fallback"] = True
                self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                                   asset_id=voucher_id, amount=amount_used, meta=meta))
                self.log.add(Event(self.tick, "ROUTE_FOUND", pool_id=source_pool.pool_id,
                                   meta={"hops": [h.__dict__ for h in plan.hops],
                                         "target": asset_out, "borrow": True, "buddy_direct": True}))
                self._producer_loan_route_found_tick += 1
                ok = self.execute_route_from_pool(
                    source_pool.pool_id, plan, amount_used, route_context="loan"
                )
                self._record_swap_attempt(source_pool.pool_id, success=ok)
                if ok:
                    amount_usd = amount_used * self._asset_value(source_pool, voucher_id)
                    self._producer_loan_executed_tick += 1
                    self._producer_loan_executed_usd_tick += amount_usd
                    self._schedule_productive_credit_inflow(source_pool.pool_id, amount_usd, voucher_id)
                    self.log.add(Event(self.tick, "LOAN_ISSUED", pool_id=source_pool.pool_id,
                                       asset_id=self.cfg.stable_symbol, amount=amount_usd,
                                       meta={"asset_in": voucher_id, "amount_in": amount_used}))
                else:
                    self._producer_loan_execution_failed_tick += 1
                return True

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
            pool_whitelist=buddy_pools,
        )
        if used_fallback:
            self.log.add(Event(self.tick, "ROUTE_REQUESTED", pool_id=source_pool.pool_id,
                               asset_id=voucher_id, amount=amount_used,
                               meta={"target_asset": self.cfg.stable_symbol, "borrow": True, "fallback": True}))

        if not plan.ok:
            self.log.add(Event(self.tick, "ROUTE_FAILED", pool_id=source_pool.pool_id,
                               asset_id=voucher_id, amount=amount_used,
                               meta={"reason": plan.reason, "target": self.cfg.stable_symbol, "borrow": True}))
            self._producer_loan_route_failed_tick += 1
            self._record_swap_attempt(source_pool.pool_id, success=False)
            self._backfill_failed_loan_route(source_pool)
            return True

        self.log.add(Event(self.tick, "ROUTE_FOUND", pool_id=source_pool.pool_id,
                           meta={"hops": [h.__dict__ for h in plan.hops],
                                 "target": self.cfg.stable_symbol, "borrow": True}))
        self._producer_loan_route_found_tick += 1
        ok = self.execute_route_from_pool(
            source_pool.pool_id, plan, amount_used, route_context="loan"
        )
        self._record_swap_attempt(source_pool.pool_id, success=ok)
        if ok:
            amount_usd = amount_used * self._asset_value(source_pool, voucher_id)
            self._producer_loan_executed_tick += 1
            self._producer_loan_executed_usd_tick += amount_usd
            self._schedule_productive_credit_inflow(source_pool.pool_id, amount_usd, voucher_id)
            self.log.add(Event(self.tick, "LOAN_ISSUED", pool_id=source_pool.pool_id,
                               asset_id=self.cfg.stable_symbol, amount=amount_usd,
                               meta={"asset_in": voucher_id, "amount_in": amount_used}))
        else:
            self._producer_loan_execution_failed_tick += 1
        return True

    def _redeem_final_route_voucher_output(
        self,
        *,
        source_pool: "Pool",
        source_asset: str,
        source_amount: float,
        voucher_id: str,
        voucher_amount: float,
        route_context: str,
        plan: RoutePlan,
    ) -> bool:
        spec = self.factory.voucher_specs.get(voucher_id)
        issuer_id = spec.issuer_id if spec else None
        if issuer_id not in self.agents:
            self.log.add(Event(
                self.tick,
                "VOUCHER_REDEEM_FAILED",
                pool_id=source_pool.pool_id,
                asset_id=voucher_id,
                amount=voucher_amount,
                meta={"reason": "missing_issuer", "level": "error"},
            ))
            return False

        issuer_agent = self.agents[issuer_id]
        issuer_pool = self.pools.get(issuer_agent.pool_id)
        if issuer_pool is None:
            self.log.add(Event(
                self.tick,
                "VOUCHER_REDEEM_FAILED",
                pool_id=source_pool.pool_id,
                asset_id=voucher_id,
                amount=voucher_amount,
                meta={"reason": "missing_issuer_pool", "level": "error"},
            ))
            return False

        self.log.add(Event(
            self.tick,
            "VOUCHER_EXIT_NETWORK",
            pool_id=source_pool.pool_id,
            asset_id=voucher_id,
            amount=voucher_amount,
            meta={
                "settlement_mode": self._voucher_settlement_mode(),
                "route_context": str(route_context or "ordinary"),
            },
        ))
        issuer_agent.issuer.return_to_issuer(voucher_amount)
        self._mark_lender_voucher_limits_dirty(voucher_id)
        self._vault_add(issuer_pool, voucher_id, voucher_amount, "redeem_receive", source_pool.pool_id)

        value = self._asset_value(issuer_pool, voucher_id)
        redeemed_usd = voucher_amount * value
        self._net_redeemed_voucher_usd_tick += redeemed_usd
        self._net_redeemed_voucher_usd_total += redeemed_usd
        self._voucher_redeemed_to_issuer_usd_tick += redeemed_usd
        self._voucher_redeemed_to_issuer_usd_total += redeemed_usd
        if source_asset == self.cfg.stable_symbol and self._is_producer_voucher(voucher_id):
            self._debt_removal_voucher_redeemed_usd_tick += redeemed_usd
            self._debt_removal_voucher_redeemed_usd_total += redeemed_usd

        self._record_route_motif(
            route_context=route_context,
            source_pool=source_pool,
            asset_in=source_asset,
            asset_out=voucher_id,
            amount_in=source_amount,
            plan=plan,
        )
        self._record_route_source_net_flow(source_pool, source_asset, source_amount, -1.0)
        self._record_route_source_net_flow(issuer_pool, voucher_id, voucher_amount, 1.0)
        self.log.add(Event(
            self.tick,
            "VOUCHER_REDEEMED",
            actor_id=issuer_id,
            asset_id=voucher_id,
            amount=voucher_amount,
            meta={
                "source_pool_id": source_pool.pool_id,
                "redeemed_usd": redeemed_usd,
                "settlement_mode": self._voucher_settlement_mode(),
            },
        ))
        return True

    def execute_route_from_pool(
        self,
        source_pool_id: str,
        plan: RoutePlan,
        amount_in: float,
        route_context: str = "ordinary",
    ) -> bool:
        """
        Withdraw asset_in from source pool into escrow, execute hop swaps, deposit output back to source.
        If output is voucher, it may exit and redeem (your 'final settlement sink').
        """
        if not plan.hops:
            return False

        source_pool = self.pools[source_pool_id]
        asset_in = plan.hops[0].asset_in
        route_context = self._route_context_for_ordinary_own_voucher_source(
            source_pool,
            asset_in,
            plan.hops[-1].asset_out,
            route_context,
        )
        if not self._vault_sub(source_pool, asset_in, amount_in, "route_withdraw", f"escrow:{source_pool_id}"):
            self.log.add(Event(self.tick, "EXEC_ROUTE_FAILED", pool_id=source_pool_id, meta={"reason": "source_insufficient"}))
            return False

        escrow: Dict[str, float] = {asset_in: amount_in}
        actor = f"escrow:{source_pool_id}"

        current_amount = amount_in
        current_asset = asset_in

        def refund_escrow() -> None:
            for a, amt in list(escrow.items()):
                if amt > 1e-9:
                    self._vault_add(source_pool, a, amt, "route_refund", "escrow")
                    escrow[a] = 0.0

        for hop in plan.hops:
            if hop.pool_id == source_pool_id:
                refund_escrow()
                self.log.add(Event(self.tick, "SWAP_FAILED", pool_id=hop.pool_id, asset_id=hop.asset_in, amount=hop.amount_in,
                                   meta={"reason": "self_swap_not_allowed"}))
                return False
            pool = self.pools[hop.pool_id]
            if not self._is_routable_pool(pool):
                refund_escrow()
                self.log.add(Event(self.tick, "SWAP_FAILED", pool_id=hop.pool_id, asset_id=hop.asset_in, amount=hop.amount_in,
                                   meta={"reason": "private_wallet_not_open", "hop": hop.__dict__}))
                return False
            if escrow.get(current_asset, 0.0) <= 1e-9:
                refund_escrow()
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
                refund_escrow()
                self.log.add(Event(self.tick, "SWAP_FAILED", pool_id=pool.pool_id, asset_id=current_asset, amount=amt_in,
                                   meta={"reason": receipt.fail_reason, "hop": hop.__dict__}))
                return False

            # update escrow balances
            escrow[current_asset] = escrow.get(current_asset, 0.0) - amt_in
            if escrow[current_asset] <= 1e-12:
                escrow.pop(current_asset, None)
            escrow[hop.asset_out] = escrow.get(hop.asset_out, 0.0) + receipt.amount_out

            self.log.add(Event(self.tick, "SWAP_EXECUTED", pool_id=pool.pool_id,
                   meta={
                       "receipt": receipt.to_dict(),
                       "route_context": str(route_context or "ordinary"),
                       "route_source_pool_id": source_pool_id,
                       "route_source_role": source_pool.policy.role,
                       "route_source_asset": asset_in,
                   }))
            gross_out = receipt.amount_out + float(receipt.fees.total_fee)
            self._update_pool_caches(pool, receipt.asset_in, float(receipt.amount_in))
            self._update_pool_caches(pool, receipt.asset_out, -float(gross_out))
            self._record_fee_cumulative(receipt)
            self._record_voucher_fee_retained_for_service(pool, receipt)
            self._record_recent_clc_fee(pool, receipt)
            self._record_clc_swap_cumulative(receipt)
            if (
                bool(self.cfg.stable_excess_sweep_after_stable_receipt)
                and pool.policy.role in ("producer", "consumer")
                and receipt.asset_in == self.cfg.stable_symbol
            ):
                self._sweep_pool_stable_excess(pool, action="stable_receipt_sweep")
            if (
                pool.policy.role == "lender"
                and receipt.asset_in == self.cfg.stable_symbol
                and self._is_producer_voucher(receipt.asset_out)
            ):
                recovered_reason = self._stable_purchase_recovery_reason(source_pool, receipt.asset_out)
                debt_reduction_reason = recovered_reason
                stable_recovery_usd = float(receipt.amount_in) * self._asset_value(pool, receipt.asset_in)
                voucher_exposure_usd = gross_out * self._asset_value(pool, receipt.asset_out)
                eligible_recovery_usd = self._reduce_lender_producer_voucher_exposure(
                    pool.pool_id,
                    receipt.asset_out,
                    voucher_exposure_usd,
                    recovered_reason,
                )
                self._record_lender_recovered_stable(
                    pool.pool_id,
                    stable_recovery_usd,
                    recovered_reason,
                    eligible_amount_usd=min(stable_recovery_usd, eligible_recovery_usd),
                )
                self._reduce_producer_debt_obligations(
                    pool.pool_id,
                    receipt.asset_out,
                    gross_out,
                    debt_reduction_reason,
                    source_pool_id=source_pool.pool_id,
                    source_role=source_pool.policy.role,
                )
            if (
                pool.policy.role == "lender"
                and source_pool.policy.role == "producer"
                and self._is_producer_voucher(receipt.asset_in)
                and receipt.asset_out == self.cfg.stable_symbol
            ):
                self._register_producer_debt_obligation(
                    source_pool.pool_id,
                    pool.pool_id,
                    receipt.asset_in,
                    float(receipt.amount_in),
                    float(receipt.amount_in) * self._asset_value(pool, receipt.asset_in),
                    debt_kind="stable",
                )
            elif (
                pool.policy.role == "lender"
                and source_pool.policy.role == "producer"
                and self._is_producer_voucher(receipt.asset_in)
                and str(receipt.asset_out or "").startswith("VCHR:")
            ):
                spec = self.factory.voucher_specs.get(receipt.asset_in)
                if spec is not None and spec.issuer_id == source_pool.steward_id:
                    self._register_producer_debt_obligation(
                        source_pool.pool_id,
                        pool.pool_id,
                        receipt.asset_in,
                        float(receipt.amount_in),
                        float(receipt.amount_in) * self._asset_value(pool, receipt.asset_in),
                        debt_kind="voucher",
                    )
                if self._is_producer_voucher(receipt.asset_out):
                    voucher_exposure_usd = gross_out * self._asset_value(pool, receipt.asset_out)
                    self._reduce_lender_producer_voucher_exposure(
                        pool.pool_id,
                        receipt.asset_out,
                        voucher_exposure_usd,
                        "producer_voucher_swap_out",
                    )
                    self._reduce_producer_debt_obligations(
                        pool.pool_id,
                        receipt.asset_out,
                        gross_out,
                        "producer_voucher_swap_out",
                        source_pool_id=source_pool.pool_id,
                        source_role=source_pool.policy.role,
                    )
            elif (
                pool.policy.role == "lender"
                and receipt.asset_in != self.cfg.stable_symbol
                and self._is_producer_voucher(receipt.asset_out)
            ):
                voucher_exposure_usd = gross_out * self._asset_value(pool, receipt.asset_out)
                self._reduce_lender_producer_voucher_exposure(
                    pool.pool_id,
                    receipt.asset_out,
                    voucher_exposure_usd,
                    "producer_voucher_swap_out",
                )
                self._reduce_producer_debt_obligations(
                    pool.pool_id,
                    receipt.asset_out,
                    gross_out,
                    "producer_voucher_swap_out",
                    source_pool_id=source_pool.pool_id,
                    source_role=source_pool.policy.role,
                )
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
            self._record_noam_fee_diagnostics(
                pool,
                receipt,
                kind="routing",
                swap_usd=swap_usd,
            )
            self._record_route_context_swap(route_context, source_pool, asset_in, swap_usd)
            self._update_affinity(source_pool_id, pool.pool_id, swap_usd)

            if pool.policy.role in ("consumer", "producer") and current_asset.startswith("VCHR:"):
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
            if self._redeem_final_voucher_outputs_enabled() or source_pool.policy.role in ("producer", "consumer"):
                ok = self._redeem_final_route_voucher_output(
                    source_pool=source_pool,
                    source_asset=asset_in,
                    source_amount=amount_in,
                    voucher_id=out_asset,
                    voucher_amount=out_amount,
                    route_context=route_context,
                    plan=plan,
                )
                if ok:
                    escrow[out_asset] = 0.0
                    self._noam_route_cache_store(source_pool_id, plan, amount_in)
                return ok
            # lenders/sys_clc keep vouchers from swaps
            self._vault_add(source_pool, out_asset, out_amount, "route_deposit", "escrow")
            self._record_route_motif(
                route_context=route_context,
                source_pool=source_pool,
                asset_in=asset_in,
                asset_out=out_asset,
                amount_in=amount_in,
                plan=plan,
            )
            self._record_route_source_net_flow(source_pool, asset_in, amount_in, -1.0)
            self._record_route_source_net_flow(source_pool, out_asset, out_amount, 1.0)
            self._noam_route_cache_store(source_pool_id, plan, amount_in)
            return True

        # normal asset: deposit back to source pool
        self._vault_add(source_pool, out_asset, out_amount, "route_deposit", "escrow")
        if out_asset == self.cfg.stable_symbol and source_pool.policy.role == "producer":
            stable_value = self._asset_value(source_pool, out_asset)
            if stable_value <= 1e-12:
                stable_value = 1.0
            self._apply_producer_stable_exit(
                source_pool,
                out_amount * stable_value,
                "route_stable_receipt",
            )
        self._record_route_motif(
            route_context=route_context,
            source_pool=source_pool,
            asset_in=asset_in,
            asset_out=out_asset,
            amount_in=amount_in,
            plan=plan,
        )
        self._record_route_source_net_flow(source_pool, asset_in, amount_in, -1.0)
        self._record_route_source_net_flow(source_pool, out_asset, out_amount, 1.0)
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
        issuer.issuer.return_to_issuer(amount)
        self._mark_lender_voucher_limits_dirty(voucher_id)
        self.log.add(Event(self.tick, "VOUCHER_REDEEMED_CONSUMER", actor_id=issuer_id,
                           pool_id=holder_pool.pool_id, asset_id=voucher_id, amount=amount))


    def snapshot_metrics(
        self,
        *,
        force: bool = False,
        force_network: bool = False,
        force_pool: bool = False,
    ) -> None:
        cfg = self.cfg
        metrics_stride = int(cfg.metrics_stride or 0)
        pool_stride = int(cfg.pool_metrics_stride or 0)
        do_network = metrics_stride > 0 and self.tick % metrics_stride == 0
        do_pool = pool_stride > 0 and self.tick % pool_stride == 0
        if force:
            do_network = True
            do_pool = True
        if force_network:
            do_network = True
        if force_pool:
            do_pool = True
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
                    lp_injected = float(self._lp_injected_usd_by_pool.get(pid, 0.0))
                    lp_returned = float(self._lp_returned_usd_by_pool.get(pid, 0.0))
                    lp_net = lp_returned - lp_injected
                    lp_roi = lp_returned / lp_injected if lp_injected > 1e-9 else 0.0
                    lp_sclc_balance = p.vault.get(cfg.sclc_symbol) if cfg.sclc_symbol else 0.0
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
                        "lp_sclc_balance": lp_sclc_balance,
                        "lp_injected_usd": lp_injected,
                        "lp_returned_usd": lp_returned,
                        "lp_net_usd": lp_net,
                        "lp_roi": lp_roi,
                    })
            if do_pool:
                self.metrics.add_pool_rows(pool_rows)

        if do_network:
            swap_receipts = sum(len(p.receipts.receipts) for p in self.pools.values() if not p.policy.system_pool)
            stable_total = sum(
                p.vault.get(cfg.stable_symbol) for p in self.pools.values()
            )
            voucher_total = sum(
                amt
                for p in self.pools.values()
                for asset_id, amt in p.vault.inventory.items()
                if asset_id.startswith("VCHR:")
            )
            active_stable_value = sum(
                p.vault.get(cfg.stable_symbol)
                for p in self.pools.values()
                if not p.policy.system_pool
            )
            active_voucher_value = sum(
                self._pool_voucher_value_usd(p)
                for p in self.pools.values()
                if not p.policy.system_pool
            )
            active_stable_voucher_value = active_stable_value + active_voucher_value
            active_stable_share = active_stable_value / max(1e-9, active_stable_voucher_value)
            active_voucher_share = active_voucher_value / max(1e-9, active_stable_voucher_value)
            active_stable_to_voucher_ratio = active_stable_value / max(1e-9, active_voucher_value)
            role_stress_counts = {role: 0 for role in ("producer", "consumer", "lender")}
            role_under_reserve = {role: 0 for role in ("producer", "consumer", "lender")}
            role_reserve_deficit = {role: 0.0 for role in ("producer", "consumer", "lender")}
            community_pools_total = 0
            community_pools_under_reserve = 0
            community_reserve_deficit = 0.0
            pools_under_reserve = 0
            stable_value = self._asset_value(next(iter(self.pools.values())), cfg.stable_symbol) if self.pools else 1.0
            if stable_value <= 1e-12:
                stable_value = 1.0
            for p in self.pools.values():
                if p.policy.system_pool:
                    continue
                role = p.policy.role
                stable_units = p.vault.get(cfg.stable_symbol)
                deficit_units = max(0.0, p.policy.min_stable_reserve - stable_units)
                under = deficit_units > 1e-9
                if under:
                    pools_under_reserve += 1
                if role in role_stress_counts:
                    role_stress_counts[role] += 1
                    if under:
                        role_under_reserve[role] += 1
                        role_reserve_deficit[role] += deficit_units * stable_value
                if role in ("producer", "consumer"):
                    community_pools_total += 1
                    if under:
                        community_pools_under_reserve += 1
                        community_reserve_deficit += deficit_units * stable_value
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
                        value = self._default_asset_value(asset_id)
                    debt_outstanding_usd += amt * value

            redeemed_total = 0.0
            outstanding_total = sum(a.issuer.outstanding_supply for a in self.agents.values())
            issued_total = sum(a.issuer.issued_total for a in self.agents.values())
            issuer_returned_total = sum(a.issuer.issuer_returned_total for a in self.agents.values())
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
                    value = self._default_asset_value(agent.voucher_spec.voucher_id)
                outstanding_value_usd += outstanding * value

            repayment_volume_usd = 0.0
            loan_issuance_usd = 0.0
            transactions_per_tick = 0
            route_requested = 0
            route_found = 0
            route_failed = 0
            route_fixed_requested = 0
            route_fixed_found = 0
            route_fixed_failed = 0
            route_substitution_requested = 0
            route_substitution_found = 0
            route_substitution_failed = 0
            vol_usd_to_vchr = 0.0
            vol_vchr_to_usd = 0.0
            vol_vchr_to_vchr = 0.0
            swap_stable_flow_value = 0.0
            swap_voucher_flow_value = 0.0
            swap_stable_net_flow_value = 0.0
            swap_voucher_net_flow_value = 0.0
            count_usd_to_vchr = 0
            count_vchr_to_usd = 0
            count_vchr_to_vchr = 0
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
                    receipt = (e.meta or {}).get("receipt") or {}
                    asset_in = receipt.get("asset_in")
                    asset_out = receipt.get("asset_out")
                    amount_in = float(receipt.get("amount_in") or 0.0)
                    amount_out = float(receipt.get("amount_out") or 0.0)
                    if amount_in > 0.0 and asset_in and asset_out:
                        pool_id = receipt.get("pool_id") or e.pool_id
                        pool = self.pools.get(pool_id)
                        value_in = pool.values.get_value(asset_in) if pool is not None else 1.0
                        if value_in <= 0.0:
                            value_in = self._default_asset_value(asset_in)
                        value_out = pool.values.get_value(asset_out) if pool is not None else 1.0
                        if value_out <= 0.0:
                            value_out = self._default_asset_value(asset_out)
                        usd = amount_in * value_in
                        out_value = amount_out * value_out
                        if asset_in == cfg.stable_symbol:
                            swap_stable_flow_value += usd
                            swap_stable_net_flow_value += usd
                        elif asset_in.startswith("VCHR:"):
                            swap_voucher_flow_value += usd
                            swap_voucher_net_flow_value += usd
                        if asset_out == cfg.stable_symbol:
                            swap_stable_flow_value += out_value
                            swap_stable_net_flow_value -= out_value
                        elif asset_out.startswith("VCHR:"):
                            swap_voucher_flow_value += out_value
                            swap_voucher_net_flow_value -= out_value
                        if asset_in == cfg.stable_symbol and asset_out.startswith("VCHR:"):
                            vol_usd_to_vchr += usd
                            count_usd_to_vchr += 1
                        elif asset_out == cfg.stable_symbol and asset_in.startswith("VCHR:"):
                            vol_vchr_to_usd += usd
                            count_vchr_to_usd += 1
                        elif asset_in.startswith("VCHR:") and asset_out.startswith("VCHR:"):
                            vol_vchr_to_vchr += usd
                            count_vchr_to_vchr += 1
                elif e.event_type == "ROUTE_REQUESTED":
                    route_requested += 1
                    kind = (e.meta or {}).get("route_attempt_kind", "fixed")
                    if kind == "substitution":
                        route_substitution_requested += 1
                    else:
                        route_fixed_requested += 1
                elif e.event_type == "ROUTE_FOUND":
                    route_found += 1
                    kind = (e.meta or {}).get("route_attempt_kind", "fixed")
                    if kind == "substitution":
                        route_substitution_found += 1
                    else:
                        route_fixed_found += 1
                elif e.event_type == "ROUTE_FAILED":
                    route_failed += 1
                    kind = (e.meta or {}).get("route_attempt_kind", "fixed")
                    if kind == "substitution":
                        route_substitution_failed += 1
                    else:
                        route_fixed_failed += 1

            swap_volume_usd_tick = float(self._swap_volume_usd_tick or 0.0)
            swap_gross_flow_value = swap_stable_flow_value + swap_voucher_flow_value
            swap_stable_flow_share = swap_stable_flow_value / max(1e-9, swap_gross_flow_value)
            utilization_rate = swap_volume_usd_tick / max(1e-9, total_pool_value)
            c_ratio = vol_vchr_to_usd / vol_usd_to_vchr if vol_usd_to_vchr > 1e-9 else 0.0
            beta_ratio = vol_vchr_to_vchr / vol_vchr_to_usd if vol_vchr_to_usd > 1e-9 else 0.0
            def context_count(context: str, *, total: bool = False) -> int:
                store = self._route_context_count_total if total else self._route_context_count_tick
                return int(store.get(context, 0))

            def context_volume(context: str, *, total: bool = False) -> float:
                store = self._route_context_volume_usd_total if total else self._route_context_volume_usd_tick
                return float(store.get(context, 0.0))

            def context_source_count(context: str, source: str, *, total: bool = False) -> int:
                if source == "stable":
                    store = (
                        self._route_context_source_stable_count_total
                        if total
                        else self._route_context_source_stable_count_tick
                    )
                else:
                    store = (
                        self._route_context_source_voucher_count_total
                        if total
                        else self._route_context_source_voucher_count_tick
                    )
                return int(store.get(context, 0))

            def context_source_volume(context: str, source: str, *, total: bool = False) -> float:
                if source == "stable":
                    store = (
                        self._route_context_source_stable_volume_usd_total
                        if total
                        else self._route_context_source_stable_volume_usd_tick
                    )
                else:
                    store = (
                        self._route_context_source_voucher_volume_usd_total
                        if total
                        else self._route_context_source_voucher_volume_usd_tick
                    )
                return float(store.get(context, 0.0))

            def motif_count(motif: str, *, total: bool = False) -> int:
                store = self._route_motif_count_total if total else self._route_motif_count_tick
                return int(store.get(motif, 0))

            def motif_volume(motif: str, *, total: bool = False) -> float:
                store = self._route_motif_volume_usd_total if total else self._route_motif_volume_usd_tick
                return float(store.get(motif, 0.0))

            def ordinary_motif_count(motif: str, *, total: bool = False) -> int:
                store = (
                    self._ordinary_route_motif_count_total
                    if total
                    else self._ordinary_route_motif_count_tick
                )
                return int(store.get(motif, 0))

            def ordinary_motif_volume(motif: str, *, total: bool = False) -> float:
                store = (
                    self._ordinary_route_motif_volume_usd_total
                    if total
                    else self._ordinary_route_motif_volume_usd_tick
                )
                return float(store.get(motif, 0.0))

            def market_motif_count(motif: str, *, total: bool = False) -> int:
                store = (
                    self._market_route_motif_count_total
                    if total
                    else self._market_route_motif_count_tick
                )
                return int(store.get(motif, 0))

            def market_motif_volume(motif: str, *, total: bool = False) -> float:
                store = (
                    self._market_route_motif_volume_usd_total
                    if total
                    else self._market_route_motif_volume_usd_tick
                )
                return float(store.get(motif, 0.0))

            def repayment_motif_count(motif: str, *, total: bool = False) -> int:
                store = (
                    self._repayment_route_motif_count_total
                    if total
                    else self._repayment_route_motif_count_tick
                )
                return int(store.get(motif, 0))

            def repayment_motif_volume(motif: str, *, total: bool = False) -> float:
                store = (
                    self._repayment_route_motif_volume_usd_total
                    if total
                    else self._repayment_route_motif_volume_usd_tick
                )
                return float(store.get(motif, 0.0))

            def loan_motif_count(motif: str, *, total: bool = False) -> int:
                store = (
                    self._loan_route_motif_count_total
                    if total
                    else self._loan_route_motif_count_tick
                )
                return int(store.get(motif, 0))

            def loan_motif_volume(motif: str, *, total: bool = False) -> float:
                store = (
                    self._loan_route_motif_volume_usd_total
                    if total
                    else self._loan_route_motif_volume_usd_tick
                )
                return float(store.get(motif, 0.0))

            def route_motif_metric_bundle(
                prefix: str,
                count_fn,
                volume_fn,
                count_tick: int,
                count_total: int,
            ) -> Dict[str, float | int]:
                return {
                    f"{prefix}_route_motif_count_tick": int(count_tick),
                    f"{prefix}_route_motif_count_total": int(count_total),
                    f"{prefix}_route_motif_voucher_to_voucher_count_tick": count_fn(
                        "voucher_to_voucher"
                    ),
                    f"{prefix}_route_motif_voucher_to_voucher_count_total": count_fn(
                        "voucher_to_voucher", total=True
                    ),
                    f"{prefix}_route_motif_voucher_to_stable_count_tick": count_fn(
                        "voucher_to_stable"
                    ),
                    f"{prefix}_route_motif_voucher_to_stable_count_total": count_fn(
                        "voucher_to_stable", total=True
                    ),
                    f"{prefix}_route_motif_stable_to_voucher_count_tick": count_fn(
                        "stable_to_voucher"
                    ),
                    f"{prefix}_route_motif_stable_to_voucher_count_total": count_fn(
                        "stable_to_voucher", total=True
                    ),
                    f"{prefix}_route_motif_voucher_to_voucher_share_total": (
                        count_fn("voucher_to_voucher", total=True) / max(1, count_total)
                    ),
                    f"{prefix}_route_motif_voucher_to_stable_share_total": (
                        count_fn("voucher_to_stable", total=True) / max(1, count_total)
                    ),
                    f"{prefix}_route_motif_stable_to_voucher_share_total": (
                        count_fn("stable_to_voucher", total=True) / max(1, count_total)
                    ),
                    f"{prefix}_route_motif_stable_involved_share_total": (
                        (
                            count_fn("voucher_to_stable", total=True)
                            + count_fn("stable_to_voucher", total=True)
                        )
                        / max(1, count_total)
                    ),
                    f"{prefix}_route_motif_voucher_to_voucher_volume_usd_tick": volume_fn(
                        "voucher_to_voucher"
                    ),
                    f"{prefix}_route_motif_voucher_to_voucher_volume_usd_total": volume_fn(
                        "voucher_to_voucher", total=True
                    ),
                    f"{prefix}_route_motif_voucher_to_stable_volume_usd_tick": volume_fn(
                        "voucher_to_stable"
                    ),
                    f"{prefix}_route_motif_voucher_to_stable_volume_usd_total": volume_fn(
                        "voucher_to_stable", total=True
                    ),
                    f"{prefix}_route_motif_stable_to_voucher_volume_usd_tick": volume_fn(
                        "stable_to_voucher"
                    ),
                    f"{prefix}_route_motif_stable_to_voucher_volume_usd_total": volume_fn(
                        "stable_to_voucher", total=True
                    ),
                }

            route_motif_count_tick = sum(int(v) for v in self._route_motif_count_tick.values())
            route_motif_count_total = sum(int(v) for v in self._route_motif_count_total.values())
            ordinary_route_motif_count_tick = sum(
                int(v) for v in self._ordinary_route_motif_count_tick.values()
            )
            ordinary_route_motif_count_total = sum(
                int(v) for v in self._ordinary_route_motif_count_total.values()
            )
            market_route_motif_count_tick = sum(
                int(v) for v in self._market_route_motif_count_tick.values()
            )
            market_route_motif_count_total = sum(
                int(v) for v in self._market_route_motif_count_total.values()
            )
            repayment_route_motif_count_tick = sum(
                int(v) for v in self._repayment_route_motif_count_tick.values()
            )
            repayment_route_motif_count_total = sum(
                int(v) for v in self._repayment_route_motif_count_total.values()
            )
            loan_route_motif_count_tick = sum(
                int(v) for v in self._loan_route_motif_count_tick.values()
            )
            loan_route_motif_count_total = sum(
                int(v) for v in self._loan_route_motif_count_total.values()
            )

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
            lp_sclc_total = 0.0
            if cfg.sclc_symbol:
                lp_sclc_total = sum(
                    p.vault.get(cfg.sclc_symbol)
                    for p in self.pools.values()
                    if not p.policy.system_pool and p.policy.role == "liquidity_provider"
                )

            fee_pool_cumulative = self._fee_pool_cumulative_usd
            producer_debt_active_count = sum(
                1 for obligation in self._producer_debt_obligations
                if obligation.remaining_voucher_units > 1e-9
                or (
                    self._producer_debt_contract_repayment_enabled()
                    and obligation.cash_service_remaining_usd > 1e-9
                )
            )
            producer_debt_active_usd = self._producer_debt_active_usd()
            producer_voucher_overlap = self._producer_voucher_overlap_diagnostics()
            lender_stable_total_usd = 0.0
            lender_stable_reserve_usd = 0.0
            lender_stable_available_above_reserve_usd = 0.0
            for pool in self.pools.values():
                if pool.policy.system_pool or pool.policy.role != "lender":
                    continue
                stable_units = pool.vault.get(cfg.stable_symbol)
                reserve_units = max(0.0, pool.policy.min_stable_reserve)
                lender_stable_total_usd += stable_units
                lender_stable_reserve_usd += reserve_units
                lender_stable_available_above_reserve_usd += max(0.0, stable_units - reserve_units)
            self.metrics.add_network({
                "tick": self.tick,
                "num_pools": num_active,
                "num_system_pools": num_system,
                "num_assets": len(self.factory.asset_universe),
                "swap_receipts_total": swap_receipts,
                "pool_total_value_usd": total_pool_value,
                "stable_total_in_pools": stable_total,
                "voucher_total_in_pools": voucher_total,
                "stable_value_total_in_active_pools": active_stable_value,
                "voucher_value_total_in_active_pools": active_voucher_value,
                "stable_value_share_in_active_pools": active_stable_share,
                "voucher_value_share_in_active_pools": active_voucher_share,
                "stable_to_voucher_value_ratio_in_active_pools": active_stable_to_voucher_ratio,
                "pools_under_stable_reserve": pools_under_reserve,
                "producer_pools_total": int(role_stress_counts["producer"]),
                "producer_pools_under_stable_reserve": int(role_under_reserve["producer"]),
                "producer_stable_reserve_stress_ratio": (
                    role_under_reserve["producer"] / max(1, role_stress_counts["producer"])
                ),
                "producer_stable_reserve_deficit_usd": float(role_reserve_deficit["producer"]),
                "consumer_pools_total": int(role_stress_counts["consumer"]),
                "consumer_pools_under_stable_reserve": int(role_under_reserve["consumer"]),
                "consumer_stable_reserve_stress_ratio": (
                    role_under_reserve["consumer"] / max(1, role_stress_counts["consumer"])
                ),
                "consumer_stable_reserve_deficit_usd": float(role_reserve_deficit["consumer"]),
                "lender_pools_total": int(role_stress_counts["lender"]),
                "lender_pools_under_stable_reserve": int(role_under_reserve["lender"]),
                "lender_stable_reserve_stress_ratio": (
                    role_under_reserve["lender"] / max(1, role_stress_counts["lender"])
                ),
                "lender_stable_reserve_deficit_usd": float(role_reserve_deficit["lender"]),
                "community_pools_total": int(community_pools_total),
                "community_pools_under_stable_reserve": int(community_pools_under_reserve),
                "community_stable_reserve_stress_ratio": (
                    community_pools_under_reserve / max(1, community_pools_total)
                ),
                "community_stable_reserve_deficit_usd": float(community_reserve_deficit),
                "debt_outstanding_units": debt_outstanding_units,
                "debt_outstanding_usd": debt_outstanding_usd,
                "redeemed_total": redeemed_total,
                "outstanding_voucher_supply_total": outstanding_total,
                "outstanding_voucher_value_usd": outstanding_value_usd,
                "issued_voucher_supply_total": issued_total,
                "issuer_returned_voucher_supply_total": issuer_returned_total,
                "net_circulating_voucher_supply_total": outstanding_total,
                "producer_deposit_stable_usd_tick": float(self._producer_deposit_stable_usd_tick),
                "producer_deposit_voucher_usd_tick": float(self._producer_deposit_voucher_usd_tick),
                "producer_deposit_stable_usd_total": float(self._producer_deposit_stable_usd_total),
                "producer_deposit_voucher_usd_total": float(self._producer_deposit_voucher_usd_total),
                "producer_deposit_credit_capacity_usd": float(self._producer_deposit_credit_capacity_usd()),
                "productive_credit_inflow_usd_tick": float(self._productive_credit_inflow_usd_tick),
                "productive_credit_inflow_usd_total": float(self._productive_credit_inflow_usd_total),
                "productive_credit_stable_retained_usd_tick": float(
                    self._productive_credit_stable_retained_usd_tick
                ),
                "productive_credit_stable_retained_usd_total": float(
                    self._productive_credit_stable_retained_usd_total
                ),
                "productive_credit_voucher_deposit_usd_tick": float(
                    self._productive_credit_voucher_deposit_usd_tick
                ),
                "productive_credit_voucher_deposit_usd_total": float(
                    self._productive_credit_voucher_deposit_usd_total
                ),
                "productive_credit_voucher_deposit_cap_clipped_usd_tick": float(
                    self._productive_credit_voucher_deposit_cap_clipped_usd_tick
                ),
                "productive_credit_voucher_deposit_cap_clipped_usd_total": float(
                    self._productive_credit_voucher_deposit_cap_clipped_usd_total
                ),
                "producer_debt_active_obligations": int(producer_debt_active_count),
                "producer_debt_active_usd": float(producer_debt_active_usd),
                "producer_debt_originated_usd_tick": float(self._producer_debt_originated_usd_tick),
                "producer_debt_originated_usd_total": float(self._producer_debt_originated_usd_total),
                "producer_debt_cash_service_due_usd_tick": float(
                    self._producer_debt_cash_service_due_usd_tick
                ),
                "producer_debt_cash_service_due_usd_total": float(
                    self._producer_debt_cash_service_due_usd_total
                ),
                "producer_debt_cash_service_paid_usd_tick": float(
                    self._producer_debt_cash_service_paid_usd_tick
                ),
                "producer_debt_cash_service_paid_usd_total": float(
                    self._producer_debt_cash_service_paid_usd_total
                ),
                "producer_debt_service_capacity_balance_usd": float(
                    self._producer_debt_service_capacity_balance_usd()
                ),
                "producer_debt_service_capacity_credited_usd_tick": float(
                    self._producer_debt_service_capacity_credited_usd_tick
                ),
                "producer_debt_service_capacity_credited_usd_total": float(
                    self._producer_debt_service_capacity_credited_usd_total
                ),
                "producer_debt_service_capacity_onramp_usd_tick": float(
                    self._producer_debt_service_capacity_onramp_usd_tick
                ),
                "producer_debt_service_capacity_onramp_usd_total": float(
                    self._producer_debt_service_capacity_onramp_usd_total
                ),
                "producer_self_repayment_swap_volume_usd_tick": float(
                    self._producer_self_repayment_swap_volume_usd_tick
                ),
                "producer_self_repayment_swap_volume_usd_total": float(
                    self._producer_self_repayment_swap_volume_usd_total
                ),
                "producer_self_repayment_voucher_removed_usd_tick": float(
                    self._producer_self_repayment_voucher_removed_usd_tick
                ),
                "producer_self_repayment_voucher_removed_usd_total": float(
                    self._producer_self_repayment_voucher_removed_usd_total
                ),
                "producer_debt_pressure_prepayment_usd_tick": float(
                    self._producer_debt_pressure_prepayment_usd_tick
                ),
                "producer_debt_pressure_prepayment_usd_total": float(
                    self._producer_debt_pressure_prepayment_usd_total
                ),
                "producer_debt_pressure_deferred_usd_tick": float(
                    self._producer_debt_pressure_deferred_usd_tick
                ),
                "producer_debt_pressure_deferred_usd_total": float(
                    self._producer_debt_pressure_deferred_usd_total
                ),
                "producer_debt_pressure_deferred_balance_usd": float(
                    self._producer_debt_pressure_deferred_balance_usd()
                ),
                "producer_debt_pressure_batched_swap_count_tick": int(
                    self._producer_debt_pressure_batched_swap_count_tick
                ),
                "producer_debt_pressure_batched_swap_count_total": int(
                    self._producer_debt_pressure_batched_swap_count_total
                ),
                "producer_debt_pressure_batched_swap_volume_usd_tick": float(
                    self._producer_debt_pressure_batched_swap_volume_usd_tick
                ),
                "producer_debt_pressure_batched_swap_volume_usd_total": float(
                    self._producer_debt_pressure_batched_swap_volume_usd_total
                ),
                "producer_debt_pressure_min_swap_usd": float(
                    self._producer_debt_pressure_min_swap_usd()
                ),
                "producer_debt_attention_pressure_usd_tick": float(
                    self._producer_debt_attention_pressure_usd_tick
                ),
                "producer_debt_attention_pressure_usd_total": float(
                    self._producer_debt_attention_pressure_usd_total
                ),
                "producer_debt_attention_suppressed_attempts_tick": int(
                    self._producer_debt_attention_suppressed_attempts_tick
                ),
                "producer_debt_attention_suppressed_attempts_total": int(
                    self._producer_debt_attention_suppressed_attempts_total
                ),
                "producer_debt_attention_suppressed_v2v_attempts_tick": int(
                    self._producer_debt_attention_suppressed_v2v_attempts_tick
                ),
                "producer_debt_attention_suppressed_v2v_attempts_total": int(
                    self._producer_debt_attention_suppressed_v2v_attempts_total
                ),
                "producer_debt_attention_share_avg_tick": (
                    self._producer_debt_attention_share_sum_tick
                    / max(1, self._producer_debt_attention_share_count_tick)
                ),
                "producer_debt_attention_share_max_tick": float(
                    self._producer_debt_attention_share_max_tick
                ),
                "producer_debt_attention_reference_usd": (
                    self._producer_debt_attention_reference_usd_sum_tick
                    / max(1, self._producer_debt_attention_reference_count_tick)
                ),
                "producer_bond_assessment_pressure_usd_tick": float(
                    self._producer_bond_assessment_pressure_usd_tick
                ),
                "producer_bond_assessment_pressure_usd_total": float(
                    self._producer_bond_assessment_pressure_usd_total
                ),
                "producer_bond_assessment_sustain_offset_attempts_tick": float(
                    self._producer_bond_assessment_sustain_offset_attempts_tick
                ),
                "producer_bond_assessment_sustain_offset_attempts_total": float(
                    self._producer_bond_assessment_sustain_offset_attempts_total
                ),
                "producer_bond_assessment_sustain_offset_v2v_attempts_tick": float(
                    self._producer_bond_assessment_sustain_offset_v2v_attempts_tick
                ),
                "producer_bond_assessment_sustain_offset_v2v_attempts_total": float(
                    self._producer_bond_assessment_sustain_offset_v2v_attempts_total
                ),
                "producer_bond_assessment_sustain_target_reduction_tick": int(
                    self._producer_bond_assessment_sustain_target_reduction_tick
                ),
                "producer_bond_assessment_sustain_target_reduction_total": int(
                    self._producer_bond_assessment_sustain_target_reduction_total
                ),
                "producer_activity_composition_pressure_usd_tick": float(
                    self._producer_activity_composition_pressure_usd_tick
                ),
                "producer_activity_composition_pressure_usd_total": float(
                    self._producer_activity_composition_pressure_usd_total
                ),
                "producer_activity_composition_shift_share_avg_tick": (
                    self._producer_activity_composition_shift_share_sum_tick
                    / max(1, self._producer_activity_composition_shift_share_count_tick)
                ),
                "producer_activity_composition_shift_share_max_tick": float(
                    self._producer_activity_composition_shift_share_max_tick
                ),
                "producer_activity_composition_reference_usd": (
                    self._producer_activity_composition_reference_usd_sum_tick
                    / max(1, self._producer_activity_composition_reference_count_tick)
                ),
                "producer_activity_composition_v2v_weight_removed_tick": float(
                    self._producer_activity_composition_v2v_weight_removed_tick
                ),
                "producer_activity_composition_v2v_weight_removed_total": float(
                    self._producer_activity_composition_v2v_weight_removed_total
                ),
                "producer_activity_composition_v2s_weight_added_tick": float(
                    self._producer_activity_composition_v2s_weight_added_tick
                ),
                "producer_activity_composition_v2s_weight_added_total": float(
                    self._producer_activity_composition_v2s_weight_added_total
                ),
                "producer_activity_composition_shifted_route_attempts_tick": int(
                    self._producer_activity_composition_shifted_route_attempts_tick
                ),
                "producer_activity_composition_shifted_route_attempts_total": int(
                    self._producer_activity_composition_shifted_route_attempts_total
                ),
                "producer_activity_composition_shifted_v2s_attempts_tick": int(
                    self._producer_activity_composition_shifted_v2s_attempts_tick
                ),
                "producer_activity_composition_shifted_v2s_attempts_total": int(
                    self._producer_activity_composition_shifted_v2s_attempts_total
                ),
                "producer_activity_composition_effective_own_voucher_stable_probability_avg_tick": (
                    self._producer_activity_composition_own_voucher_stable_probability_sum_tick
                    / max(
                        1,
                        self._producer_activity_composition_own_voucher_stable_probability_count_tick,
                    )
                ),
                "producer_activity_composition_effective_own_voucher_stable_probability_max_tick": float(
                    self._producer_activity_composition_own_voucher_stable_probability_max_tick
                ),
                "ordinary_own_voucher_stable_borrowing_enabled": int(
                    bool(getattr(self.cfg, "ordinary_own_voucher_stable_borrowing_enabled", False))
                ),
                "producer_debt_penalty_accrued_usd_tick": float(
                    self._producer_debt_penalty_accrued_usd_tick
                ),
                "producer_debt_penalty_accrued_usd_total": float(
                    self._producer_debt_penalty_accrued_usd_total
                ),
                "producer_debt_penalty_paid_usd_tick": float(
                    self._producer_debt_penalty_paid_usd_tick
                ),
                "producer_debt_penalty_paid_usd_total": float(
                    self._producer_debt_penalty_paid_usd_total
                ),
                "producer_debt_arrears_usd": float(self._producer_debt_arrears_usd()),
                "lender_stable_total_usd": float(lender_stable_total_usd),
                "lender_stable_reserve_usd": float(lender_stable_reserve_usd),
                "lender_stable_available_above_reserve_usd": float(
                    lender_stable_available_above_reserve_usd
                ),
                "lender_held_producer_voucher_inventory_usd": float(
                    self._lender_held_producer_voucher_inventory_usd()
                ),
                "active_routable_producer_voucher_float_usd": float(
                    self._active_routable_producer_voucher_float_usd()
                ),
                **producer_voucher_overlap,
                "consumer_stable_available_above_reserve_usd": float(
                    self._consumer_stable_available_above_reserve_usd()
                ),
                "lender_recovered_stable_pending_usd_total": float(
                    self._lender_recovered_stable_pending_usd_total()
                ),
                "lender_recovered_stable_sweepable_pending_usd_total": float(
                    self._lender_recovered_stable_sweepable_pending_usd_total()
                ),
                "lender_recovered_stable_usd_tick": float(self._lender_recovered_stable_usd_tick),
                "lender_recovered_stable_usd_total": float(self._lender_recovered_stable_usd_total),
                "bond_eligible_pool_exposure_recovered_stable_usd_tick": float(
                    self._bond_eligible_pool_exposure_recovered_stable_usd_tick
                ),
                "bond_eligible_pool_exposure_recovered_stable_usd_total": float(
                    self._bond_eligible_pool_exposure_recovered_stable_usd_total
                ),
                "lender_inventory_turnover_stable_usd_tick": float(
                    self._lender_inventory_turnover_stable_usd_tick
                ),
                "lender_inventory_turnover_stable_usd_total": float(
                    self._lender_inventory_turnover_stable_usd_total
                ),
                "lender_recovered_stable_borrower_self_usd_tick": float(
                    self._lender_recovered_stable_borrower_self_usd_tick
                ),
                "lender_recovered_stable_borrower_self_usd_total": float(
                    self._lender_recovered_stable_borrower_self_usd_total
                ),
                "lender_recovered_stable_borrower_regular_usd_tick": float(
                    self._lender_recovered_stable_borrower_regular_usd_tick
                ),
                "lender_recovered_stable_borrower_regular_usd_total": float(
                    self._lender_recovered_stable_borrower_regular_usd_total
                ),
                "lender_recovered_stable_borrower_maturity_usd_tick": float(
                    self._lender_recovered_stable_borrower_maturity_usd_tick
                ),
                "lender_recovered_stable_borrower_maturity_usd_total": float(
                    self._lender_recovered_stable_borrower_maturity_usd_total
                ),
                "lender_recovered_stable_consumer_purchase_usd_tick": float(
                    self._lender_recovered_stable_consumer_purchase_usd_tick
                ),
                "lender_recovered_stable_consumer_purchase_usd_total": float(
                    self._lender_recovered_stable_consumer_purchase_usd_total
                ),
                "lender_recovered_stable_external_nonproducer_purchase_usd_tick": float(
                    self._lender_recovered_stable_external_nonproducer_purchase_usd_tick
                ),
                "lender_recovered_stable_external_nonproducer_purchase_usd_total": float(
                    self._lender_recovered_stable_external_nonproducer_purchase_usd_total
                ),
                "lender_recovered_stable_other_producer_purchase_usd_tick": float(
                    self._lender_recovered_stable_other_producer_purchase_usd_tick
                ),
                "lender_recovered_stable_other_producer_purchase_usd_total": float(
                    self._lender_recovered_stable_other_producer_purchase_usd_total
                ),
                "lender_recovered_stable_third_party_purchase_usd_tick": float(
                    self._lender_recovered_stable_third_party_purchase_usd_tick
                ),
                "lender_recovered_stable_third_party_purchase_usd_total": float(
                    self._lender_recovered_stable_third_party_purchase_usd_total
                ),
                "lender_recovered_stable_other_usd_tick": float(
                    self._lender_recovered_stable_other_usd_tick
                ),
                "lender_recovered_stable_other_usd_total": float(
                    self._lender_recovered_stable_other_usd_total
                ),
                "bond_service_reserve_balance_usd": float(self._bond_service_reserve_usd_balance),
                "bond_service_lockbox_target_usd": float(self._bond_service_lockbox_target_usd()),
                "bond_service_lockbox_coverage_ratio": float(
                    self._bond_service_lockbox_coverage_ratio()
                ),
                "bond_service_lockbox_mode": self._bond_service_lockbox_mode(),
                "bond_service_reserved_usd_tick": float(self._bond_service_reserved_usd_tick),
                "bond_service_reserved_usd_total": float(self._bond_service_reserved_usd_total),
                "bond_service_paid_from_reserve_usd_tick": float(
                    self._bond_service_paid_from_reserve_usd_tick
                ),
                "bond_service_paid_from_reserve_usd_total": float(
                    self._bond_service_paid_from_reserve_usd_total
                ),
                "producer_loan_attempts_tick": int(self._producer_loan_attempts_tick),
                "producer_loan_no_lender_tick": int(self._producer_loan_no_lender_tick),
                "producer_loan_no_inventory_tick": int(self._producer_loan_no_inventory_tick),
                "producer_loan_zero_amount_tick": int(self._producer_loan_zero_amount_tick),
                "producer_loan_route_found_tick": int(self._producer_loan_route_found_tick),
                "producer_loan_route_failed_tick": int(self._producer_loan_route_failed_tick),
                "producer_loan_backfill_attempts_tick": int(self._producer_loan_backfill_attempts_tick),
                "producer_loan_backfill_executed_tick": int(self._producer_loan_backfill_executed_tick),
                "producer_loan_executed_tick": int(self._producer_loan_executed_tick),
                "producer_loan_execution_failed_tick": int(self._producer_loan_execution_failed_tick),
                "producer_loan_sampled_usd_tick": float(self._producer_loan_sampled_usd_tick),
                "producer_loan_attempted_usd_tick": float(self._producer_loan_attempted_usd_tick),
                "producer_loan_executed_usd_tick": float(self._producer_loan_executed_usd_tick),
                "producer_loan_clipped_inventory_usd_tick": float(
                    self._producer_loan_clipped_inventory_usd_tick
                ),
                "producer_loan_clipped_lender_cap_usd_tick": float(
                    self._producer_loan_clipped_lender_cap_usd_tick
                ),
                "producer_loan_clipped_lender_remaining_usd_tick": float(
                    self._producer_loan_clipped_lender_remaining_usd_tick
                ),
                "producer_loan_clipped_lender_stable_usd_tick": float(
                    self._producer_loan_clipped_lender_stable_usd_tick
                ),
                "producer_loan_clipped_combined_lender_usd_tick": float(
                    self._producer_loan_clipped_combined_lender_usd_tick
                ),
                "producer_loan_lender_collateral_cap_usd_tick": float(
                    self._producer_loan_lender_collateral_cap_usd_tick
                ),
                "producer_loan_lender_remaining_cap_usd_tick": float(
                    self._producer_loan_lender_remaining_cap_usd_tick
                ),
                "producer_loan_lender_stable_available_usd_tick": float(
                    self._producer_loan_lender_stable_available_usd_tick
                ),
                "producer_voucher_loan_attempts_tick": int(self._producer_voucher_loan_attempts_tick),
                "producer_voucher_loan_no_target_tick": int(self._producer_voucher_loan_no_target_tick),
                "producer_voucher_loan_no_inventory_tick": int(
                    self._producer_voucher_loan_no_inventory_tick
                ),
                "producer_voucher_loan_zero_amount_tick": int(
                    self._producer_voucher_loan_zero_amount_tick
                ),
                "producer_voucher_loan_route_found_tick": int(
                    self._producer_voucher_loan_route_found_tick
                ),
                "producer_voucher_loan_route_failed_tick": int(
                    self._producer_voucher_loan_route_failed_tick
                ),
                "producer_voucher_loan_executed_tick": int(
                    self._producer_voucher_loan_executed_tick
                ),
                "producer_voucher_loan_execution_failed_tick": int(
                    self._producer_voucher_loan_execution_failed_tick
                ),
                "producer_voucher_loan_attempted_usd_tick": float(
                    self._producer_voucher_loan_attempted_usd_tick
                ),
                "producer_voucher_loan_executed_usd_tick": float(
                    self._producer_voucher_loan_executed_usd_tick
                ),
                "producer_voucher_loan_clipped_lender_cap_usd_tick": float(
                    self._producer_voucher_loan_clipped_lender_cap_usd_tick
                ),
                "producer_voucher_loan_clipped_lender_remaining_usd_tick": float(
                    self._producer_voucher_loan_clipped_lender_remaining_usd_tick
                ),
                "producer_primary_voucher_loan_attempts_tick": int(
                    self._producer_primary_voucher_loan_attempts_tick
                ),
                "producer_primary_voucher_loan_attempts_total": int(
                    self._producer_primary_voucher_loan_attempts_total
                ),
                "producer_primary_voucher_loan_executed_tick": int(
                    self._producer_primary_voucher_loan_executed_tick
                ),
                "producer_primary_voucher_loan_executed_total": int(
                    self._producer_primary_voucher_loan_executed_total
                ),
                "voucher_purchase_attempts_tick": int(self._voucher_purchase_attempts_tick),
                "voucher_purchase_attempts_total": int(self._voucher_purchase_attempts_total),
                "consumer_voucher_purchase_attempts_tick": int(
                    self._consumer_voucher_purchase_attempts_tick
                ),
                "consumer_voucher_purchase_attempts_total": int(
                    self._consumer_voucher_purchase_attempts_total
                ),
                "consumer_voucher_purchase_success_tick": int(
                    self._consumer_voucher_purchase_success_tick
                ),
                "consumer_voucher_purchase_success_total": int(
                    self._consumer_voucher_purchase_success_total
                ),
                "consumer_voucher_purchase_no_stable_tick": int(
                    self._consumer_voucher_purchase_no_stable_tick
                ),
                "consumer_voucher_purchase_no_stable_total": int(
                    self._consumer_voucher_purchase_no_stable_total
                ),
                "consumer_voucher_purchase_reserve_protected_tick": int(
                    self._consumer_voucher_purchase_reserve_protected_tick
                ),
                "consumer_voucher_purchase_reserve_protected_total": int(
                    self._consumer_voucher_purchase_reserve_protected_total
                ),
                "consumer_voucher_purchase_no_route_tick": int(
                    self._consumer_voucher_purchase_no_route_tick
                ),
                "consumer_voucher_purchase_no_route_total": int(
                    self._consumer_voucher_purchase_no_route_total
                ),
                "consumer_voucher_purchase_no_target_tick": int(
                    self._consumer_voucher_purchase_no_target_tick
                ),
                "consumer_voucher_purchase_no_target_total": int(
                    self._consumer_voucher_purchase_no_target_total
                ),
                "consumer_voucher_purchase_stable_spent_usd_tick": float(
                    self._consumer_voucher_purchase_stable_spent_usd_tick
                ),
                "consumer_voucher_purchase_stable_spent_usd_total": float(
                    self._consumer_voucher_purchase_stable_spent_usd_total
                ),
                "consumer_voucher_purchase_voucher_value_acquired_usd_tick": float(
                    self._consumer_voucher_purchase_voucher_value_acquired_usd_tick
                ),
                "consumer_voucher_purchase_voucher_value_acquired_usd_total": float(
                    self._consumer_voucher_purchase_voucher_value_acquired_usd_total
                ),
                "third_party_voucher_purchase_attempts_tick": int(
                    self._third_party_voucher_purchase_attempts_tick
                ),
                "third_party_voucher_purchase_attempts_total": int(
                    self._third_party_voucher_purchase_attempts_total
                ),
                "third_party_voucher_purchase_success_tick": int(
                    self._third_party_voucher_purchase_success_tick
                ),
                "third_party_voucher_purchase_success_total": int(
                    self._third_party_voucher_purchase_success_total
                ),
                "third_party_voucher_purchase_no_stable_tick": int(
                    self._third_party_voucher_purchase_no_stable_tick
                ),
                "third_party_voucher_purchase_no_stable_total": int(
                    self._third_party_voucher_purchase_no_stable_total
                ),
                "third_party_voucher_purchase_reserve_protected_tick": int(
                    self._third_party_voucher_purchase_reserve_protected_tick
                ),
                "third_party_voucher_purchase_reserve_protected_total": int(
                    self._third_party_voucher_purchase_reserve_protected_total
                ),
                "third_party_voucher_purchase_no_route_tick": int(
                    self._third_party_voucher_purchase_no_route_tick
                ),
                "third_party_voucher_purchase_no_route_total": int(
                    self._third_party_voucher_purchase_no_route_total
                ),
                "third_party_voucher_purchase_no_target_tick": int(
                    self._third_party_voucher_purchase_no_target_tick
                ),
                "third_party_voucher_purchase_no_target_total": int(
                    self._third_party_voucher_purchase_no_target_total
                ),
                "third_party_voucher_purchase_stable_spent_usd_tick": float(
                    self._third_party_voucher_purchase_stable_spent_usd_tick
                ),
                "third_party_voucher_purchase_stable_spent_usd_total": float(
                    self._third_party_voucher_purchase_stable_spent_usd_total
                ),
                "third_party_voucher_purchase_voucher_value_acquired_usd_tick": float(
                    self._third_party_voucher_purchase_voucher_value_acquired_usd_tick
                ),
                "third_party_voucher_purchase_voucher_value_acquired_usd_total": float(
                    self._third_party_voucher_purchase_voucher_value_acquired_usd_total
                ),
                "lender_voucher_purchase_stable_budget_remaining_usd_tick": float(
                    self._lender_voucher_purchase_stable_budget_remaining_usd_tick
                ),
                "lender_voucher_purchase_stable_budget_onramp_usd_tick": float(
                    self._lender_voucher_purchase_stable_budget_onramp_usd_tick
                ),
                "lender_voucher_purchase_stable_budget_onramp_usd_total": float(
                    self._lender_voucher_purchase_stable_budget_onramp_usd_total
                ),
                "consumer_voucher_purchase_stable_budget_onramp_usd_tick": float(
                    self._consumer_voucher_purchase_stable_budget_onramp_usd_tick
                ),
                "consumer_voucher_purchase_stable_budget_onramp_usd_total": float(
                    self._consumer_voucher_purchase_stable_budget_onramp_usd_total
                ),
                "third_party_voucher_purchase_stable_budget_onramp_usd_tick": float(
                    self._third_party_voucher_purchase_stable_budget_onramp_usd_tick
                ),
                "third_party_voucher_purchase_stable_budget_onramp_usd_total": float(
                    self._third_party_voucher_purchase_stable_budget_onramp_usd_total
                ),
                "producer_stable_exited_usd_tick": float(self._producer_stable_exited_usd_tick),
                "producer_stable_exited_usd_total": float(self._producer_stable_exited_usd_total),
                "producer_stable_reuse_budget_usd_tick": float(
                    self._producer_stable_reuse_budget_usd_tick
                ),
                "producer_stable_reuse_budget_usd_total": float(
                    self._producer_stable_reuse_budget_usd_total
                ),
                "net_redeemed_voucher_usd_tick": float(self._net_redeemed_voucher_usd_tick),
                "net_redeemed_voucher_usd_total": float(self._net_redeemed_voucher_usd_total),
                "voucher_redeemed_to_issuer_usd_tick": float(
                    self._voucher_redeemed_to_issuer_usd_tick
                ),
                "voucher_redeemed_to_issuer_usd_total": float(
                    self._voucher_redeemed_to_issuer_usd_total
                ),
                "voucher_fee_retained_for_service_usd_tick": float(
                    self._voucher_fee_retained_for_service_usd_tick
                ),
                "voucher_fee_retained_for_service_usd_total": float(
                    self._voucher_fee_retained_for_service_usd_total
                ),
                "voucher_reintroduced_by_deposit_usd_tick": float(
                    self._voucher_reintroduced_by_deposit_usd_tick
                ),
                "voucher_reintroduced_by_deposit_usd_total": float(
                    self._voucher_reintroduced_by_deposit_usd_total
                ),
                "voucher_new_issuance_deposit_usd_tick": float(
                    self._voucher_new_issuance_deposit_usd_tick
                ),
                "voucher_new_issuance_deposit_usd_total": float(
                    self._voucher_new_issuance_deposit_usd_total
                ),
                "debt_removal_voucher_redeemed_usd_tick": float(
                    self._debt_removal_voucher_redeemed_usd_tick
                ),
                "debt_removal_voucher_redeemed_usd_total": float(
                    self._debt_removal_voucher_redeemed_usd_total
                ),
                "producer_debt_matured_usd_tick": float(self._producer_debt_matured_usd_tick),
                "producer_debt_matured_usd_total": float(self._producer_debt_matured_usd_total),
                "producer_debt_repaid_usd_tick": float(self._producer_debt_repaid_usd_tick),
                "producer_debt_repaid_usd_total": float(self._producer_debt_repaid_usd_total),
                "producer_debt_repaid_regular_usd_tick": float(
                    self._producer_debt_repaid_regular_usd_tick
                ),
                "producer_debt_repaid_regular_usd_total": float(
                    self._producer_debt_repaid_regular_usd_total
                ),
                "producer_debt_repaid_maturity_usd_tick": float(
                    self._producer_debt_repaid_maturity_usd_tick
                ),
                "producer_debt_repaid_maturity_usd_total": float(
                    self._producer_debt_repaid_maturity_usd_total
                ),
                "producer_debt_stable_recovered_usd_tick": float(
                    self._producer_debt_stable_recovered_usd_tick
                ),
                "producer_debt_stable_recovered_usd_total": float(
                    self._producer_debt_stable_recovered_usd_total
                ),
                "producer_debt_consumer_stable_purchase_usd_tick": float(
                    self._producer_debt_consumer_stable_purchase_usd_tick
                ),
                "producer_debt_consumer_stable_purchase_usd_total": float(
                    self._producer_debt_consumer_stable_purchase_usd_total
                ),
                "producer_debt_third_party_stable_purchase_usd_tick": float(
                    self._producer_debt_third_party_stable_purchase_usd_tick
                ),
                "producer_debt_third_party_stable_purchase_usd_total": float(
                    self._producer_debt_third_party_stable_purchase_usd_total
                ),
                "producer_debt_defaulted_usd_tick": float(self._producer_debt_defaulted_usd_tick),
                "producer_debt_defaulted_usd_total": float(self._producer_debt_defaulted_usd_total),
                "producer_debt_closed_by_circulation_usd_tick": float(
                    self._producer_debt_closed_by_circulation_usd_tick
                ),
                "producer_debt_closed_by_circulation_usd_total": float(
                    self._producer_debt_closed_by_circulation_usd_total
                ),
                "producer_debt_closed_by_voucher_swap_usd_tick": float(
                    self._producer_debt_closed_by_voucher_swap_usd_tick
                ),
                "producer_debt_closed_by_voucher_swap_usd_total": float(
                    self._producer_debt_closed_by_voucher_swap_usd_total
                ),
                "producer_debt_closed_not_held_at_maturity_usd_tick": float(
                    self._producer_debt_closed_not_held_at_maturity_usd_tick
                ),
                "producer_debt_closed_not_held_at_maturity_usd_total": float(
                    self._producer_debt_closed_not_held_at_maturity_usd_total
                ),
                "producer_debt_maturity_recovery_rate": float(
                    self.cfg.producer_debt_maturity_recovery_rate
                ),
                "repayment_volume_usd": repayment_volume_usd,
                "loan_issuance_volume_usd": loan_issuance_usd,
                "swap_volume_usd_tick": swap_volume_usd_tick,
                "swap_volume_usd_to_vchr_tick": vol_usd_to_vchr,
                "swap_volume_vchr_to_usd_tick": vol_vchr_to_usd,
                "swap_volume_vchr_to_vchr_tick": vol_vchr_to_vchr,
                "swap_stable_flow_value_tick": swap_stable_flow_value,
                "swap_voucher_flow_value_tick": swap_voucher_flow_value,
                "swap_stable_flow_share_tick": swap_stable_flow_share,
                "swap_stable_net_flow_value_tick": swap_stable_net_flow_value,
                "swap_voucher_net_flow_value_tick": swap_voucher_net_flow_value,
                "route_source_stable_net_flow_value_tick": float(
                    self._route_source_stable_net_flow_value_tick
                ),
                "route_source_voucher_net_flow_value_tick": float(
                    self._route_source_voucher_net_flow_value_tick
                ),
                "swap_count_usd_to_vchr_tick": count_usd_to_vchr,
                "swap_count_vchr_to_usd_tick": count_vchr_to_usd,
                "swap_count_vchr_to_vchr_tick": count_vchr_to_vchr,
                "swap_c_ratio": c_ratio,
                "swap_beta_ratio": beta_ratio,
                "utilization_rate": utilization_rate,
                "transactions_per_tick": transactions_per_tick,
                "route_requested_tick": int(route_requested),
                "route_found_tick": int(route_found),
                "route_failed_tick": int(route_failed),
                "route_fixed_requested_tick": int(route_fixed_requested),
                "route_fixed_found_tick": int(route_fixed_found),
                "route_fixed_failed_tick": int(route_fixed_failed),
                "route_substitution_requested_tick": int(route_substitution_requested),
                "route_substitution_found_tick": int(route_substitution_found),
                "route_substitution_failed_tick": int(route_substitution_failed),
                "route_repeat_partner_requested_tick": int(
                    self._route_repeat_partner_requested_tick
                ),
                "route_exploration_requested_tick": int(
                    self._route_exploration_requested_tick
                ),
                "route_sticky_used_tick": int(self._route_sticky_used_tick),
                "route_buddy_direct_used_tick": int(self._route_buddy_direct_used_tick),
                "route_new_target_search_tick": int(self._route_new_target_search_tick),
                "route_search_fallback_used_tick": int(
                    self._route_search_fallback_used_tick
                ),
                "route_repeat_partner_share_tick": (
                    float(self._route_repeat_partner_requested_tick)
                    / max(
                        1.0,
                        float(
                            self._route_repeat_partner_requested_tick
                            + self._route_exploration_requested_tick
                        ),
                    )
                ),
                "route_exploration_share_tick": (
                    float(self._route_exploration_requested_tick)
                    / max(
                        1.0,
                        float(
                            self._route_repeat_partner_requested_tick
                            + self._route_exploration_requested_tick
                        ),
                    )
                ),
                "route_sticky_share_tick": (
                    float(self._route_sticky_used_tick) / max(1.0, float(route_requested))
                ),
                "route_buddy_direct_share_tick": (
                    float(self._route_buddy_direct_used_tick) / max(1.0, float(route_requested))
                ),
                "route_new_target_search_share_tick": (
                    float(self._route_new_target_search_tick) / max(1.0, float(route_requested))
                ),
                "route_search_fallback_share_tick": (
                    float(self._route_search_fallback_used_tick) / max(1.0, float(route_requested))
                ),
                "noam_routing_swaps_tick": int(self._noam_routing_swaps_tick),
                "noam_clearing_swaps_tick": int(self._noam_clearing_swaps_tick),
                "noam_routing_volume_usd_tick": float(self._noam_routing_volume_usd_tick),
                "noam_clearing_volume_usd_tick": float(self._noam_clearing_volume_usd_tick),
                "noam_routing_fee_usd_tick": float(self._noam_routing_fee_usd_tick),
                "noam_clearing_fee_usd_tick": float(self._noam_clearing_fee_usd_tick),
                "noam_routing_stable_fee_usd_tick": float(self._noam_routing_stable_fee_usd_tick),
                "noam_routing_voucher_fee_usd_tick": float(self._noam_routing_voucher_fee_usd_tick),
                "noam_clearing_stable_fee_usd_tick": float(self._noam_clearing_stable_fee_usd_tick),
                "noam_clearing_voucher_fee_usd_tick": float(self._noam_clearing_voucher_fee_usd_tick),
                "noam_clearing_cycles_attempted_tick": int(self._noam_clearing_cycles_attempted_tick),
                "noam_clearing_cycles_executed_tick": int(self._noam_clearing_cycles_executed_tick),
                "noam_clearing_cycle_success_rate_tick": (
                    float(self._noam_clearing_cycles_executed_tick)
                    / max(1.0, float(self._noam_clearing_cycles_attempted_tick))
                ),
                "noam_clearing_cycle_value_usd_tick": float(
                    self._noam_clearing_cycle_value_usd_tick
                ),
                "ordinary_swap_count_tick": context_count("ordinary"),
                "ordinary_swap_count_total": context_count("ordinary", total=True),
                "ordinary_swap_volume_usd_tick": context_volume("ordinary"),
                "ordinary_swap_volume_usd_total": context_volume("ordinary", total=True),
                "ordinary_stable_source_swap_count_tick": context_source_count("ordinary", "stable"),
                "ordinary_stable_source_swap_count_total": context_source_count(
                    "ordinary", "stable", total=True
                ),
                "ordinary_stable_source_swap_volume_usd_tick": context_source_volume(
                    "ordinary", "stable"
                ),
                "ordinary_stable_source_swap_volume_usd_total": context_source_volume(
                    "ordinary", "stable", total=True
                ),
                "ordinary_voucher_source_swap_count_tick": context_source_count("ordinary", "voucher"),
                "ordinary_voucher_source_swap_count_total": context_source_count(
                    "ordinary", "voucher", total=True
                ),
                "ordinary_voucher_source_swap_volume_usd_tick": context_source_volume(
                    "ordinary", "voucher"
                ),
                "ordinary_voucher_source_swap_volume_usd_total": context_source_volume(
                    "ordinary", "voucher", total=True
                ),
                "loan_route_swap_count_tick": context_count("loan"),
                "loan_route_swap_count_total": context_count("loan", total=True),
                "loan_route_swap_volume_usd_tick": context_volume("loan"),
                "loan_route_swap_volume_usd_total": context_volume("loan", total=True),
                "voucher_loan_route_swap_count_tick": context_count("voucher_loan"),
                "voucher_loan_route_swap_count_total": context_count("voucher_loan", total=True),
                "voucher_loan_route_swap_volume_usd_tick": context_volume("voucher_loan"),
                "voucher_loan_route_swap_volume_usd_total": context_volume("voucher_loan", total=True),
                "repayment_route_swap_count_tick": context_count("repayment"),
                "repayment_route_swap_count_total": context_count("repayment", total=True),
                "repayment_route_swap_volume_usd_tick": context_volume("repayment"),
                "repayment_route_swap_volume_usd_total": context_volume("repayment", total=True),
                "loan_backfill_swap_count_tick": context_count("loan_backfill"),
                "loan_backfill_swap_count_total": context_count("loan_backfill", total=True),
                "loan_backfill_swap_volume_usd_tick": context_volume("loan_backfill"),
                "loan_backfill_swap_volume_usd_total": context_volume("loan_backfill", total=True),
                "route_motif_count_tick": int(route_motif_count_tick),
                "route_motif_count_total": int(route_motif_count_total),
                "route_motif_voucher_to_voucher_count_tick": motif_count("voucher_to_voucher"),
                "route_motif_voucher_to_voucher_count_total": motif_count(
                    "voucher_to_voucher", total=True
                ),
                "route_motif_voucher_to_stable_count_tick": motif_count("voucher_to_stable"),
                "route_motif_voucher_to_stable_count_total": motif_count(
                    "voucher_to_stable", total=True
                ),
                "route_motif_stable_to_voucher_count_tick": motif_count("stable_to_voucher"),
                "route_motif_stable_to_voucher_count_total": motif_count(
                    "stable_to_voucher", total=True
                ),
                "route_motif_other_count_tick": motif_count("other"),
                "route_motif_other_count_total": motif_count("other", total=True),
                "route_motif_voucher_to_voucher_share_total": (
                    motif_count("voucher_to_voucher", total=True)
                    / max(1, route_motif_count_total)
                ),
                "route_motif_voucher_to_stable_share_total": (
                    motif_count("voucher_to_stable", total=True)
                    / max(1, route_motif_count_total)
                ),
                "route_motif_stable_to_voucher_share_total": (
                    motif_count("stable_to_voucher", total=True)
                    / max(1, route_motif_count_total)
                ),
                "route_motif_stable_involved_share_total": (
                    (
                        motif_count("voucher_to_stable", total=True)
                        + motif_count("stable_to_voucher", total=True)
                    )
                    / max(1, route_motif_count_total)
                ),
                "route_motif_voucher_to_voucher_volume_usd_tick": motif_volume("voucher_to_voucher"),
                "route_motif_voucher_to_voucher_volume_usd_total": motif_volume(
                    "voucher_to_voucher", total=True
                ),
                "route_motif_voucher_to_stable_volume_usd_tick": motif_volume("voucher_to_stable"),
                "route_motif_voucher_to_stable_volume_usd_total": motif_volume(
                    "voucher_to_stable", total=True
                ),
                "route_motif_stable_to_voucher_volume_usd_tick": motif_volume("stable_to_voucher"),
                "route_motif_stable_to_voucher_volume_usd_total": motif_volume(
                    "stable_to_voucher", total=True
                ),
                "route_motif_stable_intermediate_count_tick": int(
                    self._route_motif_stable_intermediate_count_tick
                ),
                "route_motif_stable_intermediate_count_total": int(
                    self._route_motif_stable_intermediate_count_total
                ),
                "route_motif_stable_intermediate_volume_usd_tick": float(
                    self._route_motif_stable_intermediate_volume_usd_tick
                ),
                "route_motif_stable_intermediate_volume_usd_total": float(
                    self._route_motif_stable_intermediate_volume_usd_total
                ),
                **route_motif_metric_bundle(
                    "observed",
                    motif_count,
                    motif_volume,
                    route_motif_count_tick,
                    route_motif_count_total,
                ),
                "observed_route_motif_other_count_tick": motif_count("other"),
                "observed_route_motif_other_count_total": motif_count("other", total=True),
                "observed_route_motif_stable_involved_count_tick": (
                    motif_count("voucher_to_stable") + motif_count("stable_to_voucher")
                ),
                "observed_route_motif_stable_involved_count_total": (
                    motif_count("voucher_to_stable", total=True)
                    + motif_count("stable_to_voucher", total=True)
                ),
                "observed_route_motif_stable_involved_volume_usd_tick": (
                    motif_volume("voucher_to_stable") + motif_volume("stable_to_voucher")
                ),
                "observed_route_motif_stable_involved_volume_usd_total": (
                    motif_volume("voucher_to_stable", total=True)
                    + motif_volume("stable_to_voucher", total=True)
                ),
                "observed_route_motif_stable_intermediate_count_tick": int(
                    self._route_motif_stable_intermediate_count_tick
                ),
                "observed_route_motif_stable_intermediate_count_total": int(
                    self._route_motif_stable_intermediate_count_total
                ),
                "observed_route_motif_stable_intermediate_volume_usd_tick": float(
                    self._route_motif_stable_intermediate_volume_usd_tick
                ),
                "observed_route_motif_stable_intermediate_volume_usd_total": float(
                    self._route_motif_stable_intermediate_volume_usd_total
                ),
                "ordinary_route_motif_count_tick": int(ordinary_route_motif_count_tick),
                "ordinary_route_motif_count_total": int(ordinary_route_motif_count_total),
                "ordinary_route_motif_voucher_to_voucher_count_tick": ordinary_motif_count(
                    "voucher_to_voucher"
                ),
                "ordinary_route_motif_voucher_to_voucher_count_total": ordinary_motif_count(
                    "voucher_to_voucher", total=True
                ),
                "ordinary_route_motif_voucher_to_stable_count_tick": ordinary_motif_count(
                    "voucher_to_stable"
                ),
                "ordinary_route_motif_voucher_to_stable_count_total": ordinary_motif_count(
                    "voucher_to_stable", total=True
                ),
                "ordinary_route_motif_stable_to_voucher_count_tick": ordinary_motif_count(
                    "stable_to_voucher"
                ),
                "ordinary_route_motif_stable_to_voucher_count_total": ordinary_motif_count(
                    "stable_to_voucher", total=True
                ),
                "ordinary_route_motif_voucher_to_voucher_share_total": (
                    ordinary_motif_count("voucher_to_voucher", total=True)
                    / max(1, ordinary_route_motif_count_total)
                ),
                "ordinary_route_motif_voucher_to_stable_share_total": (
                    ordinary_motif_count("voucher_to_stable", total=True)
                    / max(1, ordinary_route_motif_count_total)
                ),
                "ordinary_route_motif_stable_to_voucher_share_total": (
                    ordinary_motif_count("stable_to_voucher", total=True)
                    / max(1, ordinary_route_motif_count_total)
                ),
                "ordinary_route_motif_stable_involved_share_total": (
                    (
                        ordinary_motif_count("voucher_to_stable", total=True)
                        + ordinary_motif_count("stable_to_voucher", total=True)
                    )
                    / max(1, ordinary_route_motif_count_total)
                ),
                "ordinary_route_motif_voucher_to_voucher_volume_usd_tick": ordinary_motif_volume(
                    "voucher_to_voucher"
                ),
                "ordinary_route_motif_voucher_to_voucher_volume_usd_total": ordinary_motif_volume(
                    "voucher_to_voucher", total=True
                ),
                "ordinary_route_motif_voucher_to_stable_volume_usd_tick": ordinary_motif_volume(
                    "voucher_to_stable"
                ),
                "ordinary_route_motif_voucher_to_stable_volume_usd_total": ordinary_motif_volume(
                    "voucher_to_stable", total=True
                ),
                "ordinary_route_motif_stable_to_voucher_volume_usd_tick": ordinary_motif_volume(
                    "stable_to_voucher"
                ),
                "ordinary_route_motif_stable_to_voucher_volume_usd_total": ordinary_motif_volume(
                    "stable_to_voucher", total=True
                ),
                "market_route_motif_count_tick": int(market_route_motif_count_tick),
                "market_route_motif_count_total": int(market_route_motif_count_total),
                "market_route_motif_voucher_to_voucher_count_tick": market_motif_count(
                    "voucher_to_voucher"
                ),
                "market_route_motif_voucher_to_voucher_count_total": market_motif_count(
                    "voucher_to_voucher", total=True
                ),
                "market_route_motif_voucher_to_stable_count_tick": market_motif_count(
                    "voucher_to_stable"
                ),
                "market_route_motif_voucher_to_stable_count_total": market_motif_count(
                    "voucher_to_stable", total=True
                ),
                "market_route_motif_stable_to_voucher_count_tick": market_motif_count(
                    "stable_to_voucher"
                ),
                "market_route_motif_stable_to_voucher_count_total": market_motif_count(
                    "stable_to_voucher", total=True
                ),
                "market_route_motif_voucher_to_voucher_share_total": (
                    market_motif_count("voucher_to_voucher", total=True)
                    / max(1, market_route_motif_count_total)
                ),
                "market_route_motif_voucher_to_stable_share_total": (
                    market_motif_count("voucher_to_stable", total=True)
                    / max(1, market_route_motif_count_total)
                ),
                "market_route_motif_stable_to_voucher_share_total": (
                    market_motif_count("stable_to_voucher", total=True)
                    / max(1, market_route_motif_count_total)
                ),
                "market_route_motif_stable_involved_share_total": (
                    (
                        market_motif_count("voucher_to_stable", total=True)
                        + market_motif_count("stable_to_voucher", total=True)
                    )
                    / max(1, market_route_motif_count_total)
                ),
                "market_route_motif_voucher_to_voucher_volume_usd_tick": market_motif_volume(
                    "voucher_to_voucher"
                ),
                "market_route_motif_voucher_to_voucher_volume_usd_total": market_motif_volume(
                    "voucher_to_voucher", total=True
                ),
                "market_route_motif_voucher_to_stable_volume_usd_tick": market_motif_volume(
                    "voucher_to_stable"
                ),
                "market_route_motif_voucher_to_stable_volume_usd_total": market_motif_volume(
                    "voucher_to_stable", total=True
                ),
                "market_route_motif_stable_to_voucher_volume_usd_tick": market_motif_volume(
                    "stable_to_voucher"
                ),
                "market_route_motif_stable_to_voucher_volume_usd_total": market_motif_volume(
                    "stable_to_voucher", total=True
                ),
                **route_motif_metric_bundle(
                    "repayment",
                    repayment_motif_count,
                    repayment_motif_volume,
                    repayment_route_motif_count_tick,
                    repayment_route_motif_count_total,
                ),
                **route_motif_metric_bundle(
                    "loan",
                    loan_motif_count,
                    loan_motif_volume,
                    loan_route_motif_count_tick,
                    loan_route_motif_count_total,
                ),
                "productive_boosted_voucher_swap_count_tick": int(
                    self._productive_boosted_voucher_swap_count_tick
                ),
                "productive_boosted_voucher_swap_count_total": int(
                    self._productive_boosted_voucher_swap_count_total
                ),
                "productive_boosted_voucher_swap_volume_usd_tick": float(
                    self._productive_boosted_voucher_swap_volume_usd_tick
                ),
                "productive_boosted_voucher_swap_volume_usd_total": float(
                    self._productive_boosted_voucher_swap_volume_usd_total
                ),
                "voucher_loan_boosted_voucher_swap_count_tick": int(
                    self._voucher_loan_boosted_voucher_swap_count_tick
                ),
                "voucher_loan_boosted_voucher_swap_count_total": int(
                    self._voucher_loan_boosted_voucher_swap_count_total
                ),
                "voucher_loan_boosted_voucher_swap_volume_usd_tick": float(
                    self._voucher_loan_boosted_voucher_swap_volume_usd_tick
                ),
                "voucher_loan_boosted_voucher_swap_volume_usd_total": float(
                    self._voucher_loan_boosted_voucher_swap_volume_usd_total
                ),
                "ordinary_stable_spend_protected_skip_count_tick": int(
                    self._ordinary_stable_spend_protected_skip_count_tick
                ),
                "ordinary_stable_spend_protected_skip_count_total": int(
                    self._ordinary_stable_spend_protected_skip_count_total
                ),
                "ordinary_stable_spend_protected_skip_value_usd_tick": float(
                    self._ordinary_stable_spend_protected_skip_value_usd_tick
                ),
                "ordinary_stable_spend_protected_skip_value_usd_total": float(
                    self._ordinary_stable_spend_protected_skip_value_usd_total
                ),
                "fee_pool_total_usd": fee_pool_total_usd,
                "fee_clc_total_usd": fee_clc_total_usd,
                "fee_pool_cumulative_usd": float(fee_pool_cumulative),
                "fee_clc_cumulative_usd": float(self._fee_clc_cumulative_usd),
                "fee_pool_cumulative_voucher": float(self._fee_pool_cumulative_voucher),
                "fee_clc_cumulative_voucher": float(self._fee_clc_cumulative_voucher),
                "fee_conversion_attempted_usd_tick": float(self._fee_conversion_attempted_usd_tick),
                "fee_conversion_success_usd_tick": float(self._fee_conversion_success_usd_tick),
                "fee_conversion_failed_usd_tick": float(self._fee_conversion_failed_usd_tick),
                "fee_conversion_attempted_usd_total": float(self._fee_conversion_attempted_usd_total),
                "fee_conversion_success_usd_total": float(self._fee_conversion_success_usd_total),
                "fee_conversion_failed_usd_total": float(self._fee_conversion_failed_usd_total),
                "fee_service_reserved_usd_tick": float(self._fee_service_reserved_usd_tick),
                "fee_service_reserved_usd_total": float(self._fee_service_reserved_usd_total),
                "fee_service_stable_reserved_usd_tick": float(
                    self._fee_service_stable_reserved_usd_tick
                ),
                "fee_service_stable_reserved_usd_total": float(
                    self._fee_service_stable_reserved_usd_total
                ),
                "fee_service_converted_voucher_reserved_usd_tick": float(
                    self._fee_service_converted_voucher_reserved_usd_tick
                ),
                "fee_service_converted_voucher_reserved_usd_total": float(
                    self._fee_service_converted_voucher_reserved_usd_total
                ),
                "lp_sclc_supply_total": float(lp_sclc_total),
                "lp_injected_usd_total": float(self._lp_injected_usd_total),
                "lp_returned_usd_total": float(self._lp_returned_usd_total),
                "lp_net_usd_total": float(self._lp_returned_usd_total - self._lp_injected_usd_total),
                "fee_in_usd_epoch": float(self._waterfall_last.get("fee_in_usd", 0.0)),
                "fee_cash_usd_epoch": float(self._waterfall_last.get("fee_cash_usd", 0.0)),
                "fee_cash_waterfall_usd_epoch": float(
                    self._waterfall_last.get("fee_cash_waterfall_usd", 0.0)
                ),
                "fee_kind_usd_epoch": float(self._waterfall_last.get("fee_kind_usd", 0.0)),
                "fee_service_reserved_usd_epoch": float(
                    self._waterfall_last.get("fee_service_reserved_usd", 0.0)
                ),
                "fee_service_stable_reserved_usd_epoch": float(
                    self._waterfall_last.get("fee_service_stable_reserved_usd", 0.0)
                ),
                "fee_service_converted_voucher_reserved_usd_epoch": float(
                    self._waterfall_last.get("fee_service_converted_voucher_reserved_usd", 0.0)
                ),
                "fee_converted_voucher_cash_usd_epoch": float(
                    self._waterfall_last.get("fee_converted_voucher_cash_usd", 0.0)
                ),
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
                "mandates_allocated_usd_total": float(self._mandates_allocated_usd_total),
                "mandates_distributed_usd_total": float(self._mandates_distributed_usd_total),
                "clc_pool_injected_usd_total": float(self._clc_pool_injected_usd_total),
                "clc_pool_swapped_out_stable_total": float(self._clc_pool_swapped_out_stable_total),
                "clc_pool_swapped_out_voucher_total": float(self._clc_pool_swapped_out_voucher_total),
                "quarterly_clearing_usd_tick": float(self._quarterly_clearing_usd_tick),
                "quarterly_clearing_usd_total": float(self._quarterly_clearing_usd_total),
                "quarterly_clearing_lender_liquidity_before_tick": float(
                    self._quarterly_clearing_lender_liquidity_before_tick
                ),
                "quarterly_clearing_lender_liquidity_after_tick": float(
                    self._quarterly_clearing_lender_liquidity_after_tick
                ),
                "fee_access_budget_usd": float(self._waterfall_last.get("fee_access_budget_usd", 0.0)),
                "claims_paid_usd_tick": float(self._claims_paid_usd_tick),
                "claims_unpaid_usd_tick": float(self._claims_unpaid_usd_tick),
                "incidents_tick": int(self._incidents_tick),
                "stable_onramp_usd_tick": float(self._stable_onramp_usd_tick),
                "stable_offramp_usd_tick": float(self._stable_offramp_usd_tick),
                "historical_stable_backing_usd_tick": float(self._historical_stable_backing_usd_tick),
                "historical_stable_backing_usd_total": float(self._historical_stable_backing_usd_total),
                "historical_stable_backing_pools_tick": int(self._historical_stable_backing_pools_tick),
                "historical_stable_backing_pools_total": int(self._historical_stable_backing_pools_total),
                "historical_stable_backing_producer_usd_total": float(
                    self._historical_stable_backing_usd_by_role.get("producer", 0.0)
                ),
                "historical_stable_backing_consumer_usd_total": float(
                    self._historical_stable_backing_usd_by_role.get("consumer", 0.0)
                ),
                "historical_stable_backing_lender_usd_total": float(
                    self._historical_stable_backing_usd_by_role.get("lender", 0.0)
                ),
                "historical_stable_backing_producer_pools_total": int(
                    self._historical_stable_backing_pools_by_role.get("producer", 0)
                ),
                "historical_stable_backing_consumer_pools_total": int(
                    self._historical_stable_backing_pools_by_role.get("consumer", 0)
                ),
                "historical_stable_backing_lender_pools_total": int(
                    self._historical_stable_backing_pools_by_role.get("lender", 0)
                ),
                "stable_excess_sweep_usd_tick": float(self._stable_excess_sweep_usd_tick),
                "stable_excess_sweep_usd_total": float(self._stable_excess_sweep_usd_total),
                "stable_excess_sweep_pools_tick": int(self._stable_excess_sweep_pools_tick),
                "stable_excess_sweep_pools_total": int(self._stable_excess_sweep_pools_total),
            })
