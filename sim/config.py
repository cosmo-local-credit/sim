from dataclasses import dataclass, field

@dataclass
class ScenarioConfig:
    # Network growth
    initial_pools: int = 10
    initial_lenders: int = 5
    initial_producers: int = 100
    initial_consumers: int = 20
    initial_liquidity_providers: int = 1
    pool_growth_rate_per_tick: float = 0.02
    pool_growth_stride_ticks: int = 4
    max_pools: int | None = 500
    add_pool_offer_assets_mean: int = 4
    add_pool_want_assets_mean: int = 6
    p_offer_overlap: float = 0.75     # offered assets drawn from existing universe
    p_want_overlap: float = 0.85      # wanted assets drawn from existing universe
    desired_assets_min_per_pool: int = 8
    desired_assets_max_per_pool: int = 100
    desired_assets_growth_per_asset: float = 0.2
    desired_assets_add_per_tick: int = 2
    desired_assets_stride_ticks: int = 4

    # Agent role mix
    p_liquidity_provider: float = 0.0
    p_lender: float = 0.25
    p_producer: float = 0.50
    p_consumer: float = 0.25

    # Stablecoin scarcity regime
    stable_symbol: str = "USD"
    initial_stable_per_pool_mean: float = 2000.0
    lender_initial_stable_mean: float = 0.0
    lp_initial_stable_mean: float = 100000.0
    stable_inflow_per_tick: float = 0.0
    producer_inflow_per_tick: float | None = None  # rate of current stable per month
    consumer_inflow_per_tick: float | None = None  # rate of current stable per month
    lender_inflow_per_tick: float = 0.0  # rate of current stable per month
    liquidity_provider_inflow_per_tick: float = 0.0  # rate of current stable per month
    stable_shock_tick: int | None = None
    stable_shock_amount: float = 0.0  # negative drains, positive adds
    stable_growth_mode: str = "per_pool"  # "per_pool" or "network_target"
    stable_growth_stride_ticks: int = 4
    stable_supply_cap: float = 100_000_000.0
    stable_supply_growth_rate: float = 0.15  # per month, toward cap
    stable_supply_noise: float = 0.05  # relative stdev
    stable_outflow_rate: float = 0.02  # per month
    stable_growth_smoothing: float = 0.25  # fraction of target gap applied per tick
    stable_flow_mode: str = "none"  # "none", "loan", "swap", "both"
    stable_flow_loan_scale: float = 1.0
    stable_flow_swap_scale: float = 0.05
    stable_flow_swap_target_usd: float = 0.0
    stable_flow_window_ticks: int = 4
    stable_inflow_activity_share: float = 0.6
    stable_inflow_activity_window_ticks: int = 12
    voucher_inflow_share: float = 0.5  # voucher USD value minted per USD stable inflow
    offramps_enabled: bool = True
    offramp_rate_min_per_tick: float = 0.0
    offramp_rate_max_per_tick: float = 0.02
    offramp_success_ema_alpha: float = 0.2
    offramp_min_attempts: int = 2
    metrics_stride: int = 5
    pool_metrics_stride: int = 10
    max_active_pools_per_tick: int | None = 100
    max_candidate_pools_per_hop: int | None = None
    event_log_maxlen: int | None = None

    # Routing
    routing_mode: str = "noam"  # "noam" or "bfs"
    max_hops: int = 4
    sticky_route_bias: float = 0.6  # 0 disables, 1 = strong preference for past counterparties
    sticky_affinity_decay: float = 0.02  # per tick
    sticky_affinity_gain: float = 0.15  # scale on log1p(USD) per successful swap
    sticky_affinity_cap: float = 50.0
    sticky_fail_threshold: int = 2  # consecutive sticky failures before falling back
    affinity_buddy_count: int = 6  # producer sticks to top N affinity buddies once reached
    noam_topk_pools_per_asset: int = 16
    noam_topm_out_per_pool: int = 16
    noam_beam_width: int = 40
    noam_max_hops: int = 5
    noam_topk_refresh_ticks: int = 4
    noam_dynamic_caps_enabled: bool = True
    noam_dynamic_cap_reference_pools: int = 50
    noam_dynamic_min_topk: int = 4
    noam_dynamic_min_topm: int = 4
    noam_dynamic_min_beam: int = 16
    noam_edge_cap_per_state: int = 30
    noam_dynamic_min_edge_cap: int = 20
    noam_overlay_enabled: bool = True
    noam_hub_asset_count: int = 60
    noam_hub_depth: int = 2
    noam_hub_candidate_limit: int = 10
    noam_overlay_top_r_paths: int = 3
    noam_overlay_max_hops: int = 3
    noam_overlay_refresh_ticks: int = 200
    noam_overlay_min_pools: int = 200
    noam_clearing_enabled: bool = True
    noam_clearing_stride_ticks: int = 2
    noam_clearing_max_cycles: int = 200
    noam_clearing_max_hops: int = 4
    noam_clearing_edge_cap_per_asset: int = 16
    noam_clearing_safety_factor: float = 0.8
    noam_clearing_budget_usd: float = 25000.0
    noam_clearing_budget_share: float = 0.01
    noam_clearing_min_cycle_value_usd: float = 1.0
    noam_success_ema_alpha: float = 0.2
    noam_success_min: float = 0.05
    noam_success_max: float = 0.98
    noam_weight_success: float = 1.2
    noam_weight_fee: float = 1.0
    noam_weight_lambda: float = 1.2
    noam_weight_benefit: float = 1.5
    noam_weight_deadend: float = 1.0
    noam_clc_edge_bonus: float = 0.75
    noam_scarcity_eta: float = 0.1
    noam_safe_budget_fraction: float = 0.2
    noam_lambda_decay: float = 0.1
    noam_usage_cap: float = 5.0
    noam_failure_ttl_ticks: int = 4
    noam_route_cache_ttl_ticks: int = 6
    noam_route_cache_bucket_usd: float = 100.0

    # Limits & fees
    default_window_len: int = 10
    default_cap_in: float = 10_000.0
    lender_voucher_cap_in: float = 2_000.0
    lender_voucher_cap_supply_fraction: float = 0.5
    lender_stable_cap_in: float = 25_000.0
    producer_voucher_cap_in: float = 15_000.0
    producer_stable_cap_in: float = 1_000_000_000.0
    pool_fee_rate: float = 0.02       # 2.0%
    clc_rake_rate: float = 1.00       # 100% of pool fees

    # Economics / Waterfall
    economics_enabled: bool = True
    waterfall_epoch_ticks: int = 4
    waterfall_include_pool_fees: bool = True
    cash_eligible_assets: list[str] = field(default_factory=lambda: ["USD"])
    cash_conversion_slippage_bps: float = 25.0
    cash_conversion_max_usd_per_epoch: float | None = None
    core_ops_budget_usd: float = 2000.0
    insurance_max_topup_usd: float = 10000.0
    liquidity_mandate_share: float = 0.50
    liquidity_mandate_max_usd: float = 0.0
    liquidity_mandate_mode: str = "lender_liquidity"
    liquidity_mandate_activity_window_ticks: int = 12
    liquidity_mandate_max_per_pool_usd: float = 2000.0
    waterfall_alpha_ops_share: float = 0.20
    waterfall_beta_liquidity_share: float = 0.40
    waterfall_gamma_insurance_share: float = 0.40
    lp_waterfall_contribution_rate: float = 1.0
    lp_sclc_supply_cap: float = 100_000_000.0
    sclc_symbol: str = "sCLC"
    sclc_fee_access_enabled: bool = True
    sclc_fee_access_share: float = 0.50
    sclc_emission_cap_usd: float = 2000.0
    sclc_requires_insurance_target: bool = True
    sclc_requires_core_ops: bool = True
    sclc_swap_window_ticks: int = 4
    sclc_swap_window_open_ticks: int = 1
    clc_pool_always_open: bool = True

    # CLC pool rebalancing (voucher -> stable)
    clc_rebalance_enabled: bool = True
    clc_rebalance_interval_ticks: int = 1
    clc_rebalance_max_swaps_per_tick: int = 2
    clc_rebalance_target_stable_ratio: float = 0.50
    clc_rebalance_swap_size_frac: float = 0.05
    clc_rebalance_min_usd: float = 25.0

    # Insurance / incidents
    insurance_target_multiplier: float = 0.02
    insurance_risk_weight_base: float = 1.0
    insurance_risk_weight_reserve_scale: float = 1.0
    insurance_risk_weight_min: float = 0.5
    insurance_risk_weight_max: float = 3.0
    incident_base_rate: float = 0.01
    incident_loss_rate: float = 0.05
    incident_min_loss_usd: float = 100.0
    incident_haircut_cap: float = 0.10
    incident_max_per_tick: int = 1
    insurance_fee_window_ticks: int = 12
    insurance_min_fee_usd: float = 25.0

    # Redemption
    base_redeem_prob: float = 0.85
    redeem_bias_swap_only: float = 0.10
    redeem_bias_mixed: float = 0.05
    redeem_bias_borrow_only: float = 0.00

    # Activity (max swap attempts per pool per tick)
    random_route_requests_per_tick: int = 4
    swap_requests_budget_per_tick: int | None = 100
    random_request_amount_mean: float = 200.0

    # Loan repayment (weeks)
    loan_term_weeks: int = 12
    loan_activity_period_ticks: int = 12  # spread loan issuance/repayment across ticks

    # Swap sizing (share of pool value, per attempt)
    swap_size_mean_frac: float = 0.02
    swap_size_min_usd: float = 1.0
    swap_size_max_usd: float | None = None
    swap_asset_selection_mode: str = "value_weighted"  # "uniform" or "value_weighted"
    swap_limits_enabled: bool = True
    swap_target_selection_mode: str = "liquidity_weighted"  # "uniform" or "liquidity_weighted"
    swap_target_retry_count: int = 2
    swap_attempts_value_scale_usd: float = 500_000.0
    swap_attempts_max_per_pool: int | None = 4
    utilization_target_rate: float = 0.02  # swap volume / pool value target (0 disables)
    utilization_boost_max: float = 3.0  # max attempts multiplier when under target

    # Debug
    debug_inventory: bool = True

    def __post_init__(self) -> None:
        if self.producer_inflow_per_tick is None:
            self.producer_inflow_per_tick = 0.05
        if self.consumer_inflow_per_tick is None:
            self.consumer_inflow_per_tick = 0.05
        if self.stable_symbol not in self.cash_eligible_assets:
            self.cash_eligible_assets.append(self.stable_symbol)
