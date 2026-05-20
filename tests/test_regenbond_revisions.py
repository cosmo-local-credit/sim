import argparse
import unittest

from scripts.run_regenbond_monte_carlo import (
    bond_metrics,
    frontier_baseline_metrics,
    scenario_config,
    summarize_frontier_cell,
)
from sim.config import ScenarioConfig
from sim.core import Event, IssuerLedger
from sim.engine import SimulationEngine


def small_config(**overrides):
    cfg = ScenarioConfig(
        initial_lenders=1,
        initial_producers=1,
        initial_consumers=0,
        initial_liquidity_providers=1,
        max_pools=3,
        debug_inventory=False,
        metrics_stride=0,
        pool_metrics_stride=0,
        event_log_maxlen=None,
        stable_supply_growth_rate=0.0,
        stable_supply_noise=0.0,
        stable_inflow_per_tick=0.0,
        producer_inflow_per_tick=0.0,
        consumer_inflow_per_tick=0.0,
        lender_inflow_per_tick=0.0,
        offramps_enabled=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class RegenBondRevisionTests(unittest.TestCase):
    def test_issuer_ledger_tracks_net_outstanding_and_cumulative_returns(self):
        ledger = IssuerLedger("VCHR:test")
        ledger.issue(100.0)
        ledger.return_to_issuer(30.0)
        ledger.redeem(100.0)
        ledger.redeem(10.0)

        self.assertEqual(ledger.issued_total, 100.0)
        self.assertEqual(ledger.issuer_returned_total, 30.0)
        self.assertEqual(ledger.redeemed_total, 100.0)
        self.assertEqual(ledger.outstanding_supply, 0.0)

    def test_deposit_based_lender_cap_uses_five_times_deposited_value(self):
        engine = SimulationEngine(small_config(producer_deposits_enabled=True))
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        lender_pool.values.set_value(voucher_id, 1.0)
        engine._producer_deposit_value_by_voucher[voucher_id] = 100.0

        self.assertAlmostEqual(engine._lender_voucher_cap(voucher_id, lender_pool), 500.0)

    def test_producer_loan_diagnostics_capture_live_limit_clipping(self):
        engine = SimulationEngine(
            small_config(
                producer_deposits_enabled=True,
                swap_size_min_usd=100.0,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        producer_pool.values.set_value(voucher_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=5.0)
        lender_pool.policy.min_stable_reserve = 0.0
        engine._producer_deposit_value_by_voucher[voucher_id] = 1000.0
        engine._vault_add(producer_pool, voucher_id, 200.0, "test_seed", "test")
        engine._vault_add(lender_pool, stable_id, 1000.0, "test_seed", "test")
        engine.tick = 1

        attempted = engine._attempt_new_loan(producer_pool, voucher_id)

        self.assertTrue(attempted)
        self.assertEqual(engine._producer_loan_attempts_tick, 1)
        self.assertAlmostEqual(engine._producer_loan_lender_remaining_cap_usd_tick, 5.0)
        self.assertGreater(engine._producer_loan_clipped_lender_remaining_usd_tick, 0.0)
        self.assertLessEqual(engine._producer_loan_attempted_usd_tick, 5.0 + 1e-9)

    def test_snapshot_reports_lender_stable_available_above_reserve(self):
        engine = SimulationEngine(small_config())
        stable_id = engine.cfg.stable_symbol
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        current_stable = lender_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(lender_pool, stable_id, current_stable, "test_clear", "test")
        lender_pool.policy.min_stable_reserve = 25.0
        engine._vault_add(lender_pool, stable_id, 100.0, "test_seed", "test")
        engine.tick = 1

        engine.snapshot_metrics(force_network=True)
        latest = engine.metrics.network_rows[-1]

        self.assertAlmostEqual(latest["lender_stable_total_usd"], 100.0)
        self.assertAlmostEqual(latest["lender_stable_reserve_usd"], 25.0)
        self.assertAlmostEqual(latest["lender_stable_available_above_reserve_usd"], 75.0)

    def test_snapshot_reports_role_specific_stable_reserve_stress(self):
        engine = SimulationEngine(
            small_config(initial_consumers=1, max_pools=4)
        )
        stable_id = engine.cfg.stable_symbol
        producer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "producer")
        consumer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "consumer")
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        for pool in (producer_pool, consumer_pool, lender_pool):
            current_stable = pool.vault.get(stable_id)
            if current_stable > 0.0:
                engine._vault_sub(pool, stable_id, current_stable, "test_clear", "test")
        producer_pool.policy.min_stable_reserve = 100.0
        consumer_pool.policy.min_stable_reserve = 50.0
        lender_pool.policy.min_stable_reserve = 25.0
        engine._vault_add(producer_pool, stable_id, 40.0, "test_seed", "test")
        engine._vault_add(consumer_pool, stable_id, 50.0, "test_seed", "test")
        engine._vault_add(lender_pool, stable_id, 10.0, "test_seed", "test")
        engine.tick = 1

        engine.snapshot_metrics(force_network=True)
        latest = engine.metrics.network_rows[-1]

        self.assertEqual(latest["producer_pools_under_stable_reserve"], 1)
        self.assertAlmostEqual(latest["producer_stable_reserve_deficit_usd"], 60.0)
        self.assertEqual(latest["consumer_pools_under_stable_reserve"], 0)
        self.assertEqual(latest["lender_pools_under_stable_reserve"], 1)
        self.assertAlmostEqual(latest["lender_stable_reserve_deficit_usd"], 15.0)
        self.assertEqual(latest["community_pools_under_stable_reserve"], 1)
        self.assertAlmostEqual(latest["community_stable_reserve_stress_ratio"], 0.5)

    def test_ordinary_stable_spend_protection_caps_stable_source_amount(self):
        engine = SimulationEngine(
            small_config(
                ordinary_stable_spend_protection_enabled=True,
                ordinary_stable_spend_buffer_voucher_share=0.05,
                swap_size_min_usd=1000.0,
                swap_size_mean_frac=1.0,
                stable_source_swap_size_multiplier=1.0,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        voucher_id = producer.voucher_spec.voucher_id
        current_stable = producer_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(producer_pool, stable_id, current_stable, "test_clear", "test")
        producer_pool.policy.min_stable_reserve = 100.0
        producer_pool.values.set_value(voucher_id, 1.0)
        current_voucher = producer_pool.vault.get(voucher_id)
        if current_voucher > 0.0:
            engine._vault_sub(producer_pool, voucher_id, current_voucher, "test_clear", "test")
        engine._vault_add(producer_pool, stable_id, 150.0, "test_seed", "test")
        engine._vault_add(producer_pool, voucher_id, 200.0, "test_seed", "test")

        self.assertAlmostEqual(
            engine._ordinary_source_spendable_amount(producer_pool, stable_id),
            40.0,
        )
        self.assertLessEqual(engine._sample_amount_in(producer_pool, stable_id), 40.0 + 1e-9)

    def test_failed_loan_route_backfill_attempts_ordinary_activity(self):
        engine = SimulationEngine(
            small_config(
                producer_loan_failure_backfill_enabled=True,
                producer_loan_failure_backfill_max_attempts=1,
            )
        )
        producer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "producer")
        calls = []

        def fake_random_route_request(source_pool=None, max_assets=None):
            calls.append((source_pool, max_assets))
            engine._noam_routing_swaps_tick += 1
            return 1

        engine._random_route_request = fake_random_route_request

        engine._backfill_failed_loan_route(producer_pool)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], producer_pool)
        self.assertEqual(calls[0][1], 1)
        self.assertEqual(engine._producer_loan_backfill_attempts_tick, 1)
        self.assertEqual(engine._producer_loan_backfill_executed_tick, 1)

    def test_community_deficit_then_lender_mandate_fills_community_first(self):
        engine = SimulationEngine(
            small_config(
                initial_consumers=1,
                max_pools=4,
                liquidity_mandate_mode="community_deficit_then_lender",
            )
        )
        stable_id = engine.cfg.stable_symbol
        producer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "producer")
        consumer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "consumer")
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        mandates_pool = engine.pools[engine.mandates_pool_id]
        for pool in (producer_pool, consumer_pool, lender_pool, mandates_pool):
            current_stable = pool.vault.get(stable_id)
            if current_stable > 0.0:
                engine._vault_sub(pool, stable_id, current_stable, "test_clear", "test")
        producer_pool.policy.min_stable_reserve = 50.0
        consumer_pool.policy.min_stable_reserve = 30.0
        engine._vault_add(consumer_pool, stable_id, 10.0, "test_seed", "test")
        engine._vault_add(mandates_pool, stable_id, 100.0, "test_seed", "test")

        distributed = engine._distribute_liquidity_mandates(100.0)

        self.assertAlmostEqual(distributed, 100.0)
        self.assertAlmostEqual(producer_pool.vault.get(stable_id), 50.0)
        self.assertAlmostEqual(consumer_pool.vault.get(stable_id), 30.0)
        self.assertAlmostEqual(lender_pool.vault.get(stable_id), 30.0)

    def test_frontier_bond_principal_is_seeded_directly_to_lenders(self):
        args = argparse.Namespace(
            pool_metrics_stride=0,
            max_active_pools_per_tick=None,
            _calibration_kes_per_usd=128.0,
            _voucher_unit_value_usd=1.0 / 128.0,
            _current_principal_usd=1000.0,
            _current_scale_factor=1.0,
            _current_bond_fee_service_share=1.0,
            _current_initial_lenders=4,
            _current_initial_producers=2,
            _current_initial_consumers=3,
            issuer_reserve_share=0.10,
            issuer_payment_stride=13,
            _frontier_producer_stable_deposit_rate_per_month=0.0,
            _frontier_producer_voucher_deposit_rate_per_month=0.0,
            _frontier_productive_credit_return_rate=0.0,
            _frontier_productive_credit_lag_ticks=2,
            _frontier_productive_credit_voucher_deposit_share=0.384157,
            _frontier_productive_credit_voucher_deposit_cap_rate_per_month=0.143206,
            _frontier_producer_debt_maturity_recovery_rate=0.0,
            _frontier_quarterly_clearing_surplus_share=1.0,
            _frontier_route_requests_per_tick=1,
            _frontier_swap_floor_per_tick=0,
            _frontier_historical_cash_backing_total_usd=0.0,
            _frontier_historical_voucher_backing_total_usd=0.0,
        )

        cfg = scenario_config("bond_issuer_frontier", 0.06, 260, args)

        self.assertEqual(cfg.initial_liquidity_providers, 0)
        self.assertAlmostEqual(cfg.lp_initial_stable_mean, 0.0)
        self.assertAlmostEqual(cfg.lender_initial_stable_mean, 250.0)
        self.assertAlmostEqual(cfg.bond_gross_principal_usd, 1000.0)
        self.assertAlmostEqual(cfg.bond_deployed_principal_usd, 1000.0)
        self.assertAlmostEqual(cfg.issuer_reserve_share, 0.0)
        self.assertAlmostEqual(cfg.producer_debt_maturity_recovery_rate, 1.0)
        self.assertTrue(cfg.bond_service_reserve_enabled)
        self.assertEqual(cfg.bond_service_lockbox_mode, "remaining_schedule")
        self.assertAlmostEqual(cfg.bond_service_lockbox_coverage_ratio, 1.25)
        self.assertTrue(cfg.producer_debt_contract_repayment_enabled)
        self.assertAlmostEqual(cfg.producer_debt_contract_service_margin_rate, 0.50)
        self.assertAlmostEqual(cfg.producer_debt_contract_revenue_rate, 1.50)
        self.assertTrue(cfg.productive_credit_voucher_feedback_enabled)
        self.assertAlmostEqual(cfg.productive_credit_voucher_deposit_share, 0.384157)
        self.assertAlmostEqual(cfg.productive_credit_voucher_deposit_cap_rate_per_month, 0.143206)
        self.assertTrue(cfg.productive_credit_voucher_activity_boost_enabled)
        self.assertAlmostEqual(cfg.productive_credit_voucher_source_weight_boost, 0.50)
        self.assertAlmostEqual(cfg.productive_credit_voucher_source_size_multiplier, 1.25)
        self.assertTrue(cfg.ordinary_stable_spend_protection_enabled)
        self.assertAlmostEqual(cfg.ordinary_stable_spend_buffer_voucher_share, 0.05)
        self.assertTrue(cfg.producer_loan_failure_backfill_enabled)
        self.assertEqual(cfg.producer_loan_failure_backfill_max_attempts, 1)
        self.assertEqual(cfg.liquidity_mandate_mode, "community_deficit_then_lender")
        self.assertEqual(cfg.max_pools, 9)

        engine = SimulationEngine(cfg, seed=7)
        stable_id = cfg.stable_symbol
        lenders = [
            pool for pool in engine.pools.values()
            if not pool.policy.system_pool and pool.policy.role == "lender"
        ]
        liquidity_providers = [
            pool for pool in engine.pools.values()
            if not pool.policy.system_pool and pool.policy.role == "liquidity_provider"
        ]

        self.assertEqual(len(lenders), 4)
        self.assertEqual(len(liquidity_providers), 0)
        self.assertAlmostEqual(sum(pool.vault.get(stable_id) for pool in lenders), 1000.0)

    def test_voucher_unit_value_prices_one_ksh_voucher_against_usd_stable(self):
        engine = SimulationEngine(small_config(kes_per_usd=128.0, voucher_unit_value_usd=1.0 / 128.0))
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        voucher_id = producer.voucher_spec.voucher_id

        self.assertAlmostEqual(producer_pool.values.get_value(voucher_id), 1.0 / 128.0)
        ok, reason, amount_out, fee = producer_pool.quote_swap(engine.cfg.stable_symbol, 1.0, voucher_id)

        self.assertTrue(ok, reason)
        self.assertAlmostEqual(amount_out + fee, 128.0)
        self.assertAlmostEqual(amount_out, 125.44)

    def test_productive_credit_inflow_is_scheduled_and_recorded_as_deposit(self):
        engine = SimulationEngine(
            small_config(
                producer_deposits_enabled=True,
                productive_credit_enabled=True,
                productive_credit_return_rate=0.5,
                productive_credit_lag_ticks=2,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        engine.tick = 1
        engine._schedule_productive_credit_inflow(producer.pool_id, 200.0, producer.voucher_spec.voucher_id)
        engine.tick = 3
        before = producer_pool.vault.get(engine.cfg.stable_symbol)

        engine._apply_productive_credit_inflows()

        self.assertAlmostEqual(producer_pool.vault.get(engine.cfg.stable_symbol) - before, 100.0)
        self.assertAlmostEqual(engine._productive_credit_inflow_usd_tick, 100.0)
        self.assertGreaterEqual(
            engine._producer_deposit_value_by_voucher[producer.voucher_spec.voucher_id],
            100.0,
        )

    def test_contract_productive_credit_uses_revenue_rate_when_larger(self):
        engine = SimulationEngine(
            small_config(
                producer_deposits_enabled=True,
                productive_credit_enabled=True,
                productive_credit_return_rate=0.5,
                productive_credit_lag_ticks=1,
                producer_debt_contract_repayment_enabled=True,
                producer_debt_contract_revenue_rate=1.35,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        engine.tick = 1
        engine._schedule_productive_credit_inflow(producer.pool_id, 200.0, producer.voucher_spec.voucher_id)
        engine.tick = 2
        before = producer_pool.vault.get(engine.cfg.stable_symbol)

        engine._apply_productive_credit_inflows()

        self.assertAlmostEqual(producer_pool.vault.get(engine.cfg.stable_symbol) - before, 270.0)
        self.assertAlmostEqual(engine._productive_credit_inflow_usd_tick, 270.0)

    def test_productive_credit_voucher_feedback_splits_inflow_without_double_counting(self):
        engine = SimulationEngine(
            small_config(
                producer_deposits_enabled=True,
                productive_credit_enabled=True,
                productive_credit_return_rate=1.0,
                productive_credit_lag_ticks=1,
                productive_credit_voucher_feedback_enabled=True,
                productive_credit_voucher_deposit_share=0.40,
                productive_credit_voucher_deposit_cap_rate_per_month=0.0,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        voucher_id = producer.voucher_spec.voucher_id
        value = engine._asset_value(producer_pool, voucher_id)
        before_stable = producer_pool.vault.get(engine.cfg.stable_symbol)
        before_voucher = producer_pool.vault.get(voucher_id)
        before_capacity = engine._producer_deposit_credit_capacity_usd()

        engine.tick = 1
        engine._schedule_productive_credit_inflow(producer.pool_id, 100.0, voucher_id)
        engine.tick = 2
        engine._apply_productive_credit_inflows()

        self.assertAlmostEqual(engine._productive_credit_inflow_usd_tick, 100.0)
        self.assertAlmostEqual(engine._productive_credit_stable_retained_usd_tick, 60.0)
        self.assertAlmostEqual(engine._productive_credit_voucher_deposit_usd_tick, 40.0)
        self.assertAlmostEqual(producer_pool.vault.get(engine.cfg.stable_symbol) - before_stable, 60.0)
        self.assertAlmostEqual(producer_pool.vault.get(voucher_id) - before_voucher, 40.0 / value)
        self.assertAlmostEqual(
            engine._producer_deposit_credit_capacity_usd() - before_capacity,
            100.0 * engine.cfg.lender_voucher_cap_deposit_multiple,
        )

    def test_productive_credit_voucher_feedback_cap_bounds_issued_vouchers(self):
        engine = SimulationEngine(
            small_config(
                producer_deposits_enabled=True,
                productive_credit_enabled=True,
                productive_credit_return_rate=1.0,
                productive_credit_lag_ticks=1,
                productive_credit_voucher_feedback_enabled=True,
                productive_credit_voucher_deposit_share=1.0,
                productive_credit_voucher_deposit_cap_rate_per_month=0.20,
                random_request_amount_mean=100.0,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        current_stable = producer_pool.vault.get(engine.cfg.stable_symbol)
        if current_stable > 0.0:
            engine._vault_sub(producer_pool, engine.cfg.stable_symbol, current_stable, "test_clear", "test")
        current_voucher = producer_pool.vault.get(producer.voucher_spec.voucher_id)
        if current_voucher > 0.0:
            engine._vault_sub(producer_pool, producer.voucher_spec.voucher_id, current_voucher, "test_clear", "test")

        engine.tick = 1
        engine._schedule_productive_credit_inflow(producer.pool_id, 100.0, producer.voucher_spec.voucher_id)
        engine.tick = 2
        engine._apply_productive_credit_inflows()

        self.assertAlmostEqual(engine._productive_credit_voucher_deposit_usd_tick, 5.0)
        self.assertAlmostEqual(engine._productive_credit_stable_retained_usd_tick, 95.0)
        self.assertAlmostEqual(engine._productive_credit_voucher_deposit_cap_clipped_usd_tick, 95.0)

    def test_productive_credit_voucher_activity_boosts_own_voucher_source_weight(self):
        engine = SimulationEngine(
            small_config(
                productive_credit_voucher_activity_boost_enabled=True,
                productive_credit_voucher_activity_boost_window_ticks=4,
                productive_credit_voucher_source_weight_boost=1.0,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        voucher_id = producer.voucher_spec.voucher_id
        stable_id = engine.cfg.stable_symbol
        current_stable = producer_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(producer_pool, stable_id, current_stable, "test_clear", "test")
        current_voucher = producer_pool.vault.get(voucher_id)
        if current_voucher > 0.0:
            engine._vault_sub(producer_pool, voucher_id, current_voucher, "test_clear", "test")
        engine._vault_add(producer_pool, stable_id, 100.0, "test_seed", "test")
        engine._vault_add(producer_pool, voucher_id, 100.0, "test_seed", "test")
        engine.tick = 10
        before = engine._source_asset_selection_weights(producer_pool, [stable_id, voucher_id])

        engine._mark_productive_credit_voucher_activity(producer_pool.pool_id, voucher_id)
        after = engine._source_asset_selection_weights(producer_pool, [stable_id, voucher_id])

        self.assertAlmostEqual(after[0], before[0])
        self.assertAlmostEqual(after[1], before[1] * 2.0)

    def test_historical_voucher_backing_is_recorded_as_producer_deposit(self):
        engine = SimulationEngine(
            small_config(
                kes_per_usd=100.0,
                voucher_unit_value_usd=0.01,
                historical_voucher_backing_tick=2,
                historical_voucher_backing_total_usd=250.0,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        voucher_id = producer.voucher_spec.voucher_id
        before_units = producer_pool.vault.get(voucher_id)
        before_deposit_value = engine._producer_deposit_value_by_voucher.get(voucher_id, 0.0)
        before_credit_capacity = engine._producer_deposit_credit_capacity_usd()

        engine.tick = 2
        engine._apply_historical_voucher_backing()

        self.assertAlmostEqual(engine._producer_deposit_voucher_usd_total, 250.0)
        self.assertAlmostEqual(
            engine._producer_deposit_value_by_voucher[voucher_id] - before_deposit_value,
            250.0,
        )
        self.assertAlmostEqual(producer_pool.vault.get(voucher_id) - before_units, 25_000.0)
        self.assertAlmostEqual(
            engine._producer_deposit_credit_capacity_usd() - before_credit_capacity,
            1_250.0,
        )

    def test_historical_stable_backing_applies_once_to_eligible_roles(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=2,
                initial_consumers=1,
                initial_liquidity_providers=0,
                max_pools=4,
                historical_stable_backing_tick=2,
                historical_stable_backing_total_usd=120.0,
                historical_stable_backing_roles=("producer", "consumer"),
            )
        )
        stable_id = engine.cfg.stable_symbol
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        eligible_pools = [
            pool for pool in engine.pools.values() if pool.policy.role in ("producer", "consumer")
        ]
        before = {pool.pool_id: pool.vault.get(stable_id) for pool in [lender_pool, *eligible_pools]}

        engine.tick = 2
        engine._apply_historical_stable_backing()

        self.assertAlmostEqual(engine._historical_stable_backing_usd_total, 120.0)
        self.assertAlmostEqual(engine._stable_onramp_usd_tick, 120.0)
        self.assertEqual(engine._historical_stable_backing_pools_total, len(eligible_pools))
        self.assertAlmostEqual(lender_pool.vault.get(stable_id), before[lender_pool.pool_id])
        for pool in eligible_pools:
            self.assertAlmostEqual(pool.vault.get(stable_id) - before[pool.pool_id], 40.0)

    def test_consumer_weighted_source_selection_can_choose_voucher_with_stable_present(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=0,
                initial_producers=1,
                initial_consumers=1,
                initial_liquidity_providers=0,
                max_pools=2,
                consumer_stable_source_bias=0.0,
            )
        )
        consumer = next(agent for agent in engine.agents.values() if agent.role == "consumer")
        consumer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "consumer")
        voucher_id = consumer.voucher_spec.voucher_id
        consumer_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=1, cap_in=1e12)
        engine._vault_add(consumer_pool, engine.cfg.stable_symbol, 100.0, "test_seed", "test")
        engine._vault_add(consumer_pool, voucher_id, 100.0, "test_seed", "test")

        weights = engine._source_asset_selection_weights(
            consumer_pool,
            [engine.cfg.stable_symbol, voucher_id],
        )

        self.assertAlmostEqual(float(weights[0]), 0.0)
        self.assertGreater(float(weights[1]), 0.0)

    def test_producer_weighted_source_selection_can_choose_voucher_with_stable_present(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=0,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=1,
                producer_stable_source_bias=0.0,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        voucher_id = producer.voucher_spec.voucher_id
        engine._vault_add(producer_pool, engine.cfg.stable_symbol, 100.0, "test_seed", "test")
        engine._vault_add(producer_pool, voucher_id, 100.0, "test_seed", "test")

        weights = engine._source_asset_selection_weights(
            producer_pool,
            [engine.cfg.stable_symbol, voucher_id],
        )

        self.assertAlmostEqual(float(weights[0]), 0.0)
        self.assertGreater(float(weights[1]), 0.0)

    def test_stable_excess_sweep_preserves_reserve_and_records_offramp(self):
        engine = SimulationEngine(
            small_config(
                stable_excess_sweep_enabled=True,
                stable_excess_sweep_buffer_voucher_share=0.0,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "producer")
        current_stable = producer_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(producer_pool, stable_id, current_stable, "test_clear", "test")
        producer_pool.policy.min_stable_reserve = 25.0
        engine._vault_add(producer_pool, stable_id, 100.0, "test_seed", "test")

        engine._apply_stable_excess_sweep()

        self.assertAlmostEqual(producer_pool.vault.get(stable_id), 25.0)
        self.assertAlmostEqual(engine._stable_offramp_usd_tick, 75.0)
        self.assertAlmostEqual(engine._stable_excess_sweep_usd_total, 75.0)
        self.assertEqual(engine._stable_excess_sweep_pools_total, 1)

    def test_voucher_fee_conversion_records_failed_routed_conversion(self):
        engine = SimulationEngine(
            small_config(
                voucher_fee_conversion_enabled=True,
                voucher_fee_conversion_max_swaps_per_epoch=1,
                voucher_fee_conversion_min_usd=1.0,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        clc_pool = engine.pools[engine.clc_pool_id]
        attempted, converted, kind, _remaining = engine._attempt_fee_voucher_conversion(
            clc_pool,
            producer.voucher_spec.voucher_id,
            10.0,
            1.0,
            None,
        )

        self.assertAlmostEqual(attempted, 10.0)
        self.assertAlmostEqual(converted, 0.0)
        self.assertAlmostEqual(kind, 10.0)
        self.assertAlmostEqual(engine._fee_conversion_failed_usd_total, 10.0)

    def test_stable_fee_service_reserves_before_waterfall(self):
        engine = SimulationEngine(
            small_config(
                bond_return_mode="issuer_cashflow",
                bond_service_reserve_enabled=True,
                bond_service_lockbox_mode="remaining_schedule",
                bond_service_lockbox_coverage_ratio=1.25,
                bond_fee_service_share=0.5,
                bond_gross_principal_usd=1000.0,
                bond_term_ticks=260,
                insurance_max_topup_usd=0.0,
                core_ops_budget_usd=0.0,
                liquidity_mandate_share=0.0,
                liquidity_mandate_bootstrap_epochs=0,
            )
        )
        source_pool = next(pool for pool in engine.pools.values() if not pool.policy.system_pool)
        source_pool.fee_ledger_pool[engine.cfg.stable_symbol] = 100.0

        engine._apply_waterfall()

        self.assertAlmostEqual(engine._fee_service_reserved_usd_total, 50.0)
        self.assertAlmostEqual(engine._fee_service_stable_reserved_usd_total, 50.0)
        self.assertAlmostEqual(engine._fee_service_converted_voucher_reserved_usd_total, 0.0)
        self.assertAlmostEqual(engine._bond_service_reserve_usd_balance, 50.0)
        self.assertAlmostEqual(engine._waterfall_last["fee_cash_usd"], 100.0)
        self.assertAlmostEqual(engine._waterfall_last["fee_cash_waterfall_usd"], 50.0)
        self.assertAlmostEqual(engine._waterfall_last["clc_alloc_usd"], 50.0)

    def test_fee_service_share_zero_preserves_waterfall_behavior(self):
        engine = SimulationEngine(
            small_config(
                bond_return_mode="issuer_cashflow",
                bond_service_reserve_enabled=True,
                bond_fee_service_share=0.0,
                bond_gross_principal_usd=1000.0,
                bond_term_ticks=260,
                insurance_max_topup_usd=0.0,
                core_ops_budget_usd=0.0,
                liquidity_mandate_share=0.0,
                liquidity_mandate_bootstrap_epochs=0,
            )
        )
        source_pool = next(pool for pool in engine.pools.values() if not pool.policy.system_pool)
        source_pool.fee_ledger_pool[engine.cfg.stable_symbol] = 100.0

        engine._apply_waterfall()

        self.assertAlmostEqual(engine._fee_service_reserved_usd_total, 0.0)
        self.assertAlmostEqual(engine._bond_service_reserve_usd_balance, 0.0)
        self.assertAlmostEqual(engine._waterfall_last["fee_cash_waterfall_usd"], 100.0)
        self.assertAlmostEqual(engine._waterfall_last["clc_alloc_usd"], 100.0)

    def test_fee_service_reservation_stops_when_lockbox_target_is_met(self):
        engine = SimulationEngine(
            small_config(
                bond_return_mode="issuer_cashflow",
                bond_service_reserve_enabled=True,
                bond_service_lockbox_mode="remaining_schedule",
                bond_service_lockbox_coverage_ratio=1.25,
                bond_fee_service_share=1.0,
                bond_gross_principal_usd=1000.0,
                bond_term_ticks=260,
                insurance_max_topup_usd=0.0,
                core_ops_budget_usd=0.0,
                liquidity_mandate_share=0.0,
                liquidity_mandate_bootstrap_epochs=0,
            )
        )
        engine._bond_service_reserve_usd_balance = engine._bond_service_lockbox_target_usd()
        source_pool = next(pool for pool in engine.pools.values() if not pool.policy.system_pool)
        source_pool.fee_ledger_pool[engine.cfg.stable_symbol] = 100.0

        engine._apply_waterfall()

        self.assertAlmostEqual(engine._fee_service_reserved_usd_total, 0.0)
        self.assertAlmostEqual(engine._waterfall_last["fee_cash_waterfall_usd"], 100.0)
        self.assertAlmostEqual(engine._waterfall_last["clc_alloc_usd"], 100.0)

    def test_converted_voucher_fee_service_reserves_before_waterfall(self):
        engine = SimulationEngine(
            small_config(
                voucher_fee_conversion_enabled=True,
                bond_return_mode="issuer_cashflow",
                bond_service_reserve_enabled=True,
                bond_service_lockbox_mode="remaining_schedule",
                bond_service_lockbox_coverage_ratio=1.25,
                bond_fee_service_share=0.5,
                bond_gross_principal_usd=1000.0,
                bond_term_ticks=260,
                insurance_max_topup_usd=0.0,
                core_ops_budget_usd=0.0,
                liquidity_mandate_share=0.0,
                liquidity_mandate_bootstrap_epochs=0,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        source_pool = next(pool for pool in engine.pools.values() if not pool.policy.system_pool)
        clc_pool = engine.pools[engine.clc_pool_id]
        voucher_id = producer.voucher_spec.voucher_id
        source_pool.values.set_value(voucher_id, 1.0)
        source_pool.fee_ledger_pool[voucher_id] = 100.0

        def fake_conversion(clc_pool, asset_id, amount, avg_value, remaining_cap_usd):
            engine._vault_add(clc_pool, engine.cfg.stable_symbol, 80.0, "test_conversion", "test")
            return 100.0, 80.0, 20.0, remaining_cap_usd

        engine._attempt_fee_voucher_conversion = fake_conversion

        engine._apply_waterfall()

        self.assertAlmostEqual(engine._fee_service_reserved_usd_total, 40.0)
        self.assertAlmostEqual(engine._fee_service_stable_reserved_usd_total, 0.0)
        self.assertAlmostEqual(engine._fee_service_converted_voucher_reserved_usd_total, 40.0)
        self.assertAlmostEqual(engine._bond_service_reserve_usd_balance, 40.0)
        self.assertAlmostEqual(engine._waterfall_last["fee_cash_usd"], 80.0)
        self.assertAlmostEqual(engine._waterfall_last["fee_cash_waterfall_usd"], 40.0)
        self.assertAlmostEqual(engine._waterfall_last["fee_kind_usd"], 20.0)
        self.assertAlmostEqual(engine._waterfall_last["fee_converted_voucher_cash_usd"], 40.0)
        self.assertAlmostEqual(engine._waterfall_last["clc_alloc_usd"], 40.0)

    def test_quarterly_clearing_moves_only_recovered_lender_surplus_to_clc(self):
        engine = SimulationEngine(
            small_config(
                quarterly_clearing_enabled=True,
                quarterly_clearing_stride_ticks=13,
                bond_gross_principal_usd=1000.0,
                bond_term_ticks=260,
            )
        )
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        clc_pool = engine.pools[engine.clc_pool_id]
        engine._vault_add(lender_pool, engine.cfg.stable_symbol, 100.0, "test_seed", "test")
        lender_pool.policy.min_stable_reserve = 25.0
        engine._lender_recovered_stable_by_pool[lender_pool.pool_id] = 80.0
        engine.tick = 13
        clc_before = clc_pool.vault.get(engine.cfg.stable_symbol)

        engine._apply_quarterly_clearing()

        self.assertAlmostEqual(engine._quarterly_clearing_usd_total, 50.0)
        self.assertAlmostEqual(clc_pool.vault.get(engine.cfg.stable_symbol) - clc_before, 50.0)
        self.assertAlmostEqual(lender_pool.vault.get(engine.cfg.stable_symbol), 50.0)

    def test_issuer_cashflow_clearing_pays_bondholders_without_clc_waterfall(self):
        engine = SimulationEngine(
            small_config(
                quarterly_clearing_enabled=True,
                quarterly_clearing_stride_ticks=13,
                bond_return_mode="issuer_cashflow",
                bond_gross_principal_usd=1000.0,
                bond_term_ticks=260,
            )
        )
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        clc_pool = engine.pools[engine.clc_pool_id]
        engine._vault_add(lender_pool, engine.cfg.stable_symbol, 100.0, "test_seed", "test")
        lender_pool.policy.min_stable_reserve = 25.0
        engine._lender_recovered_stable_by_pool[lender_pool.pool_id] = 80.0
        engine.tick = 13
        clc_before = clc_pool.vault.get(engine.cfg.stable_symbol)

        engine._apply_quarterly_clearing()

        self.assertAlmostEqual(engine._quarterly_clearing_usd_total, 50.0)
        self.assertAlmostEqual(engine._lp_returned_usd_total, 50.0)
        self.assertAlmostEqual(engine._stable_offramp_usd_tick, 50.0)
        self.assertAlmostEqual(clc_pool.vault.get(engine.cfg.stable_symbol), clc_before)
        self.assertAlmostEqual(lender_pool.vault.get(engine.cfg.stable_symbol), 50.0)

    def test_bond_service_reserve_prioritizes_scheduled_debt_service(self):
        engine = SimulationEngine(
            small_config(
                quarterly_clearing_enabled=True,
                quarterly_clearing_stride_ticks=13,
                bond_return_mode="issuer_cashflow",
                bond_gross_principal_usd=1000.0,
                bond_term_ticks=260,
                bond_service_reserve_enabled=True,
                bond_service_reserve_recovery_share=1.0,
            )
        )
        stable_id = engine.cfg.stable_symbol
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        current_stable = lender_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(lender_pool, stable_id, current_stable, "test_clear", "test")
        engine._vault_add(lender_pool, stable_id, 100.0, "test_recovery", "test")

        engine.tick = 1
        engine._record_lender_recovered_stable(lender_pool.pool_id, 100.0, "borrower_stable_repayment")

        self.assertAlmostEqual(engine._bond_service_reserve_usd_balance, 50.0)
        self.assertAlmostEqual(engine._bond_service_reserved_usd_total, 50.0)
        self.assertAlmostEqual(engine._lender_recovered_stable_by_pool[lender_pool.pool_id], 50.0)
        self.assertAlmostEqual(lender_pool.vault.get(stable_id), 50.0)

        engine.tick = 13
        engine._apply_quarterly_clearing()

        self.assertAlmostEqual(engine._bond_service_paid_from_reserve_usd_total, 50.0)
        self.assertAlmostEqual(engine._bond_service_reserve_usd_balance, 0.0)
        self.assertAlmostEqual(engine._lp_returned_usd_total, 50.0)
        self.assertAlmostEqual(engine._quarterly_clearing_usd_total, 0.0)

    def test_remaining_schedule_lockbox_reserves_full_term_service_with_buffer(self):
        engine = SimulationEngine(
            small_config(
                quarterly_clearing_enabled=True,
                quarterly_clearing_stride_ticks=13,
                bond_return_mode="issuer_cashflow",
                bond_gross_principal_usd=1000.0,
                bond_term_ticks=260,
                bond_service_reserve_enabled=True,
                bond_service_reserve_recovery_share=1.0,
                bond_service_lockbox_mode="remaining_schedule",
                bond_service_lockbox_coverage_ratio=1.25,
            )
        )
        stable_id = engine.cfg.stable_symbol
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        current_stable = lender_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(lender_pool, stable_id, current_stable, "test_clear", "test")
        engine._vault_add(lender_pool, stable_id, 2000.0, "test_recovery", "test")

        engine.tick = 1
        engine._record_lender_recovered_stable(lender_pool.pool_id, 2000.0, "borrower_stable_repayment")

        self.assertAlmostEqual(engine._bond_service_lockbox_target_usd(), 1250.0)
        self.assertAlmostEqual(engine._bond_service_reserve_usd_balance, 1250.0)
        self.assertAlmostEqual(engine._bond_service_reserved_usd_total, 1250.0)
        self.assertAlmostEqual(engine._lender_recovered_stable_by_pool[lender_pool.pool_id], 750.0)
        self.assertAlmostEqual(lender_pool.vault.get(stable_id), 750.0)

        engine.tick = 13
        engine._apply_quarterly_clearing()

        self.assertAlmostEqual(engine._bond_service_paid_from_reserve_usd_total, 50.0)
        self.assertAlmostEqual(engine._bond_service_reserve_usd_balance, 1200.0)
        self.assertAlmostEqual(engine._lp_returned_usd_total, 50.0)
        self.assertAlmostEqual(engine._quarterly_clearing_usd_total, 0.0)

    def test_non_issuer_cashflow_mode_does_not_reserve_recovered_stable(self):
        engine = SimulationEngine(
            small_config(
                bond_return_mode="lp_sclc",
                bond_gross_principal_usd=1000.0,
                bond_term_ticks=260,
                bond_service_reserve_enabled=True,
                bond_service_reserve_recovery_share=1.0,
                bond_service_lockbox_mode="remaining_schedule",
                bond_service_lockbox_coverage_ratio=1.25,
            )
        )
        stable_id = engine.cfg.stable_symbol
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        current_stable = lender_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(lender_pool, stable_id, current_stable, "test_clear", "test")
        engine._vault_add(lender_pool, stable_id, 100.0, "test_recovery", "test")

        engine.tick = 1
        engine._record_lender_recovered_stable(lender_pool.pool_id, 100.0, "borrower_stable_repayment")

        self.assertAlmostEqual(engine._bond_service_reserve_usd_balance, 0.0)
        self.assertAlmostEqual(engine._bond_service_reserved_usd_total, 0.0)
        self.assertAlmostEqual(engine._lender_recovered_stable_by_pool[lender_pool.pool_id], 100.0)
        self.assertAlmostEqual(lender_pool.vault.get(stable_id), 100.0)

    def test_issuer_cashflow_bond_metrics_do_not_haircut_principal_service_share(self):
        cfg = small_config(
            bond_return_mode="issuer_cashflow",
            bond_gross_principal_usd=1000.0,
            bond_term_ticks=260,
            bond_fee_service_share=0.5,
        )

        metrics = bond_metrics({"lp_returned_usd_total": 100.0}, cfg, 13)

        self.assertAlmostEqual(metrics["bond_cumulative_fee_return_usd"], 100.0)
        self.assertAlmostEqual(metrics["issuer_eligible_fee_service_inflow_usd"], 100.0)
        self.assertAlmostEqual(metrics["issuer_actual_bondholder_payment_usd"], 50.0)
        self.assertAlmostEqual(metrics["issuer_paid_coverage_ratio"], 1.0)
        self.assertAlmostEqual(metrics["issuer_service_cash_headroom_ratio"], 2.0)

    def test_frontier_safety_splits_payment_coverage_from_cash_headroom(self):
        row = {
            "bond_principal_usd": 1000.0,
            "principal_ratio": 0.05,
            "network_scale": "current",
            "coupon_target_annual": 0.0,
            "bond_fee_service_share": 0.5,
            "issuer_service_coverage_ratio": 1.0,
            "issuer_paid_coverage_ratio": 1.0,
            "issuer_service_cash_headroom_ratio": 1.5,
            "issuer_scheduled_debt_service_due_usd": 100.0,
            "issuer_actual_bondholder_payment_usd": 100.0,
            "issuer_unpaid_scheduled_claim_usd": 0.0,
            "route_success_rate_cumulative": 1.0,
            "realized_edge_top_share": 0.0,
        }

        summary = summarize_frontier_cell([row] * 5, {}, 0.85, "diagnostic")

        self.assertEqual(summary["safe"], 1)
        self.assertAlmostEqual(summary["scheduled_payment_coverage_p50"], 1.0)
        self.assertAlmostEqual(summary["service_cash_headroom_p50"], 1.5)
        self.assertNotIn("p50_service_coverage", summary["binding_constraint"])

    def test_frontier_safety_requires_cash_headroom_above_payment_coverage(self):
        row = {
            "bond_principal_usd": 1000.0,
            "principal_ratio": 0.05,
            "network_scale": "current",
            "coupon_target_annual": 0.0,
            "bond_fee_service_share": 0.5,
            "issuer_service_coverage_ratio": 1.0,
            "issuer_paid_coverage_ratio": 1.0,
            "issuer_service_cash_headroom_ratio": 1.0,
            "issuer_scheduled_debt_service_due_usd": 100.0,
            "issuer_actual_bondholder_payment_usd": 100.0,
            "issuer_unpaid_scheduled_claim_usd": 0.0,
            "route_success_rate_cumulative": 1.0,
            "realized_edge_top_share": 0.0,
        }

        summary = summarize_frontier_cell([row] * 5, {}, 0.85, "diagnostic")

        self.assertEqual(summary["safe"], 0)
        self.assertIn("p50_available_service_cash_headroom", summary["binding_constraint"])

    def test_frontier_safety_decomposes_material_decline_reasons(self):
        row = {
            "bond_principal_usd": 1000.0,
            "principal_ratio": 0.05,
            "network_scale": "current",
            "coupon_target_annual": 0.0,
            "bond_fee_service_share": 0.5,
            "issuer_service_coverage_ratio": 1.0,
            "issuer_paid_coverage_ratio": 1.0,
            "issuer_service_cash_headroom_ratio": 2.0,
            "issuer_available_service_cash_headroom_ratio": 2.0,
            "issuer_scheduled_debt_service_due_usd": 100.0,
            "issuer_actual_bondholder_payment_usd": 100.0,
            "issuer_unpaid_scheduled_claim_usd": 0.0,
            "route_success_rate_cumulative": 1.0,
            "realized_edge_top_share": 0.0,
            "swap_volume_usd_total": 20.0,
            "transactions_total": 100.0,
            "swap_count_vchr_to_vchr_total": 80.0,
            "swap_volume_vchr_to_vchr_total": 50.0,
            "stable_value_share_in_active_pools": 0.10,
            "voucher_value_share_in_active_pools": 0.90,
            "consumer_stable_reserve_stress_ratio": 0.0,
            "community_stable_reserve_stress_ratio": 0.0,
            "stable_liquidity_leakage_ratio_cumulative": 0.0,
        }
        baseline = {
            "route_success_p50": 1.0,
            "swap_volume_p50": 100.0,
            "voucher_to_voucher_count_p50": 80.0,
            "voucher_to_voucher_share_p50": 0.80,
            "stable_value_share_p95": 0.10,
            "voucher_value_share_p50": 0.90,
            "consumer_cash_stress_p50": 0.0,
            "consumer_cash_stress_p95": 0.0,
            "community_cash_stress_p50": 0.0,
            "community_cash_stress_p95": 0.0,
            "liquidity_leakage_p50": 0.0,
            "liquidity_leakage_p95": 0.0,
        }

        summary = summarize_frontier_cell([row] * 5, baseline, 0.85, "diagnostic")

        self.assertEqual(summary["safe"], 0)
        self.assertEqual(summary["material_decline_swap_volume_decline"], 1)
        self.assertIn("swap_volume_decline", summary["material_decline_reasons"])
        self.assertIn("swap_volume_decline_vs_no_bond", summary["binding_constraint"])

    def test_frontier_safety_reports_baseline_productive_credit_feedback(self):
        baseline_row = {
            "route_success_rate_cumulative": 1.0,
            "swap_volume_usd_total": 100.0,
            "transactions_total": 100.0,
            "swap_count_vchr_to_vchr_total": 80.0,
            "stable_value_share_in_active_pools": 0.10,
            "voucher_value_share_in_active_pools": 0.90,
            "household_cash_stress_ratio": 0.0,
            "consumer_stable_reserve_stress_ratio": 0.0,
            "community_stable_reserve_stress_ratio": 0.0,
            "stable_liquidity_leakage_ratio_cumulative": 0.0,
            "producer_deposit_credit_capacity_usd": 500.0,
            "productive_credit_inflow_usd_total": 100.0,
            "productive_credit_stable_retained_usd_total": 60.0,
            "productive_credit_voucher_deposit_usd_total": 40.0,
        }
        row = {
            **baseline_row,
            "bond_principal_usd": 1000.0,
            "principal_ratio": 0.05,
            "network_scale": "current",
            "coupon_target_annual": 0.0,
            "bond_fee_service_share": 0.5,
            "issuer_service_coverage_ratio": 1.0,
            "issuer_paid_coverage_ratio": 1.0,
            "issuer_service_cash_headroom_ratio": 2.0,
            "issuer_available_service_cash_headroom_ratio": 2.0,
            "issuer_scheduled_debt_service_due_usd": 100.0,
            "issuer_actual_bondholder_payment_usd": 100.0,
            "issuer_unpaid_scheduled_claim_usd": 0.0,
            "realized_edge_top_share": 0.0,
            "producer_deposit_credit_capacity_usd": 750.0,
            "productive_credit_inflow_usd_total": 150.0,
            "productive_credit_stable_retained_usd_total": 90.0,
            "productive_credit_voucher_deposit_usd_total": 60.0,
        }

        baseline = frontier_baseline_metrics([baseline_row] * 5)
        summary = summarize_frontier_cell([row] * 5, baseline, 0.85, "diagnostic")

        self.assertAlmostEqual(summary["baseline_productive_credit_inflow_usd_total_p50"], 100.0)
        self.assertAlmostEqual(summary["baseline_productive_credit_voucher_deposit_usd_total_p50"], 40.0)
        self.assertAlmostEqual(summary["incremental_productive_credit_inflow_usd_total_p50"], 50.0)
        self.assertAlmostEqual(summary["incremental_productive_credit_voucher_deposit_usd_total_p50"], 20.0)
        self.assertAlmostEqual(summary["productive_credit_voucher_deposit_ratio_vs_baseline"], 1.5)
        self.assertAlmostEqual(summary["producer_deposit_credit_capacity_ratio_vs_baseline"], 1.5)

    def test_producer_debt_maturity_repayment_recovers_lender_stable(self):
        engine = SimulationEngine(
            small_config(
                producer_debt_maturity_enabled=True,
                producer_debt_maturity_ticks=1,
                producer_debt_maturity_recovery_rate=1.0,
                producer_debt_maturity_preserve_reserve=False,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=1e12)
        engine._vault_add(lender_pool, voucher_id, 50.0, "test_debt_seed", producer_pool.pool_id)
        engine._vault_add(producer_pool, stable_id, 50.0, "test_stable_seed", "test")
        engine.tick = 1
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            50.0,
            50.0,
        )

        engine.tick = 2
        lender_stable_before = lender_pool.vault.get(stable_id)
        engine._apply_producer_debt_maturities()

        self.assertAlmostEqual(lender_pool.vault.get(stable_id) - lender_stable_before, 50.0)
        self.assertAlmostEqual(engine._lender_recovered_stable_by_pool[lender_pool.pool_id], 50.0)
        self.assertAlmostEqual(engine._lender_recovered_stable_usd_total, 50.0)
        self.assertAlmostEqual(engine._lender_recovered_stable_borrower_maturity_usd_total, 50.0)
        self.assertAlmostEqual(engine._producer_debt_repaid_usd_total, 50.0)
        self.assertAlmostEqual(engine._producer_debt_repaid_maturity_usd_total, 50.0)
        self.assertAlmostEqual(engine._producer_debt_stable_recovered_usd_total, 50.0)
        self.assertAlmostEqual(engine._producer_debt_defaulted_usd_total, 0.0)
        self.assertEqual(len(engine._producer_debt_obligations), 0)

    def test_producer_debt_maturity_applies_default_rate(self):
        engine = SimulationEngine(
            small_config(
                producer_debt_maturity_enabled=True,
                producer_debt_maturity_ticks=1,
                producer_debt_maturity_recovery_rate=0.5,
                producer_debt_maturity_preserve_reserve=False,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=1e12)
        engine._vault_add(lender_pool, voucher_id, 100.0, "test_debt_seed", producer_pool.pool_id)
        engine._vault_add(producer_pool, stable_id, 100.0, "test_stable_seed", "test")
        engine.tick = 1
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
        )

        engine.tick = 2
        engine._apply_producer_debt_maturities()

        self.assertAlmostEqual(engine._producer_debt_repaid_usd_total, 50.0)
        self.assertAlmostEqual(engine._producer_debt_defaulted_usd_total, 50.0)
        self.assertAlmostEqual(lender_pool.vault.get(voucher_id), 0.0)

    def test_producer_debt_obligation_can_close_when_lender_swaps_out_voucher(self):
        engine = SimulationEngine(small_config(producer_debt_maturity_enabled=True))
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        lender_pool.values.set_value(voucher_id, 1.0)
        engine.tick = 1
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
        )
        self.assertAlmostEqual(engine._producer_debt_obligations[0].cash_service_remaining_usd, 0.0)

        reduced = engine._reduce_producer_debt_obligations(
            lender_pool.pool_id,
            voucher_id,
            40.0,
            "test",
        )

        self.assertAlmostEqual(reduced, 40.0)
        self.assertAlmostEqual(engine._producer_debt_obligations[0].remaining_voucher_units, 60.0)
        self.assertAlmostEqual(engine._producer_debt_closed_by_circulation_usd_total, 40.0)

    def test_contract_cash_service_can_recover_stable_after_voucher_circulation(self):
        engine = SimulationEngine(
            small_config(
                producer_debt_maturity_enabled=True,
                producer_debt_contract_repayment_enabled=True,
                producer_debt_contract_service_margin_rate=0.25,
                producer_debt_maturity_preserve_reserve=False,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        lender_pool.values.set_value(voucher_id, 1.0)
        engine.tick = 1
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
        )
        obligation = engine._producer_debt_obligations[0]
        self.assertAlmostEqual(obligation.cash_service_remaining_usd, 125.0)
        engine._reduce_producer_debt_obligations(
            lender_pool.pool_id,
            voucher_id,
            100.0,
            "producer_voucher_swap_out",
            source_pool_id="producer:other",
            source_role="producer",
        )
        self.assertAlmostEqual(obligation.remaining_voucher_units, 0.0)
        self.assertAlmostEqual(obligation.cash_service_remaining_usd, 125.0)
        engine._vault_add(producer_pool, stable_id, 125.0, "test_revenue", "test")
        lender_stable_before = lender_pool.vault.get(stable_id)

        paid = engine._execute_producer_debt_cash_service_payment(
            obligation,
            125.0,
            "borrower_stable_repayment",
        )

        self.assertAlmostEqual(paid, 125.0)
        self.assertAlmostEqual(lender_pool.vault.get(stable_id) - lender_stable_before, 125.0)
        self.assertAlmostEqual(obligation.cash_service_remaining_usd, 0.0)
        self.assertAlmostEqual(engine._producer_debt_cash_service_paid_usd_total, 125.0)
        self.assertAlmostEqual(engine._lender_recovered_stable_borrower_regular_usd_total, 125.0)

    def test_producer_debt_reduction_splits_stable_and_circulation_paths(self):
        engine = SimulationEngine(small_config(producer_debt_maturity_enabled=True))
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        lender_pool.values.set_value(voucher_id, 1.0)
        engine.tick = 1
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
        )

        borrower_units = engine._reduce_producer_debt_obligations(
            lender_pool.pool_id,
            voucher_id,
            40.0,
            "borrower_stable_repayment",
            source_pool_id=producer_pool.pool_id,
            source_role="producer",
        )
        consumer_units = engine._reduce_producer_debt_obligations(
            lender_pool.pool_id,
            voucher_id,
            25.0,
            "consumer_stable_purchase",
            source_pool_id="consumer:test",
            source_role="consumer",
        )
        voucher_swap_units = engine._reduce_producer_debt_obligations(
            lender_pool.pool_id,
            voucher_id,
            10.0,
            "producer_voucher_swap_out",
            source_pool_id="producer:other",
            source_role="producer",
        )

        self.assertAlmostEqual(borrower_units, 40.0)
        self.assertAlmostEqual(consumer_units, 25.0)
        self.assertAlmostEqual(voucher_swap_units, 10.0)
        self.assertAlmostEqual(engine._producer_debt_repaid_usd_total, 40.0)
        self.assertAlmostEqual(engine._producer_debt_repaid_regular_usd_total, 40.0)
        self.assertAlmostEqual(engine._producer_debt_stable_recovered_usd_total, 65.0)
        self.assertAlmostEqual(engine._producer_debt_consumer_stable_purchase_usd_total, 25.0)
        self.assertAlmostEqual(engine._producer_debt_closed_by_voucher_swap_usd_total, 10.0)
        self.assertAlmostEqual(engine._producer_debt_closed_by_circulation_usd_total, 10.0)

    def test_route_metrics_separate_fixed_and_substituted_attempts(self):
        engine = SimulationEngine(small_config())
        engine.tick = 1
        engine.log.add(Event(1, "ROUTE_REQUESTED", meta={"route_attempt_kind": "fixed"}))
        engine.log.add(Event(1, "ROUTE_FAILED", meta={"route_attempt_kind": "fixed"}))
        engine.log.add(Event(1, "ROUTE_REQUESTED", meta={"route_attempt_kind": "substitution"}))
        engine.log.add(Event(1, "ROUTE_FOUND", meta={"route_attempt_kind": "substitution"}))

        engine.snapshot_metrics(force_network=True)
        latest = engine.metrics.network_rows[-1]

        self.assertEqual(latest["route_fixed_failed_tick"], 1)
        self.assertEqual(latest["route_substitution_found_tick"], 1)


if __name__ == "__main__":
    unittest.main()
