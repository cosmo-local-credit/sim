import argparse
import unittest

from scripts.run_regenbond_monte_carlo import scenario_config
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
        self.assertAlmostEqual(engine._producer_debt_repaid_usd_total, 50.0)
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

        reduced = engine._reduce_producer_debt_obligations(
            lender_pool.pool_id,
            voucher_id,
            40.0,
            "test",
        )

        self.assertAlmostEqual(reduced, 40.0)
        self.assertAlmostEqual(engine._producer_debt_obligations[0].remaining_voucher_units, 60.0)
        self.assertAlmostEqual(engine._producer_debt_closed_by_circulation_usd_total, 40.0)

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
