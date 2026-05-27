import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_regenbond_monte_carlo import (
    SHARD_CONFIG_KEYS,
    bond_metrics,
    configure_sarafu_activity_controls,
    engine_validation_moments,
    frontier_baseline_metrics,
    frontier_validation_gate_status,
    issuer_headroom_frontier_rows,
    load_calibration,
    scenario_config,
    summarize_frontier_cell,
)
from sim.config import ScenarioConfig
from sim.core import Event, IssuerLedger
from sim.engine import SimulationEngine
from sim.router import Hop, RoutePlan


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

    def test_empirical_overlap_lists_producer_voucher_on_sampled_lenders(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=3,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=4,
                producer_voucher_single_lender=False,
                producer_voucher_overlap_mode="empirical_overlap",
                producer_voucher_overlap_bucket_weights={"3": 1.0},
            ),
            seed=11,
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        voucher_id = producer.voucher_spec.voucher_id
        lender_ids = engine._producer_voucher_lender_assignments[voucher_id]
        lender_pools = [
            pool for pool in engine.pools.values()
            if not pool.policy.system_pool and pool.policy.role == "lender"
        ]

        self.assertEqual(len(lender_ids), 3)
        for pool in lender_pools:
            self.assertTrue(pool.registry.is_listed(engine.cfg.stable_symbol))
            self.assertTrue(pool.registry.is_listed(voucher_id))
        engine.snapshot_metrics(force=True)
        metrics = engine.metrics.network_rows[-1]
        self.assertAlmostEqual(metrics["producer_voucher_multi_lender_share"], 1.0)
        self.assertAlmostEqual(metrics["producer_voucher_lender_degree_p50"], 3.0)

    def test_target_asset_candidate_cache_invalidates_by_tick(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                swap_target_selection_mode="liquidity_weighted",
            ),
            seed=19,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        lender_pool = next(
            pool for pool in engine.pools.values()
            if not pool.policy.system_pool and pool.policy.role == "lender"
        )
        voucher_id = producer.voucher_spec.voucher_id
        lender_pool.values.set_value(stable_id, 1.0)
        lender_pool.values.set_value(voucher_id, 1.0)
        engine._vault_add(lender_pool, stable_id, 10.0, "test_seed", "test")
        engine._vault_add(lender_pool, voucher_id, 10.0, "test_seed", "test")

        engine.tick = 1
        first = engine._target_asset_candidate_universe(None, False)
        second = engine._target_asset_candidate_universe(None, False)
        self.assertIs(first, second)
        self.assertIn(stable_id, first)
        self.assertIn(voucher_id, first)

        engine.tick = 2
        third = engine._target_asset_candidate_universe(None, False)
        self.assertIsNot(first, third)
        self.assertEqual(set(first), set(third))

    def test_affinity_buddies_use_top_known_partner_without_full_threshold(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=2,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=3,
                affinity_buddy_count=6,
                affinity_buddy_min_count=1,
            ),
            seed=23,
        )
        producer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "producer")
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")

        engine._update_affinity(producer_pool.pool_id, lender_pool.pool_id, 100.0)

        self.assertEqual(engine._affinity_buddies_for_pool(producer_pool.pool_id), {lender_pool.pool_id})

    def test_repeat_partner_mode_uses_sticky_route_before_new_target_search(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=1,
                initial_liquidity_providers=0,
                max_pools=3,
                decision_based_activity_enabled=True,
                repeat_partner_route_share=1.0,
                route_substitution_enabled=False,
                swap_target_retry_count=1,
            ),
            seed=29,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        source_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "consumer")
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        source_pool.policy.min_stable_reserve = 0.0
        engine._vault_add(source_pool, stable_id, 25.0, "test_seed", "test")
        lender_pool.values.set_value(voucher_id, 1.0)
        lender_pool.values.set_value(stable_id, 1.0)
        engine._vault_add(lender_pool, voucher_id, 100.0, "test_seed", "test")
        engine._sticky_target_by_pool[(source_pool.pool_id, stable_id)] = voucher_id
        engine._sticky_plan_by_pool[(source_pool.pool_id, stable_id, voucher_id)] = RoutePlan(
            ok=True,
            reason="test_sticky",
            hops=[
                Hop(
                    pool_id=lender_pool.pool_id,
                    asset_in=stable_id,
                    asset_out=voucher_id,
                    amount_in=1.0,
                )
            ],
        )

        attempted = engine._random_route_request(
            source_pool=source_pool,
            max_assets=1,
            activity_mode="repeat_partner",
        )

        self.assertEqual(attempted, 1)
        self.assertEqual(engine._route_repeat_partner_requested_tick, 1)
        self.assertEqual(engine._route_sticky_used_tick, 1)
        self.assertEqual(engine._route_new_target_search_tick, 0)

    def test_ordinary_own_voucher_to_voucher_route_is_credit_origination(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=2,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=3,
                decision_based_activity_enabled=True,
                repeat_partner_route_share=1.0,
                route_substitution_enabled=False,
                swap_target_retry_count=1,
                producer_debt_maturity_enabled=True,
            ),
            seed=31,
        )
        stable_id = engine.cfg.stable_symbol
        producers = [agent for agent in engine.agents.values() if agent.role == "producer"]
        source_agent, target_agent = producers[0], producers[1]
        producer_pool = engine.pools[source_agent.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = source_agent.voucher_spec.voucher_id
        target_voucher_id = target_agent.voucher_spec.voucher_id
        for pool in (producer_pool, lender_pool):
            for asset_id in (stable_id, voucher_id, target_voucher_id):
                pool.values.set_value(asset_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(
            target_voucher_id,
            value=1.0,
            window_len=10,
            cap_in=500.0,
        )
        for asset_id in (stable_id, voucher_id, target_voucher_id):
            current = producer_pool.vault.get(asset_id)
            if current > 0.0:
                engine._vault_sub(producer_pool, asset_id, current, "test_clear", "test")
        engine._vault_add(producer_pool, voucher_id, 25.0, "test_seed", "test")
        engine._vault_add(lender_pool, target_voucher_id, 100.0, "test_seed", "test")
        engine._sticky_target_by_pool[(producer_pool.pool_id, voucher_id)] = target_voucher_id
        engine._sticky_plan_by_pool[(producer_pool.pool_id, voucher_id, target_voucher_id)] = RoutePlan(
            ok=True,
            reason="own_voucher_to_voucher",
            hops=[Hop(lender_pool.pool_id, voucher_id, target_voucher_id, 1.0)],
        )

        attempted = engine._random_route_request(
            source_pool=producer_pool,
            max_assets=1,
            activity_mode="repeat_partner",
        )

        self.assertEqual(attempted, 1)
        self.assertEqual(engine._route_sticky_used_tick, 1)
        self.assertEqual(engine._loan_route_motif_count_total.get("voucher_to_voucher"), 1)
        self.assertEqual(engine._market_route_motif_count_total.get("voucher_to_voucher", 0), 0)
        self.assertEqual(len(engine._producer_debt_obligations), 1)

    def test_ordinary_own_voucher_to_stable_route_is_not_market_circulation(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                decision_based_activity_enabled=True,
                repeat_partner_route_share=1.0,
                route_substitution_enabled=False,
                swap_target_retry_count=1,
            ),
            seed=32,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for asset_id in (stable_id, voucher_id):
            producer_pool.values.set_value(asset_id, 1.0)
            lender_pool.values.set_value(asset_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
        for asset_id in (stable_id, voucher_id):
            current = producer_pool.vault.get(asset_id)
            if current > 0.0:
                engine._vault_sub(producer_pool, asset_id, current, "test_clear", "test")
        engine._vault_add(producer_pool, voucher_id, 25.0, "test_seed", "test")
        engine._vault_add(lender_pool, stable_id, 100.0, "test_seed", "test")
        engine._sticky_target_by_pool[(producer_pool.pool_id, voucher_id)] = stable_id
        engine._sticky_plan_by_pool[(producer_pool.pool_id, voucher_id, stable_id)] = RoutePlan(
            ok=True,
            reason="own_voucher_to_stable",
            hops=[Hop(lender_pool.pool_id, voucher_id, stable_id, 1.0)],
        )

        attempted = engine._random_route_request(
            source_pool=producer_pool,
            max_assets=1,
            activity_mode="repeat_partner",
        )

        self.assertEqual(attempted, 0)
        self.assertEqual(engine._route_sticky_used_tick, 0)
        self.assertEqual(engine._market_route_motif_count_total.get("voucher_to_stable", 0), 0)

    def test_ordinary_own_voucher_to_stable_route_executes_as_loan_when_enabled(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                decision_based_activity_enabled=True,
                repeat_partner_route_share=1.0,
                route_substitution_enabled=False,
                swap_target_retry_count=1,
                ordinary_own_voucher_stable_borrowing_enabled=True,
                producer_debt_maturity_enabled=True,
                pool_fee_rate=0.0,
                clc_rake_rate=0.0,
            ),
            seed=132,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for asset_id in (stable_id, voucher_id):
            producer_pool.values.set_value(asset_id, 1.0)
            lender_pool.values.set_value(asset_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
        for asset_id in (stable_id, voucher_id):
            current = producer_pool.vault.get(asset_id)
            if current > 0.0:
                engine._vault_sub(producer_pool, asset_id, current, "test_clear", "test")
        engine._vault_add(producer_pool, voucher_id, 25.0, "test_seed", "test")
        engine._vault_add(lender_pool, stable_id, 100.0, "test_seed", "test")
        engine._sticky_target_by_pool[(producer_pool.pool_id, voucher_id)] = stable_id
        engine._sticky_plan_by_pool[(producer_pool.pool_id, voucher_id, stable_id)] = RoutePlan(
            ok=True,
            reason="own_voucher_to_stable",
            hops=[Hop(lender_pool.pool_id, voucher_id, stable_id, 10.0)],
        )

        attempted = engine._random_route_request(
            source_pool=producer_pool,
            max_assets=1,
            activity_mode="repeat_partner",
        )

        self.assertEqual(attempted, 1)
        self.assertEqual(engine._route_sticky_used_tick, 1)
        self.assertEqual(engine._loan_route_motif_count_total.get("voucher_to_stable"), 1)
        self.assertEqual(engine._market_route_motif_count_total.get("voucher_to_stable", 0), 0)
        self.assertEqual(len(engine._producer_debt_obligations), 1)
        self.assertEqual(engine._producer_debt_obligations[0].debt_kind, "stable")
        engine.snapshot_metrics(force_network=True)
        latest = engine.metrics.network_rows[-1]
        self.assertEqual(latest["observed_route_motif_voucher_to_stable_count_total"], 1)
        self.assertEqual(latest["loan_route_motif_voucher_to_stable_count_total"], 1)
        self.assertEqual(latest["market_route_motif_voucher_to_stable_count_total"], 0)

    def test_ordinary_own_voucher_stable_probability_zero_preserves_voucher_loan(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=2,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=3,
                decision_based_activity_enabled=True,
                repeat_partner_route_share=1.0,
                route_substitution_enabled=False,
                swap_target_retry_count=1,
                ordinary_own_voucher_stable_borrowing_enabled=True,
                ordinary_own_voucher_stable_borrowing_probability=0.0,
                producer_debt_maturity_enabled=True,
                pool_fee_rate=0.0,
                clc_rake_rate=0.0,
            ),
            seed=133,
        )
        stable_id = engine.cfg.stable_symbol
        producers = [agent for agent in engine.agents.values() if agent.role == "producer"]
        source_agent, target_agent = producers[0], producers[1]
        producer_pool = engine.pools[source_agent.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = source_agent.voucher_spec.voucher_id
        target_voucher_id = target_agent.voucher_spec.voucher_id
        for pool in (producer_pool, lender_pool):
            for asset_id in (stable_id, voucher_id, target_voucher_id):
                pool.values.set_value(asset_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(
            target_voucher_id,
            value=1.0,
            window_len=10,
            cap_in=500.0,
        )
        for asset_id in (stable_id, voucher_id, target_voucher_id):
            current = producer_pool.vault.get(asset_id)
            if current > 0.0:
                engine._vault_sub(producer_pool, asset_id, current, "test_clear", "test")
        engine._vault_add(producer_pool, voucher_id, 25.0, "test_seed", "test")
        engine._vault_add(lender_pool, stable_id, 100.0, "test_seed", "test")
        engine._vault_add(lender_pool, target_voucher_id, 100.0, "test_seed", "test")
        self.assertTrue(
            engine._ordinary_own_voucher_stable_target_blocked_for_attempt(
                producer_pool,
                voucher_id,
                "ordinary",
            )
        )
        engine._sticky_target_by_pool[(producer_pool.pool_id, voucher_id)] = target_voucher_id
        engine._sticky_plan_by_pool[(producer_pool.pool_id, voucher_id, target_voucher_id)] = RoutePlan(
            ok=True,
            reason="own_voucher_to_voucher",
            hops=[Hop(lender_pool.pool_id, voucher_id, target_voucher_id, 10.0)],
        )

        attempted = engine._random_route_request(
            source_pool=producer_pool,
            max_assets=1,
            activity_mode="repeat_partner",
        )

        self.assertEqual(attempted, 1)
        self.assertEqual(engine._loan_route_motif_count_total.get("voucher_to_stable", 0), 0)
        self.assertEqual(engine._loan_route_motif_count_total.get("voucher_to_voucher"), 1)
        self.assertEqual(engine._market_route_motif_count_total.get("voucher_to_voucher", 0), 0)
        self.assertEqual(len(engine._producer_debt_obligations), 1)
        self.assertEqual(engine._producer_debt_obligations[0].debt_kind, "voucher")

    def test_private_wallet_roles_do_not_create_consumer_vouchers(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=3,
                initial_producers=2,
                initial_consumers=2,
                initial_liquidity_providers=0,
                max_pools=7,
                producer_voucher_single_lender=False,
                producer_voucher_overlap_mode="empirical_overlap",
                producer_voucher_overlap_bucket_weights={"2": 1.0},
            ),
            seed=17,
        )
        stable_id = engine.cfg.stable_symbol
        producer_agents = [agent for agent in engine.agents.values() if agent.role == "producer"]
        consumer_agents = [agent for agent in engine.agents.values() if agent.role == "consumer"]

        for agent in producer_agents:
            voucher_id = agent.voucher_spec.voucher_id
            pool = engine.pools[agent.pool_id]
            self.assertIn(voucher_id, engine.factory.voucher_specs)
            self.assertTrue(pool.registry.is_listed(stable_id))
            self.assertTrue(pool.registry.is_listed(voucher_id))
            self.assertEqual(
                {
                    asset_id
                    for asset_id, policy in pool.registry.listings.items()
                    if policy.enabled
                },
                {stable_id, voucher_id},
            )
            self.assertGreater(pool.vault.get(voucher_id), 0.0)
            self.assertEqual(len(engine._producer_voucher_lender_assignments[voucher_id]), 2)

        for agent in consumer_agents:
            voucher_id = agent.voucher_spec.voucher_id
            pool = engine.pools[agent.pool_id]
            self.assertNotIn(voucher_id, engine.factory.voucher_specs)
            self.assertNotIn(voucher_id, engine.factory.asset_universe)
            self.assertTrue(pool.registry.is_listed(stable_id))
            self.assertEqual(
                {
                    asset_id
                    for asset_id, policy in pool.registry.listings.items()
                    if policy.enabled
                },
                {stable_id},
            )
            self.assertFalse(
                any(asset_id.startswith("VCHR:") for asset_id in pool.vault.inventory)
            )
            self.assertAlmostEqual(agent.issuer.issued_total, 0.0)

    def test_private_wallets_are_not_routable_or_noam_venues(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=3,
                initial_producers=2,
                initial_consumers=2,
                initial_liquidity_providers=0,
                max_pools=7,
                producer_voucher_single_lender=False,
                producer_voucher_overlap_mode="empirical_overlap",
                producer_voucher_overlap_bucket_weights={"2": 1.0},
            ),
            seed=23,
        )
        private_pool_ids = {
            pool.pool_id
            for pool in engine.pools.values()
            if pool.policy.role in ("producer", "consumer")
        }
        self.assertTrue(private_pool_ids)

        for index in (engine.accept_index, engine.offer_index):
            for pool_ids in index.values():
                self.assertTrue(private_pool_ids.isdisjoint(pool_ids))
                self.assertTrue(
                    all(engine.pools[pool_id].policy.role == "lender" for pool_id in pool_ids)
                )

        engine._refresh_noam_working_set()
        for pool_ids in engine._noam_top_pools.values():
            self.assertTrue(private_pool_ids.isdisjoint(pool_ids))
            self.assertTrue(
                all(engine.pools[pool_id].policy.role == "lender" for pool_id in pool_ids)
            )
        for pool_id, _asset_id in engine._noam_top_out:
            self.assertNotIn(pool_id, private_pool_ids)
            self.assertEqual(engine.pools[pool_id].policy.role, "lender")

    def test_open_pool_direct_voucher_to_voucher_edges_are_config_gated(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=2,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=3,
            )
        )
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        producers = [agent for agent in engine.agents.values() if agent.role == "producer"]
        voucher_in = producers[0].voucher_spec.voucher_id
        voucher_out = producers[1].voucher_spec.voucher_id
        for voucher_id in (voucher_in, voucher_out):
            lender_pool.list_asset_with_value_and_limit(
                voucher_id,
                value=1.0,
                window_len=10,
                cap_in=500.0,
            )
            lender_pool.values.set_value(voucher_id, 1.0)
        engine._vault_add(lender_pool, voucher_out, 50.0, "test_seed", "test")
        engine.tick = 1

        self.assertFalse(engine._noam_edge_allowed(lender_pool, voucher_in, voucher_out, 1.0))

        engine.cfg.open_pool_direct_voucher_to_voucher_enabled = True
        self.assertTrue(engine._noam_edge_allowed(lender_pool, voucher_in, voucher_out, 1.0))
        private_pool = engine.pools[producers[0].pool_id]
        self.assertFalse(engine._is_routable_pool(private_pool))

    def test_route_motif_counts_one_economic_route_not_each_hop(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=3,
                initial_producers=2,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=5,
                open_pool_direct_voucher_to_voucher_enabled=True,
                producer_voucher_single_lender=False,
            )
        )
        stable_id = engine.cfg.stable_symbol
        lenders = [pool for pool in engine.pools.values() if pool.policy.role == "lender"]
        producers = [agent for agent in engine.agents.values() if agent.role == "producer"]
        source_agent, target_agent = producers[0], producers[1]
        source_pool = lenders[2]
        voucher_in = source_agent.voucher_spec.voucher_id
        voucher_out = target_agent.voucher_spec.voucher_id
        for pool in [source_pool, *lenders[:2], engine.pools[target_agent.pool_id]]:
            for asset_id in (voucher_in, voucher_out, stable_id):
                pool.values.set_value(asset_id, 1.0)
        for lender in lenders:
            lender.list_asset_with_value_and_limit(voucher_in, value=1.0, window_len=10, cap_in=500.0)
            lender.list_asset_with_value_and_limit(voucher_out, value=1.0, window_len=10, cap_in=500.0)
            lender.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
        for pool, asset_id in ((source_pool, voucher_in), (lenders[0], stable_id), (lenders[1], voucher_out)):
            current = pool.vault.get(asset_id)
            if current > 0.0:
                engine._vault_sub(pool, asset_id, current, "test_clear", "test")
        engine._vault_add(source_pool, voucher_in, 20.0, "test_seed", "test")
        engine._vault_add(lenders[0], stable_id, 20.0, "test_seed", "test")
        engine._vault_add(lenders[1], voucher_out, 20.0, "test_seed", "test")
        plan = RoutePlan(
            ok=True,
            reason="test",
            hops=[
                Hop(lenders[0].pool_id, voucher_in, stable_id, 10.0),
                Hop(lenders[1].pool_id, stable_id, voucher_out, 10.0),
            ],
        )

        ok = engine.execute_route_from_pool(source_pool.pool_id, plan, 10.0)

        self.assertTrue(ok)
        self.assertEqual(engine._route_motif_count_total.get("voucher_to_voucher"), 1)
        self.assertEqual(engine._ordinary_route_motif_count_total.get("voucher_to_voucher"), 1)
        self.assertEqual(engine._market_route_motif_count_total.get("voucher_to_voucher"), 1)
        self.assertEqual(engine._route_motif_stable_intermediate_count_total, 1)
        self.assertEqual(engine._route_context_count_tick.get("ordinary"), 2)
        engine.snapshot_metrics(force_network=True)
        latest = engine.metrics.network_rows[-1]
        self.assertEqual(
            latest["observed_route_motif_voucher_to_voucher_count_total"],
            latest["route_motif_voucher_to_voucher_count_total"],
        )
        self.assertAlmostEqual(
            latest["observed_route_motif_voucher_to_voucher_share_total"],
            latest["route_motif_voucher_to_voucher_share_total"],
        )
        self.assertEqual(
            latest["observed_route_motif_stable_intermediate_count_total"],
            latest["route_motif_stable_intermediate_count_total"],
        )

    def test_repayment_and_loan_route_motifs_are_separate_from_market_motifs(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=2,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=3,
                voucher_settlement_mode="legacy",
                pool_fee_rate=0.0,
                clc_rake_rate=0.0,
            ),
            seed=27,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_a, lender_b = [pool for pool in engine.pools.values() if pool.policy.role == "lender"]
        voucher_id = producer.voucher_spec.voucher_id
        for pool in (producer_pool, lender_a, lender_b):
            pool.values.set_value(stable_id, 1.0)
            pool.values.set_value(voucher_id, 1.0)
        for lender in (lender_a, lender_b):
            lender.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
            lender.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        for pool, asset_id in (
            (producer_pool, stable_id),
            (producer_pool, voucher_id),
            (lender_a, voucher_id),
            (lender_b, stable_id),
        ):
            current = pool.vault.get(asset_id)
            if current > 0.0:
                engine._vault_sub(pool, asset_id, current, "test_clear", "test")
        engine._vault_add(producer_pool, stable_id, 20.0, "test_seed", "test")
        engine._vault_add(lender_a, voucher_id, 100.0, "test_seed", "test")
        repayment_plan = RoutePlan(
            ok=True,
            reason="repayment",
            hops=[Hop(lender_a.pool_id, stable_id, voucher_id, 10.0)],
        )

        ok = engine.execute_route_from_pool(
            producer_pool.pool_id,
            repayment_plan,
            10.0,
            route_context="repayment",
        )

        self.assertTrue(ok)
        self.assertEqual(engine._route_motif_count_total.get("stable_to_voucher"), 1)
        self.assertEqual(engine._repayment_route_motif_count_total.get("stable_to_voucher"), 1)
        self.assertEqual(engine._market_route_motif_count_total.get("stable_to_voucher", 0), 0)

        engine._vault_add(producer_pool, voucher_id, 20.0, "test_seed", "test")
        engine._vault_add(lender_b, stable_id, 100.0, "test_seed", "test")
        loan_plan = RoutePlan(
            ok=True,
            reason="loan",
            hops=[Hop(lender_b.pool_id, voucher_id, stable_id, 10.0)],
        )

        ok = engine.execute_route_from_pool(
            producer_pool.pool_id,
            loan_plan,
            10.0,
            route_context="loan",
        )

        self.assertTrue(ok)
        self.assertEqual(engine._route_motif_count_total.get("voucher_to_stable"), 1)
        self.assertEqual(engine._loan_route_motif_count_total.get("voucher_to_stable"), 1)
        self.assertEqual(engine._market_route_motif_count_total.get("voucher_to_stable", 0), 0)
        engine.snapshot_metrics(force_network=True)
        latest = engine.metrics.network_rows[-1]
        self.assertEqual(latest["observed_route_motif_stable_to_voucher_count_total"], 1)
        self.assertEqual(latest["repayment_route_motif_stable_to_voucher_count_total"], 1)
        self.assertEqual(latest["observed_route_motif_voucher_to_stable_count_total"], 1)
        self.assertEqual(latest["loan_route_motif_voucher_to_stable_count_total"], 1)
        self.assertEqual(latest["market_route_motif_stable_to_voucher_count_total"], 0)
        self.assertEqual(latest["market_route_motif_voucher_to_stable_count_total"], 0)

    def test_engine_validation_binds_ledger_observed_route_motifs(self):
        calibration = load_calibration(Path("analysis/sarafu_calibration"))
        circulation = calibration.voucher_circulation_baselines["trailing_90d"]
        observed_v2v = float(circulation["voucher_to_voucher_share"])
        observed_v2s = float(circulation["voucher_to_stable_share"])
        observed_s2v = float(circulation["stable_to_voucher_share"])
        summary = {
            "observed_route_motif_voucher_to_voucher_share_total": observed_v2v,
            "observed_route_motif_voucher_to_stable_share_total": observed_v2s,
            "observed_route_motif_stable_to_voucher_share_total": observed_s2v,
            "observed_route_motif_stable_involved_share_total": observed_v2s + observed_s2v,
            "market_route_motif_voucher_to_voucher_share_total": 0.0,
            "market_route_motif_voucher_to_stable_share_total": 1.0,
            "market_route_motif_stable_to_voucher_share_total": 0.0,
            "market_route_motif_stable_involved_share_total": 1.0,
        }

        rows = engine_validation_moments(calibration, [summary], ticks=52)
        by_moment = {(row["tier"], row["moment"]): row for row in rows}

        for moment in (
            "observed_route_motif_voucher_to_voucher_share",
            "observed_route_motif_voucher_to_stable_share",
            "observed_route_motif_stable_to_voucher_share",
            "observed_route_motif_stable_involved_share",
        ):
            row = by_moment[("all", moment)]
            self.assertEqual(row["binding"], 1)
            self.assertEqual(row["validation_status"], "pass")

        for moment in (
            "market_route_motif_voucher_to_voucher_share",
            "market_route_motif_voucher_to_stable_share",
            "market_route_motif_stable_to_voucher_share",
            "market_route_motif_stable_involved_share",
        ):
            row = by_moment[("all", moment)]
            self.assertEqual(row["binding"], 0)
            self.assertEqual(row["category"], "purpose_diagnostic")
            self.assertEqual(row["validation_status"], "reported")

        fallback_summary = {
            "route_motif_voucher_to_voucher_share_total": observed_v2v,
            "route_motif_voucher_to_stable_share_total": observed_v2s,
            "route_motif_stable_to_voucher_share_total": observed_s2v,
            "route_motif_stable_involved_share_total": observed_v2s + observed_s2v,
        }
        fallback_rows = engine_validation_moments(calibration, [fallback_summary], ticks=52)
        fallback_by_moment = {(row["tier"], row["moment"]): row for row in fallback_rows}
        self.assertEqual(
            fallback_by_moment[
                ("all", "observed_route_motif_voucher_to_stable_share")
            ]["validation_status"],
            "pass",
        )

    def test_frontier_validation_gate_uses_sim_local_summary_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "frontier_output"
            self.assertEqual(frontier_validation_gate_status(output_dir), "pass")

            sibling_validation = output_dir.parent / "engine_validation"
            sibling_validation.mkdir()
            with (sibling_validation / "engine_validation_summary.csv").open("w", encoding="utf-8") as handle:
                handle.write(
                    "scenario,runs,ticks,status,binding_pass_count,binding_review_count,binding_fail_count,reported_diagnostic_count\n"
                )
                handle.write("sarafu_engine_validation,1,1,fail,0,0,1,0\n")

            self.assertEqual(frontier_validation_gate_status(output_dir), "fail")

    def test_redeem_outputs_mode_redeems_final_lender_voucher_output(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=2,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=3,
                voucher_settlement_mode="redeem_outputs",
                pool_fee_rate=0.02,
                clc_rake_rate=1.0,
            ),
            seed=31,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        voucher_id = producer.voucher_spec.voucher_id
        source_pool, swap_pool = [
            pool for pool in engine.pools.values() if pool.policy.role == "lender"
        ]
        for pool in (source_pool, swap_pool, producer_pool):
            pool.values.set_value(stable_id, 1.0)
            pool.values.set_value(voucher_id, 1.0)
        for pool in (source_pool, swap_pool):
            pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
            pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        for pool, asset_id in (
            (source_pool, stable_id),
            (source_pool, voucher_id),
            (swap_pool, voucher_id),
            (producer_pool, voucher_id),
        ):
            current = pool.vault.get(asset_id)
            if current > 0.0:
                engine._vault_sub(pool, asset_id, current, "test_clear", "test")
        engine._vault_add(source_pool, stable_id, 20.0, "test_seed", "test")
        engine._vault_add(swap_pool, voucher_id, 100.0, "test_seed", "test")
        plan = RoutePlan(
            ok=True,
            reason="test",
            hops=[Hop(swap_pool.pool_id, stable_id, voucher_id, 10.0)],
        )

        ok = engine.execute_route_from_pool(source_pool.pool_id, plan, 10.0)

        self.assertTrue(ok)
        self.assertAlmostEqual(source_pool.vault.get(voucher_id), 0.0)
        self.assertAlmostEqual(producer_pool.vault.get(voucher_id), 9.8)
        self.assertAlmostEqual(producer.issuer.issuer_returned_total, 9.8)
        self.assertAlmostEqual(swap_pool.vault.get(voucher_id), 90.0)
        self.assertAlmostEqual(swap_pool.fee_ledger_pool.get(voucher_id, 0.0), 0.2)
        self.assertAlmostEqual(swap_pool.fee_ledger_clc.get(voucher_id, 0.0), 0.2)
        self.assertAlmostEqual(engine._net_redeemed_voucher_usd_total, 9.8)
        self.assertAlmostEqual(engine._voucher_fee_retained_for_service_usd_total, 0.2)
        self.assertAlmostEqual(engine._debt_removal_voucher_redeemed_usd_total, 9.8)

    def test_legacy_mode_keeps_final_lender_voucher_output_in_source_pool(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=2,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=3,
                voucher_settlement_mode="legacy",
                pool_fee_rate=0.02,
                clc_rake_rate=1.0,
            ),
            seed=37,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        voucher_id = producer.voucher_spec.voucher_id
        source_pool, swap_pool = [
            pool for pool in engine.pools.values() if pool.policy.role == "lender"
        ]
        for pool in (source_pool, swap_pool, producer_pool):
            pool.values.set_value(stable_id, 1.0)
            pool.values.set_value(voucher_id, 1.0)
        for pool in (source_pool, swap_pool):
            pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
            pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        for pool, asset_id in (
            (source_pool, stable_id),
            (source_pool, voucher_id),
            (swap_pool, voucher_id),
            (producer_pool, voucher_id),
        ):
            current = pool.vault.get(asset_id)
            if current > 0.0:
                engine._vault_sub(pool, asset_id, current, "test_clear", "test")
        engine._vault_add(source_pool, stable_id, 20.0, "test_seed", "test")
        engine._vault_add(swap_pool, voucher_id, 100.0, "test_seed", "test")
        plan = RoutePlan(
            ok=True,
            reason="test",
            hops=[Hop(swap_pool.pool_id, stable_id, voucher_id, 10.0)],
        )

        ok = engine.execute_route_from_pool(source_pool.pool_id, plan, 10.0)

        self.assertTrue(ok)
        self.assertAlmostEqual(source_pool.vault.get(voucher_id), 9.8)
        self.assertAlmostEqual(producer_pool.vault.get(voucher_id), 0.0)
        self.assertAlmostEqual(producer.issuer.issuer_returned_total, 0.0)
        self.assertAlmostEqual(engine._net_redeemed_voucher_usd_total, 0.0)
        self.assertAlmostEqual(engine._voucher_fee_retained_for_service_usd_total, 0.2)

    def test_producer_deposit_reintroduces_returned_vouchers_before_new_issuance(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
            ),
            seed=41,
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for pool in (producer_pool, lender_pool):
            pool.values.set_value(voucher_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        for pool in (producer_pool, lender_pool):
            current = pool.vault.get(voucher_id)
            if current > 0.0:
                engine._vault_sub(pool, voucher_id, current, "test_clear", "test")
        engine._vault_add(producer_pool, voucher_id, 8.0, "redeem_receive", "test")
        issued_before = producer.issuer.issued_total

        deposited = engine._deposit_producer_voucher_with_lenders(
            producer_pool=producer_pool,
            agent=producer,
            voucher_id=voucher_id,
            voucher_value_usd=10.0,
            source="test_deposit",
        )

        self.assertTrue(deposited)
        self.assertAlmostEqual(producer_pool.vault.get(voucher_id), 0.0)
        self.assertAlmostEqual(lender_pool.vault.get(voucher_id), 10.0)
        self.assertAlmostEqual(producer.issuer.issued_total, issued_before + 2.0)
        self.assertAlmostEqual(engine._voucher_reintroduced_by_deposit_usd_total, 8.0)
        self.assertAlmostEqual(engine._voucher_new_issuance_deposit_usd_total, 2.0)

    def test_producer_stable_exit_credits_debt_service_capacity_only_with_active_debt(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                producer_debt_pressure_enabled=True,
                producer_stable_exit_share=1.0,
                producer_debt_maturity_enabled=True,
                producer_debt_contract_repayment_enabled=True,
            ),
            seed=43,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        lender_pool.values.set_value(voucher_id, 1.0)
        engine._vault_add(producer_pool, stable_id, 20.0, "test_seed", "test")

        exited = engine._apply_producer_stable_exit(producer_pool, 10.0, "no_debt")

        self.assertAlmostEqual(exited, 10.0)
        self.assertAlmostEqual(engine._producer_stable_exited_usd_total, 10.0)
        self.assertAlmostEqual(engine._producer_debt_service_capacity_balance_usd(), 0.0)

        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            10.0,
            10.0,
        )
        engine._vault_add(producer_pool, stable_id, 10.0, "test_seed", "test")

        exited = engine._apply_producer_stable_exit(producer_pool, 10.0, "with_debt")

        self.assertAlmostEqual(exited, 10.0)
        self.assertAlmostEqual(engine._producer_stable_exited_usd_total, 20.0)
        self.assertAlmostEqual(engine._producer_debt_service_capacity_balance_usd(), 10.0)
        self.assertAlmostEqual(engine._producer_debt_service_capacity_credited_usd_total, 10.0)

    def test_debt_attention_crowdout_disabled_suppresses_nothing(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                producer_debt_pressure_enabled=True,
                producer_debt_attention_crowdout_enabled=False,
                producer_debt_maturity_enabled=True,
                producer_debt_maturity_ticks=13,
            ),
            seed=44,
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
            contract_cash_service=False,
        )

        suppressed = engine._apply_producer_debt_attention_crowdout(producer_pool, 4)

        self.assertEqual(suppressed, 0)
        self.assertEqual(engine._producer_debt_attention_suppressed_attempts_total, 0)

    def test_debt_attention_pressure_increases_with_arrears_and_deferred_due(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                producer_debt_pressure_enabled=True,
                producer_debt_attention_crowdout_enabled=True,
                producer_debt_attention_reference_usd=100.0,
                producer_debt_maturity_enabled=True,
                producer_debt_maturity_ticks=12,
                producer_debt_pressure_period_ticks=4,
            ),
            seed=45,
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            120.0,
            120.0,
            contract_cash_service=False,
        )
        base_share, base_pressure, _reference = engine._producer_debt_attention_share(producer_pool)
        obligation = engine._producer_debt_obligations[0]
        obligation.cash_service_arrears_usd = 20.0
        obligation.pressure_deferred_usd = 10.0

        stressed_share, stressed_pressure, _reference = engine._producer_debt_attention_share(producer_pool)

        self.assertGreater(stressed_pressure, base_pressure)
        self.assertGreater(stressed_share, base_share)

    def test_debt_attention_crowdout_caps_and_does_not_record_route_failure(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                producer_debt_pressure_enabled=True,
                producer_debt_attention_crowdout_enabled=True,
                producer_debt_attention_crowdout_max_share=0.50,
                producer_debt_attention_reference_usd=1.0,
                producer_debt_maturity_enabled=True,
                producer_debt_maturity_ticks=4,
                producer_debt_pressure_period_ticks=4,
            ),
            seed=46,
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
            contract_cash_service=False,
        )
        before_failures = sum(1 for event in engine.log.events if event.event_type == "ROUTE_FAILED")

        suppressed = engine._apply_producer_debt_attention_crowdout(producer_pool, 4)
        share, _pressure, _reference = engine._producer_debt_attention_share(producer_pool)
        after_failures = sum(1 for event in engine.log.events if event.event_type == "ROUTE_FAILED")

        self.assertAlmostEqual(share, 0.50)
        self.assertEqual(suppressed, 2)
        self.assertEqual(engine._producer_debt_attention_suppressed_attempts_total, 2)
        self.assertEqual(engine._producer_debt_attention_suppressed_v2v_attempts_total, 2)
        self.assertEqual(before_failures, after_failures)

    def test_bond_assessment_pressure_is_zero_without_bond_principal(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                producer_debt_pressure_enabled=True,
                producer_debt_attention_crowdout_enabled=True,
                producer_bond_assessment_pressure_enabled=True,
                producer_debt_maturity_enabled=True,
                producer_debt_maturity_ticks=13,
                bond_gross_principal_usd=0.0,
            ),
            seed=52,
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
            contract_cash_service=False,
        )
        engine.tick = 1

        self.assertAlmostEqual(engine._producer_bond_assessment_pressure_usd(producer_pool), 0.0)

    def test_bond_assessment_pressure_scales_with_coupon_and_principal(self):
        def pressure_for(*, principal: float, coupon: float) -> float:
            engine = SimulationEngine(
                small_config(
                    initial_lenders=1,
                    initial_producers=1,
                    initial_consumers=0,
                    initial_liquidity_providers=0,
                    max_pools=2,
                    producer_debt_pressure_enabled=True,
                    producer_debt_attention_crowdout_enabled=True,
                    producer_bond_assessment_pressure_enabled=True,
                    producer_debt_maturity_enabled=True,
                    producer_debt_maturity_ticks=13,
                    bond_gross_principal_usd=principal,
                    bond_term_ticks=260,
                    issuer_payment_stride_ticks=13,
                    bond_coupon_target_annual=coupon,
                ),
                seed=53,
            )
            producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
            producer_pool = engine.pools[producer.pool_id]
            lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
            voucher_id = producer.voucher_spec.voucher_id
            engine._register_producer_debt_obligation(
                producer_pool.pool_id,
                lender_pool.pool_id,
                voucher_id,
                100.0,
                100.0,
                contract_cash_service=False,
            )
            engine.tick = 1
            return engine._producer_bond_assessment_pressure_usd(producer_pool)

        low = pressure_for(principal=100.0, coupon=0.0)
        high_coupon = pressure_for(principal=100.0, coupon=0.20)
        high_principal = pressure_for(principal=200.0, coupon=0.0)

        self.assertGreater(low, 0.0)
        self.assertGreater(high_coupon, low)
        self.assertGreater(high_principal, low)

    def test_bond_assessment_pressure_adds_to_attention_share(self):
        common = dict(
            initial_lenders=1,
            initial_producers=1,
            initial_consumers=0,
            initial_liquidity_providers=0,
            max_pools=2,
            producer_debt_pressure_enabled=True,
            producer_debt_attention_crowdout_enabled=True,
            producer_debt_attention_reference_usd=100.0,
            producer_debt_maturity_enabled=True,
            producer_debt_maturity_ticks=13,
            bond_gross_principal_usd=1000.0,
            bond_term_ticks=260,
            issuer_payment_stride_ticks=13,
            bond_coupon_target_annual=0.20,
        )
        engine_without = SimulationEngine(
            small_config(**common, producer_bond_assessment_pressure_enabled=False),
            seed=54,
        )
        engine_with = SimulationEngine(
            small_config(**common, producer_bond_assessment_pressure_enabled=True),
            seed=54,
        )

        def setup(engine: SimulationEngine):
            producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
            producer_pool = engine.pools[producer.pool_id]
            lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
            voucher_id = producer.voucher_spec.voucher_id
            engine._register_producer_debt_obligation(
                producer_pool.pool_id,
                lender_pool.pool_id,
                voucher_id,
                100.0,
                100.0,
                contract_cash_service=False,
            )
            engine.tick = 1
            return producer_pool

        pool_without = setup(engine_without)
        pool_with = setup(engine_with)
        share_without, pressure_without, _reference = engine_without._producer_debt_attention_share(pool_without)
        share_with, pressure_with, _reference = engine_with._producer_debt_attention_share(pool_with)

        self.assertGreater(pressure_with, pressure_without)
        self.assertGreater(share_with, share_without)

    def test_bond_assessment_sustain_offset_reduces_only_sustain_target(self):
        def sustain_calls(offset_enabled: bool) -> tuple[int, int]:
            engine = SimulationEngine(
                small_config(
                    initial_lenders=1,
                    initial_producers=1,
                    initial_consumers=0,
                    initial_liquidity_providers=0,
                    max_pools=2,
                    swap_sustain_enabled=True,
                    swap_sustain_window_ticks=1,
                    swap_sustain_floor_per_tick=0,
                    swap_sustain_attempts_per_missing_swap=1.0,
                    swap_sustain_max_extra_attempts=0,
                    swap_sustain_max_rounds=20,
                    producer_bond_assessment_sustain_offset_enabled=offset_enabled,
                ),
                seed=55,
            )
            engine._recent_swap_counts = [10]
            engine._producer_bond_assessment_sustain_offset_attempts_tick = 3.2
            calls = 0

            def fake_random_route_request(source_pool=None, max_assets=None, route_context="ordinary"):
                nonlocal calls
                calls += 1
                engine._noam_routing_swaps_tick += 1
                return 1

            engine._random_route_request = fake_random_route_request
            engine._sustain_swap_activity()
            return calls, engine._producer_bond_assessment_sustain_target_reduction_tick

        calls_without_offset, reduction_without_offset = sustain_calls(False)
        calls_with_offset, reduction_with_offset = sustain_calls(True)

        self.assertEqual(calls_without_offset, 10)
        self.assertEqual(reduction_without_offset, 0)
        self.assertEqual(calls_with_offset, 7)
        self.assertEqual(reduction_with_offset, 3)

    def test_debt_pressure_repayment_draws_capacity_for_own_voucher_swap(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                producer_debt_pressure_enabled=True,
                producer_debt_pressure_period_ticks=1,
                producer_debt_pressure_prepay_share=0.0,
                producer_debt_maturity_enabled=True,
                producer_debt_contract_repayment_enabled=True,
                producer_debt_contract_service_margin_rate=0.0,
                producer_debt_maturity_ticks=1,
                producer_debt_penalty_enabled=False,
                pool_fee_rate=0.02,
                clc_rake_rate=1.0,
            ),
            seed=47,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for pool in (producer_pool, lender_pool):
            pool.values.set_value(stable_id, 1.0)
            pool.values.set_value(voucher_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        for pool, asset_id in (
            (producer_pool, stable_id),
            (producer_pool, voucher_id),
            (lender_pool, voucher_id),
        ):
            current = pool.vault.get(asset_id)
            if current > 0.0:
                engine._vault_sub(pool, asset_id, current, "test_clear", "test")
        engine._vault_add(lender_pool, voucher_id, 100.0, "test_seed", "test")
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
        )
        obligation = engine._producer_debt_obligations[0]
        engine._producer_debt_service_capacity_by_pool[producer_pool.pool_id] = 25.0
        engine.tick = 1

        attempted = engine._attempt_repayment(producer_pool)

        self.assertTrue(attempted)
        self.assertGreater(engine._producer_self_repayment_swap_volume_usd_total, 0.0)
        self.assertAlmostEqual(
            engine._producer_debt_service_capacity_onramp_usd_total,
            engine._producer_self_repayment_swap_volume_usd_total,
        )
        self.assertAlmostEqual(producer_pool.vault.get(voucher_id), 0.0)
        self.assertLess(lender_pool.vault.get(voucher_id), 100.0)
        self.assertLess(obligation.cash_service_remaining_usd, 100.0)
        self.assertLess(obligation.remaining_voucher_units, 100.0)
        self.assertAlmostEqual(engine._lender_recovered_stable_borrower_self_usd_total, 25.0)
        self.assertGreater(engine._voucher_fee_retained_for_service_usd_total, 0.0)
        self.assertGreater(engine._repayment_route_motif_count_total.get("stable_to_voucher", 0), 0)
        self.assertEqual(engine._market_route_motif_count_total.get("stable_to_voucher", 0), 0)

    def test_debt_pressure_penalty_accrues_and_is_paid_before_prepay(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                producer_debt_pressure_enabled=True,
                producer_debt_pressure_period_ticks=1,
                producer_debt_pressure_prepay_share=0.0,
                producer_debt_maturity_enabled=True,
                producer_debt_contract_repayment_enabled=True,
                producer_debt_contract_service_margin_rate=0.0,
                producer_debt_maturity_ticks=1,
                producer_debt_penalty_enabled=True,
                producer_debt_penalty_rate_per_period=0.10,
                pool_fee_rate=0.0,
                clc_rake_rate=0.0,
            ),
            seed=53,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for pool in (producer_pool, lender_pool):
            pool.values.set_value(stable_id, 1.0)
            pool.values.set_value(voucher_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        current = lender_pool.vault.get(voucher_id)
        if current > 0.0:
            engine._vault_sub(lender_pool, voucher_id, current, "test_clear", "test")
        engine._vault_add(lender_pool, voucher_id, 100.0, "test_seed", "test")
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
        )
        obligation = engine._producer_debt_obligations[0]
        engine.tick = 1

        attempted = engine._attempt_repayment(producer_pool)

        self.assertTrue(attempted)
        self.assertAlmostEqual(engine._producer_debt_penalty_accrued_usd_total, 10.0)
        self.assertAlmostEqual(obligation.cash_service_arrears_usd, 110.0)
        self.assertAlmostEqual(obligation.cash_service_penalty_remaining_usd, 10.0)

        engine._producer_debt_service_capacity_by_pool[producer_pool.pool_id] = 50.0
        engine.tick = 2
        attempted = engine._attempt_repayment(producer_pool)

        self.assertTrue(attempted)
        self.assertAlmostEqual(engine._producer_debt_penalty_paid_usd_total, 10.0)
        self.assertGreater(obligation.cash_service_arrears_usd, 0.0)
        self.assertAlmostEqual(engine._producer_debt_pressure_prepayment_usd_total, 0.0)

    def test_batched_debt_pressure_defers_below_minimum_without_penalty(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                producer_debt_pressure_enabled=True,
                producer_debt_pressure_batching_enabled=True,
                producer_debt_pressure_min_swap_usd=5.0,
                producer_debt_pressure_period_ticks=4,
                producer_debt_pressure_prepay_share=0.0,
                producer_debt_maturity_enabled=True,
                producer_debt_maturity_ticks=100,
                producer_debt_contract_repayment_enabled=True,
                producer_debt_contract_service_margin_rate=0.0,
                producer_debt_penalty_enabled=True,
                producer_debt_penalty_rate_per_period=0.10,
                pool_fee_rate=0.0,
                clc_rake_rate=0.0,
            ),
            seed=55,
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for pool in (producer_pool, lender_pool):
            pool.values.set_value(engine.cfg.stable_symbol, 1.0)
            pool.values.set_value(voucher_id, 1.0)
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            4.0,
            4.0,
        )
        engine.tick = 1

        attempted = engine._attempt_debt_pressure_repayment(producer_pool, voucher_id, 4)

        self.assertTrue(attempted)
        self.assertEqual(engine._repayment_route_motif_count_total.get("stable_to_voucher", 0), 0)
        self.assertAlmostEqual(engine._producer_debt_pressure_batched_swap_count_total, 0)
        self.assertGreater(engine._producer_debt_pressure_deferred_usd_total, 0.0)
        self.assertGreater(engine._producer_debt_pressure_deferred_balance_usd(), 0.0)
        self.assertAlmostEqual(engine._producer_debt_penalty_accrued_usd_total, 0.0)

    def test_batched_debt_pressure_settles_multiple_obligations_as_one_route(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                producer_debt_pressure_enabled=True,
                producer_debt_pressure_batching_enabled=True,
                producer_debt_pressure_min_swap_usd=1.0,
                producer_debt_pressure_period_ticks=1,
                producer_debt_pressure_prepay_share=0.0,
                producer_debt_maturity_enabled=True,
                producer_debt_maturity_ticks=1,
                producer_debt_contract_repayment_enabled=True,
                producer_debt_contract_service_margin_rate=0.0,
                producer_debt_penalty_enabled=True,
                producer_debt_penalty_rate_per_period=0.10,
                pool_fee_rate=0.0,
                clc_rake_rate=0.0,
            ),
            seed=56,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for pool in (producer_pool, lender_pool):
            pool.values.set_value(stable_id, 1.0)
            pool.values.set_value(voucher_id, 1.0)
            pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
            pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        current = lender_pool.vault.get(voucher_id)
        if current > 0.0:
            engine._vault_sub(lender_pool, voucher_id, current, "test_clear", "test")
        engine._vault_add(lender_pool, voucher_id, 40.0, "test_seed", "test")
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            10.0,
            10.0,
        )
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            10.0,
            10.0,
        )
        engine._producer_debt_service_capacity_by_pool[producer_pool.pool_id] = 25.0
        engine.tick = 1

        attempted = engine._attempt_debt_pressure_repayment(producer_pool, voucher_id, 1)

        self.assertTrue(attempted)
        self.assertEqual(engine._repayment_route_motif_count_total.get("stable_to_voucher"), 1)
        self.assertEqual(engine._producer_debt_pressure_batched_swap_count_total, 1)
        self.assertAlmostEqual(engine._producer_debt_pressure_batched_swap_volume_usd_total, 20.0)
        self.assertAlmostEqual(engine._producer_self_repayment_swap_volume_usd_total, 20.0)
        self.assertAlmostEqual(engine._producer_debt_pressure_deferred_balance_usd(), 0.0)
        self.assertAlmostEqual(engine._producer_debt_penalty_accrued_usd_total, 0.0)

    def test_batched_debt_pressure_above_threshold_unpaid_accrues_penalty(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=2,
                producer_debt_pressure_enabled=True,
                producer_debt_pressure_batching_enabled=True,
                producer_debt_pressure_min_swap_usd=1.0,
                producer_debt_pressure_period_ticks=1,
                producer_debt_pressure_prepay_share=0.0,
                producer_debt_maturity_enabled=True,
                producer_debt_maturity_ticks=1,
                producer_debt_contract_repayment_enabled=True,
                producer_debt_contract_service_margin_rate=0.0,
                producer_debt_penalty_enabled=True,
                producer_debt_penalty_rate_per_period=0.10,
                pool_fee_rate=0.0,
                clc_rake_rate=0.0,
            ),
            seed=57,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for pool in (producer_pool, lender_pool):
            pool.values.set_value(stable_id, 1.0)
            pool.values.set_value(voucher_id, 1.0)
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            10.0,
            10.0,
        )
        obligation = engine._producer_debt_obligations[0]
        engine.tick = 1

        attempted = engine._attempt_debt_pressure_repayment(producer_pool, voucher_id, 1)

        self.assertTrue(attempted)
        self.assertEqual(engine._producer_debt_pressure_batched_swap_count_total, 0)
        self.assertAlmostEqual(engine._producer_debt_penalty_accrued_usd_total, 1.0)
        self.assertAlmostEqual(obligation.cash_service_arrears_usd, 11.0)
        self.assertAlmostEqual(obligation.pressure_deferred_usd, 0.0)

    def test_ordinary_own_voucher_for_other_voucher_creates_producer_debt(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=2,
                initial_consumers=0,
                initial_liquidity_providers=0,
                max_pools=3,
                producer_debt_maturity_enabled=True,
                producer_debt_pressure_enabled=True,
                pool_fee_rate=0.0,
                clc_rake_rate=0.0,
            ),
            seed=61,
        )
        producers = [agent for agent in engine.agents.values() if agent.role == "producer"]
        borrower = producers[0]
        target_issuer = producers[1]
        borrower_pool = engine.pools[borrower.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        borrower_voucher = borrower.voucher_spec.voucher_id
        target_voucher = target_issuer.voucher_spec.voucher_id
        for pool in (borrower_pool, lender_pool):
            pool.values.set_value(borrower_voucher, 1.0)
            pool.values.set_value(target_voucher, 1.0)
        lender_pool.list_asset_with_value_and_limit(
            borrower_voucher, value=1.0, window_len=10, cap_in=500.0
        )
        lender_pool.list_asset_with_value_and_limit(
            target_voucher, value=1.0, window_len=10, cap_in=500.0
        )
        for pool, asset_id in (
            (borrower_pool, borrower_voucher),
            (lender_pool, target_voucher),
        ):
            current = pool.vault.get(asset_id)
            if current > 0.0:
                engine._vault_sub(pool, asset_id, current, "test_clear", "test")
        engine._vault_add(borrower_pool, borrower_voucher, 20.0, "test_seed", "test")
        engine._vault_add(lender_pool, target_voucher, 20.0, "test_seed", "test")
        plan = RoutePlan(
            ok=True,
            reason="test",
            hops=[Hop(lender_pool.pool_id, borrower_voucher, target_voucher, 10.0)],
        )

        ok = engine.execute_route_from_pool(
            borrower_pool.pool_id,
            plan,
            10.0,
            route_context="ordinary",
        )

        self.assertTrue(ok)
        self.assertEqual(len(engine._producer_debt_obligations), 1)
        obligation = engine._producer_debt_obligations[0]
        self.assertEqual(obligation.producer_pool_id, borrower_pool.pool_id)
        self.assertEqual(obligation.lender_pool_id, lender_pool.pool_id)
        self.assertEqual(obligation.voucher_id, borrower_voucher)
        self.assertEqual(obligation.debt_kind, "voucher")
        self.assertAlmostEqual(obligation.borrowed_usd, 10.0)
        self.assertAlmostEqual(obligation.remaining_voucher_units, 10.0)
        self.assertEqual(engine._route_motif_count_total.get("voucher_to_voucher"), 1)
        self.assertEqual(engine._loan_route_motif_count_total.get("voucher_to_voucher"), 1)
        self.assertEqual(engine._market_route_motif_count_total.get("voucher_to_voucher", 0), 0)
        engine.snapshot_metrics(force_network=True)
        latest = engine.metrics.network_rows[-1]
        self.assertEqual(latest["observed_route_motif_voucher_to_voucher_count_total"], 1)
        self.assertEqual(latest["loan_route_motif_voucher_to_voucher_count_total"], 1)
        self.assertEqual(latest["market_route_motif_voucher_to_voucher_count_total"], 0)

    def test_execute_route_rejects_private_wallet_hop_and_refunds_source(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=1,
                initial_consumers=1,
                initial_liquidity_providers=0,
                max_pools=3,
            ),
            seed=29,
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        consumer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "consumer")
        voucher_id = producer.voucher_spec.voucher_id

        if consumer_pool.vault.get(stable_id) < 10.0:
            engine._vault_add(consumer_pool, stable_id, 10.0, "test_seed", "test")
        stable_before = consumer_pool.vault.get(stable_id)
        voucher_before = producer_pool.vault.get(voucher_id)

        plan = RoutePlan(
            ok=True,
            reason="test_private_wallet_hop",
            hops=[
                Hop(
                    pool_id=producer_pool.pool_id,
                    asset_in=stable_id,
                    asset_out=voucher_id,
                    amount_in=10.0,
                )
            ],
        )

        ok = engine.execute_route_from_pool(consumer_pool.pool_id, plan, 10.0)

        self.assertFalse(ok)
        self.assertAlmostEqual(consumer_pool.vault.get(stable_id), stable_before)
        self.assertAlmostEqual(producer_pool.vault.get(voucher_id), voucher_before)

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
        self.assertEqual(engine._producer_voucher_loan_attempts_tick, 0)
        self.assertAlmostEqual(engine._producer_loan_lender_remaining_cap_usd_tick, 5.0)
        self.assertGreater(engine._producer_loan_clipped_lender_remaining_usd_tick, 0.0)
        self.assertLessEqual(engine._producer_loan_attempted_usd_tick, 5.0 + 1e-9)

    def test_voucher_loan_fallback_runs_only_when_stable_unavailable(self):
        engine = SimulationEngine(
            small_config(
                initial_producers=2,
                max_pools=4,
                producer_deposits_enabled=True,
                producer_voucher_loan_fallback_enabled=True,
                producer_voucher_loan_activity_boost_enabled=True,
                productive_credit_voucher_activity_boost_window_ticks=4,
                producer_debt_maturity_enabled=True,
                producer_debt_contract_repayment_enabled=True,
                swap_size_min_usd=10.0,
                random_request_amount_mean=10.0,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producers = [agent for agent in engine.agents.values() if agent.role == "producer"]
        borrower, voucher_issuer = producers[0], producers[1]
        borrower_pool = engine.pools[borrower.pool_id]
        issuer_pool = engine.pools[voucher_issuer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        borrower_voucher = borrower.voucher_spec.voucher_id
        target_voucher = voucher_issuer.voucher_spec.voucher_id
        for pool in (borrower_pool, issuer_pool, lender_pool):
            pool.values.set_value(borrower_voucher, 1.0)
            pool.values.set_value(target_voucher, 1.0)
        lender_pool.list_asset_with_value_and_limit(borrower_voucher, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(target_voucher, value=1.0, window_len=10, cap_in=500.0)
        engine.accept_index.setdefault(borrower_voucher, set()).add(lender_pool.pool_id)
        engine.accept_index.setdefault(target_voucher, set()).add(lender_pool.pool_id)
        for pool, asset in (
            (borrower_pool, borrower_voucher),
            (lender_pool, target_voucher),
            (lender_pool, stable_id),
        ):
            current = pool.vault.get(asset)
            if current > 0.0:
                engine._vault_sub(pool, asset, current, "test_clear", "test")
        engine._vault_add(borrower_pool, borrower_voucher, 50.0, "test_seed", "test")
        engine._vault_add(lender_pool, target_voucher, 100.0, "test_seed", "test")
        lender_pool.policy.min_stable_reserve = 0.0
        engine._producer_deposit_value_by_voucher[borrower_voucher] = 100.0
        engine.tick = 1
        engine._noam_last_refresh_tick = -1

        attempted = engine._attempt_new_loan(borrower_pool, borrower_voucher)

        self.assertTrue(attempted)
        self.assertEqual(engine._producer_loan_no_lender_tick, 1)
        self.assertEqual(engine._producer_voucher_loan_attempts_tick, 1)
        self.assertEqual(engine._producer_voucher_loan_executed_tick, 1)
        self.assertEqual(engine._producer_loan_executed_tick, 0)
        self.assertGreater(engine._route_context_count_tick.get("voucher_loan", 0), 0)
        self.assertGreater(engine._route_context_source_voucher_volume_usd_tick.get("voucher_loan", 0.0), 0.0)
        self.assertGreater(lender_pool.vault.get(borrower_voucher), 0.0)
        self.assertEqual(len(engine._producer_debt_obligations), 1)
        obligation = engine._producer_debt_obligations[0]
        self.assertEqual(obligation.producer_pool_id, borrower_pool.pool_id)
        self.assertEqual(obligation.lender_pool_id, lender_pool.pool_id)
        self.assertEqual(obligation.voucher_id, borrower_voucher)
        self.assertGreater(obligation.borrowed_usd, 0.0)
        self.assertEqual(obligation.debt_kind, "voucher")
        expected_cash_service = (
            obligation.borrowed_usd * engine._producer_debt_contract_service_multiplier("voucher")
        )
        self.assertGreater(expected_cash_service, 0.0)
        self.assertAlmostEqual(obligation.cash_service_due_usd, expected_cash_service)
        self.assertAlmostEqual(obligation.cash_service_remaining_usd, expected_cash_service)
        self.assertAlmostEqual(engine._productive_credit_inflow_usd_tick, 0.0)
        self.assertEqual(engine._productive_credit_queue, {})
        self.assertAlmostEqual(engine._producer_debt_stable_recovered_usd_tick, 0.0)
        self.assertAlmostEqual(engine._bond_service_reserved_usd_tick, 0.0)
        self.assertTrue(engine._producer_voucher_loan_activity_active(borrower_pool, borrower_voucher))

    def test_primary_voucher_borrowing_can_run_when_stable_is_available(self):
        engine = SimulationEngine(
            small_config(
                initial_producers=2,
                max_pools=4,
                producer_deposits_enabled=True,
                producer_primary_voucher_borrowing_enabled=True,
                producer_primary_voucher_borrowing_attempt_share=1.0,
                producer_voucher_loan_activity_boost_enabled=True,
                productive_credit_voucher_activity_boost_window_ticks=4,
                producer_debt_maturity_enabled=True,
                producer_debt_contract_repayment_enabled=True,
                loan_activity_period_ticks=1,
                swap_size_min_usd=10.0,
                random_request_amount_mean=10.0,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producers = [agent for agent in engine.agents.values() if agent.role == "producer"]
        borrower, voucher_issuer = producers[0], producers[1]
        borrower_pool = engine.pools[borrower.pool_id]
        issuer_pool = engine.pools[voucher_issuer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        borrower_voucher = borrower.voucher_spec.voucher_id
        target_voucher = voucher_issuer.voucher_spec.voucher_id
        for pool in (borrower_pool, issuer_pool, lender_pool):
            pool.values.set_value(borrower_voucher, 1.0)
            pool.values.set_value(target_voucher, 1.0)
        lender_pool.list_asset_with_value_and_limit(borrower_voucher, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(target_voucher, value=1.0, window_len=10, cap_in=500.0)
        engine.accept_index.setdefault(borrower_voucher, set()).add(lender_pool.pool_id)
        engine.accept_index.setdefault(target_voucher, set()).add(lender_pool.pool_id)
        for pool, asset in (
            (borrower_pool, borrower_voucher),
            (lender_pool, target_voucher),
        ):
            current = pool.vault.get(asset)
            if current > 0.0:
                engine._vault_sub(pool, asset, current, "test_clear", "test")
        engine._vault_add(borrower_pool, borrower_voucher, 50.0, "test_seed", "test")
        engine._vault_add(lender_pool, target_voucher, 100.0, "test_seed", "test")
        engine._vault_add(lender_pool, stable_id, 100.0, "test_seed", "test")
        lender_pool.policy.min_stable_reserve = 0.0
        engine._producer_deposit_value_by_voucher[borrower_voucher] = 100.0
        engine.tick = 1
        engine._register_producer_debt_obligation(
            issuer_pool.pool_id,
            lender_pool.pool_id,
            target_voucher,
            100.0,
            100.0,
        )
        target_obligation = engine._producer_debt_obligations[0]
        engine._noam_last_refresh_tick = -1

        attempted = engine._attempt_repayment(borrower_pool)

        self.assertTrue(attempted)
        self.assertEqual(engine._producer_primary_voucher_loan_attempts_tick, 1)
        self.assertEqual(engine._producer_primary_voucher_loan_executed_tick, 1)
        self.assertEqual(engine._producer_voucher_loan_executed_tick, 1)
        self.assertEqual(engine._producer_loan_executed_tick, 0)
        self.assertEqual(len(engine._producer_debt_obligations), 2)
        borrower_obligations = [
            obligation for obligation in engine._producer_debt_obligations
            if obligation.producer_pool_id == borrower_pool.pool_id
        ]
        self.assertEqual(len(borrower_obligations), 1)
        obligation = borrower_obligations[0]
        self.assertEqual(obligation.debt_kind, "voucher")
        expected_cash_service = (
            obligation.borrowed_usd * engine._producer_debt_contract_service_multiplier("voucher")
        )
        self.assertGreater(expected_cash_service, 0.0)
        self.assertAlmostEqual(obligation.cash_service_due_usd, expected_cash_service)
        self.assertAlmostEqual(obligation.cash_service_remaining_usd, expected_cash_service)
        self.assertLess(target_obligation.remaining_voucher_units, 100.0)
        self.assertAlmostEqual(target_obligation.cash_service_remaining_usd, 100.0)
        self.assertAlmostEqual(engine._productive_credit_inflow_usd_tick, 0.0)
        self.assertAlmostEqual(engine._producer_debt_stable_recovered_usd_tick, 0.0)
        self.assertAlmostEqual(engine._bond_service_reserved_usd_tick, 0.0)

    def test_producer_credit_budget_samples_producer_wallets_first(self):
        engine = SimulationEngine(
            small_config(
                initial_lenders=1,
                initial_producers=5,
                initial_consumers=8,
                initial_liquidity_providers=0,
                max_pools=14,
                producer_credit_request_budget_share=1.0,
            ),
            seed=3,
        )
        pools = [
            pool for pool in engine.pools.values()
            if not pool.policy.system_pool and pool.policy.role != "liquidity_provider"
        ]
        attempted_pool_ids = []

        def fake_attempt(pool):
            attempted_pool_ids.append(pool.pool_id)
            return True

        engine._attempt_repayment = fake_attempt

        remaining = engine._apply_producer_credit_request_budget(pools, 3)

        self.assertEqual(remaining, 0)
        self.assertEqual(len(attempted_pool_ids), 3)
        self.assertTrue(
            all(engine.pools[pool_id].policy.role == "producer" for pool_id in attempted_pool_ids)
        )

    def test_voucher_loan_fallback_is_disabled_by_default(self):
        engine = SimulationEngine(
            small_config(
                initial_producers=2,
                max_pools=4,
                producer_deposits_enabled=True,
                swap_size_min_usd=10.0,
                random_request_amount_mean=10.0,
            )
        )
        stable_id = engine.cfg.stable_symbol
        borrower = next(agent for agent in engine.agents.values() if agent.role == "producer")
        borrower_pool = engine.pools[borrower.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = borrower.voucher_spec.voucher_id
        current_stable = lender_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(lender_pool, stable_id, current_stable, "test_clear", "test")
        current_voucher = borrower_pool.vault.get(voucher_id)
        if current_voucher > 0.0:
            engine._vault_sub(borrower_pool, voucher_id, current_voucher, "test_clear", "test")
        engine._vault_add(borrower_pool, voucher_id, 50.0, "test_seed", "test")
        engine._producer_deposit_value_by_voucher[voucher_id] = 100.0
        engine.tick = 1

        attempted = engine._attempt_new_loan(borrower_pool, voucher_id)

        self.assertFalse(attempted)
        self.assertEqual(engine._producer_voucher_loan_attempts_tick, 0)

    def test_consumer_voucher_purchase_recovers_stable_and_preserves_reserve(self):
        engine = SimulationEngine(
            small_config(
                initial_consumers=1,
                max_pools=4,
                producer_deposits_enabled=True,
                producer_debt_maturity_enabled=True,
                producer_debt_contract_repayment_enabled=True,
                ordinary_stable_spend_protection_enabled=True,
                ordinary_stable_spend_buffer_voucher_share=0.0,
                lender_voucher_purchase_demand_enabled=True,
                lender_voucher_purchase_min_usd=1.0,
                lender_voucher_purchase_inventory_share=0.05,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        consumer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "consumer")
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for pool in (producer_pool, consumer_pool, lender_pool):
            pool.values.set_value(voucher_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
        current_voucher = lender_pool.vault.get(voucher_id)
        if current_voucher > 0.0:
            engine._vault_sub(lender_pool, voucher_id, current_voucher, "test_clear", "test")
        current_stable = consumer_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(consumer_pool, stable_id, current_stable, "test_clear", "test")
        engine._vault_add(lender_pool, voucher_id, 40.0, "test_seed", "test")
        engine._vault_add(consumer_pool, stable_id, 20.0, "test_seed", "test")
        consumer_pool.policy.min_stable_reserve = 10.0
        engine.tick = 1
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            voucher_units=40.0,
            borrowed_usd=40.0,
        )

        purchased = engine._attempt_lender_voucher_purchase("consumer")

        self.assertTrue(purchased)
        self.assertEqual(engine._consumer_voucher_purchase_attempts_tick, 1)
        self.assertEqual(engine._consumer_voucher_purchase_success_tick, 1)
        self.assertGreater(engine._consumer_voucher_purchase_stable_spent_usd_tick, 0.0)
        self.assertGreater(engine._consumer_voucher_purchase_voucher_value_acquired_usd_tick, 0.0)
        self.assertGreater(engine._lender_recovered_stable_consumer_purchase_usd_tick, 0.0)
        self.assertGreater(engine._producer_debt_consumer_stable_purchase_usd_tick, 0.0)
        self.assertGreaterEqual(consumer_pool.vault.get(stable_id), consumer_pool.policy.min_stable_reserve)

    def test_consumer_voucher_purchase_reports_reserve_protected_failure(self):
        engine = SimulationEngine(
            small_config(
                initial_consumers=1,
                max_pools=4,
                ordinary_stable_spend_protection_enabled=True,
                ordinary_stable_spend_buffer_voucher_share=0.0,
                lender_voucher_purchase_demand_enabled=True,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        consumer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "consumer")
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for pool in (producer_pool, consumer_pool, lender_pool):
            pool.values.set_value(voucher_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
        current_voucher = lender_pool.vault.get(voucher_id)
        if current_voucher > 0.0:
            engine._vault_sub(lender_pool, voucher_id, current_voucher, "test_clear", "test")
        current_stable = consumer_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(consumer_pool, stable_id, current_stable, "test_clear", "test")
        engine._vault_add(lender_pool, voucher_id, 40.0, "test_seed", "test")
        engine._vault_add(consumer_pool, stable_id, 10.0, "test_seed", "test")
        consumer_pool.policy.min_stable_reserve = 10.0
        engine.tick = 1

        purchased = engine._attempt_lender_voucher_purchase("consumer")

        self.assertFalse(purchased)
        self.assertEqual(engine._consumer_voucher_purchase_attempts_tick, 1)
        self.assertEqual(engine._consumer_voucher_purchase_success_tick, 0)
        self.assertEqual(engine._consumer_voucher_purchase_reserve_protected_tick, 1)

    def test_consumer_voucher_purchase_budget_supports_purchase_above_reserve(self):
        engine = SimulationEngine(
            small_config(
                initial_consumers=1,
                max_pools=4,
                ordinary_stable_spend_protection_enabled=True,
                ordinary_stable_spend_buffer_voucher_share=0.0,
                lender_voucher_purchase_demand_enabled=True,
                lender_voucher_purchase_stable_budget_usd_per_tick=5.0,
                lender_voucher_purchase_min_usd=1.0,
                lender_voucher_purchase_inventory_share=0.05,
            )
        )
        stable_id = engine.cfg.stable_symbol
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        consumer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "consumer")
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        for pool in (producer_pool, consumer_pool, lender_pool):
            pool.values.set_value(voucher_id, 1.0)
        lender_pool.list_asset_with_value_and_limit(voucher_id, value=1.0, window_len=10, cap_in=500.0)
        lender_pool.list_asset_with_value_and_limit(stable_id, value=1.0, window_len=10, cap_in=500.0)
        current_voucher = lender_pool.vault.get(voucher_id)
        if current_voucher > 0.0:
            engine._vault_sub(lender_pool, voucher_id, current_voucher, "test_clear", "test")
        current_stable = consumer_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(consumer_pool, stable_id, current_stable, "test_clear", "test")
        engine._vault_add(lender_pool, voucher_id, 40.0, "test_seed", "test")
        engine._vault_add(consumer_pool, stable_id, 10.0, "test_seed", "test")
        consumer_pool.policy.min_stable_reserve = 10.0
        engine._lender_voucher_purchase_stable_budget_remaining_usd_tick = 5.0
        engine.tick = 1

        purchased = engine._attempt_lender_voucher_purchase("consumer")

        self.assertTrue(purchased)
        self.assertEqual(engine._consumer_voucher_purchase_success_tick, 1)
        self.assertGreater(engine._consumer_voucher_purchase_stable_budget_onramp_usd_tick, 0.0)
        self.assertGreater(engine._lender_voucher_purchase_stable_budget_onramp_usd_tick, 0.0)
        self.assertGreater(engine._stable_onramp_usd_tick, 0.0)
        self.assertGreaterEqual(consumer_pool.vault.get(stable_id), consumer_pool.policy.min_stable_reserve)

    def test_voucher_purchase_budget_is_private_buyer_wallet_replenishment(self):
        engine = SimulationEngine(
            small_config(
                initial_consumers=1,
                max_pools=4,
                lender_voucher_purchase_demand_enabled=True,
                lender_voucher_purchase_stable_budget_usd_per_tick=5.0,
            )
        )
        stable_id = engine.cfg.stable_symbol
        consumer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "consumer")
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        consumer_before = consumer_pool.vault.get(stable_id)
        lender_before = lender_pool.vault.get(stable_id)
        engine._lender_voucher_purchase_stable_budget_remaining_usd_tick = 5.0

        applied = engine._apply_voucher_purchase_stable_budget(
            "consumer",
            consumer_pool,
            3.0,
        )

        self.assertAlmostEqual(applied, 3.0)
        self.assertAlmostEqual(consumer_pool.vault.get(stable_id), consumer_before + 3.0)
        self.assertAlmostEqual(lender_pool.vault.get(stable_id), lender_before)
        self.assertAlmostEqual(engine._consumer_voucher_purchase_stable_budget_onramp_usd_tick, 3.0)
        self.assertAlmostEqual(engine._lender_voucher_purchase_stable_budget_onramp_usd_tick, 3.0)

    def test_purchase_budget_split_caps_other_producer_reuse(self):
        engine = SimulationEngine(small_config(initial_producers=1, initial_consumers=1, max_pools=4))
        stable_id = engine.cfg.stable_symbol
        producer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "producer")
        producer_before = producer_pool.vault.get(stable_id)
        engine._lender_voucher_purchase_stable_budget_remaining_usd_tick = 12.0
        engine._lender_voucher_purchase_stable_budget_remaining_by_kind_tick = {
            "consumer": 10.0,
            "third_party": 2.0,
        }

        applied = engine._apply_voucher_purchase_stable_budget("third_party", producer_pool, 5.0)

        self.assertAlmostEqual(applied, 2.0)
        self.assertAlmostEqual(engine._lender_voucher_purchase_stable_budget_remaining_usd_tick, 10.0)
        self.assertAlmostEqual(
            engine._lender_voucher_purchase_stable_budget_remaining_by_kind_tick["third_party"],
            0.0,
        )
        self.assertAlmostEqual(engine._producer_stable_reuse_budget_usd_tick, 2.0)
        self.assertAlmostEqual(producer_pool.vault.get(stable_id), producer_before + 2.0)

    def test_stable_purchase_recovery_reason_splits_actor_roles(self):
        engine = SimulationEngine(
            small_config(
                initial_producers=2,
                initial_consumers=1,
                initial_liquidity_providers=0,
                max_pools=5,
            ),
            seed=17,
        )
        producers = [agent for agent in engine.agents.values() if agent.role == "producer"]
        issuer = producers[0]
        other_producer = producers[1]
        consumer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "consumer")

        self.assertEqual(
            engine._stable_purchase_recovery_reason(engine.pools[issuer.pool_id], issuer.voucher_spec.voucher_id),
            "borrower_stable_repayment",
        )
        self.assertEqual(
            engine._stable_purchase_recovery_reason(consumer_pool, issuer.voucher_spec.voucher_id),
            "external_nonproducer_stable_purchase",
        )
        self.assertEqual(
            engine._stable_purchase_recovery_reason(
                engine.pools[other_producer.pool_id],
                issuer.voucher_spec.voucher_id,
            ),
            "other_producer_stable_purchase",
        )

    def test_lender_recovered_stable_reserve_uses_exposure_cap(self):
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

        engine._record_lender_recovered_stable(
            lender_pool.pool_id,
            100.0,
            "other_producer_stable_purchase",
            eligible_amount_usd=30.0,
        )

        self.assertAlmostEqual(engine._bond_service_reserved_usd_total, 30.0)
        self.assertAlmostEqual(engine._bond_eligible_pool_exposure_recovered_stable_usd_total, 30.0)
        self.assertAlmostEqual(engine._lender_inventory_turnover_stable_usd_total, 70.0)
        self.assertAlmostEqual(engine._lender_recovered_stable_by_pool.get(lender_pool.pool_id, 0.0), 0.0)
        self.assertAlmostEqual(lender_pool.vault.get(stable_id), 70.0)

    def test_nonborrower_recovery_eligibility_uses_empirical_budget_cap(self):
        engine = SimulationEngine(
            small_config(
                external_nonproducer_stable_to_voucher_budget_usd_per_tick=10.0,
                other_producer_stable_to_voucher_budget_usd_per_tick=2.0,
            )
        )
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        engine._refresh_bond_recovery_budget_caps()

        engine._record_lender_recovered_stable(
            lender_pool.pool_id,
            30.0,
            "external_nonproducer_stable_purchase",
            eligible_amount_usd=30.0,
        )
        engine._record_lender_recovered_stable(
            lender_pool.pool_id,
            5.0,
            "other_producer_stable_purchase",
            eligible_amount_usd=5.0,
        )
        engine._record_lender_recovered_stable(
            lender_pool.pool_id,
            7.0,
            "third_party_stable_purchase",
            eligible_amount_usd=7.0,
        )

        self.assertAlmostEqual(engine._bond_eligible_pool_exposure_recovered_stable_usd_total, 12.0)
        self.assertAlmostEqual(engine._lender_inventory_turnover_stable_usd_total, 30.0)

    def test_producer_stable_exit_removes_calibrated_receipt_share(self):
        engine = SimulationEngine(small_config(producer_stable_exit_share=0.75))
        stable_id = engine.cfg.stable_symbol
        producer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "producer")
        current_stable = producer_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(producer_pool, stable_id, current_stable, "test_clear", "test")
        engine._vault_add(producer_pool, stable_id, 100.0, "test_recovery", "test")

        exited = engine._apply_producer_stable_exit(producer_pool, 100.0, "test")

        self.assertAlmostEqual(exited, 75.0)
        self.assertAlmostEqual(engine._producer_stable_exited_usd_tick, 75.0)
        self.assertAlmostEqual(engine._stable_offramp_usd_tick, 75.0)
        self.assertAlmostEqual(producer_pool.vault.get(stable_id), 25.0)

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

    def test_ordinary_stable_spend_protection_records_blocked_source(self):
        engine = SimulationEngine(
            small_config(
                ordinary_stable_spend_protection_enabled=True,
                ordinary_stable_spend_buffer_voucher_share=0.05,
            )
        )
        producer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "producer")
        stable_id = engine.cfg.stable_symbol
        voucher_id = next(asset for asset in producer_pool.vault.inventory if asset.startswith("VCHR:"))
        current_stable = producer_pool.vault.get(stable_id)
        if current_stable > 0.0:
            engine._vault_sub(producer_pool, stable_id, current_stable, "test_clear", "test")
        current_voucher = producer_pool.vault.get(voucher_id)
        if current_voucher > 0.0:
            engine._vault_sub(producer_pool, voucher_id, current_voucher, "test_clear", "test")
        engine._vault_add(producer_pool, stable_id, 10.0, "test_seed", "test")
        engine._vault_add(producer_pool, voucher_id, 100.0, "test_seed", "test")
        producer_pool.policy.min_stable_reserve = 20.0

        engine._record_ordinary_stable_spend_protection_skip(producer_pool)

        self.assertEqual(engine._ordinary_stable_spend_protected_skip_count_tick, 1)
        self.assertAlmostEqual(engine._ordinary_stable_spend_protected_skip_value_usd_tick, 10.0)

    def test_route_context_metrics_split_ordinary_and_loan_swaps(self):
        engine = SimulationEngine(small_config(productive_credit_voucher_activity_boost_enabled=True))
        producer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "producer")
        stable_id = engine.cfg.stable_symbol
        voucher_id = next(asset for asset in producer_pool.vault.inventory if asset.startswith("VCHR:"))
        engine.tick = 5
        engine._mark_productive_credit_voucher_activity(producer_pool.pool_id, voucher_id)

        engine._record_route_context_swap("ordinary", producer_pool, voucher_id, 12.0)
        engine._record_route_context_swap("loan", producer_pool, voucher_id, 8.0)
        engine._record_route_context_swap("ordinary", producer_pool, stable_id, 4.0)

        self.assertEqual(engine._route_context_count_tick["ordinary"], 2)
        self.assertEqual(engine._route_context_count_tick["loan"], 1)
        self.assertAlmostEqual(engine._route_context_volume_usd_tick["ordinary"], 16.0)
        self.assertAlmostEqual(engine._route_context_source_voucher_volume_usd_tick["ordinary"], 12.0)
        self.assertAlmostEqual(engine._route_context_source_stable_volume_usd_tick["ordinary"], 4.0)
        self.assertEqual(engine._productive_boosted_voucher_swap_count_tick, 1)

    def test_failed_loan_route_backfill_attempts_ordinary_activity(self):
        engine = SimulationEngine(
            small_config(
                producer_loan_failure_backfill_enabled=True,
                producer_loan_failure_backfill_max_attempts=1,
            )
        )
        producer_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "producer")
        calls = []

        def fake_random_route_request(source_pool=None, max_assets=None, route_context="ordinary"):
            calls.append((source_pool, max_assets, route_context))
            engine._noam_routing_swaps_tick += 1
            return 1

        engine._random_route_request = fake_random_route_request

        engine._backfill_failed_loan_route(producer_pool)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], producer_pool)
        self.assertEqual(calls[0][1], 1)
        self.assertEqual(calls[0][2], "loan_backfill")
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
            issuer_payment_stride=26,
            pool_clearing_stride=13,
            producer_debt_maturity_ticks=13,
            _frontier_producer_stable_deposit_rate_per_month=0.0,
            _frontier_producer_voucher_deposit_rate_per_month=0.0,
            _frontier_productive_credit_return_rate=0.0,
            _frontier_productive_credit_lag_ticks=2,
            _frontier_productive_credit_voucher_deposit_share=0.384157,
            _frontier_productive_credit_voucher_deposit_cap_rate_per_month=0.143206,
            _frontier_producer_debt_maturity_recovery_rate=0.673,
            _frontier_quarterly_clearing_surplus_share=1.0,
            _frontier_route_requests_per_tick=1,
            _frontier_swap_floor_per_tick=0,
            _frontier_historical_cash_backing_total_usd=500.0,
            _frontier_historical_voucher_backing_total_usd=0.0,
        )

        cfg = scenario_config("bond_issuer_frontier", 0.06, 260, args)

        self.assertEqual(cfg.initial_liquidity_providers, 0)
        self.assertAlmostEqual(cfg.lp_initial_stable_mean, 0.0)
        self.assertAlmostEqual(cfg.lender_initial_stable_mean, 250.0)
        self.assertAlmostEqual(cfg.bond_gross_principal_usd, 1000.0)
        self.assertAlmostEqual(cfg.bond_deployed_principal_usd, 1000.0)
        self.assertAlmostEqual(cfg.issuer_reserve_share, 0.0)
        self.assertEqual(cfg.issuer_payment_stride_ticks, 26)
        self.assertEqual(cfg.quarterly_clearing_stride_ticks, 13)
        self.assertEqual(cfg.producer_debt_maturity_ticks, 13)
        self.assertAlmostEqual(cfg.producer_debt_maturity_recovery_rate, 0.673)
        self.assertTrue(cfg.bond_service_reserve_enabled)
        self.assertEqual(cfg.bond_service_lockbox_mode, "remaining_schedule")
        self.assertAlmostEqual(cfg.bond_service_lockbox_coverage_ratio, 1.25)
        self.assertTrue(cfg.producer_debt_contract_repayment_enabled)
        self.assertAlmostEqual(cfg.producer_debt_contract_service_margin_rate, 0.0)
        self.assertAlmostEqual(cfg.producer_stable_debt_contract_service_margin_rate, 0.0)
        self.assertAlmostEqual(cfg.producer_voucher_debt_contract_service_margin_rate, 0.0)
        self.assertAlmostEqual(cfg.producer_debt_contract_revenue_rate, 1.0)
        self.assertTrue(cfg.producer_debt_pressure_enabled)
        self.assertEqual(cfg.producer_debt_pressure_period_ticks, 4)
        self.assertAlmostEqual(cfg.producer_debt_pressure_capacity_share, 1.0)
        self.assertAlmostEqual(cfg.producer_debt_pressure_prepay_share, 0.10)
        self.assertTrue(cfg.producer_debt_pressure_batching_enabled)
        self.assertAlmostEqual(cfg.producer_debt_pressure_min_swap_usd, 1.0)
        self.assertTrue(cfg.producer_debt_attention_crowdout_enabled)
        self.assertAlmostEqual(cfg.producer_debt_attention_crowdout_scale, 1.0)
        self.assertAlmostEqual(cfg.producer_debt_attention_crowdout_max_share, 0.90)
        self.assertIsNone(cfg.producer_debt_attention_reference_usd)
        self.assertAlmostEqual(cfg.producer_debt_attention_min_pressure_usd, 0.0)
        self.assertTrue(cfg.producer_bond_assessment_pressure_enabled)
        self.assertAlmostEqual(cfg.producer_bond_assessment_pressure_scale, 1.0)
        self.assertTrue(cfg.producer_bond_assessment_sustain_offset_enabled)
        self.assertTrue(cfg.ordinary_own_voucher_stable_borrowing_enabled)
        self.assertAlmostEqual(cfg.ordinary_own_voucher_stable_borrowing_probability, 0.70)
        self.assertTrue(cfg.producer_debt_penalty_enabled)
        self.assertFalse(cfg.offramps_enabled)
        self.assertIsNone(cfg.historical_stable_backing_tick)
        self.assertAlmostEqual(cfg.historical_stable_backing_total_usd, 0.0)
        self.assertTrue(cfg.productive_credit_voucher_feedback_enabled)
        self.assertAlmostEqual(cfg.productive_credit_voucher_deposit_share, 0.384157)
        self.assertAlmostEqual(cfg.productive_credit_voucher_deposit_cap_rate_per_month, 0.143206)
        self.assertTrue(cfg.productive_credit_voucher_activity_boost_enabled)
        self.assertAlmostEqual(cfg.productive_credit_voucher_source_weight_boost, 0.0)
        self.assertAlmostEqual(cfg.productive_credit_voucher_source_size_multiplier, 1.0)
        self.assertAlmostEqual(cfg.min_stable_reserve_mean, 0.0)
        self.assertFalse(cfg.ordinary_stable_spend_protection_enabled)
        self.assertAlmostEqual(cfg.ordinary_stable_spend_buffer_voucher_share, 0.0)
        self.assertTrue(cfg.producer_loan_failure_backfill_enabled)
        self.assertEqual(cfg.producer_loan_failure_backfill_max_attempts, 1)
        self.assertFalse(cfg.producer_voucher_loan_fallback_enabled)
        self.assertFalse(cfg.producer_voucher_loan_activity_boost_enabled)
        self.assertFalse(cfg.producer_primary_voucher_borrowing_enabled)
        self.assertAlmostEqual(cfg.producer_primary_voucher_borrowing_attempt_share, 0.0)
        self.assertFalse(cfg.lender_voucher_purchase_demand_enabled)
        self.assertEqual(cfg.lender_voucher_purchase_attempts_per_tick, 0)
        self.assertEqual(cfg.liquidity_mandate_mode, "community_deficit_then_lender")
        self.assertEqual(cfg.max_pools, 9)
        self.assertEqual(cfg.max_hops, 3)
        self.assertEqual(cfg.noam_max_hops, 3)
        self.assertTrue(cfg.noam_overlay_enabled)
        self.assertTrue(cfg.noam_clearing_enabled)
        self.assertEqual(cfg.noam_clearing_stride_ticks, 13)
        self.assertTrue(cfg.settlement_motif_purchase_lane_adjustment_enabled)
        self.assertTrue(cfg.noam_clearing_budget_scale_by_stride)

        baseline_args = argparse.Namespace(**vars(args))
        baseline_args._current_principal_usd = 0.0
        cfg_baseline = scenario_config("bond_issuer_frontier", 0.06, 260, baseline_args)
        for attr in (
            "max_hops",
            "noam_max_hops",
            "noam_overlay_enabled",
            "noam_clearing_enabled",
            "noam_clearing_stride_ticks",
            "noam_clearing_max_cycles",
            "noam_clearing_max_hops",
            "noam_clearing_edge_cap_per_asset",
            "decision_based_activity_enabled",
            "repeat_partner_route_share",
            "affinity_buddy_min_count",
            "swap_sustain_max_extra_attempts",
            "swap_sustain_max_rounds",
            "swap_sustain_attempts_per_missing_swap",
            "voucher_fee_conversion_max_swaps_per_epoch",
            "voucher_fee_conversion_max_usd_per_epoch",
            "voucher_settlement_mode",
            "producer_debt_pressure_enabled",
            "producer_debt_pressure_period_ticks",
            "producer_debt_pressure_capacity_share",
            "producer_debt_pressure_prepay_share",
            "producer_debt_pressure_batching_enabled",
            "producer_debt_pressure_min_swap_usd",
            "producer_debt_attention_crowdout_enabled",
            "producer_debt_attention_crowdout_scale",
            "producer_debt_attention_crowdout_max_share",
            "producer_debt_attention_reference_usd",
            "producer_debt_attention_min_pressure_usd",
            "producer_bond_assessment_pressure_enabled",
            "producer_bond_assessment_pressure_scale",
            "producer_bond_assessment_sustain_offset_enabled",
            "ordinary_own_voucher_stable_borrowing_enabled",
            "ordinary_own_voucher_stable_borrowing_probability",
            "producer_debt_penalty_enabled",
        ):
            self.assertEqual(getattr(cfg_baseline, attr), getattr(cfg, attr))

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

        args._current_scale_factor = 2.0
        args.lender_voucher_purchase_stable_budget_usd_per_tick = 184.061305
        args.enable_lender_voucher_purchase_demand = True
        cfg_scaled = scenario_config("bond_issuer_frontier", 0.06, 260, args)
        self.assertAlmostEqual(
            cfg_scaled.lender_voucher_purchase_stable_budget_usd_per_tick,
            368.12261,
        )

    def test_frontier_probe_flags_are_preserved_in_shard_payloads(self):
        for key in (
            "enable_producer_voucher_loan_fallback",
            "enable_producer_voucher_loan_activity_boost",
            "enable_producer_primary_voucher_borrowing",
            "enable_ordinary_stable_spend_protection",
            "producer_primary_voucher_borrowing_attempt_share",
            "producer_credit_request_budget_share",
            "producer_voucher_loan_max_target_candidates",
            "producer_voucher_overlap_mode",
            "enable_lender_voucher_purchase_demand",
            "lender_voucher_purchase_attempts_per_tick",
            "lender_voucher_purchase_consumer_share",
            "lender_voucher_purchase_inventory_share",
            "lender_voucher_purchase_stable_budget_usd_per_tick",
            "max_hops",
            "noam_max_hops",
            "noam_overlay_enabled",
            "noam_clearing_enabled",
            "noam_clearing_stride_ticks",
            "noam_clearing_max_cycles",
            "noam_clearing_max_hops",
            "noam_clearing_edge_cap_per_asset",
            "swap_sustain_max_extra_attempts",
            "swap_sustain_max_rounds",
            "swap_sustain_attempts_per_missing_swap",
            "voucher_fee_conversion_max_swaps_per_epoch",
            "voucher_fee_conversion_max_usd_per_epoch",
            "voucher_settlement_mode",
            "issuer_payment_stride",
            "pool_clearing_stride",
            "producer_debt_maturity_ticks",
            "disable_producer_debt_pressure",
            "producer_debt_pressure_period_ticks",
            "producer_debt_pressure_capacity_share",
            "producer_debt_pressure_prepay_share",
            "producer_debt_pressure_min_swap_usd",
            "enable_producer_debt_attention_crowdout",
            "producer_debt_attention_crowdout_scale",
            "producer_debt_attention_crowdout_max_share",
            "producer_debt_attention_reference_usd",
            "producer_debt_attention_min_pressure_usd",
            "enable_producer_bond_assessment_pressure",
            "producer_bond_assessment_pressure_scale",
            "disable_producer_bond_assessment_sustain_offset",
            "disable_producer_debt_pressure_batching",
            "disable_producer_debt_penalty",
            "producer_debt_penalty_rate_per_period",
            "enable_ordinary_own_voucher_stable_borrowing",
            "ordinary_own_voucher_stable_borrowing_probability",
        ):
            self.assertIn(key, SHARD_CONFIG_KEYS)

    def test_validation_uses_current_network_routing_profile(self):
        args = argparse.Namespace(
            pool_metrics_stride=1,
            max_active_pools_per_tick=None,
            _calibration_kes_per_usd=128.0,
            _voucher_unit_value_usd=1.0 / 128.0,
            _current_initial_lenders=3,
            _current_initial_producers=4,
            _current_initial_consumers=5,
            _validation_swap_floor_per_tick=0,
            noam_clearing_max_cycles=17,
            noam_clearing_max_hops=3,
            noam_clearing_edge_cap_per_asset=9,
            decision_based_activity_enabled=1,
            repeat_partner_route_share=0.55,
            affinity_buddy_min_count=1,
            swap_sustain_max_extra_attempts=23,
            swap_sustain_max_rounds=4,
            swap_sustain_attempts_per_missing_swap=1.5,
            voucher_fee_conversion_max_swaps_per_epoch=7,
            voucher_fee_conversion_max_usd_per_epoch=250.0,
        )

        cfg = scenario_config("sarafu_engine_validation", 0.0, 260, args)

        self.assertEqual(cfg.max_hops, 3)
        self.assertEqual(cfg.noam_max_hops, 3)
        self.assertTrue(cfg.noam_overlay_enabled)
        self.assertTrue(cfg.noam_clearing_enabled)
        self.assertEqual(cfg.noam_clearing_stride_ticks, 13)
        self.assertTrue(cfg.noam_clearing_budget_scale_by_stride)
        self.assertEqual(cfg.noam_clearing_max_cycles, 17)
        self.assertEqual(cfg.noam_clearing_max_hops, 3)
        self.assertEqual(cfg.noam_clearing_edge_cap_per_asset, 9)
        self.assertTrue(cfg.decision_based_activity_enabled)
        self.assertAlmostEqual(cfg.repeat_partner_route_share, 0.55)
        self.assertEqual(cfg.affinity_buddy_min_count, 1)
        self.assertEqual(cfg.swap_sustain_max_extra_attempts, 23)
        self.assertEqual(cfg.swap_sustain_max_rounds, 4)
        self.assertAlmostEqual(cfg.swap_sustain_attempts_per_missing_swap, 1.5)
        self.assertEqual(cfg.voucher_fee_conversion_max_swaps_per_epoch, 7)
        self.assertAlmostEqual(cfg.voucher_fee_conversion_max_usd_per_epoch, 250.0)
        self.assertTrue(cfg.producer_debt_pressure_enabled)
        self.assertEqual(cfg.producer_debt_pressure_period_ticks, 4)
        self.assertAlmostEqual(cfg.producer_debt_pressure_capacity_share, 1.0)
        self.assertAlmostEqual(cfg.producer_debt_pressure_prepay_share, 0.10)
        self.assertTrue(cfg.producer_debt_pressure_batching_enabled)
        self.assertAlmostEqual(cfg.producer_debt_pressure_min_swap_usd, 1.0)
        self.assertTrue(cfg.producer_debt_attention_crowdout_enabled)
        self.assertAlmostEqual(cfg.producer_debt_attention_crowdout_scale, 1.0)
        self.assertAlmostEqual(cfg.producer_debt_attention_crowdout_max_share, 0.90)
        self.assertIsNone(cfg.producer_debt_attention_reference_usd)
        self.assertAlmostEqual(cfg.producer_debt_attention_min_pressure_usd, 0.0)
        self.assertTrue(cfg.producer_bond_assessment_pressure_enabled)
        self.assertAlmostEqual(cfg.producer_bond_assessment_pressure_scale, 1.0)
        self.assertFalse(cfg.producer_bond_assessment_sustain_offset_enabled)
        self.assertTrue(cfg.ordinary_own_voucher_stable_borrowing_enabled)
        self.assertAlmostEqual(cfg.ordinary_own_voucher_stable_borrowing_probability, 0.70)
        self.assertTrue(cfg.producer_debt_penalty_enabled)

    def test_sarafu_activity_controls_load_settlement_motif_targets(self):
        calibration = load_calibration(Path("analysis/sarafu_calibration"))
        args = argparse.Namespace()

        configure_sarafu_activity_controls(args, calibration, 260, "validation")

        self.assertAlmostEqual(
            args._validation_settlement_motif_voucher_to_voucher_share,
            0.814026,
        )
        self.assertAlmostEqual(
            args._validation_settlement_motif_voucher_to_stable_share,
            0.12288,
        )
        self.assertAlmostEqual(
            args._validation_settlement_motif_stable_to_voucher_share,
            0.053786,
        )
        self.assertEqual(
            args._validation_lender_voucher_purchase_empirical_window,
            "trailing_52w",
        )
        self.assertAlmostEqual(
            args._validation_lender_voucher_purchase_empirical_s2v_events,
            293.0,
        )
        self.assertAlmostEqual(
            args._validation_lender_voucher_purchase_empirical_s2v_per_week,
            293.0 / 52.29,
        )
        self.assertAlmostEqual(
            args._validation_lender_voucher_purchase_empirical_stable_input_usd,
            2206.257821,
        )
        self.assertEqual(args._validation_lender_voucher_purchase_attempts_per_tick, 6)
        self.assertAlmostEqual(
            args._validation_lender_voucher_purchase_avg_stable_spend_usd,
            2206.257821 / 293.0,
        )
        self.assertAlmostEqual(
            args._validation_lender_voucher_purchase_target_usd,
            2206.257821 / 293.0,
        )
        self.assertAlmostEqual(
            args._validation_lender_voucher_purchase_max_usd,
            2206.257821 / 293.0,
        )
        self.assertAlmostEqual(
            args._validation_lender_voucher_purchase_stable_budget_usd_per_tick,
            34.244728 + 7.951460,
        )
        self.assertAlmostEqual(
            args._validation_external_nonproducer_stable_to_voucher_budget_usd_per_tick,
            34.244728,
        )
        self.assertAlmostEqual(
            args._validation_other_producer_stable_to_voucher_budget_usd_per_tick,
            7.951460,
        )
        self.assertAlmostEqual(args._validation_lender_voucher_purchase_consumer_share, 205 / 293)
        self.assertAlmostEqual(args._validation_producer_stable_reuse_share, 0.032480)
        self.assertAlmostEqual(args._validation_producer_stable_exit_share, 0.967520)
        pool_era = calibration.stable_to_voucher_actor_split["pool_era"]
        self.assertEqual(
            sum(int(pool_era[role]["events"]) for role in pool_era),
            441,
        )
        self.assertAlmostEqual(
            sum(pool_era[role]["stable_input_value_usd"] for role in pool_era),
            10375.997821,
        )
        trailing_52w = calibration.stable_to_voucher_actor_split["trailing_52w"]
        self.assertAlmostEqual(trailing_52w["original_issuer_self"]["stable_input_value_usd"], 10.65)
        self.assertAlmostEqual(trailing_52w["other_producer"]["stable_input_value_usd"], 415.74778)
        self.assertAlmostEqual(
            trailing_52w["external_nonproducer"]["stable_input_value_usd"],
            1790.510041,
        )
        reuse = calibration.producer_stable_reuse_calibration["trailing_52w"]
        self.assertAlmostEqual(reuse["producer_stable_reuse_share"], 0.032480)

    def test_calibration_loader_reads_overlap_distribution(self):
        def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_rows(root / "monte_carlo_calibration_parameters.csv", ["parameter", "value"], [])
            write_rows(
                root / "repayment_calibration_by_tier_asset.csv",
                ["tier", "asset_class", "same_token_return_coverage", "same_token_out_value"],
                [{"tier": "strong", "asset_class": "redeemable_voucher", "same_token_return_coverage": 1, "same_token_out_value": 1}],
            )
            write_rows(
                root / "pool_report_activity.csv",
                [
                    "tier",
                    "score",
                    "swap_events",
                    "recent_swap_weeks_90d",
                    "active_weeks",
                    "swaps_per_active_week",
                    "total_users",
                    "backing_inflow",
                    "backing_cash_inflow",
                    "backing_voucher_inflow",
                    "tagged_voucher_tokens",
                    "verified_report_exposure",
                    "same_token_return_rate",
                    "same_token_out_value",
                    "same_token_matched_later_in_value",
                    "borrow_proxy_matured_events",
                    "borrow_proxy_matured_return_rate",
                    "rosca_proxy_value_return_rate",
                ],
                [{
                    "tier": "strong",
                    "score": 1,
                    "swap_events": 1,
                    "recent_swap_weeks_90d": 1,
                    "active_weeks": 1,
                    "swaps_per_active_week": 1,
                    "total_users": 1,
                    "backing_inflow": 1,
                    "backing_cash_inflow": 1,
                    "backing_voucher_inflow": 0,
                    "tagged_voucher_tokens": 1,
                    "verified_report_exposure": 1,
                    "same_token_return_rate": 1,
                    "same_token_out_value": 1,
                    "same_token_matched_later_in_value": 1,
                    "borrow_proxy_matured_events": 1,
                    "borrow_proxy_matured_return_rate": 1,
                    "rosca_proxy_value_return_rate": 1,
                }],
            )
            write_rows(
                root / "impact_projection_by_activity.csv",
                ["activity", "log_intercept", "log_slope", "verified_exposure_share"],
                [{"activity": "Test", "log_intercept": 0, "log_slope": 1, "verified_exposure_share": 1}],
            )
            write_rows(
                root / "voucher_pool_overlap_calibration.csv",
                ["metric", "value", "claim_boundary"],
                [
                    {"metric": "multi_pool_share", "value": "0.25", "claim_boundary": ""},
                    {"metric": "pool_degree_p50", "value": "2", "claim_boundary": ""},
                    {"metric": "pool_degree_p90", "value": "5", "claim_boundary": ""},
                ],
            )
            write_rows(
                root / "voucher_pool_overlap_distribution.csv",
                ["pool_degree_bucket", "min_pool_degree", "max_pool_degree", "voucher_tokens", "share"],
                [
                    {"pool_degree_bucket": "1", "min_pool_degree": 1, "max_pool_degree": 1, "voucher_tokens": 3, "share": 0.75},
                    {"pool_degree_bucket": "4-5", "min_pool_degree": 4, "max_pool_degree": 5, "voucher_tokens": 1, "share": 0.25},
                ],
            )

            calibration = load_calibration(root)

        self.assertAlmostEqual(calibration.voucher_pool_overlap_calibration["multi_pool_share"], 0.25)
        self.assertAlmostEqual(calibration.voucher_pool_overlap_distribution["1"], 0.75)
        self.assertAlmostEqual(calibration.voucher_pool_overlap_distribution["4-5"], 0.25)

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
        self.assertAlmostEqual(engine._productive_credit_stable_retained_usd_tick, 100.0)
        self.assertNotIn(producer.voucher_spec.voucher_id, engine._producer_deposit_value_by_voucher)

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
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        value = engine._asset_value(producer_pool, voucher_id)
        required_units = 40.0 / value
        current_units = producer_pool.vault.get(voucher_id)
        if current_units < required_units:
            top_up = required_units - current_units
            engine._vault_add(producer_pool, voucher_id, top_up, "test_seed", "test")
            producer.issuer.issue(top_up)
        before_stable = producer_pool.vault.get(engine.cfg.stable_symbol)
        before_voucher = producer_pool.vault.get(voucher_id)
        before_lender_voucher = lender_pool.vault.get(voucher_id)
        before_capacity = engine._producer_deposit_credit_capacity_usd()

        engine.tick = 1
        engine._schedule_productive_credit_inflow(producer.pool_id, 100.0, voucher_id)
        engine.tick = 2
        engine._apply_productive_credit_inflows()

        self.assertAlmostEqual(engine._productive_credit_inflow_usd_tick, 100.0)
        self.assertAlmostEqual(engine._productive_credit_stable_retained_usd_tick, 60.0)
        self.assertAlmostEqual(engine._productive_credit_voucher_deposit_usd_tick, 40.0)
        self.assertAlmostEqual(producer_pool.vault.get(engine.cfg.stable_symbol) - before_stable, 60.0)
        self.assertAlmostEqual(producer_pool.vault.get(voucher_id) - before_voucher, -required_units)
        self.assertAlmostEqual(lender_pool.vault.get(voucher_id) - before_lender_voucher, 40.0 / value)
        self.assertAlmostEqual(
            engine._producer_deposit_credit_capacity_usd() - before_capacity,
            40.0 * engine.cfg.lender_voucher_cap_deposit_multiple,
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

    def test_historical_voucher_backing_is_deposited_with_lender_pool(self):
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
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        required_units = 25_000.0
        current_units = producer_pool.vault.get(voucher_id)
        if current_units < required_units:
            top_up = required_units - current_units
            engine._vault_add(producer_pool, voucher_id, top_up, "test_seed", "test")
            producer.issuer.issue(top_up)
        before_units = producer_pool.vault.get(voucher_id)
        before_lender_units = lender_pool.vault.get(voucher_id)
        before_deposit_value = engine._producer_deposit_value_by_voucher.get(voucher_id, 0.0)
        before_credit_capacity = engine._producer_deposit_credit_capacity_usd()

        engine.tick = 2
        engine._apply_historical_voucher_backing()

        self.assertAlmostEqual(engine._producer_deposit_voucher_usd_total, 250.0)
        self.assertAlmostEqual(
            engine._producer_deposit_value_by_voucher[voucher_id] - before_deposit_value,
            250.0,
        )
        self.assertAlmostEqual(producer_pool.vault.get(voucher_id) - before_units, -25_000.0)
        self.assertAlmostEqual(lender_pool.vault.get(voucher_id) - before_lender_units, 25_000.0)
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

        self.assertEqual(summary["safe"], 1)
        self.assertEqual(summary["issuer_operating_risk_headroom_ge_125"], 0)
        self.assertNotIn("p50_available_service_cash_headroom", summary["binding_constraint"])

        headroom_rows = issuer_headroom_frontier_rows([summary])
        self.assertEqual(headroom_rows[0]["headroom_principal_ratio"], 0.0)
        self.assertEqual(
            headroom_rows[0]["binding_constraint_at_headroom_frontier"],
            "issuer_operating_risk_headroom_lt_1p25",
        )

    def test_frontier_safety_keeps_total_swap_volume_decline_diagnostic(self):
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

        self.assertEqual(summary["safe"], 1)
        self.assertEqual(summary["diagnostic_total_swap_volume_decline"], 1)
        self.assertEqual(summary["material_decline_swap_volume_decline"], 0)
        self.assertEqual(summary["material_decline_vs_no_bond"], 0)
        self.assertIn("total_swap_volume_decline", summary["diagnostic_decline_reasons"])
        self.assertNotIn("swap_volume_decline_vs_no_bond", summary["binding_constraint"])

    def test_frontier_safety_treats_voucher_share_decline_as_diagnostic(self):
        row = {
            "bond_principal_usd": 1000.0,
            "principal_ratio": 0.05,
            "network_scale": "current",
            "coupon_target_annual": 0.0,
            "bond_fee_service_share": 1.0,
            "issuer_service_coverage_ratio": 1.0,
            "issuer_paid_coverage_ratio": 1.0,
            "issuer_service_cash_headroom_ratio": 2.0,
            "issuer_available_service_cash_headroom_ratio": 2.0,
            "issuer_scheduled_debt_service_due_usd": 100.0,
            "issuer_actual_bondholder_payment_usd": 100.0,
            "issuer_unpaid_scheduled_claim_usd": 0.0,
            "route_success_rate_cumulative": 1.0,
            "realized_edge_top_share": 0.0,
            "swap_volume_usd_total": 500.0,
            "transactions_total": 200.0,
            "swap_count_vchr_to_vchr_total": 100.0,
            "swap_volume_vchr_to_vchr_total": 120.0,
            "ordinary_voucher_source_swap_count_total": 100.0,
            "ordinary_voucher_source_swap_volume_usd_total": 120.0,
            "stable_value_share_in_active_pools": 0.10,
            "voucher_value_share_in_active_pools": 0.90,
            "consumer_stable_reserve_stress_ratio": 0.0,
            "community_stable_reserve_stress_ratio": 0.0,
            "stable_liquidity_leakage_ratio_cumulative": 0.0,
        }
        baseline = {
            "route_success_p50": 1.0,
            "swap_volume_p50": 500.0,
            "voucher_to_voucher_count_p50": 80.0,
            "voucher_to_voucher_volume_p50": 100.0,
            "voucher_to_voucher_share_p50": 0.80,
            "ordinary_voucher_source_swap_count_total_p50": 80.0,
            "ordinary_voucher_source_swap_volume_usd_total_p50": 100.0,
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

        self.assertEqual(summary["safe"], 1)
        self.assertEqual(summary["voucher_to_voucher_share_decline_vs_no_bond"], 1)
        self.assertEqual(summary["diagnostic_voucher_to_voucher_share_decline"], 1)
        self.assertEqual(summary["material_decline_voucher_to_voucher_share_decline"], 0)
        self.assertEqual(summary["material_decline_voucher_circulation_decline"], 0)
        self.assertEqual(summary["material_decline_vs_no_bond"], 0)
        self.assertIn("voucher_to_voucher_share_decline", summary["diagnostic_decline_reasons"])
        self.assertNotIn("voucher_to_voucher_share_decline_vs_no_bond", summary["binding_constraint"])

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

    def test_producer_debt_contract_uses_channel_specific_margins(self):
        engine = SimulationEngine(
            small_config(
                producer_debt_maturity_enabled=True,
                producer_debt_contract_repayment_enabled=True,
                producer_stable_debt_contract_service_margin_rate=0.05,
                producer_voucher_debt_contract_service_margin_rate=0.02,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        lender_pool.values.set_value(voucher_id, 1.0)

        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
            debt_kind="stable",
        )
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
            debt_kind="voucher",
        )

        self.assertAlmostEqual(engine._producer_debt_obligations[0].cash_service_due_usd, 105.0)
        self.assertEqual(engine._producer_debt_obligations[0].debt_kind, "stable")
        self.assertAlmostEqual(engine._producer_debt_obligations[1].cash_service_due_usd, 102.0)
        self.assertEqual(engine._producer_debt_obligations[1].debt_kind, "voucher")

    def test_producer_debt_contract_shared_margin_applies_to_all_channels(self):
        engine = SimulationEngine(
            small_config(
                producer_debt_maturity_enabled=True,
                producer_debt_contract_repayment_enabled=True,
                producer_debt_contract_service_margin_rate=0.03,
            )
        )
        producer = next(agent for agent in engine.agents.values() if agent.role == "producer")
        producer_pool = engine.pools[producer.pool_id]
        lender_pool = next(pool for pool in engine.pools.values() if pool.policy.role == "lender")
        voucher_id = producer.voucher_spec.voucher_id
        lender_pool.values.set_value(voucher_id, 1.0)

        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
            debt_kind="stable",
        )
        engine._register_producer_debt_obligation(
            producer_pool.pool_id,
            lender_pool.pool_id,
            voucher_id,
            100.0,
            100.0,
            debt_kind="voucher",
        )

        self.assertAlmostEqual(engine._producer_debt_obligations[0].cash_service_due_usd, 103.0)
        self.assertAlmostEqual(engine._producer_debt_obligations[1].cash_service_due_usd, 103.0)

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

    def test_inventory_turnover_stable_purchase_closes_loan_but_not_bond_cash(self):
        engine = SimulationEngine(
            small_config(
                producer_debt_maturity_enabled=True,
                external_nonproducer_stable_to_voucher_budget_usd_per_tick=10.0,
                other_producer_stable_to_voucher_budget_usd_per_tick=2.0,
            )
        )
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
        engine._refresh_bond_recovery_budget_caps()

        reduced = engine._reduce_producer_debt_obligations(
            lender_pool.pool_id,
            voucher_id,
            30.0,
            "inventory_turnover_stable_purchase",
        )
        engine._record_lender_recovered_stable(
            lender_pool.pool_id,
            30.0,
            "inventory_turnover_stable_purchase",
            eligible_amount_usd=30.0,
        )

        self.assertAlmostEqual(reduced, 30.0)
        self.assertAlmostEqual(engine._producer_debt_stable_recovered_usd_total, 30.0)
        self.assertAlmostEqual(engine._bond_eligible_pool_exposure_recovered_stable_usd_total, 0.0)
        self.assertAlmostEqual(engine._lender_inventory_turnover_stable_usd_total, 30.0)

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
        engine._route_repeat_partner_requested_tick = 1
        engine._route_exploration_requested_tick = 1
        engine._route_sticky_used_tick = 1
        engine._route_buddy_direct_used_tick = 1
        engine._route_new_target_search_tick = 1
        engine._route_search_fallback_used_tick = 1

        engine.snapshot_metrics(force_network=True)
        latest = engine.metrics.network_rows[-1]

        self.assertEqual(latest["route_fixed_failed_tick"], 1)
        self.assertEqual(latest["route_substitution_found_tick"], 1)
        self.assertEqual(latest["route_repeat_partner_requested_tick"], 1)
        self.assertEqual(latest["route_exploration_requested_tick"], 1)
        self.assertAlmostEqual(latest["route_repeat_partner_share_tick"], 0.5)
        self.assertAlmostEqual(latest["route_sticky_share_tick"], 0.5)
        self.assertAlmostEqual(latest["route_buddy_direct_share_tick"], 0.5)
        self.assertAlmostEqual(latest["route_new_target_search_share_tick"], 0.5)
        self.assertAlmostEqual(latest["route_search_fallback_share_tick"], 0.5)


if __name__ == "__main__":
    unittest.main()
