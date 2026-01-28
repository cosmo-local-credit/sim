import json
import math
import time
import numpy as np
import streamlit as st
import pandas as pd

from sim.config import ScenarioConfig
from sim.engine import SimulationEngine

st.set_page_config(page_title="CLC Pool Network Simulator", layout="wide")


def get_engine() -> SimulationEngine:
    if "engine" not in st.session_state:
        cfg = ScenarioConfig()
        st.session_state.cfg = cfg
        st.session_state.seed = 1
        st.session_state.engine = SimulationEngine(cfg=cfg, seed=st.session_state.seed)
    return st.session_state.engine


def reset_engine(reset_config: bool = False) -> None:
    if reset_config:
        cfg = ScenarioConfig()
        seed = 1
        st.session_state.cfg = cfg
        st.session_state.seed = seed
    else:
        cfg = st.session_state.get("cfg", ScenarioConfig())
        seed = st.session_state.get("seed", 1)
    st.session_state.engine = SimulationEngine(cfg=cfg, seed=seed)


engine = get_engine()

st.title("CLC Pool Network Simulator (MVP-1)")
st.caption("Time model: 1 tick = 1 week (4 ticks = 1 month).")

def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    mins = int(seconds // 60)
    secs = seconds - (mins * 60)
    return f"{mins}m {secs:0.1f}s"

def _fmt(value: float) -> str:
    return f"{float(value):,.2f}"

def _render_kpi_grid(kpis, columns: int = 5) -> None:
    for idx in range(0, len(kpis), columns):
        row = kpis[idx: idx + columns]
        cols = st.columns(columns)
        for col, (label, value) in zip(cols, row):
            col.metric(label, value)

def _format_table_numbers(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    numeric_cols = formatted.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return formatted
    formatted[numeric_cols] = formatted[numeric_cols].applymap(
        lambda value: f"{value:,.2f}" if pd.notnull(value) else ""
    )
    return formatted

def _metrics_has_tick(rows: list[dict], tick: int) -> bool:
    if not rows:
        return False
    return rows[-1].get("tick") == tick

def _ensure_metrics_snapshot(engine: SimulationEngine) -> None:
    need_network = not _metrics_has_tick(engine.metrics.network_rows, engine.tick)
    need_pool = not _metrics_has_tick(engine.metrics.pool_rows, engine.tick)
    if need_network or need_pool:
        engine.snapshot_metrics(force_network=need_network, force_pool=need_pool)

def _fee_total_usd(pool, ledger: dict) -> float:
    total = 0.0
    for asset_id, amt in ledger.items():
        value = pool.values.get_value(asset_id)
        if value > 0.0 and amt > 0.0:
            total += amt * value
    return total

def _format_event_meta(meta) -> str:
    if meta is None:
        return ""
    if isinstance(meta, str):
        return meta
    try:
        return json.dumps(meta, sort_keys=True)
    except TypeError:
        return str(meta)

BATCH_PARAM_SPECS = [
    ("stable_supply_cap", "Supply cap (USD)", float),
    ("stable_supply_growth_rate", "Growth rate (per month)", float),
    ("stable_outflow_rate", "Outflow rate (per month)", float),
    ("stable_growth_smoothing", "Target adjustment per tick", float),
    ("stable_inflow_activity_share", "Activity weight in inflow", float),
    ("voucher_inflow_share", "Voucher mint share", float),
    ("stable_flow_loan_scale", "Loan flow scale", float),
    ("stable_flow_swap_scale", "Swap flow scale", float),
    ("stable_flow_swap_target_usd", "Swap volume target (USD)", float),
    ("random_route_requests_per_tick", "Max swaps per pool / tick", int),
    ("producer_offramp_rate_per_month", "Producer offramp rate (per month)", float),
    ("consumer_offramp_rate_per_month", "Consumer offramp rate (per month)", float),
    ("max_hops", "Max hops", int),
    ("swap_size_mean_frac", "Swap size mean frac", float),
    ("swap_size_min_usd", "Swap size min USD", float),
    ("utilization_target_rate", "Utilization target", float),
    ("utilization_boost_max", "Utilization boost max", float),
    ("base_redeem_prob", "Base redeem probability", float),
    ("sticky_route_bias", "Route stickiness", float),
    ("core_ops_budget_usd", "Core ops budget (USD)", float),
    ("insurance_target_multiplier", "Insurance target multiplier", float),
    ("insurance_max_topup_usd", "Insurance max top-up (USD)", float),
    ("liquidity_mandate_share", "Liquidity mandate share", float),
    ("incident_base_rate", "Incident base rate", float),
    ("clc_rebalance_target_stable_ratio", "CLC target stable ratio", float),
    ("clc_rebalance_max_swaps_per_tick", "CLC max rebalance swaps", int),
]

def _parse_sweep_values(text: str, value_type: type) -> list:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if value_type is int:
                values.append(int(float(part)))
            else:
                values.append(float(part))
        except ValueError:
            continue
    return values

def _pool_balances(engine: SimulationEngine) -> dict:
    balances = {}
    stable_id = engine.cfg.stable_symbol
    for pid, pool in engine.pools.items():
        stable = pool.vault.get(stable_id)
        vouchers = sum(
            amt for asset, amt in pool.vault.inventory.items()
            if asset.startswith("VCHR:")
        )
        balances[pid] = {"stable": stable, "vouchers": vouchers}
    return balances

def _pool_tooltips(engine: SimulationEngine, nodes, max_assets: int = 12, max_swaps: int = 5) -> dict:
    tooltips = {}
    stable_id = engine.cfg.stable_symbol
    for pid in nodes:
        pool = engine.pools.get(pid)
        if pool is None:
            continue
        lines = [
            f"Pool: {pid}",
            f"Role: {pool.policy.role} | Mode: {pool.policy.mode}",
        ]

        stable = pool.vault.get(stable_id)
        voucher_total = sum(
            amt for asset, amt in pool.vault.inventory.items()
            if asset.startswith("VCHR:")
        )
        lines.append(f"Stable: {stable:.2f}")
        lines.append(f"Vouchers: {voucher_total:.2f}")

        inv = sorted(pool.vault.inventory.items(), key=lambda kv: kv[1], reverse=True)
        lines.append("Assets:")
        if inv:
            for asset, amt in inv[:max_assets]:
                lines.append(f"  {asset}: {amt:.2f}")
            if len(inv) > max_assets:
                lines.append(f"  ... +{len(inv) - max_assets} more")
        else:
            lines.append("  (empty)")

        rules = sorted(pool.limiter.rules.items(), key=lambda kv: kv[0])
        lines.append("Limits:")
        if not pool.policy.limits_enabled:
            lines.append("  (disabled)")
        elif rules:
            for asset, rule in rules[:max_assets]:
                lines.append(f"  {asset}: cap {rule.cap_in_global:.0f} / {rule.window_len_ticks}t")
            if len(rules) > max_assets:
                lines.append(f"  ... +{len(rules) - max_assets} more")
        else:
            lines.append("  (none)")

        recs = pool.receipts.tail(max_swaps)
        lines.append("Last swaps:")
        if recs:
            for r in recs:
                lines.append(
                    f"  t{r.tick} {r.asset_in}->{r.asset_out} "
                    f"{r.amount_in:.1f}->{r.amount_out:.1f} {r.status}"
                )
        else:
            lines.append("  (none)")

        tooltips[pid] = "\n".join(lines)
    return tooltips

def _extract_swap_edges(events, tick_min: int, tick_max: int) -> dict:
    edges = {}
    for e in events:
        if e.event_type != "SWAP_EXECUTED":
            continue
        if e.tick < tick_min or e.tick > tick_max:
            continue
        receipt = e.meta.get("receipt", {})
        actor = receipt.get("actor") or e.actor_id
        if not actor:
            continue
        if actor.startswith("escrow:"):
            source = actor.split(":", 1)[1]
        else:
            source = actor
        target = receipt.get("pool_id") or e.pool_id
        if not source or not target or source == target:
            continue
        a, b = (source, target) if source < target else (target, source)
        edges[(a, b)] = edges.get((a, b), 0) + 1
    return edges

def _spring_layout(nodes, edges, width: int, height: int) -> dict:
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (width * 0.5, height * 0.5)}

    idx = {node: i for i, node in enumerate(nodes)}
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(n, 2))
    pos /= np.linalg.norm(pos, axis=1, keepdims=True) + 1e-9
    pos *= 100.0

    edges_idx = []
    weights = []
    for (a, b), w in edges.items():
        if a not in idx or b not in idx:
            continue
        edges_idx.append((idx[a], idx[b]))
        weights.append(float(w))
    if not edges_idx:
        for i in range(n):
            angle = (2.0 * math.pi * i) / n
            pos[i] = np.array([math.cos(angle), math.sin(angle)]) * 120.0

    edges_idx = np.array(edges_idx, dtype=int) if edges_idx else np.empty((0, 2), dtype=int)
    weights = np.array(weights, dtype=float) if weights else np.empty((0,), dtype=float)
    spring_weights = np.log1p(weights) if weights.size else weights

    base_len = 140.0
    iters = 40 if n <= 300 else 12
    for _ in range(iters):
        if edges_idx.size:
            i = edges_idx[:, 0]
            j = edges_idx[:, 1]
            delta = pos[j] - pos[i]
            dist = np.linalg.norm(delta, axis=1) + 1e-6
            desired = base_len / (1.0 + spring_weights)
            force = (dist - desired) * 0.01
            step = (delta / dist[:, None]) * (force * spring_weights)[:, None]
            max_step = base_len * 0.5
            np.clip(step, -max_step, max_step, out=step)
            pos[i] += step
            pos[j] -= step

        if n <= 300:
            diff = pos[:, None, :] - pos[None, :, :]
            dist2 = np.sum(diff * diff, axis=2) + np.eye(n) * 1e9
            rep = diff / np.sqrt(dist2)[:, :, None]
            pos += np.sum(rep / dist2[:, :, None], axis=1) * 2.5

        pos -= pos.mean(axis=0)
        pos *= 0.98

    if not np.isfinite(pos).all():
        # fallback to circle layout if layout diverged
        for i in range(n):
            angle = (2.0 * math.pi * i) / n
            pos[i] = np.array([math.cos(angle), math.sin(angle)]) * 120.0

    min_xy = pos.min(axis=0)
    max_xy = pos.max(axis=0)
    if not np.isfinite(min_xy).all() or not np.isfinite(max_xy).all() or np.allclose(max_xy, min_xy):
        for i in range(n):
            angle = (2.0 * math.pi * i) / n
            pos[i] = np.array([math.cos(angle), math.sin(angle)]) * 120.0
        min_xy = pos.min(axis=0)
        max_xy = pos.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = 0.9 * min(width / span[0], height / span[1])
    pos = (pos - min_xy) * scale
    pos[:, 0] += (width - (span[0] * scale)) * 0.5
    pos[:, 1] += (height - (span[1] * scale)) * 0.5

    return {node: (float(pos[i, 0]), float(pos[i, 1])) for node, i in idx.items()}

def _render_swaps_svg(positions, edges, balances, tooltips, width: int, height: int) -> str:
    if not positions:
        return ""

    max_edge = max(edges.values()) if edges else 1.0
    max_stable = max(b["stable"] for b in balances.values()) if balances else 1.0
    max_voucher = max(b["vouchers"] for b in balances.values()) if balances else 1.0

    def node_radius(stable):
        return 6.0 + 18.0 * math.log1p(stable) / math.log1p(max_stable)

    def node_color(vouchers):
        t = 0.0 if max_voucher <= 0 else min(1.0, vouchers / max_voucher)
        hue = 210.0 - (190.0 * t)
        return f"hsl({hue:.1f}, 70%, 55%)"

    lines = []
    for (a, b), w in edges.items():
        if a not in positions or b not in positions:
            continue
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        width_main = 0.5 + 2.5 * (w / max_edge)
        width_glow = width_main * 2.6
        alpha = 0.15 + 0.35 * (w / max_edge)
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="rgba(120,200,255,{alpha:.3f})" stroke-width="{width_glow:.2f}" />'
        )
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="rgba(200,240,255,0.85)" stroke-width="{width_main:.2f}" />'
        )

    def _escape_svg_text(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "&#10;")
        )

    nodes = []
    for node, (x, y) in positions.items():
        bal = balances.get(node, {"stable": 0.0, "vouchers": 0.0})
        r = node_radius(bal["stable"])
        color = node_color(bal["vouchers"])
        inner = max(3.0, r * (0.35 + 0.45 * (0.0 if max_voucher <= 0 else min(1.0, bal["vouchers"] / max_voucher))))
        tooltip = tooltips.get(node, node)
        nodes.append(
            "<g>"
            f"<title>{_escape_svg_text(tooltip)}</title>"
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" fill-opacity="0.9" />'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{inner:.1f}" fill="rgba(255,255,255,0.35)" />'
            "</g>"
        )

    svg = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="background: radial-gradient(circle at 30% 20%, #0f1e2e 0%, #09131c 45%, #070b0f 100%);">',
        "".join(lines),
        "".join(nodes),
        "</svg>",
    ]
    return "\n".join(svg)

with st.sidebar:
    st.header("Sim Controls")

    st.subheader("Run")
    if st.button("Restart simulation"):
        reset_engine(reset_config=True)
        engine = st.session_state.engine
        st.session_state.run_progress = 0.0
        st.session_state.run_progress_label = "Idle"
        st.session_state.batch_results = None
    st.caption("Restart resets the simulation to tick 0 with default settings.")
    if "seed" not in st.session_state:
        st.session_state.seed = 1
    st.number_input(
        "Random seed",
        min_value=1,
        max_value=100000,
        key="seed",
    )

    run_ticks = st.slider("Ticks to run", min_value=1, max_value=500, value=25)
    c3, c4 = st.columns(2)
    run_one = c3.button("Step 1 tick")
    run_many = c4.button("Run N ticks")
    progress_label = st.session_state.get("run_progress_label", "Idle")
    progress_value = float(st.session_state.get("run_progress", 0.0))
    progress_bar = st.progress(progress_value, text=progress_label)
    if run_one:
        progress_bar.progress(0.0, text="Run progress: 0%")
        start_ts = time.time()
        engine.step(1)
        _ensure_metrics_snapshot(engine)
        elapsed = time.time() - start_ts
        progress_bar.progress(1.0, text=f"Run progress: 100% ({_fmt_duration(elapsed)})")
        st.session_state.run_progress = 1.0
        st.session_state.run_progress_label = f"Run progress: 100% ({_fmt_duration(elapsed)})"
    if run_many:
        total = int(run_ticks)
        if total > 0:
            start_ts = time.time()
            for idx in range(total):
                engine.step(1)
                progress = (idx + 1) / total
                progress_bar.progress(progress, text=f"Run progress: {progress:.0%}")
            elapsed = time.time() - start_ts
            _ensure_metrics_snapshot(engine)
            st.session_state.run_progress = 1.0
            st.session_state.run_progress_label = f"Run progress: 100% ({_fmt_duration(elapsed)})"
            progress_bar.progress(1.0, text=st.session_state.run_progress_label)
    st.caption(f"Current tick: {engine.tick}")

    st.subheader("Growth")
    c1, c2, c3 = st.columns(3)
    if c1.button("Add 1 pool"):
        engine.add_pool()
    if c2.button("Add 1 LP"):
        engine.add_pool(role="liquidity_provider")
    if c3.button("Add 10 pools"):
        for _ in range(10):
            engine.add_pool(snapshot=False, rebuild_indexes=False)
        engine.rebuild_indexes()
        engine.snapshot_metrics()
    active_pools = sum(1 for p in engine.pools.values() if not p.policy.system_pool)
    engine.cfg.pool_growth_rate_per_tick = st.number_input(
        "Pool growth rate (per tick)",
        min_value=0.0,
        value=float(engine.cfg.pool_growth_rate_per_tick),
        step=0.005,
        help="Adds new pools each tick by this % of active pools (0.02 = 2%).",
    )
    max_pools = st.number_input(
        "Max pools (0 = no cap)",
        min_value=0,
        value=int(engine.cfg.max_pools or 0),
        step=100,
        help="Cap active pools to keep long runs from exploding in size.",
    )
    engine.cfg.max_pools = None if max_pools <= 0 else int(max_pools)
    if engine.cfg.max_pools is not None:
        st.caption(f"Current pools: {active_pools} / {engine.cfg.max_pools}")
    else:
        st.caption(f"Current pools: {active_pools}")
    engine.cfg.p_liquidity_provider = st.slider(
        "Liquidity provider share",
        0.0,
        1.0,
        float(engine.cfg.p_liquidity_provider),
        step=0.05,
        help="Role weight relative to lender/producer/consumer.",
    )

    st.subheader("Batch Runs")
    st.caption("Sweep one parameter across multiple runs to compare outcomes.")
    runs_per_value = st.number_input(
        "Runs per value", min_value=1, max_value=50, value=3, step=1
    )
    ticks_per_run = st.number_input(
        "Ticks per run", min_value=1, max_value=1000, value=20, step=1
    )
    base_seed = st.number_input(
        "Base seed",
        min_value=1,
        max_value=100000,
        value=int(st.session_state.get("seed", 1)),
        step=1,
        key="batch_seed",
    )

    label_map = {label: (name, typ) for name, label, typ in BATCH_PARAM_SPECS}
    param_labels = list(label_map.keys())
    selected_label = st.selectbox("Parameter to sweep", param_labels)
    param_name, param_type = label_map[selected_label]
    current_value = getattr(engine.cfg, param_name)
    values_text = st.text_input(
        "Values (comma-separated)",
        value=str(current_value),
        key="batch_values",
    )

    if st.button("Run batch"):
        values = _parse_sweep_values(values_text, param_type)
        if not values:
            st.warning("Enter at least one numeric value.")
        else:
            results = []
            with st.spinner("Running batch simulations..."):
                for value in values:
                    for idx in range(int(runs_per_value)):
                        cfg_dict = vars(engine.cfg).copy()
                        cfg_dict[param_name] = value
                        cfg = ScenarioConfig(**cfg_dict)
                        seed = int(base_seed) + idx
                        sim = SimulationEngine(cfg=cfg, seed=seed)
                        sim.step(int(ticks_per_run))
                        net_df = sim.metrics.network_df()
                        latest = net_df.iloc[-1].to_dict() if not net_df.empty else {}
                        results.append({
                            "param": param_name,
                            "value": value,
                            "run": idx + 1,
                            "seed": seed,
                            "tick": sim.tick,
                            "num_pools": latest.get("num_pools", 0),
                            "stable_total_in_pools": latest.get("stable_total_in_pools", 0.0),
                            "redeemed_total": latest.get("redeemed_total", 0.0),
                            "outstanding_voucher_supply_total": latest.get("outstanding_voucher_supply_total", 0.0),
                            "loan_issuance_volume_usd": latest.get("loan_issuance_volume_usd", 0.0),
                            "repayment_volume_usd": latest.get("repayment_volume_usd", 0.0),
                            "utilization_rate": latest.get("utilization_rate", 0.0),
                        })
            st.session_state.batch_results = results

tab_network, tab_clc, tab_noam_overview, tab2, tab3, tab4, tab_network_controls, tab_noam_controls, tab_clc_controls = st.tabs(
    [
        "Network Overview",
        "CLC Overview",
        "NOAM Overview",
        "Pools",
        "Events",
        "Swaps Graph",
        "Network Controls",
        "NOAM Routing & Clearing",
        "CLC Controls",
    ]
)

with tab_network_controls:
    right = st.container()
    with right:
        st.subheader("Stable Supply")
        growth_mode = st.selectbox(
            "Stable growth mode",
            ["per_pool", "network_target"],
            index=["per_pool", "network_target"].index(engine.cfg.stable_growth_mode)
            if engine.cfg.stable_growth_mode in ("per_pool", "network_target")
            else 0,
            help="per_pool compounds per pool; network_target steers total supply toward a cap.",
        )
        engine.cfg.stable_growth_mode = growth_mode
        if growth_mode == "per_pool":
            engine.cfg.producer_inflow_per_tick = st.number_input(
                "Producer inflow (per month, smoothed per tick)",
                min_value=0.0,
                value=float(engine.cfg.producer_inflow_per_tick or 0.0),
                step=0.01,
            )
            engine.cfg.consumer_inflow_per_tick = st.number_input(
                "Consumer inflow (per month, smoothed per tick)",
                min_value=0.0,
                value=float(engine.cfg.consumer_inflow_per_tick or 0.0),
                step=0.01,
            )
            engine.cfg.lender_inflow_per_tick = st.number_input(
                "Lender inflow (per month, smoothed per tick)",
                min_value=0.0,
                value=float(engine.cfg.lender_inflow_per_tick or 0.0),
                step=0.01,
            )
            engine.cfg.liquidity_provider_inflow_per_tick = st.number_input(
                "Liquidity provider inflow (per month, smoothed per tick)",
                min_value=0.0,
                value=float(engine.cfg.liquidity_provider_inflow_per_tick or 0.0),
                step=0.01,
            )
            engine.cfg.stable_inflow_per_tick = st.number_input(
                "Generic inflow (per month, smoothed per tick)",
                min_value=0.0,
                value=float(engine.cfg.stable_inflow_per_tick or 0.0),
                step=0.01,
            )
        else:
            engine.cfg.stable_supply_cap = st.number_input(
                "Supply cap (USD)",
                min_value=0.0,
                value=float(engine.cfg.stable_supply_cap),
                step=1000.0,
            )
            engine.cfg.stable_supply_growth_rate = st.number_input(
                "Growth rate (per month)",
                min_value=0.0,
                value=float(engine.cfg.stable_supply_growth_rate),
                step=0.01,
            )
            engine.cfg.stable_supply_noise = st.number_input(
                "Supply noise (stdev)",
                min_value=0.0,
                value=float(engine.cfg.stable_supply_noise),
                step=0.01,
            )
            engine.cfg.stable_outflow_rate = st.number_input(
                "Outflow rate (per month)",
                min_value=0.0,
                value=float(engine.cfg.stable_outflow_rate),
                step=0.01,
            )
            engine.cfg.stable_growth_smoothing = st.slider(
                "Target adjustment per tick",
                0.0,
                1.0,
                float(engine.cfg.stable_growth_smoothing),
                step=0.05,
                help="Fraction of the gap to target applied each tick.",
            )
            engine.cfg.voucher_inflow_share = st.number_input(
                "Voucher mint share (per USD stable inflow)",
                min_value=0.0,
                value=float(engine.cfg.voucher_inflow_share),
                step=0.05,
                help="Minted voucher value distributed alongside stable inflow.",
            )
            engine.cfg.stable_inflow_activity_share = st.slider(
                "Activity weight in inflow",
                0.0,
                1.0,
                float(engine.cfg.stable_inflow_activity_share),
                step=0.05,
                help="Blend reserve deficits with recent swap activity when distributing new stable.",
            )
            engine.cfg.stable_inflow_activity_window_ticks = st.number_input(
                "Activity window (ticks)",
                min_value=1,
                max_value=52,
                value=int(engine.cfg.stable_inflow_activity_window_ticks),
                step=1,
            )
        engine.cfg.stable_growth_stride_ticks = st.number_input(
            "Stable growth stride (ticks)",
            min_value=1,
            value=int(engine.cfg.stable_growth_stride_ticks),
            step=1,
            help="Apply stable growth every N ticks (scaled to preserve the rate).",
        )

        st.subheader("Issuance/Redemption")
        flow_mode = st.selectbox(
            "Issuance/redemption driver",
            ["none", "loan", "swap", "both"],
            index=["none", "loan", "swap", "both"].index(engine.cfg.stable_flow_mode)
            if engine.cfg.stable_flow_mode in ("none", "loan", "swap", "both")
            else 0,
            help="Ties net stable issuance to loan activity and/or swap volume.",
        )
        engine.cfg.stable_flow_mode = flow_mode
        engine.cfg.stable_flow_window_ticks = st.number_input(
            "Flow window (ticks)",
            min_value=1,
            max_value=52,
            value=int(engine.cfg.stable_flow_window_ticks),
            step=1,
        )
        if flow_mode in ("loan", "both"):
            engine.cfg.stable_flow_loan_scale = st.number_input(
                "Loan flow scale (USD per USD net issuance)",
                min_value=0.0,
                value=float(engine.cfg.stable_flow_loan_scale),
                step=0.05,
            )
        if flow_mode in ("swap", "both"):
            engine.cfg.stable_flow_swap_target_usd = st.number_input(
                "Swap volume target (USD per window)",
                min_value=0.0,
                value=float(engine.cfg.stable_flow_swap_target_usd),
                step=1000.0,
            )
            engine.cfg.stable_flow_swap_scale = st.number_input(
                "Swap flow scale (USD per USD above/below target)",
                value=float(engine.cfg.stable_flow_swap_scale),
                step=0.01,
            )

        st.subheader("Offramping")
        engine.cfg.producer_offramp_rate_per_month = st.number_input(
            "Producer offramp rate (per month)",
            min_value=0.0,
            value=float(engine.cfg.producer_offramp_rate_per_month),
            step=0.01,
            help="Monthly fraction of producer stable cashed out to fiat.",
        )
        engine.cfg.consumer_offramp_rate_per_month = st.number_input(
            "Consumer offramp rate (per month)",
            min_value=0.0,
            value=float(engine.cfg.consumer_offramp_rate_per_month),
            step=0.01,
            help="Monthly fraction of consumer stable cashed out to fiat.",
        )
        engine.cfg.offramps_enabled = st.checkbox(
            "Enable swap-driven offramping",
            value=bool(engine.cfg.offramps_enabled),
            help="Pools offramp stable to fiat when swap attempts fail; success lowers offramp rate.",
        )
        if engine.cfg.offramps_enabled:
            engine.cfg.offramp_rate_min_per_tick = st.number_input(
                "Offramp min rate (per tick)",
                min_value=0.0,
                value=float(engine.cfg.offramp_rate_min_per_tick),
                step=0.001,
            )
            engine.cfg.offramp_rate_max_per_tick = st.number_input(
                "Offramp max rate (per tick)",
                min_value=0.0,
                value=float(engine.cfg.offramp_rate_max_per_tick),
                step=0.001,
            )
            engine.cfg.offramp_success_ema_alpha = st.slider(
                "Offramp success EMA alpha",
                0.0,
                1.0,
                float(engine.cfg.offramp_success_ema_alpha),
                step=0.05,
                help="Higher responds faster to recent successes/failures.",
            )
            engine.cfg.offramp_min_attempts = st.number_input(
                "Min swap attempts per tick",
                min_value=0,
                value=int(engine.cfg.offramp_min_attempts),
                step=1,
            )

        st.subheader("Desired Assets")
        engine.cfg.desired_assets_min_per_pool = st.number_input(
            "Min desired assets per pool",
            min_value=0,
            value=int(engine.cfg.desired_assets_min_per_pool),
            step=1,
        )
        engine.cfg.desired_assets_max_per_pool = st.number_input(
            "Max desired assets per pool",
            min_value=0,
            value=int(engine.cfg.desired_assets_max_per_pool),
            step=1,
        )
        engine.cfg.desired_assets_growth_per_asset = st.number_input(
            "Desired assets growth per network asset",
            min_value=0.0,
            value=float(engine.cfg.desired_assets_growth_per_asset),
            step=0.05,
            help="Each new asset in the network raises the desired listing target.",
        )
        engine.cfg.desired_assets_add_per_tick = st.number_input(
            "Max new desired assets per pool per tick",
            min_value=0,
            value=int(engine.cfg.desired_assets_add_per_tick),
            step=1,
        )

        st.subheader("Routing")
        engine.cfg.max_hops = st.slider("Max hops", 1, 6, int(engine.cfg.max_hops))
        engine.router.max_hops = engine.cfg.max_hops
        engine.cfg.random_route_requests_per_tick = st.slider(
            "Max swaps per pool / tick", 0, 10, int(engine.cfg.random_route_requests_per_tick)
        )
        engine.cfg.swap_asset_selection_mode = st.selectbox(
            "Swap asset selection",
            ["uniform", "value_weighted"],
            index=["uniform", "value_weighted"].index(engine.cfg.swap_asset_selection_mode)
            if engine.cfg.swap_asset_selection_mode in ("uniform", "value_weighted")
            else 0,
            help="value_weighted prioritizes assets with larger USD value in the pool.",
        )
        engine.cfg.swap_target_selection_mode = st.selectbox(
            "Swap target selection",
            ["uniform", "liquidity_weighted"],
            index=["uniform", "liquidity_weighted"].index(engine.cfg.swap_target_selection_mode)
            if engine.cfg.swap_target_selection_mode in ("uniform", "liquidity_weighted")
            else 0,
            help="liquidity_weighted prioritizes assets with deeper network liquidity.",
        )
        engine.cfg.swap_target_retry_count = st.number_input(
            "Swap target retries per asset",
            min_value=1,
            value=int(engine.cfg.swap_target_retry_count),
            step=1,
            help="Additional target choices if no route is found.",
        )
        engine.cfg.swap_attempts_value_scale_usd = st.number_input(
            "Extra swap attempts per USD",
            min_value=0.0,
            value=float(engine.cfg.swap_attempts_value_scale_usd),
            step=10000.0,
            help="Additional attempts per pool scale with pool value (value/this).",
        )
        engine.cfg.swap_attempts_max_per_pool = st.number_input(
            "Max swap attempts per pool (0 = unlimited)",
            min_value=0,
            value=int(engine.cfg.swap_attempts_max_per_pool or 0),
            step=1,
        )
        if engine.cfg.swap_attempts_max_per_pool == 0:
            engine.cfg.swap_attempts_max_per_pool = None
        engine.cfg.utilization_target_rate = st.number_input(
            "Utilization target (swap volume / pool value)",
            min_value=0.0,
            max_value=1.0,
            value=float(engine.cfg.utilization_target_rate),
            step=0.005,
            help="Boost swap attempts when utilization falls below this target.",
        )
        engine.cfg.utilization_boost_max = st.number_input(
            "Max utilization boost (attempts multiplier)",
            min_value=1.0,
            value=float(engine.cfg.utilization_boost_max),
            step=0.1,
        )
        engine.cfg.swap_limits_enabled = st.checkbox(
            "Enable swap caps",
            value=bool(engine.cfg.swap_limits_enabled),
            help="When off, per-asset swap caps are ignored.",
        )
        for pool in engine.pools.values():
            if pool.policy.system_pool:
                continue
            pool.policy.limits_enabled = engine.cfg.swap_limits_enabled

        st.subheader("Stickiness")
        engine.cfg.sticky_route_bias = st.slider(
            "Route stickiness",
            0.0,
            1.0,
            float(engine.cfg.sticky_route_bias),
            step=0.05,
            help="Bias routing toward pools that swapped successfully with each other in the past.",
        )
        engine.cfg.sticky_affinity_decay = st.slider(
            "Affinity decay (per tick)",
            0.0,
            0.2,
            float(engine.cfg.sticky_affinity_decay),
            step=0.01,
        )
        engine.cfg.sticky_affinity_gain = st.number_input(
            "Affinity gain",
            min_value=0.0,
            value=float(engine.cfg.sticky_affinity_gain),
            step=0.05,
        )
        engine.cfg.sticky_affinity_cap = st.number_input(
            "Affinity cap",
            min_value=1.0,
            value=float(engine.cfg.sticky_affinity_cap),
            step=1.0,
        )

        st.subheader("Redemption")
        engine.cfg.base_redeem_prob = st.slider(
            "Base redeem probability", 0.0, 1.0, float(engine.cfg.base_redeem_prob)
        )

        st.subheader("Loans")
        engine.cfg.loan_activity_period_ticks = st.number_input(
            "Loan activity period (ticks)",
            min_value=1,
            max_value=52,
            value=int(engine.cfg.loan_activity_period_ticks),
            step=1,
            help="Producers only issue/repay on their assigned tick within this period.",
        )

        st.subheader("Performance")
        metrics_stride = st.number_input(
            "Network metrics stride (ticks, 0 disables)",
            min_value=0,
            max_value=1000,
            value=int(engine.cfg.metrics_stride or 0),
            step=1,
        )
        engine.cfg.metrics_stride = int(metrics_stride)
        pool_stride = st.number_input(
            "Pool metrics stride (ticks, 0 disables)",
            min_value=0,
            max_value=1000,
            value=int(engine.cfg.pool_metrics_stride or 0),
            step=1,
        )
        engine.cfg.pool_metrics_stride = int(pool_stride)
        max_active = st.number_input(
            "Max active pools per tick (0 = all)",
            min_value=0,
            max_value=100000,
            value=int(engine.cfg.max_active_pools_per_tick or 0),
            step=100,
        )
        engine.cfg.max_active_pools_per_tick = None if max_active <= 0 else int(max_active)
        swap_budget = st.number_input(
            "Swap request budget per tick (0 = unlimited)",
            min_value=0,
            max_value=100000,
            value=int(engine.cfg.swap_requests_budget_per_tick or 0),
            step=50,
        )
        engine.cfg.swap_requests_budget_per_tick = None if swap_budget <= 0 else int(swap_budget)
        engine.cfg.pool_growth_stride_ticks = st.number_input(
            "Pool growth stride (ticks)",
            min_value=1,
            max_value=1000,
            value=int(engine.cfg.pool_growth_stride_ticks),
            step=1,
        )
        engine.cfg.desired_assets_stride_ticks = st.number_input(
            "Desired assets stride (ticks)",
            min_value=1,
            max_value=1000,
            value=int(engine.cfg.desired_assets_stride_ticks),
            step=1,
        )
        max_candidates = st.number_input(
            "Max candidate pools per hop (0 = all)",
            min_value=0,
            max_value=100000,
            value=int(engine.cfg.max_candidate_pools_per_hop or 0),
            step=100,
        )
        engine.cfg.max_candidate_pools_per_hop = None if max_candidates <= 0 else int(max_candidates)
        log_maxlen = st.number_input(
            "Event log max size (0 = unlimited)",
            min_value=0,
            max_value=200000,
            value=int(engine.cfg.event_log_maxlen or 0),
            step=1000,
        )
        engine.cfg.event_log_maxlen = None if log_maxlen <= 0 else int(log_maxlen)
        st.caption("Event log size applies after reset.")

    if st.session_state.get("batch_results"):
        st.subheader("Batch Results")
        st.dataframe(pd.DataFrame(st.session_state.batch_results), use_container_width=True)

with tab_noam_controls:
    right = st.container()
    with right:
        st.subheader("NOAM Routing")
        engine.cfg.routing_mode = st.selectbox(
            "Routing mode",
            ["noam", "bfs"],
            index=["noam", "bfs"].index(engine.cfg.routing_mode)
            if engine.cfg.routing_mode in ("noam", "bfs")
            else 0,
            help="noam uses Top-K/Top-M beam search with network-aware scoring; bfs is the legacy router.",
        )
        if engine.cfg.routing_mode == "noam":
            st.subheader("NOAM Routing (Phase 1)")
            engine.cfg.noam_topk_pools_per_asset = st.number_input(
                "NOAM Top-K pools per asset",
                min_value=1,
                value=int(engine.cfg.noam_topk_pools_per_asset),
                step=1,
            )
            engine.cfg.noam_topm_out_per_pool = st.number_input(
                "NOAM Top-M outs per pool",
                min_value=1,
                value=int(engine.cfg.noam_topm_out_per_pool),
                step=1,
            )
            engine.cfg.noam_beam_width = st.number_input(
                "NOAM beam width",
                min_value=1,
                value=int(engine.cfg.noam_beam_width),
                step=5,
            )
            engine.cfg.noam_max_hops = st.number_input(
                "NOAM max hops",
                min_value=1,
                value=int(engine.cfg.noam_max_hops),
                step=1,
            )
            engine.cfg.noam_topk_refresh_ticks = st.number_input(
                "NOAM Top-K refresh (ticks)",
                min_value=1,
                value=int(engine.cfg.noam_topk_refresh_ticks),
                step=1,
            )
            engine.cfg.noam_success_ema_alpha = st.slider(
                "NOAM success EMA alpha",
                0.0,
                1.0,
                float(engine.cfg.noam_success_ema_alpha),
                step=0.05,
            )
            engine.cfg.noam_scarcity_eta = st.number_input(
                "NOAM scarcity eta",
                min_value=0.0,
                value=float(engine.cfg.noam_scarcity_eta),
                step=0.01,
            )
            engine.cfg.noam_safe_budget_fraction = st.slider(
                "NOAM safe budget fraction",
                0.0,
                1.0,
                float(engine.cfg.noam_safe_budget_fraction),
                step=0.05,
            )
            engine.cfg.noam_weight_success = st.number_input(
                "NOAM weight: success",
                min_value=0.0,
                value=float(engine.cfg.noam_weight_success),
                step=0.1,
            )
            engine.cfg.noam_weight_fee = st.number_input(
                "NOAM weight: fee",
                min_value=0.0,
                value=float(engine.cfg.noam_weight_fee),
                step=0.1,
            )
            engine.cfg.noam_weight_lambda = st.number_input(
                "NOAM weight: scarcity",
                min_value=0.0,
                value=float(engine.cfg.noam_weight_lambda),
                step=0.1,
            )
            engine.cfg.noam_weight_benefit = st.number_input(
                "NOAM weight: benefit",
                min_value=0.0,
                value=float(engine.cfg.noam_weight_benefit),
                step=0.1,
            )
            engine.cfg.noam_weight_deadend = st.number_input(
                "NOAM weight: dead-end",
                min_value=0.0,
                value=float(engine.cfg.noam_weight_deadend),
                step=0.1,
            )
            engine.cfg.noam_clc_edge_bonus = st.number_input(
                "NOAM CLC edge bonus",
                min_value=0.0,
                value=float(engine.cfg.noam_clc_edge_bonus),
                step=0.1,
                help="Score bonus applied when routing or clearing through the CLC pool.",
            )
            st.subheader("NOAM Adaptive Caps")
            engine.cfg.noam_dynamic_caps_enabled = st.checkbox(
                "Enable adaptive caps",
                value=bool(engine.cfg.noam_dynamic_caps_enabled),
            )
            engine.cfg.noam_dynamic_cap_reference_pools = st.number_input(
                "Adaptive cap reference pools",
                min_value=1,
                value=int(engine.cfg.noam_dynamic_cap_reference_pools),
                step=10,
            )
            engine.cfg.noam_dynamic_min_topk = st.number_input(
                "Adaptive min Top-K",
                min_value=1,
                value=int(engine.cfg.noam_dynamic_min_topk),
                step=1,
            )
            engine.cfg.noam_dynamic_min_topm = st.number_input(
                "Adaptive min Top-M",
                min_value=1,
                value=int(engine.cfg.noam_dynamic_min_topm),
                step=1,
            )
            engine.cfg.noam_dynamic_min_beam = st.number_input(
                "Adaptive min beam width",
                min_value=1,
                value=int(engine.cfg.noam_dynamic_min_beam),
                step=1,
            )
            engine.cfg.noam_edge_cap_per_state = st.number_input(
                "Edge cap per state",
                min_value=0,
                value=int(engine.cfg.noam_edge_cap_per_state),
                step=5,
                help="0 disables the cap.",
            )
            engine.cfg.noam_dynamic_min_edge_cap = st.number_input(
                "Adaptive min edge cap",
                min_value=1,
                value=int(engine.cfg.noam_dynamic_min_edge_cap),
                step=5,
            )
            st.subheader("NOAM Caching (Phase 3)")
            engine.cfg.noam_failure_ttl_ticks = st.number_input(
                "Failure TTL (ticks)",
                min_value=0,
                value=int(engine.cfg.noam_failure_ttl_ticks),
                step=1,
                help="0 disables the failure cache.",
            )
            engine.cfg.noam_route_cache_ttl_ticks = st.number_input(
                "Route cache TTL (ticks)",
                min_value=0,
                value=int(engine.cfg.noam_route_cache_ttl_ticks),
                step=1,
                help="0 disables the route cache.",
            )
            engine.cfg.noam_route_cache_bucket_usd = st.number_input(
                "Route cache bucket (USD)",
                min_value=0.0,
                value=float(engine.cfg.noam_route_cache_bucket_usd),
                step=10.0,
            )
            st.subheader("NOAM Overlay (Phase 2)")
            engine.cfg.noam_overlay_enabled = st.checkbox(
                "Enable NOAM overlay routing",
                value=bool(engine.cfg.noam_overlay_enabled),
            )
            engine.cfg.noam_hub_asset_count = st.number_input(
                "NOAM hub asset count",
                min_value=1,
                value=int(engine.cfg.noam_hub_asset_count),
                step=10,
            )
            engine.cfg.noam_hub_depth = st.number_input(
                "NOAM hub search depth",
                min_value=0,
                value=int(engine.cfg.noam_hub_depth),
                step=1,
            )
            engine.cfg.noam_hub_candidate_limit = st.number_input(
                "NOAM hub candidate limit",
                min_value=1,
                value=int(engine.cfg.noam_hub_candidate_limit),
                step=1,
            )
            engine.cfg.noam_overlay_top_r_paths = st.number_input(
                "NOAM overlay top-R paths",
                min_value=1,
                value=int(engine.cfg.noam_overlay_top_r_paths),
                step=1,
            )
            engine.cfg.noam_overlay_max_hops = st.number_input(
                "NOAM overlay max hops",
                min_value=1,
                value=int(engine.cfg.noam_overlay_max_hops),
                step=1,
            )
            engine.cfg.noam_overlay_refresh_ticks = st.number_input(
                "NOAM overlay refresh (ticks)",
                min_value=1,
                value=int(engine.cfg.noam_overlay_refresh_ticks),
                step=1,
            )
            engine.cfg.noam_overlay_min_pools = st.number_input(
                "NOAM overlay min pools",
                min_value=0,
                value=int(engine.cfg.noam_overlay_min_pools),
                step=10,
                help="Overlay routing only activates when pool count reaches this threshold.",
            )
            st.subheader("NOAM Clearing")
            engine.cfg.noam_clearing_enabled = st.checkbox(
                "Enable NOAM clearing",
                value=bool(engine.cfg.noam_clearing_enabled),
            )
            engine.cfg.noam_clearing_stride_ticks = st.number_input(
                "Clearing stride (ticks)",
                min_value=1,
                value=int(engine.cfg.noam_clearing_stride_ticks),
                step=1,
            )
            engine.cfg.noam_clearing_max_cycles = st.number_input(
                "Max clearing cycles",
                min_value=1,
                value=int(engine.cfg.noam_clearing_max_cycles),
                step=1,
            )
            engine.cfg.noam_clearing_max_hops = st.number_input(
                "Clearing max hops",
                min_value=2,
                value=int(engine.cfg.noam_clearing_max_hops),
                step=1,
            )
            engine.cfg.noam_clearing_edge_cap_per_asset = st.number_input(
                "Clearing edge cap per asset",
                min_value=1,
                value=int(engine.cfg.noam_clearing_edge_cap_per_asset),
                step=1,
            )
            engine.cfg.noam_clearing_safety_factor = st.slider(
                "Clearing safety factor",
                0.1,
                1.0,
                float(engine.cfg.noam_clearing_safety_factor),
                step=0.05,
                help="Scales clearing cycle sizes to improve execution success.",
            )
            engine.cfg.noam_clearing_budget_usd = st.number_input(
                "Clearing budget (USD)",
                min_value=0.0,
                value=float(engine.cfg.noam_clearing_budget_usd),
                step=100.0,
            )
            engine.cfg.noam_clearing_budget_share = st.slider(
                "Clearing budget share (of network value)",
                0.0,
                0.10,
                float(engine.cfg.noam_clearing_budget_share),
                step=0.005,
            )
            engine.cfg.noam_clearing_min_cycle_value_usd = st.number_input(
                "Min cycle value (USD)",
                min_value=0.0,
                value=float(engine.cfg.noam_clearing_min_cycle_value_usd),
                step=10.0,
            )
        else:
            st.caption("NOAM controls are hidden when routing mode is bfs.")

with tab_clc_controls:
    right = st.container()
    with right:
        st.subheader("Fees")
        engine.cfg.pool_fee_rate = st.number_input(
            "Pool fee rate",
            min_value=0.0,
            value=float(engine.cfg.pool_fee_rate),
            step=0.001,
            help="Applied to gross output on each swap; CLC rake is taken from this fee.",
        )
        engine.cfg.clc_rake_rate = st.slider(
            "CLC rake (share of pool fee)",
            0.0,
            1.0,
            float(engine.cfg.clc_rake_rate),
            step=0.01,
        )
        for pool in engine.pools.values():
            if pool.policy.system_pool:
                continue
            pool.fees.pool_fee_rate = engine.cfg.pool_fee_rate
            pool.fees.clc_rake_rate = engine.cfg.clc_rake_rate

        st.subheader("Economics (Waterfall)")
        engine.cfg.economics_enabled = st.checkbox(
            "Enable economics layer (requires reset)",
            value=bool(engine.cfg.economics_enabled),
        )
        st.caption("Changing economics enablement requires resetting the simulation.")
        engine.cfg.waterfall_epoch_ticks = st.number_input(
            "Waterfall epoch (ticks)",
            min_value=1,
            max_value=52,
            value=int(engine.cfg.waterfall_epoch_ticks),
            step=1,
        )
        engine.cfg.waterfall_include_pool_fees = st.checkbox(
            "Include pool fees in waterfall",
            value=bool(engine.cfg.waterfall_include_pool_fees),
        )
        engine.cfg.core_ops_budget_usd = st.number_input(
            "Core ops budget (USD / epoch)",
            min_value=0.0,
            value=float(engine.cfg.core_ops_budget_usd),
            step=100.0,
        )
        engine.cfg.insurance_max_topup_usd = st.number_input(
            "Insurance max top-up (USD / epoch)",
            min_value=0.0,
            value=float(engine.cfg.insurance_max_topup_usd),
            step=100.0,
        )
        engine.cfg.insurance_target_multiplier = st.number_input(
            "Insurance target multiplier",
            min_value=0.0,
            value=float(engine.cfg.insurance_target_multiplier),
            step=0.01,
        )
        engine.cfg.liquidity_mandate_share = st.slider(
            "Liquidity mandate share (of remaining cash)",
            0.0,
            1.0,
            float(engine.cfg.liquidity_mandate_share),
            step=0.05,
        )
        engine.cfg.liquidity_mandate_bootstrap_share = st.slider(
            "Liquidity mandate bootstrap share",
            0.0,
            1.0,
            float(engine.cfg.liquidity_mandate_bootstrap_share),
            step=0.05,
        )
        engine.cfg.liquidity_mandate_bootstrap_epochs = st.number_input(
            "Liquidity mandate bootstrap epochs",
            min_value=0,
            value=int(engine.cfg.liquidity_mandate_bootstrap_epochs),
        )
        engine.cfg.liquidity_mandate_max_usd = st.number_input(
            "Liquidity mandate cap (USD / epoch, 0 = no cap)",
            min_value=0.0,
            value=float(engine.cfg.liquidity_mandate_max_usd),
            step=100.0,
        )
        mandate_modes = [
            "lender_liquidity",
            "activity_weighted",
            "deficit_weighted",
            "utilization_weighted",
        ]
        engine.cfg.liquidity_mandate_mode = st.selectbox(
            "Liquidity mandate weighting",
            mandate_modes,
            index=mandate_modes.index(engine.cfg.liquidity_mandate_mode)
            if engine.cfg.liquidity_mandate_mode in mandate_modes
            else 0,
        )
        engine.cfg.liquidity_mandate_activity_window_ticks = st.number_input(
            "Liquidity mandate activity window (ticks)",
            min_value=1,
            max_value=52,
            value=int(engine.cfg.liquidity_mandate_activity_window_ticks),
            step=1,
        )
        engine.cfg.liquidity_mandate_max_per_pool_usd = st.number_input(
            "Liquidity mandate max per pool (USD)",
            min_value=0.0,
            value=float(engine.cfg.liquidity_mandate_max_per_pool_usd),
            step=100.0,
        )
        engine.cfg.waterfall_alpha_ops_share = st.slider(
            "Waterfall alpha (ops share)",
            0.0,
            1.0,
            float(engine.cfg.waterfall_alpha_ops_share),
            step=0.05,
        )
        engine.cfg.waterfall_beta_liquidity_share = st.slider(
            "Waterfall beta (liquidity share)",
            0.0,
            1.0,
            float(engine.cfg.waterfall_beta_liquidity_share),
            step=0.05,
        )
        engine.cfg.waterfall_gamma_insurance_share = st.slider(
            "Waterfall gamma (insurance share)",
            0.0,
            1.0,
            float(engine.cfg.waterfall_gamma_insurance_share),
            step=0.05,
        )
        engine.cfg.lp_waterfall_contribution_rate = st.number_input(
            "LP waterfall contribution rate (per tick)",
            min_value=0.0,
            value=float(engine.cfg.lp_waterfall_contribution_rate),
            step=0.005,
        )
        engine.cfg.lp_sclc_supply_cap = st.number_input(
            "LP sCLC supply cap",
            min_value=0.0,
            value=float(engine.cfg.lp_sclc_supply_cap),
            step=1000.0,
        )
        cash_assets_text = st.text_input(
            "Cash-eligible assets (comma-separated)",
            value=",".join(engine.cfg.cash_eligible_assets),
        )
        cash_assets = [a.strip() for a in cash_assets_text.split(",") if a.strip()]
        if engine.cfg.stable_symbol not in cash_assets:
            cash_assets.append(engine.cfg.stable_symbol)
        engine.cfg.cash_eligible_assets = cash_assets
        engine.cfg.cash_conversion_slippage_bps = st.number_input(
            "Conversion slippage (bps)",
            min_value=0.0,
            value=float(engine.cfg.cash_conversion_slippage_bps),
            step=5.0,
        )
        engine.cfg.cash_conversion_max_usd_per_epoch = st.number_input(
            "Conversion max (USD / epoch, 0 = no cap)",
            min_value=0.0,
            value=float(engine.cfg.cash_conversion_max_usd_per_epoch or 0.0),
            step=100.0,
        )
        if engine.cfg.cash_conversion_max_usd_per_epoch == 0.0:
            engine.cfg.cash_conversion_max_usd_per_epoch = None
        engine.cfg.sclc_fee_access_enabled = st.checkbox(
            "Enable sCLC fee access",
            value=bool(engine.cfg.sclc_fee_access_enabled),
        )
        engine.cfg.sclc_requires_insurance_target = st.checkbox(
            "sCLC requires insurance target",
            value=bool(engine.cfg.sclc_requires_insurance_target),
        )
        engine.cfg.sclc_requires_core_ops = st.checkbox(
            "sCLC requires core ops funding",
            value=bool(engine.cfg.sclc_requires_core_ops),
        )
        engine.cfg.sclc_fee_access_share = st.slider(
            "sCLC fee access share (of CLC stable)",
            0.0,
            1.0,
            float(engine.cfg.sclc_fee_access_share),
            step=0.05,
        )
        engine.cfg.sclc_emission_cap_usd = st.number_input(
            "sCLC emission cap (USD / epoch)",
            min_value=0.0,
            value=float(engine.cfg.sclc_emission_cap_usd),
            step=100.0,
        )
        engine.cfg.sclc_swap_window_ticks = st.number_input(
            "sCLC swap window cadence (ticks)",
            min_value=1,
            max_value=52,
            value=int(engine.cfg.sclc_swap_window_ticks),
            step=1,
        )
        engine.cfg.sclc_swap_window_open_ticks = st.number_input(
            "sCLC swap window open (ticks)",
            min_value=0,
            max_value=52,
            value=int(engine.cfg.sclc_swap_window_open_ticks),
            step=1,
        )
        engine.cfg.clc_pool_always_open = st.checkbox(
            "Keep CLC pool open for swaps",
            value=bool(engine.cfg.clc_pool_always_open),
            help="When on, CLC pool always allows stable outflow (no access window gating).",
        )
        st.subheader("CLC Pool Rebalancing")
        engine.cfg.clc_rebalance_enabled = st.checkbox(
            "Enable CLC voucher->stable rebalancing",
            value=bool(engine.cfg.clc_rebalance_enabled),
        )
        engine.cfg.clc_rebalance_interval_ticks = st.number_input(
            "CLC rebalance interval (ticks)",
            min_value=1,
            max_value=52,
            value=int(engine.cfg.clc_rebalance_interval_ticks),
            step=1,
        )
        engine.cfg.clc_rebalance_max_swaps_per_tick = st.number_input(
            "CLC max rebalance swaps per tick",
            min_value=0,
            max_value=20,
            value=int(engine.cfg.clc_rebalance_max_swaps_per_tick),
            step=1,
        )
        engine.cfg.clc_rebalance_target_stable_ratio = st.slider(
            "CLC target stable ratio",
            0.0,
            1.0,
            float(engine.cfg.clc_rebalance_target_stable_ratio),
            step=0.05,
        )
        engine.cfg.clc_rebalance_swap_size_frac = st.slider(
            "CLC rebalance swap size (share of CLC value)",
            0.0,
            0.5,
            float(engine.cfg.clc_rebalance_swap_size_frac),
            step=0.01,
        )
        engine.cfg.clc_rebalance_min_usd = st.number_input(
            "CLC rebalance min size (USD)",
            min_value=0.0,
            value=float(engine.cfg.clc_rebalance_min_usd),
            step=10.0,
        )

        st.subheader("Insurance / Incidents")
        engine.cfg.insurance_fee_window_ticks = st.number_input(
            "Insurance fee eligibility window (ticks)",
            min_value=1,
            max_value=52,
            value=int(engine.cfg.insurance_fee_window_ticks),
            step=1,
            help="Claims are only eligible if the pool paid enough CLC fees in this window.",
        )
        engine.cfg.insurance_min_fee_usd = st.number_input(
            "Insurance min CLC fees (USD, per window)",
            min_value=0.0,
            value=float(engine.cfg.insurance_min_fee_usd),
            step=1.0,
            help="Set to 0 to disable fee-based claim eligibility.",
        )
        engine.cfg.insurance_risk_weight_base = st.number_input(
            "Risk weight base",
            min_value=0.0,
            value=float(engine.cfg.insurance_risk_weight_base),
            step=0.1,
        )
        engine.cfg.insurance_risk_weight_reserve_scale = st.number_input(
            "Risk weight reserve scale",
            min_value=0.0,
            value=float(engine.cfg.insurance_risk_weight_reserve_scale),
            step=0.1,
        )
        engine.cfg.insurance_risk_weight_min = st.number_input(
            "Risk weight min",
            min_value=0.0,
            value=float(engine.cfg.insurance_risk_weight_min),
            step=0.1,
        )
        engine.cfg.insurance_risk_weight_max = st.number_input(
            "Risk weight max",
            min_value=0.0,
            value=float(engine.cfg.insurance_risk_weight_max),
            step=0.1,
        )
        engine.cfg.incident_base_rate = st.slider(
            "Incident base rate (per tick)",
            0.0,
            0.2,
            float(engine.cfg.incident_base_rate),
            step=0.01,
        )
        engine.cfg.incident_loss_rate = st.slider(
            "Incident loss rate (share of voucher value)",
            0.0,
            1.0,
            float(engine.cfg.incident_loss_rate),
            step=0.01,
        )
        engine.cfg.incident_min_loss_usd = st.number_input(
            "Incident min loss (USD)",
            min_value=0.0,
            value=float(engine.cfg.incident_min_loss_usd),
            step=50.0,
        )
        engine.cfg.incident_haircut_cap = st.slider(
            "Incident haircut cap",
            0.0,
            1.0,
            float(engine.cfg.incident_haircut_cap),
            step=0.05,
        )
        engine.cfg.incident_max_per_tick = st.number_input(
            "Max incidents per tick",
            min_value=0,
            max_value=50,
            value=int(engine.cfg.incident_max_per_tick),
            step=1,
        )

# Data
net_df = engine.metrics.network_df()
pool_df = engine.metrics.pool_df()

with tab_network:
    st.subheader("Network KPIs")
    if net_df.empty:
        st.info("No metrics yet. Add pools or run ticks.")
    else:
        latest = net_df.iloc[-1].to_dict()
        kpis = [
            ("Pools", _fmt(latest.get("num_pools", len(engine.pools)))),
            ("Vouchers", _fmt(latest["num_assets"])),
            ("Stable total", _fmt(latest["stable_total_in_pools"])),
            ("Pools under reserve", _fmt(latest["pools_under_stable_reserve"])),
            ("Redeemed (units per tick)", _fmt(latest["redeemed_total"])),
            ("Debt outstanding (USD)", _fmt(latest.get("debt_outstanding_usd", 0.0))),
            ("Debt outstanding (units)", _fmt(latest.get("debt_outstanding_units", 0.0))),
            ("Transactions per tick", _fmt(latest.get("transactions_per_tick", 0))),
        ]
        _render_kpi_grid(kpis, columns=4)

        st.subheader("Swap Flow Breakdown (USD per tick)")
        st.line_chart(
            net_df,
            x="tick",
            y=[
                "swap_volume_usd_to_vchr_tick",
                "swap_volume_vchr_to_usd_tick",
                "swap_volume_vchr_to_vchr_tick",
            ],
        )

        st.subheader("Transactions per tick (swaps)")
        st.line_chart(
            net_df,
            x="tick",
            y=["transactions_per_tick"],
        )

        st.subheader("Repayment vs Loan Issuance vs Debt Outstanding (USD)")
        st.line_chart(
            net_df,
            x="tick",
            y=["repayment_volume_usd", "loan_issuance_volume_usd", "debt_outstanding_usd"],
        )

        st.subheader("Pool Utilization (swap volume / pool value)")
        st.line_chart(
            net_df,
            x="tick",
            y=["utilization_rate"],
        )

        st.subheader("Pools vs Vouchers (count)")
        voucher_offset = 1 + (1 if engine.cfg.sclc_symbol else 0)
        counts_df = net_df.copy()
        counts_df["voucher_count"] = counts_df["num_assets"] - voucher_offset
        counts_df["voucher_count"] = counts_df["voucher_count"].clip(lower=0)
        st.line_chart(
            counts_df,
            x="tick",
            y=["num_pools", "voucher_count"],
        )

        st.subheader("Current Supply (point-in-time)")
        st.line_chart(
            net_df,
            x="tick",
            y=[
                "stable_total_in_pools",
                "voucher_total_in_pools",
            ],
        )

        st.subheader("Redeemed (units per tick) vs Outstanding Voucher Value (USD)")
        st.line_chart(
            net_df,
            x="tick",
            y=["redeemed_total", "outstanding_voucher_value_usd"],
        )

        st.subheader("Flow Multipliers (c and Beta)")
        ratio_cols = st.columns(2)
        ratio_cols[0].metric("c (V→USD / USD→V)", _fmt(latest.get("swap_c_ratio", 0.0)))
        ratio_cols[1].metric("Beta (V→V / V→USD)", _fmt(latest.get("swap_beta_ratio", 0.0)))
        st.line_chart(
            net_df,
            x="tick",
            y=["swap_c_ratio", "swap_beta_ratio"],
        )

        st.subheader("Fiat On/Off-Ramps (USD per tick)")
        st.line_chart(
            net_df,
            x="tick",
            y=["stable_onramp_usd_tick", "stable_offramp_usd_tick"],
        )

with tab_clc:
    st.subheader("CLC KPIs")
    if net_df.empty:
        st.info("No metrics yet. Add pools or run ticks.")
    else:
        latest = net_df.iloc[-1].to_dict()
        st.caption(
            "CLC is routable and holds in-kind fee assets (including vouchers). When inventory exists, "
            "routing/clearing can use CLC as a stable->voucher venue."
        )
        fee_cols = st.columns(4)
        fee_cols[0].metric("Total pool fees (USD)", _fmt(latest.get("fee_pool_cumulative_usd", 0.0)))
        fee_cols[1].metric("Total pool fees (Vouchers)", _fmt(latest.get("fee_pool_cumulative_voucher", 0.0)))
        fee_cols[2].metric("Total CLC fees (USD)", _fmt(latest.get("fee_clc_cumulative_usd", 0.0)))
        fee_cols[3].metric("Total CLC fees (Vouchers)", _fmt(latest.get("fee_clc_cumulative_voucher", 0.0)))
        kpis = [
            ("Insurance fund (USD)", _fmt(latest.get("insurance_fund_usd", 0.0))),
            ("Insurance target (USD)", _fmt(latest.get("insurance_target_usd", 0.0))),
            ("Insurance coverage", _fmt(latest.get("insurance_coverage_ratio", 0.0))),
            ("Fee chi", _fmt(latest.get("fee_chi", 0.0))),
            ("Ops pool (USD) cumulative", _fmt(latest.get("ops_pool_usd", 0.0))),
            ("Mandates pool (USD, cumulative)", _fmt(latest.get("mandates_allocated_usd_total", 0.0))),
            ("CLC pool (USD) injection cumulative", _fmt(latest.get("clc_pool_injected_usd_total", 0.0))),
            ("Fee access budget (USD)", _fmt(latest.get("fee_access_budget_usd", 0.0))),
            ("Mandates distributed (USD, cumulative)", _fmt(latest.get("mandates_distributed_usd_total", 0.0))),
            ("CLC pool swapped out stables (cumulative)", _fmt(latest.get("clc_pool_swapped_out_stable_total", 0.0))),
            ("CLC vouchers swapped out (cumulative)", _fmt(latest.get("clc_pool_swapped_out_voucher_total", 0.0))),
            ("Conversion used (USD)", _fmt(latest.get("fee_conversion_used_usd_epoch", 0.0))),
        ]
        _render_kpi_grid(kpis, columns=4)

        st.subheader("LP (sCLC -> USD)")
        lp_injected = float(latest.get("lp_injected_usd_total", 0.0))
        lp_returned = float(latest.get("lp_returned_usd_total", 0.0))
        lp_net = lp_returned - lp_injected
        lp_roi = lp_returned / lp_injected if lp_injected > 1e-9 else 0.0
        total_ticks = max(1, int(latest.get("tick", 0)))
        years_elapsed = total_ticks / 52.0
        lp_net_roi = lp_net / lp_injected if lp_injected > 1e-9 else 0.0
        lp_apr = lp_net_roi / years_elapsed if years_elapsed > 0 else 0.0
        lp_cagr = 0.0
        if lp_injected > 1e-9 and years_elapsed > 0.0:
            ratio = lp_returned / lp_injected
            if ratio > 0.0:
                lp_cagr = (ratio ** (1.0 / years_elapsed)) - 1.0
        kpis = [
            ("LP sCLC supply", _fmt(latest.get("lp_sclc_supply_total", 0.0))),
            ("LP injected (USD)", _fmt(lp_injected)),
            ("LP returned (USD)", _fmt(lp_returned)),
            ("LP net (USD)", _fmt(lp_net)),
            ("LP ROI", _fmt(lp_roi)),
            ("LP APR (est.)", f"{lp_apr * 100:.2f}%"),
            ("LP CAGR (est.)", f"{lp_cagr * 100:.2f}%"),
            ("Years elapsed", f"{years_elapsed:.2f}"),
        ]
        _render_kpi_grid(kpis, columns=4)

        ticks_per_month = 4.0
        avg_return_per_tick = lp_returned / total_ticks if total_ticks > 0 else 0.0
        if lp_injected > 1e-9 and avg_return_per_tick > 1e-9:
            remaining = max(0.0, lp_injected - lp_returned)
            est_payback_tick = total_ticks + (remaining / avg_return_per_tick)
            est_payback_months = est_payback_tick / ticks_per_month
            next_year_ticks = int(12 * ticks_per_month)
            next_year_profit = avg_return_per_tick * next_year_ticks
            next_year_apr = next_year_profit / lp_injected
            est_kpis = [
                ("Estimate LP payback (tick)", f"{est_payback_tick:.1f}"),
                ("Estimate LP payback (months)", f"{est_payback_months:.2f}"),
                ("Estimated next-year profit (USD)", _fmt(next_year_profit)),
                ("Estimated next-year APR", f"{next_year_apr * 100:.2f}%"),
            ]
            _render_kpi_grid(est_kpis, columns=4)
        else:
            st.info("Not enough LP return history to estimate payback/APR.")

        st.subheader("Cumulative Fees (USD)")
        st.line_chart(
            net_df,
            x="tick",
            y=["fee_pool_cumulative_usd", "fee_clc_cumulative_usd"],
        )

        st.subheader("Cumulative Fees (Vouchers)")
        st.line_chart(
            net_df,
            x="tick",
            y=["fee_pool_cumulative_voucher", "fee_clc_cumulative_voucher"],
        )

        st.subheader("Waterfall Fees (USD per epoch)")
        st.line_chart(
            net_df,
            x="tick",
            y=["fee_in_usd_epoch", "fee_cash_usd_epoch", "fee_kind_usd_epoch"],
        )

        st.subheader("Waterfall Allocations (USD per epoch)")
        st.line_chart(
            net_df,
            x="tick",
            y=[
                "waterfall_ops_alloc_usd_epoch",
                "waterfall_insurance_alloc_usd_epoch",
                "waterfall_mandates_alloc_usd_epoch",
                "waterfall_clc_alloc_usd_epoch",
            ],
        )

        st.subheader("Insurance Coverage")
        st.line_chart(
            net_df,
            x="tick",
            y=["insurance_fund_usd", "insurance_target_usd"],
        )

        st.subheader("Incidents and Claims (per tick)")
        st.line_chart(
            net_df,
            x="tick",
            y=["incidents_tick", "claims_paid_usd_tick", "claims_unpaid_usd_tick"],
        )

        #st.line_chart(net_df.set_index("tick")[[
        #    "stable_total_in_pools",
        #    "redeemed_total",
        #    "outstanding_voucher_supply_total"
        #]])

with tab_noam_overview:
    st.subheader("NOAM Swaps (per tick)")
    if net_df.empty:
        st.info("No metrics yet. Add pools or run ticks.")
    else:
        st.line_chart(
            net_df,
            x="tick",
            y=["noam_routing_swaps_tick", "noam_clearing_swaps_tick"],
        )
        st.subheader("Routing Outcomes (per tick)")
        st.line_chart(
            net_df,
            x="tick",
            y=["route_requested_tick", "route_found_tick", "route_failed_tick"],
        )

with tab2:
    st.subheader("Pools (latest tick)")
    if pool_df.empty:
        st.info("No pool rows yet.")
    else:
        latest_tick = pool_df["tick"].max()
        cur = pool_df[pool_df["tick"] == latest_tick]
        cur = cur.sort_values(["tick", "pool_id"]).drop_duplicates(["pool_id"], keep="last")
        cur = cur.sort_values(["role", "mode", "stable"], ascending=[True, True, False])
        vouchers_by_pool = {
            pid: engine._pool_voucher_value_usd(p) for pid, p in engine.pools.items()
        }
        cur = cur.copy()
        cur["vouchers"] = cur["pool_id"].map(vouchers_by_pool).fillna(0.0)
        if "min_stable_reserve" in cur.columns:
            cur = cur.drop(columns=["min_stable_reserve"])
        if "stable" in cur.columns and "vouchers" in cur.columns:
            cols = list(cur.columns)
            cols.remove("vouchers")
            cols.insert(cols.index("stable") + 1, "vouchers")
            cur = cur[cols]
        st.dataframe(_format_table_numbers(cur), use_container_width=True)

        active_pools = [p for p in engine.pools.values() if not p.policy.system_pool]
        stable_total = sum(p.vault.get(engine.cfg.stable_symbol) for p in active_pools)
        voucher_total = sum(
            amt
            for p in active_pools
            for asset, amt in p.vault.inventory.items()
            if asset.startswith("VCHR:")
        )
        grand_total = stable_total + voucher_total

        total_cols = st.columns(3)
        total_cols[0].metric("Total stable in pools", _fmt(stable_total))
        total_cols[1].metric("Total vouchers in pools", _fmt(voucher_total))
        total_cols[2].metric("Grand total", _fmt(grand_total))

        total_ticks = max(1, int(engine.tick))
        per_tick_cols = st.columns(3)
        per_tick_cols[0].metric("Total stable in pools / ticks", _fmt(stable_total / total_ticks))
        per_tick_cols[1].metric("Total vouchers in pools / ticks", _fmt(voucher_total / total_ticks))
        per_tick_cols[2].metric("Grand total / ticks", _fmt(grand_total / total_ticks))

        pool_ids = list(engine.pools.keys())
        sel = st.selectbox("Select pool", pool_ids)
        p = engine.pools[sel]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Role", p.policy.role)
        c2.metric("Mode", p.policy.mode)
        c3.metric("Stable", _fmt(p.vault.get(engine.cfg.stable_symbol)))
        c4.metric("Min reserve", _fmt(p.policy.min_stable_reserve))

        inv = pd.DataFrame([{"asset": a, "amount": amt} for a, amt in p.vault.inventory.items()])
        if not inv.empty:
            inv = inv.sort_values("amount", ascending=False)
        st.write("**Inventory**")
        st.dataframe(inv, use_container_width=True)

        lst = pd.DataFrame([
            {"asset": a, "listed": pol.enabled, "value": p.values.get_value(a)}
            for a, pol in p.registry.listings.items()
        ])
        st.write("**Listings (wants)**")
        st.dataframe(lst, use_container_width=True)

        details = _pool_tooltips(engine, [sel]).get(sel, "")
        st.text_area("Pool details (copyable)", details, height=300)

with tab3:
    st.subheader("Event Log (latest 300)")
    tail = engine.log.tail(300)
    if not tail:
        st.info("No events yet.")
    else:
        df = pd.DataFrame([e.__dict__ for e in tail])
        df["_order"] = range(len(df))
        df = df.sort_values(["tick", "_order"], ascending=False).drop(columns="_order")
        if "meta" in df.columns:
            df["meta"] = df["meta"].apply(_format_event_meta)
        st.dataframe(df, use_container_width=True)

with tab4:
    st.subheader("Swap Graph")
    window_value = st.session_state.get("graph_window", "Last 10 ticks")
    window_ticks_value = int(st.session_state.get("graph_window_ticks", 50))
    max_nodes_value = int(st.session_state.get("graph_max_nodes", 2000))
    max_edges_value = int(st.session_state.get("graph_max_edges", 5000))
    width_value = int(st.session_state.get("graph_width", 1100))
    height_value = int(st.session_state.get("graph_height", 700))

    tick_max = engine.tick
    if window_value == "Current tick":
        tick_min = tick_max
    elif window_value == "Last 10 ticks":
        tick_min = max(0, tick_max - 9)
    elif window_value == "Last N ticks":
        n_ticks = max(1, int(window_ticks_value))
        tick_min = max(0, tick_max - (n_ticks - 1))
    else:
        tick_min = 0

    edges = _extract_swap_edges(engine.log.events, tick_min, tick_max)
    if not edges:
        st.info("No swaps in the selected window.")
    else:
        activity = {}
        for (a, b), w in edges.items():
            activity[a] = activity.get(a, 0) + w
            activity[b] = activity.get(b, 0) + w

        nodes = sorted(activity.keys(), key=lambda k: activity[k], reverse=True)
        if len(nodes) > max_nodes_value:
            nodes = nodes[: int(max_nodes_value)]
        nodes_set = set(nodes)

        edges = {k: v for k, v in edges.items() if k[0] in nodes_set and k[1] in nodes_set}
        if len(edges) > max_edges_value:
            top = sorted(edges.items(), key=lambda kv: kv[1], reverse=True)[: int(max_edges_value)]
            edges = dict(top)

        balances = _pool_balances(engine)
        balances = {pid: balances.get(pid, {"stable": 0.0, "vouchers": 0.0}) for pid in nodes}
        tooltips = _pool_tooltips(engine, nodes)
        positions = _spring_layout(nodes, edges, int(width_value), int(height_value))
        svg = _render_swaps_svg(positions, edges, balances, tooltips, int(width_value), int(height_value))
        st.markdown(svg, unsafe_allow_html=True)

    st.divider()
    st.subheader("Graph Settings")
    window_options = ["Current tick", "Last 10 ticks", "Last N ticks", "All ticks"]
    window_index = window_options.index(window_value) if window_value in window_options else 0
    window = st.selectbox("Window", window_options,
                          index=window_index,
                          key="graph_window")
    st.number_input(
        "Window ticks (for Last N ticks)",
        min_value=1,
        max_value=10000,
        value=window_ticks_value,
        step=10,
        key="graph_window_ticks",
    )
    max_nodes = st.number_input("Max nodes to render", min_value=50, max_value=10000,
                                value=max_nodes_value, step=50, key="graph_max_nodes")
    max_edges = st.number_input("Max edges to render", min_value=100, max_value=50000,
                                value=max_edges_value, step=100, key="graph_max_edges")
    width = st.number_input("Graph width (px)", min_value=600, max_value=1600,
                            value=width_value, step=50, key="graph_width")
    height = st.number_input("Graph height (px)", min_value=400, max_value=1200,
                             value=height_value, step=50, key="graph_height")
