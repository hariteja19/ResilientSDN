"""Flow-level simulator: model invariants and baseline sanity."""
import math

import networkx as nx
import pytest

from sim.network import FluidNetwork, SimFlow
from sim.run import run_single
from sim.scenarios import SIM_SCENARIOS


def line():
    g = nx.Graph()
    g.add_edge('a', 'b', bw=8, latency=1.0)   # 1e6 bytes/s
    g.add_edge('b', 'c', bw=8, latency=1.0)
    return g


def test_udp_overload_drops_in_proportion():
    net = FluidNetwork(line())
    for i, mbps in enumerate((8, 8)):
        f = SimFlow(id=str(i), src='a', dst='c', kind='attack', rate_fn=lambda t, m=mbps: m)
        f.paths = {('a', 'b', 'c'): 1.0}
        net.add(f)
    carried = net.step(0, True)
    assert carried[('a', 'b')] == pytest.approx(1e6)
    for f in net.flows.values():
        assert f.delivered == pytest.approx(0.5e6)       # each keeps half
        assert f.delivered <= f.sent


def test_tcp_shares_residual_max_min_and_finishes():
    net = FluidNetwork(line())
    u = SimFlow(id='u', src='a', dst='c', kind='legit', rate_fn=lambda t: 4)   # half the link
    u.paths = {('a', 'b', 'c'): 1.0}
    net.add(u)
    tcps = []
    for i in range(2):
        f = SimFlow(id=f't{i}', src='a', dst='c', kind='tcp', tcp_cap_mbps=100, size_bytes=1e6)
        f.paths = {('a', 'b', 'c'): 1.0}
        net.add(f)
        tcps.append(f)
    net.step(0, True)
    assert [f.measured_rate for f in tcps] == pytest.approx([0.25e6, 0.25e6])
    for t in range(1, 6):
        net.step(t, True)
    assert all(f.fcts and f.fcts[0][1] == pytest.approx(4.0) for f in tcps)


def test_runs_are_deterministic_and_static_never_reroutes():
    a = run_single('fat_tree', 'static', 'S3', 0)
    b = run_single('fat_tree', 'static', 'S3', 0)
    for k in ('time_above_s', 'victim_loss_pct', 'mlu_mean'):
        assert a[k] == b[k]
    assert a['reroutes'] == 0 and a['time_above_s'] > 0


def test_no_attack_no_reroute():
    for system in ('simple_reactive', 'resilient_sdn', 'te_lp'):
        r = run_single('fat_tree', system, 'S1', 0)
        assert r['reroutes'] == 0 and r['time_above_s'] == 0


def test_resilient_relieves_overload_faster_than_static():
    s = run_single('fat_tree', 'static', 'S11', 1)
    r = run_single('fat_tree', 'resilient_sdn', 'S11', 1)
    assert r['time_above_s'] < s['time_above_s'] / 2
    assert r['victim_loss_pct'] < s['victim_loss_pct']


@pytest.mark.parametrize('scenario', ['S12', 'S13'])
def test_multi_link_scenarios_have_several_targets(scenario):
    r = run_single('fat_tree', 'static', scenario, 0)
    assert r['n_targets'] == SIM_SCENARIOS[scenario]['targets']
    assert r['link_time_above_s'] > r['time_above_s']     # more than one link is hot


def test_adaptive_attacker_follows_the_victims():
    static = run_single('nsfnet', 'static', 'S14', 0)
    rsdn = run_single('nsfnet', 'resilient_sdn', 'S14', 0)
    assert static['time_above_s'] > 0
    # the attacker re-aims after each reroute, so ResilientSDN cannot keep the target cool all along
    assert rsdn['time_above_s'] > 6


def test_tcp_mode_reports_goodput_and_fct():
    r = run_single('fat_tree', 'resilient_sdn', 'S3', 0, legit_mode='tcp')
    assert r['transfers'] > 0 and r['tcp_goodput_mbps'] > 0 and not math.isnan(r['fct_mean_s'])


def test_results_do_not_depend_on_the_hash_seed():
    """Runs are compared across processes; set iteration order must not leak into results."""
    import os
    import subprocess
    import sys
    code = ("from sim.run import run_single; r = run_single('fat_tree', '{s}', 'S12', 0); "
            "print([r[k] for k in ('time_above_s', 'mlu_mean', 'reroutes', 'rule_changes')])")
    for system in ('te_lp', 'resilient_sdn', 'mp_ksp'):
        outs = {subprocess.run([sys.executable, '-c', code.format(s=system)], capture_output=True,
                               text=True, env=dict(os.environ, PYTHONHASHSEED=seed),
                               check=True).stdout for seed in ('1', '2', '3')}
        assert len(outs) == 1, (system, outs)


def test_excess_lp_relieves_target_when_another_link_pins_the_min_max_optimum():
    """GEANT S11 run 1: a 5 Mbps flow on a 1.55 Mbps link fixes U* > 1 whatever the routing, so
    min-max TE leaves the target overloaded; minimising total excess (Eq. 3 relaxed) does not."""
    minmax = run_single('geant', 'te_lp', 'S11', 1)
    excess = run_single('geant', 'te_lp_excess', 'S11', 1)
    assert minmax['time_above_s'] >= 100
    assert excess['time_above_s'] <= 6 and excess['victim_loss_pct'] < minmax['victim_loss_pct']


def test_duty_cycle_trigger_catches_pulses_without_false_alarms_on_bursts():
    assert run_single('nsfnet', 'resilient_sdn', 'S7', 0)['time_above_s'] == 32   # pulse evades N=3
    assert run_single('nsfnet', 'rsdn_duty', 'S7', 0)['time_above_s'] < 16
    for system in ('rsdn_duty', 'resilient_sdn_v3'):
        assert run_single('fat_tree', system, 'SB', 0)['reroutes'] == 0          # ablation D bursts
    assert run_single('fat_tree', 'simple_reactive', 'SB', 0)['reroutes'] > 0
