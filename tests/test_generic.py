"""Random instances of the general problem, the Phi* oracle and the sweep."""
import networkx as nx
import pytest

from sim.generic import REAL, build_instance, random_graph, sample_params
from sim.oracle import cut_bound, phi_star
from sim.sweep import run_instance


def diamond():
    g = nx.Graph()
    for a, b in [('s1', 's2'), ('s2', 's4'), ('s1', 's3'), ('s3', 's4')]:
        g.add_edge(a, b, bw=8, latency=1.0)          # 1e6 bytes/s per direction
    return g


def test_oracle_splits_optimally_and_detects_defendable_instances():
    assert phi_star(diamond(), {('s1', 's4'): 1.8e6}, 0.8)[0] == pytest.approx(0.4, abs=1e-6)
    assert phi_star(diamond(), {('s1', 's4'): 1.5e6}, 0.8)[0] == pytest.approx(0.0, abs=1e-6)


def test_cut_bound_never_exceeds_the_optimum():
    """Proposition 4 is a lower bound on Phi*, which is a lower bound on every routing."""
    import random
    rng = random.Random(1)
    for _ in range(20):
        g = random_graph('waxman', 10, rng)
        nodes = sorted(g.nodes)
        dem = {(rng.choice(nodes), rng.choice(nodes)): rng.uniform(1e6, 2e7) for _ in range(8)}
        dem = {k: v for k, v in dem.items() if k[0] != k[1]}
        assert cut_bound(g, dem, 0.8) <= phi_star(g, dem, 0.8)[0] + 1e-6


def test_oracle_is_a_lower_bound_for_every_system():
    rows = run_instance(4)                            # Barabasi-Albert, 3 targets
    assert rows and all(r['gap'] >= -1e-6 for r in rows)


def test_sampled_space_covers_real_and_random_topologies_and_all_patterns():
    ps = [sample_params(i) for i in range(400)]
    assert {p['topology'] for p in ps} >= set(REAL) | {'waxman', 'barabasi_albert'}
    assert {p['pattern'] for p in ps} == {'constant', 'pulse', 'timed', 'rolling'}
    assert {p['targets'] for p in ps} == {1, 2, 3}


def test_instances_are_reproducible_and_have_the_requested_attack():
    from controller.path_engine import path_links
    p = sample_params(4)
    assert p['pattern'] == 'constant'
    (b1, f1), (b2, f2) = build_instance(p), build_instance(p)
    assert f1 == f2 and [f.id for f in b1['flows']] == [f.id for f in b2['flows']]
    assert f1['n_targets'] == p['targets'] and f1['n_attack_flows'] == p['targets'] * p['bots']
    g = b1['graph']
    for t in b1['plan']['targets']:
        # offered load on the target under static (lowest-latency) routing = alpha * capacity
        load = sum(f.rate_fn(20) for f in b1['flows']
                   if t in path_links(nx.shortest_path(g, f.src, f.dst, weight='latency')))
        assert load == pytest.approx(p['alpha'] * g.edges[t]['bw'], rel=1e-6)
