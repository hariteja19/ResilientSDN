"""Random instances of the general problem of Sec. III-C (revision 4).

An instance is the tuple (G, c, r(t), A, B_a, T_a, pi_a) of the formulation, drawn from the space below
instead of being hand-made; S1-S14 are single points of this space. Everything is seeded by the instance
id, so every system sees the same instance.

  topology   8 real networks (fat-tree, NSFNET, Internet2, GEANT, Abilene, Arnes, Renater 2010,
             SURFnet; Topology Zoo) or a random graph: Waxman (15-40 nodes) or Barabasi-Albert
             (15-40 nodes, m = 2), link capacities 100 or a mix of 10/40/100 Mbps
  demands    gravity model: node masses ~ Exp(1), n_flows in [10, 60] pairs drawn with probability
             proportional to the product of masses, rates proportional to it, scaled so that static
             routing loads the busiest link to L_base ~ U[0.2, 0.6]
  attack     m in {1, 2, 3} target links (most used by legitimate traffic under static routing,
             fastest class, not bridges, no shared switch), k in {1..5} bots per target on host pairs
             whose static route crosses the target; intensity alpha ~ U[0.9, 1.6]: the attack brings
             the target's offered load to alpha * c
  pattern    constant | pulse (period 4 or 8 s, duty 1/3, 1/2 or 2/3, low phase at 0.65 c) |
             timed (8 s bursts every 31 s) | rolling (re-aims every 10 s at the victims' shared link,
             routing-aware, as S14)
  timing     legitimate traffic from t = 0, attack from t = 10 s for 60 s, 3 s tail
"""
import math
import random

import networkx as nx

from attack.attacker import TrafficPlanner
from controller.path_engine import path_links
from sim.network import SimFlow
from sim.scenarios import MultiPlanner, const

REAL = ['fat_tree', 'nsfnet', 'internet2', 'geant', 'abilene', 'arnes', 'renater2010', 'surfnet']
PATTERNS = ['constant', 'pulse', 'timed', 'rolling']
T0, DURATION, TAIL = 10, 60, 3
FIBRE_KM_PER_MS = 200.0


class GraphTopo:
    """The minimum of the topology interface TrafficPlanner needs, for a bare graph: one host per
    switch (h_i on s_i), addresses 10.0.0.i."""

    def __init__(self, graph):
        self.g = graph
        self.sw = sorted(graph.nodes, key=lambda v: int(str(v)[1:]))
        self.max_path_hops = None

    def get_switch_graph(self):
        return self.g

    def hosts(self):
        return [f'h{i}' for i in range(1, len(self.sw) + 1)]

    def host_info(self):
        return {f'10.0.0.{i}': {'name': f'h{i}', 'switch': s} for i, s in enumerate(self.sw, start=1)}


def random_graph(kind, n, rng):
    """Connected random topology with switches s1..sn, 'bw' (Mbps) and 'latency' (ms) on links."""
    seed = rng.randrange(2 ** 31)
    if kind == 'waxman':
        g = nx.waxman_graph(n, beta=0.4, alpha=0.25, seed=seed)
    else:
        g = nx.barabasi_albert_graph(n, 2, seed=seed)
    pos = {v: (rng.random(), rng.random()) for v in g.nodes}
    if kind == 'waxman':
        pos = {v: g.nodes[v].get('pos', pos[v]) for v in g.nodes}
    # Waxman graphs can be disconnected: join components through their geographically nearest nodes
    comps = [set(c) for c in nx.connected_components(g)]
    while len(comps) > 1:
        a, b = min(((u, v) for u in comps[0] for c in comps[1:] for v in c),
                   key=lambda e: math.dist(pos[e[0]], pos[e[1]]))
        g.add_edge(a, b)
        comps = [set(c) for c in nx.connected_components(g)]
    mixed = rng.random() < 0.5
    h = nx.Graph()
    for a, b in g.edges:
        km = 4000 * math.dist(pos[a], pos[b])   # unit square scaled to a 4000 km continent
        bw = rng.choice([10, 40, 100, 100]) if mixed else 100
        h.add_edge(f's{a + 1}', f's{b + 1}', bw=bw, latency=round(max(0.1, km / FIBRE_KM_PER_MS), 2))
    return h


def sample_params(instance_id):
    rng = random.Random(f'instance-{instance_id}')
    topo = rng.choice(REAL + ['waxman', 'barabasi_albert'] * 4)   # half real, half random
    return {
        'instance': instance_id, 'topology': topo,
        'nodes': rng.randint(15, 40) if topo in ('waxman', 'barabasi_albert') else None,
        'n_flows': rng.randint(10, 60), 'base_load': rng.uniform(0.2, 0.6),
        'targets': rng.choice([1, 1, 2, 3]), 'bots': rng.randint(1, 5),
        'alpha': rng.uniform(0.9, 1.6), 'pattern': rng.choice(PATTERNS),
        'pulse_period': rng.choice([4, 8]), 'pulse_duty': rng.choice([1 / 3, 1 / 2, 2 / 3]),
    }


def build_instance(p):
    """Instance dict in the format sim.run.simulate expects, plus descriptive features."""
    from topology import load_topology
    rng = random.Random(f'build-{p["instance"]}')
    if p['topology'] in REAL:
        topo = load_topology(p['topology'])
    else:
        topo = GraphTopo(random_graph(p['topology'], p['nodes'], rng))
    planner = MultiPlanner(topo)
    g = planner.g
    pairs = [q for q in planner.paths]
    # --- gravity-model demands, scaled to the base load
    hosts = sorted({h for q in pairs for h in q})
    mass = {h: rng.expovariate(1.0) for h in hosts}
    weights = [mass[a] * mass[b] for a, b in pairs]
    chosen = set()
    while len(chosen) < min(p['n_flows'], len(pairs)):
        chosen.add(rng.choices(range(len(pairs)), weights=weights)[0])
    legit = [pairs[i] for i in sorted(chosen)]
    load = {}
    for i in sorted(chosen):
        for l in path_links(planner.paths[pairs[i]]):
            load[l] = load.get(l, 0.0) + weights[i] / (g.edges[l]['bw'])
    scale = p['base_load'] / max(load.values())          # Mbps per unit weight
    rate = {pairs[i]: weights[i] * scale for i in sorted(chosen)}
    legit_on = {}
    for q, r in rate.items():
        for l in path_links(planner.paths[q]):
            legit_on[l] = legit_on.get(l, 0.0) + r
    # --- targets: most legit-used fastest non-bridge links with attack pairs, no shared switch
    bridges = {frozenset(e) for e in nx.bridges(g)}
    top = max(d['bw'] for *_, d in g.edges(data=True))
    cand = sorted((l for l in legit_on if frozenset(l) not in bridges and g.edges[l]['bw'] == top),
                  key=lambda l: (-legit_on[l], str(l)))
    used = set(legit)
    targets, attack = [], []
    t1 = T0 + DURATION
    for l in cand:
        if len(targets) == p['targets']:
            break
        if any(set(l) & set(t) for t in targets):
            continue
        bots = planner.pick([q for q in planner.pairs_crossing(l) if q not in used], p['bots'], used, rng)
        if not bots:
            continue
        used |= set(bots)
        c = g.edges[l]['bw']
        per_bot = max(0.0, p['alpha'] * c - legit_on.get(l, 0.0)) / len(bots)
        low = max(0.0, 0.65 * c - legit_on.get(l, 0.0)) / len(bots)
        targets.append(l)
        attack += [(q, per_bot, low) for q in bots]
    if not targets:
        return None                                        # no attackable link (e.g. a tree)
    flows = []

    def mk(q, kind, fn):
        a, b = q
        f = SimFlow(id=f'{planner.ip_of[a]}->{planner.ip_of[b]}', src=planner.switch_of[a],
                    dst=planner.switch_of[b], kind=kind, rate_fn=fn)
        flows.append(f)
        return f

    for q in legit:
        crosses = any(t in path_links(planner.paths[q]) for t in targets)
        mk(q, 'legit' if crosses else 'background', const(rate[q], 0, t1 + TAIL))
    for q, hi, lo in attack:
        mk(q, 'attack', _pattern_fn(p, hi, lo))
    deg = 2 * g.number_of_edges() / g.number_of_nodes()
    features = {'n_nodes': g.number_of_nodes(), 'n_links': g.number_of_edges(), 'avg_degree': round(deg, 2),
                'edge_connectivity': nx.edge_connectivity(g), 'n_targets': len(targets),
                'n_attack_flows': len(attack), 'n_legit_flows': len(legit),
                'n_victim_flows': sum(f.kind == 'legit' for f in flows)}
    adaptive = p['pattern'] == 'rolling'
    b = {'planner': planner, 'graph': g, 'flows': flows, 'rng': rng, 'randomise': False,
         'plan': {'targets': targets, 'legit': legit, 'background': [], 'attack': [q for q, *_ in attack]},
         's': {'adaptive': adaptive}, 's_id': p['pattern'], 't0': T0, 't1': t1, 'end': t1 + TAIL,
         'attack_fn': const(attack[0][1], T0, t1) if adaptive else None, 'ecmp_aware': False}
    return b, features


def _pattern_fn(p, hi, lo):
    t1 = T0 + DURATION
    if p['pattern'] == 'pulse':
        per, duty = p['pulse_period'], p['pulse_duty']
        return lambda t: ((hi if ((t - T0) % per) < duty * per else lo) if T0 <= t < t1 else 0.0)
    if p['pattern'] == 'timed':
        return lambda t: hi if T0 <= t < t1 and (t - T0) % 31 < 8 else 0.0
    return const(hi, T0, t1)                               # constant, and rolling (re-aimed later)
