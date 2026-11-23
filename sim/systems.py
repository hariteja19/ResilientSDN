"""Routing systems for the simulator.

static, static_ecmp, simple_reactive and resilient_sdn behave as their Ryu apps (the ResilientSDN
variants call the very same decision code, controller/decision.py). The others are new baselines
that exist only in simulation:
  mp_ksp   oblivious multipath: every flow split equally over its k lowest-latency paths, never
           changed (k = 3; a static, traffic-unaware multipath scheme in the spirit of
           ECMP/WCMP-style spreading).
  te_lp    reactive min-max-utilisation traffic engineering: when any link is above the
           threshold, solve the path-flow LP  min U  s.t.  sum_p x_fp = 1,
           sum_{f,p: e in p} r_f x_fp <= U c_e  over K = 6 lowest-latency paths per flow, with
           the measured rates of ALL flows (the controller cannot tell attack flows apart) and
           fractional splitting. An idealised TE upper bound: it needs select groups / weighted
           splitting that the emulated controllers do not use.
"""
import copy
import itertools
import random

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from controller.decision import Env, ResilientDecision
from controller.path_engine import ecmp_choice, path_links


def lowest_latency(g, s, d):
    return nx.shortest_path(g, s, d, weight='latency')


_KP_CACHE = {}


def k_paths(g, s, d, k, max_hops=None, examine=60):
    """Up to k lowest-latency simple paths within max_hops, looking at no more than `examine`
    paths in latency order (unbounded search hangs on GEANT). Cached per graph."""
    sig = g.graph.get('_sig')
    if sig is None:   # structural signature: ids can be reused after a graph is freed
        sig = g.graph['_sig'] = hash(frozenset((u, v, d['latency']) for u, v, d in g.edges(data=True)))
    key = (sig, s, d, k, max_hops)
    if key in _KP_CACHE:
        return _KP_CACHE[key]
    out = []
    for i, p in enumerate(nx.shortest_simple_paths(g, s, d, weight='latency')):
        if i >= examine:
            break
        if max_hops and len(p) - 1 > max_hops:
            continue
        out.append(tuple(p))
        if len(out) == k:
            break
    if not out:   # nothing within the hop limit: fall back to the lowest-latency path
        out = [tuple(nx.shortest_path(g, s, d, weight='latency'))]
    _KP_CACHE[key] = out
    return out


class System:
    multipath = False

    def __init__(self, graph, cfg, rng):
        self.g, self.cfg, self.rng = graph, cfg, rng
        self.reroutes, self.rule_changes = 0, 0
        self.history = {}      # flow -> list of single paths used (oscillation count)
        self.oscillations = 0
        self.moved_onto = {}   # link -> first time a reroute put traffic on it
        self.now = 0.0

    def initial_paths(self, flow):
        return {tuple(lowest_latency(self.g, flow.src, flow.dst)): 1.0}

    def on_interval(self, net, link_stats):
        pass

    # bookkeeping shared by single-path systems
    def _move(self, flow, new_path):
        old = next(iter(flow.paths))
        new = tuple(new_path)
        self.reroutes += 1
        self.rule_changes += len(set(new) - set(old)) + len(set(old) - set(new))
        seen = self.history.setdefault(flow.id, [old])
        self.oscillations += new in seen
        seen.append(new)
        for l in set(path_links(new)) - set(path_links(old)):
            self.moved_onto.setdefault(l, self.now)
        flow.paths = {new: 1.0}


class Static(System):
    pass


class StaticEcmp(System):
    def initial_paths(self, flow):
        return {tuple(ecmp_choice(self.g, flow.src, flow.dst, flow.id)): 1.0}


class SimpleReactive(System):
    """baselines/simple_reactive.py, line for line."""

    def on_interval(self, net, link_stats):
        th = self.cfg.utilisation_threshold
        for link, stats in link_stats.items():
            if stats['utilisation'] <= th:
                continue
            g2 = self.g.copy()
            g2.remove_edge(*link)
            on = [f for f in net.flows.values() if f.active
                  and link in path_links(next(iter(f.paths)))]
            on.sort(key=lambda f: -f.measured_rate)
            for f in on:
                try:
                    alt = nx.shortest_path(g2, f.src, f.dst, weight='latency')
                except nx.NetworkXNoPath:
                    continue
                self._move(f, alt)


class Resilient(System):
    def __init__(self, graph, cfg, rng, overrides=None):
        super().__init__(graph, copy.deepcopy(cfg), rng)
        for k, v in (overrides or {}).items():
            setattr(self.cfg, k, v)
        self.core = ResilientDecision(graph, self.cfg, rng=rng, clock=lambda: self.now)

    def on_interval(self, net, link_stats):
        def flows_on(link):
            on = [f for f in net.flows.values() if f.active
                  and link in path_links(next(iter(f.paths)))]
            return sorted(on, key=lambda f: -f.measured_rate)

        def set_path(flow, path, why):
            self._move(flow, path)
            return True

        lat = {}
        for a, b, d in self.g.edges(data=True):
            lat[(a, b)] = lat[(b, a)] = d['latency']
        env = Env(flows_on=flows_on, flow_rate=lambda fid: net.flows[fid].measured_rate,
                  switch_path=lambda fid: list(next(iter(net.flows[fid].paths))),
                  set_path=set_path, latency=lambda: lat)
        self.core.on_interval(link_stats, env)


class MultipathKsp(System):
    multipath = True
    K = 3

    def initial_paths(self, flow):
        ps = k_paths(self.g, flow.src, flow.dst, self.K, getattr(self.cfg, 'max_path_hops', None))
        return {p: 1.0 / len(ps) for p in ps}


class MinMaxLP(System):
    """Reactive min-max link utilisation LP (see module docstring)."""
    multipath = True
    K = 6
    CHANGE = 0.01   # a flow counts as rerouted if any path fraction moves by more than this
    OBJECTIVE = 'minmax'

    def __init__(self, graph, cfg, rng):
        super().__init__(graph, cfg, rng)
        self.cands = {}
        self.solves = 0

    def on_interval(self, net, link_stats):
        th = self.cfg.utilisation_threshold
        if not any(s['utilisation'] > th for s in link_stats.values()):
            return
        flows = [f for f in net.flows.values() if f.active and f.measured_rate > 0]
        if not flows:
            return
        hops = getattr(self.cfg, 'max_path_hops', None)
        var = []    # (flow, path)
        for f in flows:
            if f.id not in self.cands:
                self.cands[f.id] = k_paths(self.g, f.src, f.dst, self.K, hops)
            # sorted: set order depends on PYTHONHASHSEED, and with several optimal vertices the
            # LP solution depends on variable order, which made runs differ between processes
            for p in sorted(set(self.cands[f.id]) | set(f.paths)):
                var.append((f, p))
        links = list(net.cap)
        lidx = {l: i for i, l in enumerate(links)}
        nv, nl = len(var), len(links)
        excess = self.OBJECTIVE == 'excess'
        # variables: x (nv), then U, then (excess objective only) one slack s_e per link
        n = nv + 1 + (nl if excess else 0)
        c = np.zeros(n)
        c[nv] = 1e-2 if excess else 1.0      # excess mode: U only breaks ties between equal-Phi splits
        if excess:
            c[nv + 1:] = 1.0                 # Phi = sum_e s_e,  s_e >= u_e - theta
        # tie-break towards low latency and the current split, so equal-U solutions do not churn
        for j, (f, p) in enumerate(var):
            c[j] = 1e-4 * (len(p) - 1) - 1e-3 * f.paths.get(p, 0.0)
        U = np.zeros((nl, n))
        for j, (f, p) in enumerate(var):
            for l in path_links(p):
                U[lidx[l], j] += f.measured_rate / net.cap[l]
        rows, rhs = [], []
        A = U.copy()
        A[:, nv] = -1.0                      # u_e <= U
        rows.append(A), rhs.append(np.zeros(nl))
        if excess:
            B = U.copy()
            B[np.arange(nl), nv + 1 + np.arange(nl)] = -1.0   # u_e - s_e <= theta
            rows.append(B), rhs.append(np.full(nl, th))
        Aeq = np.zeros((len(flows), n))
        fidx = {f.id: i for i, f in enumerate(flows)}
        for j, (f, p) in enumerate(var):
            Aeq[fidx[f.id], j] = 1.0
        res = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), A_eq=Aeq,
                      b_eq=np.ones(len(flows)),
                      bounds=[(0, 1)] * nv + [(0, None)] * (n - nv), method='highs')
        self.solves += 1
        if not res.success:
            return
        new = {}
        for j, (f, p) in enumerate(var):
            if res.x[j] > 1e-6:
                new.setdefault(f.id, {})[p] = res.x[j]
        for f in flows:
            split = new.get(f.id)
            if not split:
                continue
            tot = sum(split.values())
            split = {p: x / tot for p, x in split.items()}
            keys = set(split) | set(f.paths)
            if max(abs(split.get(p, 0) - f.paths.get(p, 0)) for p in keys) > self.CHANGE:
                self.reroutes += 1
                old_nodes = set(itertools.chain.from_iterable(f.paths))
                new_nodes = set(itertools.chain.from_iterable(split))
                self.rule_changes += len(new_nodes ^ old_nodes) + 1   # +1: group weights rewritten
                for p in split:
                    for l in path_links(p):
                        if not any(l in path_links(q) for q in f.paths):
                            self.moved_onto.setdefault(l, self.now)
                f.paths = split


class ExcessLP(MinMaxLP):
    """LP relaxation of the paper's objective, Eq. (3) with beta = gamma = 0: minimise the total
    excess Phi = sum_e [u_e - theta]^+ (U as a tie-break). Unlike min-max it still relieves the
    target when some other link is saturated whatever the routing (seen on GEANT)."""
    OBJECTIVE = 'excess'


V2 = {'order_by_severity': True, 'fallback_least_loaded': True,
      'dpp_bottleneck': True, 'dpp_weight_V0': 0.1}

DUTY = {'duty_window': 6, 'duty_fraction': 0.5}

SYSTEMS = {
    'static': Static,
    'static_ecmp': StaticEcmp,
    'simple_reactive': SimpleReactive,
    'resilient_sdn': Resilient,
    'resilient_sdn_v2': lambda g, c, r: Resilient(g, c, r, V2),
    'mp_ksp': MultipathKsp,
    'te_lp': MinMaxLP,
    'te_lp_excess': ExcessLP,
    # revision-4 ablation of v2: one extension at a time
    'rsdn_severity': lambda g, c, r: Resilient(g, c, r, {'order_by_severity': True}),
    'rsdn_fallback': lambda g, c, r: Resilient(g, c, r, {'fallback_least_loaded': True}),
    'rsdn_bottleneck': lambda g, c, r: Resilient(g, c, r, {'dpp_bottleneck': True, 'dpp_weight_V0': 0.1}),
    # revision 4, duty-cycle trigger against pulsing attacks (controller/detector.py)
    'rsdn_duty': lambda g, c, r: Resilient(g, c, r, DUTY),
    'resilient_sdn_v3': lambda g, c, r: Resilient(g, c, r, {**V2, **DUTY}),
}


def make_system(name, graph, cfg, seed):
    return SYSTEMS[name](graph, cfg, random.Random(seed))
