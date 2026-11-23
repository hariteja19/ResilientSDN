"""Traffic of one simulated run.

S1-S11 use the scenario table of the emulation (attack/scenarios.py) and its TrafficPlanner, so the
target link, the host pairs and the attack rates are chosen exactly as in Mininet for the same seed.
New in revision 4 (simulation only):
  S12  multi-link:  two target links attacked at once, 3 attack flows x 30 Mbps each (S11-level
                    overload per link), 2 legitimate flows per target, 120 s
  S13  multi-link:  as S12 with three target links
  S14  adaptive rolling attacker (Crossfire-style): the same 3 x 30 Mbps budget, but every
                    ADAPT_S seconds the attacker observes the victims' current routes (as with
                    traceroute) and moves its flows onto host pairs whose route crosses the link
                    now shared by most victim flows; 120 s
"""
from collections import Counter

import networkx as nx

from attack.adversarial import PRELOAD_LEAD_S, per_flow_mbps
from attack.attacker import TrafficPlanner
from attack.scenarios import SCENARIOS as EMU_SCENARIOS
from controller.path_engine import path_links

SIM_SCENARIOS = dict(EMU_SCENARIOS)
SIM_SCENARIOS.update({
    'S12': {'name': 'Multi-link 2x', 'targets': 2, 'attackers': 3, 'rate_mbps': 30, 'duration': 120},
    'S13': {'name': 'Multi-link 3x', 'targets': 3, 'attackers': 3, 'rate_mbps': 30, 'duration': 120},
    'SB': {'name': 'Bursty legitimate (false-alarm test, ablation D)', 'attackers': 0, 'rate_mbps': 0,
           'bursty': True},
    'S14': {'name': 'Adaptive rolling', 'adaptive': True, 'attackers': 3, 'rate_mbps': 30,
            'duration': 120},
})
ADAPT_S = 10


def const(mbps, t0, t1):
    return lambda t: mbps if t0 <= t < t1 else 0.0


def schedule(s, plan, legit_total, t0, t1):
    """Per-attack-flow rate function (Mbps) for scenario s, as attack/adversarial.py does it."""
    n, target = len(plan['attack']), plan['target_mbps']
    if s.get('pattern') == 'pulse':
        hi = per_flow_mbps(s['high'], target, legit_total, n)
        lo = per_flow_mbps(s['low'], target, legit_total, n)
        return lambda t: (hi if int((t - t0) // s['interval']) % 2 == 0 else lo) if t0 <= t < t1 else 0.0
    if s.get('pattern') == 'timed':
        r = per_flow_mbps(s['rate_pct'], target, legit_total, n)
        return lambda t: r if t0 <= t < t1 and (t - t0) % s['interval'] < s['burst'] else 0.0
    if s.get('pattern') == 'presaturate':
        return const(per_flow_mbps(s['attack_pct'], target, legit_total, n), t0 + PRELOAD_LEAD_S, t1)
    return const(s.get('rate_mbps', 0), t0, t1)


class MultiPlanner(TrafficPlanner):
    """TrafficPlanner with several targets: the most used fast non-bridge links that share no
    switch with an earlier target (so the attacks are on separate parts of the network)."""

    def choose_target_links(self, m):
        use = Counter(l for p in self.paths.values() for l in path_links(p))
        bridges = {frozenset(e) for e in nx.bridges(self.g)}
        top_bw = max(d['bw'] for *_, d in self.g.edges(data=True))
        ok = sorted((l for l in use if frozenset(l) not in bridges
                     and self.g.edges[l]['bw'] == top_bw), key=lambda l: (-use[l], l))
        chosen = []
        for l in ok:
            if all(not set(l) & set(c) for c in chosen) and self.pairs_crossing(l):
                chosen.append(l)
            if len(chosen) == m:
                break
        return chosen

    def plan_multi(self, m, legit_cfg, n_attack, rng):
        targets = self.choose_target_links(m)
        used, legit, attack = set(), [], []
        per_target = {}
        for t in targets:
            crossing = self.pairs_crossing(t)
            lg = self.pick(crossing, legit_cfg['target_flows'], used, rng)
            used |= set(lg)
            at = self.pick(crossing, n_attack, used, rng)
            used |= set(at)
            legit += lg
            attack += at
            per_target[t] = {'legit': lg, 'attack': at}
        crossing_any = {p for t in targets for p in self.pairs_crossing(t)}
        others = [p for p in self.paths if p not in crossing_any]
        background = self.pick(others, legit_cfg['background_flows'], used, rng)
        return {'targets': targets, 'legit': legit, 'background': background, 'attack': attack,
                'preload': [], 'per_target': per_target,
                'target_mbps': self.g.edges[targets[0]]['bw']}

    def adaptive_target(self, victim_paths):
        """Link shared by most victim flows on their current routes (fastest, not a bridge,
        with pairs whose static route crosses it); ties broken by name."""
        use = Counter(l for p in victim_paths for l in path_links(p))
        bridges = {frozenset(e) for e in nx.bridges(self.g)}
        top_bw = max(d['bw'] for *_, d in self.g.edges(data=True))
        ok = [l for l in use if frozenset(l) not in bridges and self.g.edges[l]['bw'] == top_bw
              and self.pairs_crossing(l)]
        return max(ok, key=lambda l: (use[l], l)) if ok else None
