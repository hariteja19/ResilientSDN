"""ResilientSDN decision core (Algorithm 1 of the paper), free of Ryu.

The Ryu app (controller/resilient_sdn.py) and the flow-level simulator (sim/) both call
ResilientDecision.on_interval(), so the algorithm evaluated in simulation is the code that runs
in the emulated network, line for line.

The host environment is reached only through the callbacks of `Env`:
    flows_on(link)            flows whose current switch path uses directed link, largest first
    flow_rate(flow_id)        measured rate (bytes/s)
    switch_path(flow_id)      current switch path (list of switch names)
    set_path(flow, path, why) install a new path; returns False if it could not be installed
    latency()                 {directed link: ms}
    log(kind, **data)         event log
Flows only need .id, .src and .dst (edge switches).

Per interval t (paper, Algorithm 1):
  1. D(t)  = links above the threshold theta for N consecutive polls (N-1 on a link attacked
             within repeat_attack_window).
  2. For every e in D(t) (detection order; by projected utilisation, highest first, when
     order_by_severity is set) shed flows, largest first, until the projected load of e is
     <= theta * c_e:
       a. candidates P = simple paths of the flow, <= H hops, avoiding e, whose every link stays
          <= theta after the move (capacity filter, evaluated on the projected loads without
          the flow);
       b. p* = argmax w1*C + w2*L - w3*K over P (multi-objective score);
       c. move only if DPP(p*) - DPP(current) > epsilon and the flow's cooldown has elapsed;
       d. update the projected loads, so later moves in the same interval see this one.
  3. Excess still left on e is logged as 'unresolved'.
Two extensions, both off by default so the published configuration is unchanged:
  order_by_severity     process the most overloaded link first (several attacked links)
  fallback_least_loaded when no candidate passes the filter, move the flow to the path with the
                        lowest post-move bottleneck utilisation if that is below the projected
                        utilisation of e (partial relief instead of none; cooldown still applies)
"""
from dataclasses import dataclass
from typing import Callable

from controller.detector import CongestionDetector
from controller.hysteresis import HysteresisManager
from controller.path_engine import PathEngine, path_links
from controller.scorer import PathScorer


def link_key(link):
    return f'{link[0]}-{link[1]}'


def move_load(stats, path, traffic):
    """Add traffic (bytes/s, may be negative) to every link of a switch path."""
    for link in path_links(path):
        if link in stats:
            s = stats[link]
            s['bytes_per_sec'] = max(0.0, s['bytes_per_sec'] + traffic)
            s['utilisation'] = s['bytes_per_sec'] / s['capacity']


@dataclass
class Env:
    flows_on: Callable
    flow_rate: Callable
    switch_path: Callable
    set_path: Callable
    latency: Callable
    log: Callable = lambda kind, **data: None


class ResilientDecision:
    def __init__(self, graph, cfg, rng=None, clock=None):
        ablation = getattr(cfg, 'ablation', {}) or {}
        self.use_capacity_filter = ablation.get('capacity_filter', True)
        self.use_hysteresis = ablation.get('hysteresis', True)
        self.use_scoring = ablation.get('scoring', 'multi_objective') == 'multi_objective'
        self.relief_bypass = getattr(cfg, 'relief_bypass', False)
        self.gate_all = getattr(cfg, 'dpp_gate_all_candidates', False)
        self.order_by_severity = getattr(cfg, 'order_by_severity', False)
        self.fallback = getattr(cfg, 'fallback_least_loaded', False)
        if not ablation.get('trend_trigger', True):
            cfg.N_intervals = 1
        kw = {'rng': rng} if clock is None else {'rng': rng, 'clock': clock}
        self.detector = CongestionDetector(cfg, **kw)
        self.path_engine = PathEngine(graph, cfg)
        self.scorer = PathScorer(cfg)
        self.hysteresis = HysteresisManager(cfg, **kw)
        self.ablation = ablation

    def on_interval(self, link_stats, env: Env):
        congested = self.detector.update(link_stats)
        if not congested:
            return
        env.log('congestion', links=[link_key(l) for l in congested],
                confidence={link_key(l): self.detector.get_attack_confidence(l)
                            for l in congested})
        projected = {l: dict(s) for l, s in link_stats.items()}
        if self.order_by_severity:
            congested = sorted(congested, key=lambda l: -projected[l]['utilisation'])
        for link in congested:
            excess = (projected[link]['bytes_per_sec']
                      - self.detector.threshold * projected[link]['capacity'])
            for flow in env.flows_on(link):
                if excess <= 0:
                    break
                traffic = env.flow_rate(flow.id)
                if traffic > 0 and self._try_reroute(flow, link, traffic, projected, env):
                    excess -= traffic
            if excess > 0:
                # No multipath actuation: one src/dst flow is not split (see the paper, Sec. III).
                env.log('unresolved', link=link_key(link), excess=round(excess, 1))

    def _try_reroute(self, flow, link, traffic, projected, env) -> bool:
        current = env.switch_path(flow.id)
        without = {l: dict(s) for l, s in projected.items()}
        move_load(without, current, -traffic)   # the network if this flow left

        avoiding = [p for p in self.path_engine.all_paths(flow.src, flow.dst)
                    if link not in path_links(p) and p != current]
        if self.use_capacity_filter:
            candidates = [p for p in avoiding
                          if self.path_engine._passes_capacity_filter(p, traffic, without)]
        else:
            candidates = avoiding
        fallback = False
        if not candidates and self.fallback and avoiding:
            best = self._least_loaded(avoiding, traffic, without)
            if best is not None and (self.path_engine.compute_post_reroute_utilisation(
                    best, traffic, without) < projected[link]['utilisation']):
                candidates, fallback = [best], True
        if not candidates:
            env.log('no_candidate', flow=flow.id, link=link_key(link))
            return False

        latency = env.latency()
        if self.gate_all and self.use_hysteresis and not fallback:
            self.scorer.update_stats(projected)
            cur = self.scorer.dpp_score(current, 0, current)
            self.scorer.update_stats(without)
            dpp = {tuple(p): self.scorer.dpp_score(p, traffic, current) for p in candidates}
            gated = [p for p in candidates if dpp[tuple(p)] - cur > self.hysteresis.epsilon]
            if not gated:
                env.log('held', flow=flow.id, new_dpp=round(max(dpp.values()), 4),
                        current_dpp=round(cur, 4), gated=len(candidates))
                return False
            candidates = gated
        self.scorer.update_stats(without)
        if self.use_scoring:
            best = self.scorer.rank_candidates(candidates, traffic, current, latency)[0][0]
        else:
            best = min(candidates, key=lambda p: self.scorer.path_latency(p, latency))
        self._log_decision(env, flow, candidates, best, traffic, current, latency)

        new_dpp = self.scorer.dpp_score(best, traffic, current)   # stats: without the flow
        self.scorer.update_stats(projected)
        current_dpp = self.scorer.dpp_score(current, 0, current)  # stats: with the flow
        if self.use_hysteresis and not self.hysteresis.should_reroute(
                flow.id, new_dpp, current_dpp, link,
                relief=self.relief_bypass or fallback):
            env.log('held', flow=flow.id, new_dpp=round(new_dpp, 4),
                    current_dpp=round(current_dpp, 4))
            return False

        why = f'congestion on {link_key(link)}' + (' (fallback)' if fallback else '')
        if not env.set_path(flow, best, why):
            return False
        self.hysteresis.record_reroute(flow.id, new_dpp, best, old_path=current)
        move_load(projected, current, -traffic)
        move_load(projected, best, traffic)
        return True

    def _least_loaded(self, paths, traffic, stats):
        known = [p for p in paths if all(l in stats for l in path_links(p))]
        if not known:
            return None
        return min(known, key=lambda p: (self.path_engine.compute_post_reroute_utilisation(
            p, traffic, stats), len(p)))

    def _log_decision(self, env, flow, candidates, best, traffic, current, latency):
        """Objectives of every candidate, for the Pareto analysis."""
        best_latency = self.scorer.best_latency(candidates, latency)
        objectives = [[round(x, 4) for x in self.scorer.objectives(
                          p, traffic, current, latency, best_latency)]
                      for p in candidates]
        env.log('decision', flow=flow.id, objectives=objectives,
                selected=candidates.index(best))
