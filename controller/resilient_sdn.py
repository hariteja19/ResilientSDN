"""ResilientSDN: trend-based detection, capacity-filtered candidates,
multi-objective scoring and DPP hysteresis.

Run: ryu-manager controller/resilient_sdn.py

Each polling interval, for every link the detector confirms as congested,
flows on it are moved (largest first) until the link's projected load is back
under the threshold. After each move the projected link loads are updated, so
later decisions in the same round see where earlier flows went.
The decision logic (Algorithm 1) lives in controller/decision.py, shared with the
flow-level simulator in sim/. Ablation switches (config `ablation:`) turn single
mechanisms off.
"""

import random

from controller.decision import Env, ResilientDecision, move_load  # noqa: F401 (move_load re-exported)
from controller.routing_base import RoutingBase


class ResilientSDN(RoutingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        rng = random.Random(getattr(cfg, 'seed', None))  # seeded per run so S10 is reproducible
        self.core = ResilientDecision(self.graph, cfg, rng=rng)
        # kept as attributes for code and tools that read them from the app
        self.detector, self.path_engine = self.core.detector, self.core.path_engine
        self.scorer, self.hysteresis = self.core.scorer, self.core.hysteresis
        self.env = Env(flows_on=self.flows_on, flow_rate=self.monitor.get_flow_rate,
                       switch_path=self.switch_path, set_path=self.set_path,
                       latency=self._latency, log=self.log_event)
        self.log_event('config', ablation=self.core.ablation, N=cfg.N_intervals)

    def on_interval(self, link_stats):
        self.core.on_interval(link_stats, self.env)

    def _latency(self):
        """Measured latency where available, configured latency otherwise."""
        latency = {}
        for a, b, d in self.graph.edges(data=True):
            latency[(a, b)] = latency[(b, a)] = d['latency']
        latency.update(self.monitor.latency)
        return latency
