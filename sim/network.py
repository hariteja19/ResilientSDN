"""Fluid (flow-level) network model, one step per second.

Model (stated in the paper, Sec. V-F):
  * Directed links, capacity c_e = bw * 1e6 / 8 bytes/s in each direction.
  * UDP flows are inelastic: they send at their scheduled rate. A link whose arriving UDP load
    exceeds its capacity drops the excess in proportion to each flow's arriving rate (drop-tail
    at a full queue); drops upstream reduce what arrives downstream (fixed-point over the path).
  * TCP flows are elastic transfers of a fixed size, capped at an access rate. They share what
    UDP leaves on each link by max-min fairness (progressive filling). A link fully taken by UDP
    still leaves TCP a floor of TCP_FLOOR of its capacity in total (TCP keeps a small share of a
    drop-tail queue instead of starving completely); this is an assumption, not a measurement.
  * Queueing delay, TCP slow start, reordering on a path change and control-plane delay below one
    second are not modelled. Rule installation takes effect at the next step.
  * A flow may be split over several paths (fractions summing to 1); only the multipath baselines
    use this. Measured link load = carried bytes (<= capacity), as OpenFlow port counters report;
    measured flow rate = offered rate at the ingress switch, as OpenFlow flow counters report.
"""
from dataclasses import dataclass, field

from controller.path_engine import path_links

TCP_FLOOR = 0.05


@dataclass
class SimFlow:
    id: str
    src: str                 # ingress switch
    dst: str                 # egress switch
    kind: str                # legit | background | attack | preload | tcp
    rate_fn: object = None   # t -> Mbps offered (UDP); None for TCP
    tcp_cap_mbps: float = 0.0
    paths: dict = field(default_factory=dict)   # {tuple(path): fraction}
    active: bool = True
    # TCP transfer state
    size_bytes: float = 0.0
    remaining: float = 0.0
    started_at: float = 0.0
    think_s: float = 1.0
    next_start: float = 0.0
    fcts: list = field(default_factory=list)
    # accounting
    sent: float = 0.0
    delivered: float = 0.0
    sent_attack_window: float = 0.0
    delivered_attack_window: float = 0.0
    measured_rate: float = 0.0  # bytes/s at the ingress, last step


class FluidNetwork:
    def __init__(self, graph, tcp_floor=TCP_FLOOR):
        self.g = graph
        self.tcp_floor = tcp_floor
        self.cap = {}
        for a, b, d in graph.edges(data=True):
            c = d['bw'] * 1e6 / 8
            self.cap[(a, b)] = self.cap[(b, a)] = c
        self.flows = {}

    def add(self, flow):
        self.flows[flow.id] = flow

    def step(self, t, in_window):
        """Advance one second. Returns {link: carried bytes/s}."""
        udp = [f for f in self.flows.values() if f.rate_fn is not None and f.active]
        tcp = [f for f in self.flows.values() if f.rate_fn is None and f.active]
        # --- UDP: offered per (flow, path), drop fixed-point
        segs = []   # (flow, links, offered bytes/s)
        for f in udp:
            r = max(0.0, f.rate_fn(t)) * 1e6 / 8
            f.measured_rate = r
            for p, frac in f.paths.items():
                if frac > 0 and r > 0:
                    segs.append((f, path_links(p), r * frac))
        # offered load: every UDP segment at its sending rate on every link of its path (no drops);
        # the paper's objective Phi is defined on offered, not carried, utilisation
        offered = {l: 0.0 for l in self.cap}
        for f, links, r in segs:
            for l in links:
                offered[l] += r
        factor = {l: 1.0 for l in self.cap}
        for _ in range(6):
            arriving = {}
            for f, links, r in segs:
                x = r
                for l in links:
                    arriving[l] = arriving.get(l, 0.0) + x
                    x *= factor[l]
            factor = {l: (min(1.0, self.cap[l] / arriving[l]) if arriving.get(l) else 1.0)
                      for l in self.cap}
        carried = {l: 0.0 for l in self.cap}
        delivered = {}
        for f, links, r in segs:
            x = r
            for l in links:
                x *= factor[l]
                carried[l] += x
            delivered[f.id] = delivered.get(f.id, 0.0) + x
        for f in udp:
            f.sent += f.measured_rate
            f.delivered += delivered.get(f.id, 0.0)
            if in_window:
                f.sent_attack_window += f.measured_rate
                f.delivered_attack_window += delivered.get(f.id, 0.0)
        # --- TCP: max-min fair over the residual, per path segment
        residual = {}
        for l, c in self.cap.items():
            residual[l] = max(c - carried[l], self.tcp_floor * c if carried[l] >= c * 0.999 else 0.0)
        running = []
        for f in tcp:
            if f.remaining <= 0 and t >= f.next_start:
                f.remaining, f.started_at = f.size_bytes, t
            if f.remaining > 0:
                for p, frac in f.paths.items():
                    if frac > 0:
                        running.append([f, path_links(p), f.tcp_cap_mbps * 1e6 / 8 * frac, 0.0])
        alloc = self._max_min(running, residual)
        got = {}
        for (f, links, _, _), a in zip(running, alloc):
            got[f.id] = got.get(f.id, 0.0) + a
            for l in links:
                carried[l] += a
                offered[l] += a
        for f in tcp:
            rate = got.get(f.id, 0.0)
            f.measured_rate = rate
            if f.remaining > 0:
                moved = min(rate, f.remaining)
                f.remaining -= moved
                f.delivered += moved
                f.sent += moved
                if in_window:
                    f.delivered_attack_window += moved
                    f.sent_attack_window += moved
                if f.remaining <= 1e-6:
                    # finished within this second: interpolate the completion instant
                    frac = moved / rate if rate > 0 else 1.0
                    f.fcts.append((f.started_at, t + frac - f.started_at))
                    f.remaining, f.next_start = 0.0, t + frac + f.think_s
        self.last_offered = offered
        return {l: min(carried[l], self.cap[l]) for l in self.cap}

    @staticmethod
    def _max_min(segs, residual):
        """Progressive filling with per-segment caps."""
        n = len(segs)
        alloc = [0.0] * n
        frozen = [False] * n
        res = dict(residual)
        for _ in range(n + 1):
            live = [i for i in range(n) if not frozen[i]]
            if not live:
                break
            users = {}
            for i in live:
                for l in segs[i][1]:
                    users[l] = users.get(l, 0) + 1
            inc = min([res[l] / k for l, k in users.items()] +
                      [segs[i][2] - alloc[i] for i in live])
            inc = max(inc, 0.0)
            for i in live:
                alloc[i] += inc
                for l in segs[i][1]:
                    res[l] -= inc
            for i in live:
                if alloc[i] >= segs[i][2] - 1e-9 or any(res[l] <= 1e-9 for l in segs[i][1]):
                    frozen[i] = True
        return alloc
