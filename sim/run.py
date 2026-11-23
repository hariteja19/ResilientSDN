"""Flow-level simulation campaign (revision 4). Runs on Windows, no Mininet needed:

    .venv\\Scripts\\python -m sim.run --out results/sim --topologies fat_tree nsfnet \\
        --systems static simple_reactive resilient_sdn te_lp --scenarios S3 S11 S12 --runs 30

Timeline per run (same as the emulation): legitimate traffic from t = 0, attack from
t = attack_delay_seconds for the scenario's duration, 3 s tail. The controller polls every
polling_interval s and sees the mean carried link load and the ingress flow rates of the last
interval. Seeds: traffic is drawn from (topology, scenario, run), identical for every system.
"""
import argparse
import copy
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

from controller.config import PROJECT_ROOT, load_config, load_yaml
from controller.path_engine import ecmp_choice, path_links
from sim.network import FluidNetwork, SimFlow
from sim.scenarios import ADAPT_S, SIM_SCENARIOS, MultiPlanner, const, schedule
from sim.systems import SYSTEMS, make_system

MAIN_SYSTEMS = ['static', 'static_ecmp', 'simple_reactive', 'resilient_sdn', 'resilient_sdn_v2',
                'mp_ksp', 'te_lp', 'te_lp_excess']
from topology import load_topology

FIELDS = ['system', 'scenario', 'pattern_scenario', 'topology', 'run', 'legit_mode',
          'time_above_s', 'link_time_above_s', 'mlu_mean', 'mlu_p95',
          'victim_loss_pct', 'legit_loss_pct', 'tcp_goodput_mbps', 'fct_mean_s', 'fct_p95_s',
          'transfers', 'latency_ms', 'reroutes', 'oscillations', 'rule_changes',
          'secondary_congestion', 'n_targets', 'attacker', 'wall_s']

TCP = {'size_mb': 5.0, 'cap_mbps': 20.0, 'think_s': 1.0}


def ip(planner, host):
    return planner.ip_of[host]


def build(topology, scenario, run_id, legit_mode, exp_cfg, ecmp_aware=False):
    rng = random.Random(f'{topology}-{scenario}-{run_id}')
    s_id = scenario
    randomise = bool(SIM_SCENARIOS[scenario].get('randomise'))
    if randomise:
        s_id = rng.choice(SIM_SCENARIOS[scenario]['same_as'])
    s = SIM_SCENARIOS[s_id]
    topo = load_topology(topology)
    planner = MultiPlanner(topo, ecmp_aware=ecmp_aware)
    legit = exp_cfg['legit']
    if s.get('targets'):
        plan = planner.plan_multi(s['targets'], legit, s['attackers'], rng)
    elif s.get('bursty'):   # ablation D: S1 traffic plus legitimate bursts
        plan = planner.plan('S1', legit, rng)
        plan['targets'] = [plan['target_link']]
    elif s.get('adaptive'):
        plan = planner.plan('S11', legit, rng)
        plan['targets'] = [plan['target_link']]
    else:
        plan = planner.plan(s_id, legit, rng)
        plan['targets'] = [plan['target_link']]
    delay = exp_cfg['attack_delay_seconds']
    seconds = s.get('duration', exp_cfg['duration_seconds'])
    t0, t1 = delay, delay + seconds
    flows = []

    def mk(pair, kind, rate_fn=None):
        a, b = pair
        f = SimFlow(id=f'{ip(planner, a)}->{ip(planner, b)}', src=planner.switch_of[a],
                    dst=planner.switch_of[b], kind=kind, rate_fn=rate_fn)
        flows.append(f)
        return f

    end = t1 + 3
    for p in plan['legit']:
        if legit_mode == 'tcp':
            f = mk(p, 'tcp')
            f.tcp_cap_mbps, f.think_s = TCP['cap_mbps'], TCP['think_s']
            f.size_bytes = TCP['size_mb'] * 1e6
        elif s.get('bursty'):
            # ablation D (experiment_config 'bursty'): the legitimate pair itself bursts, as
            # LegitimateTraffic.start_bursty does, so it is the same flow (same address pair)
            bc = exp_cfg['bursty']
            base, burst = legit['rate_mbps'], bc['burst_mbps']
            per, dur = bc['period_seconds'], bc['burst_seconds']
            mk(p, 'legit', lambda t, base=base, burst=burst, per=per, dur=dur:
               (base + (burst if t >= t0 and (t - t0) % per < dur else 0.0)) if t < end else 0.0)
        else:
            mk(p, 'legit', const(legit['rate_mbps'], 0, end))
    for p in plan['background']:
        mk(p, 'background', const(legit['background_rate_mbps'], 0, end))
    legit_total = legit['rate_mbps'] * len(plan['legit'])
    if s.get('targets') or s.get('adaptive'):
        attack_fn = const(s['rate_mbps'], t0, t1)
    else:
        attack_fn = schedule(s, plan, legit_total, t0, t1)
    for p in plan['attack']:
        mk(p, 'attack', attack_fn)
    for (a, b, link_mbps, _) in plan.get('preload', []):
        mk((a, b), 'preload', const(s['preload_alt_paths'] * link_mbps, t0, t1))
    return {'planner': planner, 'plan': plan, 'flows': flows, 's': s, 's_id': s_id,
            'randomise': randomise, 't0': t0, 't1': t1, 'end': end, 'rng': rng,
            'attack_fn': attack_fn, 'graph': planner.g, 'ecmp_aware': ecmp_aware}


def controller_cfg(topology, scenario, run_id, randomise):
    cfg = load_config()
    hops = getattr(load_topology(topology), 'max_path_hops', None)
    if hops:
        cfg.max_path_hops = hops
    cfg.seed = f'{topology}-{scenario}-{run_id}'
    if randomise:
        cfg.adversarial = dict(cfg.adversarial, randomise_N=True, randomise_cooldown=True)
    return cfg


def run_single(topology, system, scenario, run_id, legit_mode='udp', exp_cfg=None,
               ecmp_aware=False, tcp_floor=None):
    wall = time.time()
    exp_cfg = exp_cfg or load_yaml('config/experiment_config.yaml')
    b = build(topology, scenario, run_id, legit_mode, exp_cfg, ecmp_aware)
    cfg = controller_cfg(topology, scenario, run_id, b['randomise'])
    st = simulate(b, system, cfg, tcp_floor)
    return _metrics(b, st['net'], st['sysm'], st['samples'], cfg.utilisation_threshold,
                    cfg.polling_interval, st['victims'], st['latency_acc'], st['latency_n'],
                    system, scenario, topology, run_id, legit_mode, wall)


def simulate(b, system, cfg, tcp_floor=None, oracle=False):
    """Run one built instance b under one system. Besides what _metrics needs, returns the
    objective per second of the attack window (phi: sum_e [u_e - theta]^+ on offered load) and, with
    oracle=True, its per-poll optimum over all routings (sim/oracle.py)."""
    g, planner, plan = b['graph'], b['planner'], b['plan']
    sysm = make_system(system, g, cfg, cfg.seed)
    net = FluidNetwork(g) if tcp_floor is None else FluidNetwork(g, tcp_floor)
    for f in b['flows']:
        f.paths = sysm.initial_paths(f)
        net.add(f)
    th, poll = cfg.utilisation_threshold, cfg.polling_interval
    targets = list(plan['targets'])
    samples = []        # (t, {link: util}) at each poll
    acc = {l: 0.0 for l in net.cap}
    adaptive = b['s'].get('adaptive')
    victims = [f for f in b['flows'] if f.kind in ('legit', 'tcp')]
    attack_flows = [f for f in b['flows'] if f.kind == 'attack']
    used_pairs = set(plan['legit']) | set(plan['background']) | set(plan['attack'])
    latency_acc, latency_n = 0.0, 0
    lat = {}
    for a, c, d in g.edges(data=True):
        lat[(a, c)] = lat[(c, a)] = d['latency']
    phi, phi_star_sum, cache = 0.0, 0.0, {}
    t = 0
    while t < b['end']:
        in_window = b['t0'] <= t < b['t1']
        if adaptive and in_window and (t - b['t0']) % ADAPT_S == 0 and t > b['t0']:
            attack_flows, targets = _adapt(planner, net, victims, attack_flows, b, sysm,
                                           used_pairs, targets)
        sysm.now = t
        carried = net.step(t, in_window)
        for l, v in carried.items():
            acc[l] += v
        if in_window:
            phi += sum(max(0.0, v / net.cap[l] - th) for l, v in net.last_offered.items())
            if oracle and (t + 1) % poll == 0:
                dem = {}
                for f in net.flows.values():
                    if f.active and f.rate_fn is not None and f.measured_rate > 0:
                        dem[(f.src, f.dst)] = dem.get((f.src, f.dst), 0.0) + f.measured_rate
                key = tuple(sorted((k, round(v)) for k, v in dem.items()))
                if key not in cache:
                    from sim.oracle import phi_star
                    cache[key] = phi_star(g, dem, th)[0]
                phi_star_sum += cache[key] * poll
            for f in victims:
                latency_acc += sum(fr * sum(lat[l] for l in path_links(p))
                                   for p, fr in f.paths.items())
                latency_n += 1
        if (t + 1) % poll == 0:
            stats = {l: {'bytes_per_sec': acc[l] / poll, 'capacity': net.cap[l],
                         'utilisation': acc[l] / poll / net.cap[l]} for l in net.cap}
            acc = {l: 0.0 for l in net.cap}
            samples.append((t + 1, {l: s['utilisation'] for l, s in stats.items()},
                            list(targets)))
            sysm.now = t + 1
            sysm.on_interval(net, stats)
        t += 1
    return {'net': net, 'sysm': sysm, 'samples': samples, 'victims': victims,
            'latency_acc': latency_acc, 'latency_n': latency_n, 'phi': phi,
            'phi_star': phi_star_sum if oracle else float('nan')}


def _adapt(planner, net, victims, attack_flows, b, sysm, used, targets):
    """Rolling attacker: re-aim at the link most victim routes share now."""
    victim_paths = [max(f.paths, key=f.paths.get) for f in victims]
    link = planner.adaptive_target(victim_paths)
    if link is None:
        return attack_flows, targets
    keep = [f for f in attack_flows
            if any(link in path_links(p) for p in f.paths)]
    need = len(attack_flows) - len(keep)
    if need == 0:
        return attack_flows, [link]
    for f in attack_flows:
        if f not in keep:
            f.active = False
    # Routing-aware: the attacker probes how the defence routes a new flow of each candidate pair
    # (as with traceroute) and prefers pairs that put the largest share of their rate on the link.
    def share(pair):
        a, c = pair
        probe = SimFlow(id=f'{planner.ip_of[a]}->{planner.ip_of[c]}', src=planner.switch_of[a],
                        dst=planner.switch_of[c], kind='attack')
        return sum(fr for p, fr in sysm.initial_paths(probe).items() if link in path_links(p))
    pool = [p for p in planner.paths if p not in used and planner.switch_of[p[0]] != planner.switch_of[p[1]]]
    b['rng'].shuffle(pool)
    pool = sorted(((share(p), p) for p in pool), key=lambda x: -x[0])
    pool = [p for sh, p in pool if sh > 0]
    new = []
    for pair in pool[:need]:
        used.add(pair)
        a, c = pair
        f = SimFlow(id=f'{planner.ip_of[a]}->{planner.ip_of[c]}', src=planner.switch_of[a],
                    dst=planner.switch_of[c], kind='attack', rate_fn=b['attack_fn'])
        f.paths = sysm.initial_paths(f)
        net.add(f)
        new.append(f)
    return keep + new, [link]


def _metrics(b, net, sysm, samples, th, poll, victims, latency_acc, latency_n,
             system, scenario, topology, run_id, legit_mode, wall):
    win = [(t, u, tg) for t, u, tg in samples if b['t0'] < t <= b['t1']]
    time_above = sum(poll for _, u, tg in win if any(u[l] > th for l in tg))
    link_above = sum(poll * sum(u[l] > th for l in tg) for _, u, tg in win)
    mlu = [max(u.values()) for _, u, _ in win]
    all_targets = {l for _, _, tg in win for l in tg}
    secondary = set()
    for t, u, _ in win:
        for l, v in u.items():
            if v > th and l not in all_targets and l in sysm.moved_onto and t > sysm.moved_onto[l]:
                secondary.add(l)
    flows = list(net.flows.values())

    def loss(kinds, window=True):
        s = sum(f.sent_attack_window if window else f.sent for f in flows if f.kind in kinds)
        d = sum(f.delivered_attack_window if window else f.delivered for f in flows if f.kind in kinds)
        return 100.0 * (1 - d / s) if s > 0 else float('nan')

    tcp = [f for f in flows if f.kind == 'tcp']
    fcts = [d for f in tcp for (st, d) in f.fcts if b['t0'] <= st < b['t1']]
    win_s = b['t1'] - b['t0']
    return {
        'system': system, 'scenario': scenario, 'pattern_scenario': b['s_id'],
        'topology': topology, 'run': run_id, 'legit_mode': legit_mode,
        'time_above_s': time_above, 'link_time_above_s': link_above,
        'mlu_mean': round(float(np.mean(mlu)), 4) if mlu else 0.0,
        'mlu_p95': round(float(np.percentile(mlu, 95)), 4) if mlu else 0.0,
        'victim_loss_pct': round(loss({'legit'}), 4) if legit_mode == 'udp' else float('nan'),
        'legit_loss_pct': round(loss({'legit', 'background'}), 4),
        'tcp_goodput_mbps': (round(sum(f.delivered_attack_window for f in tcp) * 8 / 1e6
                                   / win_s / len(tcp), 4) if tcp else float('nan')),
        'fct_mean_s': round(float(np.mean(fcts)), 4) if fcts else float('nan'),
        'fct_p95_s': round(float(np.percentile(fcts, 95)), 4) if fcts else float('nan'),
        'transfers': len(fcts),
        'latency_ms': round(latency_acc / latency_n, 3) if latency_n else float('nan'),
        'reroutes': sysm.reroutes, 'oscillations': sysm.oscillations,
        'rule_changes': sysm.rule_changes, 'secondary_congestion': len(secondary),
        'n_targets': len(b['plan']['targets']),
        'attacker': 'ecmp_aware' if b.get('ecmp_aware') else 'static_routes', 'wall_s': round(time.time() - wall, 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/sim')
    ap.add_argument('--topologies', nargs='+', default=['fat_tree', 'nsfnet', 'internet2', 'geant'])
    ap.add_argument('--systems', nargs='+', default=MAIN_SYSTEMS, choices=list(SYSTEMS))
    ap.add_argument('--scenarios', nargs='+', default=['S1', 'S3', 'S5', 'S6', 'S7', 'S8', 'S9',
                                                        'S10', 'S11', 'S12', 'S13', 'S14'])
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--legit', choices=['udp', 'tcp'], default='udp')
    ap.add_argument('--tcp-floor', type=float, default=None,
                    help='TCP share of a saturated link (model assumption; default sim.network.TCP_FLOOR)')
    ap.add_argument('--ecmp-aware', action='store_true',
                    help='attacker picks pairs that cross the target under static AND ECMP routing')
    a = ap.parse_args(argv)
    out = PROJECT_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    floor = '' if a.tcp_floor is None else f'_floor{a.tcp_floor:g}'
    path = out / f'sim_{a.legit}{"_ecmpaware" if a.ecmp_aware else ""}{floor}.csv'
    done = set()
    if path.exists():
        with open(path, newline='') as f:
            done = {(r['topology'], r['system'], r['scenario'], int(r['run'])) for r in csv.DictReader(f)}
    exp_cfg = load_yaml('config/experiment_config.yaml')
    new = not path.exists()
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for topo in a.topologies:
            for scen in a.scenarios:
                for r in range(a.runs):
                    for sysname in a.systems:
                        if (topo, sysname, scen, r) in done:
                            continue
                        try:
                            row = run_single(topo, sysname, scen, r, a.legit, exp_cfg, a.ecmp_aware,
                                             a.tcp_floor)
                        except Exception as e:   # record and continue; reported at the end
                            print(f'FAILED {topo} {sysname} {scen} {r}: {e!r}', file=sys.stderr)
                            continue
                        w.writerow(row)
                    fh.flush()
                print(f'{topo} {scen} done', flush=True)
    print(path)


if __name__ == '__main__':
    main()
