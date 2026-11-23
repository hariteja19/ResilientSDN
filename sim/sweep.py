"""Randomised sweep over the general problem space (revision 4; sim/generic.py defines the space).

    .venv\\Scripts\\python -m sim.sweep --start 0 --count 250 --out results/sweep

For every instance: the same seeded traffic for every system, the objective Phi of each system
(integral over the attack of sum_e [u_e - theta]^+ on offered load, in excess-seconds) and its optimum
Phi* over all routings (sim/oracle.py), so gap = Phi - Phi* >= 0 and an instance is *defendable* when
Phi* = 0. Controller settings for the sweep: hop limit H = max(6, diameter + 2) on random graphs (the
configured value on real ones) and candidate cap K = 20 lowest-latency paths (path_engine.max_candidates),
because enumerating all simple paths is intractable on dense random graphs.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import networkx as nx

from controller.config import PROJECT_ROOT, load_config
from sim.generic import REAL, build_instance, sample_params
from sim.run import _metrics, simulate

SYSTEMS = ['static', 'mp_ksp', 'simple_reactive', 'te_lp_excess', 'resilient_sdn', 'resilient_sdn_v2',
           'resilient_sdn_v3']
MAX_CANDIDATES = 20
PARAMS = ['instance', 'topology', 'nodes', 'n_flows', 'base_load', 'targets', 'bots', 'alpha', 'pattern',
          'pulse_period', 'pulse_duty']
FEATURES = ['n_nodes', 'n_links', 'avg_degree', 'edge_connectivity', 'n_targets', 'n_attack_flows',
            'n_legit_flows', 'n_victim_flows']
METRICS = ['time_above_s', 'link_time_above_s', 'mlu_mean', 'victim_loss_pct', 'legit_loss_pct',
           'reroutes', 'oscillations', 'rule_changes', 'secondary_congestion']
FIELDS = PARAMS + FEATURES + ['system'] + METRICS + ['phi', 'phi_star', 'gap', 'defendable', 'wall_s']


def controller_cfg(p, graph, instance):
    cfg = load_config()
    if p['topology'] in REAL:
        from topology import load_topology
        hops = getattr(load_topology(p['topology']), 'max_path_hops', None)
    else:
        hops = max(6, nx.diameter(graph) + 2)
    if hops:
        cfg.max_path_hops = hops
    cfg.max_candidates = MAX_CANDIDATES
    cfg.seed = f'instance-{instance}'
    return cfg


def run_instance(instance, systems=SYSTEMS):
    p = sample_params(instance)
    built = build_instance(p)
    if built is None:
        return []
    rows, phi_star_shared = [], None
    for system in systems:
        wall = time.time()
        b, feat = build_instance(p)                     # fresh flows for every system
        cfg = controller_cfg(p, b['graph'], instance)
        rolling = p['pattern'] == 'rolling'
        need_oracle = rolling or phi_star_shared is None
        st = simulate(b, system, cfg, oracle=need_oracle)
        if not rolling and need_oracle:
            phi_star_shared = st['phi_star']
        ps = st['phi_star'] if need_oracle else phi_star_shared
        m = _metrics(b, st['net'], st['sysm'], st['samples'], cfg.utilisation_threshold,
                     cfg.polling_interval, st['victims'], st['latency_acc'], st['latency_n'],
                     system, p['pattern'], p['topology'], instance, 'udp', wall)
        row = {k: p[k] for k in PARAMS}
        row.update(feat)
        row.update({k: m[k] for k in METRICS})
        row.update({'system': system, 'phi': round(st['phi'], 4), 'phi_star': round(ps, 4),
                    'gap': round(st['phi'] - ps, 4), 'defendable': ps < 1e-6,
                    'wall_s': round(time.time() - wall, 2)})
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--out', default='results/sweep')
    ap.add_argument('--systems', nargs='+', default=SYSTEMS)
    a = ap.parse_args(argv)
    out = PROJECT_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'sweep_{a.start:05d}.csv'
    done = set()
    if path.exists():
        with open(path, newline='') as f:
            done = {int(r['instance']) for r in csv.DictReader(f)}
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for i in range(a.start, a.start + a.count):
            if i in done:
                continue
            try:
                rows = run_instance(i, a.systems)
            except Exception as e:      # recorded, the sweep continues
                print(f'FAILED instance {i}: {e!r}', file=sys.stderr, flush=True)
                continue
            for r in rows:
                w.writerow(r)
            fh.flush()                  # one instance = all its systems, written together
            print(f'instance {i} done ({len(rows)} rows)', flush=True)
    print(path)


if __name__ == '__main__':
    main()
