"""Per-instance optimum of the paper's objective (revision 4).

phi_star(graph, demands, theta) solves the arc-based multicommodity LP
    min  sum_e s_e
    s.t. sum_k f_{k,e} / c_e - s_e <= theta              for every directed link e
         flow conservation of every commodity k
         f, s >= 0
with commodities aggregated by source node (exact for this objective, which depends only on the
total flow per link). Its value Phi* is a lower bound on Phi = sum_e [u_e - theta]^+ for EVERY routing
of the same demands -- any paths, split or unsplit, any hop count -- so the gap Phi(system) - Phi*
measures how far a system is from the best any controller could do with the same traffic
(Sec. III-C, Eq. 3 relaxed). Phi* = 0 means the instance is defendable: some routing keeps every link
at or below theta.

cut_bound(graph, demands, theta) is the closed-form lower bound of Proposition 4 over single-link and
single-node cuts.
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

EPS_FLOW = 1e-7    # tiny cost per unit of flow: removes useless cycles, does not change Phi*


def _arcs(graph):
    arcs, cap = [], []
    for a, b, d in graph.edges(data=True):
        c = d['bw'] * 1e6 / 8
        arcs += [(a, b), (b, a)]
        cap += [c, c]
    return arcs, np.array(cap)


def phi_star(graph, demands, theta):
    """demands: {(src_switch, dst_switch): bytes/s}. Returns (Phi*, {arc: optimal utilisation})."""
    demands = {k: v for k, v in demands.items() if v > 0 and k[0] != k[1]}
    if not demands:
        return 0.0, {}
    arcs, cap = _arcs(graph)
    nodes = list(graph.nodes)
    nidx = {v: i for i, v in enumerate(nodes)}
    by_src = defaultdict(dict)
    for (s, d), r in demands.items():
        by_src[s][d] = by_src[s].get(d, 0.0) + r
    sources = sorted(by_src, key=str)
    A, K = len(arcs), len(sources)
    nf = K * A                         # f[k, a] at k * A + a, then s[a] at nf + a
    n = nf + A
    c = np.concatenate([np.full(nf, EPS_FLOW / cap.mean()), np.ones(A)])
    # capacity / excess rows: sum_k f_ka / c_a - s_a <= theta
    rows, cols, vals = [], [], []
    for k in range(K):
        rows += list(range(A)); cols += list(range(k * A, k * A + A)); vals += list(1.0 / cap)
    rows += list(range(A)); cols += list(range(nf, n)); vals += [-1.0] * A
    A_ub = coo_matrix((vals, (rows, cols)), shape=(A, n)).tocsr()
    b_ub = np.full(A, theta)
    # conservation rows: for source k, node v: out - in = supply
    rows, cols, vals, b_eq = [], [], [], []
    scale = cap.mean()
    for k, s in enumerate(sources):
        base = k * len(nodes)
        for a, (u, v) in enumerate(arcs):
            rows += [base + nidx[u], base + nidx[v]]
            cols += [k * A + a, k * A + a]
            vals += [1.0 / scale, -1.0 / scale]
        supply = np.zeros(len(nodes))
        for d, r in by_src[s].items():
            supply[nidx[s]] += r / scale
            supply[nidx[d]] -= r / scale
        b_eq += list(supply)
    A_eq = coo_matrix((vals, (rows, cols)), shape=(K * len(nodes), n)).tocsr()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array(b_eq),
                  bounds=(0, None), method='highs')
    if not res.success:
        raise RuntimeError(f'oracle LP failed: {res.message}')
    f = res.x[:nf].reshape(K, A).sum(axis=0)
    util = {arcs[a]: f[a] / cap[a] for a in range(A)}
    phi = float(sum(max(0.0, u - theta) for u in util.values()))
    return phi, util


def cut_bound(graph, demands, theta):
    """Proposition 4 over every single-node cut ({v} vs the rest, both directions) and every
    single-link cut of a bridge: Phi >= (D_cut - theta * c(cut)) / c_max(cut)."""
    import networkx as nx
    best = 0.0
    cap = {}
    for a, b, d in graph.edges(data=True):
        cap[(a, b)] = cap[(b, a)] = d['bw'] * 1e6 / 8
    cuts = [({v}, set(graph.nodes) - {v}) for v in graph.nodes]
    for a, b in nx.bridges(graph):
        h = graph.copy()
        h.remove_edge(a, b)
        side = nx.node_connected_component(h, a)
        cuts.append((side, set(graph.nodes) - side))
    for S, T in cuts:
        for X, Y in ((S, T), (T, S)):
            links = [(u, v) for u in X for v in graph.neighbors(u) if v in Y]
            if not links:
                continue
            demand = sum(r for (s, d), r in demands.items() if s in X and d in Y)
            ccut = sum(cap[l] for l in links)
            best = max(best, (demand - theta * ccut) / max(cap[l] for l in links))
    return best
