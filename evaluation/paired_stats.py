"""Paired statistics (revision 4), for emulation CSVs (all_results.csv) and simulation CSVs.

Every system in run r of a (topology, scenario) cell faces the same seeded traffic, so the
comparison is paired. For each cell, metric and system X against a reference system R:
  * median and mean of the per-run difference X - R, with a 95% percentile-bootstrap CI of the
    mean difference (10,000 resamples, fixed seed)
  * Wilcoxon signed-rank test (two-sided, zero differences dropped: 'wilcox' zero_method; reported
    as NaN when every difference is zero)
  * matched-pairs rank-biserial correlation r_rb in [-1, 1] as effect size
    (negative = X lower than R)
  * Holm-Bonferroni correction over every test in one family (one call = one family)

    .venv\\Scripts\\python -m evaluation.paired_stats results/sim/all_udp.csv --ref resilient_sdn \\
        --metrics time_above_s victim_loss_pct reroutes
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

BOOT = 10_000


def bootstrap_ci(diff, rng, level=0.95):
    diff = np.asarray(diff, float)
    if len(diff) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return lo, hi


def rank_biserial(diff):
    d = np.asarray(diff, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    pos, neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return (pos - neg) / ranks.sum()


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    order = np.argsort(p[ok])
    m = ok.sum()
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[ok][i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def paired_table(df, ref, metrics, keys=('topology', 'scenario'), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cell, g in df.groupby(list(keys)):
        base = g[g.system == ref].set_index('run')
        for system, sg in g.groupby('system'):
            if system == ref:
                continue
            sg = sg.set_index('run')
            common = base.index.intersection(sg.index)
            for m in metrics:
                a = sg.loc[common, m].astype(float)
                b = base.loc[common, m].astype(float)
                ok = a.notna() & b.notna()
                diff = (a[ok] - b[ok]).to_numpy()
                if len(diff) == 0:
                    continue
                if np.all(diff == 0):
                    p = np.nan
                else:
                    try:
                        p = stats.wilcoxon(diff, zero_method='wilcox').pvalue
                    except ValueError:
                        p = np.nan
                lo, hi = bootstrap_ci(diff, rng)
                rows.append({**dict(zip(keys, cell if isinstance(cell, tuple) else (cell,))),
                             'system': system, 'reference': ref, 'metric': m, 'n': len(diff),
                             'mean_system': a[ok].mean(), 'mean_ref': b[ok].mean(),
                             'mean_diff': diff.mean(), 'median_diff': float(np.median(diff)),
                             'ci_lo': lo, 'ci_hi': hi, 'p_wilcoxon': p,
                             'rank_biserial': rank_biserial(diff),
                             'wins': int((diff < 0).sum()), 'ties': int((diff == 0).sum()),
                             'losses': int((diff > 0).sum())})
    out = pd.DataFrame(rows)
    if len(out):
        out['p_holm'] = holm(out['p_wilcoxon'].to_numpy())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--ref', default='resilient_sdn')
    ap.add_argument('--metrics', nargs='+', default=['time_above_s', 'victim_loss_pct', 'reroutes'])
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    df = pd.read_csv(a.csv)
    t = paired_table(df, a.ref, a.metrics)
    out = a.out or a.csv.replace('.csv', f'_paired_vs_{a.ref}.csv')
    t.to_csv(out, index=False)
    print(out)


if __name__ == '__main__':
    sys.exit(main())
