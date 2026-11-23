"""Tables, figures and statistics for the revision-4 sections, from the result CSVs:

    .venv\\Scripts\\python paper_rev4/make_assets.py

Inputs (whatever exists is used, nothing is invented for missing cells):
    results/sim/<topology>/sim_udp.csv, sim_tcp.csv, sim_udp_ecmpaware.csv   (sim.run)
    results/wan_ext/all_results.csv                                          (Mininet, run_wan_ext.sh)
    paper_rev4/data/emulation_v2_headline.csv                                (Oct 2026 campaign)
Outputs: paper_rev4/generated/*.tex, paper_rev4/figures/*.pdf|png, paper_rev4/generated/*.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.style import SYSTEM_COLORS, SYSTEM_LABELS, save, setup, systems_in  # noqa: E402
from evaluation.paired_stats import paired_table  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
GEN, FIG = HERE / 'generated', HERE / 'figures'
GEN.mkdir(exist_ok=True)
TOPO = {'fat_tree': 'Fat-tree', 'nsfnet': 'NSFNET', 'internet2': 'Internet2', 'geant': "G\\'EANT"}
TOPO_PLAIN = {'fat_tree': 'Fat-tree', 'nsfnet': 'NSFNET', 'internet2': 'Internet2', 'geant': 'GÉANT'}
SYS = ['static', 'static_ecmp', 'mp_ksp', 'simple_reactive', 'te_lp', 'te_lp_excess', 'resilient_sdn',
       'resilient_sdn_v2']
LAB = {s: SYSTEM_LABELS.get(s, s) for s in SYS}
SHORT = {'rsdn_duty': '+Duty', 'resilient_sdn_v3': 'RSDN-v3', 'rsdn_severity': '+Severity', 'rsdn_fallback': '+Fallback', 'rsdn_bottleneck': '+Bottleneck',
         'static': 'Static', 'static_ecmp': 'ECMP', 'mp_ksp': 'k-MP', 'simple_reactive': 'Reactive',
         'te_lp': 'TE-LP', 'te_lp_excess': r'TE-$\Phi$', 'resilient_sdn': 'RSDN', 'resilient_sdn_v2': 'RSDN-v2'}


def load(name, folder='sim'):
    parts = sorted((ROOT / 'results' / folder).glob(f'*/{name}.csv'))
    if not parts:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(p, on_bad_lines='skip') for p in parts], ignore_index=True)
    return d.drop_duplicates(['topology', 'system', 'scenario', 'run'])


def ci95(x):
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 2:
        return np.nan
    from scipy import stats
    return stats.sem(x) * stats.t.ppf(0.975, len(x) - 1)


def fmt(m, c, nd=1):
    if pd.isna(m):
        return '--'
    return f'{m:.{nd}f}' if pd.isna(c) or c == 0 else f'{m:.{nd}f}$\\pm${c:.{nd}f}'


def main_table(df, scenarios, metric, nd, caption, label, fname):
    systems = [s for s in SYS if s in set(df.system)]
    lines = ['\\begin{table*}[t]\\centering\\scriptsize', f'\\caption{{{caption}}}\\label{{{label}}}',
             '\\setlength{\\tabcolsep}{3pt}',
             '\\begin{tabular}{ll' + 'r' * len(systems) + '}', '\\toprule',
             'Topology & Scen. & ' + ' & '.join(SHORT[s] for s in systems) + '\\\\', '\\midrule']
    for t in TOPO:
        sub = df[df.topology == t]
        if sub.empty:
            continue
        first = True
        for sc in scenarios:
            cell = sub[sub.scenario == sc]
            if cell.empty:
                continue
            vals = {s: (cell[cell.system == s][metric].mean(), ci95(cell[cell.system == s][metric]))
                    for s in systems}
            finite = [v[0] for v in vals.values() if not pd.isna(v[0])]
            best = min(finite) if finite else None
            row = []
            for s in systems:
                txt = fmt(*vals[s], nd)
                if best is not None and not pd.isna(vals[s][0]) and np.isclose(vals[s][0], best):
                    txt = '\\textbf{' + txt + '}'
                row.append(txt)
            lines.append((TOPO[t] if first else '') + f' & {sc} & ' + ' & '.join(row) + '\\\\')
            first = False
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines += ['\\end{tabular}', '\\end{table*}']
    (GEN / fname).write_text('\n'.join(lines), encoding='utf-8')


def validation(df):
    """Simulation vs the October emulation, per cell present in both."""
    emu = pd.read_csv(HERE / 'data' / 'emulation_v2_headline.csv', comment='#')
    sim = (df[df.legit_mode == 'udp'].groupby(['topology', 'scenario', 'system'])
           [['time_above_s', 'victim_loss_pct', 'reroutes']].mean().reset_index())
    m = emu.merge(sim, on=['topology', 'scenario', 'system'], how='inner')
    m.to_csv(GEN / 'validation_sim_vs_emulation.csv', index=False)
    if m.empty:
        return m
    from scipy import stats
    rho_t = stats.spearmanr(m.time_above_s_x, m.time_above_s_y).correlation
    rho_l = stats.spearmanr(m.loss_pct, m.victim_loss_pct).correlation
    # does the simulator rank the three systems the same way as the emulation, per cell?
    agree = 0
    cells = 0
    for _, g in m.groupby(['topology', 'scenario']):
        if len(g) == 3:
            cells += 1
            agree += list(g.sort_values('time_above_s_x').system) == list(g.sort_values('time_above_s_y').system)
    mae = (m.time_above_s_x - m.time_above_s_y).abs().mean()
    with open(GEN / 'validation_macros.tex', 'w', encoding='utf-8') as f:
        f.write(f'\\newcommand{{\\valCells}}{{{cells}}}\n\\newcommand{{\\valAgree}}{{{agree}}}\n')
        f.write(f'\\newcommand{{\\valRhoTime}}{{{rho_t:.2f}}}\n\\newcommand{{\\valRhoLoss}}{{{rho_l:.2f}}}\n')
        f.write(f'\\newcommand{{\\valMAE}}{{{mae:.1f}}}\n\\newcommand{{\\valPoints}}{{{len(m)}}}\n')
    setup()
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for s, g in m.groupby('system'):
        ax.scatter(g.time_above_s_x, g.time_above_s_y, s=14, color=SYSTEM_COLORS[s], label=SYSTEM_LABELS[s])
    lim = max(m.time_above_s_x.max(), m.time_above_s_y.max()) * 1.05
    ax.plot([1, lim], [1, lim], color='#999', lw=0.8, ls='--')
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xlabel('Emulation: s above threshold (mean of 5)')
    ax.set_ylabel('Simulation: s above threshold (mean of 30)')
    ax.legend(loc='upper left')
    save(fig, FIG, 'validation')
    return m


def bar_figure(df, scenarios, metric, ylabel, name, systems=None, log=False):
    setup()
    systems = systems or [s for s in SYS if s in set(df.system)]
    topos = [t for t in TOPO if t in set(df.topology)]
    fig, axes = plt.subplots(1, len(topos), figsize=(7.16, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.8 / len(systems)
    for ax, t in zip(axes, topos):
        sub = df[df.topology == t]
        for i, s in enumerate(systems):
            means = [sub[(sub.scenario == sc) & (sub.system == s)][metric].mean() for sc in scenarios]
            errs = [ci95(sub[(sub.scenario == sc) & (sub.system == s)][metric]) for sc in scenarios]
            lower = [min(e, m) if not (pd.isna(e) or pd.isna(m)) else 0 for e, m in zip(errs, means)]
            upper = [0 if pd.isna(e) else e for e in errs]
            ax.bar(np.arange(len(scenarios)) + i * w - 0.4 + w / 2, means, w, yerr=[lower, upper],
                   color=SYSTEM_COLORS[s], label=SYSTEM_LABELS[s], error_kw={'lw': 0.6})
        ax.set_xticks(range(len(scenarios)), scenarios)
        ax.set_title(TOPO_PLAIN[t])
        if log:
            ax.set_yscale('symlog', linthresh=1)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(ncol=len(systems), loc='lower left', bbox_to_anchor=(0, 1.12), fontsize=6)
    save(fig, FIG, name)


def tradeoff(df):
    """Reroutes (disruption) against seconds above threshold, per system, attack scenarios."""
    setup()
    d = df[(df.legit_mode == 'udp') & (df.scenario != 'S1')]
    g = d.groupby('system')[['reroutes', 'time_above_s', 'link_time_above_s']].mean()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for s in systems_in(g.index):
        ax.scatter(g.loc[s, 'reroutes'], g.loc[s, 'time_above_s'], color=SYSTEM_COLORS[s], s=30)
        ax.annotate(SHORT.get(s, s), (g.loc[s, 'reroutes'], g.loc[s, 'time_above_s']),
                    textcoords='offset points', xytext=(4, 2), fontsize=6)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=-0.3)
    ax.set_xlabel('Moves per run (symmetric log; 0 = never moves)')
    ax.set_ylabel('s with a target above $\\theta$')
    save(fig, FIG, 'tradeoff')
    g.to_csv(GEN / 'tradeoff.csv')


def tcp_figure(tcp):
    setup()
    scen = [s for s in ['S3', 'S6', 'S9', 'S11', 'S12', 'S14'] if s in set(tcp.scenario)]
    systems = [s for s in SYS if s in set(tcp.system)]
    bar_figure(tcp, scen, 'fct_mean_s', 'Mean FCT of 5 MB (s)', 'tcp_fct', systems, log=True)
    bar_figure(tcp, scen, 'tcp_goodput_mbps', 'Victim TCP goodput (Mbps)', 'tcp_goodput', systems)


def paired(df, mode):
    metrics = ['time_above_s', 'link_time_above_s', 'victim_loss_pct', 'reroutes',
               'secondary_congestion', 'mlu_mean']
    if mode == 'tcp':
        metrics = ['time_above_s', 'tcp_goodput_mbps', 'fct_mean_s', 'fct_p95_s', 'reroutes']
    t = paired_table(df, 'resilient_sdn', metrics)
    t.to_csv(GEN / f'paired_vs_resilient_sdn_{mode}.csv', index=False)
    return t


def paired_summary_tex(t, systems, metric, fname, caption, label):
    """Wins/ties/losses of each system against ResilientSDN over all cells (paired by seed),
    and how many cells differ significantly after Holm correction."""
    lines = ['\\begin{table}[t]\\centering\\scriptsize', f'\\caption{{{caption}}}\\label{{{label}}}',
             '\\begin{tabular}{lrrrrr}', '\\toprule',
             'System vs RSDN & cells & lower & higher & n.s. & median $\\Delta$\\\\', '\\midrule']
    for s in systems:
        g = t[(t.system == s) & (t.metric == metric)]
        if g.empty:
            continue
        sig = g.p_holm < 0.05
        lower = int((sig & (g.mean_diff < 0)).sum())
        higher = int((sig & (g.mean_diff > 0)).sum())
        lines.append(f'{LAB[s]} & {len(g)} & {lower} & {higher} & {len(g) - lower - higher} & '
                     f'{g.median_diff.median():+.1f}\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (GEN / fname).write_text('\n'.join(lines), encoding='utf-8')


def wan_ext_table():
    p = ROOT / 'results' / 'wan_ext' / 'all_results.csv'
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.run.astype(str).str.isdigit()]
    agg = d.groupby(['topology', 'scenario', 'system']).agg(
        n=('run', 'count'), loss=('packet_loss_pct', 'mean'), above=('time_above_s', 'mean'),
        reroutes=('reroutes', 'mean'), osc=('oscillations', 'mean'),
        sec=('secondary_congestion', 'mean')).reset_index()
    agg.to_csv(GEN / 'wan_ext_summary.csv', index=False)
    lines = ['\\begin{table}[t]\\centering\\scriptsize',
             '\\caption{Mininet emulation of the scenarios added on the WANs in revision 4 (means over $n$ '
             'runs): legitimate loss (\\%), seconds the target was above $\\theta$, reroutes.}',
             '\\label{tab:wan_ext}', '\\setlength{\\tabcolsep}{2.5pt}',
             '\\begin{tabular}{llrccc}', '\\toprule',
             'Topology & Scen. & $n$ & Static & Reactive & RSDN\\\\', '\\midrule']
    for t in TOPO:
        for sc in ['S6', 'S7', 'S8', 'S10']:
            cell = agg[(agg.topology == t) & (agg.scenario == sc)].set_index('system')
            if cell.empty:
                continue
            n = int(cell.n.min())
            row = []
            for s in ['static', 'simple_reactive', 'resilient_sdn']:
                if s in cell.index:
                    c = cell.loc[s]
                    row.append(f'{c.loss:.2f} | {c.above:.0f} | {c.reroutes:.1f}')
                else:
                    row.append('--')
            lines.append(f'{TOPO[t]} & {sc} & {n} & ' + ' & '.join(row) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (GEN / 'tab_wan_ext.tex').write_text('\n'.join(lines), encoding='utf-8')
    return agg


def ablation_table(abl):
    """Which v2 extension does what: each alone against the published configuration."""
    order = ['resilient_sdn', 'rsdn_severity', 'rsdn_fallback', 'rsdn_bottleneck', 'resilient_sdn_v2']
    t = paired_table(abl, 'resilient_sdn', ['time_above_s', 'link_time_above_s', 'victim_loss_pct',
                                              'secondary_congestion', 'reroutes'])
    t.to_csv(GEN / 'ablation_v2_paired.csv', index=False)
    agg = abl.groupby(['scenario', 'system'])[['link_time_above_s', 'victim_loss_pct',
                                                'secondary_congestion']].mean().reset_index()
    lines = [r'\begin{table}[t]\centering\scriptsize',
             r'\caption{ResilientSDN-v2 ablation (simulation, mean over 4 topologies $\times$ 30 seeds): '
             r'link-seconds above $\theta$ / victim loss (\%) / secondary congestion. Each extension alone, '
             r'then all three (v2).}', r'\label{tab:ablation_v2}', r'\setlength{\tabcolsep}{2pt}',
             r'\begin{tabular}{l' + 'c' * len(order) + '}', r'\toprule',
             'Scen. & ' + ' & '.join(SHORT.get(s, s) for s in order) + r'\\', r'\midrule']
    for sc in sorted(agg.scenario.unique(), key=lambda x: int(x[1:])):
        row = []
        for s in order:
            c = agg[(agg.scenario == sc) & (agg.system == s)]
            row.append('--' if c.empty else f'{c.link_time_above_s.iloc[0]:.1f}/{c.victim_loss_pct.iloc[0]:.2f}/'
                       f'{c.secondary_congestion.iloc[0]:.2f}')
        lines.append(f'{sc} & ' + ' & '.join(row) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (GEN / 'tab_ablation_v2.tex').write_text('\n'.join(lines), encoding='utf-8')


def v3_table(v3):
    """Duty-cycle trigger: pulse relief (S7, S10) against false alarms (S1, SB) and all attacks."""
    order = [s for s in ['resilient_sdn', 'rsdn_duty', 'resilient_sdn_v2', 'resilient_sdn_v3']
             if s in set(v3.system)]
    t = paired_table(v3[v3.system.isin(order)], 'resilient_sdn',
                     ['time_above_s', 'reroutes', 'victim_loss_pct', 'secondary_congestion'])
    t.to_csv(GEN / 'v3_paired.csv', index=False)
    attacks = v3[~v3.scenario.isin(['S1', 'SB'])]
    rows = [('False moves, S1', v3[v3.scenario == 'S1'], 'reroutes', 2),
            ('False moves, bursts (SB)', v3[v3.scenario == 'SB'], 'reroutes', 2),
            (r's above $\theta$, S7 pulse', v3[v3.scenario == 'S7'], 'time_above_s', 1),
            (r's above $\theta$, S10', v3[v3.scenario == 'S10'], 'time_above_s', 1),
            (r's above $\theta$, all attacks', attacks, 'time_above_s', 1),
            (r'Victim loss \%, all attacks', attacks, 'victim_loss_pct', 2),
            ('Secondary congestion', attacks, 'secondary_congestion', 2),
            ('Moves per run, all attacks', attacks, 'reroutes', 1)]
    lines = [r'\begin{table}[t]\centering\scriptsize',
             r'\caption{Duty-cycle trigger (simulation, 4 topologies $\times$ 30 seeds): +Duty adds it to '
             r'the published ResilientSDN, v3 adds it to v2. False moves are reroutes when no attack is '
             r'present.}', r'\label{tab:v3}', r'\begin{tabular}{l' + 'r' * len(order) + '}', r'\toprule',
             ' & ' + ' & '.join(SHORT[s] for s in order) + r'\\', r'\midrule']
    for name, d, m, nd in rows:
        vals = [d[d.system == s][m].mean() for s in order]
        lines.append(name + ' & ' + ' & '.join('--' if pd.isna(v) else f'{v:.{nd}f}' for v in vals) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (GEN / 'tab_v3.tex').write_text('\n'.join(lines), encoding='utf-8')
    sb_all = pd.concat([v3[v3.scenario == 'SB'], load('sim_udp', 'sim_v3_sb')], ignore_index=True)
    sb = sb_all.groupby('system')[['reroutes', 'time_above_s']].agg(['mean', 'max'])
    sb.to_csv(GEN / 'false_alarm_bursty.csv')


def floor_sensitivity(tcp):
    """Mean FCT per cell at TCP floor 0 / 0.05 (main campaign) / 0.1, and the largest change."""
    parts = sorted((ROOT / 'results' / 'sim_floor').glob('*/sim_tcp_floor*.csv'))
    if not parts:
        return
    alt = pd.concat([pd.read_csv(p).assign(floor=float(p.stem.split('floor')[1])) for p in parts],
                    ignore_index=True)
    keys = ['topology', 'scenario', 'system']
    base = tcp[tcp.set_index(keys).index.isin(alt.set_index(keys).index)].assign(floor=0.05)
    allf = pd.concat([base, alt], ignore_index=True)
    piv = allf.groupby(keys + ['floor']).fct_mean_s.mean().unstack('floor')
    piv.to_csv(GEN / 'tcp_floor_sensitivity.csv')
    rel = (piv.max(axis=1) - piv.min(axis=1)) / piv[0.05]
    # does the ranking of systems by mean FCT change in any (topology, scenario) cell?
    flips = 0
    cells = 0
    for _, g in piv.groupby(level=['topology', 'scenario']):
        cells += 1
        orders = {f: list(g[f].sort_values().index.get_level_values('system')) for f in piv.columns}
        flips += len({tuple(o) for o in orders.values()}) > 1
    with open(GEN / 'floor_macros.tex', 'w', encoding='utf-8') as f:
        f.write(f'\\newcommand{{\\floorMaxRel}}{{{100 * rel.max():.1f}}}\n')
        f.write(f'\\newcommand{{\\floorMedRel}}{{{100 * rel.median():.1f}}}\n')
        f.write(f'\\newcommand{{\\floorCells}}{{{cells}}}\n\\newcommand{{\\floorFlips}}{{{flips}}}\n')


SWEEP_SYS = ['static', 'mp_ksp', 'simple_reactive', 'te_lp_excess', 'resilient_sdn', 'resilient_sdn_v2',
             'resilient_sdn_v3']


def load_sweep():
    parts = sorted((ROOT / 'results' / 'sweep').glob('sweep_*.csv'))
    if not parts:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(p, on_bad_lines='skip') for p in parts], ignore_index=True)
    d = d.drop_duplicates(['instance', 'system'])
    complete = d.groupby('instance').system.nunique()
    d = d[d.instance.isin(complete[complete == len(SWEEP_SYS)].index)].copy()   # all systems ran
    st = d[d.system == 'static'].set_index('instance').phi
    d['static_phi'] = d.instance.map(st)
    avoidable = (d.static_phi - d.phi_star).clip(lower=0)
    # remaining share of the avoidable excess: 0 = optimal, 1 = no better than static routing
    # Under the rolling attack the demand depends on the routing, so a system's own Phi* can be >= the static
    # Phi (its reroutes provoked a worse attack than even an optimal router could absorb). Such rows are not
    # dropped (that would hide exactly the bad cases): they count as infinitely bad if Phi > Phi_static.
    d['norm_gap'] = np.where(avoidable > 1e-6, (d.phi - d.phi_star) / avoidable.where(avoidable > 1e-6),
                             np.where(d.phi > d.static_phi + 1e-6, np.inf, 1.0))
    d['family'] = np.where(d.topology.isin(['waxman', 'barabasi_albert']), d.topology, 'real')
    return d


def sweep_assets(d):
    n_inst = d.instance.nunique()
    defend = d[d.system == 'static'].defendable.mean()
    # --- summary table
    lines = [r'\begin{table}[t]\centering\scriptsize',
             r'\caption{Randomised sweep over the problem space (' + f'{n_inst}' + r' instances, '
             r'Sec.~\ref{sec:sweep}). Remaining excess: $(\Phi-\Phi^\star)/(\Phi_{\mathrm{static}}-\Phi^\star)$, '
             r'0 = optimal, 1 = no better than static routing (median [IQR]); $>$static: share of instances '
             r'with a remaining excess above 1. Relieved: share of defendable '
             r'instances ($\Phi^\star=0$) in which no target stayed above $\theta$ for more than 10\,s. '
             r'Victim loss and moves (flow reroutes) are means over all instances.}',
             r'\label{tab:sweep}', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'System & remaining excess & $>$static & relieved & victim loss \% & moves\\', r'\midrule']
    dd = d[d.defendable]
    for s in SWEEP_SYS:
        x = d[d.system == s]
        q = x.norm_gap.quantile([0.25, 0.5, 0.75])
        rel = (dd[dd.system == s].time_above_s <= 10).mean()
        worse = (x.norm_gap > 1 + 1e-4).mean()
        lines.append(f'{SHORT.get(s, s)} & {q[0.5]:.2f} [{q[0.25]:.2f}, {q[0.75]:.2f}] & {100 * worse:.0f}\\% & '
                     f'{100 * rel:.0f}\\% & '
                     f'{x.victim_loss_pct.mean():.2f} & {x.reroutes.mean():.1f}' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (GEN / 'tab_sweep.tex').write_text('\n'.join(lines), encoding='utf-8')
    # --- by attack pattern (tests Proposition 3 on the general space)
    shown_p = ['mp_ksp', 'simple_reactive', 'te_lp_excess', 'resilient_sdn', 'resilient_sdn_v2', 'resilient_sdn_v3']
    pats = ['constant', 'timed', 'pulse', 'rolling']
    lp = [r'\begin{table}[t]\centering\scriptsize',
          r'\caption{Sweep by attack pattern: median remaining excess / share of defendable instances relieved '
          r'(definitions as in Table~\ref{tab:sweep}). Instances per pattern: '
          + ', '.join(f'{p} {d[(d.system == "static") & (d.pattern == p)].instance.nunique()}' for p in pats)
          + r'.}', r'\label{tab:sweep_pattern}', r'\setlength{\tabcolsep}{3pt}',
          r'\begin{tabular}{l' + 'r' * len(pats) + '}', r'\toprule',
          'System & ' + ' & '.join(pats) + r'\\', r'\midrule']
    for s in shown_p:
        cells = []
        for p in pats:
            g = d[(d.system == s) & (d.pattern == p)].norm_gap.median()
            r = (dd[(dd.system == s) & (dd.pattern == p)].time_above_s <= 10).mean()
            cells.append(f'{g:.2f} / {100 * r:.0f}\\%')
        lp.append(f'{SHORT.get(s, s)} & ' + ' & '.join(cells) + r'\\')
    lp += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (GEN / 'tab_sweep_pattern.tex').write_text('\n'.join(lp), encoding='utf-8')
    # defendability by topology family
    fam = d[d.system == 'static'].groupby('family').defendable.agg(['mean', 'size'])
    fam.to_csv(GEN / 'sweep_family.csv')
    # --- paired tests over instances (one family: every system vs ResilientSDN, two metrics)
    t = paired_table(d.assign(topology='all', scenario='sweep', run=d.instance), 'resilient_sdn',
                     ['phi', 'victim_loss_pct'])
    t.to_csv(GEN / 'sweep_paired.csv', index=False)
    # --- macros
    by = d[d.system == 'static']
    with open(GEN / 'sweep_macros.tex', 'w', encoding='utf-8') as f:
        f.write(f'\\newcommand{{\\sweepN}}{{{n_inst:,}}}\n'.replace(',', '{,}'))
        f.write(f'\\newcommand{{\\sweepRuns}}{{{len(d):,}}}\n'.replace(',', '{,}'))
        f.write(f'\\newcommand{{\\sweepDefend}}{{{100 * defend:.0f}}}\n')
        f.write(f'\\newcommand{{\\sweepTopos}}{{{d.topology.nunique()}}}\n')
        for s in SWEEP_SYS:
            m = d[d.system == s].norm_gap.median()
            f.write(f'\\newcommand{{\\sweepGap{s.replace("_", "").replace("2", "Two").replace("3", "Three")}}}'
                    f'{{{m:.2f}}}\n')
    # defendability against attack intensity and number of targets (the cut condition, Prop. 4)
    by.groupby([pd.cut(by.alpha, [0.9, 1.1, 1.3, 1.6]), 'n_targets'], observed=True).defendable.mean() \
      .unstack().to_csv(GEN / 'sweep_defendable.csv')
    # --- figure: remaining excess by factor
    setup()
    shown = ['simple_reactive', 'mp_ksp', 'resilient_sdn', 'resilient_sdn_v3', 'te_lp_excess']
    factors = [('alpha', 'attack intensity $\\alpha$', pd.cut(d.alpha, [0.9, 1.1, 1.3, 1.6])),
               ('n_targets', 'attacked links', d.n_targets),
               ('pattern', 'pattern', d.pattern),
               ('family', 'topology', d.family)]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.3), sharey=True)
    for ax, (name, label, key) in zip(axes, factors):
        levels = [l for l in pd.unique(key.dropna()) if l == l]
        levels = sorted(levels, key=str)
        w = 0.8 / len(shown)
        for i, s in enumerate(shown):
            med = [d[(key == lv) & (d.system == s)].norm_gap.median() for lv in levels]
            ax.bar(np.arange(len(levels)) + i * w - 0.4 + w / 2, med, w, color=SYSTEM_COLORS[s],
                   label=SYSTEM_LABELS[s])
        ax.set_xticks(range(len(levels)), [str(l).replace('barabasi_albert', 'BA').replace('(', '').replace(']', '')
                                           for l in levels], rotation=30, fontsize=6)
        ax.set_xlabel(label)
    axes[0].set_ylabel('remaining excess (median)')
    axes[0].legend(ncol=len(shown), loc='lower left', bbox_to_anchor=(0, 1.04), fontsize=6)
    save(fig, FIG, 'sweep_factors')
    # --- figure: CDF of the remaining excess
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for s in shown + ['static']:
        x = np.sort(d[d.system == s].norm_gap.dropna().clip(upper=1.5))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=SYSTEM_COLORS[s], label=SHORT.get(s, s), lw=1.5)
    ax.set_xlabel('remaining excess (0 = optimal, 1 = static)')
    ax.set_ylabel('share of instances')
    ax.legend(fontsize=6)
    save(fig, FIG, 'sweep_cdf')


def counts(udp, tcp, eaw):
    with open(GEN / 'count_macros.tex', 'w', encoding='utf-8') as f:
        f.write(f'\\newcommand{{\\simRunsUDP}}{{{len(udp):,}}}\n'.replace(',', '{,}'))
        f.write(f'\\newcommand{{\\simRunsTCP}}{{{len(tcp):,}}}\n'.replace(',', '{,}'))
        f.write(f'\\newcommand{{\\simRunsEcmp}}{{{len(eaw):,}}}\n'.replace(',', '{,}'))
        f.write(f'\\newcommand{{\\simSeeds}}{{{int(udp.run.max()) + 1 if len(udp) else 0}}}\n')


def main():
    udp, tcp, eaw = load('sim_udp'), load('sim_tcp'), load('sim_udp_ecmpaware')
    counts(udp, tcp, eaw)
    out = {}
    if len(udp):
        validation(udp)
        main_table(udp, ['S3', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14'],
                   'time_above_s', 0,
                   'Simulation, UDP legitimate traffic: seconds with a target link above $\\theta=0.8$ '
                   '(mean $\\pm$ 95\\% CI over 30 seeds; lowest in bold).', 'tab:sim_time', 'tab_sim_time.tex')
        main_table(udp, ['S6', 'S9', 'S11', 'S12', 'S13', 'S14'], 'victim_loss_pct', 2,
                   'Simulation: loss of the legitimate flows that cross the target(s), \\% '
                   '(mean $\\pm$ 95\\% CI, 30 seeds).', 'tab:sim_loss', 'tab_sim_loss.tex')
        main_table(udp, ['S3', 'S6', 'S9', 'S11', 'S12', 'S13', 'S14'], 'reroutes', 1,
                   'Simulation: flow moves per run (mean $\\pm$ 95\\% CI, 30 seeds). For TE-LP a move '
                   'is any change of a flow\'s split by more than 1\\%.', 'tab:sim_reroutes',
                   'tab_sim_reroutes.tex')
        bar_figure(udp, ['S6', 'S7', 'S8', 'S10'], 'time_above_s', 's above $\\theta$', 'wan_adversarial',
                   ['static', 'simple_reactive', 'te_lp', 'te_lp_excess', 'resilient_sdn', 'resilient_sdn_v2'], log=True)
        bar_figure(udp, ['S11', 'S12', 'S13', 'S14'], 'link_time_above_s', 'link-seconds above $\\theta$',
                   'multilink', ['static', 'simple_reactive', 'te_lp', 'te_lp_excess', 'resilient_sdn',
                                 'resilient_sdn_v2'],
                   log=True)
        tradeoff(udp)
        t = paired(udp, 'udp')
        others = [s for s in SYS if s != 'resilient_sdn' and s in set(udp.system)]
        paired_summary_tex(t, others, 'time_above_s', 'tab_paired_time.tex',
                           'Paired comparison with ResilientSDN over all (topology, scenario) cells, '
                           'seconds above $\\theta$: number of cells in which the system is significantly '
                           'lower / higher (Wilcoxon signed-rank on 30 seed-paired runs, Holm-corrected '
                           'over all tests, $\\alpha=0.05$) and median per-cell median difference.',
                           'tab:paired_time')
        paired_summary_tex(t, others, 'victim_loss_pct', 'tab_paired_loss.tex',
                           'As Table~\\ref{tab:paired_time}, for loss of the legitimate flows on the targets (\\%).',
                           'tab:paired_loss')
        out['udp'] = len(udp)
    if len(tcp):
        tcp_figure(tcp)
        main_table(tcp, ['S3', 'S6', 'S9', 'S11', 'S12', 'S13', 'S14'], 'fct_mean_s', 1,
                   'Simulation, TCP legitimate traffic: mean flow-completion time of 5\\,MB transfers that '
                   'start during the attack, s (mean $\\pm$ 95\\% CI, 30 seeds; ideal 2.0\\,s).',
                   'tab:sim_fct', 'tab_sim_fct.tex')
        main_table(tcp, ['S3', 'S6', 'S9', 'S11', 'S12', 'S13', 'S14'], 'fct_p95_s', 1,
                   'Simulation, TCP: 95th-percentile flow-completion time, s.', 'tab:sim_fct95',
                   'tab_sim_fct95.tex')
        paired(tcp, 'tcp')
        floor_sensitivity(tcp)
        out['tcp'] = len(tcp)
    if len(eaw):
        main_table(eaw, ['S3', 'S5', 'S9', 'S11'], 'time_above_s', 0,
                   'Simulation with an ECMP-aware attacker (attack pairs cross the target under both '
                   'static and ECMP routing): seconds above $\\theta$.', 'tab:sim_ecmp', 'tab_sim_ecmp.tex')
        out['ecmp_aware'] = len(eaw)
    sw = load_sweep()
    if len(sw):
        sweep_assets(sw)
        out['sweep_instances'] = sw.instance.nunique()
    v3 = load('sim_udp', 'sim_v3')
    if len(v3):
        v3_table(v3)
        out['v3'] = len(v3)
    abl = load('sim_udp', 'sim_ablation')
    if len(abl):
        ablation_table(abl)
        out['ablation'] = len(abl)
    if wan_ext_table() is not None:
        out['wan_ext'] = True
    print(out)


if __name__ == '__main__':
    main()
