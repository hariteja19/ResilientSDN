import numpy as np
import pandas as pd
import pytest

from evaluation.paired_stats import holm, paired_table, rank_biserial


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(holm([np.nan, 0.01])[0])


def test_rank_biserial_extremes():
    assert rank_biserial([-1, -2, -3]) == -1.0
    assert rank_biserial([1, 2, 3]) == 1.0
    assert rank_biserial([0, 0]) == 0.0


def test_paired_table_counts_wins_and_detects_shift():
    rows = []
    for r in range(20):
        rows.append({'topology': 't', 'scenario': 'S', 'run': r, 'system': 'ref', 'm': 10.0 + r})
        rows.append({'topology': 't', 'scenario': 'S', 'run': r, 'system': 'x', 'm': 5.0 + r})
    t = paired_table(pd.DataFrame(rows), 'ref', ['m'])
    row = t.iloc[0]
    assert row.wins == 20 and row.losses == 0 and row.mean_diff == -5.0
    assert row.p_wilcoxon < 1e-4 and row.ci_lo == row.ci_hi == -5.0
