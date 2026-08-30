"""Tests for src/significance.py.

These tests matter more than most: this module is what decides whether a
backtest result gets believed. If it says "significant" too easily it is
worse than useless, so the cases below deliberately include selections
that must be rejected.
"""
import numpy as np
import pytest

from src.significance import (binomial_ci, binomial_pvalue, breakeven_win_rate,
                              deflated_pvalue, describe_result,
                              permutation_test)


def test_breakeven_matches_barrier_geometry():
    # 1:1 payoff needs 50%; 2:1 needs 33.3%; 1:2 needs 66.7%.
    assert breakeven_win_rate(1.0) == pytest.approx(0.5)
    assert breakeven_win_rate(2.0) == pytest.approx(1 / 3)
    assert breakeven_win_rate(0.5) == pytest.approx(2 / 3)


def test_breakeven_rises_with_costs():
    assert breakeven_win_rate(1.0, cost_per_trade=0.004) > 0.5


def test_ci_contains_point_estimate_and_widens_when_small():
    lo_big, hi_big = binomial_ci(600, 1000)
    lo_small, hi_small = binomial_ci(6, 10)
    assert lo_big < 0.6 < hi_big
    assert lo_small < 0.6 < hi_small
    # A 10-trade sample must be far less certain than a 1000-trade one.
    assert (hi_small - lo_small) > 4 * (hi_big - lo_big)


def test_ci_handles_degenerate_all_wins():
    lo, hi = binomial_ci(10, 10)
    assert hi == 1.0
    assert lo < 1.0  # never claims certainty


def test_permutation_detects_a_genuinely_skilful_selection():
    """A selection that really concentrates winners must come out significant."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=1000)
    # Pick 100 trades that are mostly winners -- real skill.
    win_idx = np.flatnonzero(labels == 1)[:90]
    lose_idx = np.flatnonzero(labels == 0)[:10]
    mask = np.zeros(1000, dtype=bool)
    mask[win_idx] = True
    mask[lose_idx] = True

    res = permutation_test(labels, mask, n_iter=2000)
    assert res['statistic'] > 0.85
    assert res['p_value'] < 0.01
    assert res['null_mean'] == pytest.approx(labels.mean(), abs=0.05)


def test_permutation_rejects_a_random_selection():
    """This is the important one: no signal must yield no significance."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=1000)
    mask = np.zeros(1000, dtype=bool)
    mask[rng.choice(1000, size=85, replace=False)] = True

    res = permutation_test(labels, mask, n_iter=2000, random_state=7)
    assert res['p_value'] > 0.05, (
        f"random selection was called significant (p={res['p_value']}) -- "
        "the null model is broken")


def test_permutation_null_mean_tracks_base_rate_not_selection_size():
    """The null must be centred on the pool base rate for any selection size."""
    labels = np.array([1] * 300 + [0] * 700)
    for n_sel in (20, 100, 400):
        mask = np.zeros(1000, dtype=bool)
        mask[:n_sel] = True
        res = permutation_test(labels, mask, n_iter=1500)
        assert res['null_mean'] == pytest.approx(0.3, abs=0.03)


def test_permutation_handles_mean_return_statistic():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=500)
    returns = np.where(labels == 1, 2.0, -1.0)
    mask = np.zeros(500, dtype=bool)
    mask[np.flatnonzero(labels == 1)[:50]] = True

    res = permutation_test(labels, mask, n_iter=1500,
                           statistic='mean_return', returns=returns)
    assert res['statistic'] == pytest.approx(2.0)
    assert res['p_value'] < 0.01


def test_permutation_rejects_degenerate_selections():
    labels = np.array([1, 0, 1, 0])
    assert permutation_test(labels, np.zeros(4, dtype=bool))['reason']
    assert permutation_test(labels, np.ones(4, dtype=bool))['reason']


def test_permutation_validates_input_length():
    with pytest.raises(ValueError):
        permutation_test([1, 0, 1], [True, False])


def test_mean_return_requires_returns():
    with pytest.raises(ValueError):
        permutation_test([1, 0], [True, False], statistic='mean_return')


def test_deflated_pvalue_penalises_multiple_attempts():
    p = 0.02
    assert deflated_pvalue(p, 1) == pytest.approx(p)
    assert deflated_pvalue(p, 20) > 0.05  # 20 tries makes p=0.02 unremarkable
    assert deflated_pvalue(p, 20) < 1.0


def test_describe_result_flags_thin_sample_as_not_significant():
    """62.4% on 85 trades at 1:1 payoff, after trying ~6 configs."""
    res = describe_result(n_wins=53, n_trades=85, payoff_ratio=1.0,
                          cost_per_trade=0.004, n_configurations_tried=6)
    assert res['breakeven_wr'] > 0.5
    assert res['ci_low'] < res['breakeven_wr'] + 0.05  # CI hugs breakeven
    assert res['verdict'] != 'significant'


def test_describe_result_accepts_a_large_clean_edge():
    res = describe_result(n_wins=1200, n_trades=2000, payoff_ratio=1.0,
                          cost_per_trade=0.004, n_configurations_tried=6)
    assert res['verdict'] == 'significant'
