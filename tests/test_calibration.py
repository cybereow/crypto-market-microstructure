import numpy as np

from src.calibration import (top_quantile_precision, calibrate_threshold_for_precision,
                            precision_threshold_table)


def test_top_quantile_precision_rewards_ranking_at_the_top():
    """A model that ranks all its winners highest must score 1.0, while a
    model whose scores are unrelated to outcome must score near the base
    rate. This is the property that makes it a better model-selection
    metric than F1 for a selective strategy.
    """
    y = np.array([0] * 50 + [1] * 50)
    perfect = np.linspace(0, 1, 100)            # winners have the highest scores
    assert top_quantile_precision(y, perfect, quantile=0.8) == 1.0

    rng = np.random.default_rng(0)
    noise = rng.random(100)
    assert 0.2 < top_quantile_precision(y, noise, quantile=0.8) < 0.8


def test_top_quantile_precision_rejects_degenerate_tiny_slices():
    """Returns 0 when the confident slice is below min_support, so the
    hyperparameter search cannot 'win' by scoring 100% on 2 samples."""
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert top_quantile_precision(y, scores, quantile=0.8, min_support=10) == 0.0


def test_calibrate_threshold_reaches_target_precision():
    """With a well-separated score distribution the calibrator must find a
    threshold that actually delivers the requested precision."""
    y = np.array([0] * 500 + [1] * 500)
    scores = np.concatenate([
        np.random.default_rng(1).uniform(0.0, 0.55, 500),
        np.random.default_rng(2).uniform(0.45, 1.0, 500),
    ])

    res = calibrate_threshold_for_precision(y, scores, target_precision=0.80,
                                            min_trades=30)

    assert res['target_met'] is True
    assert res['precision'] >= 0.80
    assert res['n_trades'] >= 30


def test_calibrate_threshold_respects_min_trades_floor():
    """The guard against the classic trap: do NOT report a spectacular win
    rate achieved on a handful of trades. Asking for 100% precision with a
    large min_trades floor must fail honestly rather than return a
    threshold that admits almost nothing.
    """
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    scores = rng.random(400)

    res = calibrate_threshold_for_precision(y, scores, target_precision=0.99,
                                            min_trades=100)

    assert res['target_met'] is False
    assert 'no threshold reached' in res['reason']
    assert res['n_trades'] > 0  # still returns a usable fallback


def test_calibrate_picks_the_most_permissive_qualifying_threshold():
    """Among thresholds that hit the target, the one admitting the MOST
    trades is preferred — clearing a win-rate goal on many trades is more
    trustworthy than clearing it on a few.
    """
    y = np.array([0] * 100 + [1] * 300)
    scores = np.concatenate([np.full(100, 0.1), np.linspace(0.6, 0.99, 300)])

    res = calibrate_threshold_for_precision(y, scores, target_precision=0.90,
                                            min_trades=50)

    assert res['target_met'] is True
    assert res['n_trades'] >= 200  # took the permissive end, not the top sliver


def test_precision_threshold_table_reports_volume_alongside_win_rate():
    """The table must always carry trade counts, so a win rate can never be
    read without the sample size that qualifies it."""
    y = np.array([0, 1] * 100)
    scores = np.linspace(0, 1, 200)

    table = precision_threshold_table(y, scores)

    assert {'threshold', 'n_trades', 'win_rate', 'pct_of_candidates'}.issubset(table.columns)
    assert (table['n_trades'].diff().dropna() <= 0).all()  # monotone non-increasing
