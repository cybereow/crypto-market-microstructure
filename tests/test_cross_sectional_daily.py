"""Tests for the daily cross-sectional long/short engine.

These lock in the properties the backtest's honesty depends on:
  - resampling and panel alignment don't invent prices;
  - the feature is point-in-time (no forward leak);
  - the book is dollar-neutral and exactly 2M positions/day;
  - the cost model tracks turnover (a name that keeps its side pays nothing);
  - a known long/short outcome nets out to the hand-computed number.
"""
import numpy as np
import pandas as pd
import pytest

from src.cross_sectional_daily import (
    resample_daily, build_close_panel, cross_sectional_feature,
    long_short_book, backtest_long_short, equity_stats,
    daily_funding_panel, apply_funding,
)


def _hourly(prices, start='2023-01-01'):
    idx = pd.date_range(start, periods=len(prices), freq='h')
    return pd.DataFrame({'open': prices, 'high': prices, 'low': prices,
                         'close': prices, 'volume': 1.0}, index=idx)


def test_resample_daily_aggregates_ohlc():
    # 48 hourly bars -> 2 daily bars; open=first, high=max, low=min, close=last.
    prices = list(range(1, 49))
    daily = resample_daily(_hourly(prices))
    assert len(daily) == 2
    assert daily.iloc[0]['open'] == 1
    assert daily.iloc[0]['close'] == 24
    assert daily.iloc[0]['high'] == 24
    assert daily.iloc[1]['open'] == 25
    assert daily.iloc[1]['close'] == 48


def test_build_panel_does_not_forward_fill():
    a = pd.DataFrame({'close': [1.0, 2.0, 3.0]},
                     index=pd.date_range('2023-01-01', periods=3))
    b = pd.DataFrame({'close': [10.0, 20.0]},
                     index=pd.date_range('2023-01-02', periods=2))
    panel = build_close_panel({'A': a, 'B': b})
    # B has no price on day 0 -> stays NaN, never back/forward-filled.
    assert np.isnan(panel.iloc[0]['B'])
    assert panel.iloc[1]['B'] == 10.0


def test_feature_is_point_in_time():
    # momentum feature on day t must use only returns up to t-1 (shifted),
    # so it can never correlate with the same-day forward return by leak.
    closes = pd.DataFrame({
        'A': np.linspace(1, 2, 40), 'B': np.linspace(2, 1, 40),
    }, index=pd.date_range('2023-01-01', periods=40))
    feat = cross_sectional_feature(closes, 'momentum', 5)
    # The first 5 (lookback) + 1 (shift) rows are undefined.
    assert feat.iloc[:6].isna().all().all()
    assert feat.iloc[10].notna().all()


def test_reversal_is_negated_momentum():
    closes = pd.DataFrame({
        'A': np.linspace(1, 3, 40), 'B': np.linspace(3, 1, 40),
    }, index=pd.date_range('2023-01-01', periods=40))
    mom = cross_sectional_feature(closes, 'momentum', 7)
    rev = cross_sectional_feature(closes, 'reversal', 7)
    pd.testing.assert_frame_equal(rev, -mom)


def test_book_is_dollar_neutral_and_right_size():
    feat = pd.DataFrame({
        'A': [0.5], 'B': [0.3], 'C': [0.1], 'D': [-0.2], 'E': [-0.4], 'F': [-0.6],
    }, index=pd.DatetimeIndex(['2023-06-01']))
    w = long_short_book(feat, m_per_side=2)
    row = w.iloc[0]
    assert (row > 0).sum() == 2 and (row < 0).sum() == 2  # exactly 2M positions
    assert row.sum() == pytest.approx(0.0)                # dollar-neutral
    assert row.abs().sum() == pytest.approx(1.0)          # gross exposure 1
    # The two highest features go long, the two lowest go short.
    assert row['A'] > 0 and row['B'] > 0
    assert row['E'] < 0 and row['F'] < 0


def test_book_skips_days_with_too_few_assets():
    feat = pd.DataFrame({'A': [0.5], 'B': [np.nan], 'C': [np.nan]},
                        index=pd.DatetimeIndex(['2023-06-01']))
    w = long_short_book(feat, m_per_side=2)  # needs 4, only 1 valid
    assert (w.iloc[0] == 0).all()


def test_cost_tracks_turnover_zero_when_book_unchanged():
    # Two days, identical weights -> no turnover on day 2 -> no cost on day 2.
    idx = pd.date_range('2023-01-01', periods=3)
    closes = pd.DataFrame({'A': [10, 11, 12], 'B': [10, 9, 8],
                           'C': [10, 10, 10], 'D': [10, 10, 10]}, index=idx)
    weights = pd.DataFrame({'A': [0.25, 0.25], 'B': [-0.25, -0.25],
                            'C': [0.25, 0.25], 'D': [-0.25, -0.25]},
                           index=idx[:2])
    bt = backtest_long_short(closes, weights, cost_per_side=0.001)
    # Day 2's turnover (weights unchanged from day 1) must be zero.
    assert bt['turnover'].iloc[1] == pytest.approx(0.0)
    assert bt['cost'].iloc[1] == pytest.approx(0.0)


def test_backtest_matches_hand_computed_return():
    # One held day: A +10%, B -20%; long A short B, weight 0.5 each.
    # gross = 0.5*0.10 - 0.5*(-0.20) = 0.05 + 0.10 = 0.15
    # first-day turnover = |0.5| + |-0.5| = 1.0; cost = 1.0*0.001 = 0.001
    idx = pd.date_range('2023-01-01', periods=2)
    closes = pd.DataFrame({'A': [100.0, 110.0], 'B': [100.0, 80.0]}, index=idx)
    weights = pd.DataFrame({'A': [0.5], 'B': [-0.5]}, index=idx[:1])
    bt = backtest_long_short(closes, weights, cost_per_side=0.001)
    assert bt['gross'].iloc[0] == pytest.approx(0.15)
    assert bt['cost'].iloc[0] == pytest.approx(0.001)
    assert bt['net'].iloc[0] == pytest.approx(0.149)


def test_funding_sign_long_pays_short_receives():
    # One day, weights long A (+0.5) / short B (-0.5), both with +1% daily
    # funding. Long pays -> -0.5*0.01; short receives -> -(-0.5)*0.01 = +0.005.
    # Net funding P&L = -(0.5*0.01 + (-0.5)*0.01) = 0. Then make them differ.
    idx = pd.DatetimeIndex(['2023-06-01'])
    weights = pd.DataFrame({'A': [0.5], 'B': [-0.5]}, index=idx)
    bt = pd.DataFrame({'gross': [0.0], 'cost': [0.0], 'net': [0.0],
                       'turnover': [1.0]}, index=idx)
    # Equal funding -> nets to zero on a dollar-neutral book.
    feq = pd.DataFrame({'A': [0.01], 'B': [0.01]}, index=idx)
    out = apply_funding(bt, weights, feq)
    assert out['funding'].iloc[0] == pytest.approx(0.0)
    # Long side funded higher than short side -> a net drag on the book.
    fdiff = pd.DataFrame({'A': [0.02], 'B': [0.00]}, index=idx)
    out2 = apply_funding(bt, weights, fdiff)
    assert out2['funding'].iloc[0] == pytest.approx(-(0.5 * 0.02 + (-0.5) * 0.0))
    assert out2['funding'].iloc[0] < 0  # long pays high funding -> negative


def test_daily_funding_panel_sums_three_charges():
    # Hourly-ffilled funding constant at 0.0001 -> daily total = 3 charges =
    # mean*3 = 0.0003.
    idx = pd.date_range('2023-01-01', periods=48, freq='h')
    fund = {'A': pd.DataFrame({'funding_rate': [0.0001] * 48}, index=idx)}
    days = pd.date_range('2023-01-01', periods=2)
    panel = daily_funding_panel(fund, days)
    assert panel.loc[days[0], 'A'] == pytest.approx(0.0003)


def test_equity_stats_on_known_series():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    st = equity_stats(r, periods_per_year=365)
    assert st['n_days'] == 5
    assert st['total_return'] == pytest.approx((1.01 * 0.995 * 1.02 * 1.0 * 0.99) - 1)
    assert 0 <= st['hit_rate'] <= 1
    assert st['max_drawdown'] <= 0
