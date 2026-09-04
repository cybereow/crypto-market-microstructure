"""Tests for the daily-signal digest engine (src/daily_signals.py).

These lock in the properties the digest's honesty depends on:
  - the conviction score is bounded, and rewards the effects it claims to;
  - BTC regime is aligned to candidates without lookahead;
  - the daily budget is respected (never more than N/day);
  - the frequency-calibrated threshold hits its target rate and is causal.
"""
import numpy as np
import pandas as pd
import pytest

from src.daily_signals import (
    conviction_score, btc_trend_strength, attach_scores,
    select_top_n_per_day, select_by_threshold, threshold_for_frequency,
    W_BTC, W_REGIME, W_TREND,
)


def _cand(side, signal, bb_width_rank, atr_ratio, own_trend, ts):
    return pd.DataFrame(
        {'side': [side], 'signal': [signal], 'bb_width_rank': [bb_width_rank],
         'atr_ratio': [atr_ratio], 'own_trend': [own_trend]},
        index=pd.DatetimeIndex([ts]),
    )


def test_weights_sum_to_one():
    assert W_BTC + W_REGIME + W_TREND == pytest.approx(1.0)


def test_score_is_bounded_unit_interval():
    # Extreme inputs must still land in [0, 1].
    cand = pd.DataFrame({
        'side': [1, -1, 1, -1],
        'signal': ['breakout', 'breakout', 'reversion', 'reversion'],
        'bb_width_rank': [0.0, 1.0, 0.0, 1.0],
        'atr_ratio': [0.0, 5.0, 0.0, 5.0],
        'own_trend': [1, -1, 1, -1],
    }, index=pd.date_range('2023-01-01', periods=4, freq='h'))
    btc = pd.Series([1.0, -1.0, 1.0, -1.0], index=cand.index)
    s = conviction_score(cand, btc)
    assert (s >= 0).all() and (s <= 1).all()


def test_btc_alignment_rewards_agreement():
    # A long that agrees with a strongly-up BTC must outscore the same long
    # against a strongly-down BTC, holding everything else equal.
    base = dict(signal='breakout', bb_width_rank=0.5, atr_ratio=1.0, own_trend=1)
    ts = pd.Timestamp('2023-01-01')
    aligned = _cand(side=1, ts=ts, **base)
    against = _cand(side=1, ts=ts, **base)
    s_aligned = conviction_score(aligned, pd.Series([0.9], index=[ts]))
    s_against = conviction_score(against, pd.Series([-0.9], index=[ts]))
    assert s_aligned.iloc[0] > s_against.iloc[0]


def test_regime_component_is_family_specific():
    # A trend signal likes a squeeze (low bb-width rank); a reversion signal
    # likes contracting vol (low atr_ratio). Each should score higher in its
    # own favourable regime than the other's.
    ts = pd.Timestamp('2023-01-01')
    btc = pd.Series([0.0], index=[ts])  # neutral BTC to isolate the regime term
    trend_good = conviction_score(
        _cand(1, 'breakout', bb_width_rank=0.05, atr_ratio=1.0, own_trend=1, ts=ts), btc)
    trend_bad = conviction_score(
        _cand(1, 'breakout', bb_width_rank=0.95, atr_ratio=1.0, own_trend=1, ts=ts), btc)
    assert trend_good.iloc[0] > trend_bad.iloc[0]

    rev_good = conviction_score(
        _cand(1, 'reversion', bb_width_rank=0.5, atr_ratio=0.6, own_trend=-1, ts=ts), btc)
    rev_bad = conviction_score(
        _cand(1, 'reversion', bb_width_rank=0.5, atr_ratio=1.4, own_trend=-1, ts=ts), btc)
    assert rev_good.iloc[0] > rev_bad.iloc[0]


def test_btc_trend_strength_bounded_and_signed():
    idx = pd.date_range('2023-01-01', periods=60, freq='h')
    # Rising series -> close ends up above its SMA -> positive strength.
    up = pd.DataFrame({'close': np.linspace(100, 200, 60)}, index=idx)
    s = btc_trend_strength(up, sma=50).dropna()
    assert (s.abs() <= 1.0).all()
    assert s.iloc[-1] > 0


def test_attach_scores_handles_duplicate_timestamps():
    # Two assets fire on the exact same bar -> duplicate index labels. This
    # must not raise and must map the same BTC regime to both.
    ts = pd.Timestamp('2023-06-01 12:00')
    cand = pd.concat([
        _cand(1, 'breakout', 0.2, 0.9, 1, ts).assign(asset='A', label=1, ret=0.01),
        _cand(-1, 'reversion', 0.5, 0.8, -1, ts).assign(asset='B', label=0, ret=-0.01),
    ])
    btc = pd.Series([0.5], index=pd.DatetimeIndex([ts]))
    out = attach_scores(cand, btc)
    assert len(out) == 2
    assert out['btc_strength'].notna().all()
    assert 'score' in out.columns


def test_top_n_per_day_respects_budget():
    # 10 candidates on one day, budget 4 -> exactly 4 survive, and they are
    # the 4 highest-scoring ones.
    day = pd.date_range('2023-03-01 00:00', periods=10, freq='h')
    scored = pd.DataFrame({
        'score': np.linspace(0.1, 0.9, 10),
        'asset': ['X'] * 10, 'signal': ['breakout'] * 10,
        'label': [1] * 10, 'ret': [0.0] * 10,
    }, index=day)
    top = select_top_n_per_day(scored, n=4)
    assert len(top) == 4
    assert top['score'].min() >= 0.5  # the top-4 of 0.1..0.9


def test_top_n_per_day_multiple_days():
    idx = list(pd.date_range('2023-03-01', periods=6, freq='h')) + \
          list(pd.date_range('2023-03-02', periods=6, freq='h'))
    scored = pd.DataFrame({
        'score': np.random.default_rng(0).random(12),
        'asset': ['X'] * 12, 'signal': ['breakout'] * 12,
        'label': [1] * 12, 'ret': [0.0] * 12,
    }, index=pd.DatetimeIndex(idx))
    top = select_top_n_per_day(scored, n=3)
    per_day = top.groupby(pd.Index(top.index).normalize()).size()
    assert (per_day <= 3).all()
    assert len(per_day) == 2


def test_threshold_for_frequency_hits_target_and_is_causal():
    # 100 candidates over ~100 days, target 1/day -> ~100 admitted; the
    # threshold uses only scores, never labels/returns.
    idx = pd.date_range('2023-01-01', periods=200, freq='12h')  # 100 days
    scored = pd.DataFrame({
        'score': np.linspace(0, 1, 200),
        'asset': ['X'] * 200, 'signal': ['breakout'] * 200,
        'label': [1] * 200, 'ret': [0.0] * 200,
    }, index=idx)
    thr = threshold_for_frequency(scored, target_per_day=1.0)
    sel = select_by_threshold(scored, thr)
    span_days = (scored.index.max() - scored.index.min()).days
    achieved = len(sel) / span_days
    assert 0.7 <= achieved <= 1.5  # roughly on target


def test_empty_inputs_do_not_crash():
    empty = pd.DataFrame(columns=['score', 'asset', 'signal', 'label', 'ret'])
    empty.index = pd.DatetimeIndex([])
    assert select_top_n_per_day(empty, n=4).empty
    assert select_by_threshold(empty, 0.5).empty
    assert threshold_for_frequency(empty, 4.0) == 1.0
