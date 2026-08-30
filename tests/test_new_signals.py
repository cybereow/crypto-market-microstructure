"""Tests for the regime-conditional primary signals in src/labeling.py.

The bug these guard against is real and was hit during development:
`bb_position` is centred on zero and spans roughly [-1, +1], but was
initially compared against 0.1/0.9 as if it were a 0..1 percentile. That
fired on 38% of all bars with a 5:1 long/short skew — a "range fade"
signal that was really "price is below its mean". Cheap unit tests on
synthetic series catch this class of mistake immediately; a backtest does
not, because a broken signal still produces plausible-looking numbers.
"""
import numpy as np
import pandas as pd
import pytest

from src.labeling import (range_fade_entries, trend_pullback_entries,
                          volatility_breakout_entries)


def _base(n=300):
    idx = pd.date_range('2024-01-01', periods=n, freq='4h')
    close = pd.Series(100.0, index=idx)
    return pd.DataFrame({
        'open': close, 'high': close, 'low': close, 'close': close,
        'volume': 1000.0,
    }, index=idx)


# --------------------------------------------------------------------------
# range_fade
# --------------------------------------------------------------------------

def test_range_fade_does_not_fire_on_most_bars():
    """Regression: the percentile/centred-units mixup fired on ~38% of bars."""
    rng = np.random.default_rng(0)
    df = _base(1000)
    df['bb_position'] = rng.normal(0, 0.5, size=1000)  # realistic spread
    df['ATR_ratio'] = 1.0

    e = range_fade_entries(df)
    rate = (e != 0).mean()
    assert rate < 0.10, f"range_fade fired on {rate:.1%} of bars -- far too many"


def test_range_fade_is_roughly_side_balanced_on_symmetric_input():
    """Regression: the same bug produced a 5:1 long skew."""
    rng = np.random.default_rng(1)
    df = _base(4000)
    df['bb_position'] = rng.normal(0, 0.6, size=4000)
    df['ATR_ratio'] = 1.0

    e = range_fade_entries(df)
    longs, shorts = int((e == 1).sum()), int((e == -1).sum())
    assert longs > 0 and shorts > 0
    ratio = max(longs, shorts) / min(longs, shorts)
    assert ratio < 2.0, f"symmetric input gave {longs}L/{shorts}S (ratio {ratio:.1f})"


def test_range_fade_requires_a_cross_back_inside_the_band():
    df = _base(10)
    # Sits pinned outside the lower band the whole time: extended, never crossing.
    df['bb_position'] = [-1.5] * 10
    df['ATR_ratio'] = 1.0
    assert (range_fade_entries(df) != 0).sum() == 0

    # Now cross back inside at bar 5 -> exactly one long.
    df['bb_position'] = [-1.5] * 5 + [-0.5] * 5
    e = range_fade_entries(df)
    assert int((e == 1).sum()) == 1
    assert e.iloc[5] == 1


def test_range_fade_blocked_when_volatility_expanding():
    """The ATR guard is the whole point: fading an expansion is the losing trade."""
    df = _base(10)
    df['bb_position'] = [-1.5] * 5 + [-0.5] * 5
    df['ATR_ratio'] = 1.5  # expanding
    assert (range_fade_entries(df) != 0).sum() == 0


# --------------------------------------------------------------------------
# volatility_breakout
# --------------------------------------------------------------------------

def test_vol_breakout_requires_a_squeeze():
    n = 200
    df = _base(n)
    # A clean upward step at the end so the Donchian condition triggers.
    close = np.concatenate([np.full(n - 1, 100.0), [130.0]])
    df['close'] = close
    df['high'] = close
    df['low'] = close

    # Wide bands everywhere -> no squeeze -> no entry.
    df['bb_width'] = np.linspace(0.5, 1.0, n)
    assert (volatility_breakout_entries(df) != 0).sum() == 0

    # Narrow width on the bar BEFORE the breakout -> squeeze -> entry allowed.
    # The squeeze rank is deliberately .shift(1)'d inside the builder so the
    # decision uses only information available before the breakout bar; the
    # squeeze must therefore be present on bar n-2, not on the breakout bar.
    df['bb_width'] = np.concatenate([np.linspace(0.5, 1.0, n - 2), [0.01, 0.01]])
    assert (volatility_breakout_entries(df) == 1).sum() == 1


def test_vol_breakout_is_a_subset_of_plain_breakout():
    """It only ever filters; it must never invent entries Donchian would not give."""
    from src.labeling import donchian_breakout_entries
    rng = np.random.default_rng(3)
    n = 800
    df = _base(n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=df.index)
    df['close'], df['high'], df['low'] = close, close * 1.002, close * 0.998
    df['bb_width'] = pd.Series(rng.uniform(0.05, 0.5, n), index=df.index)

    plain = donchian_breakout_entries(df, 20)
    squeezed = volatility_breakout_entries(df, 20)
    fired = squeezed != 0
    assert (squeezed[fired] == plain[fired]).all()
    assert int(fired.sum()) <= int((plain != 0).sum())


# --------------------------------------------------------------------------
# trend_pullback
# --------------------------------------------------------------------------

def test_trend_pullback_longs_only_in_uptrend():
    df = _base(10)
    df['close_to_sma20'] = 0.05
    df['close_to_sma50'] = 0.05          # uptrend
    df['RSI_14'] = [40] * 5 + [50] * 5   # crosses up through 45 at bar 5

    e = trend_pullback_entries(df)
    assert e.iloc[5] == 1
    assert int((e == -1).sum()) == 0


def test_trend_pullback_shorts_only_in_downtrend():
    df = _base(10)
    df['close_to_sma20'] = -0.05
    df['close_to_sma50'] = -0.05         # downtrend
    df['RSI_14'] = [60] * 5 + [50] * 5   # crosses down through 55 at bar 5

    e = trend_pullback_entries(df)
    assert e.iloc[5] == -1
    assert int((e == 1).sum()) == 0


def test_trend_pullback_silent_when_trend_is_mixed():
    """Conflicting SMAs mean no established trend -- the signal must abstain."""
    df = _base(10)
    df['close_to_sma20'] = 0.05
    df['close_to_sma50'] = -0.05         # conflict
    df['RSI_14'] = [40] * 5 + [50] * 5
    assert (trend_pullback_entries(df) != 0).sum() == 0


def test_signals_return_series_aligned_to_input_index():
    """Every builder must return an index-aligned Series of the right length."""
    rng = np.random.default_rng(5)
    n = 300
    df = _base(n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=df.index)
    df['close'], df['high'], df['low'] = close, close * 1.002, close * 0.998
    df['bb_width'] = rng.uniform(0.05, 0.5, n)
    df['bb_position'] = rng.normal(0, 0.6, n)
    df['ATR_ratio'] = rng.uniform(0.8, 1.2, n)
    df['close_to_sma20'] = rng.normal(0, 0.05, n)
    df['close_to_sma50'] = rng.normal(0, 0.05, n)
    df['RSI_14'] = rng.uniform(20, 80, n)

    for fn in (range_fade_entries, volatility_breakout_entries, trend_pullback_entries):
        e = fn(df)
        assert isinstance(e, pd.Series)
        assert len(e) == n
        assert e.index.equals(df.index)
        assert set(e.unique()) <= {-1, 0, 1}
