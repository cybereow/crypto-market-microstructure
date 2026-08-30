import numpy as np
import pandas as pd

from src.execution import simulate_maker_fills, triple_barrier_from_fill


def test_maker_fill_fills_when_price_pulls_back():
    """Long signal at close=100, ATR=2, offset_mult=0.5 -> limit at 99. The
    very next bar's low dips to 99 -> should fill at bar 1, price 99.
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 99.5, 100.0, 100.0, 100.0],
        'high':  [100.0, 100.0, 100.0, 100.0, 100.0],
        'low':   [100.0, 99.0, 99.5, 100.0, 100.0],
    }, index=dates)
    entries = pd.Series([1, 0, 0, 0, 0], index=dates)
    atr = pd.Series(2.0, index=dates)

    fills = simulate_maker_fills(df, entries, atr, offset_mult=0.5, queue_timeout=3)

    assert len(fills) == 1
    row = fills.iloc[0]
    assert row['filled']
    assert row['limit_price'] == 99.0
    assert row['fill_price'] == 99.0
    assert row['wait_bars'] == 1


def test_maker_fill_never_touched_is_unfilled():
    """Price keeps running away from the limit -> never fills within the
    queue timeout, and no trade is produced.
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 105.0, 110.0, 115.0, 120.0],
        'high':  [100.0, 106.0, 111.0, 116.0, 121.0],
        'low':   [100.0, 104.0, 109.0, 114.0, 119.0],
    }, index=dates)
    entries = pd.Series([1, 0, 0, 0, 0], index=dates)
    atr = pd.Series(2.0, index=dates)  # limit at 100 - 0.5*2 = 99, never touched

    fills = simulate_maker_fills(df, entries, atr, offset_mult=0.5, queue_timeout=3)

    assert len(fills) == 1
    assert not fills.iloc[0]['filled']
    assert np.isnan(fills.iloc[0]['fill_price'])
    assert np.isnan(fills.iloc[0]['wait_bars'])


def test_maker_fill_short_direction():
    """Short signal prices its passive ask ABOVE the signal close."""
    dates = pd.date_range("2023-01-01", periods=4, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 100.0, 100.0, 100.0],
        'high':  [100.0, 101.0, 100.0, 100.0],
        'low':   [100.0, 100.0, 100.0, 100.0],
    }, index=dates)
    entries = pd.Series([-1, 0, 0, 0], index=dates)
    atr = pd.Series(2.0, index=dates)  # limit at 100 + 0.5*2 = 101

    fills = simulate_maker_fills(df, entries, atr, offset_mult=0.5, queue_timeout=2)

    assert fills.iloc[0]['limit_price'] == 101.0
    assert fills.iloc[0]['filled']
    assert fills.iloc[0]['fill_price'] == 101.0


def test_triple_barrier_from_fill_only_labels_filled_orders():
    """Two candidates: one fills and rallies to its profit target, the other
    never fills. Only the filled one should produce a labeled trade.
    """
    dates = pd.date_range("2023-01-01", periods=8, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 99.0, 100.0, 104.0, 100.0, 200.0, 205.0, 210.0],
        'high':  [100.0, 99.0, 100.0, 104.0, 100.0, 200.0, 205.0, 210.0],
        'low':   [100.0, 98.0, 100.0, 100.0, 100.0, 200.0, 205.0, 210.0],
    }, index=dates)
    # bar0: long candidate that will fill on bar1 (low=98 <= limit=99).
    # bar4: long candidate that never fills (price only goes up from 200).
    entries = pd.Series([1, 0, 0, 0, 1, 0, 0, 0], index=dates)
    atr = pd.Series(2.0, index=dates)  # offset 0.5*2=1 -> limits at 99 and 199

    fills = simulate_maker_fills(df, entries, atr, offset_mult=0.5, queue_timeout=2)
    assert fills['filled'].tolist() == [True, False]

    labels = triple_barrier_from_fill(df, fills, atr, pt_mult=1.5, sl_mult=1.0, max_holding=3)

    assert len(labels) == 1
    row = labels.iloc[0]
    assert row['side'] == 1
    # Entry at fill price 99 (bar1), pt at 99+1.5*2=102, hit by bar3's close=104.
    assert row['label'] == 1
    assert row['wait_bars'] == 1


def test_triple_barrier_from_fill_empty_when_nothing_fills():
    dates = pd.date_range("2023-01-01", periods=4, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 200.0, 300.0, 400.0],
        'high':  [100.0, 200.0, 300.0, 400.0],
        'low':   [100.0, 200.0, 300.0, 400.0],
    }, index=dates)
    entries = pd.Series([1, 0, 0, 0], index=dates)
    atr = pd.Series(2.0, index=dates)

    fills = simulate_maker_fills(df, entries, atr, offset_mult=0.5, queue_timeout=2)
    labels = triple_barrier_from_fill(df, fills, atr, pt_mult=1.5, sl_mult=1.0, max_holding=3)

    assert labels.empty
