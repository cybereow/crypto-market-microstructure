import pandas as pd

from src.labeling import (donchian_breakout_entries, rsi_reversion_entries,
                          obi_momentum_entries, triple_barrier_labels)


def test_triple_barrier_profit_take_hit():
    """A long entry whose price rallies straight through the profit-take
    level before ever touching the stop should be labeled a win, exiting at
    the profit-take price.
    """
    dates = pd.date_range("2023-01-01", periods=6, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 100.0, 106.0, 100.0, 100.0, 100.0],
        'high':  [100.0, 100.0, 106.0, 100.0, 100.0, 100.0],
        'low':   [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    }, index=dates)
    entries = pd.Series([0, 1, 0, 0, 0, 0], index=dates)
    atr = pd.Series(2.0, index=dates)  # pt at 100+1.5*2=103, sl at 100-1*2=98

    labels = triple_barrier_labels(df, entries, atr, pt_mult=1.5, sl_mult=1.0, max_holding=3)

    assert len(labels) == 1
    row = labels.iloc[0]
    assert row['side'] == 1
    assert row['label'] == 1
    assert row['hold'] == 1


def test_triple_barrier_stop_loss_hit():
    """Symmetric case: price drops through the stop first -> loss."""
    dates = pd.date_range("2023-01-01", periods=6, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 100.0, 94.0, 100.0, 100.0, 100.0],
        'high':  [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        'low':   [100.0, 100.0, 94.0, 100.0, 100.0, 100.0],
    }, index=dates)
    entries = pd.Series([0, 1, 0, 0, 0, 0], index=dates)
    atr = pd.Series(2.0, index=dates)

    labels = triple_barrier_labels(df, entries, atr, pt_mult=1.5, sl_mult=1.0, max_holding=3)

    assert len(labels) == 1
    assert labels.iloc[0]['label'] == 0


def test_triple_barrier_vertical_barrier():
    """Neither barrier touched within max_holding -> label by the sign of
    the realized return at the vertical (time) barrier.
    """
    dates = pd.date_range("2023-01-01", periods=6, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 100.0, 100.5, 100.5, 100.5, 100.5],
        'high':  [100.0, 100.0, 100.5, 100.5, 100.5, 100.5],
        'low':   [100.0, 100.0, 100.5, 100.5, 100.5, 100.5],
    }, index=dates)
    entries = pd.Series([0, 1, 0, 0, 0, 0], index=dates)
    atr = pd.Series(2.0, index=dates)  # wide barriers, never touched

    labels = triple_barrier_labels(df, entries, atr, pt_mult=1.5, sl_mult=1.0, max_holding=2)

    assert len(labels) == 1
    row = labels.iloc[0]
    assert row['hold'] == 2
    assert row['label'] == 1  # price ended slightly above entry


def test_donchian_breakout_entries_direction():
    dates = pd.date_range("2023-01-01", periods=6, freq="4h")
    df = pd.DataFrame({
        'high': [100, 101, 102, 103, 110, 100],
        'low':  [99, 100, 101, 102, 103, 90],
        'close': [100, 101, 102, 103, 109, 91],
    }, index=dates)
    entries = donchian_breakout_entries(df, lookback=3)
    # Bar 4 (110 close) breaks above the prior 3-bar high (103) -> long.
    assert entries.iloc[4] == 1
    # Bar 5 (91 close) breaks below the prior 3-bar low (101) -> short.
    assert entries.iloc[5] == -1


def test_rsi_reversion_entries_direction():
    dates = pd.date_range("2023-01-01", periods=4, freq="4h")
    df = pd.DataFrame({'RSI_14': [25, 35, 75, 65]}, index=dates)
    entries = rsi_reversion_entries(df)
    assert entries.iloc[1] == 1   # crossed back up through 30 -> long
    assert entries.iloc[3] == -1  # crossed back down through 70 -> short


def test_obi_momentum_entries_direction():
    """OBI sits flat at its baseline, then spikes to a new extreme high
    (heavy bid imbalance) and later a new extreme low (heavy ask
    imbalance). The crossing bars should fire long/short respectively.
    """
    dates = pd.date_range("2023-01-01", periods=12, freq="5min")
    df = pd.DataFrame({
        'obi': [0.50, 0.50, 0.50, 0.50, 0.50,   # flat baseline
                0.90,                            # spike up -> cross upper quantile -> long
                0.50, 0.50, 0.50, 0.50, 0.50,    # back to flat baseline
                0.10],                           # spike down -> cross lower quantile -> short
    }, index=dates)
    entries = obi_momentum_entries(df, lookback=5, quantile=0.8)
    assert entries.iloc[5] == 1
    assert entries.iloc[11] == -1
    # No entries anywhere else.
    assert (entries.drop(entries.index[[5, 11]]) == 0).all()


def test_obi_momentum_entries_no_repeat_while_extreme_persists():
    """Firing is on the CROSSING bar only, not every bar OBI stays extreme
    -- otherwise a persistent imbalance would generate overlapping
    candidate trades for every bar of its duration.
    """
    dates = pd.date_range("2023-01-01", periods=8, freq="5min")
    df = pd.DataFrame({'obi': [0.50, 0.50, 0.50, 0.50, 0.50, 0.90, 0.91, 0.92]}, index=dates)
    entries = obi_momentum_entries(df, lookback=5, quantile=0.8)
    assert entries.iloc[5] == 1
    assert entries.iloc[6] == 0
    assert entries.iloc[7] == 0
