import numpy as np
import pandas as pd

from src.labeling import (donchian_breakout_entries, rsi_reversion_entries,
                          obi_momentum_entries, funding_extreme_reversion_entries,
                          funding_reversion_confirmed_entries,
                          funding_reversion_regime_filtered_entries,
                          obv_divergence_entries, btc_lead_lag_entries,
                          triple_barrier_labels)


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


def test_funding_extreme_reversion_entries_fades_not_follows():
    """Mirror image of OBI momentum's mechanics (cross a rolling quantile)
    but the OPPOSITE direction: extreme positive funding (crowded longs)
    fades with a SHORT, extreme negative funding (crowded shorts) fades
    with a LONG. Getting the sign backwards here would silently turn this
    into a momentum signal instead of the mean-reversion one it's meant
    to be -- exactly the kind of mistake test_new_signals.py's range_fade
    tests already caught once for a different signal.
    """
    dates = pd.date_range("2023-01-01", periods=12, freq="4h")
    df = pd.DataFrame({
        'funding_rate': [0.0001, 0.0001, 0.0001, 0.0001, 0.0001,   # flat baseline
                         0.01,                                     # spike up -> fade -> SHORT
                         0.0001, 0.0001, 0.0001, 0.0001, 0.0001,   # back to flat baseline
                         -0.01],                                   # spike down -> fade -> LONG
    }, index=dates)
    entries = funding_extreme_reversion_entries(df, lookback=5, quantile=0.8)
    assert entries.iloc[5] == -1
    assert entries.iloc[11] == 1
    assert (entries.drop(entries.index[[5, 11]]) == 0).all()


def test_funding_extreme_reversion_entries_no_repeat_while_extreme_persists():
    dates = pd.date_range("2023-01-01", periods=8, freq="4h")
    df = pd.DataFrame({'funding_rate': [0.0001] * 5 + [0.01, 0.011, 0.012]}, index=dates)
    entries = funding_extreme_reversion_entries(df, lookback=5, quantile=0.8)
    assert entries.iloc[5] == -1
    assert entries.iloc[6] == 0
    assert entries.iloc[7] == 0


def test_funding_extreme_reversion_entries_uses_custom_column_name():
    dates = pd.date_range("2023-01-01", periods=6, freq="4h")
    df = pd.DataFrame({'my_funding': [0.0001] * 5 + [0.01]}, index=dates)
    entries = funding_extreme_reversion_entries(df, lookback=5, quantile=0.8,
                                                funding_col='my_funding')
    assert entries.iloc[5] == -1


def test_funding_reversion_confirmed_entries_fires_when_price_confirms():
    dates = pd.date_range("2023-01-01", periods=12, freq="4h")
    df = pd.DataFrame({
        'funding_rate': [0.0001] * 5 + [0.01] + [0.0001] * 5 + [-0.01],
        'bb_position': [0.0] * 5 + [0.7] + [0.0] * 5 + [-0.7],
    }, index=dates)
    entries = funding_reversion_confirmed_entries(df, lookback=5, quantile=0.8, bb_threshold=0.5)
    assert entries.iloc[5] == -1
    assert entries.iloc[11] == 1


def test_funding_reversion_confirmed_entries_filters_out_unconfirmed_spike():
    """Funding fires its own extreme, but price is NOT extended in the
    same direction -- the confirmed signal must abstain (this is the
    whole point of the filter).
    """
    dates = pd.date_range("2023-01-01", periods=6, freq="4h")
    df = pd.DataFrame({
        'funding_rate': [0.0001] * 5 + [0.01],
        'bb_position': [0.0] * 6,
    }, index=dates)
    entries = funding_reversion_confirmed_entries(df, lookback=5, quantile=0.8, bb_threshold=0.5)
    assert (entries == 0).all()


def test_funding_reversion_confirmed_entries_is_a_subset_of_the_base_signal():
    """It only ever filters; it must never invent entries
    funding_extreme_reversion_entries would not give.
    """
    rng = np.random.default_rng(7)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="4h")
    df = pd.DataFrame({
        'funding_rate': rng.normal(0, 0.005, n),
        'bb_position': rng.normal(0, 0.6, n),
    }, index=dates)

    base = funding_extreme_reversion_entries(df, lookback=20, quantile=0.9)
    confirmed = funding_reversion_confirmed_entries(df, lookback=20, quantile=0.9)
    fired = confirmed != 0
    assert (confirmed[fired] == base[fired]).all()
    assert int(fired.sum()) <= int((base != 0).sum())


def test_funding_reversion_regime_filtered_entries_passes_through_when_not_expanding():
    dates = pd.date_range("2023-01-01", periods=6, freq="4h")
    df = pd.DataFrame({
        'funding_rate': [0.0001] * 5 + [0.01],
        'ATR_ratio': [1.0] * 6,  # stable/contracting -> not expanding
    }, index=dates)
    entries = funding_reversion_regime_filtered_entries(df, lookback=5, quantile=0.8)
    assert entries.iloc[5] == -1


def test_funding_reversion_regime_filtered_entries_blocked_when_volatility_expanding():
    """A cascading-liquidation regime (ATR_ratio >= 1.05, the SAME cutoff
    range_fade_entries uses) must block the trade even though funding
    itself is extreme -- the whole point of section 20's filter.
    """
    dates = pd.date_range("2023-01-01", periods=6, freq="4h")
    df = pd.DataFrame({
        'funding_rate': [0.0001] * 5 + [0.01],
        'ATR_ratio': [1.2] * 6,  # expanding
    }, index=dates)
    entries = funding_reversion_regime_filtered_entries(df, lookback=5, quantile=0.8)
    assert (entries == 0).all()


def test_funding_reversion_regime_filtered_entries_is_a_subset_of_the_base_signal():
    rng = np.random.default_rng(11)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="4h")
    df = pd.DataFrame({
        'funding_rate': rng.normal(0, 0.005, n),
        'ATR_ratio': rng.uniform(0.7, 1.4, n),
    }, index=dates)

    base = funding_extreme_reversion_entries(df, lookback=20, quantile=0.9)
    filtered = funding_reversion_regime_filtered_entries(df, lookback=20, quantile=0.9)
    fired = filtered != 0
    assert (filtered[fired] == base[fired]).all()
    assert int(fired.sum()) <= int((base != 0).sum())


def test_obv_divergence_entries_bearish_when_price_high_not_confirmed_by_obv():
    """Price prints a new 3-bar high at the last bar, but that bar's up-move
    happens on tiny volume (5) versus an earlier, bigger up-move (50) that
    was then partly given back -- OBV's own prior high (50) is never
    retested (ends at 45), so the breakout is unconfirmed -> fade short.
    """
    dates = pd.date_range("2023-01-01", periods=4, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 105.0, 103.0, 106.0],
        'volume': [10.0, 50.0, 10.0, 5.0],
    }, index=dates)
    entries = obv_divergence_entries(df, lookback=3)
    assert entries.iloc[3] == -1


def test_obv_divergence_entries_bullish_when_price_low_not_confirmed_by_obv():
    """Symmetric case: a new 3-bar price LOW on tiny volume, while OBV's
    own prior low is never retested -> fade long.
    """
    dates = pd.date_range("2023-01-01", periods=4, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 95.0, 97.0, 94.0],
        'volume': [10.0, 50.0, 10.0, 5.0],
    }, index=dates)
    entries = obv_divergence_entries(df, lookback=3)
    assert entries.iloc[3] == 1


def test_obv_divergence_entries_silent_when_obv_confirms_the_breakout():
    """Same price shape as a breakout, but volume tracks price move-for-move
    so OBV makes its own new high right alongside price -- no divergence,
    no signal.
    """
    dates = pd.date_range("2023-01-01", periods=4, freq="4h")
    df = pd.DataFrame({
        'close': [100.0, 101.0, 102.0, 103.0],
        'volume': [10.0, 10.0, 10.0, 10.0],
    }, index=dates)
    entries = obv_divergence_entries(df, lookback=3)
    assert (entries == 0).all()


def test_btc_lead_lag_entries_direction():
    """BTC's trailing 5-bar return sits flat, spikes up through +3%
    (crossing bar fires long), returns to flat, then spikes down through
    -3% (crossing bar fires short).
    """
    dates = pd.date_range("2023-01-01", periods=12, freq="4h")
    df = pd.DataFrame({
        'btc_ret_5': [0.0] * 5 + [0.05] + [0.0] * 5 + [-0.05],
    }, index=dates)
    entries = btc_lead_lag_entries(df, threshold=0.03)
    assert entries.iloc[5] == 1
    assert entries.iloc[11] == -1
    assert (entries.drop(entries.index[[5, 11]]) == 0).all()


def test_btc_lead_lag_entries_no_repeat_while_extreme_persists():
    dates = pd.date_range("2023-01-01", periods=8, freq="4h")
    df = pd.DataFrame({'btc_ret_5': [0.0] * 5 + [0.05, 0.06, 0.07]}, index=dates)
    entries = btc_lead_lag_entries(df, threshold=0.03)
    assert entries.iloc[5] == 1
    assert entries.iloc[6] == 0
    assert entries.iloc[7] == 0


def test_btc_lead_lag_entries_uses_custom_column_name():
    dates = pd.date_range("2023-01-01", periods=6, freq="4h")
    df = pd.DataFrame({'my_btc_ret': [0.0] * 5 + [0.05]}, index=dates)
    entries = btc_lead_lag_entries(df, threshold=0.03, btc_ret_col='my_btc_ret')
    assert entries.iloc[5] == 1
