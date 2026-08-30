import numpy as np
import pandas as pd

from src.regime import build_btc_regime, add_alignment_features, REGIME_FEATURE_COLS


def _btc_frame(closes):
    idx = pd.date_range("2023-01-01", periods=len(closes), freq="4h")
    return pd.DataFrame({
        'open': closes, 'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes], 'close': closes,
        'volume': [1.0] * len(closes),
    }, index=idx)


def test_btc_regime_marks_uptrend_bullish():
    """A monotonically rising BTC series must end up flagged as a bullish
    regime by both the MACD and the SMA measures."""
    btc = _btc_frame(list(np.linspace(100, 200, 120)))
    regime = build_btc_regime(btc)

    assert regime['btc_trend'].iloc[-1] == 1.0
    assert regime['btc_above_sma'].iloc[-1] == 1.0
    assert set(REGIME_FEATURE_COLS[:6]).issubset(regime.columns)


def test_btc_regime_marks_downtrend_bearish():
    btc = _btc_frame(list(np.linspace(200, 100, 120)))
    regime = build_btc_regime(btc)

    assert regime['btc_trend'].iloc[-1] == -1.0
    assert regime['btc_above_sma'].iloc[-1] == -1.0


def test_alignment_is_the_side_times_trend_interaction():
    """btc_alignment must be 1 when the trade's side agrees with BTC's
    trend and 0 when it fights it — this is the whole point of computing
    the interaction explicitly instead of leaving it to the trees.
    """
    idx = pd.date_range("2023-01-01", periods=4, freq="4h")
    trades = pd.DataFrame({'side': [1, -1, 1, -1]}, index=idx)
    regime = pd.DataFrame({
        'btc_trend': [1.0, 1.0, -1.0, -1.0],
        'btc_trend_strength': [0.02, 0.02, -0.02, -0.02],
        'btc_above_sma': [1.0, 1.0, -1.0, -1.0],
        'btc_ret_5': [0.05, 0.05, -0.05, -0.05],
        'btc_vol': [0.5] * 4,
        'btc_vol_ratio': [1.0] * 4,
    }, index=idx)

    out = add_alignment_features(trades, regime)

    # long in an uptrend -> aligned; short in an uptrend -> misaligned;
    # long in a downtrend -> misaligned; short in a downtrend -> aligned.
    assert list(out['btc_alignment']) == [1.0, 0.0, 0.0, 1.0]
    assert list(out['btc_alignment_sma']) == [1.0, 0.0, 0.0, 1.0]


def test_alignment_strength_is_signed_by_side():
    """A short in a strongly bearish regime should get a POSITIVE strength
    (a tailwind), mirroring a long in a bullish one."""
    idx = pd.date_range("2023-01-01", periods=2, freq="4h")
    trades = pd.DataFrame({'side': [1, -1]}, index=idx)
    regime = pd.DataFrame({
        'btc_trend': [1.0, -1.0], 'btc_trend_strength': [0.03, -0.03],
        'btc_above_sma': [1.0, -1.0], 'btc_ret_5': [0.1, -0.1],
        'btc_vol': [0.5, 0.5], 'btc_vol_ratio': [1.0, 1.0],
    }, index=idx)

    out = add_alignment_features(trades, regime)

    assert out['btc_alignment_strength'].iloc[0] > 0
    assert out['btc_alignment_strength'].iloc[1] > 0
    assert out['btc_ret_5_aligned'].iloc[1] > 0  # short profiting from a drop


def test_missing_regime_data_yields_nan_not_a_fabricated_neutral():
    """Trades whose timestamp has no BTC regime coverage must be NaN so the
    caller drops them — silently filling a 'neutral' regime would be
    indistinguishable from a real observation.
    """
    trades = pd.DataFrame(
        {'side': [1]}, index=pd.date_range("2030-01-01", periods=1, freq="4h"))
    regime = pd.DataFrame({
        'btc_trend': [1.0], 'btc_trend_strength': [0.01], 'btc_above_sma': [1.0],
        'btc_ret_5': [0.0], 'btc_vol': [0.4], 'btc_vol_ratio': [1.0],
    }, index=pd.date_range("2023-01-01", periods=1, freq="4h"))

    out = add_alignment_features(trades, regime)

    assert np.isnan(out['btc_alignment'].iloc[0])
