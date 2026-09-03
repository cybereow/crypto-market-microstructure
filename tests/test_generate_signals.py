import numpy as np
import pandas as pd
import pytest

from scripts.generate_signals import scan_universe_signals, calculate_vwap


def _make_dummy_ohlcv(n: int = 100, base_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    t = np.linspace(0, 1, n)
    close = base_price * (1.0 + 0.05 * t + 0.01 * np.sin(np.linspace(0, 10 * np.pi, n)))
    high = close * 1.01
    low = close * 0.99
    open_p = (high + low) / 2.0
    vol = np.full(n, 1000.0)
    return pd.DataFrame({
        "open": open_p, "high": high, "low": low, "close": close, "volume": vol
    }, index=idx)


def test_calculate_vwap():
    df = _make_dummy_ohlcv(50)
    vwap = calculate_vwap(df)
    assert len(vwap) == len(df)
    assert not vwap.dropna().empty


def test_scan_universe_signals():
    dfs = {
        "BTC_USDT": _make_dummy_ohlcv(100, base_price=50000.0),
        "ETH_USDT": _make_dummy_ohlcv(100, base_price=3000.0),
        "SOL_USDT": _make_dummy_ohlcv(100, base_price=150.0),
    }
    res = scan_universe_signals(dfs, donchian_lookback=10, squeeze_pct=0.90, rs_window=5)
    assert "btc_regime" in res
    assert "live_signals" in res
    assert "watchlist" in res
    assert len(res["watchlist"]) == 3
