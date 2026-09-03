import numpy as np
import pandas as pd
import pytest

from scripts.backtest_daily_alpha import run_strategy_backtest, compute_metrics


def _make_dummy_ohlcv(n: int = 150, base_price: float = 100.0, trend: float = 0.05) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    t = np.linspace(0, 1, n)
    close = base_price * (1.0 + trend * t + 0.01 * np.sin(np.linspace(0, 10 * np.pi, n)))
    high = close * 1.01
    low = close * 0.99
    open_p = (high + low) / 2.0
    vol = np.full(n, 1000.0)
    return pd.DataFrame({
        "open": open_p, "high": high, "low": low, "close": close, "volume": vol
    }, index=idx)


def test_run_strategy_backtest_dummy():
    dfs = {
        "BTC_USDT": _make_dummy_ohlcv(150, base_price=50000.0, trend=0.10),
        "ETH_USDT": _make_dummy_ohlcv(150, base_price=3000.0, trend=0.12),
        "SOL_USDT": _make_dummy_ohlcv(150, base_price=150.0, trend=0.08),
    }
    trades = run_strategy_backtest(dfs, donchian_lookback=10, squeeze_pct=0.90, rs_window=5)
    assert isinstance(trades, pd.DataFrame)
    if not trades.empty:
        assert "symbol" in trades.columns
        assert "ret_gross" in trades.columns
        assert "entry_time" in trades.columns


def test_compute_metrics_dummy():
    idx = pd.date_range("2026-01-01", periods=4, freq="1D")
    dummy_trades = pd.DataFrame([
        {"symbol": "ETH_USDT", "entry_time": idx[0], "exit_time": idx[1], "ret_gross": 0.03},
        {"symbol": "SOL_USDT", "entry_time": idx[1], "exit_time": idx[2], "ret_gross": -0.01},
        {"symbol": "BTC_USDT", "entry_time": idx[2], "exit_time": idx[3], "ret_gross": 0.02},
    ])
    metrics = compute_metrics(dummy_trades, total_days=4.0, cost_per_trade=0.0016)
    assert metrics["total_trades"] == 3
    assert metrics["signals_per_day"] == 0.75
    assert metrics["win_rate_pct"] == pytest.approx(66.6666, rel=1e-2)
    assert metrics["profit_factor"] > 1.0
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_pct" in metrics
