"""Test filtered Liquidity Sweep and VWAP Mean-Reversion Oscillation Strategies."""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.backtest_daily_alpha import download_or_load_data, DEFAULT_SYMBOLS
from src.regime import build_btc_regime
from research.alpha_research import evaluate_trades, simulate_advanced_barrier


def calculate_vwap_bands(df, window=24):
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    vp = typical_price * df['volume']
    rolling_vp = vp.rolling(window).sum()
    rolling_v = df['volume'].rolling(window).sum()
    vwap = rolling_vp / (rolling_v + 1e-9)

    rolling_var = ((typical_price - vwap) ** 2 * df['volume']).rolling(window).sum() / (rolling_v + 1e-9)
    vwap_std = np.sqrt(np.maximum(rolling_var, 0))
    return vwap, vwap_std


def test_filtered_oscillation(dfs):
    btc_df = dfs['BTC_USDT']
    btc_regime = build_btc_regime(btc_df)
    symbols = list(dfs.keys())
    total_days = len(btc_df) / 24.0

    print("Testing Filtered Oscillation & Mean-Reversion across universe...")

    results = []
    for s_type in ['sweep', 'vwap', 'bb_rsi']:
        for pt_m in [1.5, 2.0, 2.5, 3.0]:
            for sl_m in [1.0, 1.2, 1.5]:
                trades = []
                for s in symbols:
                    df = dfs[s]
                    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
                    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
                    atr = tr.rolling(14).mean()

                    # ADX
                    up_move = h - h.shift(1)
                    down_move = l.shift(1) - l
                    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
                    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
                    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / (atr + 1e-9))
                    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / (atr + 1e-9))
                    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
                    adx = dx.rolling(14).mean()
                    is_chop = adx < 22.0

                    # RSI
                    delta = c.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs_calc = gain / (loss + 1e-9)
                    rsi = 100.0 - (100.0 / (1.0 + rs_calc))

                    # VWAP
                    vwap, vwap_std = calculate_vwap_bands(df, window=24)

                    hi24 = h.rolling(24).max().shift(1)
                    lo24 = l.rolling(24).min().shift(1)

                    if s_type == 'sweep':
                        # Liquidity sweep in chop with RSI confirmation
                        long_sig = (l < lo24) & (c > lo24) & (c > o) & is_chop & (rsi < 40)
                        short_sig = (h > hi24) & (c < hi24) & (c < o) & is_chop & (rsi > 60)
                    elif s_type == 'vwap':
                        # VWAP band fade
                        long_sig = (c < vwap - 1.8 * vwap_std) & (rsi < 32) & is_chop
                        short_sig = (c > vwap + 1.8 * vwap_std) & (rsi > 68) & is_chop
                    else:
                        # Bollinger Reversion
                        sma20 = c.rolling(20).mean()
                        std20 = c.rolling(20).std()
                        bb_upper = sma20 + 2.0 * std20
                        bb_lower = sma20 - 2.0 * std20
                        long_sig = (l < bb_lower) & (c > bb_lower) & (rsi < 30) & is_chop
                        short_sig = (h > bb_upper) & (c < bb_upper) & (rsi > 70) & is_chop

                    highs = h.to_numpy()
                    lows = l.to_numpy()
                    closes = c.to_numpy()
                    atr_vals = atr.to_numpy()
                    n = len(df)

                    for i in range(n):
                        side = 0
                        if long_sig.iloc[i]:
                            side = 1
                        elif short_sig.iloc[i]:
                            side = -1

                        if side == 0 or np.isnan(atr_vals[i]):
                            continue

                        entry_p = closes[i]
                        pt_dist = pt_m * atr_vals[i]
                        sl_dist = sl_m * atr_vals[i]

                        outcome = simulate_advanced_barrier(highs, lows, closes, i, side, entry_p,
                                                            pt_dist, sl_dist, max_hold=24)
                        exit_idx = min(i + outcome['hold'], n - 1)
                        trades.append({
                            'symbol': s,
                            'entry_time': df.index[i],
                            'exit_time': df.index[exit_idx],
                            'side': side,
                            'ret_gross': side * (outcome['exit_price'] / entry_p - 1.0)
                        })

                tdf = pd.DataFrame(trades)
                if len(tdf) > 30:
                    m_m = evaluate_trades(tdf, total_days, cost=0.0006)
                    m_t = evaluate_trades(tdf, total_days, cost=0.0016)
                    results.append({
                        'type': s_type, 'pt': pt_m, 'sl': sl_m, 'trades': len(tdf),
                        'per_day': len(tdf) / total_days, 'win_pct': m_m['win_rate'],
                        'm_sharpe': m_m['sharpe'], 'm_pf': m_m['pf'], 'm_ret': m_m['ret_pct'],
                        't_sharpe': m_t['sharpe'], 't_pf': m_t['pf'], 't_ret': m_t['ret_pct']
                    })

    rdf = pd.DataFrame(results).sort_values('m_sharpe', ascending=False)
    print("TOP OSCILLATION STRATEGIES:")
    print(rdf.head(15).to_string(index=False))


if __name__ == '__main__':
    dfs = download_or_load_data(DEFAULT_SYMBOLS, timeframe='1h', limit=2500)
    test_filtered_oscillation(dfs)
