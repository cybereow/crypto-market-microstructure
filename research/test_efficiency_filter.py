import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_daily_alpha import download_or_load_data
from src.regime import build_btc_regime
from research.alpha_research import evaluate_trades, simulate_advanced_barrier
import pandas as pd
import numpy as np

EXPANDED_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'SUI/USDT', 'NEAR/USDT',
    'APT/USDT', 'INJ/USDT', 'RENDER/USDT', 'FET/USDT', 'OP/USDT', 'TIA/USDT'
]

dfs = download_or_load_data(EXPANDED_SYMBOLS, timeframe='1h', limit=2500)
symbols = list(dfs.keys())
btc_regime = build_btc_regime(dfs['BTC_USDT'])
total_days = len(dfs['BTC_USDT']) / 24.0

print(f"Testing Kaufman Efficiency Ratio (ER) filter across {len(symbols)} assets...")

for er_thresh in [0.0, 0.20, 0.25, 0.30, 0.35]:
    rets_df = pd.DataFrame({s: dfs[s]['close'].pct_change(24) for s in symbols})
    ranks_df = rets_df.rank(axis=1, ascending=False)
    top_cutoff = max(1, int(len(symbols) * 0.38))

    trades = []
    for s in symbols:
        df = dfs[s]
        h, l, c, v = df['high'], df['low'], df['close'], df['volume']
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        ema100 = c.ewm(span=100).mean()

        # ADX
        up_move = h - h.shift(1)
        down_move = l.shift(1) - l
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / (atr + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / (atr + 1e-9))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(14).mean()
        adx_ok = adx >= 20.0

        # Kaufman Efficiency Ratio (24-period)
        change_24 = (c - c.shift(24)).abs()
        volatility_24 = (c - c.shift(1)).abs().rolling(24).sum()
        er_24 = change_24 / (volatility_24 + 1e-9)
        er_ok = (er_24 >= er_thresh) if er_thresh > 0 else pd.Series(True, index=df.index)

        hi = h.rolling(48).max().shift(1)
        lo = l.rolling(48).min().shift(1)
        vol_sma20 = v.rolling(20).mean()
        vol_ok = v >= 1.05 * vol_sma20

        reg_bull = btc_regime['btc_trend'] > 0
        reg_bear = btc_regime['btc_trend'] < 0

        is_top_rs = ranks_df[s] <= top_cutoff
        is_bot_rs = ranks_df[s] >= (len(symbols) - top_cutoff + 1)

        long_bo = (c > hi) & (c.shift(1) <= hi.shift(1)) & (c > ema100) & is_top_rs & reg_bull & adx_ok & vol_ok & er_ok
        short_bo = (c < lo) & (c.shift(1) >= lo.shift(1)) & (c < ema100) & is_bot_rs & reg_bear & adx_ok & vol_ok & er_ok

        highs = h.to_numpy()
        lows = l.to_numpy()
        closes = c.to_numpy()
        atr_vals = atr.to_numpy()
        n = len(df)

        for i in range(n):
            side = 0
            if long_bo.iloc[i]:
                side = 1
            elif short_bo.iloc[i]:
                side = -1

            if side == 0 or np.isnan(atr_vals[i]):
                continue

            entry_p = closes[i]
            pt_dist = 4.0 * atr_vals[i]
            sl_dist = 1.2 * atr_vals[i]

            outcome = simulate_advanced_barrier(highs, lows, closes, i, side, entry_p, pt_dist, sl_dist, max_hold=36)
            exit_idx = min(i + outcome['hold'], n - 1)
            ret_gross = side * (outcome['exit_price'] / entry_p - 1.0)
            trades.append({
                'symbol': s,
                'entry_time': df.index[i],
                'exit_time': df.index[exit_idx],
                'side': side,
                'ret_gross': ret_gross
            })

    tdf = pd.DataFrame(trades)
    m_m = evaluate_trades(tdf, total_days, cost=0.0006)
    m_t = evaluate_trades(tdf, total_days, cost=0.0016)
    print(f"ER Threshold: {er_thresh:.2f} -> Trades: {len(tdf):<3d} ({len(tdf)/total_days:.2f}/d) | "
          f"Maker: Sharpe={m_m['sharpe']:.2f} PF={m_m['pf']:.2f} Ret={m_m['ret_pct']:+.2f}% DD={m_m['max_dd']:.2f}% | "
          f"Taker: Sharpe={m_t['sharpe']:.2f} PF={m_t['pf']:.2f} Ret={m_t['ret_pct']:+.2f}% DD={m_t['max_dd']:.2f}%")
