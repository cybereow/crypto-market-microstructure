"""Research script to explore alpha improvements for Multi-Asset 1H Strategy.
Testing:
  1. Breakeven / Trailing Stops
  2. Volume confirmation on breakouts
  3. Liquidity Sweep & Mean Reversion (Oscillation engine)
  4. Non-overlapping position constraints
  5. Multi-factor ranking (RS + Momentum + Volatility)
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.backtest_daily_alpha import download_or_load_data, DEFAULT_SYMBOLS
from src.regime import build_btc_regime


def simulate_advanced_barrier(highs, lows, closes, start_idx, side, entry_p,
                              pt_dist, sl_dist, max_hold,
                              breakeven_trigger_atr=0.0, atr_val=0.0):
    """Triple barrier with optional breakeven ratcheting."""
    n = len(closes)
    end_idx = min(start_idx + max_hold, n - 1)
    effective_sl_dist = sl_dist
    breakeven_active = False

    for j in range(start_idx + 1, end_idx + 1):
        h, l, c = highs[j], lows[j], closes[j]

        # Check if breakeven triggered
        if breakeven_trigger_atr > 0 and not breakeven_active:
            favorable_dist = (h - entry_p) if side == 1 else (entry_p - l)
            if favorable_dist >= breakeven_trigger_atr * atr_val:
                breakeven_active = True
                # Move stop to entry + 0.1 ATR (or entry)
                effective_sl_dist = -0.1 * atr_val

        # Check barriers
        if side == 1:
            if h >= entry_p + pt_dist:
                return {'exit_price': entry_p + pt_dist, 'hold': j - start_idx, 'label': 1}
            if l <= entry_p - effective_sl_dist:
                return {'exit_price': entry_p - effective_sl_dist, 'hold': j - start_idx, 'label': -1}
        else:
            if l <= entry_p - pt_dist:
                return {'exit_price': entry_p - pt_dist, 'hold': j - start_idx, 'label': 1}
            if h >= entry_p + effective_sl_dist:
                return {'exit_price': entry_p + effective_sl_dist, 'hold': j - start_idx, 'label': -1}

    # Time expiration
    final_c = closes[end_idx]
    label = 1 if side * (final_c - entry_p) > 0 else -1
    return {'exit_price': final_c, 'hold': end_idx - start_idx, 'label': label}


def test_variations(dfs):
    btc_df = dfs['BTC_USDT']
    btc_regime = build_btc_regime(btc_df)
    symbols = list(dfs.keys())
    total_days = len(btc_df) / 24.0

    print(f"Data loaded: {len(symbols)} symbols, {len(btc_df)} bars (~{total_days:.1f} days)\n")

    # Grid parameters
    donchian_lookbacks = [36, 48, 60]
    pt_mults = [2.5, 3.0, 3.5, 4.0]
    sl_mults = [1.2, 1.5, 1.8]
    be_triggers = [0.0, 1.2, 1.5, 1.8]  # breakeven trigger ATR
    vol_filters = [False, True]
    non_overlaps = [False, True]

    # Let's run a baseline and systematic parameter tests
    results = []

    # Test baseline first
    for dl in [36, 48]:
        for pt in [3.0, 3.5, 4.0]:
            for sl in [1.2, 1.5]:
                for be in [0.0, 1.5]:
                    for vol_f in [False, True]:
                        for non_ovlp in [False, True]:
                            trades = run_simulation(dfs, btc_regime, symbols,
                                                    donchian_lb=dl, pt_m=pt, sl_m=sl,
                                                    be_trigger=be, vol_filter=vol_f,
                                                    non_overlap=non_ovlp)
                            if len(trades) < 50:
                                continue
                            m_maker = evaluate_trades(trades, total_days, cost=0.0006)
                            m_taker = evaluate_trades(trades, total_days, cost=0.0016)
                            results.append({
                                'dl': dl, 'pt': pt, 'sl': sl, 'be': be, 'vol': vol_f, 'non_ovlp': non_ovlp,
                                'trades': len(trades), 'per_day': len(trades) / total_days,
                                'win_rate': m_maker['win_rate'],
                                'maker_sharpe': m_maker['sharpe'], 'maker_pf': m_maker['pf'],
                                'maker_ret': m_maker['ret_pct'], 'maker_dd': m_maker['max_dd'],
                                'taker_sharpe': m_taker['sharpe'], 'taker_pf': m_taker['pf'],
                                'taker_ret': m_taker['ret_pct'], 'taker_dd': m_taker['max_dd'],
                            })

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values('maker_sharpe', ascending=False)
    print("TOP 10 CONFIGURATIONS BY MAKER SHARPE:")
    print(rdf.head(10).to_string(index=False))

    print("\nTOP 10 CONFIGURATIONS BY TAKER SHARPE (With >= 3.0 trades/day):")
    high_freq = rdf[rdf['per_day'] >= 3.0].sort_values('taker_sharpe', ascending=False)
    print(high_freq.head(10).to_string(index=False))


def run_simulation(dfs, btc_regime, symbols, donchian_lb=48, pt_m=3.5, sl_m=1.5,
                   be_trigger=0.0, vol_filter=False, non_overlap=False,
                   rs_window=24, min_adx=20.0, max_hold=36):
    rets_df = pd.DataFrame({s: dfs[s]['close'].pct_change(rs_window) for s in symbols})
    ranks_df = rets_df.rank(axis=1, ascending=False)
    top_cutoff = max(1, int(len(symbols) * 0.38))

    trades = []

    for s in symbols:
        df = dfs[s]
        h, l, c, v = df['high'], df['low'], df['close'], df['volume']

        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        ema100 = c.ewm(span=100).mean()

        up_move = h - h.shift(1)
        down_move = l.shift(1) - l
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / (atr + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / (atr + 1e-9))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(14).mean()
        adx_ok = (adx >= min_adx) if min_adx > 0 else pd.Series(True, index=df.index)

        hi = h.rolling(donchian_lb).max().shift(1)
        lo = l.rolling(donchian_lb).min().shift(1)

        vol_sma20 = v.rolling(20).mean()
        vol_ok = (v >= 1.05 * vol_sma20) if vol_filter else pd.Series(True, index=df.index)

        reg_bull = (btc_regime['btc_trend'] > 0)
        reg_bear = (btc_regime['btc_trend'] < 0)

        is_top_rs = ranks_df[s] <= top_cutoff
        is_bot_rs = ranks_df[s] >= (len(symbols) - top_cutoff + 1)

        long_bo = (c > hi) & (c.shift(1) <= hi.shift(1)) & (c > ema100) & is_top_rs & reg_bull & adx_ok & vol_ok
        short_bo = (c < lo) & (c.shift(1) >= lo.shift(1)) & (c < ema100) & is_bot_rs & reg_bear & adx_ok & vol_ok

        highs = h.to_numpy()
        lows = l.to_numpy()
        closes = c.to_numpy()
        atr_vals = atr.to_numpy()
        n = len(df)

        last_exit_idx = -1

        for i in range(n):
            if non_overlap and i <= last_exit_idx:
                continue

            side = 0
            if long_bo.iloc[i]:
                side = 1
            elif short_bo.iloc[i]:
                side = -1

            if side == 0 or np.isnan(atr_vals[i]):
                continue

            entry_p = closes[i]
            entry_time = df.index[i]
            pt_dist = pt_m * atr_vals[i]
            sl_dist = sl_m * atr_vals[i]

            outcome = simulate_advanced_barrier(highs, lows, closes, i, side, entry_p,
                                                pt_dist, sl_dist, max_hold,
                                                breakeven_trigger_atr=be_trigger,
                                                atr_val=atr_vals[i])
            exit_idx = min(i + outcome['hold'], n - 1)
            last_exit_idx = exit_idx
            exit_time = df.index[exit_idx]
            ret_gross = side * (outcome['exit_price'] / entry_p - 1.0)

            trades.append({
                'symbol': s,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'side': side,
                'ret_gross': ret_gross
            })

    return pd.DataFrame(trades)


def evaluate_trades(trades_df, total_days, cost=0.0006):
    df = trades_df.copy()
    df['ret_net'] = df['ret_gross'] - cost
    n = len(df)
    wins = df[df['ret_net'] > 0]
    losses = df[df['ret_net'] <= 0]
    win_rate = len(wins) / n * 100.0 if n > 0 else 0.0

    gp = wins['ret_net'].sum()
    gl = abs(losses['ret_net'].sum())
    pf = gp / gl if gl > 0 else 99.0

    capital = 10000.0
    equity = [capital]
    for _, r in df.sort_values('exit_time').iterrows():
        capital += capital * 0.20 * r['ret_net']
        equity.append(capital)

    eq = pd.Series(equity)
    cum_ret = (capital / 10000.0 - 1.0) * 100.0
    dd = (eq - eq.cummax()) / eq.cummax()
    max_dd = abs(dd.min()) * 100.0

    # Hourly Sharpe annualized
    daily_rets = df.groupby(df['exit_time'].dt.date)['ret_net'].sum()
    mean_d = daily_rets.mean()
    std_d = daily_rets.std() + 1e-9
    sharpe = (mean_d / std_d) * np.sqrt(365) if len(daily_rets) > 5 else 0.0

    return {
        'win_rate': win_rate,
        'pf': pf,
        'sharpe': sharpe,
        'ret_pct': cum_ret,
        'max_dd': max_dd
    }


if __name__ == '__main__':
    dfs = download_or_load_data(DEFAULT_SYMBOLS, timeframe='1h', limit=2500)
    test_variations(dfs)
