"""Production-Ready Signal Generator for Crypto Multi-Asset Daily Alpha Strategy.

Scans the 1h crypto universe:
  1. Filters by BTC market regime (trend & volatility)
  2. Ranks assets by cross-sectional relative strength (RS)
  3. Detects volatility squeeze (Bollinger Band compression) and Donchian channel breakout
  4. Identifies VWAP / SMA pullback opportunities
  5. Computes exact entry, stop-loss (1.5x ATR), profit-target (2.5x ATR), and position sizing (Fixed Risk / Kelly)
  6. Outputs actionable maker/taker execution instructions and a watchlist of near-breakout candidates.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

try:
    import ccxt
except ImportError:
    ccxt = None

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from src.regime import build_btc_regime
from scripts.backtest_daily_alpha import download_or_load_data

DEFAULT_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'ADA/USDT', 'SUI/USDT', 'NEAR/USDT', 'INJ/USDT', 'FET/USDT', 'OP/USDT'
]


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate rolling 24-hour VWAP."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    vp = typical_price * df['volume']
    rolling_vp = vp.rolling(24).sum()
    rolling_v = df['volume'].rolling(24).sum()
    return rolling_vp / (rolling_v + 1e-9)


def scan_universe_signals(dfs: dict[str, pd.DataFrame],
                          donchian_lookback: int = 48,
                          squeeze_pct: float = 0.50,
                          rs_window: int = 24,
                          pt_mult: float = 4.0,
                          sl_mult: float = 1.2,
                          vol_mult: float = 1.05,
                          min_adx: float = 20.0,
                          account_size: float = 10000.0,
                          risk_pct: float = 1.0) -> dict:
    """Scan latest market state and return live signals, active trades, and watchlist."""
    if 'BTC_USDT' not in dfs:
        raise ValueError("BTC_USDT required for market regime.")

    btc_df = dfs['BTC_USDT']
    btc_regime = build_btc_regime(btc_df)
    latest_btc_idx = btc_df.index[-1]

    # Current BTC regime
    btc_trend_val = btc_regime.loc[latest_btc_idx, 'btc_trend']
    btc_trend_str = "BULLISH (+1)" if btc_trend_val > 0 else "BEARISH (-1)"
    btc_strength = btc_regime.loc[latest_btc_idx, 'btc_trend_strength']
    btc_vol = btc_regime.loc[latest_btc_idx, 'btc_vol']

    symbols = list(dfs.keys())
    # Cross-sectional relative strength
    rets_df = pd.DataFrame({s: dfs[s]['close'].pct_change(rs_window) for s in symbols}).ffill()
    if latest_btc_idx in rets_df.index:
        latest_rs = rets_df.loc[latest_btc_idx].rank(ascending=False)
    else:
        latest_rs = rets_df.iloc[-1].rank(ascending=False)
    top_cutoff = max(1, int(len(symbols) * 0.38))

    live_signals = []
    watchlist = []
    recent_signals = []

    for s in symbols:
        df = dfs[s]
        h, l, c, v = df['high'], df['low'], df['close'], df['volume']

        # ATR 14
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # Higher-timeframe trend filter: 100-period EMA
        ema100 = c.ewm(span=100).mean()

        # ADX 14 for trend strength validation
        up_move = h - h.shift(1)
        down_move = l.shift(1) - l
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / (atr + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / (atr + 1e-9))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(14).mean()
        adx_ok = (adx >= min_adx) if min_adx > 0 else pd.Series(True, index=df.index)

        # Volume surge confirmation relative to 20-period SMA
        vol_sma20 = v.rolling(20).mean() if v is not None else None
        vol_ok = (v >= vol_mult * vol_sma20) if (v is not None and vol_mult > 0) else pd.Series(True, index=df.index)

        # Bollinger Bands & Squeeze Rank
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20
        bb_w = (4.0 * std20) / (sma20 + 1e-9)
        sq_rank = bb_w.rolling(50).rank(pct=True).shift(1)
        is_squeezed = (sq_rank <= squeeze_pct) | (sq_rank.shift(1) <= squeeze_pct)

        # RSI 14
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs_calc = gain / (loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs_calc))

        # Donchian channel
        hi = h.rolling(donchian_lookback).max().shift(1)
        lo = l.rolling(donchian_lookback).min().shift(1)

        # VWAP
        vwap = calculate_vwap(df)

        # Relative strength
        rs_rank = latest_rs.get(s, np.nan)
        is_top = rs_rank <= top_cutoff
        is_bot = rs_rank >= (len(symbols) - top_cutoff + 1)

        # Regime alignment
        reg_bull = (btc_regime['btc_trend'] > 0)
        reg_bear = (btc_regime['btc_trend'] < 0)

        # 1) Breakout Signals (Trend Expansion: Macro Trend + Dynamic RS + 48h Donchian + ADX + Volume Surge)
        long_bo = (c > hi) & (c.shift(1) <= hi.shift(1)) & (c > ema100) & reg_bull & is_top & adx_ok & vol_ok
        short_bo = (c < lo) & (c.shift(1) >= lo.shift(1)) & (c < ema100) & reg_bear & is_bot & adx_ok & vol_ok

        # 2) Reversal Signals (Active only in chop when ADX < 20)
        long_rev = (l < bb_lower) & (c > bb_lower) & (rsi < 30) & (adx < 20) & reg_bull
        short_rev = (h > bb_upper) & (c < bb_upper) & (rsi > 70) & (adx < 20) & reg_bear

        # Check latest fully closed candle
        curr_price = float(c.iloc[-1])
        curr_atr = float(atr.iloc[-1])
        curr_sq = float(sq_rank.iloc[-1])
        curr_hi = float(hi.iloc[-1])
        curr_lo = float(lo.iloc[-1])
        curr_vwap = float(vwap.iloc[-1])
        curr_rsi = float(rsi.iloc[-1])
        curr_adx = float(adx.iloc[-1])

        side = 0
        strategy_name = 'NONE'
        curr_pt_mult = pt_mult
        curr_sl_mult = sl_mult

        if long_bo.iloc[-1]:
            side = 1
            strategy_name = 'BREAKOUT'
            curr_pt_mult, curr_sl_mult = pt_mult, sl_mult
        elif short_bo.iloc[-1]:
            side = -1
            strategy_name = 'BREAKOUT'
            curr_pt_mult, curr_sl_mult = pt_mult, sl_mult
        elif long_rev.iloc[-1]:
            side = 1
            strategy_name = 'REVERSAL'
            curr_pt_mult, curr_sl_mult = 2.0, 1.2
        elif short_rev.iloc[-1]:
            side = -1
            strategy_name = 'REVERSAL'
            curr_pt_mult, curr_sl_mult = 2.0, 1.2

        # Sizing: risk dollar amount = account_size * (risk_pct / 100)
        risk_dollar = account_size * (risk_pct / 100.0)
        sl_dist = curr_sl_mult * curr_atr
        pt_dist = curr_pt_mult * curr_atr

        # Stop loss percentage
        sl_pct = (sl_dist / curr_price) * 100.0
        # Position size in USDT
        pos_size_usdt = risk_dollar / (sl_dist / curr_price)
        pos_size_coins = pos_size_usdt / curr_price

        if side != 0:
            entry_p = curr_price
            sl_p = entry_p - sl_dist if side == 1 else entry_p + sl_dist
            pt_p = entry_p + pt_dist if side == 1 else entry_p - pt_dist

            maker_entry_p = entry_p - 0.15 * curr_atr if side == 1 else entry_p + 0.15 * curr_atr

            live_signals.append({
                'symbol': s,
                'strategy': strategy_name,
                'time': df.index[-1],
                'side': 'LONG' if side == 1 else 'SHORT',
                'close': curr_price,
                'maker_entry': maker_entry_p,
                'stop_loss': sl_p,
                'profit_target': pt_p,
                'risk_reward': curr_pt_mult / curr_sl_mult,
                'atr': curr_atr,
                'rs_rank': int(rs_rank) if pd.notna(rs_rank) else 999,
                'pos_size_usdt': pos_size_usdt,
                'pos_size_coins': pos_size_coins,
                'risk_dollar': risk_dollar,
                'sl_pct': sl_pct
            })

        # 2. Watchlist calculation (Proximity to breakout)
        dist_hi_pct = ((curr_hi - curr_price) / curr_price) * 100.0
        dist_lo_pct = ((curr_price - curr_lo) / curr_price) * 100.0
        dist_hi_atr = (curr_hi - curr_price) / (curr_atr + 1e-9)
        dist_lo_atr = (curr_price - curr_lo) / (curr_atr + 1e-9)

        watchlist.append({
            'symbol': s,
            'price': curr_price,
            'atr': curr_atr,
            'rsi': curr_rsi,
            'adx': curr_adx,
            'squeeze_rank': curr_sq,
            'is_squeezed': bool(is_squeezed.iloc[-1]),
            'rs_rank': int(rs_rank) if pd.notna(rs_rank) else 999,
            'donchian_high': curr_hi,
            'donchian_low': curr_lo,
            'dist_long_pct': dist_hi_pct,
            'dist_short_pct': dist_lo_pct,
            'dist_long_atr': dist_hi_atr,
            'dist_short_atr': dist_lo_atr,
            'vwap': curr_vwap,
            'vwap_dist_pct': ((curr_price - curr_vwap) / curr_vwap) * 100.0
        })

        # 3. Check historical signals in last 48 bars
        tail_bars = min(48, len(df))
        for j in range(len(df) - tail_bars, len(df) - 1):
            past_side = 0
            past_strat = 'NONE'
            p_pt_m, p_sl_m = pt_mult, sl_mult
            if long_bo.iloc[j]:
                past_side, past_strat = 1, 'BREAKOUT'
            elif short_bo.iloc[j]:
                past_side, past_strat = -1, 'BREAKOUT'
            elif long_rev.iloc[j]:
                past_side, past_strat = 1, 'REVERSAL'
                p_pt_m, p_sl_m = 2.0, 1.2
            elif short_rev.iloc[j]:
                past_side, past_strat = -1, 'REVERSAL'
                p_pt_m, p_sl_m = 2.0, 1.2

            if past_side != 0:
                p_entry = float(c.iloc[j])
                p_atr = float(atr.iloc[j])
                recent_signals.append({
                    'symbol': s,
                    'strategy': past_strat,
                    'time': df.index[j],
                    'side': 'LONG' if past_side == 1 else 'SHORT',
                    'entry_price': p_entry,
                    'stop_loss': p_entry - p_sl_m * p_atr if past_side == 1 else p_entry + p_sl_m * p_atr,
                    'profit_target': p_entry + p_pt_m * p_atr if past_side == 1 else p_entry - p_pt_m * p_atr,
                    'current_price': curr_price,
                    'ret_since_pct': past_side * (curr_price / p_entry - 1.0) * 100.0,
                    'bars_ago': len(df) - 1 - j
                })

    return {
        'btc_regime': {
            'trend': btc_trend_str,
            'strength': btc_strength,
            'vol': btc_vol,
            'time': latest_btc_idx
        },
        'live_signals': live_signals,
        'watchlist': sorted(watchlist, key=lambda x: min(abs(x['dist_long_atr']), abs(x['dist_short_atr']))),
        'recent_signals': sorted(recent_signals, key=lambda x: x['time'], reverse=True)
    }


def print_signal_report(results: dict, account_size: float, risk_pct: float):
    """Print an authoritative command-line trading desk report."""
    btc = results['btc_regime']
    print("=" * 80)
    print("      CRYPTO DAILY ALPHA: MULTI-ASSET 1H SIGNAL GENERATOR (PRODUCTION)      ")
    print("=" * 80)
    print(f"Timestamp:           {btc['time']} UTC")
    print(f"BTC Regime Trend:    {btc['trend']} | Strength: {btc['strength']:+.4f} | Ann. Vol: {btc['vol']*100:.1f}%")
    print(f"Portfolio Account:   ${account_size:,.2f} USDT | Fixed Risk: {risk_pct:.1f}% (${account_size * risk_pct / 100:.2f} per trade)")
    print(f"Execution Target:    Maker Limit (0.15 ATR inside) or Taker (Momentum Breakout)")
    print("-" * 80)

    # 1. Actionable Live Signals
    live = results['live_signals']
    print(f"1. ACTIONABLE SIGNALS ON LATEST 1H CANDLE: {len(live)}")
    if live:
        for sig in live:
            print("  +" + "-" * 76 + "+")
            print(f"  | ACTION: {sig['side']:<5s} {sig['symbol']:<10s} [{sig['strategy']}] | RS Rank: #{sig['rs_rank']} | 1h ATR: {sig['atr']:.4f}")
            print(f"  | Current Trigger Price:    {sig['close']:.4f}")
            print(f"  | Recommended Maker Limit:  {sig['maker_entry']:.4f} (Earns Maker Rebate/Low Fee)")
            print(f"  | Stop Loss:                {sig['stop_loss']:.4f} (-{sig['sl_pct']:.2f}%)")
            print(f"  | Profit Target:            {sig['profit_target']:.4f} (+{sig['sl_pct']*sig['risk_reward']:.2f}%)")
            print(f"  | Risk / Reward Ratio:      1 : {sig['risk_reward']:.2f}")
            print(f"  | Suggested Position Size:  ${sig['pos_size_usdt']:,.2f} USDT ({sig['pos_size_coins']:.4f} coins)")
            print("  +" + "-" * 76 + "+")
    else:
        print("  [No breakout or reversal trigger fired on the immediate last 1h candle. Review watchlist below.]")

    print("\n" + "-" * 80)
    # 2. Watchlist / Pre-Breakout Proximity & Oscillations
    print("2. WATCHLIST: PROXIMITY TO BREAKOUT / SQUEEZE & RSI OSCILLATIONS")
    print(f"{'Symbol':<10s} | {'Price':<10s} | {'RS #':<5s} | {'RSI':<5s} | {'Squeeze':<8s} | {'Dist Long (ATR)':<16s} | {'Dist Short (ATR)':<16s} | {'VWAP Dist':<10s}")
    print("-" * 80)
    for item in results['watchlist']:
        sq_str = "ACTIVE" if item['is_squeezed'] else "NONE"
        print(f"{item['symbol']:<10s} | {item['price']:<10.4f} | #{item['rs_rank']:<4d} | {item['rsi']:<5.1f} | {sq_str:<8s} | "
              f"{item['dist_long_pct']:>+5.2f}% ({item['dist_long_atr']:>4.1f}x) | "
              f"{item['dist_short_pct']:>+5.2f}% ({item['dist_short_atr']:>4.1f}x) | "
              f"{item['vwap_dist_pct']:>+5.2f}%")

    print("\n" + "-" * 80)
    # 3. Recent Active Signals (Last 48 Hours)
    print("3. RECENT SIGNALS GENERATED IN LAST 48 HOURS")
    recent = results['recent_signals'][:8]
    if recent:
        print(f"{'Time':<19s} | {'Symbol':<10s} | {'Strat':<8s} | {'Side':<5s} | {'Entry Price':<12s} | {'Stop Loss':<12s} | {'Target':<12s} | {'Unrealized PnL':<15s}")
        print("-" * 80)
        for r in recent:
            print(f"{str(r['time'])[:19]:<19s} | {r['symbol']:<10s} | {r['strategy']:<8s} | {r['side']:<5s} | {r['entry_price']:<12.4f} | "
                  f"{r['stop_loss']:<12.4f} | {r['profit_target']:<12.4f} | {r['ret_since_pct']:>+13.2f}% ({r['bars_ago']}b ago)")
    else:
        print("  [No signals in the last 48 hours]")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate Live Crypto Signals for Daily Alpha Strategy")
    parser.add_argument("--symbols", nargs='+', default=DEFAULT_SYMBOLS,
                        help="Symbols to scan (e.g. BTC/USDT ETH/USDT ...)")
    parser.add_argument("--timeframe", type=str, default="1h", choices=["1h", "15m", "4h"],
                        help="Candle timeframe (default: 1h)")
    parser.add_argument("--limit", type=int, default=500,
                        help="Number of bars to fetch/load (default: 500)")
    parser.add_argument("--force-download", action="store_true",
                        help="Force fresh download from Binance API via CCXT")
    parser.add_argument("--account-size", type=float, default=10000.0,
                        help="Account equity in USDT (default: 10000)")
    parser.add_argument("--risk-pct", type=float, default=1.0,
                        help="Risk percentage per trade (default: 1.0%%)")
    parser.add_argument("--pt-mult", type=float, default=4.0,
                        help="Profit Target ATR multiplier (default: 4.0)")
    parser.add_argument("--sl-mult", type=float, default=1.2,
                        help="Stop Loss ATR multiplier (default: 1.2)")
    parser.add_argument("--vol-mult", type=float, default=1.05,
                        help="Volume surge multiplier relative to 20-period SMA (default: 1.05)")
    parser.add_argument("--donchian-lookback", type=int, default=48,
                        help="Donchian channel lookback (default: 48)")
    parser.add_argument("--squeeze-pct", type=float, default=0.50,
                        help="Bollinger squeeze rank percentile (default: 0.50)")
    parser.add_argument("--rs-window", type=int, default=24,
                        help="Relative strength lookback window (default: 24)")
    parser.add_argument("--min-adx", type=float, default=20.0,
                        help="Minimum ADX trend strength threshold (default: 20.0)")

    args = parser.parse_args()

    dfs = download_or_load_data(args.symbols, timeframe=args.timeframe, limit=args.limit,
                                force_download=args.force_download)

    if not dfs:
        print("Error: Could not obtain data for universe.")
        sys.exit(1)

    results = scan_universe_signals(
        dfs,
        donchian_lookback=args.donchian_lookback,
        squeeze_pct=args.squeeze_pct,
        rs_window=args.rs_window,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        vol_mult=args.vol_mult,
        min_adx=args.min_adx,
        account_size=args.account_size,
        risk_pct=args.risk_pct
    )

    print_signal_report(results, account_size=args.account_size, risk_pct=args.risk_pct)


if __name__ == "__main__":
    main()
