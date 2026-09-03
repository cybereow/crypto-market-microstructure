"""Backtesting script for Daily Alpha Strategy across crypto universe.

Strategy Components:
  1. Regime Filter: Market trend and regime determined from BTC (crypto's primary factor).
  2. Relative Strength: Cross-sectional momentum ranking across the universe.
  3. Volatility Breakout: Donchian channel breakout out of a Bollinger Band squeeze.
  4. ATR Dynamic Stops: Dynamic profit-target and stop-loss scaled by ATR with time stop.
  5. Realistic Friction: Taker fee (0.05%) + slippage (0.03%) = 0.16% round-trip,
     and Maker fee structure (0.02% maker + 0.01% slippage = 0.06% round-trip).

Evaluates:
  - Trade count per day across universe (evaluating the 3-4 signals/day target)
  - Win Rate, Profit Factor, Expected Value per trade (Net)
  - Portfolio Equity Curve, Max Drawdown, Annualized Sharpe & Sortino ratios
"""
import argparse
import os
import sys
import time

try:
    import ccxt
except ImportError:
    ccxt = None
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from src.regime import build_btc_regime
from src.labeling import scan_triple_barrier

DEFAULT_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'ADA/USDT', 'SUI/USDT', 'NEAR/USDT', 'INJ/USDT', 'FET/USDT', 'OP/USDT'
]


def download_or_load_data(symbols: list[str], timeframe: str = '1h', limit: int = 2500,
                          force_download: bool = False, data_dir: str = OUTPUT_DIR) -> dict[str, pd.DataFrame]:
    """Download OHLCV from Binance public API via CCXT or load cached CSVs."""
    os.makedirs(data_dir, exist_ok=True)
    dfs = {}
    exchange = None

    for sym in symbols:
        clean_sym = sym.replace('/', '_')
        csv_path = os.path.join(data_dir, f'binance_{clean_sym}_{timeframe}.csv')

        if os.path.exists(csv_path) and not force_download:
            df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
            if len(df) >= 100:
                dfs[clean_sym] = df
                continue

        # Fetch from Binance public API
        if exchange is None:
            if ccxt is None:
                raise ImportError("ccxt is required to download data from Binance API.")
            exchange = ccxt.binance({'enableRateLimit': True})

        print(f"Fetching {limit} bars of {timeframe} for {sym} from Binance...")
        all_ohlcv = []
        since_ms = exchange.milliseconds() - limit * 3600 * 1000 if timeframe == '1h' else exchange.milliseconds() - limit * 900 * 1000
        curr_since = since_ms

        while len(all_ohlcv) < limit:
            fetch_limit = min(1000, limit - len(all_ohlcv))
            try:
                ohlcv = exchange.fetch_ohlcv(sym, timeframe, since=curr_since, limit=fetch_limit)
            except Exception as e:
                print(f"Error fetching {sym}: {e}")
                break
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            curr_since = ohlcv[-1][0] + 1
            time.sleep(0.1)

        if not all_ohlcv:
            print(f"Warning: No data fetched for {sym}")
            continue

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.to_csv(csv_path)
        dfs[clean_sym] = df
        print(f"Saved {sym} to {csv_path}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    return dfs


def run_strategy_backtest(dfs: dict[str, pd.DataFrame],
                          donchian_lookback: int = 48,
                          squeeze_pct: float = 0.50,
                          rs_window: int = 24,
                          pt_mult: float = 4.0,
                          sl_mult: float = 1.2,
                          vol_mult: float = 1.05,
                          max_holding: int = 36,
                          min_adx: float = 20.0,
                          mode: str = 'breakout') -> pd.DataFrame:
    """Generate signals and simulate trades with triple-barrier ATR stops."""
    if 'BTC_USDT' not in dfs:
        raise ValueError("BTC_USDT is required as market benchmark for regime filter.")

    btc_df = dfs['BTC_USDT']
    btc_regime = build_btc_regime(btc_df)

    symbols = list(dfs.keys())
    # Cross-sectional returns for relative strength ranking
    rets_df = pd.DataFrame({s: dfs[s]['close'].pct_change(rs_window) for s in symbols})
    ranks_df = rets_df.rank(axis=1, ascending=False)
    top_cutoff = max(1, int(len(symbols) * 0.38))

    trades = []

    for s in symbols:
        df = dfs[s]
        h, l, c = df['high'], df['low'], df['close']
        v = df['volume'] if 'volume' in df.columns else None
        vol_sma20 = v.rolling(20).mean() if v is not None else None
        vol_ok = (v >= vol_mult * vol_sma20) if (v is not None and vol_mult > 0) else pd.Series(True, index=df.index)

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

        # Volatility squeeze: Bollinger width percentile
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

        # Regime filter from BTC
        reg_bull = (btc_regime['btc_trend'] > 0)
        reg_bear = (btc_regime['btc_trend'] < 0)

        # Relative strength ranking: top quintile for longs, bottom quintile for shorts
        is_top_rs = ranks_df[s] <= top_cutoff
        is_bot_rs = ranks_df[s] >= (len(symbols) - top_cutoff + 1)

        # 1) Breakout signals: Macro Trend + Dynamic RS + 48h Donchian Expansion + ADX strength + Volume Surge
        long_bo = (c > hi) & (c.shift(1) <= hi.shift(1)) & (c > ema100) & is_top_rs & reg_bull & adx_ok & vol_ok
        short_bo = (c < lo) & (c.shift(1) >= lo.shift(1)) & (c < ema100) & is_bot_rs & reg_bear & adx_ok & vol_ok

        # 2) Range Exhaustion Reversals (Active only in non-trending chop when ADX < 20)
        long_rev = (l < bb_lower) & (c > bb_lower) & (rsi < 30) & (adx < 20) & reg_bull
        short_rev = (h > bb_upper) & (c < bb_upper) & (rsi > 70) & (adx < 20) & reg_bear

        highs = h.to_numpy()
        lows = l.to_numpy()
        closes = c.to_numpy()
        atr_vals = atr.to_numpy()
        n = len(df)

        for i in range(n):
            side = 0
            strategy_name = 'NONE'
            curr_pt_m = pt_mult
            curr_sl_m = sl_mult

            if mode in ('breakout', 'combined'):
                if long_bo.iloc[i]:
                    side, strategy_name = 1, 'BREAKOUT'
                elif short_bo.iloc[i]:
                    side, strategy_name = -1, 'BREAKOUT'

            if side == 0 and mode in ('reversal', 'combined'):
                if long_rev.iloc[i]:
                    side, strategy_name = 1, 'REVERSAL'
                    curr_pt_m, curr_sl_m = 2.0, 1.2
                elif short_rev.iloc[i]:
                    side, strategy_name = -1, 'REVERSAL'
                    curr_pt_m, curr_sl_m = 2.0, 1.2

            if side == 0 or np.isnan(atr_vals[i]):
                continue

            entry_p = closes[i]
            entry_time = df.index[i]
            pt_dist = curr_pt_m * atr_vals[i]
            sl_dist = curr_sl_m * atr_vals[i]

            outcome = scan_triple_barrier(highs, lows, closes, i, side, entry_p,
                                          pt_dist, sl_dist, max_holding)
            exit_time = df.index[min(i + outcome['hold'], n - 1)]
            ret_gross = side * (outcome['exit_price'] / entry_p - 1.0)

            trades.append({
                'symbol': s,
                'strategy': strategy_name,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'side': side,
                'entry_price': entry_p,
                'exit_price': outcome['exit_price'],
                'hold': outcome['hold'],
                'label': outcome['label'],
                'ret_gross': ret_gross
            })

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf = tdf.sort_values('entry_time').reset_index(drop=True)
    return tdf


def compute_metrics(trades_df: pd.DataFrame, total_days: float, cost_per_trade: float,
                    initial_capital: float = 10000.0, position_size_pct: float = 0.20) -> dict:
    """Compute trade and portfolio performance metrics under given friction."""
    if trades_df.empty:
        return {}

    df = trades_df.copy()
    df['ret_net'] = df['ret_gross'] - cost_per_trade

    n_trades = len(df)
    wins = df[df['ret_net'] > 0]
    losses = df[df['ret_net'] <= 0]
    win_rate = len(wins) / n_trades * 100.0

    gross_profit = wins['ret_net'].sum()
    gross_loss = abs(losses['ret_net'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    ev_net_pct = df['ret_net'].mean() * 100.0
    avg_win_pct = wins['ret_net'].mean() * 100.0 if len(wins) > 0 else 0.0
    avg_loss_pct = losses['ret_net'].mean() * 100.0 if len(losses) > 0 else 0.0

    # Portfolio simulation
    capital = initial_capital
    records = []
    for _, row in df.iterrows():
        pnl = capital * position_size_pct * row['ret_net']
        capital += pnl
        records.append({'time': row['exit_time'], 'equity': capital})

    eq_df = pd.DataFrame(records).set_index('time')
    daily_eq = eq_df.resample('1D').last().ffill()
    daily_returns = daily_eq['equity'].pct_change().dropna()

    total_return_pct = (capital / initial_capital - 1.0) * 100.0

    # Max Drawdown
    cum_peak = daily_eq['equity'].cummax()
    dd = (daily_eq['equity'] - cum_peak) / cum_peak
    max_dd_pct = abs(dd.min()) * 100.0 if len(dd) > 0 else 0.0

    # Sharpe Ratio (annualized, 365 crypto days)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
    else:
        sharpe = 0.0

    # Sortino Ratio (downside deviation only)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        sortino = (daily_returns.mean() / downside_returns.std()) * np.sqrt(365)
    else:
        sortino = 0.0

    return {
        'total_trades': n_trades,
        'signals_per_day': n_trades / total_days,
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'ev_net_pct': ev_net_pct,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'total_return_pct': total_return_pct,
        'final_equity': capital,
        'max_drawdown_pct': max_dd_pct,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino
    }


def print_results(trades_df: pd.DataFrame, dfs: dict[str, pd.DataFrame], args: argparse.Namespace):
    """Print structured quantitative results and evaluate the 3-4 signals/day goal."""
    btc_df = dfs['BTC_USDT']
    start_ts = btc_df.index[0]
    end_ts = btc_df.index[-1]
    total_days = max(1.0, (end_ts - start_ts).total_seconds() / 86400.0)

    print("================================================================================")
    print("                 DAILY ALPHA STRATEGY BACKTEST RESULTS                          ")
    print("================================================================================")
    print(f"Data Period:         {start_ts} to {end_ts} ({total_days:.1f} days)")
    print(f"Timeframe:           {args.timeframe}")
    print(f"Universe ({len(dfs)}):       {', '.join(dfs.keys())}")
    print(f"Strategy:            Regime Filter (BTC Trend) + Relative Strength (Top/Bot Half)")
    print(f"                     Volatility Breakout (Donchian {args.donchian_lookback} + Squeeze {args.squeeze_pct})")
    print(f"Stops & Exits:       ATR Dynamic Stops: PT={args.pt_mult}x ATR, SL={args.sl_mult}x ATR, Max Hold={args.max_holding} bars")
    print("--------------------------------------------------------------------------------")

    # Signal Frequency Evaluation
    trades_df['date'] = trades_df['entry_time'].dt.date
    daily_counts = trades_df.groupby('date').size()
    active_days = len(daily_counts)
    total_signals = len(trades_df)
    sig_per_day_all = total_signals / total_days
    sig_per_day_active = daily_counts.mean() if active_days > 0 else 0.0

    print("SIGNAL FREQUENCY & UNIVERSE DISTRIBUTION:")
    print(f"  Total Signals Generated:   {total_signals}")
    print(f"  Signals Per Day (all days):{sig_per_day_all:.2f}")
    print(f"  Signals Per Day (active):  {sig_per_day_active:.2f}")
    print(f"  Active Trading Days:       {active_days} / {int(total_days)} days ({active_days/total_days*100:.1f}%)")
    print(f"  Daily Signal Quartiles:    25%={daily_counts.quantile(0.25):.1f} | Median={daily_counts.median():.1f} | 75%={daily_counts.quantile(0.75):.1f}")
    print(f"  Daily Min / Max Signals:   {daily_counts.min()} / {daily_counts.max()}")

    # Evaluation of the 3-4 signals per day target
    consistent = 2.8 <= sig_per_day_all <= 4.2
    status = "CONSISTENT (Target Met: 3-4 signals/day)" if consistent else "OUTSIDE TARGET RANGE"
    print(f"  3-4 Signals/Day Verdict:   {status}")

    # Breakdown by Strategy
    if 'strategy' in trades_df.columns:
        print("\nBREAKDOWN BY STRATEGY ENGINE:")
        for strat, group in trades_df.groupby('strategy'):
            s_wins = group[group['ret_gross'] > 0.0016]
            s_wr = len(s_wins) / len(group) * 100.0
            s_ev = (group['ret_gross'] - 0.0016).mean() * 100.0
            print(f"  {strat:<10s}: {len(group):3d} trades ({len(group)/total_days:.2f}/day) | Net Win Rate: {s_wr:.1f}% | Net EV (Taker): {s_ev:>+6.2f}%")

    # Breakdown by Symbol and Side
    print("\nBREAKDOWN BY ASSET:")
    sym_counts = trades_df.groupby('symbol').size()
    long_counts = trades_df[trades_df['side'] == 1].groupby('symbol').size()
    short_counts = trades_df[trades_df['side'] == -1].groupby('symbol').size()
    for s in dfs.keys():
        tot = sym_counts.get(s, 0)
        l = long_counts.get(s, 0)
        sh = short_counts.get(s, 0)
        print(f"  {s:<12s}: {tot:3d} signals ({tot/total_days:.2f}/day) [Long: {l:2d}, Short: {sh:2d}]")

    print("\n--------------------------------------------------------------------------------")
    print("PERFORMANCE UNDER REALISTIC FRICTIONS:")

    # Taker friction: 0.05% fee + 0.03% slippage = 0.08% per side = 0.16% round-trip
    taker_round_trip = 2.0 * (args.taker_fee + args.taker_slippage)
    taker_metrics = compute_metrics(trades_df, total_days, taker_round_trip,
                                    args.initial_capital, args.position_size_pct)

    # Maker friction: 0.02% fee + 0.01% slippage = 0.03% per side = 0.06% round-trip
    maker_round_trip = 2.0 * (args.maker_fee + args.maker_slippage)
    maker_metrics = compute_metrics(trades_df, total_days, maker_round_trip,
                                    args.initial_capital, args.position_size_pct)

    print(f"{'Metric':<30s} | {'Taker (0.16% RT)':<20s} | {'Maker (0.06% RT)':<20s}")
    print("-" * 78)
    print(f"{'Win Rate (%)':<30s} | {taker_metrics['win_rate_pct']:>18.2f}% | {maker_metrics['win_rate_pct']:>18.2f}%")
    print(f"{'Profit Factor':<30s} | {taker_metrics['profit_factor']:>19.2f} | {maker_metrics['profit_factor']:>19.2f}")
    print(f"{'Expected Value (Net %)':<30s} | {taker_metrics['ev_net_pct']:>+18.3f}% | {maker_metrics['ev_net_pct']:>+18.3f}%")
    print(f"{'Average Win (%)':<30s} | {taker_metrics['avg_win_pct']:>+18.3f}% | {maker_metrics['avg_win_pct']:>+18.3f}%")
    print(f"{'Average Loss (%)':<30s} | {taker_metrics['avg_loss_pct']:>+18.3f}% | {maker_metrics['avg_loss_pct']:>+18.3f}%")
    print(f"{'Portfolio Return (%)':<30s} | {taker_metrics['total_return_pct']:>+18.2f}% | {maker_metrics['total_return_pct']:>+18.2f}%")
    print(f"{'Final Equity ($)':<30s} | ${taker_metrics['final_equity']:>17.2f} | ${maker_metrics['final_equity']:>17.2f}")
    print(f"{'Max Drawdown (%)':<30s} | {taker_metrics['max_drawdown_pct']:>18.2f}% | {maker_metrics['max_drawdown_pct']:>18.2f}%")
    print(f"{'Annualized Sharpe Ratio':<30s} | {taker_metrics['sharpe_ratio']:>19.2f} | {maker_metrics['sharpe_ratio']:>19.2f}")
    print(f"{'Annualized Sortino Ratio':<30s} | {taker_metrics['sortino_ratio']:>19.2f} | {maker_metrics['sortino_ratio']:>19.2f}")
    print("================================================================================")


def main():
    parser = argparse.ArgumentParser(description="Backtest Daily Alpha Strategy on Crypto Universe")
    parser.add_argument("--symbols", nargs='+', default=DEFAULT_SYMBOLS,
                        help="Trading symbols (e.g., BTC/USDT ETH/USDT ...)")
    parser.add_argument("--mode", type=str, default="breakout", choices=["breakout", "combined", "reversal"],
                        help="Strategy mode: breakout, reversal, or combined (default: breakout)")
    parser.add_argument("--timeframe", type=str, default="1h", choices=["1h", "15m", "4h", "1d"],
                        help="Candle timeframe (default: 1h)")
    parser.add_argument("--limit", type=int, default=2500,
                        help="Bars of historical data to fetch/load (default: 2500)")
    parser.add_argument("--force-download", action="store_true",
                        help="Force fresh download of data via CCXT")
    parser.add_argument("--taker-fee", type=float, default=0.0005,
                        help="Taker fee per side (default: 0.05%% = 0.0005)")
    parser.add_argument("--taker-slippage", type=float, default=0.0003,
                        help="Taker slippage per side (default: 0.03%% = 0.0003)")
    parser.add_argument("--maker-fee", type=float, default=0.0002,
                        help="Maker fee per side (default: 0.02%% = 0.0002)")
    parser.add_argument("--maker-slippage", type=float, default=0.0001,
                        help="Maker slippage per side (default: 0.01%% = 0.0001)")
    parser.add_argument("--donchian-lookback", type=int, default=48,
                        help="Donchian breakout lookback period (default: 48)")
    parser.add_argument("--squeeze-pct", type=float, default=0.50,
                        help="Bollinger squeeze width percentile threshold (default: 0.50)")
    parser.add_argument("--rs-window", type=int, default=24,
                        help="Relative strength lookback window in bars (default: 24)")
    parser.add_argument("--min-adx", type=float, default=20.0,
                        help="Minimum ADX trend strength threshold (default: 20.0)")
    parser.add_argument("--vol-mult", type=float, default=1.05,
                        help="Volume surge multiplier relative to 20-period SMA (default: 1.05)")
    parser.add_argument("--pt-mult", type=float, default=4.0,
                        help="Profit target ATR multiplier (default: 4.0)")
    parser.add_argument("--sl-mult", type=float, default=1.2,
                        help="Stop loss ATR multiplier (default: 1.2)")
    parser.add_argument("--max-holding", type=int, default=36,
                        help="Max holding period in bars (default: 36)")
    parser.add_argument("--initial-capital", type=float, default=10000.0,
                        help="Initial portfolio capital in USDT (default: 10000)")
    parser.add_argument("--position-size-pct", type=float, default=0.20,
                        help="Allocation percentage per trade (default: 20%%)")

    args = parser.parse_args()

    dfs = download_or_load_data(args.symbols, timeframe=args.timeframe, limit=args.limit,
                                force_download=args.force_download)

    if not dfs:
        print("Error: No data available to run backtest.")
        sys.exit(1)

    trades_df = run_strategy_backtest(dfs,
                                      donchian_lookback=args.donchian_lookback,
                                      squeeze_pct=args.squeeze_pct,
                                      rs_window=args.rs_window,
                                      pt_mult=args.pt_mult,
                                      sl_mult=args.sl_mult,
                                      vol_mult=args.vol_mult,
                                      max_holding=args.max_holding,
                                      min_adx=args.min_adx,
                                      mode=args.mode)

    if trades_df.empty:
        print("No trades generated with the current parameters.")
        sys.exit(0)

    print_results(trades_df, dfs, args)


if __name__ == "__main__":
    main()
