import argparse
import os
import sys
import glob
import numpy as np

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from src.strategies.pairs_trading import PairsTradingStrategy

def main():
    parser = argparse.ArgumentParser(description="Backtest Statistical Arbitrage (Pairs Trading) Strategy.")
    parser.add_argument("--asset1", type=str, required=True, help="Filename of first asset CSV (e.g. kraken_BTC_USDT_1d.csv)")
    parser.add_argument("--asset2", type=str, required=True, help="Filename of second asset CSV")
    parser.add_argument("--z-entry", type=float, default=2.0, help="Z-Score entry threshold")
    parser.add_argument("--z-exit", type=float, default=0.5, help="Z-Score exit threshold")
    parser.add_argument("--window", type=int, default=30, help="Rolling window for Z-Score calculation")
    args = parser.parse_args()

    path1 = os.path.join(OUTPUT_DIR, args.asset1)
    path2 = os.path.join(OUTPUT_DIR, args.asset2)

    if not os.path.exists(path1) or not os.path.exists(path2):
        print("Error: One or both data files do not exist.")
        sys.exit(1)

    df1 = pd.read_csv(path1, index_col='timestamp', parse_dates=True)
    df2 = pd.read_csv(path2, index_col='timestamp', parse_dates=True)

    # To prevent lookahead bias, we only backtest on the Out-of-Sample data (last 20%)
    split_idx = int(len(df1) * 0.8)
    s1 = df1['close'].iloc[split_idx:]
    s2 = df2['close'].iloc[split_idx:]

    print(f"Backtesting Pairs Strategy on Out-of-Sample data for {args.asset1} and {args.asset2}")
    print(f"Params: Z-Entry={args.z_entry}, Z-Exit={args.z_exit}, Window={args.window}")

    strategy = PairsTradingStrategy(z_entry_threshold=args.z_entry, z_exit_threshold=args.z_exit, window=args.window)
    signals_df = strategy.generate_signals(s1, s2)

    if signals_df.empty:
        print("Not enough data to backtest.")
        return

    results_df = strategy.calculate_returns(signals_df)

    total_return = results_df['cum_ret'].iloc[-1] - 1
    win_rate = (results_df['strat_ret'] > 0).sum() / (results_df['strat_ret'] != 0).sum() if (results_df['strat_ret'] != 0).sum() > 0 else 0

    daily_rf = 0.0
    sharpe = np.sqrt(365) * (results_df['strat_ret'].mean() - daily_rf) / (results_df['strat_ret'].std() + 1e-9)
    roll_max = results_df['cum_ret'].cummax()
    drawdown = results_df['cum_ret'] / roll_max - 1.0
    max_dd = drawdown.min()

    print("-" * 40)
    print(f"Out-of-Sample Strategy Return (with 0.1% fees): {total_return:.2%}")
    print(f"Win Rate on Active Days: {win_rate:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()
