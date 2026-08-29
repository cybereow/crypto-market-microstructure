import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from src.strategies.grid_trading import GridTradingStrategy

def main():
    parser = argparse.ArgumentParser(description="Backtest Grid Trading Strategy.")
    parser.add_argument("--data", type=str, required=True, help="Filename of the asset CSV (e.g. kraken_BTC_USDT_1d.csv)")
    parser.add_argument("--grids", type=int, default=10, help="Number of grid levels")
    parser.add_argument("--range-pct", type=float, default=0.2, help="Grid range as a percentage of start price (e.g., 0.2 for +/- 10%)")
    parser.add_argument("--grid-type", type=str, default="arithmetic", choices=["arithmetic", "geometric"], help="Type of grid spacing")
    parser.add_argument("--adaptive-atr", type=int, default=None, help="If set, uses ATR over this period to dynamically calculate grid range")
    parser.add_argument("--atr-multiplier", type=float, default=2.0, help="Multiplier for ATR when using adaptive grid")
    args = parser.parse_args()

    data_path = os.path.join(OUTPUT_DIR, args.data)
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist.")
        sys.exit(1)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

    if args.adaptive_atr:
        print(f"Running Grid Trading Backtest (Grids: {args.grids}, Adaptive ATR({args.adaptive_atr}) x {args.atr_multiplier}, Type: {args.grid_type})...")
    else:
        print(f"Running Grid Trading Backtest (Grids: {args.grids}, Range: {args.range_pct*100}%, Type: {args.grid_type})...")

    strategy = GridTradingStrategy(
        num_grids=args.grids,
        grid_range_pct=args.range_pct,
        grid_type=args.grid_type,
        adaptive_atr_period=args.adaptive_atr,
        atr_multiplier=args.atr_multiplier
    )
    results = strategy.backtest(df)

    if not results:
        print("Backtest returned no results.")
        return

    df_result = results['equity_curve']
    daily_rf = 0.0
    sharpe = np.sqrt(365) * (df_result['return'].mean() - daily_rf) / (df_result['return'].std() + 1e-9)
    roll_max = df_result['equity'].cummax()
    drawdown = df_result['equity'] / roll_max - 1.0
    max_dd = drawdown.min()

    print("-" * 40)
    print("Grid Trading Results (with 0.1% fees):")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Buy & Hold Return: {results['buy_hold_return']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()
