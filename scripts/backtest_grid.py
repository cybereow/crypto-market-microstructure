import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from src.strategies.grid_trading import GridTradingStrategy

def main():
    parser = argparse.ArgumentParser(description="Backtest Grid Trading Strategy.")
    parser.add_argument("--data", type=str, required=True, help="Filename of the asset CSV (e.g. kraken_BTC_USDT_1d.csv)")
    parser.add_argument("--grids", type=int, default=10, help="Number of grid levels")
    parser.add_argument("--range-pct", type=float, default=0.2, help="Grid range as a percentage of start price (e.g., 0.2 for +/- 10%)")
    args = parser.parse_args()

    data_path = os.path.join(OUTPUT_DIR, args.data)
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist.")
        sys.exit(1)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

    print(f"Running Grid Trading Backtest (Grids: {args.grids}, Range: {args.range_pct*100}%)...")
    strategy = GridTradingStrategy(num_grids=args.grids, grid_range_pct=args.range_pct)
    results = strategy.backtest(df)

    if not results:
        print("Backtest returned no results.")
        return

    print("-" * 40)
    print("Grid Trading Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Buy & Hold Return: {results['buy_hold_return']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    print("-" * 40)

if __name__ == "__main__":
    main()
