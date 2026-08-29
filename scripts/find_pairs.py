import argparse
import os
import sys
import glob

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR

def find_cointegrated_pairs(prices_df: pd.DataFrame, p_value_threshold=0.05):
    """
    Finds cointegrated pairs from a DataFrame of prices.
    Returns a list of tuples: (asset1, asset2, p_value)
    """
    n = prices_df.shape[1]
    keys = prices_df.keys()
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            S1 = prices_df[keys[i]]
            S2 = prices_df[keys[j]]

            # Check for valid data (drop NaNs from overlapping periods)
            common = pd.concat([S1, S2], axis=1).dropna()
            if len(common) < 30: # Need enough data points
                continue

            score, pvalue, _ = coint(common.iloc[:, 0], common.iloc[:, 1])
            if pvalue < p_value_threshold:
                pairs.append((keys[i], keys[j], pvalue))

    return pairs

def main():
    parser = argparse.ArgumentParser(description="Find cointegrated pairs among downloaded CSV data.")
    parser.add_argument("--directory", type=str, default=OUTPUT_DIR, help="Directory containing price CSV files")
    parser.add_argument("--p-value", type=float, default=0.05, help="P-value threshold for cointegration")
    args = parser.parse_args()

    # Load all CSVs in the directory
    csv_files = glob.glob(os.path.join(args.directory, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.directory}")
        sys.exit(1)

    prices = {}
    for f in csv_files:
        try:
            # Assuming filename format: exchange_SYMBOL_timeframe.csv
            basename = os.path.basename(f)
            symbol = basename.split("_")[1] + "/" + basename.split("_")[2].replace(".csv", "")

            df = pd.read_csv(f, index_col='timestamp', parse_dates=True)
            # We use 'close' prices for cointegration testing
            prices[symbol] = df['close']
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not prices:
        print("No valid price data could be loaded.")
        sys.exit(1)

    prices_df = pd.DataFrame(prices)

    print(f"Testing {len(prices_df.columns)} assets for cointegration...")
    pairs = find_cointegrated_pairs(prices_df, args.p_value)

    if pairs:
        print(f"Found {len(pairs)} cointegrated pair(s):")
        # Sort by p-value
        pairs.sort(key=lambda x: x[2])
        for p in pairs:
            print(f"  {p[0]} and {p[1]} - p-value: {p[2]:.4f}")
    else:
        print("No cointegrated pairs found.")

if __name__ == "__main__":
    main()
