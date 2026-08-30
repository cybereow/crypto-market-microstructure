"""Backtest the two-sided market-making simulator (src/market_making.py) on
real tick-aggregated data, ablating the two skews (inventory, OBI) the same
way README section 7 ablated its four ML ideas -- so their contribution is
MEASURED, not assumed.

This is the "option 2" alternative to every directional strategy in this
repo: instead of fighting transaction cost for a directional edge, capture
the spread on both sides of the book. See src/market_making.py's docstring
for why the quotes are priced off realized volatility rather than the raw
top-of-book spread (the raw spread turns out to be far too thin to clear
even a retail maker fee on BTC perpetuals).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from src.market_making import simulate_market_making
from src.significance import bootstrap_mean_pvalue


def run_config(df: pd.DataFrame, name: str, **kwargs) -> dict:
    result = simulate_market_making(df, **kwargs)
    n_round_trips = min(result['n_bid_fills'], result['n_ask_fills'])
    print(f"  {name:<28} bid_fills={result['n_bid_fills']:6d}  ask_fills={result['n_ask_fills']:6d}  "
          f"round_trips~{n_round_trips:6d}  total_pnl={result['total_pnl']:+10.4f}  "
          f"avg_captured_spread={result['avg_captured_spread']:.5f}  "
          f"max_drawdown={result['max_drawdown']:.4f}")
    result['name'] = name
    result['n_round_trips'] = n_round_trips
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Ablate inventory/OBI skew in the market-making simulator on real tick data.")
    parser.add_argument("--data", type=str, required=True,
                         help="CSV from download_l2_obi.py with open/high/low/close/bid_close/"
                              "ask_close/obi columns.")
    parser.add_argument("--vol-lookback", type=int, default=60)
    parser.add_argument("--k-spread", type=float, default=3.0,
                         help="Half-spread in vol multiples. Chosen from cost, not fit to "
                              "results: real BTC 5s-bar range averages ~0.013% of price, so "
                              "k_spread=1.5 would price a round-trip spread roughly EQUAL to "
                              "the 0.04% assumed round-trip fee (a coin flip before any edge). "
                              "3.0 prices it at ~2x the fee, leaving real margin to measure.")
    parser.add_argument("--k-inventory", type=float, default=0.5)
    parser.add_argument("--k-obi", type=float, default=1.0)
    parser.add_argument("--max-inventory", type=float, default=5.0)
    parser.add_argument("--fee-pct", type=float, default=0.0002,
                         help="Per-fill fee (or, if negative, rebate) as a fraction of price. "
                              "0.0002 matches this repo's assumed retail maker fee elsewhere.")
    args = parser.parse_args()

    path = os.path.join(OUTPUT_DIR, args.data)
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    missing = [c for c in ['open', 'high', 'low', 'close', 'bid_close', 'ask_close', 'obi']
               if c not in df.columns]
    if missing:
        print(f"Error: {args.data} is missing columns {missing} -- re-run download_l2_obi.py "
              f"(needs the bid_close/ask_close update) to regenerate it.")
        sys.exit(1)

    spread_pct = ((df['ask_close'] - df['bid_close']) / df['close']).mean()
    range_pct = ((df['high'] - df['low']) / df['close']).mean()
    print(f"Data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    print(f"  mean top-of-book spread: {spread_pct:.4%} of price")
    print(f"  mean bar range: {range_pct:.4%} of price")
    print(f"  assumed fee: {args.fee_pct:.4%} per fill ({2 * args.fee_pct:.4%} round trip)\n")

    print("Ablation (same volatility-scaled quote width in every row -- only the skews change):")
    configs = [
        ('no skew (naive symmetric)', dict(k_inventory=0.0, k_obi=0.0)),
        ('+ inventory skew', dict(k_inventory=args.k_inventory, k_obi=0.0)),
        ('+ inventory + OBI skew', dict(k_inventory=args.k_inventory, k_obi=args.k_obi)),
    ]
    results = []
    for name, skew_kwargs in configs:
        result = run_config(df, name, vol_lookback=args.vol_lookback, k_spread=args.k_spread,
                             max_inventory=args.max_inventory, fee_pct=args.fee_pct, **skew_kwargs)
        results.append(result)

    print("\nSignificance of per-bar equity change (bootstrap, best config by total PnL):")
    best = max(results, key=lambda r: r['total_pnl'])
    equity_diffs = np.diff(best['equity'])
    if len(equity_diffs) > 1:
        p = bootstrap_mean_pvalue(equity_diffs, n_iter=3000)
        print(f"  {best['name']}: n={len(equity_diffs)} bars, mean equity change/bar={equity_diffs.mean():+.6f}, "
              f"p={p:.4f}")
        print("  (tests whether mean per-bar PnL is distinguishable from zero over the whole run --")
        print("   a coarser but unbiased check than trying to hand-pair individual bid/ask fills "
              "into round trips.)")


if __name__ == "__main__":
    main()
