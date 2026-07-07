"""
Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --symbols BTCUSDT --start 2024-01 --end 2026-06
    python scripts/run_backtest.py --symbols BTCUSDT --leverage 2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.strategies.funding import FundingArb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(config.SYMBOLS))
    ap.add_argument("--start", default=config.FUNDING_START)
    ap.add_argument("--end", default=config.FUNDING_END)
    ap.add_argument("--leverage", type=float, default=config.LEVERAGE)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    results = []
    for symbol in symbols:
        print(f"\n{'='*64}\n[{symbol}] downloading funding-rate history {args.start}..{args.end} ...", flush=True)
        strat = FundingArb(symbol, leverage=args.leverage)
        try:
            history = strat.fetch_history(args.start, args.end)
            print(f"[{symbol}] {len(history)} funding payments, "
                  f"{history['ts'].min().date()} -> {history['ts'].max().date()}")
            result = strat.backtest({"funding_history": history}, capital=config.CAPITAL)
            print(f"[{symbol}] naive-APY={result['naive_annualized_pct']}%  "
                  f"net_apy(taker)={result['apy']['net_apy_pct']}%")
            if "rolling" in result:
                r = result["rolling"]
                print(f"[{symbol}] rolling {r['window_days']}d annualized yield: "
                      f"min={r['min_pct']}%  median={r['median_pct']}%  max={r['max_pct']}%  "
                      f"({r['pct_windows_negative']}% of windows net negative)")
            results.append(result)
            history.to_csv(os.path.join(config.OUTPUT_DIR, f"{symbol}_funding_history.csv"), index=False)
        except Exception as e:
            print(f"[{symbol}] ERROR: {e}")

    with open(os.path.join(config.OUTPUT_DIR, "funding_summary.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nSaved to {config.OUTPUT_DIR}/funding_summary.json")


if __name__ == "__main__":
    main()
