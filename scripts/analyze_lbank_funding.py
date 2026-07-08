"""
Analyze funding-rate snapshots collected by scan_lbank_funding.py so far.

Usage:
    python scripts/analyze_lbank_funding.py
    python scripts/analyze_lbank_funding.py --symbol BTCUSDT
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lbank_storage import LBankStorage
from src.strategies.lbank_funding_arb import LBankFundingArb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--leverage", type=float, default=1.0)
    args = ap.parse_args()

    storage = LBankStorage()
    snapshots = storage.get_snapshots(args.symbol)

    strat = LBankFundingArb(leverage=args.leverage)
    result = strat.backtest({"snapshots": snapshots}, capital=args.capital)

    if "per_symbol" in result:
        result["per_symbol"] = dict(list(result["per_symbol"].items())[:args.top])
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
