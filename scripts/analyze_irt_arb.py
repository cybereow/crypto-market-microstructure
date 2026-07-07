"""
Analyze snapshots collected by scan_irt_arb.py so far.

Usage:
    python scripts/analyze_irt_arb.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage import Storage
from src.strategies.irt_arbitrage import IRTArbitrage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USDTIRT")
    args = ap.parse_args()

    storage = Storage()
    snapshots = storage.get_snapshots(args.pair)

    strat = IRTArbitrage()
    result = strat.backtest({"snapshots": snapshots})
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
