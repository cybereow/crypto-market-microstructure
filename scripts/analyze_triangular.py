"""
Analyze snapshots collected by scan_triangular.py so far.

Usage:
    python scripts/analyze_triangular.py
    python scripts/analyze_triangular.py --base-asset BTC
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage import Storage
from src.strategies.triangular_arbitrage import TriangularArbitrage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-asset", default=None)
    args = ap.parse_args()

    storage = Storage()
    snapshots = storage.get_snapshots(args.base_asset)

    strat = TriangularArbitrage()
    result = strat.backtest({"snapshots": snapshots})
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
