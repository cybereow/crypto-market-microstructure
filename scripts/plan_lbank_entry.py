"""
Pre-entry planning calculator for LBank funding arb -- run this BEFORE
opening a real position. Uses real historical daily klines for the
liquidation safety check. Places no orders.

Usage:
    python scripts/plan_lbank_entry.py --symbol BTCUSDT --capital 400 --leverage 2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.lbank_funding_arb import LBankFundingArb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--capital", type=float, default=400.0)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--days-history", type=int, default=60)
    args = ap.parse_args()

    strat = LBankFundingArb(args.symbol, leverage=args.leverage)
    plan = strat.plan_entry(args.capital, args.leverage, args.days_history)
    print(json.dumps(plan, indent=2, default=str))


if __name__ == "__main__":
    main()
