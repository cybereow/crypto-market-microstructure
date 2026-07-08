"""
Live LBank funding-rate collector: one API call per cycle covers every
perpetual-swap symbol at once (much cheaper than Tabdeal's per-asset
depth calls), logging each symbol's current funding rate to SQLite so a
real dataset builds up over time. No public historical archive exists for
this, so self-collection is the only option -- same reasoning as the
Tabdeal triangular scanner.

Does NOT place any orders.

Usage:
    python scripts/scan_lbank_funding.py
    python scripts/scan_lbank_funding.py --once
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.lbank_storage import LBankStorage
from src.strategies.lbank_funding_arb import LBankFundingArb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one poll and exit")
    args = ap.parse_args()

    strat = LBankFundingArb()
    storage = LBankStorage()

    print(f"Polling LBank perpetual funding rates every {config.LBANK_POLL_INTERVAL_SECONDS}s "
          f"(productGroup={config.LBANK_PRODUCT_GROUP})\n")

    while True:
        try:
            rows = strat.fetch_snapshot()
            storage.save_snapshots(rows)
            top = sorted(rows, key=lambda r: r["funding_rate"], reverse=True)[:5]
            print(f"[{len(rows)} symbols] top funding rates: " +
                  ", ".join(f"{r['symbol']}={r['funding_rate']*100:.4f}%" for r in top))
        except Exception as e:
            print(f"ERROR: {e}")

        if args.once:
            break
        time.sleep(config.LBANK_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
