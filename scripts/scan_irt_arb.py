"""
Live cross-exchange spread scanner: Tabdeal vs Nobitex, USDT/IRT.

This does NOT place any orders -- it only polls both exchanges' public
order books, logs every snapshot to SQLite, and prints a line whenever the
net edge (after both exchanges' taker fees) clears the configured
threshold. Let it run for days/weeks to build your own historical dataset,
since no public archive exists for this pair.

IMPORTANT: this could not be tested from the build environment -- every
attempt to reach api1.tabdeal.org or apiv2.nobitex.ir returned HTTP 503
through that sandbox's outbound proxy (for both exchanges, and unrelated
Iranian domains too), so this is unverified against live traffic. Run it
yourself and confirm you actually get real bid/ask numbers before trusting
any output.

Usage:
    python scripts/scan_irt_arb.py
    python scripts/scan_irt_arb.py --once
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.storage import Storage
from src.strategies.irt_arbitrage import IRTArbitrage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabdeal-symbol", default="USDTIRT")
    ap.add_argument("--nobitex-symbol", default="USDTIRT")
    ap.add_argument("--once", action="store_true", help="fetch one snapshot and exit")
    args = ap.parse_args()

    strat = IRTArbitrage(args.tabdeal_symbol, args.nobitex_symbol)
    storage = Storage()

    print(f"Scanning {args.tabdeal_symbol} (Tabdeal) vs {args.nobitex_symbol} (Nobitex), "
          f"min_net_edge_bps={config.MIN_NET_EDGE_BPS}, poll every {config.POLL_INTERVAL_SECONDS}s")
    print("NOTE: fee_bps for Nobitex in config.yaml is a placeholder -- confirm the real value "
          "before trusting net_edge_bps.\n")

    while True:
        try:
            snapshot = strat.fetch_snapshot()
            storage.save_snapshot(snapshot)
            flag = " <-- VIABLE" if snapshot["net_edge_bps"] >= config.MIN_NET_EDGE_BPS else ""
            print(f"[{snapshot['pair']}] tabdeal={snapshot['tabdeal_bid']}/{snapshot['tabdeal_ask']}  "
                  f"nobitex={snapshot['nobitex_bid']}/{snapshot['nobitex_ask']}  "
                  f"gross={snapshot['gross_edge_bps']:.1f}bps  net={snapshot['net_edge_bps']:.1f}bps  "
                  f"dir={snapshot['direction']}{flag}")
        except Exception as e:
            print(f"ERROR fetching snapshot: {e}")

        if args.once:
            break
        time.sleep(config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
