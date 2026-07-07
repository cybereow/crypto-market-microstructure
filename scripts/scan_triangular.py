"""
Live triangular-arbitrage scanner across Tabdeal's own markets: for every
asset with both a USDT and an IRT market (discovered from exchangeInfo,
capped at config.yaml's triangular.max_symbols), compares X/USDT x USDT/IRT
against the real X/IRT price.

Does NOT place any orders -- it only polls public order books, logs every
snapshot to SQLite, and prints a line whenever the net edge (after 3 legs
of taker fees) clears the configured threshold. Let it run for a while to
build a real historical dataset, since none exists publicly for this.

IMPORTANT: this could not be tested from the build environment -- every
attempt to reach api1.tabdeal.org returned HTTP 503 through that sandbox's
outbound proxy. Run it yourself and confirm you get real numbers (not
errors) before trusting any output.

Usage:
    python scripts/scan_triangular.py
    python scripts/scan_triangular.py --once
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.exchanges.tabdeal import TabdealClient
from src.storage import Storage
from src.strategies.triangular_arbitrage import TriangularArbitrage


def run_cycle(strat: TriangularArbitrage, client: TabdealClient, storage: Storage, assets: list[str]):
    usdt_irt_bid, usdt_irt_ask = client.best_bid_ask(config.ANCHOR_SYMBOL)
    for asset in assets:
        try:
            snapshot = strat.fetch_snapshot(asset, usdt_irt_bid, usdt_irt_ask)
            storage.save_snapshot(snapshot)
            flag = " <-- VIABLE" if snapshot["net_edge_bps"] >= config.MIN_NET_EDGE_BPS else ""
            print(f"[{asset}] gross={snapshot['gross_edge_bps']:+.1f}bps  "
                  f"net={snapshot['net_edge_bps']:+.1f}bps  dir={snapshot['direction']}{flag}")
        except Exception as e:
            print(f"[{asset}] ERROR: {e}")
        time.sleep(config.REQUEST_DELAY_MS / 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one scan cycle and exit")
    args = ap.parse_args()

    client = TabdealClient()
    strat = TriangularArbitrage()
    storage = Storage()

    print("Discovering eligible assets from exchangeInfo...")
    assets = strat.eligible_assets()
    print(f"Scanning {len(assets)} assets (capped at max_symbols={config.MAX_SYMBOLS}): {assets}")
    print(f"min_net_edge_bps={config.MIN_NET_EDGE_BPS}, poll every {config.POLL_INTERVAL_SECONDS}s\n")

    while True:
        run_cycle(strat, client, storage, assets)
        if args.once:
            break
        time.sleep(config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
