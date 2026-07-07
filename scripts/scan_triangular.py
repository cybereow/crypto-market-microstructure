"""
Live triangular-arbitrage scanner across Tabdeal's own markets: for every
asset with both a USDT and an IRT market (discovered from exchangeInfo,
capped at config.yaml's triangular.max_symbols), compares X/USDT x USDT/IRT
against the real X/IRT price.

Requests within a cycle run in parallel (triangular.max_workers) so a full
scan covers many assets in roughly the time of one request round-trip, not
one request per asset sequentially -- this matters because a real
mispricing may only last minutes, and wider/faster coverage per cycle is
the only real lever for catching more of them sooner.

Does NOT place any orders -- it only polls public order books, logs every
snapshot to SQLite, and prints a line whenever the net edge (after 3 legs
of taker fees) clears the configured threshold. Let it run for a while to
build a real historical dataset, since none exists publicly for this.

Usage:
    python scripts/scan_triangular.py
    python scripts/scan_triangular.py --once
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.exchanges.tabdeal import TabdealClient
from src.storage import Storage
from src.strategies.triangular_arbitrage import TriangularArbitrage


def run_cycle(strat: TriangularArbitrage, client: TabdealClient, storage: Storage, assets: list[str]):
    usdt_irt_bid, usdt_irt_ask = client.best_bid_ask(config.ANCHOR_SYMBOL)

    snapshots = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {
            pool.submit(strat.fetch_snapshot, asset, usdt_irt_bid, usdt_irt_ask): asset
            for asset in assets
        }
        for future in as_completed(futures):
            asset = futures[future]
            try:
                snapshots.append(future.result())
            except Exception as e:
                print(f"[{asset}] ERROR: {e}")

    for snapshot in snapshots:
        storage.save_snapshot(snapshot)

    viable = [s for s in snapshots if s["net_edge_bps"] >= config.MIN_NET_EDGE_BPS]
    for s in sorted(viable, key=lambda s: -s["net_edge_bps"]):
        print(f"  VIABLE [{s['base_asset']}] net={s['net_edge_bps']:+.1f}bps  dir={s['direction']}")

    if snapshots:
        best = max(snapshots, key=lambda s: s["net_edge_bps"])
        print(f"cycle: {len(snapshots)} assets scanned, {len(viable)} viable, "
              f"best={best['base_asset']} ({best['net_edge_bps']:+.1f}bps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one scan cycle and exit")
    args = ap.parse_args()

    client = TabdealClient()
    strat = TriangularArbitrage()
    storage = Storage()

    print("Discovering eligible assets from exchangeInfo...")
    assets = strat.eligible_assets()
    print(f"Scanning {len(assets)} assets (capped at max_symbols={config.MAX_SYMBOLS}, "
          f"{config.MAX_WORKERS} parallel requests)")
    print(f"min_net_edge_bps={config.MIN_NET_EDGE_BPS}, poll every {config.POLL_INTERVAL_SECONDS}s\n")

    while True:
        t0 = time.time()
        run_cycle(strat, client, storage, assets)
        print(f"(cycle took {time.time() - t0:.1f}s)\n")
        if args.once:
            break
        time.sleep(config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
